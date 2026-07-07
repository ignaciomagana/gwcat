"""Raw metadata response caching + offline-mode gate for ``gwcat.fetch`` (PR 8).

The handoff's "Online Data Strategy" draws a hard line between *release
manifests* (declarative, bundled), *online discovery* (Zenodo file listings,
GWOSC event/BBH-name queries), and the *local cache* ("source of
reproducibility").  This module implements that local-cache layer:

  * ``write_metadata_cache`` / ``read_metadata_cache`` persist the raw parsed
    JSON payload of one online metadata response under
    ``<cache_dir>/metadata/<key>.json``, wrapped with a fetch timestamp so the
    cache file is self-describing.
  * ``is_offline`` resolves whether a call should avoid the network at all:
    an explicit ``offline=True/False`` always wins; otherwise the
    ``GWCAT_OFFLINE`` environment variable is consulted (any of
    ``"1"/"true"/"yes"/"on"``, case-insensitive, means offline).
  * ``OfflineCacheMissError`` is raised — naming the exact cache file that is
    missing — when offline mode is requested but nothing has been cached yet.
    Offline mode never falls back to a network call.

Nothing here performs I/O over the network; it is pure local file handling so
it needs no mocking in tests beyond a temporary directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional, Union

__all__ = [
    "ENV_OFFLINE",
    "OfflineCacheMissError",
    "is_offline",
    "metadata_cache_dir",
    "metadata_cache_path",
    "write_metadata_cache",
    "read_metadata_cache",
    "zenodo_cache_key",
    "gwosc_cache_key",
]

#: Environment variable that forces offline mode when no explicit ``offline``
#: argument is passed to a fetch/metadata function.
ENV_OFFLINE = "GWCAT_OFFLINE"

_TRUE_VALUES = {"1", "true", "yes", "on"}


class OfflineCacheMissError(RuntimeError):
    """Offline mode was requested but the needed cache file does not exist.

    The message always names the exact path that was expected, so the caller
    can tell precisely what to populate (by running once online with the same
    ``cache_dir``) rather than guessing.
    """


def is_offline(offline: Optional[bool] = None) -> bool:
    """Resolve effective offline-mode state.

    An explicit ``True``/``False`` always wins.  ``None`` (the default) falls
    back to the ``GWCAT_OFFLINE`` environment variable.
    """
    if offline is not None:
        return bool(offline)
    return os.environ.get(ENV_OFFLINE, "").strip().lower() in _TRUE_VALUES


def metadata_cache_dir(cache_dir: Union[str, Path]) -> Path:
    """Return ``<cache_dir>/metadata`` (not created)."""
    return Path(cache_dir) / "metadata"


def metadata_cache_path(cache_dir: Union[str, Path], key: str) -> Path:
    """Return the on-disk path for one cached metadata response."""
    return metadata_cache_dir(cache_dir) / f"{key}.json"


def write_metadata_cache(cache_dir: Union[str, Path], key: str, payload: Any) -> Path:
    """Write ``payload`` (already-parsed JSON-able data) to the metadata cache.

    Wraps the payload with a fetch timestamp (unix + ISO-8601 UTC) and the
    cache key, so the file on disk is self-describing.  Returns the path
    written.
    """
    path = metadata_cache_path(cache_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "key": key,
        "fetched_at": time.time(),
        "fetched_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "payload": payload,
    }
    path.write_text(json.dumps(record, indent=2, default=str))
    return path


def read_metadata_cache(cache_dir: Union[str, Path], key: str) -> Any:
    """Read back a cached payload written by :func:`write_metadata_cache`.

    Raises :class:`OfflineCacheMissError` (naming the missing path) if the
    cache file does not exist -- this is the "clear error" contract for
    offline mode: it never falls through to a network call.
    """
    path = metadata_cache_path(cache_dir, key)
    if not path.exists():
        raise OfflineCacheMissError(
            f"Offline mode: no cached metadata response at {path} "
            f"(key={key!r}). Run the equivalent fetch once online with "
            f"cache_dir={str(cache_dir)!r} to populate it, or pass "
            "offline=False / unset GWCAT_OFFLINE."
        )
    with open(path, "r") as f:
        record = json.load(f)
    return record["payload"]


def zenodo_cache_key(record_id) -> str:
    """Cache key for one Zenodo record's raw file-listing response."""
    return f"zenodo_{record_id}"


def gwosc_cache_key(name: str) -> str:
    """Cache key for a GWOSC query, derived from a human-readable ``name``.

    Kept human-readable when it only contains filesystem-safe characters;
    otherwise falls back to a short stable hash so arbitrary query strings
    (long, containing odd characters, etc.) never produce an unsafe or
    excessively long filename.
    """
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(name))
    if safe == str(name) and len(safe) <= 120:
        return f"gwosc_{safe}"
    digest = hashlib.sha256(str(name).encode("utf-8")).hexdigest()[:16]
    return f"gwosc_{digest}"
