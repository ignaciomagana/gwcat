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
    "resolution_cache_key",
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

    The write is atomic: the record is serialized to a temporary file in the
    same directory and then ``os.replace``d onto the final path, so an
    interrupted write (Ctrl-C, full disk, killed job) can never leave a
    half-written cache file that a later offline replay would read as if it
    were a complete record of the online run.
    """
    path = metadata_cache_path(cache_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "key": key,
        "fetched_at": time.time(),
        "fetched_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "payload": payload,
    }
    text = json.dumps(record, indent=2, default=str)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    finally:
        if tmp.exists():                       # os.replace failed
            tmp.unlink()
    return path


def read_metadata_cache(cache_dir: Union[str, Path], key: str) -> Any:
    """Read back a cached payload written by :func:`write_metadata_cache`.

    Raises :class:`OfflineCacheMissError` (naming the missing path) if the
    cache file does not exist -- this is the "clear error" contract for
    offline mode: it never falls through to a network call.

    A file that exists but is truncated/unparseable, or that lacks the
    ``payload`` key, raises the same error: a damaged cache is a cache miss,
    not a usable record of an online run.  It must never surface as a raw
    ``JSONDecodeError``/``KeyError`` that a caller might mistake for a data
    problem rather than a cache problem.
    """
    path = metadata_cache_path(cache_dir, key)
    if not path.exists():
        raise OfflineCacheMissError(
            f"Offline mode: no cached metadata response at {path} "
            f"(key={key!r}). Run the equivalent fetch once online with "
            f"cache_dir={str(cache_dir)!r} to populate it, or pass "
            "offline=False / unset GWCAT_OFFLINE."
        )
    try:
        with open(path, "r") as f:
            record = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise OfflineCacheMissError(
            f"Offline mode: cached metadata response at {path} (key={key!r}) "
            f"is unreadable/truncated ({exc}). Delete it and re-run the "
            f"equivalent fetch once online with cache_dir={str(cache_dir)!r}."
        ) from exc
    if not isinstance(record, dict) or "payload" not in record:
        raise OfflineCacheMissError(
            f"Offline mode: cached metadata response at {path} (key={key!r}) "
            "has no 'payload' -- it was not written by "
            "gwcat.fetch_cache.write_metadata_cache, or the write was "
            "interrupted. Delete it and re-run the equivalent fetch online."
        )
    return record["payload"]


def zenodo_cache_key(record_id) -> str:
    """Cache key for one Zenodo record's raw file-listing response."""
    return f"zenodo_{record_id}"


def resolution_cache_key(catalog: str) -> str:
    """Cache key for one release's *resolved* Zenodo record IDs.

    ``fetch_catalog(resolve=True)`` follows each concept DOI to whatever record
    is latest at that moment.  Which records those were is part of what an
    online run did, so it is cached under this key and read back in offline
    mode -- otherwise offline replay silently falls back to the *pinned*
    records and reproduces a different file set than the online run used.
    """
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(catalog))
    return f"resolution_{safe}"


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
