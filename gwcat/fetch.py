"""Fetch GWTC PE data releases from Zenodo, filter to cosmo-only, and build the store.

Usage (CLI):
    gwcat-fetch --catalog GWTC-2.1 GWTC-3 GWTC-4.1 GWTC-5 --out store.h5
    gwcat-fetch --catalog all --data-dir ./GWTC --out store.h5
    gwcat-fetch --catalog GWTC-5 --dry-run

Usage (Python):
    from gwcat.fetch import fetch_and_build, fetch_catalog, RELEASES

    # Download + ingest in one shot
    fetch_and_build(["GWTC-2.1", "GWTC-3", "GWTC-4.1", "GWTC-5"], out="store.h5")

    # Or step by step
    paths = fetch_catalog("GWTC-5", data_dir="./GWTC")

By default the fetcher resolves each Zenodo concept DOI to its latest version,
so you always get the most recent release without editing record IDs.

Requires: pip install gwcat[fetch]   (requests + tqdm)

This module deliberately keeps two separate concerns apart (PR 8):

  * **FILE discovery/download** (Zenodo) -- "which files exist in a release,
    and how do I get them onto disk": :func:`list_files`, :func:`resolve_latest`,
    :func:`download_file`, :func:`fetch_catalog`, :func:`fetch_and_build`.
  * **EVENT-METADATA discovery** (GWOSC) -- "what does the public event
    catalog say about FAR/p_astro/BBH membership for named events, which
    online metadata cannot be assumed complete for": :func:`fetch_bbh_names_gwosc`,
    :func:`fetch_event_table_gwosc`.

Neither path calls into the other.  Merging online metadata with manifest
defaults / user overrides, and recording per-field provenance, is a further
layer on top of the raw GWOSC calls here -- see :mod:`gwcat.event_metadata`.

Both discovery paths support the same local-cache / offline-mode contract
(see :mod:`gwcat.fetch_cache`): pass ``cache_dir=...`` to persist the raw
online response under ``<cache_dir>/metadata/`` (with a fetch timestamp), and
``offline=True`` (or set ``GWCAT_OFFLINE=1``) to force reading that cache
instead of making any network call -- raising a clear error naming the
missing cache file if it was never populated.  Neither argument changes
default (``cache_dir=None, offline=None/False``) behavior: no cache_dir means
no caching side effect, exactly as before this PR.

Reproducibility extends to *which Zenodo records* a run used: with
``resolve=True`` those are whatever each concept DOI pointed at when the online
run ran, so the resolution is cached alongside the file listings and read back
offline.  Offline mode therefore replays the online run's records rather than
the registry's pinned ones (GW-15).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union
from urllib.parse import urlencode

import numpy as np
from urllib.request import urlopen, Request
from urllib.error import HTTPError

from . import fetch_cache
from .event_metadata import resolve_pastro
from .manifests import (
    ManifestValidationError,
    ReleaseManifest,
    get_manifest,
    list_injection_manifests,
    list_release_manifests,
)

# ---------------------------------------------------------------------------
# Release registry — built from declarative manifests (PR 7)
# ---------------------------------------------------------------------------
# Release/injection metadata (Zenodo record IDs, per-release file-name
# filters, descriptions, observing runs) used to be hardcoded here as Python
# dicts.  It now lives in YAML manifests bundled under gwcat/manifests/
# (releases/*.yaml, injections/*.yaml) and is loaded via gwcat.manifests.
# Adding a new release requires only a new manifest file — see
# gwcat.manifests for the schema and gwcat.manifests.get_manifest for how
# user-supplied manifest paths are also accepted.
#
# record_ids  : pinned version records (one per Zenodo deposit).
#               GWTC-5 is split across two Zenodo deposits.
# concept_ids : version-agnostic record IDs; resolve_latest() follows
#               these to find the newest version.  Use these by default.

@dataclass
class ReleaseInfo:
    """Metadata for one release registered for fetch_catalog.

    Built from a ``gwcat.manifests.ReleaseManifest`` (see ``_release_info_from_manifest``);
    ``file_filter`` is the bound ``ProductSpec.matches`` of that manifest's single
    product.
    """
    record_ids: List[int]
    concept_ids: List[Optional[int]]
    file_filter: Callable[[str], bool]
    description: str
    observing_run: str
    manifest: Optional[ReleaseManifest] = field(default=None, repr=False)


def _primary_product(manifest: ReleaseManifest):
    """Return the single product spec fetch.py should use to select files.

    fetch.py currently downloads exactly one product family per release
    (PE samples, or one injection set); manifests with more than one
    product need a future fetch.py extension to disambiguate.
    """
    if len(manifest.products) != 1:
        raise ManifestValidationError(
            f"{manifest.source_path}: fetch.py expects exactly one product "
            f"per manifest, found {sorted(manifest.products)}"
        )
    return next(iter(manifest.products.values()))


def _release_info_from_manifest(manifest: ReleaseManifest) -> ReleaseInfo:
    product = _primary_product(manifest)
    return ReleaseInfo(
        record_ids=list(manifest.record_ids),
        concept_ids=list(manifest.concept_ids),
        file_filter=product.matches,
        description=manifest.description,
        observing_run=manifest.observing_run,
        manifest=manifest,
    )


def _build_registry(names: List[str]) -> Dict[str, ReleaseInfo]:
    """Build a {name: ReleaseInfo} registry from bundled manifest names,
    also registering each manifest's declared aliases (e.g. "GWTC-4")."""
    registry: Dict[str, ReleaseInfo] = {}
    for name in names:
        manifest = get_manifest(name)
        info = _release_info_from_manifest(manifest)
        registry[name] = info
        for alias in manifest.aliases:
            registry[alias] = info
    return registry


#: PE data releases only (GWTC-2.1, GWTC-3, GWTC-4.1 [+ "GWTC-4" alias], GWTC-5).
RELEASES: Dict[str, ReleaseInfo] = _build_registry(list_release_manifests())

#: Injection/selection-function releases only.
INJECTION_RELEASES: Dict[str, ReleaseInfo] = _build_registry(list_injection_manifests())

# Merge injection records into RELEASES so fetch_catalog finds everything.
RELEASES.update(INJECTION_RELEASES)

#: Registry keys that are aliases of another key (e.g. "GWTC-4" -> "GWTC-4.1"),
#: hidden from CLI help / error listings so each release is only shown once.
_ALIAS_NAMES = {
    alias
    for info in RELEASES.values()
    if info.manifest is not None
    for alias in info.manifest.aliases
}

ZENODO_API = "https://zenodo.org/api/records"


# ---------------------------------------------------------------------------
# FILE DISCOVERY (Zenodo) — stdlib only, no requests needed for metadata queries
# ---------------------------------------------------------------------------
def _zenodo_get(url: str, timeout: int = 30) -> dict:
    """GET a Zenodo API endpoint and return parsed JSON."""
    req = Request(url, headers={"Accept": "application/json"})
    for attempt in range(3):
        try:
            with urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as e:
            if e.code == 429:
                wait = int(e.headers.get("Retry-After", 5 * (attempt + 1)))
                warnings.warn(f"Zenodo rate-limited; retrying in {wait}s")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Zenodo API failed after retries: {url}")


def list_files(
    record_id: int,
    cache_dir: Optional[Union[str, Path]] = None,
    offline: Optional[bool] = None,
) -> List[dict]:
    """Return the file list for a Zenodo record.

    Parameters
    ----------
    record_id : int
        Zenodo record ID.
    cache_dir : str or Path, optional
        When given, the raw Zenodo record JSON response is written to
        ``<cache_dir>/metadata/zenodo_<record_id>.json`` (with a fetch
        timestamp) after a live fetch.  ``None`` (default) disables caching
        entirely -- no cache file is written and behavior is unchanged from
        before PR 8.
    offline : bool, optional
        If true (or ``GWCAT_OFFLINE`` is set and ``offline`` is not passed),
        read the cached response from ``cache_dir`` instead of making a
        network call.  Raises :class:`gwcat.fetch_cache.OfflineCacheMissError`
        naming the missing cache file if it was never populated; requires
        ``cache_dir``.
    """
    offline_mode = fetch_cache.is_offline(offline)
    key = fetch_cache.zenodo_cache_key(record_id)
    if offline_mode:
        if cache_dir is None:
            raise fetch_cache.OfflineCacheMissError(
                f"list_files(record_id={record_id}, offline=True) requires "
                "cache_dir to locate the previously cached Zenodo response."
            )
        data = fetch_cache.read_metadata_cache(cache_dir, key)
    else:
        data = _zenodo_get(f"{ZENODO_API}/{record_id}")
        if cache_dir is not None:
            fetch_cache.write_metadata_cache(cache_dir, key, data)

    files = data.get("files", [])
    if not files:
        raise RuntimeError(
            f"No files found in Zenodo record {record_id}. "
            "The record may be embargoed or the API schema changed."
        )
    return files


def resolve_latest(concept_id: int) -> int:
    """Resolve a Zenodo concept DOI to its latest version record ID."""
    data = _zenodo_get(f"{ZENODO_API}/{concept_id}")
    latest_url = data.get("links", {}).get("latest", "")
    if latest_url:
        latest_data = _zenodo_get(latest_url)
        return int(latest_data["id"])
    return int(data["id"])


# ---------------------------------------------------------------------------
# FILE DISCOVERY (Zenodo) — download helpers
# ---------------------------------------------------------------------------
def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256(path: str) -> str:
    """sha256 of a local file, used for download-provenance wiring (PR 8).

    Distinct from :func:`_md5`, which verifies against the checksum Zenodo
    publishes for a file; this is the hash recorded into the store's per-row
    ``file_checksum`` meta column via ``build_store(file_provenance=...)``.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_url_for(file_entry: dict) -> str:
    """Extract the download URL from a Zenodo file entry."""
    links = file_entry.get("links", {})
    for key in ("content", "self"):
        if key in links:
            return links[key]
    raise KeyError(f"Cannot find download URL in file entry: {file_entry.get('key', '?')}")


#: hashlib algorithms a Zenodo ``checksum`` entry may name, and the hex length
#: that identifies each when an entry carries no ``<algo>:`` prefix.
_CHECKSUM_HEX_LENGTHS = {"md5": 32, "sha1": 40, "sha256": 64, "sha512": 128}


def _parse_checksum(file_entry: dict) -> Optional[tuple]:
    """``(algorithm, hex digest)`` of a Zenodo file entry, or ``None``.

    Zenodo writes ``md5:<hex>``.  Any other prefix used to be passed on
    verbatim as if it were an md5, so a ``sha256:...`` entry never matched a
    good file: every call re-downloaded it, and the download was then
    "corrupt".  Entries are now read as ``<algo>:<hex>`` for any algorithm in
    :data:`_CHECKSUM_HEX_LENGTHS`; a bare digest is identified by its length
    (md5 otherwise, the historical reading, so a garbled one still fails the
    comparison loudly).  An algorithm this module cannot evaluate gives
    ``None`` with a warning -- the file is then unverified, which a release
    that requires checksums refuses.
    """
    cs = str(file_entry.get("checksum") or "").strip()
    if not cs:
        return None
    algo, sep, digest = cs.partition(":")
    if sep:
        algo = algo.strip().lower()
    else:
        digest = cs
        algo = next((a for a, n in _CHECKSUM_HEX_LENGTHS.items()
                     if n == len(cs)), "md5")
    if algo not in _CHECKSUM_HEX_LENGTHS:
        warnings.warn(
            f"{file_entry.get('key', '?')}: cannot evaluate the checksum "
            f"{cs!r} (supported: {sorted(_CHECKSUM_HEX_LENGTHS)}); the file "
            f"is unverified.")
        return None
    return algo, digest.strip().lower()


def _file_digests(path: str, algos) -> Dict[str, str]:
    """Hex digests of one file for every algorithm in ``algos``, in ONE read.

    Verifying a cached file against Zenodo's md5 and recording its sha256 for
    provenance used to read the file twice.
    """
    hashers = {a: hashlib.new(a) for a in sorted(set(algos))}
    if not hashers:
        return {}
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            for h in hashers.values():
                h.update(chunk)
    return {a: h.hexdigest() for a, h in hashers.items()}


def _require_checksums(info: "ReleaseInfo") -> bool:
    """Whether this release's manifest demands a checksum for every file.

    Declared as ``validation.require_checksums`` and schema-checked at load, but
    read by nothing until GW-37 -- a declaration nothing consults is a comment.
    Releases registered without a manifest state nothing and default to False.
    """
    manifest = getattr(info, "manifest", None)
    if manifest is None:
        return False
    return bool((manifest.validation or {}).get("require_checksums", False))


def _contained_dest(dest_dir, key: str) -> str:
    """``dest_dir/key``, refusing a remote-supplied name that escapes it.

    ``key`` comes from the Zenodo API, and it was joined onto the data dir with
    no check (GW-37): a record listing ``../../.ssh/authorized_keys`` -- or an
    absolute path, which ``/`` discards the left operand for -- wrote wherever
    it liked. Nothing about the fetch path authenticates the record contents, so
    the destination has to be constrained here.

    The check is on the NAME, lexically: no absolute path, no ``..`` or empty
    component, no backslash.  It used to resolve the joined path, which
    follows a symlink at the destination -- so a data directory whose files
    link into a shared group copy (matching checksums, nothing to download)
    was refused, even offline, as a record naming a path outside it.  Links
    the user made inside the data directory are the user's, and the remote
    name can no longer climb out of it.  Only the base is resolved.
    """
    from pathlib import Path, PurePosixPath
    base = Path(dest_dir).resolve()
    name = str(key)
    parts = PurePosixPath(name).parts
    if (not name or "\\" in name or "\x00" in name
            or PurePosixPath(name).is_absolute()
            or any(p in ("", ".", "..") for p in name.split("/"))
            or not parts):
        raise RuntimeError(
            f"refusing to download {key!r}: the remote file name is not a "
            f"plain relative path inside the data directory {base} (it is "
            f"absolute, empty, or has a '.', '..' or empty component). A "
            f"release record that names a path outside its own download "
            f"directory is not one to trust.")
    return str(base.joinpath(*parts))


def download_file(url: str, dest: str, expected_md5: Optional[str] = None,
                  show_progress: bool = True,
                  expected_checksum: Optional[tuple] = None) -> str:
    """Download a single file with progress bar and checksum verification.
    Skips download if dest exists and checksum matches.

    ``expected_checksum`` is an ``(algorithm, hex digest)`` pair (see
    :func:`_parse_checksum`); ``expected_md5`` is the older md5-only spelling
    of the same thing.
    """
    dest = str(dest)
    if expected_checksum is None and expected_md5:
        expected_checksum = ("md5", str(expected_md5).lower())
    if os.path.exists(dest) and expected_checksum:
        algo, digest = expected_checksum
        if _file_digests(dest, [algo])[algo] == digest:
            return dest

    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        raise ImportError(
            "The fetch module requires 'requests' and 'tqdm'. "
            "Install them with:  pip install gwcat[fetch]"
        )

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("Content-Length", 0))

    # Staged, then renamed (GW-37).  Writing straight to `dest` meant a dropped
    # connection, a walltime kill, or the checksum refusal below all left a
    # truncated file at exactly the path the toolchain consumes -- and the
    # refusal declared the file corrupt while leaving it there.  The temp name
    # carries the pid so two concurrent fetches into one data dir cannot write
    # the same bytes over each other.
    tmp = f"{dest}.{os.getpid()}.part"
    try:
        n_written = 0
        with open(tmp, "wb") as f:
            bar = tqdm(total=total, unit="B", unit_scale=True,
                       desc=os.path.basename(dest)[:40], leave=False) \
                  if show_progress and total else None
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                n_written += len(chunk)
                if bar:
                    bar.update(len(chunk))
            if bar:
                bar.close()

        # A short body is a failed download, not a small file.  Without this a
        # 10-byte response against Content-Length 1000 was returned as success.
        if total and n_written != total:
            raise RuntimeError(
                f"Truncated download for {dest}: the server declared "
                f"Content-Length={total} but {n_written} byte(s) arrived. "
                f"Retry the fetch; nothing was written to {dest}.")

        if expected_checksum:
            algo, digest = expected_checksum
            actual = _file_digests(tmp, [algo])[algo]
            if actual != digest:
                raise RuntimeError(
                    f"Checksum mismatch for {dest}: expected {algo} {digest}, "
                    f"got {actual}. The download was discarded.")
        os.replace(tmp, dest)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return dest


# ---------------------------------------------------------------------------
# Core: resolve record IDs to their latest versions
# ---------------------------------------------------------------------------
def _resolve_record_ids(info: ReleaseInfo, catalog: str,
                        failures: Optional[list] = None) -> List[int]:
    """Resolve every concept_id in a ReleaseInfo to its latest record.
    Falls back to the pinned record_id on failure.

    Each concept that could not be resolved is appended to ``failures`` when
    given: the fallback is part of what the run did, and it used to be cached
    as a successful resolution.
    """
    resolved = []
    for rid, cid in zip(info.record_ids, info.concept_ids):
        if cid is None:
            resolved.append(rid)
            continue
        try:
            latest = resolve_latest(cid)
            if latest != rid:
                print(f"  [{catalog}] concept {cid} → latest record {latest} "
                      f"(pinned was {rid})")
            resolved.append(latest)
        except Exception as e:
            warnings.warn(f"Could not resolve concept {cid} for {catalog}: {e}; "
                          f"using pinned record {rid}")
            resolved.append(rid)
            if failures is not None:
                failures.append(cid)
    return resolved


def _cache_record_resolution(
    cache_dir: Union[str, Path],
    catalog: str,
    info: ReleaseInfo,
    rids: Sequence[int],
    resolved: bool,
    dry_run: bool = False,
    failed_concept_ids: Sequence[Optional[int]] = (),
) -> None:
    """Record which Zenodo records this run actually used (GW-15).

    Which records an online run used is part of what that run did: with
    ``resolve=True`` (the default) they are whatever each concept DOI pointed
    at *at that moment*, which is not recoverable from the registry pins
    afterwards.  Caching it is what lets an offline replay read the same
    records instead of silently falling back to the pins.

    The record is what an offline ``resolve=True`` replay reads, so only a run
    that resolved writes it: a ``--no-resolve`` or explicit-``record_ids`` run
    is reproducible from the pins or its arguments and used to overwrite the
    resolved record with its own.  A dry run writes it only when no real run
    has: it downloaded nothing, so the data directory still holds what the
    last real run fetched.  A concept that failed to resolve (the run fell
    back to its pin) is recorded as such, with ``resolved_latest=False``.
    """
    key = fetch_cache.resolution_cache_key(catalog)
    if dry_run:
        try:
            previous = fetch_cache.read_metadata_cache(cache_dir, key)
        except fetch_cache.OfflineCacheMissError:
            previous = None
        if isinstance(previous, dict) and not previous.get("dry_run", False):
            return
    failed = [c for c in failed_concept_ids]
    fetch_cache.write_metadata_cache(
        cache_dir,
        key,
        {
            "catalog": catalog,
            "record_ids": [int(r) for r in rids],
            "resolved_latest": bool(resolved) and not failed,
            "failed_concept_ids": failed,
            "dry_run": bool(dry_run),
            "pinned_record_ids": [int(r) for r in info.record_ids],
            "concept_ids": list(info.concept_ids),
        },
    )


def _cached_resolved_record_ids(
    catalog: str, cache_dir: Union[str, Path]
) -> List[int]:
    """Return the record IDs an earlier online run resolved for ``catalog``.

    Offline mode used to fall back to ``info.record_ids`` -- the *pinned*
    records -- while an online run with the default ``resolve=True`` had
    followed each concept DOI to whatever was latest.  As soon as any release
    is versioned past its pin (exactly what ``resolve_latest`` exists for), the
    offline replay looked for a different ``zenodo_<rid>.json`` than the online
    run cached: either a spurious cache miss, or a silent replay of an older
    file set.  Reading the resolution back from the cache is what makes offline
    replay reproduce the online run.

    Raises :class:`gwcat.fetch_cache.OfflineCacheMissError` (naming the file)
    when the resolution was never cached or the cache is truncated -- never a
    silent fallback to the pins -- and when the record does not say
    ``resolved_latest``: a record written by a run that did not resolve, or
    whose resolution failed and fell back to the pins, is not a resolution.
    """
    key = fetch_cache.resolution_cache_key(catalog)
    payload = fetch_cache.read_metadata_cache(cache_dir, key)
    rids = payload.get("record_ids") if isinstance(payload, dict) else None
    if not rids:
        raise fetch_cache.OfflineCacheMissError(
            f"Offline mode: the cached record resolution for {catalog!r} in "
            f"{str(cache_dir)!r} records no record_ids. Re-run the fetch once "
            "online with the same cache_dir, or pass resolve=False to use the "
            "pinned record IDs deliberately."
        )
    if payload.get("resolved_latest") is not True:
        failed = payload.get("failed_concept_ids") or []
        why = (f"the online run could not resolve concept(s) {failed} and "
               f"fell back to the pinned record(s)" if failed else
               "it was written by a run that did not resolve the concept DOIs "
               "(--no-resolve or explicit record IDs)")
        raise fetch_cache.OfflineCacheMissError(
            f"Offline mode: the cached record resolution "
            f"{fetch_cache.metadata_cache_path(cache_dir, key)} for "
            f"{catalog!r} is not a resolution to the latest records: {why}. "
            "Re-run the fetch once online with resolve=True, or pass "
            "resolve=False to use the pinned record IDs deliberately."
        )
    return [int(r) for r in rids]


def _canonical_catalog(catalog: str) -> str:
    """The release name ``catalog`` is an alias of ("GWTC-4" -> "GWTC-4.1").

    An alias used to get its own download directory and cache keys, so naming
    both spellings downloaded the release twice and ingested every event of it
    twice.
    """
    manifest = getattr(RELEASES.get(catalog), "manifest", None)
    if (manifest is not None and catalog in manifest.aliases
            and manifest.release in RELEASES):
        return manifest.release
    return catalog


def _catalog_dir_name(catalog: str) -> str:
    return catalog.replace(".", "p")  # GWTC-4.1 → GWTC-4p1


def _local_copy_ok(path: str, checksum: Optional[tuple], size,
                   extra_algos=()) -> tuple:
    """``(ok, digests)`` for an existing local copy of one listed file.

    With a published checksum the copy must match it.  Without one -- on a
    release that does not require checksums -- a copy of the listed size is
    accepted unverified: downloads are staged and renamed only once complete,
    so a file at the destination is a finished download.  That used to be
    re-downloaded on every call, and refused offline.  ``digests`` holds every
    algorithm in ``extra_algos`` too, from the same read.
    """
    if not os.path.isfile(path):
        return False, {}
    algos = set(extra_algos) | ({checksum[0]} if checksum else set())
    digests = _file_digests(path, algos)
    if checksum is not None:
        return digests[checksum[0]] == checksum[1], digests
    if size is not None:
        try:
            if os.path.getsize(path) != int(size):
                return False, digests
        except (TypeError, ValueError):
            pass
    return True, digests


#: The file ``fetch_catalog`` writes in each release directory: which records
#: and files the last run used, so a directory can be told apart from the
#: superseded files a re-release leaves beside the current ones.
FETCH_MANIFEST_NAME = "gwcat_fetch_manifest.json"


def _write_fetch_manifest(dest_dir: Path, payload: dict) -> Optional[Path]:
    """Write :data:`FETCH_MANIFEST_NAME` atomically; warn (not raise) on OSError.

    A read-only data directory (a shared copy) must still verify offline.
    """
    path = Path(dest_dir) / FETCH_MANIFEST_NAME
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        os.replace(tmp, path)
    except OSError as e:
        warnings.warn(f"could not write the fetch manifest {path}: {e}")
        return None
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return path


# ---------------------------------------------------------------------------
# Public API: fetch one catalog
# ---------------------------------------------------------------------------
def fetch_catalog(
    catalog: str,
    data_dir: str = "./GWTC",
    record_ids: Optional[List[int]] = None,
    resolve: bool = True,
    show_progress: bool = True,
    dry_run: bool = False,
    cache_dir: Optional[Union[str, Path]] = None,
    offline: Optional[bool] = None,
    provenance: Optional[Dict[str, dict]] = None,
) -> List[str]:
    """Download PE files for one GWTC catalog release.

    Parameters
    ----------
    catalog : str
        Key in RELEASES: "GWTC-2.1", "GWTC-3", "GWTC-4.1" (or "GWTC-4"),
        "GWTC-5".  An alias is the release it names: same directory, same
        cache keys.
    data_dir : str
        Root directory; files go into {data_dir}/{catalog}/.
    record_ids : list of int, optional
        Override the Zenodo record ID(s).
    resolve : bool
        If True (default), query Zenodo for the latest version of each
        concept DOI.  Set False to use pinned records without network.
        Offline it selects *which records the earlier online run used*: the
        resolution is read back from ``cache_dir`` (a missing/truncated one,
        or one that records no successful resolution, raises
        :class:`~gwcat.fetch_cache.OfflineCacheMissError`), so an
        offline replay reproduces the online file set instead of silently
        falling back to the pins.  ``resolve=False`` offline is the deliberate
        opt-in to the pinned records.
    show_progress : bool
        Show tqdm progress bars during download.
    dry_run : bool
        List files that would be downloaded without actually downloading.
    cache_dir : str or Path, optional
        Passed to :func:`list_files` to cache/read the raw Zenodo file-listing
        response (see :mod:`gwcat.fetch_cache`), and used to cache/read the
        record-ID resolution itself.  ``None`` (default) disables caching --
        unchanged, byte-identical default behavior.
    offline : bool, optional
        If true (or ``GWCAT_OFFLINE`` is set), never make a network call:
        file listings come from ``cache_dir`` (required in that case) and
        every file must already exist locally under ``data_dir`` with a
        matching checksum -- a missing/mismatched local file raises a clear
        error instead of downloading it.
    provenance : dict, optional
        If given, populated in place as ``{file_name: {"record_id": str,
        "file_checksum": sha256_hex}}`` for every file this call resolves
        (downloaded or already cached on disk).  Pass the same dict on to
        ``build_store(..., file_provenance=provenance)`` to populate the
        store's per-row ``record_id`` / ``file_checksum`` meta columns.  Never
        populated automatically -- opt in only.

    A run that is not a dry run writes :data:`FETCH_MANIFEST_NAME` into the
    release directory (records, how they were chosen, and every listed file
    with its size and published checksum), and warns about files there that
    match the release but are in none of the records used -- what a
    re-release leaves behind, and what a ``gwcat ingest --glob`` over the
    directory would ingest beside the current files.  Nothing is deleted.

    Returns
    -------
    list of str
        Paths to the downloaded PE files, sorted.
    """
    if catalog not in RELEASES:
        available = sorted(k for k in RELEASES if k not in _ALIAS_NAMES)
        raise ValueError(
            f"Unknown catalog {catalog!r}. Available: {available}"
        )

    offline_mode = fetch_cache.is_offline(offline)
    if offline_mode and cache_dir is None:
        raise fetch_cache.OfflineCacheMissError(
            f"fetch_catalog({catalog!r}, offline=True) requires cache_dir "
            "pointing at a previously populated metadata cache."
        )

    requested_as = catalog
    catalog = _canonical_catalog(catalog)
    info = RELEASES[catalog]
    failed_concepts: List[Optional[int]] = []
    if offline_mode:
        # Resolving the latest version always requires a network call, so an
        # offline run reads back WHICH RECORDS THE ONLINE RUN RESOLVED (GW-15).
        # Falling back to the pinned records here -- as this did -- means an
        # offline replay looks for a different Zenodo record than the online
        # run cached the moment any release is versioned past its pin.
        # resolve=False is the deliberate opt-in to the pins.
        if record_ids:
            rids, record_source = list(record_ids), "explicit"
        elif resolve:
            rids = _cached_resolved_record_ids(catalog, cache_dir)
            record_source = "offline_replay_of_resolved"
        else:
            rids, record_source = list(info.record_ids), "pinned"
    else:
        if record_ids:
            rids, record_source = list(record_ids), "explicit"
        elif resolve:
            rids = _resolve_record_ids(info, catalog, failures=failed_concepts)
            record_source = ("resolved_latest" if not failed_concepts
                             else "resolved_with_pinned_fallback")
        else:
            rids, record_source = list(info.record_ids), "pinned"
        if cache_dir is not None and resolve and not record_ids:
            _cache_record_resolution(cache_dir, catalog, info, rids,
                                     resolved=True, dry_run=dry_run,
                                     failed_concept_ids=failed_concepts)

    dest_dir = Path(data_dir) / _catalog_dir_name(catalog)
    # Where an alias used to put the same files: reused in place when they
    # verify, rather than downloaded a second time.
    legacy_dirs = [Path(data_dir) / _catalog_dir_name(a)
                   for a in sorted({requested_as, *(
                       info.manifest.aliases if info.manifest else ())})
                   if _catalog_dir_name(a) != _catalog_dir_name(catalog)]
    all_paths = []
    listed_keys: set = set()
    manifest_files: List[dict] = []

    # Only pass the new cache/offline kwargs through when actually requested,
    # so a caller (or test) that monkeypatches list_files with the pre-PR8
    # single-argument signature ``list_files(record_id)`` keeps working
    # unchanged in the (byte-identical) default case.
    list_kwargs = {}
    if cache_dir is not None:
        list_kwargs["cache_dir"] = cache_dir
    if offline_mode:
        list_kwargs["offline"] = offline_mode

    for part_idx, rid in enumerate(rids, 1):
        part_label = f" part {part_idx}/{len(rids)}" if len(rids) > 1 else ""
        print(f"[{catalog}{part_label}] querying Zenodo record {rid} ...")
        all_files = list_files(rid, **list_kwargs)
        pe_files = [f for f in all_files if info.file_filter(f["key"])]
        rejected = [f["key"] for f in all_files if not info.file_filter(f["key"])]

        if not pe_files:
            raise RuntimeError(
                f"No PE files matched filter for {catalog} in record {rid}. "
                f"Total files: {len(all_files)}, rejected: {rejected[:5]}. "
                "The file naming convention may have changed."
            )

        total_bytes = sum(f.get("size", 0) for f in pe_files)
        print(f"[{catalog}{part_label}] {len(pe_files)} PE files, "
              f"{total_bytes / 1e9:.1f} GB  (rejected {len(rejected)} non-PE files)")

        if dry_run:
            for f in pe_files:
                sz = f.get("size", 0) / 1e6
                print(f"  + {f['key']}  ({sz:.1f} MB)")
            for fn in rejected:
                print(f"  - {fn}  (skipped)")
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(pe_files, 1):
            fname = f["key"]
            dest = _contained_dest(dest_dir, fname)
            url = _download_url_for(f)
            checksum = _parse_checksum(f)
            if checksum is None and _require_checksums(info):
                # The manifests declare `validation.require_checksums: true`,
                # the loader schema-checks it, and nothing consulted it (GW-37)
                # -- so a Zenodo entry that shipped no checksum silently
                # downgraded to zero verification on a release that demands it.
                raise RuntimeError(
                    f"{catalog}: record {rid} lists {fname!r} with no checksum, "
                    f"but this release's manifest declares "
                    f"validation.require_checksums: true. An unverified file "
                    f"cannot be distinguished from a corrupted one; refusing to "
                    f"download it.")
            extra = ("sha256",) if provenance is not None else ()
            cached_ok, digests = _local_copy_ok(dest, checksum, f.get("size"),
                                                extra)
            where = ""
            if not cached_ok:
                for legacy in legacy_dirs:
                    alt = _contained_dest(legacy, fname)
                    ok, alt_digests = _local_copy_ok(alt, checksum,
                                                     f.get("size"), extra)
                    if ok:
                        dest, cached_ok, digests = alt, True, alt_digests
                        where = f", under {legacy.name}/"
                        break
            if cached_ok:
                note = "" if checksum else ", unverified: no checksum published"
                print(f"  [{i}/{len(pe_files)}] {fname} (cached{where}{note})")
            elif offline_mode:
                raise fetch_cache.OfflineCacheMissError(
                    f"Offline mode: {dest} is missing or does not match the "
                    f"expected checksum, and network downloads are disabled. "
                    "Populate data_dir by running once online, or pass "
                    "offline=False."
                )
            else:
                print(f"  [{i}/{len(pe_files)}] {fname}")
                download_file(url, dest, expected_checksum=checksum,
                              show_progress=show_progress)
                digests = {}
            all_paths.append(dest)
            listed_keys.add(fname)
            entry = {"key": fname, "record_id": int(rid),
                     "size": f.get("size"), "checksum": f.get("checksum") or None,
                     "verified": checksum is not None, "path": dest}
            if provenance is not None:
                sha256 = (digests.get("sha256")
                          or _file_digests(dest, ["sha256"])["sha256"])
                provenance[fname] = {
                    "record_id": str(rid),
                    "file_checksum": sha256,
                }
                entry["sha256"] = sha256
            manifest_files.append(entry)

    if not dry_run:
        superseded = sorted(
            p.name for p in dest_dir.iterdir()
            if p.name not in listed_keys and p.is_file()
            and info.file_filter(p.name)) if dest_dir.is_dir() else []
        if superseded:
            shown = ", ".join(superseded[:5]) + (
                f", ... ({len(superseded) - 5} more)" if len(superseded) > 5
                else "")
            warnings.warn(
                f"{catalog}: {len(superseded)} file(s) in {dest_dir} match "
                f"this release but are in none of the record(s) {list(rids)} "
                f"just used -- left by an earlier version of the release? A "
                f"`gwcat ingest --glob` over this directory would ingest them "
                f"beside the current files. They are listed as "
                f"superseded_files in {FETCH_MANIFEST_NAME}: {shown}")
        _write_fetch_manifest(dest_dir, {
            "catalog": catalog,
            "requested_as": requested_as,
            "record_ids": [int(r) for r in rids],
            "record_source": record_source,
            "failed_concept_ids": failed_concepts,
            "pinned_record_ids": [int(r) for r in info.record_ids],
            "concept_ids": list(info.concept_ids),
            "offline": bool(offline_mode),
            "written_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                            time.gmtime()),
            "files": manifest_files,
            "superseded_files": superseded,
        })
        print(f"[{catalog}] done: {len(all_paths)} files in {dest_dir}")
    return sorted(all_paths)


def is_injection_catalog(catalog: str) -> bool:
    """True when ``catalog`` names an injection/selection-function release.

    Membership in :data:`INJECTION_RELEASES` -- which is built from the
    injection manifests and registers their aliases too -- not a
    ``startswith("injections")`` test on the name (GW-15).  Injection files are
    not PESummary per-event files, so handing them to ``build_store`` crashes
    in ``_read_event_pesummary``; a name-shaped guard silently stops holding
    the moment a manifest declares a release (or alias) that does not happen to
    start with "injections".
    """
    if catalog in INJECTION_RELEASES:
        return True
    # Defence in depth: a key registered in RELEASES that *is* one of the
    # injection ReleaseInfo objects (e.g. via some future aliasing path).
    info = RELEASES.get(catalog)
    return info is not None and any(info is i for i in INJECTION_RELEASES.values())


def _unique_releases(catalogs: Sequence[str]) -> List[str]:
    """``catalogs`` with aliases mapped to their release and repeats dropped.

    Naming ``GWTC-4`` and ``GWTC-4.1`` together fetched the release twice and
    handed every O4a file to ``build_store`` twice.  Unknown names pass
    through unchanged, for the caller's own error.
    """
    out: List[str] = []
    for c in catalogs:
        canonical = _canonical_catalog(c) if c in RELEASES else c
        if canonical not in out:
            out.append(canonical)
    return out


def split_pe_and_injection_catalogs(
    catalogs: Sequence[str],
) -> tuple[List[str], List[str]]:
    """Split requested catalog names into (PE releases, injection releases).

    One shared classifier for every caller that must not feed injection files
    to ``build_store``: the library entry point (:func:`fetch_and_build`) and
    the CLI used to carry two different, drifting guards.
    """
    pe = [c for c in catalogs if not is_injection_catalog(c)]
    injections = [c for c in catalogs if is_injection_catalog(c)]
    return pe, injections


# ---------------------------------------------------------------------------
# Public API: fetch + build in one shot
# ---------------------------------------------------------------------------
def fetch_and_build(
    catalogs: Sequence[str] = ("GWTC-2.1", "GWTC-3", "GWTC-4.1", "GWTC-5"),
    data_dir: str = "./GWTC",
    out: str = "store.h5",
    event_table: Optional[dict] = None,
    resolve: bool = True,
    show_progress: bool = True,
    ingest_cfg=None,
    extra_params: Optional[list] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    offline: Optional[bool] = None,
    provenance: Optional[Dict[str, dict]] = None,
) -> str:
    """Fetch PE files from Zenodo and build the gwcat store.

    Parameters
    ----------
    catalogs : sequence of str
        Which catalogs to include.
    data_dir, out, event_table, resolve, show_progress :
        See fetch_catalog and build_store.
    ingest_cfg : IngestConfig, optional
    extra_params : list, optional
    cache_dir, offline : optional
        Forwarded to :func:`fetch_catalog` (file listings) and to
        ``build_store``'s ``event_table`` auto-fetch (GWOSC).  ``None``/unset
        (the defaults) leave behavior unchanged from before PR 8.
    provenance : dict, optional
        If given, populated in place across all fetched catalogs as
        ``{file_name: {"record_id", "file_checksum"}}`` and forwarded to
        ``build_store(file_provenance=...)``.  Not populated unless passed in
        (opt-in; avoids hashing every downloaded file by default).

    Returns
    -------
    str : path to the output store file.
    """
    from .ingest import build_store, IngestConfig

    # Injection releases are in RELEASES so fetch_catalog can find them, but
    # their files are search-sensitivity sets, not PESummary per-event files:
    # build_store would hand them to _read_event_pesummary and crash after
    # downloading tens of GB.  Reject by registry membership (GW-15) -- a
    # public path such as fetch_and_build(list_releases()) walks straight into
    # it, and a name-prefix guard would stop holding on the first injection
    # manifest not named "injections-*".
    _, injections = split_pe_and_injection_catalogs(catalogs)
    if injections:
        raise ValueError(
            f"fetch_and_build builds a PE store, but {sorted(injections)} "
            f"{'are' if len(injections) > 1 else 'is'} injection/selection "
            "release(s) whose files are not per-event PE files. Fetch them "
            "separately with fetch_catalog(), and build the selection function "
            "from them with gwcat.selection."
        )

    cfg = ingest_cfg or IngestConfig()
    all_paths = []
    for cat in _unique_releases(catalogs):
        paths = fetch_catalog(cat, data_dir=data_dir, resolve=resolve,
                              show_progress=show_progress, cache_dir=cache_dir,
                              offline=offline, provenance=provenance)
        all_paths.extend(paths)

    if not all_paths:
        raise RuntimeError("No PE files downloaded; cannot build store.")

    print(f"\n--- Building store from {len(all_paths)} files ---")
    build_store(all_paths, out, cfg=cfg, event_table=event_table,
                extra_params=extra_params, cache_dir=cache_dir, offline=offline,
                file_provenance=provenance)
    return out


# ---------------------------------------------------------------------------
# EVENT-METADATA DISCOVERY (GWOSC) — separate from Zenodo file discovery
# above: nothing in this section touches Zenodo, and nothing above touches
# GWOSC.  See the module docstring and gwcat.event_metadata for how callers
# combine this raw metadata with manifest defaults / user overrides.
# ---------------------------------------------------------------------------
#: How many BBH names the LVK-only GWOSC query is expected to return.
#:
#: This used to be 259 -- the size of the curated GWTC-5 BBH *population*
#: sample -- and it silently passed only because the query was contaminated
#: (GW-15).  Unfiltered, the live query returns 286 names across eight
#: catalogs, 41 of them third-party IAS-O3a entries that supersede the LVK
#: version of the same event under ``lastver=true``.
#:
#: It was then 244, measured with the LVK-catalog filter still applied to a
#: ``lastver=true`` response -- and that number was itself short: under
#: ``lastver`` GWOSC returns ONE version per event, which for ~30 O3a events is
#: the IAS-O3a one, so dropping non-LVK entries dropped those genuine LVK
#: events with them (GW190408_181802, GW190412_053044, ...), and a guard
#: baselined on the loss could never report it.  The query now asks for every
#: version and keeps the highest LVK version of each event.
#:
#: 272 is the same selection measured offline on the GWOSC cumulative GWTC
#: event table -- the latest LVK version of every event that LVK's latest
#: release still lists as confident -- as cached on 2026-09-24: GWTC-5.0 104,
#: GWTC-4.1 86, GWTC-2.1-confident 50, GWTC-3-confident 32 (the O1/O2 events
#: counted under GWOSC's date-only names).  The table leaves out the events
#: LVK demoted in a later release (GW190424_180648 and GW190909_114149), and
#: so does the query, which lets a GWTC release's non-confident list take part
#: in the version ordering; one live run that ignored those lists returned
#: these two on top of 272.  It has not been re-measured against the live
#: event-versions endpoint since.  The guard is two-sided against this
#: number: *fewer* means an incomplete live index, *more* means entries the
#: LVK-catalog filter did not expect to admit.  It is a warning, not an
#: assertion: it exists to be noticed and re-baselined by whoever re-runs the
#: query.
_GWOSC_BBH_EXPECTED_NAMES = 272

#: Cache key stem of the all-versions BBH query.  Deliberately not the stem
#: the ``lastver=true`` query cached under (``bbh_names_m2min_<m2>``): a cache
#: written by that query holds one version per event, so replaying it through
#: the all-versions logic would silently reproduce the short list.  An old
#: cache is therefore a cache miss, named as one.
_GWOSC_BBH_CACHE_STEM = "bbh_names_allversions_m2min"

#: GWOSC catalog labels the BBH query accepts: the LVK GWTC releases, in any of
#: the spellings GWOSC uses ("GWTC-2", "GWTC-2.1-confident", "GWTC-4.1",
#: "GWTC-5.0").  Deliberately a *pattern* and not a hardcoded list, so a future
#: "GWTC-6.0" is admitted automatically while third-party catalogs (IAS-O3a,
#: OGC-*), marginal-candidate catalogs (O3_IMBH_marginal, GWTC-*-marginal) and
#: discovery-paper collections (O4_Discovery_Papers) stay out.
_GWOSC_LVK_CATALOG_RE = re.compile(r"GWTC-\d+(?:\.\d+)?(?:-CONFIDENT)?")

#: The LVK GWTC releases' *non-confident* lists ("GWTC-2.1-marginal",
#: "GWTC-3-marginal", "GWTC-2.1-auxiliary", ...): any GWTC release label with
#: a qualifier other than "confident".  They never admit an event, but they do
#: take part in the version ordering.  An event LVK demoted in a later release
#: (GW190424_180648 and GW190909_114149 were confident in GWTC-2 and are not in
#: GWTC-2.1-confident) has its newest LVK version in one of these lists; with
#: them ignored, the older confident version decided and the event came back.
_GWOSC_LVK_NONCONFIDENT_CATALOG_RE = re.compile(
    r"GWTC-\d+(?:\.\d+)?-(?!CONFIDENT$)[A-Z0-9_.-]+")

#: A GWOSC name is only usable as an LVK event name in this package's canonical
#: form (``GW230529`` / ``GW230529_181500``).  Anything else -- e.g. the IMBH
#: marginal candidate ``200114_020818`` -- is not a GWTC event name and must
#: never enter the whitelist.
_GWOSC_EVENT_NAME_RE = re.compile(r"GW\d{6}(?:_\d{6})?$")


def _is_lvk_gwtc_catalog(label: object) -> bool:
    """True when a GWOSC catalog label identifies an LVK GWTC release."""
    if label is None:
        return False
    normalized = re.sub(r"\s+", "", str(label)).upper()
    return bool(_GWOSC_LVK_CATALOG_RE.fullmatch(normalized))


def _is_lvk_gwtc_nonconfident_catalog(label: object) -> bool:
    """True when a GWOSC catalog label is a GWTC release's non-confident list."""
    if label is None:
        return False
    normalized = re.sub(r"\s+", "", str(label)).upper()
    return bool(_GWOSC_LVK_NONCONFIDENT_CATALOG_RE.fullmatch(normalized))


def _gwosc_event_catalog(event: dict) -> Optional[str]:
    """Return the catalog label of one GWOSC event-versions entry, if any.

    GWOSC has spelled this field several ways across API versions; a dict-valued
    catalog (``{"name": ...}``) is unwrapped.  ``None`` means the entry carries
    no catalog information at all, which is treated as "not an LVK entry" --
    and, if *no* entry in a whole response carries it, as an API shape change
    worth failing on rather than silently returning an empty whitelist.
    """
    for key in ("catalog", "catalog_shortName", "catalogShortName",
                "catalog_name", "release"):
        value = event.get(key)
        if isinstance(value, dict):
            value = (value.get("shortName") or value.get("short_name")
                     or value.get("name"))
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


@lru_cache(maxsize=1)
def _curated_bbh_names() -> frozenset:
    """The curated BBH population whitelist, as a set (cached)."""
    from .bbh_allowed_names import BBH_ALL

    return frozenset(BBH_ALL)


@lru_cache(maxsize=1)
def _non_bbh_name_stems() -> frozenset:
    """Date stems (``GW190814``) of every curated non-BBH exclusion.

    Derived from :data:`gwcat.bbh_allowed_names.NON_BBH_EXCLUSIONS` -- the one
    curated source, kept as a bundled data file -- instead of the second
    hand-copied set that used to live here.  The two lists had already drifted:
    this one spelled the lower-mass-gap system ``GW190814`` and the curated one
    ``GW190814_211039``, so the third spelling GWOSC also carries for it,
    ``GW190814_192009``, matched neither and entered the BBH whitelist.
    Matching on the stem catches every spelling of the same event.
    """
    from .bbh_allowed_names import NON_BBH_EXCLUSIONS

    return frozenset(_gwosc_name_stem(name) for name in NON_BBH_EXCLUSIONS)


def _gwosc_name_stem(name: str) -> str:
    """``GW190814_192009`` -> ``GW190814`` (the GPS-time suffix removed)."""
    return re.sub(r"_\d{6}$", "", str(name))


def _is_non_bbh_name(name: str) -> bool:
    """True when ``name`` is any spelling of a curated non-BBH event.

    Exact membership first, then the date stem -- but never for a name the
    curated BBH population list itself contains, so a genuine second event on
    the same day as an excluded one (GW190828_063405 / GW190828_065509 is the
    shape of it) can never be dropped by the stem rule.
    """
    from .bbh_allowed_names import NON_BBH_EXCLUSIONS

    if name in NON_BBH_EXCLUSIONS:
        return True
    if name in _curated_bbh_names():
        return False
    return _gwosc_name_stem(name) in _non_bbh_name_stems()


def _clean_gwosc_event_name(name: object) -> str:
    """Return a GWOSC event name without an API version suffix."""
    return re.sub(r"-v\d+$", "", str(name or ""))


def _gwosc_event_version(event: dict) -> Optional[int]:
    """The version number of one GWOSC event-versions entry, if it says.

    Read from the entry's ``version`` field, else from a ``-vN`` suffix on its
    name.  ``None`` when neither is present.
    """
    value = event.get("version")
    if value is not None and not isinstance(value, bool):
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    for key in ("name", "shortName"):
        match = re.search(r"-v(\d+)$", str(event.get(key) or ""))
        if match:
            return int(match.group(1))
    return None


def _gwosc_json(url: str, timeout: int) -> dict:
    """Fetch one GWOSC JSON page."""
    req = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "gwcat-fetch/0.1 (+https://github.com/ignaciomagana/gwcat)",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _gwosc_best_parameter(parameters: object, *names: str) -> Optional[float]:
    """Extract a parameter's best value from GWOSC parameter objects."""
    wanted = set(names)
    if not isinstance(parameters, list):
        return None
    for param in parameters:
        if not isinstance(param, dict) or param.get("name") not in wanted:
            continue
        value = param.get("best")
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _cached_pages(cached, cache_dir, key: str) -> list:
    """The ``pages`` list of a cached paginated GWOSC response.

    Both GWOSC readers indexed ``cached["pages"]`` unguarded, so a payload
    without one -- not written by this module, or written by a version that
    stored something else -- surfaced as a raw ``KeyError``/``TypeError`` out
    of the offline path.  It is a cache miss, named as one, the same contract
    :func:`gwcat.fetch_cache.read_metadata_cache` applies to a truncated file.
    """
    pages = cached.get("pages") if isinstance(cached, dict) else None
    if not isinstance(pages, list) or not all(isinstance(p, dict)
                                              for p in pages):
        raise fetch_cache.OfflineCacheMissError(
            f"Offline mode: the cached GWOSC response at "
            f"{fetch_cache.metadata_cache_path(cache_dir, key)} (key={key!r}) "
            "records no list of response pages. Delete it and re-run the "
            f"equivalent fetch once online with cache_dir={str(cache_dir)!r}."
        )
    return pages


def fetch_bbh_names_gwosc(
    m2_min: float = 3.0,
    verbose: bool = True,
    timeout: int = 30,
    cache_dir: Optional[Union[str, Path]] = None,
    offline: Optional[bool] = None,
) -> list[str]:
    """Return BBH event names from the paginated GWOSC v2 event API.

    The GWOSC endpoint is queried for **every version** of every event, with
    default parameters included.  Of each event's versions only those from an
    **LVK GWTC catalog** count, and the highest of them decides: the event is
    kept when that version is in a confident list, the event carries a
    canonical GWTC event name, is not a curated non-BBH event under any
    spelling, and that version's default PE parameters contain
    ``mass_2_source`` (or the legacy alias ``m2_source``) above ``m2_min``.
    A GWTC release's non-confident list (``GWTC-2.1-marginal``,
    ``GWTC-3-marginal``, ...) takes part in that ordering but never admits an
    event: when an event's highest LVK version is in one, LVK demoted it in
    that release (GW190424_180648 and GW190909_114149, confident in GWTC-2, are
    not in GWTC-2.1-confident) and it is dropped.

    The catalog restriction is the point (GW-15).  GWOSC indexes third-party
    catalogs beside GWTC, and under ``lastver=true`` a non-LVK version
    *supersedes* the LVK one for the same event: the unfiltered query returns
    286 names across eight catalogs, and 41 O3a events were being admitted on
    the IAS pipeline's ``mass_2_source`` rather than on the LVK PE this package
    actually ingests.  Filtering is done client-side on each entry's catalog
    label rather than by adding a ``catalog=`` query parameter, so a change in
    what the server accepts can only ever make this stricter, never silently
    return an unfiltered list.

    Neither ``lastver`` nor ``min-mass-2-source`` is sent.  ``lastver=true``
    returns one version per event, so for the ~30 O3a events whose latest
    version is the IAS one, dropping non-LVK entries dropped the event and its
    LVK version was never seen.  A server-side mass cut is applied to each
    version on its own, so it could hide an event's current LVK version while
    returning an older one; the cut is applied here, to the version that
    decides.

    cache_dir / offline : see :mod:`gwcat.fetch_cache`.  ``None``/unset (the
    defaults) disable caching/offline-mode entirely -- unchanged, byte-identical
    default behavior.  When caching, every raw page of the paginated response is
    written under one cache key so an offline replay reconstructs the exact same
    ``names`` set via the same per-event filter logic.  A cache written by the
    earlier ``lastver=true`` query is not read (it lives under another key).
    """
    offline_mode = fetch_cache.is_offline(offline)
    key = fetch_cache.gwosc_cache_key(f"{_GWOSC_BBH_CACHE_STEM}_{m2_min}")

    if offline_mode:
        if cache_dir is None:
            raise fetch_cache.OfflineCacheMissError(
                "fetch_bbh_names_gwosc(offline=True) requires cache_dir to "
                "locate the previously cached GWOSC response."
            )
        cached = fetch_cache.read_metadata_cache(cache_dir, key)
        pages = _cached_pages(cached, cache_dir, key)
    else:
        query = urlencode(
            {
                "include-default-parameters": "true",
                "pagesize": 100,
            }
        )
        url = f"https://gwosc.org/api/v2/event-versions?{query}"
        pages = []
        page = 0
        seen_urls: set[str] = set()

        while url:
            if url in seen_urls:
                raise RuntimeError(f"GWOSC pagination loop detected at {url}")
            seen_urls.add(url)
            page += 1
            if verbose:
                print(f"fetch_bbh_names_gwosc: fetching page {page}: {url}")

            data = _gwosc_json(url, timeout=timeout)
            pages.append(data)
            url = data.get("next")

        if cache_dir is not None:
            fetch_cache.write_metadata_cache(
                cache_dir, key,
                {"m2_min": m2_min, "query": "all versions, no server-side cut",
                 "pages": pages})

    n_entries = 0
    n_with_catalog = 0
    dropped_catalogs: Dict[str, int] = {}
    dropped_names: set[str] = set()
    # name -> (version, confident, m2_source, catalog) of the highest LVK
    # version seen so far.  A non-confident list's version takes part in the
    # ordering, so an event LVK demoted in a later release is decided by that
    # release and dropped, not by the older confident version.
    lvk_versions: Dict[str, tuple] = {}
    for data in pages:
        for event in data.get("results", []):
            if not isinstance(event, dict):
                continue
            n_entries += 1

            catalog = _gwosc_event_catalog(event)
            if catalog is not None:
                n_with_catalog += 1
            confident = _is_lvk_gwtc_catalog(catalog)
            if not confident and not _is_lvk_gwtc_nonconfident_catalog(catalog):
                label = catalog or "<no catalog field>"
                dropped_catalogs[label] = dropped_catalogs.get(label, 0) + 1
                continue

            name = _clean_gwosc_event_name(
                event.get("name") or event.get("shortName") or event.get("grace_id")
            )
            if not name or not _GWOSC_EVENT_NAME_RE.fullmatch(name):
                if name:
                    dropped_names.add(name)
                continue

            version = _gwosc_event_version(event)
            params = event.get("default_parameters")
            m2_source = (_gwosc_best_parameter(params, "mass_2_source",
                                               "m2_source")
                         if confident else None)
            entry = (version, confident, m2_source, catalog)
            prev = lvk_versions.get(name)
            if prev is not None:
                if prev[0] is None or version is None or prev[0] == version:
                    if prev[:3] == entry[:3]:
                        continue            # the same entry twice
                    # Two LVK entries for one event that cannot be ordered: no
                    # rule says which one is current, and they disagree.
                    raise RuntimeError(
                        f"GWOSC returned two LVK entries for {name} that "
                        f"cannot be ordered (versions {prev[0]!r} and "
                        f"{version!r}, catalogs {prev[3]!r} and {catalog!r}, "
                        f"mass_2_source {prev[2]!r} and {m2_source!r}), so "
                        "which one is current is unknown. The event-versions "
                        "API shape may have changed; update "
                        "gwcat.fetch._gwosc_event_version before trusting this "
                        "list.")
                if version < prev[0]:
                    continue
            lvk_versions[name] = entry

    names: set[str] = set()
    demoted: Dict[str, str] = {}
    for name, (_version, confident, m2_source, catalog) in lvk_versions.items():
        if not confident:
            demoted[name] = catalog
            continue
        if _is_non_bbh_name(name):
            continue
        if m2_source is None or m2_source <= m2_min:
            continue
        names.add(name)

    if n_entries and not n_with_catalog:
        # Every entry lacking a catalog label means the response shape changed,
        # not that GWOSC published nothing from GWTC.  Failing here is the only
        # honest outcome: silently returning an empty whitelist would look like
        # a legitimately empty query.
        raise RuntimeError(
            f"GWOSC returned {n_entries} event-version entries, none of which "
            "carries a catalog label, so the LVK-catalog restriction cannot be "
            "applied. The event-versions API shape has changed; update "
            "gwcat.fetch._gwosc_event_catalog before trusting this list."
        )

    result = sorted(names)
    if verbose:
        print(f"fetch_bbh_names_gwosc: selected {len(result)} BBH candidates "
              f"from {n_entries} GWOSC entries")
        if dropped_catalogs:
            summary = ", ".join(
                f"{label}: {count}"
                for label, count in sorted(dropped_catalogs.items())
            )
            print(f"fetch_bbh_names_gwosc: dropped non-LVK catalogs ({summary})")
        if dropped_names:
            print("fetch_bbh_names_gwosc: dropped non-GWTC event names "
                  f"({sorted(dropped_names)})")
        if demoted:
            listed = ", ".join(f"{n} [{c}]" for n, c in sorted(demoted.items()))
            print(f"fetch_bbh_names_gwosc: dropped {len(demoted)} event(s) "
                  "whose latest LVK version is in a non-confident list "
                  f"({listed})")
    if len(result) != _GWOSC_BBH_EXPECTED_NAMES:
        direction = ("only " if len(result) < _GWOSC_BBH_EXPECTED_NAMES
                     else "as many as ")
        cause = ("Callers may be relying on an incomplete live GWOSC index."
                 if len(result) < _GWOSC_BBH_EXPECTED_NAMES else
                 "More names than the LVK GWTC catalogs were expected to "
                 "supply -- check for catalog contamination (see GW-15) or "
                 "re-baseline against a new observing run.")
        warnings.warn(
            f"GWOSC returned {direction}"
            f"{len(result)} LVK BBH names with PE mass_2_source > {m2_min}; "
            f"{_GWOSC_BBH_EXPECTED_NAMES} were expected. {cause}",
            RuntimeWarning,
            stacklevel=2,
        )
    return result


def _coerce_float(value):
    """A finite float from a GWOSC field, else NaN (never fabricated)."""
    if value is None:
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _parse_gwosc_event_table_page(data: dict, table: dict) -> None:
    """Merge one raw GWOSC event-API page's events into ``table`` in place.

    Factored out so the live-fetch and offline-cache-replay code paths in
    :func:`fetch_event_table_gwosc` share identical parsing logic -- caching
    can never silently drift from what a live call would have computed.
    """
    events = data.get("events", {})
    for name, info in events.items():
        clean = re.sub(r"-v\d+$", "", name)
        if not isinstance(info, dict):
            continue
        # GWOSC returns `far` and `p_astro` as TOP-LEVEL fields of each event
        # (GW-15).  The previous code read them out of an
        # ``info["parameters"][<pipeline>]`` sub-dict that the event API does not
        # return at all, so ``info.get("parameters", {})`` was always ``{}`` and
        # EVERY event was stored with far = p_astro = NaN.  Verified live: the
        # response has no "parameters" key and does carry e.g.
        # far=140.0, p_astro=0.61501 at the top level.
        far = _coerce_float(info.get("far"))
        # Either spelling of the one quantity (GW-14): the live API says
        # `p_astro`, a recorded/older payload may say `pastro`, and the table
        # this builds is keyed `pastro` for every downstream reader.
        pastro = resolve_pastro(info)
        # Keep the legacy sub-dict as a fallback so a future API shape (or a
        # recorded old payload) still parses, but never let it override a real
        # top-level value.
        params = info.get("parameters")
        if isinstance(params, dict):
            for _key, pset in params.items():
                if not isinstance(pset, dict):
                    continue
                if not np.isfinite(far):
                    far = _coerce_float(pset.get("far"))
                if not np.isfinite(pastro):
                    pastro = resolve_pastro(pset)
        # A later page for the same event must not clobber a finite value with
        # a NaN (the cumulative and per-catalog endpoints overlap).
        prev = table.get(clean)
        if prev is not None:
            if not np.isfinite(far):
                far = prev.get("far", float("nan"))
            if not np.isfinite(pastro):
                pastro = resolve_pastro(prev)
        table[clean] = {"far": far, "pastro": pastro}


def _add_historical_name_aliases(table: dict) -> dict:
    """Index each date-only GWOSC entry under its full GWTC name too.

    GWOSC keys the O1/O2 events by their GWTC-1 names (``GW150914``), while
    every store row carries the timestamped name of its GWTC-2.1 release file
    (``GW150914_095045``), so those ten rows never matched the table and read
    ``metadata_source='pe_file_only'``: a FAR or p_astro GWOSC published for
    them would have been dropped without a word.  The mapping is the fixed
    table of historical names
    (:data:`gwcat.population_samples.HISTORICAL_DATE_ONLY_NAMES`), and an entry
    GWOSC publishes under the full name itself is never overwritten.
    """
    from .population_samples import HISTORICAL_DATE_ONLY_NAMES
    for short, full in HISTORICAL_DATE_ONLY_NAMES.items():
        if short in table and full not in table:
            table[full] = dict(table[short])
    return table


def fetch_event_table_gwosc(
    catalog_tag: str = "GWTC",
    timeout: int = 30,
    cache_dir: Optional[Union[str, Path]] = None,
    offline: Optional[bool] = None,
) -> dict:
    """Fetch FAR and p_astro from the GWOSC event API.

    Returns {event_name: {'far': float, 'pastro': float}}.  FAR/p_astro are
    genuinely absent from some public GWOSC entries; missing values come back
    as NaN (never fabricated), which is what lets
    ``gwcat.ingest.build_store`` record ``far_available=False`` explicitly.
    An O1/O2 event GWOSC lists under its date-only name is also indexed under
    its full name (see :func:`_add_historical_name_aliases`).

    catalog_tag : "GWTC" (cumulative), "GWTC-2.1-confident", etc.
    cache_dir / offline : see :mod:`gwcat.fetch_cache`.  ``None``/unset (the
    defaults) disable caching/offline-mode entirely -- unchanged, byte-identical
    default behavior.
    """
    offline_mode = fetch_cache.is_offline(offline)
    key = fetch_cache.gwosc_cache_key(f"event_table_{catalog_tag}")

    if offline_mode:
        if cache_dir is None:
            raise fetch_cache.OfflineCacheMissError(
                "fetch_event_table_gwosc(offline=True) requires cache_dir to "
                "locate the previously cached GWOSC response."
            )
        cached = fetch_cache.read_metadata_cache(cache_dir, key)
        table: dict = {}
        for page in _cached_pages(cached, cache_dir, key):
            _parse_gwosc_event_table_page(page, table)
        return _add_historical_name_aliases(table)

    base = f"https://gwosc.org/eventapi/json/{catalog_tag}/"
    table = {}
    pages = []
    url = base

    while url:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        pages.append(data)
        _parse_gwosc_event_table_page(data, table)
        url = data.get("links", {}).get("next")

    if cache_dir is not None:
        fetch_cache.write_metadata_cache(
            cache_dir, key, {"catalog_tag": catalog_tag, "pages": pages})

    return _add_historical_name_aliases(table)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_AVAILABLE = sorted(k for k in RELEASES if k not in _ALIAS_NAMES)  # hide aliases
_PE_CATALOGS = list_release_manifests()

def _cli(
    argv=None,
    _deprecated: bool = True,
    default_write_summary: bool = False,
    prog: Optional[str] = None,
):
    """Fetch CLI. Also the implementation behind the unified ``gwcat fetch``
    subcommand (PR 10), which calls this with ``_deprecated=False,
    default_write_summary=True`` so every flag defined here is automatically
    available under both surfaces.

    argv : list of str, optional
        Parsed instead of ``sys.argv[1:]`` when given.
    _deprecated : bool
        When True (the default, used by the standalone ``gwcat-fetch``
        console script), print a one-line pointer to ``gwcat fetch`` on
        stderr before continuing with unchanged behavior.
    default_write_summary : bool
        Whether a ``--out`` build gets a validation summary by default
        (``--no-summary`` always disables it). False for the deprecated
        standalone script; the unified CLI passes True.
    prog : str, optional
        Program identity shown by argparse.  The standalone entry point keeps
        ``gwcat-fetch``; the unified dispatcher supplies ``gwcat fetch`` (or
        the name of a future replacement entry point).
    """
    import argparse
    if _deprecated:
        print("gwcat-fetch is deprecated; use `gwcat fetch` instead "
              "(same options; see `gwcat fetch --help`).", file=sys.stderr)

    ap = argparse.ArgumentParser(
        prog=prog or "gwcat-fetch",
        description="Download GWTC PE samples and injection sets from Zenodo, "
                    "and optionally build the gwcat store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available catalogs: {', '.join(_AVAILABLE)}\n"
               f"'all' expands to all of them.\n"
               f"'pe' expands to PE catalogs only ({', '.join(_PE_CATALOGS)}).\n\n"
               "By default, each catalog's Zenodo concept DOI is resolved to\n"
               "the latest version.  Use --no-resolve to skip this and use\n"
               "the pinned record IDs.",
    )
    ap.add_argument(
        "--catalog", nargs="+",
        default=_PE_CATALOGS,
        metavar="NAME",
        help="Catalogs to download.  Default: PE catalogs only.  "
             "Use 'all' for PE + injections, or name specific ones.",
    )
    ap.add_argument("--data-dir", default="./GWTC",
                    help="Root directory for downloaded files (default: ./GWTC)")
    ap.add_argument("--out", default=None, metavar="STORE.h5",
                    help="Build the store after download.  Omit to download only.")
    ap.add_argument("--no-resolve", action="store_true",
                    help="Use pinned record IDs instead of resolving latest.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List files without downloading.")
    ap.add_argument("--no-event-table", action="store_true",
                    help="Skip auto-fetching FAR/p_astro from GWOSC during build.")
    ap.add_argument("--no-progress", action="store_true",
                    help="Disable progress bars.")
    ap.add_argument("--record-ids", type=int, nargs="+", default=None,
                    help="Override Zenodo record ID(s) (only with a single --catalog).")
    ap.add_argument("--cache-dir", default=None, metavar="DIR",
                    help="Cache raw Zenodo/GWOSC metadata responses under DIR "
                         "(see gwcat.fetch_cache). Omit to disable caching.")
    ap.add_argument("--offline", action="store_true",
                    help="Never touch the network: read file listings and "
                         "event metadata from --cache-dir (required), and "
                         "require every file to already exist locally. Same "
                         "as setting GWCAT_OFFLINE=1.")
    ap.add_argument("--no-summary", action="store_true",
                    help="Skip writing validation_summary.json/.md next to "
                         "--out.")

    args = ap.parse_args(argv)

    catalogs = args.catalog
    if catalogs == ["all"] or catalogs == "all":
        catalogs = list(_AVAILABLE)
    elif catalogs == ["pe"] or catalogs == "pe":
        catalogs = list(_PE_CATALOGS)

    for c in catalogs:
        if c not in RELEASES:
            ap.error(f"Unknown catalog {c!r}. Available: {_AVAILABLE}")
    catalogs = _unique_releases(catalogs)
    if args.record_ids and len(catalogs) != 1:
        ap.error("--record-ids requires exactly one --catalog")

    show_progress = not args.no_progress
    resolve = not args.no_resolve
    # None (not False) when --offline is absent, so GWCAT_OFFLINE can still
    # activate offline mode; the flag only ever turns it on explicitly.
    offline = True if args.offline else None

    all_paths = []
    pe_paths = []          # only PE files go to build_store
    # Which Zenodo record and which bytes each file is: with --out it goes to
    # build_store, whose per-row record_id / file_checksum columns no CLI path
    # used to fill.  (Every run also writes the per-release fetch manifest.)
    provenance = {} if (args.out and not args.dry_run) else None
    for cat in catalogs:
        rids = args.record_ids if (len(catalogs) == 1 and args.record_ids) else None
        paths = fetch_catalog(
            cat, data_dir=args.data_dir, record_ids=rids,
            resolve=resolve, show_progress=show_progress,
            dry_run=args.dry_run, cache_dir=args.cache_dir,
            offline=offline,
            provenance=None if is_injection_catalog(cat) else provenance,
        )
        all_paths.extend(paths)
        if not is_injection_catalog(cat):   # membership, not a name prefix
            pe_paths.extend(paths)

    if args.dry_run:
        return

    if args.out:
        if not pe_paths:
            print("No PE files to ingest (only injection files downloaded).")
            return
        # event_table=None lets build_store auto-fetch from GWOSC;
        # event_table={} skips the fetch.
        event_table = {} if args.no_event_table else None

        from .ingest import build_store
        print(f"\n--- Building store from {len(pe_paths)} PE files ---")
        write_summary = default_write_summary and not args.no_summary
        build_store(pe_paths, args.out, event_table=event_table,
                    cache_dir=args.cache_dir, offline=offline,
                    write_summary=write_summary, file_provenance=provenance)
    else:
        print(f"\nDownloaded {len(all_paths)} files to {args.data_dir}/")
        if pe_paths:
            print("To build the store, re-run with --out store.h5")


if __name__ == "__main__":
    _cli()