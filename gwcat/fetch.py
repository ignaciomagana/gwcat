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


def _checksum_for(file_entry: dict) -> Optional[str]:
    cs = file_entry.get("checksum", "")
    return cs[4:] if cs.startswith("md5:") else (cs or None)


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
    """
    from pathlib import Path
    base = Path(dest_dir).resolve()
    target = (base / str(key)).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise RuntimeError(
            f"refusing to download {key!r}: the remote file name resolves to "
            f"{target}, outside the data directory {base}. A release record "
            f"that names a path outside its own download directory is not one "
            f"to trust.")
    return str(target)


def download_file(url: str, dest: str, expected_md5: Optional[str] = None,
                  show_progress: bool = True) -> str:
    """Download a single file with progress bar and checksum verification.
    Skips download if dest exists and checksum matches."""
    dest = str(dest)
    if os.path.exists(dest) and expected_md5:
        if _md5(dest) == expected_md5:
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

        if expected_md5:
            actual = _md5(tmp)
            if actual != expected_md5:
                raise RuntimeError(
                    f"Checksum mismatch for {dest}: expected {expected_md5}, "
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
def _resolve_record_ids(info: ReleaseInfo, catalog: str) -> List[int]:
    """Resolve every concept_id in a ReleaseInfo to its latest record.
    Falls back to the pinned record_id on failure."""
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
    return resolved


def _cache_record_resolution(
    cache_dir: Union[str, Path],
    catalog: str,
    info: ReleaseInfo,
    rids: Sequence[int],
    resolved: bool,
) -> None:
    """Record which Zenodo records this run actually used (GW-15).

    Which records an online run used is part of what that run did: with
    ``resolve=True`` (the default) they are whatever each concept DOI pointed
    at *at that moment*, which is not recoverable from the registry pins
    afterwards.  Caching it is what lets an offline replay read the same
    records instead of silently falling back to the pins.
    """
    fetch_cache.write_metadata_cache(
        cache_dir,
        fetch_cache.resolution_cache_key(catalog),
        {
            "catalog": catalog,
            "record_ids": [int(r) for r in rids],
            "resolved_latest": bool(resolved),
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
    silent fallback to the pins.
    """
    payload = fetch_cache.read_metadata_cache(
        cache_dir, fetch_cache.resolution_cache_key(catalog))
    rids = payload.get("record_ids") if isinstance(payload, dict) else None
    if not rids:
        raise fetch_cache.OfflineCacheMissError(
            f"Offline mode: the cached record resolution for {catalog!r} in "
            f"{str(cache_dir)!r} records no record_ids. Re-run the fetch once "
            "online with the same cache_dir, or pass resolve=False to use the "
            "pinned record IDs deliberately."
        )
    return [int(r) for r in rids]


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
        "GWTC-5".
    data_dir : str
        Root directory; files go into {data_dir}/{catalog}/.
    record_ids : list of int, optional
        Override the Zenodo record ID(s).
    resolve : bool
        If True (default), query Zenodo for the latest version of each
        concept DOI.  Set False to use pinned records without network.
        Offline it selects *which records the earlier online run used*: the
        resolution is read back from ``cache_dir`` (a missing/truncated one
        raises :class:`~gwcat.fetch_cache.OfflineCacheMissError`), so an
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

    info = RELEASES[catalog]
    if offline_mode:
        # Resolving the latest version always requires a network call, so an
        # offline run reads back WHICH RECORDS THE ONLINE RUN RESOLVED (GW-15).
        # Falling back to the pinned records here -- as this did -- means an
        # offline replay looks for a different Zenodo record than the online
        # run cached the moment any release is versioned past its pin.
        # resolve=False is the deliberate opt-in to the pins.
        if record_ids:
            rids = list(record_ids)
        elif resolve:
            rids = _cached_resolved_record_ids(catalog, cache_dir)
        else:
            rids = list(info.record_ids)
    else:
        rids = record_ids or (
            _resolve_record_ids(info, catalog) if resolve else list(info.record_ids)
        )
        if cache_dir is not None:
            _cache_record_resolution(cache_dir, catalog, info, rids,
                                     resolved=bool(resolve and not record_ids))

    dest_dir = Path(data_dir) / catalog.replace(".", "p")  # GWTC-4.1 → GWTC-4p1
    all_paths = []

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
            md5 = _checksum_for(f)
            if md5 is None and _require_checksums(info):
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
            cached_ok = os.path.exists(dest) and md5 and _md5(dest) == md5
            if cached_ok:
                print(f"  [{i}/{len(pe_files)}] {fname} (cached)")
            elif offline_mode:
                raise fetch_cache.OfflineCacheMissError(
                    f"Offline mode: {dest} is missing or does not match the "
                    f"expected checksum, and network downloads are disabled. "
                    "Populate data_dir by running once online, or pass "
                    "offline=False."
                )
            else:
                print(f"  [{i}/{len(pe_files)}] {fname}")
                download_file(url, dest, expected_md5=md5,
                              show_progress=show_progress)
            all_paths.append(dest)
            if provenance is not None:
                provenance[fname] = {
                    "record_id": str(rid),
                    "file_checksum": _sha256(dest),
                }

    if not dry_run:
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
    for cat in catalogs:
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
#: version of the same event under ``lastver=true``.  Restricted to LVK GWTC
#: catalogs the same query returns ~244 (the review measured 244; an
#: independent re-run of the identical filter measured 243) -- fewer than the
#: curated population sample, because the population sample is a curated
#: membership list, not "every GWOSC event with a PE mass_2_source above 3
#: Msun".  The guard is two-sided against this number: *fewer* means an
#: incomplete live index, *more* means entries the LVK-catalog filter did not
#: expect to admit.  It is a warning, not an assertion: it exists to be noticed
#: and re-baselined by whoever re-runs the query, which is exactly what the old
#: one-sided 259 could never do.
_GWOSC_BBH_EXPECTED_NAMES = 244

#: GWOSC catalog labels the BBH query accepts: the LVK GWTC releases, in any of
#: the spellings GWOSC uses ("GWTC-2", "GWTC-2.1-confident", "GWTC-4.1",
#: "GWTC-5.0").  Deliberately a *pattern* and not a hardcoded list, so a future
#: "GWTC-6.0" is admitted automatically while third-party catalogs (IAS-O3a,
#: OGC-*), marginal-candidate catalogs (O3_IMBH_marginal, GWTC-*-marginal) and
#: discovery-paper collections (O4_Discovery_Papers) stay out.
_GWOSC_LVK_CATALOG_RE = re.compile(r"GWTC-\d+(?:\.\d+)?(?:-CONFIDENT)?")

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


def fetch_bbh_names_gwosc(
    m2_min: float = 3.0,
    verbose: bool = True,
    timeout: int = 30,
    cache_dir: Optional[Union[str, Path]] = None,
    offline: Optional[bool] = None,
) -> list[str]:
    """Return BBH event names from the paginated GWOSC v2 event API.

    The GWOSC endpoint is filtered for the latest event version with a
    secondary source-frame mass above ``m2_min`` and default parameters included.
    Events are kept only when they come from an **LVK GWTC catalog**, carry a
    canonical GWTC event name, are not a curated non-BBH event under any
    spelling, and the returned default PE parameters contain ``mass_2_source``
    (or the legacy alias ``m2_source``) above the threshold.

    The catalog restriction is the point (GW-15).  GWOSC indexes third-party
    catalogs beside GWTC, and under ``lastver=true`` a non-LVK version
    *supersedes* the LVK one for the same event: the unfiltered query returns
    286 names across eight catalogs, and 41 O3a events were being admitted on
    the IAS pipeline's ``mass_2_source`` rather than on the LVK PE this package
    actually ingests.  Filtering is done client-side on each entry's catalog
    label rather than by adding a ``catalog=`` query parameter, so a change in
    what the server accepts can only ever make this stricter, never silently
    return an unfiltered list.

    cache_dir / offline : see :mod:`gwcat.fetch_cache`.  ``None``/unset (the
    defaults) disable caching/offline-mode entirely -- unchanged, byte-identical
    default behavior.  When caching, every raw page of the paginated response is
    written under one cache key so an offline replay reconstructs the exact same
    ``names`` set via the same per-event filter logic.
    """
    offline_mode = fetch_cache.is_offline(offline)
    key = fetch_cache.gwosc_cache_key(f"bbh_names_m2min_{m2_min}")

    if offline_mode:
        if cache_dir is None:
            raise fetch_cache.OfflineCacheMissError(
                "fetch_bbh_names_gwosc(offline=True) requires cache_dir to "
                "locate the previously cached GWOSC response."
            )
        cached = fetch_cache.read_metadata_cache(cache_dir, key)
        pages = cached["pages"]
    else:
        query = urlencode(
            {
                "include-default-parameters": "true",
                "lastver": "true",
                "min-mass-2-source": m2_min,
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
                cache_dir, key, {"m2_min": m2_min, "pages": pages})

    names: set[str] = set()
    n_entries = 0
    n_with_catalog = 0
    dropped_catalogs: Dict[str, int] = {}
    dropped_names: set[str] = set()
    for data in pages:
        for event in data.get("results", []):
            if not isinstance(event, dict):
                continue
            n_entries += 1

            catalog = _gwosc_event_catalog(event)
            if catalog is not None:
                n_with_catalog += 1
            if not _is_lvk_gwtc_catalog(catalog):
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
            if _is_non_bbh_name(name):
                continue

            params = event.get("default_parameters")
            m2_source = _gwosc_best_parameter(params, "mass_2_source", "m2_source")
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
        for page in cached["pages"]:
            _parse_gwosc_event_table_page(page, table)
        return table

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

    return table


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
    if args.record_ids and len(catalogs) != 1:
        ap.error("--record-ids requires exactly one --catalog")

    show_progress = not args.no_progress
    resolve = not args.no_resolve
    # None (not False) when --offline is absent, so GWCAT_OFFLINE can still
    # activate offline mode; the flag only ever turns it on explicitly.
    offline = True if args.offline else None

    all_paths = []
    pe_paths = []          # only PE files go to build_store
    for cat in catalogs:
        rids = args.record_ids if (len(catalogs) == 1 and args.record_ids) else None
        paths = fetch_catalog(
            cat, data_dir=args.data_dir, record_ids=rids,
            resolve=resolve, show_progress=show_progress,
            dry_run=args.dry_run, cache_dir=args.cache_dir,
            offline=offline,
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
                    write_summary=write_summary)
    else:
        print(f"\nDownloaded {len(all_paths)} files to {args.data_dir}/")
        if pe_paths:
            print("To build the store, re-run with --out store.h5")


if __name__ == "__main__":
    _cli()