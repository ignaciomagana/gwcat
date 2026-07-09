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
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from urllib.error import HTTPError

from . import fetch_cache
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

    with open(dest, "wb") as f:
        bar = tqdm(total=total, unit="B", unit_scale=True,
                   desc=os.path.basename(dest)[:40], leave=False) \
              if show_progress and total else None
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            if bar:
                bar.update(len(chunk))
        if bar:
            bar.close()

    if expected_md5:
        actual = _md5(dest)
        if actual != expected_md5:
            raise RuntimeError(
                f"Checksum mismatch for {dest}: expected {expected_md5}, got {actual}"
            )
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
        Ignored (treated as False) when ``offline`` is true, since resolving
        the latest version always requires a network call.
    show_progress : bool
        Show tqdm progress bars during download.
    dry_run : bool
        List files that would be downloaded without actually downloading.
    cache_dir : str or Path, optional
        Passed to :func:`list_files` to cache/read the raw Zenodo file-listing
        response (see :mod:`gwcat.fetch_cache`).  ``None`` (default) disables
        caching -- unchanged, byte-identical default behavior.
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
        # Resolving the latest version always requires a network call;
        # offline mode uses the pinned/explicit record IDs unconditionally.
        rids = record_ids or list(info.record_ids)
    else:
        rids = record_ids or (
            _resolve_record_ids(info, catalog) if resolve else list(info.record_ids)
        )

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
            dest = str(dest_dir / fname)
            url = _download_url_for(f)
            md5 = _checksum_for(f)
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
_GWOSC_BBH_EXPECTED_NAMES = 259
_GWOSC_KNOWN_NON_BBH = {
    # BNS / NSBH / mass-gap candidates that may appear in broad GWOSC
    # mass-threshold queries. Keep this lower-level guard in sync with
    # gwcat.bbh_allowed_names.NON_BBH_EXCLUSIONS so callers of
    # fetch_bbh_names_gwosc() and fetch_bbh_list() get the same BBH selection.
    "GW170817",
    "GW190425",
    "GW190425_232155",
    "GW190426_152155",
    "GW190814",
    "GW190917_114630",
    "GW200105_162426",
    "GW200115_042309",
    "GW230518_125908",
    "GW230529_181500",
    "GW240525_031210",
}


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
    Events are kept only when the returned default PE parameters contain
    ``mass_2_source`` (or the legacy alias ``m2_source``) above the threshold.
    Known BNS/NSBH events are explicitly removed as a safety guard.

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
    for data in pages:
        for event in data.get("results", []):
            if not isinstance(event, dict):
                continue
            name = _clean_gwosc_event_name(
                event.get("name") or event.get("shortName") or event.get("grace_id")
            )
            if not name or name in _GWOSC_KNOWN_NON_BBH:
                continue

            params = event.get("default_parameters")
            m2_source = _gwosc_best_parameter(params, "mass_2_source", "m2_source")
            if m2_source is None or m2_source <= m2_min:
                continue
            names.add(name)

    result = sorted(names)
    if verbose:
        print(f"fetch_bbh_names_gwosc: selected {len(result)} BBH candidates")
    if len(result) < _GWOSC_BBH_EXPECTED_NAMES:
        warnings.warn(
            "GWOSC returned only "
            f"{len(result)} BBH names with PE mass_2_source > {m2_min}; "
            f"GWTC-5-era data are expected to contain at least "
            f"{_GWOSC_BBH_EXPECTED_NAMES}. Callers may be relying on an "
            "incomplete live GWOSC index.",
            RuntimeWarning,
            stacklevel=2,
        )
    return result


def _parse_gwosc_event_table_page(data: dict, table: dict) -> None:
    """Merge one raw GWOSC event-API page's events into ``table`` in place.

    Factored out so the live-fetch and offline-cache-replay code paths in
    :func:`fetch_event_table_gwosc` share identical parsing logic -- caching
    can never silently drift from what a live call would have computed.
    """
    events = data.get("events", {})
    for name, info in events.items():
        clean = re.sub(r"-v\d+$", "", name)
        far = pastro = float("nan")
        params = info.get("parameters", {})
        for _key, pset in params.items():
            if isinstance(pset, dict):
                if "far" in pset and pset["far"] is not None:
                    far = float(pset["far"])
                if "p_astro" in pset and pset["p_astro"] is not None:
                    pastro = float(pset["p_astro"])
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
    ap.add_argument("--metadata-overrides", default=None, metavar="PATH",
                    help="YAML/CSV event-metadata overrides used when --out "
                         "builds a store. Values take precedence over GWOSC.")
    ap.add_argument("--metadata-diagnostics", default=None, metavar="PATH",
                    help="Write per-event metadata provenance diagnostics as "
                         "JSON. With --metadata-overrides, defaults to "
                         "<out>.metadata_diagnostics.json.")
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
    if (args.metadata_overrides or args.metadata_diagnostics) and not args.out:
        ap.error("--metadata-overrides/--metadata-diagnostics require --out")

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
        if not cat.startswith("injections"):
            pe_paths.extend(paths)

    if args.dry_run:
        return

    if args.out:
        if not pe_paths:
            print("No PE files to ingest (only injection files downloaded).")
            return
        from .ingest import build_store, event_name_from_path

        event_table = {} if args.no_event_table else None
        summary_context = None
        if args.metadata_overrides or args.metadata_diagnostics:
            from .event_metadata import (
                assemble_event_metadata,
                load_user_overrides,
            )

            overrides = (load_user_overrides(args.metadata_overrides)
                         if args.metadata_overrides else {})
            online_table = {}
            if not args.no_event_table:
                online_table = fetch_event_table_gwosc(
                    cache_dir=args.cache_dir, offline=offline)

            event_names = list(dict.fromkeys(
                event_name_from_path(path) for path in pe_paths))
            event_table, diagnostics = assemble_event_metadata(
                event_names,
                online_table=online_table,
                user_overrides=overrides,
            )

            diagnostics_path = args.metadata_diagnostics
            if args.metadata_overrides and diagnostics_path is None:
                diagnostics_path = f"{args.out}.metadata_diagnostics.json"
            if diagnostics_path is not None:
                with open(diagnostics_path, "w") as f:
                    json.dump(diagnostics, f, indent=2)
                    f.write("\n")

            if args.metadata_overrides:
                summary_context = {
                    "metadata_overrides_path": str(args.metadata_overrides),
                    "metadata_diagnostics_path": str(diagnostics_path),
                    "n_metadata_overrides": len(overrides),
                }

        print(f"\n--- Building store from {len(pe_paths)} PE files ---")
        write_summary = default_write_summary and not args.no_summary
        build_kwargs = {
            "event_table": event_table,
            "cache_dir": args.cache_dir,
            "offline": offline,
            "write_summary": write_summary,
        }
        if summary_context is not None:
            build_kwargs["summary_context"] = summary_context
        build_store(pe_paths, args.out, **build_kwargs)
    else:
        print(f"\nDownloaded {len(all_paths)} files to {args.data_dir}/")
        if pe_paths:
            print("To build the store, re-run with --out store.h5")


if __name__ == "__main__":
    _cli()
