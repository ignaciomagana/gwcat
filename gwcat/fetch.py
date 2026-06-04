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
from typing import Dict, List, Optional, Sequence
from urllib.request import urlopen, Request
from urllib.error import HTTPError

# ---------------------------------------------------------------------------
# File filters — decide which files in a Zenodo record are PE cosmo samples
# ---------------------------------------------------------------------------
# Each filter returns True for files we WANT.
# The O3 releases (GWTC-2.1, GWTC-3) ship separate cosmo / nocosmo files.
# The O4 releases (GWTC-4.1, GWTC-5) ship a single combined file per event.
# In both cases we must exclude PESummaryTable, skymaps, notebooks, tarballs.

_JUNK_SUFFIXES = (".tar.gz", ".tar", ".ipynb", ".txt", ".md", ".fits", ".json")
_JUNK_SUBSTRINGS = ("PESummaryTable", "Skymap", "skymap", "Archived_Skymaps")


def _is_junk(fn: str) -> bool:
    """True for non-PE files that live alongside PE samples in the record."""
    if any(fn.endswith(s) for s in _JUNK_SUFFIXES):
        return True
    if any(sub in fn for sub in _JUNK_SUBSTRINGS):
        return True
    return False


def _has_event_name(fn: str) -> bool:
    """True if the filename contains a GW event identifier."""
    return bool(re.search(r"GW\d{6}", fn))


def _is_gwtc21_cosmo(fn: str) -> bool:
    """GWTC-2.1: per-event cosmo files only (reject nocosmo)."""
    return ("GWTC2p1" in fn
            and fn.endswith("_cosmo.h5")
            and _has_event_name(fn)
            and "nocosmo" not in fn)


def _is_gwtc3_cosmo(fn: str) -> bool:
    """GWTC-3: per-event cosmo files only (reject nocosmo)."""
    return ("GWTC3" in fn
            and fn.endswith("_cosmo.h5")
            and _has_event_name(fn)
            and "nocosmo" not in fn)


def _is_o4_pe(fn: str) -> bool:
    """GWTC-4.1 / GWTC-5: combined per-event HDF5.
    No cosmo/nocosmo split in O4 — just one file per event.
    """
    if not fn.endswith(".hdf5"):
        return False
    if _is_junk(fn):
        return False
    return _has_event_name(fn)


# ---------------------------------------------------------------------------
# Release registry
# ---------------------------------------------------------------------------
# record_ids  : pinned version records (one per Zenodo deposit).
#               GWTC-5 is split across two Zenodo deposits.
# concept_ids : version-agnostic record IDs; resolve_latest() follows
#               these to find the newest version.  Use these by default.

@dataclass
class ReleaseInfo:
    """Metadata for one GWTC PE data release on Zenodo."""
    record_ids: List[int]
    concept_ids: List[Optional[int]]
    file_filter: callable
    description: str
    observing_run: str


RELEASES: Dict[str, ReleaseInfo] = {
    "GWTC-2.1": ReleaseInfo(
        record_ids=[6513631],
        concept_ids=[5117702],
        file_filter=_is_gwtc21_cosmo,
        description="O1+O2+O3a cosmo PE samples (GWTC-2.1 v2)",
        observing_run="O1+O2+O3a",
    ),
    "GWTC-3": ReleaseInfo(
        record_ids=[8177023],
        concept_ids=[5546662],
        file_filter=_is_gwtc3_cosmo,
        description="O3b cosmo PE samples (GWTC-3 v2, Oct 2023 update)",
        observing_run="O3b",
    ),
    "GWTC-4.1": ReleaseInfo(
        record_ids=[20275769],
        concept_ids=[20275768],
        file_filter=_is_o4_pe,
        description="O4a PE samples (GWTC-4.1, supersedes GWTC-4.0)",
        observing_run="O4a",
    ),
    "GWTC-5": ReleaseInfo(
        record_ids=[20348005, 20348006],         # Part 1, Part 2
        concept_ids=[20276105, 20291739],
        file_filter=_is_o4_pe,
        description="O4b PE samples (GWTC-5.0, split across two Zenodo records)",
        observing_run="O4b",
    ),
}

# Convenience alias: "GWTC-4" → "GWTC-4.1"
RELEASES["GWTC-4"] = RELEASES["GWTC-4.1"]


# ---------------------------------------------------------------------------
# Injection / selection function records
# ---------------------------------------------------------------------------
def _is_injection_hdf(fn: str) -> bool:
    """Select injection HDF files, exclude docs/PSDs/tarballs."""
    if not (fn.endswith(".hdf") or fn.endswith(".hdf5")):
        return False
    if _is_junk(fn):
        return False
    return True


def _is_o3_bbhpop_full(fn: str) -> bool:
    """Select only the full-O3 BBH injection file (not O3a/O3b splits)."""
    return (fn.startswith("endo3_bbhpop")
            and fn.endswith(".hdf5")
            and fn.count("-") == 2)  # no GPS-time splits in filename


INJECTION_RELEASES: Dict[str, ReleaseInfo] = {
    "injections-O1O2O3O4": ReleaseInfo(
        record_ids=[19500052],
        concept_ids=[None],
        file_filter=_is_injection_hdf,
        description="Cumulative O1+O2+O3+O4a+O4b search sensitivity (GWTC-5.0)",
        observing_run="O1-O4b",
    ),
    "injections-O4ab": ReleaseInfo(
        record_ids=[19500064],
        concept_ids=[None],
        file_filter=_is_injection_hdf,
        description="O4a+O4b-only search sensitivity (GWTC-5.0)",
        observing_run="O4a+O4b",
    ),
    "injections-O3-BBH": ReleaseInfo(
        record_ids=[7890437],
        concept_ids=[None],
        file_filter=_is_o3_bbhpop_full,
        description="O3 BBH search sensitivity (GWTC-3, full O1+O2+O3)",
        observing_run="O1+O2+O3",
    ),
}

# Merge injection records into RELEASES so fetch_catalog finds everything
RELEASES.update(INJECTION_RELEASES)

ZENODO_API = "https://zenodo.org/api/records"


# ---------------------------------------------------------------------------
# Zenodo API helpers (stdlib only — no requests needed for metadata queries)
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


def list_files(record_id: int) -> List[dict]:
    """Return the file list for a Zenodo record."""
    data = _zenodo_get(f"{ZENODO_API}/{record_id}")
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
# Download helpers
# ---------------------------------------------------------------------------
def _md5(path: str) -> str:
    h = hashlib.md5()
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
    show_progress : bool
        Show tqdm progress bars during download.
    dry_run : bool
        List files that would be downloaded without actually downloading.

    Returns
    -------
    list of str
        Paths to the downloaded PE files, sorted.
    """
    if catalog not in RELEASES:
        available = sorted(k for k in RELEASES if k != "GWTC-4")  # hide alias
        raise ValueError(
            f"Unknown catalog {catalog!r}. Available: {available}"
        )

    info = RELEASES[catalog]
    rids = record_ids or (
        _resolve_record_ids(info, catalog) if resolve else list(info.record_ids)
    )

    dest_dir = Path(data_dir) / catalog.replace(".", "p")  # GWTC-4.1 → GWTC-4p1
    all_paths = []

    for part_idx, rid in enumerate(rids, 1):
        part_label = f" part {part_idx}/{len(rids)}" if len(rids) > 1 else ""
        print(f"[{catalog}{part_label}] querying Zenodo record {rid} ...")
        all_files = list_files(rid)
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
            if os.path.exists(dest) and md5 and _md5(dest) == md5:
                print(f"  [{i}/{len(pe_files)}] {fname} (cached)")
            else:
                print(f"  [{i}/{len(pe_files)}] {fname}")
                download_file(url, dest, expected_md5=md5,
                              show_progress=show_progress)
            all_paths.append(dest)

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

    Returns
    -------
    str : path to the output store file.
    """
    from .ingest import build_store, IngestConfig

    cfg = ingest_cfg or IngestConfig()
    all_paths = []
    for cat in catalogs:
        paths = fetch_catalog(cat, data_dir=data_dir, resolve=resolve,
                              show_progress=show_progress)
        all_paths.extend(paths)

    if not all_paths:
        raise RuntimeError("No PE files downloaded; cannot build store.")

    print(f"\n--- Building store from {len(all_paths)} files ---")
    build_store(all_paths, out, cfg=cfg, event_table=event_table,
                extra_params=extra_params)
    return out


# ---------------------------------------------------------------------------
# Helpers: FAR / p_astro from GWOSC event API
# ---------------------------------------------------------------------------
def fetch_event_table_gwosc(
    catalog_tag: str = "GWTC",
    timeout: int = 30,
) -> dict:
    """Fetch FAR and p_astro from the GWOSC event API.

    Returns {event_name: {'far': float, 'pastro': float}}.
    catalog_tag : "GWTC" (cumulative), "GWTC-2.1-confident", etc.
    """
    base = f"https://gwosc.org/eventapi/json/{catalog_tag}/"
    table = {}
    url = base

    while url:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
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
        url = data.get("links", {}).get("next")

    return table


def fetch_bbh_names_gwosc(
    m2_min: float = 3.0,
    timeout: int = 30,
    verbose: bool = True,
) -> list:
    """Fetch the canonical BBH event name list from the GWOSC v2 API.

    Queries the GWOSC /api/v2/event-versions endpoint for all events that have
    PE results (i.e. chirp_mass_source is present) and whose secondary mass
    m2_source > m2_min (default 3.0 Msun), which is the standard threshold
    used in GWTC papers to classify an event as BBH rather than NSBH or BNS.

    This automatically covers all releases the GWOSC API knows about —
    currently GWTC-1 through GWTC-4.0, and GWTC-5.0 once GWOSC updates their
    database (expected within weeks of the May 2026 paper release).

    Parameters
    ----------
    m2_min : float
        Minimum secondary source-frame mass in Msun. 3.0 is the LVK standard
        NS/BH boundary used in GWTC-4/5 population papers.
    timeout : int
        HTTP timeout per request in seconds.
    verbose : bool
        Print progress.

    Returns
    -------
    list of str
        Sorted event names, e.g. ["GW150914", "GW151012", ...].

    Notes
    -----
    - Events without PE (search-only, no chirp_mass_source) are automatically
      excluded since the mass filter only applies to events with mass parameters.
    - The GWOSC API is paginated; this function follows all pages.
    - Safe to call with no internet: raises URLError; caller should fall back to
      the static BBH_ALL list in gwcat.bbh_allowed_names.
    """
    base = "https://gwosc.org/api/v2/event-versions"
    params = f"lastver=true&min-mass-2-source={m2_min}&include-default-parameters=true&pagesize=100"
    url = f"{base}?{params}"

    names = []
    page = 1
    while url:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())

        results = data.get("results", [])
        n_total = data.get("results_count", "?")
        if verbose and page == 1:
            print(f"GWOSC BBH query: {n_total} total events with m2_source > {m2_min} Msun")

        for ev in results:
            name = ev.get("name", "")
            if name:
                names.append(name)

        # Follow pagination
        next_url = data.get("next")
        if next_url:
            # next_url is wrapped in angle brackets in some responses
            next_url = next_url.strip("<>")
        url = next_url
        page += 1

    names = sorted(set(names))
    if verbose:
        print(f"  → {len(names)} BBH events with PE (m2_source > {m2_min} Msun)")
    return names


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_AVAILABLE = sorted(k for k in RELEASES if k != "GWTC-4")  # hide alias in help
_PE_CATALOGS = ["GWTC-2.1", "GWTC-3", "GWTC-4.1", "GWTC-5"]

def _cli():
    import argparse

    ap = argparse.ArgumentParser(
        prog="gwcat-fetch",
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

    args = ap.parse_args()

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

    all_paths = []
    pe_paths = []          # only PE files go to build_store
    for cat in catalogs:
        rids = args.record_ids if (len(catalogs) == 1 and args.record_ids) else None
        paths = fetch_catalog(
            cat, data_dir=args.data_dir, record_ids=rids,
            resolve=resolve, show_progress=show_progress,
            dry_run=args.dry_run,
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
        # event_table=None lets build_store auto-fetch from GWOSC;
        # event_table={} skips the fetch.
        event_table = {} if args.no_event_table else None

        from .ingest import build_store
        print(f"\n--- Building store from {len(pe_paths)} PE files ---")
        build_store(pe_paths, args.out, event_table=event_table)
    else:
        print(f"\nDownloaded {len(all_paths)} files to {args.data_dir}/")
        if pe_paths:
            print("To build the store, re-run with --out store.h5")


if __name__ == "__main__":
    _cli()