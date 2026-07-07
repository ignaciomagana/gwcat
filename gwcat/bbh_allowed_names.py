"""Canonical BBH event whitelist for gwcat — O1 through O4b (GWTC-5.0).

Two modes of operation
----------------------
1. **Dynamic** (preferred): call `fetch_bbh_list()` at runtime, which queries
   the GWOSC v2 API for all events with source-frame secondary mass above the
   selected threshold, then applies the explicit non-BBH exclusion guard.

2. **Static fallback**: use `BBH_ALL` directly when offline or for
   reproducibility. The static list is populated from the GWTC-3/GWTC-4.1/
   GWTC-5.0 population data-release event lists, with known BNS/NSBH/non-BBH
   names removed by `NON_BBH_EXCLUSIONS`.

Usage
-----
    from gwcat.bbh_allowed_names import BBH_ALL, fetch_bbh_list

    # Dynamic — always up to date:
    bbh_names = fetch_bbh_list()            # queries GWOSC live
    cat.select(compact_type="BBH", allowed_names=bbh_names)

    # Static — reproducible / offline:
    cat.select(compact_type="BBH", allowed_names=BBH_ALL)

    # Refresh the static list and print it (run once, paste result back):
    refresh_bbh_list()

GWTC-5.0 context (arxiv:2605.27225, submitted 26 May 2026)
-----------------------------------------------------------
- O4b ran 2024 Apr 10 – 2025 Jan 28
- 103 non-excluded O4b events are retained by the static/cache BBH sample
- Combined O1–O4b: 259 BBH/mass-gap events with PE measurements
- Explicit non-BBH exclusions include BNS/NSBH events such as
  GW190425_232155, GW200105_162426, GW200115_042309,
  GW230518_125908, and GW230529_181500
"""
from __future__ import annotations

import re
import warnings
from importlib import resources
from pathlib import Path
from typing import Optional, Union

# ── Bundled event-list data files ─────────────────────────────────────────────
# The BBH event lists used to be Python literals in this module.  They now live
# as plain-text data files under ``gwcat/data/event_lists/`` (with a
# ``provenance.yaml`` recording where each list came from) so the sample can be
# updated, diffed, and packaged without editing Python.  These helpers read them
# back into the module-level names the rest of the code (and users) rely on.
_EVENT_LIST_DIRNAME = ("data", "event_lists")


def _event_list_path(fname: str) -> Path:
    """Return the on-disk path to a bundled event-list data file."""
    try:
        base = resources.files("gwcat")
        for part in _EVENT_LIST_DIRNAME:
            base = base / part
        candidate = base / fname
        # ``resources.files`` may return a non-filesystem traversable; fall back
        # below if it cannot be represented as a real path.
        return Path(str(candidate))
    except (ModuleNotFoundError, AttributeError, TypeError, NotImplementedError):
        return Path(__file__).parent.joinpath(*_EVENT_LIST_DIRNAME, fname)


def _read_event_list_file(fname: str) -> list[str]:
    """Read an event-list data file, one name per line.

    Blank lines and ``#`` comments (including trailing inline comments) are
    ignored.  Order is preserved as it appears in the file.
    """
    path = _event_list_path(fname)
    if not path.exists():
        path = Path(__file__).parent.joinpath(*_EVENT_LIST_DIRNAME, fname)
    names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.append(line)
    return names


# ── O1–O4b BBH samples (loaded from bundled data files) ───────────────────────
BBH_O1O2 = _read_event_list_file("bbh_o1o2.txt")
BBH_O3A = _read_event_list_file("bbh_o3a.txt")
BBH_O3B = _read_event_list_file("bbh_o3b.txt")
BBH_O4A = _read_event_list_file("bbh_o4a.txt")
BBH_O4B = _read_event_list_file("bbh_o4b.txt")

# Names that must never be admitted to the BBH whitelist.  This includes
# established BNS/NSBH events and low-mass/mass-gap systems that can appear in
# broad PE/cache manifests but are outside this package's BBH-only selection.
# Loaded from data/event_lists/non_bbh_exclusions.txt (see provenance.yaml).
NON_BBH_EXCLUSIONS = set(_read_event_list_file("non_bbh_exclusions.txt"))


def _unique_sorted_bbh_names(*groups: list[str]) -> list[str]:
    """Return deterministic unique BBH names after explicit exclusions."""
    names = {name for group in groups for name in group}
    return sorted(names - NON_BBH_EXCLUSIONS)


# ── Combined static list (O1–O4b) ────────────────────────────────────────────
BBH_ALL: list[str] = _unique_sorted_bbh_names(
    BBH_O1O2, BBH_O3A, BBH_O3B, BBH_O4A, BBH_O4B
)


# Backward-compatible alias for dynamic cache/store discovery guards.
KNOWN_NON_BBH_NAMES = NON_BBH_EXCLUSIONS


def validate_bbh_allowed_names(
    expected_total: int = 259,
    expected_o4b_count: int = 103,
) -> None:
    """Validate the static BBH whitelist against GWTC-5 expectations.

    Checks that the raw static sections have no duplicate names, that the
    combined :data:`BBH_ALL` whitelist contains no explicit non-BBH exclusions,
    and that the populated O4b/static counts match the expected GWTC-5 sample.

    Parameters
    ----------
    expected_total : int
        Expected number of events in :data:`BBH_ALL` once O4b is populated.
    expected_o4b_count : int
        Expected number of non-excluded O4b names in the static/cache sample.

    Raises
    ------
    AssertionError
        If any whitelist invariant is violated.
    """
    grouped_names = BBH_O1O2 + BBH_O3A + BBH_O3B + BBH_O4A + BBH_O4B
    duplicate_grouped = sorted(
        {name for name in grouped_names if grouped_names.count(name) > 1}
    )
    if duplicate_grouped:
        raise AssertionError(f"duplicate static BBH names: {duplicate_grouped}")

    duplicate_all = sorted({name for name in BBH_ALL if BBH_ALL.count(name) > 1})
    if duplicate_all:
        raise AssertionError(f"duplicate BBH_ALL names: {duplicate_all}")

    excluded = sorted(set(BBH_ALL) & NON_BBH_EXCLUSIONS)
    if excluded:
        raise AssertionError(f"excluded non-BBH names present in BBH_ALL: {excluded}")

    o4b_count = len([name for name in BBH_O4B if name not in NON_BBH_EXCLUSIONS])
    if expected_o4b_count is not None and BBH_O4B and o4b_count != expected_o4b_count:
        raise AssertionError(
            f"O4b BBH count {o4b_count} != expected {expected_o4b_count}"
        )

    if expected_total is not None and BBH_O4B and len(BBH_ALL) != expected_total:
        raise AssertionError(
            f"BBH_ALL count {len(BBH_ALL)} != expected {expected_total}"
        )


_EVENT_NAME_RE = re.compile(r"GW\d{6}_\d{6}|GW\d{6}")
_GWTC5_DIR_NAMES = ("GWTC-5", "GWTC-5p0", "GWTC-5.0")


def _decode_hdf5_str(value) -> str:
    """Decode an HDF5 scalar/string-ish value into a Python string."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _is_gwtc5_or_o4b_catalog(label: str) -> bool:
    """Return True when a metadata catalog label identifies GWTC-5/O4b."""
    normalized = re.sub(r"[._\s]", "-", str(label)).upper()
    return "GWTC-5" in normalized or "GWTC5" in normalized or "O4B" in normalized


def _event_name_from_candidate(path: Path) -> Optional[str]:
    """Extract a GW event name using the ingest.event_name_from_path regex."""
    # Keep this regex synchronized with gwcat.ingest.event_name_from_path.
    match = _EVENT_NAME_RE.search(path.name)
    return match.group(0) if match else None


def _gwtc5_cache_dirs(data_dir: Union[str, Path]) -> list[Path]:
    """Return existing likely GWTC-5 cache directories in deterministic order."""
    root = Path(data_dir)
    candidates = []

    if root.name in _GWTC5_DIR_NAMES or _is_gwtc5_or_o4b_catalog(root.name):
        candidates.append(root)
    candidates.extend(root / dirname for dirname in _GWTC5_DIR_NAMES)

    # Include any existing catalog directory matching the naming convention used
    # by fetch_catalog("GWTC-5") even if callers chose a custom spelling.
    if root.exists() and root.is_dir():
        try:
            for child in root.iterdir():
                if child.is_dir() and _is_gwtc5_or_o4b_catalog(child.name):
                    candidates.append(child)
        except OSError:
            pass

    seen = set()
    out = []
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.exists() else candidate.absolute()
        if resolved in seen or not candidate.exists() or not candidate.is_dir():
            continue
        seen.add(resolved)
        out.append(candidate)
    return sorted(out, key=lambda p: str(p))


def _discover_cached_gwtc5_names(data_dir: Union[str, Path]) -> set[str]:
    """Discover cached GWTC-5/O4b event names from local PE-file paths."""
    names: set[str] = set()
    for directory in _gwtc5_cache_dirs(data_dir):
        try:
            paths = sorted(directory.rglob("*"), key=lambda p: str(p))
        except OSError:
            continue
        for path in paths:
            if not path.is_file():
                continue
            name = _event_name_from_candidate(path)
            if name and name not in KNOWN_NON_BBH_NAMES:
                names.add(name)
    return names


def _read_store_gwtc5_names(store_path: Union[str, Path]) -> set[str]:
    """Read GWTC-5/O4b event names from a gwcat HDF5 store, if possible."""
    names: set[str] = set()
    try:
        import h5py
    except Exception as exc:
        warnings.warn(
            "get_bbh_allowed_names: cannot read HDF5 store without "
            f"h5py ({exc})"
        )
        return names

    try:
        with h5py.File(store_path, "r") as f:
            if "index/event_names" not in f or "meta/catalog" not in f:
                return names
            event_names = [_decode_hdf5_str(n) for n in f["index/event_names"][:]]
            catalogs = [_decode_hdf5_str(c) for c in f["meta/catalog"][:]]
            for name, catalog in zip(event_names, catalogs):
                if (
                    _is_gwtc5_or_o4b_catalog(catalog)
                    and name not in KNOWN_NON_BBH_NAMES
                ):
                    names.add(name)
    except Exception as exc:
        warnings.warn(
            f"get_bbh_allowed_names: failed to read store {store_path!r} "
            f"({exc})"
        )
    return names


def get_bbh_allowed_names(
    data_dir: Union[str, Path] = "./GWTC",
    store_path: Optional[Union[str, Path]] = None,
    prefer_gwosc: bool = True,
    expected: int = 259,
) -> list[str]:
    """Return the best available BBH allowed-name list for catalog selection.

    The hardcoded :data:`BBH_ALL` list is always the base.  When it has fewer
    than ``expected`` names, local GWTC-5/O4b cache directories are scanned for
    event names using the same event-name regex as
    :func:`gwcat.ingest.event_name_from_path`.  GWTC-5/O4b PE files are treated
    as BBH by construction except for explicit known BNS/NSBH names.

    If ``store_path`` is supplied, names in ``index/event_names`` whose matching
    ``meta/catalog`` entry identifies GWTC-5/O4b are also added.  Finally, when
    ``prefer_gwosc`` is true, the live GWOSC BBH list is used only if it returns
    at least as many names as the static/cache/store combination.

    A warning is emitted whenever the final count differs from ``expected``;
    the warning includes per-source counts to aid cache/GWOSC debugging.
    """
    static_names = set(BBH_ALL)
    cache_names: set[str] = set()
    store_names: set[str] = set()
    gwosc_names: set[str] = set()
    gwosc_used = False

    combined = set(static_names)
    if len(static_names) < expected:
        cache_names = _discover_cached_gwtc5_names(data_dir)
        combined.update(cache_names)

    if store_path is not None:
        store_names = _read_store_gwtc5_names(store_path)
        combined.update(store_names)

    final = set(combined)
    if prefer_gwosc:
        try:
            gwosc_names = set(fetch_bbh_list(verbose=False)) - NON_BBH_EXCLUSIONS
            if len(gwosc_names) >= len(combined):
                final = set(gwosc_names)
                gwosc_used = True
        except Exception as exc:
            warnings.warn(f"get_bbh_allowed_names: GWOSC BBH query failed ({exc})")

    if len(final) != expected:
        gwosc_store_names = gwosc_names | store_names
        warnings.warn(
            "get_bbh_allowed_names: final BBH count "
            f"{len(final)} != expected {expected} "
            f"(static={len(static_names)}, cache={len(cache_names)}, "
            f"GWOSC/store={len(gwosc_store_names)}, "
            f"GWOSC_used={gwosc_used})"
        )

    final -= NON_BBH_EXCLUSIONS
    return sorted(final)


# ── Dynamic loader ────────────────────────────────────────────────────────────
def fetch_bbh_list(m2_min: float = 3.0, verbose: bool = True) -> list:
    """Return the live BBH event list from GWOSC (requires network).

    Queries the GWOSC v2 API for all events with m2_source > m2_min Msun and
    PE measurements present. Explicitly known non-BBH exclusions are removed
    from the returned live list.

    Falls back to the static BBH_ALL if the network is unavailable.

    Parameters
    ----------
    m2_min : float
        Secondary mass threshold in Msun (default 3.0 = LVK BBH threshold).
    verbose : bool
        Print progress.

    Returns
    -------
    list of str : sorted event names.
    """
    try:
        from .fetch import fetch_bbh_names_gwosc
        names = fetch_bbh_names_gwosc(m2_min=m2_min, verbose=verbose)
        return sorted(set(names) - NON_BBH_EXCLUSIONS)
    except Exception as e:
        import warnings
        warnings.warn(
            f"fetch_bbh_list: GWOSC query failed ({e}); using static BBH_ALL "
            f"({len(BBH_ALL)} events). Run refresh_bbh_list() when online to update."
        )
        return list(BBH_ALL)


def refresh_bbh_list(m2_min: float = 3.0) -> list:
    """Query GWOSC, print the result as Python code, and return the list.

    Run this once when online after GWOSC indexes GWTC-5.0 to get the full
    259-event list you can paste back into BBH_O4B above.

    Example::

        python -c "from gwcat.bbh_allowed_names import refresh_bbh_list; refresh_bbh_list()"
    """
    names = fetch_bbh_list(m2_min=m2_min, verbose=True)
    names = sorted(set(names) - NON_BBH_EXCLUSIONS)
    # Separate out O4b events (GW24* and GW25*)
    o4b = sorted(n for n in names if n.startswith(("GW240", "GW241", "GW242", "GW250", "GW251")))
    known = set(BBH_O1O2 + BBH_O3A + BBH_O3B + BBH_O4A) - NON_BBH_EXCLUSIONS
    new_other = sorted(n for n in names if n not in known and n not in o4b)

    print(f"\n# Total BBH from GWOSC: {len(names)}")
    print(f"# O4b (GW24x/GW25x): {len(o4b)}")
    if new_other:
        print(f"# Other new events not in static list: {new_other}")
    print("\n# Paste this into BBH_O4B:")
    print("BBH_O4B = [")
    for i, n in enumerate(o4b):
        comma = "," if i < len(o4b) - 1 else ""
        print(f'    "{n}"{comma}')
    print("]")
    return names


if __name__ == "__main__":
    print(f"O1+O2 : {len(BBH_O1O2):>4} events")
    print(f"O3a   : {len(BBH_O3A):>4} events")
    print(f"O3b   : {len(BBH_O3B):>4} events")
    print(f"O4a   : {len(BBH_O4A):>4} events")
    print(f"O4b   : {len(BBH_O4B):>4} events (static; run refresh_bbh_list() for live)")
    print(f"Total : {len(BBH_ALL):>4} events (static)")
    dupes = sorted(set(n for n in BBH_ALL if BBH_ALL.count(n) > 1))
    if dupes:
        print(f"WARNING: duplicates: {dupes}")
    else:
        print("No duplicates.")
    validate_bbh_allowed_names()
    print("Static whitelist sanity checks passed.")
    print("\nFetching live list from GWOSC...")
    refresh_bbh_list()
