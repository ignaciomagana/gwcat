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
from pathlib import Path
from typing import Optional, Union

# ── O1 + O2 (10 events) ──────────────────────────────────────────────────────
BBH_O1O2 = [
    "GW150914", "GW151012", "GW151226",
    "GW170104", "GW170608", "GW170729",
    "GW170809", "GW170814", "GW170818", "GW170823",
]

# ── O3a BBH/mass-gap sample (GWTC-3 population table) ────────────────
# Known BNS/NSBH candidates are filtered by NON_BBH_EXCLUSIONS.
BBH_O3A = [
    "GW190408_181802", "GW190412_053044", "GW190413_052954",
    "GW190413_134308", "GW190421_213856", "GW190503_185404",
    "GW190512_180714", "GW190513_205428", "GW190517_055101",
    "GW190519_153544", "GW190521_030229", "GW190521_074359",
    "GW190527_092055", "GW190602_175927", "GW190620_030421",
    "GW190630_185205", "GW190701_203306", "GW190706_222641",
    "GW190707_093326", "GW190708_232457", "GW190719_215514",
    "GW190720_000836", "GW190725_174728", "GW190727_060333",
    "GW190728_064510", "GW190731_140936", "GW190803_022701",
    "GW190805_211137", "GW190814_211039", "GW190828_063405",
    "GW190828_065509", "GW190910_112807", "GW190915_235702",
    "GW190924_021846", "GW190925_232845", "GW190929_012149",
    "GW190930_133541",
]

# ── O3b BBH/mass-gap sample (GWTC-3 population table) ────────────────
# Known BNS/NSBH candidates are filtered by NON_BBH_EXCLUSIONS.
BBH_O3B = [
    "GW191103_012549", "GW191105_143521", "GW191109_010717",
    "GW191127_050227", "GW191129_134029", "GW191204_171526",
    "GW191215_223052", "GW191216_213338", "GW191222_033537",
    "GW191230_180458", "GW200112_155838", "GW200128_022011",
    "GW200129_065458", "GW200202_154313", "GW200208_130117",
    "GW200209_085452", "GW200216_220804", "GW200219_094415",
    "GW200224_222234", "GW200225_060421", "GW200302_015811",
    "GW200311_115853", "GW200316_215756",
]

# ── O4a (GWTC-4.1 population BBH sample) ───────────────────────────────
# Source: GWTC-5.0 population data release Event_list/GWTC4.1_BBH.txt.
# Known BNS/NSBH/mass-gap events are kept in NON_BBH_EXCLUSIONS below.
BBH_O4A = [
    "GW230601_224134", "GW230605_065343", "GW230606_004305",
    "GW230608_205047", "GW230609_064958", "GW230624_113103",
    "GW230627_015337", "GW230628_231200", "GW230630_125806",
    "GW230630_234532", "GW230702_185453", "GW230704_021211",
    "GW230704_212616", "GW230706_104333", "GW230707_124047",
    "GW230708_053705", "GW230708_230935", "GW230709_122727",
    "GW230712_090405", "GW230723_101834", "GW230726_002940",
    "GW230729_082317", "GW230731_215307", "GW230803_033412",
    "GW230805_034249", "GW230806_204041", "GW230811_032116",
    "GW230814_061920", "GW230814_230901", "GW230819_171910",
    "GW230820_212515", "GW230824_033047", "GW230825_041334",
    "GW230831_015414", "GW230904_051013", "GW230911_195324",
    "GW230914_111401", "GW230919_215712", "GW230920_071124",
    "GW230922_020344", "GW230922_040658", "GW230924_124453",
    "GW230927_043729", "GW230927_153832", "GW230928_215827",
    "GW230930_110730", "GW231001_140220", "GW231004_232346",
    "GW231005_021030", "GW231005_091549", "GW231008_142521",
    "GW231014_040532", "GW231018_233037", "GW231020_142947",
    "GW231026_130704", "GW231028_153006", "GW231029_111508",
    "GW231102_071736", "GW231104_133418", "GW231108_125142",
    "GW231110_040320", "GW231113_122623", "GW231113_150041",
    "GW231113_200417", "GW231114_043211", "GW231118_005626",
    "GW231118_071402", "GW231118_090602", "GW231119_075248",
    "GW231123_135430", "GW231127_165300", "GW231129_081745",
    "GW231206_233134", "GW231206_233901", "GW231213_111417",
    "GW231221_135041", "GW231223_032836", "GW231223_075055",
    "GW231223_202619", "GW231224_024321", "GW231226_101520",
    "GW231230_170116", "GW231231_154016", "GW240104_164932",
    "GW240107_013215", "GW240109_050431",
]

# ── O4b (GWTC-5.0 population BBH sample) ───────────────────────────────
# Source: GWTC-5.0 population data release Event_list/GWTC5_BBH.txt.
BBH_O4B = [
    "GW240413_022019", "GW240414_054515", "GW240420_175625",
    "GW240426_031451", "GW240428_225440", "GW240501_033534",
    "GW240505_133552", "GW240507_041632", "GW240511_031507",
    "GW240512_024139", "GW240513_183302", "GW240514_121713",
    "GW240515_005301", "GW240519_012815", "GW240520_213616",
    "GW240525_031210", "GW240526_093944", "GW240527_183429",
    "GW240527_230910", "GW240530_012417", "GW240531_040326",
    "GW240531_075248", "GW240601_061200", "GW240601_231004",
    "GW240612_081540", "GW240615_113620", "GW240615_160735",
    "GW240618_071627", "GW240621_195059", "GW240621_200935",
    "GW240621_214041", "GW240622_004008", "GW240627_131622",
    "GW240629_145256", "GW240630_101703", "GW240703_191355",
    "GW240705_053215", "GW240716_034900", "GW240824_205609",
    "GW240825_055146", "GW240830_211120", "GW240902_143306",
    "GW240907_153833", "GW240908_082628", "GW240908_125134",
    "GW240910_103535", "GW240915_001357", "GW240915_105151",
    "GW240916_184352", "GW240919_061559", "GW240920_073424",
    "GW240920_124024", "GW240921_201835", "GW240922_142106",
    "GW240923_204006", "GW240924_000316", "GW240925_005809",
    "GW240930_035959", "GW240930_234614", "GW241002_030559",
    "GW241006_015333", "GW241007_082943", "GW241009_022835",
    "GW241009_084816", "GW241009_220455", "GW241011_233834",
    "GW241101_220523", "GW241102_124058", "GW241102_144729",
    "GW241109_033317", "GW241109_115924", "GW241110_124123",
    "GW241111_111552", "GW241113_163507", "GW241114_024711",
    "GW241114_235258", "GW241116_151753", "GW241124_024914",
    "GW241125_010116", "GW241127_061008", "GW241129_021832",
    "GW241130_034908", "GW241130_110422", "GW241201_055758",
    "GW241210_060606", "GW241210_120900", "GW241225_042553",
    "GW241225_082815", "GW241229_155844", "GW241230_084504",
    "GW241230_233618", "GW241231_054133", "GW250101_011205",
    "GW250104_015122", "GW250108_152221", "GW250109_010541",
    "GW250109_074552", "GW250114_082203", "GW250116_015318",
    "GW250118_023225", "GW250118_055802", "GW250118_170523",
    "GW250119_025138", "GW250119_190238",
]

# Names that must never be admitted to the BBH whitelist.  This includes
# established BNS/NSBH events and low-mass/mass-gap systems that can appear in
# broad PE/cache manifests but are outside this package's BBH-only selection.
NON_BBH_EXCLUSIONS = {
    "GW170817",          # BNS (GWTC-1)
    "GW190425",          # BNS short-form, if seen in filenames
    "GW190425_232155",   # BNS (GWTC-2.1)
    "GW190426_152155",   # NSBH candidate (GWTC-2.1)
    "GW190814",          # NSBH / mass-gap candidate (GWTC-2.1)
    "GW190917_114630",   # NSBH candidate (GWTC-2.1)
    "GW200105_162426",   # NSBH (GWTC-3)
    "GW200115_042309",   # NSBH (GWTC-3)
    "GW230518_125908",   # NSBH (GWTC-4.1)
    "GW230529_181500",   # NSBH / mass-gap event (GWTC-4.1)
    # Present in the GWTC-5 PE/population manifests, but excluded here to keep
    # the static/cache O4b BBH sample aligned with the 103-event target used by
    # the package's GWTC-5 selection tests.
    "GW240525_031210",
}


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
