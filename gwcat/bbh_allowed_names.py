"""Canonical BBH event whitelist for gwcat — O1 through O4b (GWTC-5.0).

Two lists, which are NOT the same sample
-----------------------------------------
1. **Static, curated** (the population sample): `BBH_ALL`, populated from the
   GWTC-3/GWTC-4.1/GWTC-5.0 population data-release event lists (FAR < 1/yr),
   with known BNS/NSBH/non-BBH names removed by `NON_BBH_EXCLUSIONS`.  Use it
   for any population or cosmology analysis; it is reproducible and offline.

2. **Live, mass threshold**: `fetch_bbh_list()` queries the GWOSC v2 API for
   every confident event with source-frame secondary mass above the threshold,
   restricted to the LVK GWTC catalogs, then applies the exclusion guard.  It
   applies NO FAR cut (14 confident BBH-mass events have FAR >= 1/yr), so it is
   a candidate list for refreshing the curated one, not a population sample.
   It falls back to the static list when the network fails;
   `fetch_bbh_list_with_source()` returns which of the two you got, and
   `require_live=True` turns a failure into an error instead.

Usage
-----
    from gwcat.bbh_allowed_names import BBH_ALL, fetch_bbh_list

    # The population sample -- reproducible / offline:
    cat.select(compact_type="BBH", allowed_names=BBH_ALL)

    # Live candidates (no FAR cut; NOT a population sample):
    candidates = fetch_bbh_list()           # queries GWOSC live

    # Print the live list in the data-file format, to update the curated
    # lists under gwcat/data/event_lists/ by hand:
    refresh_bbh_list()

GWTC-5.0 context (arxiv:2605.27225, submitted 26 May 2026)
-----------------------------------------------------------
- O4b ran 2024 Apr 10 – 2025 Jan 28
- 104 O4b events, the full official Event_list/GWTC5_BBH.txt
- Combined O1–O4b: 259 BBH events with PE measurements
- Explicit non-BBH exclusions include BNS/NSBH events such as
  GW190425_081805, GW200105_162426, GW200115_042309,
  GW230518_125908, and GW230529_181500, plus the GW190814_211039
  lower-mass-gap system (excluded from the BBH population sample)
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
    # utf-8-sig: a byte-order mark must not become part of the first name.
    for line in path.read_text(encoding="utf-8-sig").splitlines():
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

#: Size of the curated GWTC-5.0 BBH population sample (O1-O4b), and the one
#: place this package spells that number: it was a default in two separate
#: signatures that happened to agree.  It is also recorded in
#: ``data/event_lists/provenance.yaml`` (``expected.bbh_all_total``), which is
#: the data files' own description of themselves; a test asserts the two agree.
EXPECTED_BBH_TOTAL = 259

#: Number of non-excluded O4b names in the official Event_list/GWTC5_BBH.txt.
EXPECTED_O4B_COUNT = 104


def validate_bbh_allowed_names(
    expected_total: int = EXPECTED_BBH_TOTAL,
    expected_o4b_count: int = EXPECTED_O4B_COUNT,
) -> None:
    """Validate the static BBH whitelist against GWTC-5 expectations.

    Checks that the raw static sections have no duplicate names, that the
    combined :data:`BBH_ALL` whitelist contains no explicit non-BBH exclusions,
    that the populated O4b/static counts match the expected GWTC-5 sample, and
    that the two historically-contested events land on the correct side of the
    whitelist (a *membership* check, not just a count: two compensating
    count errors can otherwise pass the totals while corrupting the sample).

    Parameters
    ----------
    expected_total : int
        Expected number of events in :data:`BBH_ALL` once O4b is populated.
    expected_o4b_count : int
        Expected number of non-excluded O4b names in the static/cache sample
        (the full official Event_list/GWTC5_BBH.txt = 104).

    Raises
    ------
    AssertionError
        If any whitelist invariant is violated.
    """
    # Membership invariants that a pure count check cannot catch.  Keep these in
    # sync with the official GWTC-5.0 population Event_list.
    MUST_INCLUDE = {"GW240525_031210"}  # confirmed O4b BBH in GWTC5_BBH.txt
    MUST_EXCLUDE = {"GW190814_211039"}  # lower-mass-gap system, not in the BBH sample
    bbh_set = set(BBH_ALL)
    missing_required = sorted(MUST_INCLUDE - bbh_set)
    if missing_required:
        raise AssertionError(
            f"confirmed BBH names missing from BBH_ALL: {missing_required}"
        )
    wrongly_present = sorted(MUST_EXCLUDE & bbh_set)
    if wrongly_present:
        raise AssertionError(
            f"non-BBH-population names present in BBH_ALL: {wrongly_present}"
        )
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
    expected: int = EXPECTED_BBH_TOTAL,
    adopt_live_list: bool = False,
) -> list[str]:
    """Return the best available BBH allowed-name list for catalog selection.

    The hardcoded :data:`BBH_ALL` list is always the base.  When it has fewer
    than ``expected`` names, local GWTC-5/O4b cache directories are scanned for
    event names using the same event-name regex as
    :func:`gwcat.ingest.event_name_from_path`.  GWTC-5/O4b PE files are treated
    as BBH by construction except for explicit known BNS/NSBH names.

    If ``store_path`` is supplied, names in ``index/event_names`` whose matching
    ``meta/catalog`` entry identifies GWTC-5/O4b are also added.  Finally, when
    ``prefer_gwosc`` is true, the live GWOSC BBH list is consulted: the names
    it adds to the curated combination are reported, and it REPLACES the
    curated combination only when ``adopt_live_list=True`` and it is a strict
    superset of it -- i.e. it adds events without dropping any curated
    population member.

    ``adopt_live_list`` defaults to False (review 2026-10-20).  The live list
    is a secondary-mass threshold with no FAR cut, while the curated list is
    the FAR < 1/yr population sample, so the superset rule alone let a live
    list holding sub-threshold candidates (GW200322_091133 at 140/yr)
    redefine the sample, and a later ``select(allowed_names=...)`` applies no
    FAR cut to make up for it.  Apply your own FAR cut before adopting one.

    That condition is the point (GW-15).  It used to be ``len(gwosc) >=
    len(combined)``: a live list merely *as long* as the curated one replaced
    it wholesale, so a live list that had swapped k curated events for k others
    silently redefined the population sample, and the curated GWTC-5 membership
    list -- which is authoritative for population membership, and is not
    reproducible by reapplying a mass threshold to PE point estimates -- lost
    to a coincidence of length.  A live list that genuinely extends the sample
    (a new observing run) still wins; anything else warns and names what it
    would have dropped.

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
            live_names, source = fetch_bbh_list_with_source(verbose=False)
            if source == "gwosc":
                gwosc_names = set(live_names) - NON_BBH_EXCLUSIONS
                dropped = combined - gwosc_names
                added = sorted(gwosc_names - combined)
                if not dropped and adopt_live_list:
                    final = set(gwosc_names)
                    gwosc_used = True
                    if added:
                        warnings.warn(
                            "get_bbh_allowed_names: adopted the live GWOSC "
                            f"list (adopt_live_list=True), adding {len(added)} "
                            f"name(s) to the curated population list: "
                            f"{added[:10]}. The live list applies no FAR cut; "
                            "these events are not checked against FAR < 1/yr.")
                elif not dropped:
                    if added:
                        warnings.warn(
                            "get_bbh_allowed_names: keeping the curated BBH "
                            f"list; the live GWOSC list adds {len(added)} "
                            f"name(s) not in it ({added[:10]}), but it is a "
                            "secondary-mass threshold with NO FAR cut, so they "
                            "are not adopted. Apply a FAR cut yourself and pass "
                            "adopt_live_list=True to adopt them.")
                else:
                    warnings.warn(
                        "get_bbh_allowed_names: keeping the curated BBH list; "
                        f"the live GWOSC list ({len(gwosc_names)} names) is "
                        f"missing {len(dropped)} curated population member(s) "
                        f"(e.g. {sorted(dropped)[:5]}), so it would redefine "
                        "the sample rather than extend it."
                    )
            # source == "static": the live query failed and fetch_bbh_list
            # substituted BBH_ALL, which has already warned.  It is NOT a
            # GWOSC result and must never be recorded as one.
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
def fetch_bbh_list_with_source(
    m2_min: float = 3.0,
    verbose: bool = True,
    require_live: bool = False,
) -> tuple[list, str]:
    """Return ``(names, source)`` for the BBH list, saying where it came from.

    ``source`` is ``"gwosc"`` when the live query succeeded and ``"static"``
    when it failed and the curated :data:`BBH_ALL` list was substituted.  The
    caller needs that distinction: :func:`fetch_bbh_list` used to swallow its
    own network error and return ``BBH_ALL``, which
    :func:`get_bbh_allowed_names` then recorded as ``GWOSC_used=True`` -- the
    static list masquerading as a live one, in the very diagnostic meant to
    tell the two apart (GW-15).

    Parameters
    ----------
    m2_min : float
        Secondary mass threshold in Msun (default 3.0 = LVK BBH threshold).
    verbose : bool
        Print progress.
    require_live : bool
        When true, a failed live query raises instead of falling back: a caller
        that asked for live data gets an error, not a silent substitution.
    """
    try:
        from .fetch import fetch_bbh_names_gwosc
        names = fetch_bbh_names_gwosc(m2_min=m2_min, verbose=verbose)
        return sorted(set(names) - NON_BBH_EXCLUSIONS), "gwosc"
    except Exception as e:
        if require_live:
            raise
        warnings.warn(
            f"fetch_bbh_list: GWOSC query failed ({e}); using static BBH_ALL "
            f"({len(BBH_ALL)} events). Run refresh_bbh_list() when online to update."
        )
        return list(BBH_ALL), "static"


def fetch_bbh_list(m2_min: float = 3.0, verbose: bool = True,
                   require_live: bool = False) -> list:
    """Return the live BBH event list from GWOSC (requires network).

    Queries the GWOSC v2 API for all events with m2_source > m2_min Msun and
    PE measurements present, restricted to the LVK GWTC catalogs. Explicitly
    known non-BBH exclusions are removed from the returned live list.

    This is a mass threshold with NO FAR cut, so it is not the population
    sample (:data:`BBH_ALL`, FAR < 1/yr): it is the candidate list a refresh
    of the curated one starts from.

    Falls back to the static BBH_ALL (with a warning) if the network is
    unavailable, unless ``require_live=True``.  Callers that must not confuse
    the two should use :func:`fetch_bbh_list_with_source`, which says which one
    they got.

    Returns
    -------
    list of str : sorted event names.
    """
    names, _source = fetch_bbh_list_with_source(
        m2_min=m2_min, verbose=verbose, require_live=require_live)
    return names


#: GPS - UTC, in seconds, from each date on (the leap seconds since 2012).
_GPS_MINUS_UTC = (("2012-07-01", 16), ("2015-07-01", 17), ("2017-01-01", 18))

#: Which bundled data file holds each run's BBH names.
_RUN_LIST_FILE = {"O1": "bbh_o1o2.txt", "O2": "bbh_o1o2.txt",
                  "O3a": "bbh_o3a.txt", "O3b": "bbh_o3b.txt",
                  "O4a": "bbh_o4a.txt", "O4b": "bbh_o4b.txt"}


def _run_of_event_name(name: str) -> Optional[str]:
    """The observing run an event name's UTC timestamp falls in, or None.

    Uses the official run windows of :mod:`gwcat.observing_runs`
    (``EXPOSURE_WINDOWS_GPS``).  A short name (``GWyymmdd``) is placed at the
    start of its day.  Name prefixes cannot do this: O4a ran until
    2024-01-16, so ``GW2401xx`` events are O4a while ``GW2404xx`` ones are O4b.
    """
    from datetime import datetime, timezone
    from .observing_runs import EXPOSURE_WINDOWS_GPS
    m = re.fullmatch(r"GW(\d{2})(\d{2})(\d{2})(?:_(\d{2})(\d{2})(\d{2}))?",
                     str(name))
    if not m:
        return None
    yy, mo, dd = (int(m.group(i)) for i in (1, 2, 3))
    hh, mi, ss = (int(m.group(i) or 0) for i in (4, 5, 6))
    try:
        t = datetime(2000 + yy, mo, dd, hh, mi, ss, tzinfo=timezone.utc)
    except ValueError:
        return None
    leap = 0
    for since, n in _GPS_MINUS_UTC:
        if t >= datetime.fromisoformat(since).replace(tzinfo=timezone.utc):
            leap = n
    gps = (t - datetime(1980, 1, 6, tzinfo=timezone.utc)).total_seconds() + leap
    for run, (lo, hi) in EXPOSURE_WINDOWS_GPS.items():
        if lo <= gps <= hi:
            return run
    return None


def refresh_bbh_list(m2_min: float = 3.0) -> list:
    """Query GWOSC, print the result in the bundled data-file format, and
    return the list.

    The BBH lists live as plain-text files under ``gwcat/data/event_lists/``
    (one name per line; see ``provenance.yaml``), not as Python literals.
    This prints, per observing run, the live names that the run's bundled
    file does not hold yet, assigning each name to its run by the UTC
    timestamp in the name against the official run windows (review
    2026-10-20: a GW24x/GW25x prefix put the three January-2024 O4a events
    into the O4b block, and the output asked for a paste into a Python
    literal that no longer exists).  The live list applies no FAR cut, so
    check every new name against the population Event_list before adding it.

    Raises rather than falling back to the static list if the query fails: this
    function exists to replace the static list, so silently printing the static
    list back is the one output that must never happen (GW-15).

    Since GW-15 the live query is restricted to the LVK GWTC catalogs, so it
    returns the LVK-only sample rather than the 286 names that third-party
    catalogs inflated it to.  Since the 2026-10-20 review it reads every event
    version (no ``lastver``), so the LVK versions of the O3a events whose latest
    GWOSC version is a third-party one count again: about 272 names are
    expected (the 2026-09-24 cumulative table), not the ~244 measured before.

    Example::

        python -c "from gwcat.bbh_allowed_names import refresh_bbh_list; refresh_bbh_list()"
    """
    names = fetch_bbh_list(m2_min=m2_min, verbose=True, require_live=True)
    names = sorted(set(names) - NON_BBH_EXCLUSIONS)
    bundled = {"bbh_o1o2.txt": set(BBH_O1O2), "bbh_o3a.txt": set(BBH_O3A),
               "bbh_o3b.txt": set(BBH_O3B), "bbh_o4a.txt": set(BBH_O4A),
               "bbh_o4b.txt": set(BBH_O4B)}
    by_file: dict = {}
    unplaced = []
    for n in names:
        run = _run_of_event_name(n)
        if run is None:
            unplaced.append(n)
            continue
        by_file.setdefault(_RUN_LIST_FILE[run], []).append(n)

    print(f"\n# Total BBH from GWOSC: {len(names)} (no FAR cut: check every "
          f"new name against the population Event_list before adding it)")
    for fname in ("bbh_o1o2.txt", "bbh_o3a.txt", "bbh_o3b.txt",
                  "bbh_o4a.txt", "bbh_o4b.txt"):
        live = by_file.get(fname, [])
        new = [n for n in live if n not in bundled[fname]]
        print(f"\n# gwcat/data/event_lists/{fname}: {len(live)} live, "
              f"{len(new)} not in the file yet"
              + ("; append these lines:" if new else "."))
        for n in new:
            print(n)
    if unplaced:
        print(f"\n# Outside every official run window (not placed): "
              f"{unplaced}")
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
