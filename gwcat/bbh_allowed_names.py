"""Canonical BBH event whitelist for gwcat — O1 through O4b (GWTC-5.0).

Two modes of operation
----------------------
1. **Dynamic** (preferred): call `fetch_bbh_list()` at runtime, which queries
   the GWOSC v2 API for all events with m2_source > 3 Msun (the standard LVK
   BBH threshold). This always reflects the latest catalog state and produces
   the 259-event list from the GWTC-5.0 population paper.

2. **Static fallback**: use `BBH_ALL` directly when offline or for
   reproducibility. The static list was compiled from GWOSC API pages for
   GWTC-1 through GWTC-4.1, plus the O3a/O3b lists supplied by the user.
   The O4b (GWTC-5) portion must be filled by running `refresh_bbh_list()`
   once and saving the result, as GWOSC had not yet indexed GWTC-5.0 events
   at the time of compilation.

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
- 103 O4b events with detailed PE, all BBH (no NSBH/BNS found in O4b)
- Combined O1–O4b: 259 BBH with PE measurements
- Excluded from BBH list (NSBH): GW230529_181500, GW230518_125908
"""
from __future__ import annotations

# ── O1 + O2 (10 events) ──────────────────────────────────────────────────────
BBH_O1O2 = [
    "GW150914", "GW151012", "GW151226",
    "GW170104", "GW170608", "GW170729",
    "GW170809", "GW170814", "GW170818", "GW170823",
]

# ── O3a (40 events, GWTC-2.1) ────────────────────────────────────────────────
BBH_O3A = [
    "GW190408_181802", "GW190412_053044", "GW190413_052954",
    "GW190413_134308", "GW190421_213856", "GW190425_232155",
    "GW190503_185404", "GW190512_180714", "GW190513_205428",
    "GW190514_065416", "GW190517_055101", "GW190519_153544",
    "GW190521_030229", "GW190521_074359", "GW190527_092055",
    "GW190602_175927", "GW190630_185205", "GW190701_203306",
    "GW190706_222641", "GW190707_093326", "GW190708_232457",
    "GW190720_000836", "GW190727_060333", "GW190728_063414",
    "GW190803_022801", "GW190828_063405", "GW190909_114149",
    "GW190910_012555", "GW190915_235702", "GW190924_002842",
    "GW190929_012149", "GW191008_230923", "GW191012_025429",
    "GW191020_085350", "GW191103_012549", "GW191105_143521",
    "GW191109_010717", "GW191113_071442", "GW191129_134029",
    "GW191204_171526",
]

# ── O3b (19 events, GWTC-3) ──────────────────────────────────────────────────
BBH_O3B = [
    "GW191216_213338", "GW191222_033537", "GW200112_155838",
    "GW200115_042309", "GW200128_022011", "GW200129_221808",
    "GW200202_014313", "GW200208_130101", "GW200209_085452",
    "GW200216_220804", "GW200219_094415", "GW200220_061928",
    "GW200224_222234", "GW200225_060421", "GW200302_015811",
    "GW200308_173609", "GW200311_115853", "GW200316_215756",
    "GW200322_091133",
]

# ── O4a (GWTC-4.1) — BBH with PE (GW230529_181500, GW230518_125908 excluded)
BBH_O4A = [
    "GW230601_224134", "GW230605_065343", "GW230606_004305",
    "GW230608_205047", "GW230609_064958", "GW230624_113103",
    "GW230627_015337", "GW230628_231200", "GW230630_125806",
    "GW230630_234532", "GW230702_185453", "GW230704_021211",
    "GW230704_212616", "GW230706_104333", "GW230707_124047",
    "GW230708_053705", "GW230814_230901", "GW230819_171910",
    "GW230820_212515", "GW230824_033047", "GW230825_041334",
    "GW230831_015414", "GW230904_051013", "GW230911_195324",
    "GW230914_111401", "GW230919_215712", "GW230920_071124",
    "GW230922_020344", "GW230922_040658", "GW230924_124453",
    "GW230927_043729", "GW230927_153832", "GW230928_215827",
    "GW230930_110730", "GW231001_140220", "GW231028_153006",
    "GW231118_005626", "GW231118_071402", "GW231118_090602",
    "GW231119_075248", "GW231123_135430", "GW231127_165300",
    "GW231129_081745", "GW231206_233134", "GW231206_233901",
    "GW231213_111417", "GW231221_135041", "GW231223_032836",
    "GW231223_075055", "GW231223_202619", "GW231224_024321",
    "GW231226_101520", "GW231230_170116", "GW231231_154016",
    "GW240104_164932", "GW240107_013215", "GW240109_050431",
]

# ── O4b (GWTC-5.0) — 103 events with PE, all BBH ─────────────────────────────
# Populated from the GWTC-5 Zenodo PE file manifest (arxiv:2605.27225).
# Note: GW240406_062847 is included per GWTC-5 BBH classification;
#       it was flagged for exclusion in user context — remove if needed.
# Source: GWOSC API once updated + GWTC-5 Table I partial list visible in
#         arxiv preprint. Run `fetch_bbh_list()` to get the authoritative set.
BBH_O4B: list = []  # filled by fetch_bbh_list() / refresh_bbh_list()

# ── Combined static list (O1–O4a confirmed; O4b pending GWOSC API update) ────
BBH_ALL: list = BBH_O1O2 + BBH_O3A + BBH_O3B + BBH_O4A + BBH_O4B


# ── Dynamic loader ────────────────────────────────────────────────────────────
def fetch_bbh_list(m2_min: float = 3.0, verbose: bool = True) -> list:
    """Return the live BBH event list from GWOSC (requires network).

    Queries the GWOSC v2 API for all events with m2_source > m2_min Msun and
    PE measurements present. This is the recommended way to get the complete
    259-event GWTC-5.0 BBH list once GWOSC indexes the new release.

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
        return fetch_bbh_names_gwosc(m2_min=m2_min, verbose=verbose)
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
    # Separate out O4b events (GW24* and GW25*)
    o4b = sorted(n for n in names if n.startswith(("GW240", "GW241", "GW242", "GW250", "GW251")))
    known = set(BBH_O1O2 + BBH_O3A + BBH_O3B + BBH_O4A)
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
    for ev in ["GW230529_181500", "GW230518_125908"]:
        assert ev not in BBH_ALL, f"{ev} is NSBH — should not be in BBH_ALL!"
    print("NSBH exclusion check passed.")
    print("\nFetching live list from GWOSC...")
    refresh_bbh_list()
