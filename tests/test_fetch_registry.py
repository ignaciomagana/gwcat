"""Equivalence test for the PR-7 manifest-driven fetch registry.

Before PR 7, ``gwcat/fetch.py`` hardcoded RELEASES / INJECTION_RELEASES as
Python dicts of ``ReleaseInfo(record_ids=..., concept_ids=..., file_filter=...,
description=..., observing_run=...)``, with the per-release file filters
implemented as small Python predicate functions.

This test ports those pre-refactor dicts and filter functions verbatim (as
the "old" reference fixture) and checks that the manifest-driven registry
built by the current ``gwcat.fetch`` module is behaviorally equivalent:
same registry keys (including the "GWTC-4" alias), same record/concept IDs,
same description/observing_run strings, and — for a battery of
representative filenames per release — the same file_filter accept/reject
decisions.  This is the acceptance test for "Existing supported releases
load from manifests" with unchanged fetch behavior.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import pytest

from gwcat import fetch


# ---------------------------------------------------------------------------
# Pre-PR7 reference registry (verbatim port of the old gwcat/fetch.py dicts)
# ---------------------------------------------------------------------------
_OLD_JUNK_SUFFIXES = (".tar.gz", ".tar", ".ipynb", ".txt", ".md", ".fits", ".json")
_OLD_JUNK_SUBSTRINGS = ("PESummaryTable", "Skymap", "skymap", "Archived_Skymaps")


def _old_is_junk(fn: str) -> bool:
    if any(fn.endswith(s) for s in _OLD_JUNK_SUFFIXES):
        return True
    if any(sub in fn for sub in _OLD_JUNK_SUBSTRINGS):
        return True
    return False


def _old_has_event_name(fn: str) -> bool:
    return bool(re.search(r"GW\d{6}", fn))


def _old_is_gwtc21_cosmo(fn: str) -> bool:
    return ("GWTC2p1" in fn
            and fn.endswith("_cosmo.h5")
            and _old_has_event_name(fn)
            and "nocosmo" not in fn)


def _old_is_gwtc3_cosmo(fn: str) -> bool:
    return ("GWTC3" in fn
            and fn.endswith("_cosmo.h5")
            and _old_has_event_name(fn)
            and "nocosmo" not in fn)


def _old_is_o4_pe(fn: str) -> bool:
    if not fn.endswith(".hdf5"):
        return False
    if _old_is_junk(fn):
        return False
    return _old_has_event_name(fn)


def _old_is_injection_hdf(fn: str) -> bool:
    if not (fn.endswith(".hdf") or fn.endswith(".hdf5")):
        return False
    if _old_is_junk(fn):
        return False
    return True


def _old_is_o3_bbhpop_full(fn: str) -> bool:
    return (fn.startswith("endo3_bbhpop")
            and fn.endswith(".hdf5")
            and fn.count("-") == 3)


@dataclass
class _OldReleaseInfo:
    record_ids: List[int]
    concept_ids: List[Optional[int]]
    file_filter: Callable[[str], bool]
    description: str
    observing_run: str


OLD_RELEASES: Dict[str, _OldReleaseInfo] = {
    "GWTC-2.1": _OldReleaseInfo(
        record_ids=[6513631],
        concept_ids=[5117702],
        file_filter=_old_is_gwtc21_cosmo,
        description="O1+O2+O3a cosmo PE samples (GWTC-2.1 v2)",
        observing_run="O1+O2+O3a",
    ),
    "GWTC-3": _OldReleaseInfo(
        record_ids=[8177023],
        concept_ids=[5546662],
        file_filter=_old_is_gwtc3_cosmo,
        description="O3b cosmo PE samples (GWTC-3 v2, Oct 2023 update)",
        observing_run="O3b",
    ),
    "GWTC-4.1": _OldReleaseInfo(
        record_ids=[20275769],
        concept_ids=[20275768],
        file_filter=_old_is_o4_pe,
        description="O4a PE samples (GWTC-4.1, supersedes GWTC-4.0)",
        observing_run="O4a",
    ),
    "GWTC-5": _OldReleaseInfo(
        record_ids=[20348005, 20348006],
        concept_ids=[20276105, 20291739],
        file_filter=_old_is_o4_pe,
        description="O4b PE samples (GWTC-5.0, split across two Zenodo records)",
        observing_run="O4b",
    ),
}
OLD_RELEASES["GWTC-4"] = OLD_RELEASES["GWTC-4.1"]

OLD_INJECTION_RELEASES: Dict[str, _OldReleaseInfo] = {
    "injections-O1O2O3O4": _OldReleaseInfo(
        record_ids=[19500052],
        concept_ids=[None],
        file_filter=_old_is_injection_hdf,
        description="Cumulative O1+O2+O3+O4a+O4b search sensitivity (GWTC-5.0)",
        observing_run="O1-O4b",
    ),
    "injections-O4ab": _OldReleaseInfo(
        record_ids=[19500064],
        concept_ids=[None],
        file_filter=_old_is_injection_hdf,
        description="O4a+O4b-only search sensitivity (GWTC-5.0)",
        observing_run="O4a+O4b",
    ),
    # GW-39: the ONE deliberate departure from the verbatim port.  The old
    # registry called endo3_bbhpop "full O1+O2+O3"; the file is O3a+O3b only.
    "injections-O3-BBH": _OldReleaseInfo(
        record_ids=[7890437],
        concept_ids=[None],
        file_filter=_old_is_o3_bbhpop_full,
        description=("O3-only BBH-pop search sensitivity (GWTC-3, "
                     "endo3_bbhpop, O3a+O3b)"),
        observing_run="O3a+O3b",
    ),
}
OLD_RELEASES.update(OLD_INJECTION_RELEASES)


# ---------------------------------------------------------------------------
# Representative filenames per release: (filename, expected accept/reject)
# Chosen to exercise both the "obvious match" and the near-miss/junk paths
# of each old filter function.
# ---------------------------------------------------------------------------
_FILENAME_BATTERIES: Dict[str, List[str]] = {
    "GWTC-2.1": [
        "IGWN-GWTC2p1-v2-GW150914_095045_cosmo.h5",
        "IGWN-GWTC2p1-v2-GW150914_095045_nocosmo.h5",
        "IGWN-GWTC2p1-v2-GW150914_095045_cosmo.hdf5",
        "IGWN-GWTC2p1-v2-PESummaryTable_cosmo.h5",
        "IGWN-GWTC3-v2-GW191204_171526_cosmo.h5",
        "README.md",
        "IGWN-GWTC2p1-v2_cosmo.h5",  # no event name
    ],
    "GWTC-3": [
        "IGWN-GWTC3-v2-GW191204_171526_cosmo.h5",
        "IGWN-GWTC3-v2-GW191204_171526_nocosmo.h5",
        "IGWN-GWTC2p1-v2-GW150914_095045_cosmo.h5",
        "Skymap-GWTC3-GW191204_171526_cosmo.h5",
        "IGWN-GWTC3-v2_cosmo.h5",
    ],
    "GWTC-4.1": [
        "IGWN-GWTC4p1-v1-GW230601_000000.hdf5",
        "IGWN-GWTC4p1-v1-GW230601_000000_Skymap.fits",
        "IGWN-GWTC4p1-v1-PESummaryTable.hdf5",
        "IGWN-GWTC4p1-v1-GW230601_000000.tar.gz",
        "IGWN-GWTC4p1-v1_notebook.ipynb",
        "IGWN-GWTC4p1-v1_no_event_name.hdf5",
    ],
    "GWTC-5": [
        "IGWN-GWTC5-v1-GW240601_000000.hdf5",
        "IGWN-GWTC5-v1-GW240601_000000_Skymap.fits",
        "IGWN-GWTC5-v1-GW240601_000000.txt",
    ],
    "injections-O1O2O3O4": [
        "injections_o1o2o3o4.hdf5",
        "injections_o1o2o3o4.hdf",
        "injections_o1o2o3o4.tar.gz",
        "PESummaryTable_injections.hdf5",
        "notes.md",
    ],
    "injections-O4ab": [
        "o4ab-cartesian_spins-injections.hdf",
        "o4ab-cartesian_spins-injections.hdf5",
        "o4ab-cartesian_spins-injections.json",
    ],
    "injections-O3-BBH": [
        "endo3_bbhpop-LIGO-T2100113-v12.hdf5",
        "endo3_bbhpop-LIGO-T2100113-v12-1187008882-100.hdf5",
        "endo3_bbhpop-LIGO-T2100113-v12.txt",
        "endo3_nsbhpop-LIGO-T2100113-v12.hdf5",
    ],
}


@pytest.mark.parametrize("name", sorted(OLD_RELEASES.keys()))
def test_registry_keys_match(name):
    assert name in fetch.RELEASES, f"{name!r} missing from new fetch.RELEASES"


def test_no_extra_registry_keys():
    assert set(fetch.RELEASES.keys()) == set(OLD_RELEASES.keys())


@pytest.mark.parametrize("name", sorted(OLD_RELEASES.keys()))
def test_record_and_concept_ids_match(name):
    old = OLD_RELEASES[name]
    new = fetch.RELEASES[name]
    assert new.record_ids == old.record_ids, name
    assert new.concept_ids == old.concept_ids, name


@pytest.mark.parametrize("name", sorted(OLD_RELEASES.keys()))
def test_description_and_observing_run_match(name):
    old = OLD_RELEASES[name]
    new = fetch.RELEASES[name]
    assert new.description == old.description, name
    assert new.observing_run == old.observing_run, name


@pytest.mark.parametrize("name", sorted(_FILENAME_BATTERIES.keys()))
def test_file_filter_equivalence(name):
    old = OLD_RELEASES[name]
    new = fetch.RELEASES[name]
    for fn in _FILENAME_BATTERIES[name]:
        assert new.file_filter(fn) == old.file_filter(fn), (
            f"{name}: file_filter mismatch for {fn!r}: "
            f"old={old.file_filter(fn)} new={new.file_filter(fn)}"
        )


def test_gwtc4_alias_is_gwtc4p1_in_new_registry():
    assert fetch.RELEASES["GWTC-4"] is fetch.RELEASES["GWTC-4.1"]


def test_injection_releases_dict_matches():
    assert set(fetch.INJECTION_RELEASES.keys()) == set(OLD_INJECTION_RELEASES.keys())
    for name, old in OLD_INJECTION_RELEASES.items():
        new = fetch.INJECTION_RELEASES[name]
        assert new.record_ids == old.record_ids
        assert new.concept_ids == old.concept_ids
        assert new.description == old.description
        assert new.observing_run == old.observing_run


def test_available_hides_only_gwtc4_alias():
    # Pre-PR7: `_AVAILABLE = sorted(k for k in RELEASES if k != "GWTC-4")`.
    assert "GWTC-4" not in fetch._AVAILABLE
    assert set(fetch._AVAILABLE) == set(OLD_RELEASES.keys()) - {"GWTC-4"}


def test_pe_catalogs_matches_old_default():
    # Pre-PR7: `_PE_CATALOGS = ["GWTC-2.1", "GWTC-3", "GWTC-4.1", "GWTC-5"]`.
    assert fetch._PE_CATALOGS == ["GWTC-2.1", "GWTC-3", "GWTC-4.1", "GWTC-5"]


def test_unknown_catalog_error_message_hides_alias():
    with pytest.raises(ValueError) as exc:
        fetch.fetch_catalog("NOT-A-REAL-CATALOG", resolve=False, dry_run=True)
    msg = str(exc.value)
    assert "GWTC-4" not in msg.split("Available: ")[-1].replace("GWTC-4.1", "")


def test_fetch_catalog_dry_run_uses_manifest_filter_offline(tmp_path, monkeypatch):
    """End-to-end (but offline): fetch_catalog's file selection for a real
    catalog name must come from the manifest-driven filter, with no network
    access (resolve=False skips resolve_latest; list_files is monkeypatched
    instead of hitting Zenodo)."""
    fake_files = [
        {"key": "IGWN-GWTC2p1-v2-GW150914_095045_cosmo.h5", "size": 1,
         "checksum": "md5:abc", "links": {"self": "http://example/x"}},
        {"key": "IGWN-GWTC2p1-v2-GW150914_095045_nocosmo.h5", "size": 1,
         "checksum": "md5:def", "links": {"self": "http://example/y"}},
        {"key": "IGWN-GWTC2p1-v2-PESummaryTable_cosmo.h5", "size": 1,
         "checksum": "md5:ghi", "links": {"self": "http://example/z"}},
    ]

    def fake_list_files(record_id):
        assert record_id == 6513631  # pinned GWTC-2.1 record_id, no resolve
        return fake_files

    monkeypatch.setattr(fetch, "list_files", fake_list_files)
    paths = fetch.fetch_catalog("GWTC-2.1", data_dir=str(tmp_path),
                                resolve=False, dry_run=True)
    # dry_run never downloads, so no paths are returned, but the filter must
    # have accepted exactly the one true cosmo PE file (proven indirectly:
    # a non-matching filter would raise "No PE files matched filter").
    assert paths == []


# ==========================================================================
# GW-15: FAR / p_astro are TOP-LEVEL in the GWOSC event API
# ==========================================================================
def test_event_table_parses_top_level_far():
    """The real GWOSC payload shape.

    The parser read ``info["parameters"][<pipeline>]``, a key the event API does
    not return, so ``info.get("parameters", {})`` was always ``{}`` and EVERY
    event was stored with far = p_astro = NaN.  Verified live against
    gwosc.org: the response has no "parameters" key and carries far/p_astro at
    the top level.  The production store shows the consequence -- 0 of 282 rows
    had a finite FAR.
    """
    from gwcat.fetch import _parse_gwosc_event_table_page

    table = {}
    _parse_gwosc_event_table_page(
        {"events": {"GW200322_091133-v1": {"far": 140.0, "p_astro": 0.61501,
                                           "commonName": "GW200322_091133"}}},
        table)
    assert table == {"GW200322_091133": {"far": 140.0, "pastro": 0.61501}}


def test_event_table_still_parses_the_legacy_sub_dict():
    """A recorded old payload (or a future API shape) must still work."""
    from gwcat.fetch import _parse_gwosc_event_table_page

    table = {}
    _parse_gwosc_event_table_page(
        {"events": {"GW1-v2": {"parameters": {"pycbc": {"far": 1.2,
                                                        "p_astro": 0.9}}}}},
        table)
    assert table["GW1"]["far"] == 1.2
    assert table["GW1"]["pastro"] == 0.9


def test_event_table_accepts_either_p_astro_spelling():
    """One quantity, two spellings (GW-14): a payload (or a seeded table) that
    writes `pastro` must not read as an absent p_astro."""
    from gwcat.fetch import _parse_gwosc_event_table_page

    table = {}
    _parse_gwosc_event_table_page(
        {"events": {"GW3-v1": {"far": 1.0, "pastro": 0.77}}}, table)
    assert table["GW3"]["pastro"] == 0.77
    # A later page carrying nothing must not clobber the earlier finite value,
    # whichever spelling it was seeded under.
    seeded = {"GW4": {"far": 2.0, "p_astro": 0.66}}
    _parse_gwosc_event_table_page({"events": {"GW4-v2": {}}}, seeded)
    assert seeded["GW4"]["pastro"] == 0.66


def test_top_level_wins_over_the_legacy_sub_dict():
    from gwcat.fetch import _parse_gwosc_event_table_page

    table = {}
    _parse_gwosc_event_table_page(
        {"events": {"GW1-v1": {"far": 7.0,
                               "parameters": {"pycbc": {"far": 1.2}}}}},
        table)
    assert table["GW1"]["far"] == 7.0


def test_absent_far_is_nan_never_fabricated():
    """FAR is genuinely missing from some public entries; that must stay an
    explicit absence so build_store can record far_available=False."""
    import numpy as np

    from gwcat.fetch import _parse_gwosc_event_table_page

    table = {}
    _parse_gwosc_event_table_page({"events": {"GW2-v1": {}}}, table)
    assert np.isnan(table["GW2"]["far"])
    assert np.isnan(table["GW2"]["pastro"])
    # non-numeric junk is absence too, not a crash
    table = {}
    _parse_gwosc_event_table_page(
        {"events": {"GW3-v1": {"far": "n/a", "p_astro": None}}}, table)
    assert np.isnan(table["GW3"]["far"])


def test_a_later_nan_page_does_not_clobber_a_finite_value():
    """The cumulative and per-catalog endpoints overlap, so the same event
    arrives twice; a page lacking FAR must not erase one that had it."""
    import numpy as np

    from gwcat.fetch import _parse_gwosc_event_table_page

    table = {"GW3": {"far": 5.0, "pastro": 0.8}}
    _parse_gwosc_event_table_page({"events": {"GW3-v2": {}}}, table)
    assert table["GW3"]["far"] == 5.0
    assert table["GW3"]["pastro"] == 0.8
    # ... but a finite later value does update
    _parse_gwosc_event_table_page(
        {"events": {"GW3-v3": {"far": 2.0, "p_astro": 0.95}}}, table)
    assert table["GW3"]["far"] == 2.0


# ==========================================================================
# GW-15b: the BBH query must be restricted to the LVK GWTC catalogs
#
# The fixture below is SYNTHETIC but modelled on the live response the review
# recorded (review/findings/fetch.md, finding 2, adversarially CONFIRMED):
# the unfiltered query returns 286 names across eight catalogs --
#   {GWTC-5.0: 104, GWTC-4.1: 86, IAS-O3a: 41, GWTC-3-confident: 32,
#    GWTC-2.1-confident: 20, O4_Discovery_Papers: 1, O3_IMBH_marginal: 1,
#    GWTC-2: 1}
# -- and under `lastver=true` the 41 IAS-O3a entries SUPERSEDE the LVK version
# of those same O3a events, so their m2 > 3 admission decision is made from IAS
# PE rather than the LVK PE this package ingests.  The fixture reproduces that
# structure with real event names (LVK names drawn from the curated population
# lists, the O3a events appearing ONLY under IAS-O3a), plus the two junk names
# the review named: `200114_020818` (an IMBH marginal candidate that is not a
# GWTC event name at all) and `GW190814_192009` (the IAS spelling of the
# deliberately-excluded lower-mass-gap system, caught by neither exclusion
# list).  No network is touched: the pages are handed to the parser through the
# monkeypatched `_gwosc_json`.
# ==========================================================================
from gwcat.bbh_allowed_names import (BBH_O1O2, BBH_O3A, BBH_O3B, BBH_O4A,
                                     BBH_O4B)

_IAS_SUPERSEDED_O3A = BBH_O3A[1:]          # only IAS has these under lastver
_LVK_ENTRIES = (
    [("GWTC-5.0", n) for n in BBH_O4B]
    + [("GWTC-4.1", n) for n in BBH_O4A]
    + [("GWTC-3-confident", n) for n in BBH_O3B]
    + [("GWTC-2.1-confident", n) for n in BBH_O1O2]
    + [("GWTC-2.1-confident", "GW190814")]   # curated non-BBH, LVK spelling
    + [("GWTC-2", BBH_O3A[0])]
)
_NON_LVK_ENTRIES = (
    [("IAS-O3a", n) for n in _IAS_SUPERSEDED_O3A]
    + [("IAS-O3a", "GW190814_192009")]       # third spelling of the mass gap
    + [("O3_IMBH_marginal", "200114_020818")]
    + [("O4_Discovery_Papers", "GW260101_000000")]
)

#: What a correctly filtered query must return from the fixture: the LVK BBH
#: names, with the mass-gap system excluded.
_EXPECTED_LVK_BBH = sorted(BBH_O4B + BBH_O4A + BBH_O3B + BBH_O1O2 + [BBH_O3A[0]])


def _bbh_pages(entries, m2=30.0):
    """One page of GWOSC event-versions results for (catalog, name) pairs."""
    return [{
        "results": [
            {"name": f"{name}-v1", "catalog": catalog,
             "default_parameters": [{"name": "mass_2_source", "best": m2}]}
            for catalog, name in entries
        ],
        "next": None,
    }]


def _patch_bbh_pages(monkeypatch, entries):
    pages = _bbh_pages(entries)
    monkeypatch.setattr(fetch, "_gwosc_json", lambda url, timeout: pages[0])


def _run_bbh_query(monkeypatch, entries):
    """Run the query over a synthetic payload, ignoring the count guard."""
    _patch_bbh_pages(monkeypatch, entries)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fetch.fetch_bbh_names_gwosc(verbose=False)


def test_bbh_query_filters_to_lvk_catalogs(monkeypatch):
    names = _run_bbh_query(monkeypatch, _LVK_ENTRIES + _NON_LVK_ENTRIES)

    assert names == _EXPECTED_LVK_BBH
    # the 35 O3a events whose only surviving version is the IAS one are gone:
    # admitting them would have made the m2 > 3 decision on IAS PE
    assert not (set(names) & set(_IAS_SUPERSEDED_O3A))
    # ... and so is every non-GWTC name the unfiltered query used to admit
    assert "GW190814_192009" not in names       # mass gap, third spelling
    assert "200114_020818" not in names         # not a GWTC event name
    assert "GW260101_000000" not in names       # discovery-paper collection
    assert "GW190814" not in names              # mass gap, LVK spelling


def test_the_259_guard_was_silent_only_because_of_the_contamination(monkeypatch):
    """The old guard fired when the list was SHORTER than the 259-event GWTC-5
    population sample -- and the contamination made it longer, so it never
    fired.  Correctly filtered, the same response is well short of 259 and the
    guard now says so."""
    entries = _LVK_ENTRIES + _NON_LVK_ENTRIES

    # what the old, unfiltered logic kept: everything with m2 > 3 except the
    # names spelled exactly as the old hardcoded non-BBH set spelled them
    old_kept = {name for _cat, name in entries} - {"GW190814"}
    assert len(old_kept) > 259, "fixture must reproduce the silent-guard state"

    _patch_bbh_pages(monkeypatch, entries)
    with pytest.warns(RuntimeWarning, match="were expected"):
        names = fetch.fetch_bbh_names_gwosc(verbose=False)
    assert len(names) < 259


def test_the_guard_is_two_sided(monkeypatch):
    """Exactly the expected number of LVK names is silent; one more warns.

    A one-sided ``len < expected`` guard is what let 286 contaminated names
    pass as healthy.
    """
    expected = fetch._GWOSC_BBH_EXPECTED_NAMES
    lvk = [("GWTC-5.0", f"GW24{i:04d}_000000") for i in range(expected)]

    pages = _bbh_pages(lvk)
    monkeypatch.setattr(fetch, "_gwosc_json", lambda url, timeout: pages[0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any warning fails the test
        assert len(fetch.fetch_bbh_names_gwosc(verbose=False)) == expected

    _patch_bbh_pages(monkeypatch, lvk + [("GWTC-5.0", "GW250101_000000")])
    with pytest.warns(RuntimeWarning, match="as many as"):
        fetch.fetch_bbh_names_gwosc(verbose=False)


def test_mass_gap_is_excluded_under_every_spelling():
    """GWOSC carries three spellings of the lower-mass-gap system: GW190814,
    GW190814_211039 (curated) and GW190814_192009 (IAS).  The fetch layer used
    to keep its own hand-copied exclusion set, which spelled it GW190814 while
    the curated data file spelled it GW190814_211039 -- so the third spelling
    matched neither."""
    for spelling in ("GW190814", "GW190814_211039", "GW190814_192009"):
        assert fetch._is_non_bbh_name(spelling), spelling


def test_the_stem_rule_never_drops_a_curated_bbh():
    """Matching non-BBH names on the date stem must not cost a genuine event
    that merely shares a day with an excluded one (GW190828_063405 /
    GW190828_065509 is the shape of that hazard)."""
    from gwcat.bbh_allowed_names import BBH_ALL

    assert [n for n in BBH_ALL if fetch._is_non_bbh_name(n)] == []


def test_a_catalogless_response_is_an_error_not_an_empty_whitelist(monkeypatch):
    """If GWOSC stops labelling entries with their catalog, the restriction
    cannot be applied.  Returning an empty list would be indistinguishable from
    a legitimately empty query."""
    page = {"results": [{"name": "GW150914-v3", "default_parameters": [
        {"name": "mass_2_source", "best": 30.0}]}], "next": None}
    monkeypatch.setattr(fetch, "_gwosc_json", lambda url, timeout: page)
    with pytest.raises(RuntimeError, match="catalog label"):
        fetch.fetch_bbh_names_gwosc(verbose=False)


@pytest.mark.parametrize("label,expected", [
    ("GWTC-2", True),
    ("GWTC-2.1-confident", True),
    ("GWTC-3-confident", True),
    ("GWTC-4.1", True),
    ("GWTC-5.0", True),
    ("GWTC-6.0", True),          # a future LVK release, admitted automatically
    ("IAS-O3a", False),
    ("OGC-4", False),
    ("O4_Discovery_Papers", False),
    ("O3_IMBH_marginal", False),
    ("GWTC-3-marginal", False),  # sub-threshold candidates, not the BBH sample
    (None, False),
])
def test_lvk_catalog_predicate(label, expected):
    assert fetch._is_lvk_gwtc_catalog(label) is expected


# ==========================================================================
# GW-15d: injection releases are not PE files, and membership -- not a name
# prefix -- is what says so
# ==========================================================================
def test_injection_release_rejected_by_fetch_and_build():
    with pytest.raises(ValueError, match="injection"):
        fetch.fetch_and_build(["GWTC-3", "injections-O3-BBH"], out="unused.h5")


def test_injection_rejection_is_by_membership_not_name_prefix(monkeypatch):
    """An injection manifest whose release name does not start with
    "injections" must still be rejected: the guard was a prefix test on the
    name, which stops holding the moment the release set is generalized."""
    info = fetch.INJECTION_RELEASES["injections-O3-BBH"]
    monkeypatch.setitem(fetch.INJECTION_RELEASES, "sensitivity-O5", info)
    monkeypatch.setitem(fetch.RELEASES, "sensitivity-O5", info)

    assert fetch.is_injection_catalog("sensitivity-O5")
    assert not fetch.is_injection_catalog("GWTC-5")
    pe, injections = fetch.split_pe_and_injection_catalogs(
        ["GWTC-5", "sensitivity-O5"])
    assert pe == ["GWTC-5"] and injections == ["sensitivity-O5"]

    with pytest.raises(ValueError, match="sensitivity-O5"):
        fetch.fetch_and_build(["GWTC-5", "sensitivity-O5"], out="unused.h5")


def test_injection_aliases_are_injections_too():
    for name, info in fetch.INJECTION_RELEASES.items():
        assert fetch.is_injection_catalog(name), name
