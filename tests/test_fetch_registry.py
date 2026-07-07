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
    "injections-O3-BBH": _OldReleaseInfo(
        record_ids=[7890437],
        concept_ids=[None],
        file_filter=_old_is_o3_bbhpop_full,
        description="O3 BBH search sensitivity (GWTC-3, full O1+O2+O3)",
        observing_run="O1+O2+O3",
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
