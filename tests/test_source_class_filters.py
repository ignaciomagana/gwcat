"""Source-class filtering tests (PR 2).

Covers selecting BBH-only, NSBH-only, BNS-only, and all-CBC from a tiny
synthetic mixed catalog, plus the ``cbc`` keyword meaning "all compact-binary
classes" and the source-class provenance attribute written by to_darksirens.

Fixtures are tiny synthetic HDF5 stores built directly with h5py to match the
schema GWCatalog reads -- no network, no pesummary/ingest dependency.
"""
import numpy as np
import h5py
import pytest

from gwcat.catalog import GWCatalog
from gwcat.source_class import (normalize_source_class, resolve_filter_classes,
                                SOURCE_CLASSES, SourceClassMeta)


DARKSIRENS_PARAMS = ["mass_1", "mass_2", "luminosity_distance", "ra", "dec",
                     "chi_eff", "p_dL_pe"]


def build_mixed_store(tmp_path, events, H0=67.74, Om0=0.3089, seed=7,
                      include_far_available=True, name="mixed_store.h5"):
    """Write a synthetic store for a mixed source-class catalog.

    events : list of dicts, each with keys:
        name (str), source_class (str), far (float, np.nan = missing),
        and optionally n (int, samples per event).
    """
    rng = np.random.default_rng(seed)
    offsets = [0]
    cols = {p: [] for p in DARKSIRENS_PARAMS}
    meta = {k: [] for k in
            ["source_class", "compact_type", "far", "far_available", "pastro",
             "dL_prior_H0", "dL_prior_Om0"]}
    names = []
    for ev in events:
        n = int(ev.get("n", 15))
        cols["mass_1"].append(rng.uniform(20, 45, n))
        cols["mass_2"].append(rng.uniform(8, 20, n))
        cols["luminosity_distance"].append(rng.uniform(300, 800, n))
        cols["ra"].append(rng.uniform(0, 2 * np.pi, n))
        cols["dec"].append(rng.uniform(-np.pi / 2, np.pi / 2, n))
        cols["chi_eff"].append(rng.uniform(-0.3, 0.3, n))
        cols["p_dL_pe"].append(rng.uniform(0.1, 1.0, n))
        offsets.append(offsets[-1] + n)

        far = float(ev.get("far", np.nan))
        sc = normalize_source_class(ev["source_class"])
        names.append(ev["name"])
        meta["source_class"].append(sc)
        meta["compact_type"].append(sc)
        meta["far"].append(far)
        meta["far_available"].append(1.0 if np.isfinite(far) else 0.0)
        meta["pastro"].append(float(ev.get("pastro", np.nan)))
        meta["dL_prior_H0"].append(H0)
        meta["dL_prior_Om0"].append(Om0)

    path = tmp_path / name
    with h5py.File(path, "w") as f:
        f.attrs["param_names"] = np.array(DARKSIRENS_PARAMS,
                                          dtype=h5py.string_dtype())
        idx = f.create_group("index")
        idx.create_dataset("offsets", data=np.array(offsets, dtype="i8"))
        idx.create_dataset("event_names",
                           data=np.array(names, dtype=h5py.string_dtype()))
        mg = f.create_group("meta")
        for k in ["source_class", "compact_type"]:
            mg.create_dataset(k, data=np.array(meta[k], dtype=h5py.string_dtype()))
        for k in ["far", "pastro", "dL_prior_H0", "dL_prior_Om0"]:
            mg.create_dataset(k, data=np.asarray(meta[k], dtype="f8"))
        if include_far_available:
            mg.create_dataset("far_available",
                              data=np.asarray(meta["far_available"], dtype="f8"))
        sg = f.create_group("samples")
        for p in DARKSIRENS_PARAMS:
            sg.create_dataset(p, data=np.concatenate(cols[p]))
    return str(path)


MIXED_EVENTS = [
    {"name": "GW900001_000001", "source_class": "BBH", "far": 1e-3},
    {"name": "GW900002_000002", "source_class": "BBH", "far": 5e-4},
    {"name": "GW900003_000003", "source_class": "NSBH", "far": 2e-3},
    {"name": "GW900004_000004", "source_class": "BNS", "far": 1e-5},
    {"name": "GW900005_000005", "source_class": "MassGap", "far": 4e-2},
]


# ── unit tests of the source-class helpers ───────────────────────────────────
def test_normalize_source_class_aliases():
    assert normalize_source_class("bbh") == "BBH"
    assert normalize_source_class("NSBH") == "NSBH"
    assert normalize_source_class("bhns") == "NSBH"
    assert normalize_source_class("bns") == "BNS"
    assert normalize_source_class("mass_gap") == "MassGap"
    assert normalize_source_class("mass-gap") == "MassGap"
    assert normalize_source_class("") == "Unknown"
    assert normalize_source_class(None) == "Unknown"
    assert normalize_source_class("something weird") == "Unknown"
    assert normalize_source_class(b"BBH") == "BBH"


def test_resolve_filter_classes_keywords():
    assert resolve_filter_classes("bbh") == {"BBH"}
    assert resolve_filter_classes("nsbh") == {"NSBH"}
    assert resolve_filter_classes("bns") == {"BNS"}
    # cbc = all compact-binary classes
    assert {"BBH", "NSBH", "BNS"} <= resolve_filter_classes("cbc")
    # None = no restriction
    assert resolve_filter_classes(None) == set(SOURCE_CLASSES)
    # iterable of keywords
    assert resolve_filter_classes(["bbh", "nsbh"]) == {"BBH", "NSBH"}


def test_source_class_meta_far_absence_default():
    m = SourceClassMeta(event_name="GWx")
    assert m.source_class == "Unknown"
    assert m.far_available is False
    assert not np.isfinite(m.far)
    # supplying a finite far flips availability on
    m2 = SourceClassMeta(event_name="GWy", far=1e-3, source_class="bbh")
    assert m2.far_available is True
    assert m2.source_class == "BBH"


# ── selection tests on a mixed catalog ───────────────────────────────────────
def test_select_bbh_only(tmp_path):
    cat = GWCatalog(build_mixed_store(tmp_path, MIXED_EVENTS))
    bbh = cat.select(source_class="bbh")
    assert bbh.n_events == 2
    assert set(bbh.event_names) == {"GW900001_000001", "GW900002_000002"}
    assert set(bbh.source_class[bbh._sel]) == {"BBH"}


def test_select_nsbh_only(tmp_path):
    cat = GWCatalog(build_mixed_store(tmp_path, MIXED_EVENTS))
    nsbh = cat.select(source_class="nsbh")
    assert nsbh.n_events == 1
    assert list(nsbh.event_names) == ["GW900003_000003"]


def test_select_bns_only(tmp_path):
    cat = GWCatalog(build_mixed_store(tmp_path, MIXED_EVENTS))
    bns = cat.select(source_class="bns")
    assert bns.n_events == 1
    assert list(bns.event_names) == ["GW900004_000004"]


def test_select_all_cbc(tmp_path):
    cat = GWCatalog(build_mixed_store(tmp_path, MIXED_EVENTS))
    cbc = cat.select(source_class="cbc")
    # all five compact-binary events (incl. MassGap) are retained
    assert cbc.n_events == len(MIXED_EVENTS)
    assert set(cbc.event_names) == {e["name"] for e in MIXED_EVENTS}


def test_select_source_class_falls_back_to_compact_type(tmp_path):
    """A store lacking a source_class column still filters via compact_type."""
    path = build_mixed_store(tmp_path, MIXED_EVENTS)
    # rewrite without the source_class column to emulate a legacy store
    with h5py.File(path, "a") as f:
        del f["meta/source_class"]
    cat = GWCatalog(path)
    assert cat.select(source_class="bbh").n_events == 2
    assert cat.select(source_class="bns").n_events == 1


def test_source_class_iterable_filter(tmp_path):
    cat = GWCatalog(build_mixed_store(tmp_path, MIXED_EVENTS))
    sub = cat.select(source_class=["bbh", "bns"])
    assert sub.n_events == 3
    assert set(sub.source_class[sub._sel]) == {"BBH", "BNS"}


def test_to_darksirens_records_source_class_filter(tmp_path):
    cat = GWCatalog(build_mixed_store(tmp_path, MIXED_EVENTS))
    out = tmp_path / "bbh_export.h5"
    cat.to_darksirens(str(out), source_class="bbh", nsamp=8, seed=0,
                      cosmology=(67.74, 0.3089))
    with h5py.File(out, "r") as f:
        assert f.attrs["nobs"] == 2
        assert f.attrs["source_class_filter"] == "bbh"
        assert set(f.attrs["event_names"]) == {"GW900001_000001",
                                               "GW900002_000002"}
