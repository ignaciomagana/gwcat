"""Tests for gwcat.validation_summary (PR 10).

Covers:
  * ``value_counts`` / ``package_version`` unit behavior.
  * ``summarize_catalog`` on a known mixed-source-class fixture (exact counts),
    including that it respects an already-``select()``ed subset.
  * ``render_markdown`` renders the SAME data as ``write_validation_summary``'s
    JSON (not an independent computation).
  * the ``write_summary=True`` hooks in ``build_store``,
    ``GWCatalog.to_darksirens`` (incl. the missing-FAR contract -- handoff
    test case 14), and ``SelectionSet``/``CombinedSelectionSet.to_darksirens``
    produce the documented keys with correct values.
  * ``write_summary`` defaults to False everywhere: no extra files appear
    unless explicitly requested (existing behavior unchanged).

Fixtures are tiny synthetic HDF5 files/an in-process fake PESummary reader,
following the same approach as tests/test_source_class_filters.py,
tests/test_waveform_policy.py, and tests/test_selection_cbc.py -- no network
access anywhere in this file.
"""
import json
from pathlib import Path

import numpy as np
import h5py
import pytest

from gwcat.catalog import GWCatalog
from gwcat.selection import SelectionSet, CombinedSelectionSet
from gwcat.validation_summary import (value_counts, summarize_catalog,
                                      render_markdown, write_validation_summary,
                                      package_version)


# ==========================================================================
# Fixture builders (mirrors tests/test_cli.py's, kept self-contained per the
# suite's existing per-file-fixture convention)
# ==========================================================================
DARKSIRENS_PARAMS = ["mass_1", "mass_2", "luminosity_distance", "ra", "dec",
                     "chi_eff", "p_dL_pe"]


def _build_mixed_store(tmp_path, events, H0=67.74, Om0=0.3089, seed=41,
                       name="store.h5"):
    rng = np.random.default_rng(seed)
    offsets = [0]
    cols = {p: [] for p in DARKSIRENS_PARAMS}
    meta = {k: [] for k in ["source_class", "compact_type", "far",
                            "far_available", "pastro", "p_astro",
                            "dL_prior_H0", "dL_prior_Om0"]}
    names = []
    for ev in events:
        n = int(ev.get("n", 30))
        cols["mass_1"].append(rng.uniform(20, 45, n))
        cols["mass_2"].append(rng.uniform(8, 20, n))
        cols["luminosity_distance"].append(rng.uniform(300, 800, n))
        cols["ra"].append(rng.uniform(0, 2 * np.pi, n))
        cols["dec"].append(rng.uniform(-np.pi / 2, np.pi / 2, n))
        cols["chi_eff"].append(rng.uniform(-0.3, 0.3, n))
        cols["p_dL_pe"].append(rng.uniform(0.1, 1.0, n))
        offsets.append(offsets[-1] + n)

        far = float(ev.get("far", np.nan))
        names.append(ev["name"])
        meta["source_class"].append(ev["source_class"])
        meta["compact_type"].append(ev["source_class"])
        meta["far"].append(far)
        meta["far_available"].append(1.0 if np.isfinite(far) else 0.0)
        meta["pastro"].append(float(ev.get("pastro", np.nan)))
        meta["p_astro"].append(float(ev.get("pastro", np.nan)))
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
        for k in ["far", "far_available", "pastro", "p_astro",
                  "dL_prior_H0", "dL_prior_Om0"]:
            mg.create_dataset(k, data=np.asarray(meta[k], dtype="f8"))
        sg = f.create_group("samples")
        for p in DARKSIRENS_PARAMS:
            sg.create_dataset(p, data=np.concatenate(cols[p]))
    return str(path)


#: 2 BBH (one with NO FAR -- far_available=False), 1 NSBH, 1 BNS.
MIXED_EVENTS = [
    {"name": "GW920001_000001", "source_class": "BBH", "far": 5e-4,
     "pastro": 0.99},
    {"name": "GW920002_000002", "source_class": "BBH", "far": float("nan")},
    {"name": "GW920003_000003", "source_class": "NSBH", "far": 2e-3,
     "pastro": 0.95},
    {"name": "GW920004_000004", "source_class": "BNS", "far": 1e-6,
     "pastro": 0.999},
]


def _summary_json(out_path) -> dict:
    return json.loads(Path(str(out_path) + ".validation_summary.json").read_text())


# ==========================================================================
# value_counts / package_version
# ==========================================================================
def test_value_counts_basic():
    assert value_counts(["BBH", "BBH", "", "NSBH"]) == {
        "BBH": 2, "NSBH": 1, "unknown": 1}


def test_value_counts_decodes_bytes():
    assert value_counts([b"BBH", b"BBH", b"BNS"]) == {"BBH": 2, "BNS": 1}


def test_value_counts_empty():
    assert value_counts([]) == {}


def test_package_version_returns_nonempty_string():
    v = package_version()
    assert isinstance(v, str) and v


# ==========================================================================
# summarize_catalog
# ==========================================================================
def test_summarize_catalog_fields(tmp_path):
    store = _build_mixed_store(tmp_path, MIXED_EVENTS)
    cat = GWCatalog(store)
    info = summarize_catalog(cat)

    assert info["n_events"] == 4
    assert info["source_class_counts"] == {"BBH": 2, "NSBH": 1, "BNS": 1}
    assert info["far_missing_count"] == 1
    assert info["far_available_count"] == 3
    assert info["p_astro_available_count"] == 3
    assert info["missing_required_parameters"] == []
    assert "mass_ratio" in info["missing_optional_parameters"]
    assert info["per_event_cosmology_present"] is True
    assert info["per_event_cosmology_varies"] is False
    assert info["package_version"]
    assert "schema_version" in info


def test_summarize_catalog_respects_selection_subset(tmp_path):
    store = _build_mixed_store(tmp_path, MIXED_EVENTS)
    cat = GWCatalog(store)
    bbh = cat.select(source_class="bbh")
    info = summarize_catalog(bbh)
    assert info["n_events"] == 2
    assert info["source_class_counts"] == {"BBH": 2}
    # one of the two selected BBH events has no FAR
    assert info["far_missing_count"] == 1


# ==========================================================================
# render_markdown / write_validation_summary
# ==========================================================================
def test_render_markdown_matches_json_content(tmp_path):
    summary = {
        "kind": "ingest", "output_path": "foo.h5",
        "n_events": 3, "source_class_counts": {"BBH": 2, "BNS": 1},
        "event_names": ["a", "b", "c"], "missing_required_parameters": [],
    }
    json_path, md_path = write_validation_summary(tmp_path / "foo.h5", summary)
    assert Path(json_path).exists() and Path(md_path).exists()

    loaded = json.loads(Path(json_path).read_text())
    assert loaded == summary

    md_text = Path(md_path).read_text()
    assert "ingest" in md_text
    assert "n_events" in md_text and "3" in md_text
    assert "BBH" in md_text and "2" in md_text
    # .md is a rendering of the SAME dict passed to write_validation_summary
    assert render_markdown(summary) == md_text


def test_write_validation_summary_no_md(tmp_path):
    json_path, md_path = write_validation_summary(
        tmp_path / "bar.h5", {"kind": "x", "output_path": "bar.h5"},
        write_md=False)
    assert md_path is None
    assert Path(json_path).exists()
    assert not Path(str(tmp_path / "bar.h5") + ".validation_summary.md").exists()


# ==========================================================================
# build_store(write_summary=True)
# ==========================================================================
def _fake_pesummary_analyses(seed=0, n=15):
    rng = np.random.default_rng(seed)
    return {"C00:Mixed": {
        "luminosity_distance": rng.uniform(300, 800, n),
        "mass_1": rng.uniform(25, 50, n),
        "mass_2": rng.uniform(10, 25, n),
        "ra": rng.uniform(0, 2 * np.pi, n),
        "dec": rng.uniform(-np.pi / 2, np.pi / 2, n),
        "chi_eff": rng.uniform(-0.4, 0.4, n),
    }}


class _FakeData:
    """Minimal stand-in for a pesummary read() result (no config -> f_ref NaN)."""


def test_build_store_write_summary(tmp_path, monkeypatch):
    import gwcat.ingest as ing

    analyses = _fake_pesummary_analyses(seed=1)

    def _fake_reader(path):
        return _FakeData(), analyses, list(analyses.keys()), {}

    monkeypatch.setattr(ing, "_read_event_pesummary", _fake_reader)

    path = tmp_path / "GWTC-5_GW920101_010101_cosmo.h5"
    path.write_bytes(b"")  # existence only; the reader is faked
    out = tmp_path / "store.h5"
    ing.build_store([str(path)], str(out), event_table={},
                    cfg=ing.IngestConfig(validate_prior=False),
                    write_summary=True)

    assert Path(str(out) + ".validation_summary.json").exists()
    assert Path(str(out) + ".validation_summary.md").exists()
    summary = _summary_json(out)
    assert summary["kind"] == "ingest"
    assert summary["n_files_provided"] == 1
    assert summary["n_unique_events_ingested"] == 1
    assert summary["n_rows_ingested"] == 1
    assert summary["package_version"]
    assert "schema_version" in summary


def test_build_store_default_no_summary_files(tmp_path, monkeypatch):
    """write_summary defaults to False: no extra files unless opted in."""
    import gwcat.ingest as ing

    analyses = _fake_pesummary_analyses(seed=2)

    def _fake_reader(path):
        return _FakeData(), analyses, list(analyses.keys()), {}

    monkeypatch.setattr(ing, "_read_event_pesummary", _fake_reader)

    path = tmp_path / "GWTC-5_GW920102_020202_cosmo.h5"
    path.write_bytes(b"")
    out = tmp_path / "store_nosum.h5"
    ing.build_store([str(path)], str(out), event_table={},
                    cfg=ing.IngestConfig(validate_prior=False))
    assert not Path(str(out) + ".validation_summary.json").exists()


# ==========================================================================
# GWCatalog.to_darksirens(write_summary=True) -- incl. handoff test case 14
# ==========================================================================
def test_to_darksirens_write_summary_records_missing_far(tmp_path):
    """Handoff test case 14: validation_summary.json/.md record skipped
    events and missing FAR metadata."""
    store = _build_mixed_store(tmp_path, MIXED_EVENTS)
    cat = GWCatalog(store)
    out = tmp_path / "gw_bbh.h5"
    cat.to_darksirens(str(out), source_class="bbh", far_max=1.0,
                      allow_missing_far=True, cosmology=(67.74, 0.3089),
                      nsamp=8, seed=0, write_summary=True)

    summary = _summary_json(out)
    assert summary["kind"] == "darksirens_export"
    assert summary["n_events_considered"] == 2
    assert summary["n_events_exported"] == 2
    assert summary["n_events_missing_far"] == 1
    assert summary["far_policy"] == "allow_missing"
    assert summary["source_class_filter"] == "bbh"
    assert summary["spin_prior_mode"] == "include"
    assert summary["cosmology_mode"] == "override"
    assert summary["cosmology_override_used"] is True
    assert set(summary["event_names_exported"]) == {"GW920001_000001",
                                                     "GW920002_000002"}

    md_text = Path(str(out) + ".validation_summary.md").read_text()
    assert "n_events_missing_far" in md_text
    assert "far_policy" in md_text
    assert "allow_missing" in md_text


def test_to_darksirens_default_no_summary(tmp_path):
    store = _build_mixed_store(tmp_path, MIXED_EVENTS)
    cat = GWCatalog(store)
    out = tmp_path / "gw_bbh_nosum.h5"
    cat.to_darksirens(str(out), source_class="bbh", cosmology=(67.74, 0.3089),
                      nsamp=8, seed=0)
    assert not Path(str(out) + ".validation_summary.json").exists()


def test_to_darksirens_summary_context_overlay(tmp_path):
    store = _build_mixed_store(tmp_path, MIXED_EVENTS)
    cat = GWCatalog(store)
    out = tmp_path / "gw_bbh_ctx.h5"
    cat.to_darksirens(str(out), source_class="bbh", cosmology=(67.74, 0.3089),
                      nsamp=8, seed=0, write_summary=True,
                      summary_context={"release_manifest": "gwtc-5-fake"})
    summary = _summary_json(out)
    assert summary["release_manifest"] == "gwtc-5-fake"


# ==========================================================================
# SelectionSet / CombinedSelectionSet .to_darksirens(write_summary=True)
# ==========================================================================
_O4_FIELDS = [
    ("mass1_source", "f8"), ("mass2_source", "f8"),
    ("mass1_detector", "f8"), ("mass2_detector", "f8"),
    ("luminosity_distance", "f8"), ("z", "f8"),
    ("dluminosity_distance_dredshift", "f8"),
    ("right_ascension", "f8"), ("declination", "f8"),
    ("spin1x", "f8"), ("spin1y", "f8"), ("spin1z", "f8"),
    ("spin2x", "f8"), ("spin2y", "f8"), ("spin2z", "f8"),
    ("chi_eff", "f8"), ("weights", "f8"),
    ("lnpdraw_mass1_source", "f8"),
    ("lnpdraw_mass2_source_GIVEN_mass1_source", "f8"),
    ("lnpdraw_z", "f8"),
    ("pycbc_far", "f8"),
]
_CLASS_MASSES = {"BBH": (35.0, 30.0), "NSBH": (12.0, 1.4), "BNS": (1.6, 1.3)}


def _write_o4_injections(path, class_counts, detected=True,
                         total_generated=1000):
    rows = []
    for cls, n in class_counts.items():
        m1s, m2s = _CLASS_MASSES[cls]
        rows.extend([(m1s, m2s)] * n)
    n = len(rows)
    ev = np.zeros(n, dtype=_O4_FIELDS)
    z = 0.1
    for i, (m1s, m2s) in enumerate(rows):
        ev["mass1_source"][i] = m1s
        ev["mass2_source"][i] = m2s
        ev["z"][i] = z
        ev["mass1_detector"][i] = m1s * (1 + z)
        ev["mass2_detector"][i] = m2s * (1 + z)
        ev["luminosity_distance"][i] = 450.0
        ev["dluminosity_distance_dredshift"][i] = 4500.0
        ev["right_ascension"][i] = 1.0
        ev["declination"][i] = 0.5
        ev["spin1z"][i] = 0.2
        ev["spin2z"][i] = -0.1
        ev["chi_eff"][i] = (m1s * 0.2 + m2s * -0.1) / (m1s + m2s)
        ev["weights"][i] = 2.0
        ev["lnpdraw_mass1_source"][i] = -1.0
        ev["lnpdraw_mass2_source_GIVEN_mass1_source"][i] = -2.0
        ev["lnpdraw_z"][i] = -3.0
    far = 0.1 if detected else 5.0
    ev["pycbc_far"] = far
    with h5py.File(path, "w") as f:
        f.attrs["total_analysis_time"] = 365.25 * 24 * 3600
        f.attrs["total_generated"] = total_generated
        f.attrs["searches"] = np.array([b"pycbc"])
        f.create_dataset("events", data=ev)
    return str(path)


def test_selection_set_write_summary(tmp_path):
    inj = _write_o4_injections(tmp_path / "inj.hdf", {"BBH": 4, "NSBH": 2})
    sel = SelectionSet(inj)
    out = tmp_path / "sel.h5"
    sel.to_darksirens(str(out), far_threshold=1.0, source_class="bbh",
                      write_summary=True)

    summary = _summary_json(out)
    assert summary["kind"] == "selection_export"
    assert summary["n_detected"] == 4
    assert summary["n_campaigns"] == 1
    assert summary["source_class_filter"] == "bbh"
    assert summary["source_class_counts_detected"] == {"BBH": 4}
    assert summary["p_astro_available"] is False
    assert summary["spin_prior_mode"] == "include"


def test_selection_set_default_no_summary(tmp_path):
    inj = _write_o4_injections(tmp_path / "inj.hdf", {"BBH": 3})
    sel = SelectionSet(inj)
    out = tmp_path / "sel_nosum.h5"
    sel.to_darksirens(str(out), far_threshold=1.0)
    assert not Path(str(out) + ".validation_summary.json").exists()


def test_combined_selection_set_write_summary(tmp_path):
    inj1 = _write_o4_injections(tmp_path / "inj1.hdf", {"BBH": 4})
    inj2 = _write_o4_injections(tmp_path / "inj2.hdf", {"BBH": 3})
    combined = CombinedSelectionSet([SelectionSet(inj1), SelectionSet(inj2)])
    out = tmp_path / "sel_combined.h5"
    combined.to_darksirens(str(out), far_threshold=1.0, write_summary=True)

    summary = _summary_json(out)
    assert summary["kind"] == "selection_export"
    assert summary["n_campaigns"] == 2
    assert summary["n_detected"] == 7
    assert summary["campaign_ndraws"] == [1000, 1000]
