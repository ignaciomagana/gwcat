"""Tests for the unified `gwcat` CLI (PR 10).

Covers, per subcommand, argument parsing + a happy path on tiny synthetic
fixtures (invoking ``gwcat.cli.main`` directly with an argv list -- no
subprocess, no network):

  * ``inspect``            -- table + ``--json`` diagnostics on a mixed store.
  * ``export-darksirens``  -- GWCatalog.to_darksirens, including the
                              missing-FAR contract (handoff test case 14) and
                              --no-summary.
  * ``selection``          -- SelectionSet / CombinedSelectionSet.to_darksirens.
  * ``validate``           -- exit code reflects validate_export's result.
  * the acceptance criterion: ONE command sequence (build fixture store ->
    `export-darksirens` -> `validate`) runs end-to-end.
  * the deprecated ``gwcat-ingest``/``gwcat-fetch`` wrappers still work and
    print the deprecation pointer on stderr before delegating unchanged.

Fixtures are tiny synthetic HDF5 files built directly with h5py, matching the
schema GWCatalog/SelectionSet read (same approach as
tests/test_source_class_filters.py and tests/test_selection_cbc.py) -- no
network access anywhere in this file.
"""
import json
from pathlib import Path

import numpy as np
import h5py
import pytest

from gwcat.cli import main
from gwcat.catalog import GWCatalog


# ==========================================================================
# Fixture builders
# ==========================================================================
DARKSIRENS_PARAMS = ["mass_1", "mass_2", "luminosity_distance", "ra", "dec",
                     "chi_eff", "p_dL_pe"]


def _build_mixed_store(tmp_path, events, H0=67.74, Om0=0.3089, seed=31,
                       name="store.h5"):
    """Tiny synthetic store with source-class + FAR + per-event-cosmology
    metadata (matches the on-disk schema GWCatalog reads)."""
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
    {"name": "GW910001_000001", "source_class": "BBH", "far": 5e-4,
     "pastro": 0.99},
    {"name": "GW910002_000002", "source_class": "BBH", "far": float("nan")},
    {"name": "GW910003_000003", "source_class": "NSBH", "far": 2e-3,
     "pastro": 0.95},
    {"name": "GW910004_000004", "source_class": "BNS", "far": 1e-6,
     "pastro": 0.999},
]

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
    """Tiny O4 'events'-format injection file (same shape as
    tests/test_selection_cbc.py's write_o4)."""
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


def _summary_json(out_path) -> dict:
    return json.loads(Path(str(out_path) + ".validation_summary.json").read_text())


# ==========================================================================
# inspect
# ==========================================================================
def test_inspect_happy_path(tmp_path, capsys):
    store = _build_mixed_store(tmp_path, MIXED_EVENTS)
    rc = main(["inspect", store])
    assert rc == 0
    out = capsys.readouterr().out
    assert "4 events" in out
    assert "far_missing_count: 1 / 4" in out
    assert "p_astro_available_count: 3 / 4" in out


def test_inspect_json(tmp_path, capsys):
    store = _build_mixed_store(tmp_path, MIXED_EVENTS)
    rc = main(["inspect", store, "--json"])
    assert rc == 0
    info = json.loads(capsys.readouterr().out)
    assert info["n_events"] == 4
    assert info["source_class_counts"] == {"BBH": 2, "NSBH": 1, "BNS": 1}
    assert info["far_missing_count"] == 1
    assert info["far_available_count"] == 3
    assert info["p_astro_available_count"] == 3
    assert info["missing_required_parameters"] == []
    assert "mass_ratio" in info["missing_optional_parameters"]


# ==========================================================================
# export-darksirens
# ==========================================================================
def test_export_darksirens_happy_path(tmp_path):
    store = _build_mixed_store(tmp_path, MIXED_EVENTS)
    out = tmp_path / "gw_bbh.h5"
    rc = main(["export-darksirens", store, "--out", str(out),
              "--source-class", "bbh", "--far-max", "1.0",
              "--allow-missing-far", "--cosmology", "67.74,0.3089",
              "--nsamp", "8", "--seed", "0"])
    assert rc == 0
    assert out.exists()

    summary = _summary_json(out)
    assert summary["kind"] == "darksirens_export"
    assert summary["n_events_exported"] == 2
    assert summary["n_events_missing_far"] == 1
    assert summary["far_policy"] == "allow_missing"
    assert summary["source_class_filter"] == "bbh"
    assert summary["spin_prior_mode"] == "include"
    assert summary["cosmology_mode"] == "override"

    md_path = Path(str(out) + ".validation_summary.md")
    assert md_path.exists()
    md_text = md_path.read_text()
    assert "n_events_missing_far" in md_text
    assert "darksirens_export" in md_text


def test_export_darksirens_no_summary_flag(tmp_path):
    store = _build_mixed_store(tmp_path, MIXED_EVENTS)
    out = tmp_path / "gw_bbh_nosum.h5"
    rc = main(["export-darksirens", store, "--out", str(out),
              "--source-class", "bbh", "--cosmology", "67.74,0.3089",
              "--nsamp", "8", "--seed", "0", "--no-summary"])
    assert rc == 0
    assert out.exists()
    assert not Path(str(out) + ".validation_summary.json").exists()


def test_export_darksirens_require_far_fails_loud(tmp_path):
    store = _build_mixed_store(tmp_path, MIXED_EVENTS)
    out = tmp_path / "should_not_exist.h5"
    with pytest.raises(ValueError, match="require_far"):
        main(["export-darksirens", store, "--out", str(out),
              "--source-class", "bbh", "--far-max", "1.0",
              "--require-far", "--cosmology", "67.74,0.3089",
              "--nsamp", "8", "--seed", "0"])
    assert not out.exists()


def test_export_darksirens_comma_separated_source_class(tmp_path):
    store = _build_mixed_store(tmp_path, MIXED_EVENTS)
    out = tmp_path / "gw_nsbh_bns.h5"
    rc = main(["export-darksirens", store, "--out", str(out),
              "--source-class", "nsbh,bns", "--cosmology", "67.74,0.3089",
              "--nsamp", "8", "--seed", "0", "--no-summary"])
    assert rc == 0
    with h5py.File(out, "r") as f:
        assert set(f.attrs["event_names"]) == {"GW910003_000003",
                                               "GW910004_000004"}


# ==========================================================================
# selection
# ==========================================================================
def test_selection_single_file(tmp_path):
    inj = _write_o4_injections(tmp_path / "inj.hdf", {"BBH": 4, "NSBH": 2})
    out = tmp_path / "sel.h5"
    rc = main(["selection", "--injections", inj, "--out", str(out),
              "--far-threshold", "1.0", "--source-class", "bbh"])
    assert rc == 0
    assert out.exists()
    summary = _summary_json(out)
    assert summary["kind"] == "selection_export"
    assert summary["n_detected"] == 4
    assert summary["n_campaigns"] == 1
    assert summary["source_class_filter"] == "bbh"


def test_selection_combined_multiple_files(tmp_path):
    inj1 = _write_o4_injections(tmp_path / "inj1.hdf", {"BBH": 4})
    inj2 = _write_o4_injections(tmp_path / "inj2.hdf", {"BBH": 3})
    out = tmp_path / "sel_combined.h5"
    rc = main(["selection", "--injections", inj1, inj2, "--out", str(out)])
    assert rc == 0
    summary = _summary_json(out)
    assert summary["n_campaigns"] == 2
    assert summary["n_detected"] == 7


# ==========================================================================
# validate
# ==========================================================================
def test_validate_happy_path(tmp_path):
    store = _build_mixed_store(tmp_path, MIXED_EVENTS)
    pe = tmp_path / "gw_bbh.h5"
    assert main(["export-darksirens", store, "--out", str(pe),
                "--source-class", "bbh", "--cosmology", "67.74,0.3089",
                "--nsamp", "8", "--seed", "0", "--no-summary"]) == 0

    inj = _write_o4_injections(tmp_path / "inj.hdf", {"BBH": 4})
    sel = tmp_path / "sel.h5"
    assert main(["selection", "--injections", inj, "--out", str(sel),
                "--source-class", "bbh", "--H0", "67.74", "--Om0", "0.3089",
                "--no-summary"]) == 0

    assert main(["validate", str(pe), str(sel)]) == 0


def test_validate_mismatch_returns_nonzero(tmp_path):
    store = _build_mixed_store(tmp_path, MIXED_EVENTS)
    pe = tmp_path / "gw_bbh.h5"
    assert main(["export-darksirens", store, "--out", str(pe),
                "--source-class", "bbh", "--cosmology", "67.74,0.3089",
                "--nsamp", "8", "--seed", "0", "--no-summary"]) == 0

    inj = _write_o4_injections(tmp_path / "inj.hdf", {"BBH": 4})
    sel = tmp_path / "sel.h5"
    # No --source-class on the selection: mismatches the PE export's "bbh".
    assert main(["selection", "--injections", inj, "--out", str(sel),
                "--H0", "67.74", "--Om0", "0.3089", "--no-summary"]) == 0

    assert main(["validate", str(pe), str(sel)]) == 1


# ==========================================================================
# Acceptance criterion: fixture store -> export-darksirens -> validate, e2e.
# ==========================================================================
def test_end_to_end_fixture_store_to_validated_export(tmp_path):
    """PR10 acceptance criterion: ONE command sequence goes from a fixture
    store to a validated export -- build the store, `export-darksirens`, then
    `validate` -- asserting exit 0 and the expected outputs."""
    store = _build_mixed_store(tmp_path, MIXED_EVENTS)
    out = tmp_path / "e2e_gw_bbh.h5"

    rc_export = main(["export-darksirens", store, "--out", str(out),
                      "--source-class", "bbh", "--far-max", "1.0",
                      "--allow-missing-far", "--cosmology", "67.74,0.3089",
                      "--nsamp", "16", "--seed", "0"])
    assert rc_export == 0
    assert out.exists()
    assert Path(str(out) + ".validation_summary.json").exists()
    assert Path(str(out) + ".validation_summary.md").exists()

    rc_validate = main(["validate", str(out)])
    assert rc_validate == 0

    # Sanity: the source store is untouched by the export.
    cat = GWCatalog(store)
    assert cat.n_events == 4


# ==========================================================================
# main() argument-parsing edge cases
# ==========================================================================
def test_main_no_args_exits_nonzero():
    with pytest.raises(SystemExit):
        main([])


def test_main_unknown_command_exits_nonzero():
    with pytest.raises(SystemExit):
        main(["bogus-command"])


# ==========================================================================
# Deprecated gwcat-ingest / gwcat-fetch wrappers still work + warn
# ==========================================================================
def test_ingest_deprecated_wrapper_warns_and_delegates(capsys):
    from gwcat import ingest as ing
    with pytest.raises(SystemExit):
        ing._cli(argv=["--glob", "/nonexistent/path/*.h5"])
    err = capsys.readouterr().err
    assert "deprecated" in err
    assert "gwcat ingest" in err
    assert "no files matched" in err


def test_fetch_deprecated_wrapper_warns_and_delegates(capsys):
    from gwcat import fetch as ft
    with pytest.raises(SystemExit):
        ft._cli(argv=["--catalog", "not-a-real-catalog"])
    err = capsys.readouterr().err
    assert "deprecated" in err
    assert "gwcat fetch" in err


def test_cli_ingest_dispatch_does_not_print_deprecation(capsys):
    """`gwcat ingest` (the unified CLI path) must NOT print the deprecation
    notice -- only the standalone `gwcat-ingest` script does."""
    with pytest.raises(SystemExit):
        main(["ingest", "--glob", "/nonexistent/path/*.h5"])
    err = capsys.readouterr().err
    assert "deprecated" not in err
    assert "no files matched" in err
