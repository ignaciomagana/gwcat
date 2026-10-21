"""Regression tests for the 2026-10-20 review of the catalog layer.

Covers the event-selection view (``GWCatalog.select`` / ``SelectionSpec``),
the frozen v1 exporter and validator (``to_darksirens`` / ``validate_export``),
the source-class pairing judgement, the bundled BBH lists and the user
metadata overrides.  Each test starts from the review's failure scenario and
fails on the reviewed commit (3ba1dbf).

Fixtures are tiny synthetic stores written with h5py; one row per
``(event, sample set)``, so a multi-sample-set store is a list with repeated
names.  No network.
"""
import json
import warnings

import h5py
import numpy as np
import pytest

from gwcat.catalog import (GWCatalog, SelectionSpec, UnpairableSelectionCut,
                           validate_export)

from test_selection_cbc import _make_combined_selection

_COSMO = (67.74, 0.3089)
_PARAMS = ["mass_1", "mass_2", "luminosity_distance", "ra", "dec", "chi_eff",
           "p_dL_pe"]
_STR_META = ("source_class", "compact_type", "sample_set_name", "approximant",
             "waveform", "prior_source_kind_spin", "mass_prior_kind")
_FLOAT_META = ("far", "far_available", "pastro", "snr_med", "m1_src_med",
               "m2_src_med", "sky_area_90", "is_preferred", "is_mixed",
               "priority_rank", "spin_amax_1", "spin_amax_2", "dL_prior_H0",
               "dL_prior_Om0")


def _row(name, **kw):
    """One store row; every meta field has an explicit default."""
    r = {"name": name, "source_class": "BBH", "far": 1e-3, "pastro": 0.99,
         "snr_med": 12.0, "m1_src_med": 30.0, "m2_src_med": 20.0,
         "sky_area_90": 50.0, "is_preferred": 1.0, "is_mixed": 0.0,
         "priority_rank": 0.0, "spin_amax_1": 0.99, "spin_amax_2": 0.99,
         "sample_set_name": "C01:IMRPhenomXPHM",
         "approximant": "IMRPhenomXPHM",
         "prior_source_kind_spin": "own_analytic",
         "mass_prior_kind": "uniform_detector_frame", "n": 40}
    r.update(kw)
    r.setdefault("compact_type", r["source_class"])
    r.setdefault("waveform", r["approximant"])
    return r


def _store(tmp_path, rows, name="store.h5", seed=3, drop_meta=()):
    rng = np.random.default_rng(seed)
    offsets = [0]
    cols = {p: [] for p in _PARAMS}
    for r in rows:
        n = int(r["n"])
        cols["mass_1"].append(r.get("mass_1", rng.uniform(25, 45, n)))
        cols["mass_2"].append(r.get("mass_2", rng.uniform(10, 22, n)))
        cols["luminosity_distance"].append(
            r.get("luminosity_distance", rng.uniform(300, 800, n)))
        cols["ra"].append(rng.uniform(0, 2 * np.pi, n))
        cols["dec"].append(rng.uniform(-np.pi / 2, np.pi / 2, n))
        cols["chi_eff"].append(r.get("chi_eff", rng.uniform(-0.3, 0.3, n)))
        cols["p_dL_pe"].append(rng.uniform(0.1, 1.0, n))
        offsets.append(offsets[-1] + n)
    path = tmp_path / name
    with h5py.File(path, "w") as f:
        f.attrs["param_names"] = np.array(_PARAMS, dtype=h5py.string_dtype())
        idx = f.create_group("index")
        idx.create_dataset("offsets", data=np.array(offsets, dtype="i8"))
        idx.create_dataset("event_names", data=np.array(
            [r["name"] for r in rows], dtype=h5py.string_dtype()))
        mg = f.create_group("meta")
        for k in _STR_META:
            if k not in drop_meta:
                mg.create_dataset(k, data=np.array(
                    [str(r[k]) for r in rows], dtype=h5py.string_dtype()))
        for k in _FLOAT_META:
            if k in drop_meta:
                continue
            if k == "far_available":
                vals = [1.0 if np.isfinite(r["far"]) else 0.0 for r in rows]
            elif k in ("dL_prior_H0", "dL_prior_Om0"):
                vals = [_COSMO[0] if k == "dL_prior_H0" else _COSMO[1]
                        for _ in rows]
            else:
                vals = [float(r[k]) for r in rows]
            mg.create_dataset(k, data=np.asarray(vals, dtype="f8"))
        sg = f.create_group("samples")
        for p in _PARAMS:
            sg.create_dataset(p, data=np.concatenate(cols[p]))
    return str(path)


def _three(tmp_path, **kw):
    rows = [_row("GW900001_000001", m1_src_med=25.0),
            _row("GW900002_000002", m1_src_med=60.0),
            _row("GW900003_000003", m1_src_med=90.0)]
    return _store(tmp_path, rows, **kw)


def _attr(f, k):
    v = f.attrs[k]
    return v.decode() if isinstance(v, bytes) else v


# ==========================================================================
# Cuts on posterior-median masses or sky area are refused at export
# ==========================================================================
@pytest.mark.parametrize("cut", [{"m1_src_range": (20.0, 70.0)},
                                 {"m2_src_range": (3.0, 1e9)},
                                 {"sky_area_max": 100.0}])
def test_point_estimate_cuts_are_refused_by_both_exporters(tmp_path, cut):
    view = GWCatalog(_three(tmp_path)).select(**cut)
    with pytest.raises(UnpairableSelectionCut, match="POINT ESTIMATES"):
        view.to_darksirens(str(tmp_path / "v1.h5"), nsamp=8, seed=0,
                           cosmology=_COSMO)
    with pytest.raises(UnpairableSelectionCut, match="POINT ESTIMATES"):
        view.export(str(tmp_path / "v2.h5"), spin_basis="chieff", nsamp=8,
                    seed=0, cosmology=_COSMO)
    assert not (tmp_path / "v1.h5").exists()
    assert not (tmp_path / "v2.h5").exists()


def test_a_declared_point_estimate_cut_is_written_recorded_and_refused_in_a_pair(
        tmp_path):
    cat = GWCatalog(_three(tmp_path))
    view = cat.select(m1_src_range=(20.0, 70.0),
                      allow_unpaired_point_estimate_cuts=True)
    # The declaration survives a further select() (the exporters re-select).
    out = tmp_path / "pe.h5"
    view.select(far_max=1.0).to_darksirens(str(out), nsamp=8, seed=0,
                                           cosmology=_COSMO)
    with h5py.File(out, "r") as f:
        assert _attr(f, "point_estimate_cuts") == "m1_src_range"
        spec = json.loads(_attr(f, "selection_spec"))
        assert spec["point_estimate_cuts"] == ["m1_src_range"]
        assert spec["point_estimate_cuts_allowed"] is True
        # The median-mass range is a class cut under another spelling.
        assert _attr(f, "source_class_cut_estimator") == "posterior_median_mass"
        assert int(f.attrs["nobs"]) == 2
    sel = _make_combined_selection(tmp_path)
    with pytest.raises(ValueError, match="xcheck_point_estimate_cuts"):
        validate_export(str(out), sel)


def test_the_unfiltered_spec_record_is_unchanged():
    """The new keys appear only when such a cut is in effect."""
    spec = SelectionSpec().refine(far_max=1.0)
    d = spec.to_dict()
    assert "point_estimate_cuts" not in d
    assert "point_estimate_cuts" not in spec.to_attrs()
