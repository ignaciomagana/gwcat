"""Review 2026-10-20, selection package: regression tests from the findings'
failure scenarios (gwcat/selection.py, gwcat/selection_spin.py,
gwcat/observing_runs.py).

Each test is written from the review's failure scenario and fails on 3ba1dbf.
Synthetic fixtures only (the ``tests/test_selection_spin.py`` writers); no
real LVK file is read.
"""
import json
import warnings

import h5py
import numpy as np
import pytest

from gwcat.selection import SelectionSet, CombinedSelectionSet

from test_selection_spin import write_o4_full, write_endo3_full


def _declare_domain(path, **attrs):
    """Stamp endo3-style draw-domain attrs on a fixture's root."""
    with h5py.File(path, "r+") as f:
        for k, v in attrs.items():
            f.attrs[k] = v
    return path


# ======================================================================
# high-4: the O3 injection campaign's declared draw domain
# ======================================================================
def _o3_o4_pair(tmp_path):
    # The O3-like campaign declares m1, m2 <= 40 and z <= 0.3 (endo3 declares
    # 100 and 1.9); the O4-like one declares nothing and draws m1 to 45 and z
    # to 0.5, i.e. past both bounds.
    o3 = _declare_domain(write_endo3_full(tmp_path / "o3.hdf", n=400, seed=3),
                         min_mass1=2.0, max_mass1=40.0, min_mass2=2.0,
                         max_mass2=40.0, max_redshift=0.3, max_z=0.3)
    o4 = write_o4_full(tmp_path / "o4.hdf", n=300, seed=4)
    return o3, o4


def test_declared_draw_domain_is_read_and_recorded_per_campaign(tmp_path):
    o3, o4 = _o3_o4_pair(tmp_path)
    s3, s4 = SelectionSet(o3), SelectionSet(o4)
    assert s3.declared_draw_domain == {"mass1_source": [2.0, 40.0],
                                       "mass2_source": [2.0, 40.0],
                                       "redshift": [None, 0.3]}
    assert s4.declared_draw_domain == {"mass1_source": [None, None],
                                       "mass2_source": [None, None],
                                       "redshift": [None, None]}
    assert s4.draw_extent["mass1_source"][1] > 40.0

    out = tmp_path / "sel.h5"
    with pytest.warns(UserWarning, match="declared injection draw domains"):
        CombinedSelectionSet([s3, s4]).export(str(out),
                                              spin_basis="component")
    with h5py.File(out, "r") as f:
        rows = json.loads(f.attrs["injection_draw_domain_per_campaign"])
        assert rows[0]["declared"]["mass1_source"] == [2.0, 40.0]
        assert rows[0]["declared_source"] == "file_attrs"
        assert rows[1]["declared_source"] == "undeclared"
        assert bool(f.attrs["injection_draw_domain_mismatch"]) is True
        detail = json.loads(f.attrs["injection_draw_domain_mismatch_detail"])
        crossed = {(d["campaign"], d["quantity"], d["bound"]) for d in detail}
        assert (0, "mass1_source", "max") in crossed
        assert (0, "redshift", "max") in crossed
        assert f.attrs["injection_draw_domain_policy"] == "not_applied"
        assert json.loads(f.attrs["injection_draw_domain_applied"]) == {}
        # Default: nothing subset -- every detected row of both campaigns.
        assert int(f.attrs["n_detected"]) == 400 + 300


def test_draw_domain_intersection_subsets_and_keeps_ndraw(tmp_path):
    o3, o4 = _o3_o4_pair(tmp_path)
    s3, s4 = SelectionSet(o3), SelectionSet(o4)
    out = tmp_path / "sel_dom.h5"
    with pytest.warns(UserWarning, match="draw_domain='intersection'"):
        CombinedSelectionSet([s3, s4]).export(
            str(out), spin_basis="component", draw_domain="intersection")

    def inside(s):
        return ((s._m1src >= 2.0) & (s._m1src <= 40.0) & (s._m2src >= 2.0)
                & (s._m2src <= 40.0) & (s._z <= 0.3))

    n_in = int(inside(s3).sum() + inside(s4).sum())
    assert 0 < n_in < 700
    with h5py.File(out, "r") as f:
        assert int(f.attrs["n_detected"]) == n_in
        assert f["pdraw"].shape == (n_in,)
        assert float(np.max(f["m1src"][:])) <= 40.0
        assert float(np.max(f["redshift"][:])) <= 0.3
        # Subsetting, not reweighting: ndraw is the campaigns' total draws.
        assert int(f.attrs["ndraw"]) == s3.n_generated + s4.n_generated
        assert f.attrs["injection_draw_domain_policy"] == "intersection"
        assert json.loads(f.attrs["injection_draw_domain_applied"]) == {
            "mass1_source": [2.0, 40.0], "mass2_source": [2.0, 40.0],
            "redshift": [None, 0.3]}


def test_draw_domain_intersection_refused_without_any_declaration(tmp_path):
    o4 = write_o4_full(tmp_path / "o4.hdf", n=50, seed=5)
    with pytest.raises(ValueError, match="declares a draw domain"):
        SelectionSet(o4).export(str(tmp_path / "x.h5"),
                                spin_basis="component",
                                draw_domain="intersection")


def test_v1_combined_export_records_the_draw_domains(tmp_path):
    o3, o4 = _o3_o4_pair(tmp_path)
    out = tmp_path / "v1.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        CombinedSelectionSet([SelectionSet(o3), SelectionSet(o4)]).to_darksirens(
            str(out), amax=0.999, strict=False)
    with h5py.File(out, "r") as f:
        assert bool(f.attrs["injection_draw_domain_mismatch"]) is True
        rows = json.loads(f.attrs["injection_draw_domain_per_campaign"])
        assert [r["declared_source"] for r in rows] == ["file_attrs",
                                                        "undeclared"]
