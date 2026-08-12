"""PR5: selection-side export builder + "gwcat-selection-2.0" writer.

Reuses the fixture writers from ``tests/test_selection_spin.py`` (``write_o4_full``
/ ``write_endo3_full`` / ``write_mixture``) and checks the three spin bases:

  * component -- pdraw vs an INDEPENDENT numpy hand-computation from file
    contents (all three fixture formats), and correct Essick fractions in the
    two-campaign case;
  * chieff -- exact byte parity with the legacy ``to_darksirens`` pdraw + the
    other nine columns, single AND combined;
  * chieff_chip -- pdraw vs an independent computation with the DETECTED amax on
    a uniform-isotropic file, and :class:`SpinBasisError` (naming file + reason)
    on a mixture file.

Plus: polar/cartesian mixture end-to-end equivalence, the optional
``snr_threshold`` OR-branch, the attrs contract, and a CLI smoke test.
"""
import json
import warnings

import numpy as np
import h5py
import pytest

from gwcat.selection import (SelectionSet, CombinedSelectionSet, _ddL_dz,
                             PDRAW_STATE_COMPONENT)
from gwcat.spin import chi_eff_chi_p_prior_logprob
from gwcat.export import SpinBasisError

# Reuse the PR4 fixture writers verbatim.
from test_selection_spin import (write_o4_full, write_endo3_full, write_mixture,
                                 _LN_2PI)


_MIX_CART_KEY = ("lnpdraw_mass1_source_mass2_source_redshift_"
                 "spin1x_spin1y_spin1z_spin2x_spin2y_spin2z")


# ======================================================================
# Independent (numpy, from-file) component-basis pdraw recomputations
# ======================================================================
def _o4_component_pdraw(path):
    with h5py.File(path, "r") as f:
        ev = f["events"]
        lnp_mz = (ev["lnpdraw_mass1_source"][:]
                  + ev["lnpdraw_mass2_source_GIVEN_mass1_source"][:]
                  + ev["lnpdraw_z"][:])
        m1det, z = ev["mass1_detector"][:], ev["z"][:]
        ddL, w = ev["dluminosity_distance_dredshift"][:], ev["weights"][:]
        lnp_mag1, lnp_mag2 = (ev["lnpdraw_spin1_magnitude"][:],
                              ev["lnpdraw_spin2_magnitude"][:])
        lnp_pol1, lnp_pol2 = (ev["lnpdraw_spin1_polar_angle"][:],
                              ev["lnpdraw_spin2_polar_angle"][:])
        st1 = np.sin(ev["spin1_polar_angle"][:])
        st2 = np.sin(ev["spin2_polar_angle"][:])
    ln_p_comp = (lnp_mz + lnp_mag1 + lnp_pol1 - np.log(st1)
                 + lnp_mag2 + lnp_pol2 - np.log(st2))
    return np.exp(ln_p_comp) * m1det / (1 + z) ** 2 / ddL / 1.0 / w


def _endo3_component_pdraw(path, H0, Om0):
    with h5py.File(path, "r") as f:
        inj = f["injections"]
        sampling_pdf = inj["sampling_pdf"][:]
        m1det, z, dL = inj["mass1"][:], inj["redshift"][:], inj["distance"][:]
        a1 = np.sqrt(inj["spin1x"][:] ** 2 + inj["spin1y"][:] ** 2
                     + inj["spin1z"][:] ** 2)
        a2 = np.sqrt(inj["spin2x"][:] ** 2 + inj["spin2y"][:] ** 2
                     + inj["spin2z"][:] ** 2)
    ddL = _ddL_dz(z, dL, H0, Om0)
    ln_p_comp = np.log(sampling_pdf) + 2 * _LN_2PI + 2 * np.log(a1) + 2 * np.log(a2)
    return np.exp(ln_p_comp) * m1det / (1 + z) ** 2 / ddL / 1.0


def _mix_cart_component_pdraw(path):
    with h5py.File(path, "r") as f:
        ev = f["events"]
        lnp_joint = ev[_MIX_CART_KEY][:]
        m1det, z = ev["mass1_detector"][:], ev["z"][:]
        ddL, w = ev["dluminosity_distance_dredshift"][:], ev["weights"][:]
        a1 = np.sqrt(ev["spin1x"][:] ** 2 + ev["spin1y"][:] ** 2
                     + ev["spin1z"][:] ** 2)
        a2 = np.sqrt(ev["spin2x"][:] ** 2 + ev["spin2y"][:] ** 2
                     + ev["spin2z"][:] ** 2)
    ln_p_comp = lnp_joint + 2 * _LN_2PI + 2 * np.log(a1) + 2 * np.log(a2)
    return np.exp(ln_p_comp) * m1det / (1 + z) ** 2 / ddL / w


# ======================================================================
# 1. component pdraw vs independent hand computation (all three formats)
# ======================================================================
def test_component_pdraw_o4(tmp_path):
    path = write_o4_full(tmp_path / "o4.hdf", n=60, amax=(0.8, 0.6), seed=6)
    out = tmp_path / "sel.h5"
    SelectionSet(path).export(str(out), spin_basis="component")
    expected = _o4_component_pdraw(path)      # all injections detected (far 0.1)
    with h5py.File(out, "r") as f:
        np.testing.assert_allclose(f["pdraw"][:], expected, rtol=1e-12)
        assert f.attrs["format_version"] == "gwcat-selection-2.0"


def test_component_pdraw_endo3(tmp_path):
    path = write_endo3_full(tmp_path / "endo3.hdf", n=50, max_spin=0.9, seed=11)
    s = SelectionSet(path)
    s._load()
    out = tmp_path / "sel.h5"
    s.export(str(out), spin_basis="component")
    expected = _endo3_component_pdraw(path, s.H0, s.Om0)
    with h5py.File(out, "r") as f:
        np.testing.assert_allclose(f["pdraw"][:], expected, rtol=1e-10)


def test_component_pdraw_mixture_both_flavors(tmp_path):
    car = write_mixture(tmp_path / "mix_cart.hdf", "cartesian", seed=23)
    pol = write_mixture(tmp_path / "mix_polar.hdf", "polar", seed=23)
    out_c = tmp_path / "sel_c.h5"
    out_p = tmp_path / "sel_p.h5"
    SelectionSet(car).export(str(out_c), spin_basis="component")
    SelectionSet(pol).export(str(out_p), spin_basis="component")
    expected = _mix_cart_component_pdraw(car)
    with h5py.File(out_c, "r") as f:
        np.testing.assert_allclose(f["pdraw"][:], expected, rtol=1e-12)
    with h5py.File(out_p, "r") as f:
        np.testing.assert_allclose(f["pdraw"][:], expected, rtol=1e-12)


# ======================================================================
# 2. CHIEFF PARITY: v2 chieff basis == legacy to_darksirens (single + combined)
# ======================================================================
_SHARED_COLS = ["m1det", "m2det", "dL", "chieff", "ra", "dec",
                "m1src", "m2src", "redshift"]


def test_chieff_parity_single(tmp_path):
    path = write_o4_full(tmp_path / "o4.hdf", n=70, amax=(0.9, 0.7), seed=3)
    v1 = tmp_path / "v1.h5"
    v2 = tmp_path / "v2.h5"
    SelectionSet(path).to_darksirens(str(v1), far_threshold=1.0)
    SelectionSet(path).export(str(v2), spin_basis="chieff", far_threshold=1.0)
    with h5py.File(v1, "r") as a, h5py.File(v2, "r") as b:
        np.testing.assert_array_equal(b["pdraw"][:], a["pdraw"][:])
        for c in _SHARED_COLS:
            np.testing.assert_array_equal(b[c][:], a[c][:])


def test_chieff_parity_combined(tmp_path):
    o3 = write_endo3_full(tmp_path / "endo3.hdf", n=50, seed=10)
    o4 = write_o4_full(tmp_path / "o4.hdf", n=60, seed=32)
    v1 = tmp_path / "cv1.h5"
    v2 = tmp_path / "cv2.h5"
    CombinedSelectionSet([SelectionSet(o3), SelectionSet(o4)]).to_darksirens(
        str(v1), far_threshold=1.0)
    CombinedSelectionSet([SelectionSet(o3), SelectionSet(o4)]).export(
        str(v2), spin_basis="chieff", far_threshold=1.0)
    with h5py.File(v1, "r") as a, h5py.File(v2, "r") as b:
        np.testing.assert_array_equal(b["pdraw"][:], a["pdraw"][:])
        for c in _SHARED_COLS:
            np.testing.assert_array_equal(b[c][:], a[c][:])


# ======================================================================
# 3. chieff_chip: detected-amax independent computation + mixture -> error
# ======================================================================
def test_chieff_chip_uniform_isotropic_o4(tmp_path):
    # Equal amax so the DETECTED amax is unambiguous (no amax_1 != amax_2 warn).
    path = write_o4_full(tmp_path / "o4.hdf", n=50, amax=(0.9, 0.9), seed=15)
    s = SelectionSet(path)
    s._load()
    det_amax = s.spin_meta["amax_detected"]
    assert det_amax[0] == pytest.approx(0.9, rel=1e-12)

    out = tmp_path / "cc.h5"
    SelectionSet(path).export(str(out), spin_basis="chieff_chip")

    # Independent expected pdraw: legacy _pdraw * exp(clip(joint lnprob, -50)).
    logp = chi_eff_chi_p_prior_logprob(s._chieff, s._chi_p, s._m1src, s._m2src,
                                       amax=0.9)
    expected = s._pdraw * np.exp(np.clip(logp, -50.0, None))
    with h5py.File(out, "r") as f:
        np.testing.assert_allclose(f["pdraw"][:], expected, rtol=1e-12)
        assert f.attrs["chi_eff_chi_p_amax_detected_per_campaign"][0] == \
            pytest.approx(0.9, rel=1e-12)


def test_chieff_chip_amax_mismatch_warns(tmp_path):
    # amax_1 != amax_2 -> joint prior uses amax_1 and warns.
    path = write_o4_full(tmp_path / "o4.hdf", n=30, amax=(0.9, 0.7), seed=16)
    out = tmp_path / "cc.h5"
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        SelectionSet(path).export(str(out), spin_basis="chieff_chip")
    assert any("amax_1" in str(w.message) for w in rec)
    with h5py.File(out, "r") as f:
        assert f.attrs["chi_eff_chi_p_amax_detected_per_campaign"][0] == \
            pytest.approx(0.9, rel=1e-12)


def test_chieff_chip_mixture_raises_spin_basis_error(tmp_path):
    car = write_mixture(tmp_path / "mix.hdf", "cartesian", seed=2)
    with pytest.raises(SpinBasisError) as ei:
        SelectionSet(car).export(str(tmp_path / "cc.h5"),
                                 spin_basis="chieff_chip")
    msg = str(ei.value)
    assert str(car) in msg                       # names the file
    assert "uniform-isotropic" in msg            # names the reason
    assert "component" in msg                     # points at the fix


# ======================================================================
# 4. Two-campaign component basis: Essick fractions correct
# ======================================================================
def test_component_two_campaign_essick_fractions(tmp_path):
    o3 = write_endo3_full(tmp_path / "endo3.hdf", n=40, max_spin=0.9, seed=31,
                          total_generated=2000)
    o4 = write_o4_full(tmp_path / "o4.hdf", n=30, amax=(0.8, 0.6), seed=32,
                       total_generated=1000)
    s3, s4 = SelectionSet(o3), SelectionSet(o4)
    out = tmp_path / "sel.h5"
    CombinedSelectionSet([s3, s4]).export(str(out), spin_basis="component",
                                          far_threshold=1.0)

    # Hand-computed expected: per campaign, component_pdraw * (N_k / N_total),
    # concatenated in campaign order (all injections detected: far 0.1).
    s3._load(); s4._load()
    n3, n4 = s3._ndraw, s4._ndraw
    total = n3 + n4
    exp3 = s3.component_pdraw() * (n3 / total)
    exp4 = s4.component_pdraw() * (n4 / total)
    expected = np.concatenate([exp3, exp4])
    with h5py.File(out, "r") as f:
        np.testing.assert_allclose(f["pdraw"][:], expected, rtol=1e-12)
        assert int(f.attrs["n_detected"]) == len(expected)
        assert list(f.attrs["campaign_ndraws"]) == [n3, n4]


# ======================================================================
# 5. polar vs cartesian mixture: end-to-end exported files agree
# ======================================================================
def test_mixture_flavors_end_to_end_equal(tmp_path):
    car = write_mixture(tmp_path / "mix_cart.hdf", "cartesian", seed=21)
    pol = write_mixture(tmp_path / "mix_polar.hdf", "polar", seed=21)
    out_c = tmp_path / "sel_c.h5"
    out_p = tmp_path / "sel_p.h5"
    SelectionSet(car).export(str(out_c), spin_basis="component")
    SelectionSet(pol).export(str(out_p), spin_basis="component")
    with h5py.File(out_c, "r") as fc, h5py.File(out_p, "r") as fp:
        np.testing.assert_allclose(fp["pdraw"][:], fc["pdraw"][:], rtol=1e-12)
        for c in _SHARED_COLS + ["chip"]:
            np.testing.assert_allclose(fp[c][:], fc[c][:], rtol=1e-12)


# ======================================================================
# 6. snr_threshold OR-branch
# ======================================================================
def _write_o4_with_snr(path, n, far_vals, snr_vals, seed=40):
    """An O4 factored fixture with an explicit per-injection FAR and SNR column."""
    write_o4_full(path, n=n, amax=(0.9, 0.9), seed=seed)
    with h5py.File(path, "r+") as f:
        ev = f["events"]
        ev["pycbc_far"][:] = np.asarray(far_vals, dtype=float)
        ev.create_dataset("semianalytic_observed_phase_maximized_snr_net",
                          data=np.asarray(snr_vals, dtype=float))
    return str(path)


def test_snr_threshold_or_branch(tmp_path):
    n = 40
    far = np.full(n, 5.0)
    far[:20] = 0.1                     # first 20 detected by FAR
    snr = np.zeros(n)
    snr[25:30] = 15.0                  # 5 far-undetected injections pass SNR>10
    path = _write_o4_with_snr(tmp_path / "o4snr.hdf", n, far, snr)

    far_mask = far < 1.0
    snr_mask = snr > 10.0
    n_union = int((far_mask | snr_mask).sum())
    n_far_only = int(far_mask.sum())
    assert n_union == 25 and n_far_only == 20      # sanity on the fixture

    # Default None: FAR cut only (unchanged behavior).
    out_none = tmp_path / "none.h5"
    SelectionSet(path).export(str(out_none), spin_basis="component")
    with h5py.File(out_none, "r") as f:
        assert int(f.attrs["n_detected"]) == n_far_only
        assert "significance_snr_threshold" not in f.attrs

    # With threshold: detection = FAR OR SNR.
    out_snr = tmp_path / "snr.h5"
    SelectionSet(path).export(str(out_snr), spin_basis="component",
                              snr_threshold=10.0)
    with h5py.File(out_snr, "r") as f:
        assert int(f.attrs["n_detected"]) == n_union
        assert f["pdraw"].shape[0] == n_union
        assert f.attrs["significance_snr_column"] == \
            "semianalytic_observed_phase_maximized_snr_net"
        assert float(f.attrs["significance_snr_threshold"]) == 10.0
        assert f.attrs["significance_type"] == "far_or_snr"


def test_snr_threshold_missing_column_errors(tmp_path):
    # A plain o4 fixture has no semianalytic SNR column.
    path = write_o4_full(tmp_path / "o4.hdf", n=10, seed=41)
    with pytest.raises(ValueError, match="snr_threshold"):
        SelectionSet(path).export(str(tmp_path / "x.h5"),
                                  spin_basis="component", snr_threshold=8.0)


# ======================================================================
# 7. attrs contract
# ======================================================================
def test_attrs_contract(tmp_path):
    o3 = write_endo3_full(tmp_path / "endo3.hdf", n=20, seed=31)
    o4 = write_o4_full(tmp_path / "o4.hdf", n=15, amax=(0.9, 0.7), seed=32)
    out = tmp_path / "sel.h5"
    CombinedSelectionSet([SelectionSet(o3), SelectionSet(o4)]).export(
        str(out), spin_basis="component")
    with h5py.File(out, "r") as f:
        assert f.attrs["format_version"] == "gwcat-selection-2.0"
        assert f.attrs["spin_basis"] == "component"
        assert f.attrs["pdraw_state"] == PDRAW_STATE_COMPONENT
        assert bool(f.attrs["chi_eff_prior_applied_to_pdraw"]) is False
        # Per-campaign injected-spin provenance (campaign order).
        fmts = [c.decode() if isinstance(c, bytes) else c
                for c in f.attrs["injected_spin_format"]]
        assert fmts == ["endo3_factored", "o4_factored"]
        amax_det = np.asarray(f.attrs["injected_spin_amax_detected"])
        assert amax_det.shape == (2, 2)
        # o4 campaign detected (0.9, 0.7).
        np.testing.assert_allclose(amax_det[1], [0.9, 0.7], rtol=1e-9)
        assert list(np.asarray(f.attrs["injected_spin_uniform_isotropic"])) \
            == [True, True]
        checks = json.loads(f.attrs["injected_spin_checks"])
        assert isinstance(checks, list) and len(checks) == 2
        assert int(f.attrs["n_campaigns"]) == 2


def test_attrs_chieff_basis_records_swap(tmp_path):
    path = write_o4_full(tmp_path / "o4.hdf", n=20, seed=5)
    out = tmp_path / "sel.h5"
    SelectionSet(path).export(str(out), spin_basis="chieff", amax=0.95)
    with h5py.File(out, "r") as f:
        assert f.attrs["spin_basis"] == "chieff"
        assert bool(f.attrs["chi_eff_swap_applied"]) is True
        assert bool(f.attrs["chi_eff_prior_applied_to_pdraw"]) is True
        assert float(f.attrs["chi_eff_amax"]) == 0.95
        assert f.attrs["spin_prior_mode"] == "include"


# ======================================================================
# 8. CLI smoke
# ======================================================================
def test_cli_export_selection_smoke(tmp_path):
    from gwcat.cli import main
    inj = write_o4_full(tmp_path / "o4.hdf", n=20, seed=50)
    out = tmp_path / "sel.h5"
    rc = main(["export", "selection", str(inj), "--out", str(out),
               "--spin-basis", "component", "--no-summary"])
    assert rc == 0
    assert out.exists()
    with h5py.File(out, "r") as f:
        assert f.attrs["format_version"] == "gwcat-selection-2.0"
        assert f.attrs["spin_basis"] == "component"


def test_cli_export_selection_combined_chieff(tmp_path):
    from gwcat.cli import main
    o3 = write_endo3_full(tmp_path / "endo3.hdf", n=20, seed=51)
    o4 = write_o4_full(tmp_path / "o4.hdf", n=15, seed=52)
    out = tmp_path / "sel.h5"
    rc = main(["export", "selection", str(o3), str(o4), "--out", str(out),
               "--spin-basis", "chieff", "--far-threshold", "1.0",
               "--no-summary"])
    assert rc == 0
    with h5py.File(out, "r") as f:
        assert int(f.attrs["n_campaigns"]) == 2
        assert f.attrs["spin_basis"] == "chieff"


# ======================================================================
# validation summary written on request
# ======================================================================
def test_write_summary(tmp_path):
    path = write_o4_full(tmp_path / "o4.hdf", n=20, seed=60)
    out = tmp_path / "sel.h5"
    SelectionSet(path).export(str(out), spin_basis="component",
                              write_summary=True)
    assert (tmp_path / "sel.h5.validation_summary.json").exists()
    with open(tmp_path / "sel.h5.validation_summary.json") as fh:
        summary = json.load(fh)
    assert summary["kind"] == "selection_export"
    assert summary["spin_basis"] == "component"
    assert summary["schema_version"] == "gwcat-selection-2.0"


# ==========================================================================
# GW-06: the chieff swap is gated on a uniform-isotropic injected draw
# ==========================================================================
def _fake_set(path, *, uniform, checks=None):
    class _S:
        pass
    s = _S()
    s.path = path
    s.spin_meta = {"spin_format": "o4_factored",
                   "uniform_isotropic": uniform,
                   "checks": checks or {}}
    return s


def test_chieff_raises_on_a_verified_non_uniform_campaign():
    """The measured O4ab case: magnitude uniformity and isotropy both FAIL, so
    the analytic chi_eff swap is invalid and must be refused."""
    from gwcat.export.selection_builder import (BlockCampaignMismatch,
                                                _check_chieff_swap_valid)

    sets = [_fake_set("o4ab.hdf", uniform=False,
                      checks={"magnitude_uniform": (False, False),
                              "isotropy_dev": (0.6419, 0.6419)})]
    with pytest.raises(BlockCampaignMismatch) as exc:
        _check_chieff_swap_valid(sets, strict=True, violations=[])
    msg = str(exc.value)
    assert "o4ab.hdf" in msg
    assert "0.6419" in msg
    assert "component" in msg          # names the remedy


def test_chieff_non_strict_warns_and_records_the_violation():
    from gwcat.export.selection_builder import _check_chieff_swap_valid

    viol = []
    sets = [_fake_set("o4ab.hdf", uniform=False,
                      checks={"magnitude_uniform": (False, False),
                              "isotropy_dev": (0.6419, 0.6419)})]
    with pytest.warns(UserWarning, match="did NOT draw spins that way"):
        _check_chieff_swap_valid(sets, strict=False, violations=viol)
    assert len(viol) == 1
    assert viol[0]["verified"] is True
    assert viol[0]["isotropy_dev"] == [0.6419, 0.6419]


def test_chieff_passes_a_uniform_isotropic_campaign():
    from gwcat.export.selection_builder import _check_chieff_swap_valid

    viol = []
    sets = [_fake_set("endo3.hdf", uniform=True,
                      checks={"magnitude_uniform": (True, True),
                              "isotropy_dev": (1e-9, 1e-9)})]
    _check_chieff_swap_valid(sets, strict=True, violations=viol)   # no raise
    assert viol == []


def test_unverifiable_campaign_warns_but_does_not_refuse():
    """A file carrying NO spin draw densities is unknown, not contradicted.

    Refusing it would break every legacy spin-less campaign, for which the
    chi_eff swap is the only basis available -- there is no component density to
    fall back to.  So "checked and failed" and "never checked" must not be
    conflated, even though both leave uniform_isotropic False.
    """
    from gwcat.export.selection_builder import _check_chieff_swap_valid

    viol = []
    sets = [_fake_set("legacy.hdf", uniform=False, checks={})]
    with pytest.warns(UserWarning, match="could not be CHECKED"):
        _check_chieff_swap_valid(sets, strict=True, violations=viol)  # no raise
    assert len(viol) == 1 and viol[0]["verified"] is False


def test_only_the_offending_campaign_is_named_in_a_mixed_export():
    from gwcat.export.selection_builder import (BlockCampaignMismatch,
                                                _check_chieff_swap_valid)

    sets = [_fake_set("endo3.hdf", uniform=True,
                      checks={"magnitude_uniform": (True, True)}),
            _fake_set("o4ab.hdf", uniform=False,
                      checks={"magnitude_uniform": (False, False),
                              "isotropy_dev": (0.6419, 0.6419)})]
    with pytest.raises(BlockCampaignMismatch) as exc:
        _check_chieff_swap_valid(sets, strict=True, violations=[])
    assert "o4ab.hdf" in str(exc.value)
    assert "endo3.hdf" not in str(exc.value)
    assert "1 campaign(s)" in str(exc.value)
