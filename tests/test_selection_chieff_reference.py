"""The ``chieff_reference`` selection basis: chi_eff against a DECLARED reference.

The basis exists because two things were true at once: the marked-transition
likelihood needs a draw density in ``(m1det, q, dL, chi_eff)`` expressed against
the same isotropic uniform-magnitude spin prior the PE product divides out, and
``spin_basis='chieff'`` -- the only basis that wrote a chi_eff density -- must
refuse the O4ab campaign, because it SUBSTITUTES an assumed spin law for the
campaign's real one (GW-06).  The reference basis REWEIGHTS instead: the
campaign's exact component draw stays in the numerator and the declared
reference is divided in.

What is checked here:

  1. algebra, uniform-isotropic campaign -- the reference pdraw differs from the
     substituting swap's by exactly the constant ``a_ref^2/(amax_1 amax_2)``,
     and coincides with it when the campaign's ceiling IS the reference;
  2. algebra, a campaign that is neither uniform in magnitude nor isotropic in
     tilt -- ``chieff`` refuses it (the guard is untouched), and the reference
     pdraw matches a from-file numpy computation to 1e-12;
  3. the reference support -- rows with ``a_i > a_ref`` carry the zero-weight
     sentinel, every other row is finite and positive, and nothing but ``pdraw``
     moves relative to the component export it is built from;
  4. ``a_ref`` is required, is refused on the bases that have no reference, and
     is recorded;
  5. coverage -- a campaign whose magnitudes stop below ``a_ref`` raises under
     strict (proven ceiling) or warns (sample maximum only);
  6. end to end: two campaigns through the CLI onto disk, with the Essick
     fractions, the attrs contract and the summary.
"""
import json
import warnings

import numpy as np
import h5py
import pytest

from gwcat.selection import (SelectionSet, PDRAW_STATE_CHIEFF_REFERENCE,
                             PDRAW_STATE_COMPONENT)
from gwcat.spin import chi_eff_prior_logprob_in_support
from gwcat.export import SpinBasisError, build_selection_product
from gwcat.export.selection_builder import (BlockCampaignMismatch,
                                            OUT_OF_REFERENCE_PDRAW)
from gwcat.cli import main

from test_selection_spin import (write_o4_full, write_endo3_full,
                                 _write_events_group, _LN_2PI)

_YEAR_S = 365.25 * 24 * 3600
_A_REF = 0.99


# ======================================================================
# A campaign spin_basis='chieff' has to refuse: p(a) ∝ a^2, p(cosθ) tilted
# ======================================================================
def write_o4_nonuniform(path, n=400, seed=3, a_pow=2.0, tilt_slope=0.8,
                        total_generated=4000, mag_max=1.0):
    """O4-factored fixture with a NON-uniform, NON-isotropic spin draw.

    ``p(a) = (a_pow+1) a^a_pow / mag_max^(a_pow+1)`` on ``[0, mag_max]`` and
    ``p(cosθ) = (1 + tilt_slope cosθ)/2``.  Every ``lnpdraw`` column is the
    analytic density of the draw actually taken, so a test can recompute the
    component density (and the reference one) from the file alone.
    """
    rng = np.random.default_rng(seed)
    m1s = rng.uniform(30.0, 45.0, n)
    m2s = rng.uniform(8.0, 25.0, n)
    m1s, m2s = np.maximum(m1s, m2s), np.minimum(m1s, m2s)
    z = rng.uniform(0.05, 0.5, n)

    def _mag():
        a = mag_max * rng.uniform(0.0, 1.0, n) ** (1.0 / (a_pow + 1.0))
        p = (a_pow + 1.0) * a ** a_pow / mag_max ** (a_pow + 1.0)
        return a, np.log(p)

    def _tilt():
        u = rng.uniform(0.0, 1.0, n)
        m = tilt_slope
        c = (-1.0 + np.sqrt(1.0 + 2.0 * m * (2.0 * u - 1.0) + m ** 2)) / m
        return c, np.log((1.0 + m * c) / 2.0)

    a1, lnp_mag1 = _mag()
    a2, lnp_mag2 = _mag()
    cost1, lnp_c1 = _tilt()
    cost2, lnp_c2 = _tilt()
    st1 = np.sqrt(1.0 - cost1 ** 2)
    st2 = np.sqrt(1.0 - cost2 ** 2)
    th1, th2 = np.arccos(cost1), np.arccos(cost2)
    ph1 = rng.uniform(0.0, 2 * np.pi, n)
    ph2 = rng.uniform(0.0, 2 * np.pi, n)

    s1x, s1y, s1z = a1 * st1 * np.cos(ph1), a1 * st1 * np.sin(ph1), a1 * cost1
    s2x, s2y, s2z = a2 * st2 * np.cos(ph2), a2 * st2 * np.sin(ph2), a2 * cost2
    chieff = (m1s * s1z + m2s * s2z) / (m1s + m2s)

    cols = {
        "mass1_source": m1s, "mass2_source": m2s,
        "mass1_detector": m1s * (1 + z), "mass2_detector": m2s * (1 + z),
        "luminosity_distance": rng.uniform(200.0, 900.0, n),
        "z": z, "dluminosity_distance_dredshift": rng.uniform(3000.0, 6000.0, n),
        "right_ascension": rng.uniform(0, 2 * np.pi, n),
        "declination": rng.uniform(-np.pi / 2, np.pi / 2, n),
        "spin1x": s1x, "spin1y": s1y, "spin1z": s1z,
        "spin2x": s2x, "spin2y": s2y, "spin2z": s2z,
        "spin1_magnitude": a1, "spin1_polar_angle": th1,
        "spin1_azimuthal_angle": ph1,
        "spin2_magnitude": a2, "spin2_polar_angle": th2,
        "spin2_azimuthal_angle": ph2,
        "chi_eff": chieff, "weights": np.full(n, 1.5),
        "lnpdraw_mass1_source": -np.log(15.0) * np.ones(n),
        "lnpdraw_mass2_source_GIVEN_mass1_source": -np.log(17.0) * np.ones(n),
        "lnpdraw_z": -np.log(0.45) * np.ones(n),
        "lnpdraw_spin1_magnitude": lnp_mag1,
        # p(θ) = p(cosθ) sinθ.
        "lnpdraw_spin1_polar_angle": lnp_c1 + np.log(st1),
        "lnpdraw_spin1_azimuthal_angle": -_LN_2PI * np.ones(n),
        "lnpdraw_spin2_magnitude": lnp_mag2,
        "lnpdraw_spin2_polar_angle": lnp_c2 + np.log(st2),
        "lnpdraw_spin2_azimuthal_angle": -_LN_2PI * np.ones(n),
        "pycbc_far": np.full(n, 0.1),
    }
    attrs = {"total_analysis_time": _YEAR_S, "total_generated": total_generated}
    return _write_events_group(path, cols, attrs, [b"pycbc"])


def _reference_from_file_o4(path, a_ref):
    """pdraw in the reference basis, recomputed from the FILE in numpy."""
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
        a1, a2 = ev["spin1_magnitude"][:], ev["spin2_magnitude"][:]
        m1s, m2s, chieff = (ev["mass1_source"][:], ev["mass2_source"][:],
                            ev["chi_eff"][:])
    # p_base(m1det, q, dL) * p_draw(a1, cosθ1, a2, cosθ2), the component basis.
    ln_p_comp = (lnp_mz + lnp_mag1 + lnp_pol1 - np.log(st1)
                 + lnp_mag2 + lnp_pol2 - np.log(st2))
    pdraw_comp = np.exp(ln_p_comp) * m1det / (1 + z) ** 2 / ddL / 1.0 / w
    # ... reweighted: * p_iso(chieff | q, a_ref) / p_ref, p_ref = 1/(4 a_ref^2).
    lnp_chi, chi_sup = chi_eff_prior_logprob_in_support(
        chieff, m1s, m2s, amax=a_ref)
    in_ref = chi_sup & (a1 <= a_ref) & (a2 <= a_ref)
    out = pdraw_comp * np.exp(lnp_chi) * 4.0 * a_ref ** 2
    return np.where(in_ref, out, OUT_OF_REFERENCE_PDRAW), in_ref


# ======================================================================
# 1. Uniform-isotropic campaign: the difference from the swap is a CONSTANT
# ======================================================================
@pytest.mark.parametrize("amax", [(0.99, 0.99), (0.8, 0.6)])
def test_reference_vs_swap_is_a_constant_on_a_uniform_isotropic_campaign(
        tmp_path, amax):
    """On the one campaign type where both bases are defined, they differ by
    ``a_ref^2 / (amax_1 amax_2)`` -- a constant, and exactly 1 when the campaign
    drew at the reference ceiling.

    This is the sharpest statement of what the reference basis does: for a
    campaign that really is uniform-isotropic, ``p_draw = 1/(4 amax_1 amax_2)``
    and the reweight collapses to that ratio of ceilings.  It is also why the
    substituting swap is not merely 'the same thing with a different amax' on
    any OTHER campaign -- there ``p_draw`` is not constant at all.
    """
    path = write_o4_full(tmp_path / "iso.hdf", n=150, amax=amax, seed=6)

    swap = build_selection_product([SelectionSet(path)], spin_basis="chieff",
                                   amax=_A_REF)
    ref = build_selection_product([SelectionSet(path)],
                                  spin_basis="chieff_reference",
                                  spin_reference_amax=_A_REF, strict=False)
    in_ref = ((ref.columns["a1"] <= _A_REF) & (ref.columns["a2"] <= _A_REF))
    ratio = ref.columns["pdraw"][in_ref] / swap.columns["pdraw"][in_ref]
    expected = _A_REF ** 2 / (amax[0] * amax[1])
    np.testing.assert_allclose(ratio, expected, rtol=1e-12)


# ======================================================================
# 2. A campaign chieff must refuse: the reference basis builds it, exactly
# ======================================================================
def test_chieff_still_refuses_the_campaign_the_reference_basis_accepts(tmp_path):
    """The GW-06 guard is untouched: the new basis is a second route, not a way
    around the first one's refusal."""
    path = write_o4_nonuniform(tmp_path / "nonuni.hdf", n=200, seed=3)
    meta = SelectionSet(path).spin_meta
    assert meta["uniform_isotropic"] is False

    with pytest.raises(BlockCampaignMismatch) as ei:
        build_selection_product([SelectionSet(path)], spin_basis="chieff")
    assert "did NOT draw spins that way" in str(ei.value)

    # The same campaign, in the reference basis, builds without complaint.
    prod = build_selection_product([SelectionSet(path)],
                                   spin_basis="chieff_reference",
                                   spin_reference_amax=_A_REF)
    assert prod.attrs["spin_reference_amax"] == _A_REF


def test_reference_pdraw_matches_a_from_file_computation(tmp_path):
    """The exported pdraw against an INDEPENDENT numpy computation that never
    calls the builder -- on the non-uniform, non-isotropic campaign."""
    path = write_o4_nonuniform(tmp_path / "nonuni.hdf", n=300, seed=5)
    out = tmp_path / "sel.h5"
    SelectionSet(path).export(str(out), spin_basis="chieff_reference",
                              spin_reference_amax=_A_REF)
    expected, in_ref = _reference_from_file_o4(path, _A_REF)
    with h5py.File(out, "r") as f:
        got = f["pdraw"][:]
        assert f.attrs["spin_basis"] == "chieff_reference"
        assert f.attrs["spin_reference_excluded_rows"] == int((~in_ref).sum())
    np.testing.assert_allclose(got[in_ref], expected[in_ref], rtol=1e-12)
    np.testing.assert_array_equal(got[~in_ref], expected[~in_ref])


# ======================================================================
# 3. The reference SUPPORT: zero weight, not a fatal contradiction
# ======================================================================
def test_out_of_reference_rows_get_the_sentinel_and_nothing_else_moves(tmp_path):
    """Rows the reference excludes carry a sentinel pdraw whose inverse weight
    underflows to zero; every other column is the component export's, bit for
    bit.

    Note what does NOT happen: the substituting bases treat an out-of-support
    DETECTED injection as fatal (GW-03), because there the zero is an assumed
    prior contradicting a draw that really happened. Here the draw is untouched
    and it is the DECLARED reference that stops, so a zero weight is the answer,
    not an error.
    """
    # amax above the reference, so some injections really do fall outside it.
    path = write_o4_full(tmp_path / "iso.hdf", n=250, amax=(0.998, 0.998),
                         seed=9)
    comp = build_selection_product([SelectionSet(path)], spin_basis="component")
    ref = build_selection_product([SelectionSet(path)],
                                  spin_basis="chieff_reference",
                                  spin_reference_amax=_A_REF)

    outside = (comp.columns["a1"] > _A_REF) | (comp.columns["a2"] > _A_REF)
    assert outside.sum() > 0, "fixture drew nothing outside the reference"
    assert np.all(ref.columns["pdraw"][outside] == OUT_OF_REFERENCE_PDRAW)
    assert int(ref.attrs["spin_reference_excluded_rows"]) == int(outside.sum())
    assert np.all(np.isfinite(ref.columns["pdraw"]))
    assert np.all(ref.columns["pdraw"] > 0)
    # The row's importance weight p_pop*J/pdraw is ~1e-300 against O(1) weights
    # elsewhere: the zero the physics calls for, to every purpose a float has.
    assert np.all(1.0 / ref.columns["pdraw"][outside] < 1e-299)
    inside = ~outside
    assert (1.0 / ref.columns["pdraw"][outside]).max() < \
        1e-280 * np.median(1.0 / ref.columns["pdraw"][inside])

    for name, arr in comp.columns.items():
        if name == "pdraw":
            continue
        np.testing.assert_array_equal(arr, ref.columns[name],
                                      err_msg=f"column {name} moved")


def test_reference_starts_from_the_component_density_bit_for_bit(tmp_path):
    """``pdraw_ref / pdraw_component`` is the reference factor and nothing else
    -- the component product is the literal starting point, not a re-derivation.
    """
    path = write_o4_nonuniform(tmp_path / "nonuni.hdf", n=200, seed=11)
    comp = build_selection_product([SelectionSet(path)], spin_basis="component")
    ref = build_selection_product([SelectionSet(path)],
                                  spin_basis="chieff_reference",
                                  spin_reference_amax=_A_REF)
    lnp_chi, chi_sup = chi_eff_prior_logprob_in_support(
        comp.columns["chieff"], comp.columns["m1src"], comp.columns["m2src"],
        amax=_A_REF)
    in_ref = (chi_sup & (comp.columns["a1"] <= _A_REF)
              & (comp.columns["a2"] <= _A_REF))
    expected = comp.columns["pdraw"] * np.exp(lnp_chi + np.log(4 * _A_REF ** 2))
    np.testing.assert_array_equal(ref.columns["pdraw"][in_ref],
                                  expected[in_ref])


# ======================================================================
# 4. a_ref is declared, never defaulted
# ======================================================================
def test_reference_amax_is_required(tmp_path):
    path = write_o4_full(tmp_path / "iso.hdf", n=40, amax=(0.998, 0.998), seed=2)
    with pytest.raises(ValueError) as ei:
        build_selection_product([SelectionSet(path)],
                                spin_basis="chieff_reference")
    assert "spin_reference_amax" in str(ei.value)
    assert "no honest default" in str(ei.value)


@pytest.mark.parametrize("basis", ["component", "chieff"])
def test_reference_amax_refused_on_a_basis_with_no_reference(tmp_path, basis):
    path = write_o4_full(tmp_path / "iso.hdf", n=40, amax=(0.99, 0.99), seed=2)
    with pytest.raises(ValueError) as ei:
        build_selection_product([SelectionSet(path)], spin_basis=basis,
                                spin_reference_amax=0.99)
    assert "has no reference prior" in str(ei.value)


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5, float("nan")])
def test_reference_amax_must_be_a_physical_ceiling(tmp_path, bad):
    path = write_o4_full(tmp_path / "iso.hdf", n=40, amax=(0.99, 0.99), seed=2)
    with pytest.raises(ValueError):
        build_selection_product([SelectionSet(path)],
                                spin_basis="chieff_reference",
                                spin_reference_amax=bad)


# ======================================================================
# 5. Coverage: the campaign must reach the reference ceiling
# ======================================================================
def test_coverage_hole_raises_when_the_campaign_ceiling_proves_it(tmp_path):
    """A uniform-magnitude campaign has a DETECTED ceiling, so a reference above
    it is a proven hole: the reference puts probability where the campaign drew
    nothing, and alpha comes out biased low."""
    path = write_o4_full(tmp_path / "low.hdf", n=80, amax=(0.5, 0.5), seed=4)
    with pytest.raises(BlockCampaignMismatch) as ei:
        build_selection_product([SelectionSet(path)],
                                spin_basis="chieff_reference",
                                spin_reference_amax=_A_REF)
    msg = str(ei.value)
    assert "biased LOW" in msg and "0.5" in msg

    # strict=False exports anyway, and the file says so about itself.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        prod = build_selection_product([SelectionSet(path)],
                                       spin_basis="chieff_reference",
                                       spin_reference_amax=_A_REF, strict=False)
    assert any("biased LOW" in str(x.message) for x in w)
    assert prod.attrs["spin_reference_coverage_ok"] is False
    assert list(prod.attrs["spin_reference_coverage_per_campaign"]) == [False]


def test_coverage_only_warns_when_the_evidence_is_a_sample_maximum(tmp_path):
    """No detectable ceiling => the bound is the largest magnitude DRAWN, which
    is evidence of a hole, not proof of one. It warns and exports."""
    path = write_o4_nonuniform(tmp_path / "nonuni.hdf", n=120, seed=7,
                               mag_max=0.6)
    assert SelectionSet(path).spin_meta["amax_detected"] in (None, (None, None))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        prod = build_selection_product([SelectionSet(path)],
                                       spin_basis="chieff_reference",
                                       spin_reference_amax=_A_REF)
    assert any("sample maximum, not a support bound" in str(x.message)
               for x in w)
    assert prod.attrs["spin_reference_coverage_ok"] is False
    assert (prod.attrs["spin_reference_coverage_source_per_campaign"][0]
            == "drawn_max")


def test_coverage_ok_is_recorded_with_its_evidence(tmp_path):
    path = write_o4_full(tmp_path / "iso.hdf", n=60, amax=(0.998, 0.998), seed=8)
    prod = build_selection_product([SelectionSet(path)],
                                   spin_basis="chieff_reference",
                                   spin_reference_amax=_A_REF)
    assert prod.attrs["spin_reference_coverage_ok"] is True
    assert (prod.attrs["spin_reference_coverage_source_per_campaign"][0]
            == "detected")
    np.testing.assert_allclose(
        prod.attrs["spin_reference_coverage_bound_per_campaign"], 0.998,
        rtol=1e-6)


# ======================================================================
# 6. End to end: two campaigns, CLI, on disk
# ======================================================================
def test_end_to_end_two_campaigns_through_the_cli(tmp_path):
    """The shipped route: two injection files, one command, one file -- with the
    Essick N_k/N_total fractions, the reference applied per campaign and the
    contract attrs a consumer reads."""
    o3 = write_endo3_full(tmp_path / "endo3.hdf", n=90, max_spin=0.998, seed=1,
                          total_generated=2000)
    o4 = write_o4_nonuniform(tmp_path / "o4.hdf", n=140, seed=13,
                             total_generated=8000)
    out = tmp_path / "sel_ref.h5"
    assert main(["export", "selection", str(o3), str(o4), "--out", str(out),
                 "--parameter-space", "chieff_reference",
                 "--spin-reference-amax", str(_A_REF)]) == 0

    # The same product built in Python, per campaign, for the comparison below.
    sets = [SelectionSet(o3), SelectionSet(o4)]
    comp = build_selection_product(sets, spin_basis="component")

    with h5py.File(out, "r") as f:
        attrs = dict(f.attrs)
        pdraw = f["pdraw"][:]
        a1, a2 = f["a1"][:], f["a2"][:]
        chieff = f["chieff"][:]
        m1src, m2src = f["m1src"][:], f["m2src"][:]

    assert attrs["spin_basis"] == "chieff_reference"
    assert attrs["spin_prior_mode"] == "include"
    assert bool(attrs["chi_eff_swap_applied"]) is True
    assert bool(attrs["chi_eff_prior_applied_to_pdraw"]) is True
    assert bool(attrs["component_spin_draw_retained"]) is False
    assert attrs["chi_eff_prior_source"] == "reweighted_from_component_draw"
    assert attrs["spin_reference_amax"] == _A_REF
    assert attrs["chi_eff_amax"] == _A_REF
    assert attrs["pdraw_state"] == PDRAW_STATE_CHIEFF_REFERENCE
    assert attrs["pdraw_state"] != PDRAW_STATE_COMPONENT
    assert bool(attrs["spin_removal_amax_cancels"]) is True
    assert int(attrs["ndraw"]) == 2000 + 8000
    assert list(attrs["campaign_ndraws"]) == [2000, 8000]
    assert int(attrs["n_detected"]) == len(pdraw) == 90 + 140

    # Essick fractions survive the reweight: the reference factor is per
    # injection and does not touch N_k/N_total.
    lnp_chi, chi_sup = chi_eff_prior_logprob_in_support(
        chieff, m1src, m2src, amax=_A_REF)
    in_ref = chi_sup & (a1 <= _A_REF) & (a2 <= _A_REF)
    expected = comp.columns["pdraw"] * np.exp(lnp_chi + np.log(4 * _A_REF ** 2))
    np.testing.assert_allclose(pdraw[in_ref], expected[in_ref], rtol=1e-12)
    assert np.all(pdraw[~in_ref] == OUT_OF_REFERENCE_PDRAW)

    summary = json.loads(
        (tmp_path / "sel_ref.h5.validation_summary.json").read_text())
    assert summary["spin_basis"] == "chieff_reference"
    assert summary["spin_reference_amax"] == _A_REF
    assert summary["spin_reference_excluded_rows"] == int((~in_ref).sum())
    assert summary["spin_reference_coverage_per_campaign"] == [True, True]
