"""Tests for the chi_p (effective precessing spin) foundations in gwcat.spin.

Covers the three additions:

  * ``chi_p_from_components`` -- Schmidt et al. (2015) chi_p from component
    spins (hand cases, broadcasting, mass-frame invariance);
  * ``component_spin_prior_lnpdf`` -- ln pdf of the standard uniform-magnitude /
    isotropic component-spin PE prior;
  * ``ChiEffChiPPrior`` / ``chi_eff_chi_p_prior_logprob`` -- the joint prior
    p(chi_eff, chi_p | q, amax), validated by Monte Carlo against the analytic
    density, plus conditional-normalisation, marginal-consistency and
    boundary-robustness checks.

All tests are self-contained (numpy only, no network / no fixtures).
"""
import numpy as np
import pytest

from gwcat.spin import (
    ChiEffPrior,
    ChiEffChiPPrior,
    chi_eff_chi_p_prior_logprob,
    chi_p_from_components,
    component_spin_prior_lnpdf,
)

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


# ==================================================================
# Helpers: Monte-Carlo draws of (chi_eff, chi_p)
# ==================================================================
def _draw_chi_eff_chi_p(q, amax, n, seed=0):
    """Draw (chi_eff, chi_p) under isotropic uniform-magnitude component spins.

    m1 (primary) = 1, m2 = q so that mass_2/mass_1 = q.
    """
    rng = np.random.default_rng(seed)
    a1 = rng.uniform(0.0, amax, n)
    a2 = rng.uniform(0.0, amax, n)
    c1 = rng.uniform(-1.0, 1.0, n)
    c2 = rng.uniform(-1.0, 1.0, n)
    s1z = a1 * c1
    s2z = a2 * c2
    chi_eff = (s1z + q * s2z) / (1.0 + q)
    # Reuse the module function for chi_p (also exercises it on big arrays).
    chi_p = chi_p_from_components(a1, a2, c1, c2, np.ones(n), np.full(n, q))
    return chi_eff, chi_p


# ==================================================================
# 1. chi_p_from_components
# ==================================================================
def test_chi_p_scalar_hand_cases():
    # Aligned spins (cos_tilt = 1 -> sin_tilt = 0) => chi_p = 0.
    assert chi_p_from_components(0.9, 0.9, 1.0, 1.0, 30.0, 25.0) == 0.0
    assert chi_p_from_components(0.9, 0.9, -1.0, -1.0, 30.0, 25.0) == 0.0

    # q = 1 -> k = 1 -> chi_p = max(a1 sin1, a2 sin2).
    # spin1 fully in-plane (cos1=0 => sin1=1), spin2 aligned (sin2=0).
    assert chi_p_from_components(0.8, 0.5, 0.0, 1.0, 30.0, 30.0) == pytest.approx(0.8)
    # both in-plane, secondary larger but k=1 -> max picks 0.6.
    assert chi_p_from_components(0.4, 0.6, 0.0, 0.0, 30.0, 30.0) == pytest.approx(0.6)


def test_chi_p_secondary_branch_and_k():
    # q = 0.5 -> k = 0.5*(2+3)/(4+1.5) = 2.5/5.5.
    q = 0.5
    k = q * (4.0 * q + 3.0) / (4.0 + 3.0 * q)
    # spin1 aligned (no in-plane), spin2 in-plane -> chi_p = k * a2 * sin2.
    val = chi_p_from_components(0.7, 0.9, 1.0, 0.0, 40.0, 20.0)
    assert val == pytest.approx(k * 0.9)
    # sanity: k < 1 strictly for q < 1.
    assert 0.0 < k < 1.0


def test_chi_p_broadcasting():
    a1 = np.array([0.5, 0.8, 0.2])
    a2 = np.array([0.9, 0.1, 0.7])
    c1 = np.array([0.0, 1.0, 0.5])
    c2 = np.array([0.0, 0.0, -0.5])
    m1 = np.array([30.0, 40.0, 50.0])
    m2 = np.array([30.0, 20.0, 10.0])
    out = chi_p_from_components(a1, a2, c1, c2, m1, m2)
    assert out.shape == (3,)
    # elementwise reconstruction
    for i in range(3):
        exp = chi_p_from_components(a1[i], a2[i], c1[i], c2[i], m1[i], m2[i])
        assert out[i] == pytest.approx(exp)


def test_chi_p_mass_frame_invariance():
    """Detector-frame and source-frame masses give an identical chi_p, since
    only the ratio enters (the redshift factor cancels)."""
    z = 1.7
    src = chi_p_from_components(0.6, 0.4, 0.3, -0.2, 35.0, 21.0)
    det = chi_p_from_components(0.6, 0.4, 0.3, -0.2, 35.0 * (1 + z), 21.0 * (1 + z))
    assert det == pytest.approx(src, rel=0, abs=0)


# ==================================================================
# 2. component_spin_prior_lnpdf
# ==================================================================
def test_component_prior_in_range_value():
    amax1, amax2 = 0.99, 0.5
    val = component_spin_prior_lnpdf(0.3, 0.2, 0.1, -0.4, amax1, amax2)
    assert val == pytest.approx(-np.log(4.0 * amax1 * amax2))


def test_component_prior_out_of_range():
    # a > amax, |cos| > 1, a < 0 all give -inf.
    assert component_spin_prior_lnpdf(1.5, 0.2, 0.1, 0.1, 0.99, 0.99) == -np.inf
    assert component_spin_prior_lnpdf(0.3, 0.2, 1.5, 0.1, 0.99, 0.99) == -np.inf
    assert component_spin_prior_lnpdf(-0.1, 0.2, 0.1, 0.1, 0.99, 0.99) == -np.inf


def test_component_prior_array_and_scalar_amax():
    a1 = np.array([0.1, 0.6, 0.95])
    a2 = np.array([0.1, 0.2, 0.30])
    c1 = np.array([0.0, 0.5, -1.0])
    c2 = np.array([0.0, -0.5, 1.0])
    # per-sample amax arrays
    amax1 = np.array([0.99, 0.5, 0.99])
    amax2 = np.array([0.99, 0.99, 0.2])
    out = component_spin_prior_lnpdf(a1, a2, c1, c2, amax1, amax2)
    assert out.shape == (3,)
    # sample 1: a2=0.2 > amax2=0.99? no; a1=0.6<=0.5? NO -> out of range
    assert out[1] == -np.inf
    # sample 2: a2=0.30 > amax2=0.2 -> out of range
    assert out[2] == -np.inf
    # sample 0: in range
    assert out[0] == pytest.approx(-np.log(4.0 * 0.99 * 0.99))

    # scalar amax broadcasts against arrays
    out2 = component_spin_prior_lnpdf(a1, a2, c1, c2, 0.99, 0.99)
    assert np.all(np.isfinite(out2))
    assert np.allclose(out2, -np.log(4.0 * 0.99 * 0.99))


def test_component_prior_is_normalised():
    """The exp of the lnpdf integrates to 1 over the (a1,a2,cos1,cos2) box."""
    amax1, amax2 = 0.99, 0.7
    lnp = component_spin_prior_lnpdf(0.5, 0.5, 0.0, 0.0, amax1, amax2)
    dens = np.exp(lnp)
    volume = amax1 * amax2 * 2.0 * 2.0  # a in [0,amax], cos in [-1,1]
    assert dens * volume == pytest.approx(1.0)


# ==================================================================
# 3. ChiEffChiPPrior: conditional normalisation (exact quadrature)
# ==================================================================
@pytest.mark.parametrize("q,amax", [(1.0, 0.99), (0.5, 0.99), (0.3, 0.998)])
def test_conditional_normalisation(q, amax):
    """int p(chi_p | chi_eff, q, amax) dchi_p = 1 for several chi_eff."""
    prior = ChiEffChiPPrior(amax=amax)
    xs = np.linspace(0.0, amax, 4000)
    for chi_eff in [0.0, 0.2 * amax, 0.5 * amax, 0.8 * amax]:
        p = prior.cond_prob_chi_p(xs, chi_eff, q)
        integral = _trapz(p, xs)
        assert integral == pytest.approx(1.0, abs=2e-3)


# ==================================================================
# 4. ChiEffChiPPrior: marginal consistency with ChiEffPrior
# ==================================================================
@pytest.mark.parametrize("q,amax", [(1.0, 0.99), (0.6, 0.99), (0.3, 0.99)])
def test_marginal_consistency(q, amax):
    """int p(chi_eff, chi_p) dchi_p ~= ChiEffPrior.p(chi_eff) (both are the
    marginal; the reused ChiEffPrior is itself grid-interpolated, hence ~1-2%)."""
    joint = ChiEffChiPPrior(amax=amax)
    marg = ChiEffPrior(amax=amax)
    m1, m2 = 1.0, q                      # mass_2/mass_1 = q
    xp = np.linspace(0.0, amax, 2000)
    for chi_eff in [0.0, 0.15, 0.35, 0.6]:
        dens = np.exp(joint.logprob(chi_eff, xp, m1, m2))
        integrated = _trapz(dens, xp)
        expected = marg.prob(chi_eff, m1, m2)
        assert integrated == pytest.approx(expected, rel=2e-2)


# ==================================================================
# 5. ChiEffChiPPrior: Monte-Carlo validation of the JOINT density
# ==================================================================
@pytest.mark.parametrize("q,amax", [(1.0, 0.99), (0.5, 0.99), (0.8, 0.4), (0.3, 0.998)])
def test_joint_density_monte_carlo(q, amax):
    """2-D histogram of MC-drawn (chi_eff, chi_p) must match the analytic joint
    density (bin-averaged prediction) in well-populated bins."""
    n = 2_500_000
    chi_eff, chi_p = _draw_chi_eff_chi_p(q, amax, n, seed=12345)

    nb = 34
    ce_edges = np.linspace(-amax, amax, nb + 1)
    cp_edges = np.linspace(0.0, amax, nb + 1)
    counts, _, _ = np.histogram2d(chi_eff, chi_p, bins=[ce_edges, cp_edges])
    area = (ce_edges[1] - ce_edges[0]) * (cp_edges[1] - cp_edges[0])
    hist = counts / (n * area)                   # empirical density

    # Bin-averaged analytic prediction via a 3x3 sub-grid per bin, to remove
    # curvature bias vs the (bin-averaged) histogram.
    m1, m2 = 1.0, q
    sub = np.array([1.0, 3.0, 5.0]) / 6.0
    ce_c = ce_edges[:-1][:, None] + np.diff(ce_edges)[:, None] * sub[None, :]
    cp_c = cp_edges[:-1][:, None] + np.diff(cp_edges)[:, None] * sub[None, :]
    # build [nb, nb] averaged prediction
    pred = np.zeros((nb, nb))
    for a in range(3):
        for b in range(3):
            CE, CP = np.meshgrid(ce_c[:, a], cp_c[:, b], indexing="ij")
            pred += np.exp(chi_eff_chi_p_prior_logprob(
                CE.ravel(), CP.ravel(), m1, m2, amax=amax)).reshape(nb, nb)
    pred /= 9.0

    # Only judge well-populated bins (MC noise ~ 1/sqrt(count)).
    well = counts > 1500
    assert well.sum() > 30                        # enough bins to be meaningful
    rel = np.abs(pred[well] - hist[well]) / hist[well]
    # Robust to Poisson noise: median small, tail bounded.
    assert np.median(rel) < 0.05
    assert np.percentile(rel, 90) < 0.15


# ==================================================================
# 6. Boundary robustness: chi_p -> 0 and edges give finite/clipped logprob
# ==================================================================
def test_chi_p_zero_and_boundary_finite():
    prior = ChiEffChiPPrior(amax=0.99)
    # chi_p exactly 0 and tiny, various chi_eff incl. near +/- amax.
    chi_eff = np.array([0.0, 0.0, 0.5, -0.5, 0.985, -0.985, 0.0])
    chi_p = np.array([0.0, 1e-9, 0.0, 1e-6, 0.0, 0.5, 0.99])
    lp = prior.logprob(chi_eff, chi_p, 1.0, 0.8)
    pr = prior.prob(chi_eff, chi_p, 1.0, 0.8)
    # Never NaN -- that is the corruption this test guards against.
    assert not np.any(np.isnan(lp))
    # -inf occurs exactly where the density is zero, and nowhere else (GW-03).
    # These are all inside the BOX, but the true support is not a box: the
    # density vanishes at chi_p = 0 and pinches out near |chi_eff| -> amax, so a
    # -inf here is the correct answer rather than a floor.
    np.testing.assert_array_equal(np.isneginf(lp), pr == 0.0)
    # The genuinely interior point is a real number.
    assert np.isfinite(lp[1])
    # support() is necessary but not sufficient; the containment holds one way.
    assert np.all(np.isfinite(lp) <= prior.support(chi_eff, chi_p))


def test_logprob_scalar_and_array_shapes():
    prior = ChiEffChiPPrior(amax=0.99)
    s = prior.logprob(0.1, 0.3, 30.0, 25.0)
    assert np.isscalar(s) or np.ndim(s) == 0
    v = prior.logprob(np.array([0.1, 0.2]), np.array([0.3, 0.4]), 30.0, 25.0)
    assert v.shape == (2,)


def test_module_cache_reuse():
    """Repeated module-level calls reuse a cached instance per amax."""
    from gwcat import spin as _spin
    a = chi_eff_chi_p_prior_logprob(0.1, 0.2, 30.0, 25.0, amax=0.99)
    b = chi_eff_chi_p_prior_logprob(0.1, 0.2, 30.0, 25.0, amax=0.99)
    assert a == pytest.approx(b)
    # Keyed by (chi_eff marginal impl, amax) since GW-40i; exact by default.
    assert ("exact", 0.99) in _spin._CHIP_CACHE
    assert _spin._CHIP_CACHE[("exact", 0.99)].impl == "exact"


def test_out_of_support_chi_p_is_minus_inf_not_a_floor():
    """chi_p beyond amax is unreachable -> density 0 -> logprob -inf (GW-03).

    This test previously asserted the -50 sentinel.  That floor is the defect:
    exp(-50) = 2e-22 in a DENOMINATOR gives the sample ~1e21 times the median
    weight, so a handful of them captured essentially all of an event's
    reweighting (measured ESS = 1.0 out of 3337 on GW150914).
    """
    prior = ChiEffChiPPrior(amax=0.99)
    lp = prior.logprob(0.0, 1.2, 1.0, 1.0)      # chi_p=1.2 > amax
    assert lp == -np.inf
    assert not prior.support(0.0, 1.2)
    # ... and the support predicate agrees with the density everywhere.
    chi_eff = np.array([0.0, 0.5, 1.5, -1.2, 0.0])
    chi_p = np.array([0.5, 0.5, 0.2, 0.1, 1.5])
    lp_arr = prior.logprob(chi_eff, chi_p, 30.0, 25.0)
    assert np.all(np.isfinite(lp_arr) <= prior.support(chi_eff, chi_p))


# ==========================================================================
# GW-08: one chi_p definition, enforced
# ==========================================================================
def test_chi_p_impls_agree_between_ingest_and_spin():
    """ingest carried a byte-for-byte duplicate of the Schmidt formula with a
    "unify once both have merged" TODO.  It is now the same object, so the two
    cannot drift."""
    from gwcat.ingest import _chi_p_from_samples
    from gwcat.spin import chi_p_from_components

    assert _chi_p_from_samples is chi_p_from_components

    rng = np.random.default_rng(11)
    n = 5000
    m1 = rng.uniform(5.0, 80.0, n)
    m2 = m1 * rng.uniform(0.05, 1.0, n)      # guarantees m2 <= m1
    a1 = rng.uniform(0.0, 0.99, n)
    a2 = rng.uniform(0.0, 0.99, n)
    c1 = rng.uniform(-1.0, 1.0, n)
    c2 = rng.uniform(-1.0, 1.0, n)
    np.testing.assert_allclose(_chi_p_from_samples(a1, a2, c1, c2, m1, m2),
                               chi_p_from_components(a1, a2, c1, c2, m1, m2),
                               rtol=1e-12, atol=0)


def test_chi_p_reference_values_independent_of_the_implementation():
    """Hand-computed Schmidt values, so the test cannot be tautological.

    Three existing tests in this module compare the function against a fixture
    whose chi_p was written BY the function -- they would pass under any formula.
    These numbers come from k = q(4q+3)/(3q+4) evaluated by hand.
    """
    from gwcat.spin import chi_p_from_components

    # 1. Equal masses (q = 1): k = 1*7/7 = 1, so chi_p = max(a1 sinθ1, a2 sinθ2).
    #    a1=0.8, θ1=90deg -> 0.8 ;  a2=0.5, θ2=90deg -> 0.5 ; max = 0.8
    assert chi_p_from_components(0.8, 0.5, 0.0, 0.0, 30.0, 30.0) == \
        pytest.approx(0.8)

    # 2. q = 0.5: k = 0.5*(2+3)/(1.5+4) = 2.5/5.5 = 5/11.
    #    a1=0.1, θ1=90deg -> 0.1 ; a2=0.9, θ2=90deg -> (5/11)*0.9 = 0.409090...
    assert chi_p_from_components(0.1, 0.9, 0.0, 0.0, 40.0, 20.0) == \
        pytest.approx(5.0 / 11.0 * 0.9)

    # 3. Aligned spins (cos θ = 1) have no in-plane component at all.
    assert chi_p_from_components(0.9, 0.9, 1.0, 1.0, 40.0, 20.0) == \
        pytest.approx(0.0)

    # 4. Only the mass RATIO enters: detector- and source-frame masses agree.
    z = 0.3
    assert chi_p_from_components(0.4, 0.7, 0.2, -0.3, 40.0, 20.0) == \
        pytest.approx(chi_p_from_components(0.4, 0.7, 0.2, -0.3,
                                           40.0 * (1 + z), 20.0 * (1 + z)))


def test_chi_p_refuses_unsorted_masses_instead_of_computing_k_gt_1():
    """m2 > m1 makes the Schmidt coefficient exceed 1, so the "max" picks the
    wrong branch and the result is not chi_p."""
    from gwcat.spin import chi_p_from_components

    with pytest.raises(ValueError, match="primary"):
        chi_p_from_components(0.5, 0.5, 0.0, 0.0, 20.0, 40.0)
    # the message names how badly, and how many
    with pytest.raises(ValueError, match="2 sample"):
        chi_p_from_components(np.full(3, 0.5), np.full(3, 0.5),
                              np.zeros(3), np.zeros(3),
                              np.array([40.0, 20.0, 20.0]),
                              np.array([20.0, 40.0, 30.0]))
    # equal masses are fine (the boundary is inclusive)
    assert chi_p_from_components(0.5, 0.5, 0.0, 0.0, 30.0, 30.0) == \
        pytest.approx(0.5)


def test_chieff_prior_uses_the_exact_q_reflection_not_a_clamp():
    """ChiEffPrior tabulates q = m_primary/(m1+m2) on [0.5, 1].

    Passing the lighter body first used to give q < 0.5, which np.interp silently
    CLAMPED to the q = 0.5 edge -- an equal-mass density for an unequal-mass
    system.  chi_eff is symmetric under relabelling the two bodies at a single
    amax, so the correct mapping is the exact reflection q -> 1 - q.
    """
    from gwcat.spin import ChiEffPrior

    prior = ChiEffPrior(amax=0.99, nq=64, nchi=256)
    chi = np.linspace(-0.8, 0.8, 33)
    m_heavy, m_light = 40.0, 10.0            # q = 0.8 vs the clamped 0.5

    swapped = prior.prob(chi, m_light, m_heavy)
    ordered = prior.prob(chi, m_heavy, m_light)
    np.testing.assert_allclose(swapped, ordered, rtol=1e-12, atol=0)

    # ... and that is NOT the equal-mass density the clamp would have returned.
    equal = prior.prob(chi, 25.0, 25.0)
    assert np.max(np.abs(ordered - equal)) > 1e-3


# ==========================================================================
# GW-03: -inf, support predicates, and the missing factor 1/2
# ==========================================================================
def test_chieff_logprob_is_neg_inf_outside_support():
    from gwcat.spin import ChiEffPrior

    prior = ChiEffPrior(amax=0.99)
    assert prior.logprob(1.5, 30.0, 25.0) == -np.inf
    assert prior.logprob(-1.5, 30.0, 25.0) == -np.inf
    assert np.isfinite(prior.logprob(0.2, 30.0, 25.0))
    assert not prior.support(1.5)
    assert prior.support(0.2)
    # array form, and the containment direction that always holds
    chi = np.array([-1.4, -0.5, 0.0, 0.5, 1.4])
    lp = prior.logprob(chi, 30.0, 25.0)
    assert np.all(np.isfinite(lp) <= prior.support(chi))
    assert not np.any(np.isnan(lp))


# ==========================================================================
# GW-31: "finite log-density" is NOT "in support"
# ==========================================================================
def test_finite_logprob_just_outside_amax_is_not_in_support():
    """The exact reproduction from the review, pinned.

    ``ChiEffPrior.logprob`` deliberately leaves the support unmasked, and the
    grid clamp hands back a small but FINITE density beyond ``amax``.  Anything
    that reads support off ``isfinite(logprob)`` therefore accepts a sample the
    prior excludes -- carrying a density ~1e12 too small into a DENOMINATOR.
    """
    from gwcat.spin import ChiEffPrior

    prior = ChiEffPrior(amax=0.99)
    lp = prior.logprob(0.995, 50.0, 25.0)
    assert np.isfinite(lp)                       # the trap ...
    assert np.exp(lp) == pytest.approx(2.73e-12, rel=0.05)
    assert not prior.support(0.995)              # ... and the truth


def test_logprob_in_support_applies_the_support_mask():
    from gwcat.spin import chi_eff_prior_logprob_in_support

    chi = np.array([0.0, 0.5, 0.995, 1.5, -0.995])
    lp, sup = chi_eff_prior_logprob_in_support(chi, 50.0, 25.0, amax=0.99)
    assert list(sup) == [True, True, False, False, False]
    assert np.all(np.isfinite(lp[sup]))
    assert np.all(lp[~sup] == -np.inf)
    # exp(-inf) is exactly zero density, not a small one
    assert np.all(np.exp(lp[~sup]) == 0.0)


def test_logprob_in_support_carries_the_per_body_ceilings():
    """A restricted secondary must not be widened to the primary's ceiling."""
    from gwcat.spin import (ChiEffPrior, ChiEffPriorExact,
                            chi_eff_prior_logprob_in_support)

    chi = np.array([0.1, 0.3])
    # Legacy grid: bit-identical to the table class it wraps.
    lp, sup = chi_eff_prior_logprob_in_support(chi, 40.0, 10.0,
                                               amax=0.99, amax_2=0.05,
                                               impl="grid")
    ref = ChiEffPrior(amax=0.99, amax_2=0.05)
    np.testing.assert_allclose(lp, ref.logprob(chi, 40.0, 10.0), rtol=0,
                               atol=0)
    assert bool(sup.all())
    # ... and that is NOT the symmetric-ceiling density.
    sym = ChiEffPrior(amax=0.99).logprob(chi, 40.0, 10.0)
    assert not np.allclose(lp, sym, rtol=1e-6)

    # The default (exact) carries the per-body ceilings the same way.
    lp, sup = chi_eff_prior_logprob_in_support(chi, 40.0, 10.0,
                                               amax=0.99, amax_2=0.05)
    ref = ChiEffPriorExact(amax=0.99, amax_2=0.05)
    np.testing.assert_allclose(lp, ref.logprob(chi, 40.0, 10.0), rtol=0,
                               atol=0)
    assert bool(sup.all())
    sym = ChiEffPriorExact(amax=0.99).logprob(chi, 40.0, 10.0)
    assert not np.allclose(lp, sym, rtol=1e-6)


def test_no_minus_fifty_sentinel_survives_anywhere():
    """The floor was a magic -50 in two prior classes and four call sites; a
    grep-level guard is the cheapest way to stop it coming back."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "gwcat"
    offenders = []
    for path in root.rglob("*.py"):
        # skip stale duplicate trees (build/, .ipynb_checkpoints/) if any return
        if any(part.startswith(".") or part == "build" for part in path.parts):
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if "np.clip" not in line:
                continue
            if "-50.0" in line or "-50," in line:
                offenders.append(f"{path.relative_to(root)}:{i}: {line.strip()}")
    assert not offenders, "a -50 clip came back:\n" + "\n".join(offenders)


def test_single_spin_pdf_normalises_to_one():
    """p(s) = -ln(|s|/amax) / (2*amax) integrates to 1 over [-amax, amax].

    The factor 1/2 was missing.  Both in-tree consumers renormalise their
    convolution afterwards so it was harmless there, but the density was wrong by
    2x for anyone reusing it standalone.
    """
    from gwcat.spin import ChiEffPrior

    amax = 0.99
    s = np.linspace(-amax, amax, 400001)
    p = ChiEffPrior._single_spin_pdf(s, amax)
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    # 1e-3 not 1e-4: the density has an integrable log singularity at s = 0 that
    # the implementation caps at a finite value, so a uniform-grid trapezoid
    # slightly over-counts the central bins.
    assert trapz(p, s) == pytest.approx(1.0, rel=1e-3)
    # symmetric (to grid round-off), and peaked at the integrable log singularity
    np.testing.assert_allclose(p, p[::-1], rtol=1e-9)
    assert p[len(p) // 2] > p[0]


def test_single_spin_pdf_matches_monte_carlo():
    """Independent check: draw a ~ U(0, amax), cos(theta) ~ U(-1, 1) and compare
    the histogram of s = a*cos(theta) against the analytic density."""
    from gwcat.spin import ChiEffPrior

    amax = 0.99
    rng = np.random.default_rng(4)
    n = 4_000_000
    s = rng.uniform(0.0, amax, n) * rng.uniform(-1.0, 1.0, n)
    edges = np.linspace(-amax, amax, 61)
    hist, _ = np.histogram(s, bins=edges, density=True)
    mid = 0.5 * (edges[1:] + edges[:-1])
    pred = ChiEffPrior._single_spin_pdf(mid, amax)
    # skip the two bins straddling the log singularity at s = 0
    keep = np.abs(mid) > 2.0 * (edges[1] - edges[0])
    rel = np.abs(hist[keep] / pred[keep] - 1.0)
    assert np.median(rel) < 0.02, f"median rel dev {np.median(rel):.4f}"


def test_chieff_prior_renormalisation_is_unchanged_by_the_factor():
    """The convolved chi_eff prior still integrates to 1, i.e. the 1/2 fix did
    not move any number the two in-tree consumers see."""
    from gwcat.spin import ChiEffPrior

    prior = ChiEffPrior(amax=0.99, nq=32, nchi=1024)
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    for m1, m2 in ((30.0, 25.0), (40.0, 10.0), (30.0, 30.0)):
        p = prior.prob(prior.chi_grid, m1, m2)
        assert trapz(p, prior.chi_grid) == pytest.approx(1.0, rel=2e-3)


# ==========================================================================
# GW-04: independent amax_1 / amax_2
# ==========================================================================
def test_equal_amax_is_bit_identical_to_the_single_amax_construction():
    """The default path must not move: passing amax_2 explicitly equal to amax
    reproduces the historical table exactly."""
    from gwcat.spin import ChiEffPrior

    a = ChiEffPrior(amax=0.99, nq=32, nchi=512)
    b = ChiEffPrior(amax=0.99, amax_2=0.99, nq=32, nchi=512)
    np.testing.assert_array_equal(a.table, b.table)
    assert (a.amax_1, a.amax_2, a.amax) == (0.99, 0.99, 0.99)


def test_restricted_secondary_amax_changes_the_density():
    """A low-spin secondary (a_2 ~ U(0, 0.05)) is a genuinely different prior.

    Forcing one shared amax inflated the secondary's support ~20x; the two
    densities must not agree.
    """
    from gwcat.spin import ChiEffPrior

    shared = ChiEffPrior(amax=0.99, nq=48, nchi=1024)
    split = ChiEffPrior(amax=0.99, amax_2=0.05, nq=48, nchi=1024)
    assert (split.amax_1, split.amax_2) == (0.99, 0.05)
    # the chi_eff support bound is max(amax_1, amax_2)
    assert split.amax == 0.99
    chi = np.linspace(-0.5, 0.5, 41)
    assert np.max(np.abs(split.prob(chi, 40.0, 5.0)
                         - shared.prob(chi, 40.0, 5.0))) > 0.1


@pytest.mark.parametrize("amax_2", [0.99, 0.5, 0.05])
def test_split_amax_prior_still_normalises(amax_2):
    from gwcat.spin import ChiEffPrior

    prior = ChiEffPrior(amax=0.99, amax_2=amax_2, nq=48, nchi=2048)
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    for m1, m2 in ((30.0, 25.0), (40.0, 5.0)):
        p = prior.prob(prior.chi_grid, m1, m2)
        assert trapz(p, prior.chi_grid) == pytest.approx(1.0, rel=5e-3)


def test_split_amax_refuses_swapped_masses():
    """With amax_1 != amax_2 the bodies are distinguishable, so the q -> 1-q
    reflection is invalid and m1 must really be the primary."""
    from gwcat.spin import ChiEffPrior

    prior = ChiEffPrior(amax=0.99, amax_2=0.05, nq=32, nchi=512)
    assert np.isfinite(prior.prob(0.1, 40.0, 5.0))          # ordered: fine
    with pytest.raises(ValueError, match="more massive body as m1"):
        prior.prob(0.1, 5.0, 40.0)
    # a shared amax keeps the reflection, since the bodies are exchangeable
    sym = ChiEffPrior(amax=0.99, nq=32, nchi=512)
    assert sym.prob(0.1, 5.0, 40.0) == pytest.approx(sym.prob(0.1, 40.0, 5.0))
