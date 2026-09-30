"""The exact isotropic uniform-magnitude chi_eff prior (GW-40i).

``gwcat.spin.ChiEffPrior`` interpolates a 200 x 2000 table and is off by up to
~1e-2 in ln p (4.2e-3 relative on the 259-event PE, gate round 1).  GW-40i
evaluates the convolution exactly (:mod:`gwcat.chi_eff_exact`).  These tests pin
it against an INDEPENDENT mpmath evaluation of the defining integral
(``fixtures/chi_eff_iso_mpmath_reference.json``, made by
``fixtures/make_chi_eff_iso_mpmath_reference.py``), over the full support
including q -> 1, q -> 1e-8, chi_eff -> 0, every kink of the piecewise form
and |chi_eff| -> amax, to <= 1e-12 relative; and check the switch that keeps
the legacy grid reachable (and only reachable explicitly).
"""
import json
import pathlib
import time

import numpy as np
import pytest
from scipy import integrate

from gwcat.chi_eff_exact import chi_eff_iso_prob, chi_eff_iso_prob_q
from gwcat.spin import (CHI_EFF_PRIOR_IMPLS, CHI_EFF_PRIOR_METHODS,
                        ChiEffChiPPrior, ChiEffPrior, ChiEffPriorExact,
                        chi_eff_prior_impl, chi_eff_prior_logprob,
                        chi_eff_prior_logprob_in_support,
                        current_chi_eff_prior_impl, get_chi_eff_prior,
                        resolve_chi_eff_prior_impl)

FIXTURE = (pathlib.Path(__file__).resolve().parent / "fixtures"
           / "chi_eff_iso_mpmath_reference.json")
RTOL = 1e-12


@pytest.fixture(scope="module")
def ref():
    rows = json.loads(FIXTURE.read_text())["rows"]
    chi = np.array([r["chi_eff"] for r in rows])
    m1 = np.array([r["m1"] for r in rows])
    m2 = np.array([r["m2"] for r in rows])
    a1 = np.array([r["amax_1"] for r in rows])
    shared = np.array([r["amax_2"] is None for r in rows])
    a2 = np.where(shared, a1, [r["amax_2"] if r["amax_2"] is not None
                               else np.nan for r in rows])
    p = np.array([float(r["p"]) for r in rows])
    return dict(chi=chi, m1=m1, m2=m2, a1=a1, a2=a2, shared=shared, p=p)


def _exact(ref):
    out = np.empty_like(ref["p"])
    s = ref["shared"]
    out[s] = chi_eff_iso_prob(ref["chi"][s], ref["m1"][s], ref["m2"][s],
                              ref["a1"][s])
    out[~s] = chi_eff_iso_prob(ref["chi"][~s], ref["m1"][~s], ref["m2"][~s],
                               ref["a1"][~s], ref["a2"][~s])
    return out


# --------------------------------------------------------------------------
# 1. Accuracy against the independent mpmath reference
# --------------------------------------------------------------------------
def test_fixture_covers_the_edges(ref):
    """The reference must actually contain the regimes the claim is about."""
    q = np.minimum(ref["m1"], ref["m2"]) / np.maximum(ref["m1"], ref["m2"])
    top = (ref["a1"] * ref["m1"] + ref["a2"] * ref["m2"]) / (ref["m1"]
                                                             + ref["m2"])
    x = np.abs(ref["chi"])
    assert np.sum(q > 1 - 1e-6) >= 200                     # q -> 1
    assert np.sum(q < 1e-5) >= 200                         # extreme q
    assert np.sum(x == 0.0) >= 50 and np.sum((x > 0) & (x < 1e-9)) >= 100
    assert np.sum((top - x) / top < 1e-9) >= 100           # |chi| -> amax
    assert np.sum(ref["p"] > 0) == ref["p"].size
    assert np.sum(~ref["shared"]) >= 300                   # amax_1 != amax_2
    assert np.sum(ref["m2"] > ref["m1"]) >= 100            # either mass order


def test_exact_matches_mpmath_to_1e12_everywhere(ref):
    p = _exact(ref)
    rel = np.abs(p / ref["p"] - 1.0)
    worst = int(np.argmax(rel))
    assert rel.max() <= RTOL, (
        f"max rel err {rel.max():.3e} at chi={ref['chi'][worst]!r}, "
        f"m1={ref['m1'][worst]!r}, m2={ref['m2'][worst]!r}, "
        f"amax=({ref['a1'][worst]!r}, {ref['a2'][worst]!r})")


def test_q_interface_is_the_same_density(ref):
    s = ref["shared"] & (ref["m1"] == 1.0)
    q = ref["m2"][s]
    p = chi_eff_iso_prob_q(ref["chi"][s], q, ref["a1"][s])
    assert np.max(np.abs(p / ref["p"][s] - 1.0)) <= RTOL


def test_exact_matches_live_mpmath_on_fresh_random_points():
    """A second, freshly drawn sample, when mpmath is importable."""
    mp = pytest.importorskip("mpmath")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mkref", FIXTURE.parent / "make_chi_eff_iso_mpmath_reference.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    rng = np.random.default_rng(12345)
    rows = []
    for _ in range(40):
        m1 = float(rng.uniform(3, 200))
        m2 = float(m1 * 10 ** rng.uniform(-3, 0))
        chi = float(rng.uniform(-0.99, 0.99))
        rows.append((chi, m1, m2, 0.99, None))
    refv = np.array([float(mk.reference(r)) for r in rows])
    p = chi_eff_iso_prob([r[0] for r in rows], [r[1] for r in rows],
                         [r[2] for r in rows], 0.99)
    assert mp.__version__
    assert np.max(np.abs(p / refv - 1.0)) <= RTOL


# --------------------------------------------------------------------------
# 2. Structural properties
# --------------------------------------------------------------------------
@pytest.mark.parametrize("q", [1.0, 0.8, 0.5, 0.1, 0.01, 1e-3])
@pytest.mark.parametrize("amax", [(0.99, None), (0.5, None), (0.99, 0.05)])
def test_normalised_on_its_support(q, amax):
    a1, a2 = amax
    a2e = a1 if a2 is None else a2
    A1, A2 = a1 / (1 + q), a2e * q / (1 + q)
    top = A1 + A2
    kinks = sorted({0.0, A1, A2, abs(A1 - A2)} - {top})

    def f(x):
        return float(chi_eff_iso_prob(x, 1.0, q, a1, a2))
    edges = [k for k in kinks if 0 <= k < top] + [top]
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        v, _ = integrate.quad(f, lo, hi, epsabs=0, epsrel=1e-13, limit=400)
        total += v
    assert abs(2.0 * total - 1.0) < 1e-10


def test_symmetric_in_chi_and_in_the_mass_labels_when_amax_is_shared():
    rng = np.random.default_rng(3)
    chi = rng.uniform(-0.99, 0.99, 2000)
    m1 = rng.uniform(5, 80, 2000)
    m2 = rng.uniform(5, 80, 2000)
    p = chi_eff_iso_prob(chi, m1, m2, 0.99)
    np.testing.assert_array_equal(p, chi_eff_iso_prob(-chi, m1, m2, 0.99))
    np.testing.assert_allclose(chi_eff_iso_prob(chi, m2, m1, 0.99), p,
                               rtol=1e-13, atol=0)


def test_per_body_ceilings_follow_the_body_not_the_heavier_mass():
    """Body 1 keeps amax_1 whichever body is heavier (no reflection)."""
    chi = np.array([0.05, 0.2, 0.4])
    a = chi_eff_iso_prob(chi, 10.0, 30.0, 0.99, 0.05)     # light body: 0.99
    b = chi_eff_iso_prob(chi, 30.0, 10.0, 0.05, 0.99)     # same system, relabelled
    np.testing.assert_allclose(a, b, rtol=1e-13, atol=0)
    assert not np.allclose(a, chi_eff_iso_prob(chi, 10.0, 30.0, 0.05, 0.99),
                           rtol=1e-3)


def test_zero_on_and_past_the_edge_positive_just_inside():
    for m1, m2 in [(30.0, 30.0), (40.0, 10.0), (100.0, 1.0)]:
        assert chi_eff_iso_prob(0.99, m1, m2, 0.99) == 0.0
        assert chi_eff_iso_prob(-0.995, m1, m2, 0.99) == 0.0
        assert chi_eff_iso_prob(np.nextafter(0.99, 0), m1, m2, 0.99) > 0.0
    lp, sup = chi_eff_prior_logprob_in_support([0.99, 0.995, 0.98],
                                               30.0, 30.0, amax=0.99)
    assert lp[0] == -np.inf and lp[1] == -np.inf and np.isfinite(lp[2])
    assert list(sup) == [False, False, True]


def test_finite_and_nonnegative_arbitrarily_close_to_the_edge():
    """Down to the last representable chi below the edge, for shared and
    per-body ceilings (whose edge distance is formed in double-double and can
    be far below 1 ulp of amax), the density is finite, >= 0, and shrinks."""
    for m1, m2, a1, a2 in [(30.0, 30.0, 0.99, None), (40.0, 7.0, 0.99, 0.05),
                           (7.0, 40.0, 0.99, 0.998)]:
        a2e = a1 if a2 is None else a2
        top = (a1 * m1 + a2e * m2) / (m1 + m2)
        chi = [np.nextafter(top, 0), top * (1 - 1e-15), top * (1 - 1e-12),
               top * (1 - 1e-9)]
        p = chi_eff_iso_prob(chi, m1, m2, a1, a2)
        assert np.all(np.isfinite(p)) and np.all(p >= 0), (m1, m2, a1, a2, p)
        assert np.all(np.diff(p) >= 0)


def test_invalid_masses_give_nan_density_and_minus_inf_logprob():
    p = chi_eff_iso_prob([0.1, 0.1, 0.1], [np.nan, -1.0, 30.0],
                         [10.0, 10.0, 0.0], 0.99)
    assert np.all(np.isnan(p))
    lp = ChiEffPriorExact(0.99).logprob([0.1, 0.1], [np.nan, 30.0],
                                        [10.0, 0.0])
    assert np.all(lp == -np.inf)


def test_small_q_tends_to_the_single_spin_density():
    """q -> 0: chi_eff -> w1 s1, p -> ln(A1/|chi|)/(2 A1)."""
    q = 1e-7
    A1 = 0.99 / (1 + q)
    chi = np.array([0.05, 0.3, 0.7])
    single = np.log(A1 / chi) / (2 * A1)
    np.testing.assert_allclose(chi_eff_iso_prob_q(chi, q, 0.99), single,
                               rtol=1e-5)


@pytest.mark.parametrize("q", [1.0, 0.5, 0.1, 1e-3])
def test_value_at_zero_is_the_known_closed_form(q):
    """p(0 | q, amax) = (1 + q)(2 - ln q) / (2 amax): the chi = 0 case of the
    closed form in Callister 2021 (arXiv:2104.09508); 2/amax at q = 1."""
    amax = 0.99
    expected = (1 + q) * (2 - np.log(q)) / (2 * amax)
    assert chi_eff_iso_prob_q(0.0, q, amax) == pytest.approx(expected,
                                                             rel=1e-14)


# --------------------------------------------------------------------------
# 3. The implementation switch, and what the legacy grid still is
# --------------------------------------------------------------------------
def test_default_is_exact_and_grid_only_on_request():
    assert CHI_EFF_PRIOR_IMPLS == ("exact", "grid")
    assert current_chi_eff_prior_impl() == "exact"
    assert isinstance(get_chi_eff_prior(0.99), ChiEffPriorExact)
    assert isinstance(get_chi_eff_prior(0.99, impl="grid"), ChiEffPrior)
    with chi_eff_prior_impl("grid"):
        assert current_chi_eff_prior_impl() == "grid"
        assert isinstance(get_chi_eff_prior(0.99), ChiEffPrior)
        with chi_eff_prior_impl("exact"):
            assert current_chi_eff_prior_impl() == "exact"
        assert current_chi_eff_prior_impl() == "grid"
    assert current_chi_eff_prior_impl() == "exact"
    with pytest.raises(ValueError, match="impl must be one of"):
        resolve_chi_eff_prior_impl("interp")
    with pytest.raises(ValueError):
        with chi_eff_prior_impl("bilinear"):
            pass
    assert set(CHI_EFF_PRIOR_METHODS) == set(CHI_EFF_PRIOR_IMPLS)


def test_scope_restores_even_on_error():
    with pytest.raises(RuntimeError):
        with chi_eff_prior_impl("grid"):
            raise RuntimeError("boom")
    assert current_chi_eff_prior_impl() == "exact"


def test_legacy_grid_is_bit_identical_to_the_table_class():
    rng = np.random.default_rng(5)
    chi = rng.uniform(-0.9, 0.9, 500)
    m1 = rng.uniform(10, 60, 500)
    m2 = rng.uniform(3, 10, 500)
    np.testing.assert_array_equal(
        chi_eff_prior_logprob(chi, m1, m2, amax=0.99, impl="grid"),
        ChiEffPrior(amax=0.99).logprob(chi, m1, m2))


def test_grid_differs_from_exact_only_by_its_interpolation_error():
    """The documented size of what GW-40i removes, on isotropic draws.

    Measured: max |d ln p| 9.2e-3 (at chi ~ 0 for q ~ 0.05, where the grid
    cannot resolve the log peak), 99.9th percentile 3.2e-3, median 2.1e-4.
    """
    rng = np.random.default_rng(0)
    n = 200000
    m1 = rng.uniform(5, 100, n)
    q = rng.uniform(0.05, 1, n)
    m2 = m1 * q
    a1, a2 = rng.uniform(0, 0.99, n), rng.uniform(0, 0.99, n)
    c1, c2 = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    chi = (m1 * a1 * c1 + m2 * a2 * c2) / (m1 + m2)
    d = np.abs(chi_eff_prior_logprob(chi, m1, m2, impl="grid")
               - chi_eff_prior_logprob(chi, m1, m2))
    assert d.max() < 1.2e-2
    assert np.median(d) < 5e-4
    assert d.max() > 1e-4     # the switch is real


def test_chi_p_joint_prior_uses_the_selected_marginal():
    ex = ChiEffChiPPrior(amax=0.99)
    gr = ChiEffChiPPrior(amax=0.99, impl="grid")
    assert ex.impl == "exact" and gr.impl == "grid"
    chi, chip, m1, m2 = 0.1, 0.3, 30.0, 20.0
    ratio = ex.prob(chi, chip, m1, m2) / gr.prob(chi, chip, m1, m2)
    expected = (ChiEffPriorExact(0.99).prob(chi, m1, m2)
                / ChiEffPrior(0.99).prob(chi, m1, m2))
    assert ratio == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------
# 4. Fast enough for the production products
# --------------------------------------------------------------------------
def test_speed_for_the_pe_and_injection_row_counts():
    """~1.1M PE rows and ~2.5M injection rows (measured ~1 s and ~2 s)."""
    rng = np.random.default_rng(1)
    budget = 0.0
    for n in (1_100_000, 2_500_000):
        m1 = rng.uniform(5, 300, n)
        m2 = m1 * rng.uniform(0.01, 1, n)
        a1, a2 = rng.uniform(0, 0.99, n), rng.uniform(0, 0.99, n)
        c1, c2 = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
        chi = (m1 * a1 * c1 + m2 * a2 * c2) / (m1 + m2)
        t0 = time.perf_counter()
        lp, sup = chi_eff_prior_logprob_in_support(chi, m1, m2, amax=0.99)
        budget += time.perf_counter() - t0
        assert np.all(np.isfinite(lp)) and np.all(sup)
    assert budget < 60.0, f"3.6M evaluations took {budget:.1f} s"
