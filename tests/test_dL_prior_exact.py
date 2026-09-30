"""The exact UniformSourceFrame distance prior (GW-40i).

bilby's ``UniformSourceFrame`` -- what gwcat used to divide out -- interpolates
its density from a 1000-point grid across ``[dmin, dmax]``; its ln-shape is off
the exact density by up to 3.4e-3 on the 259-event PE (gate round 1, G5) and by
more near a low ``dmin``.  ``impl="exact"`` computes the density from accurate
cosmology integrals, normalised over the DECLARED bounds.  These tests pin it
against an independent mpmath evaluation
(``fixtures/usf_mpmath_reference.json``, made by
``fixtures/make_usf_mpmath_reference.py``) to <= 1e-12 relative -- in value,
so a fortiori in shape -- at LAL Planck15 (67.90, 0.3065) and two other
cosmologies, inside and outside the bounds, for UniformSourceFrame and
UniformComovingVolume.
"""
import json
import pathlib
import time

import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM, Planck15
from scipy import integrate

from gwcat.cosmology import (LAL_PLANCK15, USF_IMPL_DEFAULT, USF_IMPLS,
                             _exact_distances, dL_prior_prob, make_cosmology,
                             resolve_usf_impl, uniform_source_frame_prob,
                             usf_prob_exact)
from gwcat.ingest import IngestConfig

FIXTURE = (pathlib.Path(__file__).resolve().parent / "fixtures"
           / "usf_mpmath_reference.json")


@pytest.fixture(scope="module")
def blocks():
    return json.loads(FIXTURE.read_text())["rows"]


def test_fixture_covers_the_production_cosmology_and_both_sides_of_bounds(
        blocks):
    assert any(b["H0"] == 67.90 and b["Om0"] == 0.3065 for b in blocks)
    assert any(not b["time_dilation"] for b in blocks)
    assert all(min(b["dL"]) < b["dmin"] and max(b["dL"]) > b["dmax"]
               for b in blocks)


def test_exact_matches_mpmath_to_1e12(blocks):
    worst = 0.0
    for b in blocks:
        cos = FlatLambdaCDM(H0=b["H0"], Om0=b["Om0"])
        dL = np.asarray(b["dL"], dtype=float)
        ref = np.array([float(x) for x in b["p"]])
        p = usf_prob_exact(dL, cos, b["dmin"], b["dmax"],
                           time_dilation=b["time_dilation"])
        rel = np.abs(p / ref - 1.0)
        worst = max(worst, float(rel.max()))
        assert rel.max() <= 1e-12, (b["H0"], b["Om0"], b["dmin"], b["dmax"],
                                    b["time_dilation"], rel.max())
        # the public entry point is the same numbers
        kind = ("UniformSourceFrame" if b["time_dilation"]
                else "UniformComovingVolume")
        q = dL_prior_prob(dL, kind=kind, cosmology=cos, dmin=b["dmin"],
                          dmax=b["dmax"])
        np.testing.assert_array_equal(q, p)
    assert worst < 1e-13


@pytest.mark.parametrize("bounds", [(1.0, 15000.0), (10.0, 10000.0),
                                    (400.0, 900.0)])
@pytest.mark.parametrize("td", [True, False])
def test_normalised_over_the_declared_bounds(bounds, td):
    dmin, dmax = bounds
    v, _ = integrate.quad(
        lambda x: float(usf_prob_exact(np.array([x]), LAL_PLANCK15, dmin,
                                       dmax, time_dilation=td)[0]),
        dmin, dmax, epsabs=0, epsrel=1e-13, limit=400)
    assert abs(v - 1.0) < 1e-11


def test_distances_agree_with_astropy_closed_form_and_round_trip():
    cos = FlatLambdaCDM(H0=67.9, Om0=0.3065)
    ex = _exact_distances(cos)
    # astropy's hypergeometric closed form loses digits to cancellation as
    # z -> 0 (1e-13 at z = 0.01, 2.5e-9 at z = 1e-8), so compare above 0.05;
    # the low-z accuracy of the exact path is pinned by the mpmath fixture.
    z = np.concatenate([np.geomspace(0.05, 20.0, 400), [0.3, 1.0, 3.0]])
    np.testing.assert_allclose(ex.comoving(z), cos.comoving_distance(z).value,
                               rtol=1e-13, atol=0)
    d = np.geomspace(0.1, 2.0e5, 2000)
    np.testing.assert_allclose(ex.luminosity(ex.z_of_dL(d)), d, rtol=1e-15,
                               atol=0)


def test_general_E_of_z_is_honoured_radiation_and_neutrinos():
    """astropy Planck15 carries radiation and massive neutrinos: the exact
    path integrates its OWN 1/E(z), so it agrees with astropy's distances."""
    ex = _exact_distances(Planck15)
    z = np.geomspace(0.01, 5.0, 60)
    np.testing.assert_allclose(ex.comoving(z),
                               Planck15.comoving_distance(z).value,
                               rtol=1e-10, atol=0)


def test_not_widened_and_untruncated_outside_the_bounds():
    dL = np.array([5.0, 300.0, 3000.0, 6000.0])
    p, info = uniform_source_frame_prob(dL, LAL_PLANCK15, 10.0, 4000.0,
                                        return_info=True)
    assert info["impl"] == "exact"
    assert info["widened"] is False
    assert (info["eval_min"], info["eval_max"]) == (10.0, 4000.0)
    assert info["n_below_dmin"] == 1 and info["n_above_dmax"] == 1
    assert np.all(p > 0)
    # the same continuous formula on both sides of the bound: the value at a
    # point outside equals the in-bounds normalisation times the shape
    wide = usf_prob_exact(dL, LAL_PLANCK15, 1.0, 10000.0)
    np.testing.assert_allclose(p / wide, (p / wide)[0], rtol=1e-13)


def test_invalid_distances_and_bounds():
    p = usf_prob_exact(np.array([-1.0, np.nan, 0.0]), LAL_PLANCK15, 1.0, 10.0)
    assert np.isnan(p[0]) and np.isnan(p[1]) and p[2] == 0.0
    with pytest.raises(ValueError, match="dmax > dmin"):
        usf_prob_exact(np.array([1.0]), LAL_PLANCK15, 10.0, 10.0)


def test_non_flat_cosmology_is_refused():
    from astropy.cosmology import LambdaCDM
    with pytest.raises(NotImplementedError, match="FLAT"):
        usf_prob_exact(np.array([100.0]),
                       LambdaCDM(H0=70, Om0=0.3, Ode0=0.6), 1.0, 1000.0)


def test_default_is_exact_and_the_legacy_impls_stay_reachable():
    assert USF_IMPLS == ("exact", "bilby", "astropy")
    assert USF_IMPL_DEFAULT == "exact"
    assert resolve_usf_impl() == "exact"
    assert IngestConfig().dL_prior_impl == "exact"
    pytest.importorskip("bilby")
    assert resolve_usf_impl("auto") == "bilby"


def test_legacy_bilby_differs_only_by_its_interpolation_error():
    """What GW-40i removes, measured: bilby's ln-shape error over typical
    posterior ranges (2.3e-3 over 100-8000 Mpc at [10, 10000]; 3.2e-5 over
    300-800 Mpc at [10, 4000])."""
    pytest.importorskip("bilby")
    for dmin, dmax, lo, hi, bound in [(10.0, 10000.0, 100.0, 8000.0, 3e-3),
                                      (10.0, 4000.0, 300.0, 800.0, 1e-4)]:
        x = np.linspace(lo, hi, 5001)
        d = (np.log(uniform_source_frame_prob(x, LAL_PLANCK15, dmin, dmax,
                                              impl="bilby"))
             - np.log(uniform_source_frame_prob(x, LAL_PLANCK15, dmin, dmax)))
        assert np.ptp(d) < bound
        assert np.ptp(d) > 1e-7        # the switch is real


def test_speed_one_million_samples():
    x = np.random.default_rng(0).uniform(10.0, 15000.0, 1_000_000)
    cos = make_cosmology(67.9, 0.3065)
    t0 = time.perf_counter()
    p = usf_prob_exact(x, cos, 1.0, 15000.0)
    dt = time.perf_counter() - t0
    assert np.all(p > 0)
    assert dt < 30.0, f"1e6 exact dL-prior evaluations took {dt:.1f} s"
