"""Tests for the distance-prior CLASS contract (GW-02).

The pre-GW-02 parser read only ``minimum``/``maximum``/``cosmology`` out of the
analytic ``luminosity_distance`` prior repr, so:

  * every row was evaluated as UniformSourceFrame regardless of the class it
    declared -- 81 of the 282 real rows declare ``PowerLaw(alpha=2)``; and
  * ``if "Planck15" in tok`` matched ``'Planck15_LAL'`` as well as
    ``'Planck15'``, silently giving 190 O4 rows astropy's (67.74, 0.3075)
    instead of LAL's (67.90, 0.3065).

What the class means depends on the release flavour, and that is the subtlety
this module pins.  For the GWTC-2.1/3 ``*_cosmo.h5`` files gwcat ingests, the
posteriors were reweighted by the LVK to a comoving-volume prior while
``priors/analytic`` still records the original dL^2 SAMPLING prior -- verified on
real files, where the prior samples follow dL^2 (KS 0.012) and not
UniformSourceFrame (KS 0.271).  So the effective prior stays UniformSourceFrame
there, and both classes are recorded rather than conflated.

All fixtures are synthetic repr strings and synthetic prior samples -- no
network, no real data files.
"""
import warnings

import numpy as np
import pytest

from gwcat.cosmology import (DL_PRIOR_KINDS, LAL_PLANCK15, NAMED_COSMOLOGIES,
                            PLANCK15, DistancePriorKindError, dL_prior_prob,
                            make_cosmology, power_law_dL_prob,
                            uniform_source_frame_prob)
from gwcat.ingest import (IngestConfig, PriorMismatchError, _parse_analytic_dL,
                          detect_release_flavour, resolve_dL_prior,
                          validate_prior_against_samples)

# The three repr shapes the real releases actually use, verbatim.
REPR_POWERLAW = ("PowerLaw(alpha=2, minimum=10, maximum=10000, "
                 "name='luminosity_distance', latex_label='$d_L$', unit='Mpc', "
                 "boundary=None)")
REPR_USF_LAL = ("bilby.gw.prior.UniformSourceFrame(minimum=10.0, "
                "maximum=4000.0, cosmology='Planck15_LAL', "
                "name='luminosity_distance', latex_label='$d_L$', unit='Mpc', "
                "boundary=None)")
REPR_USF_INLINE = ("bilby.gw.prior.UniformSourceFrame(minimum=1.0, "
                   "maximum=500.0, cosmology=LambdaCDM(name=None, H0=67.9, "
                   "Om0=0.3065, Ode0=0.6935, Tcmb0=0., Neff=3.04, m_nu=None, "
                   "Ob0=None), name='luminosity_distance', unit='Mpc')")
REPR_USF_ASTROPY = ("bilby.gw.prior.UniformSourceFrame(minimum=10.0, "
                    "maximum=4000.0, cosmology='Planck15', "
                    "name='luminosity_distance')")


# --------------------------------------------------------------------------
# 1. The parser reads the class, and maps the cosmology token EXACTLY
# --------------------------------------------------------------------------
def test_parses_power_law_class_and_alpha():
    p = _parse_analytic_dL(REPR_POWERLAW)
    assert p.kind == "PowerLaw"
    assert p.alpha == 2.0
    assert (p.dmin, p.dmax) == (10.0, 10000.0)
    # A power law in dL needs no cosmology, and none is declared.
    assert p.H0 is None and p.cosmology_name == ""


def test_planck15_lal_is_not_astropy_planck15():
    """The substring bug: 'Planck15' in 'Planck15_LAL' is True."""
    assert "Planck15" in "Planck15_LAL"          # the old test that misfired
    lal = _parse_analytic_dL(REPR_USF_LAL)
    astro = _parse_analytic_dL(REPR_USF_ASTROPY)

    assert lal.cosmology_name == "Planck15_LAL"
    assert (lal.H0, lal.Om0) == pytest.approx((67.90, 0.3065))
    assert astro.cosmology_name == "Planck15"
    assert (astro.H0, astro.Om0) == pytest.approx((67.74, 0.3075))
    # ... and the two are genuinely different cosmologies.
    assert lal.H0 != astro.H0 and lal.Om0 != astro.Om0


def test_inline_cosmology_repr_is_read_whole():
    """``cosmology=LambdaCDM(name=None, H0=..., Om0=...)`` must be parsed from
    INSIDE the nested repr, not by scanning the whole prior string (where a
    recalibration parameter could match H0=)."""
    p = _parse_analytic_dL(REPR_USF_INLINE)
    assert p.kind == "UniformSourceFrame"
    assert (p.H0, p.Om0) == pytest.approx((67.9, 0.3065))
    assert (p.dmin, p.dmax) == (1.0, 500.0)
    assert p.cosmology_recognized is True


def test_nested_repr_class_is_not_mistaken_for_the_prior_class():
    """The class must come from the FIRST ``Name(``, so a dotted module path is
    stripped and the nested ``LambdaCDM(`` cannot win."""
    assert _parse_analytic_dL(REPR_USF_INLINE).kind == "UniformSourceFrame"


def test_unknown_cosmology_token_is_recorded_not_guessed():
    p = _parse_analytic_dL(REPR_USF_LAL.replace("Planck15_LAL", "WMAP9_LAL"))
    assert p.cosmology_name == "WMAP9_LAL"
    assert p.cosmology_recognized is False
    assert p.H0 is None, "an unknown cosmology must NOT be silently resolved"


def test_named_cosmologies_are_distinct_objects():
    assert set(NAMED_COSMOLOGIES) == {"Planck15", "Planck15_LAL"}
    assert NAMED_COSMOLOGIES["Planck15_LAL"] is LAL_PLANCK15
    assert NAMED_COSMOLOGIES["Planck15"] is PLANCK15


# --------------------------------------------------------------------------
# 2. Release flavour
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name,expect", [
    ("IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5", "cosmo"),
    ("IGWN-GWTC3p0-v2-GW191103_012549_PEDataRelease_mixed_cosmo.h5", "cosmo"),
    ("IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_nocosmo.h5",
     "nocosmo"),
    ("IGWN-GWTC4p1-18965dda8_5-GW230529_181500-combined_PEDataRelease.hdf5",
     "native"),
    ("IGWN-GWTC5p0-29ebe06b7_25-GW240413_022019-combined_PEDataRelease.hdf5",
     "native"),
])
def test_detect_release_flavour(name, expect):
    assert detect_release_flavour(name) == expect
    assert detect_release_flavour(f"/some/dir/{name}") == expect


# --------------------------------------------------------------------------
# 3. resolve_dL_prior: effective vs sampling class
# --------------------------------------------------------------------------
def _priors(repr_string, analysis="C01:IMRPhenomXPHM", samples=None):
    out = {"analytic": {analysis: {"luminosity_distance": repr_string}}}
    if samples is not None:
        out["samples"] = {analysis: {"luminosity_distance": samples}}
    return out


def _resolve(repr_string, *, flavour, analysis="C01:Mixed",
             sibling="C01:IMRPhenomXPHM", dL=None, cfg=None):
    dL = np.linspace(300.0, 3000.0, 200) if dL is None else dL
    return resolve_dL_prior("GWTC-3", analysis, [analysis, sibling],
                            _priors(repr_string, sibling), dL,
                            cfg or IngestConfig(), flavour=flavour)


def test_cosmo_flavour_keeps_uniform_source_frame_and_records_the_sampling_class():
    """The reweighted release: declared PowerLaw is the SAMPLING prior, the
    effective prior stays UniformSourceFrame, and the store says why."""
    r = _resolve(REPR_POWERLAW, flavour="cosmo")
    assert r.kind == "UniformSourceFrame"
    assert r.alpha is None
    assert r.sampling_kind == "PowerLaw"
    assert r.sampling_alpha == 2.0
    assert r.basis == "release_reweighted"
    assert r.flavour == "cosmo"
    # bounds still come from the declared prior
    assert (r.dmin, r.dmax) == (10.0, 10000.0)


def test_nocosmo_flavour_dispatches_on_the_declared_class():
    """The un-reweighted sibling release: the declared class IS the prior."""
    r = _resolve(REPR_POWERLAW, flavour="nocosmo")
    assert r.kind == "PowerLaw"
    assert r.alpha == 2.0
    assert r.basis == "analytic_declared"


def test_native_flavour_dispatches_on_the_declared_class():
    r = _resolve(REPR_USF_LAL, flavour="native", analysis="C00:Mixed",
                 sibling="C00:IMRPhenomXPHM")
    assert r.kind == "UniformSourceFrame"
    assert r.sampling_kind == "UniformSourceFrame"
    assert r.basis == "analytic_declared"
    assert r.cosmology_name == "Planck15_LAL"
    assert (r.H0, r.Om0) == pytest.approx((67.90, 0.3065))


def test_o4_row_gets_lal_cosmology_not_astropy():
    """The 190-row fix, at the resolution level."""
    r = _resolve(REPR_USF_LAL, flavour="native", analysis="C00:Mixed",
                 sibling="C00:IMRPhenomXPHM")
    assert (r.H0, r.Om0) != pytest.approx((PLANCK15.H0.value, PLANCK15.Om0))
    assert (r.H0, r.Om0) == pytest.approx((LAL_PLANCK15.H0.value,
                                           LAL_PLANCK15.Om0))


def test_unknown_declared_class_raises_rather_than_substituting():
    bad = REPR_POWERLAW.replace("PowerLaw(", "LogNormalMagic(")
    with pytest.raises(DistancePriorKindError, match="LogNormalMagic"):
        _resolve(bad, flavour="nocosmo")


def test_power_law_without_alpha_raises():
    bad = REPR_POWERLAW.replace("alpha=2, ", "")
    with pytest.raises(ValueError, match="alpha"):
        _resolve(bad, flavour="nocosmo")


def test_unrecognised_cosmology_warns_and_falls_back():
    bad = REPR_USF_LAL.replace("Planck15_LAL", "WMAP9_LAL")
    with pytest.warns(UserWarning, match="unrecognised cosmology"):
        r = _resolve(bad, flavour="native", analysis="C00:Mixed",
                     sibling="C00:IMRPhenomXPHM")
    assert r.cosmology_name == "WMAP9_LAL"
    assert "default_cosmo" in r.source


def test_no_analytic_anywhere_still_resolves_and_says_so():
    dL = np.linspace(300.0, 3000.0, 200)
    r = resolve_dL_prior("GWTC-2.1", "C01:Mixed", ["C01:Mixed"], {}, dL,
                         IngestConfig(), flavour="cosmo")
    assert r.kind == "UniformSourceFrame"
    assert r.sampling_kind == ""
    assert r.source == "default(no_analytic)"
    assert r.basis == "release_reweighted"
    # bounds fall back to the sample range
    assert (r.dmin, r.dmax) == pytest.approx((300.0, 3000.0))


def test_bad_flavour_rejected():
    with pytest.raises(ValueError, match="flavour"):
        _resolve(REPR_POWERLAW, flavour="reweighted")


# --------------------------------------------------------------------------
# 4. The KS check now compares against the right distribution
# --------------------------------------------------------------------------
def _draw_power_law(alpha, lo, hi, n, seed=0):
    """Exact inverse-CDF draws from p(x) propto x**alpha on [lo, hi]."""
    u = np.random.default_rng(seed).uniform(size=n)
    a1 = alpha + 1.0
    return (lo ** a1 + u * (hi ** a1 - lo ** a1)) ** (1.0 / a1)


def test_ks_passes_for_the_sampling_class_and_fails_for_the_effective_one():
    """On a reweighted release the prior samples follow the SAMPLING class.

    The pre-GW-02 code compared UniformSourceFrame against these dL^2 draws,
    measured KS ~ 0.27 on every GWTC-2.1/3 event, and blamed the cosmology.
    """
    lo, hi = 10.0, 10000.0
    dlp = _draw_power_law(2.0, lo, hi, 20000, seed=3)
    r = _resolve(REPR_POWERLAW, flavour="cosmo")
    v = validate_prior_against_samples(
        {"analytic": {"C01:IMRPhenomXPHM":
                      {"luminosity_distance": REPR_POWERLAW}},
         "samples": {"C01:IMRPhenomXPHM": {"luminosity_distance": dlp}}},
        ["C01:Mixed", "C01:IMRPhenomXPHM"], r, impl="astropy")

    assert v is not None
    assert v["sampling_kind"] == "PowerLaw"
    assert v["effective_kind"] == "UniformSourceFrame"
    # the declared sampling class describes the prior draws ...
    assert v["ks"] < 0.02, v
    # ... and the effective (reweighted) prior deliberately does not.
    assert v["ks_effective"] > 0.2, v


def test_ks_matches_for_a_native_release():
    """No reweighting: the two classes coincide, so the two KS values do too."""
    lo, hi = 10.0, 4000.0
    cosmo = LAL_PLANCK15
    grid = np.linspace(lo, hi, 200000)
    pdf = uniform_source_frame_prob(grid, cosmo, lo, hi, impl="astropy")
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    u = np.random.default_rng(5).uniform(size=20000)
    dlp = np.interp(u, cdf, grid)

    r = _resolve(REPR_USF_LAL, flavour="native", analysis="C00:Mixed",
                 sibling="C00:IMRPhenomXPHM")
    v = validate_prior_against_samples(
        _priors(REPR_USF_LAL, "C00:IMRPhenomXPHM", samples=dlp),
        ["C00:Mixed", "C00:IMRPhenomXPHM"], r, impl="astropy")
    assert v["ks"] == pytest.approx(v["ks_effective"])
    assert v["ks"] < 0.02, v


def test_ks_can_be_made_fatal():
    """prior_ks_fatal turns a rejected parse into a hard failure."""
    lo, hi = 10.0, 10000.0
    # draws that match NEITHER candidate (uniform in dL)
    dlp = np.random.default_rng(7).uniform(lo, hi, 20000)
    cfg = IngestConfig(prior_ks_fatal=True)
    r = _resolve(REPR_POWERLAW, flavour="nocosmo", cfg=cfg)
    v = validate_prior_against_samples(
        _priors(REPR_POWERLAW, samples=dlp),
        ["C01:Mixed", "C01:IMRPhenomXPHM"], r, impl="astropy")
    assert v["ks"] > cfg.prior_ks_max
    # the raise itself lives in build_store; assert the exception type exists
    # and is a ValueError so callers can catch it uniformly
    assert issubclass(PriorMismatchError, ValueError)


def test_no_prior_samples_returns_none():
    r = _resolve(REPR_POWERLAW, flavour="cosmo")
    assert validate_prior_against_samples(
        _priors(REPR_POWERLAW), ["C01:Mixed", "C01:IMRPhenomXPHM"], r) is None


# --------------------------------------------------------------------------
# 5. The densities themselves
# --------------------------------------------------------------------------
def test_power_law_normalises_and_is_evaluated_outside_the_bounds():
    lo, hi = 10.0, 10000.0
    grid = np.linspace(lo, hi, 200001)
    p = power_law_dL_prob(grid, 2.0, lo, hi)
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    assert trapz(p, grid) == pytest.approx(1.0, rel=1e-6)
    # GW-01's rule holds here too: no truncation at the bounds.
    out = power_law_dL_prob(np.array([1.0, 20000.0]), 2.0, lo, hi)
    assert np.all(out > 0)
    # exact shape
    assert out[1] / out[0] == pytest.approx((20000.0 / 1.0) ** 2)


def test_power_law_alpha_minus_one_uses_the_log_normalisation():
    lo, hi = 10.0, 1000.0
    grid = np.linspace(lo, hi, 200001)
    p = power_law_dL_prob(grid, -1.0, lo, hi)
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    assert trapz(p, grid) == pytest.approx(1.0, rel=1e-5)


def test_power_law_negative_dL_is_nan_not_a_sign_flip():
    p = power_law_dL_prob(np.array([-100.0, 100.0]), 2.0, 10.0, 1000.0)
    assert np.isnan(p[0]) and p[1] > 0


def test_power_law_alpha2_is_not_uniform_source_frame():
    """The two densities differ by an O(1), dL-dependent factor -- which is why
    dispatching on the wrong one cannot be absorbed by the per-event
    normalisation."""
    lo, hi = 10.0, 10000.0
    dL = np.array([100.0, 500.0, 2000.0, 8000.0])
    pl = power_law_dL_prob(dL, 2.0, lo, hi)
    usf = uniform_source_frame_prob(dL, LAL_PLANCK15, lo, hi, impl="astropy")
    ratio = pl / usf
    # not a constant: the ratio spans more than a factor of 5 across the range
    assert np.max(ratio) / np.min(ratio) > 5.0


def test_uniform_comoving_volume_differs_from_source_frame_by_time_dilation():
    lo, hi = 10.0, 8000.0
    dL = np.array([500.0, 2000.0, 6000.0])
    usf = dL_prior_prob(dL, kind="UniformSourceFrame", cosmology=LAL_PLANCK15,
                        dmin=lo, dmax=hi, impl="astropy")
    ucv = dL_prior_prob(dL, kind="UniformComovingVolume",
                        cosmology=LAL_PLANCK15, dmin=lo, dmax=hi,
                        impl="astropy")
    assert np.all(usf > 0) and np.all(ucv > 0)
    ratio = ucv / usf
    assert np.max(ratio) / np.min(ratio) > 1.1   # the (1+z) factor, not a const


@pytest.mark.parametrize("kind", DL_PRIOR_KINDS)
def test_dispatcher_covers_every_declared_kind(kind):
    dL = np.linspace(50.0, 3000.0, 64)
    p, info = dL_prior_prob(dL, kind=kind, cosmology=LAL_PLANCK15, dmin=10.0,
                            dmax=4000.0, alpha=2.0, impl="astropy",
                            return_info=True)
    assert info["kind"] == kind
    assert np.all(np.isfinite(p)) and np.all(p > 0)


def test_dispatcher_rejects_unknown_kind_and_missing_inputs():
    dL = np.linspace(50.0, 3000.0, 8)
    with pytest.raises(DistancePriorKindError, match="Nonsense"):
        dL_prior_prob(dL, kind="Nonsense", dmin=10.0, dmax=4000.0)
    with pytest.raises(ValueError, match="requires a cosmology"):
        dL_prior_prob(dL, kind="UniformSourceFrame", dmin=10.0, dmax=4000.0)
    with pytest.raises(ValueError, match="requires alpha"):
        dL_prior_prob(dL, kind="PowerLaw", dmin=10.0, dmax=4000.0)


def test_uniform_kind_is_flat():
    dL = np.array([100.0, 1000.0, 3000.0])
    p = dL_prior_prob(dL, kind="Uniform", dmin=10.0, dmax=4000.0)
    assert np.allclose(p, p[0])
    assert p[0] == pytest.approx(1.0 / (4000.0 - 10.0))


def test_closed_form_kinds_report_analytic_impl():
    _, info = dL_prior_prob(np.array([100.0]), kind="PowerLaw", alpha=2.0,
                            dmin=10.0, dmax=4000.0, return_info=True)
    assert info["impl"] == "analytic"
    assert info["widened"] is False
