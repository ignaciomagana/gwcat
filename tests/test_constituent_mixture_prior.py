"""GW-40f: the prior of a combined C00:Mixed set is the equal-weight mixture of
its constituents' own normalised priors -- including the q = 1/6 step.

GWTC-4.1 C00:Mixed concatenates equal counts of NRSur7dq4 (q >= 1/6),
IMRPhenomXPHM-SpinTaylor and SEOBNRv5PHM (q >= 0.05).  Borrowing one sibling's
smooth analytic prior misses the step and the per-analysis normalisation.
"""
import warnings

import h5py
import numpy as np
import pytest

from gwcat.catalog import GWCatalog
from gwcat.constituent_mixture import (ConstituentMixtureError,
                                       match_constituent_rows,
                                       mixture_prior_densities,
                                       truncated_dL_density, uic_density_m1q,
                                       uic_normalisation)
from gwcat.cosmology import LAL_PLANCK15
from gwcat.ingest import IngestConfig, build_store

NR, XP, MX = "C00:NRSur7dq4", "C00:IMRPhenomXPHM-SpinTaylor", "C00:Mixed"
MC = (5.0, 60.0)
QNR, QXP = 0.16666667, 0.05
USF = ("bilby.gw.prior.UniformSourceFrame(minimum=100.0, maximum=6000.0, "
       "cosmology='Planck15_LAL', name='luminosity_distance')")


def _analytic(qmin, dl=USF, amax=0.99):
    return {
        "chirp_mass": f"bilby.gw.prior.UniformInComponentsChirpMass("
                      f"minimum={MC[0]}, maximum={MC[1]}, name='chirp_mass')",
        "mass_ratio": f"bilby.gw.prior.UniformInComponentsMassRatio("
                      f"minimum={qmin}, maximum=1.0, name='mass_ratio')",
        "mass_1": "Constraint(minimum=1, maximum=1000, name='mass_1')",
        "mass_2": "Constraint(minimum=1, maximum=1000, name='mass_2')",
        "a_1": f"Uniform(minimum=0.0, maximum={amax}, name='a_1')",
        "a_2": f"Uniform(minimum=0.0, maximum={amax}, name='a_2')",
        "tilt_1": "Sine(name='tilt_1', minimum=0.0, maximum=3.141592653589793)",
        "tilt_2": "Sine(name='tilt_2', minimum=0.0, maximum=3.141592653589793)",
        "luminosity_distance": dl,
    }


def _bounds(qmin):
    return dict(mc_min=MC[0], mc_max=MC[1], q_min=qmin, q_max=1.0,
                m1_min=1.0, m1_max=1000.0, m2_min=1.0, m2_max=1000.0)


# --------------------------------------------------------------------------
# The normalisation and the step, analytically
# --------------------------------------------------------------------------
def test_uic_normalisation_matches_brute_force_area():
    b = _bounds(QXP)
    Z = uic_normalisation(**b)
    # brute force: area of the (m1, m2) support on a fine grid
    m1 = np.linspace(1, 400, 2001)
    m2 = np.linspace(1, 400, 2001)
    M1, M2 = np.meshgrid(m1, m2, indexing="ij")
    q = M2 / M1
    mc = (M1 * M2) ** 0.6 / (M1 + M2) ** 0.2
    inside = (q >= QXP) & (q <= 1) & (mc >= MC[0]) & (mc <= MC[1])
    area = inside.sum() * (m1[1] - m1[0]) * (m2[1] - m2[0])
    assert Z == pytest.approx(area, rel=5e-3)


def test_the_q_one_sixth_step_is_reproduced():
    """At fixed (m1, dL): just below q = 1/6 only XPHM-ST contributes; just
    above, both do.  The step ratio is (1/Z_x) / (1/Z_x + 1/Z_n)."""
    bx, bn = _bounds(QXP), _bounds(QNR)
    Zx, Zn = uic_normalisation(**bx), uic_normalisation(**bn)
    assert Zx > Zn
    pd = truncated_dL_density("UniformSourceFrame", LAL_PLANCK15, 100.0,
                              6000.0)
    comps = [dict(bn, Z=Zn, p_dL=pd), dict(bx, Z=Zx, p_dL=pd)]
    m1 = np.array([60.0, 60.0])
    q = np.array([QNR - 1e-4, QNR + 1e-4])
    dL = np.array([1500.0, 1500.0])
    joint, marg = mixture_prior_densities(m1, q, dL, comps)
    step = joint[0] / joint[1]
    assert step == pytest.approx((1 / Zx) / (1 / Zx + 1 / Zn), rel=1e-6)
    assert step < 1.0 - 1e-3          # a real step, not a smooth prior
    # the density is m1/Z per component, times the shared distance prior
    np.testing.assert_allclose(
        joint[1], 0.5 * (60 / Zx + 60 / Zn) * pd(np.array([1500.0]))[0],
        rtol=1e-12)
    np.testing.assert_allclose(marg, pd(dL), rtol=1e-12)
    # and a single borrowed prior (XPHM-ST) has NO step there
    single = uic_density_m1q(m1, q, Z=Zx, **bx)
    assert single[0] == pytest.approx(single[1], rel=1e-12)


def test_joint_mixture_is_not_the_product_of_marginals():
    """Different distance bounds: sum_k pi_k(m) pi_k(dL) != [sum pi_k(m)]
    [sum pi_k(dL)] / K."""
    bx, bn = _bounds(QXP), _bounds(QNR)
    Zx, Zn = uic_normalisation(**bx), uic_normalisation(**bn)
    p1 = truncated_dL_density("UniformSourceFrame", LAL_PLANCK15, 100.0, 3000.)
    p2 = truncated_dL_density("UniformSourceFrame", LAL_PLANCK15, 100.0, 6000.)
    comps = [dict(bn, Z=Zn, p_dL=p1), dict(bx, Z=Zx, p_dL=p2)]
    m1, q, dL = np.array([60.0]), np.array([0.5]), np.array([1000.0])
    joint, marg = mixture_prior_densities(m1, q, dL, comps)
    pm = 0.5 * (uic_density_m1q(m1, q, Z=Zn, **bn)
                + uic_density_m1q(m1, q, Z=Zx, **bx))
    assert not np.isclose(joint[0], pm[0] * marg[0], rtol=1e-6)
    # beyond one constituent's distance bound it contributes nothing
    j2, _ = mixture_prior_densities(m1, q, np.array([4000.0]), comps)
    np.testing.assert_allclose(
        j2, 0.5 * uic_density_m1q(m1, q, Z=Zx, **bx) * p2(np.array([4000.])),
        rtol=1e-12)


# --------------------------------------------------------------------------
# Row-by-row verification of the mixing fractions
# --------------------------------------------------------------------------
def _constituent_samples(rng, qmin, n):
    # inside every constituent's support: Mc <= 60 needs m1 <= ~68 at q = 1
    m1 = rng.uniform(30, 60, n)
    q = rng.uniform(max(qmin, 0.06), 1.0, n)
    return {"mass_1": m1, "mass_2": q * m1,
            "luminosity_distance": rng.uniform(300, 3000, n),
            "ra": rng.uniform(0, 2 * np.pi, n),
            "dec": rng.uniform(-np.pi / 2, np.pi / 2, n),
            "chi_eff": rng.uniform(-0.3, 0.3, n),
            "a_1": rng.uniform(0, 0.99, n), "a_2": rng.uniform(0, 0.99, n),
            "tilt_1": rng.uniform(0, np.pi, n),
            "tilt_2": rng.uniform(0, np.pi, n)}


def _concat(*parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _take(d, idx):
    return {k: np.asarray(v)[idx] for k, v in d.items()}


def test_match_rows_verifies_equal_fractions():
    rng = np.random.default_rng(0)
    nr = _constituent_samples(rng, QNR, 300)
    xp = _constituent_samples(rng, QXP, 500)
    mixed = _concat(_take(nr, np.arange(200)), _take(xp, np.arange(200)))
    labels, counts = match_constituent_rows(mixed, {NR: nr, XP: xp})
    assert labels == [NR, XP] and counts == {NR: 200, XP: 200}

    unequal = _concat(_take(nr, np.arange(150)), _take(xp, np.arange(250)))
    with pytest.raises(ConstituentMixtureError, match="not equal"):
        match_constituent_rows(unequal, {NR: nr, XP: xp})

    stray = {k: v.copy() for k, v in mixed.items()}
    stray["mass_1"][0] += 1e-9
    with pytest.raises(ConstituentMixtureError, match="equal no constituent"):
        match_constituent_rows(stray, {NR: nr, XP: xp})


# --------------------------------------------------------------------------
# Ingest + export
# --------------------------------------------------------------------------
_NAME = "IGWN-GWTC4p1-18965dda8_5-GW950301_000301-combined_PEDataRelease.hdf5"


def _ingest(tmp_path, monkeypatch, *, mixed_parts=(200, 200), flag=True,
            nr_amax=0.99, name="store.h5"):
    import gwcat.ingest as ing
    rng = np.random.default_rng(7)
    nr = _constituent_samples(rng, QNR, 300)
    xp = _constituent_samples(rng, QXP, 500)
    # XPHM-ST samples below the NRSur floor exist (this is the whole point)
    assert np.any(xp["mass_2"] / xp["mass_1"] < QNR)
    mixed = _concat(_take(nr, np.arange(mixed_parts[0])),
                    _take(xp, np.arange(mixed_parts[1])))
    analyses = {XP: xp, MX: mixed, NR: nr}
    priors = {"analytic": {NR: _analytic(QNR, amax=nr_amax),
                           XP: _analytic(QXP)}}

    class _D:
        config = {}
    monkeypatch.setattr(ing, "_read_event_pesummary",
                        lambda path: (_D(), analyses, list(analyses), priors))
    raw = tmp_path / _NAME
    raw.write_bytes(b"")
    out = tmp_path / name
    build_store([str(raw)], str(out), event_table={},
                cfg=IngestConfig(validate_prior=False,
                                 constituent_mixture_prior=flag,
                                 dL_prior_impl="astropy"))
    return str(out), analyses


def _meta0(store, col):
    with h5py.File(store, "r") as f:
        v = f[f"meta/{col}"][0]
    return v.decode() if isinstance(v, (bytes, bytearray)) else v


def test_ingest_builds_and_records_the_mixture(tmp_path, monkeypatch):
    store, an = _ingest(tmp_path, monkeypatch)
    assert _meta0(store, "analysis_used") == MX
    assert _meta0(store, "mass_prior_kind") == "constituent_mixture"
    for p in ("mass", "spin", "dL"):
        assert _meta0(store, f"prior_source_kind_{p}") == "constituent_mixture"
        assert _meta0(store, f"prior_source_label_{p}") == f"{XP}+{NR}"  # file order
    assert _meta0(store, "constituent_mixture_labels") == f"{XP}+{NR}"  # file order
    assert _meta0(store, "constituent_mixture_counts") == "200,200"
    assert float(_meta0(store, "mass_prior_q_min")) == pytest.approx(QXP)
    assert (float(_meta0(store, "dL_prior_H0")),
            float(_meta0(store, "dL_prior_Om0"))) == (67.90, 0.3065)

    with h5py.File(store, "r") as f:
        pmd = f["samples/p_mass_dL_pe"][:]
    mixed = an[MX]
    m1 = mixed["mass_1"]
    q = mixed["mass_2"] / m1
    bx, bn = _bounds(QXP), _bounds(QNR)
    pd = truncated_dL_density("UniformSourceFrame", LAL_PLANCK15, 100.0,
                              6000.0, impl="astropy")
    expected = 0.5 * (uic_density_m1q(m1, q, Z=uic_normalisation(**bn), **bn)
                      + uic_density_m1q(m1, q, Z=uic_normalisation(**bx), **bx)
                      ) * pd(mixed["luminosity_distance"])
    np.testing.assert_allclose(pmd, expected, rtol=1e-10)
    # the step is visible in the stored prior: below 1/6 only XPHM-ST's term
    below = q < QNR
    assert below.any()
    ratio = pmd[below] / (m1[below] * pd(mixed["luminosity_distance"][below]))
    np.testing.assert_allclose(ratio, 0.5 / uic_normalisation(**bx),
                               rtol=1e-10)


def test_flag_off_keeps_the_sibling_borrowed_prior(tmp_path, monkeypatch):
    store, _ = _ingest(tmp_path, monkeypatch, flag=False)
    assert _meta0(store, "mass_prior_kind") == "uniform_detector_frame"
    assert _meta0(store, "prior_source_kind_mass") == "sibling_inherited"
    assert _meta0(store, "constituent_mixture_labels") == ""
    with h5py.File(store, "r") as f:
        assert "p_mass_dL_pe" not in f["samples"]


def test_unequal_mixing_fractions_refuse_the_build(tmp_path, monkeypatch):
    with pytest.raises(ConstituentMixtureError, match="not equal"):
        _ingest(tmp_path, monkeypatch, mixed_parts=(150, 250))


def test_samples_outside_every_support_are_refused(tmp_path, monkeypatch):
    import gwcat.ingest as ing
    orig = ing._constituent_mixture_prior

    def _shrunk(catalog, analysis, analyses, samples_dict, priors, cfg, fl):
        p2 = {"analytic": {k: dict(v) for k, v in priors["analytic"].items()}}
        for k in p2["analytic"]:
            p2["analytic"][k]["chirp_mass"] = (
                "bilby.gw.prior.UniformInComponentsChirpMass(minimum=5.0, "
                "maximum=20.0, name='chirp_mass')")
        return orig(catalog, analysis, analyses, samples_dict, p2, cfg, fl)
    monkeypatch.setattr(ing, "_constituent_mixture_prior", _shrunk)
    with pytest.raises(ConstituentMixtureError, match="outside EVERY"):
        _ingest(tmp_path, monkeypatch)


def test_disagreeing_constituent_spin_priors_are_refused(tmp_path,
                                                          monkeypatch):
    with pytest.raises(ConstituentMixtureError, match="different spin"):
        _ingest(tmp_path, monkeypatch, nr_amax=0.8)


def test_export_uses_the_mixture_density(tmp_path, monkeypatch):
    from gwcat.spin import chi_eff_prior_logprob
    store, _ = _ingest(tmp_path, monkeypatch)
    out = tmp_path / "pe.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        GWCatalog(store).export(str(out), format="gwcat2",
                                spin_basis="chieff", nsamp=256, seed=0)
    with h5py.File(out, "r") as f:
        cols = {k: f[k][:] for k in ("m1det", "q", "dL", "chieff", "p_pe",
                                     "m1src", "m2src")}
        attrs = dict(f.attrs)
    mb = attrs["mass_prior_basis"]
    assert (mb.decode() if isinstance(mb, bytes) else mb) == \
        "constituent_mixture"
    assert bool(attrs["mass_prior_verified"]) is True
    bx, bn = _bounds(QXP), _bounds(QNR)
    pd = truncated_dL_density("UniformSourceFrame", LAL_PLANCK15, 100.0,
                              6000.0, impl="astropy")
    m1, q = cols["m1det"], cols["q"]
    mix = 0.5 * (uic_density_m1q(m1, q, Z=uic_normalisation(**bn), **bn)
                 + uic_density_m1q(m1, q, Z=uic_normalisation(**bx), **bx)
                 ) * pd(cols["dL"])
    chi = np.exp(chi_eff_prior_logprob(cols["chieff"], cols["m1src"],
                                       cols["m2src"], amax=0.99, amax_2=0.99))
    np.testing.assert_allclose(cols["p_pe"], mix * chi, rtol=1e-10)
    # and it is NOT the borrowed m1det * p_dL shape
    assert not np.allclose(cols["p_pe"] / chi / pd(cols["dL"]) / m1,
                           np.mean(cols["p_pe"] / chi / pd(cols["dL"]) / m1),
                           rtol=1e-6)


def test_v1_exporter_refuses_a_mixture_row(tmp_path, monkeypatch):
    """The frozen v1 exporter applies m1det * p_dL; it must not silently do so
    for a mixture row, whose mass class it does not support."""
    store, _ = _ingest(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="constituent_mixture"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            GWCatalog(store).to_darksirens(str(tmp_path / "v1.h5"), nsamp=16)


# --------------------------------------------------------------------------
# Review 2026-10-20: every Constraint enters the support (NRSur7dq4 total mass)
# --------------------------------------------------------------------------
#: GW230708_053705's own C00:NRSur7dq4 group: 20.18 <= Mc <= 46.55,
#: q >= 1/6, 1 <= m_i <= 1000 and Constraint(60 <= total_mass <= 400).
_NR_MC = (20.181598915151937, 46.54562366822528)
_NR_TOTAL_MASS = "Constraint(minimum=60, maximum=400, name='total_mass')"


def _area_reference(mc_min, mc_max, q_min, q_max, m_min, m_max, mt_min,
                    mt_max):
    """The support's (m1, m2) area by an independent route: for each m1, the
    m2 interval cut by every bound (Mc is monotone in m2), integrated over m1
    with adaptive quadrature."""
    from scipy.integrate import quad
    from scipy.optimize import brentq

    def mc(m1, m2):
        return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2

    def m2_at(m1, target):
        return brentq(lambda m2: mc(m1, m2) - target, 1e-12, 1e6,
                      xtol=1e-14, rtol=1e-15)

    def length(m1):
        lo = max(q_min * m1, m_min, mt_min - m1, m2_at(m1, mc_min))
        hi = min(q_max * m1, m_max, mt_max - m1, m2_at(m1, mc_max))
        return max(hi - lo, 0.0)

    val, _err = quad(length, m_min, min(m_max, mt_max), limit=2000,
                     epsabs=0.0, epsrel=1e-11)
    return val


def test_total_mass_constraint_enters_the_normalisation():
    """The NRSur7dq4 Mtot cut binds: without it Z is ~1.076x too large for
    GW230708 (bilby's own normalisation), with it Z is the true area."""
    b = dict(mc_min=_NR_MC[0], mc_max=_NR_MC[1], q_min=QNR, q_max=1.0,
             m1_min=1.0, m1_max=1000.0, m2_min=1.0, m2_max=1000.0)
    Z_cut = uic_normalisation(**b, mt_min=60.0, mt_max=400.0)
    ref = _area_reference(_NR_MC[0], _NR_MC[1], QNR, 1.0, 1.0, 1000.0, 60.0,
                          400.0)
    assert Z_cut == pytest.approx(ref, rel=1e-8)
    Z_uncut = uic_normalisation(**b)
    assert Z_uncut / Z_cut == pytest.approx(1.076, abs=1e-3)
    # and the density is exactly zero below the cut, m1/Z above it
    m1 = np.array([40.0, 40.0])
    q = np.array([0.45, 0.55])            # Mtot 58 and 62
    d = uic_density_m1q(m1, q, Z=Z_cut, **b, mt_min=60.0, mt_max=400.0)
    assert d[0] == 0.0 and d[1] == pytest.approx(40.0 / Z_cut, rel=1e-15)


def _ingest_with(tmp_path, monkeypatch, nr, xp, nr_analytic, xp_analytic,
                 cfg_kw=None):
    import gwcat.ingest as ing
    mixed = _concat(nr, xp)
    analyses = {XP: xp, MX: mixed, NR: nr}
    priors = {"analytic": {NR: nr_analytic, XP: xp_analytic}}

    class _D:
        config = {}
    monkeypatch.setattr(ing, "_read_event_pesummary",
                        lambda path: (_D(), analyses, list(analyses), priors))
    raw = tmp_path / _NAME
    raw.write_bytes(b"")
    out = tmp_path / "store_cut.h5"
    build_store([str(raw)], str(out), event_table={},
                cfg=IngestConfig(validate_prior=False,
                                 constituent_mixture_prior=True,
                                 dL_prior_impl="astropy", **(cfg_kw or {})))
    return str(out), analyses


def _nr_xp_samples_around_the_cut(n=200):
    rng = np.random.default_rng(11)
    # NRSur rows obey their own prior: q >= 1/6 and Mtot >= 60.
    nr = _constituent_samples(rng, QNR, n)
    nr["mass_1"] = rng.uniform(52.0, 60.0, n)
    nr["mass_2"] = rng.uniform(max(QNR, 0.2), 1.0, n) * nr["mass_1"]
    xp = _constituent_samples(rng, QXP, n)
    return nr, xp


def test_mixture_ingest_applies_the_nrsur_total_mass_cut(tmp_path,
                                                          monkeypatch):
    """Failure scenario of the review: Mixed rows from the other constituent
    with Mtot < 60 must get NO NRSur7dq4 density, and Z_NRSur must be the
    cut area."""
    nr, xp = _nr_xp_samples_around_the_cut()
    nr_an = dict(_analytic(QNR), total_mass=_NR_TOTAL_MASS)
    store, an = _ingest_with(tmp_path, monkeypatch, nr, xp, nr_an,
                             _analytic(QXP))
    with h5py.File(store, "r") as f:
        pmd = f["samples/p_mass_dL_pe"][:]
    mixed = an[MX]
    m1 = mixed["mass_1"]
    q = mixed["mass_2"] / m1
    mtot = m1 * (1.0 + q)
    bx, bn = _bounds(QXP), _bounds(QNR)
    Zx = uic_normalisation(**bx)
    pd = truncated_dL_density("UniformSourceFrame", LAL_PLANCK15, 100.0,
                              6000.0, impl="astropy")
    # Inside NRSur's (Mc, q) bounds but below its Mtot cut: XPHM-ST alone.
    low = (mtot < 60.0) & (q >= QNR)
    assert low.sum() > 5
    np.testing.assert_allclose(
        pmd[low], 0.5 * m1[low] / Zx * pd(mixed["luminosity_distance"][low]),
        rtol=1e-10)
    Zn = uic_normalisation(**bn, mt_min=60.0, mt_max=400.0)
    expected = 0.5 * (
        uic_density_m1q(m1, q, Z=Zn, **bn, mt_min=60.0, mt_max=400.0)
        + uic_density_m1q(m1, q, Z=Zx, **bx)) * pd(mixed["luminosity_distance"])
    np.testing.assert_allclose(pmd, expected, rtol=1e-10)


def test_a_constraint_the_support_does_not_implement_is_refused(tmp_path,
                                                                 monkeypatch):
    nr, xp = _nr_xp_samples_around_the_cut()
    nr_an = dict(_analytic(QNR), comoving_distance=(
        "Constraint(minimum=10, maximum=5000, name='comoving_distance')"))
    with pytest.raises(ConstituentMixtureError, match="does not implement"):
        _ingest_with(tmp_path, monkeypatch, nr, xp, nr_an, _analytic(QXP))
