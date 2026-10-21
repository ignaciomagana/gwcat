"""Regression tests for the gwcat review of 2026-10-20, ``gwcat/ingest.py``.

Each test reproduces one finding's failure scenario.  The fixtures use the real
GWTC-4.1 / GWTC-3 label sets and the real analytic-prior repr shapes (copied
from the local release files), driven through the repo's fake-PESummary
pattern: no pesummary, bilby or network.

Names below are the review's finding IDs, kept only to tie a test to its
record in the review's FINAL.json.
"""
import json
import os
import warnings

import h5py
import numpy as np
import pytest

import gwcat.ingest as ing

# --------------------------------------------------------------------------
# Real repr shapes (IGWN-GWTC4p1-18965dda8_5-GW230702_185453)
# --------------------------------------------------------------------------
UIC_MC = ("bilby.gw.prior.UniformInComponentsChirpMass(minimum=16.4, "
          "maximum=52.08, name='chirp_mass', latex_label='$\\mathcal{M}$', "
          "unit=None, boundary=None)")


def _uic_q(qmin):
    return (f"bilby.gw.prior.UniformInComponentsMassRatio(minimum={qmin}, "
            f"maximum=1.0, name='mass_ratio', latex_label='$q$', unit=None, "
            f"boundary=None, equal_mass=False)")


DL_USF = ("bilby.gw.prior.UniformSourceFrame(minimum=100.0, maximum=10000.0, "
          "cosmology='Planck15_LAL', name='luminosity_distance', "
          "latex_label='$d_L$', unit='Mpc', boundary=None)")
MTOT_CON = ("Constraint(minimum=60, maximum=400, name='total_mass', "
            "latex_label='$M$', unit=None)")


def _a(i, amax):
    return (f"Uniform(minimum=0, maximum={amax}, name='a_{i}', "
            f"latex_label='$a_{i}$', unit=None, boundary=None)")


def _m_con(i):
    return (f"Constraint(minimum=1, maximum=1000, name='mass_{i}', "
            f"latex_label='$m_{i}$', unit=None)")


def _aligned(amax):
    return ("bilby.gw.prior.AlignedSpin(a_prior=Uniform(minimum=0, "
            f"maximum={amax}, name=None, latex_label=None, unit=None, "
            "boundary=None), z_prior=Uniform(minimum=-1, maximum=1, name=None, "
            "latex_label=None, unit=None, boundary=None), name=None, "
            "latex_label=None, unit=None, boundary=None, minimum=-1.0, "
            "maximum=1.0, num_interp=None)")


def _analytic(qmin=0.05, amax=(0.99, 0.99), total_mass=False):
    d = {"chirp_mass": UIC_MC, "mass_ratio": _uic_q(qmin),
         "mass_1": _m_con(1), "mass_2": _m_con(2),
         "luminosity_distance": DL_USF,
         "a_1": _a(1, amax[0]), "a_2": _a(2, amax[1])}
    if total_mass:
        d["total_mass"] = MTOT_CON
    return d


class _Data:
    """A pesummary read() stand-in with the real layout: ``extra_kwargs`` is a
    LIST aligned with ``samples_dict``'s keys."""

    def __init__(self, samples_dict, extra=None, config=None):
        self.samples_dict = samples_dict
        self.config = config or {}
        self.extra_kwargs = [(extra or {}).get(lab, {})
                             for lab in samples_dict]


def _samples(rng, n, amax=0.9, gps=1372000000.0):
    # inside every constituent's chirp-mass and mass-ratio support
    m1 = rng.uniform(30.0, 50.0, n)
    q = rng.uniform(0.5, 1.0, n)
    return {"mass_1": m1, "mass_2": q * m1,
            "luminosity_distance": rng.uniform(500.0, 2000.0, n),
            "ra": rng.uniform(0.0, 2 * np.pi, n),
            "dec": np.arcsin(rng.uniform(-1.0, 1.0, n)),
            "a_1": rng.uniform(0.0, amax, n), "a_2": rng.uniform(0.0, amax, n),
            "geocent_time": gps + rng.uniform(-0.01, 0.01, n)}


def _take(s, idx):
    return {k: np.asarray(v)[idx] for k, v in s.items()}


def _concat(*parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


XP, MX, NR, SE = ("C00:IMRPhenomXPHM-SpinTaylor", "C00:Mixed",
                  "C00:NRSur7dq4", "C00:SEOBNRv5PHM")
O4_NAME = "IGWN-GWTC4p1-18965dda8_5-GW950702_185453-combined_PEDataRelease.hdf5"


def _gw230702_like(seed=0, n=120):
    """C00:Mixed = equal thirds of XPHM-ST, NRSur7dq4 and SEOBNRv5PHM, with
    the meta_data/combined record the real file carries."""
    rng = np.random.default_rng(seed)
    xp, nr, se = (_samples(rng, n) for _ in range(3))
    k = n // 2
    mixed = _concat(_take(xp, np.arange(k)), _take(nr, np.arange(k)),
                    _take(se, np.arange(k)))
    sd = {XP: xp, MX: mixed, NR: nr, SE: se}
    priors = {"analytic": {XP: _analytic(), SE: _analytic(),
                           NR: _analytic(qmin=0.16666667, total_mass=True)}}
    extra = {MX: {"meta_data": {"combined": f"{XP}, {NR}, {SE}"}}}
    return _Data(sd, extra=extra), sd, priors


_CFG = dict(validate_prior=False, dL_prior_impl="astropy")


def _ingest(tmp_path, monkeypatch, files, cfg=None, name="store.h5", **kw):
    """Run the REAL build_store; ``files`` maps basename -> (data, sd, priors)."""
    def _read(path):
        data, sd, priors = files[os.path.basename(path)]
        return data, sd, list(sd), priors
    monkeypatch.setattr(ing, "_read_event_pesummary", _read)
    paths = []
    for base in files:
        p = tmp_path / base
        p.write_bytes(b"")
        paths.append(str(p))
    out = tmp_path / name
    ing.build_store(paths, str(out), event_table={},
                    cfg=cfg or ing.IngestConfig(**_CFG), **kw)
    return str(out)


def _meta(store, col, row=0):
    with h5py.File(store, "r") as f:
        v = f[f"meta/{col}"][row]
    return v.decode() if isinstance(v, (bytes, bytearray)) else v


# ==========================================================================
# critical-1: Mixed sets whose constituents' priors differ
# ==========================================================================
def test_mixed_constituent_prior_difference_is_recorded_and_warned(
        tmp_path, monkeypatch):
    """GW230702-like: NRSur7dq4's q >= 1/6 and total-mass Constraint differ
    from its siblings; the default keeps the borrowed prior but says so."""
    files = {O4_NAME: _gw230702_like()}
    with pytest.warns(UserWarning, match="declare different analytic priors"):
        store = _ingest(tmp_path, monkeypatch, files)
    assert _meta(store, "analysis_used") == MX
    assert _meta(store, "mixed_constituent_labels") == f"{XP}+{NR}+{SE}"
    assert _meta(store, "mixed_constituent_priors") == (
        "differ:mass_ratio,total_mass")


def test_mixed_constituent_prior_difference_refused_on_request(
        tmp_path, monkeypatch):
    files = {O4_NAME: _gw230702_like()}
    cfg = ing.IngestConfig(mixed_prior_mismatch="raise", **_CFG)
    with pytest.raises(ing.MixedPriorMismatchError, match="NRSur7dq4"):
        _ingest(tmp_path, monkeypatch, files, cfg=cfg)


def test_mixed_constituents_that_agree_are_recorded_as_agreeing(
        tmp_path, monkeypatch):
    data, sd, priors = _gw230702_like()
    priors["analytic"][NR] = _analytic()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*different analytic priors")
        store = _ingest(tmp_path, monkeypatch, {O4_NAME: (data, sd, priors)},
                        cfg=ing.IngestConfig(mixed_prior_mismatch="raise",
                                             **_CFG))
    assert _meta(store, "mixed_constituent_priors") == "agree"


# ==========================================================================
# high-2: C01:Mixed samples above the 0.99 ceiling (SEOBNRv4PHM half)
# ==========================================================================
C1X, C1M, C1S = "C01:IMRPhenomXPHM", "C01:Mixed", "C01:SEOBNRv4PHM"
O3_NAME = "IGWN-GWTC3p0-v2-GW951109_010717_PEDataRelease_mixed_cosmo.h5"


def _gw191109_like(seed=1, n=200):
    """XPHM samples stop at 0.99; SEOBNRv4PHM's reach ~1.0 (no analytic group,
    config ceiling 0.99) -- the real C01:Mixed layout."""
    rng = np.random.default_rng(seed)
    xp = _samples(rng, n, amax=0.99, gps=1257296855.2)
    se = _samples(rng, n, amax=1.0, gps=1257296855.2)
    mixed = _concat(_take(xp, np.arange(n // 2)), _take(se, np.arange(n // 2)))
    sd = {C1X: xp, C1M: mixed, C1S: se}
    dl = ("PowerLaw(alpha=2, minimum=10, maximum=10000, "
          "name='luminosity_distance', latex_label='$d_L$', unit='Mpc', "
          "boundary=None)")
    priors = {"analytic": {C1X: dict(_analytic(), luminosity_distance=dl)}}
    extra = {C1M: {"meta_data": {"combined": "Prod5, Prod6_extended"}}}
    return _Data(sd, extra=extra), sd, priors


def test_spin_samples_above_the_ceiling_are_recorded_per_constituent(
        tmp_path, monkeypatch):
    files = {O3_NAME: _gw191109_like()}
    with pytest.warns(UserWarning, match="samples exceed resolved amax"):
        store = _ingest(tmp_path, monkeypatch, files)
    assert _meta(store, "analysis_used") == C1M
    assert float(_meta(store, "spin_amax_1")) == pytest.approx(0.99)
    smax = max(float(_meta(store, "spin_sample_max_1")),
               float(_meta(store, "spin_sample_max_2")))
    assert smax > 0.995
    per = json.loads(_meta(store, "spin_sample_max_per_constituent"))
    assert set(per) == {C1X, C1S}
    assert max(per[C1X]) <= 0.99 and max(per[C1S]) > 0.995
    # GWTC-3 names run tags, not labels, in meta_data/combined
    assert _meta(store, "mixed_constituent_priors") == (
        "unverified:no_constituent_labels")


def test_spin_samples_above_the_ceiling_refused_on_request(
        tmp_path, monkeypatch):
    files = {O3_NAME: _gw191109_like()}
    cfg = ing.IngestConfig(spin_exceeds_amax="raise", **_CFG)
    with pytest.raises(ing.SpinCeilingExceededError, match="0.99"):
        _ingest(tmp_path, monkeypatch, files, cfg=cfg)


# ==========================================================================
# high-11: aligned-spin labels inherited a precessing sibling's prior
# ==========================================================================
# The real analysis labels in
# IGWN-GWTC4p1-18965dda8_5-GW230529_181500-combined_PEDataRelease.hdf5
GW230529_LABELS = [
    "C00:IMRPhenomNSBH", "C00:IMRPhenomPv2_NRTidalv2",
    "C00:IMRPhenomXAS:HighSpin", "C00:IMRPhenomXAS:LowSpinSecondary",
    "C00:IMRPhenomXHM", "C00:IMRPhenomXP", "C00:IMRPhenomXPHM:HighSpin",
    "C00:IMRPhenomXPHM:LowSpin", "C00:IMRPhenomXPHM:LowSpinSecondary",
    "C00:Mixed:HighSpin", "C00:Mixed:LowSpinSecondary",
    "C00:SEOBNRv4_ROM_NRTidalv2_NSBH", "C00:SEOBNRv5PHM:HighSpin",
    "C00:SEOBNRv5PHM:LowSpinSecondary"]


def _gw230529_priors():
    """Each label's real spin declaration (a_1/a_2 or AlignedSpin chi_1/chi_2)."""
    prec = {"C00:IMRPhenomPv2_NRTidalv2": (0.99, 0.05),
            "C00:IMRPhenomXP": (0.99, 0.99),
            "C00:IMRPhenomXPHM:HighSpin": (0.99, 0.99),
            "C00:IMRPhenomXPHM:LowSpin": (0.05, 0.05),
            "C00:IMRPhenomXPHM:LowSpinSecondary": (0.99, 0.05),
            "C00:SEOBNRv5PHM:HighSpin": (0.99, 0.99),
            "C00:SEOBNRv5PHM:LowSpinSecondary": (0.99, 0.05)}
    aligned = {"C00:IMRPhenomNSBH": (0.5, 0.05),
               "C00:IMRPhenomXAS:HighSpin": (0.99, 0.99),
               "C00:IMRPhenomXAS:LowSpinSecondary": (0.5, 0.05),
               "C00:IMRPhenomXHM": (0.99, 0.99),
               "C00:SEOBNRv4_ROM_NRTidalv2_NSBH": (0.9, 0.05)}
    an = {lab: {"a_1": _a(1, a1), "a_2": _a(2, a2)}
          for lab, (a1, a2) in prec.items()}
    an.update({lab: {"chi_1": _aligned(a1), "chi_2": _aligned(a2)}
               for lab, (a1, a2) in aligned.items()})
    return {"analytic": an}


@pytest.mark.parametrize("label, ceilings", [
    ("C00:IMRPhenomNSBH", (0.5, 0.05)),
    ("C00:IMRPhenomXHM", (0.99, 0.99)),
    ("C00:SEOBNRv4_ROM_NRTidalv2_NSBH", (0.9, 0.05)),
    ("C00:IMRPhenomXAS:LowSpinSecondary", (0.5, 0.05)),
])
def test_aligned_spin_label_uses_its_own_ceilings(label, ceilings):
    rng = np.random.default_rng(3)
    a1 = rng.uniform(0, ceilings[0], 50)
    a2 = rng.uniform(0, ceilings[1], 50)
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # no samples-exceed warning
        sp = ing.resolve_spin_prior_full(label, GW230529_LABELS,
                                         _gw230529_priors(), a1, a2)
    assert (sp.amax_1, sp.amax_2) == pytest.approx(ceilings)
    assert sp.kind == "aligned_spin"
    assert sp.source_kind == "own_analytic" and sp.source_label == label


def test_mixed_label_still_borrows_a_precessing_same_variant_sibling():
    sp = ing.resolve_spin_prior_full("C00:Mixed:HighSpin", GW230529_LABELS,
                                     _gw230529_priors(), None, None)
    assert (sp.amax_1, sp.amax_2) == pytest.approx((0.99, 0.99))
    assert sp.kind == "uniform_magnitude_isotropic"
    assert sp.source_kind == "sibling_inherited"


def test_same_variant_siblings_with_different_ceilings_are_refused():
    """Which sibling an unsuffixed label borrows from followed file order."""
    analyses = ["C01:IMRPhenomA", "C01:IMRPhenomB", "C01:Mixed"]
    priors = {"analytic": {"C01:IMRPhenomA": {"a_1": _a(1, 0.99),
                                              "a_2": _a(2, 0.99)},
                           "C01:IMRPhenomB": {"a_1": _a(1, 0.8),
                                              "a_2": _a(2, 0.8)}}}
    with pytest.raises(ing.SpinPriorMismatchError, match="different spin"):
        ing.resolve_spin_prior_full("C01:Mixed", analyses, priors, None, None)
    with pytest.warns(UserWarning, match="different spin ceilings"):
        sp = ing.resolve_spin_prior_full("C01:Mixed", analyses, priors, None,
                                         None, allow_variant_mismatch=True)
    assert "candidates_disagree" in sp.source


# ==========================================================================
# high-13: --constituent-mixture-prior on files that ship C00:Mixed+XO4a
# ==========================================================================
XO, MXO = "C00:IMRPhenomXO4a", "C00:Mixed+XO4a"
O4_NAME_XO4A = (
    "IGWN-GWTC4p1-18965dda8_5-GW950601_224134-combined_PEDataRelease.hdf5")


def _gw230601_like(seed=2, n=120):
    """The real GW230601_224134 layout: C00:Mixed+XO4a has exactly the
    C00:Mixed rows and the same meta_data/combined record; C00:IMRPhenomXO4a
    contributes no row to either."""
    data, sd, priors = _gw230702_like(seed=seed, n=n)
    rng = np.random.default_rng(seed + 100)
    sd = {XO: _samples(rng, n), XP: sd[XP], MX: sd[MX],
          MXO: {k: v.copy() for k, v in sd[MX].items()}, NR: sd[NR],
          SE: sd[SE]}
    priors["analytic"][XO] = _analytic()
    rec = {"meta_data": {"combined": f"{XP}, {SE}, {NR}"}}
    return _Data(sd, extra={MX: rec, MXO: rec}), sd, priors


def test_constituent_mixture_builds_on_a_file_with_mixed_plus_xo4a(
        tmp_path, monkeypatch):
    cfg = ing.IngestConfig(constituent_mixture_prior=True, **_CFG)
    store = _ingest(tmp_path, monkeypatch, {O4_NAME_XO4A: _gw230601_like()},
                    cfg=cfg)
    assert _meta(store, "analysis_used") == MX
    assert _meta(store, "mass_prior_kind") == "constituent_mixture"
    assert _meta(store, "constituent_mixture_labels") == f"{XP}+{NR}+{SE}"
    assert _meta(store, "constituent_mixture_counts") == "60,60,60"


# ==========================================================================
# medium-20: a mass prior in other coordinates is not a sibling's UIC prior
# ==========================================================================
_ETA = ("bilby.core.prior.Uniform(minimum=0.1, maximum=0.25, "
        "name='symmetric_mass_ratio')")
_MTOT = "bilby.core.prior.Uniform(minimum=20, maximum=200, name='total_mass')"
_M1U = "Uniform(minimum=5, maximum=100, name='mass_1')"


@pytest.mark.parametrize("own", [
    {"chirp_mass": UIC_MC, "symmetric_mass_ratio": _ETA},
    {"total_mass": _MTOT, "mass_ratio": _uic_q(0.05)},
    {"mass_1": _M1U, "mass_2": _M1U.replace("mass_1", "mass_2")},
    {"chirp_mass": UIC_MC, "mass_ratio": _uic_q(0.05), "total_mass": _MTOT},
])
def test_mass_prior_in_other_coordinates_is_unrecognized_not_borrowed(own):
    analyses = ["C00:Odd", "C00:IMRPhenomXPHM-SpinTaylor"]
    priors = {"analytic": {"C00:Odd": own,
                           "C00:IMRPhenomXPHM-SpinTaylor": _analytic()}}
    mp = ing.resolve_mass_prior("C00:Odd", analyses, priors)
    assert mp.kind == "unrecognized"
    assert mp.source_kind == "own_analytic" and mp.source_label == "C00:Odd"


def test_constraint_only_mass_keys_still_borrow_and_the_real_pair_verifies():
    analyses = ["C00:Mixed", XP, NR]
    priors = {"analytic": {XP: _analytic(),
                           NR: _analytic(qmin=0.16666667, total_mass=True),
                           "C00:Mixed": {"mass_1": _m_con(1),
                                         "mass_2": _m_con(2)}}}
    mp = ing.resolve_mass_prior("C00:Mixed", analyses, priors)
    assert mp.kind == "uniform_detector_frame"
    assert mp.source_kind == "sibling_inherited"
    nr = ing.resolve_mass_prior(NR, analyses, priors)
    assert nr.kind == "uniform_detector_frame" and nr.q_min == 0.16666667
