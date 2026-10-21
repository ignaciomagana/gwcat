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
    m1 = rng.uniform(30.0, 60.0, n)
    q = rng.uniform(0.2, 1.0, n)
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
