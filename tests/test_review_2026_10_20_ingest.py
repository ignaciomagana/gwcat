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


# ==========================================================================
# medium-24: duplicate (event, sample set) rows in one build
# ==========================================================================
def _one_label_file(seed, label=C1X):
    rng = np.random.default_rng(seed)
    sd = {label: _samples(rng, 80, amax=0.9)}
    return _Data(sd), sd, {"analytic": {label: _analytic()}}


def test_two_releases_of_one_event_are_refused_not_silently_kept(
        tmp_path, monkeypatch):
    files = {"IGWN-GWTC2p1-v1-GW950101_000001_PEDataRelease_mixed_cosmo.h5":
             _one_label_file(1),
             "IGWN-GWTC2p1-v2-GW950101_000001_PEDataRelease_mixed_cosmo.h5":
             _one_label_file(2)}
    with pytest.raises(ing.DuplicateSampleSetError, match="twice"):
        _ingest(tmp_path, monkeypatch, files)


def test_two_preferred_rows_for_one_event_warn(tmp_path, monkeypatch):
    files = {"IGWN-GWTC2p1-v1-GW950101_000001_PEDataRelease_mixed_cosmo.h5":
             _one_label_file(1, label=C1X),
             "IGWN-GWTC3p0-v1-GW950101_000001_PEDataRelease_mixed_cosmo.h5":
             _one_label_file(2, label=C1S)}
    with pytest.warns(UserWarning, match="more than one preferred"):
        store = _ingest(tmp_path, monkeypatch, files)
    with h5py.File(store, "r") as f:
        assert len(f["index/event_names"]) == 2


def test_cli_ingests_a_file_matched_by_two_globs_once(tmp_path, monkeypatch):
    for b in ("GW950101_000001_a.h5", "GW950102_000002_b.h5"):
        (tmp_path / b).write_bytes(b"")
    seen = {}
    monkeypatch.setattr(ing, "build_store",
                        lambda paths, out, **kw: seen.setdefault("p", paths))
    ing._cli(["--glob", str(tmp_path / "*.h5"), "--glob",
              str(tmp_path / "GW950101*.h5"), "--no-event-table",
              "--out", str(tmp_path / "s.h5")], _deprecated=False)
    assert [os.path.basename(p) for p in seen["p"]] == [
        "GW950101_000001_a.h5", "GW950102_000002_b.h5"]


# ==========================================================================
# medium-25: a re-release whose preferred label changed
# ==========================================================================
_R4 = "IGWN-GWTC4-v1-GW950529_181500-combined_PEDataRelease.hdf5"
_R41 = "IGWN-GWTC4p1-v1-GW950529_181500-combined_PEDataRelease.hdf5"


def _rerelease(tmp_path, monkeypatch, how):
    files = {_R4: _one_label_file(1, label="C00:IMRPhenomXPHM"),
             _R41: _one_label_file(2, label="C00:SEOBNRv5PHM")}
    store = _ingest(tmp_path, monkeypatch, {_R4: files[_R4]}, name="a.h5")
    p41 = tmp_path / _R41
    p41.write_bytes(b"")

    def _read(path):
        data, sd, priors = files[os.path.basename(path)]
        return data, sd, list(sd), priors
    monkeypatch.setattr(ing, "_read_event_pesummary", _read)
    out = tmp_path / f"merged_{how}.h5"
    ing.merge_store(store, [str(p41)], out_path=str(out),
                    cfg=ing.IngestConfig(**_CFG), event_table={},
                    on_duplicate_key=how)
    with h5py.File(out, "r") as f:
        return [x.decode() if isinstance(x, bytes) else x
                for x in f["meta/sample_set_name"][:]]


def test_refresh_replaces_the_preferred_row_when_the_label_changed(
        tmp_path, monkeypatch):
    with pytest.warns(UserWarning, match="Refreshed"):
        labels = _rerelease(tmp_path, monkeypatch, "refresh")
    assert labels == ["C00:SEOBNRv5PHM"]


def test_skip_keeps_the_old_preferred_row_when_the_label_changed(
        tmp_path, monkeypatch):
    with pytest.warns(UserWarning, match="skipped"):
        labels = _rerelease(tmp_path, monkeypatch, "skip")
    assert labels == ["C00:IMRPhenomXPHM"]


# ==========================================================================
# medium-44: a _nocosmo O1-O3 release with no analytic distance prior
# ==========================================================================
_NOCOSMO = "IGWN-GWTC2p1-v2-GW950608_020116_PEDataRelease_mixed_nocosmo.h5"


def _gw170608_like(config=True, seed=4, n=150):
    """The real GW170608 layout: no analytic group on any label; the two
    LALInference constituents declare distance-max 1000 and 10000."""
    rng = np.random.default_rng(seed)
    xp, se = _samples(rng, n), _samples(rng, n)
    for smp in (xp, se):
        smp["luminosity_distance"] = rng.uniform(150.0, 600.0, n)
    mixed = _concat(_take(xp, np.arange(n // 2)), _take(se, np.arange(n // 2)))
    sd = {C1X: xp, C1M: mixed, C1S: se}
    eng = {C1X: {"a_spin1-max": b"0.99", "a_spin2-max": b"0.99",
                 "distance-max": b"1000"},
           C1S: {"a_spin1-max": b"0.99", "a_spin2-max": b"0.99",
                 "distance-max": b"10000"}}
    cfg = {lab: {"engine": e} for lab, e in eng.items()} if config else {}
    return _Data(sd, config=cfg), sd, {}


def test_nocosmo_lalinference_rows_divide_out_dL_squared(tmp_path,
                                                          monkeypatch):
    from gwcat.cosmology import dL_prior_prob, make_cosmology
    data, sd, priors = _gw170608_like()
    store = _ingest(tmp_path, monkeypatch, {_NOCOSMO: (data, sd, priors)})
    assert _meta(store, "dL_prior_kind") == "PowerLaw"
    assert float(_meta(store, "dL_prior_alpha")) == 2.0
    assert _meta(store, "dL_prior_basis") == "config_file_declared"
    assert _meta(store, "prior_source_kind_dL") == "config_file_declared"
    assert (float(_meta(store, "dL_prior_min")),
            float(_meta(store, "dL_prior_max"))) == (1.0, 10000.0)
    with h5py.File(store, "r") as f:
        p = f["samples/p_dL_pe"][:]
    dL = sd[C1M]["luminosity_distance"]
    expected = dL_prior_prob(dL, kind="PowerLaw", alpha=2.0,
                             cosmology=make_cosmology(67.74, 0.3075),
                             dmin=1.0, dmax=10000.0, impl="astropy")
    np.testing.assert_allclose(p, expected, rtol=1e-12)
    np.testing.assert_allclose(p / dL ** 2, (p / dL ** 2)[0], rtol=1e-10)


def test_nocosmo_row_with_no_declaration_at_all_is_refused(tmp_path,
                                                           monkeypatch):
    from gwcat.cosmology import DistancePriorKindError
    with pytest.raises(DistancePriorKindError, match="refusing"):
        _ingest(tmp_path, monkeypatch,
                {_NOCOSMO: _gw170608_like(config=False)})


# ==========================================================================
# medium-45: the config-file spin ceilings borrowed across spin variants
# ==========================================================================
def _cfg_data(engines):
    return _Data({lab: {} for lab in engines},
                 config={lab: {"engine": e} for lab, e in engines.items()})


def test_config_ceilings_are_not_borrowed_across_spin_variants():
    labels = ["C01:X:HighSpin", "C01:X:LowSpin"]
    data = _cfg_data({"C01:X:HighSpin": {},
                      "C01:X:LowSpin": {"a_spin1-max": "0.99",
                                        "a_spin2-max": "0.05"}})
    with pytest.raises(ing.SpinPriorMismatchError, match="HighSpin"):
        ing.resolve_spin_prior_full("C01:X:HighSpin", labels, {}, None, None,
                                    data=data)


def test_config_ceilings_of_the_same_variant_are_still_used():
    labels = ["C01:Mixed", "C01:IMRPhenomXPHM", "C01:SEOBNRv4PHM"]
    eng = {"a_spin1-max": b"0.99", "a_spin2-max": b"0.99"}
    data = _cfg_data({"C01:Mixed": {}, "C01:IMRPhenomXPHM": eng,
                      "C01:SEOBNRv4PHM": eng})
    sp = ing.resolve_spin_prior_full("C01:Mixed", labels, {}, None, None,
                                     data=data)
    assert (sp.amax_1, sp.amax_2) == (0.99, 0.99)
    assert sp.source_kind == "config_file_declared"
    assert sp.source_label == "C01:IMRPhenomXPHM,C01:SEOBNRv4PHM"


# ==========================================================================
# medium-47: `gwcat ingest --inspect` ignored the prior-changing flags
# ==========================================================================
def test_cli_inspect_uses_the_build_flags(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(ing, "inspect",
                        lambda path, cfg=None: seen.setdefault("cfg", cfg))
    tab = tmp_path / "table.yaml"
    ing._cli(["--inspect", str(tmp_path / "x.h5"),
              "--release-reweight-cosmology-table", str(tab),
              "--legacy-grid-priors", "--constituent-mixture-prior"],
             _deprecated=False)
    cfg = seen["cfg"]
    assert cfg is not None
    assert cfg.release_reweight_cosmology_table == str(tab)
    assert cfg.dL_prior_impl == "auto"
    assert cfg.constituent_mixture_prior is True


def test_inspect_reports_the_constituent_mixture(tmp_path, monkeypatch,
                                                 capsys):
    data, sd, priors = _gw230702_like()
    monkeypatch.setattr(ing, "_read_event_pesummary",
                        lambda path: (data, sd, list(sd), priors))
    path = tmp_path / O4_NAME
    path.write_bytes(b"")
    info = ing.inspect(str(path), ing.IngestConfig(
        constituent_mixture_prior=True, **_CFG))
    assert info["constituent_mixture"]["labels"] == [XP, NR, SE]
    assert info["mixed_constituents"]["priors"] == (
        "differ:mass_ratio,total_mass")
    assert info["ingest_config"]["dL_prior_impl"] == "astropy"


# ==========================================================================
# medium-48: the prior-sample bounds and KS checks against sampling noise
# ==========================================================================
_PL2 = ("PowerLaw(alpha=2, minimum=10, maximum=10000, "
        "name='luminosity_distance', latex_label='$d_L$', unit='Mpc', "
        "boundary=None)")
_O4_PL = "IGWN-GWTC4p1-18965dda8_5-GW950101_000001-combined_PEDataRelease.hdf5"


def _pl2_draws(rng, n, lo=10.0, hi=10000.0):
    u = rng.uniform(0.0, 1.0, n)
    return (lo ** 3 + u * (hi ** 3 - lo ** 3)) ** (1.0 / 3.0)


def _prior_sample_file(n_prior, seed, hi=10000.0):
    rng = np.random.default_rng(seed)
    sd = {XP: _samples(rng, 100)}
    priors = {"analytic": {XP: dict(_analytic(), luminosity_distance=_PL2)},
              "samples": {XP: {"luminosity_distance":
                               _pl2_draws(rng, n_prior, hi=hi)}}}
    return _Data(sd), sd, priors


@pytest.mark.parametrize("n_prior, seed", [(50, 11), (200, 12), (5000, 13)])
def test_correct_prior_draws_pass_the_audited_gates(tmp_path, monkeypatch,
                                                    n_prior, seed):
    cfg = ing.IngestConfig(validate_prior=True, prior_ks_fatal=True,
                           dL_prior_impl="astropy")
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*describe different ranges")
        store = _ingest(tmp_path, monkeypatch,
                        {_O4_PL: _prior_sample_file(n_prior, seed)}, cfg=cfg)
    assert float(_meta(store, "dL_prior_n_prior_samples")) == n_prior
    assert float(_meta(store, "dL_prior_frac_samples_outside")) == 0.0
    assert 0.0 < float(_meta(store, "dL_prior_frac_mass_uncovered")) < 0.2
    assert float(_meta(store, "dL_prior_bounds_pvalue")) > 1e-3
    assert float(_meta(store, "dL_prior_ks")) <= float(
        _meta(store, "dL_prior_ks_threshold"))


def test_a_genuine_bounds_mismatch_still_warns(tmp_path, monkeypatch):
    cfg = ing.IngestConfig(validate_prior=True, dL_prior_impl="astropy")
    with pytest.warns(UserWarning, match="describe different ranges"):
        store = _ingest(tmp_path, monkeypatch,
                        {_O4_PL: _prior_sample_file(5000, 14, hi=5000.0)},
                        cfg=cfg)
    assert float(_meta(store, "dL_prior_bounds_pvalue")) < 1e-3


# ==========================================================================
# low-28: a NaN KS (bounds miss the prior samples) bypassed prior_ks_fatal
# ==========================================================================
def test_prior_samples_outside_the_parsed_bounds_fail_the_audited_gate(
        tmp_path, monkeypatch):
    data, sd, priors = _prior_sample_file(500, 15)
    rng = np.random.default_rng(16)
    priors["samples"][XP]["luminosity_distance"] = rng.uniform(
        20000.0, 30000.0, 500)
    cfg = ing.IngestConfig(validate_prior=True, prior_ks_fatal=True,
                           dL_prior_impl="astropy")
    with pytest.raises(ing.PriorMismatchError, match="do not overlap"):
        _ingest(tmp_path, monkeypatch, {_O4_PL: (data, sd, priors)}, cfg=cfg)


# ==========================================================================
# medium-49: sky_area_90 limited by the sample count
# ==========================================================================
def test_sparse_broad_posterior_gets_no_fake_small_sky_area():
    """GW200322_091133-like: 69 samples spread over the sky were stored at
    ~0.9 N pixels of 0.84 deg^2 = ~53 deg^2."""
    rng = np.random.default_rng(17)
    ra = rng.uniform(0.0, 2 * np.pi, 69)
    dec = np.arcsin(rng.uniform(-1.0, 1.0, 69))
    area = ing._sky_area_90(ra, dec)
    assert not (area < 1000.0)            # NaN, or a genuinely large area
    a2, ns = ing._sky_area_90_info(ra, dec)
    assert np.isnan(a2) and np.isnan(ns)


def test_well_sampled_posterior_keeps_the_nside_64_value():
    # healpy is optional (no extra declares it): without it the sky area is NaN.
    hp = pytest.importorskip("healpy")
    rng = np.random.default_rng(18)
    n = 20000
    ra = 1.0 + 0.03 * rng.standard_normal(n)
    dec = 0.3 + 0.03 * rng.standard_normal(n)
    area, ns = ing._sky_area_90_info(ra, dec)
    pix = hp.ang2pix(64, np.pi / 2 - dec, ra)
    c = np.cumsum(np.sort(np.bincount(pix, minlength=hp.nside2npix(64)))[::-1])
    old = (int(np.searchsorted(c, 0.9 * c[-1]) + 1)
           * hp.nside2pixarea(64, degrees=True))
    assert ns == 64.0 and area == old


def test_sky_area_nside_is_stored(tmp_path, monkeypatch):
    store = _ingest(tmp_path, monkeypatch, {O3_NAME: _gw191109_like()})
    assert np.isnan(float(_meta(store, "sky_area_90_nside")))
    assert np.isnan(float(_meta(store, "sky_area_90")))


# ==========================================================================
# medium-54: merge_store could not flag appended synthetic PE as mock
# ==========================================================================
_E1 = "IGWN-GWTC3p0-v1-GW950101_000001_PEDataRelease_mixed_cosmo.h5"
_E2 = "IGWN-GWTC3p0-v1-GW950102_000002_PEDataRelease_mixed_cosmo.h5"


def _merge_new(tmp_path, monkeypatch, *, existing_mock, **kw):
    files = {_E1: _one_label_file(1), _E2: _one_label_file(2)}
    store = _ingest(tmp_path, monkeypatch, {_E1: files[_E1]}, name="a.h5",
                    mock_data=existing_mock)
    (tmp_path / _E2).write_bytes(b"")
    monkeypatch.setattr(ing, "_read_event_pesummary",
                        lambda path: (lambda d, sd, pr: (d, sd, list(sd), pr))(
                            *files[os.path.basename(path)]))
    out = tmp_path / "merged.h5"
    ing.merge_store(store, [str(tmp_path / _E2)], out_path=str(out),
                    cfg=ing.IngestConfig(**_CFG), event_table={}, **kw)
    with h5py.File(out, "r") as f:
        return ing.read_store_mock_data(f)


def test_merge_store_marks_appended_synthetic_pe_as_mock(tmp_path,
                                                          monkeypatch):
    with pytest.warns(UserWarning, match="mock"):
        assert _merge_new(tmp_path, monkeypatch, existing_mock=False,
                          mock_data=True) is True


def test_merge_store_into_a_mock_store_stays_mock_without_warning(
        tmp_path, monkeypatch):
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*mock-data store with")
        assert _merge_new(tmp_path, monkeypatch, existing_mock=True) is True


# ==========================================================================
# low-10: a C00:Mixed row's priors are searched in its constituents first
# ==========================================================================
def test_mixed_row_borrows_from_a_constituent_not_from_xo4a(tmp_path,
                                                             monkeypatch):
    """GW230601-like: C00:IMRPhenomXO4a sorts first but is not a constituent."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store = _ingest(tmp_path, monkeypatch,
                        {O4_NAME_XO4A: _gw230601_like()})
    assert _meta(store, "analysis_used") == MX
    for p in ("dL", "mass", "spin"):
        assert _meta(store, f"prior_source_label_{p}") == XP
    cfgs = json.loads(_meta(store, "spin_amax_config_per_constituent"))
    assert sorted(cfgs) == sorted([XP, NR, SE])


def test_sibling_search_order_puts_constituents_then_same_variant_first():
    order = ing._sibling_search_order(
        "C00:Mixed:HighSpin",
        ["C00:IMRPhenomNSBH", "C00:IMRPhenomXPHM:LowSpin", "C00:Mixed:HighSpin",
         "C00:SEOBNRv5PHM:HighSpin", "C00:IMRPhenomXPHM:HighSpin"],
        constituents=["C00:IMRPhenomXPHM:HighSpin"])
    assert order == ["C00:IMRPhenomXPHM:HighSpin", "C00:SEOBNRv5PHM:HighSpin",
                     "C00:IMRPhenomNSBH", "C00:IMRPhenomXPHM:LowSpin"]


# ==========================================================================
# low-73: C00:Mixed+XO4a is a combined set, ranked after C00:Mixed
# ==========================================================================
GW230601_LABELS = [XO, XP, MX, MXO, NR, SE]


def test_mixed_plus_xo4a_is_combined_and_ranked_after_plain_mixed():
    cfg = ing.IngestConfig()
    assert ing.find_mixed_analyses(GW230601_LABELS, "C00") == [MX, MXO]
    assert ing.select_analysis(GW230601_LABELS, "C00", cfg,
                               catalog="GWTC-4.1") == MX
    ranked = ing.rank_analyses(GW230601_LABELS, "C00", cfg, catalog="GWTC-4.1")
    assert ranked[:2] == [MX, MXO]
    meta = ing._sample_set_meta(MXO, MX, ranked, O4_NAME_XO4A, "all")
    assert meta["is_mixed"] == 1.0 and meta["priority_rank"] == 1.0


def test_mixed_plus_xo4a_row_carries_its_constituent_record(tmp_path,
                                                            monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store = _ingest(tmp_path, monkeypatch,
                        {O4_NAME_XO4A: _gw230601_like()},
                        sample_sets=[MXO])
    assert _meta(store, "analysis_used") == MXO
    assert float(_meta(store, "is_mixed")) == 1.0
    assert sorted(json.loads(_meta(store, "spin_amax_config_per_constituent"))
                  ) == sorted([XP, NR, SE])
    assert _meta(store, "mixed_constituent_labels") == f"{XP}+{SE}+{NR}"


# ==========================================================================
# low-27: a defaulted cosmology the KS gate cannot catch
# ==========================================================================
_USF_ODD = ("bilby.gw.prior.UniformSourceFrame(minimum=100.0, maximum=10000.0, "
            "cosmology='Planck99_XYZ', name='luminosity_distance')")
# A cosmology-dependent prior that names no cosmology at all: the only case
# still evaluated at the catalog default (an unknown token is refused at
# parse time by the priors/spin package of the same review).
_USF_NONE = ("bilby.gw.prior.UniformSourceFrame(minimum=100.0, "
             "maximum=10000.0, name='luminosity_distance')")


def _odd_cosmology_file(dl=_USF_ODD):
    rng = np.random.default_rng(19)
    sd = {XP: _samples(rng, 80)}
    return _Data(sd), sd, {"analytic": {
        XP: dict(_analytic(), luminosity_distance=dl)}}


def test_defaulted_cosmology_is_refused_in_an_audited_build(tmp_path,
                                                            monkeypatch):
    cfg = ing.IngestConfig(defaulted_cosmology_fatal=True, **_CFG)
    with pytest.raises(ing.PriorMismatchError, match="catalog default"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ingest(tmp_path, monkeypatch,
                    {_O4_PL: _odd_cosmology_file(_USF_NONE)}, cfg=cfg)


def test_defaulted_cosmology_is_recorded_by_default(tmp_path, monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store = _ingest(tmp_path, monkeypatch,
                        {_O4_PL: _odd_cosmology_file(_USF_NONE)})
    assert _meta(store, "dL_prior_cosmology_source") == "catalog_default"


def test_unknown_cosmology_token_is_refused_not_defaulted(tmp_path,
                                                         monkeypatch):
    from gwcat.cosmology import UnrecognisedCosmologyError
    with pytest.raises(UnrecognisedCosmologyError, match="Planck99_XYZ"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ingest(tmp_path, monkeypatch, {_O4_PL: _odd_cosmology_file()})


# ==========================================================================
# low-31: the store records the build that wrote it
# ==========================================================================
def test_build_store_records_writer_and_config(tmp_path, monkeypatch):
    store = _ingest(tmp_path, monkeypatch, {_E1: _one_label_file(1)})
    with h5py.File(store, "r") as f:
        prov = ing.read_store_provenance(f)
    assert prov["store_writer"] == "build_store"
    assert prov["gwcat_writer_commit"] and prov["gwcat_writer_version"]
    conf = json.loads(prov["ingest_config_json"])
    assert conf["dL_prior_impl"] == "astropy"
    assert conf["validate_prior"] is False
    assert json.loads(prov["store_inputs_json"])["files"] == [_E1]
    assert len(json.loads(prov["release_reweight_table_sha256"])) == 1


def test_merge_stores_records_its_inputs(tmp_path, monkeypatch):
    a = _ingest(tmp_path, monkeypatch, {_E1: _one_label_file(1)},
                name="a.h5")
    b = _ingest(tmp_path, monkeypatch, {_E2: _one_label_file(2)},
                name="b.h5")
    out = tmp_path / "m.h5"
    ing.merge_stores(a, b, str(out))
    with h5py.File(out, "r") as f:
        prov = ing.read_store_provenance(f)
    assert prov["store_writer"] == "merge_stores"
    ins = json.loads(prov["store_inputs_json"])["stores"]
    assert [i["path"] for i in ins] == ["a.h5", "b.h5"]
    assert all(i["store_writer"] == "build_store" for i in ins)


# ==========================================================================
# low-32: constituent-mixture rows kept the sibling's distance-prior record
# ==========================================================================
_DL_NARROW = DL_USF.replace("maximum=10000.0", "maximum=600.0")


def test_mixture_row_distance_record_describes_the_mixture(tmp_path,
                                                           monkeypatch):
    data, sd, priors = _gw230702_like()
    priors["analytic"][XP]["luminosity_distance"] = _DL_NARROW
    cfg = ing.IngestConfig(constituent_mixture_prior=True, **_CFG)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store = _ingest(tmp_path, monkeypatch, {O4_NAME: (data, sd, priors)},
                        cfg=cfg)
    assert _meta(store, "dL_prior_basis") == "constituent_mixture"
    assert (float(_meta(store, "dL_prior_min")),
            float(_meta(store, "dL_prior_max"))) == (100.0, 10000.0)
    assert float(_meta(store, "n_samples_outside_dL_prior_bounds")) == 0.0
    assert _meta(store, "dL_prior_sampling_kind") == "UniformSourceFrame"
    assert np.isnan(float(_meta(store, "dL_prior_ks")))


# ==========================================================================
# low-33: observing_run from the GPS time, not the event-name date
# ==========================================================================
def test_engineering_run_event_is_not_stored_as_o4a():
    """GW230518_125908: GPS 1368449966, before O4a's official start."""
    smp = {"geocent_time": np.full(10, 1368449966.4)}
    assert ing._observing_run_for("GW230518_125908", smp) == (
        "", "gps_outside_exposure_windows")
    assert ing._observing_run_for(
        "GW230601_224134", {"geocent_time": np.full(5, 1369694512.0)}) == (
        "O4a", "gps_exposure_windows")
    assert ing._observing_run_for("GW230518_125908", {}) == (
        "O4a", "event_name")


def test_store_records_how_the_run_was_assigned(tmp_path, monkeypatch):
    store = _ingest(tmp_path, monkeypatch, {O3_NAME: _gw191109_like()})
    assert _meta(store, "observing_run") == "O3b"
    assert _meta(store, "observing_run_source") == "gps_exposure_windows"


# ==========================================================================
# low-71: the merged store's mock flag follows the rows that survive
# ==========================================================================
def test_merge_whose_mock_rows_are_all_skipped_is_real(tmp_path,
                                                       monkeypatch):
    real = _ingest(tmp_path, monkeypatch, {_E1: _one_label_file(1)},
                   name="real.h5")
    mock = _ingest(tmp_path, monkeypatch, {_E1: _one_label_file(1)},
                   name="mock.h5", mock_data=True)
    out = tmp_path / "m.h5"
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*mock-data store with")
        with pytest.warns(UserWarning, match="skipped"):
            ing.merge_stores(real, mock, str(out))
    with h5py.File(out, "r") as f:
        assert ing.read_store_mock_data(f) is False
        assert len(f["index/event_names"]) == 1
