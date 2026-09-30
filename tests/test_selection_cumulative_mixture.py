"""GW-39: the LVK cumulative O1-O4b mixture (Zenodo 19500052), run by run.

One file, six runs: semianalytic O1/O2 rows (detected by the simulated observed
SNR, every FAR +inf, no sky), real O3a/O3b/O4a/O4b rows (detected by the run's
own search FARs), per-row mixture weights and one joint cartesian draw density.
The synthetic fixture below has rows in all six injection-support windows and
the same column layout as the release (compound ``events`` dataset, no
``right_ascension``/``declination``, no detector-frame masses).

Covered (plan section 4.1):
  (a) the run-aware mask equals the README ``(snr>thr)|(far<thr)``;
  (b) FAR-only on a file with semianalytic rows raises;
  (c) a nonzero SNR on an O3 row, or a finite FAR on an O1 row, raises;
  (d) the per-run attrs are written, including ``T_definition_per_run``;
  (e) the mixture plus an overlapping endo3-like campaign raises;
  (f) pdraw = exp(lnpdraw) J / (w T) per row, and a toy pdet equals the
      README estimator to 1e-12;
  (g) spin-reference coverage is recorded per run under ``chieff_reference``;
  (h) ``--sky-marginal`` writes no ra/dec and sets the attr;
  (i) ``chieff`` on a joint-cartesian file raises;
  (j) ``run_of_gps`` with each window table (a GW150914-like time is in the
      exposure table, not the support table).

Plus the data-gated integration test on the real release files, skipped unless
``GWCAT_REAL_DATA=1``.
"""
import json
import os
import warnings

import h5py
import numpy as np
import pytest

from gwcat.cli import main
from gwcat.export import build_selection_product, validate_export_v2
from gwcat.export.selection_builder import (BlockCampaignMismatch,
                                            OverlappingCampaignError,
                                            SpinBasisError)
from gwcat.observing_runs import (EXPOSURE_WINDOWS_GPS, INJECTION_SUPPORT_GPS,
                                  RUN_LABELS, RunWindowError,
                                  derive_mixture_bookkeeping, run_of_gps)
from gwcat.selection import (MixtureDetectionError, MixtureInvariantError,
                             SelectionSet, SNR_COLUMN)
from gwcat.spin import chi_eff_prior_logprob_in_support

from test_selection_spin import write_endo3_full, write_o4_full

_YEAR_S = 365.25 * 24 * 3600
_LN_2PI = np.log(2.0 * np.pi)
_JOINT = ("lnpdraw_mass1_source_mass2_source_redshift_"
          "spin1x_spin1y_spin1z_spin2x_spin2y_spin2z")
_SEARCHES = ["o3_cwb", "o3_gstlal", "o3_mbta", "o3_pycbc_bbh",
             "o3_pycbc_hyperbank", "o4a_cwb-bbh", "o4a_gstlal", "o4a_mbta",
             "o4a_pycbc", "o4b_cwb-bbh", "o4b_gstlal", "o4b_mbta",
             "o4b_pycbc"]

# The fixture's exposure components: N_k draws over T_k seconds, so the
# per-row weight is (T_k/T)/(N_k/N).  O4 is split into two "months" with
# different weights, as the release's monthly O4 weights are.
_N = {"O1": 4000, "O2": 9000, "O3": 5000, "O4": 10000}
_T = {"O1": 2.0e6, "O2": 3.0e6, "O3": 5.0e6, "O4": 8.0e6}
_ANCHORS = {"O3": {"T_s": _T["O3"], "N": _N["O3"], "source": "fixture"},
            "O4": {"T_s": _T["O4"], "N": _N["O4"], "source": "fixture"}}
_COMPONENT = {"O1": "O1", "O2": "O2", "O3a": "O3", "O3b": "O3",
              "O4a": "O4", "O4b": "O4"}


def write_cumulative_mixture(path, n_per_run=48, seed=7, amax=None,
                             corrupt=None):
    """Write a release-shaped cumulative mixture with rows in all six runs.

    ``amax`` optionally maps a run to its magnitude ceiling (default 0.999).
    ``corrupt`` breaks one invariant: ``"snr_on_o3"``, ``"far_on_o1"``,
    ``"cross_run_far"`` or ``"time_outside"``.
    """
    rng = np.random.default_rng(seed)
    amax = dict(amax or {})
    Ntot = sum(_N.values())
    Ttot = sum(_T.values())
    cols = {k: [] for k in (
        "mass1_source", "mass2_source", "luminosity_distance", "redshift",
        "spin1x", "spin1y", "spin1z", "spin2x", "spin2y", "spin2z",
        "time_geocenter", "dluminosity_distance_dredshift", _JOINT,
        SNR_COLUMN, "weights")}
    for s in _SEARCHES:
        cols[s + "_far"] = []
    for r in RUN_LABELS:
        n = n_per_run
        lo, hi = INJECTION_SUPPORT_GPS[r]
        m1 = rng.uniform(20.0, 60.0, n)
        m2 = rng.uniform(5.0, 20.0, n)
        cols["mass1_source"].append(np.maximum(m1, m2))
        cols["mass2_source"].append(np.minimum(m1, m2))
        z = rng.uniform(0.05, 1.2, n)
        cols["redshift"].append(z)
        cols["luminosity_distance"].append(z * 5000.0 * (1 + 0.3 * z))
        cols["dluminosity_distance_dredshift"].append(
            rng.uniform(3000.0, 9000.0, n))
        a_top = amax.get(r, 0.999)
        for i in (1, 2):
            a = rng.uniform(0.0, 1.0, n) * a_top
            a[0] = a_top                     # the ceiling is actually drawn
            cost = rng.uniform(-1.0, 1.0, n)
            st = np.sqrt(1.0 - cost ** 2)
            ph = rng.uniform(0.0, 2 * np.pi, n)
            cols[f"spin{i}x"].append(a * st * np.cos(ph))
            cols[f"spin{i}y"].append(a * st * np.sin(ph))
            cols[f"spin{i}z"].append(a * cost)
        cols["time_geocenter"].append(np.sort(rng.uniform(lo, hi, n)))
        cols[_JOINT].append(rng.normal(-12.0, 1.5, n))
        comp = _COMPONENT[r]
        w = (_T[comp] / Ttot) / (_N[comp] / Ntot)
        wr = np.full(n, w)
        if r == "O4b":                       # a second O4 "month"
            wr[n // 2:] = w * 1.07
        cols["weights"].append(wr)
        far = {s: np.full(n, np.inf) for s in _SEARCHES}
        if r in ("O1", "O2"):
            cols[SNR_COLUMN].append(rng.uniform(0.0, 20.0, n))
        else:
            cols[SNR_COLUMN].append(np.zeros(n))
            prefix = {"O3a": "o3_", "O3b": "o3_", "O4a": "o4a_",
                      "O4b": "o4b_"}[r]
            for s in _SEARCHES:
                if s.startswith(prefix):
                    v = 10 ** rng.uniform(-3.0, 1.5, n)
                    v[rng.uniform(size=n) < 0.3] = np.inf
                    far[s] = v
        for s in _SEARCHES:
            cols[s + "_far"].append(far[s])
    data = {k: np.concatenate(v) for k, v in cols.items()}
    run = np.concatenate([[r] * n_per_run for r in RUN_LABELS])
    if corrupt == "snr_on_o3":
        data[SNR_COLUMN][np.flatnonzero(run == "O3a")[3]] = 12.0
    elif corrupt == "far_on_o1":
        data["o3_gstlal_far"][np.flatnonzero(run == "O1")[5]] = 0.5
    elif corrupt == "cross_run_far":
        data["o3_mbta_far"][np.flatnonzero(run == "O4a")[2]] = 0.2
    elif corrupt == "time_outside":
        # GW150914: inside the O1 exposure window, before the O1 support.
        data["time_geocenter"][np.flatnonzero(run == "O1")[0]] = 1126259462.4
    elif corrupt is not None:
        raise ValueError(corrupt)

    dtype = [(k, "f8") for k in data]
    rec = np.zeros(run.size, dtype=dtype)
    for k, v in data.items():
        rec[k] = v
    with h5py.File(path, "w") as f:
        f.attrs["total_generated"] = float(Ntot)
        f.attrs["total_analysis_time"] = float(Ttot)
        f.attrs["searches"] = _SEARCHES
        f.create_dataset("events", data=rec)
    return str(path), data, run


def _readme_mask(data, far_thr=1.0, snr_thr=10.0):
    """The release README rule, from the raw columns alone."""
    far = np.min([data[s + "_far"] for s in _SEARCHES], axis=0)
    return (data[SNR_COLUMN] > snr_thr) | (far < far_thr)


def _build(path, **kw):
    kw.setdefault("spin_basis", "component")
    kw.setdefault("snr_threshold", 10.0)
    kw.setdefault("mixture_anchors", _ANCHORS)
    return build_selection_product(SelectionSet(path), **kw)


# ======================================================================
# (j) run_of_gps with each window table
# ======================================================================
def test_run_of_gps_support_and_exposure_tables():
    gw150914 = 1126259462.4
    assert run_of_gps(gw150914, EXPOSURE_WINDOWS_GPS) == "O1"
    with pytest.raises(RunWindowError, match="INJECTION_SUPPORT_GPS"):
        run_of_gps(gw150914, INJECTION_SUPPORT_GPS)
    # Mid-window times resolve identically under both tables.
    mids = [0.5 * (lo + hi) for lo, hi in INJECTION_SUPPORT_GPS.values()]
    for table in (INJECTION_SUPPORT_GPS, EXPOSURE_WINDOWS_GPS):
        assert run_of_gps(np.array(mids), table).tolist() == list(RUN_LABELS)
    # Closed intervals at both ends; gaps between runs are refused.
    for r, (lo, hi) in INJECTION_SUPPORT_GPS.items():
        assert run_of_gps(lo) == r and run_of_gps(hi) == r
    with pytest.raises(RunWindowError):
        run_of_gps([1200000000.0])           # between O2 and O3a
    with pytest.raises(RunWindowError):
        run_of_gps([np.nan])
    # Every support window lies inside its exposure window.
    for r in RUN_LABELS:
        (slo, shi), (elo, ehi) = INJECTION_SUPPORT_GPS[r], EXPOSURE_WINDOWS_GPS[r]
        assert elo <= slo and shi <= ehi


def test_run_of_gps_refuses_overlapping_table():
    with pytest.raises(ValueError, match="overlap"):
        run_of_gps([1.0], {"A": (0.0, 2.0), "B": (1.5, 3.0)})


# ======================================================================
# Loading: campaign kind and per-row runs
# ======================================================================
def test_mixture_is_recognised_and_rows_assigned(tmp_path):
    path, data, run = write_cumulative_mixture(tmp_path / "mix.hdf")
    s = SelectionSet(path)
    assert s.campaign_kind == "cumulative_mixture"
    assert s.run.tolist() == run.tolist()
    assert s._sky_position_available is False
    assert s.spin_meta["spin_format"] == "joint_cartesian"
    # A joint file WITHOUT the semianalytic SNR column is a single campaign.
    o4 = write_o4_full(tmp_path / "o4.hdf", n=20)
    assert SelectionSet(o4).campaign_kind == "single_campaign"
    assert SelectionSet(o4).run is None


# ======================================================================
# (a) run-aware mask == README rule
# ======================================================================
def test_run_aware_mask_equals_readme_rule(tmp_path):
    path, data, run = write_cumulative_mixture(tmp_path / "mix.hdf")
    s = SelectionSet(path)
    for far_thr, snr_thr in [(1.0, 10.0), (0.1, 9.0), (3.0, 11.0)]:
        want = _readme_mask(data, far_thr, snr_thr)
        got = s.detected_mask(far_thr, snr_threshold=snr_thr)
        np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(
            s.detected_mask(far_thr, snr_threshold=snr_thr, per_run=False),
            want)
        # The per-run rule really is per run: O1/O2 by SNR alone, O3/O4 by
        # their own FARs alone.
        semi = np.isin(run, ["O1", "O2"])
        np.testing.assert_array_equal(got[semi],
                                      data[SNR_COLUMN][semi] > snr_thr)
    # Non-trivial fixture: detections and non-detections in every run.
    det = s.detected_mask(1.0, snr_threshold=10.0)
    for r in RUN_LABELS:
        assert 0 < det[run == r].sum() < (run == r).sum(), r


# ======================================================================
# (b) FAR-only on a file with semianalytic rows raises
# ======================================================================
def test_far_only_on_mixture_raises_everywhere(tmp_path):
    path, data, run = write_cumulative_mixture(tmp_path / "mix.hdf")
    s = SelectionSet(path)
    with pytest.raises(MixtureDetectionError, match="O1/O2"):
        s.detected_mask(1.0)
    with pytest.raises(MixtureDetectionError, match="O1/O2"):
        build_selection_product(s, spin_basis="component")
    # The v1 exporter cannot express the per-run rule; it refuses too.
    with pytest.raises(MixtureDetectionError):
        s.to_darksirens(str(tmp_path / "v1.h5"), strict=False)
    # CLI: a diagnosed refusal (exit 1) that explains the exposure bias.
    rc = main(["export", "selection", path, "--out",
               str(tmp_path / "cli.h5"), "--parameter-space", "component"])
    assert rc == 1
    # lvk-cumulative without an SNR threshold is refused before any load.
    rc = main(["export", "selection", path, "--out",
               str(tmp_path / "cli2.h5"), "--parameter-space", "component",
               "--detection-policy", "lvk-cumulative"])
    assert rc == 1
    assert not (tmp_path / "cli.h5").exists()
    assert not (tmp_path / "cli2.h5").exists()


def test_far_only_error_text_explains_the_exposure_bias(tmp_path, capsys):
    path, _, _ = write_cumulative_mixture(tmp_path / "mix.hdf")
    main(["export", "selection", path, "--out", str(tmp_path / "x.h5"),
          "--parameter-space", "component"])
    err = capsys.readouterr().err
    assert "exposure" in err and "O1/O2" in err and "--snr-threshold" in err


def test_acknowledged_far_only_detects_no_semianalytic_row(tmp_path):
    path, data, run = write_cumulative_mixture(tmp_path / "mix.hdf")
    p = _build(path, snr_threshold=None,
               acknowledge_semianalytic_excluded=True)
    n_per_run = dict(zip([str(x) for x in p.attrs["run_labels"]],
                         p.attrs["n_detected_per_run"].tolist()))
    assert n_per_run["O1"] == 0 and n_per_run["O2"] == 0
    assert p.attrs["acknowledge_semianalytic_excluded"] is True
    rules = [str(x) for x in p.attrs["detection_rule_per_run"]]
    assert rules[0] == "none(semianalytic_rows_excluded)"
    # ndraw is NOT touched: dropping rows is subsetting.
    assert p.attrs["ndraw"] == sum(_N.values())


def test_lvk_cumulative_refuses_a_non_mixture(tmp_path):
    o4 = write_o4_full(tmp_path / "o4.hdf", n=20)
    with pytest.raises(ValueError, match="none of the campaigns"):
        build_selection_product(SelectionSet(o4), spin_basis="component",
                                snr_threshold=10.0,
                                detection_policy="lvk-cumulative")


# ======================================================================
# (c) invariants
# ======================================================================
@pytest.mark.parametrize("corrupt,match", [
    ("snr_on_o3", "nonzero semianalytic SNR"),
    ("far_on_o1", "finite FAR on a semianalytic"),
    ("cross_run_far", "another run's search"),
    ("time_outside", "outside every injection-support window"),
])
def test_invariant_violations_raise(tmp_path, corrupt, match):
    path, _, _ = write_cumulative_mixture(tmp_path / f"{corrupt}.hdf",
                                          corrupt=corrupt)
    with pytest.raises(MixtureInvariantError, match=match):
        SelectionSet(path)._load()


# ======================================================================
# (d) per-run attrs
# ======================================================================
def test_per_run_attrs_written(tmp_path):
    path, data, run = write_cumulative_mixture(tmp_path / "mix.hdf")
    out = tmp_path / "sel.h5"
    SelectionSet(path).export(str(out), spin_basis="component",
                              snr_threshold=10.0,
                              detection_policy="lvk-cumulative",
                              mixture_anchors=_ANCHORS)
    det = _readme_mask(data)
    with h5py.File(out, "r") as f:
        a = dict(f.attrs)
    s = lambda v: [x.decode() if isinstance(x, bytes) else str(x) for x in v]
    assert a["cumulative_mixture"]
    assert a["detection_policy"] == "lvk-cumulative"
    assert s(a["run_labels"]) == list(RUN_LABELS)
    assert a["n_rows_per_run"].tolist() == [(run == r).sum() for r in RUN_LABELS]
    assert a["n_detected_per_run"].tolist() == [
        int((det & (run == r)).sum()) for r in RUN_LABELS]
    assert int(a["n_detected"]) == int(det.sum())
    rules = s(a["detection_rule_per_run"])
    assert rules[:2] == [f"{SNR_COLUMN}>10"] * 2
    assert rules[2] == rules[3] and rules[2].startswith("min(o3_cwb_far,")
    assert rules[4].startswith("min(o4a_") and rules[5].startswith("min(o4b_")
    assert rules[5].endswith(")<1")
    np.testing.assert_array_equal(
        a["injection_support_gps_per_run"],
        [INJECTION_SUPPORT_GPS[r] for r in RUN_LABELS])
    np.testing.assert_array_equal(
        a["exposure_windows_gps_per_run"],
        [EXPOSURE_WINDOWS_GPS[r] for r in RUN_LABELS])
    assert s(a["T_definition_per_run"]) == [
        "coincident_livetime_semianalytic", "coincident_livetime_semianalytic",
        "endo3_analysis_time", "monthly_wall_clock"]
    assert s(a["mixture_components"]) == ["O1", "O2", "O3", "O4"]
    np.testing.assert_allclose(a["N_per_run"],
                               [_N[c] for c in ("O1", "O2", "O3", "O4")],
                               rtol=1e-12)
    np.testing.assert_allclose(a["T_per_run_s"],
                               [_T[c] for c in ("O1", "O2", "O3", "O4")],
                               rtol=1e-12)
    assert np.all(np.abs(a["N_per_run_integrality_residual"]) < 1e-6)
    assert a["mixture_bookkeeping_status"] == "derived_from_weights_and_anchors"
    w = json.loads(a["mixture_weights_per_run"])
    assert len(w["O4b"]) == 2 and len(w["O1"]) == 1
    np.testing.assert_allclose(
        a["z_draw_max_per_run"],
        [data["redshift"][run == r].max() for r in RUN_LABELS])
    assert a["o3_draw_density_source"] == "mixture_joint_lnpdraw"
    assert not a["cosmology_override_used"]
    assert a["injected_spin_uniform_isotropic"].tolist() == [False]
    assert s(a["campaign_kind_per_campaign"]) == ["cumulative_mixture"]


def test_bookkeeping_without_anchor_is_nan_and_warns(tmp_path):
    path, _, _ = write_cumulative_mixture(tmp_path / "mix.hdf")
    with pytest.warns(UserWarning, match="no O3/O4 component anchors"):
        p = build_selection_product(SelectionSet(path),
                                    spin_basis="component",
                                    snr_threshold=10.0)
    assert np.all(np.isnan(p.attrs["N_per_run"]))
    assert p.attrs["mixture_bookkeeping_status"] == "unavailable_no_anchor"


def test_wrong_anchor_is_refused(tmp_path):
    path, _, _ = write_cumulative_mixture(tmp_path / "mix.hdf")
    bad_o3 = {"O3": dict(_ANCHORS["O3"], N=_N["O3"] + 7), "O4": _ANCHORS["O4"]}
    with pytest.raises(MixtureInvariantError, match="O3 anchor"):
        _build(path, mixture_anchors=bad_o3)
    bad_o4 = {"O3": _ANCHORS["O3"],
              "O4": dict(_ANCHORS["O4"], T_s=_T["O4"] + 777.7)}
    with pytest.raises(MixtureInvariantError, match="not positive integers"):
        _build(path, mixture_anchors=bad_o4)


def test_derive_bookkeeping_matches_release_numbers():
    """The release's own weights and anchors give integer N_O1, N_O2."""
    run = np.array(["O1", "O2", "O3a", "O3b", "O4a"])
    w = np.array([0.3595637562455014, 0.3955440252726014, 5.43372670317606,
                  5.43372670317606, 0.9])
    from gwcat.observing_runs import MIXTURE_RELEASE_ANCHORS
    anchors = MIXTURE_RELEASE_ANCHORS[(1568035640, 96114016)]
    book = derive_mixture_bookkeeping(run, w, 1568035640, 96114016.0, anchors)
    np.testing.assert_allclose(book["N"][:2], [189781582, 421480890],
                               atol=1e-5)
    np.testing.assert_allclose(book["T_s"][:2], [4182739, 10218885], atol=1e-3)
    assert sum(book["N"]) == pytest.approx(1568035640, abs=1e-4)
    assert sum(book["T_s"]) == pytest.approx(96114016, abs=1e-4)


# ======================================================================
# (e) mixture + overlapping campaign
# ======================================================================
def _add_gps_time(path, t):
    with h5py.File(path, "r+") as f:
        f["injections"].create_dataset("gps_time", data=np.asarray(t))
    return path


def test_mixture_plus_overlapping_endo3_raises(tmp_path):
    path, _, _ = write_cumulative_mixture(tmp_path / "mix.hdf")
    endo3 = write_endo3_full(tmp_path / "endo3.hdf", n=30)
    lo, hi = INJECTION_SUPPORT_GPS["O3a"]
    _add_gps_time(endo3, np.linspace(lo, hi, 30))
    with pytest.raises(OverlappingCampaignError, match="double-counts"):
        build_selection_product([SelectionSet(path), SelectionSet(endo3)],
                                spin_basis="component", snr_threshold=10.0,
                                mixture_anchors=_ANCHORS)


def test_mixture_plus_campaign_of_unknown_range_raises(tmp_path):
    path, _, _ = write_cumulative_mixture(tmp_path / "mix.hdf")
    endo3 = write_endo3_full(tmp_path / "endo3.hdf", n=30)     # no times
    with pytest.raises(OverlappingCampaignError, match="unknown"):
        build_selection_product([SelectionSet(path), SelectionSet(endo3)],
                                spin_basis="component", snr_threshold=10.0,
                                mixture_anchors=_ANCHORS)


def test_mixture_plus_disjoint_campaign_is_allowed(tmp_path):
    """The refusal is about overlap, not about combining as such."""
    path, _, _ = write_cumulative_mixture(tmp_path / "mix.hdf")
    later = write_o4_full(tmp_path / "later.hdf", n=20)
    with h5py.File(later, "r+") as f:
        f["events"].create_dataset("time_geocenter",
                                   data=np.linspace(1.5e9, 1.51e9, 20))
        # snr_threshold applies to every campaign (unchanged behaviour), so
        # the single campaign carries the (all-zero) SNR column too.
        f["events"].create_dataset(SNR_COLUMN, data=np.zeros(20))
    p = build_selection_product([SelectionSet(path), SelectionSet(later)],
                                spin_basis="component", snr_threshold=10.0,
                                mixture_anchors=_ANCHORS)
    assert p.attrs["n_campaigns"] == 2 and p.attrs["cumulative_mixture"]


# ======================================================================
# (f) pdraw formula per row, and a toy pdet against the README estimator
# ======================================================================
def test_pdraw_per_row_and_toy_pdet_match_readme(tmp_path):
    path, data, run = write_cumulative_mixture(tmp_path / "mix.hdf")
    out = tmp_path / "sel.h5"
    SelectionSet(path).export(str(out), spin_basis="component",
                              snr_threshold=10.0,
                              detection_policy="lvk-cumulative",
                              mixture_anchors=_ANCHORS)
    sel = _readme_mask(data)
    N = float(sum(_N.values()))
    T_yr = sum(_T.values()) / _YEAR_S
    m1s, m2s, z = data["mass1_source"], data["mass2_source"], data["redshift"]
    a1 = np.sqrt(data["spin1x"] ** 2 + data["spin1y"] ** 2
                 + data["spin1z"] ** 2)
    a2 = np.sqrt(data["spin2x"] ** 2 + data["spin2y"] ** 2
                 + data["spin2z"] ** 2)
    ddL, w, lnp_draw = (data["dluminosity_distance_dredshift"],
                        data["weights"], data[_JOINT])
    m1det = m1s * (1 + z)
    # (m1s, m2s, z) -> (m1det, q, dL): |J| = (1+z) ddL / m1s; cartesian spin
    # -> (a, cos t) with azimuth marginalised: x 2 pi a^2 per body.
    jac = m1det / (1 + z) ** 2 / ddL * (2 * np.pi) ** 2 * a1 ** 2 * a2 ** 2
    want = np.exp(lnp_draw) * jac / (w * T_yr)
    with h5py.File(out, "r") as f:
        pdraw = f["pdraw"][:]
        np.testing.assert_array_equal(f["m1src"][:], m1s[sel])
        ndraw = float(f.attrs["ndraw"])
    np.testing.assert_allclose(pdraw, want[sel], rtol=1e-12)

    # A toy population in the README's own coordinates (source masses,
    # redshift, cartesian spins; azimuth-independent) ...
    amax = 0.999
    lnp_pop = (-0.5 * ((m1s - 35.0) / 8.0) ** 2 - 0.5 * ((z - 0.4) / 0.3) ** 2
               - np.log(4 * np.pi * a1 ** 2 * amax)
               - np.log(4 * np.pi * a2 ** 2 * amax))
    pdet_readme = np.sum(w[sel] * np.exp(lnp_pop[sel] - lnp_draw[sel])) / N
    # ... and the same population in the export's coordinates.
    p_pop_export = np.exp(lnp_pop) * jac
    pdet_gwcat = np.sum(p_pop_export[sel] / pdraw) / ndraw / T_yr
    assert pdet_gwcat == pytest.approx(pdet_readme, rel=1e-12)


# ======================================================================
# (g) reference coverage per run
# ======================================================================
def test_reference_coverage_recorded_per_run(tmp_path):
    # O3 draws stop at 0.95 < a_ref, while the file as a whole reaches 0.999:
    # the per-campaign check passes and only the per-run check sees the hole.
    path, _, _ = write_cumulative_mixture(tmp_path / "mix.hdf",
                                          amax={"O3a": 0.95, "O3b": 0.95})
    with pytest.warns(UserWarning, match=r"run\(s\) O3a"):
        p = _build(path, spin_basis="chieff_reference",
                   spin_reference_amax=0.99)
    a = p.attrs
    assert a["spin_reference_coverage_per_campaign"].tolist() == [True]
    assert [str(x) for x in a["spin_reference_coverage_runs"]] == \
        list(RUN_LABELS)
    assert a["spin_reference_coverage_per_run"].tolist() == [
        True, True, False, False, True, True]
    np.testing.assert_allclose(a["spin_reference_coverage_bound_per_run"][2:4],
                               0.95)
    assert a["spin_reference_coverage_ok"] is False

    good, _, _ = write_cumulative_mixture(tmp_path / "good.hdf")
    p = _build(good, spin_basis="chieff_reference", spin_reference_amax=0.99)
    assert p.attrs["spin_reference_coverage_per_run"].tolist() == [True] * 6
    assert p.attrs["spin_reference_coverage_ok"] is True


def test_chieff_reference_pdraw_and_sentinels_on_mixture(tmp_path):
    path, data, run = write_cumulative_mixture(tmp_path / "mix.hdf")
    comp = _build(path, spin_basis="component")
    ref = _build(path, spin_basis="chieff_reference", spin_reference_amax=0.99)
    c = comp.columns
    lnp, sup = chi_eff_prior_logprob_in_support(
        c["chieff"], c["m1src"], c["m2src"], amax=0.99)
    inref = sup & (c["a1"] <= 0.99) & (c["a2"] <= 0.99)
    want = c["pdraw"] * np.exp(lnp + np.log(4 * 0.99 ** 2))
    np.testing.assert_allclose(ref.columns["pdraw"][inref], want[inref],
                               rtol=1e-12)
    assert np.all(ref.columns["pdraw"][~inref] == 1e300)
    assert ref.attrs["spin_reference_excluded_rows"] == int((~inref).sum())


# ======================================================================
# (h) --sky-marginal
# ======================================================================
def test_sky_marginal_writes_no_radec_and_validates(tmp_path):
    path, _, _ = write_cumulative_mixture(tmp_path / "mix.hdf")
    out = tmp_path / "sel.h5"
    # The CLI has no anchor argument, so this also exercises the NaN path.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = main(["export", "selection", path, "--out", str(out),
                   "--parameter-space", "component",
                   "--detection-policy", "lvk-cumulative",
                   "--snr-threshold", "10", "--sky-marginal", "--no-summary"])
    assert rc == 0
    with h5py.File(out, "r") as f:
        assert "ra" not in f and "dec" not in f
        assert bool(f.attrs["sky_marginalized"]) is True
        assert f.attrs["detection_policy"] == "lvk-cumulative"
    # Without the flag the (NaN) sky columns are still written, as before.
    p = _build(path)
    assert "ra" in p.columns and p.attrs["sky_marginalized"] is False

    # The validator accepts the missing ra/dec only because the file says so.
    from test_validate_v2 import _pe_component
    pe = _pe_component(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = validate_export_v2(str(pe), str(out))
    assert results["sel_sky_marginalized_has_no_radec"] is True
    assert "sel_has_ra" not in results
    assert results["sel_mixture_n_detected_per_run_sum"] is True
    assert all(v for k, v in results.items() if k.startswith("sel_")), \
        [k for k, v in results.items() if k.startswith("sel_") and not v]


def test_sky_marginal_refused_by_the_21_writer(tmp_path):
    path, _, _ = write_cumulative_mixture(tmp_path / "mix.hdf")
    with pytest.raises(ValueError, match="2.1"):
        SelectionSet(path).export(str(tmp_path / "x.h5"), format="gwcat2.1",
                                  spin_basis="component", snr_threshold=10.0,
                                  sky_marginal=True, mixture_anchors=_ANCHORS)


# ======================================================================
# (i) chieff / chieff_chip on a joint-cartesian file
# ======================================================================
def test_chieff_on_joint_cartesian_raises(tmp_path):
    path, _, _ = write_cumulative_mixture(tmp_path / "mix.hdf")
    with pytest.raises(BlockCampaignMismatch, match="JOINT"):
        _build(path, spin_basis="chieff")
    with pytest.raises(SpinBasisError):
        _build(path, spin_basis="chieff_chip")
    # strict=False lets it through, and the file says so about itself.
    with pytest.warns(UserWarning, match="JOINT"):
        p = _build(path, spin_basis="chieff", strict=False)
    assert p.attrs["spin_basis_assumption_unverified"] is True
    viol = json.loads(p.attrs["spin_basis_assumption_violations"])
    assert viol and viol[0]["spin_format"] == "joint_cartesian"


def test_chieff_on_plain_joint_cartesian_file_raises(tmp_path):
    """The refusal is about the joint density, not only the mixture."""
    from test_selection_spin import write_mixture
    car = write_mixture(tmp_path / "joint.hdf", "cartesian", seed=3)
    with pytest.raises(BlockCampaignMismatch, match="JOINT"):
        build_selection_product(SelectionSet(car), spin_basis="chieff")


# ======================================================================
# Data-gated integration test on the released files
# ======================================================================
_REAL = os.environ.get("GWCAT_REAL_DATA") == "1"


def _real_path(var):
    p = os.environ.get(var)
    if not p or not os.path.exists(p):
        pytest.fail(f"GWCAT_REAL_DATA=1 but ${var} does not name an existing "
                    f"file (got {p!r}).")
    return p


@pytest.fixture(scope="module")
def real_mixture():
    s = SelectionSet(_real_path("GWCAT_MIXTURE_CARTESIAN"))
    s._load()
    return s


@pytest.mark.skipif(not _REAL, reason="set GWCAT_REAL_DATA=1 and "
                    "GWCAT_MIXTURE_CARTESIAN / GWCAT_RPO4AB / "
                    "GWCAT_ENDO3_MIXTURE to run on the released files")
class TestRealCumulativeMixture:
    """GWTC-5.0 cumulative O1-O4b (Zenodo 19500052), cartesian flavour."""

    def test_identity_and_per_run_counts(self, real_mixture):
        s = real_mixture
        assert s.campaign_kind == "cumulative_mixture"
        assert s.n_generated == 1568035640
        assert s._total_analysis_time_s == 96114016.0
        det = s.detected_mask(1.0, snr_threshold=10.0)
        run = s.run
        got = {r: int((det & (run == r)).sum()) for r in RUN_LABELS}
        assert got == {"O1": 109093, "O2": 260902, "O3a": 115779,
                       "O3b": 105545, "O4a": 482163, "O4b": 504666}
        assert int(det.sum()) == 1578148

    def test_bookkeeping_and_rpo4ab_attrs(self, real_mixture):
        with h5py.File(_real_path("GWCAT_RPO4AB"), "r") as f:
            T_rpo = float(f.attrs["total_analysis_time"])
            N_rpo = int(f.attrs["total_generated"])
        with h5py.File(_real_path("GWCAT_ENDO3_MIXTURE"), "r") as f:
            T_e3 = float(f.attrs["analysis_time_s"])
            N_e3 = int(f.attrs["total_generated"])
        p = build_selection_product(real_mixture,
                                    spin_basis="chieff_reference",
                                    spin_reference_amax=0.99,
                                    detection_policy="lvk-cumulative",
                                    snr_threshold=10.0, sky_marginal=True)
        a = p.attrs
        assert a["mixture_bookkeeping_status"] == \
            "derived_from_weights_and_anchors"
        assert a["T_per_run_s"][3] == T_rpo and a["N_per_run"][3] == N_rpo
        assert a["T_per_run_s"][2] == T_e3 and a["N_per_run"][2] == N_e3
        assert np.all(np.abs(a["N_per_run_integrality_residual"]) < 1e-6)
        assert sum(a["N_per_run"]) == pytest.approx(1568035640, abs=1e-3)
        assert sum(a["T_per_run_s"]) == pytest.approx(96114016.0, abs=1e-3)
        assert a["n_detected"] == 1578148
        assert a["ndraw"] == 1568035640
        assert a["spin_reference_excluded_rows"] == 9120
        assert a["spin_reference_coverage_ok"] is True
        assert a["injected_spin_uniform_isotropic"].tolist() == [False]
        assert "ra" not in p.columns

    def test_o4_rows_against_rpo4ab(self, real_mixture):
        s = real_mixture
        r = SelectionSet(_real_path("GWCAT_RPO4AB"),
                         strict_spin_checks="off")
        r._load()
        o4 = np.isin(s.run, ["O4a", "O4b"])
        t_s5 = np.asarray(s._time)[o4]
        t_rpo = np.asarray(r._time)
        order = np.argsort(t_rpo)
        ts = t_rpo[order]
        assert np.unique(ts).size == ts.size
        pos = np.searchsorted(ts, t_s5)
        pos = np.clip(pos, 0, ts.size - 1)
        assert np.array_equal(ts[pos], t_s5), "S5 O4 rows missing in rpo4ab"
        idx = order[pos]
        assert idx.size == 1585620
        np.testing.assert_array_equal(np.asarray(r._m1src)[idx],
                                      np.asarray(s._m1src)[o4])
        lr = (np.log(s.component_pdraw()[o4])
              - np.log(r.component_pdraw()[idx]))
        const = np.log(r.n_generated / s.n_generated)
        # Equivalently ln(T_rpo/T_S5) - ln(w_S5/w_rpo), from the attrs alone.
        assert abs(np.mean(lr) - const) <= 1e-9
        assert np.std(lr) < 1e-10

    def test_o3_rows_equal_endo3_mixture(self, real_mixture):
        s = real_mixture
        path = _real_path("GWCAT_ENDO3_MIXTURE")
        with h5py.File(path, "r") as f:
            inj = f["injections"]
            t_e3 = inj["gps_time"][:]
            ln_e3 = np.log(inj["sampling_pdf"][:])
            m1_e3 = inj["mass1_source"][:]
            far_e3 = np.min([inj[k][:] for k in
                             ("far_cwb", "far_gstlal", "far_mbta",
                              "far_pycbc_bbh", "far_pycbc_hyperbank")],
                            axis=0)
        with h5py.File(s.path, "r") as f:
            ln_s5 = f["events"].fields([_JOINT])[:][_JOINT]
        o3 = np.isin(s.run, ["O3a", "O3b"])
        t_s5 = np.asarray(s._time)[o3]
        order = np.argsort(t_e3)
        ts = t_e3[order]
        pos = np.clip(np.searchsorted(ts, t_s5), 0, ts.size - 1)
        assert np.array_equal(ts[pos], t_s5), "S5 O3 rows missing in endo3"
        idx = order[pos]
        assert idx.size == 326028
        np.testing.assert_array_equal(m1_e3[idx], np.asarray(s._m1src)[o3])
        assert np.max(np.abs(ln_s5[o3] - ln_e3[idx])) <= 1e-12
        det_s5 = s.detected_mask(1.0, snr_threshold=10.0)[o3]
        np.testing.assert_array_equal(det_s5, far_e3[idx] < 1.0)
        assert int(det_s5.sum()) == 221324
