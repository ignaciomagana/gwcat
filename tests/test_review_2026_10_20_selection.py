"""Review 2026-10-20, selection package: regression tests from the findings'
failure scenarios (gwcat/selection.py, gwcat/selection_spin.py,
gwcat/observing_runs.py).

Each test is written from the review's failure scenario and fails on 3ba1dbf.
Synthetic fixtures only (the ``tests/test_selection_spin.py`` writers); no
real LVK file is read.
"""
import json
import warnings

import h5py
import numpy as np
import pytest

from gwcat.selection import SelectionSet, CombinedSelectionSet

from test_selection_spin import write_o4_full, write_endo3_full


def _declare_domain(path, **attrs):
    """Stamp endo3-style draw-domain attrs on a fixture's root."""
    with h5py.File(path, "r+") as f:
        for k, v in attrs.items():
            f.attrs[k] = v
    return path


# ======================================================================
# high-4: the O3 injection campaign's declared draw domain
# ======================================================================
def _o3_o4_pair(tmp_path):
    # The O3-like campaign declares m1, m2 <= 40 and z <= 0.3 (endo3 declares
    # 100 and 1.9); the O4-like one declares nothing and draws m1 to 45 and z
    # to 0.5, i.e. past both bounds.
    o3 = _declare_domain(write_endo3_full(tmp_path / "o3.hdf", n=400, seed=3),
                         min_mass1=2.0, max_mass1=40.0, min_mass2=2.0,
                         max_mass2=40.0, max_redshift=0.3, max_z=0.3)
    o4 = write_o4_full(tmp_path / "o4.hdf", n=300, seed=4)
    return o3, o4


def test_declared_draw_domain_is_read_and_recorded_per_campaign(tmp_path):
    o3, o4 = _o3_o4_pair(tmp_path)
    s3, s4 = SelectionSet(o3), SelectionSet(o4)
    assert s3.declared_draw_domain == {"mass1_source": [2.0, 40.0],
                                       "mass2_source": [2.0, 40.0],
                                       "redshift": [None, 0.3]}
    assert s4.declared_draw_domain == {"mass1_source": [None, None],
                                       "mass2_source": [None, None],
                                       "redshift": [None, None]}
    assert s4.draw_extent["mass1_source"][1] > 40.0

    out = tmp_path / "sel.h5"
    with pytest.warns(UserWarning, match="declared injection draw domains"):
        CombinedSelectionSet([s3, s4]).export(str(out),
                                              spin_basis="component")
    with h5py.File(out, "r") as f:
        rows = json.loads(f.attrs["injection_draw_domain_per_campaign"])
        assert rows[0]["declared"]["mass1_source"] == [2.0, 40.0]
        assert rows[0]["declared_source"] == "file_attrs"
        assert rows[1]["declared_source"] == "undeclared"
        assert bool(f.attrs["injection_draw_domain_mismatch"]) is True
        detail = json.loads(f.attrs["injection_draw_domain_mismatch_detail"])
        crossed = {(d["campaign"], d["quantity"], d["bound"]) for d in detail}
        assert (0, "mass1_source", "max") in crossed
        assert (0, "redshift", "max") in crossed
        assert f.attrs["injection_draw_domain_policy"] == "not_applied"
        assert json.loads(f.attrs["injection_draw_domain_applied"]) == {}
        # Default: nothing subset -- every detected row of both campaigns.
        assert int(f.attrs["n_detected"]) == 400 + 300


def test_draw_domain_intersection_subsets_and_keeps_ndraw(tmp_path):
    o3, o4 = _o3_o4_pair(tmp_path)
    s3, s4 = SelectionSet(o3), SelectionSet(o4)
    out = tmp_path / "sel_dom.h5"
    with pytest.warns(UserWarning, match="draw_domain='intersection'"):
        CombinedSelectionSet([s3, s4]).export(
            str(out), spin_basis="component", draw_domain="intersection")

    def inside(s):
        return ((s._m1src >= 2.0) & (s._m1src <= 40.0) & (s._m2src >= 2.0)
                & (s._m2src <= 40.0) & (s._z <= 0.3))

    n_in = int(inside(s3).sum() + inside(s4).sum())
    assert 0 < n_in < 700
    with h5py.File(out, "r") as f:
        assert int(f.attrs["n_detected"]) == n_in
        assert f["pdraw"].shape == (n_in,)
        assert float(np.max(f["m1src"][:])) <= 40.0
        assert float(np.max(f["redshift"][:])) <= 0.3
        # Subsetting, not reweighting: ndraw is the campaigns' total draws.
        assert int(f.attrs["ndraw"]) == s3.n_generated + s4.n_generated
        assert f.attrs["injection_draw_domain_policy"] == "intersection"
        assert json.loads(f.attrs["injection_draw_domain_applied"]) == {
            "mass1_source": [2.0, 40.0], "mass2_source": [2.0, 40.0],
            "redshift": [None, 0.3]}


def test_draw_domain_intersection_refused_without_any_declaration(tmp_path):
    o4 = write_o4_full(tmp_path / "o4.hdf", n=50, seed=5)
    with pytest.raises(ValueError, match="declares a draw domain"):
        SelectionSet(o4).export(str(tmp_path / "x.h5"),
                                spin_basis="component",
                                draw_domain="intersection")


def test_v1_combined_export_records_the_draw_domains(tmp_path):
    o3, o4 = _o3_o4_pair(tmp_path)
    out = tmp_path / "v1.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        CombinedSelectionSet([SelectionSet(o3), SelectionSet(o4)]).to_darksirens(
            str(out), amax=0.999, strict=False)
    with h5py.File(out, "r") as f:
        assert bool(f.attrs["injection_draw_domain_mismatch"]) is True
        rows = json.loads(f.attrs["injection_draw_domain_per_campaign"])
        assert [r["declared_source"] for r in rows] == ["file_attrs",
                                                        "undeclared"]


# ======================================================================
# medium-14: the v1 chi_eff-swap exporters record what strict=False let in
# ======================================================================
def _core_spin_swap_problems(attrs):
    """darksirens-core's _require_valid_spin_swap, on the attrs alone."""
    problems = []
    if "injected_spin_uniform_isotropic" in attrs:
        if not np.atleast_1d(attrs["injected_spin_uniform_isotropic"]).all():
            problems.append("not uniform-isotropic")
    v = attrs.get("spin_basis_assumption_violations", "")
    if v and v not in ("[]", "{}"):
        problems.append("violations")
    return problems


def test_v1_strict_false_records_the_swap_violation(tmp_path):
    from test_selection_chieff_reference import write_o4_nonuniform
    path = write_o4_nonuniform(tmp_path / "nonuni.hdf", n=200, seed=3)
    out = tmp_path / "v1.h5"
    with pytest.warns(UserWarning, match="did NOT draw spins that way"):
        SelectionSet(path, strict_spin_checks="off").to_darksirens(
            str(out), amax=1.0, strict=False)
    with h5py.File(out, "r") as f:
        attrs = dict(f.attrs)
    assert list(attrs["injected_spin_uniform_isotropic"]) == [False]
    viol = json.loads(attrs["spin_basis_assumption_violations"])
    assert len(viol) == 1 and viol[0]["verified"] is True
    assert bool(attrs["spin_basis_assumption_violated"]) is True
    # core's only spin-swap guard now refuses the file.
    assert _core_spin_swap_problems(attrs)


def test_v1_valid_swap_records_a_clean_gate(tmp_path):
    o3 = write_endo3_full(tmp_path / "o3.hdf", n=60, seed=8)
    o4 = write_o4_full(tmp_path / "o4.hdf", n=60, seed=9)
    out = tmp_path / "v1c.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        CombinedSelectionSet([SelectionSet(o3), SelectionSet(o4)]).to_darksirens(
            str(out))
    with h5py.File(out, "r") as f:
        attrs = dict(f.attrs)
    assert list(attrs["injected_spin_uniform_isotropic"]) == [True, True]
    assert attrs["spin_basis_assumption_violations"] == "[]"
    assert not _core_spin_swap_problems(attrs)


# ======================================================================
# medium-15 / medium-16: the spin-ceiling detectors
# ======================================================================
def _rewrite_endo3_spins(path, a1, a2, p_a1, p_a2, seed=0):
    """Give an endo3 fixture isotropic spins with magnitudes ``a_i`` drawn
    from ``p(a_i)``, and the per-spin cartesian pdfs / joint sampling_pdf of
    exactly that draw: ``p_cart = p(a) / (4 pi a^2)``."""
    rng = np.random.default_rng(seed)
    with h5py.File(path, "r+") as f:
        g = f["injections"]
        n = len(g["mass1_source"])
        cols = {}
        for i, (a, pa) in enumerate(((a1, p_a1), (a2, p_a2)), start=1):
            cost = rng.uniform(-1.0, 1.0, n)
            st = np.sqrt(1.0 - cost ** 2)
            ph = rng.uniform(0.0, 2 * np.pi, n)
            cols[f"spin{i}x"] = a * st * np.cos(ph)
            cols[f"spin{i}y"] = a * st * np.sin(ph)
            cols[f"spin{i}z"] = a * cost
            cols[f"spin{i}x_spin{i}y_spin{i}z_sampling_pdf"] = (
                pa / (4.0 * np.pi * a ** 2))
        cols["sampling_pdf"] = (
            g["mass1_source_mass2_source_sampling_pdf"][:]
            * g["redshift_sampling_pdf"][:]
            * cols["spin1x_spin1y_spin1z_sampling_pdf"]
            * cols["spin2x_spin2y_spin2z_sampling_pdf"])
        for k, v in cols.items():
            del g[k]
            g.create_dataset(k, data=v)
    return path


def _mixture_mags(rng, n):
    """90% U(0, 0.95) + 10% U(0.40, 0.42), and its density."""
    pick = rng.uniform(size=n) < 0.9
    a = np.where(pick, rng.uniform(0.0, 0.95, n), rng.uniform(0.40, 0.42, n))
    p = 0.9 / 0.95 + np.where((a >= 0.40) & (a <= 0.42), 0.1 / 0.02, 0.0)
    return a, p


def test_cartesian_detector_gives_no_ceiling_for_a_non_uniform_draw(tmp_path):
    rng = np.random.default_rng(21)
    path = write_endo3_full(tmp_path / "o3mix.hdf", n=600, seed=21)
    (a1, p1), (a2, p2) = _mixture_mags(rng, 600), _mixture_mags(rng, 600)
    _rewrite_endo3_spins(path, a1, a2, p1, p2)
    s = SelectionSet(path, strict_spin_checks="off")
    meta = s.spin_meta
    assert meta["amax_detected"] == (None, None)        # was 1.056
    assert meta["uniform_isotropic"] is False
    # The chieff_reference coverage check no longer takes a fabricated
    # ceiling above 1 as proof that magnitudes reach a_ref = 0.99: nothing
    # above 0.95 was drawn, and the file says so.
    out = tmp_path / "ref.h5"
    with pytest.warns(UserWarning, match="sample maximum"):
        s.export(str(out), spin_basis="chieff_reference",
                 spin_reference_amax=0.99)
    with h5py.File(out, "r") as f:
        assert bool(f.attrs["spin_reference_coverage_ok"]) is False
        src = [x.decode() if isinstance(x, bytes) else x
               for x in f.attrs["spin_reference_coverage_source_per_campaign"]]
        assert src == ["drawn_max"]


def test_cartesian_detector_does_not_refuse_a_covering_non_uniform_draw(
        tmp_path):
    # p(a) = 2a on [0, 1]: the old detector called its ceiling 0.71 and a
    # strict a_ref = 0.99 export was refused as a proven coverage hole.
    rng = np.random.default_rng(22)
    path = write_endo3_full(tmp_path / "o3lin.hdf", n=600, seed=22)
    a1, a2 = np.sqrt(rng.uniform(size=600)), np.sqrt(rng.uniform(size=600))
    _rewrite_endo3_spins(path, a1, a2, 2 * a1, 2 * a2)
    s = SelectionSet(path, strict_spin_checks="off")
    assert s.spin_meta["amax_detected"] == (None, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s.export(str(tmp_path / "ref.h5"), spin_basis="chieff_reference",
                 spin_reference_amax=0.99)


def test_constant_density_not_from_zero_is_not_a_ceiling_cartesian(tmp_path):
    # a ~ U(0.05, 0.99): constant density 1/0.94, but no draw below 0.05.
    rng = np.random.default_rng(23)
    path = write_endo3_full(tmp_path / "o3shift.hdf", n=600, seed=23)
    a1, a2 = rng.uniform(0.05, 0.99, 600), rng.uniform(0.05, 0.99, 600)
    p = np.full(600, 1.0 / 0.94)
    _rewrite_endo3_spins(path, a1, a2, p, p)
    s = SelectionSet(path, strict_spin_checks="off")
    assert s.spin_meta["amax_detected"] == (None, None)  # was (0.94, 0.94)
    assert s.spin_meta["uniform_isotropic"] is False
    from gwcat.export.selection_builder import BlockCampaignMismatch
    with pytest.raises(BlockCampaignMismatch):
        s.export(str(tmp_path / "chieff.h5"), spin_basis="chieff")


def test_constant_density_not_from_zero_is_not_a_ceiling_factored(tmp_path):
    path = write_o4_full(tmp_path / "o4shift.hdf", n=600, seed=24)
    rng = np.random.default_rng(24)
    with h5py.File(path, "r+") as f:
        g = f["events"]
        for i in (1, 2):
            a = rng.uniform(0.05, 0.99, 600)
            th = g[f"spin{i}_polar_angle"][:]
            ph = g[f"spin{i}_azimuthal_angle"][:]
            new = {f"spin{i}_magnitude": a,
                   f"spin{i}x": a * np.sin(th) * np.cos(ph),
                   f"spin{i}y": a * np.sin(th) * np.sin(ph),
                   f"spin{i}z": a * np.cos(th),
                   f"lnpdraw_spin{i}_magnitude": np.full(600, -np.log(0.94))}
            for k, v in new.items():
                del g[k]
                g.create_dataset(k, data=v)
    s = SelectionSet(path, strict_spin_checks="off")
    assert s.spin_meta["amax_detected"] == (None, None)  # was (0.94, 0.94)
    assert s.spin_meta["uniform_isotropic"] is False


def test_detectors_keep_a_genuine_uniform_ceiling():
    from gwcat import selection_spin as sspin
    rng = np.random.default_rng(25)
    a = rng.uniform(0.0, 0.998, 5000)
    amax, ok = sspin.detect_max_spin_cart(1.0 / (4 * np.pi * a ** 2 * 0.998), a)
    assert ok and np.isclose(amax, 0.998, rtol=1e-12)
    amax, ok = sspin.detect_uniform_amax_from_lnmag(
        np.full(5000, -np.log(0.998)), a=a)
    assert ok and np.isclose(amax, 0.998, rtol=1e-12)
    # A "ceiling" above 1 is never physical.
    amax, ok = sspin.detect_max_spin_cart(
        1.0 / (4 * np.pi * a ** 2 * 1.2), a * 1.2 / 0.998)
    assert (amax, ok) == (None, False)


# ======================================================================
# medium-31 / low-36: FAR-column discovery
# ======================================================================
def _events_far_file(path, far_cols, searches=None, compound=True):
    """A minimal events-format file with the given FAR columns (dict)."""
    n = len(next(iter(far_cols.values())))
    base = {"mass1_source": np.full(n, 30.0), "mass2_source": np.full(n, 20.0),
            "z": np.full(n, 0.1), "mass1_detector": np.full(n, 33.0),
            "mass2_detector": np.full(n, 22.0),
            "luminosity_distance": np.full(n, 450.0),
            "dluminosity_distance_dredshift": np.full(n, 4500.0),
            "right_ascension": np.full(n, 1.0), "declination": np.full(n, 0.5),
            "spin1z": np.full(n, 0.2), "spin2z": np.full(n, -0.1),
            "chi_eff": np.full(n, 0.08), "weights": np.full(n, 2.0),
            "lnpdraw_mass1_source": np.full(n, -1.0),
            "lnpdraw_mass2_source_GIVEN_mass1_source": np.full(n, -2.0),
            "lnpdraw_z": np.full(n, -3.0)}
    for k in ("spin1x", "spin1y", "spin2x", "spin2y"):
        base[k] = np.zeros(n)
    cols = dict(base, **far_cols)
    with h5py.File(path, "w") as f:
        f.attrs["total_analysis_time"] = 365.25 * 24 * 3600
        f.attrs["total_generated"] = 100
        if searches is not None:
            f.attrs["searches"] = np.array(searches, dtype=h5py.string_dtype())
        if compound:
            dt = np.dtype([(k, np.asarray(v).dtype) for k, v in cols.items()])
            rec = np.zeros(n, dtype=dt)
            for k, v in cols.items():
                rec[k] = v
            f.create_dataset("events", data=rec)
        else:
            g = f.create_group("events")
            for k, v in cols.items():
                g.create_dataset(k, data=v)
    return str(path)


def test_declared_search_without_a_column_is_not_dropped_silently(tmp_path):
    # Row 1 is found ONLY by MBTA, whose column is spelled 'mbta-bbh_far'.
    path = _events_far_file(tmp_path / "far.hdf", {
        "pycbc_far": np.array([0.5, 5.0, 5.0]),
        "mbta-bbh_far": np.array([5.0, 0.2, 5.0])},
        searches=["pycbc", "mbta"])
    s = SelectionSet(path, strict_spin_checks="off")
    with pytest.warns(UserWarning, match=r"declares search\(es\) \['mbta'\]"):
        s._load()
    assert s._far_columns == ["pycbc_far"]
    out = tmp_path / "sel.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s.export(str(out), spin_basis="chieff", amax=0.99, strict=False)
    with h5py.File(out, "r") as f:
        disc = json.loads(f.attrs["significance_far_discovery_per_campaign"])
    assert disc[0]["unmatched_searches"] == ["mbta"]
    assert disc[0]["unclaimed_columns"] == ["mbta-bbh_far"]
    assert disc[0]["declared"] == ["pycbc", "mbta"]


def test_unreadable_far_column_is_raised_not_skipped(tmp_path):
    path = _events_far_file(tmp_path / "bad.hdf", {
        "pycbc_far": np.array([0.5, 5.0, 5.0]),
        "gstlal_far": np.array([b"x", b"0.1", b"y"])},
        searches=["pycbc", "gstlal"], compound=False)
    # h5py raises OSError (no conversion path) for this column; the old
    # loader caught it and thresholded on pycbc alone.
    with pytest.raises((ValueError, OSError)):
        SelectionSet(path, strict_spin_checks="off")._load()


def test_fallback_far_scan_order_does_not_depend_on_the_hash_seed(tmp_path):
    import os
    import subprocess
    import sys
    names = ["zz_far", "far_b", "mm_far", "far_a", "cc_far", "far_q"]
    path = _events_far_file(tmp_path / "scan.hdf",
                            {k: np.array([0.5, 5.0]) for k in names})
    code = ("import sys, h5py; from gwcat.selection import _events_far_columns;"
            "f = h5py.File(sys.argv[1], 'r');"
            "print(','.join(_events_far_columns(f['events'], f)))")
    seen = set()
    for seed in ("0", "1", "2", "3"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        out = subprocess.run([sys.executable, "-c", code, path], env=env,
                             capture_output=True, text=True, check=True)
        seen.add(out.stdout.strip())
    assert seen == {",".join(names)}          # file order, every seed


# ======================================================================
# medium-33: an explicit cosmology against the detected generation one
# ======================================================================
def _endo3_at_cosmology(path, cosmo, n=400, seed=31):
    """An endo3 fixture whose (z, dL) pairs were made with ``cosmo``."""
    write_endo3_full(path, n=n, seed=seed)
    z = np.sort(np.random.default_rng(seed).uniform(0.02, 1.5, n))
    with h5py.File(path, "r+") as f:
        g = f["injections"]
        m1s, m2s = g["mass1_source"][:], g["mass2_source"][:]
        for k, v in {"redshift": z,
                     "distance": cosmo.luminosity_distance(z).value,
                     "mass1": m1s * (1 + z), "mass2": m2s * (1 + z)}.items():
            del g[k]
            g.create_dataset(k, data=v)
    return str(path)


def test_override_against_a_detected_generation_cosmology_warns_and_records(
        tmp_path):
    from astropy.cosmology import FlatLambdaCDM
    path = _endo3_at_cosmology(tmp_path / "o3.hdf",
                               FlatLambdaCDM(H0=67.90, Om0=0.3065))
    s = SelectionSet(path, H0=67.74, Om0=0.3089, strict_spin_checks="off")
    with pytest.warns(UserWarning, match="differs from the generation"):
        s._load()
    assert s._cosmology_source == "override"
    assert np.isclose(s._cosmology_generation_H0, 67.90, rtol=1e-5)
    out = tmp_path / "sel.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s.export(str(out), spin_basis="component")
    with h5py.File(out, "r") as f:
        gen = np.asarray(f.attrs["cosmology_generation_H0_per_campaign"])
        used = np.asarray(f.attrs["cosmology_H0_per_campaign"])
    assert np.isclose(gen[0], 67.90, rtol=1e-5) and used[0] == 67.74


def test_no_override_records_the_detected_generation_cosmology(tmp_path):
    from astropy.cosmology import FlatLambdaCDM
    path = _endo3_at_cosmology(tmp_path / "o3.hdf",
                               FlatLambdaCDM(H0=67.90, Om0=0.3065))
    s = SelectionSet(path, strict_spin_checks="off")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        s._load()
    assert s._cosmology_source == "detected"
    assert np.isclose(s._cosmology_generation_Om0, 0.3065, rtol=1e-5)


# ======================================================================
# medium-56: the per-campaign 1/T_k with non-unit, unequal times
# ======================================================================
def _set_time(path, years, fmt):
    with h5py.File(path, "r+") as f:
        if fmt == "events":
            f.attrs["total_analysis_time"] = years * 365.25 * 24 * 3600
        else:
            f["injections"].attrs["analysis_time_s"] = (
                years * 365.25 * 24 * 3600)
    return path


@pytest.mark.parametrize("fmt", ["events", "injections"])
def test_pdraw_scales_as_one_over_each_campaigns_own_time(tmp_path, fmt):
    writer = write_o4_full if fmt == "events" else write_endo3_full
    a = writer(tmp_path / "a.hdf", n=80, seed=41)
    b = _set_time(writer(tmp_path / "b.hdf", n=80, seed=41), 0.911, fmt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sa, sb = SelectionSet(a), SelectionSet(b)
        sa._load(), sb._load()
    np.testing.assert_allclose(sb._pdraw / sa._pdraw, 1.0 / 0.911, rtol=1e-12)


def _component_density_from_file(path):
    """The o4_factored component draw density in (m1det, q, dL, a, cos),
    from the file's own columns (no gwcat call), and the row weights."""
    with h5py.File(path, "r") as f:
        g = f["events"]
        c = {k: g[k][:] for k in g}
    ln = (c["lnpdraw_mass1_source"] + c["lnpdraw_mass2_source_GIVEN_mass1_source"]
          + c["lnpdraw_z"])
    for i in (1, 2):
        ln = ln + (c[f"lnpdraw_spin{i}_magnitude"]
                   + c[f"lnpdraw_spin{i}_polar_angle"]
                   - np.log(np.sin(c[f"spin{i}_polar_angle"])))
    jac = c["mass1_detector"] / (1 + c["z"]) ** 2 / c[
        "dluminosity_distance_dredshift"]
    return np.exp(ln) * jac, c["weights"], c["pycbc_far"]


def test_combined_mu_is_the_sum_of_T_k_times_each_campaigns_efficiency(
        tmp_path):
    """Essick: mu = sum_k T_k P_det,k for the campaigns' common draw density,
    with unequal non-unit T_k, partial detection, and one campaign that
    detects nothing (skipped, but its N_k stays in ndraw)."""
    times, gens = (0.5, 2.25, 1.7), (900, 3000, 1500)
    paths = []
    for k, (T, N) in enumerate(zip(times, gens)):
        p = write_o4_full(tmp_path / f"c{k}.hdf", n=120, seed=50 + k,
                          total_generated=N)
        _set_time(p, T, "events")
        with h5py.File(p, "r+") as f:
            far = np.full(120, 5.0) if k == 2 else np.where(
                np.arange(120) % 3 == 0, 0.1, np.inf)
            del f["events"]["pycbc_far"]
            f["events"].create_dataset("pycbc_far", data=far)
        paths.append(p)
    out = tmp_path / "sel.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        CombinedSelectionSet([SelectionSet(p) for p in paths]).export(
            str(out), spin_basis="component")
    lam, expected = [], 0.0
    for p, T, N in zip(paths, times, gens):
        dens, w, far = _component_density_from_file(p)
        det = far < 1.0
        lam.append(dens[det])
        expected += T * float(np.sum(w[det])) / N
    lam = np.concatenate(lam)
    with h5py.File(out, "r") as f:
        ndraw = int(f.attrs["ndraw"])
        pdraw = f["pdraw"][:]
        np.testing.assert_array_equal(f.attrs["campaign_ndraws"], gens)
    assert ndraw == sum(gens)
    np.testing.assert_allclose(np.sum(lam / pdraw) / ndraw, expected,
                               rtol=1e-12)


# ======================================================================
# medium-57: the FAR cut is an OR over every pipeline, not the first one
# ======================================================================
def test_far_cut_ors_every_pipeline_events_format(tmp_path):
    inf, nan = np.inf, np.nan
    # Each row fires on ONE pipeline (never only the first), plus no-fire
    # rows with inf / NaN.
    cols = {"pycbc_far":   np.array([5.0, 5.0, 0.3, inf, nan, nan, 5.0]),
            "gstlal_far":  np.array([0.2, 5.0, inf, inf, 5.0, nan, nan]),
            "mbta_far":    np.array([inf, 0.9, 5.0, nan, inf, 0.5, 5.0]),
            "cwb-bbh_far": np.array([nan, inf, 5.0, 5.0, inf, 5.0, 0.99])}
    path = _events_far_file(tmp_path / "or.hdf", cols,
                            searches=["pycbc", "gstlal", "mbta", "cwb-bbh"])
    s = SelectionSet(path, strict_spin_checks="off")
    assert s.detected_mask(1.0).tolist() == [True, True, True, False, False,
                                             True, True]
    assert int(s.detected_mask(1.0).sum()) == 5
    # Every column alone detects something the first one does not.
    fars = s._fars
    for j in range(1, fars.shape[1]):
        assert np.any((fars[:, j] < 1.0) & ~(fars[:, 0] < 1.0))


def test_far_cut_ors_every_pipeline_injections_format(tmp_path):
    path = write_endo3_full(tmp_path / "o3.hdf", n=6, seed=61)
    inf, nan = np.inf, np.nan
    cols = {"far_gstlal":          np.array([5.0, inf, nan, 5.0, inf, 5.0]),
            "far_pycbc_bbh":       np.array([inf, 0.4, 5.0, nan, inf, 5.0]),
            "far_pycbc_hyperbank": np.array([5.0, 5.0, 0.7, inf, inf, 5.0]),
            "far_mbta":            np.array([nan, 5.0, 5.0, 0.1, inf, 5.0]),
            "far_cwb":             np.array([0.6, 5.0, inf, 5.0, nan, 5.0])}
    with h5py.File(path, "r+") as f:
        g = f["injections"]
        for k, v in cols.items():
            if k in g:
                del g[k]
            g.create_dataset(k, data=v)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = SelectionSet(path)
        mask = s.detected_mask(1.0)
    assert s._far_columns == ["far_gstlal", "far_pycbc_bbh",
                              "far_pycbc_hyperbank", "far_mbta", "far_cwb"]
    assert mask.tolist() == [True, True, True, True, False, False]


# ======================================================================
# low-5: the exact pdraw normalisation is stated, with each campaign's T_k
# ======================================================================
def test_pdraw_normalisation_and_campaign_times_are_recorded(tmp_path):
    a = _set_time(write_o4_full(tmp_path / "a.hdf", n=40, seed=71), 0.5,
                  "events")
    b = _set_time(write_endo3_full(tmp_path / "b.hdf", n=40, seed=72), 1.25,
                  "injections")
    out = tmp_path / "sel.h5"
    v1 = tmp_path / "v1.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        comb = CombinedSelectionSet([SelectionSet(a), SelectionSet(b)])
        comb.export(str(out), spin_basis="component")
        comb.to_darksirens(str(v1))
    for path in (out, v1):
        with h5py.File(path, "r") as f:
            text = f.attrs["pdraw_normalisation"]
            assert "N_k / N_total" in text and "ITS OWN analysis time" in text
            np.testing.assert_allclose(f.attrs["campaign_T_obs_yr"],
                                       [0.5, 1.25], rtol=1e-12)
            assert np.isclose(float(f.attrs["T_obs_yr"]), 1.75, rtol=1e-12)


# ======================================================================
# low-13: the spin-removal ceiling check can fire on a joint-format file
# ======================================================================
def test_removal_ceiling_check_fires_on_a_joint_file_drawn_past_it(tmp_path):
    from test_selection_spin import write_mixture
    path = write_mixture(tmp_path / "mix.hdf", "cartesian", n=3000, seed=81,
                         amax=(0.4, 0.998))
    s = SelectionSet(path, strict_spin_checks="off")
    with pytest.warns(UserWarning, match="spin magnitudes reach"):
        s._load()
    drawn = max(float(np.max(s.a1)), float(np.max(s.a2)))
    assert drawn > 0.99
    assert s._removal_amax_reference == drawn
    assert np.isclose(s._removal_amax_factor, (0.99 / drawn) ** 2)
    out = tmp_path / "sel.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s.export(str(out), spin_basis="component")
    with h5py.File(out, "r") as f:
        assert np.isclose(
            f.attrs["spin_removal_amax_reference_per_campaign"][0], drawn)


def test_removal_ceiling_check_is_silent_below_the_assumed_ceiling(tmp_path):
    from test_selection_spin import write_mixture
    path = write_mixture(tmp_path / "mix.hdf", "polar", n=200, seed=82,
                         amax=(0.4, 0.9))
    s = SelectionSet(path, strict_spin_checks="off")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        s._load()
    assert s._removal_amax_factor is None


# ======================================================================
# low-15: v1 combined per-campaign ceilings align with the campaigns
# ======================================================================
def test_v1_combined_ceilings_have_one_entry_per_campaign(tmp_path):
    none = write_o4_full(tmp_path / "none.hdf", n=30, seed=91, detected=False)
    o3 = write_endo3_full(tmp_path / "o3.hdf", n=30, seed=92)
    out = tmp_path / "v1.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        CombinedSelectionSet([SelectionSet(none), SelectionSet(o3)]
                             ).to_darksirens(str(out))
    with h5py.File(out, "r") as f:
        a1 = np.asarray(f.attrs["chi_eff_amax_1_per_campaign"])
        src = list(f.attrs["chi_eff_amax_source_per_campaign"])
        n_cosmo = len(f.attrs["cosmology_source_per_campaign"])
        scalar = float(f.attrs["chi_eff_amax"])
    assert a1.shape == (2,) == (n_cosmo,)
    assert np.isnan(a1[0]) and np.isclose(a1[1], 0.998)
    src = [x.decode() if isinstance(x, bytes) else x for x in src]
    assert src == ["no_detections", "detected"]
    assert np.isclose(scalar, 0.998)
