"""Regression tests for the 2026-10-20 review of the catalog layer.

Covers the event-selection view (``GWCatalog.select`` / ``SelectionSpec``),
the frozen v1 exporter and validator (``to_darksirens`` / ``validate_export``),
the source-class pairing judgement, the bundled BBH lists and the user
metadata overrides.  Each test starts from the review's failure scenario and
fails on the reviewed commit (3ba1dbf).

Fixtures are tiny synthetic stores written with h5py; one row per
``(event, sample set)``, so a multi-sample-set store is a list with repeated
names.  No network.
"""
import json
import warnings

import h5py
import numpy as np
import pytest

from gwcat.catalog import (GWCatalog, SelectionSpec, UnpairableSelectionCut,
                           validate_export)

from test_selection_cbc import _make_combined_selection

_COSMO = (67.74, 0.3089)
_PARAMS = ["mass_1", "mass_2", "luminosity_distance", "ra", "dec", "chi_eff",
           "p_dL_pe"]
_STR_META = ("source_class", "compact_type", "sample_set_name", "approximant",
             "waveform", "prior_source_kind_spin", "mass_prior_kind")
_FLOAT_META = ("far", "far_available", "pastro", "snr_med", "m1_src_med",
               "m2_src_med", "sky_area_90", "is_preferred", "is_mixed",
               "priority_rank", "spin_amax_1", "spin_amax_2", "dL_prior_H0",
               "dL_prior_Om0")


def _row(name, **kw):
    """One store row; every meta field has an explicit default."""
    r = {"name": name, "source_class": "BBH", "far": 1e-3, "pastro": 0.99,
         "snr_med": 12.0, "m1_src_med": 30.0, "m2_src_med": 20.0,
         "sky_area_90": 50.0, "is_preferred": 1.0, "is_mixed": 0.0,
         "priority_rank": 0.0, "spin_amax_1": 0.99, "spin_amax_2": 0.99,
         "sample_set_name": "C01:IMRPhenomXPHM",
         "approximant": "IMRPhenomXPHM",
         "prior_source_kind_spin": "own_analytic",
         "mass_prior_kind": "uniform_detector_frame", "n": 40}
    r.update(kw)
    r.setdefault("compact_type", r["source_class"])
    r.setdefault("waveform", r["approximant"])
    return r


def _store(tmp_path, rows, name="store.h5", seed=3, drop_meta=()):
    rng = np.random.default_rng(seed)
    offsets = [0]
    cols = {p: [] for p in _PARAMS}
    for r in rows:
        n = int(r["n"])
        cols["mass_1"].append(r.get("mass_1", rng.uniform(25, 45, n)))
        cols["mass_2"].append(r.get("mass_2", rng.uniform(10, 22, n)))
        cols["luminosity_distance"].append(
            r.get("luminosity_distance", rng.uniform(300, 800, n)))
        cols["ra"].append(rng.uniform(0, 2 * np.pi, n))
        cols["dec"].append(rng.uniform(-np.pi / 2, np.pi / 2, n))
        cols["chi_eff"].append(r.get("chi_eff", rng.uniform(-0.3, 0.3, n)))
        cols["p_dL_pe"].append(rng.uniform(0.1, 1.0, n))
        offsets.append(offsets[-1] + n)
    path = tmp_path / name
    with h5py.File(path, "w") as f:
        f.attrs["param_names"] = np.array(_PARAMS, dtype=h5py.string_dtype())
        idx = f.create_group("index")
        idx.create_dataset("offsets", data=np.array(offsets, dtype="i8"))
        idx.create_dataset("event_names", data=np.array(
            [r["name"] for r in rows], dtype=h5py.string_dtype()))
        mg = f.create_group("meta")
        for k in _STR_META:
            if k not in drop_meta:
                mg.create_dataset(k, data=np.array(
                    [str(r[k]) for r in rows], dtype=h5py.string_dtype()))
        for k in _FLOAT_META:
            if k in drop_meta:
                continue
            if k == "far_available":
                vals = [1.0 if np.isfinite(r["far"]) else 0.0 for r in rows]
            elif k in ("dL_prior_H0", "dL_prior_Om0"):
                vals = [_COSMO[0] if k == "dL_prior_H0" else _COSMO[1]
                        for _ in rows]
            else:
                vals = [float(r[k]) for r in rows]
            mg.create_dataset(k, data=np.asarray(vals, dtype="f8"))
        sg = f.create_group("samples")
        for p in _PARAMS:
            sg.create_dataset(p, data=np.concatenate(cols[p]))
    return str(path)


def _three(tmp_path, **kw):
    rows = [_row("GW900001_000001", m1_src_med=25.0),
            _row("GW900002_000002", m1_src_med=60.0),
            _row("GW900003_000003", m1_src_med=90.0)]
    return _store(tmp_path, rows, **kw)


def _attr(f, k):
    v = f.attrs[k]
    return v.decode() if isinstance(v, bytes) else v


# ==========================================================================
# Cuts on posterior-median masses or sky area are refused at export
# ==========================================================================
@pytest.mark.parametrize("cut", [{"m1_src_range": (20.0, 70.0)},
                                 {"m2_src_range": (3.0, 1e9)},
                                 {"sky_area_max": 100.0}])
def test_point_estimate_cuts_are_refused_by_both_exporters(tmp_path, cut):
    view = GWCatalog(_three(tmp_path)).select(**cut)
    with pytest.raises(UnpairableSelectionCut, match="POINT ESTIMATES"):
        view.to_darksirens(str(tmp_path / "v1.h5"), nsamp=8, seed=0,
                           cosmology=_COSMO)
    with pytest.raises(UnpairableSelectionCut, match="POINT ESTIMATES"):
        view.export(str(tmp_path / "v2.h5"), spin_basis="chieff", nsamp=8,
                    seed=0, cosmology=_COSMO)
    assert not (tmp_path / "v1.h5").exists()
    assert not (tmp_path / "v2.h5").exists()


def test_a_declared_point_estimate_cut_is_written_recorded_and_refused_in_a_pair(
        tmp_path):
    cat = GWCatalog(_three(tmp_path))
    view = cat.select(m1_src_range=(20.0, 70.0),
                      allow_unpaired_point_estimate_cuts=True)
    # The declaration survives a further select() (the exporters re-select).
    out = tmp_path / "pe.h5"
    view.select(far_max=1.0).to_darksirens(str(out), nsamp=8, seed=0,
                                           cosmology=_COSMO)
    with h5py.File(out, "r") as f:
        assert _attr(f, "point_estimate_cuts") == "m1_src_range"
        spec = json.loads(_attr(f, "selection_spec"))
        assert spec["point_estimate_cuts"] == ["m1_src_range"]
        assert spec["point_estimate_cuts_allowed"] is True
        # The median-mass range is a class cut under another spelling.
        assert _attr(f, "source_class_cut_estimator") == "posterior_median_mass"
        assert int(f.attrs["nobs"]) == 2
    sel = _make_combined_selection(tmp_path)
    with pytest.raises(ValueError, match="xcheck_point_estimate_cuts"):
        validate_export(str(out), sel)


def test_the_unfiltered_spec_record_is_unchanged():
    """The new keys appear only when such a cut is in effect."""
    spec = SelectionSpec().refine(far_max=1.0)
    d = spec.to_dict()
    assert "point_estimate_cuts" not in d
    assert "point_estimate_cuts" not in spec.to_attrs()


# ==========================================================================
# snr_min says which events it dropped for a missing SNR
# ==========================================================================
def test_snr_min_names_the_events_it_drops_for_a_missing_snr(tmp_path):
    """57 of the 259 production BBH events have no network_optimal_snr, so
    select(snr_min=0.0) dropped them -- GW150914 among them -- silently."""
    rows = [_row("GW150914_095045", snr_med=np.nan),
            _row("GW900002_000002", snr_med=9.0),
            _row("GW900003_000003", snr_med=30.0)]
    cat = GWCatalog(_store(tmp_path, rows))
    with pytest.warns(UserWarning, match="GW150914_095045"):
        sub = cat.select(snr_min=0.0)
    assert sorted(sub.event_names) == ["GW900002_000002", "GW900003_000003"]
    spec = sub.selection_spec.to_dict()
    assert spec["n_events_missing_snr"] == 1


def test_snr_min_on_an_all_nan_column_is_a_configuration_error(tmp_path):
    rows = [_row("GW900001_000001", snr_med=np.nan),
            _row("GW900002_000002", snr_med=np.nan)]
    with pytest.raises(ValueError, match="snr_med"):
        GWCatalog(_store(tmp_path, rows)).select(snr_min=8.0)


def test_snr_min_on_a_store_without_the_column_is_refused(tmp_path):
    rows = [_row("GW900001_000001")]
    cat = GWCatalog(_store(tmp_path, rows, drop_meta=("snr_med",)))
    with pytest.raises(ValueError, match="'snr_med' meta column"):
        cat.select(snr_min=8.0)


# ==========================================================================
# The v1 exporter records its z_max truncation, and the v1 pair refuses it
# ==========================================================================
def _zmax_store(tmp_path):
    rng = np.random.default_rng(5)
    rows = [_row("GW900001_000001", n=60,
                 luminosity_distance=rng.uniform(300, 800, 60)),
            _row("GW900002_000002", n=60,
                 luminosity_distance=rng.uniform(5000, 6000, 60))]
    return _store(tmp_path, rows, name="zmax_store.h5")


def test_v1_z_max_is_recorded_and_a_whole_event_drop_is_named(tmp_path):
    out = tmp_path / "pe_z.h5"
    with pytest.warns(UserWarning, match="GW900002_000002"):
        GWCatalog(_zmax_store(tmp_path)).to_darksirens(
            str(out), nsamp=16, seed=0, cosmology=_COSMO, z_max=0.12)
    with h5py.File(out, "r") as f:
        assert float(f.attrs["z_max"]) == 0.12
        assert int(f.attrs["nobs"]) == 1
        cut = np.asarray(f.attrs["n_samples_cut_by_z_max"])
        assert cut.shape == (1,) and 0 < cut[0] < 60
        dropped = [x.decode() if isinstance(x, bytes) else x
                   for x in f.attrs["events_dropped_by_z_max"]]
        assert dropped == ["GW900002_000002"]


def test_v1_truncated_pe_file_is_refused_against_a_v1_selection(tmp_path):
    out = tmp_path / "pe_z.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        GWCatalog(_zmax_store(tmp_path)).to_darksirens(
            str(out), nsamp=16, seed=0, cosmology=_COSMO, z_max=0.12)
    sel = _make_combined_selection(tmp_path)
    with pytest.raises(ValueError, match="xcheck_z_max"):
        validate_export(str(out), sel)


def test_v1_untruncated_file_records_no_z_max(tmp_path):
    out = tmp_path / "pe_noz.h5"
    GWCatalog(_three(tmp_path)).to_darksirens(str(out), nsamp=8, seed=0,
                                              cosmology=_COSMO)
    with h5py.File(out, "r") as f:
        assert np.isnan(float(f.attrs["z_max"]))
    results = validate_export(str(out), _make_combined_selection(tmp_path))
    assert results["xcheck_z_max"] is True


# ==========================================================================
# A name whitelist is not "reproducible by construction"
# ==========================================================================
def test_a_whitelist_against_an_unfiltered_selection_warns():
    """The bundled 259-event BBH list equals {posterior-median m2 >= 3} on the
    production store, and the production selection applies no class cut, yet
    the pairing judgement passed it as reproducible by construction."""
    from gwcat.source_class import (assess_cut_estimator_pair,
                                    CUT_ESTIMATOR_NAME_WHITELIST,
                                    CUT_ESTIMATOR_INJECTED_TRUTH,
                                    CUT_ESTIMATOR_NONE, CUT_ESTIMATOR_REMEDY)
    verdict, msg = assess_cut_estimator_pair(
        CUT_ESTIMATOR_NAME_WHITELIST, CUT_ESTIMATOR_NONE, False, False)
    assert verdict == "warn"
    assert "m2" in msg and "posterior-median" in msg
    # A selection that does cut its injections on a class is the declared
    # approximation and still passes.
    assert assess_cut_estimator_pair(
        CUT_ESTIMATOR_NAME_WHITELIST, CUT_ESTIMATOR_INJECTED_TRUTH,
        False, True) == ("pass", "")
    # The remedy no longer sends a refused median cut back as an event list.
    assert "reproducible on the injection side, so a" not in CUT_ESTIMATOR_REMEDY


def test_v1_whitelist_pair_validates_with_the_warning(tmp_path):
    from test_selection_cbc import _make_pe_export
    pe = _make_pe_export(tmp_path, event_list=["GW900001_000001"],
                         name="pe_wl.h5")
    sel = _make_combined_selection(tmp_path, name="sel_wl.h5")
    with pytest.warns(UserWarning, match="selected by NAME"):
        results = validate_export(pe, sel)
    assert results["xcheck_source_class_estimator"] is True


# ==========================================================================
# Per-sample-set cuts apply to the sample set the policy chose
# ==========================================================================
def _multiset_rows():
    """Event A's Mixed set (preferred) and SEOBNR set carry no SNR, its XPHM
    set does -- the shape of a GWTC-2.1/3 event ingested with every set."""
    a, b = "GW900001_000001", "GW900002_000002"
    mixed = dict(sample_set_name="C01:Mixed", approximant="Mixed",
                 is_mixed=1.0, is_preferred=1.0, priority_rank=0.0)
    return [
        _row(a, snr_med=np.nan, **mixed),
        _row(a, snr_med=np.nan, sample_set_name="C01:SEOBNRv4PHM",
             approximant="SEOBNRv4PHM", is_preferred=0.0, priority_rank=2.0),
        _row(a, snr_med=12.0, sample_set_name="C01:IMRPhenomXPHM",
             approximant="IMRPhenomXPHM", is_preferred=0.0, priority_rank=1.0),
        _row(b, snr_med=15.0, **mixed),
        _row(b, snr_med=15.0, sample_set_name="C01:IMRPhenomXPHM",
             approximant="IMRPhenomXPHM", is_preferred=0.0, priority_rank=1.0),
    ]


def test_a_cut_does_not_switch_an_event_to_another_waveform(tmp_path):
    cat = GWCatalog(_store(tmp_path, _multiset_rows()))
    with pytest.warns(UserWarning, match="chose for them"):
        sub = cat.select(snr_min=8.0, waveform_policy="mixed-first")
    assert list(sub.event_names) == ["GW900002_000002"]
    assert [cat.meta["sample_set_name"][r] for r in sub._sel] == ["C01:Mixed"]
    assert list(sub._selection_reasons) == ["mixed-first:is_mixed"]


def test_the_multiset_and_one_row_stores_select_the_same_events(tmp_path):
    rows = _multiset_rows()
    one = [r for r in rows if r["sample_set_name"] == "C01:Mixed"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = GWCatalog(_store(tmp_path, rows, name="all.h5")).select(
            snr_min=8.0, waveform_policy="mixed-first")
        b = GWCatalog(_store(tmp_path, one, name="one.h5")).select(
            snr_min=8.0, waveform_policy="mixed-first")
    assert list(a.event_names) == list(b.event_names)
    assert a.selection_spec.digest() == b.selection_spec.digest()


def test_policy_all_still_cuts_row_by_row(tmp_path):
    """Unchanged: under "all" every passing row is kept, so the cut stays
    row-wise."""
    cat = GWCatalog(_store(tmp_path, _multiset_rows()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sub = cat.select(snr_min=8.0, waveform_policy="all")
    got = sorted((str(cat.names[r]), str(cat.meta["sample_set_name"][r]))
                 for r in sub._sel)
    assert got == [("GW900001_000001", "C01:IMRPhenomXPHM"),
                   ("GW900002_000002", "C01:IMRPhenomXPHM"),
                   ("GW900002_000002", "C01:Mixed")]
    assert sub._homogeneous_sample_sets is False


# ==========================================================================
# The inclusive FAR cut says so, and names events on the rounded threshold
# ==========================================================================
def test_far_cut_names_events_on_the_rounded_threshold(tmp_path):
    rows = [_row("GW231002_143916", far=1.0), _row("GW900002_000002", far=0.96),
            _row("GW900003_000003", far=1e-3)]
    cat = GWCatalog(_store(tmp_path, rows))
    with pytest.warns(UserWarning, match="GW231002_143916") as rec:
        sub = cat.select(far_max=1.0)
    assert not any("GW900002_000002" in str(r.message) for r in rec)
    # The cut itself is unchanged: the event on the threshold is kept.
    assert "GW231002_143916" in set(sub.event_names)
    assert sub.selection_spec.to_dict()["far_cut_operator"] == "<="
    assert sub.selection_spec.to_attrs()["far_cut_operator"] == "<="


# ==========================================================================
# v1 'exclude' files state one spin ceiling only when it is one ceiling
# ==========================================================================
def test_v1_exclude_scalar_ceiling_is_nan_for_asymmetric_spins(tmp_path):
    """Core applies chi_eff_amax to both spins in exclude mode; an event at
    (0.99, 0.05) was written as 0.99 and evaluated at (0.99, 0.99)."""
    rows = [_row("GW900001_000001", spin_amax_1=0.99, spin_amax_2=0.05),
            _row("GW900002_000002", spin_amax_1=0.99, spin_amax_2=0.05)]
    out = tmp_path / "exc.h5"
    with pytest.warns(UserWarning, match="chi_eff_amax is NaN"):
        GWCatalog(_store(tmp_path, rows)).to_darksirens(
            str(out), nsamp=8, seed=0, cosmology=_COSMO,
            spin_prior_mode="exclude")
    with h5py.File(out, "r") as f:
        assert np.isnan(float(f.attrs["chi_eff_amax"]))
        np.testing.assert_array_equal(f.attrs["chi_eff_amax_2_per_event"],
                                      [0.05, 0.05])


def test_v1_symmetric_ceiling_is_still_written(tmp_path):
    out = tmp_path / "exc_sym.h5"
    GWCatalog(_three(tmp_path)).to_darksirens(
        str(out), nsamp=8, seed=0, cosmology=_COSMO,
        spin_prior_mode="exclude")
    with h5py.File(out, "r") as f:
        assert float(f.attrs["chi_eff_amax"]) == 0.99


# ==========================================================================
# The v1 exporter gates out-of-support samples like the 2.0 builder
# ==========================================================================
def test_v1_out_of_support_samples_are_refused_by_default(tmp_path):
    """`gwcat export-darksirens --amax 0.3` wrote 19% zero-weight samples with
    only a warning and no record."""
    from gwcat.export.pe_builder import OutOfSupportError
    out = tmp_path / "oos.h5"
    with pytest.raises(OutOfSupportError, match="max_out_of_support_frac"):
        GWCatalog(_three(tmp_path)).to_darksirens(
            str(out), nsamp=32, seed=0, cosmology=_COSMO, amax=0.1)
    assert not out.exists()


def test_v1_allowed_out_of_support_zeros_are_recorded_and_validate(tmp_path):
    out = tmp_path / "oos_ok.h5"
    with pytest.warns(UserWarning, match="outside the chi_eff prior's support"):
        GWCatalog(_three(tmp_path)).to_darksirens(
            str(out), nsamp=32, seed=0, cosmology=_COSMO, amax=0.1,
            allow_out_of_support=True)
    with h5py.File(out, "r") as f:
        n_oos = int(f.attrs["n_samples_out_of_support"])
        assert n_oos > 0
        assert int(np.sum(np.asarray(f["p_pe"]) == 0)) == n_oos
        assert int(np.sum(f.attrs["n_out_of_support_per_event"])) == n_oos
    results = validate_export(str(out))
    # The recorded spin-support zeros are not a distance-prior failure.
    assert results["pe_p_pe_positive"] is True


# ==========================================================================
# compact_type and source_class together record their intersection
# ==========================================================================
def _classes(tmp_path):
    rows = [_row("GW900001_000001", source_class="BBH"),
            _row("GW900002_000002", source_class="NSBH",
                 m2_src_med=1.5),
            _row("GW900003_000003", source_class="BBH")]
    return _store(tmp_path, rows, name="classes.h5")


@pytest.mark.parametrize("sc,expect", [("cbc", "BBH"),
                                       (["bbh", "nsbh"], "BBH")])
def test_compact_type_survives_a_source_class_request(tmp_path, sc, expect):
    view = GWCatalog(_classes(tmp_path)).select(compact_type="BBH")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sub = view.select(source_class=sc)
        out = tmp_path / "v2.h5"
        sub.export(str(out), spin_basis="chieff", nsamp=8, seed=0,
                   cosmology=_COSMO)
    assert sorted(sub.event_names) == ["GW900001_000001", "GW900003_000003"]
    assert sub.selection_spec.source_class_filter == expect
    with h5py.File(out, "r") as f:
        assert _attr(f, "source_class_filter") == expect


def test_disjoint_compact_type_and_source_class_are_refused():
    with pytest.raises(ValueError, match="share no class"):
        SelectionSpec().refine(compact_type="BBH").refine(source_class="nsbh")
    with pytest.raises(ValueError, match="share no class"):
        SelectionSpec().refine(compact_type="BBH").refine(compact_type="NSBH")


# ==========================================================================
# A chained view keeps the waveform policy that chose its sample sets
# ==========================================================================
def _strs(v):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in v]


def test_an_exported_view_records_the_policy_that_chose_it(tmp_path):
    cat = GWCatalog(_store(tmp_path, _multiset_rows()))
    view = cat.select(waveform_policy="mixed-first")
    v1, v2 = tmp_path / "v1.h5", tmp_path / "v2.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        view.to_darksirens(str(v1), nsamp=8, seed=0, cosmology=_COSMO)
        view.export(str(v2), spin_basis="chieff", nsamp=8, seed=0,
                    cosmology=_COSMO)
    for out in (v1, v2):
        with h5py.File(out, "r") as f:
            assert _attr(f, "waveform_policy") == "mixed-first"
            assert _strs(f.attrs["sample_set_selection_reason"]) == [
                "mixed-first:is_mixed", "mixed-first:is_mixed"]
            assert json.loads(_attr(f, "selection_spec"))[
                "waveform_policy"] == "mixed-first"


def test_an_event_map_view_keeps_its_map_through_the_export(tmp_path):
    cat = GWCatalog(_store(tmp_path, _multiset_rows()))
    smap = {"GW900001_000001": "C01:IMRPhenomXPHM",
            "GW900002_000002": "C01:Mixed"}
    view = cat.select(waveform_policy="event-map", sample_set_map=smap)
    out = tmp_path / "emap.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        view.export(str(out), spin_basis="chieff", nsamp=8, seed=0,
                    cosmology=_COSMO)
    with h5py.File(out, "r") as f:
        assert _attr(f, "waveform_policy") == "event-map"
        assert _attr(f, "sample_set_map_sha256") != ""
        assert _strs(f.attrs["sample_set_selection_reason"]) == [
            "event-map:mapped:C01:IMRPhenomXPHM", "event-map:mapped:C01:Mixed"]


def test_a_different_policy_cannot_rechoose_a_resolved_view(tmp_path):
    cat = GWCatalog(_store(tmp_path, _multiset_rows()))
    with pytest.raises(ValueError, match="already chosen"):
        cat.select().select(waveform_policy="mixed-first")
    # A view under "all" still holds every set, so a policy may choose.
    sub = cat.select(waveform_policy="all").select(waveform_policy="mixed-first")
    assert sub._waveform_policy == "mixed-first"
    assert list(sub._selection_reasons) == ["mixed-first:is_mixed"] * 2


# ==========================================================================
# The live GWOSC mass-threshold list does not replace the curated sample
# ==========================================================================
def test_a_live_superset_does_not_replace_the_curated_list_by_default(
        tmp_path, monkeypatch):
    """The live list keeps every confident m2 > 3 event with no FAR cut
    (GW200322_091133 at 140/yr); the superset rule let it replace the curated
    FAR < 1/yr population sample."""
    from gwcat import bbh_allowed_names as ban
    from gwcat import fetch

    extended = sorted(set(ban.BBH_ALL) | {"GW200322_091133"})
    monkeypatch.setattr(fetch, "fetch_bbh_names_gwosc", lambda **kw: extended)
    with pytest.warns(UserWarning, match="NO FAR cut") as rec:
        names = ban.get_bbh_allowed_names(data_dir=str(tmp_path),
                                          prefer_gwosc=True)
    assert names == sorted(ban.BBH_ALL)
    assert any("GW200322_091133" in str(r.message) for r in rec)


# ==========================================================================
# compact_type is one canonical class, matched case-insensitively
# ==========================================================================
def test_compact_type_is_normalised(tmp_path):
    cat = GWCatalog(_classes(tmp_path))
    sub = cat.select(compact_type="bbh")
    assert sorted(sub.event_names) == ["GW900001_000001", "GW900003_000003"]
    assert sub.selection_spec.compact_type == ("BBH",)
    with pytest.raises(ValueError, match="unrecognised source-class"):
        cat.select(compact_type="BBHs")
    with pytest.raises(ValueError, match="ONE class"):
        cat.select(compact_type="cbc")


def test_compact_type_on_a_store_without_the_column_is_refused(tmp_path):
    cat = GWCatalog(_store(tmp_path, [_row("GW900001_000001")],
                           drop_meta=("compact_type",)))
    with pytest.raises(ValueError, match="'compact_type' meta column"):
        cat.select(compact_type="BBH")


def test_v1_refuses_to_write_a_zero_event_file(tmp_path):
    out = tmp_path / "empty.h5"
    with pytest.raises(ValueError, match="left 0 of"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            GWCatalog(_three(tmp_path)).to_darksirens(
                str(out), nsamp=8, seed=0, cosmology=_COSMO, z_max=1e-6)
    assert not out.exists()
