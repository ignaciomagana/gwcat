"""Regression tests for the 2026-10-20 review of the v2 PE builder.

Each test reproduces a review failure scenario on a tiny synthetic store
(the fixtures of ``tests/test_export_v2_pe.py``) and fails on 3ba1dbf.
"""
import warnings

import h5py
import numpy as np
import pytest

from gwcat.catalog import GWCatalog

from test_export_v2_pe import _FULL_SPIN, _build_spin_store, _read

_COSMO = (67.74, 0.3089)


def _names(v):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in v]


def _ess(w):
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / (w ** 2).sum())


# ==========================================================================
# The ESS of a bootstrapped event, over distinct samples (review Medium 6)
# ==========================================================================
def test_ess_over_distinct_samples_for_a_bootstrapped_event(tmp_path):
    """An event with 40 samples drawn to 64 rows repeats rows; its row ESS
    counts each repeat again, so it can exceed the 40 distinct samples behind
    it.  The distinct-sample ESS counts each sample once."""
    events = [{"name": "GWe_000001", "n": 40}, {"name": "GWe_000002", "n": 300}]
    store, _raw = _build_spin_store(tmp_path, events, name="ess_store.h5")
    out = tmp_path / "ess.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        GWCatalog(store).export(str(out), format="gwcat2", spin_basis="chieff",
                                nsamp=64, seed=0, cosmology=_COSMO)
    cols, attrs = _read(out)
    nsamp = int(attrs["nsamp"])
    ess_rows = np.asarray(attrs["prior_reweight_ess_per_event"], float)
    ess_dist = np.asarray(attrs["prior_reweight_ess_distinct_per_event"],
                          float)
    n_unique = np.asarray(attrs["n_unique_samples_per_event"], int)
    assert n_unique[0] < 40 + 1 and n_unique[1] == nsamp

    # Independent: a distinct sample is a distinct m1det value; weight each once.
    for i in range(2):
        sl = slice(i * nsamp, (i + 1) * nsamp)
        _u, first = np.unique(cols["m1det"][sl], return_index=True)
        first.sort()
        assert first.size == n_unique[i]
        want = _ess(1.0 / cols["p_pe"][sl][first])
        assert ess_dist[i] == pytest.approx(want, rel=1e-12)
        assert ess_dist[i] <= n_unique[i] + 1e-9
    # The bootstrapped event's row ESS overstates; the other is unchanged.
    assert ess_rows[0] > ess_dist[0]
    assert ess_dist[1] == ess_rows[1]


# ==========================================================================
# Samples above the spin ceiling are counted with the cut off (High 1, Med 5)
# ==========================================================================
def _store_above(tmp_path, n_above=(60, 0), amax=0.9, name="above.h5"):
    from test_pe_spin_ceiling_cut import _store_with_spins_above
    return _store_with_spins_above(tmp_path, n_above=n_above, amax=amax,
                                   name=name)


def test_spin_ceiling_counts_are_written_without_the_cut(tmp_path):
    """The production default (no cut) wrote n_above_spin_ceiling_raw as
    zeros although the raw labels hold samples above the ceiling, and nothing
    counted the exported rows a chieff_reference selection weights at zero."""
    store, per = _store_above(tmp_path)
    out = tmp_path / "pe.h5"
    nsamp = 64
    with pytest.warns(UserWarning, match="above the ceiling their chi_eff"):
        GWCatalog(store).export(str(out), format="gwcat2",
                                spin_basis="chieff", nsamp=nsamp, seed=0,
                                cosmology=_COSMO)
    cols, attrs = _read(out)
    above = [(p["a1"] > 0.9) | (p["a2"] > 0.9) for p in per]
    assert [int(x) for x in attrs["n_above_spin_ceiling_raw_per_event"]] \
        == [int(a.sum()) for a in above] == [60, 0]
    # The exported rows: reproduce the (single shared) draw independently.
    rng = np.random.default_rng(0)
    want = [int(above[i][rng.choice(a.size, size=nsamp, replace=False)].sum())
            for i, a in enumerate(above)]
    assert want[0] > 0
    assert [int(x) for x in attrs["n_exported_above_spin_ceiling_per_event"]] \
        == want
    assert bool(attrs["spin_ceiling_counted"]) is True
    # Nothing was removed: the counts describe what the file kept.
    assert bool(attrs["spin_ceiling_cut_applied"]) is False
    assert [int(x) for x in attrs["n_dropped_spin_above_ceiling_per_event"]] \
        == [0, 0]


def test_the_cut_leaves_no_exported_row_above_the_ceiling(tmp_path):
    store, _per = _store_above(tmp_path)
    out = tmp_path / "pe_cut.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        GWCatalog(store).export(str(out), format="gwcat2",
                                spin_basis="chieff", nsamp=64, seed=0,
                                cosmology=_COSMO,
                                drop_spin_above_ceiling=True)
    _cols, attrs = _read(out)
    assert [int(x) for x in attrs["n_exported_above_spin_ceiling_per_event"]] \
        == [0, 0]
    assert [int(x) for x in attrs["n_above_spin_ceiling_raw_per_event"]] \
        == [60, 0]


def test_an_uncountable_event_is_minus_one_not_zero(tmp_path):
    """No a_1/a_2 samples (or a basis with no ceiling): -1, never a 0 that
    reads as 'none above the ceiling'."""
    events = [{"name": "GWns_000001", "provide": {"cos_tilt_1"}},
              {"name": "GWns_000002"}]
    store, _ = _build_spin_store(tmp_path, events, name="nospin_cols.h5")
    out = tmp_path / "pe.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        GWCatalog(store).export(str(out), format="gwcat2",
                                spin_basis="chieff", nsamp=32, seed=0,
                                cosmology=_COSMO)
    _cols, attrs = _read(out)
    raw = [int(x) for x in attrs["n_above_spin_ceiling_raw_per_event"]]
    assert raw[0] == -1 and raw[1] == 0
    assert int(attrs["n_exported_above_spin_ceiling_per_event"][0]) == -1
    assert bool(attrs["spin_ceiling_counted"]) is False

    out2 = tmp_path / "pe_nospin.h5"
    GWCatalog(store).export(str(out2), format="gwcat2", spin_basis="nospin",
                            nsamp=32, seed=0, cosmology=_COSMO)
    _cols, a2 = _read(out2)
    assert [int(x) for x in a2["n_above_spin_ceiling_raw_per_event"]] \
        == [-1, -1]


def _sel_reference(tmp_path, a_ref):
    from gwcat.selection import SelectionSet
    from test_selection_spin import write_o4_full
    inj = write_o4_full(tmp_path / "inj_ref.hdf", n=60, amax=(0.998, 0.998),
                        seed=6)
    out = tmp_path / "sel_ref.h5"
    SelectionSet(inj).export(str(out), spin_basis="chieff_reference",
                             spin_reference_amax=a_ref)
    return out


def test_reference_pair_refuses_pe_rows_above_a_ref_under_strict(tmp_path):
    """The reference-pair check compared declared ceilings only, so a chieff
    PE file keeping rows above a_ref passed --strict against a selection that
    gives them zero density."""
    from gwcat.export import validate_export_v2

    store, _per = _store_above(tmp_path)
    pe = tmp_path / "pe_keep.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        GWCatalog(store).export(str(pe), format="gwcat2", spin_basis="chieff",
                                nsamp=64, seed=0, cosmology=_COSMO)
    sel = _sel_reference(tmp_path, 0.9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises((ValueError, AssertionError),
                           match="xcheck_reference_spin_support"):
            validate_export_v2(str(pe), str(sel), strict=True)
    with pytest.warns(UserWarning, match="different spin supports"):
        res = validate_export_v2(str(pe), str(sel))
    assert res["xcheck_reference_spin_support"] is False

    # The same pair built with the cut validates.
    pe_cut = tmp_path / "pe_cut.h5"
    GWCatalog(store).export(str(pe_cut), format="gwcat2", spin_basis="chieff",
                            nsamp=64, seed=0, cosmology=_COSMO,
                            drop_spin_above_ceiling=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = validate_export_v2(str(pe_cut), str(sel), strict=True)
    assert res["xcheck_reference_spin_support"] is True


# ==========================================================================
# The scalar PE cosmology comes from the exported events (review Medium 7)
# ==========================================================================
def _store_cosmologies(tmp_path, H0s, Om0s, far_first=False, name="cos.h5"):
    events = [{"name": f"GWk_00000{i + 1}"} for i in range(len(H0s))]
    store, _ = _build_spin_store(tmp_path, events, name=name)
    with h5py.File(store, "r+") as f:
        f["meta/dL_prior_H0"][...] = np.asarray(H0s, float)
        f["meta/dL_prior_Om0"][...] = np.asarray(Om0s, float)
        if far_first:
            off = f["index/offsets"][:]
            dL = f["samples/luminosity_distance"][:]
            dL[int(off[0]):int(off[1])] = 60000.0     # every sample at z > 1
            f["samples/luminosity_distance"][...] = dL
    return store


def test_pe_cosmology_scalar_is_an_exported_events_cosmology(tmp_path):
    """Selected event 0 (the only one at the first cosmology) is dropped by
    z_max; the scalar used to keep describing it, with
    cosmology_per_event_varies=False, so the file stated a cosmology none of
    its events was analysed under."""
    store = _store_cosmologies(tmp_path, [70.74, 67.74, 67.74],
                               [0.30, 0.3089, 0.3089], far_first=True)
    out = tmp_path / "pe.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        GWCatalog(store).export(str(out), format="gwcat2",
                                spin_basis="chieff", nsamp=32, seed=0,
                                z_max=1.0)
    _cols, attrs = _read(out)
    assert _names(attrs["event_names"]) == ["GWk_000002", "GWk_000003"]
    assert bool(attrs["cosmology_per_event_varies"]) is False
    assert float(attrs["pe_cosmology_H0"]) == 67.74
    assert float(attrs["pe_cosmology_Om0"]) == 0.3089
    src = attrs["pe_cosmology_scalar_source"]
    assert (src.decode() if isinstance(src, bytes) else src) \
        == "all_exported_events"
    assert int(attrs["pe_cosmology_scalar_n_events"]) == 2


def test_pe_cosmology_scalar_says_when_it_describes_only_some_events(tmp_path):
    store = _store_cosmologies(tmp_path, [67.74, 67.90, 67.90],
                               [0.3075, 0.3065, 0.3065], name="cos2.h5")
    out = tmp_path / "pe.h5"
    GWCatalog(store).export(str(out), format="gwcat2", spin_basis="chieff",
                            nsamp=32, seed=0)
    _cols, attrs = _read(out)
    assert bool(attrs["cosmology_per_event_varies"]) is True
    src = attrs["pe_cosmology_scalar_source"]
    assert (src.decode() if isinstance(src, bytes) else src) \
        == "first_exported_event"
    assert int(attrs["pe_cosmology_scalar_n_events"]) == 1
    # The scalar itself is unchanged when event 0 is exported.
    assert float(attrs["pe_cosmology_H0"]) == 67.74


# ==========================================================================
# f_ref is recorded, and a tilt fit across reference frequencies warns
# (review Medium 19)
# ==========================================================================
def _store_f_ref(tmp_path, f_refs, name="fref.h5"):
    events = [{"name": f"GWf_00000{i + 1}"} for i in range(len(f_refs))]
    store, _ = _build_spin_store(tmp_path, events, name=name)
    with h5py.File(store, "r+") as f:
        f["meta"].create_dataset("f_ref", data=np.asarray(f_refs, float))
        f["meta"].create_dataset(
            "f_ref_source", data=np.array(["meta_data"] * len(f_refs),
                                          dtype=h5py.string_dtype()))
    return store


def test_component_export_records_f_ref_and_warns_on_mixed_frequencies(
        tmp_path):
    store = _store_f_ref(tmp_path, [10.0, 20.0])
    out = tmp_path / "comp.h5"
    with pytest.warns(UserWarning, match="reference frequency"):
        GWCatalog(store).export(str(out), format="gwcat2",
                                spin_basis="component", nsamp=32, seed=0,
                                cosmology=_COSMO)
    _cols, attrs = _read(out)
    np.testing.assert_array_equal(attrs["f_ref_per_event"], [10.0, 20.0])
    assert _names(attrs["f_ref_source_per_event"]) == ["meta_data"] * 2


def test_one_f_ref_does_not_warn_and_chieff_never_does(tmp_path):
    store = _store_f_ref(tmp_path, [20.0, 20.0], name="fref_one.h5")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        GWCatalog(store).export(str(tmp_path / "c.h5"), format="gwcat2",
                                spin_basis="component", nsamp=32, seed=0,
                                cosmology=_COSMO)
    mixed = _store_f_ref(tmp_path, [10.0, np.nan], name="fref_mixed.h5")
    out = tmp_path / "x.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        GWCatalog(mixed).export(str(out), format="gwcat2",
                                spin_basis="chieff", nsamp=32, seed=0,
                                cosmology=_COSMO)
    _cols, attrs = _read(out)
    f_ref = np.asarray(attrs["f_ref_per_event"], float)
    assert f_ref[0] == 10.0 and np.isnan(f_ref[1])


# ==========================================================================
# An event wholly above z_max is named and recorded (review Medium 39)
# ==========================================================================
def test_an_event_wholly_above_z_max_is_named_and_recorded(tmp_path):
    """Three selected events, one with every sample at dL 5-6 Gpc: it
    vanished from the export, and the only warning counted the samples over
    the kept events (or did not fire at all)."""
    events = [{"name": f"GWz_00000{i + 1}"} for i in range(3)]
    store, _ = _build_spin_store(tmp_path, events, name="zdrop.h5")
    with h5py.File(store, "r+") as f:
        off = f["index/offsets"][:]
        dL = f["samples/luminosity_distance"][:]
        n0 = int(off[1] - off[0])
        dL[int(off[0]):int(off[1])] = np.linspace(5000.0, 6000.0, n0)
        f["samples/luminosity_distance"][...] = dL
    out = tmp_path / "pe.h5"
    with pytest.warns(UserWarning, match=r"1 of 3 selected event.*GWz_000001"):
        GWCatalog(store).export(str(out), format="gwcat2",
                                spin_basis="chieff", nsamp=32, seed=0,
                                cosmology=_COSMO, z_max=0.3)
    _cols, attrs = _read(out)
    assert _names(attrs["event_names"]) == ["GWz_000002", "GWz_000003"]
    assert _names(attrs["events_dropped_by_z_max"]) == ["GWz_000001"]
    assert [int(x) for x in attrs["n_samples_cut_by_z_max_dropped_events"]] \
        == [n0]
    # Without z_max nothing is dropped, and the record says so.
    out2 = tmp_path / "pe_noz.h5"
    GWCatalog(store).export(str(out2), format="gwcat2", spin_basis="chieff",
                            nsamp=32, seed=0, cosmology=_COSMO)
    _cols, a2 = _read(out2)
    assert len(a2["events_dropped_by_z_max"]) == 0


# ==========================================================================
# The draw is reproducible from the file's own record (review Low 2)
# ==========================================================================
def test_the_file_records_enough_to_reproduce_its_draw(tmp_path):
    """A seed-1 rebuild had the same provenance as the seed-0 file apart from
    the ESS; the seed, replace policy and pool sizes were nowhere."""
    events = [{"name": "GWd_000001", "n": 150}, {"name": "GWd_000002", "n": 40}]
    store, raw = _build_spin_store(tmp_path, events, name="draw.h5")
    attrs_by_seed = {}
    for seed in (0, 1):
        out = tmp_path / f"pe_{seed}.h5"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            GWCatalog(store).export(str(out), format="gwcat2",
                                    spin_basis="chieff", nsamp=64, seed=seed,
                                    cosmology=_COSMO)
        attrs_by_seed[seed] = _read(out)
    cols, attrs = attrs_by_seed[1]
    assert int(attrs["resample_seed"]) == 1
    assert int(attrs_by_seed[0][1]["resample_seed"]) == 0
    pol = attrs["resample_replace_policy"]
    assert (pol.decode() if isinstance(pol, bytes) else pol) == "auto"
    assert "default_rng" in str(attrs["resample_rng_scheme"])
    n_kept = [int(x) for x in attrs["n_kept_samples_per_event"]]
    assert n_kept == [150, 40]

    # Reproduce the draw from the record alone.
    nsamp = int(attrs["nsamp"])
    rng = np.random.default_rng(int(attrs["resample_seed"]))
    for i, name in enumerate(_names(attrs["event_names"])):
        idx = rng.choice(n_kept[i], size=nsamp, replace=n_kept[i] < nsamp)
        np.testing.assert_array_equal(cols["m1det"][i * nsamp:(i + 1) * nsamp],
                                      raw[name]["mass_1"][idx])


# ==========================================================================
# A forced replace=True is recorded as a draw with replacement (review Low 64)
# ==========================================================================
def test_forced_replacement_is_recorded(tmp_path):
    events = [{"name": "GWr_000001", "n": 300}, {"name": "GWr_000002", "n": 300}]
    store, _ = _build_spin_store(tmp_path, events, name="rep.h5")
    out = tmp_path / "pe.h5"
    GWCatalog(store).export(str(out), format="gwcat2", spin_basis="chieff",
                            nsamp=64, seed=0, cosmology=_COSMO, replace=True)
    _cols, attrs = _read(out)
    assert bool(attrs["resampled_with_replacement"]) is True
    assert int(attrs["n_events_resampled_with_replacement"]) == 2
    assert _names(attrs["events_resampled_with_replacement"]) \
        == ["GWr_000001", "GWr_000002"]

    # The default draw of the same store uses no replacement and says so.
    out2 = tmp_path / "pe_auto.h5"
    GWCatalog(store).export(str(out2), format="gwcat2", spin_basis="chieff",
                            nsamp=64, seed=0, cosmology=_COSMO)
    _cols, a2 = _read(out2)
    assert bool(a2["resampled_with_replacement"]) is False
    assert len(a2["events_resampled_with_replacement"]) == 0
