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
