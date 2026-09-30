"""GW-40c: ``drop_spin_above_ceiling`` -- samples above the spin ceiling are
dropped BEFORE resampling, and counted.

The GWTC analytic priors declare a_i <= 0.99 while real posteriors reach
0.9999; a projection basis (chieff) keeps the declared ceiling, so without the
cut the PE spin support is wider than a ``chieff_reference`` selection's (which
gives a_i > a_ref zero density).  For a uniform-magnitude prior the cut is
exact rejection to U(0, amax).
"""
import h5py
import numpy as np
import pytest

from gwcat.catalog import GWCatalog
from gwcat.cli import main

from test_export_v2_pe import _build_spin_store, _read

_COSMO = (67.74, 0.3089)


def _store_with_spins_above(tmp_path, n_above=(7, 0), n=300, amax=0.9,
                            name="cut_store.h5"):
    """Two events declaring amax; the first ``n_above[i]`` samples of event i
    get a_1 (and every third also a_2) pushed above the ceiling."""
    events = [{"name": "GWcut_000001", "amax1": amax, "amax2": amax, "n": n},
              {"name": "GWcut_000002", "amax1": amax, "amax2": amax, "n": n}]
    store, raw = _build_spin_store(tmp_path, events, name=name)
    with h5py.File(store, "r+") as f:
        off = f["index/offsets"][:]
        a1 = f["samples/a_1"][:]
        a2 = f["samples/a_2"][:]
        for i, k in enumerate(n_above):
            lo = int(off[i])
            a1[lo:lo + k] = amax + 0.05
            a2[lo:lo + k:3] = amax + 0.02
        f["samples/a_1"][...] = a1
        f["samples/a_2"][...] = a2
    with h5py.File(store, "r") as f:
        off = f["index/offsets"][:]
        a1 = f["samples/a_1"][:]
        a2 = f["samples/a_2"][:]
        m1 = f["samples/mass_1"][:]
    per = [dict(a1=a1[off[i]:off[i + 1]], a2=a2[off[i]:off[i + 1]],
                m1=m1[off[i]:off[i + 1]]) for i in range(2)]
    return store, per


def _independent_count(per, amax):
    return [int(np.sum((p["a1"] > amax) | (p["a2"] > amax))) for p in per]


@pytest.mark.parametrize("basis", ["chieff", "component"])
def test_samples_above_the_ceiling_are_dropped_and_counted(tmp_path, basis):
    store, per = _store_with_spins_above(tmp_path, n_above=(7, 0))
    out = tmp_path / f"pe_{basis}.h5"
    GWCatalog(store).export(str(out), format="gwcat2", spin_basis=basis,
                            nsamp=64, seed=0, cosmology=_COSMO,
                            drop_spin_above_ceiling=True)
    cols, attrs = _read(out)
    got = [int(x) for x in attrs["n_dropped_spin_above_ceiling_per_event"]]
    assert got == _independent_count(per, 0.9) == [7, 0]
    assert bool(attrs["spin_ceiling_cut_applied"]) is True
    if basis == "component":
        # no exported sample exceeds the ceiling, and none needed widening
        assert np.all(cols["a1"] <= 0.9) and np.all(cols["a2"] <= 0.9)
        assert all((s.decode() if isinstance(s, bytes) else s) == "analytic"
                   for s in attrs["spin_amax_source_1_per_event"])
    assert int(attrs["n_samples_out_of_support"]) == 0


def test_draw_is_over_the_kept_rows_only(tmp_path):
    """The first event's draw is rng.choice over the n - n_dropped kept rows,
    re-indexed into the original row numbers."""
    store, per = _store_with_spins_above(tmp_path, n_above=(7, 0))
    out = tmp_path / "pe.h5"
    nsamp = 64
    GWCatalog(store).export(str(out), format="gwcat2", spin_basis="chieff",
                            nsamp=nsamp, seed=0, cosmology=_COSMO,
                            drop_spin_above_ceiling=True)
    cols, _ = _read(out)
    p = per[0]
    kept_rows = np.nonzero((p["a1"] <= 0.9) & (p["a2"] <= 0.9))[0]
    idx = np.random.default_rng(0).choice(kept_rows.size, size=nsamp,
                                          replace=False)
    np.testing.assert_array_equal(cols["m1det"][:nsamp],
                                  p["m1"][kept_rows[idx]])


def test_flag_off_is_the_historical_draw(tmp_path):
    store, per = _store_with_spins_above(tmp_path, n_above=(7, 0))
    a = tmp_path / "off.h5"
    GWCatalog(store).export(str(a), format="gwcat2", spin_basis="component",
                            nsamp=64, seed=0, cosmology=_COSMO)
    cols, attrs = _read(a)
    assert bool(attrs["spin_ceiling_cut_applied"]) is False
    assert [int(x) for x in attrs["n_dropped_spin_above_ceiling_per_event"]] \
        == [0, 0]
    idx = np.random.default_rng(0).choice(per[0]["m1"].size, size=64,
                                          replace=False)
    np.testing.assert_array_equal(cols["m1det"][:64], per[0]["m1"][idx])


def test_cut_below_nsamp_resamples_with_replacement(tmp_path):
    store, per = _store_with_spins_above(tmp_path, n_above=(30, 0), n=70)
    out = tmp_path / "pe.h5"
    GWCatalog(store).export(str(out), format="gwcat2", spin_basis="chieff",
                            nsamp=64, seed=0, cosmology=_COSMO,
                            drop_spin_above_ceiling=True)
    _, attrs = _read(out)
    assert [int(x) for x in attrs["n_dropped_spin_above_ceiling_per_event"]] \
        == _independent_count(per, 0.9) == [30, 0]
    # 70 - 30 = 40 < 64 -> with replacement for event 1 only
    assert int(attrs["n_events_resampled_with_replacement"]) == 1
    assert int(attrs["n_unique_samples_per_event"][0]) <= 40


def test_cut_uses_a_forced_amax(tmp_path):
    store, per = _store_with_spins_above(tmp_path, n_above=(7, 0))
    out = tmp_path / "pe.h5"
    GWCatalog(store).export(str(out), format="gwcat2", spin_basis="chieff",
                            nsamp=64, seed=0, cosmology=_COSMO, amax=0.5,
                            drop_spin_above_ceiling=True)
    _, attrs = _read(out)
    assert [int(x) for x in attrs["n_dropped_spin_above_ceiling_per_event"]] \
        == _independent_count(per, 0.5)


def test_nospin_basis_refuses_the_flag(tmp_path):
    store, _ = _store_with_spins_above(tmp_path)
    with pytest.raises(ValueError, match="divides out no spin prior"):
        GWCatalog(store).export(str(tmp_path / "x.h5"), format="gwcat2",
                                spin_basis="nospin", nsamp=16, seed=0,
                                cosmology=_COSMO, drop_spin_above_ceiling=True)


def test_missing_spin_columns_refuse_the_cut(tmp_path):
    events = [{"name": "GWns_000001", "provide": {"cos_tilt_1"}},
              {"name": "GWns_000002"}]
    store, _ = _build_spin_store(tmp_path, events, name="nospin.h5")
    with pytest.raises(ValueError, match="a_1/a_2"):
        GWCatalog(store).export(str(tmp_path / "x.h5"), format="gwcat2",
                                spin_basis="chieff", nsamp=16, seed=0,
                                cosmology=_COSMO, drop_spin_above_ceiling=True)


def test_cli_flag(tmp_path):
    store, per = _store_with_spins_above(tmp_path, n_above=(5, 2))
    out = tmp_path / "cli.h5"
    rc = main(["export", "pe", store, "--out", str(out), "--parameter-space",
               "chieff", "--nsamp", "32", "--seed", "0", "--cosmology",
               "67.74,0.3089", "--drop-spin-above-ceiling", "--no-summary"])
    assert rc == 0
    _, attrs = _read(out)
    assert [int(x) for x in attrs["n_dropped_spin_above_ceiling_per_event"]] \
        == _independent_count(per, 0.9)
