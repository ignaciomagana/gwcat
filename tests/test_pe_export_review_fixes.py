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
