"""GW-41: fixes from the review of the exact-prior commit (GW-40i).

  1. The exact-dL cache is keyed on the cosmology's parameter VALUES, not its
     ``repr`` (astropy prints H0 to 8 significant digits, so two cosmologies
     1e-9 apart in H0 shared one set of cached distances).
  2. ``distance.dl``'s ``ln_prior_pe`` evaluates the prior with the store's own
     per-row implementation (``PEContext.dL_prior_impl``), so the legacy flag
     reaches it too.
  3. The validator: a one-sided ``chi_eff_prior_impl`` record fails under
     ``strict``; a PE file that mixes the exact and legacy families warns (and
     fails under ``strict``); ``require_exact_priors`` refuses anything but an
     exact product.  The PE builder warns when it writes a mixed product.
"""
import h5py
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from gwcat import cosmology as gc
from gwcat.catalog import GWCatalog
from gwcat.cli import main
from gwcat.export.contract import mixed_prior_impl_events
from gwcat.export.validate import (_check_pe_prior_impl,
                                   _xcheck_chi_eff_prior_impl,
                                   validate_export_v2)
from gwcat.params import PEContext
from gwcat.params.blocks.distance import ln_prior_pe
from gwcat.spin import chi_eff_prior_impl

from test_export_v2_pe import _build_spin_store

_COSMO = (67.74, 0.3089)


# ==========================================================================
# 1. Cache key
# ==========================================================================
def test_cache_key_distinguishes_cosmologies_repr_cannot():
    a = FlatLambdaCDM(67.9, 0.3065)
    b = FlatLambdaCDM(67.9 + 1e-9, 0.3065)
    assert repr(a) == repr(b)                       # the review's collision
    assert gc._cosmology_cache_key(a) != gc._cosmology_cache_key(b)
    # Same values, different name: the same density, so the same key.
    c = FlatLambdaCDM(67.9, 0.3065, name="lal")
    assert gc._cosmology_cache_key(a) == gc._cosmology_cache_key(c)
    # Every parameter enters, not only H0.
    assert (gc._cosmology_cache_key(FlatLambdaCDM(67.9, 0.3065, Tcmb0=2.7255))
            != gc._cosmology_cache_key(a))


def test_cached_exact_density_equals_a_fresh_evaluation():
    a = FlatLambdaCDM(67.9, 0.3065)
    b = FlatLambdaCDM(67.9 + 1e-9, 0.3065)
    dL = np.array([10.0, 400.0, 3000.0, 9000.0])
    saved = dict(gc._EX_CACHE)
    try:
        gc._EX_CACHE.clear()
        fresh_b = gc.usf_prob_exact(dL, b, 10.0, 10000.0)
        gc._EX_CACHE.clear()
        gc.usf_prob_exact(dL, a, 10.0, 10000.0)      # populate with a first
        cached_b = gc.usf_prob_exact(dL, b, 10.0, 10000.0)
    finally:
        gc._EX_CACHE.clear()
        gc._EX_CACHE.update(saved)
    np.testing.assert_array_equal(cached_b, fresh_b)
    # ... and b genuinely differs from a (by ~dH0/H0), so reuse would show.
    pa = gc.usf_prob_exact(dL, a, 10.0, 10000.0)
    assert np.any(cached_b != pa)


# ==========================================================================
# 2. distance.dl honours the store's implementation
# ==========================================================================
@pytest.mark.parametrize("impl,expect", [("", "exact"), ("exact", "exact"),
                                         ("analytic", "exact"),
                                         ("astropy", "astropy")])
def test_distance_block_uses_the_rows_impl(impl, expect):
    cosmo = gc.LAL_PLANCK15
    dL = np.array([150.0, 800.0, 2500.0, 7000.0])
    ctx = PEContext(dL_prior_kind="UniformSourceFrame",
                    dL_prior_bounds=(10.0, 10000.0), cosmology=cosmo,
                    dL_prior_impl=impl)
    got = ln_prior_pe({"dL": dL}, ctx)
    ref = np.log(gc.dL_prior_prob(dL, kind="UniformSourceFrame", dmin=10.0,
                                  dmax=10000.0, cosmology=cosmo, impl=expect))
    np.testing.assert_array_equal(got, ref)


def test_distance_block_legacy_bilby_when_available():
    pytest.importorskip("bilby.gw.prior")
    cosmo = gc.LAL_PLANCK15
    dL = np.array([150.0, 800.0, 2500.0, 7000.0])
    ctx = PEContext(dL_prior_kind="UniformSourceFrame",
                    dL_prior_bounds=(10.0, 10000.0), cosmology=cosmo,
                    dL_prior_impl="bilby")
    got = ln_prior_pe({"dL": dL}, ctx)
    ref = np.log(gc.dL_prior_prob(dL, kind="UniformSourceFrame", dmin=10.0,
                                  dmax=10000.0, cosmology=cosmo, impl="bilby"))
    np.testing.assert_array_equal(got, ref)
    exact = np.log(gc.dL_prior_prob(dL, kind="UniformSourceFrame", dmin=10.0,
                                    dmax=10000.0, cosmology=cosmo))
    assert np.any(got != exact)


# ==========================================================================
# 3. Validator and builder
# ==========================================================================
def test_mixed_prior_impl_events():
    names = ["A", "B", "C", "D"]
    dl = ["exact", "bilby", "analytic", ""]
    assert mixed_prior_impl_events("exact", dl, names) == ["B"]
    assert mixed_prior_impl_events("grid", dl, names) == ["A"]
    assert mixed_prior_impl_events(b"exact", [b"astropy"], [b"E"]) == ["E"]
    assert mixed_prior_impl_events(None, dl, names) == []


def _collect():
    fails, results = [], {}

    def _fail(name, msg):
        fails.append((name, msg))
        raise ValueError(f"{name}: {msg}")
    return fails, results, _fail


def test_one_sided_record_fails_under_strict_and_warns_otherwise():
    fails, results, _fail = _collect()
    with pytest.warns(UserWarning, match="one side only"):
        _xcheck_chi_eff_prior_impl(_fail, results,
                                   {"chi_eff_prior_impl": "exact"}, {})
    assert not fails
    with pytest.raises(ValueError, match="xcheck_chi_eff_prior_impl_recorded"):
        _xcheck_chi_eff_prior_impl(_fail, results, {},
                                   {"chi_eff_prior_impl": "exact"},
                                   strict=True)


def test_require_exact_refuses_a_legacy_selection():
    fails, results, _fail = _collect()
    pe = {"chi_eff_prior_impl": "grid"}
    sel = {"chi_eff_prior_impl": "grid", "spin_basis": "chieff_reference"}
    _xcheck_chi_eff_prior_impl(_fail, results, pe, sel)       # consistent
    with pytest.raises(ValueError, match="xcheck_chi_eff_prior_impl_exact"):
        _xcheck_chi_eff_prior_impl(_fail, results, pe, sel,
                                   require_exact=True)
    # A component-basis selection has no chi_eff factor to require.
    _xcheck_chi_eff_prior_impl(_fail, results, {}, {"spin_basis": "component"},
                               require_exact=True)


def _pe_attrs(chi, dl, basis="chieff"):
    a = {"spin_basis": basis, "event_names": np.array(
        [f"GW{i}" for i in range(len(dl))], dtype=object),
         "dL_prior_impl_per_event": np.array(dl, dtype=object)}
    if chi is not None:
        a["chi_eff_prior_impl"] = chi
    return a


def test_pe_mixed_family_warns_and_fails_under_strict():
    fails, results, _fail = _collect()
    mixed = _pe_attrs("exact", ["exact", "bilby"])
    with pytest.warns(UserWarning, match="neither the exact product"):
        _check_pe_prior_impl(_fail, results, mixed)
    with pytest.raises(ValueError, match="pe_prior_impl_consistent"):
        _check_pe_prior_impl(_fail, results, mixed, strict=True)
    # Both legacy families together: a faithful legacy regression.
    _check_pe_prior_impl(_fail, results, _pe_attrs("grid", ["bilby", "analytic"]),
                         strict=True)
    # Both exact.
    _check_pe_prior_impl(_fail, results,
                         _pe_attrs("exact", ["exact", "analytic"]),
                         strict=True, require_exact=True)
    assert results["pe_prior_impl_exact"] is True


@pytest.mark.parametrize("attrs", [
    _pe_attrs("grid", ["exact", "exact"]),                # legacy chi_eff
    _pe_attrs("grid", ["bilby", "bilby"]),                # legacy both
    _pe_attrs("exact", ["exact", ""]),                    # unrecorded dL
    _pe_attrs(None, ["exact", "exact"]),                  # unrecorded chi_eff
    {"spin_basis": "chieff", "chi_eff_prior_impl": "exact"},  # no dL record
])
def test_require_exact_refuses_non_exact_pe(attrs):
    fails, results, _fail = _collect()
    with pytest.raises(ValueError, match="pe_prior_impl_exact"):
        _check_pe_prior_impl(_fail, results, attrs, require_exact=True)


def test_require_exact_ignores_chi_eff_on_a_component_pe():
    fails, results, _fail = _collect()
    _check_pe_prior_impl(_fail, results,
                         _pe_attrs(None, ["exact"], basis="component"),
                         require_exact=True)
    assert not fails


def _store_with_dL_impl(tmp_path, impls):
    events = [{"name": f"GWr{i}_00000{i}"} for i in range(1, len(impls) + 1)]
    store, _ = _build_spin_store(tmp_path, events, n_per_event=200)
    with h5py.File(store, "r+") as f:
        f["meta"].create_dataset("dL_prior_impl", data=np.array(
            impls, dtype=h5py.string_dtype()))
    return store


def test_end_to_end_require_exact(tmp_path):
    store = _store_with_dL_impl(tmp_path, ["exact", "exact"])
    cat = GWCatalog(store)
    kw = dict(format="gwcat2", spin_basis="chieff", nsamp=64, seed=0,
              cosmology=_COSMO)
    exact, legacy = tmp_path / "pe_exact.h5", tmp_path / "pe_grid.h5"
    cat.export(str(exact), **kw)
    with pytest.warns(UserWarning, match="mixes prior implementations"):
        with chi_eff_prior_impl("grid"):
            cat.export(str(legacy), **kw)

    r = validate_export_v2(str(exact), require_exact_priors=True)
    assert r["pe_prior_impl_exact"] is True
    with pytest.raises(ValueError, match="pe_prior_impl_exact"):
        validate_export_v2(str(legacy), require_exact_priors=True)
    with pytest.raises(ValueError, match="pe_prior_impl_consistent"):
        validate_export_v2(str(legacy), strict=True)

    # CLI: exit 0 / 1.
    assert main(["validate", str(exact), "--require-exact-priors"]) == 0
    assert main(["validate", str(legacy), "--require-exact-priors"]) == 1


def test_builder_warns_on_an_exact_export_of_a_legacy_store(tmp_path):
    store = _store_with_dL_impl(tmp_path, ["bilby", "bilby"])
    out = tmp_path / "pe.h5"
    with pytest.warns(UserWarning, match="mixes prior implementations"):
        GWCatalog(store).export(str(out), format="gwcat2", spin_basis="chieff",
                                nsamp=64, seed=0, cosmology=_COSMO)
    # The legacy regression (grid over bilby) is consistent: no such warning.
    import warnings as _w
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        with chi_eff_prior_impl("grid"):
            GWCatalog(store).export(str(tmp_path / "pe2.h5"), format="gwcat2",
                                    spin_basis="chieff", nsamp=64, seed=0,
                                    cosmology=_COSMO)
    assert not [w for w in rec if "mixes prior implementations" in str(w.message)]
