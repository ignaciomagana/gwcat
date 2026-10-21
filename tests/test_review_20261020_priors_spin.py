"""Review 2026-10-20, priors and spin package: regression tests.

Each test is the review's failure scenario, reduced to a synthetic fixture.
The production exports (BBH chieff PE, O3+O4 chieff_reference selection) are
byte-identical in their datasets; these tests pin the behaviour around them.
"""
import json
import warnings

import h5py
import numpy as np
import pytest

from gwcat.catalog import GWCatalog
from gwcat.export.validate import _check_pe_prior_impl, validate_export_v2
from gwcat.release_cosmology import (LEGACY_TABLE_FILENAME, bundled_table_path,
                                     load_release_cosmology_table)

from test_export_v2_pe import _build_spin_store

_COSMO = (67.74, 0.3089)


def _collect():
    fails, results = [], {}

    def _fail(name, msg):
        fails.append(name)
        raise ValueError(f"{name}: {msg}")
    return fails, results, _fail


def _add_meta(store, **cols):
    with h5py.File(store, "r+") as f:
        for k, v in cols.items():
            if k in f["meta"]:
                del f["meta"][k]
            if isinstance(v[0], str):
                f["meta"].create_dataset(k, data=np.array(
                    v, dtype=h5py.string_dtype()))
            else:
                f["meta"].create_dataset(k, data=np.asarray(v, dtype=float))


def _export(store, out, **kw):
    kw = dict(dict(format="gwcat2", spin_basis="chieff", nsamp=64, seed=0,
                   cosmology=_COSMO), **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        GWCatalog(store).export(str(out), **kw)
    with h5py.File(out, "r") as f:
        return dict(f.attrs)


def _strs(v):
    return [x.decode() if isinstance(x, bytes) else str(x)
            for x in np.atleast_1d(v)]


# ==========================================================================
# The release-reweight table's identity reaches the PE product
# ==========================================================================
def _two_event_store(tmp_path, name="store.h5"):
    events = [{"name": "GW190521_030229"}, {"name": "GW230708_053705"}]
    store, _ = _build_spin_store(tmp_path, events, n_per_event=200, name=name)
    return store


def test_pe_product_records_the_release_reweight_table(tmp_path):
    """The failure scenario: a build with the legacy (astropy) table must be
    distinguishable from the file, and --require-exact-priors must refuse it.
    """
    legacy = load_release_cosmology_table(
        bundled_table_path(LEGACY_TABLE_FILENAME))
    store = _two_event_store(tmp_path)
    _add_meta(store,
              dL_prior_cosmology_source=["legacy_gwcat_default",
                                         "analytic_declared"],
              dL_prior_cosmology_table_sha256=[legacy.sha256, ""],
              dL_prior_impl=["exact", "exact"])
    out = tmp_path / "pe.h5"
    attrs = _export(store, out)
    assert _strs(attrs["dL_prior_cosmology_source_per_event"]) == [
        "legacy_gwcat_default", "analytic_declared"]
    assert _strs(attrs["dL_prior_cosmology_table_sha256_per_event"]) == [
        legacy.sha256, ""]
    ids = json.loads(attrs["release_reweight_cosmology_table_ids"])
    assert ids == {legacy.sha256: legacy.table_id}

    fails, results, _fail = _collect()
    with pytest.raises(ValueError, match="regression-only release-reweight"):
        _check_pe_prior_impl(_fail, results, attrs, require_exact=True)
    with pytest.raises(ValueError, match="pe_prior_impl_exact"):
        validate_export_v2(str(out), require_exact_priors=True)


def test_production_table_passes_require_exact(tmp_path):
    prod = load_release_cosmology_table()
    store = _two_event_store(tmp_path)
    _add_meta(store,
              dL_prior_cosmology_source=["documented", "analytic_declared"],
              dL_prior_cosmology_table_sha256=[prod.sha256, ""],
              dL_prior_impl=["exact", "exact"])
    out = tmp_path / "pe.h5"
    attrs = _export(store, out)
    assert json.loads(attrs["release_reweight_cosmology_table_ids"]) == {
        prod.sha256: prod.table_id}
    r = validate_export_v2(str(out), require_exact_priors=True)
    assert r["pe_prior_impl_exact"] is True


def test_a_file_without_the_record_only_warns():
    attrs = {"spin_basis": "chieff", "chi_eff_prior_impl": "exact",
             "event_names": np.array(["GW1"], dtype=object),
             "dL_prior_impl_per_event": np.array(["exact"], dtype=object)}
    fails, results, _fail = _collect()
    with pytest.warns(UserWarning, match="cannot be verified"):
        _check_pe_prior_impl(_fail, results, attrs, require_exact=True)
    assert not fails


# ==========================================================================
# The ingest-time selection reason and the mass prior source reach the product
# ==========================================================================
def test_pe_product_records_the_ingest_selection_reason(tmp_path):
    """Priority-list picks and last-resort picks must not read the same."""
    store = _two_event_store(tmp_path)
    _add_meta(store,
              selection_reason=["preferred_priority", "preferred_last_resort"],
              priority_rank=[0.0, 3.0],
              mass_prior_source=["analytic[C01:IMRPhenomXPHM]",
                                 "constituent_mixture:own_prior[C00:A+C00:B]"])
    attrs = _export(store, tmp_path / "pe.h5")
    assert _strs(attrs["sample_set_ingest_selection_reason_per_event"]) == [
        "preferred_priority", "preferred_last_resort"]
    np.testing.assert_array_equal(attrs["sample_set_priority_rank_per_event"],
                                  [0.0, 3.0])
    assert _strs(attrs["mass_prior_source_per_event"]) == [
        "analytic[C01:IMRPhenomXPHM]",
        "constituent_mixture:own_prior[C00:A+C00:B]"]


def test_store_without_the_columns_records_empty_provenance(tmp_path):
    store = _two_event_store(tmp_path)
    attrs = _export(store, tmp_path / "pe.h5")
    assert _strs(attrs["sample_set_ingest_selection_reason_per_event"]) == [
        "", ""]
    assert np.all(np.isnan(attrs["sample_set_priority_rank_per_event"]))
    assert json.loads(attrs["release_reweight_cosmology_table_ids"]) == {}
