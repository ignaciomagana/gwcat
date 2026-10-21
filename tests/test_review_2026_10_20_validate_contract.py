"""Regression tests for the 2026-10-20 review of the validator and the 2.1
pairing contract (package "validate_contract").

Each test is the finding's failure scenario: it fails on gwcat 3ba1dbf and
passes with the fix.  Fixtures are reused from the sibling validator tests.
"""
import json
import warnings

import numpy as np
import h5py
import pytest

from gwcat.catalog import GWCatalog
from gwcat.selection import SelectionSet
from gwcat.export import validate_export_v2
from gwcat.export.contract import (PAIRING_FIELDS, build_contract,
                                   contract_hash, selection_contract_fields)

from test_export_v2_pe import _build_spin_store
from test_selection_spin import write_o4_full
from test_validate_v2 import _pe_component, _sel, _COSMO


def _attrs(path):
    with h5py.File(path, "r") as f:
        return {k: (v.decode() if isinstance(v, bytes) else v)
                for k, v in f.attrs.items()}


# ======================================================================
# The class cut's estimator and threshold reach the 2.1 hash
# ======================================================================
def test_median_vs_truth_class_cut_pair_hashes_differently(tmp_path):
    """`export pe --format gwcat2.1 --source-class bbh` with `export selection
    --format gwcat2.1 --source-class bbh`: gwcat validate refuses the pair, but
    the two contract hashes were equal, so a consumer that compares only the
    hash (darksirens-core) bound it."""
    events = [{"name": "GWv0_000001", "amax1": 0.99, "amax2": 0.99},
              {"name": "GWv0_000002", "amax1": 0.90, "amax2": 0.90}]
    store, _ = _build_spin_store(tmp_path, events, name="store_cls")
    pe = tmp_path / "pe_cls21.h5"
    with pytest.warns(UserWarning, match="POSTERIOR MEDIAN"):
        GWCatalog(store).export(str(pe), format="gwcat2.1",
                                spin_basis="component", nsamp=16, seed=0,
                                cosmology=_COSMO, source_class="bbh")
    inj = write_o4_full(tmp_path / "inj_cls.hdf", n=60, amax=(0.9, 0.9),
                        seed=6)
    sel = tmp_path / "sel_cls21.h5"
    SelectionSet(inj).export(str(sel), format="gwcat2.1",
                             spin_basis="component", source_class="bbh")
    a_pe, a_sel = _attrs(pe), _attrs(sel)
    c_pe, c_sel = json.loads(a_pe["contract"]), json.loads(a_sel["contract"])
    assert c_pe["source_class"] == c_sel["source_class"] == "BBH"
    assert c_pe["source_class_cut_estimator"] == "posterior_median_mass"
    assert c_sel["source_class_cut_estimator"] == "injected_truth"
    assert c_sel["source_class_mass_threshold"] == 3.0
    assert a_pe["contract_hash"] != a_sel["contract_hash"]
    # ...and the validator still names the estimator (it runs first).
    with pytest.raises(ValueError, match="xcheck_source_class_estimator"):
        validate_export_v2(str(pe), str(sel))


def test_unfiltered_sides_state_no_class_cut():
    """A name-whitelisted PE file and an unfiltered selection file are a
    correct pair whose raw estimators differ ('name_whitelist' vs 'none'), so
    the raw estimator must not reach the hash on an unfiltered side."""
    pe = selection_contract_fields({"source_class_filter": "",
                                    "source_class_cut_estimator":
                                        "name_whitelist"})
    sel = selection_contract_fields({"source_class_filter": "",
                                     "source_class_cut_estimator": "none",
                                     "nsbh_mass_threshold": 3.0})
    assert pe["source_class_cut_estimator"] is None
    assert sel["source_class_cut_estimator"] is None
    assert sel["source_class_mass_threshold"] is None
    assert (contract_hash(build_contract(parameter_space="component", **pe))
            == contract_hash(build_contract(parameter_space="component",
                                            **sel)))


def test_contract_hash_of_a_contract_without_the_new_fields_is_unchanged():
    """The fields added to the digest enter it only when set, so a 2.1 file
    written before them (no class filter, no truncation) still pairs with
    one written now.  The digests are the ones gwcat 3ba1dbf computes."""
    a = build_contract(parameter_space="component",
                       fit_columns=["m1det", "q", "dL"],
                       advisory_columns=["m2det"], spin_basis_kind="bijective",
                       spin_density_exact=True, sky_prior_in_density=True,
                       source_class=None, z_max=None, cosmology_H0=67.74)
    assert contract_hash(a) == "86d8cb4abcce3488"
    b = build_contract(parameter_space="chieff",
                       fit_columns=["m1det", "q", "dL", "chieff"],
                       advisory_columns=["m2det"],
                       spin_basis_kind="projection", spin_density_exact=False,
                       sky_prior_in_density=True, source_class="BBH")
    assert contract_hash(b) == "96e34ec1677e2f07"


def test_z_max_reaches_the_hash():
    """The validator fails two stated truncations that differ; the hash did
    not cover z_max, so they hashed equal."""
    assert "z_max" in PAIRING_FIELDS
    trunc = build_contract(parameter_space="component", z_max=1.0)
    full = build_contract(parameter_space="component", z_max=None)
    other = build_contract(parameter_space="component", z_max=0.8)
    assert contract_hash(trunc) != contract_hash(full)
    assert contract_hash(trunc) != contract_hash(other)
    assert contract_hash(trunc) == contract_hash(
        build_contract(parameter_space="component", z_max=1.0))


# ======================================================================
# chieff_reference: the sentinel rows are exactly the out-of-reference rows
# ======================================================================
from test_validate_v2 import _pe_chieff_at, _sel_reference  # noqa: E402


def _ref_pair(tmp_path):
    pe = _pe_chieff_at(tmp_path, 0.99, name="pe_sent.h5")
    sel = _sel_reference(tmp_path, 0.99, name="sel_sent.h5")
    return pe, sel


def test_reference_sentinel_on_an_in_support_row_fails(tmp_path):
    """A builder or merge bug that puts the sentinel on an in-support
    injection biases beta low; it passed --strict."""
    pe, sel = _ref_pair(tmp_path)
    validate_export_v2(str(pe), str(sel), strict=True)      # clean first
    with h5py.File(sel, "r+") as f:
        a1, a2, pd = f["a1"][()], f["a2"][()], f["pdraw"][()]
        k = int(np.flatnonzero((a1 <= 0.99) & (a2 <= 0.99))[0])
        pd[k] = float(f.attrs["spin_reference_excluded_pdraw"])
        f["pdraw"][...] = pd
        f.attrs["spin_reference_excluded_rows"] = int(
            np.sum(pd == f.attrs["spin_reference_excluded_pdraw"]))
    with pytest.raises(AssertionError, match="sel_reference_sentinel_rows"):
        validate_export_v2(str(pe), str(sel), strict=True)


def test_reference_out_of_support_row_with_a_real_weight_fails(tmp_path):
    """An out-of-reference row with a finite weight biases beta high."""
    pe, sel = _ref_pair(tmp_path)
    with h5py.File(sel, "r+") as f:
        a1, pd = f["a1"][()], f["pdraw"][()]
        k = int(np.flatnonzero(pd != f.attrs["spin_reference_excluded_pdraw"])[0])
        a1[k] = 0.995                      # above a_ref, pdraw left real
        f["a1"][...] = a1
    res = validate_export_v2(str(pe), str(sel))
    assert res["sel_reference_sentinel_rows"] is False


def test_reference_excluded_row_count_must_match(tmp_path):
    pe, sel = _ref_pair(tmp_path)
    with h5py.File(sel, "r+") as f:
        f.attrs["spin_reference_excluded_rows"] = (
            int(f.attrs["spin_reference_excluded_rows"]) + 3)
    res = validate_export_v2(str(pe), str(sel))
    assert res["sel_reference_excluded_rows_count"] is False


def test_reference_sentinel_that_is_a_real_weight_fails(tmp_path):
    """A caller-chosen out_of_reference_pdraw of 1.0 gives the excluded rows
    a real weight while the attrs still call them zero-weight."""
    pe, sel = _ref_pair(tmp_path)
    with h5py.File(sel, "r+") as f:
        a1, pd = f["a1"][()], f["pdraw"][()]
        old = float(f.attrs["spin_reference_excluded_pdraw"])
        a1[0] = 0.995
        pd[0] = old
        pd[pd == old] = 1.0
        f["a1"][...] = a1
        f["pdraw"][...] = pd
        f.attrs["spin_reference_excluded_pdraw"] = 1.0
        f.attrs["spin_reference_excluded_rows"] = int(np.sum(pd == 1.0))
    res = validate_export_v2(str(pe), str(sel))
    assert res["sel_reference_sentinel_rows"] is True
    assert res["sel_reference_sentinel_negligible"] is False
