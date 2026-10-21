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


# ======================================================================
# A cumulative mixture must detect something in every PE event's run
# ======================================================================
def _mixture_sel_o1o2_blind(tmp_path):
    from test_selection_cumulative_mixture import (write_cumulative_mixture,
                                                   _ANCHORS)
    path, _, _ = write_cumulative_mixture(tmp_path / "mix.hdf")
    out = tmp_path / "sel_mix_ack.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        SelectionSet(path).export(str(out), spin_basis="component",
                                  snr_threshold=None,
                                  acknowledge_semianalytic_excluded=True,
                                  mixture_anchors=_ANCHORS)
    return out


def _pe_named(tmp_path, names, tag):
    events = [{"name": n, "amax1": 0.99, "amax2": 0.99} for n in names]
    store, _ = _build_spin_store(tmp_path, events, name=f"store_{tag}.h5")
    out = tmp_path / f"pe_{tag}.h5"
    GWCatalog(store).export(str(out), format="gwcat2", spin_basis="component",
                            nsamp=32, seed=0, cosmology=_COSMO)
    return out


def test_o1o2_blind_mixture_refuses_a_pe_file_with_an_o1_event(tmp_path):
    """The acknowledged FAR-only mixture export detects nothing in O1/O2; it
    passed --strict against a PE file holding O1/O2 events."""
    sel = _mixture_sel_o1o2_blind(tmp_path)
    pe = _pe_named(tmp_path, ["GW150914_095045", "GW200129_065458"], "o1")
    with pytest.raises(ValueError) as ei:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            validate_export_v2(str(pe), str(sel), strict=True)
    msg = str(ei.value)
    assert "xcheck_mixture_event_runs" in msg
    assert "GW150914_095045" in msg and "O1" in msg
    assert "GW200129_065458" not in msg


def test_o1o2_blind_mixture_accepts_an_o3_only_pe_file(tmp_path):
    sel = _mixture_sel_o1o2_blind(tmp_path)
    pe = _pe_named(tmp_path, ["GW200129_065458", "GW190521_030229"], "o3")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = validate_export_v2(str(pe), str(sel))
    assert res["xcheck_mixture_event_runs"] is True


def test_event_names_map_to_their_official_runs():
    from gwcat.export.validate import _event_gps_span, _event_run

    # GW150914's GPS time is 1126259462.4; the name carries whole seconds.
    assert _event_gps_span("GW150914_095045") == (1126259462.0, 1126259462.0)
    assert _event_run("GW150914_095045") == "O1"
    assert _event_run("GW170817_124104") == "O2"
    assert _event_run("GW230529_181500") == "O4a"
    assert _event_run("GW150914") == "O1"          # a whole day inside O1
    assert _event_gps_span("GWv0_000001") is None


# ======================================================================
# One event, one block: duplicate event names fail
# ======================================================================
def test_all_policy_export_with_one_event_three_times_fails(tmp_path):
    """waveform_policy='all' on a store with several sample sets per event
    writes that event once per set; validate --strict passed, and the event
    digest equals the de-duplicated whitelist's."""
    from test_waveform_policy import build_multiset_store, _one_event_three_sets
    from gwcat.export.contract import event_list_digest

    store = build_multiset_store(tmp_path / "multi.h5",
                                 _one_event_three_sets(), seed=2)
    out = tmp_path / "pe_all.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        GWCatalog(store).export(str(out), format="gwcat2", spin_basis="nospin",
                                waveform_policy="all", nsamp=8, seed=0,
                                cosmology=_COSMO)
    a = _attrs(out)
    assert int(a["nobs"]) == 3 and bool(a["homogeneous_sample_sets"]) is False
    assert a["event_list_digest"] == event_list_digest(["GW850101_010101"])
    res = validate_export_v2(str(out))
    assert res["pe_event_names_unique"] is False
    assert res["pe_homogeneous_sample_sets"] is False
    with pytest.raises(AssertionError, match="pe_event_names_unique"):
        validate_export_v2(str(out), strict=True)


def test_event_names_must_match_nobs(tmp_path):
    pe = _pe_component(tmp_path, name="pe_len.h5")
    with h5py.File(pe, "r+") as f:
        names = list(f.attrs["event_names"])
        f.attrs["event_names"] = np.array(names[:1], dtype=h5py.string_dtype())
    res = validate_export_v2(str(pe))
    assert res["pe_event_names_length"] is False


# ======================================================================
# Cosmology: the per-campaign record, no vacuous pass, z_max in distance
# ======================================================================
def _set_campaign_cosmology(sel, H0, Om0, scalar=(67.74, 0.3075)):
    """Make the selection file look like the production one: a scalar that is
    the constructor default, and one campaign that used (H0, Om0)."""
    with h5py.File(sel, "r+") as f:
        f.attrs["cosmology_H0"], f.attrs["cosmology_Om0"] = scalar
        f.attrs["cosmology_source_per_campaign"] = np.array(
            ["detected"], dtype=h5py.string_dtype())
        f.attrs["cosmology_H0_per_campaign"] = np.array([H0])
        f.attrs["cosmology_Om0_per_campaign"] = np.array([Om0])


def _set_pe_cosmology(pe, H0, Om0):
    with h5py.File(pe, "r+") as f:
        f.attrs["cosmology_per_event_varies"] = False
        f.attrs["pe_cosmology_H0"], f.attrs["pe_cosmology_Om0"] = H0, Om0


def test_cosmology_is_compared_with_the_campaign_not_the_scalar(tmp_path):
    """A PE export within 1 of the campaign's generation H0 was refused
    because it is more than 1 from the scalar, which no campaign used."""
    pe = _pe_component(tmp_path, name="pe_cos.h5")
    sel = _sel(tmp_path, name="sel_cos.h5")
    _set_campaign_cosmology(sel, 67.9, 0.3065)
    _set_pe_cosmology(pe, 68.8, 0.3065)
    res = validate_export_v2(str(pe), str(sel))
    assert res["xcheck_cosmology"] is True
    # ...and a PE cosmology really far from the campaign's still fails, with
    # no advice to re-export the selection under another cosmology.
    _set_pe_cosmology(pe, 70.0, 0.3065)
    with pytest.raises(ValueError, match="xcheck_cosmology") as ei:
        validate_export_v2(str(pe), str(sel))
    assert "per-campaign" in str(ei.value)
    assert "Re-export the selection" not in str(ei.value)


def test_varying_pe_cosmology_without_per_event_arrays_fails(tmp_path):
    """With cosmology_per_event_varies=True and no per-event arrays the check
    ran over empty arrays and passed."""
    pe = _pe_component(tmp_path, name="pe_var.h5")
    sel = _sel(tmp_path, name="sel_var.h5")
    _set_campaign_cosmology(sel, 50.0, 0.3)
    with h5py.File(pe, "r+") as f:
        f.attrs["cosmology_per_event_varies"] = True
        for k in ("cosmology_H0_per_event", "cosmology_Om0_per_event"):
            if k in f.attrs:
                del f.attrs[k]
    with pytest.raises(ValueError, match="cosmology_per_event_varies"):
        validate_export_v2(str(pe), str(sel))


def _set_z_max(path, z):
    with h5py.File(path, "r+") as f:
        f.attrs["z_max"] = float(z)


def test_equal_z_max_under_different_cosmologies_fails(tmp_path):
    """z_max = 1 on both sides truncated the PE samples at dL = 6798.0 Mpc
    and the injections at 6785.9 Mpc, and was reported as matched."""
    pe = _pe_component(tmp_path, name="pe_z.h5")
    sel = _sel(tmp_path, name="sel_z.h5")
    _set_pe_cosmology(pe, 67.74, 0.3075)
    _set_campaign_cosmology(sel, 67.9, 0.3065)
    for p in (pe, sel):
        _set_z_max(p, 1.0)
    with pytest.raises(ValueError, match="xcheck_z_max_distance") as ei:
        validate_export_v2(str(pe), str(sel))
    assert "6798.0" in str(ei.value) and "6785.9" in str(ei.value)
    # One cosmology on both sides: one truncation, and it passes.
    _set_pe_cosmology(pe, 67.9, 0.3065)
    res = validate_export_v2(str(pe), str(sel))
    assert res["xcheck_z_max_distance"] is True


# ======================================================================
# Recorded out-of-support zeros are not a truncated distance prior
# ======================================================================
def _pe_out_of_support(tmp_path):
    events = [{"name": "GWv0_000001"}, {"name": "GWv0_000002"}]
    store, _ = _build_spin_store(tmp_path, events, name="store_oos.h5")
    out = tmp_path / "pe_oos.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        GWCatalog(store).export(str(out), format="gwcat2", spin_basis="chieff",
                                nsamp=64, seed=0, cosmology=_COSMO, amax=0.2,
                                allow_out_of_support=True)
    return out


def test_allow_out_of_support_export_validates(tmp_path):
    """A chieff export at amax 0.2 with allow_out_of_support=True writes
    recorded zeros, and `gwcat validate` failed it with the distance-prior
    diagnosis, so no product built this way could ever validate."""
    pe = _pe_out_of_support(tmp_path)
    a = _attrs(pe)
    assert int(a["n_samples_out_of_support"]) > 0
    res = validate_export_v2(str(pe), strict=True)
    assert res["pe_p_pe_positive"] is True
    assert res["pe_out_of_support_recorded"] is True
    assert res["pe_out_of_support_p_pe_zero"] is True


def test_in_support_zero_still_fails(tmp_path):
    pe = _pe_out_of_support(tmp_path)
    with h5py.File(pe, "r+") as f:
        p, ins = f["p_pe"][()], f["in_support"][()].astype(bool)
        p[int(np.flatnonzero(ins)[0])] = 0.0
        f["p_pe"][...] = p
    res = validate_export_v2(str(pe))
    assert res["pe_p_pe_positive"] is False


def test_unrecorded_out_of_support_rows_fail(tmp_path):
    pe = _pe_out_of_support(tmp_path)
    with h5py.File(pe, "r+") as f:
        f.attrs["n_samples_out_of_support"] = 0
    res = validate_export_v2(str(pe))
    assert res["pe_out_of_support_recorded"] is False
