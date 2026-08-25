"""Regressions for the 2026-08-24 review batch (GW-37).

Each test pins one defect the review found, and each was written against the
code BEFORE the fix -- so a test here failing means the specific behaviour it
names has come back, not merely that something nearby changed.

The batch's unifying theme is that a defect fixed on the v2 export path was
left standing on the v1 path (GW-25's "recurring failure mode"), or that a
declaration was written and never read.  The tests are grouped that way.
"""
import h5py
import numpy as np
import pytest

from gwcat.catalog import GWCatalog
from gwcat.selection import SelectionSet

from test_export_v2 import _build_store, MIXED_EVENTS
from test_selection_spin import write_o4_full

_COSMO = (67.74, 0.3089)


# ======================================================================
# 1. compact_type is a posterior-median class cut and must record itself
# ======================================================================
def test_compact_type_records_the_median_cut_estimator(tmp_path):
    """``select(compact_type=...)`` used to record cut_estimator="none".

    The store's ``compact_type`` column is filled at ingest from
    ``classify_by_mass`` on the posterior MEDIAN source-frame masses, exactly
    like ``source_class`` -- so the two spellings are the same cut, and the one
    that recorded nothing routed around every guard GW-12/GW-33 built: the file
    paired clean against an all-class selection function.
    """
    from gwcat.source_class import CUT_ESTIMATOR_POSTERIOR_MEDIAN

    cat = GWCatalog(_build_store(tmp_path))
    spec = cat.select(compact_type="BBH").selection_spec

    assert spec.cut_estimator == CUT_ESTIMATOR_POSTERIOR_MEDIAN
    assert spec.source_class_filter != "", \
        "a class-cut view described itself as unfiltered"
    assert spec.is_filtered is True
    attrs = spec.to_attrs()
    assert attrs["source_class_cut_estimator"] == CUT_ESTIMATOR_POSTERIOR_MEDIAN


def test_compact_type_and_source_class_agree_on_the_estimator(tmp_path):
    """The two spellings of one cut must not describe themselves differently."""
    cat = GWCatalog(_build_store(tmp_path))
    by_compact = cat.select(compact_type="BBH").selection_spec
    by_class = cat.select(source_class="bbh").selection_spec
    assert by_compact.cut_estimator == by_class.cut_estimator


# ======================================================================
# 2. The v1 PE writer reports the mass prior it HAS, not a constant
# ======================================================================
def test_v1_export_does_not_stamp_a_verified_basis_on_unverified_rows(
        tmp_path):
    """``to_darksirens`` stamped mass_prior_basis="uniform_detector_frame".

    It was a literal, written regardless of what was ingested, so the 9 of 282
    rows in the shipped store whose analytic prior was never parsed were
    exported as verified uniform priors -- and the block gate that exists to
    decide this never ran on the v1 path at all.
    """
    cat = GWCatalog(_build_store(tmp_path))  # store states no mass_prior_kind
    out = tmp_path / "v1.h5"
    with pytest.warns(UserWarning, match="no VERIFIED uniform detector-frame"):
        cat.to_darksirens(str(out), nsamp=16, seed=0, cosmology=_COSMO,
                          far_max=1.0, allow_missing_far=True)
    with h5py.File(out, "r") as f:
        assert f.attrs["mass_prior_basis"] == "unstated"
        assert bool(f.attrs["mass_prior_verified"]) is False


def test_v1_export_refuses_an_unsupported_mass_prior(tmp_path):
    """A prior the m1det Jacobian does not describe must stop the export.

    v2 raised on this; v1 multiplied m1det in anyway.  For a prior flat in
    (Mc, q) the correct Jacobian is m1-independent, so the m1det factor
    mis-weights samples by 2x across m1det = 30 -> 60 Msun, per sample.
    """
    store = _build_store(tmp_path)
    with h5py.File(store, "r+") as f:
        n = f["index/event_names"].shape[0]
        f["meta"].create_dataset(
            "mass_prior_kind",
            data=np.array(["uniform_chirp_mass_q"] * n,
                          dtype=h5py.string_dtype()))
    with pytest.raises(ValueError, match="NOT uniform in the detector-frame"):
        GWCatalog(store).to_darksirens(
            str(tmp_path / "v1_bad.h5"), nsamp=16, seed=0, cosmology=_COSMO,
            far_max=1.0, allow_missing_far=True)


# ======================================================================
# 3. The v1 validator checks the detection cut
# ======================================================================
def test_v1_validator_fails_on_a_detection_cut_mismatch(tmp_path):
    """The v1 cross-check block never read far_threshold at all.

    Both files already exported the number; the check was simply never written,
    so a PE file cut at FAR < 2/yr paired clean with injections detected at
    FAR < 1/yr -- beta computed under a different threshold than the events.
    """
    from gwcat.catalog import validate_export as validate_v1

    # No NaN-FAR events: the allow_missing_far leg of the same check would
    # fire first and mask what this test is about.
    events = [e for e in MIXED_EVENTS if np.isfinite(e["far"])]
    cat = GWCatalog(_build_store(tmp_path, events=events, name="farok.h5"))
    pe = tmp_path / "pe_v1.h5"
    cat.to_darksirens(str(pe), nsamp=16, seed=0, cosmology=_COSMO,
                      far_max=1.0)

    sel_src = write_o4_full(tmp_path / "inj.hdf", n=200, amax=(0.9, 0.9),
                            seed=3)
    sel = tmp_path / "sel_v1.h5"
    SelectionSet(sel_src).to_darksirens(str(sel), far_threshold=1.0)

    # Same threshold: the pair is checkable and passes the new check.
    res = validate_v1(str(pe), str(sel))
    assert res.get("xcheck_detection_cut") is True

    # Now move the PE cut and the pair must be refused.
    with h5py.File(pe, "r+") as f:
        f.attrs["far_max"] = 2.0
    with pytest.raises(ValueError, match="detection cut"):
        validate_v1(str(pe), str(sel))


# ======================================================================
# 4. sky_area_max cannot record a cut it did not apply
# ======================================================================
def test_sky_area_max_refuses_a_store_without_the_column(tmp_path):
    """The cut was guarded by a presence check; the RECORD was not.

    So a store with no sky_area_90 kept every event while the spec claimed the
    threshold -- and a paired injection product built to the same criterion
    compared EQUAL on that contract field.
    """
    cat = GWCatalog(_build_store(tmp_path))  # no sky_area_90 column
    with pytest.raises(ValueError, match="sky_area_90"):
        cat.select(sky_area_max=100.0)


def test_sky_area_max_refuses_an_all_nan_column(tmp_path):
    """Silently returning zero events is a configuration error, not a cut.

    The sibling pastro_min cut raises a named error for exactly this state
    (GW-14); this one emptied the catalog with no message at all.  gwcat writes
    NaN sky areas whenever healpy is unavailable, which is not a declared
    dependency, so the state is reachable in normal use.
    """
    store = _build_store(tmp_path)
    with h5py.File(store, "r+") as f:
        n = f["index/event_names"].shape[0]
        f["meta"].create_dataset("sky_area_90",
                                 data=np.full(n, np.nan, dtype=float))
    with pytest.raises(ValueError, match="sky_area_90.* NaN|NaN for every"):
        GWCatalog(store).select(sky_area_max=100.0)


# ======================================================================
# 6. The v1 selection exporters share the v2 physics
# ======================================================================
def test_v1_selection_export_refuses_the_ungated_chieff_swap(tmp_path):
    """v1 applied the swap to ANY campaign; v2 has gated it since GW-06.

    The swap divides out the injected spin prior and multiplies in the analytic
    marginal of a uniform-magnitude/isotropic draw.  On a campaign that did not
    draw that way the exported pdraw is wrong by an O(1), chi_eff-DEPENDENT
    factor -- measured on the shipped O4ab file as an empirical/assumed density
    ratio running 0.06 to 1.48 across the band -- and that does not cancel
    between injections, so it moves posteriors rather than just log Z.
    """
    from gwcat.export.selection_builder import (BlockCampaignMismatch,
                                                _check_chieff_swap_valid)
    from gwcat.selection import _v1_chieff_swap

    # The gate the v1 exporters now call is the SAME function v2 calls; pin
    # that they share it rather than re-implementing the check.
    import gwcat.selection as sel_mod
    assert _check_chieff_swap_valid.__module__ == \
        "gwcat.export.selection_builder"

    from test_export_v2_selection import _fake_set
    bad = _fake_set("o4ab.hdf", uniform=False,
                    checks={"magnitude_uniform": (False, False),
                            "isotropy_dev": (0.6419, 0.6419)})
    with pytest.raises(BlockCampaignMismatch):
        _v1_chieff_swap([bad], [np.ones(3, dtype=bool)], 0.99, strict=True)


def test_v1_selection_export_uses_the_campaigns_own_ceiling(tmp_path):
    """v1 forced one caller amax on every campaign; "auto" is now the default.

    endo3 injects a ~ U(0, 0.998) and the old default evaluated its chi_eff
    marginal at 0.99, which is a chi_eff-dependent reweighting of injections
    against one another -- not a constant that divides out of mu.
    """
    inj = write_o4_full(tmp_path / "inj.hdf", n=200, amax=(0.998, 0.998),
                        seed=5)
    out = tmp_path / "sel_auto.h5"
    SelectionSet(inj).to_darksirens(str(out), far_threshold=1.0)
    with h5py.File(out, "r") as f:
        assert f.attrs["chi_eff_amax_mode"] == "per_campaign"
        assert f.attrs["chi_eff_amax_source_per_campaign"][0] == "detected"
        np.testing.assert_allclose(f.attrs["chi_eff_amax_1_per_campaign"],
                                   [0.998], rtol=2e-2)

    # A numeric amax still forces one ceiling, and says so.
    out_fixed = tmp_path / "sel_fixed.h5"
    SelectionSet(inj).to_darksirens(str(out_fixed), far_threshold=1.0,
                                    amax=0.99)
    with h5py.File(out_fixed, "r") as f:
        assert f.attrs["chi_eff_amax_mode"] == "fixed"
        assert f.attrs["chi_eff_amax"] == 0.99
        assert f.attrs["chi_eff_amax_source_per_campaign"][0] == "caller"


def test_v1_and_v2_selection_exports_agree_at_their_defaults(tmp_path):
    """The parity the module docstring claims, now that both default to auto."""
    from gwcat.export import build_selection_product

    inj = write_o4_full(tmp_path / "inj.hdf", n=200, amax=(0.998, 0.998),
                        seed=7)
    v1 = tmp_path / "v1.h5"
    SelectionSet(inj).to_darksirens(str(v1), far_threshold=1.0)
    product = build_selection_product(SelectionSet(inj), spin_basis="chieff",
                                      far_threshold=1.0)
    with h5py.File(v1, "r") as f:
        np.testing.assert_array_equal(np.asarray(f["pdraw"]),
                                      product.columns["pdraw"])


def test_chieff_support_is_the_priors_own_predicate_not_isfinite():
    """``isfinite(logprob)`` is not a support test -- the grid clamp fakes it.

    At amax=0.99 a sample at chi_eff=0.995 with m1=50, m2=25 comes back at a
    FINITE 2.73e-12 while ``support()`` is False, so a "finite" test admitted
    excluded samples carrying a density ~1e12 too small -- an inverse weight
    ~1e12 too LARGE in the denominator.
    """
    from gwcat.spin import (chi_eff_prior_logprob,
                            chi_eff_prior_logprob_in_support)

    unmasked = np.asarray(chi_eff_prior_logprob([0.995], 50.0, 25.0,
                                                amax=0.99), dtype=float)
    assert np.all(np.isfinite(unmasked)), \
        "the premise changed: logprob no longer returns a finite value here"

    logp, sup = chi_eff_prior_logprob_in_support([0.995], 50.0, 25.0,
                                                 amax=0.99)
    assert not sup[0]
    assert logp[0] == -np.inf


