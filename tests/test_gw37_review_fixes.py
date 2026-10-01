"""Regressions for the 2026-08-24 review batch (GW-37).

Each test pins one defect the review found, and each was written against the
code BEFORE the fix -- so a test here failing means the specific behaviour it
names has come back, not merely that something nearby changed.

The batch's unifying theme is that a defect fixed on the v2 export path was
left standing on the v1 path (GW-25's "recurring failure mode"), or that a
declaration was written and never read.  The tests are grouped that way.
"""
import json

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

    # The clamp is a property of the LEGACY grid (GW-40i made the exact prior
    # the default), so the premise is asserted on the grid explicitly.
    unmasked = np.asarray(chi_eff_prior_logprob([0.995], 50.0, 25.0,
                                                amax=0.99, impl="grid"),
                          dtype=float)
    assert np.all(np.isfinite(unmasked)), \
        "the premise changed: logprob no longer returns a finite value here"

    logp, sup = chi_eff_prior_logprob_in_support([0.995], 50.0, 25.0,
                                                 amax=0.99, impl="grid")
    assert not sup[0]
    assert logp[0] == -np.inf

    # The exact prior has no clamp: zero density past the edge, even unmasked,
    # and the gated helper agrees.
    exact = np.asarray(chi_eff_prior_logprob([0.995], 50.0, 25.0, amax=0.99),
                       dtype=float)
    assert exact[0] == -np.inf
    logp, sup = chi_eff_prior_logprob_in_support([0.995], 50.0, 25.0,
                                                 amax=0.99)
    assert not sup[0] and logp[0] == -np.inf


# ======================================================================
# 5. A zero-event export is refused, and an empty p_pe fails validation
# ======================================================================
def test_zero_event_pe_export_is_refused(tmp_path):
    """It used to build, write, validate ALL PASSED and exit 0.

    With nobs=0 every length check is `0 == 0*nsamp` and every array check is
    skipped, so the one tool whose job is to catch this reported success.
    """
    events = [dict(e, far=float("nan")) for e in MIXED_EVENTS]
    cat = GWCatalog(_build_store(tmp_path, events=events, name="allnan.h5"))
    with pytest.raises(ValueError, match="left 0 of"):
        cat.export(str(tmp_path / "empty.h5"), format="gwcat2",
                   spin_basis="chieff", nsamp=16, seed=0, cosmology=_COSMO,
                   far_max=1.0)


def test_v2_validator_fails_on_an_empty_p_pe(tmp_path):
    """The `else` the v1 validator and the v2 selection block both had."""
    from gwcat.export.validate import validate_export_v2

    cat = GWCatalog(_build_store(tmp_path))
    pe = tmp_path / "pe.h5"
    cat.export(str(pe), format="gwcat2", spin_basis="chieff", nsamp=16,
               seed=0, cosmology=_COSMO, far_max=1.0, allow_missing_far=True)
    with h5py.File(pe, "r+") as f:
        del f["p_pe"]
        f.create_dataset("p_pe", data=np.array([], dtype=float))
        f.attrs["nobs"] = 0
    results = validate_export_v2(str(pe))
    assert results["pe_p_pe_nonempty"] is False, (
        "an empty p_pe was not reported at all -- the check vanished rather "
        "than failing, which is what let a zero-event export pass ALL PASSED")


# ======================================================================
# 7. The v2 selection validator checks presence and length
# ======================================================================
def _v2_pair(tmp_path):
    events = [e for e in MIXED_EVENTS if np.isfinite(e["far"])]
    cat = GWCatalog(_build_store(tmp_path, events=events, name="v2store.h5"))
    pe = tmp_path / "pe.h5"
    cat.export(str(pe), format="gwcat2", spin_basis="chieff", nsamp=16,
               seed=0, cosmology=_COSMO, far_max=1.0)
    inj = write_o4_full(tmp_path / "inj.hdf", n=120, amax=(0.9, 0.9), seed=9)
    sel = tmp_path / "sel.h5"
    SelectionSet(inj).export(str(sel), format="gwcat2", spin_basis="chieff",
                             far_threshold=1.0)
    return pe, sel


def test_v2_selection_validator_fails_on_a_missing_dataset(tmp_path):
    """A selection file with every dataset deleted used to pass 56/56.

    The 13 guarded checks VANISHED rather than failing, so `failed` was {} and
    the CLI exited 0 -- a regression against the v1 validator this replaced,
    whose own docstring names that defect as the thing being fixed.
    """
    from gwcat.export.validate import validate_export_v2

    _, sel = _v2_pair(tmp_path)
    with h5py.File(sel, "r+") as f:
        del f["pdraw"]
        del f["ra"]
    results = validate_export_v2(str(tmp_path / "pe.h5"), str(sel))
    assert results["sel_has_pdraw"] is False
    assert results["sel_has_ra"] is False


def test_v2_selection_validator_fails_on_ragged_columns(tmp_path):
    """The dangerous variant: it never raises, so nothing downstream notices.

    A consumer reading columns independently then pairs injection i's draw
    density with injection j's masses and distance.
    """
    from gwcat.export.validate import validate_export_v2

    pe, sel = _v2_pair(tmp_path)
    with h5py.File(sel, "r+") as f:
        m1 = np.asarray(f["m1det"])[:3]
        del f["m1det"]
        f.create_dataset("m1det", data=m1)
    results = validate_export_v2(str(pe), str(sel))
    assert results["sel_m1det_length"] is False


def test_declared_fit_columns_must_exist_as_datasets(tmp_path):
    """A 2.1 selection file declared ``q`` and shipped no ``q`` dataset.

    The values were right -- pdraw genuinely is a density in (m1det, q, dL) --
    which is exactly why no numeric check tripped.  A generic check beats
    extending two more hardcoded lists.
    """
    from gwcat.export.validate import validate_export_v2

    events = [e for e in MIXED_EVENTS if np.isfinite(e["far"])]
    cat = GWCatalog(_build_store(tmp_path, events=events, name="s21.h5"))
    pe = tmp_path / "pe21.h5"
    cat.export(str(pe), format="gwcat2.1", spin_basis="chieff", nsamp=16,
               seed=0, cosmology=_COSMO, far_max=1.0)
    inj = write_o4_full(tmp_path / "inj.hdf", n=120, amax=(0.9, 0.9), seed=4)
    sel = tmp_path / "sel21.h5"
    SelectionSet(inj).export(str(sel), format="gwcat2.1", spin_basis="chieff",
                             far_threshold=1.0)

    # The dataset is emitted now ...
    with h5py.File(sel, "r") as f:
        assert "q" in f
        np.testing.assert_allclose(np.asarray(f["q"]),
                                   np.asarray(f["m2det"])
                                   / np.asarray(f["m1det"]))
        declared = [v.decode() if isinstance(v, bytes) else str(v)
                    for v in f.attrs["fit_columns"]]
        assert "q" in declared

    # ... and removing it is now detected rather than passing 70/70.
    with h5py.File(sel, "r+") as f:
        del f["q"]
    results = validate_export_v2(str(pe), str(sel))
    assert results["sel_fit_columns_present"] is False


# ======================================================================
# 8. z_max has an injection-side counterpart and is recorded
# ======================================================================
def test_z_max_is_recorded_and_must_match_on_both_sides(tmp_path):
    """The truncation appeared in no attr, no summary key, no contract field.

    So a truncated export and the full one it came from compared EQUAL on
    selection_spec_digest, event_list_digest and contract_hash alike, while
    mu kept its full-z content.
    """
    from gwcat.export.validate import validate_export_v2

    cat = GWCatalog(_build_store(tmp_path))
    pe = tmp_path / "pe_zmax.h5"
    with pytest.warns(UserWarning, match="posterior sample"):
        cat.export(str(pe), format="gwcat2", spin_basis="chieff", nsamp=16,
                   seed=0, cosmology=_COSMO, far_max=1.0,
                   allow_missing_far=True, z_max=0.10)
    with h5py.File(pe, "r") as f:
        assert float(f.attrs["z_max"]) == 0.10
        assert np.asarray(f.attrs["n_samples_cut_by_z_max"]).sum() > 0

    inj = write_o4_full(tmp_path / "inj.hdf", n=200, amax=(0.9, 0.9), seed=8)
    # An untruncated selection file against a truncated PE file is refused.
    sel_full = tmp_path / "sel_full.h5"
    SelectionSet(inj).export(str(sel_full), format="gwcat2",
                             spin_basis="chieff", far_threshold=1.0)
    with pytest.raises(ValueError, match="z_max"):
        validate_export_v2(str(pe), str(sel_full))

    # Subset the injections the same way and the pair is coherent.
    sel_cut = tmp_path / "sel_cut.h5"
    SelectionSet(inj).export(str(sel_cut), format="gwcat2",
                             spin_basis="chieff", far_threshold=1.0,
                             z_max=0.10)
    with h5py.File(sel_cut, "r") as f:
        assert float(f.attrs["z_max"]) == 0.10
        assert np.all(np.asarray(f["redshift"]) <= 0.10)
        assert f.attrs["ndraw"] == h5py.File(sel_full, "r").attrs["ndraw"], \
            "z_max is subsetting, not reweighting: ndraw must be untouched"


# ======================================================================
# 9. far_or_snr files say so, and the SNR leg is actually compared
# ======================================================================
def test_far_or_snr_is_named_in_the_contract(tmp_path):
    """``stat = None if thr is None else "far"`` ignored the SNR leg entirely.

    A selection product whose injection mask is `far-detected OR snr > t` is
    strictly looser than a FAR-only event cut, so it biases mu high and the
    inferred rate low -- and it hash-matched a FAR-only PE file.
    """
    from test_export_v2_selection import _write_o4_with_snr

    n = 40
    far = np.full(n, 5.0)
    far[:20] = 0.1                 # 20 detected by FAR
    snr = np.zeros(n)
    snr[25:30] = 15.0              # 5 far-undetected injections pass SNR > 10
    inj = _write_o4_with_snr(tmp_path / "o4snr.hdf", n, far, snr)
    sel = tmp_path / "sel_snr.h5"
    SelectionSet(inj).export(str(sel), format="gwcat2.1",
                             spin_basis="chieff", far_threshold=1.0,
                             snr_threshold=10.0)

    with h5py.File(sel, "r") as f:
        assert f.attrs["significance_type"] == "far_or_snr"
        contract = json.loads(str(f.attrs["contract"]))
    assert contract["detection_statistic"] == "far_or_snr", \
        "the file declared itself FAR-only while detecting on far OR snr"
    assert contract["snr_min"] == 10.0


def test_detection_xcheck_reads_the_attr_the_writer_actually_stamps():
    """``_xcheck_detection_cut`` read `snr_threshold`, which nothing writes.

    So the SNR arm was structurally NaN and `_same(nan, nan)` returned True --
    the check never even reached its own "stated on one side only" warning.
    """
    from gwcat.export.validate import _xcheck_detection_cut

    results = {}

    def _fail(name, msg):
        raise ValueError(msg)

    with pytest.raises(ValueError, match="SNR"):
        _xcheck_detection_cut(
            _fail, results,
            {"far_max": 1.0, "snr_min": 8.0},
            {"far_threshold": 1.0, "significance_snr_threshold": 10.0,
             "significance_type": "far_or_snr"})


# ======================================================================
# 10. A failed merge or download cannot destroy what was already there
# ======================================================================
def test_merge_store_stages_beside_the_destination(tmp_path, monkeypatch):
    """``shutil.move`` fell through EXDEV to a copy that opened the LIVE store
    'wb', truncating it before writing a byte -- on the default, in-place,
    README-documented call, with no backup and no try/except.  /tmp is a
    different device from any real store here, so that path was not
    hypothetical: it was the only path the default call ever took.
    """
    import gwcat.ingest as ing

    store = _build_store(tmp_path)
    before = open(store, "rb").read()
    staged_beside = {}

    def _boom(*a, **kw):
        # Where the intermediate files live AT THE MOMENT of the merge.
        staged_beside["dirs"] = [p.name for p in tmp_path.iterdir()
                                 if p.is_dir()
                                 and p.name.startswith(".gwcat-merge-")]
        raise RuntimeError("merge blew up")

    monkeypatch.setattr(ing, "merge_stores", _boom)
    with pytest.raises(RuntimeError, match="merge blew up"):
        ing.merge_store(store, [], event_table={})

    assert open(store, "rb").read() == before, \
        "a failed merge destroyed the store it was merging into"
    assert staged_beside.get("dirs"), \
        "staging still goes to $TMPDIR, a different device, so the final " \
        "step is a truncating copy rather than an atomic os.replace"
    assert not [p for p in tmp_path.iterdir()
                if p.is_dir() and p.name.startswith(".gwcat-merge-")], \
        "the staging directory leaked"


def test_download_file_leaves_nothing_behind_on_a_checksum_mismatch(
        tmp_path, monkeypatch):
    """The refusal declared the file corrupt and left it at the consumed path."""
    import gwcat.fetch as fetch

    class _Resp:
        headers = {"Content-Length": "4"}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=0):
            yield b"junk"

    monkeypatch.setattr(fetch, "requests",
                        type("R", (), {"get": staticmethod(
                            lambda *a, **kw: _Resp())})(),
                        raising=False)
    dest = tmp_path / "data.h5"
    # download_file() imports requests and tqdm inside the function, so the
    # stand-ins have to be in sys.modules.  They go in through monkeypatch so
    # the real modules (or their absence) come back after the test (GW-42):
    # writing sys.modules directly left a module-typed stub named "requests"
    # behind whenever requests had not been imported yet, and every later
    # "import requests.exceptions" in the session then failed with
    # "'requests' is not a package".
    import sys
    fake_tqdm = type(sys)("tqdm")
    fake_tqdm.tqdm = lambda *a, **kw: None
    fake_requests = type(sys)("requests")
    fake_requests.get = lambda *a, **kw: _Resp()
    monkeypatch.setitem(sys.modules, "tqdm", fake_tqdm)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        fetch.download_file("https://example.invalid/x", str(dest),
                            expected_md5="0" * 32, show_progress=False)
    assert not dest.exists(), "a file declared corrupt was left on disk"
    assert not list(tmp_path.glob("*.part")), "the staging file leaked"
