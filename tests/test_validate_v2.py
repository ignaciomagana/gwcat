"""PR 7: gwcat-2.0 export validator + CLI auto-dispatch + v2 summary fields.

Exercises :func:`gwcat.export.validate_export_v2` and the ``gwcat validate``
auto-detection (v1 files -> the frozen :func:`gwcat.catalog.validate_export`;
v2 files -> ``validate_export_v2``; a mixed v1/v2 pair -> a clear error):

  1. happy path -- a component-basis PE + selection pair validates clean;
  2. spin_basis mismatch (component PE vs chieff selection) raises, naming both;
  3. a corrupted range (``cost1`` patched to 1.5) fails (and raises under strict);
  4. chieff amax cross-check raises on disagreement; chieff_chip amax is RECORDED
     (warned, never failed) when the two amax legitimately differ;
  5. CLI: a v1 pair routes to the v1 validator; a mixed v1+v2 pair errors;
  6. v2 export summaries (``write_summary=True``) carry the additive spin fields.

Fixtures are reused from the sibling PE / selection test modules.
"""
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import h5py
import pytest

from gwcat.catalog import GWCatalog
from gwcat.selection import SelectionSet, CombinedSelectionSet
from gwcat.export import validate_export_v2
from gwcat.cli import main

# Reuse the sibling modules' fixture builders verbatim.
from test_export_v2_pe import _build_spin_store
from test_export_v2 import _build_store
from test_selection_spin import write_o4_full, write_endo3_full

# The PE spin-store fixtures use an override cosmology; keep the selection on the
# same (H0, Om0) so the cosmology cross-check is exercised but does not trip.
_COSMO = (67.74, 0.3089)


def _pe_component(tmp_path, name="pe.h5", basis="component", **kw):
    events = [{"name": "GWv0_000001", "amax1": 0.99, "amax2": 0.99},
              {"name": "GWv0_000002", "amax1": 0.90, "amax2": 0.90}]
    store, _ = _build_spin_store(tmp_path, events, name=f"store_{name}")
    cat = GWCatalog(store)
    out = tmp_path / name
    cat.export(str(out), format="gwcat2", spin_basis=basis, nsamp=48, seed=0,
               allow_projection_basis=True,
               cosmology=_COSMO, **kw)
    return out


def _sel(tmp_path, name="sel.h5", basis="component", seed=6, **kw):
    inj = write_o4_full(tmp_path / f"inj_{name}.hdf", n=60, amax=(0.9, 0.9),
                        seed=seed)
    out = tmp_path / name
    SelectionSet(inj).export(str(out), spin_basis=basis, **kw)
    return out


# ======================================================================
# 1. Happy path: component PE + selection validates clean
# ======================================================================
def test_component_pair_validates_clean(tmp_path):
    pe = _pe_component(tmp_path)
    sel = _sel(tmp_path)
    results = validate_export_v2(str(pe), str(sel))
    assert all(results.values()), \
        f"unexpected failures: {[k for k, v in results.items() if not v]}"
    # The key cross-checks all ran and passed.
    for key in ("xcheck_spin_basis", "xcheck_component_pdraw_state",
                "xcheck_cosmology", "xcheck_source_class"):
        assert results[key] is True
    assert results["xcheck_component_pe_flag"] is True


def test_pe_only_validates_clean(tmp_path):
    pe = _pe_component(tmp_path)
    results = validate_export_v2(str(pe))
    assert all(results.values())
    # No selection => no cross-checks recorded.
    assert not any(k.startswith("xcheck_") for k in results)


# ======================================================================
# 2. spin_basis mismatch raises, naming BOTH sides
# ======================================================================
def test_spin_basis_mismatch_raises_naming_both(tmp_path):
    pe = _pe_component(tmp_path, basis="component")
    sel = _sel(tmp_path, basis="chieff")
    with pytest.raises(ValueError) as ei:
        validate_export_v2(str(pe), str(sel))
    msg = str(ei.value)
    assert "spin_basis" in msg
    assert "component" in msg and "chieff" in msg


# ======================================================================
# 3. Corrupted range (cost1 = 1.5) fails, and raises under strict
# ======================================================================
def test_corrupted_cost1_range_fails(tmp_path):
    pe = _pe_component(tmp_path)
    with h5py.File(pe, "r+") as f:
        d = f["cost1"][:]
        d[0] = 1.5
        f["cost1"][...] = d

    results = validate_export_v2(str(pe))
    assert results["pe_cost1_range"] is False

    with pytest.raises(AssertionError):
        validate_export_v2(str(pe), strict=True)


def test_corrupted_a1_out_of_amax_fails(tmp_path):
    pe = _pe_component(tmp_path)
    with h5py.File(pe, "r+") as f:
        d = f["a1"][:]
        d[0] = 5.0                      # far above max(spin_amax)*1.001
        f["a1"][...] = d
    results = validate_export_v2(str(pe))
    assert results["pe_a1_range"] is False


# ======================================================================
# 4. chieff amax cross-check + chieff_chip amax is recorded, not failed
# ======================================================================
def test_chieff_amax_mismatch_raises(tmp_path):
    events = [{"name": "GWv1_000001"}, {"name": "GWv1_000002"}]
    store, _ = _build_spin_store(tmp_path, events, name="ce_store.h5")
    cat = GWCatalog(store)
    pe = tmp_path / "ce_pe.h5"
    cat.export(str(pe), format="gwcat2", spin_basis="chieff", nsamp=32, seed=0,
               cosmology=_COSMO, amax=0.99)
    inj = write_o4_full(tmp_path / "ce_inj.hdf", n=40, seed=6)
    sel = tmp_path / "ce_sel.h5"
    SelectionSet(inj).export(str(sel), spin_basis="chieff", amax=0.95)

    with pytest.raises(ValueError, match="chi_eff_amax"):
        validate_export_v2(str(pe), str(sel))


def test_chieff_amax_match_passes(tmp_path):
    events = [{"name": "GWv2_000001"}, {"name": "GWv2_000002"}]
    store, _ = _build_spin_store(tmp_path, events, name="ce2_store.h5")
    cat = GWCatalog(store)
    pe = tmp_path / "ce2_pe.h5"
    cat.export(str(pe), format="gwcat2", spin_basis="chieff", nsamp=32, seed=0,
               cosmology=_COSMO, amax=0.99)
    inj = write_o4_full(tmp_path / "ce2_inj.hdf", n=40, seed=6)
    sel = tmp_path / "ce2_sel.h5"
    SelectionSet(inj).export(str(sel), spin_basis="chieff", amax=0.99)
    results = validate_export_v2(str(pe), str(sel))
    assert results["xcheck_chieff_amax"] is True
    assert all(results.values())


def test_chieff_chip_amax_recorded_not_failed(tmp_path):
    # PE prior amax (0.99) vs the injected endo3 detected amax (~0.998) DIFFER on
    # purpose: the check RECORDS both and warns, but must NOT fail.
    events = [{"name": "GWv3_000001", "amax1": 0.99, "amax2": 0.99}]
    store, _ = _build_spin_store(tmp_path, events, name="cc_store.h5")
    cat = GWCatalog(store)
    pe = tmp_path / "cc_pe.h5"
    cat.export(str(pe), format="gwcat2", spin_basis="chieff_chip", allow_projection_basis=True, nsamp=32,
               seed=0, cosmology=_COSMO)
    o3 = write_endo3_full(tmp_path / "cc_endo3.hdf", n=40, max_spin=0.998,
                          seed=11)
    sel = tmp_path / "cc_sel.h5"
    SelectionSet(o3).export(str(sel), spin_basis="chieff_chip")

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        results = validate_export_v2(str(pe), str(sel))
    # Recorded, both sides' amax finite, and it did NOT fail on the difference.
    assert results["xcheck_chieff_chip_amax_recorded"] is True
    assert results["xcheck_chieff_chip_pe_amax_finite"] is True
    assert results["xcheck_chieff_chip_sel_amax_finite"] is True
    assert all(results.values())
    # A warning explains the (legitimate) amax difference.
    assert any("amax" in str(w.message) for w in rec)


# ======================================================================
# 5. CLI auto-dispatch: v1 routing, mixed pair error
# ======================================================================
def test_cli_v1_pair_routes_to_v1_validator(tmp_path, capsys):
    st = _build_store(tmp_path)
    cat = GWCatalog(st)
    v1pe = tmp_path / "v1pe.h5"
    cat.to_darksirens(str(v1pe), cosmology=_COSMO, nsamp=16, seed=0)
    inj = write_o4_full(tmp_path / "v1inj.hdf", n=40, seed=6)
    v1sel = tmp_path / "v1sel.h5"
    SelectionSet(inj).to_darksirens(str(v1sel), far_threshold=1.0)

    rc = main(["validate", str(v1pe), str(v1sel)])
    assert rc == 0
    out = capsys.readouterr().out
    # The v1 validator's banner, NOT the v2 one ("gwcat-2.0").
    assert "Validating PE export:" in out
    assert "gwcat-2.0" not in out


def test_cli_v2_pair_routes_to_v2_validator(tmp_path, capsys):
    pe = _pe_component(tmp_path)
    sel = _sel(tmp_path)
    rc = main(["validate", str(pe), str(sel)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Validating gwcat-2.0 PE export:" in out


def test_cli_mixed_v1_v2_pair_errors(tmp_path, capsys):
    st = _build_store(tmp_path)
    cat = GWCatalog(st)
    v1pe = tmp_path / "v1pe.h5"
    cat.to_darksirens(str(v1pe), cosmology=_COSMO, nsamp=16, seed=0)
    sel_v2 = _sel(tmp_path, name="v2sel.h5")

    rc = main(["validate", str(v1pe), str(sel_v2)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "mixed export generations" in err


def test_cli_v2_corrupted_returns_nonzero(tmp_path):
    pe = _pe_component(tmp_path)
    with h5py.File(pe, "r+") as f:
        d = f["cost1"][:]
        d[0] = 1.5
        f["cost1"][...] = d
    rc = main(["validate", str(pe)])
    assert rc == 1


# ======================================================================
# 6. v2 export summaries carry the additive spin fields
# ======================================================================
def test_pe_v2_summary_fields_present(tmp_path):
    pe = _pe_component(tmp_path, write_summary=True)
    sj = json.loads(Path(str(pe) + ".validation_summary.json").read_text())
    assert sj["spin_basis"] == "component"
    assert "spin_prior_provenance" in sj
    assert sj["spin_prior_provenance"]["component_spin_prior_applied_to_p_pe"] \
        is True
    assert "spin_amax_summary" in sj
    assert "spin_amax_1" in sj["spin_amax_summary"]
    # The .md renders the same enriched dict.
    md = Path(str(pe) + ".validation_summary.md").read_text()
    assert "spin_amax_summary" in md


def test_selection_v2_summary_fields_present(tmp_path):
    sel = _sel(tmp_path, write_summary=True)
    sj = json.loads(Path(str(sel) + ".validation_summary.json").read_text())
    assert sj["spin_basis"] == "component"
    assert "injected_spin_amax_detected" in sj
    assert "injected_spin_uniform_isotropic" in sj
    assert "injected_spin_checks" in sj
    assert "pdraw_state" in sj


def test_pe_chieff_chip_summary_amax_field(tmp_path):
    events = [{"name": "GWv4_000001", "amax1": 0.99, "amax2": 0.99}]
    store, _ = _build_spin_store(tmp_path, events, name="s4.h5")
    cat = GWCatalog(store)
    out = tmp_path / "cc_sum.h5"
    cat.export(str(out), format="gwcat2", spin_basis="chieff_chip", allow_projection_basis=True, nsamp=16,
               seed=0, cosmology=_COSMO, write_summary=True)
    sj = json.loads(Path(str(out) + ".validation_summary.json").read_text())
    assert "spin_amax_summary" in sj
    assert "chi_eff_chi_p_amax_per_event" in sj["spin_amax_summary"]


# ======================================================================
# 8. Selection magnitude bound: non-uniform-isotropic campaigns fall back
#    to the physical limit 1.0 (real O4 sets draw non-uniform spins that
#    extend past any other campaign's detected amax).
# ======================================================================
def test_sel_range_bound_physical_for_non_uniform_campaign(tmp_path):
    pe = _pe_component(tmp_path)
    sel = _sel(tmp_path)
    with h5py.File(sel, "r+") as f:
        # Mark the (single) campaign as non-uniform-isotropic with an
        # unreliable detected amax, and push one spin past that amax.
        f.attrs["injected_spin_uniform_isotropic"] = np.array([False])
        f.attrs["injected_spin_amax_detected"] = np.array([0.75, 0.75])
        d = f["a1"][:]
        d[0] = 0.97          # > 0.75*1.001 but physical (< 1)
        f["a1"][...] = d
    results = validate_export_v2(str(pe), str(sel))
    assert results["sel_a1_range"] is True

    # An unphysical magnitude still fails against the 1.0 bound.
    with h5py.File(sel, "r+") as f:
        d = f["a1"][:]
        d[0] = 1.2
        f["a1"][...] = d
    results = validate_export_v2(str(pe), str(sel))
    assert results["sel_a1_range"] is False


# ======================================================================
# GW-20: sky finiteness AND range are validated
# ======================================================================
def test_nan_sky_fails(tmp_path):
    """NaN ra/dec passed every check and reached hp.ang2pix in the consumer.

    The loader NaN-fills the semianalytic O1/O2 rows of cumulative-mixture
    files, the exporters write those NaNs, and neither the validator nor the
    consumer looked -- the validator's only finiteness check was on p_pe/pdraw.
    """
    pe = _pe_component(tmp_path)
    with h5py.File(pe, "r+") as f:
        d = f["dec"][:]
        d[3] = np.nan
        f["dec"][...] = d
    results = validate_export_v2(str(pe))
    assert results["pe_dec_finite"] is False
    with pytest.raises(AssertionError, match="dec_finite"):
        validate_export_v2(str(pe), strict=True)


def test_degrees_sky_fails(tmp_path):
    """A degrees ingest produces a plausible-looking but wrong pixelisation
    with no symptom anywhere; the declared radian range is what catches it."""
    pe = _pe_component(tmp_path)
    with h5py.File(pe, "r+") as f:
        f["dec"][...] = np.degrees(f["dec"][:])      # radians -> degrees
    results = validate_export_v2(str(pe))
    assert results["pe_dec_range"] is False


def test_colatitude_sky_fails(tmp_path):
    """dec in [0, pi] (colatitude) rather than [-pi/2, pi/2]."""
    pe = _pe_component(tmp_path)
    with h5py.File(pe, "r+") as f:
        f["dec"][...] = np.pi / 2.0 - f["dec"][:]    # dec -> theta
    results = validate_export_v2(str(pe))
    assert results["pe_dec_range"] is False


def test_negative_ra_fails(tmp_path):
    """ra must be in [0, 2pi); a [-pi, pi) convention is a different wrap."""
    pe = _pe_component(tmp_path)
    with h5py.File(pe, "r+") as f:
        f["ra"][...] = f["ra"][:] - np.pi
    results = validate_export_v2(str(pe))
    assert results["pe_ra_range"] is False


def test_valid_sky_passes(tmp_path):
    """The ordinary path is unaffected -- radians, in range, finite."""
    pe = _pe_component(tmp_path)
    results = validate_export_v2(str(pe))
    for k in ("pe_ra_finite", "pe_dec_finite", "pe_ra_range", "pe_dec_range"):
        assert results[k] is True, k


def test_ra_exactly_2pi_is_out_of_range(tmp_path):
    """[0, 2pi) is half-open: 2pi wraps to 0 and must not pass as itself."""
    pe = _pe_component(tmp_path)
    with h5py.File(pe, "r+") as f:
        d = f["ra"][:]
        d[0] = 2.0 * np.pi
        f["ra"][...] = d
    results = validate_export_v2(str(pe))
    assert results["pe_ra_range"] is False


# ======================================================================
# GW-11: the cross-file checks that were vacuous or absent
# ======================================================================
def test_source_class_mismatch_fails(tmp_path):
    """An nsbh+bns PE file paired with a bbh selection file must fail.

    This passed before GW-11.  Both sides wrote ``str(source_class)`` -- the
    Python API's ``"['nsbh', 'bns']"`` and the CLI's ``"nsbh,bns"`` -- and
    neither round-tripped, so ``resolve_filter_classes`` mapped both to
    ``{"Unknown"}`` and the check compared ``{"Unknown"} == {"Unknown"}``.
    """
    pe = _pe_component(tmp_path)
    sel = _sel(tmp_path)
    with h5py.File(pe, "r+") as f:
        f.attrs["source_class_filter"] = "NSBH,BNS"
    with h5py.File(sel, "r+") as f:
        f.attrs["source_class_filter"] = "BBH"

    with pytest.raises(ValueError) as ei:
        validate_export_v2(str(pe), str(sel))
    msg = str(ei.value)
    assert "xcheck_source_class" in msg
    # The message must name both sides' resolved sets, not just "mismatch".
    assert "NSBH" in msg and "BBH" in msg


def test_source_class_legacy_repr_is_refused_not_silently_passed(tmp_path):
    """A pre-GW-11 file records a repr that was never checkable."""
    pe = _pe_component(tmp_path)
    sel = _sel(tmp_path)
    with h5py.File(pe, "r+") as f:
        f.attrs["source_class_filter"] = "['nsbh', 'bns']"
    with h5py.File(sel, "r+") as f:
        f.attrs["source_class_filter"] = "BBH"

    with pytest.raises(ValueError) as ei:
        validate_export_v2(str(pe), str(sel))
    assert "xcheck_source_class" in str(ei.value)


def test_detection_cut_mismatch_fails(tmp_path):
    """Two stated FAR thresholds that differ is a hard failure."""
    pe = _pe_component(tmp_path)
    sel = _sel(tmp_path, far_threshold=1.0)
    with h5py.File(pe, "r+") as f:
        f.attrs["far_max"] = 2.0

    with pytest.raises(ValueError) as ei:
        validate_export_v2(str(pe), str(sel))
    msg = str(ei.value)
    assert "xcheck_detection_cut" in msg
    assert "2.0" in msg and "1.0" in msg


def test_detection_cut_match_passes(tmp_path):
    pe = _pe_component(tmp_path)
    sel = _sel(tmp_path, far_threshold=1.0)
    with h5py.File(pe, "r+") as f:
        f.attrs["far_max"] = 1.0
    results = validate_export_v2(str(pe), str(sel))
    assert results["xcheck_detection_cut"] is True


def test_detection_cut_stated_on_one_side_warns_not_fails(tmp_path):
    """The shipped configuration: a name-whitelisted event list, FAR-cut injections.

    The equivalence is real (checked directly on the shipped list) but no attr
    can express it, so this warns rather than refusing a correct pairing.
    """
    pe = _pe_component(tmp_path)
    sel = _sel(tmp_path, far_threshold=1.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        results = validate_export_v2(str(pe), str(sel))
    assert results["xcheck_detection_cut"] is True
    assert any("one side only" in str(x.message) for x in w)


def test_allow_missing_far_under_a_far_cut_fails(tmp_path):
    """A declaredly FAR-cut event list that kept untested events is self-contradictory."""
    pe = _pe_component(tmp_path)
    sel = _sel(tmp_path, far_threshold=1.0)
    with h5py.File(pe, "r+") as f:
        f.attrs["far_max"] = 1.0
        f.attrs["allow_missing_far"] = True
        f.attrs["n_events_missing_far"] = 3
    with pytest.raises(ValueError) as ei:
        validate_export_v2(str(pe), str(sel))
    assert "allow_missing_far" in str(ei.value)


def test_unknown_format_version_errors(tmp_path):
    """An unrecognised format must stop, not fall through to an older contract."""
    pe = _pe_component(tmp_path)
    with h5py.File(pe, "r+") as f:
        f.attrs["format_version"] = "gwcat-pe-3.0"
    with pytest.raises(ValueError) as ei:
        validate_export_v2(str(pe))
    msg = str(ei.value)
    assert "gwcat-pe-3.0" in msg
    assert "gwcat-pe-2.0" in msg and "gwcat-pe-2.1" in msg


def test_missing_format_version_errors(tmp_path):
    pe = _pe_component(tmp_path)
    with h5py.File(pe, "r+") as f:
        del f.attrs["format_version"]
    with pytest.raises(ValueError) as ei:
        validate_export_v2(str(pe))
    assert "format_version" in str(ei.value)


# ======================================================================
# GW-11: format 2.1 actually runs through the validator (GW-22 shipped
# writers whose output the validator rejected, and never tested it)
# ======================================================================
def _pair_21(tmp_path, basis="component"):
    events = [{"name": "GWv0_000001", "amax1": 0.99, "amax2": 0.99},
              {"name": "GWv0_000002", "amax1": 0.90, "amax2": 0.90}]
    store, _ = _build_spin_store(tmp_path, events, name="store_21")
    pe = tmp_path / "pe21.h5"
    GWCatalog(store).export(str(pe), format="gwcat2.1", spin_basis=basis,
                            nsamp=48, seed=0, cosmology=_COSMO)
    inj = write_o4_full(tmp_path / "inj21.hdf", n=60, amax=(0.9, 0.9), seed=6)
    sel = tmp_path / "sel21.h5"
    SelectionSet(inj).export(str(sel), format="gwcat2.1", spin_basis=basis)
    return pe, sel


def test_21_pair_validates(tmp_path):
    pe, sel = _pair_21(tmp_path)
    with h5py.File(pe, "r") as f:
        assert f.attrs["format_version"] == "gwcat-pe-2.1"
    results = validate_export_v2(str(pe), str(sel))
    assert all(results.values()), \
        f"unexpected failures: {[k for k, v in results.items() if not v]}"
    assert results["xcheck_contract_hash"] is True


def test_21_contract_hash_mismatch_reports_the_field(tmp_path):
    """A hash mismatch must name the field, not just say the hashes differ."""
    pe, sel = _pair_21(tmp_path)
    with h5py.File(sel, "r+") as f:
        c = json.loads(str(f.attrs["contract"]))
        c["spin_basis_kind"] = "projection"
        f.attrs["contract"] = json.dumps(c)
        f.attrs["contract_hash"] = "0" * 16

    with pytest.raises(ValueError) as ei:
        validate_export_v2(str(pe), str(sel))
    msg = str(ei.value)
    assert "xcheck_contract_hash" in msg
    assert "spin_basis_kind" in msg and "projection" in msg


def test_21_source_class_is_canonical_in_the_hash(tmp_path):
    """Two spellings of one request must produce the SAME contract hash.

    The hash input was ``str(source_class)``, so ``["bbh"]`` and ``"bbh"``
    described the same product and hashed differently.
    """
    events = [{"name": "GWv0_000001", "amax1": 0.99, "amax2": 0.99},
              {"name": "GWv0_000002", "amax1": 0.90, "amax2": 0.90}]
    store, _ = _build_spin_store(tmp_path, events, name="store_hash")
    hashes = []
    for spec in ("bbh", ["bbh"], ("BBH",)):
        out = tmp_path / f"pe_{len(hashes)}.h5"
        GWCatalog(store).export(str(out), format="gwcat2.1",
                                spin_basis="component", nsamp=16, seed=0,
                                cosmology=_COSMO, source_class=spec)
        with h5py.File(out, "r") as f:
            hashes.append(str(f.attrs["contract_hash"]))
    assert len(set(hashes)) == 1, hashes


# ======================================================================
# GW-22b: the no-flags pair must validate, end to end through the CLI
# ======================================================================
def test_default_pair_validates(tmp_path):
    """`export pe` and `export selection` with NO space flag must pair.

    They defaulted to `chieff` and `component` respectively, so this exact
    sequence -- the one a first-time user runs -- produced two files that fail
    `xcheck_spin_basis` by construction.
    """
    events = [{"name": "GWv0_000001", "amax1": 0.99, "amax2": 0.99},
              {"name": "GWv0_000002", "amax1": 0.90, "amax2": 0.90}]
    store, _ = _build_spin_store(tmp_path, events, name="store_default")
    inj = write_o4_full(tmp_path / "inj_default.hdf", n=60, amax=(0.9, 0.9),
                        seed=6)
    pe = tmp_path / "pe_default.h5"
    sel = tmp_path / "sel_default.h5"

    assert main(["export", "pe", str(store), "--out", str(pe),
                 "--cosmology", "67.74,0.3089", "--nsamp", "48",
                 "--seed", "0", "--no-summary"]) == 0
    assert main(["export", "selection", str(inj), "--out", str(sel),
                 "--no-summary"]) == 0

    with h5py.File(pe, "r") as f, h5py.File(sel, "r") as g:
        assert f.attrs["spin_basis"] == g.attrs["spin_basis"]

    results = validate_export_v2(str(pe), str(sel))
    assert results["xcheck_spin_basis"] is True
    assert all(results.values()), \
        f"unexpected failures: {[k for k, v in results.items() if not v]}"


def test_the_old_defaults_would_have_failed(tmp_path):
    """Pins WHY the default changed, so nobody 'restores' the old one."""
    pe = _pe_component(tmp_path, basis="chieff")     # the old PE default
    sel = _sel(tmp_path, basis="component")          # the old selection default
    with pytest.raises(ValueError) as ei:
        validate_export_v2(str(pe), str(sel))
    assert "spin_basis" in str(ei.value)
