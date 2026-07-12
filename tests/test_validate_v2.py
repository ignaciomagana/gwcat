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
    cat.export(str(pe), format="gwcat2", spin_basis="chieff_chip", nsamp=32,
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
    cat.export(str(out), format="gwcat2", spin_basis="chieff_chip", nsamp=16,
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
