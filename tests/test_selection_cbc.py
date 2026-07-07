"""Selection products for all CBC source classes (PR 9).

Covers:
  * toy O3- and O4-format selection fixtures exporting with explicit prior +
    source-class + significance provenance attrs;
  * source_class filtering on injections (bbh/nsbh/bns/cbc) via the shared
    mass-threshold classifier, with exact counts on a mixed-injection fixture;
  * default (no source_class arg) output byte-identical to the pre-PR9 export
    (fixed-inputs regression);
  * cross-validation of a combined selection file against a PE export -- happy
    path plus clear-error failures on spin-mode, cosmology, and source-class
    mismatch;
  * the PR4 per-event-cosmology arrays path in validate_export.

All fixtures are tiny synthetic HDF5 files -- no network access.
"""
import numpy as np
import h5py
import pytest

from gwcat.catalog import GWCatalog, validate_export
from gwcat.selection import SelectionSet, CombinedSelectionSet
from gwcat.spin import chi_eff_prior_logprob

# Reference cosmology shared by the PE store and selection fixtures.
_H0, _Om0 = 67.74, 0.3089

# Per-class representative source-frame masses (thr = 3.0 Msun).
_CLASS_MASSES = {
    "BBH": (35.0, 30.0),
    "NSBH": (12.0, 1.4),
    "BNS": (1.6, 1.3),
}


# ==========================================================================
# O4 'events'-format fixture (factored lnpdraw path)
# ==========================================================================
_O4_FIELDS = [
    ("mass1_source", "f8"), ("mass2_source", "f8"),
    ("mass1_detector", "f8"), ("mass2_detector", "f8"),
    ("luminosity_distance", "f8"), ("z", "f8"),
    ("dluminosity_distance_dredshift", "f8"),
    ("right_ascension", "f8"), ("declination", "f8"),
    ("spin1x", "f8"), ("spin1y", "f8"), ("spin1z", "f8"),
    ("spin2x", "f8"), ("spin2y", "f8"), ("spin2z", "f8"),
    ("chi_eff", "f8"), ("weights", "f8"),
    ("lnpdraw_mass1_source", "f8"),
    ("lnpdraw_mass2_source_GIVEN_mass1_source", "f8"),
    ("lnpdraw_z", "f8"),
    ("pycbc_far", "f8"), ("cwb-bbh_far", "f8"),
]


def _o4_events(class_counts, detected=True):
    """Build an O4 events compound array with the requested per-class counts.

    class_counts : dict like {"BBH": 4, "NSBH": 3, "BNS": 2}
    detected : if True every injection has FAR below 1.0 (all detected).
    """
    rows = []
    for cls, n in class_counts.items():
        m1s, m2s = _CLASS_MASSES[cls]
        for _ in range(n):
            rows.append((cls, m1s, m2s))
    n = len(rows)
    ev = np.zeros(n, dtype=_O4_FIELDS)
    z = 0.1
    for i, (_cls, m1s, m2s) in enumerate(rows):
        ev["mass1_source"][i] = m1s
        ev["mass2_source"][i] = m2s
        ev["z"][i] = z
        ev["mass1_detector"][i] = m1s * (1 + z)
        ev["mass2_detector"][i] = m2s * (1 + z)
        ev["luminosity_distance"][i] = 450.0
        ev["dluminosity_distance_dredshift"][i] = 4500.0
        ev["right_ascension"][i] = 1.0
        ev["declination"][i] = 0.5
        ev["spin1z"][i] = 0.2
        ev["spin2z"][i] = -0.1
        ev["chi_eff"][i] = (m1s * 0.2 + m2s * -0.1) / (m1s + m2s)
        ev["weights"][i] = 2.0
        ev["lnpdraw_mass1_source"][i] = -1.0
        ev["lnpdraw_mass2_source_GIVEN_mass1_source"][i] = -2.0
        ev["lnpdraw_z"][i] = -3.0
    far = 0.1 if detected else 5.0
    ev["pycbc_far"] = far
    ev["cwb-bbh_far"] = far
    return ev


def write_o4(path, class_counts, detected=True, total_generated=1000):
    ev = _o4_events(class_counts, detected=detected)
    with h5py.File(path, "w") as f:
        f.attrs["total_analysis_time"] = 365.25 * 24 * 3600
        f.attrs["total_generated"] = total_generated
        f.attrs["searches"] = np.array([b"pycbc", b"cwb-bbh"])
        f.create_dataset("events", data=ev)
    return str(path)


# ==========================================================================
# O3 'injections'-format fixture (factored pdf components)
# ==========================================================================
def write_o3(path, class_counts, detected=True, total_generated=2000):
    rows = []
    for cls, n in class_counts.items():
        m1s, m2s = _CLASS_MASSES[cls]
        rows.extend([(m1s, m2s)] * n)
    n = len(rows)
    m1s = np.array([r[0] for r in rows])
    m2s = np.array([r[1] for r in rows])
    z = np.full(n, 0.1)
    far = 0.1 if detected else 5.0
    with h5py.File(path, "w") as f:
        f.attrs["total_generated"] = total_generated
        inj = f.create_group("injections")
        inj.attrs["analysis_time_s"] = 365.25 * 24 * 3600
        inj.create_dataset("mass1_source", data=m1s)
        inj.create_dataset("mass2_source", data=m2s)
        inj.create_dataset("mass1", data=m1s * (1 + z))
        inj.create_dataset("mass2", data=m2s * (1 + z))
        inj.create_dataset("distance", data=np.full(n, 450.0))
        inj.create_dataset("redshift", data=z)
        inj.create_dataset("right_ascension", data=np.full(n, 1.0))
        inj.create_dataset("declination", data=np.full(n, 0.5))
        inj.create_dataset("spin1z", data=np.full(n, 0.2))
        inj.create_dataset("spin2z", data=np.full(n, -0.1))
        inj.create_dataset("mass1_source_mass2_source_sampling_pdf",
                           data=np.full(n, 1e-3))
        inj.create_dataset("redshift_sampling_pdf", data=np.full(n, 0.5))
        inj.create_dataset("far_pycbc_bbh", data=np.full(n, far))
        inj.create_dataset("far_gstlal", data=np.full(n, far))
    return str(path)


# ==========================================================================
# Minimal PE store (with source_class column + per-event cosmology)
# ==========================================================================
_PE_PARAMS = ["mass_1", "mass_2", "luminosity_distance", "ra", "dec",
              "chi_eff", "p_dL_pe"]


def build_pe_store(tmp_path, events, seed=7, name="pe_store.h5"):
    """events: list of dicts {name, source_class, H0, Om0, n}."""
    rng = np.random.default_rng(seed)
    offsets = [0]
    cols = {p: [] for p in _PE_PARAMS}
    meta = {k: [] for k in ["source_class", "compact_type",
                            "dL_prior_H0", "dL_prior_Om0"]}
    names = []
    for ev in events:
        n = int(ev.get("n", 20))
        cols["mass_1"].append(rng.uniform(25, 45, n))
        cols["mass_2"].append(rng.uniform(10, 22, n))
        cols["luminosity_distance"].append(rng.uniform(300, 800, n))
        cols["ra"].append(rng.uniform(0, 2 * np.pi, n))
        cols["dec"].append(rng.uniform(-np.pi / 2, np.pi / 2, n))
        cols["chi_eff"].append(rng.uniform(-0.3, 0.3, n))
        cols["p_dL_pe"].append(rng.uniform(0.1, 1.0, n))
        offsets.append(offsets[-1] + n)
        names.append(ev["name"])
        meta["source_class"].append(ev.get("source_class", "BBH"))
        meta["compact_type"].append(ev.get("source_class", "BBH"))
        meta["dL_prior_H0"].append(float(ev.get("H0", _H0)))
        meta["dL_prior_Om0"].append(float(ev.get("Om0", _Om0)))
    path = tmp_path / name
    with h5py.File(path, "w") as f:
        f.attrs["param_names"] = np.array(_PE_PARAMS, dtype=h5py.string_dtype())
        idx = f.create_group("index")
        idx.create_dataset("offsets", data=np.array(offsets, dtype="i8"))
        idx.create_dataset("event_names",
                           data=np.array(names, dtype=h5py.string_dtype()))
        mg = f.create_group("meta")
        for k in ["source_class", "compact_type"]:
            mg.create_dataset(k, data=np.array(meta[k], dtype=h5py.string_dtype()))
        for k in ["dL_prior_H0", "dL_prior_Om0"]:
            mg.create_dataset(k, data=np.asarray(meta[k], dtype="f8"))
        sg = f.create_group("samples")
        for p in _PE_PARAMS:
            sg.create_dataset(p, data=np.concatenate(cols[p]))
    return str(path)


# ==========================================================================
# 1. toy O3/O4 fixtures export with explicit provenance attrs
# ==========================================================================
@pytest.mark.parametrize("writer", [write_o4, write_o3])
def test_export_records_prior_and_source_class_metadata(tmp_path, writer):
    inj = writer(tmp_path / "inj.hdf", {"BBH": 3})
    out = tmp_path / "sel.h5"
    SelectionSet(inj).to_darksirens(str(out), far_threshold=1.0)

    with h5py.File(out, "r") as f:
        # spin-prior contract (PR3, unchanged)
        assert f.attrs["spin_prior_mode"] == "include"
        assert bool(f.attrs["chi_eff_prior_applied_to_pdraw"]) is True
        # pdraw / source-class provenance (PR9)
        assert "chi_eff" in f.attrs["pdraw_state"]
        assert f.attrs["source_class_filter"] == ""          # no filter
        assert f.attrs["source_class_method"] == "none"
        assert float(f.attrs["nsbh_mass_threshold"]) == 3.0
        assert int(f.attrs["n_injections_before_filter"]) == 3
        assert int(f.attrs["n_injections_after_filter"]) == 3
        # significance provenance (explicit)
        cols = [c.decode() if isinstance(c, bytes) else c
                for c in f.attrs["significance_columns"]]
        assert len(cols) >= 1
        assert f.attrs["significance_type"] == "far"
        assert float(f.attrs["significance_far_threshold"]) == 1.0
        assert bool(f.attrs["significance_available"]) is True
        assert bool(f.attrs["p_astro_available"]) is False


def test_export_records_significance_columns_o4(tmp_path):
    inj = write_o4(tmp_path / "inj.hdf", {"BBH": 3})
    out = tmp_path / "sel.h5"
    SelectionSet(inj).to_darksirens(str(out), far_threshold=1.0)
    with h5py.File(out, "r") as f:
        cols = set(c.decode() if isinstance(c, bytes) else c
                   for c in f.attrs["significance_columns"])
    assert cols == {"pycbc_far", "cwb-bbh_far"}


# ==========================================================================
# 2. source-class filtering counts (shared mass-threshold helper)
# ==========================================================================
_COUNTS = {"BBH": 4, "NSBH": 3, "BNS": 2}


@pytest.mark.parametrize("writer", [write_o4, write_o3])
@pytest.mark.parametrize("sc,expected", [
    ("bbh", 4), ("nsbh", 3), ("bns", 2), ("cbc", 9), (None, 9),
])
def test_source_class_filter_counts(tmp_path, writer, sc, expected):
    inj = writer(tmp_path / "inj.hdf", _COUNTS, detected=True)
    sel = SelectionSet(inj)
    # mask-level count
    assert int(sel.source_class_mask(sc).sum()) == expected
    out = tmp_path / "sel.h5"
    sel.to_darksirens(str(out), far_threshold=1.0, source_class=sc)
    with h5py.File(out, "r") as f:
        assert int(f.attrs["n_detected"]) == expected
        assert f["m1det"].shape[0] == expected
        assert int(f.attrs["n_injections_before_filter"]) == 9
        assert int(f.attrs["n_injections_after_filter"]) == expected
        if sc is None:
            assert f.attrs["source_class_filter"] == ""
            assert "source_class_filter_note" not in f.attrs
        else:
            assert f.attrs["source_class_filter"] == str(sc)
            assert f.attrs["source_class_method"] == "mass_threshold"
            assert "source_class_filter_note" in f.attrs


def test_source_class_filter_ndraw_unchanged(tmp_path):
    """Filtering is subsetting, not reweighting: ndraw is untouched."""
    inj = write_o4(tmp_path / "inj.hdf", _COUNTS)
    full = tmp_path / "full.h5"
    bbh = tmp_path / "bbh.h5"
    SelectionSet(inj).to_darksirens(str(full), far_threshold=1.0)
    SelectionSet(inj).to_darksirens(str(bbh), far_threshold=1.0,
                                    source_class="bbh")
    with h5py.File(full, "r") as ff, h5py.File(bbh, "r") as fb:
        assert int(ff.attrs["ndraw"]) == int(fb.attrs["ndraw"])


def test_combined_source_class_filter_counts(tmp_path):
    o3 = write_o3(tmp_path / "o3.hdf", _COUNTS)
    o4 = write_o4(tmp_path / "o4.hdf", _COUNTS)
    combined = CombinedSelectionSet(
        [SelectionSet(o3, H0=_H0, Om0=_Om0), SelectionSet(o4, H0=_H0, Om0=_Om0)])
    out = tmp_path / "sel.h5"
    combined.to_darksirens(str(out), far_threshold=1.0, source_class="bbh")
    with h5py.File(out, "r") as f:
        assert int(f.attrs["n_detected"]) == 8       # 4 BBH per campaign
        assert int(f.attrs["n_injections_before_filter"]) == 18
        assert int(f.attrs["n_injections_after_filter"]) == 8
        assert f.attrs["source_class_filter"] == "bbh"


# ==========================================================================
# 3. default output byte-identical to the pre-PR9 export
# ==========================================================================
@pytest.mark.parametrize("writer", [write_o4, write_o3])
def test_default_export_byte_identical_to_pre_pr9(tmp_path, writer):
    """With no source_class arg the exported arrays must equal the pre-PR9
    detection-only algorithm (detected_mask + chi_eff swap), byte for byte."""
    inj = writer(tmp_path / "inj.hdf", {"BBH": 3, "NSBH": 2}, detected=True)
    sel = SelectionSet(inj)
    out = tmp_path / "sel.h5"
    sel.to_darksirens(str(out), far_threshold=1.0)          # default: no filter

    # Independent reconstruction of the pre-PR9 output.
    sel._load()
    det = sel.detected_mask(1.0)
    logp = chi_eff_prior_logprob(sel._chieff[det], sel._m1src[det],
                                 sel._m2src[det], amax=0.99)
    expected_pdraw = sel._pdraw[det] * np.exp(np.clip(logp, -50.0, None))

    with h5py.File(out, "r") as f:
        np.testing.assert_array_equal(f["m1det"][:], sel._m1det[det])
        np.testing.assert_array_equal(f["m2det"][:], sel._m2det[det])
        np.testing.assert_array_equal(f["dL"][:], sel._dL[det])
        np.testing.assert_array_equal(f["redshift"][:], sel._z[det])
        np.testing.assert_array_equal(f["pdraw"][:], expected_pdraw)
        assert int(f.attrs["n_detected"]) == int(det.sum())


def test_none_and_cbc_agree_for_all_bbh(tmp_path):
    """cbc == None when every injection is BBH."""
    inj = write_o4(tmp_path / "inj.hdf", {"BBH": 5})
    a = tmp_path / "none.h5"
    b = tmp_path / "cbc.h5"
    SelectionSet(inj).to_darksirens(str(a), far_threshold=1.0)
    SelectionSet(inj).to_darksirens(str(b), far_threshold=1.0, source_class="cbc")
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        np.testing.assert_array_equal(fa["pdraw"][:], fb["pdraw"][:])
        assert int(fa["m1det"].shape[0]) == int(fb["m1det"].shape[0]) == 5


# ==========================================================================
# 4. cross-validation of a combined selection file against a PE export
# ==========================================================================
def _make_pe_export(tmp_path, events=None, cosmology=(_H0, _Om0),
                    spin_prior_mode="include", source_class=None,
                    name="pe.h5", seed=7):
    if events is None:
        events = [{"name": "GW900001_000001", "source_class": "BBH"},
                  {"name": "GW900002_000002", "source_class": "BBH"}]
    store = build_pe_store(tmp_path, events, seed=seed,
                           name=name.replace(".h5", "_store.h5"))
    out = tmp_path / name
    GWCatalog(store).to_darksirens(
        str(out), nsamp=8, seed=0, cosmology=cosmology,
        spin_prior_mode=spin_prior_mode, source_class=source_class)
    return str(out)


def _make_combined_selection(tmp_path, source_class=None, H0=_H0, Om0=_Om0,
                             name="sel.h5", counts=None):
    counts = counts or {"BBH": 4}
    o3 = write_o3(tmp_path / "xo3.hdf", counts)
    o4 = write_o4(tmp_path / "xo4.hdf", counts)
    combined = CombinedSelectionSet(
        [SelectionSet(o3, H0=H0, Om0=Om0), SelectionSet(o4, H0=H0, Om0=Om0)])
    out = tmp_path / name
    combined.to_darksirens(str(out), far_threshold=1.0, source_class=source_class)
    return str(out)


def test_validate_combined_selection_happy_path(tmp_path):
    pe = _make_pe_export(tmp_path)
    sel = _make_combined_selection(tmp_path)
    results = validate_export(pe, sel)
    assert results["xcheck_spin_prior_mode"] is True
    assert results["xcheck_chi_eff_flag"] is True
    assert results["xcheck_cosmology"] is True
    assert results["xcheck_source_class"] is True
    assert all(results.values())


def test_validate_fails_on_spin_mode_mismatch(tmp_path):
    pe = _make_pe_export(tmp_path, spin_prior_mode="exclude")   # PE excludes
    sel = _make_combined_selection(tmp_path)                    # selection includes
    with pytest.raises(ValueError, match="spin_prior_mode"):
        validate_export(pe, sel)


def test_validate_fails_on_cosmology_mismatch(tmp_path):
    pe = _make_pe_export(tmp_path)                              # H0=67.74
    sel = _make_combined_selection(tmp_path, H0=90.0, Om0=0.3)  # far off
    with pytest.raises(ValueError, match="cosmology|H0"):
        validate_export(pe, sel)


def test_validate_fails_on_source_class_mismatch(tmp_path):
    # PE filtered to BBH only; selection left unfiltered (all classes).
    pe = _make_pe_export(tmp_path, source_class="bbh")
    sel = _make_combined_selection(tmp_path, source_class=None)
    with pytest.raises(ValueError, match="source.class|source_class_filter"):
        validate_export(pe, sel)


def test_validate_source_class_match_passes(tmp_path):
    pe = _make_pe_export(tmp_path, source_class="bbh")
    sel = _make_combined_selection(tmp_path, source_class="bbh")
    results = validate_export(pe, sel)
    assert results["xcheck_source_class"] is True


# ==========================================================================
# 5. PR4 per-event-cosmology arrays path in validate_export
# ==========================================================================
def test_validate_per_event_cosmology_arrays_path(tmp_path):
    """A PE export with per-event cosmologies (cosmology_per_event_varies=True)
    validates against a single-cosmology selection when every per-event value
    is within tolerance -- exercising the PR4 array comparison branch."""
    events = [{"name": "GW900001_000001", "source_class": "BBH", "H0": 67.4},
              {"name": "GW900002_000002", "source_class": "BBH", "H0": 67.9}]
    store = build_pe_store(tmp_path, events, name="peev_store.h5")
    pe = tmp_path / "peev.h5"
    # cosmology=None -> per-event mode; the two differing H0 make it "vary".
    GWCatalog(store).to_darksirens(str(pe), nsamp=8, seed=0, cosmology=None)
    with h5py.File(pe, "r") as f:
        assert bool(f.attrs["cosmology_per_event_varies"]) is True
        assert f.attrs["cosmology_H0_per_event"].shape[0] == 2

    # selection H0 within 1.0 of both 67.4 and 67.9
    sel = _make_combined_selection(tmp_path, H0=67.65, Om0=_Om0,
                                   name="peev_sel.h5")
    results = validate_export(str(pe), sel)
    assert results["xcheck_cosmology"] is True


def test_validate_per_event_cosmology_out_of_tolerance_fails(tmp_path):
    """Per-event cosmologies that do NOT all match the single selection
    cosmology raise a clear error (arrays path)."""
    events = [{"name": "GW900001_000001", "source_class": "BBH", "H0": 60.0},
              {"name": "GW900002_000002", "source_class": "BBH", "H0": 75.0}]
    store = build_pe_store(tmp_path, events, name="pebad_store.h5")
    pe = tmp_path / "pebad.h5"
    GWCatalog(store).to_darksirens(str(pe), nsamp=8, seed=0, cosmology=None)
    sel = _make_combined_selection(tmp_path, H0=_H0, Om0=_Om0,
                                   name="pebad_sel.h5")
    with pytest.raises(ValueError, match="per-event cosmolog"):
        validate_export(str(pe), sel)
