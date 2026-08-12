"""Tests for the versioned export pipeline (PR 3).

Covers:
  * PARITY -- ``GWCatalog.export(format="gwcat2", spin_basis="chieff")`` (i.e.
    ``build_pe_product`` + the gwcat2 writer) reproduces the legacy
    ``GWCatalog.to_darksirens`` arrays byte-for-byte for identical kwargs, under
    the default case and under far_max, z_max, source_class, small-nsamp
    (replace=False) and large-nsamp (replace=True) cuts, plus per-event
    cosmology.  Exact ``np.testing.assert_array_equal`` is used throughout (the
    builder mirrors the legacy exporter's exact rng-consumption and floating
    -point operation order -- see ``gwcat/export/pe_builder.py``), so no
    ``assert_allclose`` fallback is needed.
  * ATTR CONTRACT -- the gwcat-pe-2.0 file carries ``spin_basis`` and every
    legacy provenance attr with the same value (excluding ``format_version``),
    including the legacy-compat spin attrs.
  * REGISTRY -- duplicate registration raises; unknown lookup raises listing
    known formats; ``list_formats`` works.
  * NOT-IMPLEMENTED -- ``spin_basis`` in {"component", "chieff_chip"} and the
    top-level ``export()`` on a selection object raise NotImplementedError.

Fixtures are tiny synthetic HDF5 stores built directly with h5py (matching the
on-disk schema GWCatalog reads) -- no network, no pesummary/ingest.
"""
import numpy as np
import h5py
import pytest

from gwcat.catalog import GWCatalog
from gwcat.export import (export, build_pe_product, get_exporter,
                          register_exporter, list_formats, ExportProduct)


# ==========================================================================
# Fixture builder
# ==========================================================================
DARKSIRENS_PARAMS = ["mass_1", "mass_2", "luminosity_distance", "ra", "dec",
                     "chi_eff", "p_dL_pe"]

#: 2 BBH (one with NO FAR -- far_available=False), 1 NSBH, 1 BNS.
MIXED_EVENTS = [
    {"name": "GW930001_000001", "source_class": "BBH", "far": 5e-4,
     "pastro": 0.99, "waveform": "IMRPhenomXPHM"},
    {"name": "GW930002_000002", "source_class": "BBH", "far": float("nan"),
     "pastro": float("nan"), "waveform": "IMRPhenomXPHM"},
    {"name": "GW930003_000003", "source_class": "NSBH", "far": 2e-3,
     "pastro": 0.95, "waveform": "SEOBNRv5PHM"},
    {"name": "GW930004_000004", "source_class": "BNS", "far": 1e-6,
     "pastro": 0.999, "waveform": "IMRPhenomXPHM"},
]


def _build_store(tmp_path, events=MIXED_EVENTS, n_per_event=400,
                 H0=67.74, Om0=0.3089, seed=17, name="store.h5"):
    """Tiny synthetic store with source-class + FAR + waveform + per-event
    cosmology metadata (matches the on-disk schema GWCatalog reads)."""
    rng = np.random.default_rng(seed)
    offsets = [0]
    cols = {p: [] for p in DARKSIRENS_PARAMS}
    meta = {k: [] for k in ["source_class", "compact_type", "far",
                            "far_available", "pastro", "p_astro",
                            "dL_prior_H0", "dL_prior_Om0", "waveform",
                            "approximant", "sample_set_name"]}
    names = []
    for ev in events:
        n = int(ev.get("n", n_per_event))
        cols["mass_1"].append(rng.uniform(20, 45, n))
        cols["mass_2"].append(rng.uniform(8, 20, n))
        cols["luminosity_distance"].append(rng.uniform(300, 800, n))
        cols["ra"].append(rng.uniform(0, 2 * np.pi, n))
        cols["dec"].append(rng.uniform(-np.pi / 2, np.pi / 2, n))
        cols["chi_eff"].append(rng.uniform(-0.3, 0.3, n))
        cols["p_dL_pe"].append(rng.uniform(0.1, 1.0, n))
        offsets.append(offsets[-1] + n)

        far = float(ev.get("far", np.nan))
        names.append(ev["name"])
        meta["source_class"].append(ev["source_class"])
        meta["compact_type"].append(ev["source_class"])
        meta["far"].append(far)
        meta["far_available"].append(1.0 if np.isfinite(far) else 0.0)
        meta["pastro"].append(float(ev.get("pastro", np.nan)))
        meta["p_astro"].append(float(ev.get("pastro", np.nan)))
        meta["dL_prior_H0"].append(H0)
        meta["dL_prior_Om0"].append(Om0)
        meta["waveform"].append(ev.get("waveform", "IMRPhenomXPHM"))
        meta["approximant"].append(ev.get("waveform", "IMRPhenomXPHM"))
        meta["sample_set_name"].append(f"C01:{ev.get('waveform', 'X')}")

    path = tmp_path / name
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "1.2"
        f.attrs["param_names"] = np.array(DARKSIRENS_PARAMS,
                                          dtype=h5py.string_dtype())
        idx = f.create_group("index")
        idx.create_dataset("offsets", data=np.array(offsets, dtype="i8"))
        idx.create_dataset("event_names",
                           data=np.array(names, dtype=h5py.string_dtype()))
        mg = f.create_group("meta")
        for k in ["source_class", "compact_type", "waveform", "approximant",
                  "sample_set_name"]:
            mg.create_dataset(k, data=np.array(meta[k], dtype=h5py.string_dtype()))
        for k in ["far", "far_available", "pastro", "p_astro",
                  "dL_prior_H0", "dL_prior_Om0"]:
            mg.create_dataset(k, data=np.asarray(meta[k], dtype="f8"))
        sg = f.create_group("samples")
        for p in DARKSIRENS_PARAMS:
            sg.create_dataset(p, data=np.concatenate(cols[p]))
    return str(path)


_SHARED_DATASETS = ["ra", "dec", "m1det", "m2det", "chieff", "dL", "p_pe",
                    "redshift", "m1src", "m2src"]


def _assert_parity(tmp_path, store, tag, **kwargs):
    """Legacy to_darksirens vs new export must be array-equal on all 10
    shared datasets for identical kwargs."""
    cat = GWCatalog(store)
    legacy = tmp_path / f"legacy_{tag}.h5"
    v2 = tmp_path / f"v2_{tag}.h5"
    cat.to_darksirens(str(legacy), **kwargs)
    cat.export(str(v2), format="gwcat2", spin_basis="chieff", **kwargs)
    with h5py.File(legacy, "r") as fa, h5py.File(v2, "r") as fb:
        assert int(fa.attrs["nobs"]) == int(fb.attrs["nobs"])
        for k in _SHARED_DATASETS:
            np.testing.assert_array_equal(
                fa[k][:], fb[k][:],
                err_msg=f"[{tag}] dataset {k!r} differs")
    return legacy, v2


# ==========================================================================
# Parity
# ==========================================================================
def test_parity_default(tmp_path):
    store = _build_store(tmp_path)
    _, v2 = _assert_parity(tmp_path, store, "default",
                           nsamp=64, seed=0, cosmology=(67.74, 0.3089))
    with h5py.File(v2, "r") as f:
        # Non-trivial output, and the v2 stamp.
        assert f["m1det"].shape[0] == 4 * 64
        assert f.attrs["format_version"] == "gwcat-pe-2.0"


def test_parity_far_max_cut(tmp_path):
    store = _build_store(tmp_path)
    # far_max with allow_missing_far keeps the missing-FAR BBH event.
    _assert_parity(tmp_path, store, "farmax", nsamp=48, seed=1,
                   cosmology=(67.74, 0.3089), far_max=1.0,
                   allow_missing_far=True)


def test_parity_z_max_cut(tmp_path):
    store = _build_store(tmp_path)
    # z_max drops a fraction of each event's samples (exercises idx_map).
    _assert_parity(tmp_path, store, "zmax", nsamp=32, seed=2,
                   cosmology=(67.74, 0.3089), z_max=0.12)


def test_parity_source_class_filter(tmp_path):
    store = _build_store(tmp_path)
    legacy, v2 = _assert_parity(tmp_path, store, "srcclass", nsamp=40, seed=3,
                                cosmology=(67.74, 0.3089), source_class="bbh")
    with h5py.File(v2, "r") as f:
        assert int(f.attrs["nobs"]) == 2  # only the two BBH events


def test_parity_replace_false_branch(tmp_path):
    """replace='auto' with nsamp < n_samples -> replace=False branch."""
    store = _build_store(tmp_path, n_per_event=400)
    _assert_parity(tmp_path, store, "replace_false", nsamp=64, seed=4,
                   cosmology=(67.74, 0.3089))


def test_parity_replace_true_branch(tmp_path):
    """replace='auto' with nsamp > n_samples -> replace=True branch."""
    store = _build_store(tmp_path, n_per_event=50)
    _assert_parity(tmp_path, store, "replace_true", nsamp=200, seed=5,
                   cosmology=(67.74, 0.3089))


def test_parity_per_event_cosmology(tmp_path):
    """cosmology=None reads the per-event stored PE cosmology."""
    store = _build_store(tmp_path, H0=70.0, Om0=0.3)
    legacy, v2 = _assert_parity(tmp_path, store, "per_event",
                                nsamp=32, seed=6)  # cosmology=None default
    with h5py.File(v2, "r") as f:
        assert f.attrs["cosmology_mode"] == "per-event"
        assert f.attrs["pe_cosmology_H0"] == pytest.approx(70.0)


def test_parity_top_level_export_function(tmp_path):
    """The top-level gwcat.export.export() dispatches to GWCatalog.export()."""
    store = _build_store(tmp_path)
    cat = GWCatalog(store)
    legacy = tmp_path / "legacy_fn.h5"
    v2 = tmp_path / "v2_fn.h5"
    kw = dict(nsamp=24, seed=7, cosmology=(67.74, 0.3089))
    cat.to_darksirens(str(legacy), **kw)
    export(cat, str(v2), format="gwcat2", spin_basis="chieff", **kw)
    with h5py.File(legacy, "r") as fa, h5py.File(v2, "r") as fb:
        for k in _SHARED_DATASETS:
            np.testing.assert_array_equal(fa[k][:], fb[k][:])


# ==========================================================================
# Attr contract
# ==========================================================================
def test_attr_contract_matches_legacy(tmp_path):
    """The gwcat-pe-2.0 file carries spin_basis plus every legacy provenance
    attr with the same value (excluding format_version)."""
    store = _build_store(tmp_path)
    cat = GWCatalog(store)
    legacy = tmp_path / "legacy_attrs.h5"
    v2 = tmp_path / "v2_attrs.h5"
    kw = dict(nsamp=16, seed=0, cosmology=(67.74, 0.3089), far_max=1.0,
              allow_missing_far=True)
    cat.to_darksirens(str(legacy), **kw)
    cat.export(str(v2), format="gwcat2", spin_basis="chieff", **kw)

    with h5py.File(legacy, "r") as fa, h5py.File(v2, "r") as fb:
        assert fb.attrs["format_version"] == "gwcat-pe-2.0"
        assert fb.attrs["spin_basis"] == "chieff"
        # Legacy-compat spin attrs (chieff basis is always "include").
        assert fb.attrs["spin_prior_mode"] == "include"
        assert bool(fb.attrs["chi_eff_prior_applied_to_p_pe"]) is True
        assert bool(fb.attrs["chi_eff_in_p_pe"]) is True

        # Every legacy attr except format_version must match.
        for key in fa.attrs:
            if key == "format_version":
                continue
            assert key in fb.attrs, f"v2 file missing legacy attr {key!r}"
            va, vb = fa.attrs[key], fb.attrs[key]
            if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
                assert np.array_equal(np.asarray(va), np.asarray(vb)), (
                    f"attr {key!r} differs: {va!r} vs {vb!r}")
            else:
                assert va == vb, f"attr {key!r} differs: {va!r} vs {vb!r}"


# ==========================================================================
# write_summary feed
# ==========================================================================
def test_write_summary_feed(tmp_path):
    store = _build_store(tmp_path)
    cat = GWCatalog(store)
    out = tmp_path / "with_summary.h5"
    cat.export(str(out), format="gwcat2", spin_basis="chieff",
               write_summary=True, source_class="bbh",
               cosmology=(67.74, 0.3089), nsamp=16, seed=0)
    import json
    from pathlib import Path
    sj = json.loads(
        Path(str(out) + ".validation_summary.json").read_text())
    assert sj["kind"] == "darksirens_export"
    assert sj["output_path"] == str(out)
    assert sj["n_events_exported"] == 2
    assert sj["spin_basis"] == "chieff"
    assert sj["source_class_filter"] == "BBH"
    assert Path(str(out) + ".validation_summary.md").exists()


def test_write_summary_context_merged(tmp_path):
    store = _build_store(tmp_path)
    cat = GWCatalog(store)
    out = tmp_path / "ctx_summary.h5"
    cat.export(str(out), format="gwcat2", write_summary=True,
               summary_context={"analyst": "unit-test"},
               cosmology=(67.74, 0.3089), nsamp=8, seed=0)
    import json
    from pathlib import Path
    sj = json.loads(
        Path(str(out) + ".validation_summary.json").read_text())
    assert sj["analyst"] == "unit-test"


# ==========================================================================
# Registry
# ==========================================================================
def test_list_formats_contains_gwcat2_pe():
    formats = list_formats()
    assert ("gwcat2", "pe") in formats
    # sorted, deterministic
    assert formats == sorted(formats)


def test_get_exporter_returns_callable():
    writer = get_exporter("gwcat2", "pe")
    assert callable(writer)


def test_duplicate_registration_raises():
    with pytest.raises(ValueError, match="already registered"):
        register_exporter("gwcat2", "pe")(lambda *a, **k: None)


def test_unknown_format_raises_listing_known():
    with pytest.raises(ValueError) as exc:
        get_exporter("does-not-exist", "pe")
    msg = str(exc.value)
    assert "does-not-exist" in msg
    assert "gwcat2" in msg  # lists the known format(s)


def test_registration_and_lookup_roundtrip():
    """A fresh (name, kind) registration is retrievable, then cleaned up so the
    global registry is left untouched for other tests."""
    from gwcat.export import registry as _reg

    @register_exporter("unit-test-fmt", "pe")
    def _writer(product, out_path, **kw):  # pragma: no cover - not invoked
        return out_path

    try:
        assert get_exporter("unit-test-fmt", "pe") is _writer
        assert ("unit-test-fmt", "pe") in list_formats()
    finally:
        _reg._REGISTRY.pop(("unit-test-fmt", "pe"), None)
    assert ("unit-test-fmt", "pe") not in list_formats()


# ==========================================================================
# Missing-spin-columns / type errors
# ==========================================================================
@pytest.mark.parametrize("basis", ["component", "chieff_chip"])
def test_spin_basis_requires_spin_columns(tmp_path, basis):
    """The component / chieff_chip bases (implemented in PR 6) fail loudly on a
    store that carries no spin columns -- the darksirens fixture store has no
    a_1/a_2/tilt/chi_p -- naming the missing parameters."""
    from gwcat.schema import MissingParameterError

    store = _build_store(tmp_path)
    cat = GWCatalog(store)
    with pytest.raises(MissingParameterError):
        build_pe_product(cat, spin_basis=basis, nsamp=8,
                         allow_projection_basis=True,
                         cosmology=(67.74, 0.3089))


def test_unknown_spin_basis_raises_value_error(tmp_path):
    store = _build_store(tmp_path)
    cat = GWCatalog(store)
    with pytest.raises(ValueError, match="unknown spin_basis"):
        build_pe_product(cat, spin_basis="nonsense", nsamp=8,
                         cosmology=(67.74, 0.3089))


def test_export_selection_object_dispatches(tmp_path):
    """Top-level export() on a selection object dispatches to the selection
    builder + writer (PR5), producing a gwcat-selection-2.0 file."""
    from gwcat.selection import SelectionSet

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
        ("lnpdraw_z", "f8"), ("pycbc_far", "f8"),
    ]
    inj = tmp_path / "inj.hdf"
    n = 8
    ev = np.zeros(n, dtype=_O4_FIELDS)
    ev["mass1_source"] = 32.0
    ev["mass2_source"] = 28.0
    ev["z"] = 0.1
    ev["mass1_detector"] = 32.0 * 1.1
    ev["mass2_detector"] = 28.0 * 1.1
    ev["luminosity_distance"] = 450.0
    ev["dluminosity_distance_dredshift"] = 4500.0
    ev["weights"] = 1.0
    ev["pycbc_far"] = 0.1
    with h5py.File(inj, "w") as f:
        f.attrs["total_analysis_time"] = 365.25 * 24 * 3600
        f.attrs["total_generated"] = 1000
        f.attrs["searches"] = np.array([b"pycbc"])
        f.create_dataset("events", data=ev)

    sel = SelectionSet(str(inj))
    # This minimal fixture has no per-spin draw columns, so the default
    # component basis is unavailable; the chieff basis (1-D chi_eff swap)
    # needs only chi_eff and works.
    out = tmp_path / "out.h5"
    export(sel, str(out), spin_basis="chieff")
    with h5py.File(out, "r") as f:
        assert f.attrs["format_version"] == "gwcat-selection-2.0"
        assert f.attrs["spin_basis"] == "chieff"


# ==========================================================================
# CLI smoke tests (invoke gwcat.cli.main directly -- no subprocess/network)
# ==========================================================================
def test_cli_export_pe_smoke(tmp_path):
    from gwcat.cli import main

    store = _build_store(tmp_path)
    out = tmp_path / "cli_pe.h5"
    rc = main(["export", "pe", store, "--out", str(out),
               "--source-class", "bbh", "--cosmology", "67.74,0.3089",
               "--nsamp", "16", "--seed", "0"])
    assert rc == 0
    assert out.exists()
    with h5py.File(out, "r") as f:
        assert f.attrs["format_version"] == "gwcat-pe-2.0"
        assert f.attrs["spin_basis"] == "chieff"
        assert int(f.attrs["nobs"]) == 2

    # Summary is written by default.
    from pathlib import Path
    assert Path(str(out) + ".validation_summary.json").exists()


def test_cli_export_pe_no_summary(tmp_path):
    from gwcat.cli import main
    from pathlib import Path

    store = _build_store(tmp_path)
    out = tmp_path / "cli_pe_nosum.h5"
    rc = main(["export", "pe", store, "--out", str(out),
               "--cosmology", "67.74,0.3089", "--nsamp", "8", "--seed", "0",
               "--no-summary"])
    assert rc == 0
    assert out.exists()
    assert not Path(str(out) + ".validation_summary.json").exists()


def test_cli_export_list_formats(tmp_path, capsys):
    from gwcat.cli import main

    rc = main(["export", "list-formats"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "gwcat2" in out
    assert "pe" in out
