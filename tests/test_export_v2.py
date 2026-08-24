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
from gwcat.schema import MissingParameterError


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

        # mass_prior_basis is the ONE legacy attr the v2 file deliberately
        # states differently (GW-34).  The frozen v1 writer stamps the constant
        # "uniform_detector_frame" on every file regardless of what was
        # ingested; v2 reports the class the store actually carries -- here
        # nothing, because this synthetic store predates the mass-prior ingest.
        # A file may only claim the verified basis when every row it holds
        # carries it.
        assert fa.attrs["mass_prior_basis"] == "uniform_detector_frame"
        assert fb.attrs["mass_prior_basis"] == "unstated"
        assert bool(fb.attrs["mass_prior_verified"]) is False

        # Every other legacy attr except format_version must match.  NaN is a
        # VALUE here, not a missing one -- "no cut on this statistic" is written
        # as NaN because HDF5 has no null -- so the two sides agreeing on NaN is
        # agreement, which bare `==` would call a difference.
        for key in fa.attrs:
            if key in ("format_version", "mass_prior_basis"):
                continue
            assert key in fb.attrs, f"v2 file missing legacy attr {key!r}"
            va, vb = fa.attrs[key], fb.attrs[key]
            if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
                assert np.array_equal(np.asarray(va), np.asarray(vb),
                                      equal_nan=False
                                      if np.asarray(va).dtype.kind not in "fc"
                                      else True), (
                    f"attr {key!r} differs: {va!r} vs {vb!r}")
            elif isinstance(va, float) and np.isnan(va):
                assert isinstance(vb, float) and np.isnan(vb), (
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
    cat.export(str(out), format="gwcat2", spin_basis="chieff",
               write_summary=True,
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
    # This fixture is a chi_eff-only store, so the space has to be named: the
    # shared default is now `component`, which it cannot supply (GW-22b).
    rc = main(["export", "pe", store, "--out", str(out),
               "--parameter-space", "chieff",
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
               "--spin-basis", "chieff",          # legacy alias still accepted
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


# ==========================================================================
# GW-22b: one default, a registry-driven space argument, and list-spaces
# ==========================================================================
def test_pe_and_selection_share_one_default_space():
    """The two defaults used to differ, so the no-flags pair could never pair.

    `export pe` defaulted to `chieff` and `export selection` to `component`, so
    running both with no arguments produced files that fail the cross-file basis
    check by construction. A default that cannot be used with itself is not one.
    """
    from gwcat.cli import build_parser, DEFAULT_PARAMETER_SPACE

    p = build_parser()
    pe = p.parse_args(["export", "pe", "s.h5", "--out", "o.h5"])
    sel = p.parse_args(["export", "selection", "i.hdf", "--out", "o.h5"])
    assert pe.spin_basis == sel.spin_basis == DEFAULT_PARAMETER_SPACE


def test_space_choices_are_registry_spaces_the_builder_implements():
    """The CLI must offer exactly the spaces the builder for that subcommand
    can build -- no more, no less.

    Offering the full registry advertised `component_6d`/`cartesian`/`aligned`
    (and `nospin` on the selection side), which then crashed in the builder
    with a raw traceback: spellable but unbuildable.  Hardcoding a list here
    instead would drift the other way.  So the choices are the registry order
    filtered to the builder's SUPPORTED_SPIN_BASES.
    """
    from gwcat.cli import build_parser
    from gwcat.params import list_spaces
    from gwcat.export.pe_builder import (
        SUPPORTED_SPIN_BASES as PE_SUPPORTED)
    from gwcat.export.selection_builder import (
        SUPPORTED_SPIN_BASES as SEL_SUPPORTED)

    p = build_parser()
    for sub, supported in (("pe", PE_SUPPORTED), ("selection", SEL_SUPPORTED)):
        head = {"pe": ["export", "pe", "s.h5"],
                "selection": ["export", "selection", "i.hdf"]}[sub]
        for space in list_spaces():
            if space in supported:
                args = p.parse_args(head + ["--out", "o.h5",
                                            "--parameter-space", space])
                assert args.spin_basis == space
            else:
                with pytest.raises(SystemExit):
                    p.parse_args(head + ["--out", "o.h5",
                                         "--parameter-space", space])


def test_validator_known_bases_match_the_builders():
    """The validator must accept exactly what each builder can write.

    Its own hardcoded list rejected valid nospin PE files and claimed to
    accept selection bases the selection builder never produced.
    """
    from gwcat.export import validate, pe_builder, selection_builder

    assert validate._PE_BASES == pe_builder.SUPPORTED_SPIN_BASES
    assert validate._SEL_BASES == selection_builder.SUPPORTED_SPIN_BASES


def test_spin_basis_is_still_accepted_as_an_alias():
    from gwcat.cli import build_parser

    p = build_parser()
    args = p.parse_args(["export", "pe", "s.h5", "--out", "o.h5",
                         "--spin-basis", "chieff"])
    assert args.spin_basis == "chieff"


def test_unknown_space_is_rejected():
    from gwcat.cli import build_parser

    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["export", "pe", "s.h5", "--out", "o.h5",
                      "--parameter-space", "not_a_space"])


def test_cli_export_list_spaces(capsys):
    from gwcat.cli import main
    from gwcat.params import list_spaces

    assert main(["export", "list-spaces"]) == 0
    out = capsys.readouterr().out
    for space in list_spaces():
        assert space in out
    # the load-bearing declaration, not just the names
    assert "projection" in out and "bijective" in out


def test_missing_parameter_error_names_the_usable_spaces(tmp_path):
    """A chi_eff-only store must say what it CAN export, not only what it lacks.

    With `component` the shared default this is the first error such a store
    produces, so it has to be a signpost rather than a wall.
    """
    from gwcat.cli import main

    store = _build_store(tmp_path)
    with pytest.raises(MissingParameterError) as ei:
        main(["export", "pe", store, "--out", str(tmp_path / "x.h5"),
              "--cosmology", "67.74,0.3089", "--nsamp", "8"])
    msg = str(ei.value)
    assert "a_1" in msg                    # what is missing
    assert "chieff" in msg                 # what would work instead
    assert "--parameter-space" in msg      # how to ask for it


def test_summary_requirements_follow_the_requested_space(tmp_path):
    """`missing_required_parameters` was hardcoded to the chi_eff-only list.

    A component export therefore reported nothing missing on a store that could
    not supply it -- the summary said fine right up until the builder raised.
    """
    from gwcat.catalog import GWCatalog
    from gwcat.validation_summary import summarize_catalog

    cat = GWCatalog(_build_store(tmp_path))
    chieff = summarize_catalog(cat, parameter_space="chieff")
    component = summarize_catalog(cat, parameter_space="component")

    assert chieff["missing_required_parameters"] == []
    assert set(component["missing_required_parameters"]) >= {"a_1", "a_2"}
    assert component["required_for_parameter_space"] == "component"
    # The ingest context (no export in view) keeps the legacy behaviour.
    assert summarize_catalog(cat)["required_for_parameter_space"] is None


def test_every_no_argument_export_path_agrees_on_the_default():
    """CLI and Python API must not disagree about what "the default" is.

    GW-22b fixed the CLI's two defaults but left `GWCatalog.export` on `chieff`
    while `SelectionSet.export` was on `component`, so the no-argument *Python*
    pair still failed the basis cross-check -- the same defect one layer down,
    reachable by the more commonly used path. One constant now feeds all four.
    """
    import inspect

    from gwcat.cli import build_parser
    from gwcat.params import DEFAULT_PARAMETER_SPACE
    from gwcat.catalog import GWCatalog
    from gwcat.selection import SelectionSet, CombinedSelectionSet
    from gwcat.export import build_pe_product
    from gwcat.export.selection_builder import build_selection_product

    p = build_parser()
    assert p.parse_args(["export", "pe", "s.h5", "--out", "o.h5"]
                        ).spin_basis == DEFAULT_PARAMETER_SPACE
    assert p.parse_args(["export", "selection", "i.hdf", "--out", "o.h5"]
                        ).spin_basis == DEFAULT_PARAMETER_SPACE

    # The builders default to the constant directly...
    for fn in (build_pe_product, build_selection_product):
        assert (inspect.signature(fn).parameters["spin_basis"].default
                == DEFAULT_PARAMETER_SPACE), fn.__name__

    # ...and the user-facing methods resolve None to it, so the constant is not
    # baked into a signature default that a stale import could pin.
    for meth in (GWCatalog.export, SelectionSet.export,
                 CombinedSelectionSet.export):
        assert inspect.signature(meth).parameters["spin_basis"].default is None, \
            meth.__qualname__

    # ...and the top-level gwcat.export.export() dispatcher forwards
    # spin_basis untouched instead of reinstating per-type defaults in the
    # None slot -- it carried "chieff" for PE / "component" for selection
    # after both layers around it were fixed.
    import gwcat.export as export_mod

    seen = {}
    for cls, key in ((GWCatalog, "pe"), (SelectionSet, "sel")):
        obj = cls.__new__(cls)
        obj.export = (lambda out_path, key=key, **kw:
                      seen.__setitem__(key, kw.get("spin_basis", "MISSING")))
        export_mod.export(obj, "o.h5")
    assert seen == {"pe": None, "sel": None}


def test_v1_required_datasets_match_what_the_v1_writer_emits(tmp_path):
    """The v1 validator's mandated list must be exactly what the v1 writers
    emit -- the same "does the checker know what the builder writes?" pin as
    :func:`test_validator_known_bases_match_the_builders`.

    Before GW-9 the validator had no mandated list at all: it checked a
    required dataset only when it happened to exist, so a file missing p_pe,
    the masses or the sky produced no failure.
    """
    from gwcat.catalog import V1_PE_REQUIRED

    store = _build_store(tmp_path, name="v1req_store.h5")
    out = tmp_path / "v1req.h5"
    GWCatalog(store).to_darksirens(str(out), nsamp=8, seed=0)
    with h5py.File(out, "r") as f:
        assert set(f.keys()) == set(V1_PE_REQUIRED)


def test_public_validate_export_accepts_a_freshly_built_v2_export(tmp_path):
    """gwcat.validate_export dispatches on format_version, so a gwcat-2.0 file
    is checked against the v2 schema instead of the v1 one."""
    import gwcat

    store = _build_store(tmp_path, name="pubval_store.h5")
    out = tmp_path / "pubval.h5"
    GWCatalog(store).export(str(out), format="gwcat2", spin_basis="chieff",
                            nsamp=8, seed=0)
    results = gwcat.validate_export(str(out))
    assert all(results.values()), \
        f"unexpected failures: {[k for k, v in results.items() if not v]}"
    assert "pe_has_p_pe" in results        # the v2 schema's presence checks ran
