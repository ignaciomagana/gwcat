"""GW-40i at the product level: exact priors in p_pe and pdraw, legacy on request.

Switching the chi_eff prior and the distance prior to their exact evaluations
must change the exported products by EXACTLY the interpolation error of the
replaced factor -- nothing else moves:

  1. PE (chieff): every column but ``p_pe`` is bit-identical to the legacy
     export, and ``ln p_pe`` moves by exactly ``ln p_iso_exact - ln p_iso_grid``
     at each sample (so by at most the grid's documented ~1e-2), with the
     implementation recorded; the CLI flag reproduces the legacy file.
  2. Selection (chieff_reference, chieff): the same for ``ln pdraw``.
  3. Ingest: every sample column but ``p_dL_pe`` is bit-identical, and
     ``p_dL_pe`` moves from bilby's interpolated object to the exact density;
     the implementation is recorded per row and carried to the PE export.
  4. The validator refuses a PE/selection pair built with different chi_eff
     implementations.
"""
import h5py
import numpy as np
import pytest

from gwcat.catalog import GWCatalog
from gwcat.cli import main
from gwcat.cosmology import LAL_PLANCK15, uniform_source_frame_prob
from gwcat.ingest import IngestConfig
from gwcat.selection import SelectionSet
from gwcat.spin import (CHI_EFF_PRIOR_METHODS, chi_eff_prior_impl,
                        chi_eff_prior_logprob)

from test_export_v2_pe import _build_spin_store
from test_ingest_dL_prior_wiring import REPR_USF_LAL, _ingest
from test_selection_chieff_reference import write_o4_nonuniform
from test_selection_spin import write_mixture, write_o4_full

_COSMO = (67.74, 0.3089)
#: The legacy grid's documented worst case in ln p_iso (see
#: test_chi_eff_prior_exact.test_grid_differs_from_exact_only_by_its_...).
_GRID_LN_BOUND = 1.2e-2


def _cols(path):
    with h5py.File(path, "r") as f:
        return ({k: f[k][:] for k in f.keys()
                 if isinstance(f[k], h5py.Dataset)}, dict(f.attrs))


def _s(v):
    return v.decode() if isinstance(v, bytes) else str(v)


# ==========================================================================
# 1. PE
# ==========================================================================
def _pe_pair(tmp_path):
    events = [{"name": "GWx1_000001"}, {"name": "GWx2_000002"},
              {"name": "GWx3_000003", "amax1": 0.8, "amax2": 0.8}]
    store, _ = _build_spin_store(tmp_path, events, n_per_event=400)
    cat = GWCatalog(store)
    new, old = tmp_path / "pe_exact.h5", tmp_path / "pe_grid.h5"
    kw = dict(format="gwcat2", spin_basis="chieff", nsamp=256, seed=0,
              cosmology=_COSMO)
    cat.export(str(new), **kw)
    with chi_eff_prior_impl("grid"):
        cat.export(str(old), **kw)
    return store, new, old


def test_pe_values_change_only_by_the_chi_eff_interpolation_error(tmp_path):
    _, new, old = _pe_pair(tmp_path)
    a, attrs_a = _cols(new)
    b, attrs_b = _cols(old)
    assert set(a) == set(b)
    for k in a:
        if k != "p_pe":
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)

    nsamp = int(attrs_a["nsamp"])
    amax = np.repeat(np.asarray(attrs_a["chi_eff_amax_1_per_event"], float),
                     nsamp)
    d_expected = np.empty_like(a["p_pe"])
    for am in np.unique(amax):
        m = amax == am
        d_expected[m] = (
            chi_eff_prior_logprob(a["chieff"][m], a["m1src"][m], a["m2src"][m],
                                  amax=am, impl="exact")
            - chi_eff_prior_logprob(a["chieff"][m], a["m1src"][m],
                                    a["m2src"][m], amax=am, impl="grid"))
    d = np.log(a["p_pe"]) - np.log(b["p_pe"])
    np.testing.assert_allclose(d, d_expected, rtol=0, atol=2e-15)
    assert np.max(np.abs(d)) <= _GRID_LN_BOUND
    assert np.max(np.abs(d)) > 0.0

    assert _s(attrs_a["chi_eff_prior_impl"]) == "exact"
    assert _s(attrs_b["chi_eff_prior_impl"]) == "grid"
    assert _s(attrs_a["chi_eff_prior_method"]) == CHI_EFF_PRIOR_METHODS["exact"]
    assert _s(attrs_b["chi_eff_prior_method"]) == CHI_EFF_PRIOR_METHODS["grid"]


def test_pe_cli_flag_reproduces_the_legacy_export(tmp_path):
    store, new, old = _pe_pair(tmp_path)
    args = ["export", "pe", store, "--format", "gwcat2",
            "--parameter-space", "chieff", "--nsamp", "256", "--seed", "0",
            "--cosmology", "67.74,0.3089", "--no-summary"]
    cli_old, cli_new = tmp_path / "cli_grid.h5", tmp_path / "cli_exact.h5"
    assert main(args + ["--out", str(cli_old), "--legacy-grid-priors"]) == 0
    assert main(args + ["--out", str(cli_new)]) == 0
    for got, want in ((cli_old, old), (cli_new, new)):
        g, ga = _cols(got)
        w, _ = _cols(want)
        for k in w:
            np.testing.assert_array_equal(g[k], w[k], err_msg=k)
    assert _s(_cols(cli_old)[1]["chi_eff_prior_impl"]) == "grid"
    assert _s(_cols(cli_new)[1]["chi_eff_prior_impl"]) == "exact"


def test_v1_exporter_follows_the_same_switch(tmp_path):
    events = [{"name": "GWv1_000001"}, {"name": "GWv1_000002"}]
    store, _ = _build_spin_store(tmp_path, events, n_per_event=300)
    cat = GWCatalog(store)
    new, old = tmp_path / "ds_exact.h5", tmp_path / "ds_grid.h5"
    kw = dict(nsamp=128, seed=0, cosmology=_COSMO, write_summary=False)
    cat.to_darksirens(str(new), **kw)
    with chi_eff_prior_impl("grid"):
        cat.to_darksirens(str(old), **kw)
    a, aa = _cols(new)
    b, ba = _cols(old)
    for k in a:
        if k != "p_pe":
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)
    d = np.log(a["p_pe"]) - np.log(b["p_pe"])
    d_expected = (chi_eff_prior_logprob(a["chieff"], a["m1src"], a["m2src"],
                                        impl="exact")
                  - chi_eff_prior_logprob(a["chieff"], a["m1src"], a["m2src"],
                                          impl="grid"))
    np.testing.assert_allclose(d, d_expected, rtol=0, atol=2e-15)
    assert _s(aa["chi_eff_prior_impl"]) == "exact"
    assert _s(ba["chi_eff_prior_impl"]) == "grid"


# ==========================================================================
# 2. Selection
# ==========================================================================
@pytest.mark.parametrize("basis", ["chieff_reference", "chieff",
                                   "chieff_reference_cartesian"])
def test_selection_values_change_only_by_the_chi_eff_interpolation_error(
        tmp_path, basis):
    if basis == "chieff_reference":
        path = write_o4_nonuniform(tmp_path / "o4.hdf", n=300, seed=7)
        kw = dict(spin_basis=basis, spin_reference_amax=0.99)
        amax = 0.99
    elif basis == "chieff_reference_cartesian":
        # a joint-density CARTESIAN-spin file (the O1/O2 mixture format):
        # chi_eff comes from the cartesian components, then the same factor
        path = write_mixture(tmp_path / "mix.hdf", "cartesian", seed=4,
                             amax=(0.5, 1.0), n=300)
        kw = dict(spin_basis="chieff_reference", spin_reference_amax=0.99)
        amax = 0.99
    else:
        path = write_o4_full(tmp_path / "o4.hdf", n=300, amax=(0.9, 0.9),
                             seed=7)
        kw = dict(spin_basis=basis)
        amax = 0.9
    new, old = tmp_path / "sel_exact.h5", tmp_path / "sel_grid.h5"
    SelectionSet(path).export(str(new), **kw)
    with chi_eff_prior_impl("grid"):
        SelectionSet(path).export(str(old), **kw)
    a, aa = _cols(new)
    b, ba = _cols(old)
    assert set(a) == set(b)
    for k in a:
        if k != "pdraw":
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)
    ok = (a["pdraw"] < 1e299) & (b["pdraw"] < 1e299)   # reference sentinels
    np.testing.assert_array_equal(a["pdraw"][~ok], b["pdraw"][~ok])
    d = np.log(a["pdraw"][ok]) - np.log(b["pdraw"][ok])
    d_expected = (
        chi_eff_prior_logprob(a["chieff"][ok], a["m1src"][ok], a["m2src"][ok],
                              amax=amax, impl="exact")
        - chi_eff_prior_logprob(a["chieff"][ok], a["m1src"][ok],
                                a["m2src"][ok], amax=amax, impl="grid"))
    np.testing.assert_allclose(d, d_expected, rtol=0, atol=2e-15)
    assert 0.0 < np.max(np.abs(d)) <= _GRID_LN_BOUND
    assert _s(aa["chi_eff_prior_impl"]) == "exact"
    assert _s(ba["chi_eff_prior_impl"]) == "grid"


def test_selection_cli_flag(tmp_path):
    path = write_o4_nonuniform(tmp_path / "o4.hdf", n=200, seed=9)
    base = ["export", "selection", str(path), "--parameter-space",
            "chieff_reference", "--spin-reference-amax", "0.99",
            "--no-summary"]
    o_new, o_old = tmp_path / "n.h5", tmp_path / "o.h5"
    assert main(base + ["--out", str(o_new)]) == 0
    assert main(base + ["--out", str(o_old), "--legacy-grid-priors"]) == 0
    assert _s(_cols(o_new)[1]["chi_eff_prior_impl"]) == "exact"
    assert _s(_cols(o_old)[1]["chi_eff_prior_impl"]) == "grid"
    ref = tmp_path / "ref.h5"
    with chi_eff_prior_impl("grid"):
        SelectionSet(path).export(str(ref), spin_basis="chieff_reference",
                                  spin_reference_amax=0.99,
                                  write_summary=False)
    np.testing.assert_array_equal(_cols(o_old)[0]["pdraw"],
                                  _cols(ref)[0]["pdraw"])


# ==========================================================================
# 3. Ingest: the distance prior
# ==========================================================================
def test_ingest_dL_prior_changes_only_by_bilbys_interpolation_error(
        tmp_path, monkeypatch):
    pytest.importorskip("bilby")
    (tmp_path / "new").mkdir()
    (tmp_path / "old").mkdir()
    new = _ingest(tmp_path / "new", monkeypatch, dl_repr=REPR_USF_LAL,
                  filename="GWTC-3_GW950101_000101_nocosmo.h5")
    old = _ingest(tmp_path / "old", monkeypatch, dl_repr=REPR_USF_LAL,
                  filename="GWTC-3_GW950101_000101_nocosmo.h5",
                  cfg=IngestConfig(validate_prior=False, dL_prior_impl="auto"))
    with h5py.File(new, "r") as fa, h5py.File(old, "r") as fb:
        for k in fa["samples"]:
            if k != "p_dL_pe":
                np.testing.assert_array_equal(fa["samples"][k][:],
                                              fb["samples"][k][:], err_msg=k)
        dL = fa["samples/luminosity_distance"][:]
        pa = fa["samples/p_dL_pe"][:]
        pb = fb["samples/p_dL_pe"][:]
        assert _s(fa["meta/dL_prior_impl"][0]) == "exact"
        assert _s(fb["meta/dL_prior_impl"][0]) == "bilby"
    # exact: the declared bounds [10, 4000] at LAL Planck15, no widening
    np.testing.assert_allclose(
        pa, uniform_source_frame_prob(dL, LAL_PLANCK15, 10.0, 4000.0),
        rtol=1e-15, atol=0)
    np.testing.assert_array_equal(
        pb, uniform_source_frame_prob(dL, LAL_PLANCK15, 10.0, 4000.0,
                                      impl="bilby"))
    d = np.log(pa) - np.log(pb)
    assert 0.0 < np.ptp(d) < 1e-4       # bilby's shape error over 300-800 Mpc


def test_ingest_cli_flag_selects_the_legacy_dL_impl(monkeypatch, tmp_path):
    import gwcat.ingest as ing
    seen = {}

    def fake_build_store(paths, out, **kw):
        seen["cfg"] = kw["cfg"]
    monkeypatch.setattr(ing, "build_store", fake_build_store)
    f = tmp_path / "GWTC-3_GW950101_000101_cosmo.h5"
    f.write_bytes(b"")
    ing._cli(["--glob", str(f), "--out", str(tmp_path / "s.h5")],
             _deprecated=False)
    assert seen["cfg"].dL_prior_impl == "exact"
    ing._cli(["--glob", str(f), "--out", str(tmp_path / "s.h5"),
              "--legacy-grid-priors"], _deprecated=False)
    assert seen["cfg"].dL_prior_impl == "auto"


def test_pe_export_carries_the_ingest_dL_impl(tmp_path):
    events = [{"name": "GWd1_000001"}, {"name": "GWd2_000002"}]
    store, _ = _build_spin_store(tmp_path, events, n_per_event=200)
    with h5py.File(store, "r+") as f:
        f["meta"].create_dataset("dL_prior_impl", data=np.array(
            ["exact", "bilby"], dtype=h5py.string_dtype()))
    out = tmp_path / "pe.h5"
    GWCatalog(store).export(str(out), format="gwcat2", spin_basis="chieff",
                            nsamp=64, seed=0, cosmology=_COSMO)
    with h5py.File(out, "r") as f:
        assert [_s(x) for x in f.attrs["dL_prior_impl_per_event"]] == [
            "exact", "bilby"]


# ==========================================================================
# 4. The validator refuses a mixed pair
# ==========================================================================
def test_validator_refuses_a_pair_with_different_chi_eff_impls():
    from gwcat.export.validate import _xcheck_chi_eff_prior_impl
    fails, results = [], {}

    def _fail(name, msg):
        fails.append((name, msg))
    _xcheck_chi_eff_prior_impl(_fail, results, {"chi_eff_prior_impl": "exact"},
                               {"chi_eff_prior_impl": b"grid"})
    assert fails and fails[0][0] == "xcheck_chi_eff_prior_impl"
    assert "different chi_eff densities" in fails[0][1]

    fails.clear()
    _xcheck_chi_eff_prior_impl(_fail, results, {"chi_eff_prior_impl": "exact"},
                               {"chi_eff_prior_impl": "exact"})
    assert not fails and results["xcheck_chi_eff_prior_impl"] is True

    with pytest.warns(UserWarning, match="one side only"):
        _xcheck_chi_eff_prior_impl(_fail, results,
                                   {"chi_eff_prior_impl": "exact"}, {})
    assert not fails
