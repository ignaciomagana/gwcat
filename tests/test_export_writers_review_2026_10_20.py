"""The export writers against the 2026-10-20 review (writers_gwcat2 / atomic).

Each test is the review's failure scenario, reduced to a synthetic store or a
hand-built product: no network, no ingest, no production artifact.  Every one
fails on 3ba1dbf.

  * a PE file cut on FAR AND the posterior-median SNR declared itself
    ``far_or_snr`` -- the statistic of the looser injection mask it then paired
    with -- and the validator compared thresholds only;
  * no PE export recorded the distance-prior class it divided out, so the 2.1
    contract's ``dL_prior_kind`` was always null;
  * a re-export left the previous product's validation summary beside the new
    file, the summary was written in place, and nothing tied it to its file;
  * the files recorded nothing about the run environment that sets their last
    bits;
  * the atomic write replaced a per-file symlink with a regular file;
  * chieff_chip files said ``chi_eff_prior_applied_to_p_pe=False`` while their
    summary said True;
  * the 2.1 contract hard-coded ``sky_prior_in_density=True`` although neither
    density carries a sky factor;
  * a sky-marginal (or NaN-sky) selection product was written without saying
    that darksirens-core cannot load it.
"""
import hashlib
import json
import os

import h5py
import numpy as np
import pytest

from gwcat.catalog import GWCatalog
from gwcat.export import ExportProduct, build_pe_product, get_exporter

from test_export_writer_hardening import _build_store, _selection_product

_STR = h5py.string_dtype()


def _store(tmp_path, name="store.h5", **meta):
    """The writer-hardening two-BBH store plus extra per-event meta columns."""
    path = _build_store(tmp_path, name=name)
    with h5py.File(path, "a") as f:
        for k, v in meta.items():
            v = np.asarray(v)
            if v.dtype.kind in "US":
                f["meta"].create_dataset(k, data=v.astype(object), dtype=_STR)
            else:
                f["meta"].create_dataset(k, data=v.astype("f8"))
    return path


def _contract(path):
    with h5py.File(path, "r") as f:
        return json.loads(str(f.attrs["contract"]))


def _summary_paths(out):
    return (str(out) + ".validation_summary.json",
            str(out) + ".validation_summary.md")


# ==========================================================================
# Medium 28. FAR AND posterior-median SNR is not 'far_or_snr'
# ==========================================================================
def test_a_far_and_median_snr_event_cut_is_named_for_what_it_is(tmp_path):
    """select(snr_min) cuts on the posterior-median SNR, ANDed with far_max.

    The writer inferred the statistic from which attrs exist, so far_max plus
    snr_min came out as 'far_or_snr' -- the name of the selection builder's
    OR-widened INJECTION mask, a strictly looser cut.
    """
    store = _store(tmp_path, snr_med=[12.0, 15.0])
    out = tmp_path / "pe.h5"
    product = build_pe_product(GWCatalog(store).select(snr_min=10.0),
                               spin_basis="chieff", nsamp=8, seed=0,
                               far_max=1.0)
    get_exporter("gwcat2.1", kind="pe")(product, str(out))
    contract = _contract(out)
    assert contract["detection_statistic"] == "far_and_snr_med"
    assert contract["detection_threshold"] == 1.0
    assert contract["snr_min"] == 10.0

    out2 = tmp_path / "pe_snr_only.h5"
    product = build_pe_product(GWCatalog(store).select(snr_min=10.0),
                               spin_basis="chieff", nsamp=8, seed=0)
    get_exporter("gwcat2.1", kind="pe")(product, str(out2))
    assert _contract(out2)["detection_statistic"] == "snr_med"


def test_the_validator_refuses_a_posterior_median_snr_event_cut():
    """Statistic, threshold and snr_min agreed on both sides, so the pair
    passed: 25 of 40 injections detected under the OR against 15 under the
    AND the events were selected by, mu biased high, nothing reported."""
    from gwcat.export.validate import _xcheck_detection_cut

    def _fail(name, msg):
        raise ValueError(msg)

    with pytest.raises(ValueError, match="posterior-median SNR"):
        _xcheck_detection_cut(
            _fail, {},
            {"far_max": 1.0, "snr_min": 10.0},
            {"far_threshold": 1.0, "significance_snr_threshold": 10.0,
             "significance_type": "far_or_snr"})
    # A FAR-only event cut against a FAR-only injection cut still passes.
    results = {}
    _xcheck_detection_cut(_fail, results,
                          {"far_max": 1.0, "snr_min": float("nan")},
                          {"far_threshold": 1.0, "significance_type": "far"})


# ==========================================================================
# Medium 43. The distance-prior class reaches the file and the contract
# ==========================================================================
def test_the_pe_contract_states_the_distance_prior_class(tmp_path):
    """A _nocosmo PowerLaw(2) product and a UniformSourceFrame product had the
    same dL_prior_kind (None) and the same contract."""
    pl = _store(tmp_path, name="pl.h5",
                dL_prior_kind=["PowerLaw", "PowerLaw"],
                dL_prior_alpha=[2.0, 2.0],
                dL_prior_basis=["analytic", "analytic"])
    usf = _store(tmp_path, name="usf.h5",
                 dL_prior_kind=["UniformSourceFrame", "UniformSourceFrame"],
                 dL_prior_alpha=[np.nan, np.nan],
                 dL_prior_basis=["analytic", "analytic"])
    contracts = {}
    for label, store in (("pl", pl), ("usf", usf)):
        out = tmp_path / f"pe_{label}.h5"
        product = build_pe_product(GWCatalog(store), spin_basis="chieff",
                                   nsamp=8, seed=0)
        get_exporter("gwcat2.1", kind="pe")(product, str(out))
        contracts[label] = _contract(out)
        with h5py.File(out, "r") as f:
            kinds = [x.decode() if isinstance(x, bytes) else str(x)
                     for x in f.attrs["dL_prior_kind_per_event"]]
            assert len(kinds) == int(f.attrs["nobs"])
            assert len(f.attrs["dL_prior_alpha_per_event"]) == len(kinds)
            assert len(f.attrs["dL_prior_basis_per_event"]) == len(kinds)
    assert contracts["pl"]["dL_prior_kind"] == "PowerLaw(alpha=2)"
    assert contracts["usf"]["dL_prior_kind"] == "UniformSourceFrame"


def test_a_store_without_the_class_states_nothing(tmp_path):
    """No dL_prior_kind column (a pre-GW-02 store): no guessed class."""
    out = tmp_path / "pe.h5"
    product = build_pe_product(GWCatalog(_store(tmp_path)),
                               spin_basis="chieff", nsamp=8, seed=0)
    assert "dL_prior_kind" not in product.attrs
    get_exporter("gwcat2.1", kind="pe")(product, str(out))
    assert _contract(out)["dL_prior_kind"] is None


# ==========================================================================
# Medium 53. A summary always describes the file beside it
# ==========================================================================
@pytest.mark.parametrize("kind", ["pe", "selection"])
def test_a_reexport_without_a_summary_removes_the_stale_one(tmp_path, kind):
    out = tmp_path / f"{kind}.h5"
    if kind == "pe":
        first = build_pe_product(GWCatalog(_store(tmp_path)),
                                 spin_basis="chieff", nsamp=8, seed=0)
        second = build_pe_product(GWCatalog(_store(tmp_path, name="b.h5")),
                                  spin_basis="chieff", nsamp=4, seed=1)
    else:
        first, second = _selection_product(5), _selection_product(3)
    get_exporter("gwcat2", kind=kind)(first, str(out), write_summary=True)
    assert all(os.path.exists(p) for p in _summary_paths(out))
    get_exporter("gwcat2", kind=kind)(second, str(out))   # library default
    assert not any(os.path.exists(p) for p in _summary_paths(out)), \
        "the new file sits beside a summary of the previous export"


@pytest.mark.parametrize("kind", ["pe", "selection"])
def test_the_summary_carries_the_digest_of_its_file(tmp_path, kind):
    out = tmp_path / f"{kind}.h5"
    product = (build_pe_product(GWCatalog(_store(tmp_path)),
                                spin_basis="chieff", nsamp=8, seed=0)
               if kind == "pe" else _selection_product())
    get_exporter("gwcat2", kind=kind)(product, str(out), write_summary=True)
    with open(_summary_paths(out)[0]) as f:
        summary = json.load(f)
    with open(out, "rb") as f:
        assert summary["output_sha256"] == hashlib.sha256(f.read()).hexdigest()


def test_a_failed_summary_write_leaves_the_previous_summary_whole(
        tmp_path, monkeypatch):
    """The summary was opened 'w' in place: a failure mid-write left a
    truncated JSON where a complete one had been."""
    import gwcat.validation_summary as vs

    out = tmp_path / "x.h5"
    out.write_bytes(b"not an export")
    vs.write_validation_summary(str(out), {"kind": "test", "n": 1})
    json_path = _summary_paths(out)[0]
    before = open(json_path).read()

    def _boom(obj, fp, **kw):
        fp.write('{"kind": "te')
        raise RuntimeError("disk full")

    monkeypatch.setattr(vs.json, "dump", _boom)
    with pytest.raises(RuntimeError, match="disk full"):
        vs.write_validation_summary(str(out), {"kind": "test", "n": 2})
    assert open(json_path).read() == before
    assert [p.name for p in tmp_path.iterdir() if "gwcat-tmp" in p.name] == []


# ==========================================================================
# Low 4. The run environment that sets the last bits
# ==========================================================================
@pytest.mark.parametrize("kind", ["pe", "selection"])
def test_the_file_records_its_run_environment(tmp_path, kind):
    out = tmp_path / f"{kind}.h5"
    product = (build_pe_product(GWCatalog(_store(tmp_path)),
                                spin_basis="chieff", nsamp=8, seed=0)
               if kind == "pe" else _selection_product())
    get_exporter("gwcat2", kind=kind)(product, str(out))
    with h5py.File(out, "r") as f:
        env = json.loads(str(f.attrs["writer_environment"]))
    assert env["numpy"] == np.__version__
    assert env["h5py"] == h5py.__version__
    assert env["hdf5"] == h5py.version.hdf5_version
    for key in ("python", "machine", "scipy", "astropy",
                "numpy_cpu_baseline", "numpy_cpu_dispatch_active"):
        assert key in env
    assert isinstance(env["numpy_cpu_dispatch_active"], list)


# ==========================================================================
# Low 19. A per-file symlink is written through, not replaced
# ==========================================================================
def test_the_atomic_write_keeps_a_symlinked_destination(tmp_path):
    from gwcat.export.atomic import atomic_output_path

    real_dir, link_dir = tmp_path / "data", tmp_path / "links"
    real_dir.mkdir()
    link_dir.mkdir()
    target = real_dir / "product.h5"
    target.write_bytes(b"old")
    link = link_dir / "product.h5"
    link.symlink_to(target)

    with atomic_output_path(link) as tmp:
        with open(tmp, "wb") as f:
            f.write(b"new")

    assert link.is_symlink(), "the link was replaced by a regular file"
    assert target.read_bytes() == b"new", "the link target was left stale"
    for d in (real_dir, link_dir):
        assert [p.name for p in d.iterdir() if "gwcat-tmp" in p.name] == []


# ==========================================================================
# Medium 13. A selection product core cannot load says so when written
# ==========================================================================
def _sky_marginal_product():
    p = _selection_product()
    cols = {k: v for k, v in p.columns.items() if k not in ("ra", "dec")}
    return ExportProduct(kind="selection", columns=cols,
                         attrs={**p.attrs, "sky_marginalized": True},
                         spin_basis=p.spin_basis, summary=dict(p.summary))


def _nan_sky_product():
    p = _selection_product()
    return ExportProduct(
        kind="selection", columns=p.columns,
        attrs={**p.attrs, "sky_position_available": np.array([False, True])},
        spin_basis=p.spin_basis, summary=dict(p.summary))


@pytest.mark.parametrize("make", [_sky_marginal_product, _nan_sky_product])
def test_a_selection_core_cannot_load_warns_when_written(tmp_path, make):
    out = tmp_path / "sel.h5"
    with pytest.warns(UserWarning, match="darksirens-core"):
        get_exporter("gwcat2", kind="selection")(make(), str(out))
    assert out.exists()


def test_a_sky_resolved_selection_writes_without_that_warning(tmp_path):
    import warnings

    out = tmp_path / "sel.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        get_exporter("gwcat2", kind="selection")(_selection_product(),
                                                 str(out))
