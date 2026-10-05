"""Mock-data provenance: the ``mock_data`` export attr comes from the input.

``build_pe_product`` used to stamp ``mock_data=False`` unconditionally and the
selection builder never set it, so a synthetic campaign run through gwcat's own
builders was labelled real data (darksirens announces mock data from that
attr).  The flag now travels with the data:

  * a store written with ``mock_data=True`` (``build_store`` /
    ``_write_store_from_records``) carries a file-level ``mock_data`` attr,
    which ``GWCatalog`` reads and every PE exporter stamps;
  * an injection file whose root attrs carry ``mock_data=True`` makes the
    selection export carry it;
  * real-data inputs write NO new attr anywhere, so their stores and exports
    are byte-identical to before (the default stays False).
"""
import json
import warnings

import numpy as np
import h5py
import pytest

from gwcat import ingest
from gwcat.catalog import GWCatalog, validate_export as validate_export_v1
from gwcat.selection import SelectionSet, CombinedSelectionSet
from gwcat.export import (build_pe_product, build_selection_product,
                          validate_export_v2)
from gwcat.export.product import resolve_mock_data

from test_export_v2_pe import _build_spin_store
from test_selection_spin import write_o4_full

_COSMO = (67.74, 0.3089)
_EVENTS = [{"name": "GWv0_000001", "amax1": 0.99, "amax2": 0.99},
           {"name": "GWv0_000002", "amax1": 0.90, "amax2": 0.90}]


def _rewrite_store(src, dst, *, mock_data):
    """Re-write ``src`` through gwcat's own record writer (the path the mock
    bridges use), flagged or not."""
    S = ingest._read_store(str(src))
    off = S["offsets"]
    records = [(n, int(off[i + 1] - off[i]),
                {p: S["samples"][p][off[i]:off[i + 1]] for p in S["params"]})
               for i, n in enumerate(S["names"])]
    n = len(S["names"])
    meta = {k: list(S["meta"].get(k, [np.nan] * n))
            for k in ingest.META_FLOAT_FIELDS}
    meta.update({k: list(S["meta"].get(k, [""] * n))
                 for k in ingest.META_STR_FIELDS})
    kw = {} if mock_data is None else {"mock_data": mock_data}
    ingest._write_store_from_records(str(dst), records, S["params"],
                                     list(off), list(S["names"]), meta,
                                     ingest.IngestConfig(), **kw)
    return dst


def _stores(tmp_path):
    raw, _ = _build_spin_store(tmp_path, _EVENTS, name="raw_store.h5")
    real = _rewrite_store(raw, tmp_path / "real_store.h5", mock_data=None)
    mock = _rewrite_store(raw, tmp_path / "mock_store.h5", mock_data=True)
    return real, mock


def _injections(tmp_path, name, *, mock):
    inj = write_o4_full(tmp_path / name, n=60, amax=(0.9, 0.9), seed=6)
    if mock:
        with h5py.File(inj, "r+") as f:
            f.attrs["mock_data"] = True
    return inj


def _same(x, y):
    x, y = np.asarray(x), np.asarray(y)
    try:
        return np.array_equal(x, y, equal_nan=True)
    except TypeError:                     # strings / objects: NaN-free
        return np.array_equal(x, y)


def _attrs(path):
    with h5py.File(path, "r") as f:
        return dict(f.attrs)


# ======================================================================
# Store: written, read, merged
# ======================================================================
def test_store_flag_written_only_when_mock(tmp_path):
    real, mock = _stores(tmp_path)
    assert "mock_data" not in _attrs(real)        # real store unchanged
    assert _attrs(mock)["mock_data"] is np.True_ or _attrs(mock)["mock_data"]
    assert GWCatalog(str(real)).mock_data is False
    assert GWCatalog(str(mock)).mock_data is True
    # Views re-read the store, so the flag survives select().
    assert GWCatalog(str(mock)).select(source_class="bbh").mock_data is True


def test_real_store_bytes_unchanged_by_default(tmp_path):
    """mock_data=False and the omitted argument write the identical file."""
    raw, _ = _build_spin_store(tmp_path, _EVENTS, name="raw_store.h5")
    a = _rewrite_store(raw, tmp_path / "a.h5", mock_data=None)
    b = _rewrite_store(raw, tmp_path / "b.h5", mock_data=False)
    assert a.read_bytes() == b.read_bytes()


def test_merge_of_mock_and_real_is_mock_and_warns(tmp_path):
    real, mock = _stores(tmp_path)
    # Distinct rows so nothing is skipped as a duplicate.
    raw2, _ = _build_spin_store(
        tmp_path, [{"name": "GWv0_000003", "amax1": 0.99, "amax2": 0.99}],
        name="raw2.h5", seed=3)
    real2 = _rewrite_store(raw2, tmp_path / "real2.h5", mock_data=None)

    out = tmp_path / "merged_real.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ingest.merge_stores(str(real), str(real2), str(out))
    assert "mock_data" not in _attrs(out)

    out = tmp_path / "merged_mixed.h5"
    with pytest.warns(UserWarning, match="mock-data store with a real-data"):
        ingest.merge_stores(str(mock), str(real2), str(out))
    assert GWCatalog(str(out)).mock_data is True


# ======================================================================
# PE exports
# ======================================================================
@pytest.mark.parametrize("fmt", ["gwcat2", "gwcat2.1"])
def test_pe_export_carries_the_store_flag(tmp_path, fmt):
    real, mock = _stores(tmp_path)
    for store, expect in ((real, False), (mock, True)):
        out = tmp_path / f"pe_{expect}_{fmt}.h5"
        GWCatalog(str(store)).export(str(out), format=fmt,
                                     spin_basis="component", nsamp=48, seed=0,
                                     cosmology=_COSMO)
        assert bool(_attrs(out)["mock_data"]) is expect
        results = validate_export_v2(str(out))
        assert results["pe_mock_data_attr"] is True
        assert all(results.values()), \
            [k for k, v in results.items() if not v]


def test_pe_mock_and_real_exports_differ_only_in_the_flag(tmp_path):
    """Same samples, same numbers, same contract hash: only the label moves."""
    real, mock = _stores(tmp_path)
    files = {}
    for tag, store in (("real", real), ("mock", mock)):
        out = tmp_path / f"pe_{tag}.h5"
        GWCatalog(str(store)).export(str(out), format="gwcat2.1",
                                     spin_basis="component", nsamp=48, seed=0,
                                     cosmology=_COSMO)
        files[tag] = out
    with h5py.File(files["real"], "r") as a, h5py.File(files["mock"], "r") as b:
        assert set(a.keys()) == set(b.keys())
        for k in a.keys():
            np.testing.assert_array_equal(a[k][()], b[k][()])
        assert a.attrs["contract_hash"] == b.attrs["contract_hash"]
        diff = {k for k in set(a.attrs) | set(b.attrs)
                if k not in ("writer_commit",)
                and not _same(a.attrs.get(k), b.attrs.get(k))}
        assert diff == {"mock_data"}, diff


def test_v1_pe_export_carries_the_store_flag(tmp_path):
    real, mock = _stores(tmp_path)
    for store, expect in ((real, False), (mock, True)):
        out = tmp_path / f"pe_v1_{expect}.h5"
        GWCatalog(str(store)).to_darksirens(str(out), nsamp=48, seed=0,
                                            cosmology=_COSMO)
        assert bool(_attrs(out)["mock_data"]) is expect


def test_explicit_argument_adds_but_never_removes_the_label(tmp_path):
    real, mock = _stores(tmp_path)
    kw = dict(spin_basis="component", nsamp=16, seed=0, cosmology=_COSMO)
    assert build_pe_product(GWCatalog(str(real)), **kw).attrs["mock_data"] \
        is False
    assert build_pe_product(GWCatalog(str(real)), mock_data=True,
                            **kw).attrs["mock_data"] is True
    assert build_pe_product(GWCatalog(str(mock)), mock_data=True,
                            **kw).attrs["mock_data"] is True
    with pytest.raises(ValueError, match="cannot be exported as real"):
        build_pe_product(GWCatalog(str(mock)), mock_data=False, **kw)


def test_resolve_mock_data_rules():
    assert resolve_mock_data(False, None, source="x") is False
    assert resolve_mock_data(True, None, source="x") is True
    assert resolve_mock_data(False, True, source="x") is True
    assert resolve_mock_data(False, False, source="x") is False
    with pytest.raises(ValueError):
        resolve_mock_data(True, False, source="x")
    with pytest.raises(TypeError):
        resolve_mock_data(False, "yes", source="x")


# ======================================================================
# Selection exports
# ======================================================================
@pytest.mark.parametrize("fmt", ["gwcat2", "gwcat2.1"])
def test_selection_export_carries_the_injection_flag(tmp_path, fmt):
    real = _injections(tmp_path, "inj_real.hdf", mock=False)
    mock = _injections(tmp_path, "inj_mock.hdf", mock=True)
    assert SelectionSet(str(real)).mock_data is False
    assert SelectionSet(str(mock)).mock_data is True

    out_r, out_m = tmp_path / f"sel_real_{fmt}.h5", tmp_path / f"sel_mock_{fmt}.h5"
    SelectionSet(str(real)).export(str(out_r), format=fmt,
                                   spin_basis="component")
    SelectionSet(str(mock)).export(str(out_m), format=fmt,
                                   spin_basis="component")
    assert "mock_data" not in _attrs(out_r)       # real export unchanged
    assert bool(_attrs(out_m)["mock_data"]) is True


def test_selection_combined_and_explicit(tmp_path):
    real = _injections(tmp_path, "inj_real.hdf", mock=False)
    mock = _injections(tmp_path, "inj_mock.hdf", mock=True)
    assert CombinedSelectionSet([SelectionSet(str(real)),
                                 SelectionSet(str(mock))]).mock_data is True
    p = build_selection_product(SelectionSet(str(real)),
                                spin_basis="component", mock_data=True)
    assert p.attrs["mock_data"] is True
    p = build_selection_product(SelectionSet(str(real)),
                                spin_basis="component")
    assert "mock_data" not in p.attrs
    with pytest.raises(ValueError, match="cannot be exported as real"):
        build_selection_product(SelectionSet(str(mock)),
                                spin_basis="component", mock_data=False)


def test_v1_selection_export_carries_the_injection_flag(tmp_path):
    real = _injections(tmp_path, "inj_real.hdf", mock=False)
    mock = _injections(tmp_path, "inj_mock.hdf", mock=True)
    SelectionSet(str(real)).to_darksirens(str(tmp_path / "s_r.h5"))
    SelectionSet(str(mock)).to_darksirens(str(tmp_path / "s_m.h5"))
    assert "mock_data" not in _attrs(tmp_path / "s_r.h5")
    assert bool(_attrs(tmp_path / "s_m.h5")["mock_data"]) is True


# ======================================================================
# Validation of a pair
# ======================================================================
def _pair(tmp_path, *, pe_mock, sel_mock, fmt="gwcat2.1"):
    real, mock = _stores(tmp_path)
    pe = tmp_path / f"pe_{pe_mock}{sel_mock}.h5"
    GWCatalog(str(mock if pe_mock else real)).export(
        str(pe), format=fmt, spin_basis="component", nsamp=48, seed=0,
        cosmology=_COSMO)
    inj = _injections(tmp_path, f"inj_{pe_mock}{sel_mock}.hdf", mock=sel_mock)
    sel = tmp_path / f"sel_{pe_mock}{sel_mock}.h5"
    SelectionSet(str(inj)).export(str(sel), format=fmt, spin_basis="component")
    return pe, sel


def test_mock_pair_validates_clean(tmp_path):
    pe, sel = _pair(tmp_path, pe_mock=True, sel_mock=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*mock_data.*")
        results = validate_export_v2(str(pe), str(sel))
    assert all(results.values()), [k for k, v in results.items() if not v]
    assert results["pe_mock_data_attr"] is True
    assert results["sel_mock_data_attr"] is True
    assert results["xcheck_mock_data"] is True
    assert results["xcheck_contract_hash"] is True


def test_mixed_pair_warns_not_fails(tmp_path):
    pe, sel = _pair(tmp_path, pe_mock=True, sel_mock=False)
    with pytest.warns(UserWarning, match="one side of this pair is synthetic"):
        results = validate_export_v2(str(pe), str(sel))
    assert all(results.values()), [k for k, v in results.items() if not v]


def test_non_boolean_flag_fails_validation(tmp_path):
    pe, _ = _pair(tmp_path, pe_mock=False, sel_mock=False)
    with h5py.File(pe, "r+") as f:
        f.attrs["mock_data"] = "yes"
    assert validate_export_v2(str(pe))["pe_mock_data_attr"] is False


def test_v1_validator_accepts_a_mock_pe(tmp_path):
    _, mock = _stores(tmp_path)
    out = tmp_path / "pe_v1_mock.h5"
    GWCatalog(str(mock)).to_darksirens(str(out), nsamp=48, seed=0,
                                       cosmology=_COSMO)
    results = validate_export_v1(str(out))
    assert results["pe_mock_data_attr"] is True
