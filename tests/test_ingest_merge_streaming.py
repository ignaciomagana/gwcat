"""Merging and appending must not hold whole catalogs in memory (GW-24).

The old merge read BOTH stores in full (:func:`gwcat.ingest._read_store`) and
then allocated a concatenated copy of every column, so appending one event to
the 1.7 GB production catalog cost ~6 GiB of RSS; the in-place append read the
existing store once more just to inspect its schema.

These tests pin the streaming replacement:

  * merging reads its inputs' samples only in bounded slices -- never a whole
    column, let alone a whole store;
  * appending inspects the existing store's schema without touching its samples;
  * the streamed output is what the concatenating writer produced -- same rows,
    same values, and NaN (not the HDF5 default 0.0) where a store lacks a column.

Fixtures are tiny synthetic stores written by the real ingest writer, so the
production code path is what runs.
"""
import numpy as np
import h5py
import pytest

from gwcat import ingest
from gwcat.catalog import GWCatalog
from gwcat.ingest import (_assemble_union, _copy_runs, _read_store,
                          _read_store_meta, _write_store,
                          _write_store_from_records, merge_store,
                          merge_stores, IngestConfig,
                          META_FLOAT_FIELDS, META_STR_FIELDS)


_CORE = ["mass_1", "mass_2", "luminosity_distance", "ra", "dec", "chi_eff",
         "p_dL_pe"]
_N = 20            # samples per row


def _store(path, events, seed=0, n=_N):
    """Write a store via the ingest union assembler + writer.

    ``events`` are dicts with ``name``, ``params`` and optional ``sample_set``.
    Every value is finite and distinct, so a wrong slice is visible.
    """
    rng = np.random.default_rng(seed)
    records, names, offsets, sample_sets = [], [], [0], []
    candidates = []
    for ev in events:
        rec = {p: rng.uniform(1.0, 2.0, n) for p in ev["params"]}
        records.append((ev["name"], n, rec))
        names.append(ev["name"])
        offsets.append(offsets[-1] + n)
        sample_sets.append(ev.get("sample_set", ""))
        for p in ev["params"]:
            if p not in candidates:
                candidates.append(p)
    union, columns, avail = _assemble_union(records, candidates)
    meta = {k: [np.nan] * len(names) for k in META_FLOAT_FIELDS}
    meta.update({k: [""] * len(names) for k in META_STR_FIELDS})
    meta["sample_set_name"] = sample_sets
    _write_store(str(path), union, columns, offsets, names, avail, meta,
                 IngestConfig())
    return str(path)


class _SampleReadSpy:
    """Record the size of every read from a ``samples/`` dataset."""

    def __init__(self, monkeypatch):
        self.sizes = []
        self._orig = h5py.Dataset.__getitem__
        self.on = True
        spy = self

        def __getitem__(ds, key):
            out = spy._orig(ds, key)
            if spy.on and ds.name.startswith("/samples/"):
                spy.sizes.append(int(np.size(out)))
            return out

        monkeypatch.setattr(h5py.Dataset, "__getitem__", __getitem__)


# ==========================================================================
# 1. The merge never reads a whole column
# ==========================================================================
def test_merge_reads_samples_only_in_bounded_slices(tmp_path, monkeypatch):
    """Every read from an input store is <= the copy block, so the merge's
    memory does not scale with the catalog.  The old merge read each column
    whole (`f["samples/<p>"][:]`), which is what this refuses."""
    a = _store(tmp_path / "a.h5",
               [{"name": "A1", "params": _CORE + ["psi"]},
                {"name": "A2", "params": _CORE + ["psi"]}], seed=1)
    b = _store(tmp_path / "b.h5",
               [{"name": "B1", "params": _CORE + ["lambda_1"]}], seed=2)

    block = 7
    monkeypatch.setattr(ingest, "_COPY_BLOCK", block)
    spy = _SampleReadSpy(monkeypatch)
    merge_stores(a, b, str(tmp_path / "m.h5"))
    spy.on = False

    assert spy.sizes, "the merge read no samples at all"
    assert max(spy.sizes) <= block, (
        f"a single read of {max(spy.sizes)} samples exceeds the "
        f"{block}-sample copy block")
    # Everything that had to move, moved exactly once: 8 columns x 40 samples
    # from A and 8 columns x 20 from B (each store's own parameters only).
    assert sum(spy.sizes) == 8 * 2 * _N + 8 * _N


def test_merge_skipped_rows_are_never_read(tmp_path, monkeypatch):
    """A duplicate row dropped from the second store costs no sample reads."""
    a = _store(tmp_path / "a.h5", [{"name": "GWdup", "params": _CORE}], seed=1)
    b = _store(tmp_path / "b.h5",
               [{"name": "GWdup", "params": _CORE},      # skipped
                {"name": "GWnew", "params": _CORE}], seed=2)

    spy = _SampleReadSpy(monkeypatch)
    with pytest.warns(UserWarning, match="Duplicate events skipped"):
        merge_stores(a, b, str(tmp_path / "m.h5"))
    spy.on = False
    # 7 columns x (1 kept A row + 1 kept B row), not the 3 rows on disk.
    assert sum(spy.sizes) == 7 * 2 * _N


def test_merge_store_inspects_the_existing_store_without_reading_samples(
        tmp_path, monkeypatch):
    """The append path resolves the existing schema from metadata alone.

    It used to call ``_read_store`` -- a full 2.1 GiB load of the production
    catalog -- purely to learn which columns to ingest for the new events.
    """
    store = _store(tmp_path / "old.h5",
                   [{"name": "GWold", "params": _CORE + ["psi"]}], seed=3)

    def _boom(path):
        raise AssertionError(f"merge_store read all of {path} into memory")

    monkeypatch.setattr(ingest, "_read_store", _boom)
    out = merge_store(store, [], out_path=str(tmp_path / "out.h5"),
                      event_table={})
    cat = GWCatalog(out)
    assert list(cat.event_names) == ["GWold"]
    assert "psi" in cat.params


# ==========================================================================
# 2. The streamed output is the concatenating writer's output
# ==========================================================================
def test_streamed_merge_reproduces_the_rows_exactly(tmp_path):
    a = _store(tmp_path / "a.h5",
               [{"name": "A1", "params": _CORE + ["psi"]},
                {"name": "A2", "params": _CORE + ["psi"]}], seed=4)
    b = _store(tmp_path / "b.h5",
               [{"name": "B1", "params": _CORE + ["lambda_1"]}], seed=5)
    src_a, src_b = _read_store(a), _read_store(b)

    out = str(tmp_path / "m.h5")
    merge_stores(a, b, out)
    got = _read_store(out)

    assert got["names"] == src_a["names"] + src_b["names"]
    np.testing.assert_array_equal(got["offsets"], [0, _N, 2 * _N, 3 * _N])
    for p in _CORE + ["psi"]:
        np.testing.assert_array_equal(got["samples"][p][:2 * _N],
                                      src_a["samples"][p])
    for p in _CORE + ["lambda_1"]:
        np.testing.assert_array_equal(got["samples"][p][2 * _N:],
                                      src_b["samples"][p])


def test_absent_columns_read_back_as_nan_not_zero(tmp_path):
    """The preallocated columns must FILL with NaN: a store missing a parameter
    would otherwise hand every reader a perfectly finite 0.0."""
    a = _store(tmp_path / "a.h5", [{"name": "A1", "params": _CORE}], seed=6)
    b = _store(tmp_path / "b.h5",
               [{"name": "B1", "params": _CORE + ["lambda_1"]}], seed=7)
    out = str(tmp_path / "m.h5")
    merge_stores(a, b, out)

    with h5py.File(out, "r") as f:
        lam = f["samples/lambda_1"][:]
    assert lam.shape == (2 * _N,)
    assert np.all(np.isnan(lam[:_N])), "A's rows must be NaN, not 0.0"
    assert np.all(np.isfinite(lam[_N:]))
    # And the mask agrees with the values.
    cat = GWCatalog(out)
    np.testing.assert_array_equal(cat.param_available("lambda_1"),
                                  np.array([False, True]))


def test_merge_in_place_over_an_input_store(tmp_path):
    """out_path may still BE one of the inputs: the streaming copy reads while
    it writes, so that case goes through a temp file and a rename."""
    a = _store(tmp_path / "a.h5", [{"name": "A1", "params": _CORE}], seed=9)
    b = _store(tmp_path / "b.h5", [{"name": "B1", "params": _CORE}], seed=10)
    src_a, src_b = _read_store(a), _read_store(b)

    merge_stores(a, b, a)                      # overwrite the first store

    got = _read_store(a)
    assert got["names"] == ["A1", "B1"]
    for p in _CORE:
        np.testing.assert_array_equal(got["samples"][p][:_N],
                                      src_a["samples"][p])
        np.testing.assert_array_equal(got["samples"][p][_N:],
                                      src_b["samples"][p])
    assert not list(tmp_path.glob("*.gwcat-merge-tmp"))


# ==========================================================================
# 3. Helper contracts
# ==========================================================================
def test_read_store_meta_agrees_with_the_full_read(tmp_path):
    store = _store(tmp_path / "s.h5",
                   [{"name": "A1", "params": _CORE},
                    {"name": "A2", "params": _CORE + ["psi"]}], seed=8)
    full, meta = _read_store(store), _read_store_meta(store)
    assert "samples" not in meta
    assert meta["params"] == full["params"] and meta["names"] == full["names"]
    assert meta["n_events"] == full["n_events"]
    np.testing.assert_array_equal(meta["offsets"], full["offsets"])
    np.testing.assert_array_equal(meta["avail"], full["avail"])
    assert meta["meta"].keys() == full["meta"].keys()
    assert meta["path"] == store
    assert meta["slices"] == [(0, _N), (_N, 2 * _N)]


def test_copy_runs_coalesces_contiguous_rows():
    # Whole store kept -> one run; a dropped row splits it.
    assert _copy_runs([(0, 10), (10, 25), (25, 30)], 0) == [(0, 30, 0)]
    assert _copy_runs([(0, 10), (25, 30)], 5) == [(0, 10, 5), (25, 30, 15)]
    # Empty rows contribute nothing.
    assert _copy_runs([(0, 0), (0, 4)], 0) == [(0, 4, 0)]
    assert _copy_runs([], 3) == []


# ==========================================================================
# 4. Ingest writes the same store without assembling the union in memory
# ==========================================================================
def _records():
    """Two rows, the second carrying a column the first lacks."""
    rng = np.random.default_rng(11)
    a = {p: rng.uniform(1.0, 2.0, _N) for p in _CORE}
    b = {p: rng.uniform(1.0, 2.0, _N) for p in _CORE + ["lambda_1"]}
    return [("A1", _N, a), ("B1", _N, b)]


def _empty_meta(n):
    meta = {k: [np.nan] * n for k in META_FLOAT_FIELDS}
    meta.update({k: [""] * n for k in META_STR_FIELDS})
    return meta


def test_record_writer_matches_the_concatenating_writer(tmp_path):
    """Streaming the records into preallocated columns writes what
    _assemble_union + _write_store wrote: same union, same values, same mask."""
    records = _records()
    cands = _CORE + ["lambda_1", "never_provided"]
    offsets, names, meta = [0, _N, 2 * _N], ["A1", "B1"], _empty_meta(2)
    cfg = IngestConfig()

    union, columns, avail = _assemble_union(_records(), cands)
    old = str(tmp_path / "concatenated.h5")
    _write_store(old, union, columns, offsets, names, avail, meta, cfg)

    new = str(tmp_path / "streamed.h5")
    got = _write_store_from_records(new, records, cands, offsets, names, meta,
                                    cfg)
    assert got == union            # 'never_provided' dropped by both

    with h5py.File(old, "r") as fa, h5py.File(new, "r") as fb:
        assert dict(fa.attrs).keys() == dict(fb.attrs).keys()
        assert fa.attrs["schema_version"] == fb.attrs["schema_version"]
        seen = []
        fa.visit(seen.append)
        for key in seen:
            if isinstance(fa[key], h5py.Dataset):
                x, y = fa[key][:], fb[key][:]
                assert x.dtype == y.dtype and x.shape == y.shape, key
                assert fa[key].chunks == fb[key].chunks, key
                if x.dtype.kind == "f":
                    assert np.array_equal(x, y, equal_nan=True), key
                else:
                    assert np.array_equal(x, y), key


def test_build_store_never_assembles_the_union_in_memory(tmp_path, monkeypatch):
    """Ingest streams each event's samples into the file.  Holding the finished
    union alongside the per-event records was the whole catalog twice."""
    rng = np.random.default_rng(12)

    def _samples(n, extra=()):
        s = {"mass_1": rng.uniform(25, 50, n), "mass_2": rng.uniform(10, 25, n),
             "luminosity_distance": rng.uniform(300, 800, n),
             "ra": rng.uniform(0, 2 * np.pi, n),
             "dec": rng.uniform(-np.pi / 2, np.pi / 2, n),
             "chi_eff": rng.uniform(-0.4, 0.4, n)}
        s.update({p: rng.uniform(0, 1000, n) for p in extra})
        return s

    per_file = {
        "GWTC-3_GW950101_000101_cosmo.h5": {"C01:Mixed": _samples(_N)},
        "GWTC-3_GW950102_000102_cosmo.h5": {
            "C01:Mixed": _samples(_N, extra=("lambda_1",))},
    }

    class _FakeData:
        pass

    def _fake_read(path):
        analyses = per_file[str(path).rsplit("/", 1)[-1]]
        return _FakeData(), analyses, list(analyses), {}

    monkeypatch.setattr(ingest, "_read_event_pesummary", _fake_read)

    def _boom(*a, **k):
        raise AssertionError("build_store concatenated the union in memory")

    monkeypatch.setattr(ingest, "_assemble_union", _boom)

    paths = []
    for base in per_file:
        p = tmp_path / base
        p.write_bytes(b"")
        paths.append(str(p))
    out = str(tmp_path / "store.h5")
    ingest.build_store(paths, out, params=_CORE[:-1] + ["lambda_1"],
                       event_table={}, cfg=IngestConfig(validate_prior=False))

    cat = GWCatalog(out)
    assert cat.n_events == 2
    assert "lambda_1" in cat.params
    np.testing.assert_array_equal(cat.param_available("lambda_1"),
                                  np.array([False, True]))
    lam = cat.get(["lambda_1"], per_event=True)["lambda_1"]
    assert np.all(np.isnan(lam[0])), "the row without the column must be NaN"
    assert np.all(np.isfinite(lam[1]))
    m1 = cat.get(["mass_1"], per_event=True)["mass_1"]
    np.testing.assert_array_equal(
        m1[1], per_file["GWTC-3_GW950102_000102_cosmo.h5"]["C01:Mixed"]["mass_1"])
