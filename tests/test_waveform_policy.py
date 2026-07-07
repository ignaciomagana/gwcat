"""Waveform / sample-set policy tests (PR 6).

Covers the sample-set contract from the handoff:

  * a store may hold MULTIPLE sample sets per event -- each (event, sample_set)
    pair is one row -- and ``waveform_policy="all"`` keeps them all (case 5);
  * ``build_store(..., sample_sets="all")`` ingests every analysis of a PE file
    as a separate sample-set row (ingest side of case 5);
  * ``mixed-first`` picks the ``is_mixed`` set when present;
  * ``strict-approximant`` fails loudly -- naming the events -- when one lacks
    the requested approximant (case 4), and succeeds when all have it;
  * ``preferred`` picks by ``is_preferred`` / ``priority_rank``;
  * a legacy store (no sample-set columns) loads and exports unchanged --
    byte-identical across policies with a fixed seed;
  * ``to_darksirens`` records the policy and the per-event chosen sample sets in
    the output attrs (``homogeneous_sample_sets`` false only for a real mix).

Fixtures are tiny synthetic HDF5 stores.  The multi-sample-set stores are built
through the real ingest union assembler + writer (schema 1.2), and one test
drives the real ``build_store`` ingest path with a fake PESummary reader, so the
PR-6 code -- not a re-implementation -- is under test.
"""
import numpy as np
import h5py
import pytest

from gwcat.catalog import GWCatalog
from gwcat.ingest import (_assemble_union, _write_store, IngestConfig,
                          build_store, select_analyses, rank_analyses,
                          _waveform_family, META_FLOAT_FIELDS, META_STR_FIELDS)
from gwcat.waveform_policy import resolve_policy, WAVEFORM_POLICIES


_COSMO = (67.74, 0.3089)
_CORE = ["mass_1", "mass_2", "luminosity_distance", "ra", "dec", "chi_eff",
         "p_dL_pe"]


def _rand_event(rng, n):
    return {
        "mass_1": rng.uniform(25, 50, n),
        "mass_2": rng.uniform(10, 25, n),
        "luminosity_distance": rng.uniform(300, 800, n),
        "ra": rng.uniform(0, 2 * np.pi, n),
        "dec": rng.uniform(-np.pi / 2, np.pi / 2, n),
        "chi_eff": rng.uniform(-0.4, 0.4, n),
        "p_dL_pe": rng.uniform(0.1, 1.0, n),
    }


def build_multiset_store(path, rows, H0=67.74, Om0=0.3089, n=12, seed=0):
    """Write a real schema-1.2 store from explicit sample-set rows.

    rows : list of dicts, each a (event, sample_set) ROW with keys
        name, sample_set_name, approximant, waveform (optional; derived if
        absent), is_mixed (bool), is_preferred (bool), priority_rank (float).
    """
    rng = np.random.default_rng(seed)
    records, names, offsets = [], [], [0]
    meta = {k: [np.nan] * len(rows) for k in META_FLOAT_FIELDS}
    meta.update({k: [""] * len(rows) for k in META_STR_FIELDS})
    meta["dL_prior_H0"] = [H0] * len(rows)
    meta["dL_prior_Om0"] = [Om0] * len(rows)

    for i, r in enumerate(rows):
        records.append((r["name"], n, _rand_event(rng, n)))
        names.append(r["name"])
        offsets.append(offsets[-1] + n)
        approx = r.get("approximant", "")
        meta["name"][i] = r["name"]
        meta["sample_set_name"][i] = r.get("sample_set_name", approx)
        meta["approximant"][i] = approx
        meta["waveform"][i] = r.get("waveform", _waveform_family(approx))
        meta["is_mixed"][i] = 1.0 if r.get("is_mixed") else 0.0
        meta["is_preferred"][i] = 1.0 if r.get("is_preferred") else 0.0
        meta["priority_rank"][i] = float(r.get("priority_rank", np.nan))
        meta["selection_reason"][i] = r.get("selection_reason", "")
        meta["source_class"][i] = r.get("source_class", "BBH")
        meta["compact_type"][i] = r.get("source_class", "BBH")

    union, columns, avail = _assemble_union(records, list(_CORE))
    _write_store(str(path), union, columns, offsets, names, avail, meta,
                 IngestConfig())
    return str(path)


# One event with three sample sets: IMRPhenomXPHM, SEOBNRv5PHM, and a Mixed set.
def _one_event_three_sets(name="GW850101_010101"):
    return [
        {"name": name, "sample_set_name": "C00:IMRPhenomXPHM",
         "approximant": "IMRPhenomXPHM", "is_preferred": False,
         "priority_rank": 2},
        {"name": name, "sample_set_name": "C00:SEOBNRv5PHM",
         "approximant": "SEOBNRv5PHM", "is_preferred": False,
         "priority_rank": 1},
        {"name": name, "sample_set_name": "C00:Mixed", "approximant": "Mixed",
         "is_mixed": True, "is_preferred": True, "priority_rank": 0},
    ]


# ==========================================================================
# 0. resolver unit sanity + declared policy set
# ==========================================================================
def test_policy_names_declared():
    assert WAVEFORM_POLICIES == ("preferred", "mixed-first",
                                 "strict-approximant", "all")


def test_resolve_policy_rejects_unknown():
    with pytest.raises(ValueError, match="invalid"):
        resolve_policy(np.array(["A"]), np.array([0]), {}, policy="bogus")


# ==========================================================================
# 1. waveform_policy="all" keeps multiple sample sets for one event (case 5)
# ==========================================================================
def test_all_keeps_multiple_sample_sets(tmp_path):
    store = build_multiset_store(tmp_path / "multi.h5", _one_event_three_sets(),
                                 seed=1)
    cat = GWCatalog(store)
    assert cat.n_events == 3  # three rows, all one event

    all_sets = cat.select(waveform_policy="all")
    assert all_sets.n_events == 3
    assert set(all_sets.event_names) == {"GW850101_010101"}
    assert not all_sets._homogeneous_sample_sets

    # Every other policy collapses to exactly one sample set for the event.
    for pol in ("preferred", "mixed-first"):
        one = cat.select(waveform_policy=pol)
        assert one.n_events == 1
        assert one._homogeneous_sample_sets


def test_all_export_marks_non_homogeneous(tmp_path):
    store = build_multiset_store(tmp_path / "multi.h5", _one_event_three_sets(),
                                 seed=2)
    cat = GWCatalog(store)
    out = tmp_path / "all.h5"
    cat.to_darksirens(str(out), waveform_policy="all", nsamp=8, seed=0,
                      cosmology=_COSMO)
    with h5py.File(out, "r") as f:
        assert f.attrs["waveform_policy"] == "all"
        assert bool(f.attrs["homogeneous_sample_sets"]) is False
        assert f.attrs["nobs"] == 3          # three sample-set rows written
        # the three chosen sets are the three approximants of the one event
        assert set(f.attrs["sample_set_approximant_per_event"]) == {
            "IMRPhenomXPHM", "SEOBNRv5PHM", "Mixed"}
        assert set(f.attrs["event_names"]) == {"GW850101_010101"}


# ==========================================================================
# 2. build_store(sample_sets="all") ingests every analysis as a row (case 5)
# ==========================================================================
class _FakeData:
    """Minimal stand-in for a pesummary read() result (no config → f_ref NaN)."""


def _fake_reader_factory(analyses_params):
    """Return a fake _read_event_pesummary returning the given analyses."""
    def _fake(path):
        return _FakeData(), analyses_params, list(analyses_params.keys()), {}
    return _fake


def test_build_store_all_ingests_every_analysis(tmp_path, monkeypatch):
    import gwcat.ingest as ing
    rng = np.random.default_rng(0)
    n = 20

    def _mk():
        return {"luminosity_distance": rng.uniform(300, 800, n),
                "mass_1": rng.uniform(25, 50, n),
                "mass_2": rng.uniform(10, 25, n),
                "ra": rng.uniform(0, 2 * np.pi, n),
                "dec": rng.uniform(-np.pi / 2, np.pi / 2, n),
                "chi_eff": rng.uniform(-0.4, 0.4, n)}

    analyses = {"C00:IMRPhenomXPHM": _mk(), "C00:SEOBNRv5PHM": _mk(),
                "C00:Mixed": _mk()}
    monkeypatch.setattr(ing, "_read_event_pesummary",
                        _fake_reader_factory(analyses))

    path = tmp_path / "GWTC-5_GW850101_010101_cosmo.h5"
    path.write_bytes(b"")  # existence only; the reader is faked
    out = tmp_path / "store_all.h5"
    build_store([str(path)], str(out), sample_sets="all", event_table={},
                cfg=IngestConfig(validate_prior=False))

    cat = GWCatalog(str(out))
    assert cat.n_events == 3                       # one event, three sample sets
    assert set(cat.event_names) == {"GW850101_010101"}
    assert set(cat.meta["approximant"]) == {"IMRPhenomXPHM", "SEOBNRv5PHM",
                                            "Mixed"}
    # exactly the Mixed row is preferred; schema advertises sample-set columns
    assert int(np.sum(np.asarray(cat.meta["is_preferred"]) > 0.5)) == 1
    with h5py.File(str(out), "r") as f:
        assert f.attrs["schema_version"] == "1.2"

    # default preferred ingest keeps exactly one row (the Mixed set).
    out1 = tmp_path / "store_pref.h5"
    build_store([str(path)], str(out1), event_table={},
                cfg=IngestConfig(validate_prior=False))  # sample_sets default
    cat1 = GWCatalog(str(out1))
    assert cat1.n_events == 1
    assert list(cat1.meta["sample_set_name"]) == ["C00:Mixed"]


def test_select_analyses_helpers():
    analyses = ["C00:Mixed", "C00:SEOBNRv5PHM", "C00:IMRPhenomXPHM"]
    cfg = IngestConfig()
    # preferred -> just the Mixed set (single-label heuristic)
    assert select_analyses(analyses, "C00", cfg, "preferred") == ["C00:Mixed"]
    # all -> ranked, Mixed first then priority order
    ranked = rank_analyses(analyses, "C00", cfg)
    assert ranked[0] == "C00:Mixed"
    assert set(select_analyses(analyses, "C00", cfg, "all")) == set(analyses)
    # explicit list validated
    assert select_analyses(analyses, "C00", cfg, ["C00:SEOBNRv5PHM"]) == [
        "C00:SEOBNRv5PHM"]
    with pytest.raises(ValueError, match="not present"):
        select_analyses(analyses, "C00", cfg, ["C00:DoesNotExist"])


# ==========================================================================
# 3. mixed-first picks the is_mixed set
# ==========================================================================
def test_mixed_first_picks_mixed(tmp_path):
    store = build_multiset_store(tmp_path / "multi.h5", _one_event_three_sets(),
                                 seed=3)
    cat = GWCatalog(store)
    sub = cat.select(waveform_policy="mixed-first")
    assert sub.n_events == 1
    assert list(sub.meta["sample_set_name"][sub._sel]) == ["C00:Mixed"]
    assert list(sub._selection_reasons) == ["mixed-first:is_mixed"]


def test_mixed_first_falls_back_to_preferred_when_no_mixed(tmp_path):
    # two non-mixed sets; the is_preferred one must win via the fallback.
    rows = [
        {"name": "GW860101_010101", "sample_set_name": "C00:IMRPhenomXPHM",
         "approximant": "IMRPhenomXPHM", "is_preferred": True,
         "priority_rank": 0},
        {"name": "GW860101_010101", "sample_set_name": "C00:SEOBNRv5PHM",
         "approximant": "SEOBNRv5PHM", "is_preferred": False,
         "priority_rank": 1},
    ]
    cat = GWCatalog(build_multiset_store(tmp_path / "nomix.h5", rows, seed=4))
    sub = cat.select(waveform_policy="mixed-first")
    assert sub.n_events == 1
    assert list(sub.meta["approximant"][sub._sel]) == ["IMRPhenomXPHM"]
    assert sub._selection_reasons[0].startswith("mixed-first:fallback_preferred")


# ==========================================================================
# 4. strict-approximant: fails loud when one event lacks it, else succeeds
# ==========================================================================
def _two_events_one_missing(approx_present="SEOBNRv5PHM"):
    """Event A has SEOBNRv5PHM + IMRPhenomXPHM; event B has only IMRPhenomXPHM."""
    return [
        {"name": "GW870001_000001", "sample_set_name": "C00:SEOBNRv5PHM",
         "approximant": "SEOBNRv5PHM", "priority_rank": 1, "is_preferred": True},
        {"name": "GW870001_000001", "sample_set_name": "C00:IMRPhenomXPHM",
         "approximant": "IMRPhenomXPHM", "priority_rank": 2},
        {"name": "GW870002_000002", "sample_set_name": "C00:IMRPhenomXPHM",
         "approximant": "IMRPhenomXPHM", "priority_rank": 0, "is_preferred": True},
    ]


def test_strict_approximant_fails_loud_when_missing(tmp_path):
    cat = GWCatalog(build_multiset_store(tmp_path / "strict.h5",
                                         _two_events_one_missing(), seed=5))
    out = tmp_path / "wont_exist.h5"
    with pytest.raises(ValueError) as ei:
        cat.to_darksirens(str(out), waveform_policy="strict-approximant",
                          approximant="SEOBNRv5PHM", nsamp=8, seed=0,
                          cosmology=_COSMO)
    msg = str(ei.value)
    assert "SEOBNRv5PHM" in msg and "GW870002_000002" in msg
    assert not out.exists()          # failed loudly, wrote nothing


def test_strict_approximant_succeeds_when_all_present(tmp_path):
    # both events carry IMRPhenomXPHM -> strict-approximant is satisfiable.
    cat = GWCatalog(build_multiset_store(tmp_path / "strict.h5",
                                         _two_events_one_missing(), seed=6))
    out = tmp_path / "strict_ok.h5"
    cat.to_darksirens(str(out), waveform_policy="strict-approximant",
                      approximant="IMRPhenomXPHM", nsamp=8, seed=0,
                      cosmology=_COSMO)
    with h5py.File(out, "r") as f:
        assert f.attrs["nobs"] == 2                # one set per event
        assert bool(f.attrs["homogeneous_sample_sets"]) is True
        assert f.attrs["approximant"] == "IMRPhenomXPHM"
        assert list(f.attrs["sample_set_approximant_per_event"]) == [
            "IMRPhenomXPHM", "IMRPhenomXPHM"]


def test_strict_approximant_requires_approximant_arg(tmp_path):
    cat = GWCatalog(build_multiset_store(tmp_path / "strict.h5",
                                         _one_event_three_sets(), seed=7))
    with pytest.raises(ValueError, match="requires approximant"):
        cat.select(waveform_policy="strict-approximant")


def test_strict_approximant_matches_waveform_family(tmp_path):
    """A request for the bare family matches a suffixed approximant token."""
    rows = [
        {"name": "GW880001_000001", "sample_set_name": "C00:IMRPhenomXPHM-ST",
         "approximant": "IMRPhenomXPHM-SpinTaylor",
         "waveform": _waveform_family("IMRPhenomXPHM-SpinTaylor"),
         "is_preferred": True, "priority_rank": 0},
    ]
    cat = GWCatalog(build_multiset_store(tmp_path / "fam.h5", rows, seed=8))
    sub = cat.select(waveform_policy="strict-approximant",
                     approximant="IMRPhenomXPHM")
    assert sub.n_events == 1


# ==========================================================================
# 5. preferred picks by is_preferred / priority_rank
# ==========================================================================
def test_preferred_picks_is_preferred(tmp_path):
    cat = GWCatalog(build_multiset_store(tmp_path / "multi.h5",
                                         _one_event_three_sets(), seed=9))
    sub = cat.select(waveform_policy="preferred")
    assert sub.n_events == 1
    assert list(sub.meta["sample_set_name"][sub._sel]) == ["C00:Mixed"]
    assert sub._selection_reasons[0] == "preferred:is_preferred"


def test_preferred_falls_back_to_priority_rank(tmp_path):
    # no is_preferred flag set -> pick the smallest priority_rank.
    rows = [
        {"name": "GW890001_000001", "sample_set_name": "C00:SEOBNRv5PHM",
         "approximant": "SEOBNRv5PHM", "priority_rank": 1},
        {"name": "GW890001_000001", "sample_set_name": "C00:Mixed",
         "approximant": "Mixed", "is_mixed": True, "priority_rank": 0},
    ]
    cat = GWCatalog(build_multiset_store(tmp_path / "rank.h5", rows, seed=10))
    sub = cat.select(waveform_policy="preferred")
    assert sub.n_events == 1
    assert list(sub.meta["sample_set_name"][sub._sel]) == ["C00:Mixed"]
    assert sub._selection_reasons[0] == "preferred:min_priority_rank"


# ==========================================================================
# 6. legacy store (no sample-set columns) loads + exports unchanged
# ==========================================================================
def _build_legacy_store(tmp_path, n_events=2, n=15, seed=0, H0=67.74,
                        Om0=0.3089, name="legacy.h5"):
    """A pre-PR6 store: only darksirens columns + cosmology meta, no sample-set
    columns and no schema_version attr (single sample set per event)."""
    rng = np.random.default_rng(seed)
    names = [f"GW990000_{i:06d}" for i in range(n_events)]
    cols = {p: [] for p in _CORE}
    offsets = [0]
    for _ in range(n_events):
        ev = _rand_event(rng, n)
        for p in _CORE:
            cols[p].append(ev[p])
        offsets.append(offsets[-1] + n)
    path = tmp_path / name
    with h5py.File(path, "w") as f:
        f.attrs["param_names"] = np.array(_CORE, dtype=h5py.string_dtype())
        idx = f.create_group("index")
        idx.create_dataset("offsets", data=np.array(offsets, dtype="i8"))
        idx.create_dataset("event_names",
                           data=np.array(names, dtype=h5py.string_dtype()))
        mg = f.create_group("meta")
        mg.create_dataset("dL_prior_H0", data=np.full(n_events, H0))
        mg.create_dataset("dL_prior_Om0", data=np.full(n_events, Om0))
        sg = f.create_group("samples")
        for p in _CORE:
            sg.create_dataset(p, data=np.concatenate(cols[p]))
    return str(path)


def test_legacy_store_loads_and_is_no_op(tmp_path):
    store = _build_legacy_store(tmp_path, seed=11)
    with h5py.File(store, "r") as f:
        assert "schema_version" not in f.attrs
        assert "sample_set_name" not in f["meta"]
    cat = GWCatalog(store)
    # every policy keeps all (one-row-per-event) events, homogeneous.
    for pol in ("preferred", "mixed-first", "all"):
        sub = cat.select(waveform_policy=pol)
        assert sub.n_events == 2
        assert sub._homogeneous_sample_sets


def test_legacy_export_byte_identical_across_policies(tmp_path):
    """The sample-set layer is inert for a single-sample-set store: exporting
    with the default vs waveform_policy='all' is byte-identical (fixed seed)."""
    store = _build_legacy_store(tmp_path, seed=12)
    cat = GWCatalog(store)
    kw = dict(nsamp=10, seed=0, cosmology=_COSMO)

    out_default = tmp_path / "default.h5"
    out_all = tmp_path / "all.h5"
    cat.to_darksirens(str(out_default), **kw)                       # preferred
    cat.to_darksirens(str(out_all), waveform_policy="all", **kw)

    with h5py.File(out_default, "r") as fd, h5py.File(out_all, "r") as fa:
        assert fd.attrs["nobs"] == fa.attrs["nobs"] == 2
        for key in ["ra", "dec", "m1det", "m2det", "chieff", "dL", "p_pe",
                    "redshift", "m1src", "m2src"]:
            np.testing.assert_array_equal(fd[key][:], fa[key][:])
        # legacy events carry no stored sample-set names -> empty strings
        assert list(fd.attrs["sample_set_name_per_event"]) == ["", ""]
        assert bool(fd.attrs["homogeneous_sample_sets"]) is True


# ==========================================================================
# 7. export attrs record policy + chosen sets (preferred)
# ==========================================================================
def test_export_attrs_record_preferred_choice(tmp_path):
    cat = GWCatalog(build_multiset_store(tmp_path / "multi.h5",
                                         _one_event_three_sets(), seed=13))
    out = tmp_path / "pref.h5"
    cat.to_darksirens(str(out), waveform_policy="preferred", nsamp=8, seed=0,
                      cosmology=_COSMO)
    with h5py.File(out, "r") as f:
        assert f.attrs["waveform_policy"] == "preferred"
        assert f.attrs["approximant"] == ""
        assert f.attrs["nobs"] == 1
        assert bool(f.attrs["homogeneous_sample_sets"]) is True
        assert list(f.attrs["sample_set_name_per_event"]) == ["C00:Mixed"]
        assert list(f.attrs["sample_set_approximant_per_event"]) == ["Mixed"]
        assert list(f.attrs["sample_set_selection_reason"]) == [
            "preferred:is_preferred"]
