"""Schema-preserving ingest & merge tests (PR 5).

Covers the parameter-schema contract from the handoff:

  * ingest/merge store the UNION of parameters across events, never the
    intersection -- a parameter present in only some events is kept as a full
    column, NaN-filled for the events that lack it, with a per-event x
    per-parameter availability mask (handoff cases 6 and 12);
  * a requested export fails loudly -- naming the missing parameter AND the
    offending event(s) -- when a required column is absent from the store or
    unavailable (NaN-filled) for a selected event (case 7);
  * ``GWCatalog.get`` distinguishes required (clear named error, not a bare
    KeyError) from optional (NaN-filled) access;
  * legacy stores that predate the availability mask still load and behave
    (mask derived as all-True).

Fixtures are tiny synthetic HDF5 stores.  The union assembly and store writer
from :mod:`gwcat.ingest` are exercised directly (no pesummary / network), so the
real PR-5 code path -- not a re-implementation -- is under test.
"""
import warnings

import numpy as np
import h5py
import pytest

from gwcat.catalog import GWCatalog
from gwcat.ingest import (_assemble_union, _write_store, _read_store,
                          merge_stores, IngestConfig,
                          META_FLOAT_FIELDS, META_STR_FIELDS)
from gwcat.schema import (MissingParameterError, DARKSIRENS_REQUIRED,
                         PARAMETER_GROUPS)


_COSMO = (67.74, 0.3089)
_CORE = ["mass_1", "mass_2", "luminosity_distance", "ra", "dec", "chi_eff",
         "p_dL_pe"]


def _rand_event(rng, n, params):
    """A synthetic per-event sample dict for the given parameters."""
    out = {}
    for p in params:
        if p == "p_dL_pe":
            out[p] = rng.uniform(0.1, 1.0, n)
        elif p in ("ra",):
            out[p] = rng.uniform(0, 2 * np.pi, n)
        elif p in ("dec",):
            out[p] = rng.uniform(-np.pi / 2, np.pi / 2, n)
        elif p == "chi_eff":
            out[p] = rng.uniform(-0.4, 0.4, n)
        elif p == "luminosity_distance":
            out[p] = rng.uniform(300, 800, n)
        elif p in ("mass_1",):
            out[p] = rng.uniform(25, 50, n)
        elif p in ("mass_2",):
            out[p] = rng.uniform(10, 25, n)
        else:  # tidal / anything else
            out[p] = rng.uniform(0, 1000, n)
    return out


def _build_store(path, events, H0=67.74, Om0=0.3089, candidate_params=None,
                 n=12, seed=0):
    """Write a real store.h5 via the ingest union assembler + writer.

    events : list of dicts with keys
        name (str), params (list of param names this event provides), and
        optionally source_class (str).
    """
    rng = np.random.default_rng(seed)
    records, names, offsets = [], [], [0]
    classes = []
    for ev in events:
        sd = _rand_event(rng, n, ev["params"])
        records.append((ev["name"], n, sd))
        names.append(ev["name"])
        offsets.append(offsets[-1] + n)
        classes.append(ev.get("source_class", ""))

    if candidate_params is None:
        candidate_params = []
        for ev in events:
            for p in ev["params"]:
                if p not in candidate_params:
                    candidate_params.append(p)

    union_params, columns, avail = _assemble_union(records, candidate_params)

    meta = {k: [np.nan] * len(names) for k in META_FLOAT_FIELDS}
    meta.update({k: [""] * len(names) for k in META_STR_FIELDS})
    meta["dL_prior_H0"] = [H0] * len(names)
    meta["dL_prior_Om0"] = [Om0] * len(names)
    meta["source_class"] = list(classes)
    meta["compact_type"] = list(classes)

    _write_store(str(path), union_params, columns, offsets, names, avail, meta,
                 IngestConfig())
    return str(path)


# ==========================================================================
# 0. schema declaration sanity
# ==========================================================================
def test_darksirens_required_are_declarative():
    # The exporter's need-list is exactly the declared requirement.
    assert tuple(DARKSIRENS_REQUIRED) == (
        "mass_1", "mass_2", "luminosity_distance", "ra", "dec", "chi_eff",
        "p_dL_pe")
    # tidal params live in the bns_nsbh group and are NOT required.
    assert "lambda_1" in PARAMETER_GROUPS["bns_nsbh"]
    assert "lambda_1" not in DARKSIRENS_REQUIRED


# ==========================================================================
# 1. _assemble_union: differing param sets -> union with NaN fill + mask
#    (handoff: "ingesting events with differing param sets stores the union")
# ==========================================================================
def test_assemble_union_nan_fill_and_mask():
    recs = [
        ("A", 3, {"mass_1": np.array([1., 2., 3.]),
                  "chi_eff": np.array([.1, .2, .3])}),
        ("B", 2, {"mass_1": np.array([4., 5.]),
                  "lambda_1": np.array([10., 20.])}),
    ]
    # 'psi' is a candidate no event provides -> excluded from the union.
    union, cols, avail = _assemble_union(recs, ["mass_1", "chi_eff",
                                               "lambda_1", "psi"])
    assert union == ["mass_1", "chi_eff", "lambda_1"]

    np.testing.assert_array_equal(cols["mass_1"], [1, 2, 3, 4, 5])
    # A lacks lambda_1 (first 3 NaN); B lacks chi_eff (last 2 NaN).
    assert np.all(np.isnan(cols["lambda_1"][:3]))
    np.testing.assert_array_equal(cols["lambda_1"][3:], [10, 20])
    assert np.all(np.isnan(cols["chi_eff"][3:]))
    np.testing.assert_array_equal(cols["chi_eff"][:3], [.1, .2, .3])

    # Availability mask aligned with (events, union_params).
    np.testing.assert_array_equal(
        avail, np.array([[True, True, False], [True, False, True]]))


# ==========================================================================
# 2. Mixed BBH/BNS store: tidal columns preserved w/ NaN for BBH (case 6)
# ==========================================================================
def _mixed_events():
    return [
        {"name": "GW950001_000001", "source_class": "BBH", "params": _CORE},
        {"name": "GW950002_000002", "source_class": "BBH", "params": _CORE},
        {"name": "GW950003_000003", "source_class": "BNS",
         "params": _CORE + ["lambda_1", "lambda_2"]},
    ]


def test_mixed_bbh_bns_preserves_tidal_with_nan_and_mask(tmp_path):
    store = _build_store(tmp_path / "mixed.h5", _mixed_events(), seed=3)
    cat = GWCatalog(store)

    # Union schema kept the tidal columns despite only the BNS event having them.
    assert "lambda_1" in cat.params and "lambda_2" in cat.params

    # Availability mask: False for the two BBH events, True for the BNS event.
    np.testing.assert_array_equal(cat.param_available("lambda_1"),
                                  np.array([False, False, True]))

    # Per-event values: BBH slices NaN, BNS slice finite.
    per = cat.get(["lambda_1"], per_event=True)["lambda_1"]
    assert np.all(np.isnan(per[0])) and np.all(np.isnan(per[1]))
    assert np.all(np.isfinite(per[2]))

    # A BBH-only selection still exports fine -- tidal params are optional.
    out = tmp_path / "bbh.h5"
    cat.to_darksirens(str(out), source_class="bbh", nsamp=8, seed=0,
                      cosmology=_COSMO)
    with h5py.File(out, "r") as f:
        assert f.attrs["nobs"] == 2


# ==========================================================================
# 3. Required export columns fail loudly, naming param + event (case 7)
# ==========================================================================
def test_export_fails_loud_when_required_unavailable_for_event(tmp_path):
    """chi_eff present in the store (event A has it) but NaN-filled for event B
    -> export naming BOTH must fail loudly and write nothing."""
    events = [
        {"name": "GW960001_000001", "params": _CORE},
        {"name": "GW960002_000002",
         "params": [p for p in _CORE if p != "chi_eff"]},  # no chi_eff
    ]
    store = _build_store(tmp_path / "hole.h5", events, seed=4)
    cat = GWCatalog(store)
    assert "chi_eff" in cat.params  # present in store (from event A)

    out = tmp_path / "wont_exist.h5"
    with pytest.raises(MissingParameterError) as ei:
        cat.to_darksirens(str(out), nsamp=8, seed=0, cosmology=_COSMO)
    msg = str(ei.value)
    assert "chi_eff" in msg and "GW960002_000002" in msg
    assert not out.exists()

    # Selecting only the event that HAS chi_eff exports fine.
    ok = tmp_path / "ok.h5"
    cat.select(event_list=["GW960001_000001"]).to_darksirens(
        str(ok), nsamp=8, seed=0, cosmology=_COSMO)
    assert ok.exists()


def test_export_fails_loud_when_required_absent_from_store(tmp_path):
    """No event provides chi_eff -> the column is absent from the store and the
    export names it as not-in-store."""
    events = [
        {"name": "GW970001_000001",
         "params": [p for p in _CORE if p != "chi_eff"]},
        {"name": "GW970002_000002",
         "params": [p for p in _CORE if p != "chi_eff"]},
    ]
    store = _build_store(tmp_path / "nochi.h5", events, seed=5)
    cat = GWCatalog(store)
    assert "chi_eff" not in cat.params

    out = tmp_path / "wont_exist.h5"
    with pytest.raises(MissingParameterError, match="chi_eff"):
        cat.to_darksirens(str(out), nsamp=8, seed=0, cosmology=_COSMO)
    assert not out.exists()


# ==========================================================================
# 4. GWCatalog.get: required vs optional access
# ==========================================================================
def test_get_required_raises_clear_named_error(tmp_path):
    store = _build_store(tmp_path / "s.h5", _mixed_events(), seed=6)
    cat = GWCatalog(store)
    with pytest.raises(MissingParameterError, match="not_a_param"):
        cat.get(["not_a_param"])
    # Still a KeyError subclass for legacy handlers, but not a bare one.
    assert issubclass(MissingParameterError, KeyError)


def test_get_optional_returns_nan_filled(tmp_path):
    store = _build_store(tmp_path / "s.h5", _mixed_events(), seed=6)
    cat = GWCatalog(store)
    d = cat.get(["not_a_param"], required=False)
    assert d["not_a_param"].size == cat.nsamp_per_event.sum()
    assert np.all(np.isnan(d["not_a_param"]))
    # param_available reports an absent column as all-False.
    assert not cat.param_available("not_a_param").any()


# ==========================================================================
# 5. merge_stores: union schema preserved, no column drops (case 12)
# ==========================================================================
def test_merge_stores_preserves_union_schema(tmp_path):
    # Store A (BBH) additionally carries 'psi'; store B (BNS) carries tidal params.
    a_params = _CORE + ["psi"]
    b_params = _CORE + ["lambda_1", "lambda_2"]
    store_a = _build_store(tmp_path / "a.h5",
                           [{"name": "GW980001_000001", "params": a_params},
                            {"name": "GW980002_000002", "params": a_params}],
                           seed=7)
    store_b = _build_store(tmp_path / "b.h5",
                           [{"name": "GW980003_000003", "params": b_params}],
                           seed=8)

    out = tmp_path / "merged.h5"
    merge_stores(store_a, store_b, str(out))
    cat = GWCatalog(str(out))

    # No column dropped: the merged schema is the union of both stores.
    for p in set(a_params) | set(b_params):
        assert p in cat.params, f"{p} dropped by merge"
    assert cat.n_events == 3

    # 'psi' (A-only): available for the two A events, NaN for the B event.
    np.testing.assert_array_equal(cat.param_available("psi"),
                                  np.array([True, True, False]))
    # tidal (B-only): NaN for the A events, available for the B event.
    np.testing.assert_array_equal(cat.param_available("lambda_1"),
                                  np.array([False, False, True]))

    # NaN fill is real in the sample data, not just the mask.
    psi = cat.get(["psi"], per_event=True)["psi"]
    assert np.all(np.isnan(psi[2]))               # B event lacks psi
    lam = cat.get(["lambda_1"], per_event=True)["lambda_1"]
    assert np.all(np.isnan(lam[0])) and np.all(np.isfinite(lam[2]))

    # Sample bookkeeping is intact.
    assert int(cat.offsets[-1]) == 3 * 12


def test_merge_stores_skips_duplicate_events(tmp_path):
    store_a = _build_store(tmp_path / "a.h5",
                           [{"name": "GWdup", "params": _CORE},
                            {"name": "GWa2", "params": _CORE}], seed=1)
    store_b = _build_store(tmp_path / "b.h5",
                           [{"name": "GWdup", "params": _CORE},   # duplicate
                            {"name": "GWb2", "params": _CORE}], seed=2)
    out = tmp_path / "m.h5"
    with pytest.warns(UserWarning, match="Duplicate events skipped"):
        merge_stores(store_a, store_b, str(out))
    cat = GWCatalog(str(out))
    assert list(cat.event_names) == ["GWdup", "GWa2", "GWb2"]


# ==========================================================================
# 6. Legacy store (no availability mask) still loads and behaves
# ==========================================================================
def _write_legacy_store(path, n_events=2, n=15, seed=0, H0=67.74, Om0=0.3089):
    """A schema-1.0-style store: no avail group, no schema_version attr."""
    rng = np.random.default_rng(seed)
    names = [f"GW990000_{i:06d}" for i in range(n_events)]
    cols = {p: [] for p in _CORE}
    offsets = [0]
    for _ in range(n_events):
        ev = _rand_event(rng, n, _CORE)
        for p in _CORE:
            cols[p].append(ev[p])
        offsets.append(offsets[-1] + n)
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


def test_legacy_store_without_mask_loads_all_available(tmp_path):
    store = _write_legacy_store(tmp_path / "legacy.h5", seed=2)
    with h5py.File(store, "r") as f:
        assert "avail" not in f          # genuinely mask-free
    cat = GWCatalog(store)
    # Derived mask is all-True (exact for old intersection-ingested stores).
    assert cat.avail.shape == (2, len(_CORE))
    assert cat.avail.all()
    assert cat.param_available("chi_eff").all()

    # And it still exports.
    out = tmp_path / "leg.h5"
    cat.to_darksirens(str(out), nsamp=8, seed=0, cosmology=_COSMO)
    with h5py.File(out, "r") as f:
        assert f.attrs["nobs"] == 2


def test_merge_legacy_with_new_store_unions_schema(tmp_path):
    """Merging a legacy (mask-free) store with a schema-1.1 store that has an
    extra column keeps the union; legacy events are all-True for shared columns
    and False for the new column."""
    legacy = _write_legacy_store(tmp_path / "legacy.h5", n_events=2, seed=3)
    newer = _build_store(tmp_path / "new.h5",
                         [{"name": "GWnew", "params": _CORE + ["lambda_1"]}],
                         seed=4)
    out = tmp_path / "merged.h5"
    merge_stores(legacy, newer, str(out))
    cat = GWCatalog(str(out))

    assert "lambda_1" in cat.params
    # Legacy events: all shared columns available, tidal absent.
    np.testing.assert_array_equal(cat.param_available("chi_eff"),
                                  np.array([True, True, True]))
    np.testing.assert_array_equal(cat.param_available("lambda_1"),
                                  np.array([False, False, True]))
