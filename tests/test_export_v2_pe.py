"""Component / chieff_chip spin-basis PE export tests (PR 6).

Exercises the two new spin bases added to :func:`gwcat.export.pe_builder.build_pe_product`
(the chieff basis and its parity live in ``tests/test_export_v2.py``):

  1. component ``p_pe`` carries the flat component-spin factor
     ``1/(4·amax_1·amax_2)`` with the EVENT's own amax (event-dependent constant).
  2. component emits ``a1/a2/cost1/cost2/chip`` and shares the chieff scaffold
     bit-for-bit (identical resample -> identical ``m1det``).
  3. a tilt-only store (no ``cos_tilt`` columns) derives cos at export.
  4. ``chi_p`` is taken from file when present, else derived; recorded per event.
  5. chieff_chip ``p_pe`` carries the joint (chi_eff, chi_p) prior; amax mismatch
     warns; NaN store amax falls back with a recorded attr.
  6. missing required / alternative spin params fail loudly, naming them.
  7. a store WITH spin columns still gives chieff output == to_darksirens.
  8. an ``unrecognized`` spin_prior_kind warns and is recorded.

Fixtures are tiny synthetic HDF5 stores built directly with h5py (matching the
on-disk schema GWCatalog reads, including the per-event availability mask) -- no
network, no pesummary/ingest.
"""
import numpy as np
import h5py
import pytest

from gwcat.catalog import GWCatalog
from gwcat.spin import chi_p_from_components, chi_eff_chi_p_prior_logprob


DARKSIRENS_PARAMS = ["mass_1", "mass_2", "luminosity_distance", "ra", "dec",
                     "chi_eff", "p_dL_pe"]
_CANDIDATE_EXTRAS = ["a_1", "a_2", "cos_tilt_1", "cos_tilt_2",
                     "tilt_1", "tilt_2", "chi_p"]
_FULL_SPIN = {"a_1", "a_2", "cos_tilt_1", "cos_tilt_2"}
_P_DL_CONST = 0.7   # constant stored distance prior -> exact p_pe reconstruction


def _build_spin_store(tmp_path, events, n_per_event=300, H0=67.74, Om0=0.3089,
                      seed=11, name="spin_store.h5"):
    """Synthetic store with spin sample columns + per-event spin-prior meta +
    an availability mask.  Returns ``(path, raw)`` where ``raw[name][param]`` is
    the stored per-event sample array (so tests can replicate the resample)."""
    rng = np.random.default_rng(seed)
    extras_union = [p for p in _CANDIDATE_EXTRAS
                    if any(p in ev.get("provide", _FULL_SPIN)
                           for ev in events)]
    params = DARKSIRENS_PARAMS + extras_union
    pidx = {p: j for j, p in enumerate(params)}
    n_events = len(events)
    avail = np.ones((n_events, len(params)), dtype=bool)

    offsets = [0]
    col_lists = {p: [] for p in params}
    per_event_raw = {}
    names = []
    meta = {k: [] for k in ["source_class", "compact_type", "far",
                            "far_available", "pastro", "p_astro",
                            "dL_prior_H0", "dL_prior_Om0", "waveform",
                            "approximant", "sample_set_name",
                            "spin_amax_1", "spin_amax_2",
                            "spin_prior_kind", "spin_prior_source"]}

    for ei, ev in enumerate(events):
        n = int(ev.get("n", n_per_event))
        name = ev["name"]
        names.append(name)
        amax1 = float(ev.get("amax1", 0.99))
        amax2 = float(ev.get("amax2", 0.99))
        a1max_gen = amax1 if np.isfinite(amax1) else 0.99
        a2max_gen = amax2 if np.isfinite(amax2) else 0.99

        m1 = rng.uniform(20, 45, n)
        m2 = rng.uniform(8, 20, n)
        raw = {
            "mass_1": m1,
            "mass_2": m2,
            "luminosity_distance": rng.uniform(300, 800, n),
            "ra": rng.uniform(0, 2 * np.pi, n),
            "dec": rng.uniform(-np.pi / 2, np.pi / 2, n),
            "chi_eff": rng.uniform(-0.3, 0.3, n),
            "p_dL_pe": np.full(n, float(ev.get("p_dL", _P_DL_CONST))),
        }
        a1 = rng.uniform(0, a1max_gen, n)
        a2 = rng.uniform(0, a2max_gen, n)
        tilt1 = rng.uniform(0, np.pi, n)
        tilt2 = rng.uniform(0, np.pi, n)
        cos1 = np.cos(tilt1)
        cos2 = np.cos(tilt2)
        chip = (chi_p_from_components(a1, a2, cos1, cos2, m1, m2)
                + float(ev.get("chi_p_offset", 0.0)))
        gen = {"a_1": a1, "a_2": a2, "tilt_1": tilt1, "tilt_2": tilt2,
               "cos_tilt_1": cos1, "cos_tilt_2": cos2, "chi_p": chip}

        provide = set(ev.get("provide", _FULL_SPIN))
        for p in extras_union:
            if p in provide:
                raw[p] = gen[p]
            else:
                raw[p] = np.full(n, np.nan)
                avail[ei, pidx[p]] = False

        per_event_raw[name] = raw
        for p in params:
            col_lists[p].append(raw[p])
        offsets.append(offsets[-1] + n)

        far = float(ev.get("far", 5e-4))
        meta["source_class"].append(ev.get("source_class", "BBH"))
        meta["compact_type"].append(ev.get("source_class", "BBH"))
        meta["far"].append(far)
        meta["far_available"].append(1.0 if np.isfinite(far) else 0.0)
        meta["pastro"].append(float(ev.get("pastro", 0.99)))
        meta["p_astro"].append(float(ev.get("pastro", 0.99)))
        meta["dL_prior_H0"].append(H0)
        meta["dL_prior_Om0"].append(Om0)
        wf = ev.get("waveform", "IMRPhenomXPHM")
        meta["waveform"].append(wf)
        meta["approximant"].append(wf)
        meta["sample_set_name"].append(f"C01:{wf}")
        meta["spin_amax_1"].append(amax1)
        meta["spin_amax_2"].append(amax2)
        meta["spin_prior_kind"].append(
            ev.get("kind", "uniform_magnitude_isotropic"))
        meta["spin_prior_source"].append(ev.get("prior_source", "C01:Mixed"))

    path = tmp_path / name
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "1.2"
        f.attrs["param_names"] = np.array(params, dtype=h5py.string_dtype())
        idx = f.create_group("index")
        idx.create_dataset("offsets", data=np.array(offsets, dtype="i8"))
        idx.create_dataset("event_names",
                           data=np.array(names, dtype=h5py.string_dtype()))
        f.create_group("avail").create_dataset("mask", data=avail)
        mg = f.create_group("meta")
        for k in ["source_class", "compact_type", "waveform", "approximant",
                  "sample_set_name", "spin_prior_kind", "spin_prior_source"]:
            mg.create_dataset(k, data=np.array(meta[k],
                                                dtype=h5py.string_dtype()))
        for k in ["far", "far_available", "pastro", "p_astro",
                  "dL_prior_H0", "dL_prior_Om0", "spin_amax_1", "spin_amax_2"]:
            mg.create_dataset(k, data=np.asarray(meta[k], dtype="f8"))
        sg = f.create_group("samples")
        for p in params:
            sg.create_dataset(p, data=np.concatenate(col_lists[p]))
    return str(path), per_event_raw


def _replicate_idx(n, nsamp, seed, replace="auto"):
    """Reproduce the builder's single-event resample indices (only valid when
    exactly one event reaches rng.choice, i.e. it is the first kept event)."""
    rep = (n < nsamp) if replace == "auto" else bool(replace)
    return np.random.default_rng(seed).choice(n, size=nsamp, replace=rep)


def _read(path):
    with h5py.File(path, "r") as f:
        cols = {k: f[k][:] for k in f.keys()}
        attrs = dict(f.attrs)
    return cols, attrs


# ==========================================================================
# 1. component p_pe carries the flat component-spin factor 1/(4 amax1 amax2)
# ==========================================================================
def test_component_ppe_constant_per_event(tmp_path):
    events = [
        {"name": "GWc0_000001", "amax1": 0.99, "amax2": 0.99},
        {"name": "GWc0_000002", "amax1": 0.80, "amax2": 0.50},
    ]
    store, _ = _build_spin_store(tmp_path, events)
    cat = GWCatalog(store)
    out = tmp_path / "component.h5"
    cat.export(str(out), format="gwcat2", spin_basis="component",
               nsamp=48, seed=0, cosmology=(67.74, 0.3089))

    cols, attrs = _read(out)
    kept = [n.decode() if isinstance(n, bytes) else n
            for n in attrs["event_names"]]
    nsamp = int(attrs["nsamp"])
    amax1 = np.asarray(attrs["spin_amax_1_per_event"], float)
    amax2 = np.asarray(attrs["spin_amax_2_per_event"], float)

    ratios = []
    for i in range(len(kept)):
        sl = slice(i * nsamp, (i + 1) * nsamp)
        # p_pe = m1det * p_dL_pe / (4 amax1 amax2); p_dL_pe is constant.
        ratio = cols["p_pe"][sl] / (cols["m1det"][sl] * _P_DL_CONST)
        expected = 1.0 / (4.0 * amax1[i] * amax2[i])
        np.testing.assert_allclose(ratio, expected, rtol=1e-12)
        ratios.append(expected)

    # The two events' amax differ -> different constants survive concatenation.
    assert not np.isclose(ratios[0], ratios[1])
    assert attrs["spin_basis"].decode() if isinstance(
        attrs["spin_basis"], bytes) else attrs["spin_basis"] == "component"
    assert bool(attrs["component_spin_prior_applied_to_p_pe"]) is True


# ==========================================================================
# 2. component shares the chieff scaffold; emits a1/a2/cost1/cost2/chip
# ==========================================================================
def test_component_scaffold_invariance_and_columns(tmp_path):
    events = [{"name": "GWc1_000001", "amax1": 0.99, "amax2": 0.99},
              {"name": "GWc1_000002", "amax1": 0.90, "amax2": 0.90}]
    store, raw = _build_spin_store(tmp_path, events)
    cat = GWCatalog(store)
    kw = dict(nsamp=64, seed=3, cosmology=(67.74, 0.3089))

    chieff_out = tmp_path / "chieff.h5"
    comp_out = tmp_path / "comp.h5"
    cat.export(str(chieff_out), format="gwcat2", spin_basis="chieff", **kw)
    cat.export(str(comp_out), format="gwcat2", spin_basis="component", **kw)

    cch, _ = _read(chieff_out)
    ccm, attrs = _read(comp_out)

    # Strong scaffold-invariance: shared columns are IDENTICAL (same rng stream).
    for k in ["ra", "dec", "m1det", "m2det", "chieff", "dL", "redshift",
              "m1src", "m2src"]:
        np.testing.assert_array_equal(cch[k], ccm[k], err_msg=f"{k} differs")

    # New columns present, and mutually consistent (chip == Schmidt(a,cost,m)).
    for k in ["a1", "a2", "cost1", "cost2", "chip"]:
        assert k in ccm
    chip_check = chi_p_from_components(ccm["a1"], ccm["a2"], ccm["cost1"],
                                       ccm["cost2"], ccm["m1det"], ccm["m2det"])
    np.testing.assert_allclose(ccm["chip"], chip_check, rtol=1e-12)
    assert np.all(ccm["cost1"] <= 1.0) and np.all(ccm["cost1"] >= -1.0)

    # And the emitted a1 equals the store's a_1 resampled with the same rng
    # (single-event slice reconstruction: first kept event, n=300, nsamp=64).
    name0 = (attrs["event_names"][0].decode()
             if isinstance(attrs["event_names"][0], bytes)
             else attrs["event_names"][0])
    n0 = raw[name0]["a_1"].size
    idx0 = _replicate_idx(n0, 64, 3)
    np.testing.assert_allclose(ccm["a1"][:64], raw[name0]["a_1"][idx0])


# ==========================================================================
# 3. tilt-only store -> cos derived at export == cos(resampled tilts)
# ==========================================================================
def test_component_tilt_only_derives_cos(tmp_path):
    events = [{"name": "GWc2_000001", "amax1": 0.99, "amax2": 0.99,
               "provide": {"a_1", "a_2", "tilt_1", "tilt_2"}}]
    store, raw = _build_spin_store(tmp_path, events)
    cat = GWCatalog(store)
    out = tmp_path / "tiltonly.h5"
    cat.export(str(out), format="gwcat2", spin_basis="component",
               nsamp=64, seed=1, cosmology=(67.74, 0.3089))

    cols, attrs = _read(out)
    assert "cos_tilt_1" not in cat.params   # store really had no cos columns
    name0 = raw["GWc2_000001"]
    idx0 = _replicate_idx(name0["tilt_1"].size, 64, 1)
    np.testing.assert_allclose(cols["cost1"], np.cos(name0["tilt_1"][idx0]))
    np.testing.assert_allclose(cols["cost2"], np.cos(name0["tilt_2"][idx0]))


# ==========================================================================
# 4. chi_p from file vs derived; chi_p_source_per_event records which
# ==========================================================================
def test_chi_p_source_file_vs_derived(tmp_path):
    # Event A ships chi_p (offset from the formula so "file" is distinguishable);
    # event B ships only ingredients -> chi_p derived at export.
    events = [
        {"name": "GWc3_000001", "provide": _FULL_SPIN | {"chi_p"},
         "chi_p_offset": 0.05},
        {"name": "GWc3_000002", "provide": _FULL_SPIN},
    ]
    store, raw = _build_spin_store(tmp_path, events, n_per_event=200)
    cat = GWCatalog(store)
    out = tmp_path / "chipsrc.h5"
    cat.export(str(out), format="gwcat2", spin_basis="component",
               nsamp=50, seed=2, cosmology=(67.74, 0.3089))

    cols, attrs = _read(out)
    src = [s.decode() if isinstance(s, bytes) else s
           for s in attrs["chi_p_source_per_event"]]
    assert src == ["file", "derived"]

    # File event: chip == stored chi_p resampled (carries the +0.05 offset, so
    # it is NOT the bare formula).
    idxA = _replicate_idx(raw["GWc3_000001"]["a_1"].size, 50, 2)
    np.testing.assert_allclose(cols["chip"][:50],
                               raw["GWc3_000001"]["chi_p"][idxA])
    formulaA = chi_p_from_components(
        cols["a1"][:50], cols["a2"][:50], cols["cost1"][:50],
        cols["cost2"][:50], cols["m1det"][:50], cols["m2det"][:50])
    assert np.allclose(cols["chip"][:50], formulaA + 0.05, rtol=1e-9)

    # Derived event: chip == Schmidt formula on the resampled ingredients.
    formulaB = chi_p_from_components(
        cols["a1"][50:], cols["a2"][50:], cols["cost1"][50:],
        cols["cost2"][50:], cols["m1det"][50:], cols["m2det"][50:])
    np.testing.assert_allclose(cols["chip"][50:], formulaB, rtol=1e-12)


# ==========================================================================
# 5. chieff_chip p_pe = m1det*p_dL*exp(clip(joint lnprob)); mismatch/NaN
# ==========================================================================
def test_chieff_chip_ppe_joint_prior(tmp_path):
    events = [
        {"name": "GWc4_000001", "amax1": 0.99, "amax2": 0.99},
        {"name": "GWc4_000002", "amax1": 0.80, "amax2": 0.50},   # mismatch
    ]
    store, _ = _build_spin_store(tmp_path, events)
    cat = GWCatalog(store)
    out = tmp_path / "chieffchip.h5"
    with pytest.warns(UserWarning, match="spin_amax_1 != spin_amax_2"):
        cat.export(str(out), format="gwcat2", spin_basis="chieff_chip",
                   nsamp=32, seed=0, cosmology=(67.74, 0.3089))

    cols, attrs = _read(out)
    assert bool(attrs["chi_eff_chi_p_prior_applied_to_p_pe"]) is True
    amax1 = np.asarray(attrs["chi_eff_chi_p_amax_per_event"], float)
    mism = [s.decode() if isinstance(s, bytes) else s
            for s in attrs["spin_amax_mismatch_events"]]
    assert mism == ["GWc4_000002"]
    assert "chip" in cols

    nsamp = int(attrs["nsamp"])
    for i in range(len(amax1)):
        sl = slice(i * nsamp, (i + 1) * nsamp)
        lp = chi_eff_chi_p_prior_logprob(
            cols["chieff"][sl], cols["chip"][sl], cols["m1src"][sl],
            cols["m2src"][sl], amax=float(amax1[i]))
        expected = (cols["m1det"][sl] * _P_DL_CONST
                    * np.exp(np.clip(np.asarray(lp, float), -50.0, None)))
        np.testing.assert_allclose(cols["p_pe"][sl], expected, rtol=1e-9)


def test_chieff_chip_nan_amax_falls_back(tmp_path):
    events = [{"name": "GWc5_000001", "amax1": np.nan, "amax2": np.nan}]
    store, _ = _build_spin_store(tmp_path, events)
    cat = GWCatalog(store)
    out = tmp_path / "fallback.h5"
    with pytest.warns(UserWarning, match="amax_fallback"):
        cat.export(str(out), format="gwcat2", spin_basis="chieff_chip",
                   nsamp=24, seed=0, cosmology=(67.74, 0.3089),
                   amax_fallback=0.85)
    cols, attrs = _read(out)
    fb = [s.decode() if isinstance(s, bytes) else s
          for s in attrs["spin_amax_fallback_events"]]
    assert fb == ["GWc5_000001"]
    np.testing.assert_allclose(
        np.asarray(attrs["chi_eff_chi_p_amax_per_event"], float), [0.85])


# ==========================================================================
# 6. missing required / alternative spin params fail loudly, naming them
# ==========================================================================
def test_missing_a1_for_one_event_raises(tmp_path):
    from gwcat.schema import MissingParameterError
    events = [
        {"name": "GWc6_000001", "provide": _FULL_SPIN},
        {"name": "GWc6_000002", "provide": {"a_2", "cos_tilt_1", "cos_tilt_2"}},
    ]  # event 2 lacks a_1
    store, _ = _build_spin_store(tmp_path, events)
    cat = GWCatalog(store)
    with pytest.raises(MissingParameterError) as exc:
        cat.export(str(tmp_path / "x.h5"), format="gwcat2",
                   spin_basis="component", nsamp=8, cosmology=(67.74, 0.3089))
    msg = str(exc.value)
    assert "a_1" in msg and "GWc6_000002" in msg


def test_missing_both_tilt_alternatives_raises(tmp_path):
    from gwcat.schema import MissingParameterError
    # a_1/a_2 present, but neither cos_tilt_1 nor tilt_1 anywhere -> the
    # alternative group is unsatisfied; the error names both alternatives.
    events = [{"name": "GWc7_000001",
               "provide": {"a_1", "a_2", "cos_tilt_2"}}]
    store, _ = _build_spin_store(tmp_path, events)
    cat = GWCatalog(store)
    with pytest.raises(MissingParameterError) as exc:
        cat.export(str(tmp_path / "x.h5"), format="gwcat2",
                   spin_basis="component", nsamp=8, cosmology=(67.74, 0.3089))
    msg = str(exc.value)
    assert "cos_tilt_1" in msg and "tilt_1" in msg


# ==========================================================================
# 7. a store WITH spin columns still gives chieff output == to_darksirens
# ==========================================================================
def test_chieff_parity_on_spin_store(tmp_path):
    events = [{"name": "GWc8_000001"}, {"name": "GWc8_000002"}]
    store, _ = _build_spin_store(tmp_path, events)
    cat = GWCatalog(store)
    legacy = tmp_path / "legacy.h5"
    v2 = tmp_path / "v2.h5"
    kw = dict(nsamp=40, seed=9, cosmology=(67.74, 0.3089))
    cat.to_darksirens(str(legacy), **kw)
    cat.export(str(v2), format="gwcat2", spin_basis="chieff", **kw)
    with h5py.File(legacy, "r") as fa, h5py.File(v2, "r") as fb:
        for k in ["ra", "dec", "m1det", "m2det", "chieff", "dL", "p_pe",
                  "redshift", "m1src", "m2src"]:
            np.testing.assert_array_equal(fa[k][:], fb[k][:],
                                          err_msg=f"{k} differs")
        # chieff basis carries NONE of the new component/chieff_chip attrs.
        assert "component_spin_prior_applied_to_p_pe" not in fb.attrs
        assert "chi_p_source_per_event" not in fb.attrs


# ==========================================================================
# 8. unrecognized spin_prior_kind -> warning + attr
# ==========================================================================
def test_unrecognized_spin_prior_kind_warns(tmp_path):
    events = [
        {"name": "GWc9_000001", "kind": "uniform_magnitude_isotropic"},
        {"name": "GWc9_000002", "kind": "unrecognized"},
    ]
    store, _ = _build_spin_store(tmp_path, events)
    cat = GWCatalog(store)
    out = tmp_path / "unrec.h5"
    with pytest.warns(UserWarning, match="uniform_magnitude_isotropic"):
        cat.export(str(out), format="gwcat2", spin_basis="component",
                   nsamp=16, seed=0, cosmology=(67.74, 0.3089))
    _, attrs = _read(out)
    unrec = [s.decode() if isinstance(s, bytes) else s
             for s in attrs["spin_prior_unrecognized_events"]]
    assert unrec == ["GWc9_000002"]
