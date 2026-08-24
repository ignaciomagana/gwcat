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
import json
import warnings

import numpy as np
import h5py
import pytest

from gwcat import schema
from gwcat.catalog import GWCatalog
from gwcat.spin import chi_p_from_components, chi_eff_chi_p_prior_logprob


DARKSIRENS_PARAMS = ["mass_1", "mass_2", "luminosity_distance", "ra", "dec",
                     "chi_eff", "p_dL_pe"]
_CANDIDATE_EXTRAS = ["a_1", "a_2", "cos_tilt_1", "cos_tilt_2",
                     "tilt_1", "tilt_2", "chi_p"]
_FULL_SPIN = {"a_1", "a_2", "cos_tilt_1", "cos_tilt_2"}
_P_DL_CONST = 0.7   # constant stored distance prior -> exact p_pe reconstruction


def _build_spin_store(tmp_path, events, n_per_event=300, H0=67.74, Om0=0.3089,
                      seed=11, name="spin_store.h5", mass_prior_meta=True):
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
                            "spin_prior_kind", "spin_prior_source",
                            "mass_prior_kind"]}

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
        # The parsed mass-prior class (GW-07/GW-34). The real releases put a
        # UniformInComponents* prior on (chirp_mass, mass_ratio), which IS flat
        # in the components, so "uniform_detector_frame" is the default here.
        meta["mass_prior_kind"].append(
            ev.get("mass_prior_kind", "uniform_detector_frame"))

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
        str_meta = ["source_class", "compact_type", "waveform", "approximant",
                    "sample_set_name", "spin_prior_kind", "spin_prior_source"]
        # A store written before the mass-prior ingest carries no such column
        # at all, which is a different state from "looked and found nothing".
        if mass_prior_meta:
            str_meta.append("mass_prior_kind")
        for k in str_meta:
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
    # Precedence bug (GW-08): `X.decode() if isinstance(X, bytes) else X == "c"`
    # parses as `X.decode() if ... else (X == "c")`, so under a bytes-returning
    # h5py the assert reduced to `assert b"component"` -- truthy for ANY
    # non-empty value, i.e. a no-op.  Decode first, then compare.
    basis = attrs["spin_basis"]
    basis = basis.decode() if isinstance(basis, bytes) else basis
    assert basis == "component"
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
def test_chi_p_definition_default_recomputes_schmidt_for_every_event(tmp_path):
    """GW-08: the default no longer PRESERVES a mismatched stored chi_p.

    This test used to assert the opposite -- that an event shipping a chi_p
    column offset from the Schmidt value kept that offset in the export.  That is
    exactly the defect: ChiEffChiPPrior is constructed strictly for the Schmidt
    definition, so a column that is a different quantity gets evaluated under a
    prior that does not describe it.  The default now recomputes.
    """
    events = [
        {"name": "GWc3_000001", "provide": _FULL_SPIN | {"chi_p"},
         "chi_p_offset": 0.05},
        {"name": "GWc3_000002", "provide": _FULL_SPIN},
    ]
    store, _raw = _build_spin_store(tmp_path, events, n_per_event=200)
    cat = GWCatalog(store)
    out = tmp_path / "chipsrc.h5"
    cat.export(str(out), format="gwcat2", spin_basis="component",
               nsamp=50, seed=2, cosmology=(67.74, 0.3089))

    cols, attrs = _read(out)
    src = [s.decode() if isinstance(s, bytes) else s
           for s in attrs["chi_p_source_per_event"]]
    assert src == ["schmidt_recomputed", "schmidt_recomputed"]
    defn = attrs["chi_p_definition"]
    assert (defn.decode() if isinstance(defn, bytes) else defn) \
        == "schmidt_recomputed"

    # Both events' chip is the Schmidt formula on the store's OWN components.
    formula = chi_p_from_components(
        cols["a1"], cols["a2"], cols["cost1"], cols["cost2"],
        cols["m1det"], cols["m2det"])
    np.testing.assert_allclose(cols["chip"], formula, rtol=1e-12)

    # The disagreement with the release's column is measured and recorded, not
    # silently carried: ~0.05 for the offset event, NaN where uncomparable.
    maxdiff = np.asarray(attrs["chi_p_def_maxdiff_per_event"], dtype=float)
    assert maxdiff[0] == pytest.approx(0.05, rel=1e-6)
    assert np.isnan(maxdiff[1])


def test_chi_p_definition_file_fails_on_a_mismatched_column(tmp_path):
    """``chi_p_definition="file"`` is available but refuses a column that is not
    the Schmidt chi_p of the store's own components."""
    from gwcat.export.pe_builder import ChiPDefinitionError

    events = [
        {"name": "GWc3_000001", "provide": _FULL_SPIN | {"chi_p"},
         "chi_p_offset": 0.05},
    ]
    store, _raw = _build_spin_store(tmp_path, events, n_per_event=200)
    cat = GWCatalog(store)
    with pytest.raises(ChiPDefinitionError) as exc:
        cat.export(str(tmp_path / "bad.h5"), format="gwcat2",
                   spin_basis="component", nsamp=50, seed=2,
                   cosmology=(67.74, 0.3089), chi_p_definition="file")
    msg = str(exc.value)
    assert "GWc3_000001" in msg
    assert "Schmidt" in msg


def test_chi_p_definition_file_accepts_a_consistent_column(tmp_path):
    """No offset -> the stored column IS the Schmidt value, so "file" is fine."""
    events = [{"name": "GWc3_000001", "provide": _FULL_SPIN | {"chi_p"}}]
    store, raw = _build_spin_store(tmp_path, events, n_per_event=200)
    cat = GWCatalog(store)
    out = tmp_path / "ok.h5"
    cat.export(str(out), format="gwcat2", spin_basis="component", nsamp=50,
               seed=2, cosmology=(67.74, 0.3089), chi_p_definition="file")
    cols, attrs = _read(out)
    src = [s.decode() if isinstance(s, bytes) else s
           for s in attrs["chi_p_source_per_event"]]
    assert src == ["file"]
    idx = _replicate_idx(raw["GWc3_000001"]["a_1"].size, 50, 2)
    np.testing.assert_allclose(cols["chip"],
                               raw["GWc3_000001"]["chi_p"][idx])


def test_chi_p_definition_rejects_an_unknown_value(tmp_path):
    events = [{"name": "GWc3_000001", "provide": _FULL_SPIN}]
    store, _raw = _build_spin_store(tmp_path, events, n_per_event=100)
    with pytest.raises(ValueError, match="chi_p_definition"):
        GWCatalog(store).export(str(tmp_path / "x.h5"), format="gwcat2",
                                spin_basis="component", nsamp=50, seed=0,
                                cosmology=(67.74, 0.3089),
                                chi_p_definition="whatever")


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
        cat.export(str(out), format="gwcat2", spin_basis="chieff_chip", allow_projection_basis=True,
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
    """A fabricated fallback amax is recorded -- and, since GW-03, is now also
    caught when it fails to cover the samples.

    The fixture's spins reach beyond ``amax_fallback=0.85``, so those samples are
    genuinely outside the joint prior's support and their ``p_pe`` is zero.  The
    export therefore refuses unless the caller opts in, which is the point: a
    made-up ceiling that excludes real samples is exactly the state that used to
    be papered over by the ``-50`` floor.
    """
    from gwcat.export.pe_builder import OutOfSupportError

    events = [{"name": "GWc5_000001", "amax1": np.nan, "amax2": np.nan}]
    store, _ = _build_spin_store(tmp_path, events)
    cat = GWCatalog(store)

    kw = dict(format="gwcat2", spin_basis="chieff_chip", allow_projection_basis=True, nsamp=24, seed=0,
              cosmology=(67.74, 0.3089), amax_fallback=0.85)
    with pytest.raises(OutOfSupportError, match="GWc5_000001"):
        cat.export(str(tmp_path / "refused.h5"), **kw)

    out = tmp_path / "fallback.h5"
    with pytest.warns(UserWarning):
        cat.export(str(out), allow_out_of_support=True, **kw)
    cols, attrs = _read(out)
    fb = [s.decode() if isinstance(s, bytes) else s
          for s in attrs["spin_amax_fallback_events"]]
    assert fb == ["GWc5_000001"]
    np.testing.assert_allclose(
        np.asarray(attrs["chi_eff_chi_p_amax_per_event"], float), [0.85])
    # The out-of-support accounting is recorded rather than hidden.
    assert int(attrs["n_samples_out_of_support"]) > 0
    assert float(attrs["frac_samples_out_of_support"]) > 0.0
    assert np.all(np.asarray(cols["in_support"], dtype=int) <= 1)
    n_zero = int(np.sum(np.asarray(cols["p_pe"], float) == 0.0))
    assert n_zero == int(attrs["n_samples_out_of_support"])
    # ESS is finite and no larger than nsamp.
    ess = np.asarray(attrs["prior_reweight_ess_per_event"], float)
    assert ess.size == 1 and 0.0 < ess[0] <= 24.0


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


# ==========================================================================
# GW-18: the PE builder is driven by the block registry
# ==========================================================================
def test_chieff_export_is_byte_identical_to_to_darksirens(tmp_path):
    """The parity contract GW-18 must not break.

    For spin_basis="chieff" the block-driven builder must reproduce the frozen
    v1 exporter's arrays EXACTLY -- same selection, same rng stream, same
    Jacobian, same chi_eff factor, same concatenation order.
    """
    from gwcat.catalog import GWCatalog

    events = [{"name": "GWp_000001", "provide": _FULL_SPIN},
              {"name": "GWp_000002", "provide": _FULL_SPIN}]
    store, _raw = _build_spin_store(tmp_path, events, n_per_event=300)
    cat = GWCatalog(store)

    kw = dict(nsamp=64, seed=3, cosmology=(67.74, 0.3089))
    v1 = tmp_path / "v1.h5"
    v2 = tmp_path / "v2.h5"
    cat.to_darksirens(str(v1), **kw)
    cat.export(str(v2), format="gwcat2", spin_basis="chieff", **kw)

    with h5py.File(v1, "r") as a, h5py.File(v2, "r") as b:
        assert int(a.attrs["nobs"]) == int(b.attrs["nobs"])
        assert int(a.attrs["nsamp"]) == int(b.attrs["nsamp"])
        for col in ("ra", "dec", "m1det", "m2det", "chieff", "dL", "p_pe",
                    "redshift", "m1src", "m2src"):
            np.testing.assert_array_equal(
                a[col][:], b[col][:],
                err_msg=f"chieff parity broken for {col!r}")


def test_rng_neutrality_chieff_fetches_no_extra_store_columns(tmp_path,
                                                              monkeypatch):
    """The mechanism behind the parity above, pinned directly.

    chieff parity holds only because that path fetches no EXTRA per-event
    columns and therefore consumes an identical default_rng(seed) stream.
    Assert the exact column set handed to the per-event reader, so a future
    space that quietly starts fetching spin columns for chieff fails here rather
    than silently shifting every exported sample.
    """
    from gwcat.catalog import GWCatalog, _SampleReader
    from gwcat.params import get_space

    events = [{"name": "GWr_000001", "provide": _FULL_SPIN}]
    store, _raw = _build_spin_store(tmp_path, events, n_per_event=200)
    cat = GWCatalog(store)

    calls = []
    orig = _SampleReader.read

    def spy(self, e, params, **kw):
        calls.append((params,) if isinstance(params, str) else tuple(params))
        return orig(self, e, params, **kw)

    monkeypatch.setattr(_SampleReader, "read", spy)
    cat.export(str(tmp_path / "c.h5"), format="gwcat2", spin_basis="chieff",
               nsamp=32, seed=0, cosmology=(67.74, 0.3089))

    # one fetch per event, and it is the legacy required set -- no spin extras
    assert len(calls) == 1, f"chieff made {len(calls)} store fetches: {calls}"
    assert set(calls[0]) == set(schema.EXPORT_REQUIREMENTS["gwcat2_pe:chieff"])
    assert get_space("chieff").store_params_fetched == ()

    # ... whereas component DOES fetch extras, which is why it is exempt from
    # the parity contract rather than violating it.
    calls.clear()
    cat.export(str(tmp_path / "k.h5"), format="gwcat2", spin_basis="component",
               nsamp=32, seed=0, cosmology=(67.74, 0.3089))
    assert len(calls) == 1, f"component made {len(calls)} fetches: {calls}"
    assert set(calls[0]) & {"a_1", "a_2"}


def test_requirements_come_from_the_registry_not_a_ladder(tmp_path):
    """A missing required parameter must still fail loudly, naming it -- the
    behaviour the removed per-basis ladder provided."""
    from gwcat.catalog import GWCatalog
    from gwcat.schema import MissingParameterError, export_requirements_for

    # the generated view agrees with the legacy tuple for every legacy basis
    for basis in ("chieff", "component", "chieff_chip"):
        assert set(export_requirements_for(basis)) == set(
            schema.EXPORT_REQUIREMENTS[f"gwcat2_pe:{basis}"])

    events = [{"name": "GWq_000001", "provide": {"a_2", "cos_tilt_1",
                                                 "cos_tilt_2"}}]
    store, _raw = _build_spin_store(tmp_path, events, n_per_event=100)
    with pytest.raises(MissingParameterError, match="a_1"):
        GWCatalog(store).export(str(tmp_path / "x.h5"), format="gwcat2",
                                spin_basis="component", nsamp=32, seed=0,
                                cosmology=(67.74, 0.3089))


# ==========================================================================
# GW-21: the nospin space exports
# ==========================================================================
def test_nospin_ppe_is_the_mass_jacobian_and_distance_prior_only(tmp_path):
    """No spin coordinate is fitted, so NO spin density enters p_pe -- not the
    chi_eff prior, not the component box.

    This is the design's stated mitigation if a 4-D spin population ever does
    collapse N_eff downstream, so it has to be exactly what it claims.
    """
    events = [{"name": "GWn_000001", "provide": _FULL_SPIN}]
    store, raw = _build_spin_store(tmp_path, events, n_per_event=200)
    out = tmp_path / "nospin.h5"
    GWCatalog(store).export(str(out), format="gwcat2", spin_basis="nospin",
                            nsamp=50, seed=4, cosmology=(67.74, 0.3089))
    cols, attrs = _read(out)

    idx = _replicate_idx(raw["GWn_000001"]["a_1"].size, 50, 4)
    m1 = raw["GWn_000001"]["mass_1"][idx]
    np.testing.assert_allclose(cols["p_pe"], m1 * _P_DL_CONST, rtol=1e-12)

    # exactly the legacy 10 columns plus the support mask and the published
    # density coordinate q (GW-34) -- no spin coordinates
    assert set(cols) == {"ra", "dec", "m1det", "m2det", "chieff", "dL", "p_pe",
                         "redshift", "m1src", "m2src", "in_support", "q"}
    assert np.all(np.asarray(cols["in_support"], dtype=bool))


def test_nospin_declares_that_no_chi_eff_prior_was_applied(tmp_path):
    """"absent" and "False" are different claims.

    darksirens REQUIRES chi_eff_in_p_pe on a gwcat-pe-2.0 file, so omitting it
    makes the file fail to LOAD with a member-check error rather than fail a
    physics check -- and a consumer must be able to tell "no chi_eff prior was
    applied" from "this file does not say".
    """
    events = [{"name": "GWn_000002", "provide": _FULL_SPIN}]
    store, _raw = _build_spin_store(tmp_path, events, n_per_event=100)
    outs = {}
    for basis in ("chieff", "component", "chieff_chip", "nospin"):
        p = tmp_path / f"{basis}.h5"
        GWCatalog(store).export(str(p), format="gwcat2", spin_basis=basis,
                                allow_projection_basis=True,
                                nsamp=32, seed=0, cosmology=(67.74, 0.3089))
        outs[basis] = _read(p)[1]

    for basis, attrs in outs.items():
        assert "chi_eff_in_p_pe" in attrs, f"{basis} omits chi_eff_in_p_pe"
        assert "spin_prior_mode" in attrs, f"{basis} omits spin_prior_mode"
        mode = attrs["spin_prior_mode"]
        mode = mode.decode() if isinstance(mode, bytes) else mode
        expect = {"chieff": ("include", True),
                  "component": ("component_flat", False),
                  "chieff_chip": ("chieff_chip_joint", False),
                  "nospin": ("none", False)}[basis]
        assert mode == expect[0], f"{basis}: spin_prior_mode={mode!r}"
        assert bool(attrs["chi_eff_in_p_pe"]) is expect[1]


def test_nospin_chieff_is_emitted_but_advisory():
    """chieff is written (every plot uses it) but no term for it is in p_pe, so
    fitting on it against a nospin file would be wrong.  The registry says so."""
    from gwcat.params import get_space

    sp = get_space("nospin")
    assert "chieff" in sp.advisory_columns
    assert "chieff" not in sp.fit_columns
    assert sp.is_exact is True


# ==========================================================================
# GW-33: the v2 builder records the EFFECTIVE selection, not its own arguments
# ==========================================================================
_MIXED_CLASS_EVENTS = [
    {"name": "GWs_000001", "source_class": "BBH"},
    {"name": "GWs_000002", "source_class": "BBH"},
    {"name": "GWs_000003", "source_class": "NSBH"},
]


def test_v2_export_from_a_filtered_view_states_the_filter(tmp_path):
    """The reviewer's reproduction: exporting ``cat.select(source_class="bbh")``
    kept every BBH row and recorded an empty ``source_class_filter``, no cut
    estimator and no event list -- a filtered file advertising itself as
    unfiltered, which the paired selection function cannot contradict.
    """
    store, _ = _build_spin_store(tmp_path, _MIXED_CLASS_EVENTS,
                                 n_per_event=80, name="mixed_class.h5")
    out = tmp_path / "from_view.h5"
    bbh = GWCatalog(store).select(source_class="bbh")
    with pytest.warns(UserWarning, match="POSTERIOR MEDIAN"):
        bbh.export(str(out), format="gwcat2", spin_basis="component",
                   nsamp=32, seed=0, cosmology=(67.74, 0.3089))

    _cols, attrs = _read(out)

    def _s(v):
        return v.decode() if isinstance(v, bytes) else v

    assert int(attrs["nobs"]) == 2                  # the rows were never wrong
    assert _s(attrs["source_class_filter"]) == "BBH"
    assert _s(attrs["source_class_cut_estimator"]) == "posterior_median_mass"
    assert bool(attrs["selection_filtered"]) is True
    assert _s(attrs["selection_spec_digest"])
    spec = json.loads(_s(attrs["selection_spec"]))
    assert spec["source_class"] == ["BBH"]


def test_v2_export_records_the_events_it_wrote(tmp_path):
    from gwcat.export.contract import event_list_digest

    store, _ = _build_spin_store(tmp_path, _MIXED_CLASS_EVENTS,
                                 n_per_event=80, name="digest_store.h5")
    out = tmp_path / "digest.h5"
    GWCatalog(store).export(str(out), format="gwcat2", spin_basis="component",
                            nsamp=32, seed=0, cosmology=(67.74, 0.3089))
    _cols, attrs = _read(out)
    names = [n.decode() if isinstance(n, bytes) else n
             for n in attrs["event_names"]]
    digest = attrs["event_list_digest"]
    digest = digest.decode() if isinstance(digest, bytes) else digest
    assert digest == event_list_digest(names)
    assert digest != event_list_digest(names[:1])


def test_v2_export_inherits_the_numeric_cuts_of_the_view(tmp_path):
    """far_max/snr_min are the numbers the paired selection file is checked
    against; a cut applied one call earlier used to reach the file as NaN."""
    events = [dict(e, far=f) for e, f in
              zip(_MIXED_CLASS_EVENTS, (1e-4, 1e-2, 1e-4))]
    store, _ = _build_spin_store(tmp_path, events, n_per_event=80,
                                 name="far_view.h5")
    out = tmp_path / "far_view_export.h5"
    sub = GWCatalog(store).select(far_max=1e-3)
    sub.export(str(out), format="gwcat2", spin_basis="component",
               nsamp=32, seed=0, cosmology=(67.74, 0.3089))
    _cols, attrs = _read(out)
    assert int(attrs["nobs"]) == 2
    assert float(attrs["far_max"]) == 1e-3
    assert (attrs["far_policy"].decode()
            if isinstance(attrs["far_policy"], bytes)
            else attrs["far_policy"]) == "drop_missing"


def test_v2_pastro_min_is_refused_unless_declared_unpaired(tmp_path):
    from gwcat.catalog import UnpairableSelectionCut

    store, _ = _build_spin_store(tmp_path, _MIXED_CLASS_EVENTS,
                                 n_per_event=80, name="pastro_store.h5")
    cat = GWCatalog(store)
    out = tmp_path / "pa.h5"
    with pytest.raises(UnpairableSelectionCut, match="p_astro_available=False"):
        cat.export(str(out), format="gwcat2", spin_basis="component",
                   nsamp=32, seed=0, cosmology=(67.74, 0.3089),
                   pastro_min=0.5)
    assert not out.exists()

    cat.export(str(out), format="gwcat2", spin_basis="component",
               nsamp=32, seed=0, cosmology=(67.74, 0.3089),
               pastro_min=0.5, allow_unpaired_pastro_min=True)
    _cols, attrs = _read(out)
    assert float(attrs["pastro_min"]) == 0.5


# ==========================================================================
# GW-34: the mass BLOCK drives p_pe, and the file states the prior it got
# ==========================================================================
def _decode(v):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in v]


def test_an_assumed_mass_prior_is_not_stamped_as_a_verified_one(tmp_path):
    """The reviewer's reproduction: the shipped store holds 273 rows parsed as
    ``uniform_detector_frame`` and 9 whose analytic prior was never found
    (``assumed_default``), and every one of them was exported with a constant
    ``mass_prior_basis="uniform_detector_frame"``.

    The m1det Jacobian is only VERIFIED for the parsed class; for the rest it is
    assumed, and a file that cannot tell a consumer which it got is a file that
    invites the wrong measure.
    """
    events = [{"name": "GWm_000001", "provide": _FULL_SPIN},
              {"name": "GWm_000002", "provide": _FULL_SPIN,
               "mass_prior_kind": "assumed_default"}]
    store, _raw = _build_spin_store(tmp_path, events, n_per_event=120,
                                    name="massprior_store.h5")
    out = tmp_path / "massprior.h5"
    with pytest.warns(UserWarning, match="no VERIFIED uniform detector-frame"):
        GWCatalog(store).export(str(out), format="gwcat2",
                                spin_basis="component", nsamp=32, seed=0,
                                cosmology=(67.74, 0.3089))
    _cols, attrs = _read(out)

    basis = attrs["mass_prior_basis"]
    assert (basis.decode() if isinstance(basis, bytes) else basis) == "mixed"
    assert bool(attrs["mass_prior_verified"]) is False
    assert _decode(attrs["mass_prior_kind_per_event"]) == [
        "uniform_detector_frame", "assumed_default"]
    assert _decode(attrs["mass_prior_unverified_events"]) == ["GWm_000002"]
    assert int(attrs["n_events_mass_prior_unverified"]) == 1


def test_a_fully_parsed_store_still_states_the_verified_basis(tmp_path):
    """The honest claim must still be available: a file every one of whose rows
    carries the parsed uniform detector-frame prior says so, with no warning."""
    events = [{"name": "GWm_000003", "provide": _FULL_SPIN},
              {"name": "GWm_000004", "provide": _FULL_SPIN}]
    store, _raw = _build_spin_store(tmp_path, events, n_per_event=120,
                                    name="massprior_ok_store.h5")
    out = tmp_path / "massprior_ok.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        GWCatalog(store).export(str(out), format="gwcat2",
                                spin_basis="component", nsamp=32, seed=0,
                                cosmology=(67.74, 0.3089))
    _cols, attrs = _read(out)
    basis = attrs["mass_prior_basis"]
    assert (basis.decode() if isinstance(basis, bytes) else basis) == \
        "uniform_detector_frame"
    assert bool(attrs["mass_prior_verified"]) is True
    assert int(attrs["n_events_mass_prior_unverified"]) == 0


def test_a_store_predating_the_mass_prior_ingest_says_unstated(tmp_path):
    """"Nothing was parsed" and "a prior was parsed and it was flat" are
    different claims; only the second may be stamped as verified."""
    events = [{"name": "GWm_000005", "provide": _FULL_SPIN}]
    store, _raw = _build_spin_store(tmp_path, events, n_per_event=120,
                                    name="massprior_old_store.h5",
                                    mass_prior_meta=False)
    out = tmp_path / "massprior_old.h5"
    with pytest.warns(UserWarning, match="no VERIFIED uniform detector-frame"):
        GWCatalog(store).export(str(out), format="gwcat2",
                                spin_basis="component", nsamp=32, seed=0,
                                cosmology=(67.74, 0.3089))
    _cols, attrs = _read(out)
    basis = attrs["mass_prior_basis"]
    assert (basis.decode() if isinstance(basis, bytes) else basis) == "unstated"
    assert bool(attrs["mass_prior_verified"]) is False


def test_an_unsupported_mass_prior_is_refused_by_the_blocks_gate(tmp_path):
    """The gate ``mass.det_pair`` has declared since GW-17, finally evaluated.

    A posterior sampled under a prior that is NOT flat in the detector-frame
    components has a different Jacobian, and the ratio does not cancel in the
    per-event normalisation -- so the export refuses instead of applying the
    wrong measure.  The refusal names every offending event, not just the first.
    """
    events = [{"name": "GWm_000006", "provide": _FULL_SPIN},
              {"name": "GWm_000007", "provide": _FULL_SPIN,
               "mass_prior_kind": "uniform_chirp_mass_q"},
              {"name": "GWm_000008", "provide": _FULL_SPIN,
               "mass_prior_kind": "unrecognized"}]
    store, _raw = _build_spin_store(tmp_path, events, n_per_event=100,
                                    name="massprior_bad_store.h5")
    out = tmp_path / "massprior_bad.h5"
    with pytest.raises(ValueError, match="Jacobian") as exc:
        GWCatalog(store).export(str(out), format="gwcat2",
                                spin_basis="component", nsamp=32, seed=0,
                                cosmology=(67.74, 0.3089))
    msg = str(exc.value)
    assert "GWm_000007" in msg and "GWm_000008" in msg
    assert "GWm_000006" not in msg
    assert not out.exists()

    # ...and excluding them is enough to build the rest.
    GWCatalog(store).export(str(out), format="gwcat2", spin_basis="component",
                            nsamp=32, seed=0, cosmology=(67.74, 0.3089),
                            allowed_names=["GWm_000006"],
                            allowed_names_authoritative=True)
    _cols, attrs = _read(out)
    assert int(attrs["nobs"]) == 1
    assert bool(attrs["mass_prior_verified"]) is True


def test_the_refusal_comes_from_the_block_not_a_copy_in_the_builder(tmp_path):
    """The block's gate is the single rule: widening it in the registry widens
    the export, which is what "the blocks drive the physics" has to mean."""
    from gwcat.params.blocks import mass as mass_block_mod

    events = [{"name": "GWm_000009", "provide": _FULL_SPIN,
               "mass_prior_kind": "uniform_detector_frame_v2"}]
    store, _raw = _build_spin_store(tmp_path, events, n_per_event=100,
                                    name="massprior_gate_store.h5")
    out = tmp_path / "massprior_gate.h5"
    with pytest.raises(ValueError, match="Jacobian"):
        GWCatalog(store).export(str(out), format="gwcat2",
                                spin_basis="component", nsamp=32, seed=0,
                                cosmology=(67.74, 0.3089))

    old = mass_block_mod.VALID_MASS_PRIOR_KINDS
    mass_block_mod.VALID_MASS_PRIOR_KINDS = old + ("uniform_detector_frame_v2",)
    try:
        GWCatalog(store).export(str(out), format="gwcat2",
                                spin_basis="component", nsamp=32, seed=0,
                                cosmology=(67.74, 0.3089))
    finally:
        mass_block_mod.VALID_MASS_PRIOR_KINDS = old
    _cols, attrs = _read(out)
    assert bool(attrs["mass_prior_verified"]) is True


def test_the_export_publishes_q_as_the_density_coordinate(tmp_path):
    """``p_pe`` carries ``m1det`` -- exactly ``|d(m1det,m2det)/d(m1det,q)|`` --
    so it is a density in ``(m1det, q)``.  The file used to publish
    ``(m1det, m2det)`` as its fit columns, which is a different measure.
    """
    events = [{"name": "GWq_000001", "provide": _FULL_SPIN}]
    store, raw = _build_spin_store(tmp_path, events, n_per_event=150,
                                   name="qcol_store.h5")
    out = tmp_path / "qcol.h5"
    GWCatalog(store).export(str(out), format="gwcat2", spin_basis="component",
                            nsamp=40, seed=1, cosmology=(67.74, 0.3089))
    cols, attrs = _read(out)

    assert "q" in cols
    np.testing.assert_allclose(cols["q"], cols["m2det"] / cols["m1det"],
                               rtol=0, atol=0)
    # m2det is DERIVED from the published coordinates, and says so.
    np.testing.assert_allclose(cols["q"] * cols["m1det"], cols["m2det"],
                               rtol=1e-12)
    assert np.all((cols["q"] > 0) & (cols["q"] <= 1.0))
    coord = attrs["mass_density_coordinates"]
    assert (coord.decode() if isinstance(coord, bytes) else coord) == "m1det,q"

    idx = _replicate_idx(raw["GWq_000001"]["a_1"].size, 40, 1)
    np.testing.assert_allclose(
        cols["q"], (raw["GWq_000001"]["mass_2"][idx]
                    / raw["GWq_000001"]["mass_1"][idx]), rtol=0, atol=0)


def test_p_pe_is_bit_identical_to_the_hand_written_jacobian(tmp_path):
    """Composing the block must not perturb a single float: the chieff export is
    contractually byte-identical to the frozen v1 exporter, and
    ``exp(ln_prior_pe)`` differs from ``m1det`` for most samples -- which is why
    the block declares the factor in linear space as well as in the log."""
    events = [{"name": "GWq_000002", "provide": _FULL_SPIN}]
    store, raw = _build_spin_store(tmp_path, events, n_per_event=200,
                                   name="qbit_store.h5")
    out = tmp_path / "qbit.h5"
    GWCatalog(store).export(str(out), format="gwcat2", spin_basis="nospin",
                            nsamp=64, seed=2, cosmology=(67.74, 0.3089))
    cols, _attrs = _read(out)
    idx = _replicate_idx(raw["GWq_000002"]["a_1"].size, 64, 2)
    m1 = raw["GWq_000002"]["mass_1"][idx]
    np.testing.assert_array_equal(cols["p_pe"], m1 * _P_DL_CONST)


# ==========================================================================
# GW-31: the chieff basis uses the EVENT's own ceiling, and support() gates it
# ==========================================================================
def test_chieff_ppe_uses_each_events_own_prior_ceiling(tmp_path):
    """The chi_eff prior removed from p_pe must be the prior that sampled it.

    The chieff path used to evaluate one caller-supplied ``amax`` for every
    event, so an analysis run under ``a ~ U(0, 0.5)`` had a 0.99 prior divided
    out of it -- and the chi_eff density depends on the ceiling in a
    chi_eff-DEPENDENT way, so the error survives every normalisation.
    """
    from gwcat.spin import chi_eff_prior_logprob

    events = [{"name": "GWa1_000001", "amax1": 0.99, "amax2": 0.99},
              {"name": "GWa1_000002", "amax1": 0.50, "amax2": 0.50}]
    store, _ = _build_spin_store(tmp_path, events, name="amax_store.h5")
    cat = GWCatalog(store)
    auto = tmp_path / "chieff_auto.h5"
    cat.export(str(auto), format="gwcat2", spin_basis="chieff", nsamp=48,
               seed=0, cosmology=(67.74, 0.3089))
    cols, attrs = _read(auto)

    nsamp = int(attrs["nsamp"])
    a1 = np.asarray(attrs["chi_eff_amax_1_per_event"], float)
    a2 = np.asarray(attrs["chi_eff_amax_2_per_event"], float)
    np.testing.assert_allclose(a1, [0.99, 0.50], rtol=0, atol=0)
    np.testing.assert_allclose(a2, [0.99, 0.50], rtol=0, atol=0)
    mode = attrs["chi_eff_amax_mode"]
    assert (mode.decode() if isinstance(mode, bytes) else mode) == "per_event"
    srcs = [s.decode() if isinstance(s, bytes) else s
            for s in attrs["chi_eff_amax_source_per_event"]]
    assert srcs == ["analytic", "analytic"]

    for i in range(2):
        sl = slice(i * nsamp, (i + 1) * nsamp)
        expected = cols["m1det"][sl] * _P_DL_CONST * np.exp(
            chi_eff_prior_logprob(cols["chieff"][sl], cols["m1src"][sl],
                                  cols["m2src"][sl], amax=a1[i], amax_2=a2[i]))
        np.testing.assert_allclose(cols["p_pe"][sl], expected, rtol=1e-12)

    # Forcing one ceiling (the old, and still available, behaviour) leaves the
    # 0.99 event alone and CHANGES the one whose prior is not 0.99.
    forced = tmp_path / "chieff_forced.h5"
    cat.export(str(forced), format="gwcat2", spin_basis="chieff", nsamp=48,
               seed=0, cosmology=(67.74, 0.3089), amax=0.99)
    fcols, fattrs = _read(forced)
    np.testing.assert_array_equal(fcols["p_pe"][:nsamp], cols["p_pe"][:nsamp])
    assert not np.allclose(fcols["p_pe"][nsamp:], cols["p_pe"][nsamp:],
                           rtol=1e-6)
    fmode = fattrs["chi_eff_amax_mode"]
    assert (fmode.decode() if isinstance(fmode, bytes) else fmode) == "fixed"
    assert float(fattrs["chi_eff_amax"]) == 0.99


def test_chieff_support_is_the_predicate_not_finiteness(tmp_path):
    """A sample just above the ceiling is EXCLUDED, not given a 2.7e-12 density.

    ``ChiEffPrior.logprob`` returns a finite ~1e-12 there (the grid clamp), so
    the old ``in_support = isfinite(logp)`` admitted it -- with an inverse weight
    ~1e12 times too large in the very denominator the export exists to provide.
    """
    from gwcat.export.pe_builder import OutOfSupportError

    events = [{"name": "GWa2_000001", "amax1": 0.99, "amax2": 0.99}]
    store, _ = _build_spin_store(tmp_path, events, n_per_event=64,
                                 name="oos_store.h5")
    with h5py.File(store, "r+") as f:
        ce = f["samples/chi_eff"][:]
        ce[:] = 0.995                     # above amax = 0.99, below 1
        f["samples/chi_eff"][...] = ce

    cat = GWCatalog(store)
    with pytest.raises(OutOfSupportError) as exc:
        cat.export(str(tmp_path / "oos.h5"), format="gwcat2",
                   spin_basis="chieff", nsamp=32, seed=0,
                   cosmology=(67.74, 0.3089))
    assert "outside the spin prior's support" in str(exc.value)

    # With the gate deliberately opened, the density is EXACTLY zero and the
    # count is on the file -- not a small number nobody can see.
    out = tmp_path / "oos_allowed.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cat.export(str(out), format="gwcat2", spin_basis="chieff", nsamp=32,
                   seed=0, cosmology=(67.74, 0.3089),
                   allow_out_of_support=True, allow_zero_p_pe=True)
    cols, attrs = _read(out)
    assert np.all(cols["p_pe"] == 0.0)
    assert np.all(np.asarray(cols["in_support"], dtype=bool) == False)  # noqa: E712
    assert int(attrs["n_samples_out_of_support"]) == 32


# ==========================================================================
# GW-32 (#8): the draw happens BEFORE the read
# ==========================================================================
def test_pe_export_reads_only_the_rows_it_emits(tmp_path, monkeypatch):
    """The builder must not read a posterior slice it is about to throw away.

    It used to fetch every selected slice with ``get(..., per_event=True)`` and
    index ``nsamp`` rows out of it afterwards -- 6.88M rows read to emit 1.16M
    on the shipped store, 1.20 GiB peak.  Pin the order: no whole-selection
    fetch, and the rows the per-event reader hands back are exactly the rows
    that reach the file.
    """
    from gwcat.catalog import GWCatalog as _GWCatalog, _SampleReader

    events = [{"name": "GWz_00000%d" % i, "provide": _FULL_SPIN, "n": n}
              for i, n in enumerate((400, 250, 900), start=1)]
    store, _raw = _build_spin_store(tmp_path, events, name="rowcount.h5")
    cat = GWCatalog(store)
    nsamp = 32

    rows_read = []
    orig_read = _SampleReader.read

    def spy_read(self, e, params, rows=None, **kw):
        out = orig_read(self, e, params, rows=rows, **kw)
        rows_read.append(sum(np.size(v) for v in out.values()))
        return out

    got = []
    orig_get = _GWCatalog.get

    def spy_get(self, params, **kw):
        got.append(tuple(params))
        return orig_get(self, params, **kw)

    monkeypatch.setattr(_SampleReader, "read", spy_read)
    monkeypatch.setattr(_GWCatalog, "get", spy_get)

    out = tmp_path / "rowcount.h5.out"
    cat.export(str(out), format="gwcat2", spin_basis="component",
               nsamp=nsamp, seed=0, cosmology=(67.74, 0.3089))

    with h5py.File(out, "r") as f:
        emitted = f["m1det"].size

    # The columns the component basis reads: its required set plus the extra
    # spin columns this store actually carries.
    from gwcat.params import get_space
    from gwcat.export.pe_builder import space_ordered_required
    space = get_space("component")
    ncols_read = len(set(space_ordered_required(space, "component"))
                     | {p for p in space.store_params_fetched
                        if p in cat.params})

    assert emitted == nsamp * len(events)
    # every read is a drawn-row read, and they sum to the emitted rows x columns
    assert len(rows_read) == len(events)
    assert sum(rows_read) == nsamp * len(events) * ncols_read
    # ... and not one whole-selection posterior fetch happened
    assert got == [], f"export still fetched whole slices: {got}"


def test_pe_export_z_max_reads_dL_once_then_only_the_drawn_rows(tmp_path,
                                                               monkeypatch):
    """The z_max cut is the ONE thing that needs values before the draw.

    It gets a single full ``luminosity_distance`` column per event; everything
    else is still read at the drawn rows only.
    """
    from gwcat.catalog import _SampleReader

    events = [{"name": "GWy_000001", "provide": _FULL_SPIN, "n": 500}]
    store, _raw = _build_spin_store(tmp_path, events, name="zmaxrows.h5")
    cat = GWCatalog(store)

    calls = []
    orig_read = _SampleReader.read

    def spy_read(self, e, params, rows=None, **kw):
        calls.append((params, None if rows is None else len(rows)))
        return orig_read(self, e, params, rows=rows, **kw)

    monkeypatch.setattr(_SampleReader, "read", spy_read)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cat.export(str(tmp_path / "zmax.h5"), format="gwcat2",
                   spin_basis="chieff", nsamp=16, seed=0, z_max=5.0,
                   cosmology=(67.74, 0.3089))

    assert calls[0] == ("luminosity_distance", None)
    assert all(n == 16 for _params, n in calls[1:])


def test_pe_export_emits_the_rows_the_rng_drew_from_the_full_slice(tmp_path):
    """Downsampling before the read may not move a single sample value.

    Reproduce the draw independently -- ``default_rng(seed).choice(n, nsamp)``
    per event over the WHOLE stored slice, exactly as the pre-GW-32 read-then-
    index builder did -- and require the exported columns to be those rows.
    """
    events = [{"name": "GWx_00000%d" % i, "provide": _FULL_SPIN, "n": n}
              for i, n in enumerate((120, 300), start=1)]
    store, raw = _build_spin_store(tmp_path, events, name="drawparity.h5")
    cat = GWCatalog(store)
    nsamp, seed = 24, 5

    out = tmp_path / "drawparity.out.h5"
    cat.export(str(out), format="gwcat2", spin_basis="chieff", nsamp=nsamp,
               seed=seed, cosmology=(67.74, 0.3089))

    rng = np.random.default_rng(seed)
    want = {"m1det": [], "m2det": [], "dL": [], "ra": [], "dec": [],
            "chieff": []}
    src = {"m1det": "mass_1", "m2det": "mass_2", "dL": "luminosity_distance",
           "ra": "ra", "dec": "dec", "chieff": "chi_eff"}
    for ev in events:
        n = int(ev["n"])
        idx = rng.choice(n, size=nsamp, replace=False)
        for col, param in src.items():
            want[col].append(raw[ev["name"]][param][idx])

    with h5py.File(out, "r") as f:
        for col in want:
            np.testing.assert_array_equal(
                f[col][:], np.concatenate(want[col]),
                err_msg=f"{col} is not the rows the rng drew")
