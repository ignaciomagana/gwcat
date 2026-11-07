"""Each sample of an O3 ``C01:Mixed`` set is divided by its OWN analysis's
spin prior (owner, 7 Nov 2026; :mod:`gwcat.mixed_spin_prior`).

GWTC-2.1/3 ``C01:Mixed`` concatenates C01:IMRPhenomXPHM (``a_i ~ U(0, 0.99)``)
and C01:SEOBNRv4PHM (RIFT: config ceiling 0.99, samples to ~1.0, one fixed
``geocent_time``).  The component-basis export used to give the whole set one
box widened to its largest drawn magnitude; each XPHM row then carried the
wrong box by ``(0.99/1)^2``.  The fixtures copy the real layout, including the
Mixed rows that equal no released constituent row (the cosmo-reweighted
releases dropped them) and RIFT's fixed time.
"""
import json
import warnings

import h5py
import numpy as np
import pytest

import gwcat.ingest as ing
from gwcat.catalog import GWCatalog
from gwcat.mixed_spin_prior import (KERR_BOUND, PER_SAMPLE_AMAX_PARAMS,
                                    MixedSpinPriorError, annotate_store,
                                    attribute_mixed_rows,
                                    constituent_ceilings,
                                    resolve_mixed_spin_prior)

XP, MX, SE = "C01:IMRPhenomXPHM", "C01:Mixed", "C01:SEOBNRv4PHM"
NAME = "IGWN-GWTC3p0-v2-GW951109_010717_PEDataRelease_mixed_cosmo.h5"
T_RIFT = 1257296855.216797

_CFG = dict(validate_prior=False, dL_prior_impl="astropy")


def _samples(rng, n, amax, t=None):
    m1 = rng.uniform(30.0, 50.0, n)
    q = rng.uniform(0.5, 1.0, n)
    a1, a2 = rng.uniform(0.0, amax, n), rng.uniform(0.0, amax, n)
    t1, t2 = rng.uniform(0.0, np.pi, n), rng.uniform(0.0, np.pi, n)
    chi = (a1 * np.cos(t1) + q * a2 * np.cos(t2)) / (1 + q)
    return {"mass_1": m1, "mass_2": q * m1,
            "luminosity_distance": rng.uniform(500.0, 2000.0, n),
            "ra": rng.uniform(0.0, 2 * np.pi, n),
            "dec": np.arcsin(rng.uniform(-1.0, 1.0, n)),
            "a_1": a1, "a_2": a2, "tilt_1": t1, "tilt_2": t2,
            "chi_eff": chi,
            "geocent_time": (np.full(n, T_RIFT) if t is not None
                             else 1257296855.2 + rng.uniform(-0.01, 0.01, n))}


def _take(s, idx):
    return {k: np.asarray(v)[idx] for k, v in s.items()}


def _concat(*parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _o3_mixed(seed=3, n=300):
    """XPHM block then SEOBNRv4PHM block; a third of each block equals no
    released row (as in the cosmo-reweighted releases).  Returns the
    analyses, the analytic priors and the configs, and the true owner of
    every Mixed row (0 = XPHM, 1 = SEOBNRv4PHM)."""
    rng = np.random.default_rng(seed)
    xp = _samples(rng, n, 0.99)
    xp_hidden = _samples(rng, n // 3, 0.99)          # dropped from the release
    se = _samples(rng, n, 1.0, t=T_RIFT)
    se_hidden = _samples(rng, n // 3, 1.0, t=T_RIFT)
    # every SEOB set of the releases has magnitudes above 0.99
    se["a_1"][0] = 0.9999
    se_hidden["a_2"][0] = 0.9995
    k = n // 2
    mixed = _concat(_take(xp, np.arange(k)), xp_hidden,
                    _take(se, np.arange(k)), se_hidden)
    truth = np.r_[np.zeros(k + n // 3, int), np.ones(k + n // 3, int)]
    sd = {XP: xp, MX: mixed, SE: se}
    analytic = {"chirp_mass": (
        "bilby.gw.prior.UniformInComponentsChirpMass(minimum=10.0, "
        "maximum=60.0, name='chirp_mass')"),
        "mass_ratio": ("bilby.gw.prior.UniformInComponentsMassRatio("
                       "minimum=0.05, maximum=1.0, name='mass_ratio')"),
        "luminosity_distance": (
            "PowerLaw(alpha=2, minimum=10, maximum=10000, "
            "name='luminosity_distance', latex_label='$d_L$', unit='Mpc', "
            "boundary=None)"),
        "a_1": "Uniform(minimum=0, maximum=0.99, name='a_1', "
               "latex_label='$a_1$', unit=None, boundary=None)",
        "a_2": "Uniform(minimum=0, maximum=0.99, name='a_2', "
               "latex_label='$a_2$', unit=None, boundary=None)"}
    priors = {"analytic": {XP: analytic}}
    config = {XP: {}, MX: {},
              SE: {"engine": {"a_spin1-max": "0.99", "a_spin2-max": "0.99"}}}
    return sd, priors, config, truth


class _Data:
    def __init__(self, samples_dict, extra=None, config=None):
        self.samples_dict = samples_dict
        self.config = config or {}
        self.extra_kwargs = [(extra or {}).get(lab, {})
                             for lab in samples_dict]


def _write_release(path, sd, priors, config):
    """The fixture as an HDF5 release file (the layout the annotation reads)."""
    with h5py.File(path, "w") as f:
        for lab, smp in sd.items():
            g = f.create_group(lab)
            keys = list(smp)
            arr = np.zeros(len(smp[keys[0]]),
                           dtype=[(k, "f8") for k in keys])
            for k in keys:
                arr[k] = smp[k]
            g.create_dataset("posterior_samples", data=arr)
            an = priors["analytic"].get(lab)
            pa = g.create_group("priors").create_group("analytic")
            for k, v in (an or {}).items():
                pa.create_dataset(k, data=np.array([v.encode()]))
            eng = config.get(lab, {}).get("engine")
            if eng:
                ge = g.create_group("config_file").create_group("engine")
                for k, v in eng.items():
                    ge.create_dataset(k, data=np.array([v.encode()]))


def _ingest(tmp_path, *, per_sample=True, name="store.h5", seed=3,
            write_release=True, monkeypatch=None):
    sd, priors, config, truth = _o3_mixed(seed)
    extra = {MX: {"meta_data": {"combined": "Prod5, Prod6_extended"}}}
    data = _Data(sd, extra=extra, config=config)
    monkeypatch.setattr(ing, "_read_event_pesummary",
                        lambda path: (data, sd, list(sd), priors))
    raw = tmp_path / NAME
    if write_release:
        if not raw.exists():
            _write_release(str(raw), sd, priors, config)
    else:
        raw.write_bytes(b"")
    out = tmp_path / name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ing.build_store([str(raw)], str(out), event_table={},
                        cfg=ing.IngestConfig(per_sample_spin_prior=per_sample,
                                             **_CFG))
    return str(out), sd, truth


def _export(store, out, basis="component", **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        GWCatalog(store).export(str(out), format="gwcat2", spin_basis=basis,
                                nsamp=512, seed=0, **kw)
    with h5py.File(out, "r") as f:
        cols = {k: f[k][:] for k in f.keys()}
        attrs = dict(f.attrs)
    return cols, attrs


def _s(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


# --------------------------------------------------------------------------
# Attribution of the Mixed rows
# --------------------------------------------------------------------------
def test_rows_are_attributed_by_exact_match_time_signature_and_remainder():
    sd, _p, _c, truth = _o3_mixed()
    labels, rows, methods, tsig = attribute_mixed_rows(
        sd[MX], {XP: sd[XP], SE: sd[SE]})
    assert labels == [XP, SE]
    np.testing.assert_array_equal(rows, truth)
    n = 300
    assert methods == {"exact": n, "time_signature": n // 3,
                       "remainder": n // 3}
    assert tsig == {SE: T_RIFT}


def test_attribution_refuses_rules_that_disagree():
    sd, _p, _c, _t = _o3_mixed()
    cons = {XP: sd[XP], SE: sd[SE]}
    # an XPHM row carrying RIFT's fixed time
    bad = {k: v.copy() for k, v in sd[MX].items()}
    bad["geocent_time"][0] = T_RIFT
    with pytest.raises(MixedSpinPriorError, match="time signature"):
        attribute_mixed_rows(bad, cons)
    # a row equal to a released SEOBNRv4PHM row but without its time
    bad = {k: v.copy() for k, v in sd[MX].items()}
    j = int(np.flatnonzero(bad["geocent_time"] == T_RIFT)[0])
    bad["geocent_time"][j] = 1.0
    with pytest.raises(MixedSpinPriorError, match="lack its time"):
        attribute_mixed_rows(bad, cons)
    # no time signature: the rows no released row equals have no owner
    se_t = dict(sd[SE], geocent_time=np.linspace(0, 1, 300))
    with pytest.raises(MixedSpinPriorError, match="no single"):
        attribute_mixed_rows(sd[MX], {XP: sd[XP], SE: se_t})


def test_each_analysis_keeps_its_own_ceiling():
    assert constituent_ceilings(XP, (0.99, 0.99), None, (0.99, 0.989)) == (
        (0.99, 0.99), "analytic")
    (c, src) = constituent_ceilings(SE, None, (0.99, 0.99), (0.9999, 0.9))
    assert c == (KERR_BOUND, KERR_BOUND) and src.startswith("kerr_bound")
    assert constituent_ceilings(SE, None, (0.99, 0.99), (0.98, 0.9)) == (
        (0.99, 0.99), "config_file")
    with pytest.raises(MixedSpinPriorError, match="own analytic prior"):
        constituent_ceilings(XP, (0.99, 0.99), None, (0.995, 0.9))
    with pytest.raises(MixedSpinPriorError, match="declares no spin prior"):
        constituent_ceilings(SE, None, None, (0.9999, 0.9))


def test_one_shared_ceiling_needs_no_per_sample_ceilings():
    sd, _p, _c, _t = _o3_mixed()
    sd2 = {k: dict(v) for k, v in sd.items()}
    for lab in (MX, SE):
        for a in ("a_1", "a_2"):
            sd2[lab][a] = np.minimum(sd2[lab][a], 0.98)
    got = resolve_mixed_spin_prior(
        sd2[MX], {XP: sd2[XP], SE: sd2[SE]},
        {XP: (0.99, 0.99), SE: None}, {XP: None, SE: (0.99, 0.99)})
    assert got is None


# --------------------------------------------------------------------------
# Ingest and export
# --------------------------------------------------------------------------
def test_ingest_stores_each_samples_own_ceiling(tmp_path, monkeypatch):
    store, sd, truth = _ingest(tmp_path, monkeypatch=monkeypatch)
    with h5py.File(store, "r") as f:
        a1 = f[f"samples/{PER_SAMPLE_AMAX_PARAMS[0]}"][:]
        a2 = f[f"samples/{PER_SAMPLE_AMAX_PARAMS[1]}"][:]
        names = [_s(x) for x in f["meta/analysis_used"][:]]
        prov = json.loads(_s(f["meta/spin_prior_per_sample"][0]))
    assert names == [MX]
    want = np.where(truth == 0, 0.99, 1.0)
    np.testing.assert_array_equal(a1, want)
    np.testing.assert_array_equal(a2, want)
    assert prov["ceilings"] == {XP: [0.99, 0.99], SE: [1.0, 1.0]}
    assert prov["sources"][XP] == "analytic"
    assert prov["sources"][SE].startswith("kerr_bound")
    assert prov["rows"] == {XP: 250, SE: 250}


def test_component_export_divides_each_sample_by_its_own_box(
        tmp_path, monkeypatch):
    """The finding: an XPHM row of C01:Mixed must carry U(0, 0.99)^2, a
    SEOBNRv4PHM row U(0, 1)^2 -- not one box for the whole set."""
    store, sd, truth = _ingest(tmp_path, monkeypatch=monkeypatch)
    cols, attrs = _export(store, tmp_path / "pe.h5")
    # the per-sample ceilings the rows were drawn from, recovered by matching
    # every exported row back to the Mixed set
    mixed = sd[MX]
    key = {(a, b): i for i, (a, b) in enumerate(
        zip(mixed["luminosity_distance"].tolist(), mixed["mass_1"].tolist()))}
    idx = np.array([key[(a, b)] for a, b in zip(cols["dL"].tolist(),
                                                 cols["m1det"].tolist())])
    own = np.where(truth[idx] == 0, 0.99, 1.0)
    assert set(np.unique(own)) == {0.99, 1.0}
    with h5py.File(store, "r") as f:
        p_dl = f["samples/p_dL_pe"][:][idx]
    expected = cols["m1det"] * p_dl / (4.0 * own * own)
    np.testing.assert_allclose(cols["p_pe"], expected, rtol=1e-12, atol=0)
    assert [_s(x) for x in attrs["spin_amax_source_1_per_event"]] == [
        "per_sample_constituent"]
    assert float(attrs["spin_amax_1_per_event"][0]) == 1.0
    assert float(attrs["spin_amax_inflation_per_event"][0]) == 0.0
    assert [_s(x) for x in attrs["spin_prior_per_sample_events"]] == [
        "GW951109_010717"]
    pv = json.loads(_s(attrs["spin_prior_per_sample_per_event"][0]))
    assert sum(c for _a, _b, c in pv["drawn_rows_per_ceiling"]) == 512
    assert bool(np.all(cols["in_support"]))


def test_store_without_per_sample_ceilings_exports_as_before(
        tmp_path, monkeypatch):
    """No per-sample columns: the one widened box, bit for bit as before,
    and every dataset but p_pe equals the per-sample export's."""
    old, _sd, _t = _ingest(tmp_path, per_sample=False, name="old.h5",
                           monkeypatch=monkeypatch)
    new, _sd, _t = _ingest(tmp_path, per_sample=True, name="new.h5",
                           monkeypatch=monkeypatch)
    with h5py.File(old, "r") as f:
        assert PER_SAMPLE_AMAX_PARAMS[0] not in f["samples"]
    co, ao = _export(old, tmp_path / "pe_old.h5")
    cn, an = _export(new, tmp_path / "pe_new.h5")
    assert [_s(x) for x in ao["spin_amax_source_1_per_event"]] == [
        "samples_bound"]
    a_r1 = float(ao["spin_amax_1_per_event"][0])
    a_r2 = float(ao["spin_amax_2_per_event"][0])
    assert a_r1 > 0.999
    assert "spin_prior_per_sample_events" not in ao
    for k in co:
        if k != "p_pe":
            np.testing.assert_array_equal(co[k], cn[k], err_msg=k)
    ratio = cn["p_pe"] / co["p_pe"]
    # XPHM rows: (a_r1 a_r2) / 0.99^2; SEOBNRv4PHM rows: a_r1 a_r2 / 1
    np.testing.assert_allclose(
        np.unique(np.round(ratio, 12)),
        np.round(sorted([a_r1 * a_r2, a_r1 * a_r2 / 0.99 ** 2]), 12))


def test_projection_basis_keeps_one_ceiling_and_says_so(tmp_path,
                                                        monkeypatch):
    old, _sd, _t = _ingest(tmp_path, per_sample=False, name="old.h5",
                           monkeypatch=monkeypatch)
    new, _sd, _t = _ingest(tmp_path, per_sample=True, name="new.h5",
                           monkeypatch=monkeypatch)
    kw = dict(allow_out_of_support=True, max_out_of_support_frac=1.0)
    co, _ = _export(old, tmp_path / "c_old.h5", basis="chieff", **kw)
    with pytest.warns(UserWarning, match="only the component basis"):
        GWCatalog(new).export(str(tmp_path / "c_new.h5"), format="gwcat2",
                              spin_basis="chieff", nsamp=512, seed=0, **kw)
    with h5py.File(tmp_path / "c_new.h5", "r") as f:
        for k in co:
            np.testing.assert_array_equal(co[k], f[k][:], err_msg=k)


# --------------------------------------------------------------------------
# Adding the ceilings to an existing store
# --------------------------------------------------------------------------
def test_annotating_a_store_adds_the_ceilings_and_changes_nothing_else(
        tmp_path, monkeypatch):
    old, _sd, _t = _ingest(tmp_path, per_sample=False, name="old.h5",
                           monkeypatch=monkeypatch)
    new, _sd, _t = _ingest(tmp_path, per_sample=True, name="new.h5",
                           monkeypatch=monkeypatch)
    out = str(tmp_path / "annotated.h5")
    summ = annotate_store(old, out, [str(tmp_path)])
    assert summ["n_rows_annotated"] == 1
    with h5py.File(old, "r") as f, h5py.File(out, "r") as g, \
            h5py.File(new, "r") as h:
        for grp in ("samples", "meta", "index"):
            for k in f[grp]:
                if k == "spin_prior_per_sample":
                    continue
                np.testing.assert_array_equal(f[grp][k][:], g[grp][k][:],
                                              err_msg=f"{grp}/{k}")
        n_old = f["avail/mask"].shape[1]
        np.testing.assert_array_equal(f["avail/mask"][:],
                                      g["avail/mask"][:, :n_old])
        assert g["avail/mask"][:, n_old:].all()
        for p in PER_SAMPLE_AMAX_PARAMS:
            np.testing.assert_array_equal(g[f"samples/{p}"][:],
                                          h[f"samples/{p}"][:])
        prov = json.loads(_s(g["meta/spin_prior_per_sample"][0]))
        assert prov["ceilings"] == {XP: [0.99, 0.99], SE: [1.0, 1.0]}
        assert prov["release_file"] == NAME
    # and it exports exactly as a store ingested with the ceilings does
    ca, _ = _export(out, tmp_path / "pe_a.h5")
    cn, _ = _export(new, tmp_path / "pe_n.h5")
    for k in cn:
        np.testing.assert_array_equal(ca[k], cn[k], err_msg=k)
    with pytest.raises(FileExistsError):
        annotate_store(old, out, [str(tmp_path)])


def test_annotation_refuses_a_release_whose_rows_are_not_the_stores(
        tmp_path, monkeypatch):
    old, _sd, _t = _ingest(tmp_path, per_sample=False, name="old.h5",
                           monkeypatch=monkeypatch)
    with h5py.File(tmp_path / NAME, "r+") as f:
        ps = f[MX]["posterior_samples"]
        arr = ps[:]
        arr["luminosity_distance"][3] += 1.0
        ps[...] = arr
    out = str(tmp_path / "annotated.h5")
    summ = annotate_store(old, out, [str(tmp_path)])
    assert summ["n_rows_annotated"] == 0
    row = summ["rows"][f"GW951109_010717 [{MX}]"]
    assert row["status"] == "unresolved"
    with h5py.File(out, "r") as g:
        assert not g["avail/mask"][0, -1]
    # an unresolved row exports with its one box, as before
    _co, ao = _export(out, tmp_path / "pe.h5")
    assert [_s(x) for x in ao["spin_amax_source_1_per_event"]] == [
        "samples_bound"]
