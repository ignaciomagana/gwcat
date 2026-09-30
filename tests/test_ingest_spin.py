"""PE-ingest spin-extension tests (PR 2).

Covers the spin additions to :mod:`gwcat.ingest`:

  * derived sample columns -- ``cos_tilt_i = cos(tilt_i)`` and the Schmidt 2015
    ``chi_p`` -- are computed at ingest when the file lacks them, stored as real
    columns, correctly marked available in the per-event availability mask, and
    recorded per event in the ``derived_params`` meta field;
  * when the file already provides ``chi_p`` it is KEPT verbatim and a
    definition-consistency diagnostic ``chi_p_def_maxdiff`` is recorded instead;
  * per-event spin-magnitude prior resolution mirrors the distance-prior
    resolution: analytic ``Uniform(0, amax)`` / ``Sine`` reprs are parsed
    (``kind="uniform_magnitude_isotropic"``), a missing priors group falls back
    to ``amax=0.99`` / ``kind="assumed_default"``, exotic reprs are flagged
    ``kind="unrecognized"`` with the raw repr recorded, and spin samples above
    the resolved ``amax`` warn;
  * REGRESSION: the new derivations never perturb the legacy darksirens export.

The build_store path is driven through the repo's fake-PESummary monkeypatch
pattern (see ``tests/test_waveform_policy.py``), so the real PR-2 code -- not a
re-implementation -- is under test.
"""
import numpy as np
import h5py
import pytest

from gwcat.catalog import GWCatalog
from gwcat.ingest import (build_store, IngestConfig,
                          _parse_analytic_spin, resolve_spin_prior,
                          _chi_p_from_samples, _derive_spin_columns)


_COSMO = (67.74, 0.3089)


# --------------------------------------------------------------------------
# Fake PESummary reader (no pesummary / bilby / network)
# --------------------------------------------------------------------------
class _FakeData:
    """Minimal stand-in for a pesummary read() result (no config -> f_ref NaN)."""


def _fake_reader(analyses, priors=None):
    """Return a fake ``_read_event_pesummary`` yielding the given analyses/priors."""
    priors = priors or {}

    def _fake(path):
        return _FakeData(), analyses, list(analyses.keys()), priors
    return _fake


def _core_samples(rng, n):
    """Darksirens-required posterior columns for one analysis."""
    return {
        "mass_1": rng.uniform(25, 50, n),
        "mass_2": rng.uniform(10, 25, n),
        "luminosity_distance": rng.uniform(300, 800, n),
        "ra": rng.uniform(0, 2 * np.pi, n),
        "dec": rng.uniform(-np.pi / 2, np.pi / 2, n),
        "chi_eff": rng.uniform(-0.4, 0.4, n),
    }


def _spin_samples(rng, n, amax=0.9):
    """a_1/a_2/tilt_1/tilt_2 for chi_p / cos_tilt derivation."""
    return {
        "a_1": rng.uniform(0, amax, n),
        "a_2": rng.uniform(0, amax, n),
        "tilt_1": rng.uniform(0, np.pi, n),
        "tilt_2": rng.uniform(0, np.pi, n),
    }


def _uniform_isotropic_priors(analysis, amax=0.99):
    """A bilby-style analytic priors group for one analysis."""
    return {"analytic": {analysis: {
        "a_1": f"Uniform(minimum=0.0, maximum={amax}, name='a_1', "
               f"latex_label='$a_1$', boundary=None)",
        "a_2": f"Uniform(minimum=0.0, maximum={amax}, name='a_2', "
               f"latex_label='$a_2$', boundary=None)",
        "tilt_1": "Sine(name='tilt_1', minimum=0.0, "
                  "maximum=3.141592653589793, latex_label='$\\theta_1$')",
        "tilt_2": "Sine(name='tilt_2', minimum=0.0, "
                  "maximum=3.141592653589793, latex_label='$\\theta_2$')",
    }}}


def _ingest_one(tmp_path, monkeypatch, analyses, priors=None,
                name="GWTC-3_GW950101_000101_cosmo.h5", out="store.h5"):
    """Run the real build_store with a faked reader; return a GWCatalog."""
    import gwcat.ingest as ing
    monkeypatch.setattr(ing, "_read_event_pesummary",
                        _fake_reader(analyses, priors))
    path = tmp_path / name
    path.write_bytes(b"")  # existence only; the reader is faked
    out_path = tmp_path / out
    build_store([str(path)], str(out_path), event_table={},
                cfg=IngestConfig(validate_prior=False))
    return GWCatalog(str(out_path))


# ==========================================================================
# 0. pure-unit sanity of the parse / derive helpers
# ==========================================================================
def test_parse_analytic_spin_variants():
    assert _parse_analytic_spin(
        "Uniform(minimum=0.0, maximum=0.99, name='a_1')") == ("Uniform", 0.0, 0.99)
    assert _parse_analytic_spin(
        "Sine(name='tilt_1', minimum=0.0, maximum=3.141592653589793)") == (
        "Sine", 0.0, 3.141592653589793)
    assert _parse_analytic_spin("not a prior at all") is None
    # exotic but well-formed: class + bounds still parse (caller flags kind)
    assert _parse_analytic_spin(
        "Beta(minimum=0.0, maximum=1.0, alpha=2, beta=2)") == ("Beta", 0.0, 1.0)


def test_chi_p_formula_matches_schmidt():
    a1 = np.array([0.5, 0.5]); a2 = np.array([0.5, 0.5])
    ct1 = np.cos(np.array([0.0, np.pi / 2]))
    ct2 = np.cos(np.array([np.pi / 2, 0.0]))
    m1 = np.array([30.0, 30.0]); m2 = np.array([15.0, 15.0])
    got = _chi_p_from_samples(a1, a2, ct1, ct2, m1, m2)
    q = m2 / m1
    s1 = np.sqrt(1 - ct1 ** 2); s2 = np.sqrt(1 - ct2 ** 2)
    want = np.maximum(a1 * s1, q * (4 * q + 3) / (4 + 3 * q) * a2 * s2)
    np.testing.assert_allclose(got, want, rtol=0, atol=0)


# ==========================================================================
# 1. tilt-only file -> cos_tilt / chi_p derived, available, recorded
# ==========================================================================
def test_derived_columns_from_tilts(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n = 64
    s = _core_samples(rng, n)
    s.update(_spin_samples(rng, n))          # a_i, tilt_i but NO cos_tilt/chi_p
    analyses = {"C01:Mixed": s}
    cat = _ingest_one(tmp_path, monkeypatch, analyses)

    for p in ("cos_tilt_1", "cos_tilt_2", "chi_p"):
        assert p in cat.params
        assert cat.param_available(p).all()   # derived -> available

    # values match a direct numpy computation on the source samples
    per = cat.get(["cos_tilt_1", "cos_tilt_2", "chi_p"], per_event=True)
    np.testing.assert_allclose(per["cos_tilt_1"][0], np.cos(s["tilt_1"]))
    np.testing.assert_allclose(per["cos_tilt_2"][0], np.cos(s["tilt_2"]))
    want_chi_p = _chi_p_from_samples(s["a_1"], s["a_2"], np.cos(s["tilt_1"]),
                                     np.cos(s["tilt_2"]), s["mass_1"], s["mass_2"])
    np.testing.assert_allclose(per["chi_p"][0], want_chi_p)

    # derived_params records exactly the three derived columns
    assert set(cat.meta["derived_params"][0].split(",")) == {
        "cos_tilt_1", "cos_tilt_2", "chi_p"}
    # chi_p was derived (not from file) -> no definition-mismatch diagnostic
    assert np.isnan(cat.meta["chi_p_def_maxdiff"][0])


# ==========================================================================
# 2. file-provided chi_p kept; chi_p_def_maxdiff records the mismatch
# ==========================================================================
def test_file_chi_p_kept_and_maxdiff_recorded(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    n = 48
    s = _core_samples(rng, n)
    s.update(_spin_samples(rng, n))
    formula = _chi_p_from_samples(s["a_1"], s["a_2"], np.cos(s["tilt_1"]),
                                  np.cos(s["tilt_2"]), s["mass_1"], s["mass_2"])
    # File's chi_p deliberately differs from the formula by exactly 1e-3.
    s["chi_p"] = formula + 1e-3
    analyses = {"C01:Mixed": s}
    cat = _ingest_one(tmp_path, monkeypatch, analyses)

    # The file's chi_p is kept verbatim (NOT overwritten by the formula).
    per = cat.get(["chi_p"], per_event=True)["chi_p"]
    np.testing.assert_allclose(per[0], formula + 1e-3)
    # The diagnostic captures the (constant) 1e-3 offset.
    assert cat.meta["chi_p_def_maxdiff"][0] == pytest.approx(1e-3, rel=1e-6)
    # chi_p was NOT derived (it came from the file); cos_tilt still are.
    assert "chi_p" not in cat.meta["derived_params"][0].split(",")
    assert set(cat.meta["derived_params"][0].split(",")) == {
        "cos_tilt_1", "cos_tilt_2"}


# ==========================================================================
# 3. analytic Uniform(0,0.99)/Sine priors -> parsed amax + recognized kind
# ==========================================================================
def test_spin_prior_uniform_isotropic_parsed(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    n = 40
    s = _core_samples(rng, n)
    s.update(_spin_samples(rng, n, amax=0.5))
    analyses = {"C01:Mixed": s}
    priors = _uniform_isotropic_priors("C01:Mixed", amax=0.99)
    cat = _ingest_one(tmp_path, monkeypatch, analyses, priors=priors)

    assert cat.meta["spin_amax_1"][0] == pytest.approx(0.99)
    assert cat.meta["spin_amax_2"][0] == pytest.approx(0.99)
    assert cat.meta["spin_prior_kind"][0] == "uniform_magnitude_isotropic"
    assert "C01:Mixed" in cat.meta["spin_prior_source"][0]


def test_spin_prior_sibling_search(tmp_path, monkeypatch):
    """O4-style: the chosen Mixed set has no priors; a sibling carries them."""
    rng = np.random.default_rng(3)
    n = 30
    mixed = _core_samples(rng, n); mixed.update(_spin_samples(rng, n, 0.5))
    sib = _core_samples(rng, n); sib.update(_spin_samples(rng, n, 0.5))
    analyses = {"C00:Mixed": mixed, "C00:IMRPhenomXPHM": sib}
    priors = _uniform_isotropic_priors("C00:IMRPhenomXPHM", amax=0.99)
    cat = _ingest_one(tmp_path, monkeypatch, analyses, priors=priors,
                      name="GWTC-4_GW240101_000101.hdf5")  # native O4 name (GW-40a)
    assert cat.meta["spin_prior_kind"][0] == "uniform_magnitude_isotropic"
    assert "C00:IMRPhenomXPHM" in cat.meta["spin_prior_source"][0]


# ==========================================================================
# 4. no priors group anywhere -> fallback amax=0.99, assumed_default
# ==========================================================================
def test_spin_prior_fallback_default(tmp_path, monkeypatch):
    rng = np.random.default_rng(4)
    n = 32
    s = _core_samples(rng, n)
    s.update(_spin_samples(rng, n, amax=0.5))
    analyses = {"C01:Mixed": s}
    cat = _ingest_one(tmp_path, monkeypatch, analyses, priors={})

    assert cat.meta["spin_amax_1"][0] == pytest.approx(0.99)
    assert cat.meta["spin_amax_2"][0] == pytest.approx(0.99)
    assert cat.meta["spin_prior_kind"][0] == "assumed_default"
    assert "no_analytic_prior" in cat.meta["spin_prior_source"][0]


# ==========================================================================
# 5. exotic / unparseable prior repr -> unrecognized, raw repr in source
# ==========================================================================
def test_spin_prior_unrecognized_records_raw_repr(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    n = 32
    s = _core_samples(rng, n)
    s.update(_spin_samples(rng, n, amax=0.5))
    analyses = {"C01:Mixed": s}
    exotic = "Beta(minimum=0.0, maximum=1.0, alpha=2.0, beta=5.0, name='a_1')"
    priors = {"analytic": {"C01:Mixed": {
        "a_1": exotic,
        "a_2": "Uniform(minimum=0.0, maximum=0.99, name='a_2')",
    }}}
    cat = _ingest_one(tmp_path, monkeypatch, analyses, priors=priors)

    assert cat.meta["spin_prior_kind"][0] == "unrecognized"
    src = cat.meta["spin_prior_source"][0]
    assert "unrecognized" in src and "Beta" in src   # raw repr carried through


# ==========================================================================
# 6. a_i samples above the resolved amax -> warning
# ==========================================================================
def test_spin_samples_exceeding_amax_warns(tmp_path, monkeypatch):
    import gwcat.ingest as ing
    rng = np.random.default_rng(6)
    n = 32
    s = _core_samples(rng, n)
    s.update(_spin_samples(rng, n, amax=0.5))
    s["a_1"] = np.full(n, 1.4)               # far above amax=0.99
    analyses = {"C01:Mixed": s}
    priors = _uniform_isotropic_priors("C01:Mixed", amax=0.99)

    monkeypatch.setattr(ing, "_read_event_pesummary",
                        _fake_reader(analyses, priors))
    path = tmp_path / "GWTC-3_GW950102_000102_cosmo.h5"
    path.write_bytes(b"")
    out = tmp_path / "warn.h5"
    with pytest.warns(UserWarning, match="samples exceed"):
        build_store([str(path)], str(out), event_table={},
                    cfg=IngestConfig(validate_prior=False))
    cat = GWCatalog(str(out))
    assert "sample_exceeds_amax" in cat.meta["spin_prior_source"][0]


# ==========================================================================
# 7. REGRESSION: new derivations never perturb the legacy darksirens export
# ==========================================================================
def test_derivations_do_not_perturb_darksirens_export(tmp_path, monkeypatch):
    """Ingest the SAME core posterior with and without the spin ingredients that
    trigger the new derivations, then export both to darksirens. The exported
    datasets must be byte-identical -- the added columns are inert for the
    legacy export."""
    rng = np.random.default_rng(7)
    n = 80
    core = _core_samples(rng, n)

    # "plain": only the darksirens-required columns (no tilt/a -> no derivation)
    plain = {k: np.array(v, copy=True) for k, v in core.items()}
    # "derived": identical core columns PLUS spin ingredients (-> cos_tilt/chi_p)
    derived = {k: np.array(v, copy=True) for k, v in core.items()}
    derived.update(_spin_samples(rng, n))

    cat_plain = _ingest_one(tmp_path, monkeypatch, {"C01:Mixed": plain},
                            name="GWTC-3_GW950101_000101_cosmo.h5",
                            out="plain.h5")
    cat_derived = _ingest_one(tmp_path, monkeypatch, {"C01:Mixed": derived},
                              name="GWTC-3_GW950101_000101_cosmo.h5",
                              out="derived.h5")

    # Sanity: the derived store really gained the new columns; the plain one did not.
    assert "chi_p" in cat_derived.params and "cos_tilt_1" in cat_derived.params
    assert "chi_p" not in cat_plain.params

    out_plain = tmp_path / "ds_plain.h5"
    out_derived = tmp_path / "ds_derived.h5"
    kw = dict(nsamp=32, seed=0, cosmology=_COSMO)
    cat_plain.to_darksirens(str(out_plain), **kw)
    cat_derived.to_darksirens(str(out_derived), **kw)

    with h5py.File(out_plain, "r") as fp, h5py.File(out_derived, "r") as fd:
        assert set(fp.keys()) == set(fd.keys())
        for key in fp.keys():
            np.testing.assert_array_equal(
                fp[key][:], fd[key][:],
                err_msg=f"darksirens dataset {key!r} perturbed by derivations")


# ==========================================================================
# 8. old stores (no spin meta) remain loadable via the read path
# ==========================================================================
def test_store_without_spin_meta_still_loads(tmp_path, monkeypatch):
    """A store written without the new meta fields (only cosmology meta) still
    loads through _read_store / GWCatalog -- absent meta are simply not read."""
    from gwcat.ingest import _read_store
    rng = np.random.default_rng(8)
    n = 20
    names = ["GWold_000001"]
    core = ["mass_1", "mass_2", "luminosity_distance", "ra", "dec", "chi_eff",
            "p_dL_pe"]
    cols = {p: rng.uniform(0.1, 1.0, n) for p in core}
    with h5py.File(tmp_path / "old.h5", "w") as f:
        f.attrs["param_names"] = np.array(core, dtype=h5py.string_dtype())
        idx = f.create_group("index")
        idx.create_dataset("offsets", data=np.array([0, n], dtype="i8"))
        idx.create_dataset("event_names",
                           data=np.array(names, dtype=h5py.string_dtype()))
        mg = f.create_group("meta")
        mg.create_dataset("dL_prior_H0", data=np.full(1, 67.74))
        mg.create_dataset("dL_prior_Om0", data=np.full(1, 0.3089))
        sg = f.create_group("samples")
        for p in core:
            sg.create_dataset(p, data=cols[p])

    S = _read_store(str(tmp_path / "old.h5"))
    # New spin meta fields are simply absent (not crashing, not fabricated).
    for k in ("spin_amax_1", "spin_prior_kind", "derived_params"):
        assert k not in S["meta"]
    cat = GWCatalog(str(tmp_path / "old.h5"))
    assert cat.n_events == 1
