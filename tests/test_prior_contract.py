"""Tests for the explicit spin-prior contract (PR 3).

Covers GWCatalog.to_darksirens spin_prior_mode {"include","exclude"} and the
prior-state provenance attributes written into both the PE export and the
selection export.

Contract under test (Mode A is the default):
  * include (default): the 1-D isotropic chi_eff prior is multiplied into p_pe
    exactly once.  Byte-identical to pre-PR output.
  * exclude: p_pe carries NO chi_eff prior factor.
  * invalid modes raise ValueError; "passthrough" is intentionally rejected.
  * output attrs record the contract on both PE and selection files, using a
    shared naming so the two can be cross-checked.

Fixtures are tiny synthetic HDF5 stores/injection files -- no network access.
"""
import numpy as np
import h5py
import pytest

from gwcat.catalog import GWCatalog
from gwcat.selection import SelectionSet
from gwcat.spin import chi_eff_prior_logprob


# --------------------------------------------------------------------------
# Tiny synthetic store (matches GWCatalog's on-disk schema)
# --------------------------------------------------------------------------
def _build_tiny_store(tmp_path, n_events=2, n_per_event=25, seed=1,
                      H0=67.74, Om0=0.3089, p_dL_const=None, name="store.h5"):
    """Write a minimal synthetic store.h5 for the darksirens exporter.

    If ``p_dL_const`` is given, every sample's ``p_dL_pe`` is set to that
    constant, which makes the (mass-Jacobian only) exclude-mode p_pe exactly
    ``m1det * p_dL_const`` and hence directly checkable.
    """
    rng = np.random.default_rng(seed)
    names = [f"GW999999_{i:06d}" for i in range(n_events)]
    param_list = ["mass_1", "mass_2", "luminosity_distance", "ra", "dec",
                  "chi_eff", "p_dL_pe"]

    per_event = {p: [] for p in param_list}
    offsets = [0]
    for _ in range(n_events):
        n = n_per_event
        per_event["mass_1"].append(rng.uniform(25, 50, n))
        per_event["mass_2"].append(rng.uniform(10, 25, n))
        per_event["luminosity_distance"].append(rng.uniform(300, 800, n))
        per_event["ra"].append(rng.uniform(0, 2 * np.pi, n))
        per_event["dec"].append(rng.uniform(-np.pi / 2, np.pi / 2, n))
        per_event["chi_eff"].append(rng.uniform(-0.4, 0.4, n))
        if p_dL_const is None:
            per_event["p_dL_pe"].append(rng.uniform(0.1, 1.0, n))
        else:
            per_event["p_dL_pe"].append(np.full(n, float(p_dL_const)))
        offsets.append(offsets[-1] + n)

    path = tmp_path / name
    with h5py.File(path, "w") as f:
        f.attrs["param_names"] = np.array(param_list, dtype=h5py.string_dtype())
        idx = f.create_group("index")
        idx.create_dataset("offsets", data=np.array(offsets, dtype="i8"))
        idx.create_dataset("event_names",
                           data=np.array(names, dtype=h5py.string_dtype()))
        meta = f.create_group("meta")
        meta.create_dataset("dL_prior_H0", data=np.full(n_events, H0))
        meta.create_dataset("dL_prior_Om0", data=np.full(n_events, Om0))
        samp = f.create_group("samples")
        for p in param_list:
            samp.create_dataset(p, data=np.concatenate(per_event[p]))
    return str(path)


_COSMO = (67.74, 0.3089)


# --------------------------------------------------------------------------
# 1. include mode: chi_eff prior applied exactly once
# --------------------------------------------------------------------------
def test_include_applies_chi_eff_prior_exactly_once(tmp_path):
    """p_pe(include) / p_pe(exclude) must equal the 1-D chi_eff prior evaluated
    at the exported samples -- i.e. exactly one chi_eff prior factor."""
    store = _build_tiny_store(tmp_path, seed=7)
    cat = GWCatalog(store)

    out_inc = tmp_path / "inc.h5"
    out_exc = tmp_path / "exc.h5"
    kw = dict(nsamp=12, seed=0, cosmology=_COSMO, amax=0.99)
    cat.to_darksirens(str(out_inc), spin_prior_mode="include", **kw)
    cat.to_darksirens(str(out_exc), spin_prior_mode="exclude", **kw)

    with h5py.File(out_inc, "r") as fi, h5py.File(out_exc, "r") as fe:
        # Same seed -> identical resampled samples in both files.
        for key in ["m1det", "m2det", "dL", "chieff", "m1src", "m2src", "ra", "dec"]:
            np.testing.assert_array_equal(fi[key][:], fe[key][:])

        p_inc = fi["p_pe"][:]
        p_exc = fe["p_pe"][:]
        chieff = fi["chieff"][:]
        m1src = fi["m1src"][:]
        m2src = fi["m2src"][:]

    logp = chi_eff_prior_logprob(chieff, m1src, m2src, amax=0.99)
    expected_factor = np.exp(np.clip(logp, -50.0, None))

    assert p_exc.size > 0
    assert np.all(p_exc > 0)
    ratio = p_inc / p_exc
    np.testing.assert_allclose(ratio, expected_factor, rtol=1e-12, atol=0)
    # Non-trivial: the chi_eff factor genuinely varies (not identically 1).
    assert np.ptp(expected_factor) > 1e-6


# --------------------------------------------------------------------------
# 2. exclude mode: p_pe carries no chi_eff prior factor
# --------------------------------------------------------------------------
def test_exclude_has_no_chi_eff_factor(tmp_path):
    """With a constant distance prior, exclude-mode p_pe must equal the pure
    mass Jacobian m1det * p_dL_const -- no chi_eff dependence at all."""
    c = 0.5
    store = _build_tiny_store(tmp_path, seed=11, p_dL_const=c)
    cat = GWCatalog(store)

    out_exc = tmp_path / "exc.h5"
    cat.to_darksirens(str(out_exc), spin_prior_mode="exclude",
                      nsamp=15, seed=0, cosmology=_COSMO)

    with h5py.File(out_exc, "r") as f:
        p_exc = f["p_pe"][:]
        m1det = f["m1det"][:]
    np.testing.assert_allclose(p_exc, m1det * c, rtol=1e-12, atol=0)


# --------------------------------------------------------------------------
# 3. default == include, and reproduces the pre-PR Mode A formula
# --------------------------------------------------------------------------
def test_default_matches_include_byte_for_byte(tmp_path):
    """Calling to_darksirens with no spin_prior_mode arg must be byte-identical
    to spin_prior_mode='include' (the pre-PR default)."""
    store = _build_tiny_store(tmp_path, seed=3)
    cat = GWCatalog(store)

    out_default = tmp_path / "default.h5"
    out_include = tmp_path / "include.h5"
    kw = dict(nsamp=10, seed=0, cosmology=_COSMO)
    cat.to_darksirens(str(out_default), **kw)                       # default
    cat.to_darksirens(str(out_include), spin_prior_mode="include", **kw)

    with h5py.File(out_default, "r") as fd, h5py.File(out_include, "r") as fi:
        for key in ["ra", "dec", "m1det", "m2det", "chieff", "dL", "p_pe",
                    "redshift", "m1src", "m2src"]:
            np.testing.assert_array_equal(fd[key][:], fi[key][:])
        assert fd.attrs["chi_eff_in_p_pe"] == fi.attrs["chi_eff_in_p_pe"] == True


def test_default_reproduces_mode_a_formula(tmp_path):
    """Independent reconstruction: default p_pe == m1det * p_dL_const * chi_eff
    prior factor.  Guards the Mode A math against silent regression."""
    c = 0.7
    store = _build_tiny_store(tmp_path, seed=21, p_dL_const=c)
    cat = GWCatalog(store)

    out = tmp_path / "default.h5"
    cat.to_darksirens(str(out), nsamp=15, seed=0, cosmology=_COSMO, amax=0.99)

    with h5py.File(out, "r") as f:
        p_pe = f["p_pe"][:]
        m1det = f["m1det"][:]
        chieff = f["chieff"][:]
        m1src = f["m1src"][:]
        m2src = f["m2src"][:]

    factor = np.exp(np.clip(
        chi_eff_prior_logprob(chieff, m1src, m2src, amax=0.99), -50.0, None))
    np.testing.assert_allclose(p_pe, m1det * c * factor, rtol=1e-12, atol=0)


# --------------------------------------------------------------------------
# 4. output attrs record the mode correctly
# --------------------------------------------------------------------------
def test_pe_attrs_record_mode_include(tmp_path):
    store = _build_tiny_store(tmp_path, seed=5)
    cat = GWCatalog(store)
    out = tmp_path / "inc.h5"
    cat.to_darksirens(str(out), spin_prior_mode="include", nsamp=8, seed=0,
                      cosmology=_COSMO)
    with h5py.File(out, "r") as f:
        assert f.attrs["spin_prior_mode"] == "include"
        assert bool(f.attrs["chi_eff_prior_applied_to_p_pe"]) is True
        assert bool(f.attrs["chi_eff_in_p_pe"]) is True          # legacy
        assert bool(f.attrs["mass_jacobian_applied"]) is True
        assert bool(f.attrs["distance_prior_removed"]) is False
        assert bool(f.attrs["cosmology_override_used"]) is True  # cosmology given


def test_pe_attrs_record_mode_exclude(tmp_path):
    store = _build_tiny_store(tmp_path, seed=6)
    cat = GWCatalog(store)
    out = tmp_path / "exc.h5"
    cat.to_darksirens(str(out), spin_prior_mode="exclude", nsamp=8, seed=0)
    with h5py.File(out, "r") as f:
        assert f.attrs["spin_prior_mode"] == "exclude"
        assert bool(f.attrs["chi_eff_prior_applied_to_p_pe"]) is False
        assert bool(f.attrs["chi_eff_in_p_pe"]) is False         # legacy, consistent
        assert bool(f.attrs["mass_jacobian_applied"]) is True
        # cosmology=None default -> no override used
        assert bool(f.attrs["cosmology_override_used"]) is False


# --------------------------------------------------------------------------
# 5. invalid mode raises (including the rejected "passthrough")
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["passthrough", "PASSTHROUGH", "keep", "", "on"])
def test_invalid_spin_prior_mode_raises(tmp_path, bad):
    store = _build_tiny_store(tmp_path, seed=8)
    cat = GWCatalog(store)
    out = tmp_path / "bad.h5"
    with pytest.raises(ValueError, match="spin_prior_mode"):
        cat.to_darksirens(str(out), spin_prior_mode=bad, nsamp=5, seed=0,
                          cosmology=_COSMO)
    assert not out.exists()


def test_passthrough_error_message_explains_omission(tmp_path):
    store = _build_tiny_store(tmp_path, seed=9)
    cat = GWCatalog(store)
    with pytest.raises(ValueError, match="passthrough"):
        cat.to_darksirens(str(tmp_path / "x.h5"), spin_prior_mode="passthrough")


# --------------------------------------------------------------------------
# 6. selection export records the aligned contract; PE and selection agree
# --------------------------------------------------------------------------
def _base_o4_columns(n=3):
    fields = [
        ("mass1_source", "f8"), ("mass2_source", "f8"),
        ("mass1_detector", "f8"), ("mass2_detector", "f8"),
        ("luminosity_distance", "f8"), ("z", "f8"),
        ("dluminosity_distance_dredshift", "f8"),
        ("right_ascension", "f8"), ("declination", "f8"),
        ("spin1x", "f8"), ("spin1y", "f8"), ("spin1z", "f8"),
        ("spin2x", "f8"), ("spin2y", "f8"), ("spin2z", "f8"),
        ("chi_eff", "f8"), ("weights", "f8"),
        ("lnpdraw_mass1_source", "f8"),
        ("lnpdraw_mass2_source_GIVEN_mass1_source", "f8"),
        ("lnpdraw_z", "f8"),
        ("pycbc_far", "f8"), ("cwb-bbh_far", "f8"),
    ]
    ev = np.zeros(n, dtype=fields)
    ev["mass1_source"] = 30.0
    ev["mass2_source"] = 20.0
    ev["z"] = 0.1
    ev["mass1_detector"] = 33.0
    ev["mass2_detector"] = 22.0
    ev["luminosity_distance"] = 450.0
    ev["dluminosity_distance_dredshift"] = 4500.0
    ev["right_ascension"] = 1.0
    ev["declination"] = 0.5
    ev["spin1z"] = 0.2
    ev["spin2z"] = -0.1
    ev["chi_eff"] = 0.08
    ev["weights"] = 2.0
    ev["lnpdraw_mass1_source"] = -1.0
    ev["lnpdraw_mass2_source_GIVEN_mass1_source"] = -2.0
    ev["lnpdraw_z"] = -3.0
    ev["pycbc_far"] = [0.5, 2.0, 0.2]
    ev["cwb-bbh_far"] = [2.0, 2.0, 2.0]
    return ev


def _write_o4_injection(path):
    with h5py.File(path, "w") as f:
        f.attrs["total_analysis_time"] = 365.25 * 24 * 3600
        f.attrs["total_generated"] = 100
        f.attrs["searches"] = np.array([b"pycbc", b"cwb-bbh"])
        f.create_dataset("events", data=_base_o4_columns())


def test_selection_export_records_aligned_contract(tmp_path):
    inj = tmp_path / "o4.hdf"
    _write_o4_injection(inj)
    out = tmp_path / "sel.h5"
    SelectionSet(str(inj)).to_darksirens(str(out), far_threshold=1.0)

    with h5py.File(out, "r") as f:
        assert f.attrs["spin_prior_mode"] == "include"
        assert bool(f.attrs["chi_eff_prior_applied_to_pdraw"]) is True
        assert bool(f.attrs["chi_eff_swap_applied"]) is True     # legacy
        assert bool(f.attrs["mass_jacobian_applied"]) is True
        assert bool(f.attrs["distance_prior_removed"]) is False
        # default cosmology -> no override
        assert bool(f.attrs["cosmology_override_used"]) is False


def test_selection_cosmology_override_recorded(tmp_path):
    inj = tmp_path / "o4.hdf"
    _write_o4_injection(inj)
    out = tmp_path / "sel.h5"
    SelectionSet(str(inj), H0=70.0, Om0=0.3).to_darksirens(str(out),
                                                           far_threshold=1.0)
    with h5py.File(out, "r") as f:
        assert bool(f.attrs["cosmology_override_used"]) is True


def test_pe_and_selection_agree_on_contract(tmp_path):
    """A downstream consumer can verify the PE and selection files agree on the
    spin-prior contract via the shared attribute naming."""
    # PE export (default include mode)
    store = _build_tiny_store(tmp_path, seed=31)
    pe_out = tmp_path / "pe.h5"
    GWCatalog(store).to_darksirens(str(pe_out), nsamp=8, seed=0, cosmology=_COSMO)

    # Selection export
    inj = tmp_path / "o4.hdf"
    _write_o4_injection(inj)
    sel_out = tmp_path / "sel.h5"
    SelectionSet(str(inj)).to_darksirens(str(sel_out), far_threshold=1.0)

    with h5py.File(pe_out, "r") as fp, h5py.File(sel_out, "r") as fs:
        assert fp.attrs["spin_prior_mode"] == fs.attrs["spin_prior_mode"]
        assert (bool(fp.attrs["chi_eff_prior_applied_to_p_pe"])
                == bool(fs.attrs["chi_eff_prior_applied_to_pdraw"]))
