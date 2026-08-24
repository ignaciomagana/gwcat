"""PR4: full spin ingestion + exact component-basis draw densities.

Fixtures follow the ``tests/test_selection_cbc.py`` synthetic-HDF5 style, but
use REAL random draws with analytically exact ``lnpdraw`` fields so the test can
recompute the expected component-basis draw density independently in numpy.

Covered:
  * Format B (O4 factored, ``write_o4_full``): recovered per-spin p(a,cosθ),
    amax auto-detection, uniform_isotropic, chi_p == file field == formula,
    corrupted azimuth check (warn records / raise raises), hand-computed
    component pdraw, legacy _pdraw unchanged vs the OLD formula.
  * Format A (endo3, ``write_endo3_full``): per-spin p(a,cosθ)=1/(2 max_spin),
    max_spin detection, factored-vs-joint consistency, hand-computed component
    pdraw, legacy _pdraw unchanged.
  * Format C mixtures (``write_mixture``): polar-flavour vs cartesian-flavour
    component pdraw agree to rtol=1e-12 (the key cross-check) and legacy _pdraw
    agree; module-level crosscheck_mixture_flavors.
  * Graceful component_spin_available=False / _ln_spin_component is None path.
"""
import warnings

import numpy as np
import h5py
import pytest

from gwcat.selection import SelectionSet, crosscheck_mixture_flavors
from gwcat.spin import chi_p_from_components
from gwcat import selection_spin as sspin

_YEAR_S = 365.25 * 24 * 3600
_LN_2PI = np.log(2.0 * np.pi)


# ======================================================================
# helpers to write an 'events' group with arbitrary columns
# ======================================================================
def _write_events_group(path, cols, attrs, searches):
    with h5py.File(path, "w") as f:
        for k, v in attrs.items():
            f.attrs[k] = v
        f.attrs["searches"] = np.array(searches, dtype=h5py.string_dtype())
        g = f.create_group("events")
        for name, arr in cols.items():
            g.create_dataset(name, data=np.asarray(arr))
    return str(path)


def _isotropic_cos_theta(rng, n):
    """cosθ ~ U(-1,1) (isotropic), so p(θ)=sinθ/2, p(azimuth)=1/2π."""
    return rng.uniform(-1.0, 1.0, n)


# ======================================================================
# Format B: full O4 factored fixture
# ======================================================================
def write_o4_full(path, n, amax=(0.9, 0.7), seed=0, detected=True,
                  total_generated=1000):
    rng = np.random.default_rng(seed)
    m1s = rng.uniform(30.0, 45.0, n)
    m2s = rng.uniform(8.0, 25.0, n)
    m1s, m2s = np.maximum(m1s, m2s), np.minimum(m1s, m2s)
    z = rng.uniform(0.05, 0.5, n)

    a1 = rng.uniform(0.0, amax[0], n)
    a2 = rng.uniform(0.0, amax[1], n)
    cost1 = _isotropic_cos_theta(rng, n)
    cost2 = _isotropic_cos_theta(rng, n)
    th1 = np.arccos(cost1)
    th2 = np.arccos(cost2)
    ph1 = rng.uniform(0.0, 2 * np.pi, n)
    ph2 = rng.uniform(0.0, 2 * np.pi, n)
    st1 = np.sqrt(1.0 - cost1 ** 2)
    st2 = np.sqrt(1.0 - cost2 ** 2)

    s1x, s1y, s1z = a1 * st1 * np.cos(ph1), a1 * st1 * np.sin(ph1), a1 * cost1
    s2x, s2y, s2z = a2 * st2 * np.cos(ph2), a2 * st2 * np.sin(ph2), a2 * cost2

    chieff = (m1s * s1z + m2s * s2z) / (m1s + m2s)
    chip = chi_p_from_components(a1, a2, cost1, cost2, m1s, m2s)

    # Exact factored draw log-densities.
    lnp_m1 = -np.log(15.0) * np.ones(n)          # m1 ~ U(30,45)
    lnp_m2 = -np.log(17.0) * np.ones(n)          # m2 ~ U(8,25) (indep here)
    lnp_z = -np.log(0.45) * np.ones(n)           # z ~ U(0.05,0.5)
    lnp_mag1 = -np.log(amax[0]) * np.ones(n)
    lnp_mag2 = -np.log(amax[1]) * np.ones(n)
    lnp_pol1 = np.log(st1 / 2.0)
    lnp_pol2 = np.log(st2 / 2.0)
    lnp_az1 = -_LN_2PI * np.ones(n)
    lnp_az2 = -_LN_2PI * np.ones(n)

    far = np.full(n, 0.1 if detected else 5.0)
    cols = {
        "mass1_source": m1s, "mass2_source": m2s,
        "mass1_detector": m1s * (1 + z), "mass2_detector": m2s * (1 + z),
        "luminosity_distance": rng.uniform(200.0, 900.0, n),
        "z": z, "dluminosity_distance_dredshift": rng.uniform(3000.0, 6000.0, n),
        "right_ascension": rng.uniform(0, 2 * np.pi, n),
        "declination": rng.uniform(-np.pi / 2, np.pi / 2, n),
        "spin1x": s1x, "spin1y": s1y, "spin1z": s1z,
        "spin2x": s2x, "spin2y": s2y, "spin2z": s2z,
        "spin1_magnitude": a1, "spin1_polar_angle": th1,
        "spin1_azimuthal_angle": ph1,
        "spin2_magnitude": a2, "spin2_polar_angle": th2,
        "spin2_azimuthal_angle": ph2,
        "chi_eff": chieff, "chi_p": chip, "weights": np.full(n, 2.0),
        "lnpdraw_mass1_source": lnp_m1,
        "lnpdraw_mass2_source_GIVEN_mass1_source": lnp_m2,
        "lnpdraw_z": lnp_z,
        "lnpdraw_spin1_magnitude": lnp_mag1,
        "lnpdraw_spin1_polar_angle": lnp_pol1,
        "lnpdraw_spin1_azimuthal_angle": lnp_az1,
        "lnpdraw_spin2_magnitude": lnp_mag2,
        "lnpdraw_spin2_polar_angle": lnp_pol2,
        "lnpdraw_spin2_azimuthal_angle": lnp_az2,
        "pycbc_far": far,
    }
    attrs = {"total_analysis_time": _YEAR_S, "total_generated": total_generated}
    return _write_events_group(path, cols, attrs, [b"pycbc"])


# ======================================================================
# Format A: full endo3 fixture (linear factored densities)
# ======================================================================
def write_endo3_full(path, n, max_spin=0.998, seed=1, detected=True,
                     total_generated=2000):
    rng = np.random.default_rng(seed)
    m1s = rng.uniform(30.0, 45.0, n)
    m2s = rng.uniform(8.0, 25.0, n)
    m1s, m2s = np.maximum(m1s, m2s), np.minimum(m1s, m2s)
    z = rng.uniform(0.05, 0.5, n)

    def _iso(amax):
        a = rng.uniform(0.0, amax, n)
        cost = _isotropic_cos_theta(rng, n)
        st = np.sqrt(1.0 - cost ** 2)
        ph = rng.uniform(0.0, 2 * np.pi, n)
        return a, cost, (a * st * np.cos(ph), a * st * np.sin(ph), a * cost)

    a1, cost1, (s1x, s1y, s1z) = _iso(max_spin)
    a2, cost2, (s2x, s2y, s2z) = _iso(max_spin)

    p_spin1 = 1.0 / (4.0 * np.pi * a1 ** 2 * max_spin)
    p_spin2 = 1.0 / (4.0 * np.pi * a2 ** 2 * max_spin)
    p_mass = rng.uniform(1e-4, 1e-2, n)
    p_z = rng.uniform(0.2, 0.8, n)
    sampling_pdf = p_mass * p_z * p_spin1 * p_spin2

    far = np.full(n, 0.1 if detected else 5.0)
    with h5py.File(path, "w") as f:
        f.attrs["total_generated"] = total_generated
        inj = f.create_group("injections")
        inj.attrs["analysis_time_s"] = _YEAR_S
        data = {
            "mass1_source": m1s, "mass2_source": m2s,
            "mass1": m1s * (1 + z), "mass2": m2s * (1 + z),
            "distance": rng.uniform(200.0, 900.0, n), "redshift": z,
            "right_ascension": rng.uniform(0, 2 * np.pi, n),
            "declination": rng.uniform(-np.pi / 2, np.pi / 2, n),
            "spin1x": s1x, "spin1y": s1y, "spin1z": s1z,
            "spin2x": s2x, "spin2y": s2y, "spin2z": s2z,
            "mass1_source_mass2_source_sampling_pdf": p_mass,
            "redshift_sampling_pdf": p_z,
            "spin1x_spin1y_spin1z_sampling_pdf": p_spin1,
            "spin2x_spin2y_spin2z_sampling_pdf": p_spin2,
            "sampling_pdf": sampling_pdf,
            "far_gstlal": far, "far_pycbc_bbh": far,
        }
        for k, v in data.items():
            inj.create_dataset(k, data=v)
    return str(path)


# ======================================================================
# Format C: two-component mixture, both flavours from the same draws
# ======================================================================
def _p_spin_cart(a, amax):
    """Cartesian isotropic uniform-magnitude spin marginal, 0 above amax."""
    return np.where(a < amax, 1.0 / (4.0 * np.pi * np.maximum(a, 1e-300) ** 2
                                     * amax), 0.0)


def write_mixture(path, flavor, seed=2, amax=(0.4, 0.998), n=64,
                  total_generated=5000):
    """Write a 50/50 two-component uniform-isotropic mixture.

    ``flavor`` is "polar" or "cartesian".  The SAME underlying draws are used,
    with the single joint lnpdraw key stored in the flavour's variables.  Mass
    and redshift factors are shared across components, so only the spin part is
    a mixture.
    """
    rng = np.random.default_rng(seed)
    m1s = rng.uniform(30.0, 45.0, n)
    m2s = rng.uniform(8.0, 25.0, n)
    m1s, m2s = np.maximum(m1s, m2s), np.minimum(m1s, m2s)
    z = rng.uniform(0.05, 0.5, n)
    p_mass = rng.uniform(1e-4, 1e-2, n)
    p_z = rng.uniform(0.2, 0.8, n)

    # Each injection is drawn from component 0 or 1 (50/50).
    comp = rng.integers(0, 2, n)
    amax_draw = np.where(comp == 0, amax[0], amax[1])

    def _draw():
        a = rng.uniform(0.0, 1.0, n) * amax_draw
        cost = _isotropic_cos_theta(rng, n)
        st = np.sqrt(1.0 - cost ** 2)
        ph = rng.uniform(0.0, 2 * np.pi, n)
        return a, cost, st, ph

    a1, cost1, st1, ph1 = _draw()
    a2, cost2, st2, ph2 = _draw()

    # Mixture density in CARTESIAN spin variables (a²-jacobian already NOT in).
    s_cart = 0.5 * (_p_spin_cart(a1, amax[0]) * _p_spin_cart(a2, amax[0])) \
        + 0.5 * (_p_spin_cart(a1, amax[1]) * _p_spin_cart(a2, amax[1]))
    lnp_joint_cart = np.log(np.maximum(p_mass * p_z * s_cart, 1e-300))
    # Polar joint = cartesian joint * a1² sinθ1 * a2² sinθ2 (coord Jacobian).
    lnp_joint_polar = (lnp_joint_cart
                       + 2 * np.log(a1) + np.log(st1)
                       + 2 * np.log(a2) + np.log(st2))

    th1, th2 = np.arccos(cost1), np.arccos(cost2)
    s1x, s1y, s1z = a1 * st1 * np.cos(ph1), a1 * st1 * np.sin(ph1), a1 * cost1
    s2x, s2y, s2z = a2 * st2 * np.cos(ph2), a2 * st2 * np.sin(ph2), a2 * cost2
    chieff = (m1s * s1z + m2s * s2z) / (m1s + m2s)

    far = np.full(n, 0.1)
    cols = {
        "mass1_source": m1s, "mass2_source": m2s,
        "mass1_detector": m1s * (1 + z), "mass2_detector": m2s * (1 + z),
        "luminosity_distance": rng.uniform(200.0, 900.0, n),
        "z": z, "dluminosity_distance_dredshift": rng.uniform(3000.0, 6000.0, n),
        "right_ascension": rng.uniform(0, 2 * np.pi, n),
        "declination": rng.uniform(-np.pi / 2, np.pi / 2, n),
        "chi_eff": chieff, "weights": np.full(n, 1.0),
        "pycbc_far": far,
    }
    joint_cart_key = ("lnpdraw_mass1_source_mass2_source_redshift_"
                      "spin1x_spin1y_spin1z_spin2x_spin2y_spin2z")
    joint_polar_key = ("lnpdraw_mass1_source_mass2_source_redshift_"
                       "spin1_magnitude_spin1_polar_angle_spin1_azimuthal_angle_"
                       "spin2_magnitude_spin2_polar_angle_spin2_azimuthal_angle")
    if flavor == "cartesian":
        cols.update({"spin1x": s1x, "spin1y": s1y, "spin1z": s1z,
                     "spin2x": s2x, "spin2y": s2y, "spin2z": s2z,
                     joint_cart_key: lnp_joint_cart})
    elif flavor == "polar":
        cols.update({"spin1_magnitude": a1, "spin1_polar_angle": th1,
                     "spin1_azimuthal_angle": ph1,
                     "spin2_magnitude": a2, "spin2_polar_angle": th2,
                     "spin2_azimuthal_angle": ph2,
                     joint_polar_key: lnp_joint_polar})
    else:
        raise ValueError(flavor)
    attrs = {"total_analysis_time": _YEAR_S, "total_generated": total_generated}
    return _write_events_group(path, cols, attrs, [b"pycbc"])


# ======================================================================
# Format B tests
# ======================================================================
def test_o4_full_spin_state(tmp_path):
    amax = (0.9, 0.7)
    path = write_o4_full(tmp_path / "o4.hdf", n=200, amax=amax, seed=3)
    sel = SelectionSet(path, strict_spin_checks="raise")
    sel._load()

    assert sel.spin_meta["spin_format"] == "o4_factored"
    assert sel.component_spin_available is True

    # amax auto-detection + uniform_isotropic.
    det1, det2 = sel.spin_meta["amax_detected"]
    assert det1 == pytest.approx(amax[0], rel=1e-12)
    assert det2 == pytest.approx(amax[1], rel=1e-12)
    assert sel.spin_meta["uniform_isotropic"] is True

    # Recovered per-spin p(a,cosθ) == 1/(2 amax): exp(ln_spin) == prod.
    expected_spin_factor = 1.0 / (2 * amax[0]) * 1.0 / (2 * amax[1])
    np.testing.assert_allclose(np.exp(sel._ln_spin_component),
                               expected_spin_factor, rtol=1e-12)


def test_o4_full_chi_p_matches_field_and_formula(tmp_path):
    path = write_o4_full(tmp_path / "o4.hdf", n=64, seed=4)
    sel = SelectionSet(path)
    sel._load()
    with h5py.File(path, "r") as f:
        chip_file = f["events"]["chi_p"][:]
    chip_formula = chi_p_from_components(sel.a1, sel.a2, sel.cost1, sel.cost2,
                                         sel._m1src, sel._m2src)
    np.testing.assert_allclose(sel.chi_p, chip_file, rtol=1e-12)
    np.testing.assert_allclose(sel.chi_p, chip_formula, rtol=1e-12)


def test_o4_full_legacy_pdraw_unchanged(tmp_path):
    """Existing _pdraw equals the OLD (pre-PR4) factored formula."""
    path = write_o4_full(tmp_path / "o4.hdf", n=50, seed=5)
    sel = SelectionSet(path)
    sel._load()
    with h5py.File(path, "r") as f:
        ev = f["events"]
        lnp = (ev["lnpdraw_mass1_source"][:]
               + ev["lnpdraw_mass2_source_GIVEN_mass1_source"][:]
               + ev["lnpdraw_z"][:])
        m1det = ev["mass1_detector"][:]
        z = ev["z"][:]
        ddL = ev["dluminosity_distance_dredshift"][:]
        w = ev["weights"][:]
    expected = np.exp(lnp) * m1det / (1 + z) ** 2 / ddL / 1.0 / w
    np.testing.assert_allclose(sel._pdraw, expected, rtol=1e-12)


def test_o4_full_hand_computed_component_pdraw(tmp_path):
    path = write_o4_full(tmp_path / "o4.hdf", n=40, amax=(0.8, 0.6), seed=6)
    sel = SelectionSet(path)
    sel._load()
    with h5py.File(path, "r") as f:
        ev = f["events"]
        lnp_mz = (ev["lnpdraw_mass1_source"][:]
                  + ev["lnpdraw_mass2_source_GIVEN_mass1_source"][:]
                  + ev["lnpdraw_z"][:])
        m1det, z = ev["mass1_detector"][:], ev["z"][:]
        ddL, w = ev["dluminosity_distance_dredshift"][:], ev["weights"][:]
        lnp_mag1, lnp_mag2 = (ev["lnpdraw_spin1_magnitude"][:],
                              ev["lnpdraw_spin2_magnitude"][:])
        lnp_pol1, lnp_pol2 = (ev["lnpdraw_spin1_polar_angle"][:],
                              ev["lnpdraw_spin2_polar_angle"][:])
        st1 = np.sin(ev["spin1_polar_angle"][:])
        st2 = np.sin(ev["spin2_polar_angle"][:])
    # ln p_comp over (m1s,m2s,z,a1,a2,cosθ1,cosθ2):
    ln_p_comp = (lnp_mz + lnp_mag1 + lnp_pol1 - np.log(st1)
                 + lnp_mag2 + lnp_pol2 - np.log(st2))
    expected = np.exp(ln_p_comp) * m1det / (1 + z) ** 2 / ddL / 1.0 / w
    np.testing.assert_allclose(sel.component_pdraw(), expected, rtol=1e-12)


def test_o4_full_corrupt_azimuth_warns(tmp_path):
    path = write_o4_full(tmp_path / "o4.hdf", n=32, seed=7)
    with h5py.File(path, "r+") as f:
        f["events"]["lnpdraw_spin1_azimuthal_angle"][:] += 0.5  # corrupt
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sel = SelectionSet(path, strict_spin_checks="warn")
        sel._load()
    assert any("azimuth" in str(w.message) for w in rec)
    assert sel.spin_meta["checks"]["azimuth_uniform"][0] is False


def test_o4_full_corrupt_azimuth_raises(tmp_path):
    path = write_o4_full(tmp_path / "o4.hdf", n=32, seed=8)
    with h5py.File(path, "r+") as f:
        f["events"]["lnpdraw_spin2_azimuthal_angle"][:] -= 0.3
    with pytest.raises(ValueError, match="azimuth"):
        SelectionSet(path, strict_spin_checks="raise")._load()


def test_o4_missing_spin_lnpdraw_graceful(tmp_path):
    """Factored file with cartesian spins but no spin lnpdraw columns:
    component unavailable but load succeeds and _a1 etc are still populated."""
    path = write_o4_full(tmp_path / "o4.hdf", n=16, seed=9)
    with h5py.File(path, "r+") as f:
        for k in ["lnpdraw_spin1_magnitude", "lnpdraw_spin1_polar_angle",
                  "lnpdraw_spin1_azimuthal_angle", "lnpdraw_spin2_magnitude",
                  "lnpdraw_spin2_polar_angle", "lnpdraw_spin2_azimuthal_angle"]:
            del f["events"][k]
    sel = SelectionSet(path)
    sel._load()
    assert sel.component_spin_available is False
    assert sel._ln_spin_component is None
    assert sel.a1 is not None and sel.chi_p is not None
    with pytest.raises(ValueError, match="unavailable"):
        sel.component_pdraw()


# ======================================================================
# Format A tests
# ======================================================================
def test_endo3_full_spin_state(tmp_path):
    max_spin = 0.998
    path = write_endo3_full(tmp_path / "endo3.hdf", n=200, max_spin=max_spin,
                            seed=10)
    sel = SelectionSet(path, strict_spin_checks="raise")
    sel._load()
    assert sel.spin_meta["spin_format"] == "endo3_factored"
    assert sel.component_spin_available is True
    d1, d2 = sel.spin_meta["amax_detected"]
    assert d1 == pytest.approx(max_spin, rel=1e-9)
    assert d2 == pytest.approx(max_spin, rel=1e-9)
    assert sel.spin_meta["uniform_isotropic"] is True
    # per-spin p(a,cosθ) == 1/(2 max_spin) -> product 1/(4 max_spin^2).
    np.testing.assert_allclose(np.exp(sel._ln_spin_component),
                               1.0 / (4 * max_spin ** 2), rtol=1e-9)


def test_endo3_full_hand_computed_component_pdraw(tmp_path):
    path = write_endo3_full(tmp_path / "endo3.hdf", n=40, max_spin=0.9, seed=11)
    sel = SelectionSet(path)
    sel._load()
    with h5py.File(path, "r") as f:
        inj = f["injections"]
        sampling_pdf = inj["sampling_pdf"][:]
        m1det, z = inj["mass1"][:], inj["redshift"][:]
        dL = inj["distance"][:]
        s1 = np.stack([inj["spin1x"][:], inj["spin1y"][:], inj["spin1z"][:]])
        s2 = np.stack([inj["spin2x"][:], inj["spin2y"][:], inj["spin2z"][:]])
    a1 = np.sqrt((s1 ** 2).sum(0))
    a2 = np.sqrt((s2 ** 2).sum(0))
    from gwcat.selection import _ddL_dz
    ddL = _ddL_dz(z, dL, sel.H0, sel.Om0)
    ln_p_comp = np.log(sampling_pdf) + 2 * _LN_2PI + 2 * np.log(a1) + 2 * np.log(a2)
    expected = np.exp(ln_p_comp) * m1det / (1 + z) ** 2 / ddL / 1.0
    np.testing.assert_allclose(sel.component_pdraw(), expected, rtol=1e-10)


def test_endo3_full_legacy_pdraw_unchanged(tmp_path):
    path = write_endo3_full(tmp_path / "endo3.hdf", n=30, seed=12)
    sel = SelectionSet(path)
    sel._load()
    with h5py.File(path, "r") as f:
        inj = f["injections"]
        p_mass = inj["mass1_source_mass2_source_sampling_pdf"][:]
        p_z = inj["redshift_sampling_pdf"][:]
        m1det, z, dL = inj["mass1"][:], inj["redshift"][:], inj["distance"][:]
    from gwcat.selection import _ddL_dz
    ddL = _ddL_dz(z, dL, sel.H0, sel.Om0)
    ln_no_spin = np.log(np.maximum(p_mass * p_z, 1e-300))
    expected = np.exp(ln_no_spin) * m1det / (1 + z) ** 2 / ddL / 1.0
    np.testing.assert_allclose(sel._pdraw, expected, rtol=1e-12)


def test_endo3_chi_p_formula_path(tmp_path):
    """Format A has no chi_p field -> chi_p comes from the Schmidt formula."""
    path = write_endo3_full(tmp_path / "endo3.hdf", n=32, seed=14)
    sel = SelectionSet(path)
    sel._load()
    expected = chi_p_from_components(sel.a1, sel.a2, sel.cost1, sel.cost2,
                                     sel._m1src, sel._m2src)
    np.testing.assert_allclose(sel.chi_p, expected, rtol=1e-12)
    assert np.all(np.isfinite(sel.chi_p))


def test_endo3_factored_vs_joint_consistency_recorded(tmp_path):
    path = write_endo3_full(tmp_path / "endo3.hdf", n=20, seed=13)
    sel = SelectionSet(path, strict_spin_checks="raise")
    sel._load()  # must not raise: exact product
    assert sel.spin_meta["checks"]["factored_vs_joint_dev"] < 1e-6


# ======================================================================
# Format C mixture tests
# ======================================================================
def test_mixture_polar_cartesian_component_pdraw_match(tmp_path):
    pol = write_mixture(tmp_path / "mix_polar.hdf", "polar", seed=20)
    car = write_mixture(tmp_path / "mix_cart.hdf", "cartesian", seed=20)
    sp = SelectionSet(pol)
    sc = SelectionSet(car)
    sp._load()
    sc._load()
    assert sp.spin_meta["spin_format"] == "joint_polar"
    assert sc.spin_meta["spin_format"] == "joint_cartesian"
    assert sp.spin_meta["amax_detected"] is None
    assert sp.spin_meta["uniform_isotropic"] is False

    # THE key cross-check: component pdraw agrees to rtol 1e-12.
    np.testing.assert_allclose(sp.component_pdraw(), sc.component_pdraw(),
                               rtol=1e-12)
    # legacy _pdraw also agrees between flavours.
    np.testing.assert_allclose(sp._pdraw, sc._pdraw, rtol=1e-12)


def test_mixture_chi_p_formula_path_matches_between_flavors(tmp_path):
    """Format C has no chi_p field: both flavours agree via the formula path."""
    pol = write_mixture(tmp_path / "mix_polar.hdf", "polar", seed=25)
    car = write_mixture(tmp_path / "mix_cart.hdf", "cartesian", seed=25)
    sp, sc = SelectionSet(pol), SelectionSet(car)
    sp._load()
    sc._load()
    exp = chi_p_from_components(sc.a1, sc.a2, sc.cost1, sc.cost2,
                               sc._m1src, sc._m2src)
    np.testing.assert_allclose(sc.chi_p, exp, rtol=1e-12)
    np.testing.assert_allclose(sp.chi_p, sc.chi_p, rtol=1e-12)


def test_crosscheck_mixture_flavors_ok(tmp_path):
    pol = write_mixture(tmp_path / "mix_polar.hdf", "polar", seed=21)
    car = write_mixture(tmp_path / "mix_cart.hdf", "cartesian", seed=21)
    report = crosscheck_mixture_flavors(pol, car, rtol=1e-9)
    assert report["component_pdraw_ok"] is True
    assert report["legacy_pdraw_ok"] is True
    assert report["component_pdraw_max_rel_dev"] < 1e-9


def test_crosscheck_mixture_flavors_detects_mismatch(tmp_path):
    pol = write_mixture(tmp_path / "mix_polar.hdf", "polar", seed=22)
    car = write_mixture(tmp_path / "mix_cart.hdf", "cartesian", seed=22)
    # Corrupt the cartesian joint so the component densities disagree.
    key = ("lnpdraw_mass1_source_mass2_source_redshift_"
           "spin1x_spin1y_spin1z_spin2x_spin2y_spin2z")
    with h5py.File(car, "r+") as f:
        f["events"][key][:] += 0.1
    with pytest.raises(ValueError, match="mismatch"):
        crosscheck_mixture_flavors(pol, car, rtol=1e-9)


def test_mixture_cartesian_hand_computed_component_pdraw(tmp_path):
    car = write_mixture(tmp_path / "mix_cart.hdf", "cartesian", seed=23)
    sel = SelectionSet(car)
    sel._load()
    key = ("lnpdraw_mass1_source_mass2_source_redshift_"
           "spin1x_spin1y_spin1z_spin2x_spin2y_spin2z")
    with h5py.File(car, "r") as f:
        ev = f["events"]
        lnp_joint = ev[key][:]
        m1det, z = ev["mass1_detector"][:], ev["z"][:]
        ddL, w = ev["dluminosity_distance_dredshift"][:], ev["weights"][:]
        a1 = np.sqrt(ev["spin1x"][:] ** 2 + ev["spin1y"][:] ** 2
                     + ev["spin1z"][:] ** 2)
        a2 = np.sqrt(ev["spin2x"][:] ** 2 + ev["spin2y"][:] ** 2
                     + ev["spin2z"][:] ** 2)
    ln_p_comp = lnp_joint + 2 * _LN_2PI + 2 * np.log(a1) + 2 * np.log(a2)
    expected = np.exp(ln_p_comp) * m1det / (1 + z) ** 2 / ddL / w
    np.testing.assert_allclose(sel.component_pdraw(), expected, rtol=1e-12)


def test_mixture_legacy_pdraw_unchanged_cartesian(tmp_path):
    """Legacy _pdraw on the enriched cartesian mixture equals the OLD formula."""
    car = write_mixture(tmp_path / "mix_cart.hdf", "cartesian", seed=24)
    sel = SelectionSet(car)
    sel._load()
    key = ("lnpdraw_mass1_source_mass2_source_redshift_"
           "spin1x_spin1y_spin1z_spin2x_spin2y_spin2z")
    with h5py.File(car, "r") as f:
        ev = f["events"]
        lnp_joint = ev[key][:]
        m1det, z = ev["mass1_detector"][:], ev["z"][:]
        ddL, w = ev["dluminosity_distance_dredshift"][:], ev["weights"][:]
        a1 = np.maximum(np.sqrt(ev["spin1x"][:] ** 2 + ev["spin1y"][:] ** 2
                                + ev["spin1z"][:] ** 2), 1e-30)
        a2 = np.maximum(np.sqrt(ev["spin2x"][:] ** 2 + ev["spin2y"][:] ** 2
                                + ev["spin2z"][:] ** 2), 1e-30)
    ln_no_spin = lnp_joint + np.log(16.0 * np.pi ** 2 * a1 ** 2 * a2 ** 2
                                    * 0.99 ** 2)
    expected = np.exp(ln_no_spin) * m1det / (1 + z) ** 2 / ddL / w
    np.testing.assert_allclose(sel._pdraw, expected, rtol=1e-12)


# ======================================================================
# strict_spin_checks argument validation + CombinedSelectionSet access
# ======================================================================
def test_invalid_strict_mode_raises(tmp_path):
    path = write_o4_full(tmp_path / "o4.hdf", n=4, seed=30)
    with pytest.raises(ValueError, match="strict_spin_checks"):
        SelectionSet(path, strict_spin_checks="bogus")


def test_combined_component_spin_arrays(tmp_path):
    from gwcat.selection import CombinedSelectionSet
    o3 = write_endo3_full(tmp_path / "endo3.hdf", n=20, seed=31)
    o4 = write_o4_full(tmp_path / "o4.hdf", n=15, seed=32)
    comb = CombinedSelectionSet([SelectionSet(o3), SelectionSet(o4)])
    assert comb.component_spin_available is True
    arrs = comb.component_spin_arrays(far_threshold=1.0)
    assert arrs["a1"].shape == arrs["ln_spin_component"].shape
    assert arrs["a1"].shape[0] == 35
    assert len(comb.spin_meta) == 2
    assert comb.spin_meta[0]["spin_format"] == "endo3_factored"
    assert comb.spin_meta[1]["spin_format"] == "o4_factored"


def test_events_file_without_sky_position_loads(tmp_path):
    """Semianalytic O1/O2 mixture rows carry no ra/dec: the reader NaN-fills
    and flags availability instead of failing (real files: mixture-semi_o1_o2-*)."""
    path = tmp_path / "nosky.hdf"
    write_o4_full(path, n=50, seed=3)
    import h5py
    with h5py.File(path, "r+") as f:
        ev = f["events"]
        if isinstance(ev, h5py.Dataset):
            keep = [n for n in ev.dtype.names
                    if n not in ("right_ascension", "declination")]
            sub = np.zeros(ev.shape, dtype=[(n, ev.dtype[n]) for n in keep])
            for n in keep:
                sub[n] = ev[n]
            del f["events"]
            f.create_dataset("events", data=sub)
        else:
            for n in ("right_ascension", "declination"):
                if n in ev:
                    del ev[n]
    s = SelectionSet(str(path))
    s._load()
    assert s._sky_position_available is False
    assert np.all(~np.isfinite(s._ra)) and np.all(~np.isfinite(s._dec))
    assert np.isfinite(s.component_pdraw()).all()


# ==========================================================================
# GW-05: the checks that matter are routed, and no amax is fabricated
# ==========================================================================
def test_non_uniform_magnitude_raises_under_strict(tmp_path):
    """A non-uniform spin-magnitude draw must fail strict_spin_checks="raise".

    It did not: `detect_uniform_amax_from_lnmag`'s verdict was stored in
    `checks` and never passed to `report_check`, so only the AZIMUTH checks
    could fire.  The real O4ab file fails both magnitude uniformity and
    isotropy and loaded silently -- and those two are precisely the assumptions
    the chieff / chieff_chip projections rest on, whereas azimuth uniformity
    marginalises out.
    """
    path = write_o4_full(tmp_path / "o4.hdf", n=64, seed=11)
    with h5py.File(path, "r+") as f:
        # make the magnitude density vary -> not a uniform draw
        n = f["events"]["lnpdraw_spin1_magnitude"].shape[0]
        f["events"]["lnpdraw_spin1_magnitude"][:] += np.linspace(0, 0.4, n)
    with pytest.raises(ValueError, match="magnitude_uniform"):
        SelectionSet(path, strict_spin_checks="raise")._load()


def test_non_isotropic_polar_raises_under_strict(tmp_path):
    path = write_o4_full(tmp_path / "o4.hdf", n=64, seed=12)
    with h5py.File(path, "r+") as f:
        f["events"]["lnpdraw_spin2_polar_angle"][:] += 0.2
    with pytest.raises(ValueError, match="isotropy_polar"):
        SelectionSet(path, strict_spin_checks="raise")._load()


def test_non_uniform_magnitude_warns_and_records_under_warn(tmp_path):
    path = write_o4_full(tmp_path / "o4.hdf", n=64, seed=13)
    with h5py.File(path, "r+") as f:
        n = f["events"]["lnpdraw_spin1_magnitude"].shape[0]
        f["events"]["lnpdraw_spin1_magnitude"][:] += np.linspace(0, 0.4, n)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sel = SelectionSet(path, strict_spin_checks="warn")
        sel._load()
    assert any("magnitude_uniform" in str(w.message) for w in rec)
    assert sel.spin_meta["checks"]["magnitude_uniform"][0] is False
    assert sel.spin_meta["uniform_isotropic"] is False


def test_no_amax_is_fabricated_for_a_non_uniform_draw(tmp_path):
    """`detect_uniform_amax_from_lnmag` returns None, not exp(-median(lnp)).

    For a varying log-magnitude-density that expression is a summary of a
    mixture, not a ceiling: on the real O4ab campaign it yields 0.7478/0.7397
    while the injected spins reach 0.99999, and `strict=False` fed it straight
    into the joint prior.
    """
    from gwcat.selection_spin import detect_uniform_amax_from_lnmag

    uniform = np.full(500, -np.log(0.998))
    amax, is_uni = detect_uniform_amax_from_lnmag(uniform)
    assert is_uni is True
    assert amax == pytest.approx(0.998)

    varying = uniform + np.linspace(0.0, 0.4, 500)
    amax, is_uni = detect_uniform_amax_from_lnmag(varying)
    assert is_uni is False
    assert amax is None, "a non-uniform draw must not yield a fabricated amax"


def test_chieff_chip_refuses_an_undetectable_amax(tmp_path):
    """With amax=None the joint-prior path must refuse, even at strict=False.

    It previously warned "proceeding with the detected amax if one is
    available" and then used the fabricated number.
    """
    from gwcat.export.selection_builder import (SpinBasisError,
                                                _campaign_chieff_chip_lnfactor)

    class _FakeSet:
        path = "fake_o4ab.hdf"
        spin_meta = {"uniform_isotropic": False, "amax_detected": (None, None)}

    with pytest.raises(SpinBasisError, match="undetectable"):
        _campaign_chieff_chip_lnfactor(_FakeSet(), slice(None), strict=False)
