"""Tests for the per-event cosmology contract (PR 4).

Covers GWCatalog.to_darksirens cosmology handling and the dL->z inversion:

  * cosmology=None ("per-event"): each event's z / source-frame quantities are
    computed under ITS OWN stored PE cosmology, not the first event's.  This is
    the core correctness test for mixed-release selections.
  * a single-cosmology selection is byte-identical between per-event mode and an
    explicit override with the same (H0, Om0) -- the fix does not perturb the
    common case (regression guard against the pre-fix output).
  * cosmology=(H0, Om0) ("override"): the override is applied to all events,
    cosmology_override_used=True, and the H0/Om0 are recorded in output attrs.
  * a missing per-event cosmology (NaN) fails loudly and names the event unless
    an explicit override is supplied.
  * z_of_dL does not silently clip samples beyond its interpolation range
    (handoff high-priority test case 13).

Fixtures are tiny synthetic HDF5 stores -- no network access.
"""
import warnings

import numpy as np
import h5py
import pytest

from gwcat.catalog import GWCatalog
from gwcat.cosmology import make_cosmology, z_of_dL


# --------------------------------------------------------------------------
# Tiny synthetic store with configurable per-event cosmology
# --------------------------------------------------------------------------
def _build_store(tmp_path, H0, Om0, n_events=2, n_per_event=40, seed=1,
                 name="store.h5"):
    """Write a minimal synthetic store.h5 for the darksirens exporter.

    ``H0`` / ``Om0`` may be scalars (shared by all events) or length-``n_events``
    sequences (one PE cosmology per event).  A per-event value of ``np.nan``
    models a missing stored cosmology.
    """
    H0 = np.broadcast_to(np.asarray(H0, dtype=float), (n_events,)).copy()
    Om0 = np.broadcast_to(np.asarray(Om0, dtype=float), (n_events,)).copy()

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
        per_event["p_dL_pe"].append(rng.uniform(0.1, 1.0, n))
        offsets.append(offsets[-1] + n)

    path = tmp_path / name
    with h5py.File(path, "w") as f:
        f.attrs["param_names"] = np.array(param_list, dtype=h5py.string_dtype())
        idx = f.create_group("index")
        idx.create_dataset("offsets", data=np.array(offsets, dtype="i8"))
        idx.create_dataset("event_names",
                           data=np.array(names, dtype=h5py.string_dtype()))
        meta = f.create_group("meta")
        meta.create_dataset("dL_prior_H0", data=H0)
        meta.create_dataset("dL_prior_Om0", data=Om0)
        samp = f.create_group("samples")
        for p in param_list:
            samp.create_dataset(p, data=np.concatenate(per_event[p]))
    return str(path)


def _event_slices(f):
    """Return (name, slice) pairs for each event in a darksirens export."""
    nsamp = int(f.attrs["nsamp"])
    names = [n.decode() if isinstance(n, bytes) else n
             for n in f.attrs["event_names"]]
    return [(nm, slice(i * nsamp, (i + 1) * nsamp))
            for i, nm in enumerate(names)]


# --------------------------------------------------------------------------
# 1. CORE: per-event cosmology is respected for a mixed-cosmology selection
# --------------------------------------------------------------------------
def test_per_event_cosmology_respected_for_mixed_selection(tmp_path):
    """Two events with DIFFERENT stored PE cosmologies.  Each event's exported
    redshift / source masses must be computed under ITS OWN cosmology -- and
    demonstrably NOT under the other event's cosmology (the pre-fix bug)."""
    H0 = [70.0, 50.0]
    Om0 = [0.30, 0.20]
    store = _build_store(tmp_path, H0=H0, Om0=Om0, seed=4)
    cat = GWCatalog(store)

    out = tmp_path / "mixed.h5"
    cat.to_darksirens(str(out), nsamp=20, seed=0)  # cosmology=None -> per-event

    cosmos = {0: make_cosmology(70.0, 0.30), 1: make_cosmology(50.0, 0.20)}
    with h5py.File(out, "r") as f:
        assert f.attrs["cosmology_mode"] == "per-event"
        assert bool(f.attrs["cosmology_per_event_varies"]) is True
        np.testing.assert_array_equal(f.attrs["cosmology_H0_per_event"],
                                      np.array([70.0, 50.0]))
        np.testing.assert_array_equal(f.attrs["cosmology_Om0_per_event"],
                                      np.array([0.30, 0.20]))

        for i, (_name, sl) in enumerate(_event_slices(f)):
            dL = f["dL"][sl]
            z = f["redshift"][sl]
            m1det = f["m1det"][sl]
            m2det = f["m2det"][sl]

            # Exported z matches THIS event's own cosmology, exactly.
            z_own = z_of_dL(dL, cosmos[i])
            np.testing.assert_allclose(z, z_own, rtol=1e-12, atol=0)
            # Source masses are consistent with the exported z.
            np.testing.assert_allclose(f["m1src"][sl], m1det / (1 + z),
                                       rtol=1e-12, atol=0)
            np.testing.assert_allclose(f["m2src"][sl], m2det / (1 + z),
                                       rtol=1e-12, atol=0)

            # And it is NOT the other event's cosmology (bug would make them
            # identical because the first event's cosmology was used for all).
            other = 1 - i
            z_wrong = z_of_dL(dL, cosmos[other])
            assert np.max(np.abs(z - z_wrong)) > 1e-3


def test_per_event_matches_independent_single_event_exports(tmp_path):
    """Each event's z(dL) relation in a mixed two-event per-event export must
    match an independent single-event store exported with only that event's
    cosmology."""
    store2 = _build_store(tmp_path, H0=[70.0, 55.0], Om0=[0.30, 0.25],
                          seed=9, name="two.h5")
    GWCatalog(store2).to_darksirens(str(tmp_path / "two.out.h5"),
                                    nsamp=20, seed=0)

    single = {
        0: _build_store(tmp_path, H0=70.0, Om0=0.30, n_events=1, seed=9,
                        name="s0.h5"),
        1: _build_store(tmp_path, H0=55.0, Om0=0.25, n_events=1, seed=9,
                        name="s1.h5"),
    }

    with h5py.File(tmp_path / "two.out.h5", "r") as f2:
        two_slices = _event_slices(f2)
        for i, (_nm, sl) in enumerate(two_slices):
            cosmo_i = make_cosmology(*[[70.0, 0.30], [55.0, 0.25]][i])
            # z(dL) relation from the combined per-event export.
            z_two = f2["redshift"][sl]
            dL_two = f2["dL"][sl]
            np.testing.assert_allclose(z_two, z_of_dL(dL_two, cosmo_i),
                                       rtol=1e-12, atol=0)
        # Independent single-event export uses the same cosmology curve.
        for i in (0, 1):
            out_i = tmp_path / f"single{i}.out.h5"
            GWCatalog(single[i]).to_darksirens(str(out_i), nsamp=20, seed=0)
            cosmo_i = make_cosmology(*[[70.0, 0.30], [55.0, 0.25]][i])
            with h5py.File(out_i, "r") as fi:
                np.testing.assert_allclose(
                    fi["redshift"][:], z_of_dL(fi["dL"][:], cosmo_i),
                    rtol=1e-12, atol=0)


# --------------------------------------------------------------------------
# 2. REGRESSION: single-cosmology output unchanged (per-event == override)
# --------------------------------------------------------------------------
def test_single_cosmology_per_event_equals_override(tmp_path):
    """For a selection whose events share one cosmology, per-event mode
    (cosmology=None) must be byte-identical to an explicit override with that
    same cosmology -- i.e. the fix does not perturb the common case."""
    store = _build_store(tmp_path, H0=67.74, Om0=0.3089, seed=2)
    cat = GWCatalog(store)

    out_pe = tmp_path / "per_event.h5"
    out_ov = tmp_path / "override.h5"
    cat.to_darksirens(str(out_pe), nsamp=16, seed=0)                     # None
    cat.to_darksirens(str(out_ov), nsamp=16, seed=0,
                      cosmology=(67.74, 0.3089))                         # override

    with h5py.File(out_pe, "r") as fp, h5py.File(out_ov, "r") as fo:
        for key in ["ra", "dec", "m1det", "m2det", "chieff", "dL", "p_pe",
                    "redshift", "m1src", "m2src"]:
            np.testing.assert_array_equal(fp[key][:], fo[key][:])
        # Same numerics, different provenance labelling.
        assert fp.attrs["cosmology_mode"] == "per-event"
        assert fo.attrs["cosmology_mode"] == "override"
        assert bool(fp.attrs["cosmology_per_event_varies"]) is False


# --------------------------------------------------------------------------
# 3. OVERRIDE: single cosmology applied to all events, recorded in attrs
# --------------------------------------------------------------------------
def test_override_applies_to_all_events_and_records_attrs(tmp_path):
    """An explicit override must be applied to every event (even a store with
    differing per-event cosmologies) and recorded in the output attrs."""
    store = _build_store(tmp_path, H0=[70.0, 50.0], Om0=[0.30, 0.20], seed=5)
    cat = GWCatalog(store)

    out = tmp_path / "override.h5"
    cat.to_darksirens(str(out), nsamp=18, seed=0, cosmology=(72.0, 0.28))

    cosmo = make_cosmology(72.0, 0.28)
    with h5py.File(out, "r") as f:
        assert bool(f.attrs["cosmology_override_used"]) is True
        assert f.attrs["cosmology_mode"] == "override"
        assert f.attrs["pe_cosmology_H0"] == pytest.approx(72.0)
        assert f.attrs["pe_cosmology_Om0"] == pytest.approx(0.28)
        assert bool(f.attrs["cosmology_per_event_varies"]) is False
        np.testing.assert_array_equal(f.attrs["cosmology_H0_per_event"],
                                      np.array([72.0, 72.0]))
        # Every event's z uses the single override cosmology.
        for _nm, sl in _event_slices(f):
            np.testing.assert_allclose(
                f["redshift"][sl], z_of_dL(f["dL"][sl], cosmo),
                rtol=1e-12, atol=0)


# --------------------------------------------------------------------------
# 4. MISSING per-event cosmology -> loud error; override rescues it
# --------------------------------------------------------------------------
def test_missing_per_event_cosmology_raises_and_names_event(tmp_path):
    """A NaN stored cosmology for one event must fail loudly under
    cosmology=None and name the offending event."""
    store = _build_store(tmp_path, H0=[70.0, np.nan], Om0=[0.30, 0.30], seed=6)
    cat = GWCatalog(store)
    out = tmp_path / "missing.h5"

    with pytest.raises(ValueError, match="GW999999_000001"):
        cat.to_darksirens(str(out), nsamp=10, seed=0)
    assert not out.exists()


def test_missing_per_event_cosmology_ok_with_override(tmp_path):
    """The same store exports fine when the user supplies an override."""
    store = _build_store(tmp_path, H0=[70.0, np.nan], Om0=[0.30, np.nan],
                         seed=6)
    cat = GWCatalog(store)
    out = tmp_path / "rescued.h5"

    cat.to_darksirens(str(out), nsamp=10, seed=0, cosmology=(70.0, 0.30))
    with h5py.File(out, "r") as f:
        assert f.attrs["nobs"] == 2
        assert f.attrs["cosmology_mode"] == "override"


def test_absent_cosmology_columns_raise_without_override(tmp_path):
    """A store lacking the dL_prior_H0/Om0 columns entirely must fail loudly
    under cosmology=None but succeed with an override."""
    # Build a normal store, then strip the cosmology columns.
    store = _build_store(tmp_path, H0=70.0, Om0=0.30, seed=7)
    with h5py.File(store, "r+") as f:
        del f["meta/dL_prior_H0"]
        del f["meta/dL_prior_Om0"]

    cat = GWCatalog(store)
    with pytest.raises(ValueError, match="dL_prior_H0"):
        cat.to_darksirens(str(tmp_path / "no_cols.h5"), nsamp=10, seed=0)

    out = tmp_path / "with_override.h5"
    cat.to_darksirens(str(out), nsamp=10, seed=0, cosmology=(70.0, 0.30))
    assert out.exists()


# --------------------------------------------------------------------------
# 5. z_of_dL does NOT silently clip beyond its interpolation range (case 13)
# --------------------------------------------------------------------------
def test_z_of_dL_does_not_silently_clip_out_of_range():
    """A dL beyond dL(z=zmax) must NOT be mapped to exactly zmax.  The grid is
    extended (with a warning) so the returned z inverts back to the input dL."""
    cosmo = make_cosmology(70.0, 0.3)
    zmax = 10.0
    dL_at_zmax = cosmo.luminosity_distance(zmax).to("Mpc").value

    # A distance well beyond the default z=10 grid.
    dL_far = cosmo.luminosity_distance(25.0).to("Mpc").value

    with pytest.warns(UserWarning, match="beyond"):
        z = z_of_dL(np.array([dL_far]), cosmo, zmax=zmax)[0]

    # Not clipped to zmax ...
    assert z > zmax + 0.5
    # ... and it actually inverts the input distance.
    assert cosmo.luminosity_distance(z).to("Mpc").value == pytest.approx(
        dL_far, rel=1e-3)
    # Sanity: without the fix np.interp would return exactly zmax here.
    assert abs(z - zmax) > 1.0
    # In-range values are unaffected (no warning path changes them).
    z_in = z_of_dL(np.array([dL_at_zmax * 0.5]), cosmo, zmax=zmax)
    assert np.all(np.isfinite(z_in)) and np.all(z_in < zmax)


def test_z_of_dL_in_range_is_unchanged_and_silent():
    """In-range inputs must not warn and must match the plain interpolation."""
    cosmo = make_cosmology(67.74, 0.3089)
    dL = np.linspace(100.0, 5000.0, 50)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning -> test failure
        z = z_of_dL(dL, cosmo)
    # Monotone, finite, and inverts correctly.
    assert np.all(np.isfinite(z))
    np.testing.assert_allclose(
        cosmo.luminosity_distance(z).to("Mpc").value, dL, rtol=1e-3)


# --------------------------------------------------------------------------
# 6. GW-01: the distance prior is NOT truncated at its recorded bounds
# --------------------------------------------------------------------------
from gwcat.cosmology import (  # noqa: E402  (grouped with the GW-01 tests)
    DistancePriorImplError, resolve_usf_impl, uniform_source_frame_prob,
    _usf_grid, _usf_z_grid, _USF_NEAR_ZERO_STEP, _USF_NEAR_ZERO_ZMIN,
)


def test_usf_prob_unnormalised_outside_bounds():
    """Samples outside the recorded [dmin, dmax] must get a finite, positive,
    continuous density -- not zero.

    The recorded bounds routinely come from a sibling analysis's analytic prior,
    so posterior samples legitimately fall outside them; zeroing the density
    there hands darksirens a -inf log-weight that still counts in n.
    """
    cosmo = make_cosmology(67.9, 0.3065)
    dmin, dmax = 100.0, 500.0
    dL = np.array([40.0, 100.0, 300.0, 500.0, 900.0])

    p, info = uniform_source_frame_prob(dL, cosmo, dmin, dmax,
                                        impl="astropy", return_info=True)

    assert np.all(np.isfinite(p))
    assert np.all(p > 0.0), f"truncated density: {p}"
    # p(dL) still rises with distance across the bounds (comoving-volume shape),
    # i.e. the out-of-bounds samples are on the physical curve, not a floor.
    assert np.all(np.diff(p) > 0)

    assert info["n_below_dmin"] == 1
    assert info["n_above_dmax"] == 1
    assert info["n_outside_bounds"] == 2
    assert info["frac_outside_bounds"] == pytest.approx(2 / 5)
    assert info["widened"] is True
    assert info["eval_min"] < 40.0 and info["eval_max"] > 900.0
    # The recorded bounds survive as provenance.
    assert info["dmin"] == dmin and info["dmax"] == dmax


@pytest.mark.parametrize("bound", [100.0, 500.0])
def test_usf_prob_is_continuous_across_the_recorded_bounds(bound):
    """No step at dmin/dmax -- the old np.where truncation put one there."""
    cosmo = make_cosmology(67.9, 0.3065)
    eps = 1e-3
    p = uniform_source_frame_prob(np.array([bound - eps, bound + eps]),
                                  cosmo, 100.0, 500.0, impl="astropy")
    assert np.all(p > 0)
    assert p[1] / p[0] == pytest.approx(1.0, rel=1e-3)


def test_usf_prob_in_bounds_does_not_widen():
    """When every sample is inside the recorded bounds, the evaluation range is
    exactly [dmin, dmax] -- the common case is not perturbed."""
    cosmo = make_cosmology(67.74, 0.3089)
    dL = np.linspace(150.0, 450.0, 40)
    p, info = uniform_source_frame_prob(dL, cosmo, 100.0, 500.0,
                                        impl="astropy", return_info=True)
    assert info["widened"] is False
    assert info["eval_min"] == 100.0 and info["eval_max"] == 500.0
    assert info["n_outside_bounds"] == 0
    # ... and the density is normalised on those bounds, as before.
    grid = np.linspace(100.0, 500.0, 20000)
    pg = uniform_source_frame_prob(grid, cosmo, 100.0, 500.0, impl="astropy")
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    assert trapz(pg, grid) == pytest.approx(1.0, rel=1e-4)
    assert np.all(p > 0)


def test_usf_prob_non_finite_samples_stay_non_finite_and_are_counted():
    """A NaN dL must yield NaN (which the export validator rejects), not the 0.0
    the old truncation produced by comparison-with-NaN."""
    cosmo = make_cosmology(67.9, 0.3065)
    dL = np.array([200.0, np.nan, 300.0])
    p, info = uniform_source_frame_prob(dL, cosmo, 100.0, 500.0,
                                        impl="astropy", return_info=True)
    assert info["n_nonfinite"] == 1
    assert np.isnan(p[1])
    assert np.all(p[[0, 2]] > 0)


# --- implementation provenance ---------------------------------------------
def test_resolve_usf_impl_rejects_unknown():
    with pytest.raises(ValueError, match="impl must be"):
        resolve_usf_impl("scipy")


def test_impl_astropy_is_honoured_without_importing_bilby(monkeypatch):
    """impl='astropy' must not depend on bilby at all."""
    import gwcat.cosmology as gc

    def boom(name, *a, **k):
        raise AssertionError(f"unexpected import of {name}")

    monkeypatch.setattr(gc.importlib, "import_module", boom)
    assert resolve_usf_impl("astropy") == "astropy"
    _, info = uniform_source_frame_prob(np.array([300.0]),
                                        make_cosmology(70.0, 0.3),
                                        100.0, 500.0, impl="astropy",
                                        return_info=True)
    assert info["impl"] == "astropy"


def test_missing_bilby_falls_back_only_under_auto(monkeypatch):
    """A missing bilby install degrades to astropy under 'auto' but is a loud
    error when bilby was explicitly requested (the two densities differ by a few
    percent and that does NOT cancel in the per-event normalisation)."""
    import gwcat.cosmology as gc

    def no_bilby(name, *a, **k):
        raise ImportError("No module named 'bilby'")

    monkeypatch.setattr(gc.importlib, "import_module", no_bilby)
    assert resolve_usf_impl("auto") == "astropy"
    with pytest.raises(DistancePriorImplError, match="bilby"):
        resolve_usf_impl("bilby")


def test_bilby_failure_is_not_silently_swallowed(monkeypatch):
    """The old code wrapped the bilby branch in `except Exception` and fell back
    to astropy, so a real bilby failure silently changed the density.  Any
    non-import failure must now propagate."""
    import gwcat.cosmology as gc

    monkeypatch.setattr(gc, "resolve_usf_impl", lambda impl="auto": "bilby")

    def broken(*a, **k):
        raise RuntimeError("bilby exploded")

    monkeypatch.setattr(gc, "_usf_prob_bilby", broken)
    with pytest.raises(RuntimeError, match="bilby exploded"):
        gc.uniform_source_frame_prob(np.array([300.0]),
                                     make_cosmology(70.0, 0.3), 100.0, 500.0)


def test_astropy_and_bilby_agree_in_shape_and_impl_is_recorded():
    """bilby is the object the LVK PE used; the astropy fallback reproduces the
    same shape but is NOT identical, so which one ran is provenance."""
    pytest.importorskip("bilby")
    cosmo = make_cosmology(67.9, 0.3065)
    dmin, dmax = 10.0, 10000.0
    dL = np.array([500.0, 1000.0, 2000.0, 4000.0])

    p_b, info_b = uniform_source_frame_prob(dL, cosmo, dmin, dmax,
                                            impl="bilby", return_info=True)
    p_a, info_a = uniform_source_frame_prob(dL, cosmo, dmin, dmax,
                                            impl="astropy", return_info=True)
    assert info_b["impl"] == "bilby"
    assert info_a["impl"] == "astropy"
    assert np.all(p_b > 0) and np.all(p_a > 0)
    # Same shape (ratios agree) even where the absolute normalisation differs.
    ratio = p_b / p_a
    assert np.max(np.abs(ratio / ratio[0] - 1.0)) < 5e-3
    # The legacy 'auto' picks bilby when it is importable ...
    _, info_auto = uniform_source_frame_prob(dL, cosmo, dmin, dmax,
                                             impl="auto", return_info=True)
    assert info_auto["impl"] == "bilby"
    # ... and the default is the exact density since GW-40i.
    _, info_def = uniform_source_frame_prob(dL, cosmo, dmin, dmax,
                                            return_info=True)
    assert info_def["impl"] == "exact"


def test_bilby_prob_would_zero_out_of_bounds_samples():
    """Pin the underlying defect: bilby's UniformSourceFrame.prob IS zero below
    its minimum, so the fix must widen the evaluation range rather than hand
    bilby the recorded bounds."""
    pytest.importorskip("bilby")
    from bilby.gw.prior import UniformSourceFrame
    cosmo = make_cosmology(67.9, 0.3065)
    prior = UniformSourceFrame(minimum=100.0, maximum=500.0, cosmology=cosmo,
                               name="luminosity_distance")
    assert float(prior.prob(40.0)) == 0.0
    # ... whereas gwcat's wrapper gives it a real density.
    p = uniform_source_frame_prob(np.array([40.0]), cosmo, 100.0, 500.0,
                                  impl="bilby")
    assert p[0] > 0.0


# --- E(z) comes from the cosmology object, not a hardcoded form -------------
def _old_hardcoded_grid(cosmology, zmax=10.0, ngrid=4000):
    """The pre-GW-01 astropy fallback: comoving_distance from the cosmology
    object but E(z) hardcoded as sqrt(Om0(1+z)^3 + 1-Om0).

    Evaluated on whatever z grid the current code uses, so these tests stay
    about E(z) and not about the grid layout (GW-29 refined it near z=0).
    """
    import astropy.units as u

    c_kms = 299792.458
    dH = c_kms / cosmology.H0.value
    z = _usf_z_grid(zmax, ngrid)
    DC = cosmology.comoving_distance(z).to(u.Mpc).value
    E = np.sqrt(cosmology.Om0 * (1 + z) ** 3 + (1.0 - cosmology.Om0))
    dL_grid = (1 + z) * DC
    ddL_dz = DC + (1 + z) * (dH / E)
    pz = (DC ** 2 / E) * (1.0 / (1 + z))
    return dL_grid, pz / ddL_dz


def test_usf_grid_efunc_matches_hardcoded_form_for_bare_flat_lcdm():
    """For FlatLambdaCDM(H0, Om0) -- what make_cosmology builds -- taking E(z)
    from the object is numerically identical to the old hardcoded expression, so
    this change moves no production number."""
    cosmo = make_cosmology(67.9, 0.3065)
    dL_new, p_new = _usf_grid(cosmo, 10.0, 4000)
    dL_old, p_old = _old_hardcoded_grid(cosmo)
    np.testing.assert_allclose(dL_new, dL_old, rtol=1e-13, atol=0)
    np.testing.assert_allclose(p_new, p_old, rtol=1e-12, atol=0)


def test_usf_grid_efunc_differs_for_a_cosmology_with_radiation():
    """For a cosmology carrying radiation/neutrinos the hardcoded matter+Lambda
    E(z) is simply wrong; reading efunc off the object fixes it."""
    from astropy.cosmology import Planck15

    dL_new, p_new = _usf_grid(Planck15, 10.0, 4000)
    dL_old, p_old = _old_hardcoded_grid(Planck15)
    # Same distance grid (both use the object's comoving_distance) ...
    np.testing.assert_allclose(dL_new, dL_old, rtol=1e-13, atol=0)
    # ... but the density differs because E(z) does.  (Skip the z=0 point, where
    # both densities are identically zero.)
    rel = np.abs(p_new[1:] / p_old[1:] - 1.0)
    assert np.max(rel) > 1e-4, f"max rel diff {np.max(rel):.3e}"


# --------------------------------------------------------------------------
# GW-29: the fallback must be accurate over the SUPPORTED prior-bound range
# --------------------------------------------------------------------------
# The production store records dL prior bounds as low as dmin = 1 Mpc, but the
# fallback's log-(1+z) grid put its first non-zero node at dL ~ 2.6 Mpc.  Since
# p(dL) propto dL^2 near the origin, linearly interpolating across that bin was
# +164% high at 1 Mpc, +32% at 2 Mpc, +2.8% at 5 Mpc and +1.2% at 10 Mpc.
def _exact_usf_density(cosmology, z, *, time_dilation=True):
    """(dL, p(dL)) EXACTLY at the given redshifts -- no grid, no interpolation.

    This is the same closed form the fallback interpolates, evaluated at the
    requested z, so it is an independent reference for the grid's interpolation
    error (up to the same arbitrary constant).
    """
    import astropy.units as u

    z = np.asarray(z, dtype=float)
    DC = cosmology.comoving_distance(z).to(u.Mpc).value
    E = np.asarray(cosmology.efunc(z), dtype=float)
    dH = float(cosmology.hubble_distance.to(u.Mpc).value)
    ddL_dz = DC + (1 + z) * dH / E
    p = DC ** 2 / E / ddL_dz
    if time_dilation:
        p = p / (1 + z)
    return (1 + z) * DC, p


def _z_of_dL_exact(cosmology, dL_mpc):
    import astropy.units as u
    from astropy.cosmology import z_at_value

    return np.array([float(z_at_value(cosmology.luminosity_distance,
                                      float(d) * u.Mpc, zmin=1e-12).value)
                     for d in dL_mpc])


@pytest.mark.parametrize("time_dilation", [True, False])
def test_astropy_fallback_shape_is_accurate_from_one_mpc(time_dilation):
    """1-10 Mpc is inside the supported bound range, not an academic corner."""
    cosmo = make_cosmology(67.9, 0.3065)
    dmin, dmax = 1.0, 10000.0          # the widest bounds the store records
    probe = np.array([1.0, 2.0, 5.0, 10.0, 40.0, 100.0, 1000.0])
    z = _z_of_dL_exact(cosmo, probe)
    dL, p_exact = _exact_usf_density(cosmo, z, time_dilation=time_dilation)

    p = uniform_source_frame_prob(dL, cosmo, dmin, dmax, impl="astropy",
                                  time_dilation=time_dilation)
    # Only the SHAPE matters downstream (the normalisation is a per-event
    # constant), so compare ratios anchored on the far point.
    shape = p / p_exact
    shape = shape / shape[-1]
    np.testing.assert_allclose(shape, 1.0, rtol=1e-3)


def test_astropy_fallback_matches_a_dense_reference_grid_near_zero():
    """The reviewer's measurement: the default grid against the same formula on
    a 100x denser one, at 1-10 Mpc."""
    import gwcat.cosmology as gc

    cosmo = make_cosmology(67.9, 0.3065)
    dL = np.array([1.0, 2.0, 5.0, 10.0, 1000.0])
    p = gc._usf_prob_astropy(dL, cosmo, 1.0, 10000.0)
    p_dense = gc._usf_prob_astropy(dL, cosmo, 1.0, 10000.0, 400000)
    shape = p / p_dense
    shape = shape / shape[-1]
    np.testing.assert_allclose(shape, 1.0, rtol=1e-3)


def test_usf_z_grid_keeps_the_backbone_and_bounds_the_relative_step():
    zmax, ngrid = 10.0, 4000
    backbone = np.expm1(np.linspace(np.log(1.0), np.log(1.0 + zmax), ngrid))
    z = _usf_z_grid(zmax, ngrid)

    # np.interp needs a strictly increasing x, and z=0 must stay the first node.
    assert np.all(np.diff(z) > 0)
    assert z[0] == 0.0
    assert z[-1] == pytest.approx(zmax)
    # Every backbone node survives, so nothing above the switch point moves.
    assert np.all(np.isin(backbone, z))
    # Below the switch the RELATIVE step -- which is what bounds the
    # interpolation error of a quadratic -- is held at the target.
    z_switch = backbone[1] / _USF_NEAR_ZERO_STEP
    low = z[(z > 0) & (z <= z_switch)]
    assert low.size > 100
    assert low[0] == pytest.approx(_USF_NEAR_ZERO_ZMIN)
    assert np.max(np.diff(low) / low[:-1]) < 1.05 * _USF_NEAR_ZERO_STEP


@pytest.mark.parametrize("cosmo_name", ["flat_lcdm", "planck15", "extreme"])
def test_usf_grid_is_monotonic_below_the_micro_mpc_floor(cosmo_name):
    """The refinement reaches below the 1e-6 Mpc floor uniform_source_frame_prob
    clamps its evaluation range to, and astropy's comoving_distance is still
    monotonic there -- a non-monotonic x would make np.interp return garbage."""
    from astropy.cosmology import Planck15

    cosmo = {"flat_lcdm": make_cosmology(67.9, 0.3065),
             "planck15": Planck15,
             "extreme": make_cosmology(20.0, 0.9)}[cosmo_name]
    dH = float(cosmo.hubble_distance.to("Mpc").value)
    dL, p = _usf_grid(cosmo, 10.0, 4000)
    assert np.all(np.diff(dL) > 0)
    # The first refined node sits at dL ~ dH * zmin -- 4e-7 Mpc for a realistic
    # Hubble distance, i.e. below the 1e-6 Mpc evaluation floor.
    assert dL[0] == 0.0
    assert dL[1] == pytest.approx(dH * _USF_NEAR_ZERO_ZMIN, rel=1e-3)
    assert np.all(p[1:] > 0) and np.all(np.isfinite(p))
