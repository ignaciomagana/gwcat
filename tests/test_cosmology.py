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
