"""Tests for the public GWCatalog.to_darksirens export API (PR 1).

Covers:
  * to_darksirens() produces the expected darksirens-format HDF5 output and
    matches what the old _to_darksirens_format() implementation produced
    (they now share the same body; the private name is a thin wrapper).
  * _to_darksirens_format() still works (backward compatibility) but emits a
    DeprecationWarning pointing users to to_darksirens().
  * to_darksirens() itself does not emit a DeprecationWarning.

Fixtures are tiny, synthetic, in-memory/tmpdir HDF5 stores built directly
with h5py (matching the on-disk schema GWCatalog reads) -- no network access
and no dependency on pesummary/ingest.
"""
import warnings

import numpy as np
import h5py
import pytest

from gwcat.catalog import GWCatalog


def _build_tiny_store(tmp_path, n_events=2, n_per_event=20, seed=1,
                      H0=67.74, Om0=0.3089, name="store.h5"):
    """Write a minimal synthetic store.h5 matching GWCatalog's schema.

    Only the columns needed by the darksirens exporter are populated:
    mass_1, mass_2, luminosity_distance, ra, dec, chi_eff, p_dL_pe.
    """
    rng = np.random.default_rng(seed)
    names = [f"GW999999_{i:06d}" for i in range(n_events)]
    param_list = ["mass_1", "mass_2", "luminosity_distance", "ra", "dec",
                  "chi_eff", "p_dL_pe"]

    per_event = {p: [] for p in param_list}
    offsets = [0]
    for _ in range(n_events):
        n = n_per_event
        m1 = rng.uniform(25, 50, n)
        m2 = rng.uniform(10, 25, n)
        dL = rng.uniform(300, 800, n)
        ra = rng.uniform(0, 2 * np.pi, n)
        dec = rng.uniform(-np.pi / 2, np.pi / 2, n)
        chi = rng.uniform(-0.4, 0.4, n)
        p_dL = rng.uniform(0.1, 1.0, n)

        per_event["mass_1"].append(m1)
        per_event["mass_2"].append(m2)
        per_event["luminosity_distance"].append(dL)
        per_event["ra"].append(ra)
        per_event["dec"].append(dec)
        per_event["chi_eff"].append(chi)
        per_event["p_dL_pe"].append(p_dL)
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


def test_to_darksirens_matches_legacy_private_output(tmp_path):
    """Public to_darksirens() must byte-for-byte match the old private
    exporter's output, since the private name now delegates to it with an
    identical signature and RNG seed."""
    store_path = _build_tiny_store(tmp_path)
    cat = GWCatalog(store_path)

    out_public = tmp_path / "public.h5"
    out_private = tmp_path / "private.h5"

    kwargs = dict(nsamp=10, seed=0, cosmology=(67.74, 0.3089))

    ret = cat.to_darksirens(str(out_public), **kwargs)
    assert ret == str(out_public)

    with pytest.warns(DeprecationWarning, match="to_darksirens"):
        cat._to_darksirens_format(str(out_private), **kwargs)

    with h5py.File(out_public, "r") as fp, h5py.File(out_private, "r") as fr:
        assert fp.attrs["nobs"] == fr.attrs["nobs"] == 2
        assert fp.attrs["nsamp"] == fr.attrs["nsamp"] == 10
        assert list(fp.attrs["event_names"]) == list(fr.attrs["event_names"])

        for key in ["ra", "dec", "m1det", "m2det", "chieff", "dL", "p_pe",
                    "redshift", "m1src", "m2src"]:
            np.testing.assert_array_equal(fp[key][:], fr[key][:])
            # sanity: non-trivial output, not accidentally empty
            assert fp[key].shape[0] == 2 * 10


def test_private_alias_still_works_and_warns(tmp_path):
    """_to_darksirens_format remains usable for backward compatibility but
    must emit a DeprecationWarning steering callers to to_darksirens."""
    store_path = _build_tiny_store(tmp_path, seed=2)
    cat = GWCatalog(store_path)
    out = tmp_path / "legacy_out.h5"

    with pytest.warns(DeprecationWarning, match="to_darksirens"):
        result = cat._to_darksirens_format(str(out), nsamp=5, seed=0,
                                           cosmology=(67.74, 0.3089))

    assert result == str(out)
    assert out.exists()
    with h5py.File(out, "r") as f:
        assert f.attrs["nobs"] == 2
        assert f.attrs["nsamp"] == 5
        assert f.attrs["format_version"] == "gwcat-1.0"


def test_to_darksirens_itself_does_not_warn(tmp_path):
    """The new public method must not carry the deprecation warning; only
    the legacy private alias should."""
    store_path = _build_tiny_store(tmp_path, seed=3)
    cat = GWCatalog(store_path)
    out = tmp_path / "no_warn.h5"

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        cat.to_darksirens(str(out), nsamp=5, seed=0, cosmology=(67.74, 0.3089))

    assert out.exists()


def test_to_darksirens_per_event_cosmology_default(tmp_path):
    """cosmology=None should read the per-event PE cosmology stored in
    meta/dL_prior_H0 and meta/dL_prior_Om0, matching prior behavior."""
    store_path = _build_tiny_store(tmp_path, seed=4, H0=70.0, Om0=0.3)
    cat = GWCatalog(store_path)
    out = tmp_path / "per_event_cosmo.h5"

    cat.to_darksirens(str(out), nsamp=5, seed=0)  # cosmology=None (default)

    with h5py.File(out, "r") as f:
        assert f.attrs["pe_cosmology_H0"] == pytest.approx(70.0)
        assert f.attrs["pe_cosmology_Om0"] == pytest.approx(0.3)
