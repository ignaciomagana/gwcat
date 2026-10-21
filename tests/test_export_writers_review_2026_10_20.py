"""The export writers against the 2026-10-20 review (writers_gwcat2 / atomic).

Each test is the review's failure scenario, reduced to a synthetic store or a
hand-built product: no network, no ingest, no production artifact.  Every one
fails on 3ba1dbf.

  * a PE file cut on FAR AND the posterior-median SNR declared itself
    ``far_or_snr`` -- the statistic of the looser injection mask it then paired
    with -- and the validator compared thresholds only;
  * no PE export recorded the distance-prior class it divided out, so the 2.1
    contract's ``dL_prior_kind`` was always null;
  * a re-export left the previous product's validation summary beside the new
    file, the summary was written in place, and nothing tied it to its file;
  * the files recorded nothing about the run environment that sets their last
    bits;
  * the atomic write replaced a per-file symlink with a regular file;
  * chieff_chip files said ``chi_eff_prior_applied_to_p_pe=False`` while their
    summary said True;
  * the 2.1 contract hard-coded ``sky_prior_in_density=True`` although neither
    density carries a sky factor;
  * a sky-marginal (or NaN-sky) selection product was written without saying
    that darksirens-core cannot load it.
"""
import hashlib
import json
import os

import h5py
import numpy as np
import pytest

from gwcat.catalog import GWCatalog
from gwcat.export import ExportProduct, build_pe_product, get_exporter

from test_export_writer_hardening import _build_store, _selection_product

_STR = h5py.string_dtype()


def _store(tmp_path, name="store.h5", **meta):
    """The writer-hardening two-BBH store plus extra per-event meta columns."""
    path = _build_store(tmp_path, name=name)
    with h5py.File(path, "a") as f:
        for k, v in meta.items():
            v = np.asarray(v)
            if v.dtype.kind in "US":
                f["meta"].create_dataset(k, data=v.astype(object), dtype=_STR)
            else:
                f["meta"].create_dataset(k, data=v.astype("f8"))
    return path


def _contract(path):
    with h5py.File(path, "r") as f:
        return json.loads(str(f.attrs["contract"]))


def _summary_paths(out):
    return (str(out) + ".validation_summary.json",
            str(out) + ".validation_summary.md")


# ==========================================================================
# Medium 13. A selection product core cannot load says so when written
# ==========================================================================
def _sky_marginal_product():
    p = _selection_product()
    cols = {k: v for k, v in p.columns.items() if k not in ("ra", "dec")}
    return ExportProduct(kind="selection", columns=cols,
                         attrs={**p.attrs, "sky_marginalized": True},
                         spin_basis=p.spin_basis, summary=dict(p.summary))


def _nan_sky_product():
    p = _selection_product()
    return ExportProduct(
        kind="selection", columns=p.columns,
        attrs={**p.attrs, "sky_position_available": np.array([False, True])},
        spin_basis=p.spin_basis, summary=dict(p.summary))


@pytest.mark.parametrize("make", [_sky_marginal_product, _nan_sky_product])
def test_a_selection_core_cannot_load_warns_when_written(tmp_path, make):
    out = tmp_path / "sel.h5"
    with pytest.warns(UserWarning, match="darksirens-core"):
        get_exporter("gwcat2", kind="selection")(make(), str(out))
    assert out.exists()


def test_a_sky_resolved_selection_writes_without_that_warning(tmp_path):
    import warnings

    out = tmp_path / "sel.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        get_exporter("gwcat2", kind="selection")(_selection_product(),
                                                 str(out))
