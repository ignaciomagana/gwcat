"""gwcat2 PE writer for the versioned export pipeline (PR 3).

Owns ONLY serialization: it turns an :class:`~gwcat.export.product.ExportProduct`
(built by :func:`gwcat.export.pe_builder.build_pe_product`) into a
``format_version="gwcat-pe-2.0"`` HDF5 file.  All physics is upstream in the
builder; this module never touches columns or provenance semantics beyond
stamping the format version and the chieff-basis legacy-compat spin attrs.
"""
from __future__ import annotations

from typing import Optional

import h5py

from .registry import register_exporter

#: The datasets a chieff-basis PE file writes, in legacy order.
_PE_DATASETS = ["ra", "dec", "m1det", "m2det", "chieff", "dL", "p_pe",
                "redshift", "m1src", "m2src"]


@register_exporter("gwcat2", kind="pe")
def write_pe_gwcat2(product, out_path, *, write_summary: bool = False,
                    summary_context: Optional[dict] = None):
    """Serialize a PE :class:`ExportProduct` as a ``gwcat-pe-2.0`` HDF5 file.

    Columns are written as gzip-compressed datasets (as the legacy exporter
    does); ``product.attrs`` is written verbatim; ``format_version`` is set to
    ``"gwcat-pe-2.0"``; and -- ONLY in the chieff basis -- the legacy-compat
    spin attrs (``spin_prior_mode="include"``, ``chi_eff_prior_applied_to_p_pe``
    and its ``chi_eff_in_p_pe`` alias, both ``True``) are added so downstream
    tooling that reads the gwcat-1.0 spin contract keeps working.

    Parameters
    ----------
    product : gwcat.export.product.ExportProduct
        A ``kind="pe"`` product from :func:`build_pe_product`.
    out_path : str or path-like
        Destination HDF5 path.
    write_summary : bool, default False
        When True, also write ``<out_path>.validation_summary.json`` (+ ``.md``)
        next to ``out_path`` from ``product.summary`` (see
        :mod:`gwcat.validation_summary`).
    summary_context : dict, optional
        Extra fields merged into the written summary (last, so they win).

    Returns
    -------
    str
        ``str(out_path)``.
    """
    if product.kind != "pe":
        raise ValueError(
            f"write_pe_gwcat2 expects a kind='pe' product, got "
            f"kind={product.kind!r}.")

    with h5py.File(out_path, "w") as f:
        # Provenance attrs, verbatim from the builder.
        for k, v in product.attrs.items():
            f.attrs[k] = v

        # Format version is the writer's, never the builder's.
        f.attrs["format_version"] = "gwcat-pe-2.0"

        # Legacy-compat spin attrs, ONLY in chieff basis.
        if product.spin_basis == "chieff":
            f.attrs["spin_prior_mode"] = "include"
            f.attrs["chi_eff_prior_applied_to_p_pe"] = True
            f.attrs["chi_eff_in_p_pe"] = True

        # Datasets (gzip, like the legacy exporter).  Write the 10 canonical
        # PE datasets first (stable order), then any extra columns a future
        # basis may add.
        written = set()
        for k in _PE_DATASETS:
            if k in product.columns:
                f.create_dataset(k, data=product.columns[k],
                                 compression="gzip", shuffle=False)
                written.add(k)
        for k, arr in product.columns.items():
            if k not in written:
                f.create_dataset(k, data=arr, compression="gzip",
                                 shuffle=False)

    if write_summary:
        from ..validation_summary import write_validation_summary
        summary = dict(product.summary)
        summary["output_path"] = str(out_path)
        if summary_context:
            summary.update(summary_context)
        write_validation_summary(out_path, summary)

    return str(out_path)
