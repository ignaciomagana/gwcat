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

#: The legacy selection datasets, in the order the v1 exporter writes them,
#: followed by the additive component-spin columns.
_SELECTION_DATASETS = ["m1det", "m2det", "dL", "chieff", "ra", "dec",
                       "m1src", "m2src", "redshift", "pdraw",
                       "a1", "a2", "cost1", "cost2", "chip"]


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

        # Legacy-compat spin attrs.  chieff keeps the historical values; every
        # other basis states the truth explicitly rather than omitting them
        # (GW-21).  Two reasons the omission was not safe:
        #
        #  * darksirens REQUIRES chi_eff_in_p_pe on a gwcat-pe-2.0 file
        #    (gw/utils.py required-attrs list), so a file lacking it does not
        #    fail a physics check -- it fails to load, with a member-check error
        #    that says nothing about the basis;
        #  * "absent" and "False" are different claims. A consumer must be able
        #    to distinguish "no chi_eff prior was applied" from "this file does
        #    not say", and only the first is a statement it can act on.
        if product.spin_basis == "chieff":
            f.attrs["spin_prior_mode"] = "include"
            f.attrs["chi_eff_prior_applied_to_p_pe"] = True
            f.attrs["chi_eff_in_p_pe"] = True
        else:
            mode = {"component": "component_flat",
                    "chieff_chip": "chieff_chip_joint",
                    "nospin": "none"}.get(product.spin_basis, "unknown")
            f.attrs["spin_prior_mode"] = mode
            # The 1-D chi_eff prior specifically is NOT in p_pe for any of
            # these: component carries a flat box, chieff_chip a JOINT
            # (chi_eff, chi_p) density, nospin nothing at all.  So the legacy
            # scalar flag is False in every case, and the basis-specific attrs
            # written by the builder are what say what WAS applied.
            f.attrs["chi_eff_prior_applied_to_p_pe"] = False
            f.attrs["chi_eff_in_p_pe"] = False

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


@register_exporter("gwcat2", kind="selection")
def write_selection_gwcat2(product, out_path, *, write_summary: bool = False,
                           summary_context: Optional[dict] = None):
    """Serialize a selection :class:`ExportProduct` as ``gwcat-selection-2.0``.

    Owns ONLY serialization: gzip-compressed datasets (legacy order first, then
    any extra spin columns), ``product.attrs`` verbatim, and
    ``format_version="gwcat-selection-2.0"``.  All physics -- the detection cut,
    source-class subsetting, the per-basis spin factor and the Essick fractions
    -- is upstream in :func:`gwcat.export.selection_builder.build_selection_product`.

    Parameters
    ----------
    product : gwcat.export.product.ExportProduct
        A ``kind="selection"`` product from ``build_selection_product``.
    out_path : str or path-like
        Destination HDF5 path.
    write_summary : bool, default False
        When True, also write ``<out_path>.validation_summary.json`` (+ ``.md``)
        next to ``out_path`` from ``product.summary`` (the legacy selection
        exporter's behavior, via :mod:`gwcat.validation_summary`).
    summary_context : dict, optional
        Extra fields merged into the written summary (last, so they win).

    Returns
    -------
    str
        ``str(out_path)``.
    """
    if product.kind != "selection":
        raise ValueError(
            f"write_selection_gwcat2 expects a kind='selection' product, got "
            f"kind={product.kind!r}.")

    with h5py.File(out_path, "w") as f:
        # Provenance attrs, verbatim from the builder.
        for k, v in product.attrs.items():
            f.attrs[k] = v

        # Format version is the writer's, never the builder's.
        f.attrs["format_version"] = "gwcat-selection-2.0"

        # Datasets (gzip, like the legacy selection exporter).  Legacy order
        # first (stable), then any extra columns a basis added.
        written = set()
        for k in _SELECTION_DATASETS:
            if k in product.columns:
                f.create_dataset(k, data=product.columns[k],
                                 compression="gzip")
                written.add(k)
        for k, arr in product.columns.items():
            if k not in written:
                f.create_dataset(k, data=arr, compression="gzip")

    if write_summary:
        from ..validation_summary import write_validation_summary
        summary = dict(product.summary)
        summary["output_path"] = str(out_path)
        if summary_context:
            summary.update(summary_context)
        write_validation_summary(out_path, summary)

    return str(out_path)
