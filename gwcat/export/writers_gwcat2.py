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
import numpy as np

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


# ==========================================================================
# Format 2.1 (GW-22) -- OPT-IN.  "gwcat2" still means 2.0.
# ==========================================================================
# 2.1 adds the declarative contract to the file: the parameter space, its fit
# and advisory columns, per-block provenance, and a contract_hash that makes a
# mismatched PE/selection pairing one comparison instead of a growing list of
# per-field checks.
#
# It is registered as a SEPARATE format name rather than bumping "gwcat2",
# because `_require_hdf5_format` in darksirens is a closed literal tuple: a 2.1
# file is unloadable by any consumer that has not been taught the version, and
# the plan's rule is that the consumer patch deploys FIRST.  Keeping 2.0 the
# default means nothing gwcat writes by accident becomes unreadable.
_PE_FORMAT_21 = "gwcat-pe-2.1"
_SEL_FORMAT_21 = "gwcat-selection-2.1"


def _contract_attrs(product, kind):
    """The 2.1 contract attrs for a product, plus its hash."""
    import json as _json

    from ..params import get_space
    from .contract import build_contract, contract_hash

    space = get_space(product.spin_basis)
    a = product.attrs
    spin = space.spin_block

    # The detection cut, as a (statistic, threshold) pair rather than a bare
    # number: Essick & Fishbach require the EVENT cut and the INJECTION cut to
    # be the same statistic at the same threshold, and comparing thresholds
    # without their statistic is how that check stayed vacuous.
    # Name the STATISTIC, not the side's attr: "far_max" and "far_threshold" are
    # two spellings of one cut, and recording the spellings made the two sides
    # differ by construction.
    thr = a.get("far_max") if kind == "pe" else a.get("far_threshold")
    thr = None if thr is None or not np.isfinite(float(thr)) else float(thr)
    stat = None if thr is None else "far"

    contract = build_contract(
        parameter_space=space.name,
        fit_columns=list(space.fit_columns),
        advisory_columns=list(space.advisory_columns),
        spin_basis_kind=spin.map_kind,
        spin_density_exact=bool(space.is_exact),
        mass_prior_kind=a.get("mass_prior_basis") or a.get("mass_prior_kind"),
        dL_prior_kind=a.get("dL_prior_kind"),
        sky_prior_in_density=True,      # sky.radec declares -ln4pi both sides
        source_class=a.get("source_class_filter") or None,
        detection_statistic=stat,
        detection_threshold=thr,
        allow_missing_far=bool(a.get("allow_missing_far", False)),
        cosmology_H0=(a.get("pe_cosmology_H0") if kind == "pe"
                      else a.get("cosmology_H0")),
        cosmology_Om0=(a.get("pe_cosmology_Om0") if kind == "pe"
                       else a.get("cosmology_Om0")),
    )
    return {
        "parameter_space": space.name,
        "parameter_blocks": np.array([b.name for b in space.blocks],
                                     dtype=h5py.string_dtype()),
        "fit_columns": np.array(list(space.fit_columns),
                                dtype=h5py.string_dtype()),
        "advisory_columns": np.array(list(space.advisory_columns),
                                     dtype=h5py.string_dtype()),
        "spin_basis_kind": spin.map_kind,
        "spin_density_exact": bool(space.is_exact),
        "block_provenance": _json.dumps(space.describe()),
        "contract": _json.dumps(contract),
        "contract_hash": contract_hash(contract),
    }


@register_exporter("gwcat2.1", kind="pe")
def write_pe_gwcat21(product, out_path, *, write_summary: bool = False,
                     summary_context=None):
    """Write a ``gwcat-pe-2.1`` PE export (opt-in; see the module note)."""
    path = write_pe_gwcat2(product, out_path, write_summary=write_summary,
                           summary_context=summary_context)
    with h5py.File(out_path, "r+") as f:
        f.attrs["format_version"] = _PE_FORMAT_21
        for k, v in _contract_attrs(product, "pe").items():
            f.attrs[k] = v
    return path


@register_exporter("gwcat2.1", kind="selection")
def write_selection_gwcat21(product, out_path, *, write_summary: bool = False,
                            summary_context=None):
    """Write a ``gwcat-selection-2.1`` selection export (opt-in)."""
    path = write_selection_gwcat2(product, out_path,
                                  write_summary=write_summary,
                                  summary_context=summary_context)
    with h5py.File(out_path, "r+") as f:
        f.attrs["format_version"] = _SEL_FORMAT_21
        for k, v in _contract_attrs(product, "selection").items():
            f.attrs[k] = v
    return path
