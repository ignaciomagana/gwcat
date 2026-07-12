"""Versioned export pipeline for gwcat (PR 3).

Two-stage design: a *builder* owns all physics and returns an
:class:`ExportProduct`; a *writer* (looked up in a registry by
``(format, kind)``) owns only serialization.  This keeps the frozen legacy
``GWCatalog.to_darksirens`` exporter untouched while new, versioned formats
(``gwcat-pe-2.0`` here) are added alongside it.

Public API
----------
``build_pe_product`` / ``ExportProduct``
    Build a posterior-sample product (physics) and its container.
``register_exporter`` / ``get_exporter`` / ``list_formats``
    The writer registry.
``export``
    Top-level convenience that dispatches on object type (a
    :class:`~gwcat.catalog.GWCatalog` today; selection objects land later).
"""
from __future__ import annotations

from .product import ExportProduct
from .registry import register_exporter, get_exporter, list_formats
from .pe_builder import build_pe_product

# Import writer modules for their registration side effects (module-level
# @register_exporter decorators populate the registry).
from . import writers_gwcat2  # noqa: F401

__all__ = [
    "export",
    "build_pe_product",
    "ExportProduct",
    "register_exporter",
    "get_exporter",
    "list_formats",
]


def export(obj, out_path, format="gwcat2", spin_basis="chieff",
           write_summary=False, summary_context=None, **builder_kwargs):
    """Export ``obj`` to ``out_path``, dispatching on the object's type.

    Parameters
    ----------
    obj : gwcat.catalog.GWCatalog
        The object to export.  Selection objects
        (:class:`~gwcat.selection.SelectionSet` /
        :class:`~gwcat.selection.CombinedSelectionSet`) are not yet supported
        and raise :class:`NotImplementedError` (they land in a later PR).
    out_path : str or path-like
        Destination path.
    format : str, default "gwcat2"
        Registered export format.
    spin_basis : str, default "chieff"
        Spin basis for a PE export (only ``"chieff"`` is implemented).
    write_summary : bool, default False
        Write a validation summary next to ``out_path``.
    summary_context : dict, optional
        Extra fields merged into the validation summary.
    **builder_kwargs
        Forwarded to :func:`build_pe_product`.
    """
    from ..catalog import GWCatalog

    if isinstance(obj, GWCatalog):
        return obj.export(out_path, format=format, spin_basis=spin_basis,
                          write_summary=write_summary,
                          summary_context=summary_context, **builder_kwargs)

    from ..selection import SelectionSet, CombinedSelectionSet
    if isinstance(obj, (SelectionSet, CombinedSelectionSet)):
        raise NotImplementedError(
            "export() for selection objects (SelectionSet / "
            "CombinedSelectionSet) is not implemented yet; it lands in a "
            "follow-up PR. Use SelectionSet.to_darksirens for now.")

    raise TypeError(
        f"export() does not support objects of type "
        f"{type(obj).__name__!r}; pass a GWCatalog.")
