"""The in-memory export product handed from a builder to a writer (PR 3).

The versioned export pipeline is a clean two-stage split:

  * a *builder* (see :mod:`gwcat.export.pe_builder`) owns ALL physics -- event
    selection, per-event cosmology, resampling, the mass Jacobian, the spin
    prior -- and returns an :class:`ExportProduct`; and
  * a *writer* (registered via :mod:`gwcat.export.registry`) owns ONLY
    serialization -- turning that product into an on-disk file, setting the
    ``format_version`` and any format-specific attrs.

The product is therefore a plain, serialization-agnostic container: numeric
columns, a provenance ``attrs`` dict (everything the legacy exporter recorded
*except* ``format_version``, which is the writer's), the resolved
``spin_basis``, and a ``summary`` feed for the optional validation summary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class ExportProduct:
    """Result of a builder: numeric columns + provenance, ready to serialize.

    Attributes
    ----------
    kind : str
        Product kind -- ``"pe"`` (posterior-sample export) or ``"selection"``
        (selection-function export; lands in a later PR).  Writers are
        registered per ``(format, kind)``.
    columns : dict[str, numpy.ndarray]
        The flat, concatenated numeric datasets to serialize.  For a ``"pe"``
        product this contains (at least) the legacy darksirens datasets
        ``ra, dec, m1det, m2det, chieff, dL, p_pe, redshift, m1src, m2src``.
    attrs : dict
        All provenance the writer should record verbatim, EXCEPT
        ``format_version`` (which the writer owns).  Values are plain Python /
        numpy scalars and arrays (string arrays already carry an HDF5 string
        dtype so a writer can assign them directly).
    spin_basis : str
        The resolved spin basis (``"chieff"`` in this PR).
    summary : dict
        Feed for the optional validation summary (see
        :mod:`gwcat.validation_summary`).  The writer fills in ``output_path``
        and merges any caller ``summary_context`` before writing.
    """

    kind: str
    columns: Dict[str, np.ndarray] = field(default_factory=dict)
    attrs: Dict[str, Any] = field(default_factory=dict)
    spin_basis: str = "chieff"
    summary: Dict[str, Any] = field(default_factory=dict)
