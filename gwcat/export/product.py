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


def resolve_mock_data(input_flag, explicit, *, source) -> bool:
    """The ``mock_data`` attr an export carries.

    The flag is PROVENANCE of the input, so it comes from the input: a store
    written with ``mock_data=True`` (see :data:`gwcat.ingest.STORE_MOCK_DATA_ATTR`)
    or an injection file whose root attrs carry ``mock_data=True``.  The
    builders' explicit ``mock_data`` argument can only ADD the label:

    * ``None`` (the default) -- inherit the input's flag;
    * ``True`` -- label the export mock even though the input does not say so
      (inputs written outside gwcat's writers);
    * ``False`` -- assert the input is real; raises if the input is flagged mock,
      because relabelling synthetic data as real is exactly the error the flag
      exists to prevent.

    ``source`` names the input in the error message.
    """
    input_flag = bool(input_flag)
    if explicit is None:
        return input_flag
    if not isinstance(explicit, (bool, np.bool_)):
        raise TypeError(
            f"mock_data must be None, True or False; got {explicit!r}.")
    if not explicit and input_flag:
        raise ValueError(
            f"mock_data=False was requested, but {source} is flagged "
            f"mock_data=True. A synthetic input cannot be exported as real "
            f"data; drop the argument to inherit the flag.")
    return bool(explicit) or input_flag
