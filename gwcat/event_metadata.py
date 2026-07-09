"""Event-metadata assembly + missing-field diagnostics (PR 8).

This module is the "layered metadata system" the handoff's Online Data
Strategy calls for, applied specifically to the per-event scalar fields that
:func:`gwcat.ingest.build_store` reads out of its ``event_table`` argument
(``far``, ``pastro``/``p_astro``, ``p_bbh``, ``p_nsbh``, ``p_bns``, ``p_terr``,
``source_class``, ``release``, ``observing_run``):

    1. online metadata     (e.g. gwcat.fetch.fetch_event_table_gwosc)
    2. manifest defaults    (declarative, release-level fallback values)
    3. user override file   (YAML/CSV, event_name -> {field: value})
    4. absent               (explicit, non-crashing "we do not know this")

It is deliberately decoupled from both the raw network layer (``gwcat.fetch``,
which only ever returns/consumes plain dicts) and from ``gwcat.ingest`` (which
only ever *consumes* an ``event_table`` dict -- it does not know or care where
the values came from).  That separation is the "untangling" this PR does:
fetching raw metadata, assembling/overriding it, and ingesting it into the
store are now three independent, independently testable steps.

``metadata_diagnostics`` returns a simple, JSON-serializable
``{event_name: {field: {"value": ..., "source": ...}}}`` mapping -- the
per-event, per-field provenance record the handoff's "Validation Outputs"
section asks for.  A later PR (PR 10) is expected to fold this into a full
``validation_summary.json``; this module only produces the raw ingredient.

``assemble_event_metadata`` derives the merged ``event_table`` dict AND runs
the diagnostics in one call, so callers who want overrides typically do::

    from gwcat.event_metadata import assemble_event_metadata, load_user_overrides
    from gwcat.fetch import fetch_event_table_gwosc

    online = fetch_event_table_gwosc()
    overrides = load_user_overrides("my_overrides.yaml")
    event_table, diagnostics = assemble_event_metadata(
        event_names, online_table=online, user_overrides=overrides)
    build_store(paths, "store.h5", event_table=event_table)
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

import yaml

__all__ = [
    "DEFAULT_METADATA_FIELDS",
    "load_user_overrides",
    "metadata_diagnostics",
    "assemble_event_metadata",
]

#: Per-event scalar metadata fields tracked by default.  ``far``/``pastro``
#: mirror the historical event_table keys from
#: :func:`gwcat.fetch.fetch_event_table_gwosc`; ``p_astro``/``p_bbh``/``p_nsbh``/
#: ``p_bns``/``p_terr`` mirror the source-class-contract meta columns;
#: ``source_class`` overrides the class derived from the PE mass posteriors;
#: ``release`` and ``observing_run`` override the coarse values inferred from
#: the input file/event name.  Appending these fields preserves the established
#: ordering of the original diagnostics fields.
DEFAULT_METADATA_FIELDS: Tuple[str, ...] = (
    "far", "pastro", "p_astro", "p_bbh", "p_nsbh", "p_bns", "p_terr",
    "source_class", "release", "observing_run",
)

_SOURCES_ABSENT = "absent"
_SOURCE_ONLINE = "online"
_SOURCE_MANIFEST = "manifest"
_SOURCE_USER_OVERRIDE = "user_override"


def _is_present(value: Any) -> bool:
    """True if ``value`` is a real, known value (not None/NaN/empty string)."""
    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    if isinstance(value, float) and math.isnan(value):
        return False
    return True


def _coerce_scalar(value: str):
    """Best-effort str -> float coercion for CSV cell values; else keep as str."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


# ---------------------------------------------------------------------------
# User override file loading (YAML / CSV)
# ---------------------------------------------------------------------------
def load_user_overrides(path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """Load a user-supplied event metadata override file.

    Two layouts are accepted:

    YAML (``.yaml``/``.yml``)
        Either a mapping ``{event_name: {field: value, ...}, ...}`` or a list
        of records each carrying an ``event_name`` (or ``name``) key plus the
        override fields, e.g.::

            GW150914:
              far: 1.0e-8
              p_astro: 0.999
            GW190425:
              source_class: BNS

        or::

            - event_name: GW150914
              far: 1.0e-8

    CSV (``.csv``)
        One header column must be ``event_name``; every other column is an
        override field.  Blank cells are ignored (never override with an
        empty string).  Numeric-looking cells are coerced to float.

    Returns
    -------
    dict : {event_name: {field: value}}
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        if isinstance(raw, dict):
            return {str(name): dict(fields or {}) for name, fields in raw.items()}
        if isinstance(raw, list):
            out: Dict[str, Dict[str, Any]] = {}
            for row in raw:
                if not isinstance(row, dict):
                    raise ValueError(
                        f"{path}: each list entry must be a mapping, got {row!r}"
                    )
                name = row.get("event_name") or row.get("name")
                if not name:
                    raise ValueError(
                        f"{path}: override row missing 'event_name': {row!r}"
                    )
                out[str(name)] = {
                    k: v for k, v in row.items() if k not in ("event_name", "name")
                }
            return out
        raise ValueError(
            f"{path}: unsupported YAML structure for user overrides "
            f"(expected a mapping or a list), got {type(raw)!r}"
        )

    if suffix == ".csv":
        out = {}
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "event_name" not in reader.fieldnames:
                raise ValueError(f"{path}: CSV must have an 'event_name' column")
            for row in reader:
                name = row.pop("event_name")
                if not name:
                    continue
                out[name] = {
                    k: _coerce_scalar(v) for k, v in row.items()
                    if v is not None and v != ""
                }
        return out

    raise ValueError(
        f"{path}: unsupported user-override file type {suffix!r}; "
        "expected .yaml, .yml, or .csv"
    )


# ---------------------------------------------------------------------------
# Diagnostics + assembly
# ---------------------------------------------------------------------------
def metadata_diagnostics(
    event_names: Iterable[str],
    online_table: Optional[Mapping[str, Mapping[str, Any]]] = None,
    user_overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
    manifest_defaults: Optional[Mapping[str, Any]] = None,
    fields: Iterable[str] = DEFAULT_METADATA_FIELDS,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Per-event, per-field provenance record: found vs missing, and from where.

    Parameters
    ----------
    event_names : iterable of str
        Events to report on (typically every event about to be ingested).
    online_table : {event_name: {field: value}}, optional
        Metadata fetched online (e.g. ``fetch_event_table_gwosc()``'s return,
        or any dict with the same shape).
    user_overrides : {event_name: {field: value}}, optional
        Loaded via :func:`load_user_overrides`; always wins over ``online_table``.
    manifest_defaults : {field: value} or {event_name: {field: value}}, optional
        Release-level fallback values.  A flat ``{field: value}`` dict applies
        the same defaults to every event; a nested ``{event_name: {...}}`` dict
        applies per event.  Used only when neither an override nor online value
        is present.
    fields : iterable of str
        Which fields to report on.  Default :data:`DEFAULT_METADATA_FIELDS`.

    Returns
    -------
    dict : {event_name: {field: {"value": value_or_None, "source": str}}}
        ``source`` is one of ``"user_override"``, ``"online"``, ``"manifest"``,
        or ``"absent"``.  Serializable as-is with :mod:`json`.
    """
    online_table = online_table or {}
    user_overrides = user_overrides or {}
    manifest_defaults = manifest_defaults or {}
    # A flat {field: value} manifest-defaults dict applies uniformly; a nested
    # {event_name: {field: value}} dict is keyed per event.  Distinguish by
    # whether any value is itself a mapping.
    manifest_is_nested = any(isinstance(v, Mapping) for v in manifest_defaults.values())

    fields = tuple(fields)
    diagnostics: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for name in event_names:
        online_ev = online_table.get(name, {}) or {}
        override_ev = user_overrides.get(name, {}) or {}
        if manifest_is_nested:
            manifest_ev = manifest_defaults.get(name, {}) or {}
        else:
            manifest_ev = manifest_defaults

        per_field: Dict[str, Dict[str, Any]] = {}
        for field in fields:
            if field in override_ev and _is_present(override_ev[field]):
                per_field[field] = {"value": override_ev[field],
                                    "source": _SOURCE_USER_OVERRIDE}
            elif field in online_ev and _is_present(online_ev[field]):
                per_field[field] = {"value": online_ev[field],
                                    "source": _SOURCE_ONLINE}
            elif field in manifest_ev and _is_present(manifest_ev[field]):
                per_field[field] = {"value": manifest_ev[field],
                                    "source": _SOURCE_MANIFEST}
            else:
                per_field[field] = {"value": None, "source": _SOURCES_ABSENT}
        diagnostics[name] = per_field
    return diagnostics


def assemble_event_metadata(
    event_names: Iterable[str],
    online_table: Optional[Mapping[str, Mapping[str, Any]]] = None,
    user_overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
    manifest_defaults: Optional[Mapping[str, Any]] = None,
    fields: Iterable[str] = DEFAULT_METADATA_FIELDS,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Dict[str, Any]]]]:
    """Merge online/manifest/user-override metadata into one ``event_table``.

    This is the metadata-assembly path feeding :func:`gwcat.ingest.build_store`'s
    ``event_table`` argument.  Precedence per field is
    ``user_override > online > manifest > absent``.

    Returns
    -------
    (event_table, diagnostics)
        ``event_table`` : {event_name: {field: value, "metadata_source": str}}
            Ready to pass as ``build_store(..., event_table=event_table)``.
            Fields with no known value are simply absent from the event's dict
            (never fabricated as NaN/""), which is exactly what ``build_store``
            already treats as "explicit absence" for far/p_astro/etc.
            ``metadata_source`` is a '+'-joined, sorted list of the distinct
            sources that contributed at least one field for that event (e.g.
            ``"online"``, ``"user_override"``, ``"online+user_override"``), or
            ``"absent"`` if nothing was found for that event at all.
        ``diagnostics`` : see :func:`metadata_diagnostics`.
    """
    event_names = list(event_names)
    diagnostics = metadata_diagnostics(
        event_names, online_table=online_table, user_overrides=user_overrides,
        manifest_defaults=manifest_defaults, fields=fields,
    )

    event_table: Dict[str, Dict[str, Any]] = {}
    for name in event_names:
        per_field = diagnostics[name]
        entry = {f: d["value"] for f, d in per_field.items()
                if d["source"] != _SOURCES_ABSENT}
        sources = sorted({d["source"] for d in per_field.values()
                          if d["source"] != _SOURCES_ABSENT})
        entry["metadata_source"] = "+".join(sources) if sources else _SOURCES_ABSENT
        event_table[name] = entry
    return event_table, diagnostics
