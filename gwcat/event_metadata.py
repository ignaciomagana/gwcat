"""Event-metadata assembly + missing-field diagnostics (PR 8).

This module is the "layered metadata system" the handoff's Online Data
Strategy calls for, applied specifically to the per-event scalar fields that
:func:`gwcat.ingest.build_store` reads out of its ``event_table`` argument
(``far``, ``pastro``/``p_astro``, ``p_bbh``, ``p_nsbh``, ``p_bns``, ``p_terr``,
``source_class``):

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
    "PASTRO_KEYS",
    "load_user_overrides",
    "metadata_diagnostics",
    "assemble_event_metadata",
    "resolve_pastro",
    "resolve_pastro_column",
]

#: Per-event scalar metadata fields tracked by default.  ``far``/``pastro``
#: mirror the historical event_table keys from
#: :func:`gwcat.fetch.fetch_event_table_gwosc`; ``p_astro``/``p_bbh``/``p_nsbh``/
#: ``p_bns``/``p_terr`` mirror the source-class-contract meta columns;
#: ``source_class`` is the one field ``gwcat.ingest.build_store`` will honor as
#: an explicit override (see its ``et.get("source_class")`` check) instead of
#: always deriving it from the PE mass posteriors.
DEFAULT_METADATA_FIELDS: Tuple[str, ...] = (
    "far", "pastro", "p_astro", "p_bbh", "p_nsbh", "p_bns", "p_terr",
    "source_class",
)

#: The two spellings of the SAME quantity -- the astrophysical probability of
#: an event -- that this package has to accept.  ``pastro`` is the historical
#: key returned by :func:`gwcat.fetch.fetch_event_table_gwosc` and the legacy
#: store column; ``p_astro`` is the GWOSC field name, the spelling this
#: module's own documented override example uses, and the source-class-contract
#: column.  They are ONE number: nothing anywhere writes two different values,
#: and no consumer wants "the p_astro one" as opposed to "the pastro one".
#: Resolution is therefore symmetric everywhere -- first finite value wins, in
#: this order -- and both columns are written with the resolved value, so a
#: reader that knows only one spelling cannot be told "absent" about a value
#: the store holds under the other (GW-14).
PASTRO_KEYS: Tuple[str, ...] = ("p_astro", "pastro")

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


def _field_spellings(field: str) -> Tuple[str, ...]:
    """Every key one metadata field may be written under, most-preferred first."""
    return PASTRO_KEYS if field in PASTRO_KEYS else (field,)


def _layer_value(layer: Mapping[str, Any], field: str):
    """The value one precedence layer supplies for ``field``, or None."""
    for key in _field_spellings(field):
        if key in layer and _is_present(layer[key]):
            return layer[key]
    return None


def resolve_pastro(mapping: Optional[Mapping[str, Any]]) -> float:
    """The event's astrophysical probability under either spelling.

    Reads ``p_astro`` then ``pastro`` from a per-event mapping (an
    ``event_table`` entry, a user-override record) and returns the first finite
    value, or NaN when neither spelling carries one.  Use this instead of
    ``et.get("pastro")``: a manifest or override file that supplies only
    ``p_astro`` -- the spelling this module documents -- must not read as
    "absent" (GW-14).
    """
    if not mapping:
        return float("nan")
    for key in PASTRO_KEYS:
        if key not in mapping:
            continue
        try:
            val = float(mapping[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            return val
    return float("nan")


def resolve_pastro_column(meta: Optional[Mapping[str, Any]], n: int):
    """Element-wise :func:`resolve_pastro` over a store's ``meta`` columns.

    ``meta`` is a mapping of column name -> per-row values (a ``GWCatalog.meta``
    or a merge's in-memory meta dict).  Returns a length-``n`` float array whose
    row *i* is the first finite of ``p_astro[i]``, ``pastro[i]`` -- NaN when the
    store holds neither.  A store written before one of the columns existed, or
    an ingest that only ever saw one spelling, therefore still answers a
    p_astro question correctly.
    """
    import numpy as np

    out = np.full(int(n), np.nan, dtype=float)
    if not meta:
        return out
    for key in reversed(PASTRO_KEYS):        # later keys overwritten by earlier
        if key not in meta:
            continue
        col = np.asarray(meta[key], dtype=float)
        if col.size != out.size:
            continue
        finite = np.isfinite(col)
        out[finite] = col[finite]
    return out


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
            # Precedence is over LAYERS, so a layer must be searched under every
            # spelling of the field before dropping to the next one: an override
            # file saying `p_astro` has to beat an online table saying `pastro`,
            # and vice versa, or the layering silently depends on which key each
            # source happened to use (GW-14).
            override_v = _layer_value(override_ev, field)
            online_v = _layer_value(online_ev, field)
            manifest_v = _layer_value(manifest_ev, field)
            if override_v is not None:
                per_field[field] = {"value": override_v,
                                    "source": _SOURCE_USER_OVERRIDE}
            elif online_v is not None:
                per_field[field] = {"value": online_v,
                                    "source": _SOURCE_ONLINE}
            elif manifest_v is not None:
                per_field[field] = {"value": manifest_v,
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
