"""Validation-summary outputs for ingest/export (PR 10).

The handoff's "Validation Outputs" section asks every ingest/export to
produce a machine-readable ``validation_summary.json`` and a human-readable
``validation_summary.md`` next to the output file, covering: events
discovered/ingested/skipped, source-class counts, sample-set counts per
event, waveform/approximant counts, missing required/optional parameters,
missing-FAR status, p_astro availability, prior mode, cosmology mode, output
schema version, package version, manifest name/version (when known), and
source file checksums (when known).

This module has two layers:

  * :func:`summarize_catalog` -- generic, read-only diagnostics computed
    straight from a :class:`gwcat.catalog.GWCatalog` view's already-loaded
    metadata/availability arrays.  It never fabricates a field: a value not
    present in the store (e.g. no ``waveform`` column, no per-event
    cosmology) is simply omitted or reported as an explicit ``None`` /
    ``"unknown"``.  Reused by :func:`gwcat.ingest.build_store`'s
    ``write_summary=True`` path (over the freshly written store), by the
    ``gwcat inspect`` CLI command (over an existing store), and by
    :meth:`gwcat.catalog.GWCatalog.to_darksirens`'s ``write_summary=True``
    path (over the post-``select()`` view, before overlaying export-specific
    fields such as ``spin_prior_mode`` / ``cosmology_mode`` / ``far_policy``).

  * :func:`write_validation_summary` / :func:`render_markdown` -- format- and
    write a caller-assembled summary ``dict`` (any "kind": ``"ingest"``,
    ``"darksirens_export"``, ``"selection_export"``, ...) as JSON + Markdown,
    both serializing exactly the same data (the ``.md`` is a rendering of the
    same dict as the ``.json``, not an independent computation).

No network access; no dependency on which "kind" of summary is being built.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

__all__ = [
    "package_version",
    "value_counts",
    "summarize_catalog",
    "v2_export_summary_additions",
    "render_markdown",
    "write_validation_summary",
]


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
def package_version() -> str:
    """Return the installed gwcat package version, or ``"unknown"``.

    Uses :mod:`importlib.metadata` (works for both regular and editable
    installs) rather than importing :mod:`gwcat` itself, so this module never
    risks a circular import against the top-level package `__init__`.
    """
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version("gwcat")
        except PackageNotFoundError:
            pass
    except Exception:
        pass
    try:
        from . import __version__ as v  # local import: only reached as a fallback
        return str(v)
    except Exception:
        return "unknown"


def _norm_label(x: Any) -> str:
    if isinstance(x, (bytes, bytearray)):
        x = x.decode()
    s = str(x)
    return s if s != "" else "unknown"


def value_counts(seq: Iterable[Any]) -> Dict[str, int]:
    """Return ``{label: count}`` for an iterable of (possibly bytes) labels.

    Empty-string labels are reported under the explicit key ``"unknown"``
    (an honest name for "not recorded"), never silently dropped. Sorted by
    descending count, then label, for stable/reproducible output.
    """
    counts = Counter(_norm_label(x) for x in seq)
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _json_default(o: Any):
    """``json.dump(..., default=...)`` handler for numpy / bytes / set values."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bytes, bytearray)):
        return o.decode()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return str(o)


# ---------------------------------------------------------------------------
# Generic store/catalog diagnostics
# ---------------------------------------------------------------------------
def summarize_catalog(cat, parameter_space: str = None) -> Dict[str, Any]:
    """Read-only diagnostics for a (possibly already-``select()``ed) GWCatalog.

    Parameters
    ----------
    cat : gwcat.catalog.GWCatalog
        Any view -- a fresh, unfiltered catalog over a just-written store
        (ingest context) or the result of ``.select(...)`` (export context).
    parameter_space : str, optional
        The space the summary is being written for.  ``missing_required_
        parameters`` is computed against *that* space's requirements.  Omitted
        (the ingest context, where no export is in view) it falls back to the
        legacy ``DARKSIRENS_REQUIRED`` list.

        This mattered: the requirement list was hardcoded to the chi_eff-only
        set, so a component-basis export -- which needs ``a_1``/``a_2``/the
        tilts -- reported no missing parameters on a store that could not
        supply them, and the summary said the export was fine right up until
        the builder raised.

    Returns
    -------
    dict
        Only ever contains fields actually derivable from ``cat``'s already
        -loaded metadata/availability arrays and the store's own HDF5 attrs
        -- nothing here is fabricated. See the module docstring for how
        ingest/export call sites overlay additional, context-specific keys
        on top of this dict.
    """
    import h5py
    from .schema import DARKSIRENS_REQUIRED, ALL_GROUP_PARAMS, \
        export_requirements_for

    sel = np.asarray(cat._sel)
    names = np.asarray(cat.event_names)
    n_events = int(sel.size)

    with h5py.File(cat.path, "r") as f:
        schema_version = f.attrs.get("schema_version", "unknown")
    if isinstance(schema_version, bytes):
        schema_version = schema_version.decode()

    def _meta_col(field):
        v = cat.meta.get(field)
        return None if v is None else np.asarray(v)[sel]

    source_class_counts = value_counts(cat.source_class[sel]) if n_events else {}

    waveform_counts: Dict[str, int] = {}
    wf = _meta_col("waveform")
    if wf is not None:
        waveform_counts = value_counts(wf)

    approximant_counts: Dict[str, int] = {}
    approx = _meta_col("approximant")
    if approx is not None:
        approximant_counts = value_counts(approx)

    row_counts = Counter(names.tolist())
    sample_set_counts_per_event = dict(row_counts)
    n_events_with_multiple_sample_sets = sum(1 for v in row_counts.values() if v > 1)

    far_avail = _meta_col("far_available")
    if far_avail is not None:
        far_missing = int((np.asarray(far_avail, dtype=float) <= 0.5).sum())
    else:
        far = _meta_col("far")
        far_missing = (int((~np.isfinite(np.asarray(far, dtype=float))).sum())
                       if far is not None else n_events)

    # Resolved per ROW, not per column: an older store populated `pastro` and
    # left `p_astro` NaN, and a column-level fallback that only fires when the
    # whole `p_astro` column is ABSENT reported "no p_astro" for every one of
    # those events (GW-14).
    p_astro_available = int(np.isfinite(cat._pastro_column()[sel]).sum())

    stored = list(cat.params)
    if parameter_space is None:
        required = DARKSIRENS_REQUIRED
    else:
        required = export_requirements_for(parameter_space)
    missing_required = [p for p in required if p not in stored]
    missing_optional = [p for p in ALL_GROUP_PARAMS
                        if p not in stored and p not in missing_required]

    partial_availability: Dict[str, int] = {}
    if n_events:
        for j, p in enumerate(cat.params):
            n_missing = int((~cat.avail[sel, j]).sum())
            if n_missing:
                partial_availability[p] = n_missing

    H0 = _meta_col("dL_prior_H0")
    Om0 = _meta_col("dL_prior_Om0")
    per_event_cosmology_present = H0 is not None and Om0 is not None
    per_event_cosmology_varies = None
    if per_event_cosmology_present and H0.size:
        H0f = np.asarray(H0, dtype=float)
        Om0f = np.asarray(Om0, dtype=float)
        finite = np.isfinite(H0f) & np.isfinite(Om0f)
        per_event_cosmology_varies = bool(
            finite.any()
            and (np.ptp(H0f[finite]) > 0 or np.ptp(Om0f[finite]) > 0))

    return {
        "package_version": package_version(),
        "schema_version": schema_version,
        "n_events": n_events,
        "event_names": names.tolist(),
        "stored_parameters": stored,
        # Which space the requirement list came from, so a reader can tell a
        # clean chi_eff summary from a clean component one.
        "required_for_parameter_space": parameter_space,
        "missing_required_parameters": missing_required,
        "missing_optional_parameters": missing_optional,
        "partial_availability": partial_availability,
        "source_class_counts": source_class_counts,
        "waveform_counts": waveform_counts,
        "approximant_counts": approximant_counts,
        "sample_set_counts_per_event": sample_set_counts_per_event,
        "n_events_with_multiple_sample_sets": n_events_with_multiple_sample_sets,
        "far_missing_count": far_missing,
        "far_available_count": n_events - far_missing,
        "p_astro_available_count": p_astro_available,
        "per_event_cosmology_present": per_event_cosmology_present,
        "per_event_cosmology_varies": per_event_cosmology_varies,
    }


# ---------------------------------------------------------------------------
# gwcat-2.0 export summary enrichment (additive; v1 summaries unchanged)
# ---------------------------------------------------------------------------
#: The v2 export ``format_version`` values whose summaries get enriched.  2.1 is
#: 2.0 plus the contract attrs, so it enriches identically; listing only 2.0
#: silently produced a bare summary for every 2.1 file.
_V2_PE_FORMAT = "gwcat-pe-2.0"
_V2_SELECTION_FORMAT = "gwcat-selection-2.0"
_V2_PE_FORMATS = ("gwcat-pe-2.0", "gwcat-pe-2.1")
_V2_SELECTION_FORMATS = ("gwcat-selection-2.0", "gwcat-selection-2.1")


def _decode(v: Any) -> Any:
    return v.decode() if isinstance(v, (bytes, bytearray)) else v


def _summarize_amax(arr) -> Optional[Dict[str, Any]]:
    """Summarize a per-event/per-campaign spin-amax array as min/max/uniques.

    Returns ``None`` for an empty/all-NaN array so the caller can omit the
    field entirely rather than emit a meaningless placeholder.
    """
    a = np.asarray(arr, dtype=float).ravel()
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return None
    uniques = sorted({round(float(x), 6) for x in finite})
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "unique_values": uniques,
    }


def v2_export_summary_additions(out_path) -> Dict[str, Any]:
    """Additive validation-summary fields for a gwcat-2.0 export file.

    Reads the just-written ``gwcat-pe-2.0`` / ``gwcat-selection-2.0`` HDF5 file's
    provenance attrs and returns the spin-basis / per-basis prior-provenance /
    injected-spin fields the v2 summaries should carry.  Returns ``{}`` for any
    non-v2 file (v1 darksirens / v1 selection / ingest store.h5), a missing
    file, or any read error -- so v1 summaries and non-HDF5 summary writes are
    left completely unchanged.

    This deliberately reads the file (rather than the builder's summary feed) so
    the enrichment is fully self-contained in this module: the writer serializes
    the file first, then hands its path to :func:`write_validation_summary`,
    which is where these fields are merged in (additively; existing keys win).
    """
    import os
    if not out_path or not os.path.exists(str(out_path)):
        return {}
    try:
        import h5py
        with h5py.File(str(out_path), "r") as f:
            attrs = {k: _decode(f.attrs[k]) for k in f.attrs}
    except Exception:
        return {}

    fmt = attrs.get("format_version")
    if fmt in _V2_PE_FORMATS:
        return _v2_pe_additions(attrs)
    if fmt in _V2_SELECTION_FORMATS:
        return _v2_selection_additions(attrs)
    return {}


def _v2_pe_additions(attrs: Dict[str, Any]) -> Dict[str, Any]:
    basis = attrs.get("spin_basis")
    out: Dict[str, Any] = {"spin_basis": basis}
    provenance: Dict[str, Any] = {}
    amax: Dict[str, Any] = {}
    if basis == "chieff":
        provenance["chi_eff_prior_applied_to_p_pe"] = bool(
            attrs.get("chi_eff_prior_applied_to_p_pe", False))
        if "chi_eff_amax" in attrs:
            amax["chi_eff_amax"] = float(attrs["chi_eff_amax"])
    elif basis == "component":
        provenance["component_spin_prior_applied_to_p_pe"] = bool(
            attrs.get("component_spin_prior_applied_to_p_pe", False))
        for key, label in (("spin_amax_1_per_event", "spin_amax_1"),
                           ("spin_amax_2_per_event", "spin_amax_2")):
            s = _summarize_amax(attrs.get(key, []))
            if s is not None:
                amax[label] = s
    elif basis == "chieff_chip":
        provenance["chi_eff_chi_p_prior_applied_to_p_pe"] = bool(
            attrs.get("chi_eff_chi_p_prior_applied_to_p_pe", False))
        s = _summarize_amax(attrs.get("chi_eff_chi_p_amax_per_event", []))
        if s is not None:
            amax["chi_eff_chi_p_amax_per_event"] = s
    if provenance:
        out["spin_prior_provenance"] = provenance
    if amax:
        out["spin_amax_summary"] = amax
    return out


def _v2_selection_additions(attrs: Dict[str, Any]) -> Dict[str, Any]:
    import json as _json
    out: Dict[str, Any] = {"spin_basis": attrs.get("spin_basis")}
    fmt = attrs.get("injected_spin_format")
    if fmt is not None:
        out["injected_spin_format"] = [_decode(x) for x in np.asarray(fmt).ravel()]
    uni = attrs.get("injected_spin_uniform_isotropic")
    if uni is not None:
        out["injected_spin_uniform_isotropic"] = [
            bool(x) for x in np.asarray(uni).ravel()]
    amax_det = attrs.get("injected_spin_amax_detected")
    if amax_det is not None:
        out["injected_spin_amax_detected"] = np.asarray(
            amax_det, dtype=float).tolist()
    checks = attrs.get("injected_spin_checks")
    if checks is not None:
        try:
            out["injected_spin_checks"] = _json.loads(_decode(checks))
        except Exception:
            out["injected_spin_checks"] = _decode(checks)
    if "pdraw_state" in attrs:
        out["pdraw_state"] = _decode(attrs["pdraw_state"])
    return out


# ---------------------------------------------------------------------------
# Rendering + writing
# ---------------------------------------------------------------------------
def _render_field(key: str, value: Any) -> str:
    if isinstance(value, dict):
        if not value:
            return f"- **{key}**: (none)"
        items = list(value.items())
        shown, more = (items, 0) if len(items) <= 25 else (items[:25], len(items) - 25)
        sub = "\n".join(f"  - `{k}`: {v}" for k, v in shown)
        if more:
            sub += f"\n  - ... ({more} more)"
        return f"- **{key}**:\n{sub}"
    if isinstance(value, (list, tuple)):
        if not value:
            return f"- **{key}**: (none)"
        if len(value) > 25:
            shown = ", ".join(str(v) for v in value[:25])
            return f"- **{key}** ({len(value)} total): {shown}, ..."
        return f"- **{key}**: {', '.join(str(v) for v in value)}"
    return f"- **{key}**: {value}"


def render_markdown(summary: Dict[str, Any], title: str = "Validation Summary") -> str:
    """Render a summary dict as Markdown -- the same data as the JSON file,
    never an independent computation."""
    kind = summary.get("kind", "summary")
    out = summary.get("output_path", "")
    lines = [f"# {title}", "", f"- **kind**: `{kind}`", f"- **output**: `{out}`", ""]
    lines.append("## Fields")
    lines.append("")
    for key, value in summary.items():
        if key in ("kind", "output_path"):
            continue
        lines.append(_render_field(key, value))
    lines.append("")
    return "\n".join(lines)


def write_validation_summary(
    out_path, summary: Dict[str, Any], write_md: bool = True,
) -> Tuple[str, Optional[str]]:
    """Write ``<out_path>.validation_summary.json`` (+ ``.md``) next to ``out_path``.

    Both files serialize the SAME ``summary`` dict -- the ``.md`` is a
    human-rendering of the ``.json``, never a separately computed summary.
    Returns ``(json_path, md_path_or_None)``.

    For a gwcat-2.0 export (``out_path`` is a ``gwcat-pe-2.0`` /
    ``gwcat-selection-2.0`` file), the summary is enriched *additively* with the
    spin-basis / prior-provenance / injected-spin fields read back from that
    file (see :func:`v2_export_summary_additions`).  The enrichment never
    overrides a field the caller already supplied, and is a no-op for v1 exports,
    ingest stores, and non-HDF5 summary writes, so those summaries are unchanged.
    """
    out_path = str(out_path)

    additions = v2_export_summary_additions(out_path)
    if additions:
        # Additive only: caller-supplied fields win over the reconstructed ones.
        summary = {**additions, **summary}

    json_path = out_path + ".validation_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
        f.write("\n")

    md_path = None
    if write_md:
        md_path = out_path + ".validation_summary.md"
        with open(md_path, "w") as f:
            f.write(render_markdown(summary))
    return json_path, md_path
