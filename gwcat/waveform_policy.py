"""Waveform / sample-set policy resolution (PR 6).

A store may hold more than one posterior *sample set* per event -- e.g. an
IMRPhenomXPHM analysis, a SEOBNRv5PHM analysis and a combined ``Mixed`` set for
the SAME ``event_name``.  In the ragged store each ``(event, sample_set)`` pair
is a single row (the same layout a single-set event already used); the
sample-set identity lives in the ``meta/`` columns written by
:mod:`gwcat.ingest` (``sample_set_name``, ``waveform``, ``approximant``,
``is_mixed``, ``is_preferred``, ``priority_rank`` ...).

This module resolves, at *select / export* time, WHICH sample set represents
each event under a chosen policy.  It never mutates the store; it returns the
subset of row indices to keep plus a per-row human-readable ``selection_reason``
and a ``homogeneous`` flag (``True`` iff at most one sample set per event
survives).

Policies
--------
``preferred``          : one set per event, chosen by ``is_preferred`` (then the
                         smallest ``priority_rank``).  The default; a no-op for
                         single-sample-set stores.
``mixed-first``        : prefer an ``is_mixed`` set when the event has one, else
                         fall back to ``preferred``.
``strict-approximant`` : require a given ``approximant`` for EVERY selected
                         event; fail loudly (naming the events) if any lacks it.
``all``                : keep every sample set for every event.  The output is
                         then explicitly NOT homogeneous when any event carries
                         more than one set -- callers must record that so a
                         multi-waveform file is never presented as homogeneous.
``event-map``          : (GW-40e) an explicit ``{event: sample_set_name}`` map
                         (:class:`SampleSetMap`, loaded from JSON or from a
                         popsummary's ``events``/``event_sample_IDs``).  An
                         event absent from the map, or whose mapped label the
                         store lacks, FAILS -- unless the map declares a
                         substitute for it, which is then recorded with its
                         reason.  An optional NRSur q-rule (``nrsur_q_rule``)
                         refuses an NRSur7dq4 label whose
                         C00:IMRPhenomXPHM-SpinTaylor posterior puts more than
                         that fraction of its mass at q < 1/6 (the NRSur prior
                         floor), and verifies every declared ``nrsur_q_rule``
                         substitute is justified.  See
                         :func:`resolve_event_map`.

Backward compatibility
-----------------------
A legacy / single-sample-set store has exactly one row per event and no
sample-set metadata columns.  Every event group then has a single row, so
``preferred`` / ``mixed-first`` / ``all`` keep that one row (a genuine no-op).
``strict-approximant`` still needs approximant/waveform metadata and raises a
clear error if the store has none.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: The waveform/sample-set policies understood by :func:`resolve_policy`.
WAVEFORM_POLICIES = ("preferred", "mixed-first", "strict-approximant", "all",
                     "event-map")

#: The NRSur7dq4 mass-ratio prior floor in the GWTC-4.1/5 releases
#: (``UniformInComponentsMassRatio(minimum=0.16666667)``).
NRSUR_Q_FLOOR = 1.0 / 6.0
#: The label the NRSur q-rule falls back to (and measures the q < 1/6 mass on).
NRSUR_Q_RULE_REFERENCE = "IMRPhenomXPHM-SpinTaylor"
#: The reason string a q-rule substitution carries.
NRSUR_Q_RULE_REASON = "nrsur_q_rule"


class SampleSetMapError(ValueError):
    """An event-map cannot be resolved as declared."""


class UnmappedEventError(SampleSetMapError):
    """A selected event has no entry in the sample-set map."""


class MissingSampleSetError(SampleSetMapError):
    """A mapped label is not in the store and no substitute is declared."""


class NRSurQRuleError(SampleSetMapError):
    """An NRSur label violates the q-rule, or a q-rule substitute is unjustified."""


def _label_prefix_base(label: str):
    bits = str(label).split(":")
    return (bits[0] if bits else ""), (bits[1] if len(bits) > 1 else "")


def is_nrsur_label(label: str) -> bool:
    """True for an NRSur7dq4 sample-set label (any prefix / variant)."""
    return _label_prefix_base(label)[1].startswith("NRSur7dq4")


def nrsur_reference_label(label: str) -> str:
    """The XPHM-SpinTaylor label the q-rule uses for an NRSur ``label``."""
    return f"{_label_prefix_base(label)[0]}:{NRSUR_Q_RULE_REFERENCE}"


@dataclass(frozen=True)
class SampleSetMap:
    """An explicit ``{event: sample_set_name}`` map plus declared substitutes.

    ``substitutes[event]`` is ``{"reason": str, "label": str or None,
    "original_label": str or None}``: ``label`` is a fallback used when the
    mapped label is absent from the store; with no ``label`` the entry declares
    that the MAPPED label is already a substitute (e.g. the three NRSur q-rule
    fallbacks in operator decision OD-12), recorded with its reason.
    """
    labels: Dict[str, str]
    substitutes: Dict[str, dict] = field(default_factory=dict)
    source: str = ""
    sha256: str = ""
    meta: dict = field(default_factory=dict)


def _sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _norm_substitutes(raw) -> Dict[str, dict]:
    out = {}
    for ev, v in (raw or {}).items():
        if isinstance(v, str):
            ent = {"reason": v, "label": None, "original_label": None}
        elif isinstance(v, dict):
            if not v.get("reason"):
                raise SampleSetMapError(
                    f"sample-set map: substitute for {ev!r} states no reason; "
                    f"every substitute must say why.")
            ent = {"reason": str(v["reason"]),
                   "label": (None if v.get("label") in (None, "")
                             else str(v["label"])),
                   "original_label": (None if v.get("original_label")
                                      in (None, "")
                                      else str(v["original_label"]))}
        else:
            raise SampleSetMapError(
                f"sample-set map: substitute for {ev!r} must be a reason string "
                f"or a mapping, got {type(v).__name__}.")
        out[str(ev)] = ent
    return out


def sample_set_map_from_mapping(doc: dict, *, source: str = "in-memory",
                                sha256: Optional[str] = None) -> SampleSetMap:
    """Build a :class:`SampleSetMap` from a decoded JSON document.

    Two layouts are accepted: ``{"events": {ev: label}, "substitutes": {...},
    ...}`` or the flat ``{ev: label, ..., "substitutes": {...}}`` (every key
    starting with ``GW`` is an event; other keys are metadata).
    """
    if not isinstance(doc, dict):
        raise SampleSetMapError("sample-set map must be a JSON object.")
    if "events" in doc and isinstance(doc["events"], dict):
        labels = {str(k): str(v) for k, v in doc["events"].items()}
        meta = {k: v for k, v in doc.items()
                if k not in ("events", "substitutes")}
    else:
        labels = {str(k): str(v) for k, v in doc.items()
                  if str(k).startswith("GW")}
        meta = {k: v for k, v in doc.items()
                if not str(k).startswith("GW") and k != "substitutes"}
    bad = [k for k, v in labels.items() if not v or ":" not in v]
    if bad:
        raise SampleSetMapError(
            f"sample-set map: {len(bad)} event(s) map to something that is not "
            f"a sample-set label (e.g. 'C00:NRSur7dq4'): {bad[:10]}")
    subs = _norm_substitutes(doc.get("substitutes"))
    if sha256 is None:
        sha256 = hashlib.sha256(json.dumps(
            {"events": labels, "substitutes": subs}, sort_keys=True
        ).encode()).hexdigest()
    return SampleSetMap(labels=labels, substitutes=subs, source=source,
                        sha256=sha256, meta=meta)


def load_sample_set_map(spec) -> SampleSetMap:
    """Load a sample-set map from a path, a mapping or a :class:`SampleSetMap`.

    * ``.json`` -- see :func:`sample_set_map_from_mapping`;
    * ``.h5`` / ``.hdf5`` / ``.hdf`` -- a popsummary file: its ``events`` and
      ``event_sample_IDs`` (root attributes, or datasets) give the map, with no
      substitutes.
    The file's sha256 is recorded as the map's identity.
    """
    if isinstance(spec, SampleSetMap):
        return spec
    if isinstance(spec, dict):
        return sample_set_map_from_mapping(spec)
    if not isinstance(spec, (str, os.PathLike)):
        raise SampleSetMapError(
            f"sample_set_map must be a path, a mapping or a SampleSetMap; got "
            f"{type(spec).__name__}.")
    path = os.path.abspath(os.fspath(spec))
    sha = _sha256_file(path)
    low = path.lower()
    if low.endswith((".h5", ".hdf5", ".hdf")):
        import h5py

        def _strs(x):
            return [v.decode() if isinstance(v, (bytes, bytearray)) else str(v)
                    for v in np.asarray(x).ravel()]
        with h5py.File(path, "r") as f:
            got = {}
            for key in ("events", "event_sample_IDs"):
                if key in f.attrs:
                    got[key] = _strs(f.attrs[key])
                elif key in f:
                    got[key] = _strs(f[key][()])
        if set(got) != {"events", "event_sample_IDs"}:
            raise SampleSetMapError(
                f"{path}: not a popsummary sample-set map -- it needs both "
                f"'events' and 'event_sample_IDs' (attrs or datasets); found "
                f"{sorted(got)}.")
        ev, ids = got["events"], got["event_sample_IDs"]
        if len(ev) != len(ids):
            raise SampleSetMapError(
                f"{path}: 'events' ({len(ev)}) and 'event_sample_IDs' "
                f"({len(ids)}) differ in length.")
        if len(set(ev)) != len(ev):
            raise SampleSetMapError(f"{path}: duplicate event names.")
        return sample_set_map_from_mapping(
            {"events": dict(zip(ev, ids)), "format": "popsummary"},
            source=path, sha256=sha)
    with open(path) as f:
        doc = json.load(f)
    return sample_set_map_from_mapping(doc, source=path, sha256=sha)


def _decode(x) -> str:
    if isinstance(x, (bytes, bytearray)):
        return x.decode()
    return "" if x is None else str(x)


def _truthy(x) -> bool:
    """Interpret a stored 0.0/1.0 (or NaN) flag as a boolean."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return np.isfinite(v) and v > 0.5


def resolve_policy(event_names: Sequence, sel, meta: dict,
                   policy: str = "preferred",
                   approximant: str | None = None,
                   sample_set_map=None, q_frac_fn=None, nrsur_q_rule=None,
                   nrsur_q_rule_substitute: bool = False
                   ) -> Tuple[np.ndarray, List[str], bool]:
    """Resolve the active waveform/sample-set policy over a selection.

    Parameters
    ----------
    event_names : 1-D array-like of str
        Event name for EVERY row in the store (length == number of store rows).
    sel : 1-D int array-like
        Row indices currently selected (metadata cuts already applied).
    meta : dict {field: 1-D array}
        The store's meta columns.  ``sample_set_name``, ``waveform``,
        ``approximant``, ``is_mixed``, ``is_preferred`` and ``priority_rank`` are
        used when present; a missing column falls back to an explicit-absence
        default so single-sample-set stores resolve as a no-op.
    policy : str
        One of :data:`WAVEFORM_POLICIES`.
    approximant : str or None
        Required for ``strict-approximant``; ignored otherwise.

    Returns
    -------
    kept : 1-D int ndarray
        Row indices kept -- a subset of ``sel`` in ``sel`` order, grouped by
        event.  One row per event unless ``policy == "all"``.
    reasons : list of str
        Per-kept-row selection reason (aligned with ``kept``).
    homogeneous : bool
        ``True`` iff no event contributes more than one kept row (always ``True``
        except possibly under ``all``).

    Raises
    ------
    ValueError
        For an unknown policy, for ``strict-approximant`` without an
        ``approximant`` (or without approximant/waveform metadata), or when
        ``strict-approximant`` cannot satisfy the requested approximant for one
        or more selected events (the message names them).
    """
    if policy not in WAVEFORM_POLICIES:
        raise ValueError(
            f"waveform_policy={policy!r} is invalid; choose one of "
            f"{WAVEFORM_POLICIES}.")
    if policy == "event-map":
        kept, reasons, homogeneous, _report = resolve_event_map(
            event_names, sel, meta, sample_set_map, q_frac_fn=q_frac_fn,
            nrsur_q_rule=nrsur_q_rule,
            nrsur_q_rule_substitute=nrsur_q_rule_substitute)
        return kept, reasons, homogeneous
    if sample_set_map is not None or nrsur_q_rule is not None:
        raise ValueError(
            f"sample_set_map / nrsur_q_rule apply only to "
            f"waveform_policy='event-map', not {policy!r}.")

    sel = np.asarray(sel, dtype=int)
    names = np.asarray(event_names)

    def col(field):
        v = meta.get(field)
        return None if v is None else np.asarray(v)

    approx = col("approximant")
    wf = col("waveform")
    is_mixed = col("is_mixed")
    is_pref = col("is_preferred")
    rank = col("priority_rank")

    if policy == "strict-approximant":
        if approximant is None:
            raise ValueError(
                "waveform_policy='strict-approximant' requires approximant=... "
                "(e.g. approximant='IMRPhenomXPHM').")
        if approx is None and wf is None:
            raise ValueError(
                "waveform_policy='strict-approximant' needs approximant/waveform "
                "metadata, but the store has neither column. Re-ingest recording "
                "sample-set metadata (build_store(..., sample_sets=...)).")

    def rank_of(r: int) -> float:
        # smaller is more preferred; NaN / absent rank -> +inf (least preferred)
        if rank is None:
            return np.inf
        try:
            v = float(rank[r])
        except (TypeError, ValueError):
            return np.inf
        return v if np.isfinite(v) else np.inf

    def approx_matches(r: int, want: str) -> bool:
        got_a = _decode(approx[r]) if approx is not None else ""
        got_w = _decode(wf[r]) if wf is not None else ""
        return want == got_a or want == got_w

    # Group selected rows by event, preserving first-seen (== sel) order.
    order: List = []
    groups: dict = {}
    for r in sel:
        nm = names[r]
        key = nm.item() if hasattr(nm, "item") else nm
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(int(r))

    kept: List[int] = []
    reasons: List[str] = []
    missing_strict: List = []

    for key in order:
        rows = groups[key]

        if policy == "all":
            for r in rows:
                kept.append(r)
                reasons.append("all:kept")
            continue

        if policy == "strict-approximant":
            matches = [r for r in rows if approx_matches(r, approximant)]
            if not matches:
                missing_strict.append(key)
                continue
            best = min(matches, key=rank_of)
            kept.append(best)
            reasons.append(f"strict-approximant:{approximant}")
            continue

        if policy == "mixed-first":
            mixed_rows = [r for r in rows
                          if is_mixed is not None and _truthy(is_mixed[r])]
            if mixed_rows:
                best = min(mixed_rows, key=rank_of)
                kept.append(best)
                reasons.append("mixed-first:is_mixed")
                continue
            # else fall through to the preferred logic below

        # preferred (and the mixed-first fallback)
        pref_rows = [r for r in rows
                     if is_pref is not None and _truthy(is_pref[r])]
        if pref_rows:
            best = min(pref_rows, key=rank_of)
            why = "is_preferred"
        elif rank is not None and any(np.isfinite(rank_of(r)) for r in rows):
            best = min(rows, key=rank_of)
            why = "min_priority_rank"
        else:
            best = rows[0]
            why = "single_sample_set" if len(rows) == 1 else "first_available"
        prefix = ("mixed-first:fallback_preferred:" if policy == "mixed-first"
                  else "preferred:")
        kept.append(best)
        reasons.append(prefix + why)

    if missing_strict:
        raise ValueError(
            f"waveform_policy='strict-approximant' with approximant="
            f"{approximant!r}: {len(missing_strict)} selected event(s) have no "
            f"sample set with that approximant: "
            f"{sorted(str(k) for k in missing_strict)}. Choose a different "
            f"approximant, drop those events, or use waveform_policy='all' / "
            f"'preferred'.")

    kept_arr = np.asarray(kept, dtype=int)
    kept_names = names[kept_arr] if kept_arr.size else names[:0]
    homogeneous = len(set(kept_names.tolist())) == kept_arr.size
    return kept_arr, reasons, bool(homogeneous)


def resolve_event_map(event_names: Sequence, sel, meta: dict, sample_set_map,
                      *, q_frac_fn: Optional[Callable] = None,
                      nrsur_q_rule: Optional[float] = None,
                      nrsur_q_rule_substitute: bool = False):
    """Resolve ``waveform_policy="event-map"`` (GW-40e).

    For every selected event: the map's label if the store has it; else the
    map's declared substitute label (recorded with its reason); else FAIL.  An
    event absent from the map fails too.  Every failure of one kind is
    collected and reported together, naming the events.

    NRSur q-rule (``nrsur_q_rule=frac``; requires ``q_frac_fn(event, label)``
    returning the fraction of ``label``'s posterior samples with
    ``m2/m1 < 1/6``, or None when the store has no such row):

    * an NRSur7dq4 final label whose ``{prefix}:IMRPhenomXPHM-SpinTaylor``
      fraction exceeds ``frac`` is REFUSED (:class:`NRSurQRuleError`), or --
      with ``nrsur_q_rule_substitute=True`` -- replaced by that XPHM-ST label
      and recorded as a substitute with reason ``"nrsur_q_rule"``;
    * every substitute the map DECLARES with a reason starting
      ``"nrsur_q_rule"`` is verified to exceed ``frac`` (else refused: a
      substitution the rule does not justify is not the rule).

    Returns ``(kept, reasons, homogeneous, report)`` -- ``report`` records the
    map's source/sha256, every substitute with its reason and original label,
    the rule threshold and every measured q < 1/6 fraction.
    """
    if sample_set_map is None:
        raise SampleSetMapError(
            "waveform_policy='event-map' requires sample_set_map= (a JSON "
            "file, a popsummary file, a mapping, or a SampleSetMap).")
    smap = load_sample_set_map(sample_set_map)
    if nrsur_q_rule is not None:
        thr = float(nrsur_q_rule)
        if not (0.0 <= thr < 1.0):
            raise ValueError(f"nrsur_q_rule must be a fraction in [0, 1); "
                             f"got {nrsur_q_rule!r}.")
        if q_frac_fn is None:
            raise ValueError("nrsur_q_rule needs q_frac_fn (the store's "
                             "posterior samples) to be evaluated.")
    sel = np.asarray(sel, dtype=int)
    names = np.asarray(event_names)
    ss = meta.get("sample_set_name")
    if ss is None:
        raise SampleSetMapError(
            "waveform_policy='event-map' needs the store's sample_set_name "
            "column; this store has none (re-ingest with sample-set metadata).")
    ss = np.asarray(ss)

    order: List = []
    groups: dict = {}
    for r in sel:
        nm = names[r]
        key = nm.item() if hasattr(nm, "item") else nm
        key = key.decode() if isinstance(key, (bytes, bytearray)) else str(key)
        if key not in groups:
            groups[key] = {}
            order.append(key)
        groups[key][_decode(ss[r])] = int(r)

    unmapped, missing = [], []
    q_viol, q_unjust, q_noref = [], [], []
    chosen: Dict[str, Tuple[int, str]] = {}
    substitutes: Dict[str, dict] = {}
    q_frac: Dict[str, float] = {}

    for ev in order:
        rows = groups[ev]
        if ev not in smap.labels:
            unmapped.append(ev)
            continue
        want = smap.labels[ev]
        sub = smap.substitutes.get(ev)
        if want in rows:
            label = want
            if sub is not None:
                substitutes[ev] = {"label": label, "reason": sub["reason"],
                                   "original_label": sub["original_label"]
                                   or ""}
        elif sub is not None and sub["label"] and sub["label"] in rows:
            label = sub["label"]
            substitutes[ev] = {"label": label, "reason": sub["reason"],
                               "original_label": sub["original_label"] or want}
        else:
            missing.append((ev, want, sorted(rows)))
            continue

        if nrsur_q_rule is not None:
            reason = substitutes.get(ev, {}).get("reason", "")
            declared_q = reason.startswith(NRSUR_Q_RULE_REASON)
            if is_nrsur_label(label) or declared_q:
                ref = (nrsur_reference_label(label) if is_nrsur_label(label)
                       else label)
                frac = q_frac_fn(ev, ref)
                if frac is None:
                    q_noref.append((ev, ref))
                    continue
                q_frac[ev] = float(frac)
                if is_nrsur_label(label) and frac > thr:
                    if nrsur_q_rule_substitute and ref in rows:
                        substitutes[ev] = {
                            "label": ref, "reason": NRSUR_Q_RULE_REASON,
                            "original_label": label}
                        label = ref
                    else:
                        q_viol.append((ev, label, frac))
                        continue
                elif declared_q and not is_nrsur_label(label) and frac <= thr:
                    q_unjust.append((ev, label, frac))
                    continue
        chosen[ev] = (rows[label], label)

    if unmapped:
        raise UnmappedEventError(
            f"waveform_policy='event-map': {len(unmapped)} selected event(s) "
            f"have no entry in the sample-set map {smap.source or '(in-memory)'}"
            f": {unmapped[:20]}" + ("" if len(unmapped) <= 20
                                    else f" (+{len(unmapped) - 20} more)")
            + ". Map every event explicitly (or drop it from the selection); "
              "gwcat will not choose a label for an unmapped event.")
    if missing:
        listed = "; ".join(f"{ev}: wants {w!r}, store has {have}"
                           for ev, w, have in missing[:10])
        raise MissingSampleSetError(
            f"waveform_policy='event-map': {len(missing)} event(s) map to a "
            f"sample set the store does not hold, and the map declares no "
            f"substitute for them: {listed}"
            + ("" if len(missing) <= 10 else f" (+{len(missing) - 10} more)")
            + ". Add a substitutes entry {event: {'label': ..., 'reason': ...}}"
              " to the map, or re-ingest with --sample-sets all.")
    if q_noref:
        raise NRSurQRuleError(
            f"nrsur_q_rule: cannot evaluate the rule for {len(q_noref)} "
            f"event(s) -- the store has no reference sample set: "
            f"{q_noref[:10]}. Ingest with --sample-sets all.")
    if q_viol:
        listed = ", ".join(f"{ev} ({lab}): {100 * fr:.2f}%"
                           for ev, lab, fr in q_viol[:10])
        raise NRSurQRuleError(
            f"nrsur_q_rule={nrsur_q_rule:g}: {len(q_viol)} NRSur7dq4 label(s) "
            f"put more than {100 * float(nrsur_q_rule):.2f}% of their "
            f"{NRSUR_Q_RULE_REFERENCE} posterior mass at q < 1/6, below the "
            f"NRSur prior floor: {listed}. Declare the fallback as a "
            f"substitute in the map (reason 'nrsur_q_rule'), or pass "
            f"nrsur_q_rule_substitute=True to substitute automatically.")
    if q_unjust:
        listed = ", ".join(f"{ev} ({lab}): {100 * fr:.2f}%"
                           for ev, lab, fr in q_unjust[:10])
        raise NRSurQRuleError(
            f"nrsur_q_rule={nrsur_q_rule:g}: {len(q_unjust)} declared "
            f"'nrsur_q_rule' substitute(s) are NOT justified by the rule (the "
            f"q < 1/6 mass does not exceed the threshold): {listed}.")

    kept, reasons = [], []
    for ev in order:
        r, label = chosen[ev]
        kept.append(r)
        if ev in substitutes:
            reasons.append(f"event-map:substitute({substitutes[ev]['reason']})"
                           f":{label}")
        else:
            reasons.append(f"event-map:mapped:{label}")
    kept_arr = np.asarray(kept, dtype=int)
    report = {
        "map_source": smap.source,
        "map_sha256": smap.sha256,
        "n_mapped_events": int(len(kept)),
        "n_map_entries_unused": int(len(set(smap.labels) - set(order))),
        "substitutes": substitutes,
        "nrsur_q_rule": None if nrsur_q_rule is None else float(nrsur_q_rule),
        "nrsur_q_rule_substitute": bool(nrsur_q_rule_substitute),
        "nrsur_q_frac": q_frac,
    }
    return kept_arr, reasons, True, report
