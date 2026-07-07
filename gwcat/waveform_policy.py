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

Backward compatibility
-----------------------
A legacy / single-sample-set store has exactly one row per event and no
sample-set metadata columns.  Every event group then has a single row, so
``preferred`` / ``mixed-first`` / ``all`` keep that one row (a genuine no-op).
``strict-approximant`` still needs approximant/waveform metadata and raises a
clear error if the store has none.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

#: The waveform/sample-set policies understood by :func:`resolve_policy`.
WAVEFORM_POLICIES = ("preferred", "mixed-first", "strict-approximant", "all")


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
                   approximant: str | None = None) -> Tuple[np.ndarray,
                                                             List[str], bool]:
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
