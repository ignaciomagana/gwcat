"""Helpers for reproducible GWTC population-event selection.

The public PE releases use canonical timestamped names such as
``GW150914_095045``. Historical GWTC-1 tables and user-maintained lists often
use date-only aliases such as ``GW150914``. This module resolves those aliases
without fuzzy matching and provides a single entry point for selecting the
bundled 259-event GWTC-5 BBH/mass-gap population sample.

The bundled population list is authoritative membership. It must not be
intersected with the generic source-mass classifier after name selection: the
list already encodes the official GWTC-5.0 BBH population membership, including
borderline BBH that a PE-median mass cut might drop, and excluding the
``GW190814_211039`` lower-mass-gap system.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, TYPE_CHECKING

from .bbh_allowed_names import BBH_ALL, validate_bbh_allowed_names

if TYPE_CHECKING:  # pragma: no cover
    from .catalog import GWCatalog

_SHORT_EVENT_RE = re.compile(r"^GW\d{6}$")
_FULL_EVENT_RE = re.compile(r"^GW\d{6}_\d{6}$")


def resolve_event_name_aliases(
    requested_names: Iterable[str],
    available_names: Iterable[str],
) -> tuple[list[str], dict[str, str]]:
    """Resolve exact names and unique historical date-only aliases.

    Exact matches always win. A date-only name such as ``GW150914`` may resolve
    to exactly one available name beginning with ``GW150914_``. Full timestamped
    names are never shortened or fuzzily matched.

    Parameters
    ----------
    requested_names:
        Event names from a population list or user input.
    available_names:
        Canonical event names present in a PE directory or :class:`GWCatalog`.

    Returns
    -------
    resolved, aliases:
        ``resolved`` preserves the requested order while replacing short aliases
        by canonical local names. ``aliases`` maps only changed names.

    Raises
    ------
    ValueError
        If a name is missing, a short alias is ambiguous, or two requested names
        resolve to the same local event.
    """
    available = [str(name) for name in available_names]
    available_set = set(available)
    resolved: list[str] = []
    aliases: dict[str, str] = {}
    failures: list[str] = []

    for requested in (str(name) for name in requested_names):
        if requested in available_set:
            resolved.append(requested)
            continue

        if _SHORT_EVENT_RE.fullmatch(requested):
            candidates = sorted(
                name for name in available if name.startswith(requested + "_")
            )
            if len(candidates) == 1:
                aliases[requested] = candidates[0]
                resolved.append(candidates[0])
                continue
            if not candidates:
                failures.append(
                    f"{requested}: no exact match and no {requested}_HHMMSS alias"
                )
            else:
                failures.append(f"{requested}: ambiguous aliases {candidates}")
            continue

        if _FULL_EVENT_RE.fullmatch(requested):
            failures.append(f"{requested}: canonical timestamped name not found")
        else:
            failures.append(f"{requested}: invalid or unavailable event name")

    if failures:
        raise ValueError(
            "Could not resolve population event names:\n  " + "\n  ".join(failures)
        )

    duplicate_targets = sorted(
        name for name, count in Counter(resolved).items() if count > 1
    )
    if duplicate_targets:
        raise ValueError(
            "Population event aliases resolve to duplicate local names: "
            f"{duplicate_targets}"
        )

    return resolved, aliases


def resolve_gwtc5_bbh_population_names(
    available_names: Iterable[str],
    *,
    expected_total: int = 259,
    expected_o4b_count: int = 104,
) -> tuple[list[str], dict[str, str]]:
    """Resolve the bundled GWTC-5 BBH population list locally."""
    validate_bbh_allowed_names(
        expected_total=expected_total,
        expected_o4b_count=expected_o4b_count,
    )
    resolved, aliases = resolve_event_name_aliases(BBH_ALL, available_names)
    if expected_total is not None and len(resolved) != expected_total:
        raise ValueError(
            f"resolved GWTC-5 population count {len(resolved)} != {expected_total}"
        )
    return resolved, aliases


def select_gwtc5_bbh_population(
    catalog: "GWCatalog",
    *,
    waveform_policy: str = "mixed-first",
    approximant: str | None = None,
) -> tuple["GWCatalog", dict[str, str]]:
    """Select the authoritative 259-event GWTC-5 BBH sample.

    This applies only the bundled event membership and waveform policy. It does
    not reapply ``source_class='bbh'`` because the bundled list is already the
    authoritative population membership; re-running the generic source-mass
    classifier could drop borderline BBH that the official sample retains.
    """
    resolved, aliases = resolve_gwtc5_bbh_population_names(catalog.names)
    view = catalog.select(
        allowed_names=resolved,
        allowed_names_authoritative=True,
        source_class=None,
        waveform_policy=waveform_policy,
        approximant=approximant,
    )
    if view.n_events != len(resolved):
        raise ValueError(
            f"selected {view.n_events} population events after resolving "
            f"{len(resolved)} names"
        )
    return view, aliases
