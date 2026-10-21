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

#: The date-only names under which LVK published events before every GWTC name
#: took the ``GWyymmdd_hhmmss`` form: GWTC-1's eleven, and the four O3a events
#: with their own discovery papers.  The suffix is the UTC time of the event's
#: GPS time, which is how every GWTC name is formed (GW150914: GPS
#: 1126259462.4 = 2015-09-14 09:50:45 UTC).
#:
#: A date-only name resolves ONLY through this table.  Prefix-matching it
#: against whatever the store happens to hold picked the other event of the
#: day whenever only one of a same-day pair was present (``GW200208`` became
#: the non-member GW200208_222617), and a store carrying one event in several
#: sample-set rows made the match spuriously "ambiguous".  GWTC-2's
#: ``GW190521`` is GW190521_030229, not the same-day GW190521_074359.
HISTORICAL_DATE_ONLY_NAMES = {
    "GW150914": "GW150914_095045",
    "GW151012": "GW151012_095443",
    "GW151226": "GW151226_033853",
    "GW170104": "GW170104_101158",
    "GW170608": "GW170608_020116",
    "GW170729": "GW170729_185629",
    "GW170809": "GW170809_082821",
    "GW170814": "GW170814_103043",
    "GW170817": "GW170817_124104",
    "GW170818": "GW170818_022509",
    "GW170823": "GW170823_131358",
    "GW190412": "GW190412_053044",
    "GW190425": "GW190425_081805",
    "GW190521": "GW190521_030229",
    "GW190814": "GW190814_211039",
}


def resolve_event_name_aliases(
    requested_names: Iterable[str],
    available_names: Iterable[str],
) -> tuple[list[str], dict[str, str]]:
    """Resolve exact names and historical date-only aliases.

    Exact matches always win. A date-only name such as ``GW150914`` resolves
    only when it is one of the historical GWTC names in
    :data:`HISTORICAL_DATE_ONLY_NAMES`, to the event that table names, and only
    when that event is available. Any other date-only name is refused: a date
    does not identify an event (O3 and O4 have many same-day pairs). Full
    timestamped names are never shortened or fuzzily matched.

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
        If a name is missing, a short name is not a historical GWTC name or
        names an unavailable event, or two requested names resolve to the same
        local event.
    """
    # One entry per event: a store lists an event once per sample-set row.
    available = list(dict.fromkeys(str(name) for name in available_names))
    available_set = set(available)
    resolved: list[str] = []
    aliases: dict[str, str] = {}
    failures: list[str] = []

    for requested in (str(name) for name in requested_names):
        if requested in available_set:
            resolved.append(requested)
            continue

        if _SHORT_EVENT_RE.fullmatch(requested):
            same_day = sorted(
                name for name in available if name.startswith(requested + "_")
            )
            hint = f" (available on that date: {same_day})" if same_day else ""
            canonical = HISTORICAL_DATE_ONLY_NAMES.get(requested)
            if canonical is None:
                failures.append(
                    f"{requested}: not a historical date-only GWTC name, so it "
                    f"does not identify one event{hint}; use the full "
                    f"GWyymmdd_hhmmss name"
                )
            elif canonical in available_set:
                aliases[requested] = canonical
                resolved.append(canonical)
            else:
                failures.append(
                    f"{requested}: the historical name of {canonical}, which "
                    f"is not available{hint}"
                )
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
