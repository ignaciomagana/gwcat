"""Source-class contract for gwcat (PR 2).

The package supports these canonical compact-binary source classes::

    BBH        binary black hole
    NSBH       neutron-star--black-hole
    BNS        binary neutron star
    MassGap    ambiguous / lower-mass-gap system
    Unknown    unclassified

Source class is treated as *per-event metadata*, not something inferred from a
static event-name list alone.  The full metadata model (see the handoff doc) is::

    event_name, release, observing_run,
    source_class, source_class_method, source_class_reference,
    p_astro, p_bbh, p_nsbh, p_bns, p_terr,
    far, far_available, metadata_source

This module centralises:

  * the canonical class labels and their normalisation,
  * the mapping from CLI/selection keywords (``bbh``/``nsbh``/``bns``/``cbc``)
    to sets of canonical classes,
  * ``SourceClassMeta``, a dataclass capturing the per-event model with
    explicit-absence defaults, and
  * ``load_event_list`` for user-supplied event-list files.

None of the selection helpers here require network access.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Set, Union

# ── Canonical class labels ───────────────────────────────────────────────────
BBH = "BBH"
NSBH = "NSBH"
BNS = "BNS"
MASSGAP = "MassGap"
UNKNOWN = "Unknown"

#: Canonical source classes, in a stable order.
SOURCE_CLASSES = (BBH, NSBH, BNS, MASSGAP, UNKNOWN)

# Compact-key (lowercased, punctuation-stripped) -> canonical label.
_CLASS_ALIASES = {
    "bbh": BBH,
    "binaryblackhole": BBH,
    "nsbh": NSBH,
    "bhns": NSBH,
    "neutronstarblackhole": NSBH,
    "bns": BNS,
    "binaryneutronstar": BNS,
    "massgap": MASSGAP,
    "ambiguous": MASSGAP,
    "lowermassgap": MASSGAP,
    "unknown": UNKNOWN,
    "unclassified": UNKNOWN,
    "": UNKNOWN,
}

# Selection keyword -> set of canonical classes it admits.  ``cbc`` means "all
# compact-binary classes" and is deliberately permissive so that a mixed
# BBH/NSBH/BNS catalog is returned whole.
_FILTER_MAP = {
    "bbh": frozenset({BBH}),
    "nsbh": frozenset({NSBH}),
    "bns": frozenset({BNS}),
    "massgap": frozenset({MASSGAP}),
    "cbc": frozenset({BBH, NSBH, BNS, MASSGAP, UNKNOWN}),
    "all": frozenset(SOURCE_CLASSES),
}


def _compact_key(label) -> str:
    if label is None:
        return ""
    if isinstance(label, bytes):
        label = label.decode("utf-8", "replace")
    return re.sub(r"[\s_\-/]", "", str(label).strip().lower())


def normalize_source_class(label) -> str:
    """Return the canonical source-class label for an arbitrary input.

    Case/punctuation-insensitive.  Unrecognised or empty values normalise to
    :data:`UNKNOWN` (explicit-absence), never to a crash.
    """
    return _CLASS_ALIASES.get(_compact_key(label), UNKNOWN)


def resolve_filter_classes(
    source_class: Union[str, Iterable[str], None],
) -> Set[str]:
    """Resolve a selection request into a set of canonical class labels.

    Accepts the keywords ``bbh``/``nsbh``/``bns``/``massgap``/``cbc``/``all``,
    a canonical class name, or an iterable of any of those.  ``None`` means "no
    source-class restriction" and returns every canonical class.
    """
    if source_class is None:
        return set(SOURCE_CLASSES)
    if isinstance(source_class, (list, tuple, set, frozenset)):
        out: Set[str] = set()
        for item in source_class:
            out |= resolve_filter_classes(item)
        return out
    key = _compact_key(source_class)
    if key in _FILTER_MAP:
        return set(_FILTER_MAP[key])
    # Fall back to interpreting it as a single canonical class name.
    return {normalize_source_class(source_class)}


def load_event_list(source: Union[str, Path, Sequence[str]]) -> list:
    """Load an event-name list from a file path or an in-memory sequence.

    A file is parsed one event name per line; blank lines and ``#`` comments
    (including trailing inline comments) are ignored.  A list/tuple/set is
    returned as a list of strings unchanged.  Order is preserved and duplicates
    are dropped while keeping first occurrence.
    """
    if isinstance(source, (list, tuple, set, frozenset)):
        raw = [str(x) for x in source]
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"event list file not found: {source}")
        raw = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                raw.append(line)
    seen: Set[str] = set()
    names = []
    for name in raw:
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


@dataclass
class SourceClassMeta:
    """Per-event source-class metadata with explicit-absence defaults.

    Floats default to NaN and strings to ``""`` so that "we do not know this"
    is represented explicitly rather than by a fabricated value.  ``far`` and
    ``far_available`` are decoupled on purpose: a public interface may expose
    ``p_astro`` and preferred-sample links without a machine-readable FAR, and
    that state must round-trip as ``far_available=False`` rather than pretending
    a FAR exists.
    """

    event_name: str = ""
    release: str = ""
    observing_run: str = ""
    source_class: str = UNKNOWN
    source_class_method: str = ""
    source_class_reference: str = ""
    p_astro: float = float("nan")
    p_bbh: float = float("nan")
    p_nsbh: float = float("nan")
    p_bns: float = float("nan")
    p_terr: float = float("nan")
    far: float = float("nan")
    far_available: bool = False
    metadata_source: str = ""

    def __post_init__(self):
        self.source_class = normalize_source_class(self.source_class)
        # far_available must never claim a FAR that is not finite.
        if self.far_available and not math.isfinite(self.far):
            self.far_available = False
        # A finite FAR implies availability unless explicitly overridden below.
        if math.isfinite(self.far):
            self.far_available = True

    #: Float meta fields contributed to the HDF5 ``meta/`` group.
    FLOAT_FIELDS = (
        "p_astro", "p_bbh", "p_nsbh", "p_bns", "p_terr", "far", "far_available",
    )
    #: String meta fields contributed to the HDF5 ``meta/`` group.
    STR_FIELDS = (
        "release", "observing_run", "source_class", "source_class_method",
        "source_class_reference", "metadata_source",
    )

    def float_value(self, field_name: str) -> float:
        val = getattr(self, field_name)
        if isinstance(val, bool):
            return 1.0 if val else 0.0
        return float(val)
