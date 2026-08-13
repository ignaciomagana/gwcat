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

import numpy as np

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


#: Default source-frame mass threshold (Msun) separating a NS component from a
#: BH component.  This is the SINGLE source of truth shared by PE-event
#: classification (``gwcat.ingest._classify``) and injection selection
#: (``gwcat.selection``), so the two classification paths cannot drift apart.
DEFAULT_NSBH_MASS_THRESHOLD = 3.0


def classify_by_mass(m1_source, m2_source,
                     thr: float = DEFAULT_NSBH_MASS_THRESHOLD):
    """Classify a compact binary from its source-frame component masses.

    This is the shared mass-threshold classifier.  It is used both to label
    ingested PE events (via :func:`gwcat.ingest._classify`) and to filter
    selection injections by injected mass (:mod:`gwcat.selection`), guaranteeing
    the two paths apply *identical* thresholds and can never diverge.

    A component with source-frame mass ``>= thr`` is treated as a black hole and
    ``< thr`` as a neutron star.  With the mass-ordering convention m1 >= m2:

        m1 >= thr and m2 >= thr  -> ``"BBH"``
        m1 >= thr and m2 <  thr  -> ``"NSBH"``
        otherwise (m1 < thr)     -> ``"BNS"``

    Non-finite masses yield ``"unknown"``.  These are the *legacy* compact labels
    (note the lowercase ``"unknown"``); run them through
    :func:`normalize_source_class` for the canonical form.

    Scalar inputs return a ``str``; array-like inputs return an object
    ``numpy.ndarray`` of the same shape.
    """
    scalar = np.ndim(m1_source) == 0 and np.ndim(m2_source) == 0
    m1 = np.atleast_1d(np.asarray(m1_source, dtype=float))
    m2 = np.atleast_1d(np.asarray(m2_source, dtype=float))
    m1b, m2b = np.broadcast_arrays(m1, m2)
    out = np.full(m1b.shape, "unknown", dtype=object)
    finite = np.isfinite(m1b) & np.isfinite(m2b)
    bbh = finite & (m1b >= thr) & (m2b >= thr)
    nsbh = finite & (m1b >= thr) & (m2b < thr)
    bns = finite & ~bbh & ~nsbh
    out[bbh] = "BBH"
    out[nsbh] = "NSBH"
    out[bns] = "BNS"
    if scalar:
        return str(out.reshape(-1)[0])
    return out


def canonical_classes_from_mass(m1_source, m2_source,
                                thr: float = DEFAULT_NSBH_MASS_THRESHOLD):
    """Vectorised canonical source-class labels from source-frame masses.

    Thin convenience wrapper around :func:`classify_by_mass` that maps the legacy
    compact labels onto the canonical labels (:data:`SOURCE_CLASSES`).  Returns a
    string object ndarray suitable for :func:`numpy.isin` against the class set
    returned by :func:`resolve_filter_classes`.
    """
    labels = classify_by_mass(m1_source, m2_source, thr)
    return np.array([normalize_source_class(x) for x in np.atleast_1d(labels)])


# ── Which QUANTITY the source-class cut was applied to (GW-12) ───────────────
#
# `classify_by_mass` is the single classifier and `DEFAULT_NSBH_MASS_THRESHOLD`
# the single threshold, so the two sides of an export pair cannot disagree about
# *where* the class boundary is.  They can still disagree about *what they put
# through it*, and they did: PE events are classified from the POSTERIOR MEDIAN
# source-frame masses (gwcat.ingest, `_classify` on `np.median(mass_i_source)`)
# while injections are classified from their INJECTED TRUE masses
# (SelectionSet.source_class_mask).  Sharing a threshold is not sharing a cut.
#
# The consequence is Essick & Fishbach's event-selection requirement, in its
# source-class form: beta must be the detection probability of the SAME
# procedure that produced the event list.  A median is a noisy estimator of the
# mass it estimates, so near the NS/BH boundary a fraction of true-NSBH systems
# have median m2 >= thr and enter a "BBH" event list, while true-BBH systems
# with median m2 < thr leave it -- and the two fractions do not cancel, because
# the mass distribution is not flat across the boundary and the measurement
# error is not symmetric in the source frame (it is inherited from a detector-
# frame mass divided by a redshift that is itself uncertain).  Cutting the
# injections on truth reproduces neither fraction, so the exported beta is the
# selection function of a cut nobody applied.  The bias is concentrated
# entirely in the boundary events, which is where NSBH/BNS science lives.
#
# There is no cheap fix available at export time: reproducing the median cut on
# the injection side needs a per-injection mock point estimate, which needs a
# measurement-error model this package does not have and would be a fabricated
# one if it invented it.  So the estimator is RECORDED on both sides instead,
# and the paired-file validators refuse a pair whose two sides used different
# estimators.  See :func:`assess_cut_estimator_pair`.

#: No source-class restriction was applied on this side.
CUT_ESTIMATOR_NONE = "none"
#: Events classified from posterior MEDIAN source-frame masses (PE side).
CUT_ESTIMATOR_POSTERIOR_MEDIAN = "posterior_median_mass"
#: Events selected by name from an explicit list (PE side).  Membership in a
#: fixed list is not a mass cut: it is reproducible on the injection side by
#: construction, so it does not carry the boundary bias above.
CUT_ESTIMATOR_NAME_WHITELIST = "name_whitelist"
#: Injections classified from their INJECTED TRUE masses (selection side).
CUT_ESTIMATOR_INJECTED_TRUTH = "injected_truth"

#: The attr both sides write.
CUT_ESTIMATOR_ATTR = "source_class_cut_estimator"


def pe_cut_estimator(source_class, event_list=None) -> str:
    """The estimator a PE export's source-class restriction was applied to.

    ``source_class`` is resolved by ``GWCatalog.select`` against the store's
    ``source_class`` column, which ingest fills from ``classify_by_mass`` on the
    posterior MEDIAN source-frame masses -- so any non-``None`` request is a
    median-based cut, and takes precedence in the record even when an event list
    is also supplied (the median cut is the part that cannot be reproduced).

    An ``event_list`` alone restricts by NAME.  That is membership in a fixed
    list, not a cut on a noisy statistic, so it is recorded distinctly and the
    pairing check treats it as reproducible.
    """
    if source_class is not None:
        return CUT_ESTIMATOR_POSTERIOR_MEDIAN
    if event_list is not None:
        return CUT_ESTIMATOR_NAME_WHITELIST
    return CUT_ESTIMATOR_NONE


def selection_cut_estimator(source_class) -> str:
    """The estimator a selection export's source-class restriction was applied to.

    ``SelectionSet.source_class_mask`` classifies injections from the campaign's
    own injected source-frame masses, which are exact by construction.
    """
    return (CUT_ESTIMATOR_NONE if source_class is None
            else CUT_ESTIMATOR_INJECTED_TRUTH)


def _estimator_str(raw) -> str:
    """Read a ``source_class_cut_estimator`` attr, tolerating bytes/numpy.str_."""
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    return str(raw).strip()


#: The escape hatches named in the refusal, kept in one place so the two
#: validators and the export-time warning cannot drift apart.
CUT_ESTIMATOR_REMEDY = (
    "Either (1) select the events by NAME instead -- pass event_list= (CLI: "
    "--event-list) with the names you mean and drop --source-class from the PE "
    "side; list membership is reproducible on the injection side, so a "
    "whitelisted pair is accepted -- or (2) drop the source-class filter from "
    "BOTH sides and model the classes in the population instead of cutting on "
    "them.")


def assess_cut_estimator_pair(pe_estimator, sel_estimator,
                              pe_filtered: bool, sel_filtered: bool):
    """Judge a PE/selection pair on which quantity each side's class cut used.

    Parameters are the two ``source_class_cut_estimator`` attrs (``None`` when
    the attr is absent, i.e. a file written before GW-12) and whether each side
    carries a real, non-empty ``source_class_filter``.

    Returns ``(verdict, message)`` with ``verdict`` in ``{"pass", "warn",
    "fail"}``.  The caller owns how a warn is emitted and how a fail is raised,
    so the v1 and v2 validators can share this judgement without sharing their
    reporting machinery.
    """
    pe = _estimator_str(pe_estimator)
    sel = _estimator_str(sel_estimator)

    if pe == CUT_ESTIMATOR_NAME_WHITELIST:
        # The PE restriction is membership in a fixed list of names, not a cut
        # on a measured quantity, so there is no point estimate to disagree
        # about: whatever the injection side cuts on, it is not being asked to
        # reproduce a median.  This is the shipped BBH-whitelist configuration
        # and it is unaffected by GW-12.
        return "pass", ""

    if not pe_filtered and not sel_filtered:
        return "pass", ""

    if not pe_filtered or not sel_filtered:
        # One-sided statement.  xcheck_source_class already governs whether a
        # one-sided class filter is legal at all; this check only reports that
        # there is nothing here to compare, following the same warn-not-fail
        # convention _xcheck_detection_cut uses for a one-sided cut.
        return "warn", (
            "source-class cut stated on one side only (PE estimator="
            f"{pe or 'unknown'!r}, selection estimator={sel or 'unknown'!r}); "
            "the files cannot show that the two selections agree.")

    if not pe or not sel:
        # Absent attr on a pre-GW-12 file.  "Unknown" is not "compatible", but
        # it is also not evidence of a mismatch, so warn rather than condemn a
        # pair that may well have been built the right way.
        return "warn", (
            f"{CUT_ESTIMATOR_ATTR} missing on "
            + ("both files" if not pe and not sel
               else ("the PE file" if not pe else "the selection file"))
            + " (PE=" + (pe or "unknown") + ", selection=" + (sel or "unknown")
            + "); this pair predates the record, so which quantity each "
              "source-class cut was applied to cannot be checked. Re-export to "
              "make it verifiable.")

    if pe == CUT_ESTIMATOR_POSTERIOR_MEDIAN and sel == CUT_ESTIMATOR_INJECTED_TRUTH:
        return "fail", (
            "the event source-class cut and the injection source-class cut use "
            "the same threshold on DIFFERENT quantities: the PE events were "
            f"classified from posterior MEDIAN source-frame masses ({pe!r}) "
            "while the injections were classified from their injected TRUE "
            f"masses ({sel!r}). A hard cut on a noisy point estimate is not "
            "reproduced by the same cut on true values -- systems scatter "
            "across the NS/BH mass boundary in both directions and the two "
            "fractions do not cancel -- so the exported selection function does "
            "not describe the cut that produced this event list, and it is "
            "biased exactly at the boundary where NSBH/BNS science lives "
            "(Essick & Fishbach; GW-12). " + CUT_ESTIMATOR_REMEDY)

    if pe == sel:
        return "pass", ""

    return "warn", (
        f"unrecognised {CUT_ESTIMATOR_ATTR} pair (PE={pe!r}, "
        f"selection={sel!r}); this checker knows "
        f"{[CUT_ESTIMATOR_NONE, CUT_ESTIMATOR_POSTERIOR_MEDIAN, CUT_ESTIMATOR_NAME_WHITELIST, CUT_ESTIMATOR_INJECTED_TRUTH]} "
        "and cannot judge whether these two cuts select the same systems.")


#: Warning text the PE exporters emit when a median-based class cut is built.
#: Stated at export time so the failure is understood before the paired
#: validator refuses the product, not after.
PE_MEDIAN_CUT_WARNING = (
    "source_class= applies a hard cut on POSTERIOR MEDIAN source-frame masses, "
    "but the paired selection export cuts its injections on their injected TRUE "
    "masses. The same threshold on different quantities is not the same cut: it "
    "biases the selection function at the class boundary (Essick & Fishbach; "
    "GW-12), and validate_export/validate_pair will REFUSE this pair "
    f"({CUT_ESTIMATOR_ATTR}='{CUT_ESTIMATOR_POSTERIOR_MEDIAN}' vs "
    f"'{CUT_ESTIMATOR_INJECTED_TRUTH}'). " + CUT_ESTIMATOR_REMEDY)


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

    This function classifies *data* -- a label read off an event whose class
    nobody recorded is genuinely unknown, and mapping it to :data:`UNKNOWN` is
    the right answer.  A *request* is different: see
    :func:`resolve_filter_classes`, which raises instead.
    """
    return _CLASS_ALIASES.get(_compact_key(label), UNKNOWN)


def _accepted_tokens() -> str:
    """The tokens a request may use, for an error message."""
    return (f"keywords {sorted(_FILTER_MAP)} or canonical class names "
            f"{list(SOURCE_CLASSES)} (case- and punctuation-insensitive)")


def resolve_filter_classes(
    source_class: Union[str, Iterable[str], None],
) -> Set[str]:
    """Resolve a selection request into a set of canonical class labels.

    Accepts the keywords ``bbh``/``nsbh``/``bns``/``massgap``/``cbc``/``all``,
    a canonical class name, or an iterable of any of those.  ``None`` means "no
    source-class restriction" and returns every canonical class.

    Raises ``ValueError`` on a token that is neither.  This is deliberately
    stricter than :func:`normalize_source_class`: an unrecognised *request*
    used to resolve to ``{"Unknown"}``, so a typo like ``--source-class BBHs``
    silently selected only the *unclassified* events -- zero of them on the real
    store -- and reported success.  A request names classes the caller believes
    exist; if one does not, that is an error, not an empty selection.
    """
    if source_class is None:
        return set(SOURCE_CLASSES)
    if isinstance(source_class, (list, tuple, set, frozenset)):
        # An accidentally-empty list would resolve to set() -- a zero-event
        # selection -- while its provenance serialised to "", which every
        # reader takes as "no restriction".  A silent zero-event export whose
        # attrs claim it is unfiltered is the exact class of trap this module
        # exists to close, so refuse instead: "no restriction" is spelled
        # None.
        if not source_class:
            raise ValueError(
                "empty source-class request (an empty list selects nothing, "
                "and its provenance would serialise identically to 'no "
                "restriction'); pass None for no restriction, or name the "
                f"class(es): {_accepted_tokens()}")
        out: Set[str] = set()
        for item in source_class:
            out |= resolve_filter_classes(item)
        return out
    if isinstance(source_class, bytes):
        source_class = source_class.decode("utf-8", "replace")
    # An empty/whitespace string is the same trap as the empty list -- and the
    # easiest to hit: `--source-class "$SC"` with an unset shell variable.  It
    # used to reach the alias table, whose "" -> Unknown mapping (correct for
    # EVENT metadata, where an unrecorded class IS unknown) turned it into a
    # request for only the unclassified events: zero on the real store,
    # reported as success.  A *request* spelled "" is an error.
    if isinstance(source_class, str) and not source_class.strip():
        raise ValueError(
            "empty source-class request '' (an unset shell variable?); pass "
            "None / omit the flag for no restriction, or name the class(es): "
            f"{_accepted_tokens()}. To select events whose class was never "
            f"recorded, ask for {UNKNOWN!r} explicitly.")
    # A comma-separated string is the CLI spelling of a list.  Resolving it here
    # rather than in the CLI is what makes "nsbh,bns" and ["nsbh", "bns"] the
    # same request everywhere, including on the Python API path the CLI helper
    # never touches.
    if isinstance(source_class, str) and "," in source_class:
        parts = [p for p in source_class.split(",") if p.strip()]
        if not parts:
            raise ValueError(
                f"empty source-class request {source_class!r}; accepted: "
                f"{_accepted_tokens()}")
        return resolve_filter_classes(parts)
    key = _compact_key(source_class)
    if key in _FILTER_MAP:
        return set(_FILTER_MAP[key])
    if key in _CLASS_ALIASES:
        return {_CLASS_ALIASES[key]}
    raise ValueError(
        f"unrecognised source-class request {source_class!r}; accepted: "
        f"{_accepted_tokens()}. To select events whose class was never "
        f"recorded, ask for {UNKNOWN!r} explicitly.")


def canonical_source_class(
    source_class: Union[str, Iterable[str], None],
) -> tuple:
    """The canonical, order-independent form of a source-class request.

    Returns a tuple of canonical labels in :data:`SOURCE_CLASSES` order, so that
    ``["nsbh", "bns"]``, ``"nsbh,bns"`` and ``("BNS", "NSBH")`` all produce the
    same value.  ``None`` (no restriction) returns ``()`` -- and so does an
    explicit request that admits every class (``"cbc"``/``"all"``): the two
    select identical events, so serialising them differently made the two
    pairing checks contradict each other on one pair (``xcheck_source_class``
    resolved both to the full class set and passed, while ``contract_hash``
    kept ``"BBH,...,Unknown"`` distinct from ``None`` and hard-failed).  What
    is canonicalised is the *selection*, not the spelling.
    """
    if source_class is None:
        return ()
    resolved = resolve_filter_classes(source_class)
    if resolved == set(SOURCE_CLASSES):
        return ()
    return tuple(c for c in SOURCE_CLASSES if c in resolved)


def format_source_class_filter(
    source_class: Union[str, Iterable[str], None],
) -> str:
    """Serialise a source-class request to a stable, round-trippable string.

    This is what the exporters write to ``source_class_filter`` and what
    :func:`parse_source_class_filter` reads back.  ``None`` -> ``""``.

    The exporters used to write ``str(source_class)``, whose value depends on
    how the request was *spelled*: the Python API's ``["nsbh", "bns"]`` became
    ``"['nsbh', 'bns']"`` and the CLI's ``--source-class nsbh,bns`` became
    ``"nsbh,bns"``.  Neither round-trips, so the paired-file cross-check
    resolved both to ``{"Unknown"}`` and passed on files that did not agree.
    """
    return ",".join(canonical_source_class(source_class))


def parse_source_class_filter(raw) -> tuple:
    """Read a ``source_class_filter`` attr back to canonical labels.

    Inverse of :func:`format_source_class_filter`.  An empty/absent value means
    "no restriction" and returns ``()``.  Tolerates the ``bytes`` and
    ``numpy.str_`` an HDF5 attr round-trip can produce.
    """
    if raw is None:
        return ()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    s = str(raw).strip()
    if not s:
        return ()
    parts = [p for p in s.split(",") if p.strip()]
    # A stray-separator attr like "," carries no tokens: same "no restriction"
    # reading as "" (canonical_source_class would refuse an empty request).
    return canonical_source_class(parts) if parts else ()


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
