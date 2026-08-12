"""Stage 1: ingest raw PESummary cosmo files into one fast 'store.h5'.

Design choices (locked):
  * Layout: concatenated 1-D columns + an integer offsets index. Fast bulk
    reads, good compression, trivial ragged slicing per event.
  * The store keeps a generous, waveform-complete parameter set so you never
    re-ingest for a missing column. Derived quantities are NOT stored; the
    GWCatalog API computes them on demand.
  * The store keeps the *distance* PE prior p_dL_pe (mass-prior-agnostic) plus
    the cosmology used to evaluate it. The (m1det, q, dL)-basis mass Jacobian
    is applied only at darksirens export time.

Format heterogeneity handled (verified by probe):
  * O3 (C01:*):  prefer 'C01:Mixed'. GWTC-3 has priors/analytic; GWTC-2.1 does
    NOT -> fall back to LVK default UniformSourceFrame/Planck15 and validate
    against the stored prior 'samples'.
  * O4 (C00:*):  prefer 'C00:Mixed'. 'C00:Mixed' carries NO priors group, so
    read the analytic dL prior from a sibling waveform analysis. Some O4b
    (GWTC-5) events have no Mixed set at all -> fall back to a configurable
    waveform priority list and record the choice per event.

This module uses pesummary.io.read for robustness across the above quirks.
For the very largest files you can swap _read_event_pesummary for an h5py
reader of f[analysis]['posterior_samples'] (a structured array) -- the rest of
the pipeline is agnostic to how samples are obtained.

UNTESTED against your local files: run `python -m gwcat.ingest --inspect <file>`
on one event per catalog first; it prints the chosen analysis, prior source,
and the prior-validation result before you launch the full batch.
"""
from __future__ import annotations

import os
import re
import glob
import json
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import h5py

from .cosmology import (make_cosmology, uniform_source_frame_prob,
                        dL_prior_prob, DL_PRIOR_KINDS, DL_PRIOR_NEEDS_ALPHA,
                        DistancePriorKindError, NAMED_COSMOLOGIES,
                        PLANCK15, O4_FALLBACK)
from .source_class import (normalize_source_class, classify_by_mass,
                          DEFAULT_NSBH_MASS_THRESHOLD)
from .spin import chi_p_from_components

# --------------------------------------------------------------------------
# Parameter sets
# --------------------------------------------------------------------------
# Enough to regenerate a precessing, higher-mode waveform (with f_ref stored as
# per-event metadata) plus the commonly used summaries. Anything present but not
# listed can be added via store(..., extra_params=[...]) without code changes.
WAVEFORM_PARAMS = [
    "mass_1", "mass_2",
    "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl",
    "theta_jn", "psi", "phase",
    "luminosity_distance", "ra", "dec", "geocent_time",
]
EXTRA_DEFAULT_PARAMS = [
    "mass_1_source", "mass_2_source", "redshift", "comoving_distance",
    "chi_eff", "chi_p", "mass_ratio", "chirp_mass", "chirp_mass_source",
    "iota", "cos_theta_jn",
    "spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y", "spin_2z",
    "network_optimal_snr", "network_matched_filter_snr", "log_likelihood",
    # Derived precession columns (PR 2).  Stored when the file provides them or
    # when they can be derived at ingest from tilt_i / spin ingredients (see
    # _derive_spin_columns); appended at the END so the column order stays
    # stable for stores written before this PR.
    "cos_tilt_1", "cos_tilt_2",
]
DEFAULT_PARAMS = WAVEFORM_PARAMS + EXTRA_DEFAULT_PARAMS

# Per-event metadata columns (scalars). NaN where unavailable from the PE file.
#
# Source-class metadata model (see gwcat.source_class):
#   floats  -> p_astro, p_bbh, p_nsbh, p_bns, p_terr, far, far_available
#   strings -> release, observing_run, source_class, source_class_method,
#              source_class_reference, metadata_source
# far_available is stored as a 0.0/1.0 float mask so that "FAR is genuinely
# absent" (far_available=0) is a first-class, non-crashing state independent of
# whether the far column happens to be NaN.
META_FLOAT_FIELDS = [
    "far", "pastro", "snr_med",
    "m1_src_med", "m2_src_med",
    "dL_prior_H0", "dL_prior_Om0", "dL_prior_min", "dL_prior_max",
    "f_ref", "nsamp_original", "sky_area_90",
    # source-class contract
    "p_astro", "p_bbh", "p_nsbh", "p_bns", "p_terr", "far_available",
    # distance-prior bounds diagnostic (GW-01): how many of this row's samples
    # fall outside the RECORDED [dL_prior_min, dL_prior_max].  The density is
    # evaluated everywhere regardless (see gwcat.cosmology), so this is a
    # provenance/quality signal, not a count of discarded samples.
    "n_samples_outside_dL_prior_bounds",
    # distance-prior CLASS provenance (GW-02): the exponent of the effective
    # prior when it is a power law (NaN otherwise), the exponent the file's own
    # analytic group declares, and the KS of that declared class against the
    # file's own prior samples (NaN when the file carries none).
    "dL_prior_alpha", "dL_prior_sampling_alpha", "dL_prior_ks",
    # parsed mass-prior bounds (GW-07).  The real releases put the prior on
    # (chirp_mass, mass_ratio) and record mass_1/mass_2 only as Constraints, so
    # these are the bounds that actually describe the mass prior.
    "mass_prior_chirp_min", "mass_prior_chirp_max",
    "mass_prior_q_min", "mass_prior_q_max",
]
META_STR_FIELDS = [
    "name", "catalog", "analysis_used", "dL_prior_source",
    "mass_prior_kind", "compact_type",
    # which implementation evaluated p_dL_pe ("bilby" / "astropy" / "analytic");
    # bilby and astropy differ by ~3% at the low-distance end and that does NOT
    # cancel downstream (GW-01)
    "dL_prior_impl",
    # distance-prior CLASS provenance (GW-02).  dL_prior_kind is the EFFECTIVE
    # distribution divided out of p_pe; dL_prior_sampling_kind is what the file
    # declares.  They differ for a reweighted release, where
    # dL_prior_basis="release_reweighted" and dL_prior_release_flavour="cosmo"
    # record exactly why.  dL_prior_cosmology_name is the verbatim cosmology
    # token ("Planck15_LAL" vs "Planck15" -- matched exactly, never by substring).
    "dL_prior_kind", "dL_prior_sampling_kind", "dL_prior_cosmology_name",
    "dL_prior_release_flavour", "dL_prior_basis",
    # where f_ref came from, e.g. "meta_data[C01:IMRPhenomXPHM]:sibling" (GW-07).
    # Spins, tilts and chi_p are all defined AT f_ref, so a borrowed value has to
    # say whose it is.  mass_prior_source likewise records which analysis's
    # analytic mass prior was parsed, and whether it was recognised.
    "f_ref_source", "mass_prior_source",
    # source-class contract
    "release", "observing_run", "source_class", "source_class_method",
    "source_class_reference", "metadata_source",
]

# ── Sample-set / waveform contract (PR 6) ────────────────────────────────────
# Per-ROW sample-set provenance.  Each (event, sample_set) pair is one row of
# the ragged store, so these are ordinary meta columns aligned with the rows;
# uniqueness is (event_name, sample_set_name).  Explicit-absence defaults follow
# the PR-2 pattern: NaN for the float flags, "" for the strings.  ``release``
# and ``catalog`` already exist above; the rest are new here.
#   floats  -> is_mixed, is_preferred (0.0/1.0 flags), priority_rank
#   strings -> sample_set_name, waveform (family), approximant,
#              calibration_model, selection_reason, file_name, file_checksum,
#              record_id
# available_parameters / sample_count are intentionally NOT stored: they are
# already derivable from the availability mask (avail/mask) and the offsets
# index, respectively (see GWCatalog.param_available / nsamp_per_event).
SAMPLE_SET_FLOAT_FIELDS = ["is_mixed", "is_preferred", "priority_rank"]
SAMPLE_SET_STR_FIELDS = ["sample_set_name", "waveform", "approximant",
                         "calibration_model", "selection_reason",
                         "file_name", "file_checksum", "record_id",
                         # spin-variant suffix of the label, e.g. "HighSpin" /
                         # "LowSpinSecondary" / "" (GW-07).  A restricted-spin
                         # variant is a DIFFERENT prior, so which one was
                         # ingested has to be on the row.
                         "spin_variant"]
META_FLOAT_FIELDS += SAMPLE_SET_FLOAT_FIELDS
META_STR_FIELDS += SAMPLE_SET_STR_FIELDS

# ── Spin-prior / derived-column provenance (PR 2) ────────────────────────────
# Per-event spin metadata.  Explicit-absence defaults follow the PR-2 pattern:
# NaN for the float fields, "" for the strings.
#   floats  -> spin_amax_1, spin_amax_2 (resolved analytic spin-magnitude prior
#              bounds), chi_p_def_maxdiff (max |chi_p_file - chi_p_formula| when
#              both are available; a definition-consistency diagnostic, NaN when
#              not computable).
#   strings -> spin_prior_kind ("uniform_magnitude_isotropic" / "unrecognized" /
#              "assumed_default"), spin_prior_source (which analysis / raw repr
#              the resolution came from), derived_params (comma-separated list of
#              sample columns derived at ingest for this event, may be "").
# Old stores predate these fields; the read/merge paths NaN/""-fill them (see
# _read_store, which only reads fields present in the file, and merge_stores,
# which setdefaults every declared field), so those stores remain loadable.
SPIN_FLOAT_FIELDS = ["spin_amax_1", "spin_amax_2", "chi_p_def_maxdiff"]
SPIN_STR_FIELDS = ["spin_prior_kind", "spin_prior_source", "derived_params"]
META_FLOAT_FIELDS += SPIN_FLOAT_FIELDS
META_STR_FIELDS += SPIN_STR_FIELDS

# Default waveform priority when no Mixed set exists (O4b/GWTC-5 events).
O4_WAVEFORM_PRIORITY = [
    "C00:IMRPhenomXPHM-SpinTaylor", "C00:SEOBNRv5PHM",
    "C00:IMRPhenomXPNR", "C00:NRSur7dq4",
]
O3_WAVEFORM_PRIORITY = ["C01:IMRPhenomXPHM", "C01:SEOBNRv4PHM"]


@dataclass
class IngestConfig:
    # Msun, source-frame, for classification.  Shared with injection selection
    # via gwcat.source_class.DEFAULT_NSBH_MASS_THRESHOLD so the two cannot drift.
    nsbh_mass_threshold: float = DEFAULT_NSBH_MASS_THRESHOLD
    o4_waveform_priority: list = field(default_factory=lambda: list(O4_WAVEFORM_PRIORITY))
    o3_waveform_priority: list = field(default_factory=lambda: list(O3_WAVEFORM_PRIORITY))
    o3_default_cosmo: tuple = (PLANCK15.H0.value, PLANCK15.Om0)   # used when no analytic
    o4_fallback_cosmo: tuple = (O4_FALLBACK.H0.value, O4_FALLBACK.Om0)
    validate_prior: bool = True
    compression: str = "gzip"
    #: Which UniformSourceFrame implementation evaluates p_dL_pe (GW-01).
    #: "auto" prefers bilby (the object the LVK PE used) and falls back to
    #: astropy only when bilby is not installed; "bilby"/"astropy" pin it.
    dL_prior_impl: str = "auto"
    #: Accept an analytic spin prior borrowed from a sibling analysis with a
    #: DIFFERENT spin variant (GW-07).  Off by default: a LowSpin(Secondary) run
    #: restricts a_2 to U(0, 0.05) while HighSpin uses U(0, 0.99), so a
    #: mismatched borrow inflates the secondary's prior support ~20x.
    spin_prior_allow_variant_mismatch: bool = False
    #: KS threshold above which the file's own prior samples are taken to reject
    #: the parsed distance prior (GW-02).  This compares the SAMPLING class the
    #: file declares against its own prior draws, so exceeding it means the parse
    #: / bounds / cosmology mapping is wrong.
    prior_ks_max: float = 0.05
    #: Make that a hard failure rather than a warning.  Default False so a single
    #: odd release cannot block a 282-file ingest; set True for an audited build.
    prior_ks_fatal: bool = False
    #: Warn when more than this FRACTION of a row's dL samples fall outside the
    #: recorded distance-prior bounds.  The default warns on any occurrence: the
    #: samples are no longer zeroed (GW-01), but the mismatch means the recorded
    #: bounds describe a different analysis than the one being ingested.
    dL_outside_warn_frac: float = 0.0


# --------------------------------------------------------------------------
# Catalog family detection
# --------------------------------------------------------------------------
def detect_catalog(path: str) -> str:
    b = os.path.basename(path)
    if "GWTC2p1" in b or "GWTC-2.1" in b or "GWTC2.1" in b:
        return "GWTC-2.1"
    if "GWTC3" in b or "GWTC-3" in b:
        return "GWTC-3"
    if "GWTC4p1" in b or "GWTC-4.1" in b or "GWTC4.1" in b:
        return "GWTC-4.1"
    if "GWTC4" in b or "GWTC-4" in b:
        return "GWTC-4"
    if "GWTC5" in b or "GWTC-5" in b:
        return "GWTC-5"
    # fall back to the analysis prefix
    return "unknown"


def _prefix_for(analyses) -> str:
    return "C01" if any(a.startswith("C01") for a in analyses) else "C00"


def event_name_from_path(path: str) -> str:
    m = re.search(r"(GW\d{6}_\d{6}|GW\d{6})", os.path.basename(path))
    return m.group(1) if m else os.path.splitext(os.path.basename(path))[0]


# --------------------------------------------------------------------------
# Reading one event (pesummary)
# --------------------------------------------------------------------------
def _read_event_pesummary(path: str):
    from pesummary.io import read
    data = read(path, package="gw")
    samples_dict = data.samples_dict          # {analysis: {param: array}}
    analyses = list(samples_dict.keys())
    priors = getattr(data, "priors", {}) or {}
    return data, samples_dict, analyses, priors


#: Spin-variant suffixes real releases append to an analysis label, most
#: preferred first.  GWTC-4.1 labels the combined sets ``C00:Mixed:HighSpin`` and
#: ``C00:Mixed:LowSpinSecondary``; an exact match on ``"C00:Mixed"`` finds
#: neither.  Preference order matters and is a physics choice: the high-spin
#: (unrestricted) run is the general-purpose posterior, and the low-spin runs are
#: restricted-prior variants published alongside it for NSBH/BNS candidates.
SPIN_VARIANT_PRIORITY = ("HighSpin", "", "LowSpinSecondary", "LowSpin")


def _label_parts(label: str):
    """``"C00:Mixed:HighSpin"`` -> ``("C00", "Mixed", "HighSpin")``."""
    bits = str(label).split(":")
    prefix = bits[0] if bits else ""
    base = bits[1] if len(bits) > 1 else ""
    variant = ":".join(bits[2:]) if len(bits) > 2 else ""
    return prefix, base, variant


def _variant_rank(variant: str) -> int:
    """Index of ``variant`` in :data:`SPIN_VARIANT_PRIORITY` (unknown last)."""
    try:
        return SPIN_VARIANT_PRIORITY.index(variant)
    except ValueError:
        return len(SPIN_VARIANT_PRIORITY)


def _spin_variant_token(variant: str) -> str:
    """The spin-restriction token inside a possibly compound variant.

    GWTC-3 labels NSBH events with compound variants such as
    ``C01:Mixed:NSBH:HighSpin``, where the spin restriction is only one component.
    Comparing raw variant strings would treat ``"NSBH:HighSpin"`` and
    ``"HighSpin"`` as different spin priors when they are the same one.
    """
    parts = [p for p in str(variant).split(":") if p]
    for tok in SPIN_VARIANT_PRIORITY:
        if tok and tok in parts:
            return tok
    return ""


def find_mixed_analyses(analyses, prefix: str):
    """Every combined ("Mixed") set for ``prefix``, best spin variant first.

    Matches ``{prefix}:Mixed`` and any ``{prefix}:Mixed:<variant>``.  The
    pre-GW-07 code tested ``f"{prefix}:Mixed" in analyses`` exactly, so on a real
    GWTC-4.1 file offering ``C00:Mixed:HighSpin`` and
    ``C00:Mixed:LowSpinSecondary`` it found no Mixed set, fell through the
    waveform-priority list (which lists only unsuffixed labels, so it also
    missed), and landed on the ``startswith`` last resort -- which returns
    whatever h5py yields first.  For GW230529_181500 that is
    ``C00:IMRPhenomNSBH``, an aligned-spin NSBH approximant, chosen over three
    available Mixed sets, and the reason recorded for it was
    ``"preferred_priority"``.
    """
    cands = []
    for a in analyses:
        pfx, base, variant = _label_parts(a)
        if pfx == prefix and base == "Mixed":
            cands.append((_variant_rank(variant), a))
    # stable: ties keep the file's own order
    return [a for _r, a in sorted(cands, key=lambda t: t[0])]


def _priority_matches(analyses, priority):
    """Priority-list hits, allowing a spin-variant suffix on each entry.

    ``priority`` entries are unsuffixed (``"C00:SEOBNRv5PHM"``), but real labels
    may carry a variant (``"C00:SEOBNRv5PHM:HighSpin"``).  Matching only exactly
    is why a suffixed file skipped the whole list.
    """
    out = []
    for want in priority:
        w_pfx, w_base, _ = _label_parts(want)
        hits = []
        for a in analyses:
            pfx, base, variant = _label_parts(a)
            if (pfx, base) == (w_pfx, w_base):
                hits.append((_variant_rank(variant), a))
        out.extend(a for _r, a in sorted(hits, key=lambda t: t[0]))
    return out


def select_analysis(analyses, prefix: str, cfg: IngestConfig):
    """Pick the single preferred analysis label for one PE file.

    This is the historical one-sample-set-per-event heuristic (kept as the
    default): prefer a combined ``Mixed`` set -- including the spin-variant
    spellings real releases use, see :func:`find_mixed_analyses` -- else walk the
    configured waveform-priority list, else fall back to the first analysis
    carrying the file's prefix.  :func:`select_analyses` builds on it to support
    ingesting several sample sets per event.
    """
    ordered = rank_analyses(analyses, prefix, cfg)
    if not ordered:
        raise RuntimeError(f"No usable analysis among {analyses}")
    return ordered[0]


def rank_analyses(analyses, prefix: str, cfg: IngestConfig):
    """Order the prefix's analyses by ingest preference (most preferred first).

    ``{prefix}:Mixed`` (if present) ranks first, then the configured
    waveform-priority list in order, then any remaining prefixed analyses in
    their original order.  The index into this list becomes each sample set's
    ``priority_rank``; the first element is what :func:`select_analysis` returns.
    """
    prefixed = [a for a in analyses if _label_parts(a)[0] == prefix]
    priority = cfg.o3_waveform_priority if prefix == "C01" else cfg.o4_waveform_priority
    ordered = []
    # Every Mixed set first, best spin variant first (GW-07).
    for a in find_mixed_analyses(prefixed, prefix):
        if a not in ordered:
            ordered.append(a)
    # Then the waveform-priority list, tolerating spin-variant suffixes.
    for a in _priority_matches(prefixed, priority):
        if a not in ordered:
            ordered.append(a)
    for a in prefixed:
        if a not in ordered:
            ordered.append(a)
    return ordered


def select_analyses(analyses, prefix: str, cfg: IngestConfig,
                    sample_sets="preferred"):
    """Return the list of analysis labels to ingest for one PE file.

    Parameters
    ----------
    sample_sets : {"preferred", "all"} or list of str
        * ``"preferred"`` (default): exactly the single label
          :func:`select_analysis` picks -- the historical one-row-per-event
          behavior.
        * ``"all"``: every analysis carrying the file's prefix, each ingested as
          a separate sample-set row (uniqueness is ``(event_name,
          sample_set_name)``).
        * a list/tuple of labels: exactly those labels (each validated to be
          present in the file's analyses).
    """
    if isinstance(sample_sets, str):
        if sample_sets == "preferred":
            return [select_analysis(analyses, prefix, cfg)]
        if sample_sets == "all":
            ordered = rank_analyses(analyses, prefix, cfg)
            if not ordered:
                raise RuntimeError(f"No usable analysis among {analyses}")
            return ordered
        raise ValueError(
            f"sample_sets={sample_sets!r} is invalid; use 'preferred', 'all', "
            f"or a list of analysis labels.")
    wanted = list(sample_sets)
    missing = [a for a in wanted if a not in analyses]
    if missing:
        raise ValueError(
            f"sample_sets={wanted}: label(s) {missing} not present in the "
            f"file's analyses {list(analyses)}.")
    return wanted


#: Hyphenated CONFIGURATION suffixes that may be stripped to get a waveform
#: family.  A whitelist, because a hyphen is also part of real waveform *names*:
#: splitting on the first ``'-'`` unconditionally maps
#: ``IMRPhenomPv2-NRTidalv2`` to ``IMRPhenomPv2``, i.e. reports a BBH
#: approximant for a tidal one, which is a different waveform model and not the
#: family a ``strict-approximant`` request means (GW-07).
WAVEFORM_CONFIG_SUFFIXES = ("SpinTaylor",)


def _waveform_family(approximant: str) -> str:
    """Coarse waveform family from an approximant/analysis token.

    ``'IMRPhenomXPHM-SpinTaylor' -> 'IMRPhenomXPHM'`` (a configuration suffix);
    ``'IMRPhenomPv2-NRTidalv2' -> 'IMRPhenomPv2-NRTidalv2'`` (part of the name);
    ``'SEOBNRv5PHM' -> 'SEOBNRv5PHM'``; ``'Mixed' -> 'Mixed'``.
    """
    if not approximant:
        return ""
    head, sep, tail = approximant.rpartition("-")
    if sep and tail in WAVEFORM_CONFIG_SUFFIXES:
        return head
    return approximant


def _sample_set_meta(analysis: str, preferred_label: str, ranked, path: str,
                     sample_sets, provenance: Optional[dict] = None,
                     priority_labels=()) -> dict:
    """Per-row sample-set provenance for one ingested analysis label.

    ``is_preferred`` marks the label the default (single-set) heuristic would
    have chosen, and ``priority_rank`` is its index in :func:`rank_analyses`, so
    the ``preferred`` waveform policy can reproduce that choice downstream.
    ``record_id`` / ``file_checksum`` default to "" unless ``provenance`` (a
    ``{"record_id":..., "file_checksum":...}`` dict keyed by the source file's
    basename -- see ``build_store(file_provenance=...)``, PR 8) supplies them.
    """
    provenance = provenance or {}
    # Split the label properly: "C00:Mixed:HighSpin" is approximant "Mixed" with
    # spin variant "HighSpin", not an approximant literally called
    # "Mixed:HighSpin" (which is what split(":", 1)[1] produced).
    _pfx, approximant, spin_variant = _label_parts(analysis)
    if not approximant:
        approximant = analysis
    is_mixed = 1.0 if approximant.lower() == "mixed" else 0.0
    is_preferred = 1.0 if analysis == preferred_label else 0.0
    try:
        rank = float(list(ranked).index(analysis))
    except ValueError:
        rank = np.nan
    if is_preferred:
        # Say which rule actually fired.  The pre-GW-07 code reported
        # "preferred_priority" whenever the label was not a Mixed set -- including
        # when it came from the startswith LAST RESORT, i.e. "whatever h5py
        # yielded first". For GW230529_181500 that recorded
        # selection_reason="preferred_priority" for C00:IMRPhenomNSBH, an
        # aligned-spin NSBH approximant picked over three available Mixed sets.
        if is_mixed:
            reason = "preferred_mixed"
        elif analysis in tuple(priority_labels):
            reason = "preferred_priority"
        else:
            reason = "preferred_last_resort"
    elif isinstance(sample_sets, str) and sample_sets == "all":
        reason = "ingested_all"
    else:
        reason = "ingested_explicit"
    return dict(
        sample_set_name=analysis,
        waveform=_waveform_family(approximant),
        approximant=approximant,
        spin_variant=spin_variant,
        calibration_model="",
        record_id=str(provenance.get("record_id", "")),
        file_name=os.path.basename(path),
        file_checksum=str(provenance.get("file_checksum", "")),
        is_mixed=is_mixed,
        is_preferred=is_preferred,
        priority_rank=rank,
        selection_reason=reason,
    )


# --------------------------------------------------------------------------
# Distance-prior resolution
# --------------------------------------------------------------------------
_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


@dataclass(frozen=True)
class AnalyticDLPrior:
    """One parsed analytic ``luminosity_distance`` prior repr.

    ``kind`` is the distribution CLASS -- the field the pre-GW-02 parser never
    read, which is why every row was evaluated as UniformSourceFrame regardless
    of what it declared.  ``cosmology_name`` is the verbatim token when the repr
    named one (``'Planck15_LAL'``, ``'Planck15'``), so the mapping can be exact
    rather than a substring test.
    """
    kind: Optional[str] = None
    dmin: Optional[float] = None
    dmax: Optional[float] = None
    alpha: Optional[float] = None
    H0: Optional[float] = None
    Om0: Optional[float] = None
    cosmology_name: str = ""
    cosmology_recognized: Optional[bool] = None
    raw: str = ""


def _balanced_arg(s: str, key: str) -> Optional[str]:
    """The value of ``key=`` in a repr, honouring nested parentheses.

    ``cosmology=LambdaCDM(name=None, H0=67.9, Om0=0.3065, ...)`` must come back
    whole; a naive ``[^,)]+`` capture stops at the first comma and yields
    ``"LambdaCDM(name=None"``, which then forces a search for ``H0=`` across the
    ENTIRE prior string -- where a recalibration parameter could match instead.
    """
    m = re.search(rf"{key}\s*=\s*", s)
    if not m:
        return None
    i = m.end()
    if i >= len(s):
        return None
    # Scan to the delimiter that ends this argument, but only at depth 0 -- so
    # ``LambdaCDM(name=None, H0=67.9, ...)`` comes back whole even though it does
    # not START with a bracket, and a quoted token containing a comma survives.
    pairs = {"(": ")", "[": "]", "{": "}"}
    opens, closes = set(pairs), set(pairs.values())
    depth, j, quote = 0, i, None
    while j < len(s):
        c = s[j]
        if quote is not None:
            if c == quote:
                quote = None
        elif c in "'\"":
            quote = c
        elif c in opens:
            depth += 1
        elif c in closes:
            if depth == 0:
                break          # the close of the ENCLOSING call
            depth -= 1
        elif c == "," and depth == 0:
            break
        j += 1
    return s[i:j].strip()


def _parse_analytic_dL(prior_string: str) -> AnalyticDLPrior:
    """Parse an analytic ``luminosity_distance`` prior repr.

    Handles the three shapes the real releases use::

        PowerLaw(alpha=2, minimum=10, maximum=10000, name='luminosity_distance')
        bilby.gw.prior.UniformSourceFrame(minimum=10.0, maximum=4000.0,
            cosmology='Planck15_LAL', ...)
        bilby.gw.prior.UniformSourceFrame(minimum=1.0, maximum=500.0,
            cosmology=LambdaCDM(name=None, H0=67.9, Om0=0.3065, ...), ...)

    The class name is taken from the first ``Name(`` in the string, so a dotted
    module path is stripped and a nested cosmology repr cannot be mistaken for
    it.  Cosmology resolution is by EXACT token match against
    :data:`gwcat.cosmology.NAMED_COSMOLOGIES`: the old ``"Planck15" in tok``
    substring test also matched ``'Planck15_LAL'``, silently giving 190 O4 rows
    astropy's (67.74, 0.3075) instead of LAL's (67.90, 0.3065).
    """
    s = str(prior_string)

    kind = None
    km = re.search(r"([A-Za-z_]\w*)\s*\(", s)
    if km:
        kind = km.group(1)

    def _num(key):
        v = _balanced_arg(s, key)
        if v is None:
            return None
        m = re.match(rf"^\(?\s*({_NUM})", v)
        return float(m.group(1)) if m else None

    dmin = _num("minimum")
    dmax = _num("maximum")
    alpha = _num("alpha")

    H0 = Om0 = None
    cosmology_name = ""
    cosmology_recognized = None
    tok = _balanced_arg(s, "cosmology")
    if tok is not None:
        bare = tok.strip().strip("'\"")
        if "(" in bare:
            # An inline cosmology repr: read H0/Om0 from INSIDE it only.
            h = re.search(rf"H0\s*=\s*({_NUM})", bare)
            o = re.search(rf"Om0\s*=\s*({_NUM})", bare)
            if h and o:
                H0, Om0 = float(h.group(1)), float(o.group(1))
                cosmology_name = bare.split("(", 1)[0].strip()
                cosmology_recognized = True
            else:
                cosmology_name = bare.split("(", 1)[0].strip()
                cosmology_recognized = False
        elif bare in NAMED_COSMOLOGIES:
            cosmo = NAMED_COSMOLOGIES[bare]
            H0, Om0 = cosmo.H0.value, cosmo.Om0
            cosmology_name = bare
            cosmology_recognized = True
        else:
            # A named cosmology gwcat does not know.  Do NOT guess: record the
            # token and leave H0/Om0 unresolved so the caller can be loud.
            cosmology_name = bare
            cosmology_recognized = False

    return AnalyticDLPrior(kind=kind, dmin=dmin, dmax=dmax, alpha=alpha,
                           H0=H0, Om0=Om0, cosmology_name=cosmology_name,
                           cosmology_recognized=cosmology_recognized, raw=s)


class PriorMismatchError(ValueError):
    """The file's own prior samples reject the distance prior gwcat parsed."""


#: Release flavours, i.e. whether the posterior samples in a file are still on
#: the prior its ``priors/analytic`` group records.
#:
#:   "cosmo"   -- GWTC-2.1/GWTC-3 ``*_cosmo.h5``: posteriors have been REWEIGHTED
#:                by the LVK to a distance prior uniform in comoving volume and
#:                source-frame time, while ``priors/analytic`` still records the
#:                original ``PowerLaw(alpha=2)`` SAMPLING prior.  The effective
#:                prior is therefore UniformSourceFrame, not the declared class.
#:   "nocosmo" -- the sibling release with the original priors intact: the
#:                declared class IS the effective prior.
#:   "native"  -- no flavour token in the name (the O4 combined releases).  The
#:                declared class is the effective prior; verified on real files,
#:                where the prior samples follow UniformSourceFrame to KS<0.01.
RELEASE_FLAVOURS = ("cosmo", "nocosmo", "native")


def detect_release_flavour(path: str) -> str:
    """Which :data:`RELEASE_FLAVOURS` a PE file is, from its name.

    This distinction is load-bearing and was previously undeclared anywhere in
    gwcat: the correctness of the 81 GWTC-2.1/3 rows rests entirely on the input
    being the ``_cosmo`` variant, a property of the FILENAME that nothing read.
    """
    b = os.path.basename(path).lower()
    if "nocosmo" in b:
        return "nocosmo"
    if "cosmo" in b:
        return "cosmo"
    return "native"


@dataclass(frozen=True)
class ResolvedDLPrior:
    """The distance prior gwcat will divide out, plus how it got there.

    ``kind``/``alpha`` are the EFFECTIVE prior -- the density actually evaluated
    into ``p_dL_pe``.  ``sampling_kind``/``sampling_alpha`` are what the file's
    ``priors/analytic`` group declares, which for a reweighted release is a
    different distribution.  Keeping both is the point: the two disagreeing is a
    normal, documented state for ``_cosmo`` files, and conflating them is what
    made a KS of 0.27 against the file's own prior samples look like an error.
    """
    kind: str
    H0: float
    Om0: float
    dmin: float
    dmax: float
    source: str
    alpha: Optional[float] = None
    sampling_kind: str = ""
    sampling_alpha: Optional[float] = None
    cosmology_name: str = ""
    flavour: str = "native"
    basis: str = ""

    @property
    def cosmology(self):
        return make_cosmology(self.H0, self.Om0)


def resolve_dL_prior(catalog, analysis, analyses, priors, dL_samples,
                     cfg: IngestConfig, flavour: str = "native"):
    """Resolve the effective distance prior for one ingested analysis.

    Strategy for the bounds/cosmology is unchanged: read analytic from the chosen
    analysis; if absent (e.g. the GWTC-2.1/3 ``Mixed`` sets, whose analytic AND
    prior-sample groups are both empty), search sibling analyses; if still absent
    use the catalog default cosmology with bounds from the dL sample range.

    What GW-02 adds is the distribution CLASS, which was never parsed, and an
    exact cosmology-token mapping.  The effective class is the declared one
    EXCEPT for a reweighted ``_cosmo`` release, where it is UniformSourceFrame by
    release convention -- recorded as ``basis="release_reweighted"`` so the
    assumption is visible in the store rather than implicit in a filename.

    Returns a :class:`ResolvedDLPrior`.
    """
    if flavour not in RELEASE_FLAVOURS:
        raise ValueError(f"flavour must be one of {RELEASE_FLAVOURS}; "
                         f"got {flavour!r}")

    analytic = priors.get("analytic", {}) if isinstance(priors, dict) else {}

    def _try(an):
        node = analytic.get(an) if isinstance(analytic, dict) else None
        if node and "luminosity_distance" in node:
            return _parse_analytic_dL(node["luminosity_distance"])
        return None

    parsed = _try(analysis)
    src = f"analytic[{analysis}]"
    if parsed is None:
        for an in analyses:  # sibling search (handles O4 Mixed w/o priors)
            parsed = _try(an)
            if parsed is not None:
                src = f"analytic[{an}]"
                break

    dmin = float(np.min(dL_samples))
    dmax = float(np.max(dL_samples))
    default_cosmo = (cfg.o3_default_cosmo if analysis.startswith("C01")
                     else cfg.o4_fallback_cosmo)

    if parsed is None:
        # No analytic anywhere (9 of the GWTC-2.1 rows).  A cosmo-flavour
        # release is still reweighted, so UniformSourceFrame remains the
        # effective prior; there is simply no declaration to compare it to.
        H0, Om0 = default_cosmo
        return ResolvedDLPrior(
            kind="UniformSourceFrame", H0=H0, Om0=Om0, dmin=dmin, dmax=dmax,
            source="default(no_analytic)", sampling_kind="",
            flavour=flavour,
            basis=("release_reweighted" if flavour == "cosmo"
                   else "assumed_default"))

    if parsed.dmin is not None:
        dmin = parsed.dmin
    if parsed.dmax is not None:
        dmax = parsed.dmax

    H0, Om0 = parsed.H0, parsed.Om0
    if H0 is None:
        H0, Om0 = default_cosmo
        src += "+default_cosmo"
        if parsed.cosmology_recognized is False:
            warnings.warn(
                f"{analysis}: analytic distance prior names an unrecognised "
                f"cosmology {parsed.cosmology_name!r}; falling back to "
                f"(H0={H0}, Om0={Om0}). Add it to "
                f"gwcat.cosmology.NAMED_COSMOLOGIES if it is real -- the KS "
                f"check against the file's own prior samples is the only thing "
                f"standing between this guess and a wrong p_dL_pe.")

    sampling_kind = parsed.kind or ""
    if flavour == "cosmo":
        # Reweighted release: the declared class is the sampling prior only.
        kind, alpha, basis = "UniformSourceFrame", None, "release_reweighted"
    else:
        kind = sampling_kind or "UniformSourceFrame"
        alpha = parsed.alpha
        basis = "analytic_declared" if sampling_kind else "assumed_default"
        if kind not in DL_PRIOR_KINDS:
            raise DistancePriorKindError(
                f"{analysis}: analytic distance prior declares class "
                f"{kind!r}, which gwcat cannot evaluate (known: "
                f"{list(DL_PRIOR_KINDS)}). Refusing to substitute "
                f"UniformSourceFrame -- that substitution is the GW-02 defect.")
        if kind in DL_PRIOR_NEEDS_ALPHA and alpha is None:
            raise ValueError(
                f"{analysis}: analytic distance prior declares {kind} but no "
                f"alpha could be parsed from {parsed.raw[:120]!r}.")

    return ResolvedDLPrior(
        kind=kind, H0=H0, Om0=Om0, dmin=dmin, dmax=dmax, source=src,
        alpha=alpha, sampling_kind=sampling_kind,
        sampling_alpha=parsed.alpha, cosmology_name=parsed.cosmology_name,
        flavour=flavour, basis=basis)


# --------------------------------------------------------------------------
# Spin-prior resolution + derived spin columns  (PR 2)
# --------------------------------------------------------------------------
def _parse_analytic_spin(prior_repr):
    """Parse a bilby spin-prior repr into ``(kind, minimum, maximum)``.

    Handles reprs such as
    ``"Uniform(minimum=0.0, maximum=0.99, name='a_1', ...)"``,
    ``"Sine(name='tilt_1', minimum=0.0, maximum=3.141592653589793, ...)"`` and
    the dotted spelling the O4 releases use,
    ``"bilby.gw.prior.UniformInComponentsChirpMass(minimum=..., ...)"``.
    ``kind`` is the distribution class name with any module path stripped;
    ``minimum`` / ``maximum`` are ``None`` when absent.  Returns ``None`` when
    the repr carries no recognisable class or bounds.

    The class regex is a ``search`` for the first ``Name(``, not a ``match``
    anchored at the string start: anchoring rejected every dotted repr outright
    (GW-07).  Bounds are read with :func:`_balanced_arg` so a nested repr's
    ``minimum=`` cannot be picked up instead.
    """
    s = str(prior_repr).strip()
    m = re.search(r"([A-Za-z_]\w*)\s*\(", s)
    if not m:
        return None
    kind = m.group(1)

    def _num(key):
        v = _balanced_arg(s, key)
        if v is None:
            return None
        mm = re.match(rf"^\(?\s*({_NUM})", v)
        return float(mm.group(1)) if mm else None

    lo = _num("minimum")
    hi = _num("maximum")
    if lo is None and hi is None:
        return None
    return kind, lo, hi


class SpinPriorMismatchError(ValueError):
    """The only analytic spin prior available belongs to a different spin variant."""


#: The analytic mass-prior pair that IS uniform in detector-frame component
#: masses.  bilby's ``UniformInComponents*`` classes exist precisely to make
#: sampling in ``(chirp_mass, mass_ratio)`` equivalent to a flat prior in
#: ``(m1, m2)``, which is the assumption the exported ``p_pe = m1det * p_dL_pe``
#: Jacobian encodes.  Verified across all four releases.
_UNIFORM_IN_COMPONENTS = ("UniformInComponentsChirpMass",
                          "UniformInComponentsMassRatio")


@dataclass(frozen=True)
class ResolvedMassPrior:
    """The analytic mass prior, parsed rather than assumed (GW-07).

    ``kind == "uniform_detector_frame"`` is the only value that justifies the
    ``|dm2det/dq| = m1det`` Jacobian applied at export.  Anything else is
    recorded as ``"unrecognized"`` with the raw reprs in ``source``, so the
    Jacobian's assumption becomes checkable instead of being stamped as a
    constant string.
    """
    kind: str = "unrecognized"
    chirp_min: Optional[float] = None
    chirp_max: Optional[float] = None
    q_min: Optional[float] = None
    q_max: Optional[float] = None
    source: str = ""


def resolve_mass_prior(analysis, analyses, priors):
    """Parse the analytic mass prior for one ingested analysis.

    Searches the chosen analysis then its siblings, exactly as the distance and
    spin priors do (a combined ``Mixed`` set carries no analytic group of its
    own).  Real releases record ``mass_1``/``mass_2`` as ``Constraint(1, 1000)``
    -- a *constraint*, not a prior -- with the actual prior on
    ``(chirp_mass, mass_ratio)``, so the class of that pair is what decides
    whether the mass Jacobian is right.
    """
    analytic = priors.get("analytic", {}) if isinstance(priors, dict) else {}

    def _find(an):
        node = analytic.get(an) if isinstance(analytic, dict) else None
        if not node:
            return None
        got = {k: node[k] for k in ("chirp_mass", "mass_ratio") if k in node}
        return got if len(got) == 2 else None

    found, src_an = _find(analysis), analysis
    if found is None:
        for an in analyses:
            found = _find(an)
            if found is not None:
                src_an = an
                break
    if found is None:
        return ResolvedMassPrior(kind="assumed_default",
                                 source="default(no_analytic_prior)")

    mc = _parse_analytic_spin(found["chirp_mass"])
    q = _parse_analytic_spin(found["mass_ratio"])
    kinds = (mc[0] if mc else None, q[0] if q else None)
    if kinds == _UNIFORM_IN_COMPONENTS:
        return ResolvedMassPrior(
            kind="uniform_detector_frame",
            chirp_min=None if mc[1] is None else float(mc[1]),
            chirp_max=None if mc[2] is None else float(mc[2]),
            q_min=None if q[1] is None else float(q[1]),
            q_max=None if q[2] is None else float(q[2]),
            source=f"analytic[{src_an}]")
    raw = "; ".join(f"{k}={str(found[k])[:80]!r}" for k in sorted(found))
    return ResolvedMassPrior(
        kind="unrecognized",
        source=f"analytic[{src_an}]:unrecognized({raw})")


def resolve_spin_prior(analysis, analyses, priors, a1_samples, a2_samples,
                       fallback_amax=0.99, allow_variant_mismatch=False):
    """Return ``(amax_1, amax_2, kind, source)`` for the spin-magnitude prior.

    Mirrors :func:`resolve_dL_prior`: read the analytic spin priors of the
    chosen ``analysis``; if that analysis carries no priors group (e.g. an O4
    ``Mixed`` set) search the sibling ``analyses`` -- but only siblings sharing
    the ingested label's spin variant, see the sibling-search comment below.

    ``kind`` is ``"uniform_magnitude_isotropic"`` only when both ``a_1`` and
    ``a_2`` parse as ``Uniform(0, amax_i)`` AND the tilt priors parse as
    ``Sine`` (or are absent).  Otherwise ``kind`` is ``"unrecognized"`` and the
    raw repr(s) are recorded in ``source``.  When nothing analytic is found
    anywhere, fall back to ``amax = fallback_amax`` with
    ``kind = "assumed_default"``.

    Regardless of how the bounds were obtained, ``max(|a_i samples|)`` is checked
    against ``amax_i * (1 + 1e-3)``; a violation warns and is noted in
    ``source`` (the parsed ``amax`` is kept, not clamped).
    """
    analytic = priors.get("analytic", {}) if isinstance(priors, dict) else {}

    def _node(an):
        return analytic.get(an) if isinstance(analytic, dict) else None

    def _find(an):
        node = _node(an)
        if not node:
            return None
        found = {k: node[k] for k in ("a_1", "a_2", "tilt_1", "tilt_2")
                 if k in node}
        return found or None

    found = _find(analysis)
    src_an = analysis
    if found is None:
        # Sibling search (an O4 Mixed set carries no priors group of its own).
        #
        # It must respect the SPIN VARIANT (GW-07).  Borrowing across variants is
        # a real, large error: a LowSpin(Secondary) run restricts a_2 to
        # U(0, 0.05) while the HighSpin run uses U(0, 0.99), so a mismatched
        # borrow inflates the secondary's prior support ~20x.  The pre-GW-07 code
        # took the first sibling carrying a_1/a_2 in h5py key order, i.e. by
        # alphabet.
        #
        # (Aligned-spin siblings are already skipped for free: they record
        # chi_1/chi_2 rather than a_1/a_2, so `_find` returns None for them.
        # Verified on GW230529_181500, whose C00:IMRPhenomNSBH analytic group has
        # chi_1/chi_2 and no a_1/a_2.)
        want = _spin_variant_token(_label_parts(analysis)[2])
        cands = [a for a in analyses if _find(a) is not None]
        same = [a for a in cands if _spin_variant_token(_label_parts(a)[2]) == want]
        other = [a for a in cands if a not in same]

        for an in same:
            found = _find(an)
            src_an = an
            break

        if found is None and other:
            if want:
                # The ingested label DECLARES a spin restriction and only a
                # different one is on offer.  That is a contradiction, not a
                # choice: a LowSpin(Secondary) run restricts a_2 to U(0, 0.05)
                # where HighSpin uses U(0, 0.99), so the borrow would inflate the
                # secondary's prior support ~20x.
                msg = (
                    f"{analysis}: no sibling analysis with spin variant "
                    f"{want!r} carries an analytic a_1/a_2 prior; the only "
                    f"candidates declare a different spin restriction "
                    f"({other}). Borrowing across spin variants inflates a "
                    f"restricted secondary-spin prior by up to ~20x, so gwcat "
                    f"refuses. Ingest the matching variant explicitly, or pass "
                    f"allow_variant_mismatch=True to accept the borrow with a "
                    f"warning.")
                if not allow_variant_mismatch:
                    raise SpinPriorMismatchError(msg)
                warnings.warn(msg)
                found, src_an = _find(other[0]), f"{other[0]}:variant_mismatch"
            else:
                # The ingested label declares NO spin restriction (a plain
                # `C01:Mixed`), while every candidate sibling does.  Which one
                # the combined set corresponds to is not stated in the file, so
                # this is an assumption rather than a contradiction: take the
                # best-ranked variant deterministically and record that we
                # assumed it.  Real case: the GWTC-3 NSBH events
                # GW191219_163120 / GW200105_162426 / GW200115_042309, whose
                # `C01:Mixed` set has only `C01:IMRPhenomXPHM:HighSpin` and
                # `:LowSpin` to borrow from.  HighSpin wins, which is also what
                # h5py key order happened to give before GW-07 -- so this
                # deliberately does not move any current number.
                best = sorted(
                    other,
                    key=lambda a: _variant_rank(
                        _spin_variant_token(_label_parts(a)[2])))[0]
                tok = _spin_variant_token(_label_parts(best)[2])
                warnings.warn(
                    f"{analysis}: the ingested label declares no spin variant "
                    f"but every sibling carrying an analytic a_1/a_2 prior does "
                    f"({other}). Assuming the {tok!r} variant ({best}) is the "
                    f"one the combined set corresponds to; recorded as "
                    f"variant_assumed in spin_prior_source.")
                found = _find(best)
                src_an = f"{best}:variant_assumed({tok})"

    def _is_uniform_zero(p):
        return (p is not None and p[0] == "Uniform"
                and p[1] is not None and abs(p[1]) <= 1e-9
                and p[2] is not None)

    def _is_sine_or_absent(p):
        return p is None or p[0] == "Sine"

    if found is None:
        amax_1 = amax_2 = float(fallback_amax)
        kind = "assumed_default"
        source = "default(no_analytic_prior)"
    else:
        p_a1 = _parse_analytic_spin(found["a_1"]) if "a_1" in found else None
        p_a2 = _parse_analytic_spin(found["a_2"]) if "a_2" in found else None
        p_t1 = _parse_analytic_spin(found["tilt_1"]) if "tilt_1" in found else None
        p_t2 = _parse_analytic_spin(found["tilt_2"]) if "tilt_2" in found else None
        a1_ok = _is_uniform_zero(p_a1)
        a2_ok = _is_uniform_zero(p_a2)
        tilts_ok = _is_sine_or_absent(p_t1) and _is_sine_or_absent(p_t2)
        amax_1 = float(p_a1[2]) if a1_ok else float(fallback_amax)
        amax_2 = float(p_a2[2]) if a2_ok else float(fallback_amax)
        if a1_ok and a2_ok and tilts_ok:
            kind = "uniform_magnitude_isotropic"
            source = f"analytic[{src_an}]"
        else:
            kind = "unrecognized"
            raw = "; ".join(f"{k}={found[k]!r}"
                            for k in ("a_1", "a_2", "tilt_1", "tilt_2")
                            if k in found)
            source = f"analytic[{src_an}]:unrecognized({raw})"

    # Validate the spin samples against the resolved bounds (warn, don't clamp).
    viol = []
    for lbl, samp, amax in (("a_1", a1_samples, amax_1),
                            ("a_2", a2_samples, amax_2)):
        if samp is not None and np.size(samp):
            smax = float(np.nanmax(np.abs(np.asarray(samp, float))))
            if np.isfinite(smax) and smax > amax * (1 + 1e-3):
                viol.append(f"{lbl}:max={smax:.4g}>amax={amax:.4g}")
    if viol:
        warnings.warn("spin prior: samples exceed resolved amax "
                      f"({'; '.join(viol)}); prior bounds may be wrong")
        source += " | sample_exceeds_amax(" + "; ".join(viol) + ")"

    return amax_1, amax_2, kind, source


#: The single chi_p implementation (GW-08).  ``ingest`` previously carried a
#: byte-for-byte duplicate of the Schmidt formula with a "unify the two once both
#: have merged" note; that TODO is now discharged, so a change to the definition
#: cannot land in one copy only.
_chi_p_from_samples = chi_p_from_components


def _derive_spin_columns(rec):
    """Add derived spin sample columns to a per-event ``rec`` dict *in place*.

    Returns ``(derived_names, chi_p_def_maxdiff)``:
      * ``cos_tilt_i = cos(tilt_i)`` when ``cos_tilt_i`` is absent and
        ``tilt_i`` is present.
      * ``chi_p`` (Schmidt 2015, see :func:`_chi_p_from_samples`) when absent but
        ``a_1``/``a_2``/``cos_tilt_1``/``cos_tilt_2``/``mass_1``/``mass_2`` are
        all present (``cos_tilt_i`` may itself have just been derived).
      * ``chi_p_def_maxdiff``: ``max|chi_p_file - chi_p_formula|`` when the file
        provides ``chi_p`` AND all the ingredients (a definition-consistency
        diagnostic).  The file's ``chi_p`` is never overwritten; ``NaN`` when the
        diagnostic is not computable.

    Only the derived columns and the diagnostic are produced here; the caller
    records ``derived_names`` / ``chi_p_def_maxdiff`` in the per-event meta and
    the derived columns are marked available in the store's availability mask
    (they are ordinary columns of ``rec`` from here on).
    """
    derived = []
    for i in (1, 2):
        ct, ti = f"cos_tilt_{i}", f"tilt_{i}"
        if ct not in rec and ti in rec:
            rec[ct] = np.cos(np.asarray(rec[ti], float))
            derived.append(ct)

    ingredients = ("a_1", "a_2", "cos_tilt_1", "cos_tilt_2", "mass_1", "mass_2")
    chi_p_formula = None
    if all(k in rec for k in ingredients):
        chi_p_formula = _chi_p_from_samples(
            rec["a_1"], rec["a_2"], rec["cos_tilt_1"], rec["cos_tilt_2"],
            rec["mass_1"], rec["mass_2"])

    chi_p_def_maxdiff = np.nan
    if "chi_p" not in rec:
        if chi_p_formula is not None:
            rec["chi_p"] = chi_p_formula
            derived.append("chi_p")
    elif chi_p_formula is not None:
        # File provides chi_p: keep it, record only the definition mismatch.
        chi_p_def_maxdiff = float(np.max(np.abs(
            np.asarray(rec["chi_p"], float) - chi_p_formula)))

    return derived, chi_p_def_maxdiff


def _ks_against_prior_samples(dlp, *, kind, cosmo, dmin, dmax, alpha, impl):
    """KS distance between prior samples ``dlp`` and one candidate density."""
    grid = np.linspace(max(dmin, dlp.min()), min(dmax, dlp.max()), 200)
    pdf = dL_prior_prob(grid, kind=kind, cosmology=cosmo, dmin=dmin,
                        dmax=dmax, alpha=alpha, impl=impl)
    cdf_model = np.concatenate(
        [[0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    if cdf_model[-1] <= 0:
        return float("nan")
    cdf_model = cdf_model / cdf_model[-1]
    ecdf = np.searchsorted(np.sort(dlp), grid, side="right") / dlp.size
    return float(np.max(np.abs(cdf_model - ecdf)))


def validate_prior_against_samples(priors, analyses_to_try, resolved,
                                   *, impl: str = "auto"):
    """Check the parsed distance prior against the file's own prior samples.

    The prior samples are draws from the **sampling** prior, so they validate
    ``resolved.sampling_kind`` -- the thing GW-02 taught the parser to read. They
    do NOT validate the effective prior on a reweighted ``_cosmo`` release, and
    the pre-GW-02 code conflated the two: it compared UniformSourceFrame against
    dL^2 prior samples, measured KS = 0.27 on every GWTC-2.1/3 event, and warned
    that "the assumed cosmology may be wrong". The cosmology was fine; the
    comparison was against the wrong distribution.

    Returns a dict with the KS of the sampling class (``ks``, the one that must be
    small) and of the effective class (``ks_effective``, expected to be large for
    a reweighted release), or ``None`` when the file carries no prior samples.

    pesummary keys prior samples by the *constituent* analyses, not by 'Mixed',
    so we search a list.
    """
    psamp = priors.get("samples", {}) if isinstance(priors, dict) else {}
    node, used = None, None
    for an in analyses_to_try:
        cand = psamp.get(an) if isinstance(psamp, dict) else None
        if cand and "luminosity_distance" in cand:
            node, used = cand, an
            break
    if node is None:
        return None
    dlp = np.asarray(node["luminosity_distance"], float)
    dlp = dlp[np.isfinite(dlp)]
    if dlp.size < 50:
        return None
    cosmo = resolved.cosmology
    dmin, dmax = resolved.dmin, resolved.dmax

    sampling_kind = resolved.sampling_kind or resolved.kind
    ks_sampling = _ks_against_prior_samples(
        dlp, kind=sampling_kind, cosmo=cosmo, dmin=dmin, dmax=dmax,
        alpha=resolved.sampling_alpha, impl=impl)
    ks_effective = (
        ks_sampling if sampling_kind == resolved.kind
        else _ks_against_prior_samples(
            dlp, kind=resolved.kind, cosmo=cosmo, dmin=dmin, dmax=dmax,
            alpha=resolved.alpha, impl=impl))
    return {"ks": ks_sampling, "ks_effective": ks_effective,
            "sampling_kind": sampling_kind, "effective_kind": resolved.kind,
            "n_prior_samples": int(dlp.size), "analysis": used}


# --------------------------------------------------------------------------
# Inspect a single file (smoke test before the full run)
# --------------------------------------------------------------------------
def inspect(path: str, cfg: Optional[IngestConfig] = None):
    cfg = cfg or IngestConfig()
    catalog = detect_catalog(path)
    data, samples_dict, analyses, priors = _read_event_pesummary(path)
    prefix = _prefix_for(analyses)
    analysis = select_analysis(analyses, prefix, cfg)
    ranked = rank_analyses(analyses, prefix, cfg)
    s = samples_dict[analysis]
    dL = np.asarray(s["luminosity_distance"], float)
    flavour = detect_release_flavour(path)
    res = resolve_dL_prior(catalog, analysis, analyses, priors, dL, cfg,
                           flavour=flavour)
    val = (validate_prior_against_samples(priors, [analysis] + analyses, res,
                                          impl=cfg.dL_prior_impl)
           if cfg.validate_prior else None)
    _, dL_info = dL_prior_prob(dL, kind=res.kind, cosmology=res.cosmology,
                               dmin=res.dmin, dmax=res.dmax, alpha=res.alpha,
                               impl=cfg.dL_prior_impl, return_info=True)
    f_ref, f_ref_source = _read_f_ref(data, analysis, analyses)
    a1_samp = np.asarray(s["a_1"], float) if "a_1" in s else None
    a2_samp = np.asarray(s["a_2"], float) if "a_2" in s else None
    spin_amax_1, spin_amax_2, spin_kind, spin_src = resolve_spin_prior(
        analysis, analyses, priors, a1_samp, a2_samp,
        allow_variant_mismatch=cfg.spin_prior_allow_variant_mismatch)
    mass_prior = resolve_mass_prior(analysis, analyses, priors)
    avail = [p for p in DEFAULT_PARAMS if p in s]
    missing = [p for p in WAVEFORM_PARAMS if p not in s]
    info = {
        "file": os.path.basename(path), "catalog": catalog,
        "analyses": analyses, "analysis_used": analysis,
        # Sample-set contract (PR 6): what "sample_sets='all'" would ingest,
        # ranked most-preferred first, and the single "preferred" default.
        "sample_sets_available": ranked,
        "preferred_sample_set": analysis,
        "n_samples": int(dL.size),
        "f_ref": f_ref,
        "f_ref_source": f_ref_source,
        "dL_prior": {"H0": res.H0, "Om0": res.Om0, "min": res.dmin,
                     "max": res.dmax, "source": res.source,
                     # GW-02: the effective distribution CLASS, what the file
                     # declared, and why they differ when they do.
                     "kind": res.kind, "alpha": res.alpha,
                     "sampling_kind": res.sampling_kind,
                     "sampling_alpha": res.sampling_alpha,
                     "cosmology_name": res.cosmology_name,
                     "release_flavour": res.flavour, "basis": res.basis,
                     # GW-01: which implementation evaluates p_dL_pe, and how
                     # many samples the recorded bounds fail to cover.
                     "impl": dL_info["impl"],
                     "n_samples_outside_bounds": dL_info["n_outside_bounds"],
                     "frac_outside_bounds": dL_info["frac_outside_bounds"]},
        # Spin-magnitude prior resolution (PR 2).
        "spin_prior": {"amax_1": spin_amax_1, "amax_2": spin_amax_2,
                       "kind": spin_kind, "source": spin_src},
        # Parsed mass prior (GW-07): "uniform_detector_frame" is the only value
        # that justifies the export's |dm2det/dq| = m1det Jacobian.
        "mass_prior": {"kind": mass_prior.kind, "source": mass_prior.source,
                       "chirp_min": mass_prior.chirp_min,
                       "chirp_max": mass_prior.chirp_max,
                       "q_min": mass_prior.q_min, "q_max": mass_prior.q_max},
        "prior_validation": val,
        "waveform_params_missing": missing,
        "stored_params_available": avail,
    }
    print(json.dumps(info, indent=2, default=str))
    if missing:
        warnings.warn(f"{path}: missing waveform params {missing}")
    return info


# --------------------------------------------------------------------------
# Build the store
# --------------------------------------------------------------------------
def _classify(m1_src, m2_src, thr):
    """Legacy PE-event classifier.

    Thin wrapper over the shared mass-threshold classifier
    (:func:`gwcat.source_class.classify_by_mass`) so PE-event and injection
    classification apply identical thresholds and cannot drift apart.  Returns
    the legacy compact labels ``"BBH"``/``"NSBH"``/``"BNS"``/``"unknown"``.
    """
    return classify_by_mass(m1_src, m2_src, thr)


def _observing_run_from_name(name: str) -> str:
    """Best-effort observing-run label from a GWOSC event name.

    Returns "" (explicit absence) when the name does not carry a parseable
    date.  This is a coarse mapping and callers may override it with authoritative
    release/manifest metadata later.
    """
    m = re.match(r"GW(\d{2})(\d{2})", str(name))
    if not m:
        return ""
    yy, mm = m.group(1), int(m.group(2))
    mapping = {
        "15": "O1", "16": "O1",
        "17": "O2",
        "19": "O3a" if mm <= 9 else "O3b",
        "20": "O3b",
        "23": "O4a",
        "24": "O4a" if mm <= 3 else "O4b",
        "25": "O4b",
    }
    return mapping.get(yy, "")


def _sky_area_90(ra_samples, dec_samples, nside=64):
    """Estimate the 90% credible sky area (deg^2) from posterior ra/dec samples.

    Uses healpy if available; returns NaN otherwise.  This is computed once at
    ingest and stored as metadata for fast selection.
    """
    try:
        import healpy as hp
    except ImportError:
        return np.nan
    npix = hp.nside2npix(nside)
    pix_area_deg2 = hp.nside2pixarea(nside, degrees=True)
    # Convert (ra, dec) → healpy (theta, phi)
    theta = np.pi / 2 - np.asarray(dec_samples)
    phi = np.asarray(ra_samples)
    pix = hp.ang2pix(nside, theta, phi)
    counts = np.bincount(pix, minlength=npix)
    # Sort descending; find smallest set covering 90%
    sorted_counts = np.sort(counts)[::-1]
    cumsum = np.cumsum(sorted_counts)
    threshold = 0.9 * cumsum[-1]
    n_pix_90 = int(np.searchsorted(cumsum, threshold) + 1)
    return n_pix_90 * pix_area_deg2


def _resolve_event_table(event_table, cache_dir=None, offline=None):
    """Resolve the event_table argument for build_store / merge_store.

    None (default) → auto-fetch FAR/p_astro from GWOSC (or from cache_dir when
                      offline=True / GWCAT_OFFLINE is set -- see gwcat.fetch_cache;
                      a cache miss in offline mode raises, it is not swallowed).
    {}             → skip (no network call).
    dict           → use as-is (e.g. from gwcat.event_metadata.assemble_event_metadata).
    """
    if event_table is not None:
        return event_table
    from .fetch_cache import is_offline
    from .fetch import fetch_event_table_gwosc
    if is_offline(offline):
        print(f"Reading FAR/p_astro from offline metadata cache under "
              f"{cache_dir} ...")
        table = fetch_event_table_gwosc(cache_dir=cache_dir, offline=True)
        print(f"  got {len(table)} events from cache")
        return table
    try:
        print("Fetching FAR/p_astro from GWOSC ...")
        table = fetch_event_table_gwosc(cache_dir=cache_dir)
        print(f"  got {len(table)} events from GWOSC")
        return table
    except Exception as e:
        warnings.warn(
            f"Could not auto-fetch event table from GWOSC: {e}\n"
            "  FAR/p_astro metadata will be NaN. Pass event_table={{}} to silence."
        )
        return {}


def build_store(paths, out_path, params=None, extra_params=None,
                cfg: Optional[IngestConfig] = None, event_table=None,
                sample_sets="preferred", file_provenance: Optional[dict] = None,
                cache_dir=None, offline: Optional[bool] = None,
                write_summary: bool = False,
                summary_context: Optional[dict] = None):
    """Ingest a list of cosmo-file paths into a single concatenated store.

    params       : column list to store (default DEFAULT_PARAMS).
    extra_params : appended to params (store anything extra without editing code).
    event_table  : {event_name: {'far':..,'pastro':..}} for FAR/p_astro,
                   which are NOT in per-event PE files.
                   None (default) → auto-fetch from GWOSC.
                   Pass {} to skip.
                   An entry may also carry ``source_class`` (overrides the
                   mass-threshold classification) and/or ``metadata_source``
                   (overrides the "event_table"/"pe_file_only" default label,
                   e.g. with the richer "online+user_override" string that
                   ``gwcat.event_metadata.assemble_event_metadata`` produces).
    sample_sets  : which posterior sample set(s) to ingest per PE file (PR 6).
                   "preferred" (default) keeps exactly one sample set per event
                   (the historical Mixed/priority heuristic).  "all" ingests
                   every analysis carrying the file's prefix as a SEPARATE row
                   (event identity stays event_name; uniqueness is
                   (event_name, sample_set_name)).  A list of analysis labels
                   ingests exactly those.  Sample-set provenance is recorded in
                   the meta/ columns (sample_set_name, waveform, approximant,
                   is_mixed, is_preferred, priority_rank, ...).
    file_provenance : {file_basename: {"record_id":.., "file_checksum":..}}, optional
                   (PR 8) Populates the per-row ``record_id`` / ``file_checksum``
                   meta columns for the file each row was ingested from.  See
                   ``gwcat.fetch.fetch_catalog(provenance=...)``.  Default None
                   leaves both columns "" as before this PR.
    cache_dir, offline : optional
                   Only consulted when event_table is None (auto-fetch).  See
                   gwcat.fetch_cache; None/unset leaves auto-fetch behavior
                   unchanged from before PR 8.
    write_summary : bool, default False
                   (PR 10) When True, write ``<out_path>.validation_summary.json``
                   and ``.md`` next to ``out_path`` (see
                   :mod:`gwcat.validation_summary`).  Opt-in at the library level;
                   the unified ``gwcat ingest`` CLI turns this on by default
                   (``--no-summary`` to disable).  Default False keeps
                   ``build_store`` byte-identical (no extra files written) for
                   every existing caller.
    summary_context : dict, optional
                   Extra fields merged into the written summary (e.g. a release
                   manifest name/version a caller already knows about). Never
                   populated automatically.
    """
    cfg = cfg or IngestConfig()
    params = list(params or DEFAULT_PARAMS)
    if extra_params:
        params += [p for p in extra_params if p not in params]
    event_table = _resolve_event_table(event_table, cache_dir=cache_dir,
                                       offline=offline)
    file_provenance = file_provenance or {}

    records = []   # per row: (name, n_samples, {param: array}) -- union schema
    offsets = [0]
    names, meta = [], {k: [] for k in META_FLOAT_FIELDS + META_STR_FIELDS}

    for path in paths:
        catalog = detect_catalog(path)
        name = event_name_from_path(path)
        data, samples_dict, analyses, priors = _read_event_pesummary(path)
        prefix = _prefix_for(analyses)
        # Sample-set contract (PR 6): one or more analyses per file, each a row.
        preferred_label = select_analysis(analyses, prefix, cfg)
        ranked = rank_analyses(analyses, prefix, cfg)
        labels = select_analyses(analyses, prefix, cfg, sample_sets)
        # Which labels the waveform-priority list actually matched, so
        # selection_reason can distinguish a priority hit from the last resort.
        priority_labels = _priority_matches(
            [a for a in analyses if _label_parts(a)[0] == prefix],
            cfg.o3_waveform_priority if prefix == "C01"
            else cfg.o4_waveform_priority)
        et = event_table.get(name, {})
        prov = file_provenance.get(os.path.basename(path), {})
        # Whether this release's posteriors are still on the prior its
        # priors/analytic group records (GW-02).
        flavour = detect_release_flavour(path)

        for analysis in labels:
            s = samples_dict[analysis]

            n = len(np.asarray(s["luminosity_distance"]))
            # Union schema: keep EVERY candidate parameter this event actually
            # has.  Parameters a given event lacks are NaN-filled for that
            # event's slice later; we never drop a whole column because one
            # event lacks it.
            rec = {p: np.asarray(s[p], dtype=np.float64) for p in params if p in s}

            dL = np.asarray(s["luminosity_distance"], float)
            res = resolve_dL_prior(catalog, analysis, analyses, priors, dL, cfg,
                                   flavour=flavour)
            H0, Om0, dmin, dmax, src = (res.H0, res.Om0, res.dmin, res.dmax,
                                        res.source)
            ks_val = None
            if cfg.validate_prior:
                v = validate_prior_against_samples(
                    priors, [analysis] + analyses, res, impl=cfg.dL_prior_impl)
                if v is not None:
                    ks_val = v["ks"]
                    # The KS is now against the SAMPLING class the file declares,
                    # so a failure means the parse is wrong -- not that "the
                    # assumed cosmology may be wrong", which is what the
                    # pre-GW-02 message said while comparing two different
                    # distributions on every GWTC-2.1/3 event.
                    if np.isfinite(ks_val) and ks_val > cfg.prior_ks_max:
                        msg = (
                            f"{name} [{analysis}]: the file's own prior samples "
                            f"reject the parsed distance prior "
                            f"{v['sampling_kind']}"
                            f"{'' if res.sampling_alpha is None else f'(alpha={res.sampling_alpha:g})'}"
                            f" on [{dmin:.4g}, {dmax:.4g}] Mpc with "
                            f"cosmology={res.cosmology_name or f'(H0={H0}, Om0={Om0})'}: "
                            f"KS={ks_val:.4f} > {cfg.prior_ks_max} over "
                            f"{v['n_prior_samples']} samples from "
                            f"{v['analysis']}. The parse, the bounds or the "
                            f"cosmology mapping is wrong.")
                        if cfg.prior_ks_fatal:
                            raise PriorMismatchError(msg)
                        warnings.warn(msg)
            # distance prior evaluated per sample, stored mass-prior-agnostic.
            # Dispatched on the EFFECTIVE distribution class (GW-02) and
            # evaluated over the FULL sample range rather than truncated at the
            # recorded bounds (GW-01): those bounds often come from a sibling
            # analysis, and a truncated density hands legitimate samples
            # p_dL_pe = 0 -> p_pe = 0 -> a -inf darksirens log-weight.
            p_dL, dL_info = dL_prior_prob(
                dL, kind=res.kind, cosmology=res.cosmology, dmin=dmin,
                dmax=dmax, alpha=res.alpha, impl=cfg.dL_prior_impl,
                return_info=True)
            rec["p_dL_pe"] = p_dL
            if dL_info["frac_outside_bounds"] > cfg.dL_outside_warn_frac:
                warnings.warn(
                    f"{name} [{analysis}]: {dL_info['n_outside_bounds']} of "
                    f"{dL_info['n_samples']} dL samples "
                    f"({100 * dL_info['frac_outside_bounds']:.2f}%) fall "
                    f"outside the recorded distance-prior bounds "
                    f"[{dmin:.4g}, {dmax:.4g}] Mpc from {src} "
                    f"({dL_info['n_below_dmin']} below, "
                    f"{dL_info['n_above_dmax']} above).  The density was "
                    f"evaluated over [{dL_info['eval_min']:.4g}, "
                    f"{dL_info['eval_max']:.4g}] Mpc instead of zeroing them, "
                    f"but the recorded bounds describe a different analysis "
                    f"than the one ingested.")
            if dL_info["n_nonfinite"]:
                warnings.warn(
                    f"{name} [{analysis}]: {dL_info['n_nonfinite']} non-finite "
                    f"dL sample(s); p_dL_pe is NaN for those samples and the "
                    f"export validator will reject the row.")
            # Derived spin columns (cos_tilt_i, chi_p) + chi_p definition
            # diagnostic (PR 2).  Additive: only fills columns the file lacks;
            # never overwrites the file's chi_p.
            derived_names, chi_p_def_maxdiff = _derive_spin_columns(rec)
            # Spin-magnitude prior resolution (PR 2), mirroring resolve_dL_prior.
            a1_samp = np.asarray(s["a_1"], float) if "a_1" in s else None
            a2_samp = np.asarray(s["a_2"], float) if "a_2" in s else None
            spin_amax_1, spin_amax_2, spin_kind, spin_src = resolve_spin_prior(
                analysis, analyses, priors, a1_samp, a2_samp,
                allow_variant_mismatch=cfg.spin_prior_allow_variant_mismatch)
            records.append((name, n, rec))

            # metadata
            m1s = (float(np.median(s["mass_1_source"]))
                   if "mass_1_source" in s else np.nan)
            m2s = (float(np.median(s["mass_2_source"]))
                   if "mass_2_source" in s else np.nan)
            snr = (float(np.median(s["network_optimal_snr"]))
                   if "network_optimal_snr" in s else np.nan)
            f_ref, f_ref_source = _read_f_ref(data, analysis, analyses)

            # ── Source-class contract ──────────────────────────────────────
            compact = _classify(m1s, m2s, cfg.nsbh_mass_threshold)
            far_val = float(et.get("far", np.nan))
            # far_available is an explicit state: True only when a finite FAR
            # was actually supplied by the event table (public metadata may omit
            # it).
            far_available = 1.0 if np.isfinite(far_val) else 0.0
            # p_astro / component probabilities come from the event table when
            # present; otherwise stay NaN (explicit absence).
            p_astro = float(et.get("p_astro", et.get("pastro", np.nan)))
            # metadata_source: an assembled event_table (PR 8, see
            # gwcat.event_metadata.assemble_event_metadata) may supply a richer
            # provenance string (e.g. "online+user_override", "absent") directly;
            # fall back to the historical binary label when it does not.
            metadata_source = et.get("metadata_source") or (
                "event_table" if et else "pe_file_only")
            # source_class: a user override (PR 8) takes precedence over the
            # mass-threshold classification; source_class_method/_reference
            # record which happened.
            override_source_class = et.get("source_class")
            if override_source_class:
                source_class_val = normalize_source_class(override_source_class)
                source_class_method = "user_override"
                source_class_reference = "user_override_file"
            else:
                source_class_val = normalize_source_class(compact)
                source_class_method = "mass_threshold"
                source_class_reference = (
                    f"m2_source<{cfg.nsbh_mass_threshold}Msun -> NS component")

            names.append(name)
            offsets.append(offsets[-1] + n)
            meta["name"].append(name)
            meta["catalog"].append(catalog)
            meta["analysis_used"].append(analysis)
            meta["dL_prior_source"].append(src)
            # Parsed, not stamped (GW-07): the exported p_pe applies an
            # |dm2det/dq| = m1det Jacobian that is only valid for a prior
            # uniform in detector-frame component masses, so warn loudly when
            # the file declares something else instead of asserting it silently.
            mp = resolve_mass_prior(analysis, analyses, priors)
            if mp.kind == "unrecognized":
                warnings.warn(
                    f"{name} [{analysis}]: the analytic mass prior is not "
                    f"uniform in detector-frame component masses "
                    f"({mp.source}); the exported p_pe applies an "
                    f"|dm2det/dq| = m1det Jacobian that assumes it is.")
            meta["mass_prior_kind"].append(mp.kind)
            meta["mass_prior_source"].append(mp.source)
            meta["mass_prior_chirp_min"].append(
                np.nan if mp.chirp_min is None else mp.chirp_min)
            meta["mass_prior_chirp_max"].append(
                np.nan if mp.chirp_max is None else mp.chirp_max)
            meta["mass_prior_q_min"].append(
                np.nan if mp.q_min is None else mp.q_min)
            meta["mass_prior_q_max"].append(
                np.nan if mp.q_max is None else mp.q_max)
            meta["compact_type"].append(compact)
            # canonical source-class metadata (parallel to legacy compact_type)
            meta["source_class"].append(source_class_val)
            meta["source_class_method"].append(source_class_method)
            meta["source_class_reference"].append(source_class_reference)
            meta["release"].append(catalog)
            meta["observing_run"].append(_observing_run_from_name(name))
            meta["metadata_source"].append(metadata_source)
            meta["far_available"].append(far_available)
            meta["p_astro"].append(p_astro)
            meta["p_bbh"].append(float(et.get("p_bbh", np.nan)))
            meta["p_nsbh"].append(float(et.get("p_nsbh", np.nan)))
            meta["p_bns"].append(float(et.get("p_bns", np.nan)))
            meta["p_terr"].append(float(et.get("p_terr", np.nan)))
            meta["far"].append(far_val)
            meta["pastro"].append(float(et.get("pastro", np.nan)))
            meta["snr_med"].append(snr)
            meta["m1_src_med"].append(m1s)
            meta["m2_src_med"].append(m2s)
            meta["dL_prior_H0"].append(float(H0))
            meta["dL_prior_Om0"].append(float(Om0))
            meta["dL_prior_min"].append(float(dmin))
            meta["dL_prior_max"].append(float(dmax))
            meta["dL_prior_impl"].append(dL_info["impl"])
            meta["n_samples_outside_dL_prior_bounds"].append(
                float(dL_info["n_outside_bounds"]))
            # ── Distance-prior class provenance (GW-02) ─────────────────────
            meta["dL_prior_kind"].append(res.kind)
            meta["dL_prior_sampling_kind"].append(res.sampling_kind)
            meta["dL_prior_cosmology_name"].append(res.cosmology_name)
            meta["dL_prior_release_flavour"].append(res.flavour)
            meta["dL_prior_basis"].append(res.basis)
            meta["dL_prior_alpha"].append(
                np.nan if res.alpha is None else float(res.alpha))
            meta["dL_prior_sampling_alpha"].append(
                np.nan if res.sampling_alpha is None
                else float(res.sampling_alpha))
            meta["dL_prior_ks"].append(
                np.nan if ks_val is None else float(ks_val))
            # `if f_ref` was also a truthiness bug: a legitimate f_ref of 0.0
            # would have been stored as NaN.  _read_f_ref already rejects
            # non-positive values explicitly, so test against None (GW-07).
            meta["f_ref"].append(np.nan if f_ref is None else float(f_ref))
            meta["f_ref_source"].append(f_ref_source)
            meta["nsamp_original"].append(float(n))
            # ── Spin-prior / derived-column provenance (PR 2) ───────────────
            meta["spin_amax_1"].append(float(spin_amax_1))
            meta["spin_amax_2"].append(float(spin_amax_2))
            meta["chi_p_def_maxdiff"].append(float(chi_p_def_maxdiff))
            meta["spin_prior_kind"].append(spin_kind)
            meta["spin_prior_source"].append(spin_src)
            meta["derived_params"].append(",".join(derived_names))
            # Sky area (optional; requires healpy)
            if "ra" in s and "dec" in s:
                meta["sky_area_90"].append(
                    _sky_area_90(np.asarray(s["ra"]), np.asarray(s["dec"])))
            else:
                meta["sky_area_90"].append(np.nan)
            # ── Sample-set / waveform contract (PR 6) ──────────────────────
            ss = _sample_set_meta(analysis, preferred_label, ranked, path,
                                  sample_sets, provenance=prov,
                                  priority_labels=priority_labels)
            for k, val in ss.items():
                meta[k].append(val)
            print(f"[{catalog}] {name}: {n} samp, sample_set={analysis}, "
                  f"prior={src}")

    # Assemble the UNION of parameters across events, NaN-filling event slices
    # where a parameter is absent, and build the per-event availability mask.
    # Derived spin columns (PR 2) are appended to the candidate list so they are
    # stored and correctly marked available even when a caller passed a custom
    # ``params`` that omitted them (a column absent from every rec is dropped by
    # _assemble_union, so this is harmless when nothing was derived).
    candidate_params = list(params) + ["p_dL_pe"]
    for p in ("cos_tilt_1", "cos_tilt_2", "chi_p"):
        if p not in candidate_params:
            candidate_params.append(p)
    union_params, columns, avail = _assemble_union(records, candidate_params)

    _write_store(out_path, union_params, columns, offsets, names, avail, meta, cfg)
    print(f"\nWrote {out_path}: {len(names)} events, "
          f"{offsets[-1]} total samples, params={union_params}")

    if write_summary:
        # Re-open what was just written as a fresh, unfiltered GWCatalog: reads
        # only index/meta/avail (cheap), never the (potentially large) sample
        # arrays. summarize_catalog is the single, honest source of counting
        # logic shared with `gwcat inspect` and the darksirens-export summary.
        from .catalog import GWCatalog
        from .validation_summary import summarize_catalog, write_validation_summary
        cat = GWCatalog(out_path)
        summary = summarize_catalog(cat)
        summary.update({
            "kind": "ingest",
            "output_path": str(out_path),
            "n_files_provided": len(paths),
            "n_rows_ingested": len(names),
            "n_unique_events_ingested": len(set(names)),
            "sample_sets_mode": (sample_sets if isinstance(sample_sets, str)
                                 else "explicit_list"),
        })
        if file_provenance:
            summary["source_file_checksums"] = file_provenance
        if summary_context:
            summary.update(summary_context)
        write_validation_summary(out_path, summary)

    return out_path


def _assemble_union(records, candidate_params):
    """Assemble union-schema columns + an availability mask from per-event data.

    Parameters
    ----------
    records : list of (name, n_samples, {param: 1-D array})
        One entry per event.  Each dict holds only the parameters that event
        actually provides.
    candidate_params : sequence of str
        Column order to consider.  A parameter is stored iff at least one event
        provides it; the stored order follows ``candidate_params`` (duplicates
        removed, first occurrence kept).

    Returns
    -------
    union_params : list of str
        Parameters present in >= 1 event, in ``candidate_params`` order.
    columns : dict {param: 1-D float64 array}
        Concatenated across events; NaN where an event lacks the parameter.
    avail : 2-D bool array, shape (n_events, len(union_params))
        ``avail[i, j]`` is True iff event ``i`` actually provided
        ``union_params[j]`` (False marks a NaN-filled slice).
    """
    seen, ordered = set(), []
    for p in candidate_params:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    union_params = [p for p in ordered
                    if any(p in rec for (_n, _c, rec) in records)]

    n_events = len(records)
    avail = np.zeros((n_events, len(union_params)), dtype=bool)
    columns = {}
    for j, p in enumerate(union_params):
        chunks = []
        for i, (_name, n, rec) in enumerate(records):
            if p in rec:
                chunks.append(np.asarray(rec[p], dtype=np.float64))
                avail[i, j] = True
            else:
                chunks.append(np.full(n, np.nan, dtype=np.float64))
        columns[p] = (np.concatenate(chunks) if chunks
                      else np.array([], dtype=np.float64))
    return union_params, columns, avail


_F_REF_KEYS = ("reference-frequency", "reference_frequency", "f_ref")


def _coerce_f_ref(val):
    """A finite positive float from an h5py scalar/1-element array, else None."""
    try:
        arr = np.asarray(val, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    if arr.size != 1 or not np.isfinite(arr[0]) or arr[0] <= 0:
        return None
    return float(arr[0])


def _f_ref_from_config(data, analysis):
    """``f_ref`` from the pesummary ``config`` section of one analysis."""
    try:
        cfgd = data.config[analysis] if hasattr(data, "config") else {}
    except Exception:
        return None
    if not isinstance(cfgd, dict):
        return None
    for key in _F_REF_KEYS:
        for sect in cfgd.values():
            if isinstance(sect, dict) and key in sect:
                got = _coerce_f_ref(sect[key])
                if got is not None:
                    return got
    return None


def _extra_kwargs_for(data, analysis):
    """The pesummary ``extra_kwargs`` entry for one analysis label.

    pesummary exposes this as a **list** positionally aligned with
    ``samples_dict``'s key order (verified against pesummary on the real
    releases), not as a dict keyed by label; both shapes are accepted so a
    version change cannot silently return nothing.
    """
    ek = getattr(data, "extra_kwargs", None)
    if isinstance(ek, dict):
        return ek.get(analysis)
    if isinstance(ek, (list, tuple)):
        try:
            labels = list(data.samples_dict.keys())
            return ek[labels.index(analysis)]
        except (AttributeError, ValueError, IndexError, TypeError):
            return None
    return None


def _f_ref_from_meta_data(data, analysis):
    """``f_ref`` from the analysis's ``meta_data`` block.

    This is where the real releases actually put it, and the pre-GW-07 code never
    looked: it read only ``data.config``, which is EMPTY for the combined
    ``Mixed`` sets, so 175 of 282 rows stored ``f_ref = NaN`` while a sibling's
    ``meta_data['f_ref']`` sat there with the value.  Verified on GW150914:
    ``config['C01:Mixed']`` has no sections and its meta_data block has 5 keys
    and no ``f_ref``, while ``C01:IMRPhenomXPHM`` has
    ``config = {'config': {'reference-frequency': '20'}}`` and
    ``meta_data['f_ref'] = 20.0``.

    ``f_ref`` matters because spins, tilts and chi_p are all defined *at* it.
    """
    node = _extra_kwargs_for(data, analysis)
    if isinstance(node, dict):
        for sub in ("meta_data", "other", "sampler"):
            inner = node.get(sub)
            if isinstance(inner, dict):
                for key in _F_REF_KEYS:
                    if key in inner:
                        got = _coerce_f_ref(inner[key])
                        if got is not None:
                            return got
        for key in _F_REF_KEYS:
            if key in node:
                got = _coerce_f_ref(node[key])
                if got is not None:
                    return got
    return None


def _read_f_ref(data, analysis, analyses=None):
    """Reference frequency for ``analysis``, searching siblings as a fallback.

    Order: this analysis's ``config``, then its ``meta_data``, then the same two
    for each sibling analysis.  The sibling fallback is what rescues the combined
    ``Mixed`` sets, which carry neither a config nor an ``f_ref`` of their own but
    are built FROM constituent runs that do.  Returns ``(f_ref, source)`` where
    ``source`` names where it came from ("" when nothing was found).
    """
    for getter, tag in ((_f_ref_from_config, "config"),
                        (_f_ref_from_meta_data, "meta_data")):
        got = getter(data, analysis)
        if got is not None:
            return got, f"{tag}[{analysis}]"
    for an in (analyses or []):
        if an == analysis:
            continue
        for getter, tag in ((_f_ref_from_config, "config"),
                            (_f_ref_from_meta_data, "meta_data")):
            got = getter(data, an)
            if got is not None:
                return got, f"{tag}[{an}]:sibling"
    return None, ""


#: Schema version written by build_store/merge.  1.1 adds the ``avail/mask``
#: availability dataset on top of the 1.0 layout.  Stores written as "1.0" (or
#: with no version) have no mask; readers treat every stored column as available
#: for every event (see :meth:`GWCatalog.__init__`), which is exact for legacy
#: stores because the old intersection ingest guaranteed it.
SCHEMA_VERSION = "1.1"

#: 1.2 adds the per-row sample-set/waveform meta columns (PR 6) on top of 1.1.
#: A store is written as 1.2 when any sample-set column is present; otherwise it
#: stays 1.1.  Stores predating 1.2 have no sample-set columns and load as
#: single-sample-set-per-event, so waveform-policy resolution is a no-op.
SCHEMA_VERSION_SAMPLESETS = "1.2"

#: 1.3 adds the distance-prior provenance a correct p_dL_pe depends on (GW-01,
#: GW-02): dL_prior_impl, n_samples_outside_dL_prior_bounds, and the class fields
#: dL_prior_kind / _sampling_kind / _alpha / _sampling_alpha /
#: _cosmology_name / _release_flavour / _basis / _ks.  A store is written as 1.3
#: when any of them is present.  Every one is read-optional, so 1.1/1.2 stores
#: still load -- but they carry NO record of which distribution was divided out,
#: which is why GW-16 re-ingests rather than back-filling.
SCHEMA_VERSION_DL_PRIOR = "1.3"

#: The meta columns whose presence marks a 1.3 store.
_SCHEMA_13_FIELDS = ("dL_prior_kind", "dL_prior_sampling_kind",
                     "dL_prior_cosmology_name", "dL_prior_release_flavour",
                     "dL_prior_basis", "dL_prior_impl", "dL_prior_alpha",
                     "dL_prior_sampling_alpha", "dL_prior_ks",
                     "n_samples_outside_dL_prior_bounds")


def _write_store(out_path, stored_params, columns, offsets, names, avail, meta,
                 cfg):
    """Write a store.h5 with the union parameter set + availability mask.

    ``columns`` maps each stored parameter to a full-length (already
    concatenated) 1-D array.  ``avail`` is a (n_events, n_params) bool mask
    aligned with ``names`` (rows) and ``stored_params`` (columns).
    """
    dt_str = h5py.string_dtype(encoding="utf-8")
    avail = np.asarray(avail, dtype=bool)
    # Bump the schema version to 1.2 only when sample-set columns are present,
    # so a store with none still advertises 1.1 and loads unchanged.
    has_sampleset = any(k in meta for k in
                        SAMPLE_SET_STR_FIELDS + SAMPLE_SET_FLOAT_FIELDS)
    # 1.3 when the distance-prior provenance is present (GW-01/GW-02), else 1.2
    # when sample-set columns are, else 1.1.
    has_dL_prov = any(k in meta and len(meta[k]) for k in _SCHEMA_13_FIELDS)
    if has_dL_prov:
        schema_version = SCHEMA_VERSION_DL_PRIOR
    elif has_sampleset:
        schema_version = SCHEMA_VERSION_SAMPLESETS
    else:
        schema_version = SCHEMA_VERSION
    with h5py.File(out_path, "w") as f:
        f.attrs["schema_version"] = schema_version
        f.attrs.create("param_names",
                       np.array(stored_params, dtype=h5py.string_dtype()))
        f.attrs["n_events"] = len(names)
        g = f.create_group("samples")
        for p in stored_params:
            arr = np.asarray(columns.get(p, np.array([])), dtype=np.float64)
            g.create_dataset(p, data=arr, compression=cfg.compression,
                             shuffle=True)
        idx = f.create_group("index")
        idx.create_dataset("offsets", data=np.asarray(offsets, dtype=np.int64))
        idx.create_dataset("event_names", data=np.array(names, dtype=object),
                           dtype=dt_str)
        # Per-event x per-parameter availability mask (rows aligned with
        # index/event_names, columns aligned with attrs/param_names).
        ag = f.create_group("avail")
        ag.create_dataset("mask", data=avail, compression=cfg.compression)
        mg = f.create_group("meta")
        for k in META_FLOAT_FIELDS:
            mg.create_dataset(k, data=np.asarray(meta[k], dtype=np.float64))
        for k in META_STR_FIELDS:
            mg.create_dataset(k, data=np.array(meta[k], dtype=object), dtype=dt_str)


# --------------------------------------------------------------------------
# Store read / merge helpers (schema-preserving)
# --------------------------------------------------------------------------
def _decode(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def _read_store(path):
    """Read a store.h5 into an in-memory dict (schema-agnostic).

    Derives an all-True availability mask for legacy stores that predate the
    ``avail/mask`` dataset -- exact for those stores because the old
    intersection ingest guaranteed every stored column was present for every
    event.
    """
    with h5py.File(path, "r") as f:
        params = [_decode(p) for p in f.attrs["param_names"]]
        offsets = f["index/offsets"][:].astype(np.int64)
        names = [_decode(n) for n in f["index/event_names"][:]]
        n_events = len(names)
        samples = {p: f[f"samples/{p}"][:] for p in params}
        if "avail" in f and "mask" in f["avail"]:
            avail = np.asarray(f["avail/mask"][:], dtype=bool)
        else:
            avail = np.ones((n_events, len(params)), dtype=bool)
        meta = {}
        if "meta" in f:
            for k in META_FLOAT_FIELDS:
                if k in f["meta"]:
                    meta[k] = list(f[f"meta/{k}"][:])
            for k in META_STR_FIELDS:
                if k in f["meta"]:
                    meta[k] = [_decode(v) for v in f[f"meta/{k}"][:]]
    return dict(params=params, offsets=offsets, names=names, samples=samples,
                avail=avail, meta=meta, n_events=n_events)


def _subset_store(S, keep):
    """Return a copy of an in-memory store restricted to event indices ``keep``."""
    keep = list(keep)
    slices = [(int(S["offsets"][i]), int(S["offsets"][i + 1])) for i in keep]
    samples = {}
    for p in S["params"]:
        col = S["samples"][p]
        samples[p] = (np.concatenate([col[a:b] for a, b in slices]) if slices
                      else np.array([], dtype=col.dtype))
    offs = [0]
    for a, b in slices:
        offs.append(offs[-1] + (b - a))
    avail = S["avail"][keep, :] if keep else S["avail"][:0, :]
    meta = {k: [v[i] for i in keep] for k, v in S["meta"].items()}
    return dict(params=S["params"], offsets=np.asarray(offs, dtype=np.int64),
                names=[S["names"][i] for i in keep], samples=samples,
                avail=avail, meta=meta, n_events=len(keep))


def merge_stores(store_a, store_b, out_path, cfg: Optional[IngestConfig] = None,
                 skip_duplicates: bool = True):
    """Merge two existing store.h5 files, PRESERVING the union of parameters.

    A parameter present in only one store becomes a full column in the output:
    the events from the store that lacked it are NaN-filled and marked
    unavailable in the availability mask.  No column is ever dropped because one
    store is missing it.  Meta fields merge as a union too, with explicit-absence
    defaults (NaN for floats, "" for strings).

    Events in ``store_b`` whose names already appear in ``store_a`` are skipped
    when ``skip_duplicates`` is True (a warning is emitted).

    Returns the output path.
    """
    cfg = cfg or IngestConfig()
    A = _read_store(store_a)
    B = _read_store(store_b)

    if skip_duplicates:
        dupes = set(A["names"]) & set(B["names"])
        if dupes:
            warnings.warn(f"Duplicate events skipped from the second store: "
                          f"{sorted(dupes)}")
            keep = [i for i, n in enumerate(B["names"]) if n not in dupes]
            B = _subset_store(B, keep)

    # Union parameter order: store A's columns first, then B's new columns.
    union_params = list(A["params"]) + [p for p in B["params"]
                                        if p not in A["params"]]
    a_total = int(A["offsets"][-1]) if A["n_events"] else 0
    b_total = int(B["offsets"][-1]) if B["n_events"] else 0

    columns = {}
    for p in union_params:
        a_col = (A["samples"][p] if p in A["samples"]
                 else np.full(a_total, np.nan, dtype=np.float64))
        b_col = (B["samples"][p] if p in B["samples"]
                 else np.full(b_total, np.nan, dtype=np.float64))
        columns[p] = np.concatenate([a_col, b_col])

    a_idx = {p: j for j, p in enumerate(A["params"])}
    b_idx = {p: j for j, p in enumerate(B["params"])}
    n_total = A["n_events"] + B["n_events"]
    avail = np.zeros((n_total, len(union_params)), dtype=bool)
    for j, p in enumerate(union_params):
        if p in a_idx and A["n_events"]:
            avail[:A["n_events"], j] = A["avail"][:, a_idx[p]]
        if p in b_idx and B["n_events"]:
            avail[A["n_events"]:, j] = B["avail"][:, b_idx[p]]

    offsets = (np.concatenate([A["offsets"], B["offsets"][1:] + a_total])
               .astype(np.int64) if B["n_events"] else A["offsets"])
    names = list(A["names"]) + list(B["names"])

    # Union of meta fields; explicit-absence defaults for a field a store lacks.
    merged_meta = {}
    for k in set(A["meta"]) | set(B["meta"]):
        fill = np.nan if k in META_FLOAT_FIELDS else ""
        a_v = A["meta"].get(k, [fill] * A["n_events"])
        b_v = B["meta"].get(k, [fill] * B["n_events"])
        merged_meta[k] = list(a_v) + list(b_v)
    # Ensure every declared meta field exists (writer requires all keys).
    for k in META_FLOAT_FIELDS:
        merged_meta.setdefault(k, [np.nan] * n_total)
    for k in META_STR_FIELDS:
        merged_meta.setdefault(k, [""] * n_total)

    _write_store(out_path, union_params, columns, offsets, names, avail,
                 merged_meta, cfg)
    print(f"Merged stores: {A['n_events']} + {B['n_events']} = {n_total} "
          f"events, params={union_params} → {out_path}")
    return out_path


# --------------------------------------------------------------------------
# Merge new events into an existing store
# --------------------------------------------------------------------------
def merge_store(existing_path: str, new_paths, out_path: str = None,
                cfg: Optional[IngestConfig] = None, event_table=None,
                extra_params=None, sample_sets="preferred",
                file_provenance: Optional[dict] = None, cache_dir=None,
                offline: Optional[bool] = None):
    """Append new events to an existing store without re-ingesting everything.

    Schema-preserving (PR 5): the merged store holds the UNION of parameters.
    A parameter present in only some events (e.g. BNS tidal columns absent from
    BBH events) is kept as a full column, NaN-filled and marked unavailable for
    the events that lack it -- never silently dropped by intersection.

    Parameters
    ----------
    existing_path : str
        Path to the existing store.h5.
    new_paths : list of str
        Paths to new PE files to add.
    out_path : str or None
        Output path.  None → overwrite existing_path (via a temp file for safety).
    cfg, event_table, extra_params, file_provenance, cache_dir, offline :
        Same as build_store.  event_table=None auto-fetches from GWOSC.

    Returns
    -------
    str : path to the merged store.
    """
    import shutil, tempfile

    cfg = cfg or IngestConfig()
    event_table = _resolve_event_table(event_table, cache_dir=cache_dir,
                                       offline=offline)
    out_path = out_path or existing_path

    old = _read_store(existing_path)

    # Candidate columns for the new events: the generous default set plus any
    # columns the existing store already has (minus the computed p_dL_pe, which
    # build_store always appends) plus any user extras.  This lets the new
    # events keep their own extra columns (e.g. tidal params) which merge_stores
    # then unions with the existing schema.
    candidates = []
    for p in list(DEFAULT_PARAMS) + list(old["params"]) + list(extra_params or []):
        if p != "p_dL_pe" and p not in candidates:
            candidates.append(p)

    tmpdir = tempfile.mkdtemp()
    try:
        tmp_new = os.path.join(tmpdir, "new.h5")
        build_store(new_paths, tmp_new, params=candidates, cfg=cfg,
                    event_table=event_table, sample_sets=sample_sets,
                    file_provenance=file_provenance)

        tmp_merged = os.path.join(tmpdir, "merged.h5")
        merge_stores(existing_path, tmp_new, tmp_merged, cfg=cfg,
                     skip_duplicates=True)
        shutil.move(tmp_merged, out_path)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return out_path


def _cli(
    argv=None,
    _deprecated: bool = True,
    default_write_summary: bool = False,
    prog: Optional[str] = None,
):
    """Ingest CLI. Also the implementation behind the unified ``gwcat ingest``
    subcommand (PR 10), which calls this with ``_deprecated=False,
    default_write_summary=True`` so any flag added here is picked up by both
    surfaces automatically -- no separate argument list to keep in sync.

    argv : list of str, optional
        Parsed instead of ``sys.argv[1:]`` when given (lets ``gwcat.cli``
        delegate its ``ingest`` subcommand's remaining args here directly).
    _deprecated : bool
        When True (the default, used by the standalone ``gwcat-ingest``
        console script), print a one-line pointer to ``gwcat ingest`` on
        stderr before continuing with unchanged behavior.
    default_write_summary : bool
        Whether ``--out`` writes get a validation summary by default
        (``--no-summary`` always disables it regardless). False for the
        deprecated standalone script (unchanged side effects); the unified
        CLI passes True.
    prog : str, optional
        Program identity shown by argparse.  The standalone entry point keeps
        ``gwcat-ingest``; the unified dispatcher supplies ``gwcat ingest`` (or
        the name of a future replacement entry point).
    """
    import argparse
    import sys as _sys
    if _deprecated:
        print("gwcat-ingest is deprecated; use `gwcat ingest` instead "
              "(same options; see `gwcat ingest --help`).", file=_sys.stderr)
    ap = argparse.ArgumentParser(
        prog=prog or "gwcat-ingest",
        description="Ingest GWTC cosmo files -> store.h5",
    )
    ap.add_argument("--inspect", metavar="FILE", help="probe one file and exit")
    ap.add_argument("--glob", action="append", default=[],
                    help="glob of cosmo files (repeatable, one per catalog dir)")
    ap.add_argument("--out", default="store.h5")
    ap.add_argument("--no-event-table", action="store_true",
                    help="Skip auto-fetching FAR/p_astro from GWOSC.")
    ap.add_argument("--sample-sets", default="preferred", metavar="POLICY",
                    help="Which posterior sample set(s) to ingest per PE file "
                         "(PR 6): 'preferred' (default), 'all', or a "
                         "comma-separated list of analysis labels.")
    ap.add_argument("--cache-dir", default=None, metavar="DIR",
                    help="Cache/read the auto-fetched GWOSC event table under "
                         "DIR (see gwcat.fetch_cache). Omit to disable caching.")
    ap.add_argument("--offline", action="store_true",
                    help="Never touch the network for the auto-fetched event "
                         "table; read it from --cache-dir instead (same as "
                         "GWCAT_OFFLINE=1).")
    ap.add_argument("--file-provenance", default=None, metavar="JSON_FILE",
                    help="Path to a JSON file of "
                         "{file_basename: {record_id, file_checksum}} (PR 8) "
                         "populating the per-row provenance meta columns.")
    ap.add_argument("--no-summary", action="store_true",
                    help="Skip writing validation_summary.json/.md next to "
                         "--out.")
    a = ap.parse_args(argv)
    if a.inspect:
        inspect(a.inspect)
        return
    paths = []
    for g in a.glob:
        paths += sorted(glob.glob(g))
    if not paths:
        ap.error("no files matched; pass --glob or --inspect")
    event_table = {} if a.no_event_table else None

    sample_sets = a.sample_sets
    if sample_sets not in ("preferred", "all"):
        sample_sets = [s.strip() for s in sample_sets.split(",") if s.strip()]

    file_provenance = None
    if a.file_provenance:
        with open(a.file_provenance) as f:
            file_provenance = json.load(f)

    offline = True if a.offline else None
    write_summary = default_write_summary and not a.no_summary

    build_store(paths, a.out, event_table=event_table, sample_sets=sample_sets,
                cache_dir=a.cache_dir, offline=offline,
                file_provenance=file_provenance, write_summary=write_summary)


if __name__ == "__main__":
    _cli()