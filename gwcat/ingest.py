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
from .event_metadata import PASTRO_KEYS, resolve_pastro, resolve_pastro_column
from .release_cosmology import (load_release_cosmology_table,
                                ReleaseReweightCosmologyError)

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
    # WHERE the (H0, Om0) the distance prior is evaluated at came from (GW-40a):
    # "analytic_declared" (the prior repr names it), "catalog_default" (nothing
    # declared, the per-CATALOG IngestConfig default), or -- for a reweighted
    # `_cosmo` release -- the release-reweight table row's own `source`
    # ("documented" / "inferred_from_z(dL)"), with that table's sha256 beside
    # it.  The pesummary version that wrote the file (its `version/pesummary`)
    # is recorded too: it is what the table's citations are checked against.
    "dL_prior_cosmology_source", "dL_prior_cosmology_table_sha256",
    "release_pesummary_version",
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

# ── Per-prior source provenance (GW-40b) ─────────────────────────────────────
# For each of the mass, spin and distance priors: WHICH label's declaration the
# resolved prior came from (``prior_source_label_*``, "" when none) and WHAT
# KIND of source that is (``prior_source_kind_*``), one of
# :data:`PRIOR_SOURCE_KINDS`.  The legacy ``*_prior_kind`` columns say what the
# prior IS; these say where gwcat got it -- a combined ``Mixed`` set has no
# priors group of its own, so its "analytic" prior is a sibling's, and until
# now nothing on the row said so.
PRIOR_SOURCE_KINDS = ("own_analytic", "sibling_inherited",
                      "config_file_declared", "assumed_default",
                      "release_reweighted", "constituent_mixture")
PRIOR_SOURCE_STR_FIELDS = [f"prior_source_{w}_{p}"
                           for p in ("mass", "spin", "dL")
                           for w in ("label", "kind")]
META_STR_FIELDS += PRIOR_SOURCE_STR_FIELDS
# Constituent-mixture provenance (GW-40f): which constituents a combined set's
# mixture prior was built from, and their VERIFIED row counts ("" elsewhere).
META_STR_FIELDS += ["constituent_mixture_labels", "constituent_mixture_counts"]
# The spin ceilings each constituent of a combined (``Mixed``) set DECLARES in
# its LALInference ``config_file/engine/a_spin{1,2}-max`` (approximation A5 of
# the v2 build plan): JSON ``{constituent_label: [a1_max, a2_max] or null}``
# over the same-prefix siblings, "" for a non-Mixed row.  A Mixed set's spin
# prior is inherited from ONE sibling's analytic group; this records what the
# OTHER constituents (e.g. C01:SEOBNRv4PHM, which has no analytic group)
# declared, so a per-half ceiling difference is visible on the row.
META_STR_FIELDS += ["spin_amax_config_per_constituent"]

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
    #: Cosmology for an O1-O3 row whose distance prior declares none and which
    #: is NOT a reweighted `_cosmo` release (those use the release-reweight
    #: table below).  Chosen by CATALOG since GW-40a, not by label prefix.
    o3_default_cosmo: tuple = (PLANCK15.H0.value, PLANCK15.Om0)
    o4_fallback_cosmo: tuple = (O4_FALLBACK.H0.value, O4_FALLBACK.Om0)
    #: The release-reweight cosmology table (GW-40a): path to a YAML table, or
    #: None for the bundled production table
    #: (``gwcat/data/release_reweight_cosmology.yaml``, operator decision OD-2).
    #: Every `_cosmo` (release_reweighted) row takes its (H0, Om0) from the row
    #: for its CATALOG; a catalog the table lacks is refused, never defaulted.
    release_reweight_cosmology_table: Optional[str] = None
    #: Build a ``C00:Mixed`` row's prior as the equal-weight mixture of its
    #: constituents' own normalised analytic priors (GW-40f), with the mixing
    #: fractions verified row by row, instead of borrowing one sibling's.
    #: Needed only for label policies that USE C00:Mixed (BUILD_PLAN OD-1
    #: options b/c).  Off by default so existing stores and the v1 exporter
    #: are unchanged; a row whose mixture cannot be verified is refused.
    constituent_mixture_prior: bool = False
    validate_prior: bool = True
    compression: str = "gzip"
    #: Which UniformSourceFrame implementation evaluates p_dL_pe (GW-01).
    #: "exact" (the default since GW-40i) evaluates the density from accurate
    #: cosmology integrals, normalised over the declared bounds, at each row's
    #: resolved cosmology.  The LEGACY "auto" (the pre-GW-40i default) prefers
    #: bilby's 1000-point interpolated object and falls back to astropy only
    #: when bilby is not installed; "bilby"/"astropy" pin one of those.  The
    #: CLI's --legacy-grid-priors selects "auto", to reproduce older stores.
    dL_prior_impl: str = "exact"
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


#: Catalog -> observing-run era, for choosing IngestConfig defaults (GW-40a).
#: The defaults used to be keyed by the label PREFIX (``C01`` = O3), which is
#: wrong for GWTC-5's GW240925_005809: an O4 event released with ``C01`` labels
#: was ranked with the O3 waveform list and would have been given the O3
#: default cosmology.
CATALOG_ERA = {
    "GWTC-1": "O1-O3", "GWTC-2": "O1-O3", "GWTC-2.1": "O1-O3",
    "GWTC-3": "O1-O3",
    "GWTC-4": "O4", "GWTC-4.1": "O4", "GWTC-5": "O4",
}


def _catalog_era(catalog, prefix: str = "") -> str:
    """``"O1-O3"`` or ``"O4"`` for ``catalog``; the prefix only as a last resort.

    An unrecognised catalog (a file name gwcat cannot place) keeps the historical
    prefix rule, so nothing that used to ingest changes behaviour.
    """
    era = CATALOG_ERA.get(str(catalog) if catalog is not None else "")
    if era is not None:
        return era
    return "O1-O3" if str(prefix).startswith("C01") else "O4"


def _default_cosmo_for(catalog, analysis: str, cfg: "IngestConfig"):
    """The per-catalog fallback cosmology for a prior that declares none."""
    era = _catalog_era(catalog, _label_parts(analysis)[0])
    return cfg.o3_default_cosmo if era == "O1-O3" else cfg.o4_fallback_cosmo


def _priority_for(catalog, prefix: str, cfg: "IngestConfig"):
    """The waveform-priority list for a file, chosen by CATALOG (GW-40a).

    The era's list is re-expressed with the FILE's own prefix, so an O4 file
    released with ``C01`` labels ranks ``C01:IMRPhenomXPHM-SpinTaylor`` where a
    ``C00`` file ranks ``C00:IMRPhenomXPHM-SpinTaylor``.
    """
    era = _catalog_era(catalog, prefix)
    base = cfg.o3_waveform_priority if era == "O1-O3" else cfg.o4_waveform_priority
    out = []
    for want in base:
        _p, b, v = _label_parts(want)
        lab = f"{prefix}:{b}" + (f":{v}" if v else "")
        if lab not in out:
            out.append(lab)
    return out


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


def select_analysis(analyses, prefix: str, cfg: IngestConfig, catalog=None):
    """Pick the single preferred analysis label for one PE file.

    This is the historical one-sample-set-per-event heuristic (kept as the
    default): prefer a combined ``Mixed`` set -- including the spin-variant
    spellings real releases use, see :func:`find_mixed_analyses` -- else walk the
    configured waveform-priority list, else fall back to the first analysis
    carrying the file's prefix.  :func:`select_analyses` builds on it to support
    ingesting several sample sets per event.
    """
    ordered = rank_analyses(analyses, prefix, cfg, catalog=catalog)
    if not ordered:
        raise RuntimeError(f"No usable analysis among {analyses}")
    return ordered[0]


def rank_analyses(analyses, prefix: str, cfg: IngestConfig, catalog=None):
    """Order the prefix's analyses by ingest preference (most preferred first).

    ``{prefix}:Mixed`` (if present) ranks first, then the configured
    waveform-priority list in order, then any remaining prefixed analyses in
    their original order.  The index into this list becomes each sample set's
    ``priority_rank``; the first element is what :func:`select_analysis` returns.

    ``catalog`` (GW-40a) chooses the waveform-priority list by observing-run
    era; without it the historical prefix rule applies.
    """
    prefixed = [a for a in analyses if _label_parts(a)[0] == prefix]
    if catalog is None:
        priority = (cfg.o3_waveform_priority if prefix == "C01"
                    else cfg.o4_waveform_priority)
    else:
        priority = _priority_for(catalog, prefix, cfg)
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
                    sample_sets="preferred", catalog=None):
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
            return [select_analysis(analyses, prefix, cfg, catalog=catalog)]
        if sample_sets == "all":
            ordered = rank_analyses(analyses, prefix, cfg, catalog=catalog)
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
    #: Where (H0, Om0) came from (GW-40a): "analytic_declared",
    #: "catalog_default", or a release-reweight table row's `source`.
    cosmology_source: str = ""
    #: sha256 of the release-reweight table used ("" when none was consulted).
    cosmology_table_sha256: str = ""
    #: Per-prior provenance (GW-40b): the label whose analytic group supplied
    #: the prior ("" when none did) and its kind -- "own_analytic",
    #: "sibling_inherited", "assumed_default" or "release_reweighted".
    prior_source_label: str = ""
    prior_source_kind: str = ""

    @property
    def cosmology(self):
        return make_cosmology(self.H0, self.Om0)


def resolve_dL_prior(catalog, analysis, analyses, priors, dL_samples,
                     cfg: IngestConfig, flavour: str = "native",
                     release_table=None):
    """Resolve the effective distance prior for one ingested analysis.

    Strategy for the bounds is unchanged: read analytic from the chosen
    analysis; if absent (e.g. the GWTC-2.1/3 ``Mixed`` sets, whose analytic AND
    prior-sample groups are both empty), search sibling analyses; if still absent
    use the bounds of the dL sample range.

    GW-02 added the distribution CLASS and an exact cosmology-token mapping.  The
    effective class is the declared one EXCEPT for a reweighted ``_cosmo``
    release, where it is UniformSourceFrame by release convention -- recorded as
    ``basis="release_reweighted"``.

    GW-40a: the COSMOLOGY of a ``release_reweighted`` row comes from the explicit
    per-catalog release-reweight table (``release_table``, or the one
    ``cfg.release_reweight_cosmology_table`` names, or the bundled production
    table), NEVER from the file's ``meta_data`` and never from a label-prefix
    default; a catalog the table lacks raises
    :class:`~gwcat.release_cosmology.ReleaseReweightCosmologyError`.  Any other
    row whose prior declares no cosmology takes the per-CATALOG default.

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
    src_label = analysis if parsed is not None else ""
    if parsed is None:
        for an in analyses:  # sibling search (handles O4 Mixed w/o priors)
            parsed = _try(an)
            if parsed is not None:
                src = f"analytic[{an}]"
                src_label = an
                break

    dmin = float(np.min(dL_samples))
    dmax = float(np.max(dL_samples))
    default_cosmo = _default_cosmo_for(catalog, analysis, cfg)

    reweighted = flavour == "cosmo"
    table_row, table_sha = None, ""
    if reweighted:
        table = release_table or load_release_cosmology_table(
            cfg.release_reweight_cosmology_table)
        table_row = table.lookup(catalog)
        table_sha = table.sha256

    if parsed is None:
        # No analytic anywhere (9 of the GWTC-2.1 rows).  A cosmo-flavour
        # release is still reweighted, so UniformSourceFrame remains the
        # effective prior; there is simply no declaration to compare it to.
        if reweighted:
            return ResolvedDLPrior(
                kind="UniformSourceFrame", H0=table_row.H0, Om0=table_row.Om0,
                dmin=dmin, dmax=dmax,
                source="default(no_analytic)",
                sampling_kind="", cosmology_name=table_row.name,
                flavour=flavour, basis="release_reweighted",
                cosmology_source=table_row.source,
                cosmology_table_sha256=table_sha,
                prior_source_label="", prior_source_kind="release_reweighted")
        H0, Om0 = default_cosmo
        return ResolvedDLPrior(
            kind="UniformSourceFrame", H0=H0, Om0=Om0, dmin=dmin, dmax=dmax,
            source="default(no_analytic)", sampling_kind="",
            flavour=flavour, basis="assumed_default",
            cosmology_source="catalog_default",
            prior_source_label="", prior_source_kind="assumed_default")

    if parsed.dmin is not None:
        dmin = parsed.dmin
    if parsed.dmax is not None:
        dmax = parsed.dmax

    sampling_kind = parsed.kind or ""
    if reweighted:
        # Reweighted release: the declared class is the sampling prior only,
        # and the cosmology is the table's -- whatever the stale repr says.
        H0, Om0 = table_row.H0, table_row.Om0
        src += f"+release_table[{catalog}]"
        return ResolvedDLPrior(
            kind="UniformSourceFrame", H0=H0, Om0=Om0, dmin=dmin, dmax=dmax,
            source=src, alpha=None, sampling_kind=sampling_kind,
            sampling_alpha=parsed.alpha, cosmology_name=table_row.name,
            flavour=flavour, basis="release_reweighted",
            cosmology_source=table_row.source,
            cosmology_table_sha256=table_sha,
            prior_source_label=src_label,
            prior_source_kind="release_reweighted")

    H0, Om0 = parsed.H0, parsed.Om0
    cosmology_source = "analytic_declared"
    if H0 is None:
        H0, Om0 = default_cosmo
        src += "+default_cosmo"
        cosmology_source = "catalog_default"
        if parsed.cosmology_recognized is False:
            warnings.warn(
                f"{analysis}: analytic distance prior names an unrecognised "
                f"cosmology {parsed.cosmology_name!r}; falling back to "
                f"(H0={H0}, Om0={Om0}). Add it to "
                f"gwcat.cosmology.NAMED_COSMOLOGIES if it is real -- the KS "
                f"check against the file's own prior samples is the only thing "
                f"standing between this guess and a wrong p_dL_pe.")

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

    if not sampling_kind:
        psk = "assumed_default"
    elif src_label == analysis:
        psk = "own_analytic"
    else:
        psk = "sibling_inherited"
    return ResolvedDLPrior(
        kind=kind, H0=H0, Om0=Om0, dmin=dmin, dmax=dmax, source=src,
        alpha=alpha, sampling_kind=sampling_kind,
        sampling_alpha=parsed.alpha, cosmology_name=parsed.cosmology_name,
        flavour=flavour, basis=basis, cosmology_source=cosmology_source,
        prior_source_label=src_label, prior_source_kind=psk)


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
    #: GW-40b: whose analytic group it came from, and the provenance kind
    #: ("own_analytic" / "sibling_inherited" / "assumed_default").
    source_label: str = ""
    source_kind: str = ""
    #: Parsed ``Constraint`` bounds on the detector-frame component masses
    #: (GW-40f; None when the group records none).
    m1_min: Optional[float] = None
    m1_max: Optional[float] = None
    m2_min: Optional[float] = None
    m2_max: Optional[float] = None


def resolve_mass_prior(analysis, analyses, priors, *, siblings=True):
    """Parse the analytic mass prior for one ingested analysis.

    Searches the chosen analysis then its siblings, exactly as the distance and
    spin priors do (a combined ``Mixed`` set carries no analytic group of its
    own).  Real releases record ``mass_1``/``mass_2`` as ``Constraint(1, 1000)``
    -- a *constraint*, not a prior -- with the actual prior on
    ``(chirp_mass, mass_ratio)``, so the class of that pair is what decides
    whether the mass Jacobian is right.

    ``siblings=False`` (GW-40f) restricts the search to ``analysis`` itself, for
    a constituent whose OWN prior is required.
    """
    analytic = priors.get("analytic", {}) if isinstance(priors, dict) else {}

    def _find(an):
        node = analytic.get(an) if isinstance(analytic, dict) else None
        if not node:
            return None
        got = {k: node[k] for k in ("chirp_mass", "mass_ratio") if k in node}
        if len(got) != 2:
            return None
        for k in ("mass_1", "mass_2"):
            if k in node:
                got[k] = node[k]
        return got

    found, src_an = _find(analysis), analysis
    if found is None and siblings:
        for an in analyses:
            found = _find(an)
            if found is not None:
                src_an = an
                break
    if found is None:
        return ResolvedMassPrior(kind="assumed_default",
                                 source="default(no_analytic_prior)",
                                 source_label="",
                                 source_kind="assumed_default")
    src_kind = "own_analytic" if src_an == analysis else "sibling_inherited"

    def _constraint(key):
        got = _parse_analytic_spin(found[key]) if key in found else None
        if got is None or got[0] != "Constraint":
            return None, None
        return (None if got[1] is None else float(got[1]),
                None if got[2] is None else float(got[2]))

    mc = _parse_analytic_spin(found["chirp_mass"])
    q = _parse_analytic_spin(found["mass_ratio"])
    kinds = (mc[0] if mc else None, q[0] if q else None)
    if kinds == _UNIFORM_IN_COMPONENTS:
        m1lo, m1hi = _constraint("mass_1")
        m2lo, m2hi = _constraint("mass_2")
        return ResolvedMassPrior(
            kind="uniform_detector_frame",
            chirp_min=None if mc[1] is None else float(mc[1]),
            chirp_max=None if mc[2] is None else float(mc[2]),
            q_min=None if q[1] is None else float(q[1]),
            q_max=None if q[2] is None else float(q[2]),
            source=f"analytic[{src_an}]",
            source_label=src_an, source_kind=src_kind,
            m1_min=m1lo, m1_max=m1hi, m2_min=m2lo, m2_max=m2hi)
    raw = "; ".join(f"{k}={str(found[k])[:80]!r}" for k in sorted(found)
                    if k in ("chirp_mass", "mass_ratio"))
    return ResolvedMassPrior(
        kind="unrecognized",
        source=f"analytic[{src_an}]:unrecognized({raw})",
        source_label=src_an, source_kind=src_kind)


@dataclass(frozen=True)
class ResolvedSpinPrior:
    """The spin-magnitude prior plus where it came from (GW-40b).

    ``kind``/``source`` are the legacy PR-2 fields (what the prior IS).
    ``source_label``/``source_kind`` say WHOSE declaration it is: the ingested
    label's own analytic group ("own_analytic"), a sibling's
    ("sibling_inherited"), a LALInference ``config_file/engine/a_spin{1,2}-max``
    pair ("config_file_declared"), or nothing ("assumed_default").
    """
    amax_1: float
    amax_2: float
    kind: str
    source: str
    source_label: str = ""
    source_kind: str = ""


#: The LALInference ``[engine]`` keys that declare the spin-magnitude ceilings.
_CONFIG_SPIN_KEYS = ("a_spin1-max", "a_spin2-max")


def _config_float(val):
    """A finite float from a config value (str / bytes / 1-element array)."""
    if isinstance(val, (bytes, bytearray)):
        val = val.decode()
    if isinstance(val, (list, tuple, np.ndarray)):
        arr = np.asarray(val).ravel()
        if arr.size != 1:
            return None
        val = arr[0]
        if isinstance(val, (bytes, bytearray)):
            val = val.decode()
    try:
        v = float(str(val).strip().strip("'\""))
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _spin_amax_from_config(data, analysis, analyses=()):
    """Spin ceilings DECLARED by a LALInference config (GW-40b).

    The six GWTC-2.1 LALInference events (GW170608_020116, GW190707_093326,
    GW190720_000836, GW190725_174728, GW190728_064510, GW190924_021846) carry no
    analytic priors group anywhere, so their spin prior used to be recorded as
    ``assumed_default``.  Their constituent runs' ``config_file/engine`` DOES
    declare ``a_spin1-max = a_spin2-max = 0.99``.

    Search order: ``analysis``'s own config, else every sibling that declares
    the keys.  Returns ``(amax_1, amax_2, labels, consistent)`` or ``None``;
    ``consistent`` is False when siblings declare different ceilings (then the
    caller must NOT upgrade the provenance -- a Mixed set built from runs with
    different ceilings is not a single declared prior).
    """
    cfg_all = getattr(data, "config", None)
    if not isinstance(cfg_all, dict):
        return None

    def _one(an):
        try:
            cfgd = cfg_all.get(an)
        except Exception:
            return None
        if not isinstance(cfgd, dict):
            return None
        eng = cfgd.get("engine")
        if not isinstance(eng, dict):
            return None
        vals = [_config_float(eng.get(k)) for k in _CONFIG_SPIN_KEYS]
        if any(v is None for v in vals):
            return None
        return float(vals[0]), float(vals[1])

    own = _one(analysis)
    if own is not None:
        return own[0], own[1], [analysis], True
    hits = [(an, _one(an)) for an in analyses if an != analysis]
    hits = [(an, v) for an, v in hits if v is not None]
    if not hits:
        return None
    vals = {v for _an, v in hits}
    a1, a2 = hits[0][1]
    return a1, a2, [an for an, _v in hits], len(vals) == 1


def constituent_spin_amax_config(data, analysis, analyses):
    """JSON ``{sibling: [a1_max, a2_max] | null}`` for a ``Mixed`` label.

    Every same-prefix, non-Mixed sibling of ``analysis`` is listed, with the
    ceilings its ``config_file/engine/a_spin{1,2}-max`` declares, or ``null``
    when its config declares none.  ``""`` for a label that is not a combined
    ``Mixed`` set, or when ``data`` carries no config at all.
    """
    import json as _json
    prefix, base, _variant = _label_parts(analysis)
    if base != "Mixed":
        return ""
    cfg_all = getattr(data, "config", None)
    if not isinstance(cfg_all, dict):
        return ""
    out = {}
    for an in analyses:
        p2, b2, _v2 = _label_parts(an)
        if an == analysis or p2 != prefix or b2 == "Mixed":
            continue
        got = _spin_amax_from_config(data, an, ())
        out[str(an)] = (None if got is None
                        else [float(got[0]), float(got[1])])
    return _json.dumps(out, sort_keys=True)


def resolve_spin_prior(analysis, analyses, priors, a1_samples, a2_samples,
                       fallback_amax=0.99, allow_variant_mismatch=False):
    """Return ``(amax_1, amax_2, kind, source)`` for the spin-magnitude prior.

    The legacy 4-tuple view of :func:`resolve_spin_prior_full`; see there.
    """
    r = resolve_spin_prior_full(
        analysis, analyses, priors, a1_samples, a2_samples,
        fallback_amax=fallback_amax,
        allow_variant_mismatch=allow_variant_mismatch)
    return r.amax_1, r.amax_2, r.kind, r.source


def resolve_spin_prior_full(analysis, analyses, priors, a1_samples,
                            a2_samples, fallback_amax=0.99,
                            allow_variant_mismatch=False, data=None):
    """Resolve the spin-magnitude prior AND its provenance (GW-40b).

    ``data`` (the pesummary read object) enables the LALInference config path:
    when no analytic spin prior exists for the label or any sibling, a
    ``config_file/engine/a_spin{1,2}-max`` declaration upgrades the provenance
    from ``assumed_default`` to ``config_file_declared`` and supplies the
    ceilings.  The legacy ``kind`` stays ``"assumed_default"`` there: the config
    declares the bounds, not the distribution class.

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
    src_label = analysis if found is not None else ""
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
            src_label = an
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
                src_label = other[0]
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
                src_label = best

    def _is_uniform_zero(p):
        return (p is not None and p[0] == "Uniform"
                and p[1] is not None and abs(p[1]) <= 1e-9
                and p[2] is not None)

    def _is_sine_or_absent(p):
        return p is None or p[0] == "Sine"

    src_kind = ("" if found is None else
                "own_analytic" if src_label == analysis else "sibling_inherited")
    if found is None:
        amax_1 = amax_2 = float(fallback_amax)
        kind = "assumed_default"
        source = "default(no_analytic_prior)"
        src_label, src_kind = "", "assumed_default"
        declared = (_spin_amax_from_config(data, analysis, analyses)
                    if data is not None else None)
        if declared is not None:
            c1, c2, clabels, consistent = declared
            where = ",".join(clabels)
            if consistent:
                amax_1, amax_2 = c1, c2
                source = (f"config_file[{where}]/engine/"
                          f"a_spin1-max={c1:g},a_spin2-max={c2:g}")
                src_label, src_kind = where, "config_file_declared"
            else:
                warnings.warn(
                    f"{analysis}: the constituent configs {clabels} declare "
                    f"DIFFERENT spin ceilings (engine/a_spin1-max, "
                    f"a_spin2-max); not treating them as one declared prior. "
                    f"Spin provenance stays assumed_default.")
                source += f" | config_file_inconsistent({where})"
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

    return ResolvedSpinPrior(amax_1=amax_1, amax_2=amax_2, kind=kind,
                             source=source, source_label=src_label,
                             source_kind=src_kind)


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


def _ks_against_prior_samples(dlp, *, kind, cosmo, dmin, dmax, alpha, impl,
                              return_outside=False):
    """KS distance between prior samples ``dlp`` and one candidate density.

    Both CDFs are conditioned on the SAME window (GW-37).  The model CDF was
    renormalized over ``[max(dmin, dlp.min()), min(dmax, dlp.max())]`` while the
    empirical CDF was left unconditional, so the statistic measured missing tail
    mass instead of shape: 20000 correct PowerLaw(alpha=2) draws on [10, 10000]
    score KS = 0.0087 against dmax=10000 but 0.4916 against dmax=8000 and 0.8750
    against dmax=5000 -- a correct parse rejected by a threshold of 0.05.

    A bounds mismatch is still a real defect (the module's own note says
    exceeding the threshold means "the parse / bounds / cosmology mapping is
    wrong", naming bounds first), so conditioning does not hide it: the fraction
    of prior samples falling OUTSIDE the recorded bounds is returned alongside,
    to be reported as what it is rather than disguised as a failed shape test.
    """
    lo = max(dmin, dlp.min())
    hi = min(dmax, dlp.max())
    inside = (dlp >= lo) & (dlp <= hi)
    # How much of the DECLARED prior's mass the samples never reach.  Nonzero in
    # BOTH directions of a bounds mismatch -- samples beyond the bounds leave
    # `inside` short, bounds wider than the samples leave model mass uncovered
    # -- which is what makes it a bounds diagnostic rather than half of one.
    frac_outside = float("nan")
    if not inside.any() or not np.isfinite(hi - lo) or hi <= lo:
        return (float("nan"), 1.0) if return_outside else float("nan")

    full = np.linspace(dmin, dmax, 2000)
    pdf_full = dL_prior_prob(full, kind=kind, cosmology=cosmo, dmin=dmin,
                             dmax=dmax, alpha=alpha, impl=impl)
    cdf_full = np.concatenate(
        [[0], np.cumsum(0.5 * (pdf_full[1:] + pdf_full[:-1]) * np.diff(full))])
    if cdf_full[-1] > 0:
        covered = (np.interp(hi, full, cdf_full)
                   - np.interp(lo, full, cdf_full)) / cdf_full[-1]
        # Samples outside the bounds count too: both are range disagreements.
        frac_outside = float(max(0.0, 1.0 - covered)
                             + (1.0 - inside.mean()))

    grid = np.linspace(lo, hi, 200)
    pdf = dL_prior_prob(grid, kind=kind, cosmology=cosmo, dmin=dmin,
                        dmax=dmax, alpha=alpha, impl=impl)
    cdf_model = np.concatenate(
        [[0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    if cdf_model[-1] <= 0:
        return (float("nan"), frac_outside) if return_outside else float("nan")
    cdf_model = cdf_model / cdf_model[-1]
    # Conditioned on the same window as the model, so the two are comparable.
    dlp_in = np.sort(dlp[inside])
    ecdf = np.searchsorted(dlp_in, grid, side="right") / dlp_in.size
    ks = float(np.max(np.abs(cdf_model - ecdf)))
    return (ks, frac_outside) if return_outside else ks


def validate_prior_against_samples(priors, analyses_to_try, resolved,
                                   *, impl: str = "exact"):
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
    ks_sampling, frac_outside = _ks_against_prior_samples(
        dlp, kind=sampling_kind, cosmo=cosmo, dmin=dmin, dmax=dmax,
        alpha=resolved.sampling_alpha, impl=impl, return_outside=True)
    ks_effective = (
        ks_sampling if sampling_kind == resolved.kind
        else _ks_against_prior_samples(
            dlp, kind=resolved.kind, cosmo=cosmo, dmin=dmin, dmax=dmax,
            alpha=resolved.alpha, impl=impl))
    return {"ks": ks_sampling, "ks_effective": ks_effective,
            "sampling_kind": sampling_kind, "effective_kind": resolved.kind,
            # Reported separately from the KS (GW-37) so a bounds mismatch is
            # diagnosed as a bounds mismatch instead of surfacing as a rejected
            # parse -- the two used to be conflated in one number.
            "frac_prior_bounds_uncovered": frac_outside,
            "n_prior_samples": int(dlp.size), "analysis": used}


# --------------------------------------------------------------------------
# Inspect a single file (smoke test before the full run)
# --------------------------------------------------------------------------
def inspect(path: str, cfg: Optional[IngestConfig] = None):
    cfg = cfg or IngestConfig()
    catalog = detect_catalog(path)
    data, samples_dict, analyses, priors = _read_event_pesummary(path)
    prefix = _prefix_for(analyses)
    analysis = select_analysis(analyses, prefix, cfg, catalog=catalog)
    ranked = rank_analyses(analyses, prefix, cfg, catalog=catalog)
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
    sp = resolve_spin_prior_full(
        analysis, analyses, priors, a1_samp, a2_samp,
        allow_variant_mismatch=cfg.spin_prior_allow_variant_mismatch,
        data=data)
    spin_amax_1, spin_amax_2, spin_kind, spin_src = (sp.amax_1, sp.amax_2,
                                                     sp.kind, sp.source)
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
                     "cosmology_source": res.cosmology_source,
                     "cosmology_table_sha256": res.cosmology_table_sha256,
                     "release_flavour": res.flavour, "basis": res.basis,
                     # GW-01: which implementation evaluates p_dL_pe, and how
                     # many samples the recorded bounds fail to cover.
                     "impl": dL_info["impl"],
                     "n_samples_outside_bounds": dL_info["n_outside_bounds"],
                     "frac_outside_bounds": dL_info["frac_outside_bounds"]},
        # Spin-magnitude prior resolution (PR 2).
        "spin_prior": {"amax_1": spin_amax_1, "amax_2": spin_amax_2,
                       "kind": spin_kind, "source": spin_src,
                       "source_label": sp.source_label,
                       "source_kind": sp.source_kind},
        # Per-prior source provenance (GW-40b).
        "prior_source": {
            "mass": {"label": mass_prior.source_label,
                     "kind": mass_prior.source_kind},
            "spin": {"label": sp.source_label, "kind": sp.source_kind},
            "dL": {"label": res.prior_source_label,
                   "kind": res.prior_source_kind}},
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


def _constituent_mixture_prior(catalog, analysis, analyses, samples_dict,
                               priors, cfg, flavour):
    """The equal-weight constituent-mixture prior of a ``C00:Mixed`` row.

    See :mod:`gwcat.constituent_mixture`.  Every contributing constituent must
    carry its OWN analytic mass (uniform in components), distance and spin
    priors; the spin priors and the distance-prior cosmology must agree across
    constituents; every Mixed row must equal one constituent row and each
    constituent must supply exactly ``n_Mixed / K`` rows.  Anything else is
    refused (:class:`~gwcat.constituent_mixture.ConstituentMixtureError`).
    """
    from .constituent_mixture import (ConstituentMixtureError,
                                      match_constituent_rows,
                                      mixture_prior_densities,
                                      truncated_dL_density, uic_normalisation)
    pfx, _base, variant = _label_parts(analysis)
    want = _spin_variant_token(variant)
    cands = {a: samples_dict[a] for a in analyses
             if a != analysis and _label_parts(a)[0] == pfx
             and _label_parts(a)[1] != "Mixed"
             and _spin_variant_token(_label_parts(a)[2]) == want}
    labels, counts = match_constituent_rows(samples_dict[analysis], cands)

    comps, amaxes, cosmos, dl_kinds, bad = [], set(), set(), set(), []
    for lab in labels:
        smp = samples_dict[lab]
        mp = resolve_mass_prior(lab, [lab], priors, siblings=False)
        rd = resolve_dL_prior(catalog, lab, [lab], priors,
                              np.asarray(smp["luminosity_distance"], float),
                              cfg, flavour=flavour)
        sp = resolve_spin_prior_full(
            lab, [lab], priors,
            np.asarray(smp["a_1"], float) if "a_1" in smp else None,
            np.asarray(smp["a_2"], float) if "a_2" in smp else None)
        if mp.kind != "uniform_detector_frame" or mp.source_kind != "own_analytic":
            bad.append(f"{lab}: mass prior {mp.kind}/{mp.source_kind}")
            continue
        if rd.prior_source_kind != "own_analytic":
            bad.append(f"{lab}: distance prior {rd.prior_source_kind}")
            continue
        if (sp.kind != "uniform_magnitude_isotropic"
                or sp.source_kind != "own_analytic"):
            bad.append(f"{lab}: spin prior {sp.kind}/{sp.source_kind}")
            continue
        amaxes.add((sp.amax_1, sp.amax_2))
        cosmos.add((rd.H0, rd.Om0, rd.cosmology_name))
        dl_kinds.add(rd.kind)
        bounds = dict(mc_min=mp.chirp_min, mc_max=mp.chirp_max,
                      q_min=mp.q_min, q_max=mp.q_max, m1_min=mp.m1_min,
                      m1_max=mp.m1_max, m2_min=mp.m2_min, m2_max=mp.m2_max)
        comps.append(dict(
            bounds, label=lab, Z=uic_normalisation(**bounds),
            dmin=rd.dmin, dmax=rd.dmax, dl_kind=rd.kind,
            p_dL=truncated_dL_density(rd.kind, rd.cosmology, rd.dmin, rd.dmax,
                                      rd.alpha, impl=cfg.dL_prior_impl)))
    if bad:
        raise ConstituentMixtureError(
            f"{analysis}: cannot build the constituent-mixture prior -- every "
            f"constituent must carry its own analytic priors: {bad}.")
    if len(amaxes) != 1:
        raise ConstituentMixtureError(
            f"{analysis}: constituents declare different spin priors "
            f"{sorted(amaxes)}; the chi_eff prior would not factor out of the "
            f"mixture. Refusing rather than approximating.")
    if len(cosmos) != 1:
        raise ConstituentMixtureError(
            f"{analysis}: constituents' distance priors use different "
            f"cosmologies {sorted(cosmos)}; one exported z(dL) cannot describe "
            f"them.")
    mixed = samples_dict[analysis]
    m1 = np.asarray(mixed["mass_1"], float)
    q = np.asarray(mixed["mass_2"], float) / m1
    dL = np.asarray(mixed["luminosity_distance"], float)
    joint, marg = mixture_prior_densities(m1, q, dL, comps)
    n_zero = int(np.sum(~(joint > 0)))
    if n_zero:
        raise ConstituentMixtureError(
            f"{analysis}: {n_zero} Mixed sample(s) lie outside EVERY "
            f"constituent's prior support; the mixture assigns them zero "
            f"density.")
    (a1, a2), = amaxes
    (H0, Om0, cname), = cosmos
    return dict(labels=labels, counts=counts, p_mass_dL=joint, p_dL=marg,
                amax_1=a1, amax_2=a2, H0=H0, Om0=Om0, cosmology_name=cname,
                dl_kind=(dl_kinds.pop() if len(dl_kinds) == 1 else "mixture"),
                dmin=min(c["dmin"] for c in comps),
                dmax=max(c["dmax"] for c in comps),
                chirp_min=min(c["mc_min"] for c in comps),
                chirp_max=max(c["mc_max"] for c in comps),
                q_min=min(c["q_min"] for c in comps),
                q_max=max(c["q_max"] for c in comps),
                Z={c["label"]: c["Z"] for c in comps})


def _release_pesummary_version(path) -> str:
    """The pesummary version that wrote a PE release file ("" if unreadable).

    Read from the file's own ``version/pesummary`` dataset with h5py (a few
    bytes; the samples are not touched).  It is provenance for the
    release-reweight table's citations, never an input to any density.
    """
    try:
        with h5py.File(path, "r") as f:
            if "version" in f and "pesummary" in f["version"]:
                v = np.asarray(f["version/pesummary"][()]).ravel()
                if v.size:
                    x = v[0]
                    return (x.decode() if isinstance(x, (bytes, bytearray))
                            else str(x))
    except (OSError, KeyError, TypeError, ValueError):
        pass
    return ""


def build_store(paths, out_path, params=None, extra_params=None,
                cfg: Optional[IngestConfig] = None, event_table=None,
                sample_sets="preferred", file_provenance: Optional[dict] = None,
                cache_dir=None, offline: Optional[bool] = None,
                write_summary: bool = False,
                summary_context: Optional[dict] = None,
                mock_data: bool = False):
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
    mock_data    : bool, default False
                   Flag the store as SYNTHETIC (:data:`STORE_MOCK_DATA_ATTR`).
                   The flag is part of the store, so every PE export built from
                   it carries ``mock_data=True`` without the exporter being
                   told.  Default False writes no attr: a real-data store is
                   byte-identical to one written before the flag existed.
    """
    cfg = cfg or IngestConfig()
    params = list(params or DEFAULT_PARAMS)
    if extra_params:
        params += [p for p in extra_params if p not in params]
    event_table = _resolve_event_table(event_table, cache_dir=cache_dir,
                                       offline=offline)
    file_provenance = file_provenance or {}
    # The release-reweight cosmology table (GW-40a) is loaded on first use by
    # resolve_dL_prior (cached per path+sha256), so a build with no `_cosmo`
    # file never reads it and every row records the sha256 of the bytes used.
    release_table = None

    records = []   # per row: (name, n_samples, {param: array}) -- union schema
    offsets = [0]
    names, meta = [], {k: [] for k in META_FLOAT_FIELDS + META_STR_FIELDS}

    for path in paths:
        catalog = detect_catalog(path)
        name = event_name_from_path(path)
        data, samples_dict, analyses, priors = _read_event_pesummary(path)
        prefix = _prefix_for(analyses)
        # Sample-set contract (PR 6): one or more analyses per file, each a row.
        # Defaults keyed by CATALOG, not label prefix (GW-40a).
        preferred_label = select_analysis(analyses, prefix, cfg,
                                          catalog=catalog)
        ranked = rank_analyses(analyses, prefix, cfg, catalog=catalog)
        labels = select_analyses(analyses, prefix, cfg, sample_sets,
                                 catalog=catalog)
        # Which labels the waveform-priority list actually matched, so
        # selection_reason can distinguish a priority hit from the last resort.
        priority_labels = _priority_matches(
            [a for a in analyses if _label_parts(a)[0] == prefix],
            _priority_for(catalog, prefix, cfg))
        pesummary_version = _release_pesummary_version(path)
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
                                   flavour=flavour,
                                   release_table=release_table)
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
                    # A bounds mismatch reported as a bounds mismatch (GW-37).
                    # The KS is now conditioned on the overlap window on both
                    # sides, so it no longer absorbs missing tail mass -- which
                    # means this has to be said separately or it stops being
                    # said at all.
                    frac_out = v.get("frac_prior_bounds_uncovered")
                    if (frac_out is not None and np.isfinite(frac_out)
                            and frac_out > cfg.dL_outside_warn_frac):
                        warnings.warn(
                            f"{name} [{analysis}]: the file's own distance "
                            f"PRIOR samples and the recorded prior bounds "
                            f"[{dmin:.4g}, {dmax:.4g}] Mpc from {src} describe "
                            f"different ranges -- {100 * frac_out:.2f}% of the "
                            f"declared prior's mass is unaccounted for (samples "
                            f"outside the bounds, or bounds the samples never "
                            f"reach). The KS is computed on the overlap and so "
                            f"does not see this; it usually means the bounds "
                            f"came from a sibling analysis.")
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
                if dL_info.get("widened"):
                    # Legacy interpolations: the table had to be widened.
                    how = (f"The density was evaluated over "
                           f"[{dL_info['eval_min']:.4g}, "
                           f"{dL_info['eval_max']:.4g}] Mpc instead of "
                           f"zeroing them")
                else:
                    # Exact / closed form: the same formula, normalised over
                    # the declared bounds, is evaluated at every sample.
                    how = ("The density's formula, normalised over the "
                           "declared bounds, was evaluated at those samples "
                           "instead of zeroing them")
                warnings.warn(
                    f"{name} [{analysis}]: {dL_info['n_outside_bounds']} of "
                    f"{dL_info['n_samples']} dL samples "
                    f"({100 * dL_info['frac_outside_bounds']:.2f}%) fall "
                    f"outside the recorded distance-prior bounds "
                    f"[{dmin:.4g}, {dmax:.4g}] Mpc from {src} "
                    f"({dL_info['n_below_dmin']} below, "
                    f"{dL_info['n_above_dmax']} above).  {how}, "
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
            sp = resolve_spin_prior_full(
                analysis, analyses, priors, a1_samp, a2_samp,
                allow_variant_mismatch=cfg.spin_prior_allow_variant_mismatch,
                data=data)
            spin_amax_1, spin_amax_2, spin_kind, spin_src = (
                sp.amax_1, sp.amax_2, sp.kind, sp.source)
            # ── Constituent-mixture prior for C00:Mixed (GW-40f, opt-in) ─────
            cmix = None
            if (cfg.constituent_mixture_prior
                    and _label_parts(analysis)[0] == "C00"
                    and _label_parts(analysis)[1] == "Mixed"):
                cmix = _constituent_mixture_prior(
                    catalog, analysis, analyses, samples_dict, priors, cfg,
                    flavour)
                rec["p_dL_pe"] = cmix["p_dL"]
                rec["p_mass_dL_pe"] = cmix["p_mass_dL"]
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
            # present; otherwise stay NaN (explicit absence).  The two spellings
            # are one quantity (see event_metadata.PASTRO_KEYS): resolve once
            # and write BOTH columns, so a table carrying only `p_astro` -- the
            # spelling the override/manifest documentation uses -- is not stored
            # as an absent `pastro` that every legacy reader then reads as
            # "unknown" (GW-14).
            p_astro = resolve_pastro(et)
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
            # ── Per-prior source provenance (GW-40b) ────────────────────────
            meta["prior_source_label_mass"].append(mp.source_label)
            meta["prior_source_kind_mass"].append(mp.source_kind)
            meta["prior_source_label_spin"].append(sp.source_label)
            meta["prior_source_kind_spin"].append(sp.source_kind)
            meta["spin_amax_config_per_constituent"].append(
                constituent_spin_amax_config(data, analysis, analyses))
            meta["prior_source_label_dL"].append(res.prior_source_label)
            meta["prior_source_kind_dL"].append(res.prior_source_kind)
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
            meta["pastro"].append(p_astro)
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
            meta["dL_prior_cosmology_source"].append(res.cosmology_source)
            meta["dL_prior_cosmology_table_sha256"].append(
                res.cosmology_table_sha256)
            meta["release_pesummary_version"].append(pesummary_version)
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
            if cmix is None:
                meta["constituent_mixture_labels"].append("")
                meta["constituent_mixture_counts"].append("")
            else:
                joined = "+".join(cmix["labels"])
                over = {
                    "constituent_mixture_labels": joined,
                    "constituent_mixture_counts": ",".join(
                        str(cmix["counts"][lab]) for lab in cmix["labels"]),
                    "mass_prior_kind": "constituent_mixture",
                    "mass_prior_source": f"constituent_mixture[{joined}]",
                    "mass_prior_chirp_min": cmix["chirp_min"],
                    "mass_prior_chirp_max": cmix["chirp_max"],
                    "mass_prior_q_min": cmix["q_min"],
                    "mass_prior_q_max": cmix["q_max"],
                    "dL_prior_kind": cmix["dl_kind"],
                    "dL_prior_basis": "constituent_mixture",
                    "dL_prior_source": f"constituent_mixture[{joined}]",
                    "dL_prior_min": cmix["dmin"],
                    "dL_prior_max": cmix["dmax"],
                    "dL_prior_H0": float(cmix["H0"]),
                    "dL_prior_Om0": float(cmix["Om0"]),
                    "dL_prior_cosmology_name": cmix["cosmology_name"],
                    "spin_amax_1": float(cmix["amax_1"]),
                    "spin_amax_2": float(cmix["amax_2"]),
                    "spin_prior_kind": "uniform_magnitude_isotropic",
                    "spin_prior_source": f"constituent_mixture[{joined}]",
                }
                for p in ("mass", "spin", "dL"):
                    over[f"prior_source_label_{p}"] = joined
                    over[f"prior_source_kind_{p}"] = "constituent_mixture"
                meta["constituent_mixture_labels"].append(
                    over.pop("constituent_mixture_labels"))
                meta["constituent_mixture_counts"].append(
                    over.pop("constituent_mixture_counts"))
                # Every other field was already appended for this row from the
                # sibling-borrowed resolution; the mixture REPLACES it.
                for k, val in over.items():
                    meta[k][-1] = val
                src = f"constituent_mixture[{joined}]"
            print(f"[{catalog}] {name}: {n} samp, sample_set={analysis}, "
                  f"prior={src}")

    # Assemble the UNION of parameters across events, NaN-filling event slices
    # where a parameter is absent, and build the per-event availability mask.
    # Derived spin columns (PR 2) are appended to the candidate list so they are
    # stored and correctly marked available even when a caller passed a custom
    # ``params`` that omitted them (a column absent from every rec is dropped by
    # _assemble_union, so this is harmless when nothing was derived).
    candidate_params = list(params) + ["p_dL_pe", "p_mass_dL_pe"]
    for p in ("cos_tilt_1", "cos_tilt_2", "chi_p"):
        if p not in candidate_params:
            candidate_params.append(p)
    # Written event slice by event slice into preallocated columns: the union is
    # never concatenated in memory, so the ingest peak is the records alone
    # rather than the records plus a second copy of the whole catalog (GW-24).
    union_params = _write_store_from_records(out_path, records,
                                             candidate_params, offsets, names,
                                             meta, cfg, mock_data=mock_data)
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


def _union_params_and_avail(records, candidate_params):
    """The stored column order and the per-event availability mask.

    Split out of :func:`_assemble_union` so the streaming writer
    (:func:`_write_store_from_records`) decides WHAT to store by exactly the
    same rule as the concatenating one, without building any column.
    """
    seen, ordered = set(), []
    for p in candidate_params:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    union_params = [p for p in ordered
                    if any(p in rec for (_n, _c, rec) in records)]
    avail = np.zeros((len(records), len(union_params)), dtype=bool)
    for j, p in enumerate(union_params):
        for i, (_name, _n, rec) in enumerate(records):
            avail[i, j] = p in rec
    return union_params, avail


def _assemble_union(records, candidate_params):
    """Assemble union-schema columns + an availability mask from per-event data.

    Holds the whole catalog a second time (the concatenated columns) on top of
    ``records``; :func:`_write_store_from_records` is the streaming alternative
    that ingest itself uses.

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
    union_params, avail = _union_params_and_avail(records, candidate_params)
    columns = {}
    for p in union_params:
        chunks = [np.asarray(rec[p], dtype=np.float64) if p in rec
                  else np.full(n, np.nan, dtype=np.float64)
                  for (_name, n, rec) in records]
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

#: 1.4 adds the release-reweight cosmology provenance (GW-40a:
#: dL_prior_cosmology_source / _table_sha256, release_pesummary_version) and
#: the per-prior source provenance (GW-40b: prior_source_{label,kind}_{mass,
#: spin,dL}).  Every one is read-optional, so 1.1-1.3 stores still load; they
#: simply carry no record of where each prior came from.
SCHEMA_VERSION_PRIOR_PROVENANCE = "1.4"

#: File-level store attr marking a store whose samples are SYNTHETIC (a mock
#: campaign written through gwcat's own store writer), not released PE.
#:
#: Provenance travels with the data: the flag is written into the store by
#: :func:`build_store` / :func:`_write_store_from_records` (``mock_data=True``),
#: survives :func:`merge_stores` / :func:`merge_store` (a merge holding any mock
#: row is mock), is read by :class:`gwcat.catalog.GWCatalog`, and is stamped by
#: every PE exporter as the ``mock_data`` attr darksirens reads.  It is written
#: ONLY when True, so a real-data store is byte-identical to what gwcat wrote
#: before the flag existed; an absent attr reads as False, which is what every
#: store written before it means.
STORE_MOCK_DATA_ATTR = "mock_data"


def read_store_mock_data(f) -> bool:
    """Whether an open store (or any ``h5py`` object with ``attrs``) is mock."""
    return bool(f.attrs.get(STORE_MOCK_DATA_ATTR, False))


#: The meta columns whose presence marks a 1.3 store.
_SCHEMA_13_FIELDS = ("dL_prior_kind", "dL_prior_sampling_kind",
                     "dL_prior_cosmology_name", "dL_prior_release_flavour",
                     "dL_prior_basis", "dL_prior_impl", "dL_prior_alpha",
                     "dL_prior_sampling_alpha", "dL_prior_ks",
                     "n_samples_outside_dL_prior_bounds")


#: The meta columns whose (non-empty) presence marks a 1.4 store.
_SCHEMA_14_FIELDS = ("dL_prior_cosmology_source", "prior_source_kind_spin",
                     "prior_source_kind_mass", "prior_source_kind_dL")


def _store_schema_version(meta):
    """The schema version a store carrying ``meta`` advertises."""
    # Bump the schema version to 1.2 only when sample-set columns are present,
    # so a store with none still advertises 1.1 and loads unchanged.
    has_sampleset = any(k in meta for k in
                        SAMPLE_SET_STR_FIELDS + SAMPLE_SET_FLOAT_FIELDS)
    # 1.3 when the distance-prior provenance is present (GW-01/GW-02), else 1.2
    # when sample-set columns are, else 1.1.
    has_dL_prov = any(k in meta and len(meta[k]) for k in _SCHEMA_13_FIELDS)
    has_prior_prov = any(k in meta and len(meta[k]) and any(meta[k])
                         for k in _SCHEMA_14_FIELDS)
    if has_prior_prov:
        return SCHEMA_VERSION_PRIOR_PROVENANCE
    if has_dL_prov:
        return SCHEMA_VERSION_DL_PRIOR
    if has_sampleset:
        return SCHEMA_VERSION_SAMPLESETS
    return SCHEMA_VERSION


def _write_store_attrs(f, stored_params, names, meta, mock_data=False):
    """Write the file-level attributes (schema, column names, row count).

    ``mock_data`` adds :data:`STORE_MOCK_DATA_ATTR` -- only when True, so a
    real-data store's attrs are exactly what they were before the flag existed.
    """
    f.attrs["schema_version"] = _store_schema_version(meta)
    f.attrs.create("param_names",
                   np.array(stored_params, dtype=h5py.string_dtype()))
    f.attrs["n_events"] = len(names)
    if mock_data:
        f.attrs[STORE_MOCK_DATA_ATTR] = True


def _write_store_index(f, offsets, names, avail, meta, cfg):
    """Write everything but ``samples/``: the index, the mask and the meta."""
    dt_str = h5py.string_dtype(encoding="utf-8")
    idx = f.create_group("index")
    idx.create_dataset("offsets", data=np.asarray(offsets, dtype=np.int64))
    idx.create_dataset("event_names", data=np.array(names, dtype=object),
                       dtype=dt_str)
    # Per-event x per-parameter availability mask (rows aligned with
    # index/event_names, columns aligned with attrs/param_names).
    ag = f.create_group("avail")
    ag.create_dataset("mask", data=np.asarray(avail, dtype=bool),
                      compression=cfg.compression)
    mg = f.create_group("meta")
    for k in META_FLOAT_FIELDS:
        mg.create_dataset(k, data=np.asarray(meta[k], dtype=np.float64))
    for k in META_STR_FIELDS:
        mg.create_dataset(k, data=np.array(meta[k], dtype=object), dtype=dt_str)


def _write_store(out_path, stored_params, columns, offsets, names, avail, meta,
                 cfg, mock_data=False):
    """Write a store.h5 with the union parameter set + availability mask.

    ``columns`` maps each stored parameter to a full-length (already
    concatenated) 1-D array.  ``avail`` is a (n_events, n_params) bool mask
    aligned with ``names`` (rows) and ``stored_params`` (columns).
    """
    with h5py.File(out_path, "w") as f:
        _write_store_attrs(f, stored_params, names, meta, mock_data)
        g = f.create_group("samples")
        for p in stored_params:
            arr = np.asarray(columns.get(p, np.array([])), dtype=np.float64)
            g.create_dataset(p, data=arr, compression=cfg.compression,
                             shuffle=True)
        _write_store_index(f, offsets, names, avail, meta, cfg)


def _create_sample_column(g, p, n, cfg):
    """A preallocated store column: NaN everywhere nothing is written.

    The fill value is what lets a writer skip the slices an event does not
    provide instead of materialising a NaN block for them -- an unwritten
    region reads back exactly the NaN the concatenating writer stored.
    """
    return g.create_dataset(p, shape=(int(n),), dtype=np.float64,
                            fillvalue=np.nan, compression=cfg.compression,
                            shuffle=True)


def _write_store_from_records(out_path, records, candidate_params, offsets,
                              names, meta, cfg, mock_data=False):
    """Write a store straight from per-event arrays, one event slice at a time.

    The concatenating path (:func:`_assemble_union` + :func:`_write_store`)
    holds the finished union alongside ``records``, i.e. the whole catalog twice
    -- ~4 GiB for the 282-row production ingest.  Here each event's samples go
    from the record into its slice of a preallocated column, so the union is
    never built in memory (GW-24).

    ``mock_data=True`` flags the store as synthetic (:data:`STORE_MOCK_DATA_ATTR`);
    this is how a mock campaign written through this writer is exported with
    ``mock_data=True`` instead of being labelled real data.

    Returns the stored parameter list (the union, in ``candidate_params`` order).
    """
    union_params, avail = _union_params_and_avail(records, candidate_params)
    total = int(offsets[-1]) if len(offsets) else 0
    with h5py.File(out_path, "w") as f:
        _write_store_attrs(f, union_params, names, meta, mock_data)
        g = f.create_group("samples")
        for p in union_params:
            ds = _create_sample_column(g, p, total, cfg)
            for i, (_name, _n, rec) in enumerate(records):
                if p not in rec:
                    continue                      # NaN-filled by construction
                ds[int(offsets[i]):int(offsets[i + 1])] = np.asarray(
                    rec[p], dtype=np.float64)
        _write_store_index(f, offsets, names, avail, meta, cfg)
    return union_params


# --------------------------------------------------------------------------
# Store read / merge helpers (schema-preserving)
# --------------------------------------------------------------------------
def _decode(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def _read_store_meta(path):
    """Read a store's index, mask and meta -- everything EXCEPT the samples.

    This is what merging and appending need to decide WHAT to write: the sample
    arrays themselves are then copied slice by slice straight between the HDF5
    files (see :func:`_write_store_streaming`), never materialised in memory.
    Reading the 282-row / 6.9M-sample production catalog this way costs
    milliseconds and a few MB, against ~5 s and 2.1 GiB for the full read.

    The returned dict is :func:`_read_store`'s minus ``samples``, plus:

    ``path``    the file the samples still live in;
    ``slices``  each row's ``(start, stop)`` sample range in that file's columns.

    Derives an all-True availability mask for legacy stores that predate the
    ``avail/mask`` dataset -- exact for those stores because the old
    intersection ingest guaranteed every stored column was present for every
    event.
    """
    with h5py.File(path, "r") as f:
        params = [_decode(p) for p in f.attrs["param_names"]]
        mock_data = read_store_mock_data(f)
        offsets = f["index/offsets"][:].astype(np.int64)
        names = [_decode(n) for n in f["index/event_names"][:]]
        n_events = len(names)
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
    slices = [(int(offsets[i]), int(offsets[i + 1])) for i in range(n_events)]
    return dict(params=params, offsets=offsets, names=names, avail=avail,
                meta=meta, n_events=n_events, path=str(path), slices=slices,
                mock_data=mock_data)


def _read_store(path):
    """Read a store.h5 into an in-memory dict (schema-agnostic).

    Loads every column in full, so the peak memory is the whole catalog (2.1 GiB
    for the production store).  Prefer :func:`_read_store_meta` whenever the
    samples are only going to be copied elsewhere.
    """
    S = _read_store_meta(path)
    with h5py.File(path, "r") as f:
        S["samples"] = {p: f[f"samples/{p}"][:] for p in S["params"]}
    return S


def _subset_meta(S, keep):
    """Restrict a store's metadata (:func:`_read_store_meta`) to rows ``keep``.

    The sample ranges follow the rows, so the streaming writer still knows where
    each surviving row's samples live in the source file; nothing is copied here.
    """
    keep = list(keep)
    slices = [S["slices"][i] for i in keep]
    offs = [0]
    for a, b in slices:
        offs.append(offs[-1] + (b - a))
    avail = S["avail"][keep, :] if keep else S["avail"][:0, :]
    meta = {k: [v[i] for i in keep] for k, v in S["meta"].items()}
    return dict(params=S["params"], offsets=np.asarray(offs, dtype=np.int64),
                names=[S["names"][i] for i in keep], avail=avail, meta=meta,
                n_events=len(keep), path=S["path"], slices=slices,
                mock_data=bool(S.get("mock_data", False)))


#: Largest number of float64 samples moved in one read/write step when the merge
#: streams event slices between stores (1 Mi samples = 8 MiB).  This is what
#: bounds the merge's memory by construction: one production column is 55 MB and
#: a whole catalog 2.1 GiB, and the old merge held two of those plus the
#: concatenation of both.
_COPY_BLOCK = 1 << 20


def _copy_runs(slices, dst_start):
    """Coalesce ``(start, stop)`` source ranges into ``(start, stop, dst)`` runs.

    Rows that are contiguous in the source AND land contiguously in the output
    -- the usual case, since a merge keeps whole stores -- become one run, so the
    copy costs one bounded read per block rather than one per event.
    """
    runs = []
    dst = int(dst_start)
    for a, b in slices:
        a, b = int(a), int(b)
        if b <= a:
            continue
        if runs:
            r0, r1, rd = runs[-1]
            if r1 == a and rd + (r1 - r0) == dst:      # contiguous both sides
                runs[-1] = (r0, b, rd)
                dst += b - a
                continue
        runs.append((a, b, dst))
        dst += b - a
    return runs


def _copy_column(src_ds, dst_ds, runs):
    """Copy ``runs`` from one column to another in <= ``_COPY_BLOCK`` steps."""
    for a, b, dst in runs:
        for s in range(a, b, _COPY_BLOCK):
            e = min(s + _COPY_BLOCK, b)
            dst_ds[dst + (s - a):dst + (e - a)] = src_ds[s:e]


def _write_store_streaming(out_path, stored_params, sources, offsets, names,
                           avail, meta, cfg, mock_data=False):
    """Write a store whose sample columns are COPIED from existing stores.

    ``sources`` is a list of ``(store metadata dict, destination start offset)``
    pairs, in output row order; each dict comes from :func:`_read_store_meta` (or
    :func:`_subset_meta`) and carries the source path and each kept row's sample
    range.  Columns are preallocated at their final length with a NaN fill
    value, so a parameter a source lacks needs no NaN block written at all -- it
    reads back NaN exactly as the concatenating writer produced it -- and no more
    than ``_COPY_BLOCK`` samples are in memory at a time.
    """
    total = int(offsets[-1]) if len(offsets) else 0
    with h5py.File(out_path, "w") as f:
        _write_store_attrs(f, stored_params, names, meta, mock_data)
        g = f.create_group("samples")
        dsets = {p: _create_sample_column(g, p, total, cfg)
                 for p in stored_params}
        for S, dst_start in sources:
            if not S["n_events"]:
                continue
            runs = _copy_runs(S["slices"], dst_start)
            wanted = [p for p in stored_params if p in S["params"]]
            if not (runs and wanted):
                continue
            with h5py.File(S["path"], "r") as sf:
                for p in wanted:
                    _copy_column(sf[f"samples/{p}"], dsets[p], runs)
        _write_store_index(f, offsets, names, avail, meta, cfg)


def _row_keys(S):
    """The identity of each row of an in-memory store: ``(name, sample_set)``.

    A row of the ragged store is one ``(event, sample_set)`` pair -- that is the
    uniqueness the sample-set contract declares (see SAMPLE_SET_STR_FIELDS) and
    what ``sample_sets="all"`` produces.  Stores written before the contract, and
    the single-sample-set default, carry ``sample_set_name = ""`` for every row,
    so keying on the pair reduces to keying on the name for them.
    """
    ss = S["meta"].get("sample_set_name") or [""] * S["n_events"]
    return [(n, str(ss[i]) if i < len(ss) else "")
            for i, n in enumerate(S["names"])]


def merge_stores(store_a, store_b, out_path, cfg: Optional[IngestConfig] = None,
                 skip_duplicates: bool = True,
                 on_duplicate_key: Optional[str] = None):
    """Merge two existing store.h5 files, PRESERVING the union of parameters.

    A parameter present in only one store becomes a full column in the output:
    the events from the store that lacked it are NaN-filled and marked
    unavailable in the availability mask.  No column is ever dropped because one
    store is missing it.  Meta fields merge as a union too, with explicit-absence
    defaults (NaN for floats, "" for strings).

    Row identity is ``(event_name, sample_set_name)`` -- the key the sample-set
    contract declares -- NOT the event name.  A ``store_b`` row whose event
    already appears in ``store_a`` under a DIFFERENT sample set is therefore
    added, not dropped: that is how a second waveform's samples enter an
    existing catalog (GW-14).

    on_duplicate_key : {"skip", "refresh", "keep"} or None
        What to do with a ``store_b`` row whose full key matches a ``store_a``
        row.  ``"skip"`` drops it and warns (the historical behaviour, and the
        default).  ``"refresh"`` REPLACES the matching ``store_a`` row with
        ``store_b``'s -- how one sample set's samples or metadata are updated
        without re-ingesting the whole catalog; the refreshed row moves to the
        end of the store, which is positional only (rows are addressed by key).
        ``"keep"`` appends both, leaving two rows with the same key -- a store
        state the readers treat as duplicate and which is only ever what you
        want mid-pipeline.  ``None`` derives it from the legacy
        ``skip_duplicates`` flag (True -> "skip", False -> "keep").

    Neither store is ever held in memory: only their indices, masks and meta are
    read, and the samples are then copied event slice by event slice straight
    from the input files into preallocated output columns (GW-24).  Merging into
    the 1.7 GB production catalog therefore costs a few MB rather than the ~6 GiB
    the two full reads plus their concatenation used to.

    Returns the output path.
    """
    cfg = cfg or IngestConfig()
    if on_duplicate_key is None:
        on_duplicate_key = "skip" if skip_duplicates else "keep"
    if on_duplicate_key not in ("skip", "refresh", "keep"):
        raise ValueError(
            f"on_duplicate_key={on_duplicate_key!r} is invalid; use 'skip', "
            f"'refresh' or 'keep'")
    A = _read_store_meta(store_a)
    B = _read_store_meta(store_b)

    if on_duplicate_key != "keep":
        a_keys = _row_keys(A)
        b_keys = _row_keys(B)
        dupes = set(a_keys) & set(b_keys)
        if dupes:
            listed = sorted(f"{n}[{s}]" if s else n for n, s in dupes)
            if on_duplicate_key == "skip":
                warnings.warn(f"Duplicate events skipped from the second "
                              f"store: {listed}")
                B = _subset_meta(B, [i for i, k in enumerate(b_keys)
                                     if k not in dupes])
            else:
                warnings.warn(f"Refreshed from the second store (the first "
                              f"store's rows are replaced): {listed}")
                A = _subset_meta(A, [i for i, k in enumerate(a_keys)
                                     if k not in dupes])

    # Union parameter order: store A's columns first, then B's new columns.
    union_params = list(A["params"]) + [p for p in B["params"]
                                        if p not in A["params"]]
    a_total = int(A["offsets"][-1]) if A["n_events"] else 0

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

    # p_astro / pastro are one quantity under two spellings (GW-14).  A store
    # ingested before both columns were written -- or from a table that used
    # only one key -- contributes NaN under the other, and the merged file must
    # not present that NaN as knowledge it does not have when the value is
    # sitting in the sibling column of the same row.
    resolved_pastro = resolve_pastro_column(merged_meta, n_total)
    for k in PASTRO_KEYS:
        merged_meta[k] = list(resolved_pastro)

    # The copy reads the inputs WHILE writing the output, so an in-place merge
    # -- out_path IS one of the inputs, which the old in-memory merge allowed
    # because it had already loaded both -- lands on a sibling temp file and is
    # renamed over the target once complete.
    # Staged unconditionally (GW-37), not only when out_path is an input: the
    # non-in-place branch used to open an arbitrary pre-existing out_path with
    # h5py.File(..., "w"), so a merge that raised part-way left the file that
    # was already there destroyed. A previous good store is not ours to lose
    # because the merge producing its replacement failed.
    # A merge holding ANY synthetic row is synthetic: the flag is file-level,
    # and labelling a partly-mock store as real is the error it exists to stop.
    a_mock, b_mock = bool(A.get("mock_data")), bool(B.get("mock_data"))
    mock_data = a_mock or b_mock
    if a_mock != b_mock:
        warnings.warn(
            f"merging a mock-data store with a real-data store "
            f"({store_a!r} mock_data={a_mock}, {store_b!r} mock_data={b_mock}); "
            f"the merged store is flagged mock_data=True, and so is every "
            f"export built from it.")

    write_path = f"{out_path}.gwcat-merge-tmp"
    try:
        _write_store_streaming(write_path, union_params, [(A, 0), (B, a_total)],
                               offsets, names, avail, merged_meta, cfg,
                               mock_data=mock_data)
    except BaseException:
        if os.path.exists(write_path):
            os.remove(write_path)
        raise
    os.replace(write_path, out_path)
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
                offline: Optional[bool] = None,
                on_duplicate_key: str = "skip"):
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
    on_duplicate_key : {"skip", "refresh"}
        What to do when a new file yields a row whose
        ``(event_name, sample_set_name)`` already exists in ``existing_path``.
        ``"skip"`` (default) keeps the existing row; ``"refresh"`` replaces it
        with the newly ingested one -- the way to update one waveform's samples
        without re-ingesting the catalog.  Note that a new SAMPLE SET for an
        event already in the store is not a duplicate at all: it is added under
        either setting (GW-14).

    Returns
    -------
    str : path to the merged store.
    """
    import shutil, tempfile

    cfg = cfg or IngestConfig()
    event_table = _resolve_event_table(event_table, cache_dir=cache_dir,
                                       offline=offline)
    out_path = out_path or existing_path

    # Inspection is metadata-only: the existing store's columns are needed to
    # decide which parameters to ingest for the new events, not its samples,
    # which merge_stores then copies straight from the file (GW-24).
    old = _read_store_meta(existing_path)

    # Candidate columns for the new events: the generous default set plus any
    # columns the existing store already has (minus the computed p_dL_pe, which
    # build_store always appends) plus any user extras.  This lets the new
    # events keep their own extra columns (e.g. tidal params) which merge_stores
    # then unions with the existing schema.
    candidates = []
    for p in list(DEFAULT_PARAMS) + list(old["params"]) + list(extra_params or []):
        if p != "p_dL_pe" and p not in candidates:
            candidates.append(p)

    # Stage BESIDE the destination, not in $TMPDIR (GW-37).  The default call is
    # in-place -- out_path is existing_path -- and /tmp is a different device
    # from any real store here, so `shutil.move` always fell through
    # `os.rename`'s EXDEV to the copy fallback, which opens the LIVE store 'wb'
    # and truncates it before writing a byte.  Any walltime kill, quota or
    # ENOSPC inside a multi-GB copy destroyed the catalog it was merging into.
    # A sibling temp dir makes the final step `os.replace`: atomic, same device,
    # and the original survives every failure -- the pattern `merge_stores`
    # already uses for its own in-place branch one screen up.
    dest_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    tmpdir = tempfile.mkdtemp(dir=dest_dir, prefix=".gwcat-merge-")
    try:
        tmp_new = os.path.join(tmpdir, "new.h5")
        build_store(new_paths, tmp_new, params=candidates, cfg=cfg,
                    event_table=event_table, sample_sets=sample_sets,
                    file_provenance=file_provenance)

        tmp_merged = os.path.join(tmpdir, "merged.h5")
        merge_stores(existing_path, tmp_new, tmp_merged, cfg=cfg,
                     on_duplicate_key=on_duplicate_key)
        os.replace(tmp_merged, out_path)
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
    ap.add_argument("--release-reweight-cosmology-table", default=None,
                    metavar="YAML",
                    help="Release-reweight cosmology table for the GWTC-2.1/3 "
                         "_cosmo releases (GW-40a). Default: the bundled "
                         "gwcat/data/release_reweight_cosmology.yaml (LAL "
                         "Planck15 67.90/0.3065, cited). The pre-GW-40 "
                         "astropy behaviour is bundled as "
                         "release_reweight_cosmology_legacy_astropy.yaml, for "
                         "regressions only.")
    ap.add_argument("--legacy-grid-priors", action="store_true",
                    help="Evaluate p_dL_pe with the pre-GW-40i interpolated "
                         "implementation (bilby's 1000-point UniformSourceFrame, "
                         "astropy fallback) instead of the exact one. For "
                         "reproducing stores built before GW-40i only (the v1 "
                         "regression); recorded per row as dL_prior_impl.")
    ap.add_argument("--constituent-mixture-prior", action="store_true",
                    help="Build each C00:Mixed row's prior as the equal-weight "
                         "mixture of its constituents' own normalised priors, "
                         "with the mixing fractions verified row by row "
                         "(GW-40f; needed only for label policies that use "
                         "C00:Mixed). Default: off.")
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

    cfg = IngestConfig(
        release_reweight_cosmology_table=a.release_reweight_cosmology_table,
        constituent_mixture_prior=a.constituent_mixture_prior,
        dL_prior_impl="auto" if a.legacy_grid_priors else "exact")
    build_store(paths, a.out, event_table=event_table, sample_sets=sample_sets,
                cache_dir=a.cache_dir, offline=offline, cfg=cfg,
                file_provenance=file_provenance, write_summary=write_summary)


if __name__ == "__main__":
    _cli()