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
                        PLANCK15, O4_FALLBACK)
from .source_class import (normalize_source_class, classify_by_mass,
                          DEFAULT_NSBH_MASS_THRESHOLD)

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
]
META_STR_FIELDS = [
    "name", "catalog", "analysis_used", "dL_prior_source",
    "mass_prior_kind", "compact_type",
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
                         "file_name", "file_checksum", "record_id"]
META_FLOAT_FIELDS += SAMPLE_SET_FLOAT_FIELDS
META_STR_FIELDS += SAMPLE_SET_STR_FIELDS

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


def select_analysis(analyses, prefix: str, cfg: IngestConfig):
    """Pick the single preferred analysis label for one PE file.

    This is the historical one-sample-set-per-event heuristic (kept as the
    default): prefer the combined ``{prefix}:Mixed`` set; else walk the
    configured waveform-priority list; else fall back to the first analysis
    carrying the file's prefix.  :func:`select_analyses` builds on it to support
    ingesting several sample sets per event.
    """
    mixed = f"{prefix}:Mixed"
    if mixed in analyses:
        return mixed
    priority = cfg.o3_waveform_priority if prefix == "C01" else cfg.o4_waveform_priority
    for a in priority:
        if a in analyses:
            return a
    # last resort: first non-meta analysis
    for a in analyses:
        if a.startswith(prefix):
            return a
    raise RuntimeError(f"No usable analysis among {analyses}")


def rank_analyses(analyses, prefix: str, cfg: IngestConfig):
    """Order the prefix's analyses by ingest preference (most preferred first).

    ``{prefix}:Mixed`` (if present) ranks first, then the configured
    waveform-priority list in order, then any remaining prefixed analyses in
    their original order.  The index into this list becomes each sample set's
    ``priority_rank``; the first element is what :func:`select_analysis` returns.
    """
    prefixed = [a for a in analyses if a.startswith(prefix)]
    priority = cfg.o3_waveform_priority if prefix == "C01" else cfg.o4_waveform_priority
    ordered = []
    mixed = f"{prefix}:Mixed"
    if mixed in prefixed:
        ordered.append(mixed)
    for a in priority:
        if a in prefixed and a not in ordered:
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


def _waveform_family(approximant: str) -> str:
    """Coarse waveform family from an approximant/analysis token.

    ``'IMRPhenomXPHM-SpinTaylor' -> 'IMRPhenomXPHM'``;
    ``'SEOBNRv5PHM' -> 'SEOBNRv5PHM'``; ``'Mixed' -> 'Mixed'``.  Splits off a
    trailing configuration suffix after the first ``'-'`` so a
    ``strict-approximant`` request on the bare family still matches.
    """
    if not approximant:
        return ""
    return approximant.split("-", 1)[0]


def _sample_set_meta(analysis: str, preferred_label: str, ranked, path: str,
                     sample_sets, provenance: Optional[dict] = None) -> dict:
    """Per-row sample-set provenance for one ingested analysis label.

    ``is_preferred`` marks the label the default (single-set) heuristic would
    have chosen, and ``priority_rank`` is its index in :func:`rank_analyses`, so
    the ``preferred`` waveform policy can reproduce that choice downstream.
    ``record_id`` / ``file_checksum`` default to "" unless ``provenance`` (a
    ``{"record_id":..., "file_checksum":...}`` dict keyed by the source file's
    basename -- see ``build_store(file_provenance=...)``, PR 8) supplies them.
    """
    provenance = provenance or {}
    approximant = analysis.split(":", 1)[1] if ":" in analysis else analysis
    is_mixed = 1.0 if "mixed" in analysis.lower() else 0.0
    is_preferred = 1.0 if analysis == preferred_label else 0.0
    try:
        rank = float(list(ranked).index(analysis))
    except ValueError:
        rank = np.nan
    if is_preferred:
        reason = "preferred_mixed" if is_mixed else "preferred_priority"
    elif isinstance(sample_sets, str) and sample_sets == "all":
        reason = "ingested_all"
    else:
        reason = "ingested_explicit"
    return dict(
        sample_set_name=analysis,
        waveform=_waveform_family(approximant),
        approximant=approximant,
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


def _parse_analytic_dL(prior_string: str):
    """Parse a bilby UniformSourceFrame repr into (dmin, dmax, H0, Om0|None).

    Cosmology may be a named astropy cosmology or a FlatLambdaCDM(...) repr or
    absent. Returns H0/Om0 = None when not parseable.
    """
    s = str(prior_string)
    dmin = re.search(rf"minimum\s*=\s*({_NUM})", s)
    dmax = re.search(rf"maximum\s*=\s*({_NUM})", s)
    dmin = float(dmin.group(1)) if dmin else None
    dmax = float(dmax.group(1)) if dmax else None

    H0 = Om0 = None
    cm = re.search(r"cosmology\s*=\s*([^,\)]+)", s)
    if cm:
        tok = cm.group(1).strip()
        if "Planck15" in tok:
            H0, Om0 = PLANCK15.H0.value, PLANCK15.Om0
        else:
            h = re.search(rf"H0\s*=\s*({_NUM})", s)
            o = re.search(rf"Om0\s*=\s*({_NUM})", s)
            if h and o:
                H0, Om0 = float(h.group(1)), float(o.group(1))
    return dmin, dmax, H0, Om0


def resolve_dL_prior(catalog, analysis, analyses, priors, dL_samples, cfg: IngestConfig):
    """Return (H0, Om0, dmin, dmax, source_label).

    Strategy: read analytic from the chosen analysis; if absent (e.g. O4 Mixed),
    search sibling analyses; if still absent (e.g. GWTC-2.1), use the catalog
    default cosmology with bounds from the dL sample range.
    """
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
    if parsed is not None:
        p_dmin, p_dmax, H0, Om0 = parsed
        if p_dmin is not None:
            dmin = p_dmin
        if p_dmax is not None:
            dmax = p_dmax
        if H0 is None:  # analytic gave bounds but no cosmology
            H0, Om0 = (cfg.o3_default_cosmo if analysis.startswith("C01")
                       else cfg.o4_fallback_cosmo)
            src += "+default_cosmo"
        return H0, Om0, dmin, dmax, src

    # No analytic anywhere (GWTC-2.1)
    H0, Om0 = (cfg.o3_default_cosmo if analysis.startswith("C01")
               else cfg.o4_fallback_cosmo)
    return H0, Om0, dmin, dmax, "default(no_analytic)"


def validate_prior_against_samples(priors, analyses_to_try, H0, Om0, dmin, dmax):
    """If prior 'samples' exist (under any of analyses_to_try), check our
    UniformSourceFrame reproduces their dL density. Returns {ks, n, analysis}
    or None. pesummary keys prior samples by the *constituent* analyses, not by
    'Mixed', so we search a list."""
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
    cosmo = make_cosmology(H0, Om0)
    grid = np.linspace(max(dmin, dlp.min()), min(dmax, dlp.max()), 200)
    pdf = uniform_source_frame_prob(grid, cosmo, dmin, dmax)
    cdf_model = np.concatenate([[0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    cdf_model /= cdf_model[-1]
    ecdf = np.searchsorted(np.sort(dlp), grid, side="right") / dlp.size
    ks = float(np.max(np.abs(cdf_model - ecdf)))
    return {"ks": ks, "n_prior_samples": int(dlp.size), "analysis": used}


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
    H0, Om0, dmin, dmax, src = resolve_dL_prior(catalog, analysis, analyses,
                                                priors, dL, cfg)
    val = (validate_prior_against_samples(priors, [analysis] + analyses,
                                          H0, Om0, dmin, dmax)
           if cfg.validate_prior else None)
    f_ref = _read_f_ref(data, analysis)
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
        "dL_prior": {"H0": H0, "Om0": Om0, "min": dmin, "max": dmax, "source": src},
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
        et = event_table.get(name, {})
        prov = file_provenance.get(os.path.basename(path), {})

        for analysis in labels:
            s = samples_dict[analysis]

            n = len(np.asarray(s["luminosity_distance"]))
            # Union schema: keep EVERY candidate parameter this event actually
            # has.  Parameters a given event lacks are NaN-filled for that
            # event's slice later; we never drop a whole column because one
            # event lacks it.
            rec = {p: np.asarray(s[p], dtype=np.float64) for p in params if p in s}

            dL = np.asarray(s["luminosity_distance"], float)
            H0, Om0, dmin, dmax, src = resolve_dL_prior(
                catalog, analysis, analyses, priors, dL, cfg)
            if cfg.validate_prior:
                v = validate_prior_against_samples(priors, [analysis] + analyses,
                                                   H0, Om0, dmin, dmax)
                if v and v["ks"] > 0.05:
                    warnings.warn(f"{name}: prior KS={v['ks']:.3f} > 0.05 "
                                  f"(assumed cosmology may be wrong)")
            # distance prior evaluated per sample, stored mass-prior-agnostic
            p_dL = uniform_source_frame_prob(dL, make_cosmology(H0, Om0),
                                             dmin, dmax)
            rec["p_dL_pe"] = p_dL
            records.append((name, n, rec))

            # metadata
            m1s = (float(np.median(s["mass_1_source"]))
                   if "mass_1_source" in s else np.nan)
            m2s = (float(np.median(s["mass_2_source"]))
                   if "mass_2_source" in s else np.nan)
            snr = (float(np.median(s["network_optimal_snr"]))
                   if "network_optimal_snr" in s else np.nan)
            f_ref = _read_f_ref(data, analysis)

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
            meta["mass_prior_kind"].append("uniform_detector_frame")
            meta["compact_type"].append(compact)
            # canonical source-class metadata (parallel to legacy compact_type)
            meta["source_class"].append(source_class_val)
            meta["source_class_method"].append(source_class_method)
            meta["source_class_reference"].append(source_class_reference)
            meta["release"].append(str(et.get("release", catalog)))
            meta["observing_run"].append(
                str(et.get("observing_run", _observing_run_from_name(name))))
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
            meta["f_ref"].append(float(f_ref) if f_ref else np.nan)
            meta["nsamp_original"].append(float(n))
            # Sky area (optional; requires healpy)
            if "ra" in s and "dec" in s:
                meta["sky_area_90"].append(
                    _sky_area_90(np.asarray(s["ra"]), np.asarray(s["dec"])))
            else:
                meta["sky_area_90"].append(np.nan)
            # ── Sample-set / waveform contract (PR 6) ──────────────────────
            ss = _sample_set_meta(analysis, preferred_label, ranked, path,
                                  sample_sets, provenance=prov)
            for k, val in ss.items():
                meta[k].append(val)
            print(f"[{catalog}] {name}: {n} samp, sample_set={analysis}, "
                  f"prior={src}")

    # Assemble the UNION of parameters across events, NaN-filling event slices
    # where a parameter is absent, and build the per-event availability mask.
    union_params, columns, avail = _assemble_union(
        records, list(params) + ["p_dL_pe"])

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


def _read_f_ref(data, analysis):
    try:
        cfgd = data.config[analysis] if hasattr(data, "config") else {}
        for key in ("reference-frequency", "reference_frequency", "f_ref"):
            for sect in cfgd.values() if isinstance(cfgd, dict) else []:
                if isinstance(sect, dict) and key in sect:
                    return float(sect[key])
    except Exception:
        pass
    return None


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
    schema_version = SCHEMA_VERSION_SAMPLESETS if has_sampleset else SCHEMA_VERSION
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
    ap.add_argument("--metadata-overrides", default=None, metavar="PATH",
                    help="YAML/CSV event-metadata overrides. Values take "
                         "precedence over GWOSC metadata.")
    ap.add_argument("--metadata-diagnostics", default=None, metavar="PATH",
                    help="Write per-event metadata provenance diagnostics as "
                         "JSON. With --metadata-overrides, defaults to "
                         "<out>.metadata_diagnostics.json.")
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
    sample_sets = a.sample_sets
    if sample_sets not in ("preferred", "all"):
        sample_sets = [s.strip() for s in sample_sets.split(",") if s.strip()]

    file_provenance = None
    if a.file_provenance:
        with open(a.file_provenance) as f:
            file_provenance = json.load(f)

    offline = True if a.offline else None
    write_summary = default_write_summary and not a.no_summary

    event_table = {} if a.no_event_table else None
    summary_context = None
    if a.metadata_overrides or a.metadata_diagnostics:
        from .event_metadata import assemble_event_metadata, load_user_overrides
        from .fetch import fetch_event_table_gwosc

        overrides = (load_user_overrides(a.metadata_overrides)
                     if a.metadata_overrides else {})
        online_table = {}
        if not a.no_event_table:
            online_table = fetch_event_table_gwosc(
                cache_dir=a.cache_dir, offline=offline)

        event_names = list(dict.fromkeys(event_name_from_path(p) for p in paths))
        event_table, diagnostics = assemble_event_metadata(
            event_names,
            online_table=online_table,
            user_overrides=overrides,
        )

        diagnostics_path = a.metadata_diagnostics
        if a.metadata_overrides and diagnostics_path is None:
            diagnostics_path = f"{a.out}.metadata_diagnostics.json"
        if diagnostics_path is not None:
            with open(diagnostics_path, "w") as f:
                json.dump(diagnostics, f, indent=2)
                f.write("\n")

        if a.metadata_overrides:
            summary_context = {
                "metadata_overrides_path": str(a.metadata_overrides),
                "metadata_diagnostics_path": str(diagnostics_path),
                "n_metadata_overrides": len(overrides),
            }

    build_store(
        paths,
        a.out,
        event_table=event_table,
        sample_sets=sample_sets,
        cache_dir=a.cache_dir,
        offline=offline,
        file_provenance=file_provenance,
        write_summary=write_summary,
        summary_context=summary_context,
    )


if __name__ == "__main__":
    _cli()
