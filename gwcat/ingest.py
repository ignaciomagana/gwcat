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
from .source_class import normalize_source_class

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

# Default waveform priority when no Mixed set exists (O4b/GWTC-5 events).
O4_WAVEFORM_PRIORITY = [
    "C00:IMRPhenomXPHM-SpinTaylor", "C00:SEOBNRv5PHM",
    "C00:IMRPhenomXPNR", "C00:NRSur7dq4",
]
O3_WAVEFORM_PRIORITY = ["C01:IMRPhenomXPHM", "C01:SEOBNRv4PHM"]


@dataclass
class IngestConfig:
    nsbh_mass_threshold: float = 3.0      # Msun, source-frame, for classification
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
    if not np.isfinite(m1_src) or not np.isfinite(m2_src):
        return "unknown"
    if m1_src >= thr and m2_src >= thr:
        return "BBH"
    if m1_src >= thr and m2_src < thr:
        return "NSBH"
    return "BNS"


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


def _resolve_event_table(event_table):
    """Resolve the event_table argument for build_store / merge_store.

    None (default) → auto-fetch FAR/p_astro from GWOSC.
    {}             → skip (no network call).
    dict           → use as-is.
    """
    if event_table is not None:
        return event_table
    try:
        from .fetch import fetch_event_table_gwosc
        print("Fetching FAR/p_astro from GWOSC ...")
        table = fetch_event_table_gwosc()
        print(f"  got {len(table)} events from GWOSC")
        return table
    except Exception as e:
        warnings.warn(
            f"Could not auto-fetch event table from GWOSC: {e}\n"
            "  FAR/p_astro metadata will be NaN. Pass event_table={{}} to silence."
        )
        return {}


def build_store(paths, out_path, params=None, extra_params=None,
                cfg: Optional[IngestConfig] = None, event_table=None):
    """Ingest a list of cosmo-file paths into a single concatenated store.

    params       : column list to store (default DEFAULT_PARAMS).
    extra_params : appended to params (store anything extra without editing code).
    event_table  : {event_name: {'far':..,'pastro':..}} for FAR/p_astro,
                   which are NOT in per-event PE files.
                   None (default) → auto-fetch from GWOSC.
                   Pass {} to skip.
    """
    cfg = cfg or IngestConfig()
    params = list(params or DEFAULT_PARAMS)
    if extra_params:
        params += [p for p in extra_params if p not in params]
    event_table = _resolve_event_table(event_table)

    col_chunks = {p: [] for p in params}      # only filled for params present
    stored_params = None
    offsets = [0]
    names, meta = [], {k: [] for k in META_FLOAT_FIELDS + META_STR_FIELDS}

    for path in paths:
        catalog = detect_catalog(path)
        name = event_name_from_path(path)
        data, samples_dict, analyses, priors = _read_event_pesummary(path)
        prefix = _prefix_for(analyses)
        analysis = select_analysis(analyses, prefix, cfg)
        s = samples_dict[analysis]

        present = [p for p in params if p in s]
        if stored_params is None:
            stored_params = present
        elif present != stored_params:
            # keep the intersection stable across events
            inter = [p for p in stored_params if p in present]
            for p in set(stored_params) - set(inter):
                col_chunks.pop(p, None)
            stored_params = inter
        n = len(np.asarray(s["luminosity_distance"]))
        for p in stored_params:
            col_chunks[p].append(np.asarray(s[p], dtype=np.float64))

        dL = np.asarray(s["luminosity_distance"], float)
        H0, Om0, dmin, dmax, src = resolve_dL_prior(catalog, analysis, analyses,
                                                    priors, dL, cfg)
        if cfg.validate_prior:
            v = validate_prior_against_samples(priors, [analysis] + analyses,
                                               H0, Om0, dmin, dmax)
            if v and v["ks"] > 0.05:
                warnings.warn(f"{name}: prior KS={v['ks']:.3f} > 0.05 "
                              f"(assumed cosmology may be wrong)")
        # distance prior evaluated per sample, stored mass-prior-agnostic
        p_dL = uniform_source_frame_prob(dL, make_cosmology(H0, Om0), dmin, dmax)
        col_chunks.setdefault("p_dL_pe", []).append(p_dL)

        # metadata
        m1s = float(np.median(s["mass_1_source"])) if "mass_1_source" in s else np.nan
        m2s = float(np.median(s["mass_2_source"])) if "mass_2_source" in s else np.nan
        snr = (float(np.median(s["network_optimal_snr"]))
               if "network_optimal_snr" in s else np.nan)
        f_ref = _read_f_ref(data, analysis)
        et = event_table.get(name, {})

        # ── Source-class contract ──────────────────────────────────────────
        compact = _classify(m1s, m2s, cfg.nsbh_mass_threshold)
        far_val = float(et.get("far", np.nan))
        # far_available is an explicit state: True only when a finite FAR was
        # actually supplied by the event table (public metadata may omit it).
        far_available = 1.0 if np.isfinite(far_val) else 0.0
        # p_astro / component probabilities come from the event table when
        # present; otherwise stay NaN (explicit absence).
        p_astro = float(et.get("p_astro", et.get("pastro", np.nan)))
        metadata_source = "event_table" if et else "pe_file_only"

        names.append(name)
        offsets.append(offsets[-1] + n)
        meta["name"].append(name)
        meta["catalog"].append(catalog)
        meta["analysis_used"].append(analysis)
        meta["dL_prior_source"].append(src)
        meta["mass_prior_kind"].append("uniform_detector_frame")  # validated below
        meta["compact_type"].append(compact)
        # canonical source-class metadata (parallel to legacy compact_type)
        meta["source_class"].append(normalize_source_class(compact))
        meta["source_class_method"].append("mass_threshold")
        meta["source_class_reference"].append(
            f"m2_source<{cfg.nsbh_mass_threshold}Msun -> NS component")
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
        meta["f_ref"].append(float(f_ref) if f_ref else np.nan)
        meta["nsamp_original"].append(float(n))
        # Sky area (optional; requires healpy)
        if "ra" in s and "dec" in s:
            meta["sky_area_90"].append(
                _sky_area_90(np.asarray(s["ra"]), np.asarray(s["dec"])))
        else:
            meta["sky_area_90"].append(np.nan)
        print(f"[{catalog}] {name}: {n} samp, analysis={analysis}, prior={src}")

    if "p_dL_pe" not in stored_params:
        stored_params = stored_params + ["p_dL_pe"]

    _write_store(out_path, stored_params, col_chunks, offsets, names, meta, cfg)
    print(f"\nWrote {out_path}: {len(names)} events, "
          f"{offsets[-1]} total samples, params={stored_params}")
    return out_path


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


def _write_store(out_path, stored_params, col_chunks, offsets, names, meta, cfg):
    dt_str = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out_path, "w") as f:
        f.attrs["schema_version"] = "1.0"
        f.attrs.create("param_names",
                       np.array(stored_params, dtype=h5py.string_dtype()))
        f.attrs["n_events"] = len(names)
        g = f.create_group("samples")
        for p in stored_params:
            arr = np.concatenate(col_chunks[p]) if col_chunks.get(p) else np.array([])
            g.create_dataset(p, data=arr, compression=cfg.compression, shuffle=True)
        idx = f.create_group("index")
        idx.create_dataset("offsets", data=np.asarray(offsets, dtype=np.int64))
        idx.create_dataset("event_names", data=np.array(names, dtype=object), dtype=dt_str)
        mg = f.create_group("meta")
        for k in META_FLOAT_FIELDS:
            mg.create_dataset(k, data=np.asarray(meta[k], dtype=np.float64))
        for k in META_STR_FIELDS:
            mg.create_dataset(k, data=np.array(meta[k], dtype=object), dtype=dt_str)


# --------------------------------------------------------------------------
# Merge new events into an existing store
# --------------------------------------------------------------------------
def merge_store(existing_path: str, new_paths, out_path: str = None,
                cfg: Optional[IngestConfig] = None, event_table=None,
                extra_params=None):
    """Append new events to an existing store without re-ingesting everything.

    Parameters
    ----------
    existing_path : str
        Path to the existing store.h5.
    new_paths : list of str
        Paths to new PE files to add.
    out_path : str or None
        Output path.  None → overwrite existing_path (via a temp file for safety).
    cfg, event_table, extra_params :
        Same as build_store.  event_table=None auto-fetches from GWOSC.

    Returns
    -------
    str : path to the merged store.
    """
    import shutil, tempfile

    cfg = cfg or IngestConfig()
    event_table = _resolve_event_table(event_table)
    out_path = out_path or existing_path

    # Read existing store
    with h5py.File(existing_path, "r") as f:
        old_params = [p.decode() if isinstance(p, bytes) else str(p)
                      for p in f.attrs["param_names"]]
        old_offsets = f["index/offsets"][:]
        old_names = [n.decode() if isinstance(n, bytes) else str(n)
                     for n in f["index/event_names"][:]]
        old_samples = {p: f[f"samples/{p}"][:] for p in old_params}
        old_meta = {}
        for k in META_FLOAT_FIELDS:
            if k in f["meta"]:
                old_meta[k] = list(f[f"meta/{k}"][:])
        for k in META_STR_FIELDS:
            if k in f["meta"]:
                vals = f[f"meta/{k}"][:]
                old_meta[k] = [v.decode() if isinstance(v, bytes) else str(v)
                               for v in vals]

    # Build the new events into a temporary store
    tmpdir = tempfile.mkdtemp()
    tmp_new = os.path.join(tmpdir, "new.h5")
    params = list(old_params)
    if extra_params:
        params += [p for p in extra_params if p not in params]
    build_store(new_paths, tmp_new, params=params, cfg=cfg, event_table=event_table)

    # Read the new store
    with h5py.File(tmp_new, "r") as f:
        new_params = [p.decode() if isinstance(p, bytes) else str(p)
                      for p in f.attrs["param_names"]]
        new_offsets = f["index/offsets"][:]
        new_names = [n.decode() if isinstance(n, bytes) else str(n)
                     for n in f["index/event_names"][:]]
        new_samples = {p: f[f"samples/{p}"][:] for p in new_params}
        new_meta = {}
        for k in META_FLOAT_FIELDS:
            if k in f["meta"]:
                new_meta[k] = list(f[f"meta/{k}"][:])
        for k in META_STR_FIELDS:
            if k in f["meta"]:
                vals = f[f"meta/{k}"][:]
                new_meta[k] = [v.decode() if isinstance(v, bytes) else str(v)
                               for v in vals]

    # Check for duplicate events
    dupes = set(old_names) & set(new_names)
    if dupes:
        warnings.warn(f"Duplicate events will be skipped: {dupes}")
        keep = [i for i, n in enumerate(new_names) if n not in dupes]
        if not keep:
            print("No new events to add.")
            return existing_path
        new_names = [new_names[i] for i in keep]
        # Slice new samples and meta
        new_slices = [(new_offsets[i], new_offsets[i + 1]) for i in keep]
        for p in new_params:
            new_samples[p] = np.concatenate([new_samples[p][a:b] for a, b in new_slices])
        new_offsets_clean = [0]
        for a, b in new_slices:
            new_offsets_clean.append(new_offsets_clean[-1] + (b - a))
        new_offsets = np.array(new_offsets_clean, dtype=np.int64)
        for k in new_meta:
            new_meta[k] = [new_meta[k][i] for i in keep]

    # Merge: use intersection of params
    merged_params = [p for p in old_params if p in new_params]

    # Concatenate samples
    merged_samples = {}
    old_total = old_offsets[-1]
    for p in merged_params:
        merged_samples[p] = np.concatenate([old_samples[p], new_samples[p]])

    # Merge offsets
    merged_offsets = np.concatenate([
        old_offsets,
        new_offsets[1:] + old_total
    ]).astype(np.int64)

    # Merge names and meta
    merged_names = old_names + new_names
    merged_meta = {}
    for k in set(list(old_meta.keys()) + list(new_meta.keys())):
        old_v = old_meta.get(k, [np.nan] * len(old_names) if k in META_FLOAT_FIELDS
                             else [""] * len(old_names))
        new_v = new_meta.get(k, [np.nan] * len(new_names) if k in META_FLOAT_FIELDS
                             else [""] * len(new_names))
        merged_meta[k] = list(old_v) + list(new_v)

    # Write merged store
    tmp_merged = os.path.join(tmpdir, "merged.h5")
    dt_str = h5py.string_dtype(encoding="utf-8")
    with h5py.File(tmp_merged, "w") as f:
        f.attrs["schema_version"] = "1.0"
        f.attrs.create("param_names",
                       np.array(merged_params, dtype=h5py.string_dtype()))
        f.attrs["n_events"] = len(merged_names)
        g = f.create_group("samples")
        for p in merged_params:
            g.create_dataset(p, data=merged_samples[p],
                             compression=cfg.compression, shuffle=True)
        idx = f.create_group("index")
        idx.create_dataset("offsets", data=merged_offsets)
        idx.create_dataset("event_names",
                           data=np.array(merged_names, dtype=object), dtype=dt_str)
        mg = f.create_group("meta")
        for k in META_FLOAT_FIELDS:
            if k in merged_meta:
                mg.create_dataset(k, data=np.asarray(merged_meta[k], dtype=np.float64))
        for k in META_STR_FIELDS:
            if k in merged_meta:
                mg.create_dataset(k, data=np.array(merged_meta[k], dtype=object),
                                  dtype=dt_str)

    # Move to final location
    shutil.move(tmp_merged, out_path)
    shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"Merged store: {len(old_names)} + {len(new_names)} = "
          f"{len(merged_names)} events → {out_path}")
    return out_path


def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="Ingest GWTC cosmo files -> store.h5")
    ap.add_argument("--inspect", metavar="FILE", help="probe one file and exit")
    ap.add_argument("--glob", action="append", default=[],
                    help="glob of cosmo files (repeatable, one per catalog dir)")
    ap.add_argument("--out", default="store.h5")
    ap.add_argument("--no-event-table", action="store_true",
                    help="Skip auto-fetching FAR/p_astro from GWOSC.")
    a = ap.parse_args()
    if a.inspect:
        inspect(a.inspect)
        return
    paths = []
    for g in a.glob:
        paths += sorted(glob.glob(g))
    if not paths:
        ap.error("no files matched; pass --glob or --inspect")
    event_table = {} if a.no_event_table else None
    build_store(paths, a.out, event_table=event_table)


if __name__ == "__main__":
    _cli()