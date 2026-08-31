"""Selection function processing for LVK injection sets.

Supports two injection formats:

  * Modern 'events/' format (O4 sets, Zenodo 19500064 / 19500052)
  * Legacy 'injections/' format (O3 BBH, Zenodo 7890437)

Both formats go through the same processing pipeline:
  1. Read source-frame masses, spins, sky position
  2. Compute detector-frame masses, chi_eff, redshift
  3. Remove the spin component from the draw PDF
  4. Apply (m1src,m2src,z) → (m1det,q,dL) coordinate Jacobian
  5. Normalise by observing time and injection weights
  6. Apply FAR-based detection cut

CombinedSelectionSet merges multiple campaigns (e.g. O3 + O4ab) following
the multi-campaign VT estimator in Essick et al. (2023):
  ndraw = N_O3 + N_O4
  pdraw_i *= N_k / ndraw  for injection i from campaign k

Spin-prior contract (Mode A, matching the PE export)
----------------------------------------------------
On load, the injection spin-draw distribution is removed from the draw PDF
(step 3 above).  On export (``to_darksirens``), it is REPLACED by the 1-D
isotropic chi_eff prior — the "chi_eff swap" — so the exported ``pdraw``
already contains the 1-D chi_eff prior.  This is recorded as
``chi_eff_swap_applied=True``, ``chi_eff_prior_applied_to_pdraw=True``, and
``spin_prior_mode="include"``, consistent with the PE export's ``p_pe``.
Downstream (darksirens) MUST NOT multiply the chi_eff prior again — doing so
double-counts it.

Usage:
    from gwcat.selection import SelectionSet, CombinedSelectionSet

    # Single campaign
    sel = SelectionSet("injection_file.hdf")
    sel.to_darksirens("selection.h5", far_threshold=1.0)

    # Combined O3 + O4
    sel_o3 = SelectionSet("endo3_bbhpop-...-v12.hdf5")
    sel_o4 = SelectionSet("injections-O4ab/...-cartesian_spins_*.hdf")
    combined = CombinedSelectionSet([sel_o3, sel_o4])
    combined.to_darksirens("selection_bbh.h5", far_threshold=1.0)
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import h5py

from .cosmology import PLANCK15
from .source_class import (classify_by_mass, normalize_source_class,
                           resolve_filter_classes, format_source_class_filter,
                           DEFAULT_NSBH_MASS_THRESHOLD,
                           CUT_ESTIMATOR_ATTR, selection_cut_estimator)
from . import selection_spin as _sspin
from .spin import AMAX_AUTO, chi_p_from_components

# Human-readable description of what the exported ``pdraw`` represents after all
# of the code's manipulations (see the module docstring / to_darksirens).  Both
# the single and combined exporters write it verbatim so downstream code can
# read the state instead of re-deriving it.
PDRAW_STATE = (
    "draw_density_in_(m1det,q,dL)_basis_with_1D_chi_eff_prior_included; "
    "per-injection spin draw removed at load and replaced by the isotropic "
    "chi_eff prior on export (chi_eff swap); normalised by T_obs and injection "
    "weights. Detector-frame masses in Msun, dL in Mpc."
)

# ── Per-basis pdraw_state strings for the "gwcat-selection-2.0" writer ─────────
# Each describes truthfully what the exported ``pdraw`` represents in that spin
# basis.  ``chieff`` reuses the legacy PDRAW_STATE verbatim (the v2 chieff export
# reproduces the v1 pdraw array exactly).
PDRAW_STATE_CHIEFF = PDRAW_STATE

PDRAW_STATE_COMPONENT = (
    "draw_density_in_(m1det,q,dL,a1,a2,cost1,cost2)_basis; the per-injection "
    "component spin draw (spin magnitudes a_i and tilt cosines cosθ_i, spin "
    "azimuths marginalised out) is RETAINED exactly -- it is NOT swapped for a "
    "chi_eff prior; normalised by T_obs and injection weights. Detector-frame "
    "masses in Msun, dL in Mpc."
)

PDRAW_STATE_CHIEFF_CHIP = (
    "draw_density_in_(m1det,q,dL)_basis_with_2D_(chi_eff,chi_p)_prior_included; "
    "per-injection spin draw removed at load and replaced by the isotropic joint "
    "(chi_eff, chi_p) prior on export (chi_eff/chi_p swap); normalised by T_obs "
    "and injection weights. Detector-frame masses in Msun, dL in Mpc."
)

PDRAW_STATE_CHIEFF_REFERENCE = (
    "draw_density_in_(m1det,q,dL,chieff)_basis_with_1D_chi_eff_prior_included; "
    "built from the campaign's EXACT per-injection component spin draw and "
    "REWEIGHTED to a declared isotropic uniform-magnitude reference spin prior "
    "(ceiling spin_reference_amax) -- the campaign's own spin density is "
    "divided out, not discarded, so this is valid for a campaign of any spin "
    "distribution; rows outside the reference support carry weight exactly zero "
    "via spin_reference_excluded_pdraw; normalised by T_obs and injection "
    "weights. Detector-frame masses in Msun, dL in Mpc."
)

# pdraw_state keyed by spin basis (used by gwcat.export.selection_builder).
PDRAW_STATE_BY_BASIS = {
    "chieff": PDRAW_STATE_CHIEFF,
    "component": PDRAW_STATE_COMPONENT,
    "chieff_chip": PDRAW_STATE_CHIEFF_CHIP,
    "chieff_reference": PDRAW_STATE_CHIEFF_REFERENCE,
}

# Note recorded whenever a source-class filter subsets the injections: this is
# subsetting (Essick et al.), NOT a reweighting, so ndraw is left unchanged.
SOURCE_CLASS_FILTER_NOTE = (
    "source-class filtering subsets injections by injected source-frame mass; "
    "ndraw (total_generated) is NOT rescaled. The analyst MUST pair this "
    "selection file with a PE export filtered to the same source class(es)."
)


def _h5_field_names(table):
    """Return available column names for an HDF group or compound dataset."""
    if isinstance(table, h5py.Dataset) and table.dtype.names is not None:
        return set(table.dtype.names)
    return set(table.keys())


def _h5_has_field(table, name):
    """Whether an HDF group or compound dataset has a column/field."""
    return name in _h5_field_names(table)


def _h5_read_field(table, name, dtype=float):
    """Read one column from an HDF group or compound dataset."""
    if not _h5_has_field(table, name):
        available = sorted(_h5_field_names(table))
        raise KeyError(
            f"Field {name!r} not found. Available fields include: "
            f"{available[:30]}{' ...' if len(available) > 30 else ''}"
        )
    return np.asarray(table[name], dtype)


def _h5_first_field(table, names, dtype=float):
    """Read the first available column from a list of aliases."""
    for name in names:
        if _h5_has_field(table, name):
            return _h5_read_field(table, name, dtype), name
    available = sorted(_h5_field_names(table))
    raise KeyError(
        f"None of the fields {names!r} found. Available fields include: "
        f"{available[:30]}{' ...' if len(available) > 30 else ''}"
    )


# ── The events-format FIELD PLAN (GW-32) ──────────────────────────────────────
# The O4 campaigns ship ``events`` as ONE compound dataset of 119 columns and
# 994 bytes per record (2.8 GB for the clipped set).  h5py serves
# ``dset["column"]`` by reading the whole RECORD off disk and keeping one field
# of it, so the loader's ~35 column reads were ~35 passes over those 2.8 GB:
# 16.5 s and 1.01 GiB to load one file.  The plan below states, up front, every
# column the loader may ask for, so :class:`_PlannedFields` can fetch them
# together in ONE chunked ``Dataset.fields(...)`` pass (measured 4.8x faster on
# a 6-column subset of the real file: 0.47 s versus 2.24 s).
#
# The plan is a PERFORMANCE hint and never a correctness constraint: a column it
# missed is still served, by the direct read it always was.  It is grouped by
# the branch of the loader that wants it so the two cannot drift silently.

#: Masses, distance, sky, redshift, the draw weights and the shipped Jacobian.
_EVENTS_CORE_FIELDS = (
    "mass1_source", "mass2_source", "luminosity_distance",
    "right_ascension", "declination", "z", "redshift",
    "mass1_detector", "mass2_detector", "chi_eff", "weights",
    "dluminosity_distance_dredshift",
)
#: Cartesian spin components (read verbatim when the file ships them).
_EVENTS_SPIN_CARTESIAN = ("spin1x", "spin1y", "spin1z",
                          "spin2x", "spin2y", "spin2z")
#: Polar spin magnitude/tilt, preferred by :meth:`SelectionSet._read_spin_polar`.
_EVENTS_SPIN_POLAR = ("spin1_magnitude", "spin1_polar_angle",
                      "spin2_magnitude", "spin2_polar_angle")
#: Spin azimuths -- needed ONLY to rebuild cartesian components from polar ones.
_EVENTS_SPIN_AZIMUTH = ("spin1_azimuthal_angle", "spin2_azimuthal_angle")

#: The joint (masses, redshift, cartesian spins) log draw density.
_JOINT_CART_FIELD = (
    "lnpdraw_mass1_source_mass2_source_redshift_"
    "spin1x_spin1y_spin1z_spin2x_spin2y_spin2z"
)
#: The same density in polar spin coordinates (Format-C polar flavour).
_JOINT_POLAR_FIELD = (
    "lnpdraw_mass1_source_mass2_source_redshift_"
    "spin1_magnitude_spin1_polar_angle_spin1_azimuthal_angle_"
    "spin2_magnitude_spin2_polar_angle_spin2_azimuthal_angle"
)
#: A joint density that already excludes the spins.
_JOINT_NO_SPIN_FIELDS = ("lnpdraw_mass1_source_mass2_source_redshift",
                         "lnpdraw_mass1_source_mass2_source_z")
#: The factored public-O4 spin-free density.
_EVENTS_FACTORED_FIELDS = ("lnpdraw_mass1_source",
                           "lnpdraw_mass2_source_GIVEN_mass1_source",
                           "lnpdraw_z", "lnpdraw_redshift")
#: Columns only the component-basis spin state (PR4) reads.
_EVENTS_SPIN_STATE_FIELDS = (
    "chi_p",
    "lnpdraw_spin1_magnitude", "lnpdraw_spin2_magnitude",
    "lnpdraw_spin1_polar_angle", "lnpdraw_spin2_polar_angle",
    "lnpdraw_spin1_azimuthal_angle", "lnpdraw_spin2_azimuthal_angle",
)

#: Rows per ``Dataset.fields(...)`` block.  ~36 MB of transient buffer for a
#: 35-column plan, so the peak does not scale with the file.
_EVENTS_READ_BLOCK = 1 << 17


def _search_names(f):
    """The search pipelines an injection file declares, as a list of names."""
    try:
        raw = f.attrs["searches"]
    except Exception:
        return []
    if isinstance(raw, np.ndarray):
        return [x.decode() if isinstance(x, bytes) else str(x) for x in raw.flat]
    if isinstance(raw, (list, tuple)):
        return [x.decode() if isinstance(x, bytes) else str(x) for x in raw]
    if isinstance(raw, bytes):
        return [raw.decode()]
    if isinstance(raw, str):
        return [raw]
    return [str(raw)]


def _events_far_columns(table, f):
    """The FAR columns the events loader will threshold on, in its own order.

    One declared search contributes one column (``<search>_far`` or
    ``far_<search>``); a file that declares none falls back to every column that
    looks like a FAR.  Shared with :func:`_events_field_plan` so the plan asks
    for exactly the columns the read will use.
    """
    have = _h5_field_names(table)
    columns = []
    for s in _search_names(f):
        for col in (s + "_far", "far_" + s):
            if col in have:
                columns.append(col)
                break
    if not columns:
        columns = [key for key in have
                   if isinstance(key, str)
                   and (key.endswith("_far") or key.startswith("far_"))]
    return columns


def _events_field_plan(table, f):
    """Every column :meth:`SelectionSet._read_events` may read from ``table``."""
    have = _h5_field_names(table)
    candidates = (_EVENTS_CORE_FIELDS + _EVENTS_SPIN_CARTESIAN
                  + _EVENTS_SPIN_POLAR + (_JOINT_CART_FIELD, _JOINT_POLAR_FIELD)
                  + _JOINT_NO_SPIN_FIELDS + _EVENTS_FACTORED_FIELDS
                  + _EVENTS_SPIN_STATE_FIELDS)
    # The azimuths are read only to REBUILD the cartesian components, so a file
    # that ships those components is never asked for them.
    if not all(n in have for n in _EVENTS_SPIN_CARTESIAN):
        candidates = candidates + _EVENTS_SPIN_AZIMUTH
    plan = [n for n in candidates if n in have]
    plan += [c for c in _events_far_columns(table, f) if c not in plan]
    return plan


class _PlannedFields:
    """Read-through view over a compound dataset, filled in ONE chunked pass.

    Serves the :func:`_events_field_plan` columns from memory after a single
    ``Dataset.fields(plan)`` sweep, and anything else by the direct read it
    always was -- so the plan can only make the load faster, never wrong.  It
    exposes just the two things ``_h5_field_names`` / ``_h5_read_field`` need
    (``keys`` and ``__getitem__``), so the loader below is unchanged.

    A served column is DROPPED from the cache: ``_h5_read_field`` hands the
    array itself to the loader (``np.asarray`` of a float64 array is that
    array), so what the loader keeps is what stays alive and the prefetch does
    not add a second copy of the whole plan to the peak.  A column asked for
    twice therefore costs the second read it always did.
    """

    def __init__(self, dset, plan, block=_EVENTS_READ_BLOCK):
        self._dset = dset
        self._names = set(dset.dtype.names)
        # File order, so the one pass reads each record's fields in place.
        want = [n for n in dset.dtype.names if n in set(plan)]
        n = dset.shape[0]
        cache = {k: np.empty(n, dtype=dset.dtype[k]) for k in want}
        if want:
            view = dset.fields(want)
            for i in range(0, n, block):
                j = min(i + block, n)
                chunk = view[i:j]
                for k in want:
                    cache[k][i:j] = chunk[k]
        self._cache = cache

    def keys(self):
        return set(self._names)

    def __getitem__(self, name):
        try:
            return self._cache.pop(name)
        except KeyError:
            return self._dset[name]


def _planned_events_table(table, f):
    """``table``, wrapped for one-pass reads when it is a compound dataset.

    An ``events/`` GROUP already stores one dataset per column, so a read of it
    touches only that column and there is nothing to plan.
    """
    if isinstance(table, h5py.Dataset) and table.dtype.names is not None:
        return _PlannedFields(table, _events_field_plan(table, f))
    return table


def _require_positive_finite(value, field, path, note=None):
    """Return ``value`` after refusing any non-finite or non-positive entry.

    Every quantity the draw-density arithmetic multiplies or divides by -- the
    injected mass/redshift sampling PDFs, the mixture weights, the observing
    time, ``total_generated``, and the final ``pdraw`` itself -- must be a
    finite, strictly positive number.  A zero, a negative, a NaN or an infinity
    in any of them is a MALFORMED FILE, not a small number: flooring it (the
    pre-GW-28 ``np.maximum(p, 1e-300)``) fabricates a density that was never
    drawn, and dividing by it propagates ``inf``/``NaN`` into ``mu`` and
    silently through every posterior built on it.  So it is refused here, by
    name, before the arithmetic rather than after.

    Raises
    ------
    ValueError
        Naming ``field``, the file, how many entries are bad, and the first
        offending index/value so the malformed rows can be found.
    """
    arr = np.asarray(value, dtype=float)
    bad = ~np.isfinite(arr) | (arr <= 0)
    n_bad = int(np.count_nonzero(bad))
    if n_bad:
        flat = np.atleast_1d(bad).ravel()
        first = int(np.flatnonzero(flat)[0])
        vals = np.atleast_1d(arr).ravel()
        n_total = int(flat.size)
        raise ValueError(
            f"{path}: {field!r} is not finite and strictly positive -- "
            f"{n_bad} of {n_total} entries are invalid (first at index "
            f"{first}, value {vals[first]!r}). The selection maths multiplies "
            f"or divides by this quantity, so a zero/negative/NaN/inf here "
            f"makes pdraw meaningless; it is refused rather than floored or "
            f"divided through."
            + (f" {note}" if note else ""))
    return arr if arr.ndim else float(arr)


def _selection_provenance_dict(source_class, nsbh_mass_threshold,
                               n_before, n_after, far_columns, far_threshold):
    """Return the PR9 pdraw / source-class / significance provenance as a dict.

    Single source of truth for the provenance attrs written by BOTH the frozen
    v1 :meth:`SelectionSet.to_darksirens` / :meth:`CombinedSelectionSet.to_darksirens`
    exporters (via :func:`_write_selection_provenance`) and the versioned
    "gwcat-selection-2.0" selection builder.  Values are plain scalars / numpy
    arrays (the ``significance_columns`` array already carries the HDF5 string
    dtype) so either code path can assign them to ``h5py`` attrs directly.
    Records truthfully what the code did; it changes none of the math.
    """
    d = {}
    # ── pdraw state (v1 default; the v2 builder overrides per spin basis) ───
    d["pdraw_state"] = PDRAW_STATE

    # ── Source-class filter provenance ─────────────────────────────────────
    d["source_class_filter"] = format_source_class_filter(source_class)
    d["source_class_method"] = (
        "none" if source_class is None else "mass_threshold")
    # WHICH masses went through that threshold (GW-12).  source_class_method
    # records the classifier; this records the quantity, and the two sides of an
    # export pair shared the former while differing on the latter.  The paired
    # validators refuse a median-vs-truth pair; see
    # gwcat.source_class.assess_cut_estimator_pair.
    d[CUT_ESTIMATOR_ATTR] = selection_cut_estimator(source_class)
    d["nsbh_mass_threshold"] = float(nsbh_mass_threshold)
    d["n_injections_before_filter"] = int(n_before)
    d["n_injections_after_filter"] = int(n_after)
    if source_class is not None:
        d["source_class_filter_note"] = SOURCE_CLASS_FILTER_NOTE

    # ── Search / significance provenance (explicit-absence, per the FAR
    #    contract): record which columns/pipelines were thresholded, the
    #    threshold applied, and that no per-injection p_astro was used. ──────
    cols = [str(c) for c in (far_columns or [])]
    d["significance_columns"] = np.array(cols, dtype=h5py.string_dtype())
    d["significance_type"] = "far"
    d["significance_far_threshold"] = float(far_threshold)
    d["significance_available"] = bool(len(cols) > 0)
    # No per-injection p_astro is read/used for thresholding here; record the
    # absence explicitly rather than pretending it exists.
    d["p_astro_available"] = False
    return d


def _write_selection_provenance(f, source_class, nsbh_mass_threshold,
                                n_before, n_after, far_columns, far_threshold):
    """Write the PR9 pdraw / source-class / significance provenance attrs.

    Shared by :meth:`SelectionSet.to_darksirens` and
    :meth:`CombinedSelectionSet.to_darksirens` so the two exporters record the
    same contract in the same way.  A thin wrapper over
    :func:`_selection_provenance_dict` (the shared source of truth) that writes
    each item to ``f.attrs``; assigning the ``significance_columns`` string
    array via ``f.attrs[...] =`` is byte-equivalent to the previous
    ``f.attrs.create`` call, so v1 output is unchanged.
    """
    for k, v in _selection_provenance_dict(
            source_class, nsbh_mass_threshold, n_before, n_after,
            far_columns, far_threshold).items():
        f.attrs[k] = v


#: Candidate Om0 grid for :func:`detect_generation_cosmology`.  Coarse enough to
#: stay cheap at load time, fine enough that the residual test below either
#: identifies the cosmology to ~1e-7 or rejects outright.
_OM0_GRID_LO, _OM0_GRID_HI, _OM0_GRID_STEP = 0.20, 0.40, 0.0005

#: A campaign is only declared "identified" when every sampled (z, dL) pair is
#: reproduced to better than this.  The real endo3 BBH campaign hits 1.6e-7;
#: a wrong-but-nearby cosmology (Planck15 against endo3) sits at 2.3e-3, so
#: there are four orders of magnitude of daylight between match and miss.
COSMOLOGY_DETECT_RTOL = 1e-5

#: Detected H0 outside this band is rejected (-> fallback with a warning)
#: rather than reported.  dL scales exactly as 1/H0, so a unit mistake in the
#: file (dL in Gpc, say) is otherwise absorbed into H0 perfectly and comes
#: back as a CONFIDENT detection of H0=67900 -- and a pdraw Jacobian off by
#: the unit factor relative to every other campaign in a combined product.
COSMOLOGY_DETECT_H0_BAND = (40.0, 120.0)


def detect_generation_cosmology(z, dL_mpc, n_sample: int = 4000,
                                rtol: float = COSMOLOGY_DETECT_RTOL):
    """Recover the flat-LCDM cosmology a campaign's ``(z, dL)`` pairs were made with.

    Returns ``(H0, Om0, max_rel_residual)``, or ``None`` when no cosmology
    reproduces the pairs to ``rtol`` or the implied ``H0`` falls outside
    :data:`COSMOLOGY_DETECT_H0_BAND`.

    Exploits the fact that at fixed ``Om0`` a flat-LCDM ``dL`` scales exactly as
    ``1/H0`` (verified to 1e-12), so ``H0`` is solved in closed form per ``Om0``
    rather than searched: one distance evaluation per grid point, not a 2-D
    scan.  The coarse grid alone is NOT enough to accept: a half-grid-step Om0
    error leaves a ~2e-4 residual, 20x over ``rtol``, so a campaign generated
    at an off-grid Om0 (Planck18's 0.30966, say) would be reported as "not
    identified" and silently fall back to Planck15 -- the exact wrong-Jacobian
    bug this function exists to prevent.  The best grid point is therefore
    refined locally (the residual is smooth in Om0) before the ``rtol`` test.

    Used because the file's own generation cosmology -- not the caller's, and
    not Planck15 -- is what makes ``ddL/dz`` the right Jacobian for that
    campaign.
    """
    z = np.asarray(z, dtype=float).ravel()
    dL = np.asarray(dL_mpc, dtype=float).ravel()
    good = np.isfinite(z) & np.isfinite(dL) & (z > 0) & (dL > 0)
    if good.sum() < 16:
        return None
    z, dL = z[good], dL[good]
    if z.size > n_sample:
        # Deterministic stride, not a random draw: detection must not depend on
        # an rng, and a stride over a generated set spans the z range.
        step = int(np.ceil(z.size / n_sample))
        z, dL = z[::step], dL[::step]

    from .cosmology import make_cosmology

    H0_ref = 70.0

    def _try(Om0):
        pred_ref = make_cosmology(H0_ref, float(Om0)).luminosity_distance(z).value
        # dL ∝ 1/H0  =>  H0 = H0_ref * median(pred_ref / dL)
        H0 = H0_ref * float(np.median(pred_ref / dL))
        if not np.isfinite(H0) or H0 <= 0:
            return None
        resid = float(np.max(np.abs(pred_ref * (H0_ref / H0) / dL - 1.0)))
        return (H0, float(Om0), resid)

    best = None
    for Om0 in np.arange(_OM0_GRID_LO, _OM0_GRID_HI + 1e-12, _OM0_GRID_STEP):
        cand = _try(Om0)
        if cand is not None and (best is None or cand[2] < best[2]):
            best = cand
    if best is None:
        return None

    # Local Om0 refinement around the best grid point: three rounds of an
    # 11-point sub-grid, each a decade finer, take the Om0 error from
    # half-a-grid-step (~2.5e-4, residual ~2e-4) down to ~2.5e-7 (residual
    # ~2e-7), comfortably inside rtol for a genuinely-LCDM campaign while a
    # non-LCDM (z, dL) relation still cannot get anywhere near it.
    step = _OM0_GRID_STEP
    for _ in range(3):
        step /= 10.0
        lo, hi = best[1] - 10 * step, best[1] + 10 * step
        for Om0 in np.arange(lo, hi + step / 2, step):
            if not (0.0 < Om0 < 1.0):
                continue
            cand = _try(Om0)
            if cand is not None and cand[2] < best[2]:
                best = cand

    if best[2] > rtol:
        return None
    lo_H0, hi_H0 = COSMOLOGY_DETECT_H0_BAND
    if not (lo_H0 <= best[0] <= hi_H0):
        # A perfect fit at an absurd H0 is a unit error, not a cosmology.
        return None
    return best


def _cosmo_or_nan(s, attr):
    """A recorded per-campaign cosmology value, or NaN when none applies.

    NaN is the honest answer for an events-format campaign: its pdraw used the
    stored ddL/dz and therefore NO cosmology, which is a different statement
    from "the same cosmology as everyone else".
    """
    v = getattr(s, attr, None)
    return float("nan") if v is None else float(v)


def _cosmology_is_mixed(set_list) -> bool:
    """Whether the campaigns in this product disagree about their cosmology.

    True when they used different (H0, Om0), or when some used one and others
    used none at all.  A combined O3+O4 export is the live case: an explicit
    override changes endo3's pdraw (it computes ddL/dz) and not O4ab's (which
    reads it), so one product would carry two cosmologies while reporting one.
    """
    seen = {(getattr(s, "_cosmology_source", None),
             _cosmo_or_nan(s, "_cosmology_used_H0"),
             _cosmo_or_nan(s, "_cosmology_used_Om0"))
            for s in set_list}
    # NaN != NaN, so compare the rounded tuple form rather than the raw floats.
    norm = {(src, None if h != h else round(h, 9),
             None if o != o else round(o, 9)) for src, h, o in seen}
    return len(norm) > 1


def _refuse_mixed_cosmology(set_list):
    """Refuse a product in which an override reached only SOME campaigns.

    Campaigns legitimately differ in cosmology, and that is not the fault: an
    endo3-style campaign should use its own detected generation cosmology while
    an O4-era ``events`` campaign uses the ddL/dz it ships, and both are right.
    Recording them per campaign is enough for that case.

    The corrupting case is narrower and is what this refuses: a caller supplies
    one ``H0``/``Om0`` believing it governs the product, and it silently governs
    only part of it.  Every cosmology-dependent quantity of an ``events``
    campaign -- z, dL, the detector masses and critically ddL/dz -- is read
    verbatim, so the kwarg never reaches its ``pdraw``; an ``injections``
    campaign has no stored derivative, so the same kwarg *does* change its
    ``pdraw``.  The result is one product built under two cosmologies while
    reporting a single scalar, and nothing downstream can see it: ``pdraw``
    arrives at the consumer as a bare per-injection array with no campaign
    labelling that anything reads.

    Lives here (not in the v2 builder) because BOTH selection export
    generations combine campaigns: ``CombinedSelectionSet.to_darksirens`` /
    ``to_combined_selection_file`` hit the identical corruption on the v1/CLI
    path.
    """
    if len(set_list) < 2:
        return
    sources = {getattr(s, "_cosmology_source", None) for s in set_list}
    if not ("override" in sources and "file" in sources):
        return
    rows = "; ".join(
        f"{s.path}: source={getattr(s, '_cosmology_source', None)!r}"
        for s in set_list)
    raise ValueError(
        "an explicit cosmology was supplied but reached only SOME campaigns in "
        "this export, so the product would be built under two cosmologies "
        "while reporting one -- and pdraw reaches the consumer unlabelled by "
        f"campaign, so that would be undetectable downstream. Per campaign: "
        f"{rows}. A campaign with cosmology_source='file' ships its own "
        "ddL/dz and cannot honour an override. Drop the override so every "
        "campaign uses its own generation cosmology, which is what makes each "
        "campaign's Jacobian correct.")


def _v1_chieff_swap(set_list, keeps, amax, *, strict=True):
    """Ceilings, uniform-isotropic gate and support mask for the v1 chi_eff swap.

    GW-37.  The two v1 ``to_darksirens`` exporters carried their own copy of the
    swap, and that copy is what kept the three defects GW-03/GW-31 removed from
    the v2 builder only: the swap applied to a campaign never checked for
    uniform magnitudes and isotropic tilts, the support tested as
    ``isfinite(logprob)`` (which the 1-D grid clamp makes True ~1e-12 past the
    ceiling), and one caller-supplied ``amax`` forced on every campaign when the
    density being REPLACED is that campaign's own.  Rather than patch a second
    copy, both exporters now call the v2 helpers outright -- one implementation,
    so the two generations cannot drift again.

    Returns ``(ln_factor, in_support, amax_pairs, amax_sources, mode)`` with the
    factor concatenated in campaign order over each campaign's ``keep`` mask.
    """
    # Function-local: gwcat.export.selection_builder imports this module.
    from .export.selection_builder import (
        _campaign_chieff_amax, _campaign_chieff_lnfactor,
        _check_chieff_swap_valid)
    from .spin import parse_amax_option

    forced_amax = parse_amax_option(amax, what="amax")
    _check_chieff_swap_valid(set_list, strict=strict, violations=[])

    ln_parts, sup_parts, pairs, sources = [], [], [], []
    for s, keep in zip(set_list, keeps):
        src, a1, a2 = _campaign_chieff_amax(s, forced_amax, ASSUMED_REMOVAL_AMAX)
        lnp, sup = _campaign_chieff_lnfactor(s, keep, a1, a2)
        ln_parts.append(lnp)
        sup_parts.append(sup)
        pairs.append((a1, a2))
        sources.append(src)
    mode = "fixed" if forced_amax is not None else "per_campaign"
    return (np.concatenate(ln_parts), np.concatenate(sup_parts),
            pairs, sources, mode)


def _v1_refuse_unsupported(ln_factor, in_support, amax_pairs, what):
    """Refuse detected injections the assumed chi_eff prior excludes (GW-03).

    The predicate is the prior's OWN ``support()``, carried in ``in_support``,
    not ``isfinite(ln_factor)``: past the ceiling the grid returns a finite
    ~1e-12 density, so the finiteness test admitted excluded injections with a
    ``pdraw`` ~1e12 too small -- an inverse weight ~1e12 too LARGE in the
    Monte-Carlo sum for mu.
    """
    unsupported = ~(np.asarray(in_support, dtype=bool)
                    & np.isfinite(ln_factor))
    n_unsupported = int(np.sum(unsupported))
    if not n_unsupported:
        return
    ceilings = ", ".join(f"({a1:g},{a2:g})" for a1, a2 in amax_pairs)
    raise ValueError(
        f"{what}: {n_unsupported} of {ln_factor.size} detected injections "
        f"fall outside the assumed chi_eff prior's support (per-campaign "
        f"amax {ceilings}), so their pdraw would be exactly zero. They were "
        f"drawn and detected, so dropping them biases the selection integral "
        f"mu low. Fix the amax or export in the component basis, which is "
        f"exact for any campaign.")


def _write_chieff_amax_attrs(f, amax_pairs, amax_sources, mode):
    """Stamp the ceilings the swap ACTUALLY used, per campaign (GW-31/GW-37).

    ``chi_eff_amax`` stays scalar for backward compatibility when every
    campaign resolved to the same ceiling, and is NaN otherwise so a reader
    cannot mistake one campaign's ceiling for the file's.
    """
    a1s = [a1 for a1, _ in amax_pairs]
    a2s = [a2 for _, a2 in amax_pairs]
    uniform = len(set(a1s)) == 1 and len(set(a2s)) == 1
    f.attrs["chi_eff_amax"] = float(a1s[0]) if uniform else float("nan")
    f.attrs["chi_eff_amax_1_per_campaign"] = np.asarray(a1s, dtype=float)
    f.attrs["chi_eff_amax_2_per_campaign"] = np.asarray(a2s, dtype=float)
    f.attrs["chi_eff_amax_source_per_campaign"] = [str(s) for s in amax_sources]
    f.attrs["chi_eff_amax_mode"] = str(mode)


def _ddL_dz(z, dL_mpc, H0, Om0):
    """d(dL)/dz evaluated at z.  dL in Mpc."""
    c_kms = 299792.458
    dH = c_kms / H0
    E = np.sqrt(Om0 * (1 + z) ** 3 + (1.0 - Om0))
    DC = dL_mpc / (1 + z)
    return DC + (1 + z) * dH / E


#: The spin-magnitude ceiling the cartesian/polar spin-REMOVAL step assumes when
#: subtracting the analytic injected spin prior.  It is NOT necessarily the amax a
#: campaign was injected with -- endo3 injects 0.998 -- and the difference leaves a
#: constant (assumed/injected)^2 factor in the un-normalised pdraw.  See
#: SelectionSet._check_removal_amax (GW-04).
ASSUMED_REMOVAL_AMAX = 0.99


class SelectionSet:
    """Uniform interface over LVK injection files.

    Reads one HDF file in the modern 'events/' format, processes it, and
    provides the arrays needed by darksirens.  Call ``to_darksirens()`` to
    write the output file.

    Parameters
    ----------
    path : str
        Path to an LVK injection HDF file.
    H0, Om0 : float, optional
        Reference cosmology for dL↔z conversion.  Defaults to Planck15.
    strict_spin_checks : {"warn", "raise", "off"}, optional
        Policy for the PR4 read-time spin sanity checks (uniform-azimuth,
        isotropy, uniform-magnitude ``amax`` detection, factored-vs-joint
        consistency).  ``"warn"`` (default) records the outcome in
        ``spin_meta["checks"]`` and emits a :class:`UserWarning` on failure,
        ``"raise"`` raises :class:`ValueError`, ``"off"`` records silently.
        ``True``/``False`` are accepted as aliases for ``"raise"``/``"off"``.
    """

    def __init__(self, path: str, H0: float = None, Om0: float = None,
                 nsbh_mass_threshold: float = None,
                 strict_spin_checks: str = "warn"):
        self.path = path
        self.H0 = H0 or PLANCK15.H0.value
        self.Om0 = Om0 or PLANCK15.Om0
        # Whether the caller supplied a non-default reference cosmology.
        self._cosmology_override = (H0 is not None) or (Om0 is not None)
        # Which cosmology this campaign's pdraw actually used, set on load by
        # _record_cosmology.  Unset until then rather than guessed: for an
        # events-format file the answer is "none at all" (GW-09).
        self._cosmology_source = None
        self._cosmology_used_H0 = None
        self._cosmology_used_Om0 = None
        self._cosmology_detect_residual = None
        # PR4 read-time spin checks: "warn" (default) records + emits a warning
        # on failure, "raise" raises, "off" records silently.  Normalised here
        # so an invalid value fails loudly at construction.
        self._strict_spin_checks = _sspin.normalize_strict_mode(strict_spin_checks)
        # Additive component-basis spin state (populated on load; None until
        # then / when unobtainable).  See gwcat.selection_spin.
        self._a1 = self._a2 = self._cost1 = self._cost2 = None
        self._chi_p = None
        self._ln_spin_component = None
        self._weights = None
        # False only for 'events' files without drawn sky positions (the
        # semianalytic O1/O2 rows of the cumulative mixtures) — see _read_events.
        self._sky_position_available = True
        self._spin_meta = {
            "spin_format": None,
            "amax_detected": None,
            "uniform_isotropic": False,
            "checks": {},
        }
        # Source-frame NS/BH mass threshold for source-class filtering of
        # injections.  Defaults to the SAME shared constant used by PE-event
        # classification (gwcat.ingest) so injections and events cannot drift.
        self._nsbh_mass_threshold = (
            DEFAULT_NSBH_MASS_THRESHOLD if nsbh_mass_threshold is None
            else float(nsbh_mass_threshold))
        # Names of the FAR/significance columns actually used for thresholding;
        # populated by _read_events / _read_injections (explicit provenance).
        self._far_columns = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _record_cosmology(self, source, H0, Om0, resid=None):
        """Record which cosmology this campaign's ``pdraw`` actually used.

        ``source`` is one of:

        ``"file"``
            The campaign ships ``dluminosity_distance_dredshift``, so the
            Jacobian is read verbatim and **no** cosmology enters ``pdraw`` --
            neither the default nor an override.  Every O4-era ``events`` file
            is in this class.
        ``"detected"``
            The campaign's own generation cosmology, recovered from its
            ``(z, dL)`` pairs (see :func:`detect_generation_cosmology`) and used
            for the Jacobian.  This is the correct choice for a file that does
            not ship the derivative.
        ``"override"``
            A caller-supplied ``H0``/``Om0`` was used.
        ``"default"``
            Neither detected nor supplied; the Planck15 fallback was used.
        """
        self._cosmology_source = source
        self._cosmology_used_H0 = None if H0 is None else float(H0)
        self._cosmology_used_Om0 = None if Om0 is None else float(Om0)
        self._cosmology_detect_residual = (None if resid is None
                                           else float(resid))

    def _resolve_jacobian_cosmology(self, z, dL):
        """Pick the cosmology for ``ddL/dz`` on a file that does not ship it.

        An explicit override wins -- an analyst may be deliberately probing a
        different cosmology -- but the campaign's *own* generation cosmology is
        the default, because using anything else silently mis-states the
        Jacobian that converts the generated ``(m1src, m2src, z)`` density into
        the ``(m1det, q, dL)`` one.

        The production build passed ``(67.74, 0.3089)``.  The endo3 BBH campaign
        was generated at ``(67.90, 0.3065)``, so its ``pdraw`` carried a
        z-dependent ~0.28% distortion -- small, but it does NOT cancel the way a
        constant rescaling would (see GW-04), because the error varies across
        the redshift range and therefore reweights the injections against one
        another inside `mu`.
        """
        if self._cosmology_override:
            self._record_cosmology("override", self.H0, self.Om0)
            return self.H0, self.Om0

        det = detect_generation_cosmology(z, dL)
        if det is None:
            warnings.warn(
                f"{self.path}: could not identify the campaign's generation "
                f"cosmology from its (z, dL) pairs, and the file does not ship "
                f"dluminosity_distance_dredshift; falling back to Planck15 "
                f"({self.H0}, {self.Om0}) for the ddL/dz Jacobian. The draw "
                f"density is approximate to whatever the true generation "
                f"cosmology differs by.")
            self._record_cosmology("default", self.H0, self.Om0)
            return self.H0, self.Om0

        H0, Om0, resid = det
        self._record_cosmology("detected", H0, Om0, resid)
        return H0, Om0

    def _load(self):
        if self._loaded:
            return
        with h5py.File(self.path, "r") as f:
            if "events" in f:
                self._read_events(f)
            elif "injections" in f:
                self._read_injections(f)
            else:
                raise RuntimeError(
                    f"Unrecognised injection format in {self.path}: "
                    "expected 'events/' or 'injections/' group."
                )
        # The single choke point for the finished draw density (GW-28): every
        # exporter -- the v1 to_darksirens pair and the v2 builder alike --
        # starts from self._pdraw, so validating it here covers all of them
        # before anything is written or returned.  The exporters re-check the
        # array they actually write, because the chi_eff swap and the Essick
        # N_k/N_total rescaling touch it again afterwards.
        _require_positive_finite(
            self._pdraw, "pdraw", self.path,
            note="pdraw is the per-injection draw density; the selection "
                 "integral divides by it, so a zero row is an infinite weight "
                 "in mu and a non-finite row poisons the whole sum.")
        self._loaded = True

    def _check_removal_amax(self):
        """Warn when the amax assumed by the spin-REMOVAL step is not the amax
        the campaign was actually injected with (GW-04).

        The removal subtracts the analytic 6-D cartesian spin prior
        ``ln p = -ln(16 pi^2 a1^2 a2^2 amax^2)``, whose amax dependence is the
        constant ``-2 ln(amax)``.  Using the wrong amax therefore leaves a
        CONSTANT multiplicative error ``c = (amax_assumed / amax_injected)^2``
        in ``pdraw``.  endo3 injects 0.998 while this step assumed 0.99, so
        ``c ~ 0.9840``.

        Scope: which files, and which bases, this can reach at all
        ----------------------------------------------------------
        Two containments, both verified rather than argued (GW-19):

        1. **It cancels EXACTLY for the component basis.**  ``pdraw`` for that
           basis is ``pdraw_base * exp(ln_p_comp - ln_pdraw_no_spin)`` and
           ``pdraw_base ∝ exp(ln_pdraw_no_spin)``, so the whole
           ``ln_pdraw_no_spin`` -- the only place the removal amax appears --
           divides out.  Measured on the real O4ab campaign: perturbing the
           assumed ceiling 0.99 -> 0.998 moves ``component_pdraw`` by 0 (to
           1e-14) while it moves ``pdraw_base`` by the predicted 1.0162269.
        2. **No real file reaches it.**  ``ASSUMED_REMOVAL_AMAX`` is used only in
           the ``joint_cartesian`` and ``joint_polar`` branches, and both shipped
           campaigns are factored formats (``o4_factored``, ``endo3_factored``)
           that read their spin-free density directly with no assumed-prior
           subtraction.  So this is a latent defect on a path production data
           does not take -- not a bias in any shipped product.

        For a projection basis on a joint-format file it would be live, and the
        following describes what it would then do.

        What that does and does NOT affect
        ----------------------------------
        Because ``c`` is the same for every injection, it cancels out of
        everything except the absolute evidence.  With ``w = Lambda/pdraw``:

          * ``log_mu -> log_mu - ln c`` (mu is high by 1/c ~ 1.6%);
          * ``log_sigma2 -> log_sigma2 - 2 ln c``, so
            ``N_eff = exp(2 log_mu - log_sigma2)`` is **exactly unchanged**
            (verified numerically to 3e-15) -- the sparse-selection guard and its
            variance budget are not engaged by this at all;
          * the selection correction ``-N_obs log mu`` picks up the
            Theta-INDEPENDENT constant ``+N_obs ln c``.

        So **no posterior moves**: H0, the population hyper-parameters and every
        credible interval are unaffected, and the offset cancels identically in a
        Bayes factor between two models fit against the same file.  The only
        quantity that shifts is the absolute ``log Z``, by ``N_obs ln c``
        (~ -1.4 nats at 90 BBH, ~ -4.2 at 259).  The real hazard is therefore
        comparing log-evidences ACROSS files built with different amax
        conventions -- which is exactly what a re-export creates.

        This all rests on ``c`` being genuinely constant, which holds only while
        a single amax applies to both bodies.  A real NSBH campaign with
        ``amax_2 ~ 0.05`` would make it per-injection and none of the above would
        apply; the ``np.isclose`` mismatch check (GW-04) is what surfaces that.

        The removal happens while reading the file, before the injected-spin
        state (and hence the detected amax) is known, so it cannot simply use the
        right value here.  This makes the discrepancy visible and quantified
        instead of silent; correcting the constant is deliberately left to the
        block-based rewrite (GW-19), which resolves the amax before applying any
        spin factor.
        """
        removal = getattr(self, "_removal_amax", None)
        if removal is None:
            return None
        # NB: the private attribute, not the `spin_meta` property -- that one is
        # lazy and calls _load(), which is what invoked us.
        detected = (getattr(self, "_spin_meta", None) or {}).get("amax_detected")
        if detected is None or not all(np.isfinite(a) for a in detected):
            return None
        worst = max(float(a) for a in detected)
        if np.isclose(worst, removal, rtol=1e-9, atol=1e-12):
            return None
        factor = (removal / worst) ** 2
        warnings.warn(
            f"{self.path}: the spin-removal step subtracted a cartesian spin "
            f"prior assuming amax={removal}, but this campaign's injected amax "
            f"is {worst}. The 6-D prior's amax dependence is the constant "
            f"-2*ln(amax), so pdraw carries a spurious CONSTANT factor "
            f"c={factor:.6f} and mu is high by 1/c ({100 * (1 / factor - 1):.2f}%). "
            f"Because c is the same for every injection it cancels everywhere "
            f"except the absolute evidence: N_eff is exactly unchanged, and NO "
            f"posterior moves (H0, population, intervals) -- only log Z shifts, "
            f"by N_obs*ln(c) = {np.log(factor):+.4f} nats per observed event. Do "
            f"not compare log-evidences across files built with different amax "
            f"conventions. Tracked for GW-19.")
        return factor

    def _read_events(self, f):
        """Read the O4 ``events`` format.

        The O4 Zenodo files store ``events`` as a single compound HDF5
        dataset, while some downstream/older files expose the same columns as
        datasets in an ``events/`` group.  Support both layouts here.

        The compound layout is read through :class:`_PlannedFields` (GW-32), so
        the column reads below cost ONE pass over the dataset between them
        instead of one pass each.  Nothing in the body changes because of it.
        """
        ev = _planned_events_table(f["events"], f)

        # Source-frame parameters.  The O4 release also stores detector-frame
        # masses and redshift directly; use them when present so that we do not
        # introduce small differences by re-inverting dL with our cosmology.
        m1src = _h5_read_field(ev, "mass1_source")
        m2src = _h5_read_field(ev, "mass2_source")
        dL = _h5_read_field(ev, "luminosity_distance")
        # Sky position is absent for semianalytic O1/O2 rows in the cumulative
        # mixture files ("mixture-semi_o1_o2-*"): those estimates do not track
        # the detector duty cycle, so no ra/dec was drawn (see the release
        # docs).  NaN-fill and record availability instead of failing to load.
        if _h5_has_field(ev, "right_ascension"):
            ra = _h5_read_field(ev, "right_ascension")
            dec = _h5_read_field(ev, "declination")
            self._sky_position_available = True
        else:
            ra = np.full(m1src.shape, np.nan)
            dec = np.full(m1src.shape, np.nan)
            self._sky_position_available = False

        z, _ = _h5_first_field(ev, ["z", "redshift"])
        if _h5_has_field(ev, "mass1_detector"):
            m1det = _h5_read_field(ev, "mass1_detector")
        else:
            m1det = m1src * (1 + z)
        if _h5_has_field(ev, "mass2_detector"):
            m2det = _h5_read_field(ev, "mass2_detector")
        else:
            m2det = m2src * (1 + z)

        # Spin components (cartesian) and chi_eff.  Prefer the release-provided
        # chi_eff if available, otherwise derive it from the z-components.
        # Cartesian columns are read verbatim when present (byte-identical to
        # the pre-PR4 loader); Format-C polar-flavour files that ship only polar
        # spin columns are supported by deriving the cartesian components from
        # (a, θ, φ) -- see _read_component_spins.
        s1x, s1y, s1z, s2x, s2y, s2z = self._read_component_spins(ev)
        if _h5_has_field(ev, "chi_eff"):
            chieff = _h5_read_field(ev, "chi_eff")
        else:
            chieff = (m1src * s1z + m2src * s2z) / (m1src + m2src)

        weights = _h5_read_field(ev, "weights")

        # Normalised per-spin (magnitude, cosθ), preferring the file's polar
        # columns when present (exact), else derived from the cartesian
        # components (a=|s⃗|, cosθ=s_z/a).  Used by the additive component-basis
        # spin state and by the polar-joint legacy path below.
        sa1, scost1, sa2, scost2 = self._read_spin_polar(ev, s1x, s1y, s1z,
                                                         s2x, s2y, s2z)

        # Draw probability in source-frame component masses and redshift, with
        # spins removed.  Older/current-development O4 files may contain a
        # single joint log-density over masses, redshift, and cartesian spins;
        # the public O4ab clipped release instead stores factored log-density
        # columns, including spin magnitudes/angles.  In the factored case we
        # simply omit all spin terms so darksirens can apply its chi_eff prior.
        # The names live at module scope so the field plan asks for exactly the
        # columns this branch chain reads (GW-32).
        joint_cart = _JOINT_CART_FIELD
        joint_polar = _JOINT_POLAR_FIELD
        joint_no_spin_names = list(_JOINT_NO_SPIN_FIELDS)
        # Track the spin format and the joint log density (when present) for the
        # additive component-basis spin state computed after this chain.
        spin_fmt = None
        ln_pdraw_joint = None
        if _h5_has_field(ev, joint_cart):
            ln_pdraw_joint = _h5_read_field(ev, joint_cart)
            spin_fmt = "joint_cartesian"

            # Analytical 6-D isotropic cartesian spin prior:
            #   p(s1x,s1y,s1z,s2x,s2y,s2z)
            #     = 1 / (16 pi^2 a1^2 a2^2 amax^2)
            # where ai = |si|.  Divide this out so darksirens can replace it
            # with the 1-D chi_eff marginal.
            a1 = np.sqrt(s1x ** 2 + s1y ** 2 + s1z ** 2)
            a2 = np.sqrt(s2x ** 2 + s2y ** 2 + s2z ** 2)
            amax = ASSUMED_REMOVAL_AMAX
            self._removal_amax = float(amax)
            a1 = np.maximum(a1, 1e-30)
            a2 = np.maximum(a2, 1e-30)
            ln_pdraw_spin6d = -np.log(
                16.0 * np.pi ** 2 * a1 ** 2 * a2 ** 2 * amax ** 2)
            ln_pdraw_no_spin = ln_pdraw_joint - ln_pdraw_spin6d
        elif _h5_has_field(ev, joint_polar):
            # Format-C *polar* flavour: one joint log density in polar spin
            # coordinates.  Subtract the polar-coordinate form of the SAME
            # assumed prior the cartesian branch uses, chosen so the polar file
            # and its equivalent cartesian twin yield identical legacy _pdraw
            # (derivation in gwcat.selection_spin.ln_pdraw_no_spin_from_polar_joint).
            ln_pdraw_joint = _h5_read_field(ev, joint_polar)
            spin_fmt = "joint_polar"
            self._removal_amax = float(ASSUMED_REMOVAL_AMAX)
            ln_pdraw_no_spin = _sspin.ln_pdraw_no_spin_from_polar_joint(
                ln_pdraw_joint, scost1, scost2,
                amax=ASSUMED_REMOVAL_AMAX)
        elif any(_h5_has_field(ev, name) for name in joint_no_spin_names):
            ln_pdraw_no_spin, _ = _h5_first_field(ev, joint_no_spin_names)
            spin_fmt = "joint_no_spin"
        elif (_h5_has_field(ev, "lnpdraw_mass1_source")
              and _h5_has_field(ev, "lnpdraw_mass2_source_GIVEN_mass1_source")
              and (_h5_has_field(ev, "lnpdraw_z")
                   or _h5_has_field(ev, "lnpdraw_redshift"))):
            ln_pdraw_z, _ = _h5_first_field(
                ev, ["lnpdraw_z", "lnpdraw_redshift"])
            ln_pdraw_no_spin = (
                _h5_read_field(ev, "lnpdraw_mass1_source")
                + _h5_read_field(ev, "lnpdraw_mass2_source_GIVEN_mass1_source")
                + ln_pdraw_z
            )
            spin_fmt = "o4_factored"
        else:
            lnp_fields = sorted(
                name for name in _h5_field_names(ev) if name.startswith("lnpdraw"))
            raise RuntimeError(
                "Could not construct the spin-free O4 draw PDF. Expected either "
                f"{joint_cart!r}, {joint_polar!r}, one of "
                f"{joint_no_spin_names!r}, or the factored public O4 fields "
                "'lnpdraw_mass1_source', "
                "'lnpdraw_mass2_source_GIVEN_mass1_source', and "
                "'lnpdraw_z'/'lnpdraw_redshift'. Available lnpdraw fields: "
                f"{lnp_fields}")

        # Coordinate Jacobian: (m1src, m2src, z) → (m1det, q, dL)
        #   |J| = m1det / (1+z)^2 / (ddL/dz)
        if _h5_has_field(ev, "dluminosity_distance_dredshift"):
            ddL = _h5_read_field(ev, "dluminosity_distance_dredshift")
            # The file ships the derivative, so NO cosmology enters pdraw here:
            # not the default, not an override.  This campaign's pdraw is exact
            # either way, so an override is inert rather than harmful, and it
            # only warns -- what used to make it dishonest was the export
            # stamping it as though it had been applied, and
            # cosmology_source_per_campaign now records the truth instead.
            #
            # The case that IS corrupting is a COMBINED export, where the same
            # override changes an endo3-style campaign's pdraw and not this one,
            # so one product carries two cosmologies while reporting a scalar.
            # That is refused wherever campaigns are combined: the v2 selection
            # builder AND the v1 combined exporter (both call
            # _refuse_mixed_cosmology).
            if self._cosmology_override:
                warnings.warn(
                    f"{self.path}: an explicit cosmology (H0={self.H0}, "
                    f"Om0={self.Om0}) was supplied but is INERT for this "
                    f"campaign -- it ships "
                    f"'dluminosity_distance_dredshift', so pdraw uses the "
                    f"stored derivative and no cosmology at all. The export "
                    f"records cosmology_source='file' for it. Combining it "
                    f"with a campaign that DOES honour the override is "
                    f"refused.")
            self._record_cosmology("file", None, None)
        else:
            H0_j, Om0_j = self._resolve_jacobian_cosmology(z, dL)
            ddL = _ddL_dz(z, dL, H0_j, Om0_j)
        pdraw = np.exp(ln_pdraw_no_spin) * m1det / (1 + z) ** 2 / ddL

        # Time normalisation and mixture/month weights.  The O4 examples keep
        # weights in the numerator of importance-sampling sums; equivalently,
        # divide the stored draw density by weights.  Both divisors are checked
        # first (GW-28): dividing by an unvalidated weight or observing time is
        # how an inf/NaN gets into pdraw without anything saying so.
        T_yr = _require_positive_finite(
            f.attrs["total_analysis_time"], "total_analysis_time",
            self.path) / (3600 * 24 * 365.25)
        pdraw /= T_yr
        _require_positive_finite(weights, "weights", self.path)
        pdraw /= weights

        ndraw = int(_require_positive_finite(
            f.attrs["total_generated"], "total_generated", self.path,
            note="It is the campaign's ndraw, the denominator of the "
                 "selection integral and of detection_efficiency()."))

        # FAR: discover search pipelines from the file, and handle both O4
        # names (e.g. "pycbc_far", "cwb-bbh_far") and older names.  The
        # discovery is _events_far_columns, shared with the field plan so the
        # one-pass read asks for exactly these columns (GW-32).
        fars_per_search = []
        far_columns = []
        for col in _events_far_columns(ev, f):
            try:
                fars_per_search.append(_h5_read_field(ev, col))
                far_columns.append(col)
            except Exception:
                pass
        self._fars = np.column_stack(fars_per_search) if fars_per_search else None
        self._far_columns = far_columns

        # Store
        self._m1det = m1det
        self._m2det = m2det
        self._dL = dL
        self._chieff = chieff
        self._ra = ra
        self._dec = dec
        self._m1src = m1src
        self._m2src = m2src
        self._z = z
        self._pdraw = pdraw
        self._ndraw = ndraw
        self._T_yr = T_yr
        self._weights = weights

        # ── Additive component-basis spin state (PR4) ──────────────────────
        self._compute_events_spin_state(
            ev, spin_fmt, ln_pdraw_no_spin, ln_pdraw_joint,
            sa1, scost1, sa2, scost2, m1src, m2src)
        # Now that the injected amax is known, check it against the one the
        # removal step above had to assume (GW-04).
        self._removal_amax_factor = self._check_removal_amax()

    # ------------------------------------------------------------------
    # PR4: component-basis spin helpers (events format)
    # ------------------------------------------------------------------
    def _read_component_spins(self, ev):
        """Return ``(s1x,s1y,s1z,s2x,s2y,s2z)`` cartesian spin components.

        Cartesian columns are read verbatim when present (byte-identical to the
        pre-PR4 loader).  Format-C polar-flavour files that ship only polar spin
        columns are supported by reconstructing the cartesian components from
        ``(a, θ, φ)``; the azimuth defaults to zero if absent (only the
        z-components, which are azimuth-independent, feed chi_eff).
        """
        cart = ["spin1x", "spin1y", "spin1z", "spin2x", "spin2y", "spin2z"]
        if all(_h5_has_field(ev, k) for k in cart):
            return tuple(_h5_read_field(ev, k) for k in cart)
        polar = ["spin1_magnitude", "spin1_polar_angle",
                 "spin2_magnitude", "spin2_polar_angle"]
        if all(_h5_has_field(ev, k) for k in polar):
            a1 = _h5_read_field(ev, "spin1_magnitude")
            th1 = _h5_read_field(ev, "spin1_polar_angle")
            a2 = _h5_read_field(ev, "spin2_magnitude")
            th2 = _h5_read_field(ev, "spin2_polar_angle")
            ph1 = (_h5_read_field(ev, "spin1_azimuthal_angle")
                   if _h5_has_field(ev, "spin1_azimuthal_angle")
                   else np.zeros_like(a1))
            ph2 = (_h5_read_field(ev, "spin2_azimuthal_angle")
                   if _h5_has_field(ev, "spin2_azimuthal_angle")
                   else np.zeros_like(a2))
            s1z = a1 * np.cos(th1)
            s2z = a2 * np.cos(th2)
            s1x = a1 * np.sin(th1) * np.cos(ph1)
            s1y = a1 * np.sin(th1) * np.sin(ph1)
            s2x = a2 * np.sin(th2) * np.cos(ph2)
            s2y = a2 * np.sin(th2) * np.sin(ph2)
            return s1x, s1y, s1z, s2x, s2y, s2z
        # Neither representation available: re-raise the original clear error.
        return tuple(_h5_read_field(ev, k) for k in cart)

    def _read_spin_polar(self, ev, s1x, s1y, s1z, s2x, s2y, s2z):
        """Return ``(a1, cosθ1, a2, cosθ2)`` preferring the file's polar columns.

        When ``spin{i}_magnitude`` / ``spin{i}_polar_angle`` are present they are
        used exactly (``a=magnitude``, ``cosθ=cos(polar_angle)``); otherwise the
        values are derived from the cartesian components.
        """
        if (_h5_has_field(ev, "spin1_magnitude")
                and _h5_has_field(ev, "spin1_polar_angle")
                and _h5_has_field(ev, "spin2_magnitude")
                and _h5_has_field(ev, "spin2_polar_angle")):
            sa1 = _h5_read_field(ev, "spin1_magnitude")
            scost1 = np.cos(_h5_read_field(ev, "spin1_polar_angle"))
            sa2 = _h5_read_field(ev, "spin2_magnitude")
            scost2 = np.cos(_h5_read_field(ev, "spin2_polar_angle"))
            return sa1, scost1, sa2, scost2
        sa1, scost1 = _sspin.polar_from_cartesian(s1x, s1y, s1z)
        sa2, scost2 = _sspin.polar_from_cartesian(s2x, s2y, s2z)
        return sa1, scost1, sa2, scost2

    def _compute_events_spin_state(self, ev, spin_fmt, ln_pdraw_no_spin,
                                   ln_pdraw_joint, sa1, scost1, sa2, scost2,
                                   m1src, m2src):
        """Populate the additive component-basis spin state for events files.

        Sets ``_a1/_a2/_cost1/_cost2``, ``_chi_p`` and ``_ln_spin_component``
        (the exact log factor with ``pdraw_component = _pdraw *
        exp(_ln_spin_component)``), plus ``_spin_meta``.  Never alters ``_pdraw``
        or any legacy state.
        """
        mode = self._strict_spin_checks
        self._a1, self._cost1 = np.asarray(sa1, float), np.asarray(scost1, float)
        self._a2, self._cost2 = np.asarray(sa2, float), np.asarray(scost2, float)

        # chi_p: use the file field when present (Format B), else the formula.
        if _h5_has_field(ev, "chi_p"):
            self._chi_p = _h5_read_field(ev, "chi_p")
        else:
            self._chi_p = chi_p_from_components(
                self._a1, self._a2, self._cost1, self._cost2, m1src, m2src)

        checks = {}
        amax_detected = None
        uniform_isotropic = False
        ln_spin = None

        if spin_fmt == "o4_factored":
            have_mag = (_h5_has_field(ev, "lnpdraw_spin1_magnitude")
                        and _h5_has_field(ev, "lnpdraw_spin2_magnitude"))
            have_polar = (_h5_has_field(ev, "lnpdraw_spin1_polar_angle")
                          and _h5_has_field(ev, "lnpdraw_spin2_polar_angle"))
            if have_mag and have_polar:
                lnp_mag1 = _h5_read_field(ev, "lnpdraw_spin1_magnitude")
                lnp_mag2 = _h5_read_field(ev, "lnpdraw_spin2_magnitude")
                lnp_pol1 = _h5_read_field(ev, "lnpdraw_spin1_polar_angle")
                lnp_pol2 = _h5_read_field(ev, "lnpdraw_spin2_polar_angle")
                ln_p_comp = _sspin.ln_p_component_factored(
                    ln_pdraw_no_spin, lnp_mag1, lnp_pol1, lnp_mag2, lnp_pol2,
                    scost1, scost2)
                ln_spin = ln_p_comp - ln_pdraw_no_spin

                # amax auto-detection + isotropy.
                amax1, uni1 = _sspin.detect_uniform_amax_from_lnmag(lnp_mag1)
                amax2, uni2 = _sspin.detect_uniform_amax_from_lnmag(lnp_mag2)
                iso1, dev_i1 = _sspin.check_isotropy_polar(lnp_pol1, scost1)
                iso2, dev_i2 = _sspin.check_isotropy_polar(lnp_pol2, scost2)
                amax_detected = (amax1, amax2)
                uniform_isotropic = bool(uni1 and uni2 and iso1 and iso2)
                checks["magnitude_uniform"] = (bool(uni1), bool(uni2))
                checks["isotropy_dev"] = (dev_i1, dev_i2)
                checks["isotropy_uniform"] = (bool(iso1), bool(iso2))
                # Route the MAGNITUDE and ISOTROPY checks through report_check
                # too (GW-05).  They were computed, stored in `checks`, and then
                # never acted on -- only the azimuth checks below reached
                # report_check -- so strict_spin_checks="raise" passed silently
                # on the real O4ab file even though both of these fail there
                # (isotropy_dev = 0.6419, magnitude_uniform = [False, False]).
                # Those are exactly the assumptions the chieff/chieff_chip
                # projections depend on, whereas azimuth uniformity marginalises
                # out; the two checks that mattered were the two being ignored.
                _sspin.report_check(
                    "magnitude_uniform_spin1", uni1,
                    f"amax_detected={amax1!r}", mode, self.path)
                _sspin.report_check(
                    "magnitude_uniform_spin2", uni2,
                    f"amax_detected={amax2!r}", mode, self.path)
                _sspin.report_check(
                    "isotropy_polar_spin1", iso1,
                    f"max|lnp_polar-ln(sinθ/2)|={dev_i1:.3e}", mode, self.path)
                _sspin.report_check(
                    "isotropy_polar_spin2", iso2,
                    f"max|lnp_polar-ln(sinθ/2)|={dev_i2:.3e}", mode, self.path)
                # Azimuth check (records + acts per strict mode).
                if (_h5_has_field(ev, "lnpdraw_spin1_azimuthal_angle")
                        and _h5_has_field(ev, "lnpdraw_spin2_azimuthal_angle")):
                    az1, dev_a1 = _sspin.check_uniform_azimuth(
                        _h5_read_field(ev, "lnpdraw_spin1_azimuthal_angle"))
                    az2, dev_a2 = _sspin.check_uniform_azimuth(
                        _h5_read_field(ev, "lnpdraw_spin2_azimuthal_angle"))
                    checks["azimuth_uniform"] = (bool(az1), bool(az2))
                    checks["azimuth_dev"] = (dev_a1, dev_a2)
                    _sspin.report_check("azimuth_uniform_spin1", az1,
                                        f"max|lnp_azim+ln2π|={dev_a1:.3e}",
                                        mode, self.path)
                    _sspin.report_check("azimuth_uniform_spin2", az2,
                                        f"max|lnp_azim+ln2π|={dev_a2:.3e}",
                                        mode, self.path)
            # else: factored file without spin lnpdraw columns -> component
            # spin density unobtainable (ln_spin stays None).
        elif spin_fmt == "joint_cartesian":
            ln_p_comp = _sspin.ln_p_component_joint_cartesian(
                ln_pdraw_joint, sa1, sa2)
            ln_spin = ln_p_comp - ln_pdraw_no_spin
        elif spin_fmt == "joint_polar":
            ln_p_comp = _sspin.ln_p_component_joint_polar(
                ln_pdraw_joint, scost1, scost2)
            ln_spin = ln_p_comp - ln_pdraw_no_spin
        # spin_fmt == "joint_no_spin": no spin draw info -> ln_spin None.

        self._ln_spin_component = ln_spin
        self._spin_meta = {
            "spin_format": spin_fmt,
            "amax_detected": amax_detected,
            "uniform_isotropic": uniform_isotropic,
            "checks": checks,
        }

    def _read_injections(self, f):
        """Read the O3 'injections/' format (e.g. endo3_bbhpop files).

        The O3 format stores the draw PDF in factored components, so we
        multiply mass × redshift PDFs directly instead of dividing out an
        analytical spin prior.  Detector-frame masses and redshift are also
        stored, avoiding a cosmology inversion.
        """
        inj = f["injections"]

        # Source-frame and detector-frame parameters (both stored directly)
        m1src = np.asarray(inj["mass1_source"], float)
        m2src = np.asarray(inj["mass2_source"], float)
        m1det = np.asarray(inj["mass1"], float)
        m2det = np.asarray(inj["mass2"], float)
        dL = np.asarray(inj["distance"], float)
        z = np.asarray(inj["redshift"], float)
        ra = np.asarray(inj["right_ascension"], float)
        dec = np.asarray(inj["declination"], float)

        # chi_eff from z-components
        s1z = np.asarray(inj["spin1z"], float)
        s2z = np.asarray(inj["spin2z"], float)
        chieff = (m1src * s1z + m2src * s2z) / (m1src + m2src)

        # Spin-free draw PDF from factored components.  Both factors are
        # required to be finite and strictly positive (GW-28); this used to
        # floor the product at 1e-300, which turned a malformed (zero or
        # negative) density into a fabricated ~1e-300 one that then sailed
        # through every downstream check as a merely improbable injection.
        p_mass = _require_positive_finite(
            np.asarray(inj["mass1_source_mass2_source_sampling_pdf"], float),
            "mass1_source_mass2_source_sampling_pdf", self.path)
        p_z = _require_positive_finite(
            np.asarray(inj["redshift_sampling_pdf"], float),
            "redshift_sampling_pdf", self.path)
        ln_pdraw_no_spin = np.log(p_mass * p_z)

        # Jacobian: (m1src, m2src, z) → (m1det, q, dL)
        H0_j, Om0_j = self._resolve_jacobian_cosmology(z, dL)
        ddL = _ddL_dz(z, dL, H0_j, Om0_j)
        pdraw = np.exp(ln_pdraw_no_spin) * m1det / (1 + z) ** 2 / ddL

        # Time normalisation
        T_s = f.attrs.get("analysis_time_s",
                          inj.attrs.get("analysis_time_s"))
        if T_s is None:
            raise RuntimeError(
                f"No analysis_time_s attribute found in {self.path}")
        T_yr = _require_positive_finite(
            T_s, "analysis_time_s", self.path) / (3600 * 24 * 365.25)
        pdraw /= T_yr

        # Injection weights (mixture_weight = 1.0 for single-subpop files)
        if "mixture_weight" in inj:
            weights = _require_positive_finite(
                np.asarray(inj["mixture_weight"], float),
                "mixture_weight", self.path)
            pdraw /= weights
        else:
            weights = np.ones_like(pdraw)

        # ndraw is fatal if absent, exactly as in _read_events.  It used to
        # default to 0, which zeroes this campaign's Essick fraction and sends
        # every one of its injections to Lambda/0 = inf -- a silently ruined mu
        # from a missing attribute.
        if "total_generated" in f.attrs:
            ndraw = int(_require_positive_finite(
                f.attrs["total_generated"], "total_generated", self.path,
                note="It is the campaign's ndraw, the denominator of the "
                     "selection integral and of detection_efficiency()."))
        elif "total_generated" in inj.attrs:
            ndraw = int(_require_positive_finite(
                inj.attrs["total_generated"], "total_generated", self.path,
                note="It is the campaign's ndraw, the denominator of the "
                     "selection integral and of detection_efficiency()."))
        else:
            raise RuntimeError(
                f"{self.path}: no 'total_generated' attribute on the file or "
                f"on the 'injections' group. It is the campaign's ndraw, so "
                f"without it the Essick fraction is zero and every injection "
                f"in this campaign contributes an infinite weight to mu.")

        # FAR columns: O3 uses hardcoded names
        fars_per_search = []
        far_columns = []
        for col in ["far_gstlal", "far_pycbc_bbh", "far_pycbc_hyperbank",
                     "far_mbta", "far_cwb"]:
            if col in inj:
                fars_per_search.append(np.asarray(inj[col], float))
                far_columns.append(col)
        # Also scan for any other *far* columns we might have missed
        if not fars_per_search:
            for key in inj:
                if isinstance(key, str) and key.startswith("far_"):
                    try:
                        fars_per_search.append(np.asarray(inj[key], float))
                        far_columns.append(key)
                    except Exception:
                        pass
        self._fars = np.column_stack(fars_per_search) if fars_per_search else None
        self._far_columns = far_columns

        # Store (same attributes as _read_events)
        self._m1det = m1det
        self._m2det = m2det
        self._dL = dL
        self._chieff = chieff
        self._ra = ra
        self._dec = dec
        self._m1src = m1src
        self._m2src = m2src
        self._z = z
        self._pdraw = pdraw
        self._ndraw = ndraw
        self._T_yr = T_yr
        self._weights = weights

        # ── Additive component-basis spin state (PR4, Format A / endo3) ────
        self._compute_injections_spin_state(inj, ln_pdraw_no_spin, p_mass, p_z,
                                             m1src, m2src)

    def _compute_injections_spin_state(self, inj, ln_pdraw_no_spin,
                                       p_mass, p_z, m1src, m2src):
        """Populate the additive component-basis spin state for endo3 files.

        Format A stores LINEAR densities with cartesian, isotropic
        uniform-magnitude spins.  Per spin ``p(a,cosθ) = 2π·a²·p_cart``; the
        component density follows from the joint ``sampling_pdf`` (or the
        per-spin cartesian sampling pdfs).  ``_ln_spin_component`` is set to
        ``None`` when the spin marginal is unobtainable.
        """
        mode = self._strict_spin_checks

        cart = ["spin1x", "spin1y", "spin1z", "spin2x", "spin2y", "spin2z"]
        have_cart = all(k in inj for k in cart)
        if have_cart:
            s1x, s1y, s1z = (np.asarray(inj[k], float) for k in cart[:3])
            s2x, s2y, s2z = (np.asarray(inj[k], float) for k in cart[3:])
            self._a1, self._cost1 = _sspin.polar_from_cartesian(s1x, s1y, s1z)
            self._a2, self._cost2 = _sspin.polar_from_cartesian(s2x, s2y, s2z)
            self._chi_p = chi_p_from_components(
                self._a1, self._a2, self._cost1, self._cost2, m1src, m2src)

        checks = {}
        amax_detected = None
        uniform_isotropic = False
        ln_spin = None

        spd1_key = "spin1x_spin1y_spin1z_sampling_pdf"
        spd2_key = "spin2x_spin2y_spin2z_sampling_pdf"
        have_spin_pdf = spd1_key in inj and spd2_key in inj

        if have_cart and "sampling_pdf" in inj:
            # Component density straight from the joint sampling_pdf (exact even
            # for mixtures): ln p_comp = ln(joint) + 2 ln 2π + 2 ln a1 + 2 ln a2.
            # Validated, not floored, for the same reason as the mass/redshift
            # factors above (GW-28): a zero joint density is a malformed row.
            ln_joint = np.log(_require_positive_finite(
                np.asarray(inj["sampling_pdf"], float),
                "sampling_pdf", self.path))
            ln_p_comp = _sspin.ln_p_component_joint_cartesian(
                ln_joint, self._a1, self._a2)
            ln_spin = ln_p_comp - ln_pdraw_no_spin
        elif have_cart and have_spin_pdf:
            # Fall back to the factored per-spin cartesian marginals.
            spd1 = np.asarray(inj[spd1_key], float)
            spd2 = np.asarray(inj[spd2_key], float)
            ln_spin = (_sspin.ln_p_spin_cart_component(spd1, self._a1)
                       + _sspin.ln_p_spin_cart_component(spd2, self._a2))

        # amax / uniform-isotropic detection from the per-spin cartesian pdfs.
        if have_cart and have_spin_pdf:
            spd1 = np.asarray(inj[spd1_key], float)
            spd2 = np.asarray(inj[spd2_key], float)
            ms1, uni1 = _sspin.detect_max_spin_cart(spd1, self._a1)
            ms2, uni2 = _sspin.detect_max_spin_cart(spd2, self._a2)
            amax_detected = (ms1, ms2)
            uniform_isotropic = bool(uni1 and uni2)
            checks["max_spin_uniform"] = (bool(uni1), bool(uni2))
            # Route it through report_check (GW-05): the same omission as the
            # events path -- computed, stored, and never acted on, so
            # strict_spin_checks="raise" could not fire on it.
            _sspin.report_check("max_spin_uniform_spin1", uni1,
                                f"max_spin_detected={ms1!r}", mode, self.path)
            _sspin.report_check("max_spin_uniform_spin2", uni2,
                                f"max_spin_detected={ms2!r}", mode, self.path)
            # Consistency of the factored product vs the joint sampling_pdf.
            if "sampling_pdf" in inj:
                ok, dev = _sspin.check_factored_vs_joint(
                    np.asarray(inj["sampling_pdf"], float), p_mass, p_z,
                    spd1, spd2)
                checks["factored_vs_joint_dev"] = dev
                _sspin.report_check(
                    "factored_vs_joint", ok,
                    f"max rel dev={dev:.3e}", mode, self.path)

        self._ln_spin_component = ln_spin
        self._spin_meta = {
            "spin_format": "endo3_factored",
            "amax_detected": amax_detected,
            "uniform_isotropic": uniform_isotropic,
            "checks": checks,
        }

    # ------------------------------------------------------------------
    # Detection cut
    # ------------------------------------------------------------------
    def detected_mask(self, far_threshold: float = 1.0) -> np.ndarray:
        """Boolean mask: True for injections detected below FAR threshold (yr^-1)."""
        self._load()
        if self._fars is None:
            raise ValueError("No FAR columns found in injection file.")
        return np.any(self._fars < far_threshold, axis=1)

    # ------------------------------------------------------------------
    # Source-class filtering (PR 9)
    # ------------------------------------------------------------------
    def source_class_mask(self, source_class=None) -> np.ndarray:
        """Boolean mask selecting injections in the requested source class(es).

        Injections are classified by their *injected* source-frame component
        masses using the SAME shared mass-threshold classifier as PE-event
        ingest (:func:`gwcat.source_class.classify_by_mass`), so a ``bbh``
        selection of injections is consistent with a ``bbh`` selection of PE
        events.  ``source_class=None`` (the default) applies no restriction and
        returns an all-True mask -- byte-identical to the pre-PR9 behavior.

        Accepts the ``bbh``/``nsbh``/``bns``/``massgap``/``cbc`` keywords (``cbc``
        = all compact-binary classes), a canonical class name, or an iterable of
        those.
        """
        self._load()
        n = len(self._m1src)
        if source_class is None:
            return np.ones(n, dtype=bool)
        labels = classify_by_mass(self._m1src, self._m2src,
                                  self._nsbh_mass_threshold)
        canonical = np.array([normalize_source_class(x) for x in labels])
        allowed = resolve_filter_classes(source_class)
        return np.isin(canonical, list(allowed))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def n_retained(self) -> int:
        """Number of injection rows the FILE actually carries.

        This is NOT the campaign's draw count.  A clipped release keeps only
        the rows that survived a pre-selection -- the O4ab clipped file carries
        2,959,534 rows out of 870,454,872 generated draws -- so this number is
        a property of the distribution, not of the detectors.  Use it for
        bookkeeping (array shapes, memory); never as an efficiency denominator.
        """
        self._load()
        return len(self._m1det)

    @property
    def n_generated(self) -> int:
        """Total draws the campaign generated (the file's ``total_generated``).

        The authoritative ``ndraw``: the denominator of the selection integral
        mu, and of :meth:`detection_efficiency`.
        """
        self._load()
        return int(self._ndraw)

    @property
    def n_injections(self) -> int:
        """Alias of :attr:`n_retained`, kept for backward compatibility.

        Prefer the explicit :attr:`n_retained` / :attr:`n_generated` pair: this
        name reads like the campaign's draw count and is not (GW-28).
        """
        return self.n_retained

    def detection_efficiency(self, far_threshold: float = 1.0) -> float:
        """Fraction of the campaign's GENERATED draws detected at this FAR.

        The denominator is :attr:`n_generated` (``total_generated``), not the
        number of rows the file retained.  Dividing by the retained rows (the
        pre-GW-28 behavior) reports the detected fraction OF THE CLIP, which on
        the O4ab clipped campaign is 986,829/2,959,534 = 0.3334 against a true
        generated-draw efficiency of 986,829/870,454,872 = 0.0011337 -- high by
        a factor of 294.  The detected fraction of the retained rows is
        available as ``detected_mask(far).sum() / n_retained`` for anyone who
        wants it.
        """
        return self.detected_mask(far_threshold).sum() / self.n_generated

    # ── PR4 component-basis spin accessors (additive, read-only) ───────────
    @property
    def component_spin_available(self) -> bool:
        """Whether an exact component-basis spin draw density was recovered.

        ``True`` iff ``_ln_spin_component`` is populated, i.e. the file carried
        enough spin-draw information for the exact ``(a1,a2,cosθ1,cosθ2)``
        conversion (all formats except the spin-free joint key and a factored
        file lacking the per-spin spin lnpdraw columns).
        """
        self._load()
        return self._ln_spin_component is not None

    @property
    def spin_meta(self) -> dict:
        """Copy of the component-spin metadata dict (format, amax, checks)."""
        self._load()
        return dict(self._spin_meta)

    @property
    def a1(self):
        """Primary spin magnitude ``a1`` (None if not derivable)."""
        self._load()
        return self._a1

    @property
    def a2(self):
        """Secondary spin magnitude ``a2`` (None if not derivable)."""
        self._load()
        return self._a2

    @property
    def cost1(self):
        """Primary spin tilt cosine ``cosθ1`` (None if not derivable)."""
        self._load()
        return self._cost1

    @property
    def cost2(self):
        """Secondary spin tilt cosine ``cosθ2`` (None if not derivable)."""
        self._load()
        return self._cost2

    @property
    def chi_p(self):
        """Effective precessing spin ``χ_p`` (file field or Schmidt formula)."""
        self._load()
        return self._chi_p

    @property
    def ln_spin_component(self):
        """Log factor s.t. ``pdraw_component = _pdraw * exp(ln_spin_component)``.

        ``None`` when the component-basis spin draw density is unobtainable.
        """
        self._load()
        return self._ln_spin_component

    def component_pdraw(self):
        """Exact per-year component-basis draw density in (m1det,q,dL,a1,a2,
        cosθ1,cosθ2), weight-divided.  Requires ``component_spin_available``.
        """
        self._load()
        if self._ln_spin_component is None:
            raise ValueError(
                f"Component-basis spin draw density unavailable for {self.path} "
                f"(spin_format={self._spin_meta.get('spin_format')!r}).")
        return self._pdraw * np.exp(self._ln_spin_component)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def to_darksirens(self, out_path: str, far_threshold: float = 1.0,
                      amax=AMAX_AUTO, source_class=None,
                      write_summary: bool = False,
                      summary_context: Optional[dict] = None,
                      strict: bool = True):
        """Write a pre-processed selection file for darksirens.

        Applies the 1-D chi_eff spin-prior swap: the injection spin-draw
        distribution removed at load time is replaced by the 1-D isotropic
        chi_eff prior, so the exported ``pdraw`` already contains it
        (``chi_eff_swap_applied=True``, ``chi_eff_prior_applied_to_pdraw=True``,
        ``spin_prior_mode="include"``).  This matches the PE export's Mode A
        contract: darksirens reads the result directly and MUST NOT multiply the
        chi_eff prior again — no gwdistributions needed.

        Parameters
        ----------
        out_path : str
        far_threshold : float
            FAR detection threshold in yr⁻¹.
        amax : float or ``"auto"``, default ``"auto"``
            Ceiling of the uniform spin-magnitude prior whose chi_eff marginal
            the swap multiplies in.  ``"auto"`` (GW-37, matching
            :func:`gwcat.export.build_selection_product`) uses this campaign's
            OWN detected injected ceiling -- the density being replaced is that
            campaign's, so the ceiling must be too.  A number forces one
            ceiling, which is what the pre-GW-37 default of ``0.99`` did to
            every campaign including endo3 (injected at 0.998); the resulting
            error is chi_eff-dependent and does NOT divide out of mu.
        source_class : str, iterable, or None
            Optional source-class filter (``bbh``/``nsbh``/``bns``/``massgap``/
            ``cbc`` or a canonical class).  Injections are classified by their
            injected source-frame masses with the SAME shared thresholds as PE
            events (see :meth:`source_class_mask`).  ``None`` (default) applies
            no restriction and is byte-identical to the pre-PR9 export.  Note
            that filtering is *subsetting*, not reweighting: ``ndraw`` is left
            unchanged, and the analyst must pair the file with a PE export
            filtered to the same class(es).  Recorded in the output attrs.
        write_summary : bool, default False
            (PR 10) When True, write ``<out_path>.validation_summary.json`` and
            ``.md`` next to ``out_path`` (see :mod:`gwcat.validation_summary`).
            The unified ``gwcat selection`` CLI turns this on by default
            (``--no-summary`` to disable).
        summary_context : dict, optional
            Extra fields merged into the written summary.
        strict : bool, default True
            Raise when the campaign's injected spins are measurably not
            uniform-magnitude/isotropic, which makes the swap the wrong density
            by a chi_eff-dependent O(1) factor (GW-37).  ``False`` warns.
        """
        self._load()
        det = self.detected_mask(far_threshold)
        sc_mask = self.source_class_mask(source_class)
        n_before = int(det.size)
        n_after = int(sc_mask.sum())
        keep = det & sc_mask
        n_det = int(keep.sum())
        if n_det == 0:
            raise RuntimeError(
                f"No detected injections at FAR < {far_threshold}"
                + ("" if source_class is None
                   else f" in source class {source_class!r}"))

        # Apply the 1-D chi_eff prior swap.  Ceilings, the uniform-isotropic
        # gate and the support mask all come from the v2 helpers (GW-37) so the
        # two export generations cannot disagree about the same physics.
        chieff_det = self._chieff[keep]
        m1src_det = self._m1src[keep]
        m2src_det = self._m2src[keep]
        logp_chi, in_support, amax_pairs, amax_sources, amax_mode = (
            _v1_chieff_swap([self], [keep], amax, strict=strict))
        # No -50 floor (GW-03): a detected injection with zero assumed draw
        # density is a contradiction, not a small number -- see
        # gwcat/export/selection_builder.py.  The v2 twin refuses identically.
        _v1_refuse_unsupported(logp_chi, in_support, amax_pairs,
                               "to_selection_file")
        with np.errstate(over="ignore"):
            pdraw_det = self._pdraw[keep] * np.exp(logp_chi)
        # The array that is about to be written (GW-28).  _load already vetted
        # the base density; the chi_eff swap multiplied it again, so re-check
        # the product rather than assume the multiply was harmless.
        _require_positive_finite(
            pdraw_det, "pdraw (after the chi_eff swap)", self.path,
            note=f"amax={amax_pairs[0]}, far_threshold={far_threshold}.")

        with h5py.File(out_path, "w") as f:
            f.attrs["format_version"] = "gwcat-selection-1.0"

            # Which gwcat wrote this file (DS-10 provenance; also on the v2
            # writers).  Commit, not version: an editable install moves per
            # commit while the version string stands still.
            from .validation_summary import gwcat_commit, package_version as _pkg_version
            f.attrs["writer_commit"] = gwcat_commit()
            f.attrs["writer_version"] = _pkg_version()
            f.attrs["ndraw"] = self._ndraw
            f.attrs["T_obs_yr"] = float(self._T_yr)
            f.attrs["far_threshold"] = float(far_threshold)
            f.attrs["n_detected"] = n_det
            # The cosmology pdraw ACTUALLY used (GW-09): the detected
            # generation cosmology for an injections-format campaign, the
            # override if one was given, NaN for an events-format campaign
            # whose Jacobian was read verbatim from the file (no cosmology
            # entered pdraw at all).  Stamping self.H0 here regardless -- the
            # pre-GW-09 behavior -- mis-described the file's own Jacobian the
            # moment detection started changing pdraw.
            f.attrs["cosmology_H0"] = _cosmo_or_nan(self, "_cosmology_used_H0")
            f.attrs["cosmology_Om0"] = _cosmo_or_nan(self,
                                                     "_cosmology_used_Om0")
            f.attrs["cosmology_source"] = str(self._cosmology_source)
            # False when this campaign's rows carry no drawn sky position (the
            # semianalytic O1/O2 entries of the cumulative mixtures): its
            # exported ra/dec are NaN, and a consumer that feeds them to
            # hp.ang2pix must know that is declared, not corrupt (GW-10; the
            # v2 builder has recorded this since GW-20, the v1 writer never
            # did).
            f.attrs["sky_position_available"] = bool(
                getattr(self, "_sky_position_available", True))
            f.attrs["chi_eff_swap_applied"] = True
            _write_chieff_amax_attrs(f, amax_pairs, amax_sources, amax_mode)
            # ── Spin-prior contract provenance (PR 3) ──────────────────────
            # Selection export always applies the chi_eff swap (Mode A); the
            # naming mirrors the PE export so downstream can cross-check the two.
            f.attrs["spin_prior_mode"] = "include"
            f.attrs["chi_eff_prior_applied_to_pdraw"] = True
            f.attrs["mass_jacobian_applied"] = True
            # pdraw is the injection draw density in the (m1det,q,dL) basis; no
            # distance PRIOR is removed (injections have a draw distribution).
            f.attrs["distance_prior_removed"] = False
            f.attrs["cosmology_override_used"] = bool(self._cosmology_override)
            _write_selection_provenance(
                f, source_class=source_class,
                nsbh_mass_threshold=self._nsbh_mass_threshold,
                n_before=n_before, n_after=n_after,
                far_columns=self._far_columns, far_threshold=far_threshold)

            for name, arr in [
                ("m1det", self._m1det[keep]), ("m2det", self._m2det[keep]),
                ("dL", self._dL[keep]), ("chieff", chieff_det),
                ("ra", self._ra[keep]), ("dec", self._dec[keep]),
                ("m1src", m1src_det), ("m2src", m2src_det),
                ("redshift", self._z[keep]), ("pdraw", pdraw_det),
            ]:
                f.create_dataset(name, data=arr, compression="gzip")

        if write_summary:
            from .validation_summary import (write_validation_summary,
                                            value_counts, package_version)
            classes_det = (classify_by_mass(m1src_det, m2src_det,
                                            self._nsbh_mass_threshold)
                          if n_det else [])
            summary = {
                "kind": "selection_export",
                "output_path": str(out_path),
                "package_version": package_version(),
                "schema_version": "gwcat-selection-1.0",
                "n_campaigns": 1,
                "n_injections_total": int(self.n_retained),
                "n_injections_before_filter": n_before,
                "n_injections_after_filter": n_after,
                "n_detected": n_det,
                "ndraw": int(self._ndraw),
                "T_obs_yr": float(self._T_yr),
                "far_threshold": float(far_threshold),
                "significance_columns": list(self._far_columns),
                "significance_available": bool(self._far_columns),
                "p_astro_available": False,
                "source_class_filter": (
                    None if source_class is None
                    else format_source_class_filter(source_class)),
                CUT_ESTIMATOR_ATTR: selection_cut_estimator(source_class),
                "source_class_counts_detected": (
                    value_counts([normalize_source_class(c) for c in classes_det])),
                "cosmology_H0": _cosmo_or_nan(self, "_cosmology_used_H0"),
                "cosmology_Om0": _cosmo_or_nan(self, "_cosmology_used_Om0"),
                "cosmology_source": str(self._cosmology_source),
                "cosmology_override_used": bool(self._cosmology_override),
                "spin_prior_mode": "include",
                "chi_eff_prior_applied_to_pdraw": True,
            }
            if summary_context:
                summary.update(summary_context)
            write_validation_summary(out_path, summary)

        print(f"Wrote {out_path}: n_det={n_det}, ndraw={self._ndraw}, "
              f"FAR<{far_threshold}, "
              f"H0={_cosmo_or_nan(self, '_cosmology_used_H0')}, "
              f"Om0={_cosmo_or_nan(self, '_cosmology_used_Om0')} "
              f"(source={self._cosmology_source}), "
              f"source_class={source_class}")
        return out_path

    def export(self, out_path, format="gwcat2", spin_basis=None,
               write_summary=False, summary_context=None, **builder_kwargs):
        """Export via the versioned :mod:`gwcat.export` pipeline (PR5).

        Thin dispatch mirroring :meth:`gwcat.catalog.GWCatalog.export`: build a
        selection :class:`~gwcat.export.product.ExportProduct` (which owns all
        the physics -- detection cut, source-class subsetting, the per-basis
        spin factor and the Essick fractions), look up the ``(format,
        "selection")`` writer, and serialize.  The file carries
        ``format_version="gwcat-selection-2.0"``.

        For ``spin_basis="chieff"`` the exported ``pdraw`` array is
        byte-identical to :meth:`to_darksirens` with the same kwargs (the
        legacy 1-D chi_eff swap).  ``**builder_kwargs`` are forwarded to
        :func:`gwcat.export.build_selection_product` (``far_threshold``,
        ``source_class``, ``amax``, ``snr_threshold``, ``strict``).
        """
        from .export import build_selection_product, get_exporter
        from .params import DEFAULT_PARAMETER_SPACE
        if spin_basis is None:
            spin_basis = DEFAULT_PARAMETER_SPACE
        product = build_selection_product(self, spin_basis=spin_basis,
                                          **builder_kwargs)
        writer = get_exporter(format, "selection")
        return writer(product, out_path, write_summary=write_summary,
                      summary_context=summary_context)


class CombinedSelectionSet:
    """Combine injection sets from multiple observing campaigns.

    Implements the multi-campaign VT estimator (Essick et al. 2023):
    each campaign contributes its detected injections weighted by its
    share of the total generated count, so the combined estimator is

        ⟨VT⟩ = ⟨VT⟩_A + ⟨VT⟩_B = (1/N_total) Σ_det [Λ(θ) / pdraw(θ)]

    where pdraw for injection i from campaign k is rescaled:

        pdraw_combined_i = pdraw_k_i × (N_k / N_total)

    Parameters
    ----------
    selection_sets : list of SelectionSet
        One per observing campaign (e.g. O3 and O4ab).
        All must use the same reference cosmology.

    Usage
    -----
    >>> sel_o3 = SelectionSet("endo3_bbhpop-...-v12.hdf5")
    >>> sel_o4 = SelectionSet("injections-O4ab/...-cartesian_spins_*.hdf")
    >>> combined = CombinedSelectionSet([sel_o3, sel_o4])
    >>> combined.to_darksirens("selection_bbh.h5", far_threshold=1.0)
    """

    def __init__(self, selection_sets):
        if not selection_sets:
            raise ValueError("Need at least one SelectionSet")
        self._sets = list(selection_sets)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def n_campaigns(self) -> int:
        return len(self._sets)

    @property
    def n_retained(self) -> int:
        """Injection rows carried by the campaign files, summed."""
        return sum(s.n_retained for s in self._sets)

    @property
    def n_generated(self) -> int:
        """Combined ``ndraw`` = sum of the per-campaign ``total_generated``."""
        return sum(s.n_generated for s in self._sets)

    @property
    def n_injections(self) -> int:
        """Alias of :attr:`n_retained`, kept for backward compatibility."""
        return self.n_retained

    def detection_efficiency(self, far_threshold: float = 1.0) -> float:
        """Detected fraction of the combined GENERATED draws (see the
        single-campaign twin: the denominator is ndraw, not retained rows)."""
        n_det = sum(int(s.detected_mask(far_threshold).sum()) for s in self._sets)
        return n_det / self.n_generated

    # ── PR4 per-campaign component-basis spin access ───────────────────────
    @property
    def spin_meta(self) -> list:
        """Per-campaign ``spin_meta`` dicts, in campaign order."""
        return [s.spin_meta for s in self._sets]

    @property
    def component_spin_available(self) -> bool:
        """True iff every campaign carries an exact component-basis spin draw."""
        return all(s.component_spin_available for s in self._sets)

    def component_spin_arrays(self, far_threshold: float = 1.0,
                              source_class=None) -> dict:
        """Concatenated per-campaign spin arrays for the future builder.

        Mirrors the ``keep = detected & source_class`` masking and the
        campaign ordering of :meth:`to_darksirens` (empty campaigns skipped), so
        the returned arrays align element-for-element with the combined export's
        internals.  Each of ``a1, a2, cost1, cost2, chi_p, ln_spin_component`` is
        a concatenated array, or ``None`` if any contributing campaign lacks it.
        ``ln_spin_component`` is per-injection and unaffected by the Essick
        ``N_k/N_total`` reweighting (which only scales ``_pdraw``).
        """
        for s in self._sets:
            s._load()
        keys = ["a1", "a2", "cost1", "cost2", "chi_p", "ln_spin_component"]
        attr = {"a1": "_a1", "a2": "_a2", "cost1": "_cost1", "cost2": "_cost2",
                "chi_p": "_chi_p", "ln_spin_component": "_ln_spin_component"}
        parts = {k: [] for k in keys}
        available = {k: True for k in keys}
        for s in self._sets:
            keep = s.detected_mask(far_threshold) & s.source_class_mask(source_class)
            if not keep.any():
                continue
            for k in keys:
                arr = getattr(s, attr[k])
                if arr is None:
                    available[k] = False
                else:
                    parts[k].append(np.asarray(arr)[keep])
        return {k: (np.concatenate(parts[k]) if available[k] and parts[k]
                    else None) for k in keys}

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def to_darksirens(self, out_path: str, far_threshold: float = 1.0,
                      amax=AMAX_AUTO, source_class=None,
                      write_summary: bool = False,
                      summary_context: Optional[dict] = None,
                      strict: bool = True):
        """Write a combined selection file for darksirens.

        Parameters
        ----------
        out_path : str
        far_threshold : float
            FAR detection threshold in yr⁻¹, applied per campaign.
        amax : float or ``"auto"``, default ``"auto"``
            Ceiling of the uniform spin-magnitude prior behind the chi_eff
            swap.  ``"auto"`` (GW-37) resolves it PER CAMPAIGN from each
            campaign's own injected draw; a number forces one ceiling on all of
            them, which is what the pre-GW-37 default did.
        source_class : str, iterable, or None
            Optional source-class filter applied per campaign by injected
            source-frame mass (see :meth:`SelectionSet.source_class_mask`).
            ``None`` (default) is byte-identical to the pre-PR9 export.  As in
            the single-campaign exporter this is subsetting, not reweighting:
            each campaign's ``ndraw`` share is unchanged, so the Essick et al.
            fractions ``N_k / N_total`` are identical to the unfiltered file.
        write_summary : bool, default False
            (PR 10) When True, write ``<out_path>.validation_summary.json`` and
            ``.md`` next to ``out_path``.  See :mod:`gwcat.validation_summary`.
        summary_context : dict, optional
            Extra fields merged into the written summary.
        strict : bool, default True
            Raise when a campaign's injected spins are measurably not
            uniform-magnitude/isotropic (GW-37).  ``False`` warns.
        """
        # Load all campaigns
        for s in self._sets:
            s._load()

        # Cosmology consistency check
        H0s = [s.H0 for s in self._sets]
        Om0s = [s.Om0 for s in self._sets]
        if max(H0s) - min(H0s) > 1.0 or max(Om0s) - min(Om0s) > 0.05:
            warnings.warn(
                f"Cosmology mismatch across campaigns: "
                f"H0={H0s}, Om0={Om0s}. Results may be inconsistent."
            )

        # Combined ndraw for Essick et al. reweighting
        ndraw_per = [s._ndraw for s in self._sets]
        ndraw_total = sum(ndraw_per)

        cols = {k: [] for k in ["m1det", "m2det", "dL", "chieff",
                                "ra", "dec", "m1src", "m2src", "z", "pdraw"]}
        n_det_total = 0
        n_before_total = 0
        n_after_total = 0
        far_columns_union = []
        campaign_info = []
        # The campaigns that actually CONTRIBUTED rows, in concatenation order
        # (empty ones are skipped below).  The chi_eff swap resolves its ceiling
        # per campaign, so it needs the same list in the same order (GW-37).
        contrib_sets, contrib_keeps = [], []

        for k, s in enumerate(self._sets):
            det = s.detected_mask(far_threshold)
            sc_mask = s.source_class_mask(source_class)
            keep = det & sc_mask
            n_before_total += int(det.size)
            n_after_total += int(sc_mask.sum())
            for c in s._far_columns:
                if c not in far_columns_union:
                    far_columns_union.append(c)
            n_det_k = int(keep.sum())
            if n_det_k == 0:
                warnings.warn(
                    f"Campaign {s.path}: no detected injections at "
                    f"FAR < {far_threshold}"
                    + ("" if source_class is None
                       else f" in source class {source_class!r}"))
                continue

            # Essick et al. reweighting: pdraw_i *= N_k / N_total.  Source-class
            # filtering is subsetting only -- N_k/N_total is unchanged.
            frac = ndraw_per[k] / ndraw_total
            pdraw_k = s._pdraw[keep] * frac

            cols["m1det"].append(s._m1det[keep])
            cols["m2det"].append(s._m2det[keep])
            cols["dL"].append(s._dL[keep])
            cols["chieff"].append(s._chieff[keep])
            cols["ra"].append(s._ra[keep])
            cols["dec"].append(s._dec[keep])
            cols["m1src"].append(s._m1src[keep])
            cols["m2src"].append(s._m2src[keep])
            cols["z"].append(s._z[keep])
            cols["pdraw"].append(pdraw_k)
            contrib_sets.append(s)
            contrib_keeps.append(keep)
            n_det_total += n_det_k
            campaign_info.append(
                f"{s.path}: N={ndraw_per[k]}, T={s._T_yr:.2f}yr, "
                f"n_det={n_det_k}, frac={frac:.4f}")

        if n_det_total == 0:
            raise RuntimeError(
                f"No detected injections across {len(self._sets)} campaigns "
                f"at FAR < {far_threshold}"
                + ("" if source_class is None
                   else f" in source class {source_class!r}"))

        # Concatenate
        data = {k: np.concatenate(v) for k, v in cols.items()}

        # Apply 1-D chi_eff prior swap, per-campaign ceilings and support gate
        # shared with the v2 builder (GW-37).
        logp_chi, in_support, amax_pairs, amax_sources, amax_mode = (
            _v1_chieff_swap(contrib_sets, contrib_keeps, amax, strict=strict))
        # No -50 floor (GW-03); see the sibling exporter above.
        _v1_refuse_unsupported(logp_chi, in_support, amax_pairs,
                               "to_combined_selection_file")
        with np.errstate(over="ignore"):
            data["pdraw"] *= np.exp(logp_chi)
        # The array that is about to be written (GW-28); the Essick N_k/N_total
        # rescaling and the chi_eff swap both touched it since _load vetted it.
        _require_positive_finite(
            data["pdraw"], "pdraw (after the chi_eff swap and the Essick "
            "N_k/N_total rescaling)",
            " + ".join(s.path for s in self._sets),
            note=f"amax={amax_pairs}, far_threshold={far_threshold}.")

        # An override that reached only SOME campaigns corrupts the combined
        # pdraw exactly as it does on the v2 path (GW-09); this exporter has
        # every campaign in view too, so it refuses identically.
        _refuse_mixed_cosmology(self._sets)

        # Write
        with h5py.File(out_path, "w") as f:
            f.attrs["format_version"] = "gwcat-selection-1.0"

            # Which gwcat wrote this file (DS-10 provenance; also on the v2
            # writers).  Commit, not version: an editable install moves per
            # commit while the version string stands still.
            from .validation_summary import gwcat_commit, package_version as _pkg_version
            f.attrs["writer_commit"] = gwcat_commit()
            f.attrs["writer_version"] = _pkg_version()
            f.attrs["ndraw"] = ndraw_total
            f.attrs["T_obs_yr"] = float(sum(s._T_yr for s in self._sets))
            f.attrs["far_threshold"] = float(far_threshold)
            f.attrs["n_detected"] = n_det_total
            # The scalar is the cosmology every campaign's pdraw used -- NaN
            # when the campaigns legitimately differ (each used its own
            # generation cosmology) or used none at all; the per-campaign
            # arrays below are the authoritative record.  Stamping
            # sets[0].H0 (always Planck15) mis-described every campaign whose
            # pdraw used a DETECTED cosmology.
            _mixed = _cosmology_is_mixed(self._sets)
            f.attrs["cosmology_H0"] = (
                float("nan") if _mixed
                else _cosmo_or_nan(self._sets[0], "_cosmology_used_H0"))
            f.attrs["cosmology_Om0"] = (
                float("nan") if _mixed
                else _cosmo_or_nan(self._sets[0], "_cosmology_used_Om0"))
            f.attrs["cosmology_mixed_across_campaigns"] = bool(_mixed)
            f.attrs.create(
                "cosmology_source_per_campaign",
                np.array([str(getattr(s, "_cosmology_source", None))
                          for s in self._sets], dtype=h5py.string_dtype()))
            f.attrs.create(
                "cosmology_H0_per_campaign",
                np.array([_cosmo_or_nan(s, "_cosmology_used_H0")
                          for s in self._sets], dtype=float))
            f.attrs.create(
                "cosmology_Om0_per_campaign",
                np.array([_cosmo_or_nan(s, "_cosmology_used_Om0")
                          for s in self._sets], dtype=float))
            # Per-campaign sky availability (GW-10): NaN-sky campaigns (the
            # semianalytic O1/O2 mixture rows) may be legitimately concatenated
            # with real-sky ones, but the file must say WHICH rows are which
            # class -- and mixing deserves a warning, because a consumer doing
            # sky work will silently lose the NaN campaigns' injections.
            _sky = np.array([bool(getattr(s, "_sky_position_available", True))
                             for s in self._sets], dtype=bool)
            f.attrs.create("sky_position_available", _sky)
            if _sky.any() and not _sky.all():
                warnings.warn(
                    f"{out_path}: concatenating campaign(s) WITHOUT drawn sky "
                    f"positions (exported ra/dec are NaN) with campaign(s) "
                    f"that have them: sky_position_available per campaign = "
                    f"{_sky.tolist()}. Any sky-dependent selection use will "
                    f"silently drop the NaN campaigns' injections; cut on "
                    f"campaign, not on finiteness, if that is not intended.")
            f.attrs["chi_eff_swap_applied"] = True
            _write_chieff_amax_attrs(f, amax_pairs, amax_sources, amax_mode)
            # ── Spin-prior contract provenance (PR 3) ──────────────────────
            f.attrs["spin_prior_mode"] = "include"
            f.attrs["chi_eff_prior_applied_to_pdraw"] = True
            f.attrs["mass_jacobian_applied"] = True
            f.attrs["distance_prior_removed"] = False
            f.attrs["cosmology_override_used"] = bool(
                any(getattr(s, "_cosmology_override", False)
                    for s in self._sets))
            f.attrs["n_campaigns"] = len(self._sets)
            f.attrs.create("campaign_ndraws",
                           np.array(ndraw_per, dtype=np.int64))
            _write_selection_provenance(
                f, source_class=source_class,
                nsbh_mass_threshold=self._sets[0]._nsbh_mass_threshold,
                n_before=n_before_total, n_after=n_after_total,
                far_columns=far_columns_union, far_threshold=far_threshold)

            for name, arr in [
                ("m1det", data["m1det"]), ("m2det", data["m2det"]),
                ("dL", data["dL"]), ("chieff", data["chieff"]),
                ("ra", data["ra"]), ("dec", data["dec"]),
                ("m1src", data["m1src"]), ("m2src", data["m2src"]),
                ("redshift", data["z"]), ("pdraw", data["pdraw"]),
            ]:
                f.create_dataset(name, data=arr, compression="gzip")

        if write_summary:
            from .validation_summary import (write_validation_summary,
                                            value_counts, package_version)
            classes_det = (classify_by_mass(data["m1src"], data["m2src"],
                                            self._sets[0]._nsbh_mass_threshold)
                          if n_det_total else [])
            summary = {
                "kind": "selection_export",
                "output_path": str(out_path),
                "package_version": package_version(),
                "schema_version": "gwcat-selection-1.0",
                "n_campaigns": len(self._sets),
                "campaign_paths": [s.path for s in self._sets],
                "campaign_ndraws": list(ndraw_per),
                "n_injections_total": int(self.n_retained),
                "n_injections_before_filter": n_before_total,
                "n_injections_after_filter": n_after_total,
                "n_detected": n_det_total,
                "ndraw": int(ndraw_total),
                "T_obs_yr": float(sum(s._T_yr for s in self._sets)),
                "far_threshold": float(far_threshold),
                "significance_columns": list(far_columns_union),
                "significance_available": bool(far_columns_union),
                "p_astro_available": False,
                "source_class_filter": (
                    None if source_class is None
                    else format_source_class_filter(source_class)),
                CUT_ESTIMATOR_ATTR: selection_cut_estimator(source_class),
                "source_class_counts_detected": (
                    value_counts([normalize_source_class(c) for c in classes_det])),
                "cosmology_H0": (
                    float("nan") if _mixed
                    else _cosmo_or_nan(self._sets[0], "_cosmology_used_H0")),
                "cosmology_Om0": (
                    float("nan") if _mixed
                    else _cosmo_or_nan(self._sets[0], "_cosmology_used_Om0")),
                "cosmology_mixed_across_campaigns": bool(_mixed),
                "cosmology_source_per_campaign": [
                    str(getattr(s, "_cosmology_source", None))
                    for s in self._sets],
                "cosmology_override_used": bool(
                    any(getattr(s, "_cosmology_override", False)
                        for s in self._sets)),
                "spin_prior_mode": "include",
                "chi_eff_prior_applied_to_pdraw": True,
            }
            if summary_context:
                summary.update(summary_context)
            write_validation_summary(out_path, summary)

        for info in campaign_info:
            print(f"  {info}")
        print(f"Wrote {out_path}: n_det={n_det_total}, ndraw={ndraw_total}, "
              f"FAR<{far_threshold}, campaigns={len(self._sets)}")
        return out_path

    def export(self, out_path, format="gwcat2", spin_basis=None,
               write_summary=False, summary_context=None, **builder_kwargs):
        """Export the combined campaigns via the versioned pipeline (PR5).

        Thin dispatch to :func:`gwcat.export.build_selection_product` over all
        campaigns (Essick ``N_k/N_total`` fractions applied per campaign before
        concatenation, exactly as :meth:`to_darksirens`), then the registered
        ``(format, "selection")`` writer.  See :meth:`SelectionSet.export`.
        """
        from .export import build_selection_product, get_exporter
        from .params import DEFAULT_PARAMETER_SPACE
        if spin_basis is None:
            spin_basis = DEFAULT_PARAMETER_SPACE
        product = build_selection_product(self._sets, spin_basis=spin_basis,
                                          **builder_kwargs)
        writer = get_exporter(format, "selection")
        return writer(product, out_path, write_summary=write_summary,
                      summary_context=summary_context)


# ======================================================================
# PR4: mixture-flavour cross-check
# ======================================================================
def crosscheck_mixture_flavors(path_polar, path_cartesian, rtol=1e-9,
                               strict_spin_checks="off"):
    """Cross-check the two "completely equivalent" Format-C mixture flavours.

    Loads the polar-flavour and cartesian-flavour files as
    :class:`SelectionSet` objects (which share the same underlying draws in the
    same order) and compares:

    * the exact **component-basis** draw density
      ``pdraw_component = _pdraw · exp(_ln_spin_component)``, which must agree to
      ``rtol`` -- this is the key invariance the component conversions must
      satisfy; and
    * the **legacy** ``_pdraw``, which the polar-joint subtraction is
      constructed to make byte-comparable to the cartesian branch.

    Returns a report dict.  Raises :class:`ValueError` if either comparison
    exceeds ``rtol`` or if a component-basis density is unobtainable.
    """
    sp = SelectionSet(path_polar, strict_spin_checks=strict_spin_checks)
    sc = SelectionSet(path_cartesian, strict_spin_checks=strict_spin_checks)
    sp._load()
    sc._load()

    if sp._ln_spin_component is None or sc._ln_spin_component is None:
        raise ValueError(
            "crosscheck_mixture_flavors: component-basis spin density "
            f"unavailable (polar={sp._spin_meta.get('spin_format')!r}, "
            f"cartesian={sc._spin_meta.get('spin_format')!r}).")
    if sp._pdraw.shape != sc._pdraw.shape:
        raise ValueError(
            "crosscheck_mixture_flavors: flavour files differ in length "
            f"({sp._pdraw.shape} vs {sc._pdraw.shape}).")

    comp_p = sp.component_pdraw()
    comp_c = sc.component_pdraw()

    def _max_rel(a, b):
        denom = np.where(np.abs(b) > 0, np.abs(b), 1.0)
        return float(np.max(np.abs(a - b) / denom)) if a.size else 0.0

    comp_dev = _max_rel(comp_p, comp_c)
    legacy_dev = _max_rel(sp._pdraw, sc._pdraw)
    report = {
        "n": int(sp._pdraw.size),
        "component_pdraw_max_rel_dev": comp_dev,
        "legacy_pdraw_max_rel_dev": legacy_dev,
        "rtol": float(rtol),
        "component_pdraw_ok": bool(comp_dev <= rtol),
        "legacy_pdraw_ok": bool(legacy_dev <= rtol),
        "spin_format_polar": sp._spin_meta.get("spin_format"),
        "spin_format_cartesian": sc._spin_meta.get("spin_format"),
    }
    if not (report["component_pdraw_ok"] and report["legacy_pdraw_ok"]):
        raise ValueError(
            "crosscheck_mixture_flavors mismatch: "
            f"component_pdraw max rel dev={comp_dev:.3e}, "
            f"legacy_pdraw max rel dev={legacy_dev:.3e} (rtol={rtol:.1e}). "
            f"Report: {report}")
    return report