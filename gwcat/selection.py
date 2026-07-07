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
                          resolve_filter_classes, DEFAULT_NSBH_MASS_THRESHOLD)

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


def _write_selection_provenance(f, source_class, nsbh_mass_threshold,
                                n_before, n_after, far_columns, far_threshold):
    """Write the PR9 pdraw / source-class / significance provenance attrs.

    Shared by :meth:`SelectionSet.to_darksirens` and
    :meth:`CombinedSelectionSet.to_darksirens` so the two exporters record the
    same contract in the same way.  Records truthfully what the code did; it
    changes none of the math.
    """
    # ── pdraw state ────────────────────────────────────────────────────────
    f.attrs["pdraw_state"] = PDRAW_STATE

    # ── Source-class filter provenance ─────────────────────────────────────
    f.attrs["source_class_filter"] = (
        "" if source_class is None
        else (str(source_class) if isinstance(source_class, (str, bytes))
              else ",".join(str(x) for x in source_class)))
    f.attrs["source_class_method"] = (
        "none" if source_class is None else "mass_threshold")
    f.attrs["nsbh_mass_threshold"] = float(nsbh_mass_threshold)
    f.attrs["n_injections_before_filter"] = int(n_before)
    f.attrs["n_injections_after_filter"] = int(n_after)
    if source_class is not None:
        f.attrs["source_class_filter_note"] = SOURCE_CLASS_FILTER_NOTE

    # ── Search / significance provenance (explicit-absence, per the FAR
    #    contract): record which columns/pipelines were thresholded, the
    #    threshold applied, and that no per-injection p_astro was used. ──────
    cols = [str(c) for c in (far_columns or [])]
    f.attrs.create("significance_columns",
                   np.array(cols, dtype=h5py.string_dtype()))
    f.attrs["significance_type"] = "far"
    f.attrs["significance_far_threshold"] = float(far_threshold)
    f.attrs["significance_available"] = bool(len(cols) > 0)
    # No per-injection p_astro is read/used for thresholding here; record the
    # absence explicitly rather than pretending it exists.
    f.attrs["p_astro_available"] = False


def _ddL_dz(z, dL_mpc, H0, Om0):
    """d(dL)/dz evaluated at z.  dL in Mpc."""
    c_kms = 299792.458
    dH = c_kms / H0
    E = np.sqrt(Om0 * (1 + z) ** 3 + (1.0 - Om0))
    DC = dL_mpc / (1 + z)
    return DC + (1 + z) * dH / E


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
    """

    def __init__(self, path: str, H0: float = None, Om0: float = None,
                 nsbh_mass_threshold: float = None):
        self.path = path
        self.H0 = H0 or PLANCK15.H0.value
        self.Om0 = Om0 or PLANCK15.Om0
        # Whether the caller supplied a non-default reference cosmology.
        self._cosmology_override = (H0 is not None) or (Om0 is not None)
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
        self._loaded = True

    def _read_events(self, f):
        """Read the O4 ``events`` format.

        The O4 Zenodo files store ``events`` as a single compound HDF5
        dataset, while some downstream/older files expose the same columns as
        datasets in an ``events/`` group.  Support both layouts here.
        """
        ev = f["events"]

        # Source-frame parameters.  The O4 release also stores detector-frame
        # masses and redshift directly; use them when present so that we do not
        # introduce small differences by re-inverting dL with our cosmology.
        m1src = _h5_read_field(ev, "mass1_source")
        m2src = _h5_read_field(ev, "mass2_source")
        dL = _h5_read_field(ev, "luminosity_distance")
        ra = _h5_read_field(ev, "right_ascension")
        dec = _h5_read_field(ev, "declination")

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
        s1x = _h5_read_field(ev, "spin1x")
        s1y = _h5_read_field(ev, "spin1y")
        s1z = _h5_read_field(ev, "spin1z")
        s2x = _h5_read_field(ev, "spin2x")
        s2y = _h5_read_field(ev, "spin2y")
        s2z = _h5_read_field(ev, "spin2z")
        if _h5_has_field(ev, "chi_eff"):
            chieff = _h5_read_field(ev, "chi_eff")
        else:
            chieff = (m1src * s1z + m2src * s2z) / (m1src + m2src)

        weights = _h5_read_field(ev, "weights")

        # Draw probability in source-frame component masses and redshift, with
        # spins removed.  Older/current-development O4 files may contain a
        # single joint log-density over masses, redshift, and cartesian spins;
        # the public O4ab clipped release instead stores factored log-density
        # columns, including spin magnitudes/angles.  In the factored case we
        # simply omit all spin terms so darksirens can apply its chi_eff prior.
        joint_cart = (
            "lnpdraw_mass1_source_mass2_source_redshift_"
            "spin1x_spin1y_spin1z_spin2x_spin2y_spin2z"
        )
        joint_no_spin_names = [
            "lnpdraw_mass1_source_mass2_source_redshift",
            "lnpdraw_mass1_source_mass2_source_z",
        ]
        if _h5_has_field(ev, joint_cart):
            ln_pdraw_joint = _h5_read_field(ev, joint_cart)

            # Analytical 6-D isotropic cartesian spin prior:
            #   p(s1x,s1y,s1z,s2x,s2y,s2z)
            #     = 1 / (16 pi^2 a1^2 a2^2 amax^2)
            # where ai = |si|.  Divide this out so darksirens can replace it
            # with the 1-D chi_eff marginal.
            a1 = np.sqrt(s1x ** 2 + s1y ** 2 + s1z ** 2)
            a2 = np.sqrt(s2x ** 2 + s2y ** 2 + s2z ** 2)
            amax = 0.99
            a1 = np.maximum(a1, 1e-30)
            a2 = np.maximum(a2, 1e-30)
            ln_pdraw_spin6d = -np.log(
                16.0 * np.pi ** 2 * a1 ** 2 * a2 ** 2 * amax ** 2)
            ln_pdraw_no_spin = ln_pdraw_joint - ln_pdraw_spin6d
        elif any(_h5_has_field(ev, name) for name in joint_no_spin_names):
            ln_pdraw_no_spin, _ = _h5_first_field(ev, joint_no_spin_names)
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
        else:
            lnp_fields = sorted(
                name for name in _h5_field_names(ev) if name.startswith("lnpdraw"))
            raise RuntimeError(
                "Could not construct the spin-free O4 draw PDF. Expected either "
                f"{joint_cart!r}, one of {joint_no_spin_names!r}, or the "
                "factored public O4 fields "
                "'lnpdraw_mass1_source', "
                "'lnpdraw_mass2_source_GIVEN_mass1_source', and "
                "'lnpdraw_z'/'lnpdraw_redshift'. Available lnpdraw fields: "
                f"{lnp_fields}")

        # Coordinate Jacobian: (m1src, m2src, z) → (m1det, q, dL)
        #   |J| = m1det / (1+z)^2 / (ddL/dz)
        if _h5_has_field(ev, "dluminosity_distance_dredshift"):
            ddL = _h5_read_field(ev, "dluminosity_distance_dredshift")
        else:
            ddL = _ddL_dz(z, dL, self.H0, self.Om0)
        pdraw = np.exp(ln_pdraw_no_spin) * m1det / (1 + z) ** 2 / ddL

        # Time normalisation and mixture/month weights.  The O4 examples keep
        # weights in the numerator of importance-sampling sums; equivalently,
        # divide the stored draw density by weights.
        T_yr = f.attrs["total_analysis_time"] / (3600 * 24 * 365.25)
        pdraw /= T_yr
        pdraw /= weights

        ndraw = int(f.attrs["total_generated"])

        # FAR: discover search pipelines from the file, and handle both O4
        # names (e.g. "pycbc_far", "cwb-bbh_far") and older names.
        try:
            raw = f.attrs["searches"]
            if isinstance(raw, np.ndarray):
                search_list = [x.decode() if isinstance(x, bytes) else str(x)
                               for x in raw.flat]
            elif isinstance(raw, (list, tuple)):
                search_list = [x.decode() if isinstance(x, bytes) else str(x)
                               for x in raw]
            elif isinstance(raw, bytes):
                search_list = [raw.decode()]
            elif isinstance(raw, str):
                search_list = [raw]
            else:
                search_list = [str(raw)]
        except Exception:
            search_list = []

        fars_per_search = []
        far_columns = []
        for s in search_list:
            for col in (s + "_far", "far_" + s):
                if _h5_has_field(ev, col):
                    fars_per_search.append(_h5_read_field(ev, col))
                    far_columns.append(col)
                    break
        if not fars_per_search:
            for key in _h5_field_names(ev):
                if (isinstance(key, str)
                        and (key.endswith("_far")
                             or key.startswith("far_"))):
                    try:
                        fars_per_search.append(_h5_read_field(ev, key))
                        far_columns.append(key)
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

        # Spin-free draw PDF from factored components
        p_mass = np.asarray(inj["mass1_source_mass2_source_sampling_pdf"], float)
        p_z = np.asarray(inj["redshift_sampling_pdf"], float)
        ln_pdraw_no_spin = np.log(np.maximum(p_mass * p_z, 1e-300))

        # Jacobian: (m1src, m2src, z) → (m1det, q, dL)
        ddL = _ddL_dz(z, dL, self.H0, self.Om0)
        pdraw = np.exp(ln_pdraw_no_spin) * m1det / (1 + z) ** 2 / ddL

        # Time normalisation
        T_s = f.attrs.get("analysis_time_s",
                          inj.attrs.get("analysis_time_s"))
        if T_s is None:
            raise RuntimeError(
                f"No analysis_time_s attribute found in {self.path}")
        T_yr = float(T_s) / (3600 * 24 * 365.25)
        pdraw /= T_yr

        # Injection weights (mixture_weight = 1.0 for single-subpop files)
        if "mixture_weight" in inj:
            pdraw /= np.asarray(inj["mixture_weight"], float)

        ndraw = int(f.attrs.get("total_generated",
                                inj.attrs.get("total_generated", 0)))

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
    def n_injections(self) -> int:
        self._load()
        return len(self._m1det)

    def detection_efficiency(self, far_threshold: float = 1.0) -> float:
        """Fraction of injections detected at the given FAR threshold."""
        return self.detected_mask(far_threshold).sum() / self.n_injections

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def to_darksirens(self, out_path: str, far_threshold: float = 1.0,
                      amax: float = 0.99, source_class=None,
                      write_summary: bool = False,
                      summary_context: Optional[dict] = None):
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
        amax : float
            Maximum spin magnitude for the isotropic prior (default 0.99).
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
        """
        from .spin import chi_eff_prior_logprob

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

        # Apply the 1-D chi_eff prior swap
        chieff_det = self._chieff[keep]
        m1src_det = self._m1src[keep]
        m2src_det = self._m2src[keep]
        logp_chi = chi_eff_prior_logprob(chieff_det, m1src_det, m2src_det, amax=amax)
        safe_logp = np.clip(logp_chi, a_min=-50.0, a_max=None)
        pdraw_det = self._pdraw[keep] * np.exp(safe_logp)

        with h5py.File(out_path, "w") as f:
            f.attrs["format_version"] = "gwcat-selection-1.0"
            f.attrs["ndraw"] = self._ndraw
            f.attrs["T_obs_yr"] = float(self._T_yr)
            f.attrs["far_threshold"] = float(far_threshold)
            f.attrs["n_detected"] = n_det
            f.attrs["cosmology_H0"] = float(self.H0)
            f.attrs["cosmology_Om0"] = float(self.Om0)
            f.attrs["chi_eff_swap_applied"] = True
            f.attrs["chi_eff_amax"] = float(amax)
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
                "n_injections_total": int(self.n_injections),
                "n_injections_before_filter": n_before,
                "n_injections_after_filter": n_after,
                "n_detected": n_det,
                "ndraw": int(self._ndraw),
                "T_obs_yr": float(self._T_yr),
                "far_threshold": float(far_threshold),
                "significance_columns": list(self._far_columns),
                "significance_available": bool(self._far_columns),
                "p_astro_available": False,
                "source_class_filter": (None if source_class is None
                                        else str(source_class)),
                "source_class_counts_detected": (
                    value_counts([normalize_source_class(c) for c in classes_det])),
                "cosmology_H0": float(self.H0),
                "cosmology_Om0": float(self.Om0),
                "cosmology_override_used": bool(self._cosmology_override),
                "spin_prior_mode": "include",
                "chi_eff_prior_applied_to_pdraw": True,
            }
            if summary_context:
                summary.update(summary_context)
            write_validation_summary(out_path, summary)

        print(f"Wrote {out_path}: n_det={n_det}, ndraw={self._ndraw}, "
              f"FAR<{far_threshold}, H0={self.H0}, Om0={self.Om0}, "
              f"source_class={source_class}")
        return out_path


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
    def n_injections(self) -> int:
        return sum(s.n_injections for s in self._sets)

    def detection_efficiency(self, far_threshold: float = 1.0) -> float:
        n_det = sum(int(s.detected_mask(far_threshold).sum()) for s in self._sets)
        return n_det / self.n_injections

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def to_darksirens(self, out_path: str, far_threshold: float = 1.0,
                      amax: float = 0.99, source_class=None,
                      write_summary: bool = False,
                      summary_context: Optional[dict] = None):
        """Write a combined selection file for darksirens.

        Parameters
        ----------
        out_path : str
        far_threshold : float
            FAR detection threshold in yr⁻¹, applied per campaign.
        amax : float
            Maximum spin magnitude for the chi_eff prior (default 0.99).
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
        """
        from .spin import chi_eff_prior_logprob

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

        # Apply 1-D chi_eff prior swap
        logp_chi = chi_eff_prior_logprob(
            data["chieff"], data["m1src"], data["m2src"], amax=amax)
        safe_logp = np.clip(logp_chi, a_min=-50.0, a_max=None)
        data["pdraw"] *= np.exp(safe_logp)

        # Write
        with h5py.File(out_path, "w") as f:
            f.attrs["format_version"] = "gwcat-selection-1.0"
            f.attrs["ndraw"] = ndraw_total
            f.attrs["T_obs_yr"] = float(sum(s._T_yr for s in self._sets))
            f.attrs["far_threshold"] = float(far_threshold)
            f.attrs["n_detected"] = n_det_total
            f.attrs["cosmology_H0"] = float(self._sets[0].H0)
            f.attrs["cosmology_Om0"] = float(self._sets[0].Om0)
            f.attrs["chi_eff_swap_applied"] = True
            f.attrs["chi_eff_amax"] = float(amax)
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
                "n_injections_total": int(self.n_injections),
                "n_injections_before_filter": n_before_total,
                "n_injections_after_filter": n_after_total,
                "n_detected": n_det_total,
                "ndraw": int(ndraw_total),
                "T_obs_yr": float(sum(s._T_yr for s in self._sets)),
                "far_threshold": float(far_threshold),
                "significance_columns": list(far_columns_union),
                "significance_available": bool(far_columns_union),
                "p_astro_available": False,
                "source_class_filter": (None if source_class is None
                                        else str(source_class)),
                "source_class_counts_detected": (
                    value_counts([normalize_source_class(c) for c in classes_det])),
                "cosmology_H0": float(self._sets[0].H0),
                "cosmology_Om0": float(self._sets[0].Om0),
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