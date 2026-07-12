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
from . import selection_spin as _sspin
from .spin import chi_p_from_components

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
        joint_cart = (
            "lnpdraw_mass1_source_mass2_source_redshift_"
            "spin1x_spin1y_spin1z_spin2x_spin2y_spin2z"
        )
        joint_polar = (
            "lnpdraw_mass1_source_mass2_source_redshift_"
            "spin1_magnitude_spin1_polar_angle_spin1_azimuthal_angle_"
            "spin2_magnitude_spin2_polar_angle_spin2_azimuthal_angle"
        )
        joint_no_spin_names = [
            "lnpdraw_mass1_source_mass2_source_redshift",
            "lnpdraw_mass1_source_mass2_source_z",
        ]
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
            amax = 0.99
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
            ln_pdraw_no_spin = _sspin.ln_pdraw_no_spin_from_polar_joint(
                ln_pdraw_joint, scost1, scost2, amax=0.99)
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
        self._weights = weights

        # ── Additive component-basis spin state (PR4) ──────────────────────
        self._compute_events_spin_state(
            ev, spin_fmt, ln_pdraw_no_spin, ln_pdraw_joint,
            sa1, scost1, sa2, scost2, m1src, m2src)

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
            weights = np.asarray(inj["mixture_weight"], float)
            pdraw /= weights
        else:
            weights = np.ones_like(pdraw)

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
            ln_joint = np.log(np.maximum(
                np.asarray(inj["sampling_pdf"], float), 1e-300))
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
    def n_injections(self) -> int:
        self._load()
        return len(self._m1det)

    def detection_efficiency(self, far_threshold: float = 1.0) -> float:
        """Fraction of injections detected at the given FAR threshold."""
        return self.detected_mask(far_threshold).sum() / self.n_injections

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