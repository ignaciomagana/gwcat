"""Selection function processing for LVK injection sets.

Reads the modern LVK injection format ('events/' group), which is used by
ALL current releases including the cumulative O1–O4b set (Zenodo 19500052).
The legacy O3-only 'injections/' format (Zenodo 7890437) is superseded and
not supported — use the cumulative files instead.

Processing pipeline:
  1. Read source-frame masses, spins, sky position from 'events/' group
  2. Compute detector-frame masses, chi_eff, redshift
  3. Divide out the analytical 6-D spin prior from the draw PDF
  4. Apply (m1src,m2src,z) → (m1det,q,dL) coordinate Jacobian
  5. Normalise by observing time and injection weights
  6. Apply FAR-based detection cut

The 1-D chi_eff spin-prior swap is NOT applied here — darksirens handles it
via gwdistributions.  This keeps gwcat spin-prior-agnostic.

Usage:
    from gwcat.selection import SelectionSet
    sel = SelectionSet("injection_file.hdf")
    sel.to_darksirens("selection_out.h5", far_threshold=1.0)
"""
from __future__ import annotations

import warnings
import numpy as np
import h5py

from .cosmology import make_cosmology, z_of_dL, PLANCK15


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

    def __init__(self, path: str, H0: float = None, Om0: float = None):
        self.path = path
        self.H0 = H0 or PLANCK15.H0.value
        self.Om0 = Om0 or PLANCK15.Om0
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
                raise RuntimeError(
                    f"{self.path} uses the legacy O3 'injections/' format. "
                    "This format is superseded by the cumulative GWTC-5 "
                    "injection sets (Zenodo 19500052 / 19500064) which cover "
                    "O1–O4b in the modern 'events/' format. Use those instead."
                )
            else:
                raise RuntimeError(
                    f"Unrecognised injection format in {self.path}: "
                    "expected 'events/' group."
                )
        self._loaded = True

    def _read_events(self, f):
        """Read the 'events/' format (used by all current LVK injection sets)."""
        ev = f["events"]

        # Source-frame parameters
        m1src = np.asarray(ev["mass1_source"], float)
        m2src = np.asarray(ev["mass2_source"], float)
        dL = np.asarray(ev["luminosity_distance"], float)
        ra = np.asarray(ev["right_ascension"], float)
        dec = np.asarray(ev["declination"], float)

        # Spin components (cartesian)
        s1x = np.asarray(ev["spin1x"], float)
        s1y = np.asarray(ev["spin1y"], float)
        s1z = np.asarray(ev["spin1z"], float)
        s2x = np.asarray(ev["spin2x"], float)
        s2y = np.asarray(ev["spin2y"], float)
        s2z = np.asarray(ev["spin2z"], float)

        # Derived
        chieff = (m1src * s1z + m2src * s2z) / (m1src + m2src)
        cosmo = make_cosmology(self.H0, self.Om0)
        z = z_of_dL(dL, cosmo)
        m1det = m1src * (1 + z)
        m2det = m2src * (1 + z)

        # Injection weights
        weights = np.asarray(ev["weights"], float)

        # Joint log-draw PDF (9-D: m1src, m2src, z, 6 spin components)
        ln_pdraw_joint = np.asarray(
            ev["lnpdraw_mass1_source_mass2_source_redshift_"
               "spin1x_spin1y_spin1z_spin2x_spin2y_spin2z"], float)

        # Analytical 6-D isotropic spin prior:
        #   p(s1x,s1y,s1z,s2x,s2y,s2z) = 1 / (16 pi^2 a1^2 a2^2 amax^2)
        # where ai = |si|.  Divide this out so darksirens can replace it
        # with the 1-D chi_eff marginal.
        a1 = np.sqrt(s1x ** 2 + s1y ** 2 + s1z ** 2)
        a2 = np.sqrt(s2x ** 2 + s2y ** 2 + s2z ** 2)
        amax = 0.99
        # Guard against a1 or a2 = 0 (vanishing spin magnitude)
        a1 = np.maximum(a1, 1e-30)
        a2 = np.maximum(a2, 1e-30)
        ln_pdraw_spin6d = -np.log(16.0 * np.pi ** 2 * a1 ** 2 * a2 ** 2 * amax ** 2)

        # Remove 6-D spin; do NOT add 1-D chi_eff (darksirens does that)
        ln_pdraw_no_spin = ln_pdraw_joint - ln_pdraw_spin6d

        # Coordinate Jacobian: (m1src, m2src, z) → (m1det, q, dL)
        #   |J| = m1det / (1+z)^2 / (ddL/dz)
        ddL = _ddL_dz(z, dL, self.H0, self.Om0)
        pdraw = np.exp(ln_pdraw_no_spin) * m1det / (1 + z) ** 2 / ddL

        # Time normalisation
        T_yr = f.attrs["total_analysis_time"] / (3600 * 24 * 365.25)
        pdraw /= T_yr

        # Injection weights (reweighting factor for non-uniform draw campaigns)
        pdraw /= weights

        ndraw = int(f.attrs["total_generated"])

        # FAR: per-search columns
        searches = [s.decode() if isinstance(s, bytes) else str(s)
                    for s in f.attrs["searches"]]
        fars_per_search = []
        for s in searches:
            col = f"{s}_far"
            if col in ev:
                fars_per_search.append(np.asarray(ev[col], float))
        self._fars = np.column_stack(fars_per_search) if fars_per_search else None

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
    def to_darksirens(self, out_path: str, far_threshold: float = 1.0):
        """Write a pre-processed selection file for darksirens.

        The 1-D chi_eff spin-prior swap is NOT applied (chi_eff_swap_applied=False).
        darksirens applies it via gwdistributions.
        """
        self._load()
        det = self.detected_mask(far_threshold)
        n_det = int(det.sum())
        if n_det == 0:
            raise RuntimeError(f"No detected injections at FAR < {far_threshold}")

        with h5py.File(out_path, "w") as f:
            f.attrs["format_version"] = "gwcat-selection-1.0"
            f.attrs["ndraw"] = self._ndraw
            f.attrs["T_obs_yr"] = float(self._T_yr)
            f.attrs["far_threshold"] = float(far_threshold)
            f.attrs["n_detected"] = n_det
            f.attrs["cosmology_H0"] = float(self.H0)
            f.attrs["cosmology_Om0"] = float(self.Om0)
            f.attrs["chi_eff_swap_applied"] = False

            for name, arr in [
                ("m1det", self._m1det), ("m2det", self._m2det),
                ("dL", self._dL), ("chieff", self._chieff),
                ("ra", self._ra), ("dec", self._dec),
                ("m1src", self._m1src), ("m2src", self._m2src),
                ("redshift", self._z), ("pdraw", self._pdraw),
            ]:
                f.create_dataset(name, data=arr[det], compression="gzip")

        print(f"Wrote {out_path}: n_det={n_det}, ndraw={self._ndraw}, "
              f"FAR<{far_threshold}, H0={self.H0}, Om0={self.Om0}")
        return out_path