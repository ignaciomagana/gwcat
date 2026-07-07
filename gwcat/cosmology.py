"""Cosmology helpers and the cosmo-file luminosity-distance prior.

Everything here is mass-prior-agnostic. We only ever deal with the marginal
distance prior p(dL | cosmo). The (m1det, q, dL)-basis mass Jacobian is applied
elsewhere (exclusively in GWCatalog.to_darksirens).

The cosmo files (GWTC-2.1/3 *cosmo.h5 and the O4 combined files) use a
luminosity-distance prior that is uniform in comoving volume and source-frame
time -- i.e. bilby's UniformSourceFrame. We reproduce *that exact object* when
bilby is available, so the prior we divide out is identical to the one the LVK
PE used. A self-contained astropy fallback is provided for environments without
bilby; it implements the same density up to normalisation.

Note on normalisation: the darksirens loader normalises p_pe per event, so any
per-event-constant factor (including the prior's normalisation over [dmin,dmax])
cancels. Only the *shape* of p(dL), set by the cosmology, affects the final
weights. Bounds are therefore stored for provenance but do not change results.
"""
from __future__ import annotations

import numpy as np
from astropy.cosmology import FlatLambdaCDM, Planck15
import astropy.units as u

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# LVK default cosmology used for the O3 cosmo-file reweighting (bilby default).
PLANCK15 = Planck15
# Fallback O4 PE cosmology observed in GWTC-4 configs (used only if a file's
# analytic prior string does not carry an explicit cosmology).
O4_FALLBACK = FlatLambdaCDM(H0=67.9, Om0=0.3065)


def make_cosmology(H0: float, Om0: float) -> FlatLambdaCDM:
    return FlatLambdaCDM(H0=H0, Om0=Om0)


def z_of_dL(dL_mpc, cosmology: FlatLambdaCDM, zmax: float = 10.0, n: int = 4000):
    """Invert dL(z) by monotonic interpolation. dL in Mpc."""
    z = np.expm1(np.linspace(np.log(1.0), np.log(1.0 + zmax), n))
    dl = cosmology.luminosity_distance(z).to(u.Mpc).value
    return np.interp(np.asarray(dL_mpc, dtype=float), dl, z)


def uniform_source_frame_prob(dL_mpc, cosmology: FlatLambdaCDM,
                              dmin: float, dmax: float):
    """p(dL) for a UniformSourceFrame prior. dL in Mpc.

    Prefers bilby's implementation (identical to LVK PE); falls back to an
    astropy computation of the same density.
    """
    dL = np.asarray(dL_mpc, dtype=float)
    try:
        from bilby.gw.prior import UniformSourceFrame
        prior = UniformSourceFrame(
            minimum=float(dmin), maximum=float(dmax),
            cosmology=cosmology, name="luminosity_distance",
            latex_label="$d_L$", unit="Mpc", boundary=None,
        )
        return np.asarray(prior.prob(dL), dtype=float)
    except Exception:
        return _usf_prob_astropy(dL, cosmology, dmin, dmax)


def _usf_prob_astropy(dL, cosmology, dmin, dmax, ngrid: int = 4000):
    """Astropy fallback: p(dL) propto dVc/dz * 1/(1+z) * |dz/ddL|, normalised
    on [dmin, dmax]."""
    c_kms = 299792.458
    H0 = cosmology.H0.value
    Om0 = cosmology.Om0
    dH = c_kms / H0  # Mpc

    z = np.expm1(np.linspace(np.log(1.0), np.log(1.0 + 10.0), ngrid))
    DC = cosmology.comoving_distance(z).to(u.Mpc).value
    E = np.sqrt(Om0 * (1 + z) ** 3 + (1.0 - Om0))
    dDC_dz = dH / E
    dL_grid = (1 + z) * DC
    ddL_dz = DC + (1 + z) * dDC_dz
    # p(z) propto comoving-volume element * time dilation
    pz = (DC ** 2 / E) * (1.0 / (1 + z))
    # change of variables to dL
    p_dL_grid = pz / ddL_dz

    # normalise on [dmin, dmax]
    mask = (dL_grid >= dmin) & (dL_grid <= dmax)
    norm = _trapz(p_dL_grid[mask], dL_grid[mask])
    p = np.interp(np.asarray(dL, float), dL_grid, p_dL_grid, left=0.0, right=0.0)
    out = np.where((dL >= dmin) & (dL <= dmax), p / norm, 0.0)
    return out
