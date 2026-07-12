"""1-D chi_eff prior under isotropic uniform-magnitude spins.

Replaces gwdistributions.distributions.spin.IsotropicUniformMagnitudeChiEffGivenComponentMass.

Physics:
    a_i  ~ Uniform(0, amax)          spin magnitude
    cos θ_i ~ Uniform(-1, 1)          isotropic orientation
    s_iz = a_i cos θ_i               z-component of dimensionless spin
    χ_eff = (m1 s1z + m2 s2z) / (m1 + m2)

The marginal p(s_iz) = −log(|s_iz|/amax) / amax   for |s_iz| < amax
(a log-triangular distribution).  χ_eff is a mass-weighted sum of two such
variables, so its PDF is a convolution of two scaled log-triangulars.

The class precomputes p(χ_eff | q, amax) on a (q, χ_eff) grid and evaluates
via fast bilinear interpolation — O(N) for N samples.

Usage:
    from gwcat.spin import chi_eff_prior_logprob, ChiEffPrior

    # Quick function call (builds table on first use, caches it)
    logp = chi_eff_prior_logprob(chieff, m1_source, m2_source, amax=0.99)

    # Or manage the object explicitly for repeated calls with different amax
    prior = ChiEffPrior(amax=0.99)
    logp = prior.logprob(chieff, m1_source, m2_source)
"""
from __future__ import annotations

import numpy as np

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


class ChiEffPrior:
    """Precomputed 1-D χ_eff prior on a (q, χ_eff) grid.

    Parameters
    ----------
    amax : float
        Maximum dimensionless spin magnitude (default 0.99).
    nq : int
        Number of mass-ratio grid points (q = m1/(m1+m2) ∈ [0.5, 1]).
    nchi : int
        Number of χ_eff grid points.
    ngrid_conv : int
        Internal convolution grid resolution.
    """

    def __init__(self, amax: float = 0.99, nq: int = 200,
                 nchi: int = 2000, ngrid_conv: int = 4000):
        self.amax = float(amax)
        self.q_grid = np.linspace(0.5, 1.0, nq)
        self.chi_grid = np.linspace(-amax, amax, nchi)
        self._ngrid_conv = ngrid_conv

        # Build lookup table: table[i, j] = p(chi_grid[j] | q_grid[i], amax)
        self.table = np.empty((nq, nchi))
        for i, q in enumerate(self.q_grid):
            self.table[i] = self._convolve_at_q(q)

    # ------------------------------------------------------------------
    # Internal: single-spin marginal and convolution
    # ------------------------------------------------------------------
    @staticmethod
    def _single_spin_pdf(s, amax):
        """p(s_iz) = −log(|s|/amax) / amax  for |s| < amax."""
        abs_s = np.abs(s)
        out = np.zeros_like(s)
        eps = 1e-30
        mask = abs_s > eps
        valid = mask & (abs_s < amax)
        out[valid] = -np.log(abs_s[valid] / amax) / amax
        # At |s| ≈ 0: cap at the value at eps (integrable singularity)
        out[~mask] = -np.log(eps / amax) / amax
        return out

    def _convolve_at_q(self, q):
        """Compute p(χ_eff | q) by convolving two scaled single-spin PDFs."""
        amax = self.amax
        ng = self._ngrid_conv
        # Grid for individual X_i = q_i * s_iz; total range is [-amax, amax]
        x = np.linspace(-amax * 1.05, amax * 1.05, ng)
        dx = x[1] - x[0]

        q1, q2 = q, 1.0 - q

        # PDF of X1 = q1 * s1z: p(X1) = (1/q1) * p_s(X1/q1)
        if q1 > 1e-12:
            p1 = self._single_spin_pdf(x / q1, amax) / q1
        else:
            p1 = np.zeros(ng)
            p1[ng // 2] = 1.0 / dx

        if q2 > 1e-12:
            p2 = self._single_spin_pdf(x / q2, amax) / q2
        else:
            p2 = np.zeros(ng)
            p2[ng // 2] = 1.0 / dx

        # Convolve: χ_eff = X1 + X2
        p_conv = np.convolve(p1, p2, mode="full") * dx
        n_conv = len(p_conv)
        x_conv = np.linspace(2 * x[0], 2 * x[-1], n_conv)

        # Normalise on [-amax, amax]
        in_range = (x_conv >= -amax) & (x_conv <= amax)
        norm = _trapz(p_conv[in_range], x_conv[in_range])
        if norm > 0:
            p_conv /= norm

        return np.interp(self.chi_grid, x_conv, p_conv, left=0.0, right=0.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def prob(self, chi_eff, m1, m2):
        """p(χ_eff | m1, m2, amax).  Vectorized over inputs."""
        q = np.asarray(m1, dtype=float) / (np.asarray(m1, dtype=float)
                                           + np.asarray(m2, dtype=float))
        chi = np.asarray(chi_eff, dtype=float)
        scalar = q.ndim == 0 and chi.ndim == 0
        q = np.atleast_1d(q)
        chi = np.atleast_1d(chi)

        p = self._interp2d(q, chi)
        return float(p[0]) if scalar else p

    def logprob(self, chi_eff, m1, m2):
        """log p(χ_eff | m1, m2, amax).  Clipped at −50 for safety."""
        p = self.prob(chi_eff, m1, m2)
        return np.where(p > 0, np.log(p), -50.0)

    def _interp2d(self, q, chi):
        """Bilinear interpolation on the (q_grid, chi_grid) table."""
        nq = len(self.q_grid)
        nchi = len(self.chi_grid)

        # Map q to fractional index
        q_idx = np.interp(q, self.q_grid, np.arange(nq))
        q_lo = np.floor(q_idx).astype(int).clip(0, nq - 2)
        q_hi = q_lo + 1
        q_f = q_idx - q_lo

        # Map chi_eff to fractional index
        chi_idx = np.interp(chi, self.chi_grid, np.arange(nchi))
        chi_lo = np.floor(chi_idx).astype(int).clip(0, nchi - 2)
        chi_hi = chi_lo + 1
        chi_f = chi_idx - chi_lo

        # Bilinear
        v00 = self.table[q_lo, chi_lo]
        v01 = self.table[q_lo, chi_hi]
        v10 = self.table[q_hi, chi_lo]
        v11 = self.table[q_hi, chi_hi]

        v0 = v00 * (1 - chi_f) + v01 * chi_f
        v1 = v10 * (1 - chi_f) + v11 * chi_f
        return v0 * (1 - q_f) + v1 * q_f


# ------------------------------------------------------------------
# Module-level convenience (cached singleton)
# ------------------------------------------------------------------
_CACHE = {}


def chi_eff_prior_logprob(chi_eff, m1_source, m2_source, amax=0.99):
    """log p(χ_eff | m1_source, m2_source, amax).

    Builds a ChiEffPrior on first call for each amax and caches it.
    """
    if amax not in _CACHE:
        _CACHE[amax] = ChiEffPrior(amax=amax)
    return _CACHE[amax].logprob(chi_eff, m1_source, m2_source)


# ==================================================================
# χ_p (effective precessing spin) foundations
# ==================================================================
def chi_p_from_components(a_1, a_2, cos_tilt_1, cos_tilt_2, mass_1, mass_2):
    """Effective precessing spin χ_p from component spins (Schmidt et al. 2015).

    Vectorized (numpy broadcasting).  With the *primary* mass ``mass_1`` (the
    more massive body) and ``q = mass_2 / mass_1 ≤ 1``,

        sin θ_i = sqrt(clip(1 − cos²θ_i, 0, 1))
        χ_p = max( a_1 sin θ_1 ,  q(4q+3)/(4+3q) · a_2 sin θ_2 ).

    Only the *ratio* mass_2/mass_1 enters, so the masses may be supplied in
    either the detector or the source frame — the two give an identical χ_p
    (the redshift factor cancels).  ``mass_1`` is assumed to be the primary
    (mass_1 ≥ mass_2), matching the Schmidt convention.

    .. note::
       **Mass-ratio convention trap.**  This function uses the *component*
       convention ``q = mass_2 / mass_1 ∈ (0, 1]``.  The :class:`ChiEffPrior`
       grids in this module instead use the *primary-mass-fraction*
       convention ``q_grid = m1 / (m1 + m2) ∈ [0.5, 1]`` (its ``q_grid``
       attribute).  The two are related by
       ``q_grid = 1 / (1 + q)``  ⇔  ``q = (1 − q_grid) / q_grid``.
       Do not mix them.

    Parameters
    ----------
    a_1, a_2 : array_like
        Dimensionless spin magnitudes of the primary and secondary.
    cos_tilt_1, cos_tilt_2 : array_like
        Cosines of the spin-tilt angles (aligned-spin fractions).
    mass_1, mass_2 : array_like
        Primary and secondary masses (any frame; only the ratio matters).

    Returns
    -------
    float or ndarray
        χ_p, scalar if all inputs are scalar.
    """
    a_1 = np.asarray(a_1, dtype=float)
    a_2 = np.asarray(a_2, dtype=float)
    cos_tilt_1 = np.asarray(cos_tilt_1, dtype=float)
    cos_tilt_2 = np.asarray(cos_tilt_2, dtype=float)
    mass_1 = np.asarray(mass_1, dtype=float)
    mass_2 = np.asarray(mass_2, dtype=float)

    q = mass_2 / mass_1
    k = q * (4.0 * q + 3.0) / (4.0 + 3.0 * q)
    sin_tilt_1 = np.sqrt(np.clip(1.0 - cos_tilt_1 ** 2, 0.0, 1.0))
    sin_tilt_2 = np.sqrt(np.clip(1.0 - cos_tilt_2 ** 2, 0.0, 1.0))

    chi_p = np.maximum(a_1 * sin_tilt_1, k * a_2 * sin_tilt_2)
    return float(chi_p) if chi_p.ndim == 0 else chi_p
