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
        Maximum dimensionless spin magnitude of the PRIMARY (default 0.99).
    amax_2 : float, optional
        Maximum spin magnitude of the SECONDARY.  Defaults to ``amax`` (the
        historical single-amax behaviour).  A restricted low-spin secondary is
        a real configuration -- GWTC-3 NSBH runs use ``a_2 ~ U(0, 0.05)`` where
        the primary keeps ``U(0, 0.99)`` -- and forcing one shared ``amax``
        inflated the secondary's prior support ~20x (GW-04).
    nq : int
        Number of mass-ratio grid points (q = m1/(m1+m2) ∈ [0.5, 1]).
    nchi : int
        Number of χ_eff grid points.
    ngrid_conv : int
        Internal convolution grid resolution.

    Notes
    -----
    The support of χ_eff is ``|χ_eff| ≤ max(amax_1, amax_2)``: the primary term
    ``q1·s1z`` reaches ``q1·amax_1`` and the secondary ``q2·s2z`` reaches
    ``q2·amax_2``, and with ``q1 + q2 = 1`` the sum is bounded by the larger of
    the two.  :attr:`amax` therefore reports that bound, while ``amax_1`` and
    ``amax_2`` record the per-body priors the density was built from.
    """

    def __init__(self, amax: float = 0.99, nq: int = 200,
                 nchi: int = 2000, ngrid_conv: int = 4000,
                 amax_2: float = None):
        self.amax_1 = float(amax)
        self.amax_2 = self.amax_1 if amax_2 is None else float(amax_2)
        #: The χ_eff support bound, max(amax_1, amax_2).
        self.amax = max(self.amax_1, self.amax_2)
        self.q_grid = np.linspace(0.5, 1.0, nq)
        self.chi_grid = np.linspace(-self.amax, self.amax, nchi)
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
        """p(s_iz) = −ln(|s|/amax) / (2·amax)  for |s| < amax.

        The aligned-spin component of a body with magnitude ``a ~ U(0, amax)``
        and isotropic orientation has ``s = a·cosθ`` distributed over
        ``[−amax, amax]``, so the normalisation carries a factor ½ that this
        function omitted (GW-08/GW-03): ``∫ −ln(|s|/amax) ds = 2·amax`` over the
        full range, not ``amax``.  Both in-tree consumers renormalise their
        convolution afterwards, so the omission was harmless there -- but it is
        live the moment this is reused standalone, e.g. for an aligned-spin
        block, and a density that is wrong by 2x in a denominator is not
        something to leave sitting in a shared module.
        """
        abs_s = np.abs(s)
        out = np.zeros_like(s)
        eps = 1e-30
        mask = abs_s > eps
        valid = mask & (abs_s < amax)
        out[valid] = -np.log(abs_s[valid] / amax) / (2.0 * amax)
        # At |s| ≈ 0: cap at the value at eps (integrable singularity)
        out[~mask] = -np.log(eps / amax) / (2.0 * amax)
        return out

    def _convolve_at_q(self, q):
        """Compute p(χ_eff | q) by convolving two scaled single-spin PDFs.

        Each body uses ITS OWN ``amax`` (GW-04); when the two are equal this is
        bit-identical to the previous single-amax construction.
        """
        amax = self.amax
        ng = self._ngrid_conv
        # Grid for individual X_i = q_i * s_iz; total range is [-amax, amax]
        x = np.linspace(-amax * 1.05, amax * 1.05, ng)
        dx = x[1] - x[0]

        q1, q2 = q, 1.0 - q

        # PDF of X1 = q1 * s1z: p(X1) = (1/q1) * p_s(X1/q1; amax_1)
        if q1 > 1e-12:
            p1 = self._single_spin_pdf(x / q1, self.amax_1) / q1
        else:
            p1 = np.zeros(ng)
            p1[ng // 2] = 1.0 / dx

        if q2 > 1e-12:
            p2 = self._single_spin_pdf(x / q2, self.amax_2) / q2
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
        """p(χ_eff | m1, m2, amax).  Vectorized over inputs.

        The grid is tabulated in the primary-mass fraction
        ``q = m_primary/(m1+m2) ∈ [0.5, 1]``, so an input with ``m1 < m2`` is
        mapped by the exact reflection ``q → 1 − q`` (GW-08).  Previously such an
        input produced ``q < 0.5``, which ``_interp2d``'s ``np.interp`` silently
        CLAMPED to the ``q = 0.5`` edge -- an equal-mass density for an unequal-
        mass system.  The reflection is exact because ``χ_eff`` is symmetric
        under relabelling the two bodies when they share one ``amax``.

        With ``amax_1 != amax_2`` that symmetry is broken -- the bodies have
        genuinely different priors -- so the reflection is no longer valid and
        ``m1`` must actually be the primary.  Passing them the other way round
        raises rather than silently evaluating the wrong body's prior (GW-04).
        """
        m1a = np.asarray(m1, dtype=float)
        m2a = np.asarray(m2, dtype=float)
        if self.amax_1 != self.amax_2:
            bad = np.asarray(m2a > m1a)
            if bad.any():
                raise ValueError(
                    f"ChiEffPrior.prob: amax_1={self.amax_1} != "
                    f"amax_2={self.amax_2}, so the two bodies have different "
                    f"spin priors and chi_eff is NOT symmetric under swapping "
                    f"them. {int(np.sum(bad))} input(s) have m2 > m1; pass the "
                    f"more massive body as m1.")
        q = np.maximum(m1a, m2a) / (m1a + m2a)
        chi = np.asarray(chi_eff, dtype=float)
        scalar = q.ndim == 0 and chi.ndim == 0
        q = np.atleast_1d(q)
        chi = np.atleast_1d(chi)

        p = self._interp2d(q, chi)
        return float(p[0]) if scalar else p

    def support(self, chi_eff):
        """Whether each ``chi_eff`` is inside the prior's support, ``|χ| ≤ amax``.

        An explicit predicate, so a caller can count and report out-of-support
        samples instead of discovering them as a ``-inf`` (GW-03).
        """
        chi = np.abs(np.asarray(chi_eff, dtype=float))
        return np.asarray(chi <= self.amax)

    def logprob(self, chi_eff, m1, m2):
        """log p(χ_eff | m1, m2, amax).  ``-inf`` outside the support.

        This used to return the sentinel ``-50``, i.e. a density of ``2e-22``
        rather than zero.  ``p_pe``/``pdraw`` are DENOMINATORS, so that floor
        turned an impossible sample into one carrying ~1e21 times the median
        weight and collapsing the event's Monte-Carlo integral to a single
        sample.  Out of support is now exactly zero density (GW-03); callers
        must count those samples rather than clip them back up.

        CAVEAT on the support boundary: this method applies no explicit
        ``|chi_eff| > amax`` mask of its own.  The grid interpolation clamps a
        beyond-``amax`` query to the boundary column, whose tabulated density
        is *positive at ~1e-12* for many q rows, so ``logprob`` there returns
        ~``-25``, not ``-inf`` -- "-inf outside the support" holds only where
        the interpolated density is exactly zero.  This is by design: callers
        are expected to gate on :meth:`support` first (as the export builders
        do), which is what makes out-of-support counts reportable rather than
        discovered as infinities.  darksirens' bit-for-bit port of this class
        pins the current behavior, so do not "fix" the clamp here without
        coordinating a paired change there (DS-06).
        """
        p = self.prob(chi_eff, m1, m2)
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = np.where(np.asarray(p) > 0, np.log(p), -np.inf)
        # NaN density (rather than zero) is also "no support", not a small number.
        return np.where(np.isnan(np.asarray(p, dtype=float)), -np.inf, logp)

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

    # Enforce the Schmidt convention rather than silently computing k > 1
    # (GW-08).  The formula is only defined with mass_1 the PRIMARY: for
    # mass_2 > mass_1 the coefficient q(4q+3)/(4+3q) exceeds 1 and the "max"
    # picks the wrong branch, i.e. it returns a number that is not chi_p.
    # Verified on the production store: 0 of 6.87M samples have m2 > m1, so this
    # only ever catches a caller that mixed up the two columns.
    bad = np.asarray(mass_2 > mass_1)
    if bad.any():
        n_bad = int(bad.sum())
        worst = float(np.max(np.asarray(mass_2 / mass_1)[bad]))
        raise ValueError(
            f"chi_p_from_components: mass_1 must be the primary (mass_1 >= "
            f"mass_2), but {n_bad} sample(s) have mass_2 > mass_1 (worst "
            f"mass_2/mass_1 = {worst:.6g}). Swap the (mass, a, cos_tilt) pairs "
            f"so the more massive body is body 1; do not pass them unsorted, "
            f"because q > 1 makes the Schmidt coefficient exceed 1 and the "
            f"result is not chi_p.")

    q = mass_2 / mass_1
    k = q * (4.0 * q + 3.0) / (4.0 + 3.0 * q)
    sin_tilt_1 = np.sqrt(np.clip(1.0 - cos_tilt_1 ** 2, 0.0, 1.0))
    sin_tilt_2 = np.sqrt(np.clip(1.0 - cos_tilt_2 ** 2, 0.0, 1.0))

    chi_p = np.maximum(a_1 * sin_tilt_1, k * a_2 * sin_tilt_2)
    return float(chi_p) if chi_p.ndim == 0 else chi_p


def component_spin_prior_lnpdf(a_1, a_2, cos_tilt_1, cos_tilt_2,
                               amax_1, amax_2):
    """ln p of the standard component-spin PE prior.

    The default LVK component-spin prior takes the magnitudes uniform and the
    orientations isotropic, independently for each body,

        a_i ~ Uniform(0, amax_i),   cos θ_i ~ Uniform(−1, 1),

    so the joint density is constant inside the box and zero outside,

        ln p = −ln(4 · amax_1 · amax_2)   for 0 ≤ a_i ≤ amax_i, |cos θ_i| ≤ 1
             = −∞                          otherwise.

    Vectorized (numpy broadcasting).  ``amax_1``/``amax_2`` may be scalars or
    per-sample arrays.

    Parameters
    ----------
    a_1, a_2 : array_like
        Spin magnitudes.
    cos_tilt_1, cos_tilt_2 : array_like
        Cosines of the tilt angles.
    amax_1, amax_2 : array_like
        Maximum spin magnitudes (scalar or per-sample).

    Returns
    -------
    float or ndarray
        ln p, scalar if all inputs are scalar.  Out-of-support points are
        ``-inf`` (zero density).
    """
    a_1 = np.asarray(a_1, dtype=float)
    a_2 = np.asarray(a_2, dtype=float)
    cos_tilt_1 = np.asarray(cos_tilt_1, dtype=float)
    cos_tilt_2 = np.asarray(cos_tilt_2, dtype=float)
    amax_1 = np.asarray(amax_1, dtype=float)
    amax_2 = np.asarray(amax_2, dtype=float)

    in_box = ((a_1 >= 0.0) & (a_1 <= amax_1)
              & (a_2 >= 0.0) & (a_2 <= amax_2)
              & (np.abs(cos_tilt_1) <= 1.0) & (np.abs(cos_tilt_2) <= 1.0))
    with np.errstate(divide="ignore"):
        lnnorm = -np.log(4.0 * amax_1 * amax_2)
    lnp = np.where(in_box, lnnorm, -np.inf)
    lnp = np.asarray(lnp, dtype=float)
    return float(lnp) if lnp.ndim == 0 else lnp


class ChiEffChiPPrior:
    """Joint prior p(χ_eff, χ_p | q, amax) under isotropic uniform-magnitude spins.

    Both component spins are drawn from a_i ~ Uniform(0, amax) with isotropic
    orientation (cos θ_i ~ Uniform(−1, 1)), sharing a single ``amax``.  The
    joint factorises as

        p(χ_eff, χ_p | q, amax) = p(χ_eff | q, amax) · p(χ_p | χ_eff, q, amax),

    where the *marginal* p(χ_eff | q, amax) is delegated to the existing
    :class:`ChiEffPrior` (reused verbatim) and the *conditional*
    p(χ_p | χ_eff, q, amax) is built semi-analytically here.

    Construction (Callister arXiv:2104.09508; Callister et al. arXiv:2106.00521,
    appendix).  With aligned components s_iz = a_i cos θ_i, in-plane magnitudes
    s_ip = a_i sin θ_i ≥ 0, q = m2/m1 ≤ 1 and k = q(4q+3)/(4+3q):

        χ_eff = (s_1z + q s_2z)/(1+q),   χ_p = max(s_1p, k s_2p).

    * Single-spin aligned marginal:  p_z(s) = −ln(|s|/amax)/(2 amax) on
      [−amax, amax] (reused from :meth:`ChiEffPrior._single_spin_pdf`, up to a
      constant that cancels in the weight normalisation below).
    * Conditional in-plane CDF/pdf given s_z (:meth:`_inplane_cdf_pdf`):
          F(x|s_z) = ln(1 + x²/s_z²) / ln(amax²/s_z²)   for 0 ≤ x ≤ √(amax²−s_z²)
          F = 1 above;  f = dF/dx = (2x/(s_z²+x²)) / ln(amax²/s_z²).
    * Conditioned on χ_eff, s_2z(s_1z) = ((1+q)χ_eff − s_1z)/q, so
          w(s_1z | χ_eff) ∝ p_z(s_1z) p_z(s_2z(s_1z))
      restricted to s_1z AND s_2z both in [−amax, amax] and normalised by its
      own 1-D integral (the 1/q Jacobian and the p_z prefactors cancel).
    * Since s_1p, s_2p are conditionally independent given (s_1z, s_2z),
          p(χ_p = x | χ_eff) = ∫ ds_1z w(s_1z|χ_eff) ·
              [ f(x|s_1z) F(x/k|s_2z) + F(x|s_1z) f(x/k|s_2z)/k ].
      χ_p has support [0, amax] (k ≤ 1).

    Design decision — **direct evaluation, no precomputed grid.**  Two routes
    were prototyped:

    * *grid + trilinear interpolation* (as :class:`ChiEffPrior` uses in 2-D):
      fast to evaluate but rejected — the conditional has a razor-sharp
      χ_eff→0, χ_p→0, q→1 corner (where the two weight singularities merge)
      that linear interpolation cannot resolve; even a 56×64×128 table left
      >40 % error there.
    * *direct quadrature* (chosen): the s_1z integral is done with fixed-order
      Gauss–Legendre, breaking the interval at every point where the integrand
      loses smoothness — the two weight log-singularities (s_1z = 0, s_2z = 0),
      the four in-plane-cap kinks, and a short geometric ladder of break-points
      laid *toward* each log-singularity so the (up to squared-log) endpoint is
      resolved even in that corner.  With the defaults below this holds the
      pointwise density to ≲1 % relative error wherever it exceeds ~1 % of its
      peak (≲0.1 % at ``order=32``), verified against a 2·10⁷-sample
      brute-force reference.

    Evaluation is vectorised (points × quad-nodes) and internally chunked to
    bound memory.  It is accurate and robust rather than fast: a bulk call with
    ~1e6 evaluation points takes ~1 minute (the density is transcendental-op
    bound), which is why the fast interpolated route was tempting but is
    inaccurate here.  Typical use (per-event PE resampling, ≲1e4 points, or the
    grid evaluations in the tests) is sub-second.

    Parameters
    ----------
    amax : float
        Maximum dimensionless spin magnitude (default 0.99), shared by both
        component spins.
    order : int
        Gauss–Legendre order per sub-interval (default 24; ~0.8 % worst-case
        density error, sub-percent almost everywhere.  Raise to 32 for ≲0.1 %).
    grade_levels : int
        Number of geometric break-points laid toward each log-singularity
        (default 3).  ``grade_ratio ** j`` for j = 0 … grade_levels−1.
    grade_ratio : float
        Geometric contraction ratio of that ladder (default 0.2).
    """

    def __init__(self, amax: float = 0.99, order: int = 24,
                 grade_levels: int = 3, grade_ratio: float = 0.2):
        self.amax = float(amax)
        self.order = int(order)
        self.grade_levels = int(grade_levels)
        self.grade_ratio = float(grade_ratio)
        # Reuse the existing marginal prior verbatim.
        self._chi_eff_prior = ChiEffPrior(amax=amax)
        self._gl_nodes, self._gl_weights = \
            np.polynomial.legendre.leggauss(self.order)
        # Number of sub-intervals: 7 fixed break-points (endpoints, the two
        # log-singularities, the four cap kinks) plus 4 graded points per
        # ladder level -> (8 + 4*grade_levels) break-points.
        self._n_sub = 7 + 4 * self.grade_levels
        # Cap on (points × nodes) processed per chunk.  Kept modest so the
        # transient [block, nodes] arrays stay cache-resident (larger blocks
        # measured *slower* from cache thrashing) while bounding memory.
        self._chunk_elems = 1_000_000

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _k_factor(q):
        """k = q(4q+3)/(4+3q); the Schmidt secondary-spin weighting (k ≤ 1)."""
        return q * (4.0 * q + 3.0) / (4.0 + 3.0 * q)

    @staticmethod
    def _inplane_cdf_pdf(x, s_z, amax):
        """Conditional in-plane CDF F(x|s_z) and pdf f(x|s_z).

        For a single isotropic uniform-magnitude spin, the in-plane magnitude
        s_p conditioned on the aligned component s_z has

            F(x|s_z) = ln(1 + x²/s_z²)/ln(amax²/s_z²)  for 0 ≤ x ≤ √(amax²−s_z²)
            F = 1 above,  f = dF/dx = (2x/(s_z²+x²))/ln(amax²/s_z²).

        Returns ``(F, f)`` broadcast to the shape of ``x`` and ``s_z``.  Values
        are clamped so x ≤ 0 gives (0, 0) and x beyond √(amax²−s_z²) gives
        (1, 0); s_z is floored away from 0 for numerical safety (the quadrature
        never evaluates exactly at the s_z = 0 split point).
        """
        x = np.asarray(x, dtype=float)
        s_z = np.asarray(s_z, dtype=float)
        s2 = np.maximum(s_z * s_z, 1e-300)
        x2 = x * x
        log_denom = np.log(amax * amax / s2)          # = ln(amax²/s_z²) ≥ 0
        log_denom = np.where(log_denom > 1e-300, log_denom, 1e-300)
        F = np.log1p(x2 / s2) / log_denom
        f = (2.0 * x / (s2 + x2)) / log_denom
        over = x2 >= (amax * amax - s2)               # x beyond max in-plane
        F = np.where(over, 1.0, F)
        f = np.where(over, 0.0, f)
        nonpos = x <= 0.0
        F = np.where(nonpos, 0.0, F)
        f = np.where(nonpos, 0.0, f)
        return F, f

    def _cond_prob_block(self, chi_p, chi_eff, q):
        """p(χ_p | χ_eff, q, amax) for a single 1-D block (all same length)."""
        amax = self.amax
        nodes01 = self._gl_nodes
        w01 = self._gl_weights
        n = chi_p.shape[0]

        c = (1.0 + q) * chi_eff                        # s_1z where s_2z = 0
        k = np.maximum(self._k_factor(q), 1e-12)

        # Valid s_1z interval: both s_1z and s_2z(s_1z) in [-amax, amax].
        lo = np.maximum(-amax, c - q * amax)
        hi = np.minimum(amax, c + q * amax)
        valid = hi > lo

        # Break the integral at every point where the integrand loses
        # smoothness, so fixed-order Gauss-Legendre converges cleanly:
        #   * the two integrable log-singularities of the weight w(s_1z)
        #     (s_1z = 0 and s_2z = 0, i.e. s_1z = c);
        #   * the C0 kinks where an in-plane cap turns on, i.e. where
        #     χ_p = √(amax²−s_1z²)  (spin 1)  and  χ_p/k = √(amax²−s_2z²)
        #     (spin 2), giving s_1z = ±s1★ and s_1z = c ∓ q·s2★.
        # In addition, lay a short geometric ladder of break-points *toward*
        # each log-singularity (scales χ_p near s_1z=0 and q·χ_p/k near s_1z=c,
        # contracted by grade_ratio**j).  This resolves the narrow in-plane
        # peak sitting on the (possibly squared-) log singularity in the
        # χ_eff→0, χ_p→0, q→1 corner where the two singularities merge.
        # All candidates are clamped to [lo, hi] and sorted; coincident /
        # out-of-range ones collapse to zero-width sub-intervals (weight 0).
        s1_star = np.sqrt(np.clip(amax * amax - chi_p * chi_p, 0.0, None))
        s2_star = np.sqrt(np.clip(amax * amax - (chi_p / k) ** 2, 0.0, None))
        cand = [
            lo, hi,
            np.zeros_like(c),          # s_1z = 0        (weight singularity)
            c,                          # s_2z = 0        (weight singularity)
            s1_star, -s1_star,          # spin-1 cap kinks
            c - q * s2_star, c + q * s2_star,   # spin-2 cap kinks
        ]
        scale1 = chi_p                                  # in-plane peak width
        scale2 = q * chi_p / k
        for j in range(self.grade_levels):
            g1 = scale1 * (self.grade_ratio ** j)
            g2 = scale2 * (self.grade_ratio ** j)
            cand += [g1, -g1, c + g2, c - g2]
        cand = np.stack(cand, axis=1)                   # [n, 8 + 4*grade_levels]
        bounds = np.sort(np.clip(cand, lo[:, None], hi[:, None]), axis=1)
        n_sub = bounds.shape[1] - 1

        # Gauss-Legendre nodes over each sub-interval.
        nodes = []
        effw = []
        for j in range(n_sub):
            a = bounds[:, j]
            b = bounds[:, j + 1]
            mid = 0.5 * (a + b)
            half = 0.5 * (b - a)                        # 0 for empty intervals
            nodes.append(mid[:, None] + half[:, None] * nodes01[None, :])
            effw.append(half[:, None] * w01[None, :])
        s1z = np.concatenate(nodes, axis=1)            # [n, M]
        ew = np.concatenate(effw, axis=1)              # [n, M]
        s2z = (c[:, None] - s1z) / q[:, None]

        # Weight (unnormalised): p_z(s_1z) p_z(s_2z).  The overall constant in
        # _single_spin_pdf (missing 1/2 factor) cancels against the norm below.
        w = (ChiEffPrior._single_spin_pdf(s1z, amax)
             * ChiEffPrior._single_spin_pdf(s2z, amax))

        F1, f1 = self._inplane_cdf_pdf(chi_p[:, None], s1z, amax)
        F2, f2 = self._inplane_cdf_pdf(chi_p[:, None] / k[:, None], s2z, amax)
        integrand = w * (f1 * F2 + F1 * f2 / k[:, None])

        num = np.sum(ew * integrand, axis=1)
        den = np.sum(ew * w, axis=1)                   # normalises w(s_1z)

        out = np.zeros(n)
        good = valid & (den > 0.0)
        out[good] = num[good] / den[good]
        return np.where(out > 0.0, out, 0.0)           # guard tiny negatives

    def cond_prob_chi_p(self, chi_p, chi_eff, q):
        """p(χ_p | χ_eff, q, amax).  Vectorized; chunked internally.

        ``q`` here is the *component* mass ratio m2/m1 ∈ (0, 1] (see the
        convention note on :func:`chi_p_from_components`).
        """
        chi_p = np.asarray(chi_p, dtype=float)
        chi_eff = np.asarray(chi_eff, dtype=float)
        q = np.asarray(q, dtype=float)
        scalar = chi_p.ndim == 0 and chi_eff.ndim == 0 and q.ndim == 0
        chi_p, chi_eff, q = np.broadcast_arrays(chi_p, chi_eff, q)
        shape = chi_p.shape
        cp = np.ascontiguousarray(chi_p).ravel()
        ce = np.ascontiguousarray(chi_eff).ravel()
        qq = np.ascontiguousarray(q).ravel()

        n = cp.size
        m = self._n_sub * self.order
        block = max(1, self._chunk_elems // m)
        out = np.empty(n)
        for i in range(0, n, block):
            sl = slice(i, i + block)
            out[sl] = self._cond_prob_block(cp[sl], ce[sl], qq[sl])
        out = out.reshape(shape)
        return float(out) if scalar else out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def prob(self, chi_eff, chi_p, m1, m2):
        """p(χ_eff, χ_p | m1, m2, amax).  Vectorized over inputs.

        Masses may be given in any frame and in any order; the more massive
        body is treated as the primary and ``q = m_secondary / m_primary`` is
        used for the χ_p conditional, while the same primary/secondary
        assignment is passed to the reused :class:`ChiEffPrior` marginal.
        """
        chi_eff = np.asarray(chi_eff, dtype=float)
        chi_p = np.asarray(chi_p, dtype=float)
        m1 = np.asarray(m1, dtype=float)
        m2 = np.asarray(m2, dtype=float)
        scalar = (chi_eff.ndim == 0 and chi_p.ndim == 0
                  and m1.ndim == 0 and m2.ndim == 0)

        m_hi = np.maximum(m1, m2)
        m_lo = np.minimum(m1, m2)
        q = m_lo / m_hi

        p_marg = np.asarray(self._chi_eff_prior.prob(chi_eff, m_hi, m_lo),
                            dtype=float)
        p_cond = np.asarray(self.cond_prob_chi_p(chi_p, chi_eff, q),
                            dtype=float)
        p = p_marg * p_cond
        return float(p) if scalar else p

    def support(self, chi_eff, chi_p):
        """Whether each ``(χ_eff, χ_p)`` is inside the joint prior's BOX.

        ``|χ_eff| ≤ amax`` and ``0 ≤ χ_p ≤ amax``.  This is the predicate that
        matters most in practice: χ_p reaches the ceiling on real data where
        χ_eff does not, which is why the floored samples were concentrated there.

        .. note::
           This is a **necessary, not sufficient** condition.  The true support is
           the set of ``(χ_eff, χ_p)`` reachable by some valid
           ``(a_i, cos θ_i)`` configuration, and that region is not a box: it
           pinches in near the ``|χ_eff| → amax`` corners, and the *density*
           vanishes at ``χ_p = 0`` (a continuous density may legitimately be zero
           on a boundary).  So ``logprob`` can be ``-inf`` at a point this
           predicate accepts.  ``isfinite(logprob) ⊆ support`` always holds; the
           converse does not.

           One consequence worth knowing: an aligned-spin sample set has
           ``χ_p ≡ 0``, so every one of its samples gets zero density under this
           joint prior.  That is the right answer -- a precessing-spin prior does
           not describe an aligned-spin run -- and GW-07 stopped such a run being
           ingested as a preferred sample set in the first place.
        """
        chi = np.abs(np.asarray(chi_eff, dtype=float))
        chip = np.asarray(chi_p, dtype=float)
        return np.asarray((chi <= self.amax) & (chip >= 0.0)
                          & (chip <= self.amax))

    def logprob(self, chi_eff, chi_p, m1, m2):
        """log p(χ_eff, χ_p | m1, m2, amax).  ``-inf`` outside the support.

        See :meth:`ChiEffPrior.logprob` for why the old ``-50`` sentinel was
        actively harmful rather than merely approximate (GW-03).
        """
        p = self.prob(chi_eff, chi_p, m1, m2)
        p_arr = np.asarray(p, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = np.where(p_arr > 0.0, np.log(p_arr), -np.inf)
        # A NaN density means no support either, not a small number.
        logp = np.where(np.isnan(p_arr), -np.inf, logp)
        return float(logp) if np.ndim(p) == 0 else logp


# Cached singletons keyed by amax, mirroring chi_eff_prior_logprob.
_CHIP_CACHE = {}


def chi_eff_chi_p_prior_logprob(chi_eff, chi_p, m1_source, m2_source,
                                amax=0.99):
    """log p(χ_eff, χ_p | m1_source, m2_source, amax).

    Builds a :class:`ChiEffChiPPrior` on first call for each amax and caches
    it (like :func:`chi_eff_prior_logprob`).
    """
    if amax not in _CHIP_CACHE:
        _CHIP_CACHE[amax] = ChiEffChiPPrior(amax=amax)
    return _CHIP_CACHE[amax].logprob(chi_eff, chi_p, m1_source, m2_source)
