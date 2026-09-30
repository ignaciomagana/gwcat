"""The isotropic uniform-magnitude chi_eff prior, evaluated exactly (GW-40i).

Physics.  Body ``i`` has ``a_i ~ U(0, amax_i)`` and ``cos theta_i ~ U(-1, 1)``,
so its aligned component ``s_i = a_i cos theta_i`` has the log-triangular density
``p(s) = ln(amax_i/|s|) / (2 amax_i)`` on ``|s| < amax_i``.  With the mass
fractions ``w_i = m_i/(m_1 + m_2)``,

    chi_eff = w_1 s_1 + w_2 s_2 = X + Y,   X in [-A_1, A_1],  A_i = w_i amax_i,

and the prior is the convolution (Callister 2021, arXiv:2104.09508, Sec. 3.1)

    p(chi) = 1/(4 A_1 A_2) * INT ln(A_1/|x|) ln(A_2/|chi - x|) dx          (*)

over ``|x| <= A_1, |chi - x| <= A_2``.  It is symmetric in ``chi`` and
vanishes for ``|chi| >= A_1 + A_2`` (``= amax`` when both bodies share one).

Evaluation.  :class:`gwcat.spin.ChiEffPrior` tabulates (*) on a 200 x 2000
(q, chi_eff) grid and interpolates bilinearly; that is off by up to ~1e-2 in
ln p (4.2e-3 relative on the 259-event PE, median ~2e-4) and the error enters
every exported ``p_pe`` and ``pdraw``.  This module evaluates (*) to ~1e-14 relative instead, from a closed
form assembled so that it never cancels:

* (*) is split at the two log singularities ``x = 0`` and ``x = chi`` into a
  piece left of 0, a piece between 0 and chi, and a piece right of chi.  Each is
  rewritten, by scaling its variable to its own length, as a sum of NON-NEGATIVE
  terms: products of non-negative logarithms and the three one-dimensional
  integrals

      E1(k) = -INT_0^1 ln(1 - k t) dt
            = 1 + (1-k) ln(1-k) / k,
      E2(k) =  INT_0^1 ln(1 - t) ln(1 - k t) dt
            = 2 + ((1-k)/k) [ln(1-k) - Li2(k) - ln^2(1-k)/2],
      E3(m) =  INT_0^1 ln(t) ln(1 - m t) dt
            = 2 + (1-m) ln(1-m)/m - Li2(m)/m,

  each of which is non-negative on [0, 1].  ``Li2`` is scipy's ``spence``
  (``Li2(k) = spence(1 - k)``, and ``1 - k`` is always passed in directly, never
  formed by subtraction).  Below ``k = 1/4`` the closed forms lose digits to
  cancellation, so the power series (40 terms, truncation < 1e-24) is used.
* Near the edge ``|chi| > max(A_1, A_2)`` both singularities lie OUTSIDE the
  integration range and the density falls like ``(amax - |chi|)^3``, where any
  closed form would cancel catastrophically.  There the (analytic) integrand is
  integrated directly by Gauss-Legendre quadrature, geometrically graded toward
  whichever end a singularity sits close to (ratio 1/2 between a sub-interval's
  length and its distance to the singularity, 20 nodes each, so the per-piece
  error is below 1e-23 of that piece).  Every term is again non-negative.
* The geometry -- ``A_i``, ``A_i - |chi|``, ``A_1 - A_2 -/+ |chi|`` and
  ``amax - |chi|`` -- is formed in double-double arithmetic (error-free
  transformations), so a kink location such as ``chi = A_1`` is resolved to
  ~1e-32 absolute.  Without it the result near a kink inherits the 1-ulp
  rounding of ``A_i``, which is 1e-11 relative at ``q = 1e-8``.

Measured against an independent mpmath tanh-sinh evaluation of (*) at 40+ digits
(``tests/fixtures/chi_eff_iso_mpmath_reference.json``): max relative error
3.1e-14 over 3,865 edge-case points (q from 1e-8 to 1, amax 0.05 to 1, chi at
0, 1e-300, every kink +/- 1e-14 ... 1e-3 relative, and down to 1 ulp below the
support edge), 1.2e-14 on 1,200 random physical points and 5e-15 on 300 points
with amax_1 != amax_2.  3.6M evaluations take ~3 s.
"""
from __future__ import annotations

import numpy as np
from scipy.special import spence

__all__ = ["chi_eff_iso_prob", "chi_eff_iso_prob_q", "EXACT_METHOD"]

#: Recorded next to every density this module produced.
EXACT_METHOD = ("closed_form_Li2_nonneg_split+graded_gauss_legendre_edge;"
                "double_double_geometry")

_PI2_6 = np.pi ** 2 / 6.0

# ---- power series of E1, E2, E3 below _SER_T ------------------------------
_SER_N = 40
_SER_T = 0.25
_n = np.arange(1, _SER_N + 1, dtype=float)
_HARM = np.cumsum(1.0 / np.arange(1, _SER_N + 2, dtype=float))   # H_1..H_{N+1}
_C_E1 = 1.0 / (_n * (_n + 1.0))                  # sum k^n / (n(n+1))
_C_E2 = _HARM[1:] / (_n * (_n + 1.0))           # sum k^n H_{n+1} / (n(n+1))
_C_E3 = 1.0 / (_n * (_n + 1.0) ** 2)             # sum m^n / (n (n+1)^2)

# ---- edge quadrature ------------------------------------------------------
_GL_X, _GL_W = np.polynomial.legendre.leggauss(20)
_GRADE = 3.0            # breakpoints e (3^k - 1): length / distance = 2 ... 1/2
_GRADE_CAP = 60         # 3^60 ~ 4e28 -- covers any e a double can express here


# ---------------------------------------------------------------------------
# double-double helpers (Dekker / Knuth error-free transformations)
# ---------------------------------------------------------------------------
def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _split(a):
    c = 134217729.0 * a                     # 2^27 + 1
    hi = c - (c - a)
    return hi, a - hi


def _two_prod(a, b):
    p = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def _dd_add(ah, al, bh, bl):
    s, e = _two_sum(ah, bh)
    return _two_sum(s, e + (al + bl))


def _dd_mul_d(ah, al, b):
    p, e = _two_prod(ah, b)
    return _two_sum(p, e + al * b)


def _dd_div(xh, xl, yh, yl):
    """(xh + xl) / (yh + yl) to double-double precision (three quotient digits)."""
    q1 = xh / yh
    ph, pl = _dd_mul_d(yh, yl, q1)
    rh, rl = _dd_add(xh, xl, -ph, -pl)
    q2 = rh / yh
    ph, pl = _dd_mul_d(yh, yl, q2)
    rh, rl = _dd_add(rh, rl, -ph, -pl)
    q3 = rh / yh
    s, e = _two_sum(q1, q2)
    return _dd_add(s, e, q3, 0.0)


def _dd_to_d(h, l):
    return h + l


# ---------------------------------------------------------------------------
# the one-dimensional pieces
# ---------------------------------------------------------------------------
def _series(x, coef):
    acc = np.zeros_like(x)
    for c in coef[::-1]:
        acc = (acc + c) * x
    return acc


def _xlogx(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x > 0, x * np.log(np.where(x > 0, x, 1.0)), 0.0)


def _E1(k, om):
    """-INT_0^1 ln(1 - k t) dt; ``om = 1 - k`` supplied exactly."""
    out = np.empty_like(k)
    s = k < _SER_T
    out[s] = _series(k[s], _C_E1)
    out[~s] = 1.0 + _xlogx(om[~s]) / k[~s]
    return out


def _E2(k, om):
    """INT_0^1 ln(1 - t) ln(1 - k t) dt; ``om = 1 - k`` supplied exactly."""
    out = np.empty_like(k)
    s = k < _SER_T
    out[s] = _series(k[s], _C_E2)
    o = om[~s]
    with np.errstate(divide="ignore", invalid="ignore"):
        lo = np.log(np.where(o > 0, o, 1.0))
    br = (_xlogx(o) - 0.5 * np.where(o > 0, o * lo * lo, 0.0)
          - o * spence(o))
    out[~s] = 2.0 + br / k[~s]
    return out


def _E3(m, om):
    """INT_0^1 ln(t) ln(1 - m t) dt; ``om = 1 - m`` supplied exactly."""
    out = np.empty_like(m)
    s = m < _SER_T
    out[s] = _series(m[s], _C_E3)
    o = om[~s]
    out[~s] = 2.0 + (_xlogx(o) - spence(o)) / m[~s]
    return out


def _outer_piece(S, a, b, chi):
    """INT_0^S ln(alpha/s) ln(beta/(chi + s)) ds with ``a = ln(alpha/S) >= 0``
    and ``b = ln(beta/(chi + S)) >= 0``:  s = S t gives
    ``S [a b + b + a E1(k) + E2(k)]``, ``k = S/(chi + S)``."""
    den = chi + S
    k = S / den
    om = chi / den
    return S * (a * b + b + a * _E1(k, om) + _E2(k, om))


def _edge_half(eF, eG):
    """INT_0^{1/2} log1p(u/(eF + 1 - u)) log1p((1 - u)/(eG + u)) du.

    The second factor's log singularity sits at ``u = -eG``; the interval is
    graded toward u = 0 at breakpoints ``eG (3^k - 1)``, so every sub-interval
    is at least half its length away from it.  Points sharing a level count
    are evaluated together.
    """
    out = np.zeros_like(eF)
    with np.errstate(divide="ignore", over="ignore"):
        K = np.ceil(np.log1p(0.5 / eG) / np.log(_GRADE))
    K = np.clip(np.nan_to_num(K, nan=1.0, posinf=_GRADE_CAP),
                1, _GRADE_CAP).astype(int)
    for kk in np.unique(K):
        m = K == kk
        steps = np.expm1(np.arange(kk + 1, dtype=float) * np.log(_GRADE))
        with np.errstate(invalid="ignore"):
            bp = np.minimum(eG[m][:, None] * steps[None, :], 0.5)
        bp[:, 0] = 0.0          # also when eG overflowed to inf (inf * 0)
        bp[:, -1] = 0.5
        lo, hi = bp[:, :-1], bp[:, 1:]
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        u = mid[..., None] + half[..., None] * _GL_X
        w = half[..., None] * _GL_W
        eFm = eF[m][:, None, None]
        eGm = eG[m][:, None, None]
        f = np.log1p(u / (eFm + 1.0 - u))
        g = np.log1p((1.0 - u) / (eGm + u))
        out[m] = np.sum(w * f * g, axis=(1, 2))
    return out


# ---------------------------------------------------------------------------
# the density
# ---------------------------------------------------------------------------
def _density(chi, w1h, w1l, w2h, w2l, amax1, amax2, shared):
    chi = np.abs(chi)
    A1h, A1l = _dd_mul_d(w1h, w1l, amax1)
    A2h, A2l = _dd_mul_d(w2h, w2l, amax2)
    A1 = _dd_to_d(A1h, A1l)
    A2 = _dd_to_d(A2h, A2l)
    zero = np.zeros_like(chi)
    dA1 = _dd_to_d(*_dd_add(A1h, A1l, -chi, zero))          # A1 - chi
    dA2 = _dd_to_d(*_dd_add(A2h, A2l, -chi, zero))          # A2 - chi
    Kh, Kl = _dd_add(A1h, A1l, -A2h, -A2l)                   # A1 - A2
    Dm = _dd_to_d(*_dd_add(Kh, Kl, -chi, zero))              # A1 - A2 - chi
    Sp = _dd_to_d(*_dd_add(Kh, Kl, chi, zero))               # A1 - A2 + chi
    if shared:
        edge = amax1 - chi                     # exact (Sterbenz) near the edge
    else:
        Th, Tl = _dd_add(A1h, A1l, A2h, A2l)
        edge = _dd_to_d(*_dd_add(Th, Tl, -chi, zero))       # A1 + A2 - chi

    tot = np.zeros_like(chi)
    inside = (edge > 0) & (A1 > 0) & (A2 > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        # x < 0 (exists when chi < A2): s = -x on [0, min(A1, A2 - chi)]
        m = inside & (dA2 > 0)
        if m.any():
            capped = Sp[m] <= 0.0                              # A1 <= A2 - chi
            S = np.where(capped, A1[m], dA2[m])
            a = np.where(capped, 0.0, np.log1p(Sp[m] / dA2[m]))
            b = np.where(capped, -np.log1p(Sp[m] / A2[m]), 0.0)
            tot[m] += _outer_piece(S, a, b, chi[m])
        # x > chi (exists when chi < A1): s = x - chi on [0, min(A1 - chi, A2)]
        m = inside & (dA1 > 0)
        if m.any():
            capped = Dm[m] >= 0.0                              # A2 <= A1 - chi
            S = np.where(capped, A2[m], dA1[m])
            a = np.where(capped, 0.0, np.log1p(-Dm[m] / dA1[m]))
            b = np.where(capped, np.log1p(Dm[m] / (chi[m] + A2[m])), 0.0)
            tot[m] += _outer_piece(S, a, b, chi[m])
        # 0 < x < chi, whole interval (chi <= A1 and chi <= A2)
        m = inside & (dA1 >= 0) & (dA2 >= 0) & (chi > 0)
        if m.any():
            c = chi[m]
            l1 = np.log1p(dA1[m] / c)
            l2 = np.log1p(dA2[m] / c)
            tot[m] += c * ((l1 + 1.0) * (l2 + 1.0) + 1.0 - _PI2_6)
        # chi - A2 < x < chi  (A2 < chi <= A1)
        m = inside & (dA1 >= 0) & (dA2 < 0)
        if m.any():
            c = chi[m]
            tot[m] += A2[m] * (np.log1p(dA1[m] / c)
                               + _E3(A2[m] / c, -dA2[m] / c))
        # 0 < x < A1  (A1 < chi <= A2)
        m = inside & (dA1 < 0) & (dA2 >= 0)
        if m.any():
            c = chi[m]
            tot[m] += A1[m] * (np.log1p(dA2[m] / c)
                               + _E3(A1[m] / c, -dA1[m] / c))
        # chi - A2 < x < A1  (chi > max(A1, A2)): the edge, by graded quadrature
        m = inside & (dA1 < 0) & (dA2 < 0)
        if m.any():
            d = edge[m]
            eF = -dA2[m] / d
            eG = -dA1[m] / d
            tot[m] = d * (_edge_half(eF, eG) + _edge_half(eG, eF))

    out = np.zeros_like(chi)
    out[inside] = tot[inside] / (4.0 * A1[inside] * A2[inside])
    return out


def _prepare(chi_eff, *arrays):
    chi = np.asarray(chi_eff, dtype=float)
    arrs = [np.asarray(a, dtype=float) for a in arrays]
    shape = np.broadcast_shapes(chi.shape, *[a.shape for a in arrs])
    chi = np.ascontiguousarray(np.broadcast_to(chi, shape), dtype=float).ravel()
    arrs = [np.ascontiguousarray(np.broadcast_to(a, shape), dtype=float).ravel()
            for a in arrs]
    return shape, chi, arrs


def chi_eff_iso_prob(chi_eff, m1, m2, amax_1=0.99, amax_2=None):
    """Exact ``p(chi_eff | m1, m2)`` for ``a_i ~ U(0, amax_i)``, isotropic tilts.

    Body 1 (mass ``m1``) carries ``amax_1`` and body 2 carries ``amax_2``
    (default: ``amax_1``); the masses need not be ordered and may be in any
    frame (only ``m_i/(m1 + m2)`` enters).  Vectorised with numpy broadcasting.
    Returns 0 outside the support ``|chi_eff| >= w1 amax_1 + w2 amax_2`` and NaN
    where a mass is not finite and positive.
    """
    shared = amax_2 is None
    a2_in = amax_1 if shared else amax_2
    shape, chi, (m1a, m2a, a1, a2) = _prepare(chi_eff, m1, m2, amax_1, a2_in)
    bad = ~(np.isfinite(m1a) & np.isfinite(m2a) & (m1a > 0) & (m2a > 0))
    m1s = np.where(bad, 1.0, m1a)
    m2s = np.where(bad, 1.0, m2a)
    sh, sl = _two_sum(m1s, m2s)
    zero = np.zeros_like(chi)
    w1h, w1l = _dd_div(m1s, zero, sh, sl)
    w2h, w2l = _dd_div(m2s, zero, sh, sl)
    p = _density(chi, w1h, w1l, w2h, w2l, a1, a2, shared)
    p = np.where(bad, np.nan, p)
    p = p.reshape(shape)
    return float(p) if p.ndim == 0 else p


def chi_eff_iso_prob_q(chi_eff, q, amax_1=0.99, amax_2=None):
    """As :func:`chi_eff_iso_prob` with the mass ratio ``q = m2/m1`` (any q > 0).

    ``w1 = 1/(1+q)`` and ``w2 = q/(1+q)`` are formed in double-double, so this
    is the density at exactly the double ``q`` given.
    """
    shared = amax_2 is None
    a2_in = amax_1 if shared else amax_2
    shape, chi, (qa, a1, a2) = _prepare(chi_eff, q, amax_1, a2_in)
    bad = ~(np.isfinite(qa) & (qa > 0))
    qs = np.where(bad, 1.0, qa)
    one = np.ones_like(qs)
    zero = np.zeros_like(qs)
    sh, sl = _two_sum(one, qs)
    w1h, w1l = _dd_div(one, zero, sh, sl)
    w2h, w2l = _dd_div(qs, zero, sh, sl)
    p = _density(chi, w1h, w1l, w2h, w2l, a1, a2, shared)
    p = np.where(bad, np.nan, p)
    p = p.reshape(shape)
    return float(p) if p.ndim == 0 else p
