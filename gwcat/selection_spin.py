"""Exact component-basis spin math for the LVK selection readers.

This module isolates the format-specific spin conversions used by
:mod:`gwcat.selection`, so that :class:`~gwcat.selection.SelectionSet` can stay
readable while every formula here carries its derivation sketch.  Nothing in
this module mutates the legacy ``_pdraw`` computation: it only produces the
*additive* component-basis spin state introduced in PR4.

Target variable set
-------------------
We express draw densities over

    (m1_source, m2_source, z, a1, a2, cosθ1, cosθ2)

i.e. per-spin magnitude ``a_i`` and tilt cosine ``cosθ_i``, with both spin
azimuths marginalised out.  The three building-block Jacobians used below are:

* **spherical volume element** — for a spin drawn in cartesian components,
  ``d³s = a² sinθ · da dθ dφ`` so ``p(a,θ,φ) = p_cart(s⃗) · a²·sinθ``;
* **uniform-azimuth marginalisation** — every LVK campaign draws φ ~ U[0,2π),
  so integrating a φ-independent density over φ multiplies by ``2π``;
* **θ→cosθ change of variable** — ``dcosθ = -sinθ dθ`` ⇒ divide by ``sinθ``.

The three composed give, per spin, ``p(a,cosθ) = 2π·a²·p_cart(s⃗)`` from a
cartesian density, or ``p(a,cosθ) = p(a)·p(θ)/sinθ`` from a factored polar
density.

The component-basis draw density in the exported ``(m1det,q,dL,a1,a2,cosθ1,
cosθ2)`` basis is then obtained from the legacy per-year, weight-divided
``_pdraw`` by

    pdraw_component = _pdraw · exp(_ln_spin_component)

with the cancellation identity

    _ln_spin_component := ln_p_comp_spinfull − ln_pdraw_no_spin

where ``ln_pdraw_no_spin`` is *exactly* the spin-free log density the existing
loader already computed (see :mod:`gwcat.selection`).  Because the mass/z
Jacobian, ``T_obs`` and injection-weight divisions act identically on both
sides, they cancel and ``pdraw_component`` is the exact component-basis draw
density.
"""
from __future__ import annotations

import warnings

import numpy as np

# ln(2π) and ln(16 π²) appear throughout the azimuth / cartesian-prior algebra.
LN_2PI = np.log(2.0 * np.pi)
LN_16PI2 = np.log(16.0 * np.pi ** 2)

# Numerical floors (match the guards spelled out in the PR4 plan).
_SIN_FLOOR = 1e-300
_A_FLOOR = 1e-30


# ----------------------------------------------------------------------
# Small numeric helpers
# ----------------------------------------------------------------------
def clip_a(a):
    """Floor a spin magnitude away from zero (avoids ``log(0)``)."""
    return np.clip(np.asarray(a, dtype=float), _A_FLOOR, None)


def sintheta_from_cost(cost):
    """sinθ from cosθ, floored away from zero: ``clip(sqrt(1-cos²θ), 1e-300)``."""
    cost = np.asarray(cost, dtype=float)
    return np.clip(np.sqrt(np.clip(1.0 - cost ** 2, 0.0, 1.0)), _SIN_FLOOR, None)


def sintheta_from_theta(theta):
    """sinθ from a polar angle θ (radians), floored: ``clip(sin θ, 1e-300)``."""
    return np.clip(np.sin(np.asarray(theta, dtype=float)), _SIN_FLOOR, None)


def polar_from_cartesian(sx, sy, sz):
    """Return ``(a, cosθ)`` from cartesian spin components.

    ``a = |s⃗|`` and ``cosθ = s_z / a`` with the magnitude floored (``clip_a``)
    so that a zero-spin injection yields a finite, in-range ``cosθ``.
    """
    sx = np.asarray(sx, dtype=float)
    sy = np.asarray(sy, dtype=float)
    sz = np.asarray(sz, dtype=float)
    a = np.sqrt(sx ** 2 + sy ** 2 + sz ** 2)
    cost = np.clip(sz / clip_a(a), -1.0, 1.0)
    return a, cost


# ----------------------------------------------------------------------
# Full component-basis log densities, ln p_comp over
# (m1s, m2s, z, a1, a2, cosθ1, cosθ2)
# ----------------------------------------------------------------------
def ln_p_component_factored(ln_pmass_z, lnp_mag1, lnp_polar1,
                            lnp_mag2, lnp_polar2, cost1, cost2):
    """Format B (``o4_factored``): fully factored polar draw densities.

    Derivation (per spin).  The draw factorises as ``p(a)·p(θ)·p(φ)`` with the
    file storing ``lnp_mag = ln p(a)`` and ``lnp_polar = ln p(θ)`` (a density in
    the polar *angle* θ, **not** cosθ).  Marginalising the uniform azimuth
    integrates ``p(φ)`` to 1 (dropped — exact because the factorisation is
    exact), and the θ→cosθ Jacobian divides by ``sinθ``:

        ln p(a, cosθ) = lnp_mag + lnp_polar − ln sinθ.

    Adding the spin-free mass/redshift log density ``ln_pmass_z`` (which is what
    the legacy loader already computes) gives the full component density.
    """
    st1 = sintheta_from_cost(cost1)
    st2 = sintheta_from_cost(cost2)
    return (ln_pmass_z
            + lnp_mag1 + lnp_polar1 - np.log(st1)
            + lnp_mag2 + lnp_polar2 - np.log(st2))


def ln_p_component_joint_cartesian(ln_pdraw_joint, a1, a2):
    """Format C cartesian joint (and, algebraically, Format A joint).

    Derivation.  The file stores one joint cartesian log density
    ``ln p(m1s,m2s,z, s1x..s2z)``.  Per spin, the spherical volume element gives
    ``p(a,θ,φ)=p_cart·a²·sinθ``; the θ→cosθ Jacobian removes the ``sinθ``; and
    marginalising the uniform azimuth multiplies by ``2π``:

        p(a, cosθ) = 2π · a² · p_cart.

    Hence for the two spins

        ln p_comp = ln p_joint + 2·ln(2π) + 2·ln a1 + 2·ln a2.
    """
    return ln_pdraw_joint + 2.0 * LN_2PI + 2.0 * np.log(clip_a(a1)) \
        + 2.0 * np.log(clip_a(a2))


def ln_p_component_joint_polar(ln_pdraw_joint, cost1, cost2):
    """Format C polar joint.

    Derivation.  The file stores one joint *polar* log density
    ``ln p(m1s,m2s,z, a1,θ1,φ1, a2,θ2,φ2)``.  The magnitude ``a`` and the
    spherical volume element ``a²`` are already baked into this density, so we
    only marginalise the uniform azimuth (``+2π`` per spin) and apply the
    θ→cosθ Jacobian (``−ln sinθ`` per spin):

        ln p_comp = ln p_joint + 2·ln(2π) − ln sinθ1 − ln sinθ2.
    """
    st1 = sintheta_from_cost(cost1)
    st2 = sintheta_from_cost(cost2)
    return ln_pdraw_joint + 2.0 * LN_2PI - np.log(st1) - np.log(st2)


def ln_p_spin_cart_component(spin_pdf_cart, a):
    """Per-spin ``ln p(a, cosθ)`` from a cartesian spin marginal pdf (Format A).

    ``p(a,cosθ) = 2π·a²·p_cart`` (spherical volume element + uniform-azimuth
    marginalisation, θ→cosθ Jacobian cancels the volume ``sinθ``).  For the
    endo3 uniform-isotropic draw ``p_cart = 1/(4π a² max_spin)`` this collapses
    to ``1/(2·max_spin)``.

    ``spin_pdf_cart`` must be finite and strictly positive: a zero is a
    malformed row, and flooring it at 1e-300 (as this did until review
    2026-10-20, low-17) fabricated a draw density ~1e-300 times too small.
    """
    spin_pdf_cart = np.asarray(spin_pdf_cart, dtype=float)
    bad = ~np.isfinite(spin_pdf_cart) | (spin_pdf_cart <= 0.0)
    if np.any(bad):
        first = int(np.flatnonzero(np.atleast_1d(bad))[0])
        raise ValueError(
            f"per-spin cartesian sampling pdf is not finite and strictly "
            f"positive: {int(np.sum(bad))} of {bad.size} entries (first at "
            f"index {first}, value {np.atleast_1d(spin_pdf_cart)[first]!r}); "
            f"a zero draw density is a malformed row, refused rather than "
            f"floored.")
    return np.log(spin_pdf_cart) + 2.0 * np.log(clip_a(a)) + LN_2PI


# ----------------------------------------------------------------------
# Legacy spin-free density from a Format-C *polar* joint
# ----------------------------------------------------------------------
def ln_pdraw_no_spin_from_polar_joint(ln_pdraw_joint, cost1, cost2, amax=0.99):
    """Legacy (pre-PR4) spin-free draw density from a Format-C polar joint.

    The existing cartesian branch subtracts an *assumed* isotropic
    uniform-magnitude cartesian spin prior
    ``p_assumed_cart = 1/(16π² a1² a2² amax²)`` from the cartesian joint, i.e.

        ln_pdraw_no_spin_cart = ln p_joint_cart + ln(16π² a1² a2² amax²).

    A polar joint equals its cartesian twin *times* the coordinate Jacobian of
    ``(sx,sy,sz)→(a,θ,φ)`` for each spin, ``a²·sinθ``:

        ln p_joint_polar = ln p_joint_cart + 2 ln a1 + ln sinθ1
                                          + 2 ln a2 + ln sinθ2.

    Substituting to force the *same* legacy ``_pdraw`` for both flavours, the
    ``a²`` terms cancel and the assumed prior in polar variables becomes

        ln p_assumed_polar = ln sinθ1 + ln sinθ2 − ln(16π² amax²),

    which we subtract:

        ln_pdraw_no_spin_polar = ln p_joint_polar − ln p_assumed_polar
                               = ln p_joint_polar − ln sinθ1 − ln sinθ2
                                 + ln(16π² amax²).

    By construction this is byte-identical to the cartesian branch's result for
    the equivalent draws (proved in ``tests/test_selection_spin.py``).
    """
    st1 = sintheta_from_cost(cost1)
    st2 = sintheta_from_cost(cost2)
    ln_p_assumed_polar = np.log(st1) + np.log(st2) - (LN_16PI2 + 2.0 * np.log(amax))
    return ln_pdraw_joint - ln_p_assumed_polar


# ----------------------------------------------------------------------
# Read-time checks + amax auto-detection
# ----------------------------------------------------------------------
#: Relative slack on the TOP edge of a detected uniform magnitude support: no
#: drawn magnitude may exceed the detected ceiling by more than this.  endo3
#: stores float32 spin components, which round |s| by ~1e-7 relative.
AMAX_EDGE_RTOL = 1e-5

#: The BOTTOM edge: ``N`` magnitudes drawn from ``U(0, amax)`` have
#: ``P(min a > x) = (1 - x/amax)^N``, so ``min a > LOW_EDGE_DRAWS * amax / N``
#: happens with probability below ``exp(-LOW_EDGE_DRAWS) ~ 1e-13``.  A campaign
#: whose smallest magnitude sits above that bound did not draw from 0.
LOW_EDGE_DRAWS = 30.0


def uniform_magnitude_support_ok(amax, a):
    """Whether drawn magnitudes ``a`` are consistent with ``U(0, amax)``.

    A constant magnitude density only says ``p(a) = 1/width`` on SOME interval;
    the review of 2026-10-20 (medium-16) found both detectors reading
    ``1/width`` as a ceiling for ``a ~ U(a_min, a_max)`` with ``a_min > 0``.
    This requires the support to be the one the chi_eff swap and the coverage
    check assume, ``[0, amax]``: ``0 < amax <= 1`` (a physical ceiling; a
    larger "ceiling" is an artefact, medium-15), no drawn magnitude above
    ``amax`` (within :data:`AMAX_EDGE_RTOL`), and the smallest one within
    :data:`LOW_EDGE_DRAWS` ``* amax / N`` of zero.  ``a=None`` checks
    ``amax`` alone.
    """
    if amax is None or not np.isfinite(amax):
        return False
    if not (0.0 < amax <= 1.0 + AMAX_EDGE_RTOL):
        return False
    if a is None:
        return True
    a = np.asarray(a, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if not a.size:
        return True
    if float(a.max()) > amax * (1.0 + AMAX_EDGE_RTOL):
        return False
    return bool(float(a.min()) <= amax * min(1.0, LOW_EDGE_DRAWS / a.size))


def detect_uniform_amax_from_lnmag(lnp_mag, tol_std=1e-6, a=None):
    """Detect a uniform magnitude draw and its ``amax`` from ``lnp_mag``.

    For ``a ~ U(0, amax)`` the magnitude density is ``p(a)=1/amax`` so
    ``lnp_mag = −ln amax`` is *constant*.  Returns ``(amax, is_uniform)`` where
    ``is_uniform`` is ``std(lnp_mag) < tol_std`` AND the drawn magnitudes ``a``
    (when given) span ``[0, amax]`` (:func:`uniform_magnitude_support_ok`).

    ``amax`` is ``exp(−median(lnp_mag))`` **only when the draw is actually
    uniform from zero**, and ``None`` otherwise (GW-05).  Returning the
    expression unconditionally produced a number that is not an amax at all for
    a non-uniform draw: the O4ab campaign yields 0.7478 / 0.7397 by this formula
    while its spins reach 0.99999, because the median of a *varying* log-density
    is just a summary statistic of a mixture, not a ceiling.  That fabricated
    value was fed straight into the joint (chi_eff, chi_p) prior under
    ``strict=False``, silently mis-specifying the support by ~25%.  A CONSTANT
    density on ``[a_min, a_max]`` with ``a_min > 0`` gives ``a_max - a_min``,
    which is not a ceiling either (review 2026-10-20, medium-16), so the
    support is checked too.
    """
    lnp_mag = np.asarray(lnp_mag, dtype=float)
    is_uniform = bool(np.std(lnp_mag) < tol_std)
    amax = float(np.exp(-np.median(lnp_mag))) if is_uniform else None
    if is_uniform and not uniform_magnitude_support_ok(amax, a):
        return None, False
    return amax, is_uniform


def check_isotropy_polar(lnp_polar, cost, tol=1e-3):
    """Isotropy check for a factored polar draw: ``lnp_polar ≈ ln(sinθ/2)``.

    Returns ``(passed, max_abs_dev)``.
    """
    lnp_polar = np.asarray(lnp_polar, dtype=float)
    st = sintheta_from_cost(cost)
    dev = float(np.max(np.abs(lnp_polar - np.log(st / 2.0)))) if lnp_polar.size \
        else 0.0
    return bool(dev < tol), dev


def check_uniform_azimuth(lnp_azim, tol=1e-3):
    """Uniform-azimuth check: ``lnp_azim ≈ −ln(2π)`` ⇒ ``|lnp_azim+ln2π|<tol``.

    Returns ``(passed, max_abs_dev)``.
    """
    lnp_azim = np.asarray(lnp_azim, dtype=float)
    dev = float(np.max(np.abs(lnp_azim + LN_2PI))) if lnp_azim.size else 0.0
    return bool(dev < tol), dev


def detect_max_spin_cart(spin_pdf_cart, a, tol=1e-6):
    """Detect ``max_spin`` from a cartesian spin marginal pdf (Format A).

    For ``p_cart = 1/(4π a² max_spin)`` the product ``c = p_cart·4π·a²`` is the
    constant ``1/max_spin``.  Returns ``(max_spin, is_uniform_isotropic)`` where
    uniformity is judged by the *relative* spread of ``c`` and by the drawn
    magnitudes ``a`` spanning ``[0, max_spin]``
    (:func:`uniform_magnitude_support_ok`).

    ``max_spin`` is ``None`` unless the draw is uniform from zero -- the twin of
    the O4 detector's GW-05 fix, which this one never received (review
    2026-10-20, medium-15): for a non-uniform draw ``1/median(c)`` is a summary
    statistic of the magnitude density, not a ceiling (1.056 for 90% U(0, 0.95)
    + 10% U(0.40, 0.42); 0.71 for p(a) = 2a), and it was stored as the
    campaign's detected amax and counted as proof by the reference-coverage
    check.  For a constant density on ``[a_min, a_max]``, ``a_min > 0``, it is
    ``a_max - a_min`` (medium-16).
    """
    spin_pdf_cart = np.asarray(spin_pdf_cart, dtype=float)
    c = spin_pdf_cart * 4.0 * np.pi * clip_a(a) ** 2
    med = float(np.median(c))
    if med <= 0.0 or not np.isfinite(med):
        return None, False
    rel_spread = float(np.max(np.abs(c - med)) / med) if c.size else 0.0
    if not rel_spread < tol:
        return None, False
    max_spin = 1.0 / med
    if not uniform_magnitude_support_ok(max_spin, a):
        return None, False
    return max_spin, True


def check_factored_vs_joint(sampling_pdf, p_mass, p_z,
                            spin_pdf1, spin_pdf2, rtol=1e-6):
    """Format A consistency: joint ``sampling_pdf`` == product of the factors.

    Returns ``(passed, max_rel_dev)`` comparing ``sampling_pdf`` against
    ``p_mass·p_z·spin_pdf1·spin_pdf2``.
    """
    sampling_pdf = np.asarray(sampling_pdf, dtype=float)
    product = (np.asarray(p_mass, dtype=float) * np.asarray(p_z, dtype=float)
               * np.asarray(spin_pdf1, dtype=float)
               * np.asarray(spin_pdf2, dtype=float))
    denom = np.where(np.abs(sampling_pdf) > 0, np.abs(sampling_pdf), 1.0)
    dev = float(np.max(np.abs(product - sampling_pdf) / denom)) \
        if sampling_pdf.size else 0.0
    return bool(dev <= rtol), dev


# ----------------------------------------------------------------------
# strict_spin_checks dispatch
# ----------------------------------------------------------------------
_VALID_STRICT = ("warn", "raise", "off")


def normalize_strict_mode(strict_spin_checks):
    """Validate / normalise the ``strict_spin_checks`` argument.

    Accepts ``"warn"`` (default), ``"raise"``, ``"off"``.  ``True`` maps to
    ``"raise"`` and ``False`` maps to ``"off"`` for convenience.
    """
    if strict_spin_checks is True:
        return "raise"
    if strict_spin_checks is False:
        return "off"
    mode = str(strict_spin_checks).lower()
    if mode not in _VALID_STRICT:
        raise ValueError(
            f"strict_spin_checks must be one of {_VALID_STRICT!r} "
            f"(or a bool); got {strict_spin_checks!r}")
    return mode


def report_check(name, passed, detail, mode, path=None):
    """Act on a single read-time check according to ``mode``.

    ``mode="off"`` records only; ``"warn"`` additionally emits a
    :class:`UserWarning`; ``"raise"`` raises :class:`ValueError` on failure.
    Always returns ``passed`` so callers can accumulate.
    """
    if not passed:
        where = f" in {path}" if path else ""
        msg = (f"selection spin check {name!r} failed{where}: {detail}")
        if mode == "raise":
            raise ValueError(msg)
        if mode == "warn":
            warnings.warn(msg)
    return passed
