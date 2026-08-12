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

Note on bounds (GW-01): the recorded [dmin, dmax] frequently come from a
*sibling* analysis's analytic prior (see ingest.resolve_dL_prior), so posterior
samples of the ingested analysis legitimately fall outside them.  Zeroing the
density there would hand those samples ``p_dL_pe = 0`` -> ``p_pe = 0``, which
darksirens turns into a ``-inf`` log-weight that still counts in ``n``.  Since
the normalisation cancels anyway, the density is evaluated over a range widened
to cover every finite sample instead; the recorded bounds are kept for
provenance and the count of samples outside them is reported so the mismatch is
visible rather than silently destructive.

Note on implementation choice: bilby and the astropy fallback do NOT agree to
machine precision (they differ by ~3% at the low-distance end), and that
difference does not cancel in the per-event normalisation.  Which one produced a
given ``p_dL_pe`` is therefore part of the provenance, returned as
``info["impl"]`` and stored per row as ``dL_prior_impl``.  There is no silent
fallback: only a missing bilby install falls back to astropy, and any other
failure propagates.
"""
from __future__ import annotations

import importlib

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
    """Invert dL(z) by monotonic interpolation. dL in Mpc.

    The inversion uses a log-spaced z grid on ``[0, zmax]`` plus linear
    interpolation.  ``numpy.interp`` clamps out-of-range inputs to the grid
    endpoints, which would SILENTLY map any dL beyond ``dL(zmax)`` to exactly
    ``zmax`` -- a hidden clip that fabricates redshifts.  To avoid that, the z
    grid is extended adaptively (with a loud warning) whenever an input dL
    exceeds the current grid's maximum, so no sample is silently clipped.

    For inputs already within ``[0, dL(zmax)]`` the grid and the returned
    values are byte-for-byte unchanged (the extension branch is skipped).
    """
    dL = np.asarray(dL_mpc, dtype=float)
    z = np.expm1(np.linspace(np.log(1.0), np.log(1.0 + zmax), n))
    dl = cosmology.luminosity_distance(z).to(u.Mpc).value

    finite = dL[np.isfinite(dL)]
    dL_max = float(finite.max()) if finite.size else 0.0
    if dL_max > dl[-1]:
        import warnings
        dl_hi_orig = float(dl[-1])
        n_beyond = int(np.sum(finite > dl_hi_orig))
        zmax_ext = zmax
        # Grow zmax until the grid covers the largest requested dL (hard cap
        # at z=1e4 to guarantee termination for absurd inputs).
        while dl[-1] < dL_max and zmax_ext < 1e4:
            zmax_ext *= 2.0
            z = np.expm1(np.linspace(np.log(1.0), np.log(1.0 + zmax_ext), n))
            dl = cosmology.luminosity_distance(z).to(u.Mpc).value
        if dl[-1] < dL_max:
            raise ValueError(
                f"z_of_dL: dL={dL_max:.4g} Mpc exceeds dL at the maximum "
                f"supported redshift z={zmax_ext:g}; refusing to invert "
                f"because it would clip samples to z={zmax_ext:g}.")
        warnings.warn(
            f"z_of_dL: {n_beyond} sample(s) have dL beyond dL(z={zmax:g}) "
            f"({dl_hi_orig:.4g} Mpc); extended the inversion grid to "
            f"z={zmax_ext:g} to avoid silently clipping them to z={zmax:g}.",
            stacklevel=2,
        )
    return np.interp(dL, dl, z)


#: Implementations of the UniformSourceFrame density, in preference order.
USF_IMPLS = ("bilby", "astropy")


class DistancePriorImplError(RuntimeError):
    """The requested distance-prior implementation is unavailable."""


def resolve_usf_impl(impl: str = "auto") -> str:
    """Resolve ``impl`` to a concrete implementation name.

    ``"auto"`` prefers bilby and falls back to astropy **only** when bilby is
    not importable.  ``"bilby"`` raises :class:`DistancePriorImplError` when
    bilby is missing rather than silently degrading to a density that differs
    by a few percent.
    """
    if impl not in ("auto",) + USF_IMPLS:
        raise ValueError(
            f"impl must be 'auto' or one of {USF_IMPLS}; got {impl!r}")
    if impl == "astropy":
        return "astropy"
    try:
        importlib.import_module("bilby.gw.prior")
    except ImportError as exc:
        if impl == "bilby":
            raise DistancePriorImplError(
                "impl='bilby' was requested but bilby is not importable "
                f"({exc}).  Pass impl='astropy' to accept the fallback "
                "density explicitly -- it differs from bilby's by ~3% at the "
                "low-distance end and that does not cancel downstream."
            ) from exc
        return "astropy"
    return "bilby"


def uniform_source_frame_prob(dL_mpc, cosmology: FlatLambdaCDM,
                              dmin: float, dmax: float, *,
                              impl: str = "auto", return_info: bool = False):
    """p(dL) for a UniformSourceFrame prior, evaluated everywhere. dL in Mpc.

    The density is **not** truncated at ``[dmin, dmax]``: when finite samples
    fall outside the recorded bounds, the evaluation range is widened to cover
    them (which only changes the per-event-constant normalisation, see the
    module docstring).  ``dmin``/``dmax`` are still used to count and report the
    mismatch.

    Parameters
    ----------
    impl : {"auto", "bilby", "astropy"}
        Which implementation to use.  ``"auto"`` prefers bilby, falling back to
        astropy only when bilby is not installed.
    return_info : bool
        When True, return ``(p, info)`` where ``info`` records the
        implementation actually used, the widened evaluation range, and how many
        samples fell outside the recorded bounds.
    """
    dL = np.asarray(dL_mpc, dtype=float)
    dmin = float(dmin)
    dmax = float(dmax)

    finite = dL[np.isfinite(dL)]
    n_below = int(np.sum(finite < dmin))
    n_above = int(np.sum(finite > dmax))
    n_outside = n_below + n_above

    eval_min, eval_max = dmin, dmax
    if n_outside:
        # Widen just enough to cover every finite sample, with a small pad so
        # no sample sits exactly on an implementation's support boundary.
        lo = min(dmin, float(finite.min()))
        hi = max(dmax, float(finite.max()))
        pad = max(1e-9, 1e-6 * (hi - lo))
        eval_min = max(lo - pad, 1e-6)
        eval_max = hi + pad

    impl_used = resolve_usf_impl(impl)
    if impl_used == "bilby":
        p = _usf_prob_bilby(dL, cosmology, eval_min, eval_max)
    else:
        p = _usf_prob_astropy(dL, cosmology, eval_min, eval_max)

    if not return_info:
        return p
    info = {
        "impl": impl_used,
        "dmin": dmin,
        "dmax": dmax,
        "eval_min": float(eval_min),
        "eval_max": float(eval_max),
        "widened": bool(n_outside),
        "n_samples": int(dL.size),
        "n_below_dmin": n_below,
        "n_above_dmax": n_above,
        "n_outside_bounds": n_outside,
        "frac_outside_bounds": (n_outside / dL.size) if dL.size else 0.0,
        "n_nonfinite": int(dL.size - finite.size),
    }
    return p, info


def _usf_prob_bilby(dL, cosmology, dmin, dmax):
    """bilby's UniformSourceFrame density -- the object the LVK PE itself used."""
    from bilby.gw.prior import UniformSourceFrame
    prior = UniformSourceFrame(
        minimum=float(dmin), maximum=float(dmax),
        cosmology=cosmology, name="luminosity_distance",
        latex_label="$d_L$", unit="Mpc", boundary=None,
    )
    return np.asarray(prior.prob(dL), dtype=float)


def _usf_grid(cosmology, zmax: float, ngrid: int):
    """(dL, p(dL)) on a log-spaced z grid, up to an arbitrary constant.

    ``E(z)`` comes from the cosmology object (``efunc``) rather than a
    hardcoded matter+Lambda form, so a cosmology carrying radiation, neutrinos
    or curvature is handled correctly.  For a bare ``FlatLambdaCDM(H0, Om0)``
    -- what :func:`make_cosmology` builds -- the two agree exactly.
    """
    z = np.expm1(np.linspace(np.log(1.0), np.log(1.0 + zmax), ngrid))
    DC = cosmology.comoving_distance(z).to(u.Mpc).value
    E = np.asarray(cosmology.efunc(z), dtype=float)
    dH = float(cosmology.hubble_distance.to(u.Mpc).value)
    dDC_dz = dH / E
    dL_grid = (1 + z) * DC
    ddL_dz = DC + (1 + z) * dDC_dz
    # p(z) propto comoving-volume element * time dilation
    pz = (DC ** 2 / E) * (1.0 / (1 + z))
    # change of variables to dL
    return dL_grid, pz / ddL_dz


def _usf_prob_astropy(dL, cosmology, dmin, dmax, ngrid: int = 4000):
    """Astropy fallback: p(dL) propto dVc/dz * 1/(1+z) * |dz/ddL|, normalised
    on [dmin, dmax].

    The z grid is extended until it covers ``dmax`` so that no sample is
    silently interpolated to zero off the end of the grid.
    """
    dL = np.asarray(dL, dtype=float)
    zmax = 10.0
    dL_grid, p_dL_grid = _usf_grid(cosmology, zmax, ngrid)
    while dL_grid[-1] < dmax and zmax < 1e4:
        zmax *= 2.0
        dL_grid, p_dL_grid = _usf_grid(cosmology, zmax, ngrid)
    if dL_grid[-1] < dmax:
        raise ValueError(
            f"_usf_prob_astropy: dmax={dmax:.4g} Mpc exceeds dL at z={zmax:g}; "
            "refusing to evaluate because samples beyond the grid would be "
            "interpolated to zero.")

    # Normalise on a dense LINEAR grid over [dmin, dmax] rather than on the
    # log-spaced z grid restricted to those bounds: the latter misses the partial
    # end bins and is only ~0.5% accurate for a narrow distance window.  The
    # normalisation is a per-event constant that cancels downstream either way,
    # but this makes the returned density actually normalised as documented.
    if dmax > dmin:
        fine = np.linspace(dmin, dmax, 20001)
        norm = float(_trapz(np.interp(fine, dL_grid, p_dL_grid), fine))
    else:
        norm = 0.0
    if not (norm > 0):
        raise ValueError(
            f"_usf_prob_astropy: the prior range [{dmin:.4g}, {dmax:.4g}] Mpc "
            "encloses no normalisable probability; check the recorded distance-"
            "prior bounds.")
    # No left/right zero-fill needed: the grid spans [0, dL(zmax)] >= dmax, and
    # the density is evaluated everywhere rather than truncated at the bounds.
    return np.interp(dL, dL_grid, p_dL_grid) / norm
