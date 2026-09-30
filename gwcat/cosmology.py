"""Cosmology helpers and the cosmo-file luminosity-distance prior.

Everything here is mass-prior-agnostic. We only ever deal with the marginal
distance prior p(dL | cosmo). The (m1det, q, dL)-basis mass Jacobian is applied
elsewhere (exclusively in GWCatalog.to_darksirens).

The cosmo files (GWTC-2.1/3 *cosmo.h5 and the O4 combined files) use a
luminosity-distance prior that is uniform in comoving volume and source-frame
time -- i.e. bilby's UniformSourceFrame. Since GW-40i gwcat evaluates THAT
DENSITY exactly (``impl="exact"``, below) rather than bilby's interpolated
object of it; the legacy bilby object and a self-contained astropy fallback
remain selectable for regressions against older products.

Note on normalisation: the darksirens loader normalises p_pe per event, so any
per-event-constant factor (including the prior's normalisation over [dmin,dmax])
cancels. Only the *shape* of p(dL), set by the cosmology, affects the final
weights. Bounds are therefore stored for provenance but do not change results.

Note on bounds (GW-01): the recorded [dmin, dmax] frequently come from a
*sibling* analysis's analytic prior (see ingest.resolve_dL_prior), so posterior
samples of the ingested analysis legitimately fall outside them.  Zeroing the
density there would hand those samples ``p_dL_pe = 0`` -> ``p_pe = 0``, which
darksirens turns into a ``-inf`` log-weight that still counts in ``n``.  The
density is therefore evaluated everywhere, never truncated: the exact
implementation normalises over the recorded bounds and evaluates the same
formula outside them, and the legacy (interpolated) ones widen their evaluation
range to cover every finite sample (the normalisation cancels anyway).  The
recorded bounds are kept for provenance and the count of samples outside them
is reported so the mismatch is visible rather than silently destructive.

Note on implementation choice (GW-40i): the default is ``impl="exact"``, which
evaluates the UniformSourceFrame / UniformComovingVolume density from accurate
cosmology integrals -- the comoving distance by composite Gauss-Legendre
quadrature of ``1/E(z)`` (no table), ``z(dL)`` by Newton iteration to machine
precision, and the normalisation over the DECLARED ``[dmin, dmax]`` by
composite Gauss-Legendre quadrature in ``z`` -- to <= 1.4e-15 relative against
an independent mpmath evaluation (``tests/fixtures/usf_mpmath_reference.json``:
three flat cosmologies including LAL Planck15, five bound pairs from 1 to
50000 Mpc, inside and outside the bounds); ~2 s per 1e6 samples.  It is
evaluated everywhere and never widened (see GW-01
above: the declared normalisation is kept, and samples outside the bounds get
the same untruncated formula).

The two older implementations are LEGACY, kept only to reproduce products built
before GW-40i: ``"bilby"`` (bilby's own object, which interpolates its density
from a 1000-point grid across ``[dmin, dmax]`` -- up to 3.4e-3 off in ln-shape
against the exact density) and ``"astropy"`` (a 4000-point interpolated
fallback).  ``impl="auto"`` is the pre-GW-40i default and means "bilby, else
astropy".  bilby and the astropy fallback do NOT agree to machine precision
(they differ by ~3% at the low-distance end, and by far more below ~10 Mpc,
where bilby's own interpolant is coarse while the fallback is refined there, see
``_usf_z_grid``), and neither difference cancels in the per-event
normalisation.  Which one produced a given ``p_dL_pe`` is therefore part of the
provenance, returned as ``info["impl"]`` and stored per row as
``dL_prior_impl``.  There is no silent fallback: only a missing bilby install
turns ``"auto"`` into astropy, and any other failure propagates.
"""
from __future__ import annotations

import importlib

import numpy as np
from astropy.cosmology import FlatLambdaCDM, Planck15
import astropy.units as u

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# LVK default cosmology used for the O3 cosmo-file reweighting (bilby default).
PLANCK15 = Planck15

#: LAL's Planck15, which is NOT astropy's.  GWTC-4.1/5 analytic priors record
#: ``cosmology='Planck15_LAL'`` (and GWTC-4.1 sometimes spells the same numbers
#: out as a ``LambdaCDM(H0=67.9, Om0=0.3065, ...)`` repr).  astropy's Planck15 is
#: (67.74, 0.3075); LAL's is (67.90, 0.3065).  A substring test for "Planck15"
#: matches BOTH tokens, which is how these rows silently acquired astropy's
#: values -- see :data:`NAMED_COSMOLOGIES`, which is matched exactly.
LAL_PLANCK15 = FlatLambdaCDM(H0=67.90, Om0=0.3065)

# Historical name for the same object: the fallback O4 PE cosmology, used when a
# file's analytic prior string carries no explicit cosmology.
O4_FALLBACK = LAL_PLANCK15

#: Cosmology tokens that appear verbatim in LVK analytic prior reprs, resolved by
#: EXACT match (after stripping quotes/whitespace) -- never by substring.
#:
#: ``'Planck15_lal'`` (GW-40a) is the lower-case spelling some pesummary
#: ``meta_data`` blocks and prior reprs use for the same LAL object.  It is an
#: EXACT alias -- there is still no case-folding, so ``'planck15_lal'`` or
#: ``'PLANCK15_LAL'`` remain unrecognised and fail loudly -- and it is consulted
#: only for a cosmology an analytic prior string explicitly DECLARES.  gwcat
#: never reads ``meta_data/cosmology`` to decide a reweighted release's prior
#: (that field is absent on every GWTC-2.1/3 ``C01:Mixed`` set and contradicts
#: the stored z(dL) for several GWTC-4.1/5 labels); see
#: :mod:`gwcat.release_cosmology` for where that cosmology comes from instead.
NAMED_COSMOLOGIES = {
    "Planck15": PLANCK15,
    "Planck15_LAL": LAL_PLANCK15,
    "Planck15_lal": LAL_PLANCK15,
}


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


#: Implementations of the UniformSourceFrame density.  ``"exact"`` (GW-40i) is
#: the default; ``"bilby"`` and ``"astropy"`` are the legacy interpolated ones.
USF_IMPLS = ("exact", "bilby", "astropy")

#: The default implementation of every public entry point (GW-40i).
USF_IMPL_DEFAULT = "exact"


class DistancePriorImplError(RuntimeError):
    """The requested distance-prior implementation is unavailable."""


def resolve_usf_impl(impl: str = USF_IMPL_DEFAULT) -> str:
    """Resolve ``impl`` to a concrete implementation name.

    ``"exact"`` (the default) and ``"astropy"`` need nothing optional.  The
    LEGACY ``"auto"`` prefers bilby and falls back to astropy **only** when
    bilby is not importable.  ``"bilby"`` raises :class:`DistancePriorImplError`
    when bilby is missing rather than silently degrading to a density that
    differs by a few percent.
    """
    if impl not in ("auto",) + USF_IMPLS:
        raise ValueError(
            f"impl must be 'auto' or one of {USF_IMPLS}; got {impl!r}")
    if impl in ("exact", "astropy"):
        return impl
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
                              impl: str = USF_IMPL_DEFAULT,
                              return_info: bool = False,
                              time_dilation: bool = True):
    """p(dL) for a UniformSourceFrame prior, evaluated everywhere. dL in Mpc.

    ``time_dilation=False`` gives UniformComovingVolume instead (same shape
    without the ``1/(1+z)`` source-frame-time factor).

    The density is **not** truncated at ``[dmin, dmax]``.  With ``impl="exact"``
    it is normalised over exactly the declared ``[dmin, dmax]`` and the same
    formula is evaluated outside it.  The legacy implementations interpolate a
    density that is only defined on their evaluation range, so for them, when
    finite samples fall outside the recorded bounds, the range is widened to
    cover them (which only changes the per-event-constant normalisation, see
    the module docstring).  ``dmin``/``dmax`` are always used to count and
    report the mismatch.

    Parameters
    ----------
    impl : {"exact", "auto", "bilby", "astropy"}
        Which implementation to use (default ``"exact"``, GW-40i).  The legacy
        ``"auto"`` prefers bilby, falling back to astropy only when bilby is not
        installed.
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

    impl_used = resolve_usf_impl(impl)
    eval_min, eval_max = dmin, dmax
    if n_outside and impl_used != "exact":
        # Widen just enough to cover every finite sample, with a small pad so
        # no sample sits exactly on an implementation's support boundary.
        lo = min(dmin, float(finite.min()))
        hi = max(dmax, float(finite.max()))
        pad = max(1e-9, 1e-6 * (hi - lo))
        eval_min = max(lo - pad, 1e-6)
        eval_max = hi + pad

    cls_name = ("UniformSourceFrame" if time_dilation
                else "UniformComovingVolume")
    if impl_used == "exact":
        p = usf_prob_exact(dL, cosmology, dmin, dmax,
                           time_dilation=time_dilation)
    elif impl_used == "bilby":
        p = _usf_prob_bilby(dL, cosmology, eval_min, eval_max,
                            cls_name=cls_name)
    else:
        p = _usf_prob_astropy(dL, cosmology, eval_min, eval_max,
                              time_dilation=time_dilation)

    if not return_info:
        return p
    info = {
        "impl": impl_used,
        "kind": cls_name,
        "dmin": dmin,
        "dmax": dmax,
        "eval_min": float(eval_min),
        "eval_max": float(eval_max),
        "widened": bool(eval_min != dmin or eval_max != dmax),
        "n_samples": int(dL.size),
        "n_below_dmin": n_below,
        "n_above_dmax": n_above,
        "n_outside_bounds": n_outside,
        "frac_outside_bounds": (n_outside / dL.size) if dL.size else 0.0,
        "n_nonfinite": int(dL.size - finite.size),
    }
    return p, info


def _usf_prob_bilby(dL, cosmology, dmin, dmax, *,
                    cls_name: str = "UniformSourceFrame"):
    """bilby's cosmological distance prior -- the object the LVK PE itself used."""
    import bilby.gw.prior as _bp
    cls = getattr(_bp, cls_name)
    prior = cls(
        minimum=float(dmin), maximum=float(dmax),
        cosmology=cosmology, name="luminosity_distance",
        latex_label="$d_L$", unit="Mpc", boundary=None,
    )
    return np.asarray(prior.prob(dL), dtype=float)


# --------------------------------------------------------------------------
# The exact UniformSourceFrame density (GW-40i)
# --------------------------------------------------------------------------
#: Gauss-Legendre rule of the comoving-distance and normalisation quadratures.
_EX_GL_X, _EX_GL_W = np.polynomial.legendre.leggauss(32)
#: Panel edges in z of the comoving-distance quadrature.  ``1/E(z)`` is analytic
#: with its nearest complex singularities ~1.2 from the real axis (flat LCDM,
#: Om0 ~ 0.3), so 32 nodes on a panel no longer than its distance from 0 --
#: geometric panels -- are exact to rounding.
_EX_EDGES = np.concatenate(([0.0, 0.125, 0.25, 0.5], 2.0 ** np.arange(0, 15)))
#: Sub-panels per panel for the normalisation integral (the integrand
#: ``D_C^2 / (E (1+z))`` is smooth; 8 x 32 nodes is converged to rounding).
_EX_NORM_SUB = 8
#: Newton iterations for z(dL): the interpolated start is good to ~1e-6
#: relative, so 2-3 steps reach rounding; the loop stops once converged.
_EX_NEWTON_MAX = 12
_EX_CHUNK = 262144
_EX_CACHE = {}


class _ExactFlatDistances:
    """Comoving/luminosity distance of one flat cosmology without any table.

    ``D_C(z) = D_H INT_0^z dz'/E(z')`` is the sum of precomputed whole-panel
    integrals (:data:`_EX_EDGES`) and a 32-node Gauss-Legendre integral over
    the last, partial panel; ``E`` comes from the cosmology object itself
    (``inv_efunc``), so radiation or massive neutrinos are honoured.  Only flat
    cosmologies are supported (``D_L = (1+z) D_C``); anything else raises.
    """

    def __init__(self, cosmology):
        ok0 = float(getattr(cosmology, "Ok0", 0.0))
        if abs(ok0) > 0.0:
            raise NotImplementedError(
                f"exact distance prior needs a FLAT cosmology (Ok0 = 0); got "
                f"Ok0={ok0!r} for {cosmology!r}.")
        self.cosmology = cosmology
        self.dH = float(cosmology.hubble_distance.to(u.Mpc).value)
        a, b = _EX_EDGES[:-1], _EX_EDGES[1:]
        # whole panels, each split in 4 for margin
        sub = np.linspace(0.0, 1.0, 5)
        lo = (a[:, None] + (b - a)[:, None] * sub[None, :-1]).ravel()
        hi = (a[:, None] + (b - a)[:, None] * sub[None, 1:]).ravel()
        panel = self._gl(lo, hi).reshape(a.size, 4).sum(axis=1)
        self.cum = np.concatenate(([0.0], np.cumsum(panel)))

    def inv_e(self, z):
        return np.asarray(self.cosmology.inv_efunc(z), dtype=float)

    def _gl(self, lo, hi):
        """INT_lo^hi dz/E(z) per pair, 32-node Gauss-Legendre."""
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        half = 0.5 * (hi - lo)
        mid = 0.5 * (hi + lo)
        z = mid[:, None] + half[:, None] * _EX_GL_X[None, :]
        return half * (self.inv_e(z) @ _EX_GL_W)

    def comoving(self, z):
        """D_C(z) in Mpc (z >= 0, vectorised, chunked)."""
        z = np.asarray(z, dtype=float)
        flat = z.ravel()
        if flat.size and not (np.nanmax(flat) <= _EX_EDGES[-1]):
            raise ValueError(
                f"exact distance prior: z={np.nanmax(flat):.4g} beyond the "
                f"supported z <= {_EX_EDGES[-1]:g}.")
        out = np.empty(flat.shape, dtype=float)
        for i in range(0, flat.size, _EX_CHUNK):
            zz = flat[i:i + _EX_CHUNK]
            k = np.clip(np.searchsorted(_EX_EDGES, zz, side="right") - 1,
                        0, _EX_EDGES.size - 2)
            out[i:i + _EX_CHUNK] = self.dH * (self.cum[k]
                                              + self._gl(_EX_EDGES[k], zz))
        return out.reshape(z.shape)

    def luminosity(self, z):
        z = np.asarray(z, dtype=float)
        return (1.0 + z) * self.comoving(z)

    def z_of_dL(self, dL):
        """z(dL) by Newton iteration on D_L(z) = dL, to rounding.  dL >= 0."""
        dL = np.asarray(dL, dtype=float)
        # A starting guess only: log-(1+z) grid covering max(dL).
        zmax = 2.0
        top = float(np.max(dL)) if dL.size else 0.0
        while True:
            zg = np.expm1(np.linspace(0.0, np.log1p(zmax), 2049))
            dg = self.luminosity(zg)
            if dg[-1] >= top or zmax >= _EX_EDGES[-1]:
                break
            zmax = min(2.0 * zmax, float(_EX_EDGES[-1]))
        if dg[-1] < top:
            raise ValueError(
                f"exact distance prior: dL={top:.6g} Mpc is beyond "
                f"z={_EX_EDGES[-1]:g}.")
        z = np.interp(dL, dg, zg)
        for _ in range(_EX_NEWTON_MAX):
            dc = self.comoving(z)
            f = (1.0 + z) * dc - dL
            fp = dc + (1.0 + z) * self.dH * self.inv_e(z)
            step = f / fp
            z = np.maximum(z - step, 0.0)
            # Quadratic convergence: once every step is below 1e-10 relative,
            # the one just taken has left an error of order (1e-10)^2, i.e.
            # below rounding.  (A tighter test never fires: at convergence the
            # step is rounding noise of f, a few ulp of z.)
            if not np.any(np.abs(step) > 1e-10 * np.maximum(z, 1e-300)):
                break
        return z

    def density_shape(self, z, time_dilation=True):
        """dVc/dz [/(1+z)] * dz/dL, up to the constant 4 pi D_H (Mpc)."""
        dc = self.comoving(z)
        ie = self.inv_e(z)
        num = dc * dc * ie
        if time_dilation:
            num = num / (1.0 + z)
        return num / (dc + (1.0 + z) * self.dH * ie)

    def norm(self, zlo, zhi, time_dilation=True):
        """INT_zlo^zhi D_C^2 / E [/(1+z)] dz -- the normalisation of
        :meth:`density_shape` over dL in [dL(zlo), dL(zhi)]."""
        if not zhi > zlo:
            return 0.0
        inner = _EX_EDGES[(_EX_EDGES > zlo) & (_EX_EDGES < zhi)]
        br = np.concatenate(([zlo], inner, [zhi]))
        t = np.linspace(0.0, 1.0, _EX_NORM_SUB + 1)
        lo = (br[:-1, None] + np.diff(br)[:, None] * t[None, :-1]).ravel()
        hi = (br[:-1, None] + np.diff(br)[:, None] * t[None, 1:]).ravel()
        half = 0.5 * (hi - lo)
        z = (0.5 * (hi + lo))[:, None] + half[:, None] * _EX_GL_X[None, :]
        dc = self.comoving(z)
        f = dc * dc * self.inv_e(z)
        if time_dilation:
            f = f / (1.0 + z)
        return float(np.sum(half * (f @ _EX_GL_W)))


def _exact_distances(cosmology) -> _ExactFlatDistances:
    key = ("dist", repr(cosmology))
    obj = _EX_CACHE.get(key)
    if obj is None:
        obj = _ExactFlatDistances(cosmology)
        _EX_CACHE[key] = obj
    return obj


def usf_prob_exact(dL_mpc, cosmology, dmin: float, dmax: float, *,
                   time_dilation: bool = True):
    """Exact UniformSourceFrame (``time_dilation=False``: UniformComovingVolume)
    density in dL [1/Mpc], normalised over exactly ``[dmin, dmax]``.

    ``p(dL) = [D_C^2 / (E(z) (1+z))] / (dD_L/dz) / Z``,
    ``Z = INT_{z(dmin)}^{z(dmax)} D_C^2 / (E (1+z)) dz``, with every distance
    and ``z(dL)`` computed as in :class:`_ExactFlatDistances` -- no
    interpolation anywhere in the returned value.  The formula is evaluated
    for every ``dL >= 0`` (not truncated at the bounds); ``dL < 0`` and
    non-finite ``dL`` give NaN, so a validator sees them.
    """
    dL = np.asarray(dL_mpc, dtype=float)
    dmin, dmax = float(dmin), float(dmax)
    if not (dmax > dmin >= 0.0):
        raise ValueError(
            f"exact distance prior: need dmax > dmin >= 0, got "
            f"[{dmin!r}, {dmax!r}] Mpc.")
    ex = _exact_distances(cosmology)
    zkey = ("norm", repr(cosmology), dmin, dmax, bool(time_dilation))
    Z = _EX_CACHE.get(zkey)
    if Z is None:
        zlo, zhi = ex.z_of_dL(np.array([dmin, dmax]))
        Z = ex.norm(float(zlo), float(zhi), time_dilation=time_dilation)
        if not (Z > 0 and np.isfinite(Z)):
            raise ValueError(
                f"exact distance prior: [{dmin:.6g}, {dmax:.6g}] Mpc encloses "
                f"no normalisable probability.")
        _EX_CACHE[zkey] = Z
    out = np.full(dL.shape, np.nan, dtype=float)
    ok = np.isfinite(dL) & (dL >= 0.0)
    if ok.any():
        z = ex.z_of_dL(dL[ok])
        out[ok] = ex.density_shape(z, time_dilation=time_dilation) / Z
    return out


#: Near-zero refinement of the fallback's redshift grid (GW-29).  A log-(1+z)
#: grid is uniformly spaced in ``z`` near the origin: its step is
#: ``h = log(1+zmax)/(ngrid-1) ~ 6e-4`` (``dL ~ 2.6 Mpc``) for the default
#: ``zmax=10, ngrid=4000`` and a Planck15-like cosmology, so its RELATIVE bin
#: width ``h/z`` blows up as ``z -> 0``.  Since ``p(dL) propto dL**2`` there,
#: linear interpolation across those first bins overestimates the density by
#: ``(dL-a)(b-dL)/dL**2`` -- measured against the exact density: +165% at 1 Mpc,
#: +32% at 2 Mpc, +2.8% at 5 Mpc, +1.2% at 10 Mpc.  That is squarely inside the
#: *supported* prior-bound range: LVK cosmo files record ``dmin`` down to 1 Mpc.
#:
#: The cure is a geometric segment that holds the relative bin width at
#: :data:`_USF_NEAR_ZERO_STEP` -- bounding the relative interpolation error of a
#: quadratic at ``~step**2/4`` -- from :data:`_USF_NEAR_ZERO_ZMIN`
#: (``dL ~ 4e-7 Mpc``, below the 1e-6 Mpc floor
#: :func:`uniform_source_frame_prob` clamps its evaluation range to) up to the
#: redshift where the backbone is already that fine.  Every backbone node is
#: kept, so the density above the switch point is byte-for-byte what it was.
_USF_NEAR_ZERO_ZMIN = 1e-10
_USF_NEAR_ZERO_STEP = 0.01


def _usf_z_grid(zmax: float, ngrid: int) -> np.ndarray:
    """Hybrid redshift grid: geometric near ``z=0``, log-spaced in ``(1+z)``
    above -- see :data:`_USF_NEAR_ZERO_ZMIN` for why the origin needs it.
    """
    backbone = np.expm1(np.linspace(np.log(1.0), np.log(1.0 + zmax), ngrid))
    if backbone.size < 2:
        return backbone
    # h is the backbone's first non-zero node, i.e. its (uniform) step near the
    # origin.  Below z_switch its relative step h/z exceeds the target; above,
    # the backbone is already that fine and is left alone.
    h = float(backbone[1])
    z_switch = min(h / _USF_NEAR_ZERO_STEP, float(backbone[-1]))
    if z_switch <= _USF_NEAR_ZERO_ZMIN:
        return backbone
    n = 1 + int(np.ceil(np.log(z_switch / _USF_NEAR_ZERO_ZMIN)
                        / np.log1p(_USF_NEAR_ZERO_STEP)))
    near = np.geomspace(_USF_NEAR_ZERO_ZMIN, z_switch, n)
    return np.unique(np.concatenate(([0.0], near, backbone)))


def _usf_grid(cosmology, zmax: float, ngrid: int, *,
              time_dilation: bool = True):
    """(dL, p(dL)) on the hybrid z grid, up to an arbitrary constant.

    ``E(z)`` comes from the cosmology object (``efunc``) rather than a
    hardcoded matter+Lambda form, so a cosmology carrying radiation, neutrinos
    or curvature is handled correctly.  For a bare ``FlatLambdaCDM(H0, Om0)``
    -- what :func:`make_cosmology` builds -- the two agree exactly.

    ``time_dilation=True`` gives UniformSourceFrame (uniform in comoving volume
    *and source-frame time*, the extra ``1/(1+z)``); ``False`` gives
    UniformComovingVolume.
    """
    z = _usf_z_grid(zmax, ngrid)
    DC = cosmology.comoving_distance(z).to(u.Mpc).value
    E = np.asarray(cosmology.efunc(z), dtype=float)
    dH = float(cosmology.hubble_distance.to(u.Mpc).value)
    dDC_dz = dH / E
    dL_grid = (1 + z) * DC
    ddL_dz = DC + (1 + z) * dDC_dz
    # p(z) propto comoving-volume element [* time dilation]
    pz = DC ** 2 / E
    if time_dilation:
        pz = pz / (1 + z)
    # change of variables to dL
    return dL_grid, pz / ddL_dz


def _usf_prob_astropy(dL, cosmology, dmin, dmax, ngrid: int = 4000, *,
                      time_dilation: bool = True):
    """Astropy fallback: p(dL) propto dVc/dz [* 1/(1+z)] * |dz/ddL|, normalised
    on [dmin, dmax].

    The z grid is extended until it covers ``dmax`` so that no sample is
    silently interpolated to zero off the end of the grid, and refined near
    ``z=0`` (see :func:`_usf_z_grid`) so that the *shape* is accurate to ~1e-4
    over the whole supported prior-bound range -- LVK cosmo files record
    ``dmin`` down to 1 Mpc, where a pure log-(1+z) grid was 165% high.
    """
    dL = np.asarray(dL, dtype=float)
    zmax = 10.0
    dL_grid, p_dL_grid = _usf_grid(cosmology, zmax, ngrid,
                                   time_dilation=time_dilation)
    while dL_grid[-1] < dmax and zmax < 1e4:
        zmax *= 2.0
        dL_grid, p_dL_grid = _usf_grid(cosmology, zmax, ngrid,
                                       time_dilation=time_dilation)
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


# --------------------------------------------------------------------------
# Distance-prior dispatch by distribution CLASS (GW-02)
# --------------------------------------------------------------------------
#: Distance-prior classes gwcat can evaluate.  These are the bilby class names
#: that appear verbatim at the head of an LVK analytic ``luminosity_distance``
#: prior repr.  Anything else must fail loudly rather than be silently treated
#: as UniformSourceFrame -- which is exactly the GW-02 defect: the parser read
#: only ``minimum``/``maximum``/``cosmology``, so a ``PowerLaw(alpha=2, ...)``
#: was stored with a comoving-volume density.
DL_PRIOR_KINDS = ("UniformSourceFrame", "UniformComovingVolume",
                  "PowerLaw", "Uniform")

#: Which kinds need a cosmology, and which need an ``alpha``.
DL_PRIOR_NEEDS_COSMOLOGY = ("UniformSourceFrame", "UniformComovingVolume")
DL_PRIOR_NEEDS_ALPHA = ("PowerLaw",)


class DistancePriorKindError(ValueError):
    """An analytic distance prior names a class gwcat cannot evaluate."""


def power_law_dL_prob(dL_mpc, alpha: float, dmin: float, dmax: float):
    """p(dL) proportional to ``dL**alpha``, normalised on ``[dmin, dmax]``.

    Cosmology-independent and exact in closed form, so there is no
    implementation choice and no grid.  Following GW-01 the density is evaluated
    everywhere rather than truncated at the bounds; the normalisation over
    ``[dmin, dmax]`` is a per-event constant that cancels downstream.

    ``alpha = 2`` is what GWTC-2.1/GWTC-3 record for ``luminosity_distance``
    (uniform in Euclidean volume), which is NOT the same density as
    UniformSourceFrame under any cosmology.
    """
    dL = np.asarray(dL_mpc, dtype=float)
    a = float(alpha)
    lo, hi = float(dmin), float(dmax)
    if not (hi > lo >= 0.0):
        raise ValueError(
            f"power_law_dL_prob: need dmax > dmin >= 0, got [{lo!r}, {hi!r}].")

    if np.isclose(a, -1.0):
        if lo <= 0.0:
            raise ValueError(
                "power_law_dL_prob: alpha == -1 is not normalisable with "
                f"dmin = {lo!r}.")
        norm = np.log(hi / lo)
    else:
        norm = (hi ** (a + 1.0) - lo ** (a + 1.0)) / (a + 1.0)
    if not (norm > 0):
        raise ValueError(
            f"power_law_dL_prob: alpha={a!r} on [{lo:.4g}, {hi:.4g}] encloses "
            "no normalisable probability.")

    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.power(dL, a) / norm
    # dL < 0 is unphysical; return NaN rather than a sign-flipped density so the
    # export validator rejects it loudly.
    return np.where(dL >= 0.0, p, np.nan)


def dL_prior_prob(dL_mpc, *, kind: str, dmin: float, dmax: float,
                  cosmology: FlatLambdaCDM = None, alpha: float = None,
                  impl: str = USF_IMPL_DEFAULT, return_info: bool = False):
    """Evaluate the recorded analytic distance prior, dispatched on its CLASS.

    This is the single entry point ingest should use: it guarantees the density
    gwcat divides out is the one the file actually declares, instead of assuming
    UniformSourceFrame for every row.

    Raises :class:`DistancePriorKindError` for an unknown class, and
    ``ValueError`` when a class's required inputs are missing -- a row whose
    prior gwcat cannot reproduce must fail, not be approximated.
    """
    if kind not in DL_PRIOR_KINDS:
        raise DistancePriorKindError(
            f"unknown analytic distance-prior class {kind!r}; gwcat can "
            f"evaluate {list(DL_PRIOR_KINDS)}. Refusing to substitute a "
            f"different density -- the ratio p_true/p_assumed does NOT cancel "
            f"in the per-event normalisation.")
    if kind in DL_PRIOR_NEEDS_COSMOLOGY and cosmology is None:
        raise ValueError(f"{kind} requires a cosmology.")
    if kind in DL_PRIOR_NEEDS_ALPHA and alpha is None:
        raise ValueError(f"{kind} requires alpha.")

    if kind in DL_PRIOR_NEEDS_COSMOLOGY:
        return uniform_source_frame_prob(
            dL_mpc, cosmology, dmin, dmax, impl=impl, return_info=return_info,
            time_dilation=(kind == "UniformSourceFrame"))

    # Closed-form kinds: exact, no implementation choice, no widening needed
    # (the density is defined for every dL >= 0).
    dL = np.asarray(dL_mpc, dtype=float)
    a = 0.0 if kind == "Uniform" else float(alpha)
    p = power_law_dL_prob(dL, a, dmin, dmax)
    if not return_info:
        return p

    finite = dL[np.isfinite(dL)]
    n_below = int(np.sum(finite < dmin))
    n_above = int(np.sum(finite > dmax))
    n_outside = n_below + n_above
    info = {
        "impl": "analytic",
        "kind": kind,
        "alpha": a,
        "dmin": float(dmin),
        "dmax": float(dmax),
        "eval_min": float(dmin),
        "eval_max": float(dmax),
        "widened": False,
        "n_samples": int(dL.size),
        "n_below_dmin": n_below,
        "n_above_dmax": n_above,
        "n_outside_bounds": n_outside,
        "frac_outside_bounds": (n_outside / dL.size) if dL.size else 0.0,
        "n_nonfinite": int(dL.size - finite.size),
    }
    return p, info
