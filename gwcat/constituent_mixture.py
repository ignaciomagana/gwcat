"""The prior of a combined ``C00:Mixed`` sample set (GW-40f).

GWTC-4.1 ``C00:Mixed`` is an EQUAL-COUNT concatenation of its constituent
analyses (e.g. GW230606_004305: 41919 = 3 x 13973 rows), and it carries no
``priors`` group.  Its constituents do not share one prior: NRSur7dq4 samples
``q >= 1/6`` while IMRPhenomXPHM-SpinTaylor and SEOBNRv5PHM sample
``q >= 0.05``.  Borrowing one sibling's smooth analytic prior (what the ingest
does for a ``Mixed`` set by default) therefore misses the step at ``q = 1/6``
and the per-analysis normalisation.

WHICH prior describes an equal-count mixture is a modelling choice, not a
consequence of the equal counts (review 2026-10-20).  Each constituent k is a
posterior ``L_k pi_k / E_k`` with its own likelihood ``L_k`` (its waveform) and
evidence ``E_k``, so the concatenation does not have the form ``L x pi`` for
one ``pi``.  Two readings are implemented, selected by ``convention``
(:data:`MIXTURE_CONVENTIONS`) and recorded in the store's prior sources:

* ``"equal_weight"`` (the default, unchanged): ``pi_Mixed = (1/K) sum_k pi_k``
  evaluated at EVERY sample.  It is NOT the prior implied by a common
  likelihood -- that is proportional to ``sum_k pi_k / E_k`` (inverse-evidence
  weights; not implemented, since it needs every constituent's evidence under
  ONE likelihood, which the release does not provide) -- and it sets the
  ``q = 1/6`` step height from prior areas.
* ``"own_prior"``: each row divided by the prior of the constituent it came
  from, ``pi_{k(row)}`` (waveform-model averaging: every row stays a sample of
  its own constituent's posterior).

Choosing between them is the owner's decision; neither is the default by
derivation.  This module computes them for the mass and distance priors, in
the export's coordinates ``(m1det, q, dL)``:

* mass -- ``UniformInComponentsChirpMass`` x ``UniformInComponentsMassRatio``
  (flat in the detector-frame components) restricted to its chirp-mass and
  mass-ratio bounds and EVERY ``Constraint`` its analytic group declares --
  ``mass_1``, ``mass_2`` and ``total_mass`` (GWTC-4.1/5 NRSur7dq4 declares
  ``Constraint(60 <= total_mass <= 400)`` in the detector frame); a
  ``Constraint`` on any other key is refused, never dropped
  (:func:`mass_constraint_bounds`) -- i.e. ``pi_k(m1, q) = m1 / Z_k`` on the
  support, with ``Z_k`` the area of that support in ``(m1, m2)``
  (:func:`uic_normalisation`);
* distance -- each constituent's own declared prior, normalised on its own
  bounds and zero outside them.

The spin prior must be the SAME for every constituent (it is: U(0, 0.99)
isotropic in every GWTC-4.1 constituent), so it factors out and the exporter's
``p(chi_eff | q, amax)`` applies unchanged; a mixture whose constituents
disagree on it is refused rather than approximated.

The mixing fractions are VERIFIED, not assumed: every Mixed row must equal one
constituent row exactly, and each contributing constituent must supply exactly
``n_Mixed / K`` of them (:func:`match_constituent_rows`).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

#: Columns whose exact float64 values identify a posterior row.
MATCH_KEYS = ("mass_1", "mass_2", "luminosity_distance")

#: The mixture-prior conventions (see the module docstring).  The first is the
#: default and reproduces every store built before the option existed.
MIXTURE_CONVENTIONS = ("equal_weight", "own_prior")


class ConstituentMixtureError(ValueError):
    """A combined sample set's mixture prior cannot be built or verified."""


def _chirp_factor(q):
    """``m1 / Mc`` at mass ratio ``q``: ``(1+q)^(1/5) / q^(3/5)``."""
    q = np.asarray(q, dtype=float)
    return (1.0 + q) ** 0.2 / q ** 0.6


#: The ``Constraint`` keys the mass support implements, and the bound names
#: each one fills.  All are detector-frame masses, like the samples.
MASS_CONSTRAINT_KEYS = {
    "mass_1": ("m1_min", "m1_max"),
    "mass_2": ("m2_min", "m2_max"),
    "total_mass": ("mt_min", "mt_max"),
}


def mass_constraint_bounds(node, parse) -> dict:
    """Every ``Constraint`` of one analytic prior group, as support bounds.

    ``node`` is the group (``{key: repr}``) and ``parse`` the repr parser
    returning ``(kind, minimum, maximum)`` or None (the ingest's
    ``_parse_analytic_spin``).  Returns ``{m1_min, m1_max, m2_min, m2_max,
    mt_min, mt_max}`` (None where undeclared).  A ``Constraint`` on a key the
    support does not implement raises :class:`ConstituentMixtureError`: a
    dropped constraint leaves the normalisation and the support silently wrong
    (the ``total_mass`` cut of NRSur7dq4 was dropped exactly that way).
    """
    out = {b: None for pair in MASS_CONSTRAINT_KEYS.values() for b in pair}
    unknown = []
    for key, rep in (node or {}).items():
        got = parse(rep)
        if got is None or got[0] != "Constraint":
            continue
        if key not in MASS_CONSTRAINT_KEYS:
            unknown.append(key)
            continue
        lo_name, hi_name = MASS_CONSTRAINT_KEYS[key]
        out[lo_name] = None if got[1] is None else float(got[1])
        out[hi_name] = None if got[2] is None else float(got[2])
    if unknown:
        raise ConstituentMixtureError(
            f"the analytic prior declares Constraint(s) on {sorted(unknown)}, "
            f"which the mixture support does not implement (only "
            f"{sorted(MASS_CONSTRAINT_KEYS)}); refusing rather than dropping "
            f"them.")
    return out


def _m1_limits(q, mc_min, mc_max, m1_min=None, m1_max=None, m2_min=None,
               m2_max=None, mt_min=None, mt_max=None):
    """``[a(q), b(q)]``: the m1 interval inside the support at each ``q``."""
    q = np.asarray(q, dtype=float)
    f = _chirp_factor(q)
    lo = np.asarray(mc_min * f, dtype=float)
    hi = np.asarray(mc_max * f, dtype=float)
    if m1_min is not None:
        lo = np.maximum(lo, m1_min)
    if m1_max is not None:
        hi = np.minimum(hi, m1_max)
    if m2_min is not None:
        lo = np.maximum(lo, m2_min / q)
    if m2_max is not None:
        hi = np.minimum(hi, m2_max / q)
    if mt_min is not None:
        lo = np.maximum(lo, mt_min / (1.0 + q))
    if mt_max is not None:
        hi = np.minimum(hi, mt_max / (1.0 + q))
    return lo, hi


def uic_normalisation(mc_min, mc_max, q_min, q_max, m1_min=None, m1_max=None,
                      m2_min=None, m2_max=None, mt_min=None, mt_max=None,
                      n_q=200001) -> float:
    """Area of the uniform-in-components support in ``(m1, m2)``.

    ``Z = integral over q in [q_min, q_max] of (b(q)^2 - a(q)^2) / 2 dq``, which
    is ``integral m1 dm1 dq`` -- the ``(m1, m2) -> (m1, q)`` Jacobian is
    ``m1``.  1-D quadrature on a dense grid; the integrand is continuous and
    piecewise smooth, so the relative error is ~1e-9 at the default ``n_q``.
    """
    for nm, v in (("mc_min", mc_min), ("mc_max", mc_max), ("q_min", q_min),
                  ("q_max", q_max)):
        if v is None or not np.isfinite(v):
            raise ConstituentMixtureError(
                f"uniform-in-components prior needs a finite {nm}; got {v!r}.")
    if not (0 < q_min < q_max <= 1 and 0 < mc_min < mc_max):
        raise ConstituentMixtureError(
            f"invalid uniform-in-components bounds: Mc [{mc_min}, {mc_max}], "
            f"q [{q_min}, {q_max}].")
    q = np.linspace(q_min, q_max, int(n_q))
    a, b = _m1_limits(q, mc_min, mc_max, m1_min, m1_max, m2_min, m2_max,
                      mt_min, mt_max)
    integrand = np.where(b > a, 0.5 * (b * b - a * a), 0.0)
    Z = float(_trapz(integrand, q))
    if not Z > 0:
        raise ConstituentMixtureError(
            "uniform-in-components support has zero area; check the bounds.")
    return Z


def uic_density_m1q(m1, q, *, mc_min, mc_max, q_min, q_max, Z, m1_min=None,
                    m1_max=None, m2_min=None, m2_max=None, mt_min=None,
                    mt_max=None):
    """``pi(m1det, q) = m1det / Z`` inside the support, exactly 0 outside."""
    m1 = np.asarray(m1, dtype=float)
    q = np.asarray(q, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mc = m1 / _chirp_factor(q)
    inside = (q >= q_min) & (q <= q_max) & (mc >= mc_min) & (mc <= mc_max)
    if m1_min is not None:
        inside &= m1 >= m1_min
    if m1_max is not None:
        inside &= m1 <= m1_max
    if m2_min is not None:
        inside &= q * m1 >= m2_min
    if m2_max is not None:
        inside &= q * m1 <= m2_max
    if mt_min is not None:
        inside &= m1 * (1.0 + q) >= mt_min
    if mt_max is not None:
        inside &= m1 * (1.0 + q) <= mt_max
    return np.where(inside, m1 / Z, 0.0)


def _row_keys(samples: dict, keys=MATCH_KEYS):
    cols = [np.ascontiguousarray(np.asarray(samples[k], dtype=np.float64))
            for k in keys]
    stacked = np.ascontiguousarray(np.stack(cols, axis=1))
    return [r.tobytes() for r in stacked]


def match_constituent_rows(mixed: dict, constituents: Dict[str, dict],
                           keys: Sequence[str] = MATCH_KEYS,
                           return_rows: bool = False):
    """Assign every Mixed row to the constituent row it equals, exactly.

    Returns ``(labels, counts)``: the contributing constituent labels (file
    order) and their row counts; with ``return_rows=True`` also, third, the
    index into ``labels`` of the constituent each Mixed row came from (what
    the ``"own_prior"`` convention needs).  Raises :class:`ConstituentMixtureError`
    unless EVERY Mixed row matches exactly one constituent row and each
    contributing constituent supplies exactly ``n_Mixed / K`` rows -- the
    row-by-row proof that the mixing fractions are the equal weights the prior
    assumes.
    """
    missing = [k for k in keys if k not in mixed]
    if missing:
        raise ConstituentMixtureError(
            f"cannot match Mixed rows: columns {missing} are absent.")
    owner: Dict[bytes, str] = {}
    ambiguous = 0
    for lab, smp in constituents.items():
        if any(k not in smp for k in keys):
            continue
        for kb in _row_keys(smp, keys):
            prev = owner.get(kb)
            if prev is not None and prev != lab:
                ambiguous += 1
            owner[kb] = lab
    if ambiguous:
        raise ConstituentMixtureError(
            f"{ambiguous} constituent row(s) are identical across two "
            f"analyses; the Mixed rows cannot be attributed.")
    mk = _row_keys(mixed, keys)
    counts: Dict[str, int] = {}
    unmatched = 0
    for kb in mk:
        lab = owner.get(kb)
        if lab is None:
            unmatched += 1
        else:
            counts[lab] = counts.get(lab, 0) + 1
    n = len(mk)
    if unmatched:
        raise ConstituentMixtureError(
            f"{unmatched} of {n} Mixed row(s) equal no constituent row; the "
            f"combined set is not a concatenation of {sorted(constituents)}.")
    labels = [lab for lab in constituents if counts.get(lab, 0)]
    K = len(labels)
    if K == 0:
        raise ConstituentMixtureError("no constituent contributes any row.")
    if n % K or any(counts[lab] != n // K for lab in labels):
        raise ConstituentMixtureError(
            f"the Mixed set's mixing fractions are not equal: rows per "
            f"constituent {dict((lab, counts[lab]) for lab in labels)} of "
            f"{n} (expected {n / K:g} each). An equal-weight mixture prior "
            f"would misdescribe it.")
    counts_out = {lab: counts[lab] for lab in labels}
    if not return_rows:
        return labels, counts_out
    pos = {lab: i for i, lab in enumerate(labels)}
    rows = np.fromiter((pos[owner[kb]] for kb in mk), dtype=np.int64,
                       count=n)
    return labels, counts_out, rows


def mixture_prior_densities(m1, q, dL, components: Sequence[dict],
                            convention: str = "equal_weight",
                            row_component=None):
    """``(p_mass_dL, p_dL)`` of the mixture prior, per sample.

    ``convention`` is one of :data:`MIXTURE_CONVENTIONS`.  ``"own_prior"``
    needs ``row_component``, the index into ``components`` of the constituent
    each sample came from (:func:`match_constituent_rows` with
    ``return_rows=True``), and gives each sample its own constituent's
    ``pi_k(m1, q) pi_k(dL)`` and ``pi_k(dL)``.  The rest of this docstring
    describes the default ``"equal_weight"``.

    Each component is a dict with the mass bounds (``mc_min, mc_max, q_min,
    q_max, m1_min, m1_max, m2_min, m2_max, mt_min, mt_max``), ``Z`` and a
    callable ``p_dL(dL) -> density`` (normalised on its own bounds, 0 outside).
    ``p_mass_dL`` is the JOINT mixture ``(1/K) sum_k pi_k(m1, q) pi_k(dL)`` --
    not the product of the two marginal mixtures, which differ whenever the
    constituents' distance priors differ -- and ``p_dL`` is the marginal
    distance mixture, kept for provenance.
    """
    K = len(components)
    if K == 0:
        raise ConstituentMixtureError("no components.")
    if convention not in MIXTURE_CONVENTIONS:
        raise ConstituentMixtureError(
            f"convention={convention!r}; choose one of {MIXTURE_CONVENTIONS}.")
    if convention == "own_prior":
        if row_component is None:
            raise ConstituentMixtureError(
                "convention='own_prior' needs row_component (which constituent "
                "each sample came from).")
        row_component = np.asarray(row_component)
        if (row_component.shape != np.shape(m1)
                or not np.all((row_component >= 0) & (row_component < K))):
            raise ConstituentMixtureError(
                f"row_component must give a component index in [0, {K}) for "
                f"every sample.")
    joint = np.zeros(np.shape(m1), dtype=float)
    marg = np.zeros(np.shape(m1), dtype=float)
    for k, c in enumerate(components):
        pm = uic_density_m1q(
            m1, q, mc_min=c["mc_min"], mc_max=c["mc_max"], q_min=c["q_min"],
            q_max=c["q_max"], Z=c["Z"], m1_min=c.get("m1_min"),
            m1_max=c.get("m1_max"), m2_min=c.get("m2_min"),
            m2_max=c.get("m2_max"), mt_min=c.get("mt_min"),
            mt_max=c.get("mt_max"))
        pd = np.asarray(c["p_dL"](dL), dtype=float)
        if convention == "own_prior":
            mine = row_component == k
            joint = np.where(mine, pm * pd, joint)
            marg = np.where(mine, pd, marg)
            continue
        joint += pm * pd / K
        marg += pd / K
    return joint, marg


def truncated_dL_density(kind, cosmology, dmin, dmax, alpha=None,
                         impl="exact"):
    """A constituent's distance prior normalised on ITS bounds, 0 outside.

    Unlike the single-prior ingest path (GW-01), which widens the evaluation
    range to cover every sample, a mixture component must keep its own
    normalisation: a sample from another constituent outside this one's bounds
    genuinely has zero density under it.
    """
    from .cosmology import dL_prior_prob

    def _p(dL):
        dL = np.asarray(dL, dtype=float)
        out = np.zeros(dL.shape, dtype=float)
        inside = (dL >= dmin) & (dL <= dmax)
        if inside.any():
            out[inside] = dL_prior_prob(dL[inside], kind=kind,
                                        cosmology=cosmology, dmin=dmin,
                                        dmax=dmax, alpha=alpha, impl=impl)
        return out
    return _p


def sample_q(samples: dict) -> Optional[np.ndarray]:
    """``m2/m1`` from a sample dict, or None when the masses are absent."""
    if "mass_1" not in samples or "mass_2" not in samples:
        return None
    return (np.asarray(samples["mass_2"], dtype=float)
            / np.asarray(samples["mass_1"], dtype=float))
