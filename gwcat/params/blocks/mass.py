"""Detector-frame mass block (GW-17, GW-34).

The density coordinates are ``(m1det, q)``, NOT ``(m1det, m2det)``
--------------------------------------------------------------------
``p_pe`` carries the factor ``m1det``, and that factor is exactly
``|d(m1det, m2det)/d(m1det, q)| = m1det`` -- the Jacobian of the map
``m2det = q * m1det``.  A density carrying that factor is a density in
``(m1det, q)``.  The block published ``(m1det, m2det)`` as its fit columns
anyway, so a generic contract consumer reading ``fit_columns`` off a 2.1 file
would integrate the exported weights against the wrong measure.  ``q`` is
therefore the fit coordinate and ``m2det`` is advisory: still written by every
export (every consumer and every plot uses it), but DERIVED --
``m2det = q * m1det`` -- and covered by no density of its own.
"""
from __future__ import annotations

import numpy as np

from ..block import ParameterBlock, Range

#: The only mass-prior class for which the exported ``m1det`` Jacobian is
#: VERIFIED.  Fitting in ``(m1det, q)`` with ``m2det = q*m1det`` gives
#: ``|d(m1det,m2det)/d(m1det,q)| = m1det``, and that IS the ``p_pe = m1det *
#: p_dL_pe`` factor -- correct only because the PE mass prior is uniform in
#: ``(m1det, m2det)``.  GW-07 made that verifiable instead of assumed; the real
#: releases put the prior on ``(chirp_mass, mass_ratio)`` as
#: ``UniformInComponents*``, which IS flat in the components.
VALID_MASS_PRIOR_KINDS = ("uniform_detector_frame",)

#: Recorded for a store that states no mass-prior class at all (one written
#: before the GW-07 mass-prior ingest).  Distinct from ``"assumed_default"``,
#: which is a store that DID look and found no analytic prior to parse.
UNSTATED_MASS_PRIOR = "unstated"

#: Classes for which the Jacobian is ASSUMED rather than verified: the release
#: carries no analytic ``(chirp_mass, mass_ratio)`` prior to parse
#: (``"assumed_default"`` -- 9 of the 282 rows in the shipped store), or the
#: store predates the mass-prior ingest and states nothing.  These still export
#: -- refusing a row because its release shipped no priors group would throw
#: away real events over a bookkeeping gap -- but they must NOT be stamped with
#: the verified basis, which is exactly what every export did before GW-34.
ASSUMED_MASS_PRIOR_KINDS = ("assumed_default", UNSTATED_MASS_PRIOR, "")


def classify_mass_prior(kind) -> str:
    """``"verified"`` | ``"assumed"`` | ``"unsupported"`` for one prior class.

    The single rule behind both the block's gate and the basis a builder stamps
    on the file, so "what the Jacobian requires" and "what the file claims"
    cannot drift apart again.
    """
    k = "" if kind is None else str(kind)
    if k in VALID_MASS_PRIOR_KINDS:
        return "verified"
    if k in ASSUMED_MASS_PRIOR_KINDS:
        return "assumed"
    return "unsupported"


def check_mass_prior(ctx) -> str:
    """The block's safety gate: refuse a prior the ``m1det`` factor misdescribes.

    Returns the classification (``"verified"`` or ``"assumed"``) so a caller can
    record WHICH it got; raises for a parsed prior that is not flat in the
    detector-frame components, because there the factor is not the event's
    Jacobian at all and the error does not cancel in any normalisation.
    """
    kind = "" if ctx is None else (getattr(ctx, "mass_prior_kind", "") or "")
    state = classify_mass_prior(kind)
    if state == "unsupported":
        where = getattr(ctx, "event_name", "") or ""
        where = f" (event {where})" if where else ""
        raise ValueError(
            f"mass.det_pair{where}: the ingested mass prior is {kind!r}, but "
            f"the |dm2det/dq| = m1det Jacobian is valid only for "
            f"{VALID_MASS_PRIOR_KINDS[0]!r}.")
    return state


def prior_pe_factor(cols, ctx):
    """``m1det`` -- the (m1det, q)-basis Jacobian, gated on the prior class.

    Linear space, and the stored array itself rather than a rebuilt one: the
    chieff PE export is contractually byte-identical to the frozen v1 exporter,
    and ``exp(ln_prior_pe)`` differs from ``m1det`` by an ulp for most samples.
    """
    check_mass_prior(ctx)
    return np.asarray(cols["m1det"], dtype=float)


def ln_prior_pe(cols, ctx):
    """``ln m1det`` -- the same Jacobian, in the log the block contract adds in."""
    m1 = prior_pe_factor(cols, ctx)
    with np.errstate(divide="ignore"):
        return np.where(m1 > 0, np.log(m1), -np.inf)


def ln_draw_inj(cols, ctx):
    """Zero: the mass term lives in ``pdraw_base`` (see block.py)."""
    return 0.0


MASS_DET_PAIR = ParameterBlock(
    name="mass.det_pair",
    kind="mass",
    columns=("m1det", "q"),
    advisory_columns=("m2det",),
    map_kind="bijective",
    exact_draw_density=True,
    store_required=("mass_1", "mass_2"),
    ln_prior_pe=ln_prior_pe,
    prior_pe_factor=prior_pe_factor,
    ln_draw_inj=ln_draw_inj,
    ranges={"m1det": Range(0.0, np.inf, closed="neither",
                           note="detector-frame primary mass, Msun"),
            "q": Range(0.0, 1.0, closed="right",
                       note="mass ratio m2det/m1det -- THE density coordinate: "
                            "the m1det factor in p_pe is its Jacobian"),
            "m2det": Range(0.0, np.inf, closed="neither",
                           note="detector-frame secondary mass, Msun; DERIVED "
                                "(m2det = q * m1det) and advisory")},
    notes=("The m1det factor is the (m1det, q) Jacobian, NOT a prior. It is "
           "valid only for a mass prior uniform in detector-frame component "
           "masses, which is why this block gates on mass_prior_kind -- and why "
           "q, not m2det, is the published density coordinate. A file writes "
           "both; m2det is derived (q * m1det) and carries no density of its "
           "own. A class the gate does not recognise is refused; one that was "
           "never parsed exports as 'assumed', never as the verified basis."),
)
