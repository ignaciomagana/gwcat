"""Detector-frame component-mass block (GW-17)."""
from __future__ import annotations

import numpy as np

from ..block import ParameterBlock, Range

#: The only mass-prior class for which the exported ``m1det`` Jacobian is valid.
#: Fitting in ``(m1det, q)`` with ``m2det = q*m1det`` gives
#: ``|d(m1det,m2det)/d(m1det,q)| = m1det``, and that IS the ``p_pe = m1det *
#: p_dL_pe`` factor -- correct only because the PE mass prior is uniform in
#: ``(m1det, m2det)``.  GW-07 made that verifiable instead of assumed; the real
#: releases put the prior on ``(chirp_mass, mass_ratio)`` as
#: ``UniformInComponents*``, which IS flat in the components.
VALID_MASS_PRIOR_KINDS = ("uniform_detector_frame",)


def ln_prior_pe(cols, ctx):
    """``ln m1det`` -- the (m1det, q)-basis Jacobian, gated on the prior class."""
    if ctx.mass_prior_kind and ctx.mass_prior_kind not in VALID_MASS_PRIOR_KINDS:
        raise ValueError(
            f"mass.det_pair: the ingested mass prior is "
            f"{ctx.mass_prior_kind!r}, but the |dm2det/dq| = m1det Jacobian is "
            f"valid only for {VALID_MASS_PRIOR_KINDS[0]!r}.")
    m1 = np.asarray(cols["m1det"], dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(m1 > 0, np.log(m1), -np.inf)


def ln_draw_inj(cols, ctx):
    """Zero: the mass term lives in ``pdraw_base`` (see block.py)."""
    return 0.0


MASS_DET_PAIR = ParameterBlock(
    name="mass.det_pair",
    kind="mass",
    columns=("m1det", "m2det"),
    map_kind="bijective",
    exact_draw_density=True,
    store_required=("mass_1", "mass_2"),
    ln_prior_pe=ln_prior_pe,
    ln_draw_inj=ln_draw_inj,
    ranges={"m1det": Range(0.0, np.inf, closed="neither",
                           note="detector-frame primary mass, Msun"),
            "m2det": Range(0.0, np.inf, closed="neither",
                           note="detector-frame secondary mass, Msun")},
    notes=("The m1det factor is the (m1det, q) Jacobian, NOT a prior. It is "
           "valid only for a mass prior uniform in detector-frame component "
           "masses, which is why this block gates on mass_prior_kind."),
)
