"""Luminosity-distance block (GW-17).

The density is dispatched on the PARSED prior class (GW-02), not assumed to be
UniformSourceFrame, and is evaluated everywhere rather than truncated at the
recorded bounds (GW-01).  Both were real defects: 81 of 282 rows declare
``PowerLaw(alpha=2)``, and truncation shipped an exact ``p_pe = 0``.
"""
from __future__ import annotations

import numpy as np

from ...cosmology import (DL_PRIOR_KINDS, USF_IMPL_DEFAULT, USF_IMPLS,
                          dL_prior_prob)
from ..block import ParameterBlock, Range


def ln_prior_pe(cols, ctx):
    """``ln p(dL)`` for the event's own declared prior class."""
    dL = np.asarray(cols["dL"], dtype=float)
    kind = ctx.dL_prior_kind or "UniformSourceFrame"
    if kind not in DL_PRIOR_KINDS:
        raise ValueError(
            f"distance.dl: unknown distance-prior class {kind!r}; gwcat can "
            f"evaluate {list(DL_PRIOR_KINDS)}. Refusing to substitute a "
            f"different density -- the ratio p_true/p_assumed does not cancel "
            f"in the per-event normalisation.")
    lo, hi = (ctx.dL_prior_bounds if len(ctx.dL_prior_bounds) == 2
              else (float(np.nanmin(dL)), float(np.nanmax(dL))))
    # The implementation the STORE used for this row (GW-40i): exact by
    # default, the legacy interpolation only when the row was ingested with it
    # ("analytic" -- a closed-form class -- and "" take the default, which the
    # closed-form classes ignore anyway).
    impl = (ctx.dL_prior_impl if ctx.dL_prior_impl in USF_IMPLS
            else USF_IMPL_DEFAULT)
    p = dL_prior_prob(dL, kind=kind, cosmology=ctx.cosmology, dmin=lo, dmax=hi,
                      alpha=ctx.dL_prior_alpha, impl=impl)
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(p > 0, np.log(p), -np.inf)


def ln_draw_inj(cols, ctx):
    """Zero: the distance term lives in ``pdraw_base`` (see block.py)."""
    return 0.0


DISTANCE_DL = ParameterBlock(
    name="distance.dl",
    kind="distance",
    columns=("dL",),
    advisory_columns=("redshift",),
    map_kind="bijective",
    exact_draw_density=True,
    store_required=("luminosity_distance", "p_dL_pe"),
    ln_prior_pe=ln_prior_pe,
    ln_draw_inj=ln_draw_inj,
    ranges={"dL": Range(0.0, np.inf, closed="neither", note="Mpc"),
            "redshift": Range(0.0, np.inf, closed="left")},
    notes=("Dispatched on the parsed prior CLASS and evaluated everywhere -- "
           "not truncated at the recorded bounds, which routinely come from a "
           "sibling analysis. redshift is advisory: it is derived from dL under "
           "the event's cosmology and carries no independent density."),
)
