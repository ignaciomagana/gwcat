"""Sky-position block (GW-17)."""
from __future__ import annotations

import numpy as np

from ..block import CampaignRequirement, ParameterBlock, Range

_LN_4PI = float(np.log(4.0 * np.pi))


def ln_prior_pe(cols, ctx):
    """Isotropic sky: ``-ln 4pi``, a constant.

    Declared explicitly rather than omitted.  It cancels between p_pe and pdraw
    only if BOTH sides carry it, and "it cancels" was previously an assumption
    nobody had written down -- the one finding the review left unresolved.
    """
    return -_LN_4PI


def ln_draw_inj(cols, ctx):
    """The same constant on the injection side, so the cancellation is explicit."""
    return -_LN_4PI


SKY_RADEC = ParameterBlock(
    name="sky.radec",
    kind="sky",
    columns=("ra", "dec"),
    map_kind="bijective",
    exact_draw_density=True,
    prior_is_per_event_constant=True,
    requires_campaign=CampaignRequirement(needs_sky_draw=True),
    store_required=("ra", "dec"),
    ln_prior_pe=ln_prior_pe,
    ln_draw_inj=ln_draw_inj,
    ranges={"ra": Range(0.0, 2.0 * np.pi, closed="left",
                        note="radians; a degrees ingest passes silently today"),
            "dec": Range(-np.pi / 2.0, np.pi / 2.0,
                         note="radians, NOT colatitude")},
    notes=("Sky is in every space unconditionally. The measure is declared on "
           "both sides so the cancellation is checkable rather than assumed; "
           "the ranges exist because a degrees/radians or colatitude mistake "
           "otherwise yields a plausible-looking but wrong pixelisation."),
)
