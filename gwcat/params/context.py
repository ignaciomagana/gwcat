"""Evaluation contexts handed to a block's density callables (GW-17).

A block never reaches into a builder's locals.  It receives a context carrying
the conditioners its density may depend on, which keeps the additive-log
contract honest: blocks factorise *given the context*.

The restriction worth stating explicitly: a block may condition on anything in
the context (``spin.chieff`` conditions on the source-frame masses, as it must),
but it may NOT depend on another block's OUTPUT.  That would require declared
block ordering, and the general case is deliberately not built.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class PEContext:
    """Per-event conditioners for the PE side.

    Arrays are per-sample for ONE event, already resampled -- so a block never
    sees the resampling and cannot perturb the rng stream.
    """
    event_name: str = ""
    m1det: Optional[np.ndarray] = None
    m2det: Optional[np.ndarray] = None
    m1src: Optional[np.ndarray] = None
    m2src: Optional[np.ndarray] = None
    dL: Optional[np.ndarray] = None
    redshift: Optional[np.ndarray] = None
    #: Per-body spin-magnitude ceilings resolved for THIS event, with the
    #: provenance of each ("analytic" | "samples_bound" | "fallback", GW-04).
    amax_1: float = float("nan")
    amax_2: float = float("nan")
    amax_source_1: str = ""
    amax_source_2: str = ""
    #: The parsed distance-prior contract for this event (GW-02).
    dL_prior_kind: str = ""
    dL_prior_alpha: Optional[float] = None
    dL_prior_bounds: tuple = ()
    #: Which implementation evaluated this event's distance prior at ingest
    #: (the store's per-row ``dL_prior_impl``: "exact", "analytic", or the
    #: legacy "bilby"/"astropy").  Empty means "unknown" and gets the default
    #: exact evaluation; a block that re-evaluates the prior uses THIS, so it
    #: reproduces the store under --legacy-grid-priors too (GW-40i).
    dL_prior_impl: str = ""
    cosmology: Any = None
    #: Parsed mass-prior class; the m1det Jacobian is valid only for
    #: "uniform_detector_frame" (GW-07).
    mass_prior_kind: str = ""
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InjContext:
    """Per-campaign conditioners for the injection side.

    ``ln_pdraw_base`` is NOT here: it is added by the builder outside the block
    loop, because it is the rate normalisation shared by every space (see
    :mod:`gwcat.params.block`).
    """
    campaign_path: str = ""
    m1det: Optional[np.ndarray] = None
    m2det: Optional[np.ndarray] = None
    m1src: Optional[np.ndarray] = None
    m2src: Optional[np.ndarray] = None
    dL: Optional[np.ndarray] = None
    redshift: Optional[np.ndarray] = None
    #: The campaign's injected-spin state (SelectionSet.spin_meta): the evidence
    #: a projection block's CampaignRequirement is checked against.
    spin_meta: Mapping[str, Any] = field(default_factory=dict)
    #: Detected per-body ceiling for THIS campaign; None when the draw is not a
    #: single uniform-magnitude distribution (GW-05 -- never fabricated).
    amax_1: Optional[float] = None
    amax_2: Optional[float] = None
    sky_position_available: bool = True
    extras: Mapping[str, Any] = field(default_factory=dict)
