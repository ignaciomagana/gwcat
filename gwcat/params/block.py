"""The ``ParameterBlock`` abstraction (GW-17, design PR-A).

A **block** is a group of fit coordinates that travels with its own density on
*both* sides of the likelihood, its own store requirements, its own support, and
its own provenance.  A **space** is an ordered list of blocks.

The density contract is exactly additive in the log::

    PE:         ln p_pe  = sum_blocks ln_prior_pe
    selection:  ln pdraw = ln pdraw_base + sum_blocks ln_draw_inj

``pdraw_base`` -- the campaign's spin-free draw density in ``(m1det, q, dL)``,
already divided by ``T_obs`` and the injection weights -- is deliberately **not**
a block.  It is the rate normalisation plus the coordinate Jacobian, identical
for every parameter space, and leaving it untouched is what preserves both the
chieff byte-parity contract and the Essick per-campaign fractions.  Mass and
distance blocks therefore contribute ``0`` on the injection side and carry their
terms only on the PE side, where there is no analogous base.

Two rules this abstraction exists to make declarative rather than incidental
---------------------------------------------------------------------------

**R1 (projection rule).**  ``chi_eff`` and ``chi_p`` are many-to-one projections
of the 4-D spin vector.  Converting a draw density in
``(a1, a2, cos_tilt_1, cos_tilt_2)`` into a density in ``chi_eff`` means
integrating out three spin degrees of freedom *at fixed chi_eff*, and that
integral has a closed form **only** for the uniform-magnitude / isotropic parent.
A projection block therefore carries a validity predicate and must **refuse**,
not approximate, when the parent is something else.  A bijective block
transforms pointwise by a Jacobian and assumes nothing about the parent.

**R2 (support rule).**  A density that appears in a **denominator** may never be
floored.  ``clip(ln p, -50, None)`` turns "impossible under this prior" into
"weight ~ 1e21".  Out-of-support points get a recorded mask, not a floor.

Both rules are enforced elsewhere (GW-03, GW-05, GW-06); this module is where
they become a *declared property of each block* rather than a rule remembered at
each call site.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Tuple

import numpy as np

#: Block kinds, in canonical order.
KINDS = ("mass", "distance", "sky", "spin", "tidal")

#: How a block's coordinates relate to the underlying sampled parameters.
MAP_KINDS = ("bijective", "projection")


@dataclass(frozen=True)
class Range:
    """A physical range for one exported column, used by the validator."""
    lo: float
    hi: float
    closed: str = "both"          # "both" | "left" | "right" | "neither"
    note: str = ""

    def contains(self, x):
        x = np.asarray(x, dtype=float)
        lo_ok = x >= self.lo if self.closed in ("both", "left") else x > self.lo
        hi_ok = x <= self.hi if self.closed in ("both", "right") else x < self.hi
        return np.asarray(lo_ok & hi_ok)

    def describe(self) -> dict:
        return {"lo": float(self.lo), "hi": float(self.hi),
                "closed": self.closed, "note": self.note}


@dataclass(frozen=True)
class CampaignRequirement:
    """What must be TRUE of an injection campaign for a block's ``ln_draw_inj``
    to be that campaign's ACTUAL density.

    A block whose ``ln_draw_inj`` is read numerically off the file leaves every
    field ``False`` and accepts any campaign -- which is precisely why the
    component basis is exact for the non-uniform-isotropic O4 sets while the
    projections are not.
    """
    uniform_magnitude: bool = False
    isotropic_tilts: bool = False
    single_amax: bool = False
    uniform_azimuth: bool = False
    needs_sky_draw: bool = False

    @property
    def any_required(self) -> bool:
        return any((self.uniform_magnitude, self.isotropic_tilts,
                    self.single_amax, self.uniform_azimuth,
                    self.needs_sky_draw))

    def describe(self) -> dict:
        return {"uniform_magnitude": self.uniform_magnitude,
                "isotropic_tilts": self.isotropic_tilts,
                "single_amax": self.single_amax,
                "uniform_azimuth": self.uniform_azimuth,
                "needs_sky_draw": self.needs_sky_draw}

    def unmet(self, campaign_meta: Mapping) -> Tuple[str, ...]:
        """Which requirements a campaign's ``spin_meta`` fails to satisfy.

        ``campaign_meta`` is a :attr:`gwcat.selection.SelectionSet.spin_meta`
        dict.  An empty tuple means "nothing this block needs is contradicted";
        note that an *unverifiable* campaign (one carrying no spin draw
        densities at all) is not reported as a failure here -- distinguishing
        "checked and failed" from "never checked" is the caller's job, and
        conflating them would refuse every legacy spin-less campaign (GW-06).
        """
        meta = campaign_meta or {}
        checks = meta.get("checks") or {}
        out = []
        uniform_isotropic = bool(meta.get("uniform_isotropic"))
        if (self.uniform_magnitude or self.isotropic_tilts) and not uniform_isotropic:
            ran = any(checks.get(k) is not None for k in
                      ("magnitude_uniform", "max_spin_uniform", "isotropy_dev"))
            if ran:
                if self.uniform_magnitude:
                    out.append("uniform_magnitude")
                if self.isotropic_tilts:
                    out.append("isotropic_tilts")
        if self.single_amax:
            amax = meta.get("amax_detected")
            if (amax is None
                    or any(a is None or not np.isfinite(a) for a in amax)):
                out.append("single_amax")
        if self.needs_sky_draw and meta.get("sky_position_available") is False:
            out.append("needs_sky_draw")
        return tuple(out)


@dataclass(frozen=True)
class ParameterBlock:
    """One group of fit coordinates plus its two-sided density contract.

    The callables are intentionally optional: GW-17 is a pure assembly PR, so a
    block may declare its contract (columns, ranges, map kind, campaign
    requirements) before its densities are wired to the builders.  Nothing in
    :mod:`gwcat` imports this package yet.
    """
    name: str
    kind: str
    columns: Tuple[str, ...] = ()
    advisory_columns: Tuple[str, ...] = ()

    # ── declarative physics contract ────────────────────────────────────────
    map_kind: str = "bijective"
    exact_draw_density: bool = True
    requires_campaign: CampaignRequirement = field(
        default_factory=CampaignRequirement)
    #: True when the PE prior factor is CONSTANT within an event.  Such a factor
    #: cancels identically in the consumer's per-event ``p_pe`` normalisation
    #: (darksirens ``gw/utils.py``), which is why a flat-box spin prior is both
    #: amax-robust and weight-variance-free -- measured: ESS/nsamp 0.86 for
    #: component vs 0.55 for chieff on the same 259 events.
    prior_is_per_event_constant: bool = False

    # ── store requirements ──────────────────────────────────────────────────
    store_required: Tuple[str, ...] = ()
    store_alternatives: Tuple[tuple, ...] = ()
    #: Store columns the block FETCHES per event.  The rng-neutrality gate: a
    #: space whose blocks fetch nothing extra must consume an identical
    #: ``default_rng(seed)`` stream, which is what preserves chieff byte-parity.
    store_params_fetched: Tuple[str, ...] = ()

    # ── behaviour (wired in GW-18/GW-19) ────────────────────────────────────
    materialize_pe: Optional[Callable] = None
    ln_prior_pe: Optional[Callable] = None
    #: The SAME PE prior as a multiplicative factor, for a block that can supply
    #: it exactly.  ``exp(ln_prior_pe)`` is not the factor a builder would have
    #: multiplied in -- ``exp(log(m1)) != m1`` for 68% of float64 masses -- and
    #: the chieff PE export is contractually byte-identical to the frozen v1
    #: exporter, so composing through the log would break parity by an ulp per
    #: sample.  A block declaring this is composed in linear space
    #: (:func:`gwcat.params.compose.block_prior_factor_pe`); one that does not is
    #: composed as ``exp(ln_prior_pe)``.  Both run the same gates.
    prior_pe_factor: Optional[Callable] = None
    materialize_inj: Optional[Callable] = None
    ln_draw_inj: Optional[Callable] = None

    # ── validation & provenance ─────────────────────────────────────────────
    ranges: Mapping[str, Range] = field(default_factory=dict)
    support: Optional[Callable] = None
    #: Rough cost per evaluated point, seconds.  ChiEffChiPPrior is ~1e-6 s/point
    #: (~1 min per 1e6), which is one reason chieff_chip should not be a default.
    estimated_cost_per_point: float = 0.0
    notes: str = ""

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"{self.name}: kind must be one of {KINDS}, "
                             f"got {self.kind!r}")
        if self.map_kind not in MAP_KINDS:
            raise ValueError(f"{self.name}: map_kind must be one of "
                             f"{MAP_KINDS}, got {self.map_kind!r}")
        overlap = set(self.columns) & set(self.advisory_columns)
        if overlap:
            raise ValueError(
                f"{self.name}: {sorted(overlap)} listed as both a fit column "
                f"and an advisory column; a column the density does not cover "
                f"must not be fittable.")
        unknown = set(self.ranges) - set(self.columns) - set(self.advisory_columns)
        if unknown:
            raise ValueError(
                f"{self.name}: ranges declared for non-existent column(s) "
                f"{sorted(unknown)}")
        # R1, as an invariant rather than a convention: a projection cannot
        # honestly claim an exact draw density without constraining its parent.
        if (self.map_kind == "projection" and self.exact_draw_density
                and not self.requires_campaign.any_required):
            raise ValueError(
                f"{self.name}: a projection block claiming exact_draw_density "
                f"must declare what it requires of the parent campaign (R1). "
                f"Converting a component draw density into a projected one "
                f"integrates out degrees of freedom at fixed coordinate, which "
                f"has a closed form only for a constrained parent.")

    @property
    def all_columns(self) -> Tuple[str, ...]:
        return tuple(self.columns) + tuple(self.advisory_columns)

    def describe(self) -> dict:
        """JSON-able provenance for this block (written into the export attrs)."""
        return {
            "name": self.name,
            "kind": self.kind,
            "columns": list(self.columns),
            "advisory_columns": list(self.advisory_columns),
            "map_kind": self.map_kind,
            "exact_draw_density": bool(self.exact_draw_density),
            "prior_is_per_event_constant": bool(self.prior_is_per_event_constant),
            "requires_campaign": self.requires_campaign.describe(),
            "store_required": list(self.store_required),
            "store_alternatives": [list(g) for g in self.store_alternatives],
            "store_params_fetched": list(self.store_params_fetched),
            "ranges": {k: v.describe() for k, v in self.ranges.items()},
            "estimated_cost_per_point": float(self.estimated_cost_per_point),
            "notes": self.notes,
        }
