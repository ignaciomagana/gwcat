"""Spin blocks (GW-17).

This is where R1 earns the abstraction.  Every block below declares whether it is
a **projection** of the 4-D spin vector (chi_eff, chi_p, aligned-z) or a
**bijection** of it (component polar/6-D/cartesian), and a projection declares
what it needs of the parent campaign.  That single field is why the component
basis is exact for the non-uniform-isotropic O4 campaigns and the projections
are not -- and it is now a property of the block rather than a rule remembered
at each call site.

A projection comes in two kinds, and only one of them makes a claim about the
campaign.  ``spin.chieff`` / ``spin.chieff_chip`` **substitute**: they discard the
campaign's spin draw and write the analytic marginal in its place, which is the
right density only if the campaign really drew spins that way (hence
``_PROJECTION_PARENT``).  ``spin.chieff_reference`` **reweights**: it keeps the
campaign's exact component draw and divides by a declared reference spin prior,
so it requires nothing of the campaign's spin distribution -- only that the
campaign's magnitudes COVER the reference ceiling, which is a support check, not
a distributional one.

Measured consequence of ``prior_is_per_event_constant`` (GW-24, 259 events):
the flat-box component prior gives ESS/nsamp median 0.861 with zero events below
0.1, against 0.589 and 30 events below 0.1 for chieff.
"""
from __future__ import annotations

import numpy as np

from ..block import CampaignRequirement, ParameterBlock, Range

_LN_2PI = float(np.log(2.0 * np.pi))

#: Campaigns must be a single uniform-magnitude, isotropic draw for a projected
#: spin coordinate to have a closed-form density (R1).
_PROJECTION_PARENT = CampaignRequirement(
    uniform_magnitude=True, isotropic_tilts=True, single_amax=True)


# ── projections ─────────────────────────────────────────────────────────────
def _ln_chieff_pe(cols, ctx):
    from ...spin import chi_eff_prior_logprob
    return np.asarray(chi_eff_prior_logprob(
        cols["chieff"], ctx.m1src, ctx.m2src, amax=ctx.amax_1), dtype=float)


def _ln_chieff_chip_pe(cols, ctx):
    from ...spin import chi_eff_chi_p_prior_logprob
    return np.asarray(chi_eff_chi_p_prior_logprob(
        cols["chieff"], cols["chip"], ctx.m1src, ctx.m2src,
        amax=ctx.amax_1), dtype=float)


SPIN_CHIEFF = ParameterBlock(
    name="spin.chieff",
    kind="spin",
    columns=("chieff",),
    map_kind="projection",
    exact_draw_density=False,
    requires_campaign=_PROJECTION_PARENT,
    store_required=("chi_eff",),
    ln_prior_pe=_ln_chieff_pe,
    ranges={"chieff": Range(-1.0, 1.0)},
    notes=("The analytic swap replaces the campaign's real spin-draw density "
           "with the uniform-isotropic chi_eff marginal. Measurably false for "
           "O4ab (isotropy_dev = 0.642), which is what GW-06 gates."),
)

SPIN_CHIEFF_CHIP = ParameterBlock(
    name="spin.chieff_chip",
    kind="spin",
    columns=("chieff", "chip"),
    map_kind="projection",
    exact_draw_density=False,
    requires_campaign=_PROJECTION_PARENT,
    store_required=("chi_eff",),
    store_alternatives=(
        ("chi_p", ("a_1", "a_2", "cos_tilt_1", "cos_tilt_2"),
         ("a_1", "a_2", "tilt_1", "tilt_2")),),
    store_params_fetched=("a_1", "a_2", "cos_tilt_1", "cos_tilt_2",
                          "tilt_1", "tilt_2", "chi_p"),
    ln_prior_pe=_ln_chieff_chip_pe,
    ranges={"chieff": Range(-1.0, 1.0), "chip": Range(0.0, 1.0)},
    estimated_cost_per_point=1e-6,
    notes=("chi_p reaches the assumed ceiling on real data where chi_eff does "
           "not, which is why the -50 floor concentrated its damage here. "
           "Opt-in, not a shipped product (GW-23). ~1 min per 1e6 points."),
)


def _ln_chieff_reference_pe(cols, ctx):
    """The SAME analytic marginal as ``spin.chieff`` -- see the block notes.

    The reference basis differs from ``spin.chieff`` only on the INJECTION side
    (reweight vs substitution); the PE side of both is the isotropic
    uniform-magnitude chi_eff prior, so a ``chieff`` PE export at
    ``amax = a_ref`` is exactly the PE half of this basis.
    """
    return _ln_chieff_pe(cols, ctx)


SPIN_CHIEFF_REFERENCE = ParameterBlock(
    name="spin.chieff_reference",
    kind="spin",
    columns=("chieff",),
    advisory_columns=("a1", "a2", "cost1", "cost2", "chip"),
    map_kind="projection",
    # The exported density is NOT the campaign's own: it is the campaign's own
    # density REWEIGHTED, exactly, to a declared reference spin prior.  So the
    # space is not "exact" in the sense `is_exact` means (no assumed density
    # anywhere) -- there is an assumed density, it is just the DECLARED one
    # rather than a claim about the campaign.
    exact_draw_density=False,
    # Deliberately EMPTY, and this is the whole point of the block.  A
    # substituting projection (spin.chieff) needs the parent draw to BE
    # uniform-isotropic, because it throws that draw away.  A reweighting
    # projection keeps the parent draw in the numerator and divides by the
    # reference in the denominator, so it assumes nothing about the parent --
    # it only needs the parent to COVER the reference support, which is a
    # coverage check on the injected magnitudes (enforced in the builder), not
    # a distributional requirement expressible here.
    requires_campaign=CampaignRequirement(),
    store_required=("a_1", "a_2", "chi_eff"),
    store_alternatives=(("cos_tilt_1", "tilt_1"), ("cos_tilt_2", "tilt_2")),
    store_params_fetched=("a_1", "a_2", "cos_tilt_1", "cos_tilt_2",
                          "tilt_1", "tilt_2", "chi_p"),
    ln_prior_pe=_ln_chieff_reference_pe,
    ranges={"chieff": Range(-1.0, 1.0), "a1": Range(0.0, 1.0),
            "a2": Range(0.0, 1.0), "cost1": Range(-1.0, 1.0),
            "cost2": Range(-1.0, 1.0), "chip": Range(0.0, 1.0)},
    notes=("chi_eff against a DECLARED reference spin prior (isotropic, "
           "uniform magnitude, ceiling a_ref), reached by reweighting the "
           "campaign's exact component draw rather than substituting for it: "
           "pdraw_ref = pdraw_component * p_iso(chieff|q,a_ref) / p_ref(a,cost) "
           "with p_ref = 1/(4 a_ref^2). Valid for ANY campaign, including the "
           "non-uniform-isotropic O4 sets that spin.chieff must refuse. Its PE "
           "half is a spin.chieff export at amax = a_ref, and unlike the "
           "spin.chieff pairing the two ceilings MUST match -- the reference "
           "is the same object on both sides."),
)


# ── bijections ──────────────────────────────────────────────────────────────
def _ln_component_polar_pe(cols, ctx):
    """``-ln(4 * amax_1 * amax_2)`` -- constant within an event."""
    return -np.log(4.0 * float(ctx.amax_1) * float(ctx.amax_2))


def _support_component_polar(cols, ctx):
    return np.asarray(
        (np.abs(cols["a1"]) <= ctx.amax_1) & (np.abs(cols["a2"]) <= ctx.amax_2)
        & (np.abs(cols["cost1"]) <= 1.0) & (np.abs(cols["cost2"]) <= 1.0))


SPIN_COMPONENT_POLAR = ParameterBlock(
    name="spin.component_polar",
    kind="spin",
    columns=("a1", "a2", "cost1", "cost2"),
    advisory_columns=("chieff", "chip"),
    map_kind="bijective",
    exact_draw_density=True,
    prior_is_per_event_constant=True,
    # chi_eff is required even though it is ADVISORY here: the component export
    # still writes the legacy 10 columns, so the store must carry it. A
    # deliberate, documented simplification -- the alternative is making chi_eff
    # optional-but-conditional, which buys nothing.
    store_required=("a_1", "a_2", "chi_eff"),
    store_alternatives=(("cos_tilt_1", "tilt_1"), ("cos_tilt_2", "tilt_2")),
    store_params_fetched=("a_1", "a_2", "cos_tilt_1", "cos_tilt_2",
                          "tilt_1", "tilt_2", "chi_p"),
    ln_prior_pe=_ln_component_polar_pe,
    support=_support_component_polar,
    ranges={"a1": Range(0.0, 1.0), "a2": Range(0.0, 1.0),
            "cost1": Range(-1.0, 1.0), "cost2": Range(-1.0, 1.0),
            "chieff": Range(-1.0, 1.0), "chip": Range(0.0, 1.0)},
    notes=("The default and the recommended space. Its injection density is "
           "read NUMERICALLY off the file, so it requires nothing of the "
           "campaign and is exact for any of them. chieff/chip are emitted but "
           "ADVISORY: pdraw carries no term for them, so fitting on them here "
           "would be wrong -- the hole sel_v2_component.h5 shipped with."),
)

SPIN_COMPONENT_6D = ParameterBlock(
    name="spin.component_6d",
    kind="spin",
    columns=("a1", "a2", "cost1", "cost2", "phi1", "phi2"),
    advisory_columns=("chieff", "chip"),
    map_kind="bijective",
    exact_draw_density=True,
    prior_is_per_event_constant=True,
    requires_campaign=CampaignRequirement(uniform_azimuth=True),
    store_required=("a_1", "a_2", "chi_eff"),
    store_alternatives=(("cos_tilt_1", "tilt_1"), ("cos_tilt_2", "tilt_2"),
                        ("phi_12",), ("phi_jl",)),
    ln_prior_pe=lambda cols, ctx: -np.log(
        16.0 * np.pi ** 2 * float(ctx.amax_1) * float(ctx.amax_2)),
    ranges={"a1": Range(0.0, 1.0), "a2": Range(0.0, 1.0),
            "cost1": Range(-1.0, 1.0), "cost2": Range(-1.0, 1.0),
            "phi1": Range(0.0, 2.0 * np.pi, closed="left"),
            "phi2": Range(0.0, 2.0 * np.pi, closed="left")},
    notes=("Exact and available, but PAIRING two 6-D files needs an explicit "
           "acknowledgement: PE samples azimuths in an L-aligned frame and "
           "injections in theirs. Both are verified-uniform so both marginalise "
           "out exactly in 4-D; the frames only matter if you keep them."),
)

SPIN_CARTESIAN = ParameterBlock(
    name="spin.cartesian",
    kind="spin",
    columns=("s1x", "s1y", "s1z", "s2x", "s2y", "s2z"),
    advisory_columns=("chieff", "chip"),
    map_kind="bijective",
    exact_draw_density=True,
    prior_is_per_event_constant=False,   # the a^-2 does NOT cancel
    requires_campaign=CampaignRequirement(uniform_azimuth=True),
    store_required=("spin_1x", "spin_1y", "spin_1z",
                    "spin_2x", "spin_2y", "spin_2z", "chi_eff"),
    ranges={c: Range(-1.0, 1.0) for c in
            ("s1x", "s1y", "s1z", "s2x", "s2y", "s2z")},
    notes=("The one bijective spin block whose PE prior is PER-SAMPLE: "
           "-ln(16 pi^2 amax_1 amax_2 a1^2 a2^2). The a^-2 does not cancel in "
           "the per-event normalisation, so it carries real weight variance "
           "unlike the polar/6-D forms."),
)

SPIN_ALIGNED_Z = ParameterBlock(
    name="spin.aligned_z",
    kind="spin",
    columns=("chi1z", "chi2z"),
    map_kind="projection",
    exact_draw_density=False,
    requires_campaign=_PROJECTION_PARENT,
    store_required=("spin_1z", "spin_2z", "chi_eff"),
    ranges={"chi1z": Range(-1.0, 1.0), "chi2z": Range(-1.0, 1.0)},
    notes=("p(s) = ln(amax/|s|)/(2*amax) -- the factor 1/2 that "
           "ChiEffPrior._single_spin_pdf omitted until GW-03. Harmless there "
           "because both consumers renormalise; live the moment it is used "
           "standalone, which is exactly this block."),
)

SPIN_NONE = ParameterBlock(
    name="spin.none",
    kind="spin",
    columns=(),
    advisory_columns=("chieff",),
    map_kind="bijective",
    exact_draw_density=True,
    prior_is_per_event_constant=True,
    store_required=("chi_eff",),
    ln_prior_pe=lambda cols, ctx: 0.0,
    ln_draw_inj=lambda cols, ctx: 0.0,
    ranges={"chieff": Range(-1.0, 1.0)},
    notes=("No spin coordinate is fitted, so no spin density enters either "
           "side. The honest space for a cosmology-only run, and the mitigation "
           "if a 4-D spin population ever does collapse N_eff."),
)
