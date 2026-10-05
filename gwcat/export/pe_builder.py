"""PE (posterior-sample) export builder for the versioned pipeline (PR 3 / PR 6).

The two rules this module enforces
---------------------------------
**R1 (projection rule).**  A projected spin coordinate is definable only against
a uniform-magnitude/isotropic parent, so a projection block must REFUSE -- not
approximate -- when the parent is something else.

**R2 (support rule).**  A density that appears in a denominator may never be
floored.  Out of support is zero density, counted and reported, never ``exp(-50)``.


:func:`build_pe_product` reproduces -- for the ``spin_basis="chieff"`` case --
the arrays and provenance of the legacy :meth:`gwcat.catalog.GWCatalog.to_darksirens`
EXACTLY for identical keyword arguments, and returns them as an
:class:`gwcat.export.product.ExportProduct` instead of writing a file.  PR 6 adds
the ``"component"`` and ``"chieff_chip"`` spin bases on top of the SAME shared
selection/resampling scaffold.

Parity contract (why the loop below is a near-verbatim copy of the legacy
exporter's, not a call into it)
-------------------------------------------------------------------------------
The legacy ``to_darksirens`` body is frozen (a hard project constraint: it must
stay byte-identical, and shared code may NOT be refactored out of it).  This
builder therefore *duplicates* the logic rather than sharing it.  To stay
bit-for-bit identical it mirrors, in order:

  * the exact ``select(...)`` call (same filters, same ``compact_type=None``),
  * the same required-parameter check and ``get(..., per_event=True)`` read,
  * a single ``np.random.default_rng(seed)`` consumed by per-event
    ``rng.choice(n_kept, size=nsamp, replace=rep)`` calls in the SAME event
    order with the SAME arguments (events skipped before the ``rng.choice``
    call -- empty, fully z_max-cut, or under-sampled with ``replace=False`` --
    consume no random state in either implementation),
  * the same per-event cosmology resolution and ``z_of_dL`` inversion,
  * the same ``p_pe = m1det * p_dL_pe`` mass Jacobian -- since GW-34 obtained
    from ``mass.det_pair`` through the block composer rather than written out
    here, in LINEAR space precisely so the factor is the same float it always
    was (``exp(log(m1det))`` differs from ``m1det`` for 68% of samples) -- and
    (chieff basis always
    uses "include" semantics) the same 1-D chi_eff prior factor applied to the
    concatenated ``p_pe`` -- since GW-03 with NO floor on the log-density, in
    both this builder and the frozen v1 twin, so the parity still holds -- and
  * the same concatenation order.

Since GW-31 the chi_eff factor is evaluated at each EVENT's own prior ceiling
rather than at one caller-supplied ``amax``, so array parity with the frozen v1
exporter holds for identical kwargs *including an explicit numeric* ``amax=``
(which forces the single-ceiling behaviour the v1 exporter has).  With the
default ``amax="auto"`` the two agree whenever the store's declared ceilings
equal the v1 default -- and where they do not, v1 is the one that is wrong.

The only spin-basis-specific step -- the output columns plus the ``p_pe`` spin
factor -- is isolated in :func:`_apply_chieff_basis` (and, for PR 6, in
:func:`_apply_component_basis` / :func:`_apply_chieff_chip_basis`), so the shared
scaffold is untouched.  Extra per-sample columns the new bases need
(``a_1``/``a_2``/``cos_tilt_i``/``chi_p``) are fetched and resampled ONLY for
the non-chieff bases: the chieff path never fetches them, so the rng stream and
therefore the chieff parity is preserved exactly.

Design decisions (PR 6)
-----------------------
* **component basis -- ``chi_eff`` kept required.**  The handoff's bare
  ingredient tuple omits ``chi_eff``, but the component export still writes the
  legacy 10 columns (including ``chieff``).  Rather than make ``chi_eff``
  optional-but-conditional, it is KEPT required (see
  :data:`gwcat.schema.COMPONENT_REQUIRED`) so the ``chieff`` column is always
  present -- a deliberate, documented simplification.
* **component p_pe.**  ``p_pe = m1det * p_dL_pe / (4 * amax_1 * amax_2)`` with
  the event's own ``amax`` (a per-event *constant* that nonetheless varies event
  to event, so it multiplies ``p_pe`` explicitly).  No chi_eff factor.
* **chieff_chip p_pe.**  ``p_pe = m1det * p_dL_pe * exp(joint_lnprob)`` (no
  floor since GW-03) where ``joint_lnprob = chi_eff_chi_p_prior_logprob(chieff, chip,
  m1src, m2src, amax=amax_1)`` -- the SAME source-frame mass convention the
  chieff basis uses for its 1-D chi_eff prior.  ``amax`` is the event's
  ``amax_1``; a per-event ``amax_1 != amax_2`` warns (the joint prior assumes a
  single amax) and is recorded in ``spin_amax_mismatch_events``.
* **chieff_chip output columns.**  Always ``chip`` on top of the legacy 10; the
  raw ``a1``/``a2``/``cost1``/``cost2`` are added only when available for EVERY
  kept event (so the output stays rectangular).
"""
from __future__ import annotations

import warnings

import numpy as np

# Re-exported so a caller catching this builder's refusals can import them all
# from one place (SelectionSpec.check_pairable raises it).
from ..catalog import UnpairableSelectionCut  # noqa: F401
from ..cosmology import make_cosmology, z_of_dL
from ..params import (DEFAULT_PARAMETER_SPACE, PEContext,
                      block_prior_factor_pe, get_space)
from ..params.blocks.mass import UNSTATED_MASS_PRIOR, classify_mass_prior
from ..source_class import CUT_ESTIMATOR_ATTR, PE_MEDIAN_CUT_WARNING
from ..spin import AMAX_AUTO, parse_amax_option
from .contract import event_list_digest, mixed_prior_impl_events
from .product import ExportProduct, resolve_mock_data


def space_ordered_required(space, spin_basis):
    """``space.store_required`` in the legacy schema's ORDER where one exists.

    The set is the registry's (and a test pins the two equal); the order is the
    schema tuple's, so a missing-parameter error message reads exactly as it
    always has.  For a space with no legacy counterpart the registry order is
    used directly.
    """
    from ..schema import EXPORT_REQUIREMENTS

    want = set(space.store_required)
    legacy = EXPORT_REQUIREMENTS.get(f"gwcat2_pe:{spin_basis}")
    if legacy is not None and set(legacy) == want:
        return tuple(legacy)
    return tuple(space.store_required)

#: The datasets a chieff-basis PE export writes, in legacy order.
_CHIEFF_COLUMNS = ["ra", "dec", "m1det", "m2det", "chieff", "dL", "p_pe",
                   "redshift", "m1src", "m2src"]

#: Spin bases the v2 PE builder implements.  A registry space outside this
#: tuple is declarable but not yet buildable; the CLI and the validator read
#: this tuple (as ``SUPPORTED_SPIN_BASES``) so they cannot advertise or reject
#: a different set than the builder enforces.
_KNOWN_SPIN_BASES = ("chieff", "component", "chieff_chip", "nospin")
SUPPORTED_SPIN_BASES = _KNOWN_SPIN_BASES

#: Per-sample store columns the non-chieff bases may need (fetched only then).
_EXTRA_SAMPLE_CANDIDATES = ("a_1", "a_2", "cos_tilt_1", "cos_tilt_2",
                            "tilt_1", "tilt_2", "chi_p")


def _group_available_all(sub, *members):
    """True if, for every selected event, at least one of ``members`` (a store
    parameter that is present AND available) is available."""
    if sub.n_events == 0:
        return True
    ok = np.zeros(sub.n_events, dtype=bool)
    for p in members:
        ok |= sub.param_available(p)
    return bool(ok.all())


class OutOfSupportError(ValueError):
    """Samples fall outside the spin prior's support beyond the allowed fraction.

    Their density is genuinely zero (the prior really does exclude them), so this
    is not a numerical accident -- it means the declared ``amax`` does not cover
    the samples the export is built from.
    """


def _ess_of_inverse_weights(p_pe, nobs, nsamp):
    """Per-event effective sample size of the ``1/p_pe`` reweighting.

    ``ESS = (Σ w)^2 / Σ w^2`` with ``w = 1/p_pe`` over the event's samples,
    counting a zero-density sample as ``w = 0`` (it drops out of the sum but
    still counts in ``n``, matching the consumer's convention).

    This is the diagnostic the ``-50`` floor destroyed: floored samples carried
    ~1e21 times the median weight, so a single one drove ESS to ~1 out of
    thousands, and nothing recorded it.
    """
    if not nobs or not nsamp:
        return np.array([], dtype=float)
    p = np.asarray(p_pe, dtype=float).reshape(nobs, nsamp)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(p > 0, 1.0 / p, 0.0)
    w = np.where(np.isfinite(w), w, 0.0)
    s1 = w.sum(axis=1)
    s2 = (w ** 2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ess = np.where(s2 > 0, s1 ** 2 / s2, 0.0)
    return np.asarray(ess, dtype=float)


class ProjectionBasisNotAllowed(ValueError):
    """A projected spin basis was requested without opting in (GW-23).

    Projections are demoted from shipped products because R1 makes them
    undefinable against the real O4 campaigns, and because the component basis
    is both exact and measurably better conditioned.
    """


class ChiPDefinitionError(ValueError):
    """The stored chi_p column is not the Schmidt chi_p of the store's own
    components, so a joint (chi_eff, chi_p) prior would not describe it."""


#: The mass-prior class of a constituent-mixture row (GW-40f).
CONSTITUENT_MIXTURE = "constituent_mixture"


def _mass_state(kind) -> str:
    """:func:`classify_mass_prior`, plus the constituent mixture (GW-40f).

    A ``constituent_mixture`` row's density is not the ``m1det`` Jacobian the
    mass block gates, so the block would call it unsupported; it is instead a
    parsed, row-verified prior carried as ``p_mass_dL_pe`` -- verified, but
    applied here rather than through the block.
    """
    if str(kind) == CONSTITUENT_MIXTURE:
        return "verified"
    return classify_mass_prior(kind)


def build_pe_product(cat, *, spin_basis=DEFAULT_PARAMETER_SPACE,
                     nsamp=4096, seed=0,
                     far_max=None, pastro_min=None, z_max=None,
                     replace="auto", cosmology=None, amax=AMAX_AUTO,
                     amax_fallback=0.99,
                     allowed_names=None, allowed_names_authoritative=True,
                     source_class=None, event_list=None,
                     allow_missing_far=False, require_far=False,
                     waveform_policy="preferred", approximant=None,
                     allow_zero_p_pe=False,
                     allow_unpaired_pastro_min=False,
                     chi_p_definition="schmidt_recomputed",
                     chi_p_def_tol=1e-6,
                     max_out_of_support_frac=0.0,
                     allow_out_of_support=False,
                     allow_projection_basis=False,
                     drop_spin_above_ceiling=False,
                     sample_set_map=None, nrsur_q_rule=None,
                     nrsur_q_rule_substitute=False, mock_data=None):
    """Build a PE :class:`ExportProduct` from a :class:`~gwcat.catalog.GWCatalog`.

    For ``spin_basis="chieff"`` this reproduces the legacy
    :meth:`GWCatalog.to_darksirens` arrays and provenance exactly (see the
    module docstring for the parity contract).  ``format_version`` is NOT set
    here -- it belongs to the writer.

    The ``"component"`` and ``"chieff_chip"`` bases add spin-basis-specific
    output columns and ``p_pe`` factors (see the module docstring).

    ``amax`` -- the ceiling of the spin prior this export divides out -- defaults
    to ``"auto"``: EVERY basis, chieff included, then reads the ceiling from the
    event's own prior provenance (``spin_amax_1``/``spin_amax_2`` in the store
    meta), falling back to ``amax_fallback`` (with a warning) when it is NaN (old
    stores).  Pass a number to force one ceiling on every event; that is what the
    chieff basis used to do unconditionally, which stamped the caller's default
    0.99 on events whose sampling prior said something else (GW-31).

    The assembled ``p_pe`` must be finite and strictly positive (GW-01); a zero
    weight means the store's ``p_dL_pe`` was written by a pre-GW-01 ingest that
    truncated the distance prior at its recorded bounds.  Pass
    ``allow_zero_p_pe=True`` to downgrade that to a warning.

    ``chi_p_definition`` (GW-08) selects which quantity the ``chip`` column is:

    * ``"schmidt_recomputed"`` (default) -- always recompute the Schmidt chi_p
      from the store's own ``(a_i, cos_tilt_i, m_i)``.  This is the definition
      :class:`~gwcat.spin.ChiEffChiPPrior` is constructed for, so the column and
      the prior evaluated on it describe the same spin configuration.
    * ``"file"`` -- keep the release's own ``chi_p`` column, and **fail** when it
      disagrees with the Schmidt value by more than ``chi_p_def_tol``.  For six
      GWTC-2.1/3 ``C01:Mixed`` events the stored column is not the Schmidt chi_p
      of the store's own components, so half the column would be evaluated under
      a prior that does not describe it.

    The mass density, composed from its block (GW-34)
    -------------------------------------------------
    ``p_pe``'s mass factor is obtained from ``mass.det_pair`` rather than written
    out here, so the block's gate runs on every event: an event sampled under a
    prior that is not flat in the detector-frame component masses is REFUSED
    (the ``m1det`` factor is not its Jacobian), and one whose prior was never
    parsed exports with ``mass_prior_basis="assumed_default"`` and
    ``mass_prior_verified=False`` instead of being stamped as verified.  The
    file publishes ``q = m2det/m1det`` -- the coordinate the Jacobian belongs to
    -- with ``m2det`` kept as a derived, advisory column.

    The spin-support cut (GW-40c)
    -----------------------------
    ``drop_spin_above_ceiling=True`` removes, BEFORE resampling, every raw
    sample with ``a_1 > amax_1`` or ``a_2 > amax_2`` at the event's resolved
    ceiling (the one the spin prior below is evaluated at).  The GWTC priors
    declare 0.99 while 1,810 of 1.16M samples (71 of 282 events) exceed it; a
    projection basis keeps the declared ceiling, so without the cut PE and a
    ``chieff_reference`` selection (zero density above a_ref) have different
    supports.  For a uniform-magnitude prior the cut is exact rejection to
    ``U(0, amax)``.  The cut runs AFTER the ``z_max`` cut, so the per-event
    counts written as ``n_dropped_spin_above_ceiling_per_event`` are samples
    with ``z <= z_max``; ``n_above_spin_ceiling_raw_per_event`` counts the whole
    raw label (before ``z_max``) -- the integer an independent raw-file count
    reproduces -- and ``spin_ceiling_cut_order`` records which applies.  The
    draw then sees only the kept samples, so ``replace="auto"`` resamples with replacement exactly when
    fewer than ``nsamp`` remain.  Off by default: the default draw is unchanged.

    Per-prior provenance (GW-40c)
    -----------------------------
    ``prior_source_{kind,label}_{mass,spin,dL}_per_event`` carry the store's
    GW-40b provenance onto the file; ``chi_eff_amax_source_per_event`` is the
    spin prior's source KIND when the ceiling came from the store (the
    pre-GW-40 resolution -- analytic / fallback / caller -- is kept as
    ``chi_eff_amax_resolution_per_event``); ``spin_prior_assumed_events`` lists
    the events whose spin prior is ``assumed_default``.  An older store without
    those columns yields ``"unrecorded"`` (and the legacy "analytic" source).

    Effective-selection provenance (GW-33)
    --------------------------------------
    Every selection attr is written from ``sub.selection_spec`` -- ``cat``'s own
    accumulated cuts composed with this call's -- so a product built from
    ``cat.select(source_class="bbh")`` records the BBH filter instead of this
    call's defaults.  ``pastro_min`` (from either side) is REFUSED unless
    ``allow_unpaired_pastro_min=True``: the injection campaigns carry no
    per-injection p_astro, so no selection function can reproduce that cut.

    Mock-data provenance
    --------------------
    The ``mock_data`` attr (which darksirens reads to announce "This is using
    mock data.") comes from the STORE: ``cat.mock_data``, set when the store was
    written with ``mock_data=True`` (:func:`gwcat.ingest.build_store` /
    ``_write_store_from_records``).  The ``mock_data`` argument can only add the
    label (``True``) or assert a real input (``False``, which raises on a mock
    store); ``None`` inherits.  See :func:`gwcat.export.product.resolve_mock_data`.
    """
    mock_data = resolve_mock_data(getattr(cat, "mock_data", False), mock_data,
                                  source=f"the store {getattr(cat, 'path', cat)!r}")
    if spin_basis not in _KNOWN_SPIN_BASES:
        raise ValueError(
            f"unknown spin_basis={spin_basis!r}; known bases are "
            f"{list(_KNOWN_SPIN_BASES)}.")
    # ── chieff_chip is opt-in, not a product (GW-23) ────────────────────────
    if spin_basis == "chieff_chip" and not allow_projection_basis:
        raise ProjectionBasisNotAllowed(
            "spin_basis='chieff_chip' is opt-in and is NOT a shipped product. "
            "It is a PROJECTION of the 4-D spin vector, so it is definable only "
            "against a uniform-magnitude/isotropic injected draw (R1) -- which "
            "means it cannot be built against the O4 campaigns at all "
            "(isotropy_dev = 0.642, magnitude_uniform = [False, False]). It is "
            "also where the support contract bites hardest: chi_p reaches the "
            "assumed ceiling on real data where chi_eff does not, so 41 of 282 "
            "events sit out of support and the old -50 floor drove GW150914's "
            "reweighting to ESS = 1.0 of 3337. And it costs ~1 min per 1e6 "
            "points. Use spin_basis='component' -- exact for any campaign, and "
            "measured to carry far less weight variance (ESS/nsamp median 0.861 "
            "vs 0.589 on the same 259 events); derive chi_eff/chi_p from its "
            "columns downstream with gwcat.spin. Pass "
            "allow_projection_basis=True if you specifically need the projected "
            "density and understand the above.")
    if chi_p_definition not in ("schmidt_recomputed", "file"):
        raise ValueError(
            f"chi_p_definition must be 'schmidt_recomputed' or 'file'; got "
            f"{chi_p_definition!r}.")

    space = get_space(spin_basis)
    # Whether this space needs per-sample spin columns beyond the legacy set.
    # Derived from the registry rather than asserted (GW-18).
    need_extras = bool(space.store_params_fetched)
    # ── The ceiling of the prior being divided out (GW-31) ──────────────────
    # `None` = resolve it per event from the store's own prior provenance; a
    # float = the caller forces one ceiling on every event.  The chieff basis
    # took the forced path unconditionally, so a caller-default 0.99 was applied
    # to events whose sampling prior declared a different ceiling -- and the
    # chi_eff density depends on that ceiling in a chi_eff-DEPENDENT way, so the
    # error does not cancel in any per-event normalisation.
    forced_amax = parse_amax_option(amax, what="amax")
    # Every basis that divides out a SPIN prior needs a ceiling; nospin does not.
    need_amax = need_extras or spin_basis == "chieff"
    if drop_spin_above_ceiling and not need_amax:
        raise ValueError(
            f"drop_spin_above_ceiling=True needs a spin prior ceiling to cut "
            f"at, but spin_basis={spin_basis!r} divides out no spin prior. The "
            f"cut exists to make the PE spin support equal the selection's "
            f"reference support; drop the flag for this basis.")

    # chieff basis ALWAYS uses "include" semantics: the 1-D chi_eff prior is
    # multiplied into p_pe here (Mode A), matching the legacy default.  The
    # non-chieff bases carry their own spin factors (see their apply helpers).
    if spin_basis == "chieff":
        spin_prior_mode = "include"
        chi_eff_included = True
    elif spin_basis == "component":
        spin_prior_mode = "component_flat"
        chi_eff_included = False
    elif spin_basis == "nospin":
        # No spin coordinate is fitted, so NO spin density enters p_pe at all --
        # not the chi_eff prior, not the component box.  p_pe carries only the
        # mass Jacobian and the distance prior.  This is the honest space for a
        # cosmology-only run, and the design's stated mitigation if a 4-D spin
        # population ever does collapse N_eff downstream (GW-21).
        spin_prior_mode = "none"
        chi_eff_included = False
    else:  # chieff_chip
        spin_prior_mode = "chieff_chip_joint"
        chi_eff_included = True

    # ── Event selection: mirror the legacy exporter's select() call ─────────
    # compact_type is fixed to None (the builder does not expose it); every
    # other filter is passed through identically.
    sub = cat.select(compact_type=None, far_max=far_max,
                     pastro_min=pastro_min, allowed_names=allowed_names,
                     allowed_names_authoritative=allowed_names_authoritative,
                     source_class=source_class, event_list=event_list,
                     allow_missing_far=allow_missing_far,
                     require_far=require_far,
                     waveform_policy=waveform_policy,
                     approximant=approximant,
                     sample_set_map=sample_set_map,
                     nrsur_q_rule=nrsur_q_rule,
                     nrsur_q_rule_substitute=nrsur_q_rule_substitute)

    # The EFFECTIVE selection (GW-33): this call's filters composed with those
    # `cat` already carried.  `select()` intersects rows with the view it is
    # called on, so building the product from THIS call's arguments described a
    # filtered file as unfiltered.  Everything below records `spec`.
    spec = sub.selection_spec
    spec.check_pairable(allow_unpaired_pastro_min=allow_unpaired_pastro_min)
    # ── The class cut the paired selection file cannot reproduce (GW-12) ────
    # Said here, at build time, rather than only by the validator that will
    # refuse the finished pair: the cheap alternative (event_list=) is a choice
    # about how to build THIS product, and it is useless advice after the fact.
    # Asked of the EFFECTIVE spec: a view that was already class-filtered gets
    # the warning too, which is the case nothing else would have caught.
    if spec.source_class_requested:
        warnings.warn(PE_MEDIAN_CUT_WARNING)

    # ── Required-parameter checks, driven by the space (GW-18) ──────────────
    # The three-branch ladder is gone: what a space requires is now a property
    # of its blocks (gwcat.params), and schema.EXPORT_REQUIREMENTS is a
    # generated view over the same registry -- so the two cannot disagree.
    # Order still comes from the schema tuple for the legacy bases, purely so
    # error messages read the way they always have.
    label = ("gwcat2 PE export" if spin_basis == "chieff"
             else f"gwcat2 PE export ({spin_basis} basis)")
    need = list(space_ordered_required(space, spin_basis))
    sub._require_params(need, export=label)
    if space.store_alternatives:
        sub._require_alternatives(space.store_alternatives, export=label)

    rng = np.random.default_rng(seed)

    # Extra per-sample columns.  The gate is the space's declared
    # store_params_fetched, NOT a hardcoded `spin_basis != "chieff"`: that is
    # the rng-neutrality contract made structural.  A space declaring () fetches
    # nothing, so it cannot perturb the default_rng(seed) stream, which is the
    # whole reason the chieff export stays byte-identical to to_darksirens.
    extra_params = [p for p in space.store_params_fetched
                    if p in sub._param_index]

    # ── What the loop reads, and WHEN (GW-32) ──────────────────────────────
    # It used to be one ``sub.get(need, per_event=True)`` (plus a second for the
    # extras) BEFORE the rng draw, i.e. every row of every selected event was
    # materialised so that ``nsamp`` of them could be indexed out: 6.88M rows
    # read to emit 1.16M on the shipped store, 1.20 GiB peak.  The draw does not
    # depend on the values -- only on the row COUNT, and (with z_max) on
    # luminosity_distance -- so the indices are chosen first and only they are
    # read.  The rng stream is untouched by this: the same rng.choice calls
    # happen in the same order with the same arguments, so the drawn rows, and
    # therefore every exported value, are identical.
    #
    # An extra column that is NaN-filled for an event is not read for it either:
    # every consumer of the extras below is already gated on
    # ``param_available``, so an unavailable column is fetched and then never
    # looked at.  (The required set cannot be in that state -- _require_params
    # refuses a NaN-filled required column outright.)
    extra_avail = {p: sub.param_available(p) for p in extra_params}

    # A constituent-mixture row (GW-40f) carries its joint mass x distance
    # prior as a sample column; only those events read it.
    has_cmix_col = "p_mass_dL_pe" in sub._param_index
    cmix_avail = (sub.param_available("p_mass_dL_pe") if has_cmix_col
                  else np.zeros(sub.n_events, dtype=bool))

    def _read_plan(e):
        """The columns event ``e`` will actually be asked for."""
        plan = need + [p for p in extra_params
                       if p not in need and extra_avail[p][e]]
        if _mass_kind(e) == CONSTITUENT_MIXTURE:
            plan = plan + ["p_mass_dL_pe"]
        return plan

    # ── Resolve cosmology: per-event (default) or a single override ─────────
    sel_idx = np.asarray(sub._sel)
    have_cosmo_cols = ("dL_prior_H0" in sub.meta
                       and "dL_prior_Om0" in sub.meta)
    if cosmology is not None:
        cosmology_mode = "override"
        override_H0, override_Om0 = float(cosmology[0]), float(cosmology[1])
        per_event_H0 = np.full(sub.n_events, override_H0, dtype=float)
        per_event_Om0 = np.full(sub.n_events, override_Om0, dtype=float)
        pe_H0, pe_Om0 = override_H0, override_Om0
    else:
        cosmology_mode = "per-event"
        if not have_cosmo_cols:
            raise ValueError(
                "cosmology=None requires a per-event PE cosmology in the "
                "store (meta/dL_prior_H0 and meta/dL_prior_Om0), but those "
                "columns are absent. Pass an explicit cosmology=(H0, Om0) "
                "override to apply one cosmology to all events.")
        per_event_H0 = np.asarray(sub.meta["dL_prior_H0"], dtype=float)[sel_idx]
        per_event_Om0 = np.asarray(sub.meta["dL_prior_Om0"], dtype=float)[sel_idx]
        bad = ~(np.isfinite(per_event_H0) & np.isfinite(per_event_Om0))
        if bad.any():
            bad_names = sorted(np.asarray(sub.event_names)[bad].tolist())
            raise ValueError(
                f"cosmology=None but {int(bad.sum())} selected event(s) "
                f"have no stored PE cosmology (dL_prior_H0/dL_prior_Om0 is "
                f"NaN): {bad_names}. Pass an explicit cosmology=(H0, Om0) "
                f"override to apply one cosmology to all events.")
        pe_H0 = float(per_event_H0[0]) if sub.n_events else float("nan")
        pe_Om0 = float(per_event_Om0[0]) if sub.n_events else float("nan")

    # Per-event cosmology objects, built once per unique (H0, Om0) pair.
    _cosmo_cache: dict = {}

    def _cosmo_for(e):
        key = (per_event_H0[e], per_event_Om0[e])
        c = _cosmo_cache.get(key)
        if c is None:
            c = make_cosmology(*key)
            _cosmo_cache[key] = c
        return c

    # ── The ingested mass-prior class, per selected event (GW-34) ───────────
    # The m1det factor in p_pe is the (m1det, q) Jacobian, and it describes the
    # posterior ONLY if that posterior's mass prior is flat in the detector-frame
    # components.  The builder used to multiply it in unconditionally and stamp
    # mass_prior_basis="uniform_detector_frame" on every file it wrote -- so the
    # 9 of 282 rows in the shipped store whose prior was never parsed
    # ("assumed_default") were exported as verified uniform priors, and the
    # block's gate on the class never ran at all.  The class now travels with the
    # event: into mass.det_pair's gate below, and onto the file.
    mass_block = space.mass_block
    raw_mass_kind = sub.meta.get("mass_prior_kind")
    sel_mass_kind = (np.asarray(raw_mass_kind)[sel_idx]
                     if raw_mass_kind is not None else None)

    def _mass_kind(e):
        """The parsed mass-prior class of selected event ``e``."""
        if sel_mass_kind is None:
            return UNSTATED_MASS_PRIOR
        v = sel_mass_kind[e]
        v = v.decode() if isinstance(v, (bytes, bytearray)) else str(v)
        return v or UNSTATED_MASS_PRIOR

    # Refuse the whole build, naming EVERY offending event, rather than letting
    # the per-event gate stop at the first: the remedy is to exclude them (or
    # re-ingest), and an operator cannot act on a list discovered one run at a
    # time.  The rule is the block's own classifier, so the two cannot diverge.
    unsupported_mass = [(str(sub.event_names[e]), _mass_kind(e))
                        for e in range(sub.n_events)
                        if _mass_state(_mass_kind(e)) == "unsupported"]
    no_cmix = [str(sub.event_names[e]) for e in range(sub.n_events)
               if _mass_kind(e) == CONSTITUENT_MIXTURE and not cmix_avail[e]]
    if no_cmix:
        raise ValueError(
            f"{len(no_cmix)} event(s) declare a constituent_mixture mass prior "
            f"but the store carries no p_mass_dL_pe samples for them: "
            f"{no_cmix[:10]}. Re-ingest with constituent_mixture_prior=True.")
    if unsupported_mass:
        listed = ", ".join(f"{n}: {k!r}" for n, k in unsupported_mass[:10])
        more = ("" if len(unsupported_mass) <= 10
                else f", ... (+{len(unsupported_mass) - 10} more)")
        raise ValueError(
            f"{len(unsupported_mass)} of {sub.n_events} selected event(s) were "
            f"sampled under a mass prior that is NOT uniform in the "
            f"detector-frame component masses, so the |dm2det/dq| = m1det "
            f"Jacobian this export applies is not their Jacobian and the error "
            f"does not cancel in any per-event normalisation: {listed}{more}. "
            f"Exclude them (event_list= / allowed_names=), or re-ingest so the "
            f"analytic (chirp_mass, mass_ratio) prior is parsed.")

    cols = {k: [] for k in ["m1det", "m2det", "q", "dL", "ra", "dec",
                            "chieff", "p_pe", "redshift", "m1src", "m2src"]}
    kept = []
    kept_mass_kind = []
    #: Posterior samples each kept event lost to z_max (GW-37).  The truncation
    #: used to appear in no attr, no summary key and no contract field, so a
    #: truncated export and an untruncated one compared EQUAL on
    #: selection_spec_digest, event_list_digest and contract_hash alike.
    n_cut_by_z_max = []
    kept_H0, kept_Om0 = [], []
    kept_ss_name, kept_ss_approx, kept_ss_reason = [], [], []
    # Resampling provenance (GW-13), aligned with ``kept``.
    n_unique_per_event, upsampled_events = [], []

    # ── Per-event spin-prior ceiling scaffolding (every spin basis) ─────────
    # Per-event spin amax from the store meta (aligned with selected events).
    # This is a META read, not a sample read: it fetches nothing through
    # ``sub.get`` and so cannot perturb the default_rng(seed) stream, which is
    # what keeps the chieff basis's resample identical to the frozen v1 one.
    if need_amax:
        def _meta_sel(name, dtype=float):
            v = sub.meta.get(name)
            if v is None:
                return None
            return np.asarray(v, dtype=dtype)[sel_idx]

        sel_amax1 = _meta_sel("spin_amax_1")
        sel_amax2 = _meta_sel("spin_amax_2")
        raw_kind = sub.meta.get("spin_prior_kind")
        sel_kind = (np.asarray(raw_kind)[sel_idx] if raw_kind is not None
                    else None)

        kept_amax1, kept_amax2 = [], []
        amax_src1_list, amax_src2_list, amax_infl_list = [], [], []
        samples_bound_events = []
        fallback_events, unrecognized_events, mismatch_events = [], [], []

        def _declared_ceiling(e):
            """``(amax_1, amax_2, used_fallback)`` BEFORE any samples_bound
            widening: a forced numeric ``amax`` for every event, else this
            event's stored ceilings, else ``amax_fallback``.  One function for
            both the GW-40c spin cut (before the draw) and the prior (after), so
            the cut and the density can never use different ceilings."""
            if forced_amax is not None:
                return float(forced_amax), float(forced_amax), False
            a1 = (float(sel_amax1[e]) if sel_amax1 is not None
                  else float("nan"))
            a2 = (float(sel_amax2[e]) if sel_amax2 is not None
                  else float("nan"))
            fb = False
            if not np.isfinite(a1):
                a1, fb = float(amax_fallback), True
            if not np.isfinite(a2):
                a2, fb = float(amax_fallback), True
            return a1, a2, fb

    # ── Per-prior source provenance, per selected event (GW-40c) ─────────────
    # A META read (no sample access), so the rng stream is untouched.  A store
    # written before GW-40b has no prior_source_* columns: those events read
    # "unrecorded" (or "assumed_default" where the legacy class column says so)
    # rather than being silently promoted to a verified source.
    def _meta_str_sel(name):
        v = sub.meta.get(name)
        if v is None:
            return None
        return [x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
                for x in np.asarray(v)[sel_idx]]

    _legacy_kind_col = {"mass": "mass_prior_kind", "spin": "spin_prior_kind"}
    sel_prior_kind, sel_prior_label = {}, {}
    for _p in ("mass", "spin", "dL"):
        kinds = _meta_str_sel(f"prior_source_kind_{_p}")
        labels = _meta_str_sel(f"prior_source_label_{_p}")
        if kinds is None or not any(kinds):
            legacy = (_meta_str_sel(_legacy_kind_col[_p])
                      if _p in _legacy_kind_col else
                      _meta_str_sel("dL_prior_basis"))
            kinds = ["assumed_default" if k == "assumed_default"
                     else "release_reweighted" if k == "release_reweighted"
                     else "unrecorded"
                     for k in (legacy or [""] * sub.n_events)]
            labels = [""] * sub.n_events
        sel_prior_kind[_p] = [k or "unrecorded" for k in kinds]
        sel_prior_label[_p] = labels or [""] * sub.n_events
    spin_kind_recorded = _meta_str_sel("prior_source_kind_spin") is not None
    # A5: the spin ceilings each constituent of a Mixed set declares in its
    # config (JSON per row; "" for non-Mixed rows and pre-GW-40 stores).
    sel_cfg_amax = (_meta_str_sel("spin_amax_config_per_constituent")
                    or [""] * sub.n_events)
    kept_cfg_amax = []
    # Which implementation evaluated each event's p_dL_pe AT INGEST (GW-40i:
    # "exact" by default; "bilby"/"astropy"/"analytic" in older stores or under
    # --legacy-grid-priors).  A meta read, like the provenance above.
    sel_dL_impl = _meta_str_sel("dL_prior_impl") or [""] * sub.n_events
    kept_dL_impl = []
    kept_prior_kind = {p: [] for p in ("mass", "spin", "dL")}
    kept_prior_label = {p: [] for p in ("mass", "spin", "dL")}
    #: Raw samples removed by the GW-40c spin-support cut, per kept event --
    #: counted AFTER the z_max cut (the samples the cut actually removed from
    #: the pool the draw sees) ...
    n_dropped_spin = []
    #: ... and the same count over the event's WHOLE raw label, before any
    #: z_max cut: the number an independent count of the raw file reproduces.
    n_above_spin_ceiling_raw = []
    if drop_spin_above_ceiling:
        avail_a1_cut = sub.param_available("a_1")
        avail_a2_cut = sub.param_available("a_2")
        no_spin = [str(sub.event_names[e]) for e in range(sub.n_events)
                   if not (avail_a1_cut[e] and avail_a2_cut[e])]
        if no_spin:
            raise ValueError(
                f"drop_spin_above_ceiling=True needs the a_1/a_2 sample columns "
                f"to cut on, but {len(no_spin)} selected event(s) lack them: "
                f"{no_spin[:10]}. Re-ingest with spin magnitudes, or exclude "
                f"those events.")

    if need_extras:
        avail_a1 = sub.param_available("a_1")
        avail_a2 = sub.param_available("a_2")
        avail_cos1 = sub.param_available("cos_tilt_1")
        avail_cos2 = sub.param_available("cos_tilt_2")
        avail_tilt1 = sub.param_available("tilt_1")
        avail_tilt2 = sub.param_available("tilt_2")
        avail_chip = sub.param_available("chi_p")

        # Whether the raw component columns are available for EVERY selected
        # event (=> a rectangular output is possible).
        emit_extras = (_group_available_all(sub, "a_1")
                       and _group_available_all(sub, "a_2")
                       and _group_available_all(sub, "cos_tilt_1", "tilt_1")
                       and _group_available_all(sub, "cos_tilt_2", "tilt_2"))

        a1_list, a2_list, cost1_list, cost2_list, chip_list = [], [], [], [], []
        chip_src_list = []
        chip_maxdiff_list = []
        chip_mismatch_events, chip_no_ingredient_events = [], []

    def _resolve_cost(e, avail_cos, avail_tilt, cos_name, tilt_name):
        """cos_tilt samples for the rows already drawn for event ``e``:
        prefer the stored cos_tilt, else cos(stored tilt), else None."""
        if avail_cos[e]:
            return drawn[cos_name]
        if avail_tilt[e]:
            return np.cos(drawn[tilt_name])
        return None

    sel_rows = np.asarray(sub._sel)
    reasons_arr = getattr(sub, "_selection_reasons", None)
    # One open handle on the store for the whole loop; ``drawn`` holds the rows
    # of the event being processed and nothing else.
    drawn: dict = {}
    with sub.sample_reader() as reader:
        sample_counts = reader.counts()
        for e in range(sub.n_events):
            # The slice LENGTH, from the store's offsets -- no posterior read.
            n = int(sample_counts[e])
            if n == 0:
                continue

            cosmo_e = _cosmo_for(e)

            # Per-sample z_max cut.  This is the one place the draw depends on
            # sample VALUES, so it (and only it) reads a full column, and only when
            # a cut was asked for.  ``mass_1``/``mass_2`` used to be read in full
            # and subset here too; both subsets were dead -- the kept rows are
            # re-indexed from the store below -- so they are no longer read at all.
            n_cut_e = 0
            if z_max is not None:
                dL_e = reader.read(e, "luminosity_distance")["luminosity_distance"]
                z_e = z_of_dL(dL_e, cosmo_e)
                keep = z_e <= z_max
                n_cut_e = int(np.sum(~keep))
                if not keep.any():
                    continue
                idx_map = np.nonzero(keep)[0]
            else:
                idx_map = np.arange(n)

            # ── The spin-support cut (GW-40c), BEFORE the draw ──────────────
            # Reads the event's full a_1/a_2 (a sample read, but not an rng
            # call, so with the flag off nothing here runs and the draw is the
            # historical one bit for bit).
            # Order: the z_max cut (above) runs first, so
            # n_dropped_spin_above_ceiling counts only samples with z <= z_max;
            # n_above_spin_ceiling_raw counts the whole raw label.
            n_drop_e = 0
            n_raw_e = 0
            if drop_spin_above_ceiling:
                c1, c2, _fb = _declared_ceiling(e)
                ab = reader.read(e, ["a_1", "a_2"])
                a1_raw = np.asarray(ab["a_1"], dtype=float)
                a2_raw = np.asarray(ab["a_2"], dtype=float)
                above_raw = (np.abs(a1_raw) > c1) | (np.abs(a2_raw) > c2)
                n_raw_e = int(np.sum(above_raw))
                ok = ~above_raw[idx_map]
                n_drop_e = int(np.sum(~ok))
                if not ok.any():
                    raise ValueError(
                        f"drop_spin_above_ceiling: every sample of event "
                        f"{sub.event_names[e]} has a spin magnitude above its "
                        f"ceiling ({c1:g}, {c2:g}); the event has no support "
                        f"under its own spin prior.")
                idx_map = idx_map[ok]

            n_kept = len(idx_map)
            rep = (n_kept < nsamp) if replace == "auto" else bool(replace)
            if n_kept < nsamp and not rep:
                warnings.warn(f"Event {sub.event_names[e]}: only {n_kept} samples "
                              f"after z_max cut, but replace=False and nsamp={nsamp}. "
                              f"Skipping.")
                continue
            idx_local = rng.choice(n_kept, size=nsamp, replace=rep)
            idx_orig = idx_map[idx_local]
            # Distinct posterior samples behind this event's nsamp rows.  Without
            # it a bootstrapped event's per-event ESS reads high by the duplication
            # factor -- the very diagnostic that should flag it as unusable.
            n_unique_per_event.append(int(np.unique(idx_orig).size))
            if rep and n_kept < nsamp:
                upsampled_events.append(str(sub.event_names[e]))

            # The draw is settled: NOW read, and read only the rows it kept.
            drawn = reader.read(e, _read_plan(e), rows=idx_orig)

            m1 = drawn["mass_1"]
            m2 = drawn["mass_2"]
            dL = drawn["luminosity_distance"]
            p_dL = drawn["p_dL_pe"]

            # ── The mass block's density, COMPOSED rather than written out here ──
            # p_pe = |d(m1det,m2det)/d(m1det,q)| * p(dL) = m1det * p_dL_pe.  The
            # factor comes from mass.det_pair through the block composer, so the
            # block's gate on THIS event's ingested prior class is what decides
            # whether the Jacobian may be applied to it -- the declaration and the
            # arithmetic are one thing now instead of two that agreed by habit.
            # (The distance term is the store's p_dL_pe, materialized at ingest by
            # the same function distance.dl calls; see gwcat.params.compose.)
            mass_kind_e = _mass_kind(e)
            q = m2 / m1
            if mass_kind_e == CONSTITUENT_MIXTURE:
                # GW-40f: the equal-weight mixture of the constituents' own
                # normalised (m1det, q) x dL priors, built and verified at
                # ingest -- NOT m1det * p_dL, which is one constituent's shape.
                p_pe = np.asarray(drawn["p_mass_dL_pe"], dtype=float)
            else:
                ctx_e = PEContext(event_name=str(sub.event_names[e]),
                                  m1det=m1, m2det=m2, dL=dL, cosmology=cosmo_e,
                                  mass_prior_kind=mass_kind_e,
                                  dL_prior_impl=str(sel_dL_impl[e]))
                p_pe = block_prior_factor_pe(
                    mass_block, {"m1det": m1, "m2det": m2, "q": q},
                    ctx_e) * p_dL

            # Redshift and source masses under THIS event's PE cosmology
            z = z_of_dL(dL, cosmo_e)

            cols["m1det"].append(m1)
            cols["m2det"].append(m2)
            cols["q"].append(q)
            cols["dL"].append(dL)
            cols["ra"].append(drawn["ra"])
            cols["dec"].append(drawn["dec"])
            cols["chieff"].append(drawn["chi_eff"])
            cols["p_pe"].append(p_pe)
            cols["redshift"].append(z)
            cols["m1src"].append(m1 / (1 + z))
            cols["m2src"].append(m2 / (1 + z))
            kept.append(sub.event_names[e])
            kept_mass_kind.append(mass_kind_e)
            n_cut_by_z_max.append(n_cut_e)
            n_dropped_spin.append(n_drop_e)
            n_above_spin_ceiling_raw.append(n_raw_e)
            for _p in ("mass", "spin", "dL"):
                kept_prior_kind[_p].append(sel_prior_kind[_p][e])
                kept_prior_label[_p].append(sel_prior_label[_p][e])
            kept_cfg_amax.append(sel_cfg_amax[e])
            kept_dL_impl.append(sel_dL_impl[e])
            kept_H0.append(float(per_event_H0[e]))
            kept_Om0.append(float(per_event_Om0[e]))
            row = sel_rows[e]
            kept_ss_name.append(_ss_meta(sub, row, "sample_set_name"))
            kept_ss_approx.append(_ss_meta(sub, row, "approximant"))
            kept_ss_reason.append(
                str(reasons_arr[e]) if reasons_arr is not None
                and e < len(reasons_arr) else "")

            # ── Per-event spin-prior ceiling + (non-chieff) spin columns ────────
            if need_amax:
                name = sub.event_names[e]

                # Resolve per-event amax.  A forced (numeric) `amax` overrides the
                # store for every event -- the caller has said which ceiling to use.
                # Otherwise it comes from THIS event's prior provenance, and only a
                # missing/NaN one falls back (recorded, and warned about below).
                a1max, a2max, used_fallback = _declared_ceiling(e)
                if used_fallback:
                    fallback_events.append(str(name))

                # np.isclose, not exact float equality (GW-04): a single injected
                # ceiling round-trips as 0.9980000000000001 vs 0.9979999999999999
                # through the numerical amax resolution, so `!=` fired on files with
                # no real prior asymmetry and buried the case that matters (a genuine
                # NSBH prior with amax_2 ~ 0.05).
                if (spin_basis == "chieff_chip"
                        and not np.isclose(a1max, a2max, rtol=1e-9, atol=1e-12)):
                    mismatch_events.append(str(name))

                # spin_prior_kind provenance (flat/joint assumption may not hold).
                if sel_kind is not None:
                    k = str(sel_kind[e])
                    if k and k not in ("uniform_magnitude_isotropic",
                                       "assumed_default"):
                        unrecognized_events.append(str(name))

                # Component pieces for this event (None where unavailable).
                a1_e = a2_e = None
                if need_extras:
                    a1_e = (drawn["a_1"]
                            if avail_a1[e] else None)
                    a2_e = (drawn["a_2"]
                            if avail_a2[e] else None)

                # ── Resolved-amax provenance (GW-04) ────────────────────────────
                # "analytic"      the ingested analysis's own prior covers its
                #                 samples -- the honest case;
                # "samples_bound" the parsed prior does NOT cover the samples, so
                #                 the ceiling is raised to max|a_i| and the
                #                 inflation is recorded;
                # "fallback"      no stored amax at all, a fabricated ceiling;
                # "caller"        a numeric `amax=` overrode the store (GW-31).
                #
                # samples_bound is not a nicety: the GWTC analytic priors declare
                # amax = 0.99 for all 282 rows while the posteriors reach 0.9996+,
                # so 1810 of 1.16M samples (71 of 282 events) sit outside the
                # declared box.  Under GW-03 those get p_pe = 0 and the export
                # refuses -- correctly, because a ceiling that excludes real samples
                # is wrong.  Raising it to cover them is exact for the COMPONENT
                # basis, whose density is a flat box: widening the box changes only
                # the per-event constant 1/(4*amax_1*amax_2), which cancels in the
                # per-event normalisation the consumer applies.
                src1 = src2 = ("caller" if forced_amax is not None
                               else "fallback" if used_fallback else "analytic")
                infl1 = infl2 = 0.0
                # ONLY for a flat-box (bijection-like) prior.  Widening the box
                # changes just the per-event constant 1/(4*amax_1*amax_2), which
                # cancels in the consumer's per-event normalisation -- so it is
                # exact.  For a PROJECTION (chieff / chieff_chip) the entire density
                # p(chi_eff[, chi_p] | amax) depends on the ceiling, so raising it
                # would silently change the physics rather than fix a bookkeeping
                # bound.  Those bases keep the declared amax and are refused by the
                # GW-03 support gate instead, which is the honest outcome: a
                # projection cannot be built against a prior whose support does not
                # contain the samples.
                if spin_basis == "component" and a1_e is not None and np.size(a1_e):
                    s1 = float(np.nanmax(np.abs(np.asarray(a1_e, float))))
                    if np.isfinite(s1) and s1 > a1max:
                        infl1 = s1 / a1max - 1.0
                        a1max = s1 * (1.0 + 1e-6)
                        src1 = "samples_bound"
                if spin_basis == "component" and a2_e is not None and np.size(a2_e):
                    s2 = float(np.nanmax(np.abs(np.asarray(a2_e, float))))
                    if np.isfinite(s2) and s2 > a2max:
                        infl2 = s2 / a2max - 1.0
                        a2max = s2 * (1.0 + 1e-6)
                        src2 = "samples_bound"
                if "samples_bound" in (src1, src2):
                    samples_bound_events.append(
                        (str(name), max(infl1, infl2)))
                amax_src1_list.append(src1)
                amax_src2_list.append(src2)
                amax_infl_list.append(max(infl1, infl2))

                kept_amax1.append(a1max)
                kept_amax2.append(a2max)

            # ── Non-chieff per-event spin columns (chi_p and its ingredients) ────
            if need_extras:
                cost1_e = _resolve_cost(e, avail_cos1, avail_tilt1,
                                        "cos_tilt_1", "tilt_1")
                cost2_e = _resolve_cost(e, avail_cos2, avail_tilt2,
                                        "cos_tilt_2", "tilt_2")

                # chip resolution (GW-08).  ``chi_p_definition`` decides whether the
                # exported column is the Schmidt chi_p of the store's OWN
                # (a_i, cos_tilt_i, m_i) -- the definition ChiEffChiPPrior is built
                # for -- or the release's stored column, which for six GWTC-2.1/3
                # C01:Mixed events is a different quantity.
                have_ingredients = (a1_e is not None and a2_e is not None
                                    and cost1_e is not None and cost2_e is not None)
                chip_from_file = None
                if avail_chip[e]:
                    chip_from_file = drawn["chi_p"]
                chip_derived = None
                if have_ingredients:
                    from ..spin import chi_p_from_components
                    chip_derived = chi_p_from_components(a1_e, a2_e, cost1_e,
                                                         cost2_e, m1, m2)

                # Definition-consistency diagnostic, whenever both are available.
                maxdiff = np.nan
                if chip_from_file is not None and chip_derived is not None:
                    maxdiff = float(np.max(np.abs(
                        np.asarray(chip_from_file, float) - chip_derived)))
                chip_maxdiff_list.append(maxdiff)

                if chi_p_definition == "schmidt_recomputed":
                    if chip_derived is not None:
                        chip_e, chip_src = chip_derived, "schmidt_recomputed"
                    elif chip_from_file is not None:
                        chip_e, chip_src = chip_from_file, "file_no_ingredients"
                        chip_no_ingredient_events.append(str(sub.event_names[e]))
                    else:  # pragma: no cover - guarded by requirement checks
                        chip_e, chip_src = None, ""
                else:  # "file"
                    if chip_from_file is not None:
                        chip_e, chip_src = chip_from_file, "file"
                        if np.isfinite(maxdiff) and maxdiff > chi_p_def_tol:
                            chip_mismatch_events.append(
                                (str(sub.event_names[e]), maxdiff))
                    elif chip_derived is not None:
                        chip_e, chip_src = chip_derived, "derived"
                    else:  # pragma: no cover
                        chip_e, chip_src = None, ""
                chip_list.append(chip_e)
                chip_src_list.append(chip_src)

                if emit_extras:
                    a1_list.append(a1_e)
                    a2_list.append(a2_e)
                    cost1_list.append(cost1_e)
                    cost2_list.append(cost2_e)

    nobs = len(kept)
    if nobs == 0:
        # The guard build_selection_product has had all along (GW-37).  A
        # zero-event product is not a small product: with nobs=0 every length
        # check is `0 == 0*nsamp`, every array check is skipped, and the file
        # validated ALL PASSED while `gwcat export pe` exited 0 -- so a scripted
        # export-then-validate gate reported success on an empty catalog.  The
        # realistic trigger is mundane: a GWOSC outage leaves every event with
        # far=NaN and the default missing-FAR policy drops all of them.
        raise ValueError(
            f"the export's cuts left 0 of {sub.n_events} selected event(s), so "
            f"there is no PE product to build. Effective selection: "
            f"{sub.selection_spec.to_json()}. Check far_max/pastro_min/z_max "
            f"and whether the store's FARs were populated at ingest (a store "
            f"ingested during a GWOSC outage carries far=NaN for every event, "
            f"which the default missing-FAR policy then drops).")
    data = {k: np.concatenate(v) if v else np.array([])
            for k, v in cols.items()}

    kept_H0_arr = np.asarray(kept_H0, dtype=float)
    kept_Om0_arr = np.asarray(kept_Om0, dtype=float)
    cosmology_per_event_varies = bool(
        nobs > 1 and (np.ptp(kept_H0_arr) > 0 or np.ptp(kept_Om0_arr) > 0))

    import h5py
    _str = h5py.string_dtype()

    # ── Spin-basis-specific step (columns + p_pe spin factor) ───────────────
    spin_attrs: dict = {}
    # Per-event ceilings, resolved above, broadcast to one value per sample.
    # The chieff basis needs them too now (GW-31): its 1-D chi_eff prior is a
    # PROJECTION whose whole shape depends on the ceiling, so evaluating it at a
    # caller default rather than at the event's own prior ceiling is wrong by a
    # chi_eff-dependent factor that no per-event normalisation removes.
    if need_amax:
        amax1_arr = np.asarray(kept_amax1, dtype=float)
        amax2_arr = np.asarray(kept_amax2, dtype=float)
        amax1_ps = (np.repeat(amax1_arr, nsamp) if nobs else np.array([]))
        amax2_ps = (np.repeat(amax2_arr, nsamp) if nobs else np.array([]))
        #: The file-level chi_eff support bound: max(amax_1, amax_2) over events.
        amax_bound = (float(max(amax1_arr.max(), amax2_arr.max()))
                      if nobs else float(forced_amax if forced_amax is not None
                                         else amax_fallback))
    if spin_basis == "chieff":
        columns = _apply_chieff_basis(data, amax1_ps, amax2_ps)
    elif spin_basis == "nospin":
        columns = _apply_nospin_basis(data)
    else:
        chip = (np.concatenate(chip_list) if chip_list else np.array([]))
        extras = None
        if emit_extras and nobs:
            extras = {
                "a1": np.concatenate(a1_list),
                "a2": np.concatenate(a2_list),
                "cost1": np.concatenate(cost1_list),
                "cost2": np.concatenate(cost2_list),
            }
        elif emit_extras:  # nobs == 0
            extras = {k: np.array([]) for k in ("a1", "a2", "cost1", "cost2")}

        if spin_basis == "component":
            columns = _apply_component_basis(data, amax1_ps, amax2_ps,
                                             chip, extras)
        else:  # chieff_chip
            columns = _apply_chieff_chip_basis(data, amax1_ps, chip, extras)

        # Emit warnings once (after the loop).
        if spin_basis == "chieff_chip" and mismatch_events:
            warnings.warn(
                f"spin_basis='chieff_chip': {len(mismatch_events)} event(s) "
                f"have spin_amax_1 != spin_amax_2; the joint (chi_eff, chi_p) "
                f"prior assumes a single amax and uses amax_1: "
                f"{mismatch_events}")
        # ── chi_p definition consistency (GW-08) ────────────────────────────
        if chip_mismatch_events:
            listed = ", ".join(f"{nm}: {d:.4g}" for nm, d in
                               chip_mismatch_events[:10])
            more = ("" if len(chip_mismatch_events) <= 10
                    else f", ... (+{len(chip_mismatch_events) - 10} more)")
            raise ChiPDefinitionError(
                f"chi_p_definition='file': {len(chip_mismatch_events)} event(s) "
                f"ship a chi_p column that is not the Schmidt chi_p of the "
                f"store's own (a_i, cos_tilt_i, m_i), by more than "
                f"chi_p_def_tol={chi_p_def_tol:g} "
                f"[max|chi_p_file - chi_p_schmidt|]: {listed}{more}. "
                f"ChiEffChiPPrior is constructed strictly for the Schmidt "
                f"definition, so exporting these would evaluate part of the "
                f"column under a prior that does not describe it. Use "
                f"chi_p_definition='schmidt_recomputed' (the default), or raise "
                f"chi_p_def_tol deliberately.")
        if samples_bound_events:
            worst = sorted(samples_bound_events, key=lambda t: -t[1])[:5]
            listed = ", ".join(f"{nm}: +{100 * f:.3f}%" for nm, f in worst)
            warnings.warn(
                f"spin_basis={spin_basis!r}: {len(samples_bound_events)} of "
                f"{nobs} event(s) have posterior spin samples ABOVE the amax "
                f"their analytic prior declares, so the ceiling was raised to "
                f"cover them (spin_amax_source='samples_bound'). Largest "
                f"inflations: {listed}. For the component basis this is exact "
                f"-- widening a flat box changes only the per-event constant "
                f"1/(4*amax_1*amax_2), which cancels in the consumer's "
                f"per-event normalisation -- which is why it is applied ONLY "
                f"here. A projection basis keeps its declared ceiling and is "
                f"refused by the support gate instead.")
        if chip_no_ingredient_events:
            warnings.warn(
                f"chi_p_definition='schmidt_recomputed': "
                f"{len(chip_no_ingredient_events)} event(s) lack the component "
                f"ingredients, so the release's own chi_p column was kept and "
                f"recorded as 'file_no_ingredients': "
                f"{chip_no_ingredient_events}")

        # Basis-specific provenance attrs (the chieff basis records the same
        # ceilings under its own chi_eff_* names below).
        spin_attrs = {
            "spin_amax_1_per_event": amax1_arr,
            "spin_amax_2_per_event": amax2_arr,
            "spin_amax_fallback_events": np.array(fallback_events, dtype=_str),
            "spin_prior_unrecognized_events": np.array(
                unrecognized_events, dtype=_str),
            "chi_p_source_per_event": np.array(chip_src_list, dtype=_str),
            # GW-08: the definition in force, and the measured disagreement with
            # the release's own column (NaN where it could not be compared).
            "chi_p_definition": str(chi_p_definition),
            "chi_p_def_maxdiff_per_event": np.asarray(chip_maxdiff_list,
                                                      dtype=float),
            "chi_p_def_tol": float(chi_p_def_tol),
            "spin_amax_fallback": float(amax_fallback),
            # GW-04: how each per-body ceiling was resolved, and by how much a
            # samples_bound ceiling had to be raised above the declared prior.
            "spin_amax_source_1_per_event": np.array(amax_src1_list, dtype=_str),
            "spin_amax_source_2_per_event": np.array(amax_src2_list, dtype=_str),
            "spin_amax_inflation_per_event": np.asarray(amax_infl_list,
                                                        dtype=float),
        }
        if spin_basis == "component":
            spin_attrs["component_spin_prior_applied_to_p_pe"] = True
        else:  # chieff_chip
            spin_attrs["chi_eff_chi_p_prior_applied_to_p_pe"] = True
            spin_attrs["chi_eff_chi_p_amax_per_event"] = amax1_arr
            spin_attrs["spin_amax_mismatch_events"] = np.array(
                mismatch_events, dtype=_str)

    # ── Ceiling warnings + provenance shared by every spin basis (GW-31) ────
    # These used to live in the non-chieff branch, so the ONE basis that ships
    # said nothing about where its ceiling came from.
    if need_amax:
        if fallback_events:
            warnings.warn(
                f"spin_basis={spin_basis!r}: {len(fallback_events)} event(s) "
                f"have no stored spin amax (spin_amax_1/2 NaN); using "
                f"amax_fallback={amax_fallback}: {fallback_events}")
        if unrecognized_events:
            warnings.warn(
                f"spin_basis={spin_basis!r}: {len(unrecognized_events)} "
                f"event(s) have spin_prior_kind != "
                f"'uniform_magnitude_isotropic' (the flat/joint spin-prior "
                f"assumption may not hold): {unrecognized_events}")
        if spin_basis == "chieff":
            spin_attrs.update({
                # WHICH ceiling the 1-D chi_eff prior was evaluated at, per
                # event, and how it was resolved ("analytic" = the event's own
                # sampling prior, "fallback" = amax_fallback because the store
                # declares none, "caller" = a numeric amax= overrode both).
                "chi_eff_amax_1_per_event": amax1_arr,
                "chi_eff_amax_2_per_event": amax2_arr,
                # GW-40c: the SOURCE KIND of the ceiling -- where the spin
                # prior it came from was declared (own_analytic /
                # sibling_inherited / config_file_declared / assumed_default)
                # -- whenever it came from the store; "caller"/"fallback" when
                # it did not.  A pre-GW-40b store (no provenance columns)
                # keeps the historical "analytic".  The historical resolution
                # itself is kept as chi_eff_amax_resolution_per_event.
                "chi_eff_amax_source_per_event": np.array(
                    [(k if (r == "analytic" and spin_kind_recorded) else r)
                     for r, k in zip(amax_src1_list,
                                     kept_prior_kind["spin"])], dtype=_str),
                "chi_eff_amax_resolution_per_event": np.array(
                    amax_src1_list, dtype=_str),
                "chi_eff_amax_mode": ("fixed" if forced_amax is not None
                                      else "per_event"),
                "spin_amax_fallback": float(amax_fallback),
                "spin_amax_fallback_events": np.array(fallback_events,
                                                      dtype=_str),
                "spin_prior_unrecognized_events": np.array(
                    unrecognized_events, dtype=_str),
            })

    # ── The published mass density coordinate (GW-34) ───────────────────────
    # p_pe is a density in (m1det, q, ...): the m1det factor above IS
    # |d(m1det,m2det)/d(m1det,q)|.  Every export nevertheless published
    # (m1det, m2det) as its fit columns, so a consumer reading the contract off
    # the file could integrate these weights against the wrong measure.  q is
    # written as the fit coordinate; m2det is kept -- every consumer and every
    # plot uses it -- but is DERIVED (m2det = q * m1det) and advisory.
    columns["q"] = data["q"]

    # ── Prior support accounting (GW-03) ────────────────────────────────────
    # The basis helpers return the per-sample support mask alongside the
    # columns; it is provenance, not a fit column, so it is popped here.
    in_support = np.asarray(columns.pop("_in_support",
                                        np.ones(nobs * nsamp, dtype=bool)))
    n_out = int(np.sum(~in_support))
    frac_out = (n_out / in_support.size) if in_support.size else 0.0
    per_event_out = (in_support.reshape(nobs, nsamp) if nobs and nsamp
                     else np.zeros((0, 0), dtype=bool))
    n_out_per_event = ((~per_event_out).sum(axis=1).astype(np.int64)
                       if per_event_out.size else np.array([], dtype=np.int64))
    # Effective sample size of the 1/p_pe reweighting, per event.  This is the
    # diagnostic the -50 floor destroyed: a single floored sample took
    # essentially all the weight, giving ESS ~ 1 out of thousands.
    ess_per_event = _ess_of_inverse_weights(columns["p_pe"], nobs, nsamp)

    if n_out and frac_out > max_out_of_support_frac and not allow_out_of_support:
        hit = np.nonzero(n_out_per_event)[0]
        listed = ", ".join(f"{kept[i]}: {int(n_out_per_event[i])}/{nsamp}"
                           for i in hit[:10])
        more = "" if hit.size <= 10 else f", ... (+{hit.size - 10} more)"
        raise OutOfSupportError(
            f"spin_basis={spin_basis!r}: {n_out} of {in_support.size} samples "
            f"({100 * frac_out:.3f}%) fall outside the spin prior's support, "
            f"above max_out_of_support_frac={max_out_of_support_frac:g}. "
            f"Affected event(s) [{hit.size} of {nobs}]: {listed}{more}. "
            f"Their p_pe is exactly zero, which is correct -- the prior really "
            f"does assign them no density -- but it means the declared amax "
            f"does not cover the samples. "
            + ("The component prior is a flat box, so its ceiling can be "
               "raised to cover them exactly -- see the samples_bound "
               "resolution in GW-04; reaching this message means a sample "
               "exceeded even that. "
               if spin_basis == "component"
               else "Use spin_basis='component', whose flat-box support can be "
                    "widened to cover the samples exactly (a projection's "
                    "whole density depends on the ceiling, so it cannot). ")
            + f"Or pass allow_out_of_support=True to export anyway.")
    if n_out and allow_out_of_support:
        warnings.warn(
            f"spin_basis={spin_basis!r}: exporting with {n_out} of "
            f"{in_support.size} samples ({100 * frac_out:.3f}%) outside the "
            f"spin prior's support (allow_out_of_support=True). Their p_pe is "
            f"zero, so the consumer will drop them while still counting them in "
            f"n for the per-event MC variance -- see "
            f"prior_reweight_ess_per_event for the resulting concentration.")

    # ── Exported-weight support contract (GW-01) ────────────────────────────
    from ..schema import check_p_pe_positive
    check_p_pe_positive(
        columns["p_pe"], event_names=kept, nsamp=nsamp,
        allow_zero=allow_zero_p_pe, expected_zero=~in_support,
        context=f"gwcat-pe-2.0 export (spin_basis={spin_basis!r})",
        remedy=("A zero p_pe at an IN-SUPPORT sample comes from a store whose "
                "p_dL_pe was truncated at the recorded distance-prior bounds; "
                "re-ingest the store so the distance prior is evaluated over "
                "the full sample range, or pass allow_zero_p_pe=True to write "
                "it anyway. (Out-of-support samples are expected to be zero and "
                "are excluded from this check -- see max_out_of_support_frac.)"))

    # Rectangularity.  A raise, not an assert: `python -O` strips asserts, and
    # this guards the invariant every consumer reshapes on (same rationale as
    # the v1 writer's twin check in catalog.py).
    expected = nobs * nsamp
    if columns["m1det"].size != expected:
        raise RuntimeError(
            f"internal error: assembled {columns['m1det'].size} rows but "
            f"nobs*nsamp = {nobs}*{nsamp} = {expected}. The export would not "
            f"be reshapeable by any consumer; refusing to build it.")

    # From the view's policy resolution, matching the v1 writer: False iff the
    # SELECTED view holds more than one sample set of one event (only possible
    # under waveform_policy="all") -- the case where the hierarchical
    # likelihood would count one physical event N times.  The written-file
    # uniqueness of `kept` (event NAMES) can differ when a duplicate row is
    # later skipped (z_max / undersampling); the view-level answer is the one
    # the flag exists to give, and using name-uniqueness here while the v1
    # writer used the view made the two writers disagree on the same input.
    homogeneous = bool(getattr(sub, "_homogeneous_sample_sets", True))

    # ── The mass prior as it was actually INGESTED, not as it was assumed ────
    # One string for the file when every kept event agrees, "mixed" when they do
    # not, plus the per-event classes and the list of events whose prior was
    # never verified.  "uniform_detector_frame" is now a claim the file can only
    # make when every row it holds carries that parsed class.
    mass_kinds = [str(k) for k in kept_mass_kind]
    uniq_mass_kinds = sorted(set(mass_kinds))
    mass_prior_basis = (uniq_mass_kinds[0] if len(uniq_mass_kinds) == 1
                        else ("mixed" if uniq_mass_kinds
                              else UNSTATED_MASS_PRIOR))
    mass_unverified = [str(n) for n, k in zip(kept, mass_kinds)
                       if _mass_state(k) != "verified"]
    mass_prior_verified = bool(mass_kinds) and not mass_unverified
    if mass_unverified:
        warnings.warn(
            f"{len(mass_unverified)} of {nobs} exported event(s) carry no "
            f"VERIFIED uniform detector-frame mass prior (classes "
            f"{uniq_mass_kinds}), so the m1det Jacobian is assumed for them "
            f"rather than parsed from the release: "
            f"{mass_unverified[:10]}"
            + ("" if len(mass_unverified) <= 10
               else f", ... (+{len(mass_unverified) - 10} more)")
            + f". The file records mass_prior_basis={mass_prior_basis!r} and "
              f"mass_prior_verified=False; it is NOT stamped as a verified "
              f"uniform prior.")

    if z_max is not None and any(n > 0 for n in n_cut_by_z_max):
        warnings.warn(
            f"z_max={z_max} dropped {sum(n_cut_by_z_max)} posterior sample(s) "
            f"across {sum(1 for n in n_cut_by_z_max if n)} of {nobs} event(s). "
            f"The selection product must be built with the SAME z_max "
            f"(build_selection_product(z_max=...), CLI --z-max) or mu "
            f"integrates over a redshift range the exported posteriors do not "
            f"cover; validate_export refuses a pair whose z_max disagrees.")

    # ── Provenance attrs (everything the legacy exporter records EXCEPT ──────
    # format_version, which is the writer's; plus the new spin_basis). The two
    # legacy-compat spin attrs (spin_prior_mode / chi_eff_prior_applied_to_p_pe
    # / chi_eff_in_p_pe) are added by the chieff writer, not here, so a future
    # non-chieff basis never carries them.
    attrs = {
        # darksirens core
        "nsamp": int(nsamp),
        "nobs": int(nobs),
        # From the store's provenance (or the caller's explicit True), never a
        # constant: a mock campaign run through these builders was labelled real.
        "mock_data": bool(mock_data),
        # ── The per-sample redshift truncation (GW-37) ──────────────────────
        # NaN when unused, because HDF5 has no null and "no truncation" must be
        # distinguishable from "this file predates the record".  It appeared in
        # no attr at all, so a file truncated at z<0.12 was indistinguishable
        # from the full posterior it came from -- same selection_spec_digest,
        # same event_list_digest, same contract_hash.  The injection side takes
        # the same argument now (build_selection_product(z_max=...)), and the
        # validator cross-checks the two.
        "z_max": float("nan") if z_max is None else float(z_max),
        "n_samples_cut_by_z_max": np.asarray(n_cut_by_z_max, dtype=np.int64),
        # spin basis (new in v2)
        "spin_basis": spin_basis,
        # provenance
        # The mass prior AS INGESTED (GW-34), never a constant string: a row
        # whose analytic prior was never parsed says so, and the file says
        # whether the m1det Jacobian is verified or merely assumed.
        "mass_prior_basis": mass_prior_basis,
        "mass_prior_verified": bool(mass_prior_verified),
        "mass_prior_kind_per_event": np.array(mass_kinds, dtype=_str),
        "mass_prior_unverified_events": np.array(mass_unverified, dtype=_str),
        "n_events_mass_prior_unverified": int(len(mass_unverified)),
        "mass_jacobian_applied": True,
        # WHICH coordinates p_pe is a density in: the m1det factor is the
        # (m1det, q) Jacobian, and m2det = q * m1det is derived from them.
        "mass_density_coordinates": "m1det,q",
        "distance_prior_removed": False,
        "cosmology_mode": cosmology_mode,
        "cosmology_override_used": bool(cosmology is not None),
        "source_frame_under_recorded_cosmology": True,
        "cosmology_per_event_varies": bool(cosmology_per_event_varies),
        "cosmology_H0_per_event": kept_H0_arr,
        "cosmology_Om0_per_event": kept_Om0_arr,
        # The file-level chi_eff support bound actually in force: max over the
        # exported events of max(amax_1, amax_2).  It is NO LONGER simply the
        # caller's argument -- with amax="auto" each event uses its own prior's
        # ceiling, and this scalar is the envelope of those (GW-31).
        "chi_eff_amax": (float(amax_bound) if need_amax else
                         float(forced_amax if forced_amax is not None
                               else amax_fallback)),
        "pe_cosmology_H0": pe_H0,
        "pe_cosmology_Om0": pe_Om0,
        "homogeneous_sample_sets": homogeneous,
        "n_unique_samples_per_event": np.asarray(n_unique_per_event,
                                                 dtype=np.int64),
        "resampled_with_replacement": bool(upsampled_events),
        "n_events_resampled_with_replacement": int(len(upsampled_events)),
        "sample_set_name_per_event": np.array(
            [str(x) for x in kept_ss_name], dtype=_str),
        "sample_set_approximant_per_event": np.array(
            [str(x) for x in kept_ss_approx], dtype=_str),
        "sample_set_selection_reason": np.array(
            [str(x) for x in kept_ss_reason], dtype=_str),
        "event_names": np.array([str(k) for k in kept], dtype=_str),
        # ── Prior-support accounting (GW-03) ─────────────────────────────────
        # Observability at the source: how many samples the prior excludes, and
        # what that does to the reweighting's effective sample size.  The
        # consumer masks zero-weight samples by design but reports nothing, so
        # without these a producer writing many zeros loses them silently.
        "n_samples_out_of_support": int(n_out),
        "frac_samples_out_of_support": float(frac_out),
        "n_out_of_support_per_event": n_out_per_event,
        "prior_reweight_ess_per_event": ess_per_event,
        "max_out_of_support_frac": float(max_out_of_support_frac),
        "out_of_support_allowed": bool(allow_out_of_support),
        # ── The spin-support cut (GW-40c) ────────────────────────────────────
        "spin_ceiling_cut_applied": bool(drop_spin_above_ceiling),
        # Removed from the pool the draw sees: samples with z <= z_max (every
        # sample when no z_max) whose a_1 or a_2 exceeds the ceiling.
        "n_dropped_spin_above_ceiling_per_event": np.asarray(
            n_dropped_spin if drop_spin_above_ceiling else [0] * nobs,
            dtype=np.int64),
        # The same count over the WHOLE raw label, before the z_max cut -- the
        # integer an independent raw-file count reproduces.  Equal to the
        # above when z_max is None.
        "n_above_spin_ceiling_raw_per_event": np.asarray(
            n_above_spin_ceiling_raw if drop_spin_above_ceiling
            else [0] * nobs, dtype=np.int64),
        "spin_ceiling_cut_order": ("after_z_max_cut" if z_max is not None
                                   else "no_z_max_cut"),
        # ── Per-prior source provenance (GW-40c) ─────────────────────────────
        **{f"prior_source_kind_{p}_per_event": np.array(
            kept_prior_kind[p], dtype=_str) for p in ("mass", "spin", "dL")},
        **{f"prior_source_label_{p}_per_event": np.array(
            kept_prior_label[p], dtype=_str) for p in ("mass", "spin", "dL")},
        # A5: per event, JSON {constituent: [a1_max, a2_max] | null} of the
        # ceilings a Mixed set's constituents declare in their configs.
        "spin_amax_config_per_constituent_per_event": np.array(
            kept_cfg_amax, dtype=_str),
        "spin_prior_assumed_events": np.array(
            [str(n) for n, k in zip(kept, kept_prior_kind["spin"])
             if k == "assumed_default"], dtype=_str),
        "spin_prior_non_own_analytic_events": np.array(
            [str(n) for n, k in zip(kept, kept_prior_kind["spin"])
             if k != "own_analytic"], dtype=_str),
        # ── How the two analytic prior factors were evaluated (GW-40i) ──────
        "dL_prior_impl_per_event": np.array(
            [str(x) for x in kept_dL_impl], dtype=_str),
    }
    if chi_eff_included:
        from ..spin import CHI_EFF_PRIOR_METHODS, current_chi_eff_prior_impl
        _chi_impl = current_chi_eff_prior_impl()
        attrs["chi_eff_prior_impl"] = _chi_impl
        attrs["chi_eff_prior_method"] = CHI_EFF_PRIOR_METHODS[_chi_impl]
        # A MIXED product -- exact chi_eff over a legacy-dL store, or the
        # legacy chi_eff grid over an exact store -- is legal (each factor is
        # still the declared prior to its implementation's accuracy) but is
        # neither the exact product nor a legacy regression; say so, and let
        # `gwcat validate --strict` / --require-exact-priors refuse it.
        _mixed = mixed_prior_impl_events(_chi_impl, kept_dL_impl, kept)
        if _mixed:
            warnings.warn(
                f"PE export mixes prior implementations: chi_eff_prior_impl="
                f"{_chi_impl!r} but the store evaluated p_dL_pe with a "
                f"{'legacy' if _chi_impl == 'exact' else 'exact'} "
                f"implementation for {len(_mixed)} event(s) (e.g. "
                f"{_mixed[:3]}). Re-ingest (or re-export) with "
                f"--legacy-grid-priors on both or on neither.")
    attrs.update(_population_resolver_attrs(cat, kept))
    attrs.update(_sample_set_map_attrs(
        getattr(sub, "_sample_set_map_report", None), kept))
    # ── The EFFECTIVE event selection (GW-33) ───────────────────────────────
    # The composed source_class_filter and cut estimator (GW-12: WHICH masses
    # the class threshold was applied to), every numeric cut as a number the
    # paired selection file can be checked against (NaN = no cut on that
    # statistic), the normalised name-filter digest, and the FAR policy.
    attrs.update(spec.to_attrs())
    # WHICH events this file holds, independent of how they were spelled.
    attrs["event_list_digest"] = event_list_digest(kept)
    attrs.update(spin_attrs)
    # Per-sample support mask, written as uint8 so the consumer can mask without
    # re-deriving the prior.
    columns["in_support"] = in_support.astype(np.uint8)

    # ── Validation-summary feed (writer fills output_path + summary_context) ─
    from ..validation_summary import summarize_catalog
    summary = summarize_catalog(sub, parameter_space=spin_basis)
    summary.update({
        "kind": "darksirens_export",
        "n_events_considered": int(sub.n_events),
        "n_events_exported": int(nobs),
        "n_events_skipped_after_selection": int(sub.n_events - nobs),
        "event_names_exported": [str(k) for k in kept],
        "nsamp_per_event": int(nsamp),
        "spin_basis": spin_basis,
        # The redshift truncation, in the sidecar too (GW-37).
        "z_max": None if z_max is None else float(z_max),
        "n_samples_cut_by_z_max": int(sum(n_cut_by_z_max)),
        # The EFFECTIVE selection (GW-33), the same record the attrs carry.
        "selection_spec": spec.to_dict(),
        "selection_spec_digest": spec.digest(),
        "event_list_digest": event_list_digest(kept),
        "source_class_filter": spec.source_class_filter or None,
        CUT_ESTIMATOR_ATTR: spec.cut_estimator,
        "event_list_filter": spec.event_list_filter or None,
        "far_policy": spec.far_policy,
        "allow_missing_far": bool(spec.allow_missing_far),
        "require_far": bool(spec.require_far),
        "n_events_missing_far": int(spec.n_missing_far),
        "spin_prior_mode": spin_prior_mode,
        "chi_eff_prior_applied_to_p_pe": bool(chi_eff_included),
        # The mass prior behind the m1det Jacobian (GW-34), so a summary reader
        # sees an unverified prior without opening the file's attrs.
        "mass_prior_basis": mass_prior_basis,
        "mass_prior_verified": bool(mass_prior_verified),
        "n_events_mass_prior_unverified": int(len(mass_unverified)),
        "cosmology_mode": cosmology_mode,
        "cosmology_override_used": bool(cosmology is not None),
        "cosmology_per_event_varies": bool(cosmology_per_event_varies),
        "waveform_policy": str(waveform_policy),
        "approximant": None if approximant is None else str(approximant),
        "homogeneous_sample_sets": homogeneous,
        "n_unique_samples_per_event": [int(x) for x in n_unique_per_event],
        "resampled_with_replacement": bool(upsampled_events),
        "n_events_resampled_with_replacement": int(len(upsampled_events)),
        "n_samples_out_of_support": int(n_out),
        "frac_samples_out_of_support": float(frac_out),
        "prior_reweight_ess_min": (float(np.min(ess_per_event))
                                   if ess_per_event.size else None),
        "prior_reweight_ess_median": (float(np.median(ess_per_event))
                                      if ess_per_event.size else None),
        "chi_p_definition": str(chi_p_definition),
        "spin_ceiling_cut_applied": bool(drop_spin_above_ceiling),
        "n_dropped_spin_above_ceiling": int(sum(n_dropped_spin)),
        "n_above_spin_ceiling_raw": int(sum(n_above_spin_ceiling_raw)),
        "prior_source_kind_counts": {
            p: {k: int(kept_prior_kind[p].count(k))
                for k in sorted(set(kept_prior_kind[p]))}
            for p in ("mass", "spin", "dL")},
        "population_resolver": attrs.get("population_resolver", ""),
    })

    return ExportProduct(kind="pe", columns=columns, attrs=attrs,
                         spin_basis=spin_basis, summary=summary)


#: The one resolver whose output is the bundled GWTC-5 BBH population.
POPULATION_RESOLVER = ("gwcat.population_samples."
                       "resolve_gwtc5_bbh_population_names")


def _names_sha256(names):
    """sha256 of the sorted unique names, one per line with a trailing
    newline -- the recipe the v1 ``allowed_names_digest`` reference
    (``fe6e33da...``) was computed with."""
    import hashlib
    uniq = sorted({(n.decode() if isinstance(n, (bytes, bytearray))
                    else str(n)) for n in names})
    return hashlib.sha256(("\n".join(uniq) + "\n").encode()).hexdigest()


def _population_resolver_attrs(cat, kept):
    """Whether the exported event set IS the bundled-population resolver's
    output, with the resolver's inputs (GW-40c).

    The resolver is re-run here on the store's names (a pure, local function of
    the bundled lists -- no GWOSC); when its output equals the exported event
    set, the file records the resolver's qualified name, the sha256 of every
    bundled list it read and the resolved names' digest.  Otherwise (a subset,
    a different population, a synthetic store) ``population_resolver`` is ""
    and ``population_resolver_match`` False -- recorded, never guessed.

    What this does NOT prove: it shows the exported set EQUALS the resolver's
    output on the store's names, not that the caller obtained the set by
    calling the resolver (an equal set from any other source records the same
    attrs).  A call-trace claim (BUILD_PLAN G3g) must come from the build
    wrapper's own audit log, not from these attrs.
    """
    import hashlib
    import json as _json
    from .contract import event_list_digest as _digest
    from ..bbh_allowed_names import _event_list_path

    files = ("bbh_o1o2.txt", "bbh_o3a.txt", "bbh_o3b.txt", "bbh_o4a.txt",
             "bbh_o4b.txt", "non_bbh_exclusions.txt", "provenance.yaml")
    shas = {}
    for fn in files:
        try:
            with open(_event_list_path(fn), "rb") as f:
                shas[fn] = hashlib.sha256(f.read()).hexdigest()
        except OSError:
            shas[fn] = ""
    match, resolved, aliases, err = False, [], {}, ""
    try:
        from ..population_samples import resolve_gwtc5_bbh_population_names
        resolved, aliases = resolve_gwtc5_bbh_population_names(
            [str(n) for n in cat.names])
        match = sorted(resolved) == sorted(str(k) for k in kept)
    except ValueError as exc:
        err = str(exc)[:200]
    inputs = {
        "bundled_list_sha256": shas,
        # Two digests of the SAME resolved set, under two recipes, each
        # labelled so a gate compares like with like:
        "allowed_names_digest": _digest(resolved) if resolved else "",
        "allowed_names_digest_recipe": (
            "gwcat.export.contract.event_list_digest: blake2b-64 of the JSON "
            "list of sorted unique names"),
        "allowed_names_sha256": (_names_sha256(resolved) if resolved else ""),
        "allowed_names_sha256_recipe": (
            "sha256 of the sorted unique names joined by '\\n' with a "
            "trailing '\\n' (the v1 / BUILD_PLAN G3a recipe)"),
        "n_resolved": len(resolved),
        "n_aliases": len(aliases),
        "aliases": {str(k): str(v) for k, v in sorted(aliases.items())},
        "error": err,
    }
    return {
        "population_resolver": POPULATION_RESOLVER if match else "",
        "population_resolver_match": bool(match),
        "population_resolver_inputs": _json.dumps(inputs, sort_keys=True),
    }


def _sample_set_map_attrs(report, kept):
    """The event-map provenance on the file (GW-40e); empty for other policies.

    Restricted to the EXPORTED events, in export order, so the substitute and
    q-fraction lists describe exactly what the file holds.
    """
    import h5py
    _str = h5py.string_dtype()
    kept = [str(k) for k in kept]
    report = report or {}
    subs = report.get("substitutes", {}) or {}
    qf = report.get("nrsur_q_frac", {}) or {}
    sub_ev = [k for k in kept if k in subs]
    q_ev = [k for k in kept if k in qf]
    thr = report.get("nrsur_q_rule")
    return {
        "sample_set_map_source": str(report.get("map_source", "")),
        "sample_set_map_sha256": str(report.get("map_sha256", "")),
        "sample_set_substitute_events": np.array(sub_ev, dtype=_str),
        "sample_set_substitute_labels": np.array(
            [subs[k]["label"] for k in sub_ev], dtype=_str),
        "sample_set_substitute_original_labels": np.array(
            [subs[k].get("original_label", "") for k in sub_ev], dtype=_str),
        "sample_set_substitute_reasons": np.array(
            [subs[k]["reason"] for k in sub_ev], dtype=_str),
        "nrsur_q_rule": float("nan") if thr is None else float(thr),
        "nrsur_q_rule_substitute": bool(report.get("nrsur_q_rule_substitute",
                                                   False)),
        "nrsur_q_frac_events": np.array(q_ev, dtype=_str),
        "nrsur_q_frac_below_floor": np.asarray([qf[k] for k in q_ev],
                                               dtype=float),
    }


def _ss_meta(sub, row, field):
    v = sub.meta.get(field)
    if v is None:
        return ""
    x = v[int(row)]
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def _apply_chieff_basis(data, amax1_ps, amax2_ps):
    """chieff-basis output columns + the 1-D chi_eff prior factor on p_pe.

    This is the single spin-basis-specific step.  ``data`` already carries the
    mass-Jacobian ``p_pe = m1det * p_dL_pe`` and the source-frame masses; here
    the 1-D isotropic chi_eff prior is multiplied into ``p_pe`` (chieff basis
    is always "include"), and the legacy 10 columns are returned in order.

    The ceiling is per-sample (per-event-constant, like the other bases), so an
    event whose sampling prior declares ``a_2 ~ U(0, 0.05)`` gets ITS density,
    not the caller's default (GW-31).  Unique ``(amax_1, amax_2)`` pairs are
    grouped so one prior table serves every event that shares a ceiling; with a
    single pair this is one call on the whole array, i.e. bit-identical to the
    previous single-amax evaluation.

    The ``clip(logp, -50, None)`` guard is gone (GW-03): an out-of-support sample
    now gets ``p_pe = 0`` and is counted, rather than a floored ``2e-22`` that
    dominates every downstream weight.  Support is the prior's own
    :meth:`~gwcat.spin.ChiEffPrior.support` predicate, NOT ``isfinite(logp)``:
    the grid clamp returns a finite ~1e-12 density beyond ``amax`` (2.73e-12 at
    ``amax=0.99, chi_eff=0.995, m1=50, m2=25``), so the finiteness test admitted
    excluded samples with an inverse weight ~1e12 too large.  ``in_support``
    comes back alongside the columns so the caller can account for it.
    """
    p_pe = data["p_pe"]
    in_support = np.ones(p_pe.shape, dtype=bool)
    if data["chieff"].size > 0:
        from ..spin import chi_eff_prior_logprob_in_support
        logp_chi = np.empty(p_pe.shape, dtype=float)
        pairs = np.unique(np.stack([np.asarray(amax1_ps, dtype=float),
                                    np.asarray(amax2_ps, dtype=float)],
                                   axis=1), axis=0)
        for a1, a2 in pairs:
            m = (amax1_ps == a1) & (amax2_ps == a2)
            lp, sup = chi_eff_prior_logprob_in_support(
                data["chieff"][m], data["m1src"][m], data["m2src"][m],
                amax=float(a1), amax_2=float(a2))
            logp_chi[m] = lp
            in_support[m] = sup
        with np.errstate(over="ignore"):
            p_pe = p_pe * np.exp(logp_chi)
        p_pe = np.where(in_support, p_pe, 0.0)

    return {
        "_in_support": in_support,
        "ra": data["ra"],
        "dec": data["dec"],
        "m1det": data["m1det"],
        "m2det": data["m2det"],
        "chieff": data["chieff"],
        "dL": data["dL"],
        "p_pe": p_pe,
        "redshift": data["redshift"],
        "m1src": data["m1src"],
        "m2src": data["m2src"],
    }


def _apply_nospin_basis(data):
    """nospin-basis output columns: the legacy 10, and NO spin factor on p_pe.

    ``p_pe = m1det * p_dL_pe`` exactly -- the mass Jacobian and the distance
    prior, nothing else.  ``chieff`` is still emitted (it costs nothing and every
    plot uses it) but it is ADVISORY: no term for it appears in ``p_pe``, so
    fitting on it against this file would be wrong.  The registry says so
    declaratively (``spin.none.advisory_columns``), which is the whole point of
    the fit/advisory split.

    Everything is in support: there is no spin prior to fall outside of.
    """
    p_pe = data["p_pe"]
    return {
        "_in_support": np.ones(np.shape(p_pe), dtype=bool),
        "ra": data["ra"],
        "dec": data["dec"],
        "m1det": data["m1det"],
        "m2det": data["m2det"],
        "chieff": data["chieff"],
        "dL": data["dL"],
        "p_pe": p_pe,
        "redshift": data["redshift"],
        "m1src": data["m1src"],
        "m2src": data["m2src"],
    }


def _apply_component_basis(data, amax1_ps, amax2_ps, chip, extras):
    """component-basis output columns + the flat component-spin prior on p_pe.

    ``p_pe = m1det * p_dL_pe / (4 * amax_1 * amax_2)`` with the per-sample
    (per-event-constant) spin amax; NO chi_eff factor.  Emits the legacy 10
    columns plus ``a1``/``a2``/``cost1``/``cost2``/``chip``.

    Support (GW-03): the component prior is a flat box, so a sample is out of
    support only when a magnitude exceeds its own ``amax``.  That is a genuinely
    per-sample condition here -- unlike the projections, whose whole density
    depends on the assumed ceiling -- so it is checked directly on ``a_i``.
    """
    p_pe = data["p_pe"]
    in_support = np.ones(p_pe.shape, dtype=bool)
    if p_pe.size > 0:
        p_pe = p_pe / (4.0 * amax1_ps * amax2_ps)
        if extras is not None:
            in_support = ((np.abs(extras["a1"]) <= amax1_ps)
                          & (np.abs(extras["a2"]) <= amax2_ps)
                          & (np.abs(extras["cost1"]) <= 1.0)
                          & (np.abs(extras["cost2"]) <= 1.0))
            p_pe = np.where(in_support, p_pe, 0.0)
    out = {
        "_in_support": in_support,
        "ra": data["ra"],
        "dec": data["dec"],
        "m1det": data["m1det"],
        "m2det": data["m2det"],
        "chieff": data["chieff"],
        "dL": data["dL"],
        "p_pe": p_pe,
        "redshift": data["redshift"],
        "m1src": data["m1src"],
        "m2src": data["m2src"],
        "chip": chip,
    }
    if extras is not None:
        out["a1"] = extras["a1"]
        out["a2"] = extras["a2"]
        out["cost1"] = extras["cost1"]
        out["cost2"] = extras["cost2"]
    return out


def _apply_chieff_chip_basis(data, amax1_ps, chip, extras):
    """chieff_chip-basis output columns + the joint (chi_eff, chi_p) prior on p_pe.

    ``p_pe = m1det * p_dL_pe * exp(joint_lnprob)`` where ``joint_lnprob =
    chi_eff_chi_p_prior_logprob(chieff, chip, m1src, m2src, amax=amax_1)`` -- the
    SAME source-frame mass convention the chieff basis uses.  The joint prior
    takes a scalar ``amax``, so the (per-event-constant) ``amax_1`` array is
    grouped by unique value.  Emits the legacy 10 columns plus ``chip`` (and
    ``a1``/``a2``/``cost1``/``cost2`` when available).

    This is where the ``-50`` floor did the most damage (GW-03): chi_p reaches
    the assumed ceiling on real data where chi_eff does not, so a handful of
    floored samples captured essentially all of an event's ``1/p_pe`` weight
    (measured ESS = 1.0 out of 3337 on GW150914).  Out of support is now zero
    density, counted and reported -- and taken from the prior's own
    :meth:`~gwcat.spin.ChiEffChiPPrior.support` predicate rather than inferred
    from ``isfinite(logp)`` (GW-31).  The two agree numerically for the joint
    prior; the point is that the builder no longer relies on them agreeing.
    """
    p_pe = data["p_pe"]
    in_support = np.ones(p_pe.shape, dtype=bool)
    if p_pe.size > 0:
        from ..spin import chi_eff_chi_p_prior_logprob_in_support
        logp = np.empty(p_pe.shape, dtype=float)
        for a in np.unique(amax1_ps):
            m = amax1_ps == a
            lp, sup = chi_eff_chi_p_prior_logprob_in_support(
                data["chieff"][m], chip[m], data["m1src"][m], data["m2src"][m],
                amax=float(a))
            logp[m] = lp
            in_support[m] = sup
        with np.errstate(over="ignore"):
            p_pe = p_pe * np.exp(logp)
        p_pe = np.where(in_support, p_pe, 0.0)
    out = {
        "_in_support": in_support,
        "ra": data["ra"],
        "dec": data["dec"],
        "m1det": data["m1det"],
        "m2det": data["m2det"],
        "chieff": data["chieff"],
        "dL": data["dL"],
        "p_pe": p_pe,
        "redshift": data["redshift"],
        "m1src": data["m1src"],
        "m2src": data["m2src"],
        "chip": chip,
    }
    if extras is not None:
        out["a1"] = extras["a1"]
        out["a2"] = extras["a2"]
        out["cost1"] = extras["cost1"]
        out["cost2"] = extras["cost2"]
    return out
