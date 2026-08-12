"""PE (posterior-sample) export builder for the versioned pipeline (PR 3 / PR 6).

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
  * the same ``p_pe = m1det * p_dL_pe`` mass Jacobian, and (chieff basis always
    uses "include" semantics) the same 1-D chi_eff prior factor applied to the
    concatenated ``p_pe`` -- since GW-03 with NO floor on the log-density, in
    both this builder and the frozen v1 twin, so the parity still holds -- and
  * the same concatenation order.

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

from ..cosmology import make_cosmology, z_of_dL
from .product import ExportProduct

#: The datasets a chieff-basis PE export writes, in legacy order.
_CHIEFF_COLUMNS = ["ra", "dec", "m1det", "m2det", "chieff", "dL", "p_pe",
                   "redshift", "m1src", "m2src"]

#: Spin bases the v2 builder implements.
_KNOWN_SPIN_BASES = ("chieff", "component", "chieff_chip")

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


class ChiPDefinitionError(ValueError):
    """The stored chi_p column is not the Schmidt chi_p of the store's own
    components, so a joint (chi_eff, chi_p) prior would not describe it."""


def build_pe_product(cat, *, spin_basis="chieff", nsamp=4096, seed=0,
                     far_max=None, pastro_min=None, z_max=None,
                     replace="auto", cosmology=None, amax=0.99,
                     amax_fallback=0.99,
                     allowed_names=None, allowed_names_authoritative=True,
                     source_class=None, event_list=None,
                     allow_missing_far=False, require_far=False,
                     waveform_policy="preferred", approximant=None,
                     allow_zero_p_pe=False,
                     chi_p_definition="schmidt_recomputed",
                     chi_p_def_tol=1e-6,
                     max_out_of_support_frac=0.0,
                     allow_out_of_support=False):
    """Build a PE :class:`ExportProduct` from a :class:`~gwcat.catalog.GWCatalog`.

    For ``spin_basis="chieff"`` this reproduces the legacy
    :meth:`GWCatalog.to_darksirens` arrays and provenance exactly (see the
    module docstring for the parity contract).  ``format_version`` is NOT set
    here -- it belongs to the writer.

    The ``"component"`` and ``"chieff_chip"`` bases add spin-basis-specific
    output columns and ``p_pe`` factors (see the module docstring).  ``amax`` is
    only used by the chieff basis; the new bases read a per-event ``amax`` from
    the store meta (``spin_amax_1``/``spin_amax_2``), falling back to
    ``amax_fallback`` (with a warning) when it is NaN (old stores).

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
    """
    if spin_basis not in _KNOWN_SPIN_BASES:
        raise ValueError(
            f"unknown spin_basis={spin_basis!r}; known bases are "
            f"{list(_KNOWN_SPIN_BASES)}.")
    if chi_p_definition not in ("schmidt_recomputed", "file"):
        raise ValueError(
            f"chi_p_definition must be 'schmidt_recomputed' or 'file'; got "
            f"{chi_p_definition!r}.")

    need_extras = spin_basis != "chieff"

    # chieff basis ALWAYS uses "include" semantics: the 1-D chi_eff prior is
    # multiplied into p_pe here (Mode A), matching the legacy default.  The
    # non-chieff bases carry their own spin factors (see their apply helpers).
    if spin_basis == "chieff":
        spin_prior_mode = "include"
        chi_eff_included = True
    elif spin_basis == "component":
        spin_prior_mode = "component_flat"
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
                     approximant=approximant)

    # ── Required-parameter checks (per basis) ───────────────────────────────
    from ..schema import (DARKSIRENS_REQUIRED, COMPONENT_REQUIRED,
                          COMPONENT_TILT_ALTERNATIVES,
                          CHIEFF_CHIP_CHIP_ALTERNATIVES)
    if spin_basis == "chieff":
        need = list(DARKSIRENS_REQUIRED)
        sub._require_params(need, export="gwcat2 PE export")
    elif spin_basis == "component":
        need = list(COMPONENT_REQUIRED)
        sub._require_params(need, export="gwcat2 PE export (component basis)")
        sub._require_alternatives(
            COMPONENT_TILT_ALTERNATIVES,
            export="gwcat2 PE export (component basis)")
    else:  # chieff_chip
        need = list(DARKSIRENS_REQUIRED)
        sub._require_params(need, export="gwcat2 PE export (chieff_chip basis)")
        sub._require_alternatives(
            CHIEFF_CHIP_CHIP_ALTERNATIVES,
            export="gwcat2 PE export (chieff_chip basis)")

    per = sub.get(need, per_event=True)
    rng = np.random.default_rng(seed)

    # Extra per-sample columns (fetched ONLY for non-chieff bases so the chieff
    # rng stream / parity is untouched).
    extra_params = ([p for p in _EXTRA_SAMPLE_CANDIDATES
                     if p in sub._param_index] if need_extras else [])
    per_extra = (sub.get(extra_params, per_event=True)
                 if extra_params else {})

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

    cols = {k: [] for k in ["m1det", "m2det", "dL", "ra", "dec",
                            "chieff", "p_pe", "redshift", "m1src", "m2src"]}
    kept = []
    kept_H0, kept_Om0 = [], []
    kept_ss_name, kept_ss_approx, kept_ss_reason = [], [], []

    # ── Non-chieff spin scaffolding (empty / unused for chieff) ─────────────
    # Per-event spin amax from the store meta (aligned with selected events).
    if need_extras:
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

        kept_amax1, kept_amax2 = [], []
        a1_list, a2_list, cost1_list, cost2_list, chip_list = [], [], [], [], []
        chip_src_list = []
        chip_maxdiff_list = []
        chip_mismatch_events, chip_no_ingredient_events = [], []
        amax_src1_list, amax_src2_list, amax_infl_list = [], [], []
        samples_bound_events = []
        fallback_events, unrecognized_events, mismatch_events = [], [], []

    def _resolve_cost(e, avail_cos, avail_tilt, cos_name, tilt_name):
        """cos_tilt samples for event ``e`` (idx_orig applied by caller):
        prefer the stored cos_tilt, else cos(stored tilt), else None."""
        if avail_cos[e]:
            return per_extra[cos_name][e]
        if avail_tilt[e]:
            return np.cos(per_extra[tilt_name][e])
        return None

    sel_rows = np.asarray(sub._sel)
    reasons_arr = getattr(sub, "_selection_reasons", None)
    for e in range(sub.n_events):
        n = len(per["luminosity_distance"][e])
        if n == 0:
            continue

        cosmo_e = _cosmo_for(e)

        dL_e = per["luminosity_distance"][e]
        m1_e = per["mass_1"][e]
        m2_e = per["mass_2"][e]

        # Per-sample z_max cut
        if z_max is not None:
            z_e = z_of_dL(dL_e, cosmo_e)
            keep = z_e <= z_max
            if not keep.any():
                continue
            dL_e = dL_e[keep]
            m1_e = m1_e[keep]
            m2_e = m2_e[keep]
            idx_map = np.nonzero(keep)[0]
        else:
            idx_map = np.arange(n)

        n_kept = len(idx_map)
        rep = (n_kept < nsamp) if replace == "auto" else bool(replace)
        if n_kept < nsamp and not rep:
            warnings.warn(f"Event {sub.event_names[e]}: only {n_kept} samples "
                          f"after z_max cut, but replace=False and nsamp={nsamp}. "
                          f"Skipping.")
            continue
        idx_local = rng.choice(n_kept, size=nsamp, replace=rep)
        idx_orig = idx_map[idx_local]

        m1 = per["mass_1"][e][idx_orig]
        m2 = per["mass_2"][e][idx_orig]
        dL = per["luminosity_distance"][e][idx_orig]
        p_dL = per["p_dL_pe"][e][idx_orig]

        # Jacobian: uniform detector-frame component-mass prior
        p_pe = m1 * p_dL

        # Redshift and source masses under THIS event's PE cosmology
        z = z_of_dL(dL, cosmo_e)

        cols["m1det"].append(m1)
        cols["m2det"].append(m2)
        cols["dL"].append(dL)
        cols["ra"].append(per["ra"][e][idx_orig])
        cols["dec"].append(per["dec"][e][idx_orig])
        cols["chieff"].append(per["chi_eff"][e][idx_orig])
        cols["p_pe"].append(p_pe)
        cols["redshift"].append(z)
        cols["m1src"].append(m1 / (1 + z))
        cols["m2src"].append(m2 / (1 + z))
        kept.append(sub.event_names[e])
        kept_H0.append(float(per_event_H0[e]))
        kept_Om0.append(float(per_event_Om0[e]))
        row = sel_rows[e]
        kept_ss_name.append(_ss_meta(sub, row, "sample_set_name"))
        kept_ss_approx.append(_ss_meta(sub, row, "approximant"))
        kept_ss_reason.append(
            str(reasons_arr[e]) if reasons_arr is not None
            and e < len(reasons_arr) else "")

        # ── Non-chieff per-event spin columns + amax resolution ─────────────
        if need_extras:
            name = sub.event_names[e]

            # Resolve per-event amax (NaN -> fallback, recorded).
            a1max = (float(sel_amax1[e]) if sel_amax1 is not None
                     else float("nan"))
            a2max = (float(sel_amax2[e]) if sel_amax2 is not None
                     else float("nan"))
            used_fallback = False
            if not np.isfinite(a1max):
                a1max = float(amax_fallback)
                used_fallback = True
            if not np.isfinite(a2max):
                a2max = float(amax_fallback)
                used_fallback = True
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
            a1_e = (per_extra["a_1"][e][idx_orig]
                    if avail_a1[e] else None)
            a2_e = (per_extra["a_2"][e][idx_orig]
                    if avail_a2[e] else None)

            # ── Resolved-amax provenance (GW-04) ────────────────────────────
            # "analytic"      the ingested analysis's own prior covers its
            #                 samples -- the honest case;
            # "samples_bound" the parsed prior does NOT cover the samples, so
            #                 the ceiling is raised to max|a_i| and the
            #                 inflation is recorded;
            # "fallback"      no stored amax at all, a fabricated ceiling.
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
            src1 = src2 = "fallback" if used_fallback else "analytic"
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
            cost1_full = _resolve_cost(e, avail_cos1, avail_tilt1,
                                       "cos_tilt_1", "tilt_1")
            cost2_full = _resolve_cost(e, avail_cos2, avail_tilt2,
                                       "cos_tilt_2", "tilt_2")
            cost1_e = None if cost1_full is None else cost1_full[idx_orig]
            cost2_e = None if cost2_full is None else cost2_full[idx_orig]

            # chip resolution (GW-08).  ``chi_p_definition`` decides whether the
            # exported column is the Schmidt chi_p of the store's OWN
            # (a_i, cos_tilt_i, m_i) -- the definition ChiEffChiPPrior is built
            # for -- or the release's stored column, which for six GWTC-2.1/3
            # C01:Mixed events is a different quantity.
            have_ingredients = (a1_e is not None and a2_e is not None
                                and cost1_e is not None and cost2_e is not None)
            chip_from_file = None
            if avail_chip[e]:
                chip_from_file = per_extra["chi_p"][e][idx_orig]
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
    if spin_basis == "chieff":
        columns = _apply_chieff_basis(data, amax=amax)
    else:
        amax1_arr = np.asarray(kept_amax1, dtype=float)
        amax2_arr = np.asarray(kept_amax2, dtype=float)
        amax1_ps = (np.repeat(amax1_arr, nsamp) if nobs else np.array([]))
        amax2_ps = (np.repeat(amax2_arr, nsamp) if nobs else np.array([]))
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

        # Basis-specific provenance attrs (never written for chieff).
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

    # Sanity check
    expected = nobs * nsamp
    assert columns["m1det"].size == expected, \
        f"data length {columns['m1det'].size} != nobs*nsamp = {expected}"

    homogeneous = bool(len(set(str(k) for k in kept)) == len(kept))

    # ── Provenance attrs (everything the legacy exporter records EXCEPT ──────
    # format_version, which is the writer's; plus the new spin_basis). The two
    # legacy-compat spin attrs (spin_prior_mode / chi_eff_prior_applied_to_p_pe
    # / chi_eff_in_p_pe) are added by the chieff writer, not here, so a future
    # non-chieff basis never carries them.
    attrs = {
        # darksirens core
        "nsamp": int(nsamp),
        "nobs": int(nobs),
        "mock_data": False,
        # spin basis (new in v2)
        "spin_basis": spin_basis,
        # provenance
        "compact_type": "",
        "mass_prior_basis": "uniform_detector_frame",
        "mass_jacobian_applied": True,
        "distance_prior_removed": False,
        "cosmology_mode": cosmology_mode,
        "cosmology_override_used": bool(cosmology is not None),
        "source_frame_under_recorded_cosmology": True,
        "cosmology_per_event_varies": bool(cosmology_per_event_varies),
        "cosmology_H0_per_event": kept_H0_arr,
        "cosmology_Om0_per_event": kept_Om0_arr,
        "chi_eff_amax": float(amax),
        "pe_cosmology_H0": pe_H0,
        "pe_cosmology_Om0": pe_Om0,
        "source_class_filter": ("" if source_class is None
                                else str(source_class)),
        "event_list_filter": (
            "" if event_list is None
            else (str(event_list) if isinstance(event_list, (str, bytes))
                  else "custom_sequence")),
        "far_policy": getattr(sub, "_far_policy", "none"),
        "allow_missing_far": bool(allow_missing_far),
        "require_far": bool(require_far),
        "n_events_missing_far": int(getattr(sub, "_n_missing_far", 0)),
        "waveform_policy": str(waveform_policy),
        "approximant": "" if approximant is None else str(approximant),
        "homogeneous_sample_sets": homogeneous,
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
    }
    attrs.update(spin_attrs)
    # Per-sample support mask, written as uint8 so the consumer can mask without
    # re-deriving the prior.
    columns["in_support"] = in_support.astype(np.uint8)

    # ── Validation-summary feed (writer fills output_path + summary_context) ─
    from ..validation_summary import summarize_catalog
    summary = summarize_catalog(sub)
    summary.update({
        "kind": "darksirens_export",
        "n_events_considered": int(sub.n_events),
        "n_events_exported": int(nobs),
        "n_events_skipped_after_selection": int(sub.n_events - nobs),
        "event_names_exported": [str(k) for k in kept],
        "nsamp_per_event": int(nsamp),
        "spin_basis": spin_basis,
        "source_class_filter": (None if source_class is None
                                else str(source_class)),
        "event_list_filter": (
            None if event_list is None
            else (str(event_list)
                  if isinstance(event_list, (str, bytes))
                  else "custom_sequence")),
        "far_policy": getattr(sub, "_far_policy", "none"),
        "allow_missing_far": bool(allow_missing_far),
        "require_far": bool(require_far),
        "n_events_missing_far": int(getattr(sub, "_n_missing_far", 0)),
        "spin_prior_mode": spin_prior_mode,
        "chi_eff_prior_applied_to_p_pe": bool(chi_eff_included),
        "cosmology_mode": cosmology_mode,
        "cosmology_override_used": bool(cosmology is not None),
        "cosmology_per_event_varies": bool(cosmology_per_event_varies),
        "waveform_policy": str(waveform_policy),
        "approximant": None if approximant is None else str(approximant),
        "homogeneous_sample_sets": homogeneous,
        "n_samples_out_of_support": int(n_out),
        "frac_samples_out_of_support": float(frac_out),
        "prior_reweight_ess_min": (float(np.min(ess_per_event))
                                   if ess_per_event.size else None),
        "prior_reweight_ess_median": (float(np.median(ess_per_event))
                                      if ess_per_event.size else None),
        "chi_p_definition": str(chi_p_definition),
    })

    return ExportProduct(kind="pe", columns=columns, attrs=attrs,
                         spin_basis=spin_basis, summary=summary)


def _ss_meta(sub, row, field):
    v = sub.meta.get(field)
    if v is None:
        return ""
    x = v[int(row)]
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def _apply_chieff_basis(data, *, amax):
    """chieff-basis output columns + the 1-D chi_eff prior factor on p_pe.

    This is the single spin-basis-specific step.  ``data`` already carries the
    mass-Jacobian ``p_pe = m1det * p_dL_pe`` and the source-frame masses; here
    the 1-D isotropic chi_eff prior is multiplied into ``p_pe`` (chieff basis
    is always "include"), and the legacy 10 columns are returned in order.

    The ``clip(logp, -50, None)`` guard is gone (GW-03): an out-of-support sample
    now gets ``p_pe = 0`` and is counted, rather than a floored ``2e-22`` that
    dominates every downstream weight.  ``in_support`` comes back alongside the
    columns so the caller can account for it.
    """
    p_pe = data["p_pe"]
    in_support = np.ones(p_pe.shape, dtype=bool)
    if data["chieff"].size > 0:
        from ..spin import chi_eff_prior_logprob
        logp_chi = np.asarray(
            chi_eff_prior_logprob(data["chieff"], data["m1src"],
                                  data["m2src"], amax=amax), dtype=float)
        in_support = np.isfinite(logp_chi)
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
    density, counted and reported.
    """
    p_pe = data["p_pe"]
    in_support = np.ones(p_pe.shape, dtype=bool)
    if p_pe.size > 0:
        from ..spin import chi_eff_chi_p_prior_logprob
        logp = np.empty(p_pe.shape, dtype=float)
        for a in np.unique(amax1_ps):
            m = amax1_ps == a
            lp = chi_eff_chi_p_prior_logprob(
                data["chieff"][m], chip[m], data["m1src"][m], data["m2src"][m],
                amax=float(a))
            logp[m] = np.asarray(lp, dtype=float)
        in_support = np.isfinite(logp)
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
