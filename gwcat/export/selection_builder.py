"""Selection-function export builder for the versioned pipeline (PR5).

The two rules this module enforces
---------------------------------
**R1 (projection rule).**  A projected spin coordinate is definable only against
a uniform-magnitude/isotropic injected draw, so a projection basis must REFUSE --
not approximate -- when the campaign is something else.  The component basis
assumes nothing about the campaign and is exact for all of them.

**R2 (support rule).**  A density that appears in a denominator may never be
floored.  On this side it is stronger than on the PE side: a DETECTED injection
with zero assumed draw density is a contradiction, not a small number, so the
export refuses rather than writing a zero the consumer would silently drop.


:func:`build_selection_product` reproduces -- for the ``spin_basis="chieff"``
case -- the arrays and provenance of the legacy
:meth:`gwcat.selection.SelectionSet.to_darksirens` /
:meth:`gwcat.selection.CombinedSelectionSet.to_darksirens` EXACTLY for identical
keyword arguments, and returns them as an
:class:`gwcat.export.product.ExportProduct` (``kind="selection"``) instead of
writing a file.  It additionally supports two other spin bases -- ``"component"``
and ``"chieff_chip"`` -- that differ ONLY in the per-injection spin factor
applied to ``pdraw`` and in the extra spin columns emitted.

Parity contract (why this mirrors the *combined* exporter for BOTH single and
multi-campaign inputs)
----------------------------------------------------------------------------
The legacy combined exporter, per campaign ``k`` (Essick et al. 2023):

  * ``keep = detected_mask(far_threshold) & source_class_mask(source_class)``,
  * ``frac = N_k / N_total`` with ``N_total = sum(N_k)`` (source-class filtering
    is subsetting only -- ``N_k`` is unchanged),
  * ``pdraw_k = _pdraw[keep] * frac`` appended in campaign order (empty
    campaigns skipped),

then, on the concatenated arrays, applies the 1-D chi_eff prior swap
``pdraw *= exp(chi_eff_prior_logprob(chieff, m1src, m2src, amax))``.
For a SINGLE campaign, ``N_total == N_k`` so ``frac == 1.0`` exactly
and ``_pdraw[keep] * 1.0`` is bit-identical to ``_pdraw[keep]``; running the
single case through this same code path therefore reproduces the single-campaign
legacy exporter's ``pdraw`` bit-for-bit as well.  The chi_eff swap is applied
per campaign and concatenated in the SAME order, which for one shared ceiling is
elementwise identical to the single post-concat call -- so with an explicit
numeric ``amax=`` the chieff-basis v2 ``pdraw`` still equals the v1 ``pdraw``
array under ``assert_array_equal``.

Spin-basis-specific step (the only place the bases diverge)
-----------------------------------------------------------
* **chieff**  ``pdraw *= exp(chi_eff_prior_logprob(chieff, m1src, m2src,
  amax_detected_k))`` -- the legacy swap, minus the floor (GW-03), and since
  GW-31 at each campaign's OWN detected ceiling rather than one caller default.
  The swap replaces the campaign's real spin-draw density with the analytic
  chi_eff marginal, so it must be the marginal of the density that campaign
  actually drew from: end-O3 injects ``a ~ U(0, 0.998)``, and evaluating its
  marginal at 0.99 leaves a chi_eff-DEPENDENT error in pdraw (measured: x1.008
  at chi_eff = 0, x0.79 at chi_eff = 0.9) that no per-event normalisation
  cancels.
* **component**  ``pdraw *= exp(ln_spin_component)`` with the per-injection,
  frac-independent ``ln_spin_component`` (public accessor); NO clip and NO
  chi_eff factor.  Requires ``component_spin_available`` for every campaign.
* **chieff_chip**  ``pdraw *= exp(chi_eff_chi_p_prior_logprob(chieff, chip,
  m1src, m2src, amax=amax_detected_k))`` with each campaign's own
  DETECTED amax; requires a single uniform-isotropic injected spin draw with a
  detectable amax per campaign (else :class:`SpinBasisError`).  If a campaign's
  ``amax_1 != amax_2`` the joint prior (which assumes a single amax) uses
  ``amax_1`` and warns -- mirroring the PE builder's convention.
* **chieff_reference**  the component factor FIRST (so the intermediate is the
  component-basis pdraw bit-for-bit), then one explicit change of spin
  reference::

      pdraw = pdraw_component * p_iso(chieff | m1src, m2src, a_ref) * 4 a_ref^2

  i.e. ``pdraw_component * p_iso(chieff|q,a_ref) / p_ref(a,cosθ)`` with the
  reference spin prior ``p_ref = 1/(4 a_ref^2)`` (isotropic tilts, magnitudes
  uniform on ``[0, a_ref]``).  ``a_ref`` is REQUIRED and explicit
  (``spin_reference_amax=``): it is not a property of the campaign but a
  declaration about the reference the consumer's PE side divides out, and a
  default would be a fabricated one.  See the section below for why this is not
  the ``chieff`` swap.

Why ``chieff_reference`` is not the ``chieff`` swap
---------------------------------------------------
Both write a density in ``(m1det, q, dL, chi_eff)``, and that is the whole of
the resemblance.

* ``chieff`` **substitutes**: it throws the campaign's spin draw away and writes
  the analytic marginal in its place.  That is the right density only if the
  campaign drew its spins that way, which is why ``_check_chieff_swap_valid``
  refuses it on a measured non-uniform-isotropic campaign (GW-06).  **That guard
  stays exactly as it is**; the new basis is not a way around it.
* ``chieff_reference`` **reweights**: the campaign's exact per-injection
  component density stays in the numerator and the reference is divided in, so
  nothing is assumed about how the campaign drew its spins and the basis is
  buildable on the campaigns ``chieff`` must refuse.

The estimator this makes exact.  With ``s`` the spin degrees of freedom beyond
``chi_eff``, the selection integral a ``chi_eff``-only population model needs is

    alpha = int p_pop(theta_4) E_ref[P_det | chi_eff, theta_4] d theta_4,

the non-``chi_eff`` spin content averaged over the SAME reference prior the PE
side assumes.  Monte-Carlo'd over the campaign's actual draws its weight is

    w = p_pop(theta_4) J * [ p_ref(a,cosθ) / p_iso(chi_eff|q,a_ref) ]
                         / [ p_base(m1det,q,dL) * p_draw(a,cosθ) ]
      = p_pop(theta_4) J / pdraw,

because ``p_ref / p_iso`` is precisely the reference conditional density of the
spin components given ``chi_eff`` and ``pdraw_component = p_base * p_draw``.

Out of the reference support -- and why it is NOT the GW-03 contradiction.
``p_ref`` is exactly zero where ``a_1 > a_ref`` or ``a_2 > a_ref``, so those
injections carry weight exactly zero: not a small density in a denominator, but
a region the DECLARED reference assigns no probability to at all.  Under the
substituting bases an out-of-support detected injection is fatal, because there
the zero is the assumed prior contradicting a draw that really happened.  Here
the draw and the detection are untouched -- it is the reference that stops -- so
the honest treatment is a zero weight, written as a large sentinel ``pdraw``
(``out_of_reference_pdraw``, default 1e300) whose inverse weight is ~1e-300,
counted in ``spin_reference_excluded_rows``, and never confused with the fatal
case, which still raises.

The converse hole is real and is checked: if a campaign's magnitudes stop BELOW
``a_ref`` the reference support is not covered, the reweighting has nothing to
put in the missing region, and alpha comes out biased low.  With a detectable
(uniform-magnitude) ceiling that is provable and raises under ``strict``; on a
campaign with no detectable ceiling it falls back to the largest magnitude
actually drawn and warns, because "not observed" is weaker than "not drawn".

Columns (all bases): the legacy 10 (``m1det, m2det, dL, chieff, ra, dec, m1src,
m2src, redshift, pdraw``) plus ``a1, a2, cost1, cost2, chip`` whenever every
contributing campaign carries them (they are free -- read from the additive
component-spin accessors -- so the chieff basis includes them too).
``sky_marginal=True`` drops ``ra``/``dec`` and records ``sky_marginalized``.

Cumulative multi-run mixtures (GW-39)
-------------------------------------
The GWTC-5.0 O1-O4b file is ONE campaign whose rows come from six runs.  Three
ways of exporting it look plausible and are wrong, and each is refused here
rather than written:

* a FAR-only cut (:class:`MixtureDetectionError`): the O1/O2 rows are
  semianalytic and carry no FAR, so none is detected while their draws and
  exposure stay in ``ndraw``/``T_obs`` -- the release's rule is SNR on O1/O2
  rows and search FAR on O3/O4 rows, applied per run
  (:meth:`gwcat.selection.SelectionSet.detected_mask`);
* the ``chieff``/``chieff_chip`` substitution bases: the file ships one joint
  spin density, the uniform/isotropic assumption cannot be verified on it, and
  its spins are measured non-isotropic -- ``chieff_reference`` reweights
  instead and is exact;
* combining it with a campaign that overlaps it in time
  (:class:`OverlappingCampaignError`), which double-counts exposure.

The export records the per-run provenance (``run_labels``,
``detection_rule_per_run``, ``n_detected_per_run``, the two window tables, the
per-run mixture weights, the derived ``N_per_run``/``T_per_run_s`` with
``T_definition_per_run``, ``z_draw_max_per_run``) and runs the reference
coverage check per run as well as per campaign.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import h5py

from ..params import BLOCKS, DEFAULT_PARAMETER_SPACE, get_space
from ..source_class import (format_source_class_filter, CUT_ESTIMATOR_ATTR,
                            selection_cut_estimator)
from ..spin import (AMAX_AUTO, chi_eff_chi_p_prior_logprob_in_support,
                    chi_eff_prior_logprob_in_support, parse_amax_option)
from ..selection import (SelectionSet, CombinedSelectionSet,
                         PDRAW_STATE_BY_BASIS, _selection_provenance_dict,
                         _refuse_mixed_cosmology, _cosmo_or_nan,
                         _cosmology_is_mixed, SNR_COLUMN,
                         CAMPAIGN_CUMULATIVE_MIXTURE, MixtureDetectionError,
                         MixtureInvariantError, _mixture_far_only_message)
from ..observing_runs import (RUN_LABELS, SEMIANALYTIC_RUNS,
                              INJECTION_SUPPORT_GPS, EXPOSURE_WINDOWS_GPS,
                              MIXTURE_COMPONENTS, MIXTURE_RELEASE_ANCHORS,
                              derive_mixture_bookkeeping)
from .product import ExportProduct

#: Spin bases the selection builder implements.  A registry space outside this
#: tuple is declarable but not yet buildable; the CLI and the validator read
#: this tuple (as ``SUPPORTED_SPIN_BASES``) so they cannot advertise or reject
#: a different set than the builder enforces.
_KNOWN_SPIN_BASES = ("chieff", "component", "chieff_chip",
                     "chieff_reference")
SUPPORTED_SPIN_BASES = _KNOWN_SPIN_BASES

#: Bases whose ``pdraw`` is the campaign's own density reweighted to a DECLARED
#: reference spin prior rather than the campaign's own density (or a
#: substitution for it).  Each requires an explicit ``spin_reference_amax``.
_REFERENCE_BASES = ("chieff_reference",)

#: ``pdraw`` written for a row the declared reference assigns zero density to.
#: Large rather than infinite so the file stays finite-and-positive (the loader
#: contract every consumer enforces) while the row's importance weight
#: ``p_pop * J / pdraw`` is ~1e-300 -- 290-odd orders of magnitude below any real
#: weight, and exactly zero the moment it is multiplied by a population density.
#: Same value as the sentinel the marked-transition analysis had been patching in
#: downstream, so a product built here matches one built there row for row.
OUT_OF_REFERENCE_PDRAW = 1e300

#: The cumulative-mixture SNR column used by the optional OR-branch.
_SNR_COLUMN = SNR_COLUMN

#: ``detection_policy`` values the builder (and ``export selection
#: --detection-policy``) accept.  ``far``: the FAR cut, OR-ed with the SNR
#: column when ``snr_threshold`` is given.  ``lvk-cumulative``: the release's
#: per-run rule on a cumulative mixture (SNR on O1/O2 rows, the run's own
#: search FARs on O3/O4 rows); it requires ``snr_threshold`` and a mixture.
DETECTION_POLICIES = ("far", "lvk-cumulative")

#: Spin formats that carry ONE joint (masses, redshift, spins) draw density.
#: The chi_eff/chi_eff_chip swaps need the campaign's spins to be uniform in
#: magnitude and isotropic, and a joint density admits no such check.
_JOINT_SPIN_FORMATS = ("joint_cartesian", "joint_polar")


class OverlappingCampaignError(ValueError):
    """Two campaigns in one selection product cover the same GPS time.

    A cumulative mixture already contains every run's exposure in its
    ``total_generated``/``total_analysis_time`` and per-row weights; adding a
    campaign that overlaps it in time (endo3 for O3, rpo4ab for O4) counts that
    exposure twice and double-counts its detections in the Essick sum.
    """

#: Extra per-injection spin columns (emitted when available for every campaign).
#: Read off the component block rather than re-typed (GW-19): the registry is the
#: single place the column set lives, so a new spin coordinate is one edit there
#: instead of edits here, in the PE builder, in the writers and in the validator.
_EXTRA_COLUMNS = tuple(
    c for c in BLOCKS["spin.component_polar"].columns if c not in ("chip",))


class SpinBasisError(RuntimeError):
    """A requested spin basis is incompatible with a campaign's injected draw.

    Raised (for ``spin_basis="chieff_chip"``) when a campaign's injected spin
    distribution is not a single uniform-isotropic draw or has an undetectable
    ``amax`` -- e.g. a cumulative mixture of spin sub-populations, for which no
    single-``amax`` joint (chi_eff, chi_p) prior is defined.  The message names
    the offending file and the reason, and points at ``spin_basis="component"``
    (which retains the exact per-injection component spin draw for any file).
    """


def _detect_keep(s, far_threshold, source_class, snr_threshold, z_max=None,
                 acknowledge_semianalytic_excluded=False):
    """``(keep, detect, sc_mask)`` for one campaign, mirroring the legacy masks.

    ``keep = detect & sc_mask``.  For a single campaign ``detect`` is the FAR
    cut, OR-ed with the SNR cut (``snr > snr_threshold``) when
    ``snr_threshold`` is not ``None``.  For a cumulative mixture (GW-39) it is
    the RUN-AWARE rule of :meth:`gwcat.selection.SelectionSet.detected_mask`
    -- SNR on the O1/O2 rows only, the run's own search FARs on the O3/O4 rows
    only -- and a FAR-only request on a mixture with semianalytic rows raises
    :class:`MixtureDetectionError` unless
    ``acknowledge_semianalytic_excluded=True``.  ``sc_mask`` is intersected
    with ``z <= z_max`` when a redshift truncation is in force.

    ``z_max`` is the injection-side counterpart of the PE export's per-sample
    redshift truncation (GW-37).  It had none: the PE side dropped posterior
    samples above the cut while mu kept its full-z content, so the two sides
    integrated over different redshift ranges.  Like the source-class filter
    this is SUBSETTING, not reweighting -- ``ndraw`` is untouched, so the
    Essick fractions N_k/N_total are unchanged.
    """
    if (s.campaign_kind == CAMPAIGN_CUMULATIVE_MIXTURE
            and snr_threshold is None
            and not acknowledge_semianalytic_excluded):
        n_semi = int(np.isin(np.asarray(s.run), SEMIANALYTIC_RUNS).sum())
        if n_semi:
            raise MixtureDetectionError(
                _mixture_far_only_message(s.path, n_semi))
    detect = s.detected_mask(
        far_threshold, snr_threshold=snr_threshold,
        acknowledge_semianalytic_excluded=acknowledge_semianalytic_excluded)
    sc_mask = s.source_class_mask(source_class)
    if z_max is not None:
        sc_mask = sc_mask & (np.asarray(s._z, dtype=float) <= float(z_max))
    return detect & sc_mask, detect, sc_mask


def _refuse_overlapping_campaigns(set_list):
    """Refuse a cumulative mixture combined with a campaign that overlaps it.

    A mixture's normalisation (``total_generated``, ``total_analysis_time``) and
    its per-row weights already hold every run it spans; a second campaign over
    any of that time -- endo3 for O3, rpo4ab for O4 -- would count the shared
    exposure twice.  The overlap is judged on GPS ranges (the rows' own times,
    else the campaign's ``gps_start``/``gps_end`` attrs); a campaign whose
    range is unknown is refused too, because an overlap cannot be ruled out.
    """
    if len(set_list) < 2:
        return
    mixtures = [s for s in set_list
                if s.campaign_kind == CAMPAIGN_CUMULATIVE_MIXTURE]
    for m in mixtures:
        lo, hi = m.gps_range
        for o in set_list:
            if o is m:
                continue
            rng = o.gps_range
            if rng is None:
                raise OverlappingCampaignError(
                    f"{m.path} is a cumulative multi-run mixture spanning GPS "
                    f"[{lo:.0f}, {hi:.0f}], and it is combined with {o.path}, "
                    f"whose GPS range is unknown (no row times, no "
                    f"gps_start/gps_end attrs), so an overlap cannot be ruled "
                    f"out. A mixture already contains every run's exposure; "
                    f"export it alone.")
            if rng[0] <= hi and lo <= rng[1]:
                raise OverlappingCampaignError(
                    f"{m.path} is a cumulative multi-run mixture spanning GPS "
                    f"[{lo:.0f}, {hi:.0f}], and it is combined with {o.path} "
                    f"(GPS [{rng[0]:.0f}, {rng[1]:.0f}]), which overlaps it. "
                    f"The mixture's total_generated, total_analysis_time and "
                    f"per-row weights already carry that time's exposure, so "
                    f"combining the two double-counts it (and its detected "
                    f"injections) in the selection integral. Export the "
                    f"mixture alone; to use another campaign for one run, "
                    f"build that run from its own campaign instead.")


def _mixture_run_attrs(s, keep, detect, far_threshold, snr_threshold,
                       mixture_anchors):
    """The per-run provenance attrs of one cumulative-mixture campaign (GW-39).

    Everything is read from the file itself (row times, weights, attrs) except
    the O3/O4 component (T, N) anchors used to solve the O1/O2 bookkeeping,
    which come from ``mixture_anchors`` or, by default, the bundled
    :data:`gwcat.observing_runs.MIXTURE_RELEASE_ANCHORS` entry of this release.
    """
    _str = h5py.string_dtype()
    run = np.asarray(s.run)
    runs = [r for r in RUN_LABELS if np.any(run == r)]
    z = np.asarray(s._z, dtype=float)
    w = np.asarray(s._weights, dtype=float)
    rules = s.detection_rule_per_run(far_threshold, snr_threshold)
    attrs = {
        "cumulative_mixture": True,
        "run_labels": np.array(runs, dtype=_str),
        "n_rows_per_run": np.array([int(np.sum(run == r)) for r in runs],
                                   dtype=np.int64),
        # Rows WRITTEN per run: detection AND any source-class / z_max subset,
        # so they sum to n_detected.  The detection-rule count alone is next.
        "n_detected_per_run": np.array(
            [int(np.sum(keep & (run == r))) for r in runs], dtype=np.int64),
        "n_passing_detection_rule_per_run": np.array(
            [int(np.sum(detect & (run == r))) for r in runs], dtype=np.int64),
        "injection_support_gps_per_run": np.array(
            [INJECTION_SUPPORT_GPS[r] for r in runs], dtype=float),
        "exposure_windows_gps_per_run": np.array(
            [EXPOSURE_WINDOWS_GPS[r] for r in runs], dtype=float),
        "row_time_range_gps_per_run": np.array(
            [(float(np.min(s._time[run == r])), float(np.max(s._time[run == r])))
             for r in runs], dtype=float),
        "detection_rule_per_run": np.array([rules[r] for r in runs],
                                           dtype=_str),
        "mixture_weights_per_run": json.dumps(
            {r: [float(x) for x in np.unique(w[run == r])] for r in runs}),
        "z_draw_max_per_run": np.array(
            [float(np.max(z[run == r])) for r in runs], dtype=float),
        "o3_draw_density_source": "mixture_joint_lnpdraw",
    }

    comps = [c for c, _, _ in MIXTURE_COMPONENTS]
    attrs["mixture_components"] = np.array(comps, dtype=_str)
    attrs["mixture_component_runs"] = json.dumps(
        {c: list(r) for c, r, _ in MIXTURE_COMPONENTS})
    attrs["T_definition_per_run"] = np.array(
        [d for _, _, d in MIXTURE_COMPONENTS], dtype=_str)
    key = (int(s._ndraw), int(round(s._total_analysis_time_s)))
    anchors = (mixture_anchors if mixture_anchors is not None
               else MIXTURE_RELEASE_ANCHORS.get(key))
    has_semi = bool(np.isin(run, SEMIANALYTIC_RUNS).any())
    if anchors is None or not has_semi:
        if has_semi:
            warnings.warn(
                f"{s.path}: no O3/O4 component anchors are known for this "
                f"cumulative mixture (total_generated, total_analysis_time) = "
                f"{key}, so its per-run N_k and T_k cannot be derived from the "
                f"weights; N_per_run and T_per_run_s are written as NaN. pdraw "
                f"is unaffected (it uses the weights directly).")
        attrs["N_per_run"] = np.full(len(comps), np.nan)
        attrs["T_per_run_s"] = np.full(len(comps), np.nan)
        attrs["N_per_run_integrality_residual"] = np.full(2, np.nan)
        attrs["mixture_bookkeeping_status"] = (
            "unavailable_no_anchor" if has_semi
            else "not_derived_no_semianalytic_rows")
        attrs["mixture_bookkeeping_anchor_sources"] = json.dumps({})
        return attrs
    try:
        book = derive_mixture_bookkeeping(
            run, w, s._ndraw, s._total_analysis_time_s, anchors)
    except ValueError as exc:
        raise MixtureInvariantError(f"{s.path}: {exc}") from exc
    attrs["N_per_run"] = np.array(book["N"], dtype=float)
    attrs["T_per_run_s"] = np.array(book["T_s"], dtype=float)
    attrs["N_per_run_integrality_residual"] = np.array(
        book["N_integrality_residual"], dtype=float)
    attrs["mixture_bookkeeping_status"] = "derived_from_weights_and_anchors"
    attrs["mixture_bookkeeping_anchor_sources"] = json.dumps(
        book["anchor_sources"])
    return attrs


def _reference_coverage_per_run(s, a_ref):
    """``(runs, covered, bound)`` -- the a_ref coverage check, run by run.

    The per-campaign check takes the largest magnitude drawn over the WHOLE
    file, so one run reaching a_ref hides another that stops short of it.  A
    mixture's runs were drawn separately (O3 by endo3, to 0.998), so the check
    is repeated on each run's own rows: the per-run bound is the smaller of the
    two bodies' largest drawn magnitudes.
    """
    run = np.asarray(s.run)
    runs = [r for r in RUN_LABELS if np.any(run == r)]
    a1 = np.asarray(s._a1, dtype=float)
    a2 = np.asarray(s._a2, dtype=float)
    bound = [float(min(a1[run == r].max(), a2[run == r].max())) for r in runs]
    covered = [bool(b >= a_ref) for b in bound]
    bad = [f"{r} (magnitudes reach {b:.6g})"
           for r, b, ok in zip(runs, bound, covered) if not ok]
    if bad:
        warnings.warn(
            f"spin_reference_amax={a_ref} is above the largest spin magnitude "
            f"DRAWN in run(s) {', '.join(bad)} of {s.path}. These runs carry "
            f"no detectable magnitude ceiling, so this is a sample maximum, "
            f"not a support bound -- evidence of a coverage hole in those runs, "
            f"not proof. Recorded in spin_reference_coverage_per_run.")
    return runs, covered, bound


def _campaign_chieff_amax(s, forced_amax, amax_fallback):
    """The ``(amax_1, amax_2)`` the chi_eff swap must use for one campaign.

    ``("detected", a1, a2)`` from the campaign's own injected draw -- the only
    ceiling the swap may honestly use, since it REPLACES that draw's spin
    density with the analytic marginal of a ``U(0, amax)`` magnitude prior.
    ``("caller", ...)`` when a numeric ``amax=`` was passed, and
    ``("fallback", ...)`` for a campaign carrying no spin draw densities at all
    (the legacy spin-less files, for which the swap is the only basis available
    and no ceiling is knowable) -- which warns, because a fabricated ceiling in a
    density that depends on it is exactly what GW-31 was about.
    """
    if forced_amax is not None:
        return "caller", float(forced_amax), float(forced_amax)
    detected = (s.spin_meta or {}).get("amax_detected")
    if (detected is not None
            and all(a is not None and np.isfinite(a) for a in detected[:2])):
        return "detected", float(detected[0]), float(detected[1])
    warnings.warn(
        f"{s.path}: spin_basis='chieff' needs the ceiling of the injected spin "
        f"magnitude prior, but this campaign's is undetectable (no per-spin "
        f"draw densities). Falling back to amax={amax_fallback}, recorded as "
        f"chi_eff_amax_source_per_campaign='fallback'. The chi_eff marginal "
        f"depends on that ceiling in a chi_eff-DEPENDENT way, so if the real "
        f"one differs the error does not cancel; use spin_basis='component' "
        f"where the campaign carries its exact per-injection spin draw.")
    return "fallback", float(amax_fallback), float(amax_fallback)


def _campaign_chieff_lnfactor(s, keep, amax_1, amax_2):
    """Per-injection ln chi_eff prior for one campaign, support APPLIED.

    ``support()`` is the predicate, not ``isfinite(logprob)``: beyond ``amax``
    the grid clamp returns a finite ~1e-12 density (GW-31), and on THIS side a
    detected injection admitted with a spuriously tiny pdraw carries a ~1e12
    inverse weight straight into the Monte-Carlo sum for mu.
    """
    lnp, sup = chi_eff_prior_logprob_in_support(
        s._chieff[keep], s._m1src[keep], s._m2src[keep],
        amax=amax_1, amax_2=amax_2)
    return np.asarray(lnp, dtype=float), np.asarray(sup, dtype=bool)


def _campaign_chieff_chip_lnfactor(s, keep, strict):
    """Per-injection ln (chi_eff, chi_p) prior for one campaign (no floor).

    Resolves the campaign's DETECTED amax and validates the single
    uniform-isotropic assumption; see :class:`SpinBasisError`.
    """
    meta = s.spin_meta
    uniform = bool(meta.get("uniform_isotropic"))
    amax_detected = meta.get("amax_detected")
    if not uniform:
        msg = (f"{s.path}: injected spin draw is not a single uniform-isotropic "
               f"distribution; use spin_basis='component'.")
        if strict:
            raise SpinBasisError(msg)
        warnings.warn(msg + " (strict=False: proceeding with the detected amax "
                            "if one is available)")
    # `None` now means "undetectable" rather than a fabricated median-derived
    # number (GW-05), so this refuses where it used to warn-and-proceed.
    if (amax_detected is None
            or any(a is None or not np.isfinite(a) for a in amax_detected)):
        raise SpinBasisError(
            f"{s.path}: injected spin amax is undetectable (the injected spin "
            f"draw is not a single uniform-magnitude distribution); the joint "
            f"(chi_eff, chi_p) prior needs one amax, and there is no honest "
            f"value to use -- the median of a varying log-magnitude-density is "
            f"a summary of a mixture, not a ceiling. Use "
            f"spin_basis='component', which keeps the campaign's exact "
            f"per-injection spin draw.")
    amax_1, amax_2 = float(amax_detected[0]), float(amax_detected[1])
    # np.isclose, not exact float equality (GW-04): the detected amax comes out
    # of a numerical fit, so a single injected 0.998 ceiling is recovered as
    # 0.9980000000000001 vs 0.9979999999999999 and `!=` fired on every real
    # file -- burying the warning that matters, which is a genuine NSBH prior
    # with a restricted secondary (amax_2 ~ 0.05).
    if not np.isclose(amax_1, amax_2, rtol=1e-9, atol=1e-12):
        warnings.warn(
            f"{s.path}: injected spin_amax_1={amax_1} != spin_amax_2={amax_2}; "
            f"the joint (chi_eff, chi_p) prior assumes a single amax and uses "
            f"amax_1={amax_1} (mirroring the PE builder convention).")
    lnp, sup = chi_eff_chi_p_prior_logprob_in_support(
        s._chieff[keep], s._chi_p[keep], s._m1src[keep], s._m2src[keep],
        amax=amax_1)
    # No -50 floor (GW-03).  A DETECTED injection whose assumed draw density is
    # zero is not a small number, it is a contradiction: the injection really was
    # drawn and really was detected, so a density of zero means the assumed
    # prior does not cover the campaign.  The caller refuses rather than writing
    # it, because the consumer would silently exclude that injection from the
    # selection integral and bias mu.
    return (np.asarray(lnp, dtype=float), np.asarray(sup, dtype=bool),
            amax_1, amax_2)


def _parse_reference_amax(spin_basis, spin_reference_amax):
    """The declared reference ceiling for a reference basis, validated.

    Required, never defaulted.  ``a_ref`` is not a property of the campaign that
    could be resolved from the file (that is what ``amax="auto"`` does for the
    substituting swap); it is a statement about the reference prior the
    CONSUMER's PE side divides out, and the only place that is known is the call
    site.  A default here would be a fabricated ceiling inside a density that
    depends on it -- GW-31, one basis over.
    """
    if spin_basis not in _REFERENCE_BASES:
        if spin_reference_amax is not None:
            raise ValueError(
                f"spin_reference_amax={spin_reference_amax!r} was passed with "
                f"spin_basis={spin_basis!r}, which has no reference prior. It "
                f"applies only to {list(_REFERENCE_BASES)}; the substituting "
                f"chi_eff swap takes its ceiling from `amax=` instead.")
        return None
    if spin_reference_amax is None:
        raise ValueError(
            f"spin_basis={spin_basis!r} needs an explicit "
            f"spin_reference_amax=: the reference spin prior this export "
            f"reweights TO is a declaration about the PE side it will be "
            f"paired with (typically the GWTC sampling-prior ceiling 0.99), "
            f"not a property of the injection campaign, so there is no honest "
            f"default to resolve it from.")
    a_ref = float(spin_reference_amax)
    if not np.isfinite(a_ref) or not (0.0 < a_ref <= 1.0):
        raise ValueError(
            f"spin_reference_amax={spin_reference_amax!r}: the reference spin "
            f"magnitude ceiling must be finite and in (0, 1].")
    return a_ref


def _check_reference_coverage(set_list, a_ref, *, strict):
    """Refuse/warn when a campaign's magnitudes do not COVER the reference.

    The reweighting divides the reference density by the campaign's own, so it
    can only redistribute weight where the campaign actually drew.  If a
    campaign's magnitude prior stops at ``amax_k < a_ref`` then the region
    ``a in (amax_k, a_ref]`` -- which carries reference probability
    ``1 - (amax_k/a_ref)^2`` per body -- has no draws to carry it, and every
    detected system in it is missing from the sum.  ``alpha`` is then biased
    LOW, the same direction and for the same reason as a dropped detected
    injection (GW-03).

    Two evidences, deliberately not conflated.  A campaign with a DETECTED
    (uniform-magnitude) ceiling proves the hole, so it raises under ``strict``.
    A campaign without one leaves only the largest magnitude actually drawn,
    which is a sample maximum and not a support bound -- that warns.

    Returns ``(covered_flags, evidence_per_campaign)``.
    """
    covered, evidence, proven_bad, unproven_bad = [], [], [], []
    for s in set_list:
        meta = s.spin_meta or {}
        detected = meta.get("amax_detected")
        has_det = (detected is not None
                   and all(a is not None and np.isfinite(a)
                           for a in detected[:2]))
        if has_det:
            bound = float(min(detected[0], detected[1]))
            source = "detected"
        else:
            mags = [np.asarray(a, dtype=float) for a in (s._a1, s._a2)
                    if a is not None]
            bound = float(min(m.max() for m in mags)) if mags else float("nan")
            source = "drawn_max"
        ok = bool(np.isfinite(bound) and bound >= a_ref)
        covered.append(ok)
        evidence.append({"path": str(s.path), "bound": bound,
                         "bound_source": source, "covers_a_ref": ok})
        if not ok:
            (proven_bad if source == "detected" else unproven_bad).append(
                evidence[-1])

    def _detail(rows):
        return "; ".join(f"{r['path']} (magnitudes reach {r['bound']:.6g}, "
                         f"from {r['bound_source']})" for r in rows)

    if proven_bad:
        msg = (
            f"spin_reference_amax={a_ref} exceeds the injected spin magnitude "
            f"ceiling of {len(proven_bad)} campaign(s): {_detail(proven_bad)}. "
            f"The reference prior puts probability where those campaigns drew "
            f"no injections at all, so the reweighted selection integral is "
            f"missing that region entirely and alpha is biased LOW -- which "
            f"raises the likelihood in the direction that looks like a better "
            f"fit. Lower spin_reference_amax to the campaign ceiling (and use "
            f"the SAME value on the PE side), or use spin_basis='component', "
            f"which needs no reference at all. Pass strict=False to export "
            f"anyway; the file records spin_reference_coverage_per_campaign.")
        if strict:
            raise BlockCampaignMismatch(msg)
        warnings.warn(msg)
    if unproven_bad:
        warnings.warn(
            f"spin_reference_amax={a_ref} is above the largest spin magnitude "
            f"DRAWN by {len(unproven_bad)} campaign(s): "
            f"{_detail(unproven_bad)}. These campaigns carry no detectable "
            f"magnitude ceiling, so this is a sample maximum, not a support "
            f"bound -- it is evidence of a coverage hole, not proof of one. "
            f"Recorded in spin_reference_coverage_per_campaign.")
    return covered, evidence


def _campaign_reference_lnfactor(s, keep, a_ref):
    """``(ln p_iso(chieff|q,a_ref), in_reference_support)`` for one campaign.

    The reference conditional's numerator only; the constant ``1/p_ref =
    4 a_ref^2`` is applied once, after concatenation, so the arithmetic is one
    multiplication on the finished component pdraw rather than a per-campaign
    log sum (which would move the result by an ulp against the component export
    it is built from).

    In-support means BOTH bodies inside the reference magnitude ceiling AND
    ``chi_eff`` inside the reference prior's own support predicate -- the
    predicate, not ``isfinite``, for the GW-31 reason: past ``amax`` the grid
    clamp returns a finite ~1e-12 density.
    """
    if s._a1 is None or s._a2 is None:
        raise SpinBasisError(
            f"{s.path}: spin_basis='chieff_reference' needs the injected spin "
            f"magnitudes a1/a2 to evaluate the reference support "
            f"(a_i <= a_ref), and this file carries none "
            f"(spin_format={s.spin_meta.get('spin_format')!r}).")
    lnp, chi_sup = chi_eff_prior_logprob_in_support(
        s._chieff[keep], s._m1src[keep], s._m2src[keep], amax=a_ref)
    in_ref = (np.asarray(chi_sup, dtype=bool)
              & (np.asarray(s._a1, dtype=float)[keep] <= a_ref)
              & (np.asarray(s._a2, dtype=float)[keep] <= a_ref))
    return np.asarray(lnp, dtype=float), np.asarray(in_ref, dtype=bool)


class BlockCampaignMismatch(SpinBasisError):
    """A campaign's injected draw contradicts the requested basis's assumption.

    Distinct from a bare :class:`SpinBasisError` (which reports an *unavailable*
    quantity) because this one is a physics mismatch: the density gwcat would
    write is simply not the density the campaign was drawn from.
    """


def _check_chieff_swap_valid(set_list, *, strict, violations):
    """Refuse the chi_eff swap on a campaign that is not uniform-isotropic.

    The swap divides out the injected spin prior and multiplies in the analytic
    ``p(chi_eff)`` for uniform magnitudes and isotropic orientations.  If the
    campaign did not draw its spins that way, the exported ``pdraw`` is the wrong
    density by an O(1), chi_eff-DEPENDENT factor -- the textbook Essick &
    Fishbach p_draw mismatch, which biases the chi_eff population posterior and
    leaks into masses, rate and H0 through the selection integral.

    Unlike the constant amax offset (GW-04), this one does NOT cancel: it varies
    across injections, so it moves posteriors, not just log Z.

    The evidence is already in hand -- ``s.spin_meta['uniform_isotropic']`` --
    and ``chieff_chip`` has checked it since PR5.  ``chieff`` never did, which is
    how the shipped ``selection_o3o4ab_allsky.h5`` came to apply it to an O4ab
    campaign measured at ``isotropy_dev = 0.6419``.
    """
    bad, unverifiable, joint_unverifiable = [], [], []
    for s in set_list:
        meta = s.spin_meta or {}
        if meta.get("uniform_isotropic"):
            continue
        checks = meta.get("checks") or {}
        record = {
            "path": str(s.path),
            "spin_format": meta.get("spin_format"),
            "uniform_isotropic": bool(meta.get("uniform_isotropic")),
            "magnitude_uniform": _jsonable(checks.get("magnitude_uniform")),
            "isotropy_dev": _jsonable(checks.get("isotropy_dev")),
            "max_spin_uniform": _jsonable(checks.get("max_spin_uniform")),
        }
        # "False" carries two very different meanings and they must not be
        # conflated: a check that RAN AND FAILED is a measured contradiction; a
        # check that never ran (the file carries no spin draw densities at all)
        # is simply unknown.  Refusing the second would break every legacy
        # spin-less campaign, for which the chi_eff swap is the only option
        # available -- there is no component density to fall back to.
        ran = any(checks.get(k) is not None for k in
                  ("magnitude_uniform", "max_spin_uniform", "isotropy_dev"))
        record["verified"] = bool(ran)
        if not ran and record["spin_format"] in _JOINT_SPIN_FORMATS:
            # GW-39.  A JOINT (masses, redshift, spins) draw density is not the
            # spin-less legacy case above: the campaign DID draw spins from a
            # distribution of its own, one that a joint density does not let
            # the uniform/isotropic checks inspect.  The GWTC-5.0 cumulative
            # mixture is the live case, and its spins are measured
            # non-isotropic (O1/O2 mean cos tilt 0.245, O4 0.240), which is
            # the 91f1b924 defect class -- so "unverifiable" is refused here,
            # not waved through.
            record["reason"] = ("joint spin draw density: uniform-magnitude/"
                                "isotropic assumption cannot be verified")
            joint_unverifiable.append(record)
            continue
        (bad if ran else unverifiable).append(record)

    if joint_unverifiable:
        msg = (
            f"spin_basis='chieff' replaces each campaign's real spin-draw "
            f"density with the analytic uniform-magnitude/isotropic chi_eff "
            f"marginal, but {len(joint_unverifiable)} campaign(s) ship one "
            f"JOINT (masses, redshift, spins) draw density, so whether they "
            f"drew spins that way cannot be verified: "
            + "; ".join(f"{b['path']} (spin_format={b['spin_format']!r})"
                        for b in joint_unverifiable)
            + ". If they did not, the exported pdraw is the wrong density by "
            f"an O(1), chi_eff-dependent factor that does not cancel (the "
            f"isotropic-substitution defect). Use "
            f"spin_basis='chieff_reference' (with spin_reference_amax=), which "
            f"keeps the campaign's exact per-injection spin density and "
            f"reweights it, or spin_basis='component'. Pass strict=False to "
            f"export anyway; the file then records "
            f"spin_basis_assumption_violations.")
        if strict:
            raise BlockCampaignMismatch(msg)
        warnings.warn(msg)
        violations.extend(joint_unverifiable)

    if unverifiable:
        warnings.warn(
            f"spin_basis='chieff': {len(unverifiable)} campaign(s) carry no "
            f"injected spin draw densities, so the uniform-magnitude/isotropic "
            f"assumption behind the chi_eff swap could not be CHECKED: "
            + "; ".join(str(b["path"]) for b in unverifiable)
            + ". Proceeding (the swap is the only basis such a file supports), "
            f"but the assumption is unverified rather than confirmed; recorded "
            f"in spin_basis_assumption_unverified.")
        violations.extend(unverifiable)
    if not bad:
        return
    detail = "; ".join(
        f"{b['path']} (spin_format={b['spin_format']!r}, "
        f"magnitude_uniform={b['magnitude_uniform']}, "
        f"isotropy_dev={b['isotropy_dev']})" for b in bad)
    msg = (
        f"spin_basis='chieff' replaces each campaign's real spin-draw density "
        f"with the analytic uniform-magnitude/isotropic chi_eff marginal, but "
        f"{len(bad)} campaign(s) did NOT draw spins that way: {detail}. The "
        f"exported pdraw would be the wrong density by an O(1), "
        f"chi_eff-dependent factor -- unlike a constant offset this does not "
        f"cancel, so it biases the chi_eff population posterior and leaks into "
        f"masses, rate and H0. Use spin_basis='component', which is EXACT for "
        f"any campaign because it keeps the injected per-injection spin density "
        f"instead of assuming one. Pass strict=False to export anyway; the file "
        f"then records spin_basis_assumption_violations so it is at least "
        f"self-describing.")
    if strict:
        raise BlockCampaignMismatch(msg)
    warnings.warn(msg)
    violations.extend(bad)


def _jsonable(v):
    """Coerce numpy scalars/tuples in a check value to plain Python."""
    if v is None:
        return None
    if isinstance(v, (tuple, list)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.floating, float, np.integer, int)):
        return float(v)
    return str(v)


def build_selection_product(sets, *, spin_basis=DEFAULT_PARAMETER_SPACE,
                            far_threshold=1.0,
                            source_class=None, amax=AMAX_AUTO,
                            amax_fallback=0.99, spin_reference_amax=None,
                            out_of_reference_pdraw=OUT_OF_REFERENCE_PDRAW,
                            snr_threshold=None, z_max=None, strict=True,
                            detection_policy="far",
                            acknowledge_semianalytic_excluded=False,
                            sky_marginal=False, mixture_anchors=None):
    """Build a selection :class:`ExportProduct` from one or more SelectionSets.

    Parameters
    ----------
    sets : SelectionSet, CombinedSelectionSet, or list of SelectionSet
        One campaign or several; handled uniformly (see the module docstring's
        parity contract).
    spin_basis : {"component", "chieff", "chieff_chip", "chieff_reference"}
        Per-injection spin factor / extra columns (see the module docstring).
        Defaults to ``"component"`` -- the exact component-basis draw density,
        which is well-defined for every file.  ``"chieff_reference"`` writes the
        ``(m1det, q, dL, chi_eff)`` density against a DECLARED reference spin
        prior by reweighting that exact component density; it needs
        ``spin_reference_amax`` and is valid for any campaign.
    far_threshold : float, default 1.0
        FAR detection threshold in yr^-1, applied per campaign.
    source_class : str, iterable, or None
        Optional source-class subset (see
        :meth:`gwcat.selection.SelectionSet.source_class_mask`).  Subsetting,
        not reweighting: ``ndraw`` is unchanged.
    amax : float or ``"auto"``, default ``"auto"``
        chi_eff-prior spin ceiling for the ``"chieff"`` basis.  ``"auto"`` uses
        each campaign's OWN detected injected ceiling (GW-31) -- the density the
        swap replaces is that campaign's, so the ceiling must be that campaign's
        too.  A number forces one ceiling on every campaign (the legacy
        behaviour, and what reproduces ``to_darksirens`` bit-for-bit).  Ignored
        by ``"component"``; the ``"chieff_chip"`` basis always uses the detected
        amax.
    amax_fallback : float, default 0.99
        Ceiling assumed for a ``"chieff"``-basis campaign whose injected spin
        draw carries no detectable ceiling (warned about, and recorded in
        ``chi_eff_amax_source_per_campaign``).
    spin_reference_amax : float
        REQUIRED by (and only accepted by) ``spin_basis="chieff_reference"``:
        the magnitude ceiling ``a_ref`` of the isotropic uniform-magnitude
        reference spin prior the exported ``pdraw`` is expressed against.  It
        must be the ceiling the paired PE export divides out (0.99 for the GWTC
        sampling priors) -- unlike the substituting swap's ``amax``, where the
        two sides legitimately differ, here they are the SAME object and the
        validator refuses a pair whose ceilings disagree.  No default: see
        :func:`_parse_reference_amax`.
    out_of_reference_pdraw : float, default 1e300
        ``pdraw`` written for a row the reference assigns zero density to
        (``a_i > a_ref``), so its importance weight underflows to exactly zero
        while the file keeps a finite positive ``pdraw`` everywhere.  Reference
        bases only.
    z_max : float, optional
        Redshift truncation matching the PE export's per-sample ``z_max``
        (GW-37).  Injections above it are SUBSET out (``ndraw`` untouched, so
        the Essick fractions are unchanged), which is what makes mu integrate
        over the same redshift range the truncated posteriors cover.  Leaving
        it ``None`` against a truncated PE export means the two sides cover
        different z ranges; the validator cross-checks the pair.
    snr_threshold : float, optional
        When set, detection becomes ``far-detected OR (snr > snr_threshold)``
        using the cumulative-mixture ``semianalytic_observed_phase_maximized_snr_net``
        column.  Default ``None`` reproduces the FAR-only cut exactly.
    strict : bool, default True
        For ``"chieff_chip"``: raise :class:`SpinBasisError` when a campaign is
        not verified single-uniform-isotropic.  ``False`` warns and proceeds
        when a detected amax is nonetheless available (an undetectable amax
        always raises).
    detection_policy : {"far", "lvk-cumulative"}, default "far"
        Recorded in the attrs.  ``"lvk-cumulative"`` (GW-39) requires a
        cumulative-mixture campaign and ``snr_threshold``, and states that the
        release's per-run rule is intended: semianalytic SNR on the O1/O2 rows,
        each run's own search FARs on the O3/O4 rows.  On a mixture the
        run-aware rule is applied whenever ``snr_threshold`` is given.
    acknowledge_semianalytic_excluded : bool, default False
        A FAR-only cut on a cumulative mixture with semianalytic O1/O2 rows
        raises :class:`MixtureDetectionError` (it detects none of them while
        their draws and exposure stay in the normalisation).  ``True`` allows
        it, for a deliberately O3+O4-only analysis.
    sky_marginal : bool, default False
        Omit the ``ra``/``dec`` columns and record ``sky_marginalized=True``:
        pdraw carries no sky density and the population is isotropic, so the
        sky is marginalised rather than NaN-filled.  Required in effect for
        the cumulative mixtures, which ship no sky position at all.
    mixture_anchors : dict, optional
        The O3/O4 component ``{"O3": {"T_s", "N", "source"}, "O4": {...}}``
        used to derive a mixture's per-run N_k/T_k; defaults to the bundled
        :data:`gwcat.observing_runs.MIXTURE_RELEASE_ANCHORS` entry.

    Returns
    -------
    gwcat.export.product.ExportProduct
        ``kind="selection"``; ``format_version`` is the writer's, not set here.
    """
    if spin_basis not in _KNOWN_SPIN_BASES:
        raise ValueError(
            f"unknown spin_basis={spin_basis!r}; known bases are "
            f"{list(_KNOWN_SPIN_BASES)}.")
    if detection_policy not in DETECTION_POLICIES:
        raise ValueError(
            f"unknown detection_policy={detection_policy!r}; known policies "
            f"are {list(DETECTION_POLICIES)}.")
    # `None` = each campaign's own detected ceiling; a float = one forced ceiling.
    forced_amax = parse_amax_option(amax, what="amax")
    # `None` for every non-reference basis; a validated float for a reference
    # basis, which never defaults it.
    a_ref = _parse_reference_amax(spin_basis, spin_reference_amax)

    if isinstance(sets, CombinedSelectionSet):
        set_list = list(sets._sets)
    elif isinstance(sets, SelectionSet):
        set_list = [sets]
    else:
        set_list = list(sets)
    if not set_list:
        raise ValueError("build_selection_product needs at least one SelectionSet")

    for s in set_list:
        s._load()

    # A cumulative mixture already holds every run's exposure (GW-39).
    _refuse_overlapping_campaigns(set_list)
    is_mixture = [s.campaign_kind == CAMPAIGN_CUMULATIVE_MIXTURE
                  for s in set_list]
    if detection_policy == "lvk-cumulative":
        if snr_threshold is None:
            raise ValueError(
                "detection_policy='lvk-cumulative' needs snr_threshold (the "
                "semianalytic O1/O2 SNR threshold; 10 in the LVK analyses): "
                "without it no O1/O2 row can be detected while their exposure "
                "stays in the normalisation.")
        if not any(is_mixture):
            raise ValueError(
                "detection_policy='lvk-cumulative' applies the per-run rule of "
                "a cumulative multi-run mixture, and none of the campaigns is "
                "one: " + ", ".join(str(s.path) for s in set_list))

    # Cosmology consistency check (mirror CombinedSelectionSet.to_darksirens).
    H0s = [s.H0 for s in set_list]
    Om0s = [s.Om0 for s in set_list]
    if max(H0s) - min(H0s) > 1.0 or max(Om0s) - min(Om0s) > 0.05:
        warnings.warn(
            f"Cosmology mismatch across campaigns: H0={H0s}, Om0={Om0s}. "
            f"Results may be inconsistent.")

    # The component draw density is the STARTING POINT of both the component
    # basis and every reference basis (which reweights it), so both require it.
    if spin_basis == "component" or spin_basis in _REFERENCE_BASES:
        for s in set_list:
            if not s.component_spin_available:
                raise SpinBasisError(
                    f"{s.path}: component-basis spin draw density unavailable "
                    f"(spin_format={s.spin_meta.get('spin_format')!r}); this "
                    f"file lacks the per-spin draw information needed for "
                    f"spin_basis={spin_basis!r}. Use spin_basis='chieff'.")

    #: Per-campaign reference-support coverage (reference bases only).
    reference_coverage, reference_coverage_evidence = [], []
    #: The same check run by run on each mixture campaign (GW-39).
    reference_coverage_runs = None
    if a_ref is not None:
        reference_coverage, reference_coverage_evidence = (
            _check_reference_coverage(set_list, a_ref, strict=strict))
        for s, mix in zip(set_list, is_mixture):
            if mix:
                reference_coverage_runs = _reference_coverage_per_run(s, a_ref)

    ndraw_per = [int(s._ndraw) for s in set_list]
    ndraw_total = sum(ndraw_per)

    # Per-injection column names (the legacy 9 + the frac-scaled pdraw).
    base_cols = ["m1det", "m2det", "dL", "chieff", "ra", "dec",
                 "m1src", "m2src", "z", "pdraw"]
    attr_of = {"m1det": "_m1det", "m2det": "_m2det", "dL": "_dL",
               "chieff": "_chieff", "ra": "_ra", "dec": "_dec",
               "m1src": "_m1src", "m2src": "_m2src", "z": "_z"}
    parts = {k: [] for k in base_cols}
    extra_parts = {k: [] for k in ("a1", "a2", "cost1", "cost2", "chip")}
    lnfactor_parts = []          # per-campaign ln spin factor (every basis)
    support_parts = []           # per-campaign in-support mask (projections)
    ref_lnp_parts = []           # per-campaign ln p_iso(chieff|q,a_ref)
    ref_support_parts = []       # per-campaign in-REFERENCE-support mask
    extras_available = True      # a1/a2/cost1/cost2 present for every campaign
    chip_available = True        # chi_p present for every campaign

    n_det_total = 0
    n_before_total = 0
    n_after_total = 0
    far_columns_union = []
    campaign_info = []
    chieff_chip_amax = []        # per (non-empty) campaign detected amax used
    # Per (non-empty) campaign chi_eff ceilings actually used, and how each was
    # resolved ("detected" / "caller" / "fallback").
    chieff_amax_pairs, chieff_amax_sources = [], []

    mixture_attrs = None
    for k, s in enumerate(set_list):
        keep, det, sc_mask = _detect_keep(
            s, far_threshold, source_class, snr_threshold, z_max=z_max,
            acknowledge_semianalytic_excluded=acknowledge_semianalytic_excluded)
        if is_mixture[k]:
            mixture_attrs = _mixture_run_attrs(
                s, keep, det, far_threshold, snr_threshold, mixture_anchors)
        n_before_total += int(det.size)
        n_after_total += int(sc_mask.sum())
        for c in s._far_columns:
            if c not in far_columns_union:
                far_columns_union.append(c)
        n_det_k = int(keep.sum())
        if n_det_k == 0:
            warnings.warn(
                f"Campaign {s.path}: no detected injections at "
                f"FAR < {far_threshold}"
                + ("" if source_class is None
                   else f" in source class {source_class!r}"))
            continue

        frac = ndraw_per[k] / ndraw_total
        parts["pdraw"].append(s._pdraw[keep] * frac)
        for name, attr in attr_of.items():
            parts[name].append(getattr(s, attr)[keep])

        # Extra spin columns (free from the additive accessors).
        a1, a2 = s._a1, s._a2
        cost1, cost2, chi_p = s._cost1, s._cost2, s._chi_p
        if any(v is None for v in (a1, a2, cost1, cost2)):
            extras_available = False
        else:
            extra_parts["a1"].append(np.asarray(a1)[keep])
            extra_parts["a2"].append(np.asarray(a2)[keep])
            extra_parts["cost1"].append(np.asarray(cost1)[keep])
            extra_parts["cost2"].append(np.asarray(cost2)[keep])
        if chi_p is None:
            chip_available = False
        else:
            extra_parts["chip"].append(np.asarray(chi_p)[keep])

        # Per-basis per-injection ln factor.  The chieff swap is applied HERE,
        # per campaign, because its ceiling is per campaign (GW-31); with one
        # shared ceiling this is elementwise identical to the old single
        # post-concat call, so the v1 parity survives.
        if spin_basis == "component":
            lnfactor_parts.append(np.asarray(s._ln_spin_component)[keep])
            support_parts.append(np.ones(n_det_k, dtype=bool))
        elif spin_basis == "chieff_reference":
            # The component factor, byte-for-byte what the component basis
            # applies -- the reference change is one multiplication on the
            # FINISHED component pdraw, below.  `support_parts` stays all-True:
            # the campaign's own density is positive on every drawn injection,
            # and the reference's zeros are a separate, non-fatal mask.
            lnfactor_parts.append(np.asarray(s._ln_spin_component)[keep])
            support_parts.append(np.ones(n_det_k, dtype=bool))
            lnp_ref, in_ref = _campaign_reference_lnfactor(s, keep, a_ref)
            ref_lnp_parts.append(lnp_ref)
            ref_support_parts.append(in_ref)
        elif spin_basis == "chieff_chip":
            lnfac, sup, amax_used, _ = _campaign_chieff_chip_lnfactor(
                s, keep, strict)
            lnfactor_parts.append(lnfac)
            support_parts.append(sup)
            chieff_chip_amax.append(amax_used)
        else:  # chieff
            src, a1, a2 = _campaign_chieff_amax(s, forced_amax, amax_fallback)
            lnfac, sup = _campaign_chieff_lnfactor(s, keep, a1, a2)
            lnfactor_parts.append(lnfac)
            support_parts.append(sup)
            chieff_amax_pairs.append((a1, a2))
            chieff_amax_sources.append(src)

        n_det_total += n_det_k
        campaign_info.append(
            f"{s.path}: N={ndraw_per[k]}, T={s._T_yr:.2f}yr, "
            f"n_det={n_det_k}, frac={frac:.4f}")

    _refuse_mixed_cosmology(set_list)

    if n_det_total == 0:
        raise RuntimeError(
            f"No detected injections across {len(set_list)} campaign(s) at "
            f"FAR < {far_threshold}"
            + ("" if source_class is None
               else f" in source class {source_class!r}"))

    data = {k: np.concatenate(v) for k, v in parts.items()}
    #: Campaigns whose injected draw contradicts the basis (GW-06, strict=False).
    swap_violations = []

    # ── Spin-basis-specific step: apply the per-injection pdraw factor ──────
    if spin_basis == "chieff":
        # ── Uniform-isotropic gate (GW-06) ─────────────────────────────────
        # The swap REPLACES each campaign's real spin-draw density with the
        # analytic uniform-magnitude/isotropic chi_eff marginal.  That is only
        # valid if the campaign actually drew its spins that way.  Until GW-06
        # the check existed for chieff_chip and not for chieff -- i.e. the basis
        # that ships was the one with no gate.
        _check_chieff_swap_valid(set_list, strict=strict,
                                 violations=swap_violations)
    ln_factor = np.concatenate(lnfactor_parts)
    in_support = np.concatenate(support_parts)
    with np.errstate(over="ignore"):
        data["pdraw"] = data["pdraw"] * np.exp(ln_factor)

    # ── Out-of-support DETECTED injections are fatal (GW-03) ────────────────
    # A PE sample with zero prior density can be dropped: the posterior simply
    # has no support there.  A detected injection cannot -- it was drawn from the
    # real campaign and it triggered, so the Monte-Carlo sum over draws must
    # include it.  If the ASSUMED prior gives it zero density the assumption is
    # wrong, and the consumer's `pdraw > 0` guard would quietly exclude it and
    # bias mu low.
    # Out of support is the prior's OWN predicate, not isfinite(ln_factor): the
    # 1-D chi_eff grid returns a finite ~1e-12 density beyond amax, so the
    # finiteness test let excluded injections through with a pdraw ~1e12 too
    # small -- i.e. an inverse weight ~1e12 too large in the sum for mu (GW-31).
    unsupported = ~(in_support & np.isfinite(ln_factor))
    n_unsupported = int(np.sum(unsupported))
    if n_unsupported:
        frac = n_unsupported / ln_factor.size
        raise SpinBasisError(
            f"spin_basis={spin_basis!r}: {n_unsupported} of {ln_factor.size} "
            f"detected injections ({100 * frac:.3f}%) fall outside the assumed "
            f"spin prior's support, so their pdraw would be exactly zero. "
            f"These injections were drawn and detected, so they belong in the "
            f"Monte-Carlo sum: mu = (1/Ndraw) * sum(p_pop/p_draw), where Ndraw "
            f"is the campaign's TOTAL generated count read from the file attr "
            f"and is independent of the detection mask. Dropping a detected "
            f"injection therefore removes a positive term from the numerator "
            f"while the denominator stays fixed -- mu is biased LOW, "
            f"-N_obs*log(mu) goes UP, and the likelihood is inflated in the "
            f"direction that looks like a better fit. Worse, the consumer "
            f"excludes it with the same `prior_wt > 0` mask it uses for padded "
            f"sentinel rows, so the two are indistinguishable downstream and "
            f"neither is counted. The assumed amax does not cover this "
            f"campaign: fix the amax (GW-04) or use spin_basis='component', "
            f"which is exact for any campaign because the assumed prior "
            f"cancels identically.")

    # ── Reference bases: the one explicit change of spin reference ──────────
    # `data["pdraw"]` is now the component-basis pdraw, bit-for-bit what
    # spin_basis="component" would have written.  The reference spin prior is
    # CONSTANT inside its support (isotropic tilts, magnitudes uniform on
    # [0, a_ref] => p_ref = 1/(4 a_ref^2)), so the whole change of reference is
    # the single factor p_iso(chi_eff | q, a_ref) / p_ref applied to it -- one
    # multiplication, in this association, so a reference export and the
    # component export it is built from differ by exactly that factor and by
    # nothing else the floating-point arithmetic re-orders.
    n_out_of_reference = 0
    ln_ref_norm = float("nan")
    if a_ref is not None:
        ln_ref_norm = np.log(4.0 * a_ref ** 2)
        ln_p_ref_chi = np.concatenate(ref_lnp_parts)
        in_reference = np.concatenate(ref_support_parts)
        with np.errstate(over="ignore", invalid="ignore"):
            data["pdraw"] = data["pdraw"] * np.exp(ln_p_ref_chi + ln_ref_norm)
        bad = ~np.isfinite(data["pdraw"]) | (data["pdraw"] <= 0.0)
        n_bad = int(np.sum(bad & in_reference))
        if n_bad:
            raise SpinBasisError(
                f"spin_basis={spin_basis!r}: {n_bad} of {bad.size} detected "
                f"injections are INSIDE the reference support "
                f"(a_ref={a_ref}) yet came out with a non-finite or "
                f"non-positive pdraw. That is an arithmetic failure of the "
                f"reweighting, not a support statement, and it must not be "
                f"written as a zero-weight row -- those rows are physically "
                f"in the integral.")
        excluded = ~in_reference | bad
        data["pdraw"] = np.where(excluded, float(out_of_reference_pdraw),
                                 data["pdraw"])
        n_out_of_reference = int(np.sum(excluded))

    # ── Output columns: legacy 10 + (a1,a2,cost1,cost2,chip when available) ──
    columns = {
        "m1det": data["m1det"], "m2det": data["m2det"], "dL": data["dL"],
        # GW-34 made `q` the published mass density COORDINATE registry-wide and
        # the PE builder emits it, but this builder never did -- so a 2.1
        # selection file declared `q` in its own fit_columns and shipped no such
        # dataset (GW-37).  A contract-driven consumer raised KeyError on it; one
        # that fell back to the present columns built the injection-side density
        # over (m1det, m2det) against a PE side in (m1det, q), reintroducing the
        # per-injection m1det mismatch GW-34 removed.  pdraw already IS a density
        # in (m1det, q, dL) -- only the self-description was wrong.
        "q": data["m2det"] / data["m1det"],
        "chieff": data["chieff"], "ra": data["ra"], "dec": data["dec"],
        "m1src": data["m1src"], "m2src": data["m2src"],
        "redshift": data["z"], "pdraw": data["pdraw"],
    }
    if extras_available:
        for name in _EXTRA_COLUMNS:
            columns[name] = np.concatenate(extra_parts[name])
    if chip_available:
        columns["chip"] = np.concatenate(extra_parts["chip"])
    if sky_marginal:
        # GW-39: the sky is marginalised, not missing -- no sky density is in
        # pdraw and the population models are isotropic -- so the columns are
        # omitted rather than written as NaN a consumer would have to exempt.
        del columns["ra"], columns["dec"]

    # ── Provenance attrs ────────────────────────────────────────────────────
    _str = h5py.string_dtype()
    attrs = dict(_selection_provenance_dict(
        source_class=source_class,
        nsbh_mass_threshold=set_list[0]._nsbh_mass_threshold,
        n_before=n_before_total, n_after=n_after_total,
        far_columns=far_columns_union, far_threshold=far_threshold))
    # pdraw_state is basis-specific (overrides the v1 default the dict carries).
    attrs["pdraw_state"] = PDRAW_STATE_BY_BASIS[spin_basis]

    attrs.update({
        "spin_basis": spin_basis,
        "ndraw": int(ndraw_total),
        "T_obs_yr": float(sum(s._T_yr for s in set_list)),
        "far_threshold": float(far_threshold),
        "n_detected": int(n_det_total),
        "cosmology_H0": float(set_list[0].H0),
        "cosmology_Om0": float(set_list[0].Om0),
        "cosmology_override_used": bool(
            any(getattr(s, "_cosmology_override", False) for s in set_list)),
        # Per-campaign, because a combined export legitimately mixes campaigns
        # whose pdraw used DIFFERENT cosmologies -- or none at all.  The scalar
        # above is the first campaign's reference value, kept for backward
        # compatibility; these are the authoritative record (GW-09).
        "cosmology_source_per_campaign": np.array(
            [str(getattr(s, "_cosmology_source", None)) for s in set_list],
            dtype=_str),
        "cosmology_H0_per_campaign": np.array(
            [_cosmo_or_nan(s, "_cosmology_used_H0") for s in set_list],
            dtype=float),
        "cosmology_Om0_per_campaign": np.array(
            [_cosmo_or_nan(s, "_cosmology_used_Om0") for s in set_list],
            dtype=float),
        "cosmology_mixed_across_campaigns": bool(_cosmology_is_mixed(set_list)),
        "mass_jacobian_applied": True,
        "distance_prior_removed": False,
        "n_campaigns": len(set_list),
        "campaign_ndraws": np.array(ndraw_per, dtype=np.int64),
        # Per-campaign injected-spin provenance (campaign order).
        "injected_spin_format": np.array(
            [str(s.spin_meta.get("spin_format")) for s in set_list],
            dtype=_str),
        "injected_spin_amax_detected": np.array(
            [_amax_pair(s.spin_meta.get("amax_detected")) for s in set_list],
            dtype=float),
        "injected_spin_uniform_isotropic": np.array(
            [bool(s.spin_meta.get("uniform_isotropic")) for s in set_list],
            dtype=bool),
        "injected_spin_checks": json.dumps(
            [_json_checks(s.spin_meta.get("checks", {})) for s in set_list]),
        # False for campaigns whose rows carry no drawn sky position (the
        # semianalytic O1/O2 entries of the cumulative mixtures); their
        # exported ra/dec are NaN.
        "sky_position_available": _sky_availability(set_list),
        "component_columns_emitted": bool(extras_available),
        # ── Spin-removal amax provenance (GW-19) ─────────────────────────────
        # The cartesian/polar prior-REMOVAL step has to assume a ceiling before
        # the campaign's own is known.  Its amax dependence is the constant
        # -2*ln(amax), so a wrong value leaves a constant factor
        # (assumed/injected)^2 in pdraw -- which CANCELS EXACTLY for the
        # component basis (verified: perturbing it 0.99 -> 0.998 moves
        # component_pdraw by 0 to 1e-14) but not for a projection.  Recorded so
        # the constant is knowable rather than buried; NaN where the campaign
        # never took a removal branch, which is every real file so far
        # (o4_factored / endo3_factored read their spin-free density directly).
        "spin_removal_amax_assumed_per_campaign": np.array(
            [float(getattr(s, "_removal_amax", None) or np.nan)
             for s in set_list], dtype=float),
        # Also True for a reference basis: its starting point IS the
        # component density, and the reference factor does not involve the
        # removal ceiling, so the (assumed/injected)^2 constant cancels there
        # for exactly the same reason.
        "spin_removal_amax_cancels": bool(
            spin_basis == "component" or spin_basis in _REFERENCE_BASES),
    })

    # Basis-specific spin-prior contract attrs (mirror the v1 / PE naming).
    if spin_basis == "chieff":
        attrs["spin_prior_mode"] = "include"
        attrs["chi_eff_swap_applied"] = True
        attrs["chi_eff_prior_applied_to_pdraw"] = True
        # WHICH ceiling each campaign's swap used, and where it came from.  The
        # scalar is the envelope (the support bound of the written pdraw); it is
        # no longer simply the caller's argument, and it is NOT required to
        # match the PE file's -- the two divide out different priors (GW-31).
        pairs = np.asarray(chieff_amax_pairs, dtype=float).reshape(-1, 2)
        attrs["chi_eff_amax_per_campaign"] = pairs
        attrs["chi_eff_amax_source_per_campaign"] = np.array(
            chieff_amax_sources, dtype=_str)
        attrs["chi_eff_amax_mode"] = ("fixed" if forced_amax is not None
                                      else "per_campaign")
        attrs["chi_eff_amax"] = (float(pairs.max()) if pairs.size
                                 else float(amax_fallback))
        # GW-06: empty unless strict=False let a non-uniform-isotropic campaign
        # through, in which case the file says so about itself.
        attrs["spin_basis_assumption_violations"] = json.dumps(swap_violations)
        attrs["spin_basis_assumption_violated"] = bool(
            [v for v in swap_violations if v.get("verified")])
        attrs["spin_basis_assumption_unverified"] = bool(
            [v for v in swap_violations if not v.get("verified")])
    elif spin_basis == "chieff_reference":
        attrs["spin_prior_mode"] = "include"
        # Both True, and both mean what they have always meant: the 1-D chi_eff
        # prior IS a factor of the exported pdraw (unlike chieff_chip, where a
        # JOINT prior is and these are False).  WHICH chi_eff density, and how
        # it got there, is what the spin_reference_* attrs below say -- the
        # campaign's own spin draw was divided out by the reference rather than
        # discarded, which is the difference between this basis and the swap.
        attrs["chi_eff_swap_applied"] = True
        attrs["chi_eff_prior_applied_to_pdraw"] = True
        attrs["chi_eff_prior_source"] = "reweighted_from_component_draw"
        attrs["component_spin_draw_retained"] = False
        attrs["spin_basis_source"] = "component"
        attrs["spin_reference"] = (
            f"isotropic uniform-magnitude a_max={a_ref}")
        attrs["spin_reference_amax"] = float(a_ref)
        # The support bound of the written pdraw, in the same spelling every
        # other basis uses for it.
        attrs["chi_eff_amax"] = float(a_ref)
        attrs["chi_eff_amax_mode"] = "reference"
        attrs["spin_reference_formula"] = (
            "pdraw = pdraw_component * exp(chi_eff_prior_logprob_in_support("
            "chieff, m1src, m2src, amax=a_ref)) * 4 * a_ref**2; equivalently "
            "pdraw_component * p_iso(chieff|q,a_ref) / p_ref(a1,cost1,a2,cost2) "
            "with the reference spin prior p_ref = 1/(4 a_ref^2). Rows with "
            "a1 > a_ref or a2 > a_ref (zero reference density) carry weight "
            "exactly zero and are written with spin_reference_excluded_pdraw.")
        attrs["spin_reference_excluded_rows"] = int(n_out_of_reference)
        attrs["spin_reference_excluded_pdraw"] = float(out_of_reference_pdraw)
        attrs["spin_reference_ln_norm"] = float(ln_ref_norm)
        # Whether each campaign's injected magnitudes COVER [0, a_ref], and on
        # what evidence -- a detected (uniform-magnitude) ceiling proves it, a
        # sample maximum only suggests it.
        attrs["spin_reference_coverage_per_campaign"] = np.array(
            reference_coverage, dtype=bool)
        attrs["spin_reference_coverage_bound_per_campaign"] = np.array(
            [e["bound"] for e in reference_coverage_evidence], dtype=float)
        attrs["spin_reference_coverage_source_per_campaign"] = np.array(
            [e["bound_source"] for e in reference_coverage_evidence],
            dtype=_str)
        attrs["spin_reference_coverage_ok"] = bool(all(reference_coverage))
    elif spin_basis == "component":
        attrs["spin_prior_mode"] = "component"
        # Stated False, not omitted: darksirens' loader REQUIRES this attr, so
        # omission made the file fail to load -- the exact defect GW-21 fixed
        # for chi_eff_in_p_pe on the PE side. False is also the truth: the 1-D
        # chi_eff swap is not applied; the exact component draw is retained.
        attrs["chi_eff_swap_applied"] = False
        attrs["chi_eff_prior_applied_to_pdraw"] = False
        attrs["component_spin_draw_retained"] = True
    else:  # chieff_chip
        attrs["spin_prior_mode"] = "include"
        # False for the same reason as the component branch: the attr means
        # THE 1-D chi_eff swap specifically, and here the joint (chi_eff,
        # chi_p) swap is applied instead -- recorded on its own attr below.
        attrs["chi_eff_swap_applied"] = False
        attrs["chi_eff_chi_p_swap_applied"] = True
        attrs["chi_eff_chi_p_prior_applied_to_pdraw"] = True
        attrs["chi_eff_chi_p_amax_detected_per_campaign"] = np.array(
            chieff_chip_amax, dtype=float)
        # The support bound of the joint prior in force, i.e. the largest
        # detected ceiling -- not the caller's argument, which this basis has
        # never used for anything.
        attrs["chi_eff_amax"] = (float(np.max(chieff_chip_amax))
                                 if chieff_chip_amax else float(amax_fallback))

    # ── GW-39: campaign kind, detection policy, sky, per-run bookkeeping ────
    attrs["cumulative_mixture"] = bool(any(is_mixture))
    attrs["campaign_kind_per_campaign"] = np.array(
        [str(s.campaign_kind) for s in set_list], dtype=_str)
    attrs["detection_policy"] = str(detection_policy)
    attrs["acknowledge_semianalytic_excluded"] = bool(
        acknowledge_semianalytic_excluded)
    attrs["sky_marginalized"] = bool(sky_marginal)
    if mixture_attrs is not None:
        attrs.update(mixture_attrs)
    if reference_coverage_runs is not None:
        runs_c, cov_c, bound_c = reference_coverage_runs
        attrs["spin_reference_coverage_runs"] = np.array(runs_c, dtype=_str)
        attrs["spin_reference_coverage_per_run"] = np.array(cov_c, dtype=bool)
        attrs["spin_reference_coverage_bound_per_run"] = np.array(
            bound_c, dtype=float)
        attrs["spin_reference_coverage_ok"] = bool(
            attrs.get("spin_reference_coverage_ok", True) and all(cov_c))

    if snr_threshold is not None:
        attrs["significance_snr_column"] = _SNR_COLUMN
        attrs["significance_snr_threshold"] = float(snr_threshold)
        attrs["significance_type"] = "far_or_snr"

    # The redshift truncation this product was built under (GW-37).  NaN when
    # unused, in the same spelling the PE side stamps, so the validator can
    # compare them and a reader can see that mu covers the same z range the
    # truncated posteriors do.
    attrs["z_max"] = float("nan") if z_max is None else float(z_max)

    # ── Validation-summary feed (writer fills output_path + summary_context) ─
    from ..validation_summary import (value_counts, package_version)
    from ..source_class import classify_by_mass, normalize_source_class
    classes_det = classify_by_mass(data["m1src"], data["m2src"],
                                   set_list[0]._nsbh_mass_threshold)
    summary = {
        "kind": "selection_export",
        "package_version": package_version(),
        "schema_version": "gwcat-selection-2.0",
        "spin_basis": spin_basis,
        "n_campaigns": len(set_list),
        "campaign_paths": [s.path for s in set_list],
        "campaign_ndraws": list(ndraw_per),
        "n_injections_total": int(sum(s.n_injections for s in set_list)),
        "n_injections_before_filter": n_before_total,
        "n_injections_after_filter": n_after_total,
        "n_detected": int(n_det_total),
        "ndraw": int(ndraw_total),
        "T_obs_yr": float(sum(s._T_yr for s in set_list)),
        "far_threshold": float(far_threshold),
        "snr_threshold": (None if snr_threshold is None
                          else float(snr_threshold)),
        "significance_columns": list(far_columns_union),
        "significance_available": bool(far_columns_union),
        "p_astro_available": False,
        "source_class_filter": format_source_class_filter(source_class),
        CUT_ESTIMATOR_ATTR: selection_cut_estimator(source_class),
        "source_class_counts_detected": value_counts(
            [normalize_source_class(c) for c in classes_det]),
        "cosmology_H0": float(set_list[0].H0),
        "cosmology_Om0": float(set_list[0].Om0),
        "cosmology_override_used": bool(
            any(getattr(s, "_cosmology_override", False) for s in set_list)),
        "cosmology_source_per_campaign": [
            str(getattr(s, "_cosmology_source", None)) for s in set_list],
        "cosmology_H0_per_campaign": [
            _cosmo_or_nan(s, "_cosmology_used_H0") for s in set_list],
        "cosmology_Om0_per_campaign": [
            _cosmo_or_nan(s, "_cosmology_used_Om0") for s in set_list],
        "cosmology_mixed_across_campaigns": bool(_cosmology_is_mixed(set_list)),
        "injected_spin_format": [str(s.spin_meta.get("spin_format"))
                                 for s in set_list],
        "component_columns_emitted": bool(extras_available),
    }
    if a_ref is not None:
        summary["spin_reference_amax"] = float(a_ref)
        summary["spin_reference_excluded_rows"] = int(n_out_of_reference)
        summary["spin_reference_coverage_per_campaign"] = [
            bool(x) for x in reference_coverage]

    for info in campaign_info:
        print(f"  {info}")
    print(f"Built gwcat-selection-2.0 product: basis={spin_basis}, "
          f"n_det={n_det_total}, ndraw={ndraw_total}, "
          f"campaigns={len(set_list)}")

    return ExportProduct(kind="selection", columns=columns, attrs=attrs,
                         spin_basis=spin_basis, summary=summary)





def _sky_availability(set_list):
    """Per-campaign sky availability, warning when NaN-sky meets real-sky.

    NaN-sky campaigns (the semianalytic O1/O2 mixture rows) may legitimately be
    concatenated with real-sky ones, but a consumer doing sky work will
    silently lose the NaN campaigns' injections, so mixing deserves a warning
    at build time, not only a per-campaign flag in the attrs (GW-10).
    """
    sky = np.array([bool(getattr(s, "_sky_position_available", True))
                    for s in set_list], dtype=bool)
    if sky.any() and not sky.all():
        warnings.warn(
            f"concatenating campaign(s) WITHOUT drawn sky positions (exported "
            f"ra/dec are NaN) with campaign(s) that have them: "
            f"sky_position_available per campaign = {sky.tolist()}. Any "
            f"sky-dependent selection use will silently drop the NaN "
            f"campaigns' injections; cut on campaign, not on finiteness, if "
            f"that is not intended.")
    return sky


def _amax_pair(amax_detected):
    """``(amax_1, amax_2)`` as floats; ``(nan, nan)`` when unavailable.

    A per-body ``None`` (GW-05: "not a uniform draw, so no amax exists") maps to
    NaN, which is how "undetectable" is spelled in the exported attrs.
    """
    if amax_detected is None:
        return (float("nan"), float("nan"))
    return tuple(float("nan") if a is None else float(a)
                 for a in amax_detected[:2])


def _json_checks(checks):
    """Coerce a spin-check dict into JSON-serializable primitives."""
    out = {}
    for k, v in (checks or {}).items():
        if isinstance(v, (tuple, list)):
            out[k] = [bool(x) if isinstance(x, (bool, np.bool_)) else float(x)
                      for x in v]
        elif isinstance(v, (bool, np.bool_)):
            out[k] = bool(v)
        else:
            out[k] = float(v)
    return out
