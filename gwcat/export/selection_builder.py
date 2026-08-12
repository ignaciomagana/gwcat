"""Selection-function export builder for the versioned pipeline (PR5).

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
legacy exporter's ``pdraw`` bit-for-bit as well.  The chi_eff swap is applied on
the concatenated arrays in one call, in the SAME order, so the chieff-basis v2
``pdraw`` equals the v1 ``pdraw`` array under ``assert_array_equal``.

Spin-basis-specific step (the only place the bases diverge)
-----------------------------------------------------------
* **chieff**  ``pdraw *= exp(chi_eff_prior_logprob(chieff, m1src, m2src,
  amax))`` -- the legacy swap, minus the floor (GW-03).
* **component**  ``pdraw *= exp(ln_spin_component)`` with the per-injection,
  frac-independent ``ln_spin_component`` (public accessor); NO clip and NO
  chi_eff factor.  Requires ``component_spin_available`` for every campaign.
* **chieff_chip**  ``pdraw *= exp(chi_eff_chi_p_prior_logprob(chieff, chip,
  m1src, m2src, amax=amax_detected_k))`` with each campaign's own
  DETECTED amax; requires a single uniform-isotropic injected spin draw with a
  detectable amax per campaign (else :class:`SpinBasisError`).  If a campaign's
  ``amax_1 != amax_2`` the joint prior (which assumes a single amax) uses
  ``amax_1`` and warns -- mirroring the PE builder's convention.

Columns (all bases): the legacy 10 (``m1det, m2det, dL, chieff, ra, dec, m1src,
m2src, redshift, pdraw``) plus ``a1, a2, cost1, cost2, chip`` whenever every
contributing campaign carries them (they are free -- read from the additive
component-spin accessors -- so the chieff basis includes them too).
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import h5py

from ..spin import chi_eff_prior_logprob, chi_eff_chi_p_prior_logprob
from ..selection import (SelectionSet, CombinedSelectionSet,
                         PDRAW_STATE_BY_BASIS, _selection_provenance_dict)
from .product import ExportProduct

#: Spin bases the selection builder implements.
_KNOWN_SPIN_BASES = ("chieff", "component", "chieff_chip")

#: The cumulative-mixture SNR column used by the optional OR-branch.
_SNR_COLUMN = "semianalytic_observed_phase_maximized_snr_net"

#: Extra per-injection spin columns (emitted when available for every campaign).
_EXTRA_COLUMNS = ("a1", "a2", "cost1", "cost2")


class SpinBasisError(RuntimeError):
    """A requested spin basis is incompatible with a campaign's injected draw.

    Raised (for ``spin_basis="chieff_chip"``) when a campaign's injected spin
    distribution is not a single uniform-isotropic draw or has an undetectable
    ``amax`` -- e.g. a cumulative mixture of spin sub-populations, for which no
    single-``amax`` joint (chi_eff, chi_p) prior is defined.  The message names
    the offending file and the reason, and points at ``spin_basis="component"``
    (which retains the exact per-injection component spin draw for any file).
    """


def _read_snr_column(path):
    """Read the semianalytic SNR column from an injection file, or ``None``.

    Looks in the ``events`` group/dataset (O4) or the ``injections`` group (O3).
    Returns a float array, or ``None`` when the column is absent.
    """
    with h5py.File(path, "r") as f:
        for group in ("events", "injections"):
            if group not in f:
                continue
            table = f[group]
            names = (table.dtype.names if isinstance(table, h5py.Dataset)
                     and table.dtype.names is not None else set(table.keys()))
            if _SNR_COLUMN in names:
                return np.asarray(table[_SNR_COLUMN], dtype=float)
    return None


def _detect_keep(s, far_threshold, source_class, snr_threshold):
    """``(keep, det, sc_mask)`` for one campaign, mirroring the legacy masks.

    ``keep = detect & sc_mask`` where ``detect`` is the FAR cut, OR-ed with the
    SNR cut (``snr > snr_threshold``) when ``snr_threshold`` is not ``None``.
    """
    det = s.detected_mask(far_threshold)
    sc_mask = s.source_class_mask(source_class)
    if snr_threshold is not None:
        snr = _read_snr_column(s.path)
        if snr is None:
            raise ValueError(
                f"snr_threshold={snr_threshold} was requested but injection "
                f"file {s.path} has no {_SNR_COLUMN!r} column; only cumulative "
                f"mixture files carry it. Drop snr_threshold (=None) to use the "
                f"FAR cut alone.")
        detect = det | (np.asarray(snr, dtype=float) > snr_threshold)
    else:
        detect = det
    return detect & sc_mask, det, sc_mask


def _campaign_chieff_chip_lnfactor(s, keep, amax, strict):
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
    if amax_detected is None or any(not np.isfinite(a) for a in amax_detected):
        raise SpinBasisError(
            f"{s.path}: injected spin amax is undetectable (the injected spin "
            f"draw is not a single uniform-magnitude distribution); the joint "
            f"(chi_eff, chi_p) prior needs one amax. Use spin_basis='component'.")
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
    lnp = chi_eff_chi_p_prior_logprob(
        s._chieff[keep], s._chi_p[keep], s._m1src[keep], s._m2src[keep],
        amax=amax_1)
    # No -50 floor (GW-03).  A DETECTED injection whose assumed draw density is
    # zero is not a small number, it is a contradiction: the injection really was
    # drawn and really was detected, so a density of zero means the assumed
    # prior does not cover the campaign.  The caller refuses rather than writing
    # it, because the consumer would silently exclude that injection from the
    # selection integral and bias mu.
    return np.asarray(lnp, dtype=float), amax_1


def build_selection_product(sets, *, spin_basis="component", far_threshold=1.0,
                            source_class=None, amax=0.99, snr_threshold=None,
                            strict=True):
    """Build a selection :class:`ExportProduct` from one or more SelectionSets.

    Parameters
    ----------
    sets : SelectionSet, CombinedSelectionSet, or list of SelectionSet
        One campaign or several; handled uniformly (see the module docstring's
        parity contract).
    spin_basis : {"component", "chieff", "chieff_chip"}
        Per-injection spin factor / extra columns (see the module docstring).
        Defaults to ``"component"`` -- the exact component-basis draw density,
        which is well-defined for every file.
    far_threshold : float, default 1.0
        FAR detection threshold in yr^-1, applied per campaign.
    source_class : str, iterable, or None
        Optional source-class subset (see
        :meth:`gwcat.selection.SelectionSet.source_class_mask`).  Subsetting,
        not reweighting: ``ndraw`` is unchanged.
    amax : float, default 0.99
        chi_eff-prior spin amax for the ``"chieff"`` basis (matches the legacy
        swap and the PE builder).  Ignored by ``"component"``; the
        ``"chieff_chip"`` basis uses each campaign's DETECTED amax instead.
    snr_threshold : float, optional
        When set, detection becomes ``far-detected OR (snr > snr_threshold)``
        using the cumulative-mixture ``semianalytic_observed_phase_maximized_snr_net``
        column.  Default ``None`` reproduces the FAR-only cut exactly.
    strict : bool, default True
        For ``"chieff_chip"``: raise :class:`SpinBasisError` when a campaign is
        not verified single-uniform-isotropic.  ``False`` warns and proceeds
        when a detected amax is nonetheless available (an undetectable amax
        always raises).

    Returns
    -------
    gwcat.export.product.ExportProduct
        ``kind="selection"``; ``format_version`` is the writer's, not set here.
    """
    if spin_basis not in _KNOWN_SPIN_BASES:
        raise ValueError(
            f"unknown spin_basis={spin_basis!r}; known bases are "
            f"{list(_KNOWN_SPIN_BASES)}.")

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

    # Cosmology consistency check (mirror CombinedSelectionSet.to_darksirens).
    H0s = [s.H0 for s in set_list]
    Om0s = [s.Om0 for s in set_list]
    if max(H0s) - min(H0s) > 1.0 or max(Om0s) - min(Om0s) > 0.05:
        warnings.warn(
            f"Cosmology mismatch across campaigns: H0={H0s}, Om0={Om0s}. "
            f"Results may be inconsistent.")

    # component basis: every campaign must carry the exact component spin draw.
    if spin_basis == "component":
        for s in set_list:
            if not s.component_spin_available:
                raise SpinBasisError(
                    f"{s.path}: component-basis spin draw density unavailable "
                    f"(spin_format={s.spin_meta.get('spin_format')!r}); this "
                    f"file lacks the per-spin draw information needed for "
                    f"spin_basis='component'. Use spin_basis='chieff'.")

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
    lnfactor_parts = []          # component / chieff_chip: per-campaign ln factor
    extras_available = True      # a1/a2/cost1/cost2 present for every campaign
    chip_available = True        # chi_p present for every campaign

    n_det_total = 0
    n_before_total = 0
    n_after_total = 0
    far_columns_union = []
    campaign_info = []
    chieff_chip_amax = []        # per (non-empty) campaign detected amax used

    for k, s in enumerate(set_list):
        keep, det, sc_mask = _detect_keep(s, far_threshold, source_class,
                                          snr_threshold)
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

        # Per-basis per-injection ln factor (chieff applies its swap post-concat).
        if spin_basis == "component":
            lnfactor_parts.append(np.asarray(s._ln_spin_component)[keep])
        elif spin_basis == "chieff_chip":
            lnfac, amax_used = _campaign_chieff_chip_lnfactor(
                s, keep, amax, strict)
            lnfactor_parts.append(lnfac)
            chieff_chip_amax.append(amax_used)

        n_det_total += n_det_k
        campaign_info.append(
            f"{s.path}: N={ndraw_per[k]}, T={s._T_yr:.2f}yr, "
            f"n_det={n_det_k}, frac={frac:.4f}")

    if n_det_total == 0:
        raise RuntimeError(
            f"No detected injections across {len(set_list)} campaign(s) at "
            f"FAR < {far_threshold}"
            + ("" if source_class is None
               else f" in source class {source_class!r}"))

    data = {k: np.concatenate(v) for k, v in parts.items()}

    # ── Spin-basis-specific step: apply the per-injection pdraw factor ──────
    if spin_basis == "chieff":
        # Legacy swap: one call on the concatenated arrays.  The -50 floor is
        # gone (GW-03) -- see _campaign_chieff_chip_lnfactor for why a floored
        # injection is worse here than on the PE side.
        ln_factor = np.asarray(
            chi_eff_prior_logprob(data["chieff"], data["m1src"],
                                  data["m2src"], amax=amax), dtype=float)
        with np.errstate(over="ignore"):
            data["pdraw"] = data["pdraw"] * np.exp(ln_factor)
    else:
        ln_factor = np.concatenate(lnfactor_parts)
        with np.errstate(over="ignore"):
            data["pdraw"] = data["pdraw"] * np.exp(ln_factor)

    # ── Out-of-support DETECTED injections are fatal (GW-03) ────────────────
    # A PE sample with zero prior density can be dropped: the posterior simply
    # has no support there.  A detected injection cannot -- it was drawn from the
    # real campaign and it triggered, so the Monte-Carlo sum over draws must
    # include it.  If the ASSUMED prior gives it zero density the assumption is
    # wrong, and the consumer's `pdraw > 0` guard would quietly exclude it and
    # bias mu low.
    n_unsupported = int(np.sum(~np.isfinite(ln_factor)))
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

    # ── Output columns: legacy 10 + (a1,a2,cost1,cost2,chip when available) ──
    columns = {
        "m1det": data["m1det"], "m2det": data["m2det"], "dL": data["dL"],
        "chieff": data["chieff"], "ra": data["ra"], "dec": data["dec"],
        "m1src": data["m1src"], "m2src": data["m2src"],
        "redshift": data["z"], "pdraw": data["pdraw"],
    }
    if extras_available:
        for name in _EXTRA_COLUMNS:
            columns[name] = np.concatenate(extra_parts[name])
    if chip_available:
        columns["chip"] = np.concatenate(extra_parts["chip"])

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
        "sky_position_available": np.array(
            [bool(getattr(s, "_sky_position_available", True))
             for s in set_list], dtype=bool),
        "component_columns_emitted": bool(extras_available),
    })

    # Basis-specific spin-prior contract attrs (mirror the v1 / PE naming).
    if spin_basis == "chieff":
        attrs["spin_prior_mode"] = "include"
        attrs["chi_eff_swap_applied"] = True
        attrs["chi_eff_prior_applied_to_pdraw"] = True
        attrs["chi_eff_amax"] = float(amax)
    elif spin_basis == "component":
        attrs["spin_prior_mode"] = "component"
        attrs["chi_eff_prior_applied_to_pdraw"] = False
        attrs["component_spin_draw_retained"] = True
    else:  # chieff_chip
        attrs["spin_prior_mode"] = "include"
        attrs["chi_eff_chi_p_swap_applied"] = True
        attrs["chi_eff_chi_p_prior_applied_to_pdraw"] = True
        attrs["chi_eff_amax"] = float(amax)
        attrs["chi_eff_chi_p_amax_detected_per_campaign"] = np.array(
            chieff_chip_amax, dtype=float)

    if snr_threshold is not None:
        attrs["significance_snr_column"] = _SNR_COLUMN
        attrs["significance_snr_threshold"] = float(snr_threshold)
        attrs["significance_type"] = "far_or_snr"

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
        "source_class_filter": (None if source_class is None
                                else str(source_class)),
        "source_class_counts_detected": value_counts(
            [normalize_source_class(c) for c in classes_det]),
        "cosmology_H0": float(set_list[0].H0),
        "cosmology_Om0": float(set_list[0].Om0),
        "cosmology_override_used": bool(
            any(getattr(s, "_cosmology_override", False) for s in set_list)),
        "injected_spin_format": [str(s.spin_meta.get("spin_format"))
                                 for s in set_list],
        "component_columns_emitted": bool(extras_available),
    }

    for info in campaign_info:
        print(f"  {info}")
    print(f"Built gwcat-selection-2.0 product: basis={spin_basis}, "
          f"n_det={n_det_total}, ndraw={ndraw_total}, "
          f"campaigns={len(set_list)}")

    return ExportProduct(kind="selection", columns=columns, attrs=attrs,
                         spin_basis=spin_basis, summary=summary)


def _amax_pair(amax_detected):
    """``(amax_1, amax_2)`` as floats; ``(nan, nan)`` when unavailable."""
    if amax_detected is None:
        return (float("nan"), float("nan"))
    return (float(amax_detected[0]), float(amax_detected[1]))


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
