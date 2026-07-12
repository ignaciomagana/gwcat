"""PE (posterior-sample) export builder for the versioned pipeline (PR 3).

:func:`build_pe_product` reproduces -- for the ``spin_basis="chieff"`` case --
the arrays and provenance of the legacy :meth:`gwcat.catalog.GWCatalog.to_darksirens`
EXACTLY for identical keyword arguments, and returns them as an
:class:`gwcat.export.product.ExportProduct` instead of writing a file.

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
    concatenated ``p_pe`` with the same ``clip(logp, -50, None)`` guard, and
  * the same concatenation order.

The only spin-basis-specific step -- the output columns plus the ``p_pe`` spin
factor -- is isolated in :func:`_apply_chieff_basis` so future bases
(``"component"``, ``"chieff_chip"``) slot in without touching the shared
selection/resampling scaffold.
"""
from __future__ import annotations

import numpy as np

from ..cosmology import make_cosmology, z_of_dL
from .product import ExportProduct

#: The datasets a chieff-basis PE export writes, in legacy order.
_CHIEFF_COLUMNS = ["ra", "dec", "m1det", "m2det", "chieff", "dL", "p_pe",
                   "redshift", "m1src", "m2src"]

#: Spin bases planned for the v2 builder; only "chieff" is implemented here.
_KNOWN_SPIN_BASES = ("chieff", "component", "chieff_chip")


def build_pe_product(cat, *, spin_basis="chieff", nsamp=4096, seed=0,
                     far_max=None, pastro_min=None, z_max=None,
                     replace="auto", cosmology=None, amax=0.99,
                     allowed_names=None, allowed_names_authoritative=True,
                     source_class=None, event_list=None,
                     allow_missing_far=False, require_far=False,
                     waveform_policy="preferred", approximant=None):
    """Build a PE :class:`ExportProduct` from a :class:`~gwcat.catalog.GWCatalog`.

    For ``spin_basis="chieff"`` this reproduces the legacy
    :meth:`GWCatalog.to_darksirens` arrays and provenance exactly (see the
    module docstring for the parity contract).  ``format_version`` is NOT set
    here -- it belongs to the writer.

    Only ``spin_basis="chieff"`` is implemented in this PR; ``"component"`` and
    ``"chieff_chip"`` raise :class:`NotImplementedError`.
    """
    if spin_basis != "chieff":
        if spin_basis in _KNOWN_SPIN_BASES:
            raise NotImplementedError(
                f"spin_basis={spin_basis!r} is not implemented yet; only "
                f"'chieff' is available in this release. The 'component' and "
                f"'chieff_chip' bases land in a follow-up PR.")
        raise ValueError(
            f"unknown spin_basis={spin_basis!r}; known bases are "
            f"{list(_KNOWN_SPIN_BASES)}.")

    # chieff basis ALWAYS uses "include" semantics: the 1-D chi_eff prior is
    # multiplied into p_pe here (Mode A), matching the legacy default.
    spin_prior_mode = "include"
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

    from ..schema import DARKSIRENS_REQUIRED
    need = list(DARKSIRENS_REQUIRED)
    sub._require_params(need, export="gwcat2 PE export")
    per = sub.get(need, per_event=True)
    rng = np.random.default_rng(seed)

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

    def _ss_meta(row, field):
        v = sub.meta.get(field)
        if v is None:
            return ""
        x = v[int(row)]
        return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)

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
            import warnings
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
        kept_ss_name.append(_ss_meta(row, "sample_set_name"))
        kept_ss_approx.append(_ss_meta(row, "approximant"))
        kept_ss_reason.append(
            str(reasons_arr[e]) if reasons_arr is not None
            and e < len(reasons_arr) else "")

    nobs = len(kept)
    data = {k: np.concatenate(v) if v else np.array([])
            for k, v in cols.items()}

    kept_H0_arr = np.asarray(kept_H0, dtype=float)
    kept_Om0_arr = np.asarray(kept_Om0, dtype=float)
    cosmology_per_event_varies = bool(
        nobs > 1 and (np.ptp(kept_H0_arr) > 0 or np.ptp(kept_Om0_arr) > 0))

    # ── Spin-basis-specific step (columns + p_pe spin factor) ───────────────
    columns = _apply_chieff_basis(data, amax=amax)

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
    import h5py
    _str = h5py.string_dtype()
    attrs = {
        # darksirens core
        "nsamp": int(nsamp),
        "nobs": int(nobs),
        "mock_data": False,
        # spin basis (new in v2)
        "spin_basis": "chieff",
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
    }

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
        "spin_basis": "chieff",
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
    })

    return ExportProduct(kind="pe", columns=columns, attrs=attrs,
                         spin_basis="chieff", summary=summary)


def _apply_chieff_basis(data, *, amax):
    """chieff-basis output columns + the 1-D chi_eff prior factor on p_pe.

    This is the single spin-basis-specific step.  ``data`` already carries the
    mass-Jacobian ``p_pe = m1det * p_dL_pe`` and the source-frame masses; here
    the 1-D isotropic chi_eff prior is multiplied into ``p_pe`` (chieff basis
    is always "include") with the same ``clip(logp, -50, None)`` guard the
    legacy exporter uses, and the legacy 10 columns are returned in order.
    """
    p_pe = data["p_pe"]
    if data["chieff"].size > 0:
        from ..spin import chi_eff_prior_logprob
        logp_chi = chi_eff_prior_logprob(data["chieff"], data["m1src"],
                                         data["m2src"], amax=amax)
        safe_logp = np.clip(logp_chi, a_min=-50.0, a_max=None)
        p_pe = p_pe * np.exp(safe_logp)

    return {
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
