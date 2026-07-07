"""Stage 2: fast reader over store.h5.

  * select(...)              -> a filtered GWCatalog (view, no copy of samples)
  * get(params)              -> dict of concatenated arrays for current selection
  * event(name)             -> per-event dict
  * derived methods          -> chi_eff, q, chirp_mass, source masses, redshift,
                                computed on demand (NOT stored)
  * to_darksirens()          -> the ONLY place the (m1det,q,dL)-basis mass
                                Jacobian is applied. Writes exactly what
                                darksirens.gw.utils.load_gw_samples reads.
                                (_to_darksirens_format is a deprecated alias.)
"""
from __future__ import annotations

import warnings

import numpy as np
import h5py

from .cosmology import make_cosmology, z_of_dL
from .source_class import (normalize_source_class, resolve_filter_classes,
                           load_event_list)


class GWCatalog:
    def __init__(self, store_path, _sel=None):
        self.path = store_path
        with h5py.File(store_path, "r") as f:
            self.params = [p.decode() if isinstance(p, bytes) else str(p)
                           for p in f.attrs["param_names"]]
            self.offsets = f["index/offsets"][:]
            self.names = np.array([n for n in f["index/event_names"][:]])
            self.meta = {k: f[f"meta/{k}"][:] for k in f["meta"].keys()}
        # decode bytes -> str for string meta
        for k, v in self.meta.items():
            if v.dtype.kind in ("S", "O"):
                self.meta[k] = np.array([x.decode() if isinstance(x, bytes) else x
                                         for x in v])
        self.names = np.array([n.decode() if isinstance(n, bytes) else n
                               for n in self.names])
        self._sel = np.arange(len(self.names)) if _sel is None else np.asarray(_sel)
        self._source_class_cache = None
        # Provenance of the most recent select() call (defaults for a fresh
        # catalog with no filters applied yet).
        self._far_policy = "none"
        self._n_missing_far = 0
        self._selection_source_class = None
        self._selection_event_list = None

    # ---- selection (operates on metadata only; cheap) --------------------
    @property
    def n_events(self):
        return len(self._sel)

    @property
    def event_names(self):
        return self.names[self._sel]

    @property
    def source_class(self):
        """Canonical per-event source-class labels for ALL events in the store.

        Prefers the ``meta/source_class`` column; falls back to the legacy
        ``meta/compact_type`` column; otherwise ``Unknown``.  Always returns
        canonical labels (BBH/NSBH/BNS/MassGap/Unknown).
        """
        if self._source_class_cache is None:
            if "source_class" in self.meta:
                raw = self.meta["source_class"]
            elif "compact_type" in self.meta:
                raw = self.meta["compact_type"]
            else:
                raw = ["Unknown"] * len(self.names)
            self._source_class_cache = np.array(
                [normalize_source_class(x) for x in raw])
        return self._source_class_cache

    def _far_available_mask(self):
        """Boolean per-event mask: is a usable FAR present for this event?

        Uses the explicit ``meta/far_available`` column when the store provides
        it; otherwise derives availability from a finite ``meta/far`` value.
        Absent FAR is a valid, non-crashing state.
        """
        if "far_available" in self.meta:
            return np.asarray(self.meta["far_available"], dtype=float) > 0.5
        if "far" in self.meta:
            return np.isfinite(np.asarray(self.meta["far"], dtype=float))
        return np.zeros(len(self.names), dtype=bool)

    def select(self, compact_type=None, far_max=None, pastro_min=None,
               snr_min=None, z_max=None, m1_src_range=None, m2_src_range=None,
               sky_area_max=None, names=None, allowed_names=None,
               allowed_names_authoritative=True, source_class=None,
               event_list=None, allow_missing_far=False, require_far=False):
        """Return a filtered view of the catalog (no sample copy).

        Source-class / event-list / FAR options (PR 2)
        ------------------------------------------------
        source_class : str, iterable, or None
            Filter by canonical source class using the ``bbh``/``nsbh``/``bns``/
            ``cbc`` keywords (``cbc`` = all compact-binary classes) or a
            canonical class name.  Operates on the ``meta/source_class`` column
            (falling back to ``meta/compact_type``), NOT on a static name list.
        event_list : str, path, iterable, or None
            Restrict the selection to a user event-list file (one name per line;
            ``#`` comments allowed) or an in-memory sequence of event names.
        allow_missing_far, require_far : bool
            Policy for ``far_max`` when a selected event has no FAR
            (``far_available=False``).  ``require_far=True`` fails loudly;
            ``allow_missing_far=True`` keeps the event, warns, and records the
            absence; the default drops missing-FAR events (legacy behavior).
        """
        import warnings
        if require_far and allow_missing_far:
            raise ValueError(
                "require_far and allow_missing_far are mutually exclusive")

        m = np.ones(len(self.names), dtype=bool)
        if compact_type is not None:
            apply_compact_type = (allowed_names is None or
                                  not allowed_names_authoritative)
            if apply_compact_type:
                m &= (self.meta["compact_type"] == compact_type)
        if source_class is not None:
            allowed_classes = resolve_filter_classes(source_class)
            m &= np.isin(self.source_class, list(allowed_classes))
        if pastro_min is not None:
            pa = self.meta["pastro"]
            m &= np.where(np.isnan(pa), False, pa >= pastro_min)
        if snr_min is not None:
            m &= np.where(np.isnan(self.meta["snr_med"]), False,
                          self.meta["snr_med"] >= snr_min)
        if m1_src_range is not None:
            lo, hi = m1_src_range
            m &= (self.meta["m1_src_med"] >= lo) & (self.meta["m1_src_med"] <= hi)
        if m2_src_range is not None:
            lo, hi = m2_src_range
            m &= (self.meta["m2_src_med"] >= lo) & (self.meta["m2_src_med"] <= hi)
        if sky_area_max is not None and "sky_area_90" in self.meta:
            sa = self.meta["sky_area_90"]
            m &= np.where(np.isnan(sa), False, sa <= sky_area_max)
        _whitelist = allowed_names if allowed_names is not None else names
        if _whitelist is not None:
            _whitelist_arr = np.asarray(_whitelist)
            missing = set(_whitelist_arr) - set(self.names)
            if missing:
                warnings.warn(
                    f"allowed_names: {len(missing)} name(s) not found in store "
                    f"and will be skipped: {sorted(missing)}"
                )
            if compact_type is not None and allowed_names is not None:
                present = np.isin(self.names, _whitelist_arr)
                dropped = self.names[present &
                                     (self.meta["compact_type"] != compact_type)]
                if len(dropped):
                    action = ("will not be dropped because allowed_names is "
                              "authoritative" if allowed_names_authoritative
                              else "will be dropped")
                    warnings.warn(
                        f"compact_type={compact_type!r} excludes {len(dropped)} "
                        f"allowed_names event(s); they {action}: "
                        f"{sorted(dropped.tolist())}"
                    )
            m &= np.isin(self.names, _whitelist_arr)

        # User event-list file / sequence (additional intersection gate).
        if event_list is not None:
            listed = load_event_list(event_list)
            listed_arr = np.asarray(listed)
            missing_ev = set(listed_arr) - set(self.names)
            if missing_ev:
                warnings.warn(
                    f"event_list: {len(missing_ev)} name(s) not found in store "
                    f"and will be skipped: {sorted(missing_ev)}"
                )
            m &= np.isin(self.names, listed_arr)

        # ── FAR handling with explicit missing-FAR policy ─────────────────
        far_policy = "none"
        n_missing_far = 0
        if far_max is not None:
            far = (np.asarray(self.meta["far"], dtype=float)
                   if "far" in self.meta else np.full(len(self.names), np.nan))
            fa = self._far_available_mask()
            in_scope = np.isin(np.arange(len(self.names)), self._sel)
            # Events passing every other filter and in the current selection.
            candidates = m & in_scope
            missing_mask = candidates & ~fa
            n_missing_far = int(missing_mask.sum())
            below = np.zeros(len(self.names), dtype=bool)
            below[fa] = far[fa] <= far_max
            if require_far:
                if n_missing_far > 0:
                    raise ValueError(
                        f"require_far=True but {n_missing_far} selected event(s) "
                        f"have no FAR (far_available=False): "
                        f"{sorted(self.names[missing_mask].tolist())}. "
                        f"Pass allow_missing_far=True to keep them instead.")
                m &= below
                far_policy = "require"
            elif allow_missing_far:
                if n_missing_far > 0:
                    warnings.warn(
                        f"allow_missing_far=True: keeping {n_missing_far} "
                        f"event(s) with missing FAR through the far_max cut: "
                        f"{sorted(self.names[missing_mask].tolist())}")
                m &= (below | ~fa)
                far_policy = "allow_missing"
            else:
                if n_missing_far > 0:
                    warnings.warn(
                        f"far_max cut dropped {n_missing_far} event(s) with "
                        f"missing FAR (far_available=False); pass "
                        f"allow_missing_far=True to keep them or require_far=True "
                        f"to fail loudly: {sorted(self.names[missing_mask].tolist())}")
                m &= below
                far_policy = "drop_missing"

        if pastro_min is not None and "pastro" in self.meta:
            sel_pre = np.nonzero(m & np.isin(np.arange(len(self.names)),
                                             self._sel))[0]
            if np.isnan(np.asarray(self.meta["pastro"], dtype=float)[sel_pre]).any():
                warnings.warn("p_astro NaN for some events; populate via "
                              "event_table at ingest to use these cuts.")

        sel = np.nonzero(m & np.isin(np.arange(len(self.names)), self._sel))[0]
        result = GWCatalog(self.path, _sel=sel)
        result._far_policy = far_policy
        result._n_missing_far = n_missing_far
        result._selection_source_class = source_class
        result._selection_event_list = event_list
        return result

    # ---- sample access ---------------------------------------------------
    def _slices(self):
        return [(self.offsets[i], self.offsets[i + 1]) for i in self._sel]

    def get(self, params, per_event=False):
        """Read columns for the current selection.

        per_event=False -> dict of flat concatenated arrays.
        per_event=True  -> dict of lists (one array per event).
        """
        params = [params] if isinstance(params, str) else list(params)
        out = {p: [] for p in params}
        sl = self._slices()
        with h5py.File(self.path, "r") as f:
            for p in params:
                if p not in self.params:
                    raise KeyError(f"{p} not in store; stored: {self.params}")
                d = f[f"samples/{p}"]
                out[p] = [d[a:b] for (a, b) in sl]
        if per_event:
            return out
        return {p: np.concatenate(v) if v else np.array([]) for p, v in out.items()}

    def event(self, name, params=None):
        i = int(np.nonzero(self.names == name)[0][0])
        a, b = self.offsets[i], self.offsets[i + 1]
        params = params or self.params
        with h5py.File(self.path, "r") as f:
            return {p: f[f"samples/{p}"][a:b] for p in params if p in self.params}

    # ---- derived quantities (computed, not stored) -----------------------
    def mass_ratio(self, per_event=False):
        d = self.get(["mass_1", "mass_2"], per_event=per_event)
        if per_event:
            return [m2 / m1 for m1, m2 in zip(d["mass_1"], d["mass_2"])]
        return d["mass_2"] / d["mass_1"]

    def chirp_mass(self, frame="detector", per_event=False):
        m1, m2 = ("mass_1", "mass_2") if frame == "detector" else \
                 ("mass_1_source", "mass_2_source")
        d = self.get([m1, m2], per_event=per_event)
        f = lambda a, b: (a * b) ** 0.6 / (a + b) ** 0.2
        if per_event:
            return [f(x, y) for x, y in zip(d[m1], d[m2])]
        return f(d[m1], d[m2])

    def chi_eff(self, per_event=False):
        if "chi_eff" in self.params:
            return self.get("chi_eff", per_event=per_event)["chi_eff"] \
                if not per_event else self.get("chi_eff", per_event=True)["chi_eff"]
        # derive from z-components
        d = self.get(["mass_1", "mass_2", "spin_1z", "spin_2z"], per_event=per_event)
        f = lambda m1, m2, s1, s2: (m1 * s1 + m2 * s2) / (m1 + m2)
        if per_event:
            return [f(*x) for x in zip(d["mass_1"], d["mass_2"], d["spin_1z"], d["spin_2z"])]
        return f(d["mass_1"], d["mass_2"], d["spin_1z"], d["spin_2z"])

    def source_masses(self, cosmology=None):
        """Return (m1_src, m2_src). cosmology=None uses stored redshift;
        otherwise recompute z from dL under the given (H0, Om0)."""
        if cosmology is None:
            if "redshift" in self.params:
                d = self.get(["mass_1", "mass_2", "redshift"])
                z = d["redshift"]
            else:
                raise ValueError("no stored redshift; pass cosmology=(H0,Om0)")
        else:
            d = self.get(["mass_1", "mass_2", "luminosity_distance"])
            z = z_of_dL(d["luminosity_distance"], make_cosmology(*cosmology))
        return d["mass_1"] / (1 + z), d["mass_2"] / (1 + z)

    # ---- darksirens export (Jacobian lives here, and only here) ----------
    def to_darksirens(self, out_path, compact_type=None, nsamp=4096,
                      far_max=None, pastro_min=None, z_max=None,
                      seed=0, replace="auto", cosmology=None, amax=0.99,
                      allowed_names=None,
                      allowed_names_authoritative=True,
                      source_class=None, event_list=None,
                      allow_missing_far=False, require_far=False):
        """Write an HDF5 consumable by darksirens.gw.utils.load_gw_samples.

        p_pe convention
        ---------------
        load_gw_samples expects p_pe in the (m1det, q, dL) basis.  It then
        multiplies by the 1-D chi_eff prior and normalises per event.

        For a uniform detector-frame component-mass prior the (m1det, q)-basis
        density carries a Jacobian |dm2det/dq| = m1det, so:

            p_pe = m1det * p_dL_pe          (chi_eff EXCLUDED; loader adds it)

        This is the ONLY place the mass Jacobian is applied. The store keeps the
        mass-prior-agnostic p_dL_pe.

        Parameters
        ----------
        cosmology : tuple (H0, Om0) or None
            Cosmology for z_max cut and stored source masses / redshift.
            None → read per-event PE cosmology from the store metadata.
        z_max : float or None
            Per-sample redshift cut.  Samples above z_max are dropped BEFORE
            resampling to nsamp.  Requires cosmology or stored redshift.
        compact_type : str or None
            Optional metadata compact-type cut.  The default None means no
            derived compact-type gate is applied; use compact_type="BBH" only
            when that additional metadata cut is desired.
        allowed_names_authoritative : bool
            If True, allowed_names is treated as authoritative and compact_type
            is not applied as an additional gate.  A warning is still emitted
            when compact_type would exclude allowed names.
        source_class : str, iterable, or None
            Source-class filter (``bbh``/``nsbh``/``bns``/``cbc`` or a canonical
            class).  ``cbc`` selects all compact-binary classes.  See
            :meth:`select`.
        event_list : str, path, iterable, or None
            User event-list filter (file path or in-memory sequence).
        allow_missing_far, require_far : bool
            Missing-FAR policy for the ``far_max`` cut; recorded in the output
            HDF5 provenance attributes ``far_policy``, ``allow_missing_far``,
            ``require_far``, and ``n_events_missing_far``.
        """
        sub = self.select(compact_type=compact_type, far_max=far_max,
                          pastro_min=pastro_min, allowed_names=allowed_names,
                          allowed_names_authoritative=allowed_names_authoritative,
                          source_class=source_class, event_list=event_list,
                          allow_missing_far=allow_missing_far,
                          require_far=require_far)
        need = ["mass_1", "mass_2", "luminosity_distance", "ra", "dec",
                "chi_eff", "p_dL_pe"]
        per = sub.get(need, per_event=True)
        rng = np.random.default_rng(seed)

        # Resolve cosmology for z computation
        if cosmology is not None:
            _cosmo = make_cosmology(*cosmology)
            pe_H0, pe_Om0 = float(cosmology[0]), float(cosmology[1])
        else:
            # Use per-event PE cosmology (take first event's; they should agree)
            pe_H0 = float(sub.meta["dL_prior_H0"][sub._sel[0]])
            pe_Om0 = float(sub.meta["dL_prior_Om0"][sub._sel[0]])
            _cosmo = make_cosmology(pe_H0, pe_Om0)

        cols = {k: [] for k in ["m1det", "m2det", "dL", "ra", "dec",
                                "chieff", "p_pe", "redshift", "m1src", "m2src"]}
        kept = []
        for e in range(sub.n_events):
            n = len(per["luminosity_distance"][e])
            if n == 0:
                continue

            dL_e = per["luminosity_distance"][e]
            m1_e = per["mass_1"][e]
            m2_e = per["mass_2"][e]

            # Per-sample z_max cut
            if z_max is not None:
                z_e = z_of_dL(dL_e, _cosmo)
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

            # Redshift and source masses under the PE cosmology
            z = z_of_dL(dL, _cosmo)

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

        nobs = len(kept)
        data = {k: np.concatenate(v) if v else np.array([])
                for k, v in cols.items()}

        # Apply 1-D chi_eff prior to p_pe
        if data["chieff"].size > 0:
            from .spin import chi_eff_prior_logprob
            logp_chi = chi_eff_prior_logprob(data["chieff"], data["m1src"],
                                             data["m2src"], amax=amax)
            safe_logp = np.clip(logp_chi, a_min=-50.0, a_max=None)
            data["p_pe"] = data["p_pe"] * np.exp(safe_logp)

        # Sanity check
        expected = nobs * nsamp
        assert data["m1det"].size == expected, \
            f"data length {data['m1det'].size} != nobs*nsamp = {expected}"

        with h5py.File(out_path, "w") as f:
            # --- Attributes darksirens reads ---
            f.attrs["nsamp"] = int(nsamp)
            f.attrs["nobs"] = int(nobs)
            f.attrs["mock_data"] = False
            # --- Provenance (gwcat-specific) ---
            f.attrs["format_version"] = "gwcat-1.0"
            f.attrs["compact_type"] = ("" if compact_type is None
                                       else str(compact_type))
            f.attrs["mass_prior_basis"] = "uniform_detector_frame"
            f.attrs["chi_eff_in_p_pe"] = True
            f.attrs["chi_eff_amax"] = float(amax)
            f.attrs["pe_cosmology_H0"] = pe_H0
            f.attrs["pe_cosmology_Om0"] = pe_Om0
            # --- Source-class / FAR-policy provenance (PR 2) ---
            f.attrs["source_class_filter"] = (
                "" if source_class is None else str(source_class))
            f.attrs["event_list_filter"] = (
                "" if event_list is None
                else (str(event_list) if isinstance(event_list, (str, bytes))
                      else "custom_sequence"))
            f.attrs["far_policy"] = getattr(sub, "_far_policy", "none")
            f.attrs["allow_missing_far"] = bool(allow_missing_far)
            f.attrs["require_far"] = bool(require_far)
            f.attrs["n_events_missing_far"] = int(
                getattr(sub, "_n_missing_far", 0))
            f.attrs.create("event_names",
                           np.array([str(k) for k in kept],
                                    dtype=h5py.string_dtype()))
            # --- Datasets ---
            for k in ["ra", "dec", "m1det", "m2det", "chieff", "dL", "p_pe",
                       "redshift", "m1src", "m2src"]:
                f.create_dataset(k, data=data[k], compression="gzip",
                                 shuffle=False)
        print(f"Wrote {out_path}: nobs={nobs}, nsamp={nsamp}, "
              f"H0={pe_H0}, Om0={pe_Om0}, compact_type={compact_type}")
        return out_path

    def _to_darksirens_format(self, *args, **kwargs):
        """Deprecated alias for :meth:`to_darksirens`.

        Kept for backward compatibility with existing scripts/notebooks.
        Will be removed in a future release; migrate to ``to_darksirens``.
        """
        warnings.warn(
            "GWCatalog._to_darksirens_format is deprecated and will be "
            "removed in a future release; use GWCatalog.to_darksirens "
            "instead (identical signature and behavior).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.to_darksirens(*args, **kwargs)

    # ---- diagnostics ---------------------------------------------------------
    @property
    def nsamp_per_event(self):
        """Number of posterior samples per event (array)."""
        return np.diff(self.offsets)[self._sel]

    def summary(self):
        """Print a compact event table for the current selection.

        Columns: name, catalog, analysis, nsamp, m1_src_med, m2_src_med,
        dL_med (from stored medians), compact_type, FAR, p_astro.
        """
        hdr = (f"{'name':<22} {'cat':<10} {'analysis':<22} {'nsamp':>6} "
               f"{'m1s':>6} {'m2s':>6} {'type':<5} "
               f"{'FAR':>10} {'p_astro':>7}")
        print(hdr)
        print("-" * len(hdr))
        ns = self.nsamp_per_event
        for j, i in enumerate(self._sel):
            name = self.names[i]
            cat = self.meta["catalog"][i] if "catalog" in self.meta else "?"
            ana = self.meta["analysis_used"][i] if "analysis_used" in self.meta else "?"
            m1 = self.meta["m1_src_med"][i] if "m1_src_med" in self.meta else np.nan
            m2 = self.meta["m2_src_med"][i] if "m2_src_med" in self.meta else np.nan
            ct = self.meta["compact_type"][i] if "compact_type" in self.meta else "?"
            far = self.meta["far"][i] if "far" in self.meta else np.nan
            pa = self.meta["pastro"][i] if "pastro" in self.meta else np.nan

            far_s = f"{far:.2e}" if np.isfinite(far) else "NaN"
            pa_s = f"{pa:.3f}" if np.isfinite(pa) else "NaN"
            m1_s = f"{m1:.1f}" if np.isfinite(m1) else "?"
            m2_s = f"{m2:.1f}" if np.isfinite(m2) else "?"

            print(f"{name:<22} {str(cat):<10} {str(ana):<22} {ns[j]:>6} "
                  f"{m1_s:>6} {m2_s:>6} {str(ct):<5} "
                  f"{far_s:>10} {pa_s:>7}")
        print(f"\n{self.n_events} events, "
              f"{int(ns.sum())} total samples")


def validate_export(gw_path: str, selection_path: str = None, strict: bool = False):
    """Check a darksirens PE export (and optionally a selection export) for
    internal consistency.

    Checks:
      * array lengths == nobs * nsamp
      * p_pe finite and positive
      * source masses <= detector masses
      * redshift non-negative
      * format_version present
      * if selection_path: cosmology consistent, pdraw finite/positive,
        ndraw > n_detected, chi_eff_swap_applied flag present

    Returns a dict of {check_name: passed_bool}.  If strict=True, raises on
    the first failure.
    """
    results = {}

    def _check(name, cond, msg=""):
        results[name] = bool(cond)
        if not cond and strict:
            raise AssertionError(f"validate_export FAILED: {name}. {msg}")
        if not cond:
            print(f"  FAIL: {name}  {msg}")
        return cond

    # --- PE file ---
    print(f"Validating PE export: {gw_path}")
    with h5py.File(gw_path, "r") as f:
        nobs = int(f.attrs.get("nobs", 0))
        nsamp = int(f.attrs.get("nsamp", 0))
        expected = nobs * nsamp

        _check("pe_format_version", "format_version" in f.attrs)
        _check("pe_mock_data_attr", "mock_data" in f.attrs)
        _check("pe_cosmology_H0", "pe_cosmology_H0" in f.attrs)

        for ds in ["m1det", "m2det", "dL", "chieff", "ra", "dec", "p_pe"]:
            if ds in f:
                _check(f"pe_{ds}_length", f[ds].shape[0] == expected,
                       f"{f[ds].shape[0]} != {expected}")

        if "p_pe" in f:
            p = np.array(f["p_pe"])
            if p.size > 0:
                _check("pe_p_pe_finite", np.all(np.isfinite(p)))
                _check("pe_p_pe_nonneg", np.all(p >= 0),
                       f"min={p.min():.3e}")
                n_zero = int(np.sum(p == 0))
                if n_zero > 0:
                    pct = 100 * n_zero / p.size
                    print(f"  NOTE: {n_zero} ({pct:.1f}%) p_pe samples are zero "
                          f"(expected at distance-prior tails)")
                # Check no event is entirely zero-weight
                p_ev = p.reshape(nobs, nsamp) if nobs > 0 and nsamp > 0 else p
                if p_ev.ndim == 2:
                    all_zero_events = np.sum(p_ev, axis=1) == 0
                    _check("pe_no_allzero_events", not np.any(all_zero_events),
                           f"{int(all_zero_events.sum())} event(s) have ALL p_pe=0")
            else:
                _check("pe_p_pe_nonempty", False, "p_pe is empty")

        if "m1src" in f and "m1det" in f:
            _check("pe_m1src_le_m1det",
                   np.all(np.array(f["m1src"]) <= np.array(f["m1det"]) + 1e-10))

        if "redshift" in f:
            _check("pe_redshift_nonneg", np.all(np.array(f["redshift"]) >= -1e-10))

    # --- Selection file ---
    if selection_path is not None:
        print(f"Validating selection export: {selection_path}")
        with h5py.File(selection_path, "r") as f:
            _check("sel_format_version", "format_version" in f.attrs)
            _check("sel_chi_eff_swap_flag", "chi_eff_swap_applied" in f.attrs)

            ndraw = int(f.attrs.get("ndraw", 0))
            n_det = int(f.attrs.get("n_detected", 0))
            _check("sel_ndraw_gt_ndet", ndraw > n_det,
                   f"ndraw={ndraw} <= n_detected={n_det}")

            if "pdraw" in f:
                pd = np.array(f["pdraw"])
                _check("sel_pdraw_length", pd.shape[0] == n_det)
                if pd.size > 0:
                    _check("sel_pdraw_finite", np.all(np.isfinite(pd)))
                    _check("sel_pdraw_positive", np.all(pd > 0),
                           f"min={pd.min():.3e}")
                else:
                    _check("sel_pdraw_nonempty", False, "pdraw is empty")

        # Cross-check cosmology
        with h5py.File(gw_path, "r") as fg, h5py.File(selection_path, "r") as fs:
            pe_H0 = fg.attrs.get("pe_cosmology_H0")
            sel_H0 = fs.attrs.get("cosmology_H0")
            if pe_H0 is not None and sel_H0 is not None:
                _check("cosmo_H0_consistent", abs(pe_H0 - sel_H0) < 1.0,
                       f"PE H0={pe_H0}, sel H0={sel_H0}")
            pe_Om = fg.attrs.get("pe_cosmology_Om0")
            sel_Om = fs.attrs.get("cosmology_Om0")
            if pe_Om is not None and sel_Om is not None:
                _check("cosmo_Om0_consistent", abs(pe_Om - sel_Om) < 0.05,
                       f"PE Om0={pe_Om}, sel Om0={sel_Om}")

    n_pass = sum(results.values())
    n_total = len(results)
    status = "ALL PASSED" if n_pass == n_total else f"{n_total - n_pass} FAILED"
    print(f"  {n_pass}/{n_total} checks: {status}")
    return results