"""gwcat-2.0 export validator (PR 7).

:func:`validate_export_v2` is the v2 counterpart of the frozen v1 validator
:func:`gwcat.catalog.validate_export`.  It mirrors the v1 report structure and
style exactly:

  * a ``{check_name: passed_bool}`` dict is returned;
  * internal-consistency checks respect ``strict`` (raise on the first failure
    when ``strict=True``, otherwise print ``FAIL: ...`` and record ``False``);
  * cross-file *contract* checks ALWAYS raise ``ValueError`` on mismatch,
    independent of ``strict`` -- a file that *looks* valid while carrying the
    wrong spin basis / cosmology / source class is exactly what must be stopped.

It validates ``format_version="gwcat-pe-2.0"`` PE exports (built by
:func:`gwcat.export.pe_builder.build_pe_product`) and, optionally, a paired
``format_version="gwcat-selection-2.0"`` selection export (built by
:func:`gwcat.export.selection_builder.build_selection_product`), across the three
spin bases ``chieff`` / ``component`` / ``chieff_chip``.
"""
from __future__ import annotations

import warnings

import numpy as np
import h5py

#: Spin bases the v2 pipeline implements.
_KNOWN_BASES = ("chieff", "component", "chieff_chip")

#: The legacy 10 datasets every PE export writes (any basis).
_PE_LEGACY = ["ra", "dec", "m1det", "m2det", "chieff", "dL", "p_pe",
              "redshift", "m1src", "m2src"]

#: Datasets the component PE basis adds on top of the legacy 10 (always emitted
#: for a component export, since a_1/a_2 + a tilt are required per event).
_PE_COMPONENT_EXTRA = ["a1", "a2", "cost1", "cost2", "chip"]

#: The legacy 10 datasets every selection export writes (any basis).
_SEL_LEGACY = ["m1det", "m2det", "dL", "chieff", "ra", "dec",
               "m1src", "m2src", "redshift", "pdraw"]

#: Optional per-injection spin columns a selection export may add.
_SPIN_COLUMNS = ["a1", "a2", "cost1", "cost2", "chip"]

#: Small multiplicative slack for magnitude upper bounds (float round-off).
_SLACK = 1.001
_TOL = 1e-9


def _read_export(path, want_datasets):
    """Read an export file's attrs (decoded) + the requested datasets.

    Returns ``(attrs, present, cols)`` where ``attrs`` is a plain dict of
    HDF5 attributes (bytes decoded to ``str``), ``present`` is the set of
    dataset names in the file, and ``cols`` maps each requested-and-present
    dataset name to its numpy array.
    """
    with h5py.File(path, "r") as f:
        attrs = {}
        for k in f.attrs:
            v = f.attrs[k]
            attrs[k] = v.decode() if isinstance(v, (bytes, bytearray)) else v
        present = set(f.keys())
        cols = {k: np.asarray(f[k]) for k in want_datasets if k in present}
    return attrs, present, cols


def _amax_bound(*arrays, default=1.0):
    """Largest finite spin ``amax`` across the given attr arrays (or ``default``)."""
    vals = []
    for a in arrays:
        arr = np.asarray(a, dtype=float).ravel()
        finite = arr[np.isfinite(arr)]
        if finite.size:
            vals.append(float(finite.max()))
    return max(vals) if vals else float(default)


def validate_export_v2(pe_path, selection_path=None, strict=False):
    """Validate a gwcat-2.0 PE export (and optionally a paired selection export).

    Internal-consistency checks
    ---------------------------
    PE file (``gwcat-pe-2.0``):
      * ``format_version == "gwcat-pe-2.0"`` and ``spin_basis`` valid;
      * required datasets (the legacy 10 always; ``component`` additionally
        writes ``a1/a2/cost1/cost2/chip``; ``chieff_chip`` additionally writes
        ``chip``);
      * ``p_pe`` finite and **strictly positive** -- an exact zero is a hard
        failure (GW-01), reported with a count.  It used to be documented as
        legal at "distance-prior tails", but that state only arose from the
        truncated distance prior, and darksirens turns a zero weight into a
        ``-inf`` log-weight that still counts in ``n``;
      * ``nobs * nsamp`` length consistency across the legacy datasets;
      * physical ranges when spin columns are present: ``a1``/``a2`` in
        ``[0, max(spin_amax)*1.001]``, ``|cost{1,2}| <= 1``, ``chip`` in
        ``[0, max(spin_amax)*1.001]``, ``chieff`` in ``[-1, 1]``.

    Selection file (``gwcat-selection-2.0``):
      * ``format_version == "gwcat-selection-2.0"`` and ``spin_basis`` valid;
      * ``pdraw`` finite and positive, ``pdraw`` length ``== n_detected``,
        ``ndraw > n_detected``;
      * the per-campaign injected-spin provenance attrs
        (``injected_spin_format`` / ``injected_spin_amax_detected`` /
        ``injected_spin_uniform_isotropic``) are present;
      * physical ranges as above when spin columns are present.

    Cross-file contract checks (ALWAYS raise on mismatch)
    -----------------------------------------------------
      * PE ``spin_basis`` must equal selection ``spin_basis`` (the error names
        both).
      * Basis-specific spin-amax handling (see below).
      * ``component`` basis: PE ``component_spin_prior_applied_to_p_pe`` must be
        ``True`` and the selection ``pdraw_state`` must equal the component
        constant.
      * Cosmology: the selection cosmology must match the PE cosmology (reusing
        the v1 tolerances ``|dH0| >= 1.0`` / ``|dOm0| >= 0.05``; per-event PE
        cosmologies are each compared against the single selection cosmology).
      * Source class: the PE and selection ``source_class_filter`` must resolve
        to the same canonical class set.

    The chieff_chip amax cross-check -- why the two amax are NOT required to match
    -----------------------------------------------------------------------------
    In the ``chieff_chip`` basis the PE side and the selection side each divide
    out a joint ``(chi_eff, chi_p)`` prior, but they are dividing out *different*
    densities and therefore legitimately use *different* ``amax`` values:

      * The **PE** side reweights ``p_pe`` by the analytic joint prior evaluated
        at the PE prior's spin ceiling ``chi_eff_chi_p_amax_per_event`` -- the
        ``amax`` at which the *posterior samples' own* spin prior was defined
        (typically ~0.99).  Its job is to remove the PE spin prior that was in
        force when the posterior was sampled.
      * The **selection** side reweights ``pdraw`` by the joint prior evaluated
        at each campaign's *detected injected* ceiling
        ``injected_spin_amax_detected`` -- the ``amax`` of the isotropic spin
        draw the injection campaign actually used (e.g. 0.998 for an O3 endo3
        campaign).  Its job is to swap the injected draw density.

    Requiring these two ``amax`` to be equal would be WRONG: it would force the
    injection campaign's spin ceiling to match the PE prior's ceiling, which need
    not hold.  The hierarchical likelihood is consistent by construction as long
    as *each side uses its own correct amax*.  This check therefore verifies only
    that each side used a finite, well-defined ceiling (PE: the per-event array
    is finite; selection: the detected amax is finite), RECORDS both sets of
    values (a printed NOTE), and emits a ``UserWarning`` if they differ so an
    analyst who expected equality understands why they should not.  It never
    fails on inequality.

    Parameters
    ----------
    pe_path : str or path-like
        Path to a ``gwcat-pe-2.0`` PE export.
    selection_path : str or path-like, optional
        Path to a paired ``gwcat-selection-2.0`` selection export.
    strict : bool, default False
        Raise on the first internal-consistency failure.  Cross-file contract
        checks always raise on mismatch regardless of this flag.

    Returns
    -------
    dict
        ``{check_name: passed_bool}``.
    """
    results = {}

    def _check(name, cond, msg=""):
        results[name] = bool(cond)
        if not cond and strict:
            raise AssertionError(f"validate_export_v2 FAILED: {name}. {msg}")
        if not cond:
            print(f"  FAIL: {name}  {msg}")
        return bool(cond)

    def _fail(name, msg):
        results[name] = False
        raise ValueError(f"validate_export_v2 FAILED: {name}. {msg}")

    # ── PE file (internal) ──────────────────────────────────────────────────
    print(f"Validating gwcat-2.0 PE export: {pe_path}")
    pe_attrs, pe_present, pe_cols = _read_export(
        pe_path, _PE_LEGACY + _PE_COMPONENT_EXTRA)

    _check("pe_format_version",
           pe_attrs.get("format_version") == "gwcat-pe-2.0",
           f"format_version={pe_attrs.get('format_version')!r} "
           f"(expected 'gwcat-pe-2.0')")
    pe_basis = pe_attrs.get("spin_basis")
    _check("pe_spin_basis_valid", pe_basis in _KNOWN_BASES,
           f"spin_basis={pe_basis!r} not in {_KNOWN_BASES}")

    # Required datasets per basis.
    pe_required = list(_PE_LEGACY)
    if pe_basis == "component":
        pe_required += _PE_COMPONENT_EXTRA
    elif pe_basis == "chieff_chip":
        pe_required += ["chip"]
    for ds in pe_required:
        _check(f"pe_has_{ds}", ds in pe_present, f"dataset {ds!r} missing")

    # nobs * nsamp length consistency.
    nobs = int(pe_attrs.get("nobs", 0))
    nsamp = int(pe_attrs.get("nsamp", 0))
    expected = nobs * nsamp
    for ds in _PE_LEGACY:
        if ds in pe_cols:
            _check(f"pe_{ds}_length", pe_cols[ds].shape[0] == expected,
                   f"{pe_cols[ds].shape[0]} != nobs*nsamp = {expected}")

    # p_pe finite & STRICTLY positive.  An exact zero used to be documented as
    # legal ("distance-prior tails"), but that state only ever arose from the
    # truncated distance prior GW-01 removed, and darksirens turns a zero weight
    # into a -inf log-weight that still counts in n for the per-event MC
    # variance.  It is now a hard failure, reported with a count.
    if "p_pe" in pe_cols and pe_cols["p_pe"].size:
        p = pe_cols["p_pe"]
        _check("pe_p_pe_finite", np.all(np.isfinite(p)))
        n_zero = int(np.sum(p == 0.0))
        n_neg = int(np.sum(p < 0.0))
        _check("pe_p_pe_positive", n_zero == 0 and n_neg == 0,
               f"{n_zero} zero ({100 * n_zero / p.size:.2f}%) and {n_neg} "
               f"negative of {p.size} samples; min={np.nanmin(p):.3e}. A zero "
               f"p_pe means the store's p_dL_pe was truncated at the recorded "
               f"distance-prior bounds -- re-ingest so the distance prior is "
               f"evaluated over the full sample range.")

    # Physical ranges.
    pe_amax = _amax_bound(pe_attrs.get("spin_amax_1_per_event", []),
                          pe_attrs.get("spin_amax_2_per_event", []),
                          pe_attrs.get("chi_eff_chi_p_amax_per_event", []),
                          pe_attrs.get("chi_eff_amax", []))
    _range_checks(_check, pe_cols, pe_amax, prefix="pe")
    _sky_checks(_check, pe_cols, prefix="pe")

    # ── Selection file (internal) + cross-checks ────────────────────────────
    if selection_path is not None:
        print(f"Validating gwcat-2.0 selection export: {selection_path}")
        sel_attrs, sel_present, sel_cols = _read_export(
            selection_path, _SEL_LEGACY + _SPIN_COLUMNS)

        _check("sel_format_version",
               sel_attrs.get("format_version") == "gwcat-selection-2.0",
               f"format_version={sel_attrs.get('format_version')!r} "
               f"(expected 'gwcat-selection-2.0')")
        sel_basis = sel_attrs.get("spin_basis")
        _check("sel_spin_basis_valid", sel_basis in _KNOWN_BASES,
               f"spin_basis={sel_basis!r} not in {_KNOWN_BASES}")

        ndraw = int(sel_attrs.get("ndraw", 0))
        n_det = int(sel_attrs.get("n_detected", 0))
        _check("sel_ndraw_gt_ndet", ndraw > n_det,
               f"ndraw={ndraw} <= n_detected={n_det}")

        if "pdraw" in sel_cols:
            pd = sel_cols["pdraw"]
            _check("sel_pdraw_length", pd.shape[0] == n_det,
                   f"{pd.shape[0]} != n_detected = {n_det}")
            if pd.size:
                _check("sel_pdraw_finite", np.all(np.isfinite(pd)))
                _check("sel_pdraw_positive", np.all(pd > 0),
                       f"min={pd.min():.3e}")
            else:
                _check("sel_pdraw_nonempty", False, "pdraw is empty")

        # Per-campaign injected-spin provenance attrs must be present.
        for a in ("injected_spin_format", "injected_spin_amax_detected",
                  "injected_spin_uniform_isotropic"):
            _check(f"sel_has_{a}", a in sel_attrs, f"attr {a!r} missing")

        # Magnitude bound for the range checks.  A campaign's detected amax
        # only bounds its injections when that campaign's draw really is
        # uniform-isotropic; a non-uniform campaign (e.g. the O4 sets, where
        # p(a) is not flat) has no meaningful detected amax and its spins can
        # extend to the physical limit 1.  So use max(detected amax) only when
        # EVERY campaign is uniform-isotropic with finite detected amax;
        # otherwise fall back to the physical bound 1.0.
        iso = np.asarray(
            sel_attrs.get("injected_spin_uniform_isotropic", [])).ravel()
        amax_arr = np.asarray(
            sel_attrs.get("injected_spin_amax_detected", []),
            dtype=float).ravel()
        all_iso = iso.size > 0 and bool(np.all(iso.astype(bool)))
        all_finite = amax_arr.size > 0 and bool(np.all(np.isfinite(amax_arr)))
        if all_iso and all_finite:
            sel_amax = _amax_bound(amax_arr,
                                   sel_attrs.get("chi_eff_amax", []))
        else:
            sel_amax = 1.0
        _range_checks(_check, sel_cols, sel_amax, prefix="sel")
        # A campaign that genuinely drew no sky position exempts its NaN rows,
        # but only if the file SAYS so -- the flag exists to be read (GW-20).
        _sky_avail = sel_attrs.get("sky_position_available")
        _sky_all = (None if _sky_avail is None
                    else bool(np.all(np.asarray(_sky_avail, dtype=bool))))
        _sky_checks(_check, sel_cols, prefix="sel",
                    sky_available=_sky_all)

        # ── Cross-file contract checks (ALWAYS raise on mismatch) ────────────
        # (a) spin_basis must match -- name BOTH sides.
        if pe_basis != sel_basis:
            _fail("xcheck_spin_basis",
                  f"PE spin_basis={pe_basis!r} but selection "
                  f"spin_basis={sel_basis!r}. The PE export and its selection "
                  f"function must share one spin basis; otherwise the priors "
                  f"divided out on each side are inconsistent.")
        results["xcheck_spin_basis"] = True

        # (b) Basis-specific spin-amax handling.
        if pe_basis == "chieff":
            pe_a = pe_attrs.get("chi_eff_amax")
            sel_a = sel_attrs.get("chi_eff_amax")
            if pe_a is not None and sel_a is not None:
                if abs(float(pe_a) - float(sel_a)) > 1e-6:
                    _fail("xcheck_chieff_amax",
                          f"PE chi_eff_amax={pe_a} but selection "
                          f"chi_eff_amax={sel_a} (|Δ| > 1e-6). Both sides swap "
                          f"the same 1-D chi_eff prior and must use one amax.")
                results["xcheck_chieff_amax"] = True

        elif pe_basis == "component":
            _check("xcheck_component_pe_flag",
                   bool(pe_attrs.get("component_spin_prior_applied_to_p_pe",
                                     False)),
                   "PE component_spin_prior_applied_to_p_pe is not True")
            from ..selection import PDRAW_STATE_COMPONENT
            if sel_attrs.get("pdraw_state") != PDRAW_STATE_COMPONENT:
                _fail("xcheck_component_pdraw_state",
                      f"selection pdraw_state={sel_attrs.get('pdraw_state')!r} "
                      f"does not match the component-basis constant. The "
                      f"selection file was not written with the component spin "
                      f"draw retained.")
            results["xcheck_component_pdraw_state"] = True

        elif pe_basis == "chieff_chip":
            # Record both amax; verify each side's OWN ceiling is finite; WARN
            # (never fail) if they differ -- see this function's docstring for
            # why PE-prior amax and injected-detected amax legitimately differ.
            pe_amax_arr = np.asarray(
                pe_attrs.get("chi_eff_chi_p_amax_per_event", []), dtype=float)
            sel_amax_arr = np.asarray(
                sel_attrs.get("injected_spin_amax_detected", []), dtype=float)
            _check("xcheck_chieff_chip_pe_amax_finite",
                   pe_amax_arr.size > 0 and np.all(np.isfinite(pe_amax_arr)),
                   "PE chi_eff_chi_p_amax_per_event is empty or non-finite")
            _check("xcheck_chieff_chip_sel_amax_finite",
                   sel_amax_arr.size > 0 and np.all(np.isfinite(sel_amax_arr)),
                   "selection injected_spin_amax_detected is empty or non-finite")
            pe_u = sorted({round(float(x), 6) for x in pe_amax_arr.ravel()
                           if np.isfinite(x)})
            sel_u = sorted({round(float(x), 6) for x in sel_amax_arr.ravel()
                            if np.isfinite(x)})
            print(f"  NOTE: chieff_chip amax -- PE prior amax {pe_u} vs "
                  f"selection detected amax {sel_u}. These are DIFFERENT "
                  f"quantities (PE prior ceiling vs injected-draw ceiling) and "
                  f"are NOT required to match.")
            if set(pe_u) != set(sel_u):
                warnings.warn(
                    f"chieff_chip: PE prior amax {pe_u} != selection detected "
                    f"amax {sel_u}. This is expected and consistent -- each side "
                    f"divides out its own prior at its own amax; do not 'fix' it "
                    f"by forcing the two to match.")
            results["xcheck_chieff_chip_amax_recorded"] = True

        # (c) Cosmology agreement (reuse the v1 tolerances / per-event logic).
        sel_H0 = sel_attrs.get("cosmology_H0")
        sel_Om = sel_attrs.get("cosmology_Om0")
        if sel_H0 is not None and sel_Om is not None:
            varies = bool(pe_attrs.get("cosmology_per_event_varies", False))
            if varies:
                pe_H0_arr = np.asarray(
                    pe_attrs.get("cosmology_H0_per_event", []), dtype=float)
                pe_Om_arr = np.asarray(
                    pe_attrs.get("cosmology_Om0_per_event", []), dtype=float)
                bad = ((np.abs(pe_H0_arr - float(sel_H0)) >= 1.0).any()
                       or (np.abs(pe_Om_arr - float(sel_Om)) >= 0.05).any())
                if bad:
                    h0rng = ((pe_H0_arr.min(), pe_H0_arr.max())
                             if pe_H0_arr.size else ("?", "?"))
                    omrng = ((pe_Om_arr.min(), pe_Om_arr.max())
                             if pe_Om_arr.size else ("?", "?"))
                    _fail("xcheck_cosmology",
                          f"PE export uses per-event cosmologies with H0 in "
                          f"{h0rng} and Om0 in {omrng} that do not all match the "
                          f"single selection cosmology (H0={sel_H0}, "
                          f"Om0={sel_Om}). Re-export the selection under a "
                          f"matching cosmology, or use a single-cosmology PE "
                          f"export.")
                results["xcheck_cosmology"] = True
            else:
                pe_H0 = pe_attrs.get("pe_cosmology_H0")
                pe_Om = pe_attrs.get("pe_cosmology_Om0")
                if pe_H0 is not None and abs(float(pe_H0) - float(sel_H0)) >= 1.0:
                    _fail("xcheck_cosmology",
                          f"PE cosmology H0={pe_H0} disagrees with selection "
                          f"H0={sel_H0} (|Δ| >= 1.0).")
                if pe_Om is not None and abs(float(pe_Om) - float(sel_Om)) >= 0.05:
                    _fail("xcheck_cosmology",
                          f"PE cosmology Om0={pe_Om} disagrees with selection "
                          f"Om0={sel_Om} (|Δ| >= 0.05).")
                results["xcheck_cosmology"] = True

        # (d) Source-class compatibility.
        from ..source_class import resolve_filter_classes, SOURCE_CLASSES

        def _sc_classes(raw):
            s = "" if raw is None else str(raw)
            if s == "":
                return set(SOURCE_CLASSES)
            return set(resolve_filter_classes(s))

        pe_scf = pe_attrs.get("source_class_filter", "")
        sel_scf = sel_attrs.get("source_class_filter", "")
        pe_classes = _sc_classes(pe_scf)
        sel_classes = _sc_classes(sel_scf)
        if pe_classes != sel_classes:
            _fail("xcheck_source_class",
                  f"PE source_class_filter={pe_scf!r} -> {sorted(pe_classes)} "
                  f"but selection source_class_filter={sel_scf!r} -> "
                  f"{sorted(sel_classes)}. The selection injections must cover "
                  f"the same source class(es) as the PE events.")
        results["xcheck_source_class"] = True

    n_pass = sum(results.values())
    n_total = len(results)
    status = "ALL PASSED" if n_pass == n_total else f"{n_total - n_pass} FAILED"
    print(f"  {n_pass}/{n_total} checks: {status}")
    return results


def _sky_checks(_check, cols, prefix, sky_available=None):
    """Finiteness AND range checks on ra/dec (GW-20).

    Neither existed.  ``_range_checks`` inspected only the spin columns, and the
    validator's only finiteness check was on ``p_pe``/``pdraw`` -- ra/dec were
    read into the column dict and never examined.  Two distinct defects that
    both reach ``hp.ang2pix`` in the consumer:

    * **NaN sky.**  The loader NaN-fills the semianalytic O1/O2 rows of
      cumulative-mixture files and records ``sky_position_available=False`` per
      campaign, the exporters write those NaNs, and nothing read the flag.
    * **Wrong units or convention.**  A degrees ingest, or colatitude in place of
      declination, produces a plausible-looking but wrong pixelisation with no
      symptom anywhere.  The ranges are what make that catchable: ``dec`` must be
      in ``[-pi/2, pi/2]``, so 45.0 (degrees) or 2.0 (colatitude) both fail.

    ``sky_available``, when given, is the per-campaign availability flag: rows
    from a campaign that genuinely drew no sky position are exempt from the
    finiteness check but still counted and reported, because dropping them
    silently while keeping ``ndraw`` would bias the selection integral.
    """
    for name, lo, hi_, closed in (("ra", 0.0, 2.0 * np.pi, "left"),
                                  ("dec", -np.pi / 2.0, np.pi / 2.0, "both")):
        if name not in cols or not cols[name].size:
            continue
        v = np.asarray(cols[name], dtype=float)
        finite = np.isfinite(v)
        n_nan = int((~finite).sum())
        exempt = bool(sky_available is False)
        _check(f"{prefix}_{name}_finite", n_nan == 0 or exempt,
               f"{n_nan} of {v.size} {name} values are non-finite and the "
               f"campaign does not declare sky_position_available=False; they "
               f"would reach hp.ang2pix as NaN")
        if n_nan and exempt:
            print(f"  NOTE: {n_nan} {name} values are NaN, declared via "
                  f"sky_position_available=False")
        if not finite.any():
            continue
        f_ = v[finite]
        in_hi = f_ < hi_ if closed == "left" else f_ <= hi_ + _TOL
        ok = bool(np.all(f_ >= lo - _TOL) and np.all(in_hi))
        _check(f"{prefix}_{name}_range", ok,
               f"{name} outside [{lo:.6g}, {hi_:.6g}] radians "
               f"(min={f_.min():.6g}, max={f_.max():.6g}) -- degrees, "
               f"colatitude or a sign convention?")


def _range_checks(_check, cols, amax, prefix):
    """Physical-range checks shared by the PE and selection internal blocks.

    ``amax`` bounds ``a1``/``a2``/``chip``; ``|cost{1,2}| <= 1``;
    ``chieff`` in ``[-1, 1]``.  Sky is handled by :func:`_sky_checks`.
    """
    hi = amax * _SLACK
    if "chieff" in cols and cols["chieff"].size:
        c = cols["chieff"]
        _check(f"{prefix}_chieff_range",
               np.all(c >= -1 - _TOL) and np.all(c <= 1 + _TOL),
               f"chieff out of [-1, 1] (min={c.min():.3f}, max={c.max():.3f})")
    for name in ("a1", "a2"):
        if name in cols and cols[name].size:
            a = cols[name]
            _check(f"{prefix}_{name}_range",
                   np.all(a >= -_TOL) and np.all(a <= hi),
                   f"{name} out of [0, {hi:.3f}] "
                   f"(min={a.min():.3f}, max={a.max():.3f})")
    for name in ("cost1", "cost2"):
        if name in cols and cols[name].size:
            ct = cols[name]
            _check(f"{prefix}_{name}_range", np.all(np.abs(ct) <= 1 + _TOL),
                   f"|{name}| > 1 (max abs={np.abs(ct).max():.3f})")
    if "chip" in cols and cols["chip"].size:
        cp = cols["chip"]
        _check(f"{prefix}_chip_range",
               np.all(cp >= -_TOL) and np.all(cp <= hi),
               f"chip out of [0, {hi:.3f}] "
               f"(min={cp.min():.3f}, max={cp.max():.3f})")
