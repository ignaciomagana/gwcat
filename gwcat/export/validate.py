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

:func:`validate_export_any` is the ONE public format dispatcher over both
generations -- it is what ``gwcat.validate_export`` and ``gwcat validate``
call, so a v2 file can never be handed to the v1 contract (or the reverse).
"""
from __future__ import annotations

import warnings

import numpy as np
import h5py

#: Spin bases each side of the v2 pipeline implements -- read from the
#: builders themselves, so the validator cannot reject a file the builder
#: legitimately wrote (nospin PE files failed here) or accept one it could
#: never have produced.
from .pe_builder import SUPPORTED_SPIN_BASES as _PE_BASES
from .selection_builder import SUPPORTED_SPIN_BASES as _SEL_BASES

#: Format versions this validator understands, per side.  2.1 adds the contract
#: attrs on top of 2.0 and is otherwise identical, so both run the same checks;
#: the 2.1-only pairing-hash comparison is keyed off the attr, not the version.
_PE_FORMATS = ("gwcat-pe-2.0", "gwcat-pe-2.1")
_SEL_FORMATS = ("gwcat-selection-2.0", "gwcat-selection-2.1")

#: Format versions the frozen v1 validator (:func:`gwcat.catalog.validate_export`)
#: owns.  Lives here, next to the v2 strings, so ONE module decides which
#: generation validates which file: ``gwcat.cli`` used to keep its own copy of
#: the v2 list (which went stale and silently routed 2.1 files to the v1
#: validator) while the public ``gwcat.validate_export`` dispatched on nothing
#: at all and handed every file to the v1 validator.
_V1_FORMATS = ("gwcat-1.0", "gwcat-selection-1.0")


def _validator_generation(_fail, version, kind):
    """Refuse a format this validator was never taught.

    An unrecognised ``format_version`` must stop here rather than fall through
    to the frozen v1 validator, which would check a v1 contract against a file
    written to some later one and report success.
    """
    known = _PE_FORMATS if kind == "pe" else _SEL_FORMATS
    if version is None:
        _fail(f"{kind}_format_version_present",
              f"no format_version attr; a gwcat {kind} export must declare "
              f"one of {sorted(known)}")
    if str(version) not in known:
        _fail(f"{kind}_format_version_known",
              f"format_version={version!r} is not a format validate_export_v2 "
              f"understands (knows {sorted(known)}). Refusing rather than "
              f"validating it against the wrong contract.")
    return str(version)


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

#: (PE basis, selection basis) pairs that validate WITHOUT being equal.  Only
#: one: the reference basis writes the injection-side density against the very
#: prior a chieff PE export removes, so the two are the two halves of one
#: density -- see :func:`_xcheck_reference_pair`, which then requires the single
#: thing that makes them one (the same ceiling).
_REFERENCE_PAIRS = {("chieff", "chieff_reference")}

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


def _amax_provenance(attrs, names):
    """The per-event / per-campaign ceilings a file records, as a flat array.

    Falls back to the file-level ``chi_eff_amax`` scalar for a file written
    before the per-proposal ceilings existed, so an older pair is checked rather
    than reported as missing provenance.  Empty when neither is present.
    """
    parts = [np.asarray(attrs[n], dtype=float).ravel()
             for n in names if n in attrs]
    if not parts:
        scalar = attrs.get("chi_eff_amax")
        if scalar is None:
            return np.array([], dtype=float)
        return np.asarray([scalar], dtype=float)
    return np.concatenate(parts)


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
        both) -- with the single exception of a ``chieff`` PE file paired with a
        ``chieff_reference`` selection file, which are the two halves of one
        density and are then held to the STRICTER rule that their ceilings match
        exactly (:func:`_xcheck_reference_pair`).  Note that the 2.1 pairing
        hash covers ``parameter_space`` verbatim and so does not yet recognise
        that pair; export a reference pair at 2.0.
      * Basis-specific spin-amax handling (see below).
      * ``component`` basis: PE ``component_spin_prior_applied_to_p_pe`` must be
        ``True`` and the selection ``pdraw_state`` must equal the component
        constant.
      * Cosmology: the selection cosmology must match the PE cosmology (reusing
        the v1 tolerances ``|dH0| >= 1.0`` / ``|dOm0| >= 0.05``; per-event PE
        cosmologies are each compared against the single selection cosmology).
      * Source class: the PE and selection ``source_class_filter`` must resolve
        to the same canonical class set, and -- when both sides really do carry
        a class filter -- their ``source_class_cut_estimator`` must agree.  A
        median-based event cut paired with a truth-based injection cut is
        refused: same threshold, different quantity, biased at the boundary
        (GW-12).

    The projection amax cross-check -- why the two amax are NOT required to match
    -----------------------------------------------------------------------------
    This applies to BOTH projection bases (``chieff`` since GW-31, ``chieff_chip``
    all along).  The PE side and the selection side each divide out a spin prior,
    but they are dividing out *different* densities and therefore legitimately
    use *different* ``amax`` values:

      * The **PE** side reweights ``p_pe`` by the analytic prior evaluated at the
        PE prior's spin ceiling (``chi_eff_amax_1/2_per_event`` for ``chieff``,
        ``chi_eff_chi_p_amax_per_event`` for ``chieff_chip``) -- the ``amax`` at
        which the *posterior samples' own* spin prior was defined (typically
        ~0.99).  Its job is to remove the PE spin prior that was in force when
        the posterior was sampled.
      * The **selection** side reweights ``pdraw`` by the same prior evaluated at
        each campaign's *detected injected* ceiling
        (``chi_eff_amax_per_campaign`` / ``injected_spin_amax_detected``) -- the
        ``amax`` of the isotropic spin draw the injection campaign actually used
        (0.998 for the O3 endo3 campaign).  Its job is to swap the injected draw
        density.

    Requiring these two ``amax`` to be equal would be WRONG: it would force the
    injection campaign's spin ceiling to match the PE prior's ceiling, which need
    not hold -- and the ``chieff`` branch used to FAIL on exactly that, which is
    only survivable while both sides are wrong in the same way (GW-31).  The
    hierarchical likelihood is consistent by construction as long as *each side
    uses its own correct amax*.  This check therefore verifies only that each
    side used a finite, well-defined ceiling, RECORDS both sets of values (a
    printed NOTE), and emits a ``UserWarning`` if they differ so an analyst who
    expected equality understands why they should not.  It never fails on
    inequality.

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

    _validator_generation(_fail, pe_attrs.get("format_version"), "pe")
    _check("pe_format_version",
           pe_attrs.get("format_version") in _PE_FORMATS,
           f"format_version={pe_attrs.get('format_version')!r} "
           f"(expected one of {sorted(_PE_FORMATS)})")
    pe_basis = pe_attrs.get("spin_basis")
    _check("pe_spin_basis_valid", pe_basis in _PE_BASES,
           f"spin_basis={pe_basis!r} not in {_PE_BASES}")

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
    else:
        # The `else` the frozen v1 validator and the v2 SELECTION block both
        # have, and only this one lacked (GW-37).  With nobs=0 every length
        # check above passes trivially (expected = 0*nsamp = 0) and every array
        # check is skipped, so an export whose cuts dropped every event
        # validated ALL PASSED and `gwcat validate` exited 0 -- defeating a
        # scripted exit-code gate at exactly the moment it was needed.
        _check("pe_p_pe_nonempty", False,
               f"p_pe is empty (nobs={nobs}). Every event was dropped by the "
               f"export's cuts; the file has no posterior samples at all.")

    # Physical ranges.
    pe_amax = _amax_bound(pe_attrs.get("spin_amax_1_per_event", []),
                          pe_attrs.get("spin_amax_2_per_event", []),
                          pe_attrs.get("chi_eff_chi_p_amax_per_event", []),
                          pe_attrs.get("chi_eff_amax_1_per_event", []),
                          pe_attrs.get("chi_eff_amax_2_per_event", []),
                          pe_attrs.get("chi_eff_amax", []))
    _range_checks(_check, pe_cols, pe_amax, prefix="pe")
    _sky_checks(_check, pe_cols, prefix="pe")
    _fit_columns_present(_check, pe_attrs, pe_present, prefix="pe")

    # ── Selection file (internal) + cross-checks ────────────────────────────
    if selection_path is not None:
        print(f"Validating gwcat-2.0 selection export: {selection_path}")
        sel_attrs, sel_present, sel_cols = _read_export(
            selection_path, _SEL_LEGACY + _SPIN_COLUMNS)

        _validator_generation(_fail, sel_attrs.get("format_version"),
                              "selection")
        _check("sel_format_version",
               sel_attrs.get("format_version") in _SEL_FORMATS,
               f"format_version={sel_attrs.get('format_version')!r} "
               f"(expected one of {sorted(_SEL_FORMATS)})")
        sel_basis = sel_attrs.get("spin_basis")
        _check("sel_spin_basis_valid", sel_basis in _SEL_BASES,
               f"spin_basis={sel_basis!r} not in {_SEL_BASES}")

        ndraw = int(sel_attrs.get("ndraw", 0))
        n_det = int(sel_attrs.get("n_detected", 0))
        _check("sel_ndraw_gt_ndet", ndraw > n_det,
               f"ndraw={ndraw} <= n_detected={n_det}")

        # Required datasets, and every column tied to n_detected (GW-37).  This
        # block had NEITHER -- the PE half above has both, and so does the
        # frozen v1 validator this one replaced, whose own docstring names "the
        # v1 validator checks a required dataset only when it happens to exist"
        # as the defect being fixed.  Without them a selection file with every
        # dataset deleted validated ALL PASSED (the 13 guarded checks vanish
        # rather than fail), and a RAGGED file -- pdraw at length 60 against
        # columns truncated to 3 -- passed too, which is the dangerous one: it
        # never raises, so a consumer reading columns independently pairs
        # injection i's draw density with injection j's masses and distance.
        sel_required = list(_SEL_LEGACY)
        if sel_basis == "component":
            sel_required += _SPIN_COLUMNS
        elif sel_basis == "chieff_reference":
            # a1/a2 are the coordinates of the reference SUPPORT: they are what
            # tells a reader which rows carry the zero-weight sentinel and why.
            # A reference file without them states that some rows are outside
            # the reference and offers no way to see which.
            sel_required += ["a1", "a2"]
        elif sel_basis == "chieff_chip":
            sel_required += ["chip"]
        for ds in sel_required:
            _check(f"sel_has_{ds}", ds in sel_present,
                   f"dataset {ds!r} missing; the {sel_attrs.get('format_version')} "
                   f"schema with spin_basis={sel_basis!r} mandates "
                   f"{sel_required}")
        for ds in sel_required:
            if ds in sel_cols and ds != "pdraw":
                _check(f"sel_{ds}_length", sel_cols[ds].shape[0] == n_det,
                       f"{sel_cols[ds].shape[0]} != n_detected = {n_det}; the "
                       f"columns are ragged, so a consumer reading them "
                       f"independently pairs one injection's draw density with "
                       f"another's parameters.")

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

        # Every name a 2.1 file declares in its OWN fit_columns must exist as a
        # dataset (GW-37).  A generic check beats extending two more hardcoded
        # lists: the selection file declared `q` and shipped no `q`, and nothing
        # compared the declaration against the file, so 70/70 checks passed on a
        # product whose contract described coordinates it did not contain.
        _fit_columns_present(_check, sel_attrs, sel_present, prefix="sel")

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
                                   sel_attrs.get("chi_eff_amax_per_campaign",
                                                 []),
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
        #
        # One pair is legitimate without being equal, and it is a pair about the
        # SAME density rather than a loosening: a chieff_reference selection file
        # is expressed against a declared isotropic uniform-magnitude reference
        # prior, and a chieff PE file at that same ceiling divides out exactly
        # that prior.  The two halves therefore fit -- provided the ceilings
        # match, which (b) below then REQUIRES rather than merely records.  That
        # requirement is the mirror image of the chieff/chieff rule, where the
        # two amax legitimately differ because each side removes its own prior.
        reference_pair = (pe_basis, sel_basis) in _REFERENCE_PAIRS
        if pe_basis != sel_basis and not reference_pair:
            _fail("xcheck_spin_basis",
                  f"PE spin_basis={pe_basis!r} but selection "
                  f"spin_basis={sel_basis!r}. The PE export and its selection "
                  f"function must share one spin basis; otherwise the priors "
                  f"divided out on each side are inconsistent. The only "
                  f"unequal pair that validates is "
                  f"{sorted(_REFERENCE_PAIRS)}.")
        results["xcheck_spin_basis"] = True

        # (b) Basis-specific spin-amax handling.
        if reference_pair:
            _xcheck_reference_pair(_check, _fail, results, pe_attrs, sel_attrs)
        elif pe_basis == "chieff":
            # Each side's ceiling is checked against ITS OWN provenance, and the
            # two are RECORDED rather than required to match -- see the
            # docstring section on why equality is the wrong contract (GW-31).
            pe_amax_arr = _amax_provenance(
                pe_attrs, ("chi_eff_amax_1_per_event",
                           "chi_eff_amax_2_per_event"))
            sel_amax_arr = _amax_provenance(
                sel_attrs, ("chi_eff_amax_per_campaign",))
            _check("xcheck_chieff_pe_amax_finite",
                   pe_amax_arr.size > 0 and np.all(np.isfinite(pe_amax_arr)),
                   "PE chi_eff ceiling (chi_eff_amax_1/2_per_event, or the "
                   "chi_eff_amax scalar on an older file) is empty or "
                   "non-finite")
            _check("xcheck_chieff_sel_amax_finite",
                   sel_amax_arr.size > 0 and np.all(np.isfinite(sel_amax_arr)),
                   "selection chi_eff ceiling (chi_eff_amax_per_campaign, or "
                   "the chi_eff_amax scalar on an older file) is empty or "
                   "non-finite")
            # Each side's ceiling against ITS OWN provenance: a campaign whose
            # ceiling says "detected" must carry the detected value, and a
            # fabricated ("fallback") ceiling on either side is called out.
            sel_src = [str(x) for x in np.atleast_1d(
                sel_attrs.get("chi_eff_amax_source_per_campaign", []))]
            det = np.asarray(sel_attrs.get("injected_spin_amax_detected", []),
                             dtype=float).reshape(-1, 2)
            used = np.asarray(sel_attrs.get("chi_eff_amax_per_campaign", []),
                              dtype=float).reshape(-1, 2)
            if sel_src and used.shape[0] == det.shape[0] == len(sel_src):
                rows = [k for k, s in enumerate(sel_src) if s == "detected"]
                ok = all(np.allclose(used[k], det[k], rtol=1e-9, atol=1e-12)
                         for k in rows)
                _check("xcheck_chieff_sel_amax_matches_detected", ok,
                       f"a campaign records chi_eff_amax_source='detected' but "
                       f"its chi_eff_amax_per_campaign {used.tolist()} is not "
                       f"its injected_spin_amax_detected {det.tolist()}")
            for side, srcs in (("PE", [str(x) for x in np.atleast_1d(
                    pe_attrs.get("chi_eff_amax_source_per_event", []))]),
                    ("selection", sel_src)):
                n_fab = sum(1 for s in srcs if s == "fallback")
                if n_fab:
                    warnings.warn(
                        f"chieff: {n_fab} {side} ceiling(s) were NOT resolved "
                        f"from the proposal's own prior (source='fallback'), so "
                        f"the chi_eff density divided out there rests on a "
                        f"fabricated amax. It is recorded, not verified.")

            pe_u = sorted({round(float(x), 6) for x in pe_amax_arr.ravel()
                           if np.isfinite(x)})
            sel_u = sorted({round(float(x), 6) for x in sel_amax_arr.ravel()
                            if np.isfinite(x)})
            print(f"  NOTE: chieff amax -- PE prior amax {pe_u} vs selection "
                  f"injected amax {sel_u}. These are DIFFERENT quantities (the "
                  f"PE sampling prior's ceiling vs the injected draw's) and are "
                  f"NOT required to match.")
            if set(pe_u) != set(sel_u):
                warnings.warn(
                    f"chieff: PE prior amax {pe_u} != selection injected amax "
                    f"{sel_u}. This is expected and consistent -- each side "
                    f"divides out its own prior at its own amax (end-O3 injects "
                    f"a ~ U(0, 0.998) while the GWTC PE priors declare 0.99); "
                    f"do not 'fix' it by forcing the two to match.")
            results["xcheck_chieff_amax_recorded"] = True

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
        #
        # Both sides now write the canonical comma-joined form, so this compares
        # the classes actually requested.  It used to compare
        # ``str(source_class)`` reprs -- "['nsbh', 'bns']" from the Python API
        # against "nsbh,bns" from the CLI -- and BOTH resolved to {"Unknown"},
        # so the check passed on a genuinely mismatched pair.
        from ..source_class import parse_source_class_filter, SOURCE_CLASSES

        def _sc_classes(raw):
            parsed = parse_source_class_filter(raw)
            # An empty filter means "no restriction", i.e. every class.
            return set(parsed) if parsed else set(SOURCE_CLASSES)

        pe_scf = pe_attrs.get("source_class_filter", "")
        sel_scf = sel_attrs.get("source_class_filter", "")
        try:
            pe_classes = _sc_classes(pe_scf)
            sel_classes = _sc_classes(sel_scf)
        except ValueError as exc:
            _fail("xcheck_source_class",
                  f"unparseable source_class_filter (PE={pe_scf!r}, "
                  f"selection={sel_scf!r}): {exc} Files written before the "
                  f"canonical form record a Python repr that was never "
                  f"checkable; re-export to make the pairing verifiable.")
        if pe_classes != sel_classes:
            _fail("xcheck_source_class",
                  f"PE source_class_filter={pe_scf!r} -> {sorted(pe_classes)} "
                  f"but selection source_class_filter={sel_scf!r} -> "
                  f"{sorted(sel_classes)}. The selection injections must cover "
                  f"the same source class(es) as the PE events.")
        results["xcheck_source_class"] = True

        # (d2) The class cut's ESTIMATOR: same threshold, same quantity (GW-12).
        _xcheck_source_class_estimator(_fail, results, pe_attrs, sel_attrs)

        # (e) The detection cut: same statistic, same threshold.
        _xcheck_detection_cut(_fail, results, pe_attrs, sel_attrs)

        # (e2) Per-campaign selection cosmology (GW-09).
        _xcheck_campaign_cosmology(_check, results, sel_attrs)

        # (f) 2.1 pairing hash.  Only when BOTH sides carry one -- and never as
        # a substitute for the checks above, which name the offending field.
        pe_hash = pe_attrs.get("contract_hash")
        sel_hash = sel_attrs.get("contract_hash")
        if pe_hash is not None and sel_hash is not None:
            if str(pe_hash) != str(sel_hash):
                from .contract import (contract_diff, format_diff,
                                       PAIRING_FIELDS)
                import json as _json
                try:
                    diff = contract_diff(_json.loads(str(pe_attrs["contract"])),
                                         _json.loads(str(sel_attrs["contract"])),
                                         fields=PAIRING_FIELDS)
                    detail = format_diff(diff)
                except (KeyError, ValueError):
                    detail = ("neither file records a readable 'contract' attr "
                              "to diff")
                _fail("xcheck_contract_hash",
                      f"PE contract_hash={pe_hash} != selection "
                      f"contract_hash={sel_hash}. {detail}")
            results["xcheck_contract_hash"] = True

    n_pass = sum(results.values())
    n_total = len(results)
    status = "ALL PASSED" if n_pass == n_total else f"{n_total - n_pass} FAILED"
    print(f"  {n_pass}/{n_total} checks: {status}")
    return results


# ---------------------------------------------------------------------------
# The one public format dispatcher (re-exported as ``gwcat.validate_export``)
# ---------------------------------------------------------------------------
def read_format_version(path):
    """Return an export file's ``format_version`` attr (decoded), or ``None``."""
    with h5py.File(path, "r") as f:
        v = f.attrs.get("format_version")
    if isinstance(v, (bytes, bytearray)):
        v = v.decode()
    return None if v is None else str(v)


def export_generation(version):
    """Map a ``format_version`` string to ``"v1"`` / ``"v2"`` / ``"unknown"``."""
    if version in _V1_FORMATS:
        return "v1"
    if version in _PE_FORMATS + _SEL_FORMATS:
        return "v2"
    return "unknown"


def validate_export_any(pe_path, selection_path=None, strict=False):
    """Validate an export (and optionally its paired selection export),
    dispatching on each file's declared ``format_version``.

    This is the ONE public entry point, re-exported as
    :func:`gwcat.validate_export` and used by ``gwcat validate``:

      * ``gwcat-1.0`` / ``gwcat-selection-1.0`` -> the frozen
        :func:`gwcat.catalog.validate_export`;
      * ``gwcat-pe-2.x`` / ``gwcat-selection-2.x`` ->
        :func:`validate_export_v2`;
      * a mixed v1/v2 pair, or a format this gwcat does not know, is REFUSED
        with ``ValueError`` rather than checked against the wrong contract.

    Dispatching is the whole point.  ``gwcat.validate_export`` used to be the
    v1 validator itself, and the v1 validator checks a required dataset only
    when it happens to exist -- so a gwcat-2.0 file missing ``p_pe``, the
    masses and the sky came back "ALL PASSED".  Only the CLI dispatched, so the
    library entry point was the one that could bless an unusable file.

    Parameters
    ----------
    pe_path : str or path-like
        PE export to validate.
    selection_path : str or path-like, optional
        Paired selection export.  Must be the same generation as ``pe_path``.
    strict : bool, default False
        Raise on the first internal-consistency failure.  Cross-file contract
        checks always raise on mismatch regardless of this flag.

    Returns
    -------
    dict
        ``{check_name: passed_bool}``, from whichever validator ran.
    """
    pe_version = read_format_version(pe_path)
    pe_gen = export_generation(pe_version)
    sel_version = sel_gen = None
    if selection_path is not None:
        sel_version = read_format_version(selection_path)
        sel_gen = export_generation(sel_version)
        if {pe_gen, sel_gen} == {"v1", "v2"}:
            raise ValueError(
                f"mixed export generations -- PE is {pe_gen} "
                f"({pe_version!r}) but selection is {sel_gen} "
                f"({sel_version!r}). Validate a v1 PE file against a v1 "
                f"selection file (gwcat-1.0 / gwcat-selection-1.0) or a v2 PE "
                f"file against a v2 selection file (gwcat-pe-2.0 / "
                f"gwcat-selection-2.0); the two generations cannot be paired.")

    if "unknown" in {pe_gen, sel_gen} - {None}:
        # Unknown must stop here.  Falling through to the frozen v1 validator
        # checks a v1 contract against a file written to some other one and
        # reports a verdict about a format this gwcat does not know.
        raise ValueError(
            f"unrecognised format_version (PE={pe_version!r}, "
            f"selection={sel_version!r}); this gwcat validates "
            f"{_V1_FORMATS + _PE_FORMATS + _SEL_FORMATS}. Upgrade gwcat or "
            f"check the file.")

    if pe_gen == "v2":
        return validate_export_v2(pe_path, selection_path, strict=strict)
    from ..catalog import validate_export as _validate_export_v1
    return _validate_export_v1(pe_path, selection_path, strict=strict)


def _xcheck_source_class_estimator(_fail, results, pe_attrs, sel_attrs):
    """The two source-class cuts must act on the same quantity (GW-12).

    ``xcheck_source_class`` proves the two sides asked for the same CLASSES;
    it cannot see that they asked different questions to get them.  PE events
    are classified from posterior median source-frame masses and injections
    from injected truth, so a shared threshold is not a shared cut, and the
    selection function that ships describes a cut nobody applied.  See
    :func:`gwcat.source_class.assess_cut_estimator_pair` for the mechanism.

    Absent attrs (a file written before this record existed) warn rather than
    fail, the same convention ``_xcheck_detection_cut`` uses for a cut stated on
    one side only: unknown is not evidence of a mismatch.
    """
    from ..source_class import (parse_source_class_filter, CUT_ESTIMATOR_ATTR,
                                assess_cut_estimator_pair)

    def _filtered(raw):
        try:
            return bool(parse_source_class_filter(raw))
        except ValueError:
            # Unparseable: xcheck_source_class has already failed on it, and a
            # repr nobody can read is not a filter this check can judge.
            return False

    verdict, msg = assess_cut_estimator_pair(
        pe_attrs.get(CUT_ESTIMATOR_ATTR),
        sel_attrs.get(CUT_ESTIMATOR_ATTR),
        _filtered(pe_attrs.get("source_class_filter", "")),
        _filtered(sel_attrs.get("source_class_filter", "")))
    if verdict == "fail":
        _fail("xcheck_source_class_estimator", msg)
    if verdict == "warn":
        warnings.warn(msg)
    results["xcheck_source_class_estimator"] = True


def _xcheck_detection_cut(_fail, results, pe_attrs, sel_attrs):
    """The event cut and the injection detection cut must be the same cut.

    Essick & Fishbach require the statistic and the threshold to agree: beta is
    the detection probability under *the cut the events were selected by*, so a
    selection file built at FAR < 1/yr does not describe an event list cut at
    FAR < 2/yr.  Until GW-11 the thresholds were not exported at all, so this
    could not be checked -- only the qualitative ``far_policy`` was recorded.

    Policy, and why it is not simply "the two numbers must match":

      * **Both sides state a numeric cut on the same statistic and the numbers
        differ** -> FAIL.  Unambiguous, and checkable from the files alone.
      * **One side states a cut and the other does not** -> WARN.  This is the
        real production configuration, not an error: the shipped BBH product
        selects events by *name whitelist* (``far_policy="none"``, no
        ``far_max``) while the injection set applies FAR < 1/yr.  Whether the
        whitelist is equivalent to the injection cut is a claim about the data
        -- for the shipped list it was checked directly, and 0 of 249
        resolvable events violate FAR < 1/yr -- and no attr in either file can
        establish it.  Refusing here would reject a pairing that is correct;
        warning says the equivalence has to be established elsewhere.
      * **The PE side applied a FAR cut and kept events whose FAR was unknown**
        (``allow_missing_far``) -> FAIL.  That one *is* self-contradictory
        within the file: the event list is declaredly FAR-cut and declaredly
        contains events that were never tested against the cut.
    """
    def _num(x):
        if x is None:
            return float("nan")
        try:
            return float(np.asarray(x).ravel()[0])
        except (TypeError, ValueError, IndexError):
            return float("nan")

    def _same(a, b):
        if np.isnan(a) and np.isnan(b):
            return True
        if np.isnan(a) or np.isnan(b):
            return False
        return bool(np.isclose(a, b, rtol=1e-12, atol=0.0))

    pe_far = _num(pe_attrs.get("far_max"))
    sel_far = _num(sel_attrs.get("far_threshold"))
    pe_snr = _num(pe_attrs.get("snr_min"))
    # `significance_snr_threshold` is the name the selection builder ACTUALLY
    # writes (GW-37).  This read used a name nothing in the package ever
    # stamps, so the SNR arm was structurally NaN: a selection file whose
    # injection mask is `far-detected OR snr > t` -- strictly looser than the
    # event cut -- compared equal on both legs and paired clean, biasing mu high
    # and the inferred rate low.  The legacy spelling stays as a fallback.
    sel_snr = _num(sel_attrs.get("significance_snr_threshold",
                                 sel_attrs.get("snr_threshold")))

    mism, unstated = [], []
    for label, pe_v, sel_v, pe_name, sel_name in (
            ("FAR", pe_far, sel_far, "far_max", "far_threshold"),
            ("SNR", pe_snr, sel_snr, "snr_min", "significance_snr_threshold")):
        if _same(pe_v, sel_v):
            continue
        where = (f"PE {pe_name}={pe_v} vs selection {sel_name}={sel_v}")
        if np.isnan(pe_v) or np.isnan(sel_v):
            unstated.append(f"{label}: {where}")
        else:
            mism.append(f"{label}: {where}")
    if mism:
        _fail("xcheck_detection_cut",
              "the event cut and the injection detection cut differ -- "
              + "; ".join(mism)
              + ". beta must be computed under the same statistic at the same "
                "threshold the events were selected by.")
    if unstated:
        warnings.warn(
            "detection cut stated on one side only -- " + "; ".join(unstated)
            + f" (PE far_policy={pe_attrs.get('far_policy', 'none')!r}, "
              f"event_list_filter="
              f"{pe_attrs.get('event_list_filter', '')!r}). This is legal -- a "
              f"name-whitelisted event list can satisfy the injection cut "
              f"exactly -- but the files cannot show it. Verify the "
              f"equivalence directly before quoting a rate.")

    # The redshift truncation must match too (GW-37).  The PE export drops
    # posterior samples above z_max; unless the injections are subset the same
    # way, mu keeps its full-z content and the two sides integrate over
    # different redshift ranges.  NaN on both sides ("no truncation") agrees.
    pe_zmax = _num(pe_attrs.get("z_max"))
    sel_zmax = _num(sel_attrs.get("z_max"))
    if not _same(pe_zmax, sel_zmax):
        msg = (f"PE z_max={pe_zmax} vs selection z_max={sel_zmax}. The PE "
               f"export truncated its posteriors at that redshift; the "
               f"selection product must subset its injections the same way "
               f"(build_selection_product(z_max=...)) or mu integrates over a "
               f"redshift range the events do not cover.")
        if np.isnan(pe_zmax) or np.isnan(sel_zmax):
            # One side predates the record (no z_max attr at all): unknown is
            # not evidence of a mismatch, the same convention the cut legs use.
            if "z_max" in pe_attrs and "z_max" in sel_attrs:
                _fail("xcheck_detection_cut", msg)
            else:
                warnings.warn("redshift truncation stated on one side only -- "
                              + msg)
        else:
            _fail("xcheck_detection_cut", msg)

    if (not np.isnan(pe_far)
            and bool(pe_attrs.get("allow_missing_far", False))):
        # NaN (attr absent) is truthy, so `or 0` never catches it and int(nan)
        # raises; a foreign 2.x file without the attr must read as 0, not crash.
        n_missing_raw = _num(pe_attrs.get("n_events_missing_far"))
        n_missing = 0 if np.isnan(n_missing_raw) else int(n_missing_raw)
        if n_missing:
            _fail("xcheck_detection_cut",
                  f"PE far_max={pe_far} was applied but allow_missing_far=True "
                  f"kept {n_missing} event(s) whose FAR is unknown, so the "
                  f"event list is not the FAR-cut list the injections model. "
                  f"Supply the missing FARs or drop those events.")
    results["xcheck_detection_cut"] = True


def _xcheck_campaign_cosmology(_check, results, sel_attrs):
    """The per-campaign cosmology record must be present and self-consistent.

    The scalar ``cosmology_H0``/``Om0`` cannot describe a combined export: an
    O4-era ``events`` campaign reads its own ddL/dz and uses no cosmology at
    all, while an endo3-style campaign uses its detected generation cosmology.
    Both are correct and they are different, so the honest record is per
    campaign -- ``cosmology_source_per_campaign`` with NaN H0/Om0 for the
    campaigns that used none.

    Checked here rather than merely written, because a `source` array that
    disagrees with the H0 array (a campaign claiming ``'file'`` while carrying a
    finite H0, or claiming ``'detected'`` with NaN) means the export layer and
    the reader disagree about what happened.
    """
    src = sel_attrs.get("cosmology_source_per_campaign")
    if src is None:
        # A pre-GW-09 file; nothing to check and nothing to fail.
        return
    src = [s.decode() if isinstance(s, bytes) else str(s)
           for s in np.atleast_1d(src)]
    H0 = np.atleast_1d(np.asarray(
        sel_attrs.get("cosmology_H0_per_campaign", []), dtype=float))
    Om0 = np.atleast_1d(np.asarray(
        sel_attrs.get("cosmology_Om0_per_campaign", []), dtype=float))
    n_camp = int(sel_attrs.get("n_campaigns", len(src)))

    _check("sel_cosmology_per_campaign_length",
           len(src) == n_camp == H0.size == Om0.size,
           f"n_campaigns={n_camp} but source/H0/Om0 arrays are "
           f"{len(src)}/{H0.size}/{Om0.size}")

    known = {"file", "detected", "override", "default"}
    _check("sel_cosmology_source_known",
           all(s in known for s in src),
           f"unrecognised cosmology_source value(s) "
           f"{sorted(set(src) - known)}; known are {sorted(known)}")

    if len(src) == H0.size == Om0.size:
        bad = [(s, h, o) for s, h, o in zip(src, H0, Om0)
               if (s == "file") != (not np.isfinite(h) and not np.isfinite(o))]
        _check("sel_cosmology_source_matches_values", not bad,
               f"campaign(s) whose cosmology_source disagrees with their "
               f"recorded H0/Om0: {bad}. 'file' means no cosmology entered "
               f"pdraw and must carry NaN; every other source must carry "
               f"finite values.")
    # The aggregate reflects the sub-checks; setdefault(True) here recorded a
    # passing aggregate even when they had just failed.
    subs = ("sel_cosmology_per_campaign_length", "sel_cosmology_source_known",
            "sel_cosmology_source_matches_values")
    results.setdefault(
        "xcheck_campaign_cosmology",
        all(results.get(k, True) for k in subs))


def _xcheck_reference_pair(_check, _fail, results, pe_attrs, sel_attrs):
    """A chieff PE file and a chieff_reference selection file: ONE ceiling.

    This is the cross-check that is stricter here than for chieff/chieff, and
    the reason is the whole difference between the two bases.  The substituting
    swap divides out a DIFFERENT prior on each side -- the posterior's sampling
    prior on one, the injected draw on the other -- so its two ``amax`` values
    are two different quantities and forcing them equal would be wrong (GW-31).
    The reference basis divides out the SAME prior on both sides: the reference
    is one declared object, and if the two files name different ceilings for it
    the spin density does not cancel between numerator and denominator, leaving
    a chi_eff-dependent error in every population posterior the pair produces.
    So here equality is the contract, and a mismatch fails.
    """
    a_ref = sel_attrs.get("spin_reference_amax")
    if a_ref is None:
        _fail("xcheck_reference_amax_recorded",
              "selection spin_basis='chieff_reference' but the file records no "
              "spin_reference_amax; the density it writes is a density with "
              "respect to a reference prior it does not name.")
    a_ref = float(a_ref)
    _check("xcheck_reference_amax_finite", np.isfinite(a_ref) and a_ref > 0,
           f"selection spin_reference_amax={a_ref!r} is not a positive finite "
           f"ceiling")

    pe_amax = _amax_provenance(pe_attrs, ("chi_eff_amax_1_per_event",
                                          "chi_eff_amax_2_per_event"))
    _check("xcheck_reference_pe_amax_finite",
           pe_amax.size > 0 and np.all(np.isfinite(pe_amax)),
           "PE chi_eff ceiling (chi_eff_amax_1/2_per_event, or the "
           "chi_eff_amax scalar on an older file) is empty or non-finite")
    if pe_amax.size and np.all(np.isfinite(pe_amax)):
        if not np.allclose(pe_amax, a_ref, rtol=1e-9, atol=1e-12):
            pe_u = sorted({round(float(x), 9) for x in pe_amax.ravel()})
            _fail("xcheck_reference_amax_matches",
                  f"PE chi_eff prior amax {pe_u} != selection "
                  f"spin_reference_amax {a_ref}. In this basis the two are the "
                  f"SAME object -- the selection pdraw was reweighted TO the "
                  f"prior the PE p_pe divides OUT -- so they must be equal, "
                  f"and a mismatch leaves an uncancelled chi_eff-dependent "
                  f"factor in every event's weight. Re-export the selection "
                  f"file with spin_reference_amax equal to the PE ceiling (or "
                  f"the PE file at that amax).")
        results["xcheck_reference_amax_matches"] = True

    from ..selection import PDRAW_STATE_CHIEFF_REFERENCE
    if sel_attrs.get("pdraw_state") != PDRAW_STATE_CHIEFF_REFERENCE:
        _fail("xcheck_reference_pdraw_state",
              f"selection pdraw_state={sel_attrs.get('pdraw_state')!r} does "
              f"not match the chieff_reference constant. The file was not "
              f"written by the reference-basis builder.")
    results["xcheck_reference_pdraw_state"] = True

    # Coverage is a physics fact about the campaigns, recorded at build time;
    # the validator reads it rather than re-deriving it, and says so out loud
    # because a coverage hole biases alpha LOW.
    cov = sel_attrs.get("spin_reference_coverage_ok")
    if cov is not None and not bool(cov):
        warnings.warn(
            f"chieff_reference: the selection file records "
            f"spin_reference_coverage_ok=False -- at least one campaign's "
            f"injected spin magnitudes do not reach a_ref={a_ref}, so the "
            f"reference support has a region with no draws in it and the "
            f"selection integral is biased low. See "
            f"spin_reference_coverage_per_campaign.")


def _fit_columns_present(_check, attrs, present, prefix):
    """Every coordinate a file DECLARES must exist as a dataset in it (GW-37).

    ``fit_columns`` is the contract's statement of which coordinates the
    exported density is a density in, and a generic consumer iterates it
    directly.  Nothing compared it against the file, so a 2.1 selection product
    could declare ``q`` -- the mass density coordinate GW-34 made canonical --
    and ship no such dataset while passing every check.  A declaration nothing
    verifies is documentation, not a contract.
    """
    declared = attrs.get("fit_columns")
    if declared is None:
        return  # a 2.0 file states no fit_columns; nothing to check.
    names = [v.decode() if isinstance(v, (bytes, bytearray)) else str(v)
             for v in np.atleast_1d(declared)]
    missing = [n for n in names if n not in present]
    _check(f"{prefix}_fit_columns_present", not missing,
           f"declares fit_columns={names} but the file has no dataset(s) "
           f"{missing}; a consumer iterating fit_columns raises KeyError, and "
           f"one that falls back to the present columns builds the density in "
           f"the wrong coordinates.")


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
