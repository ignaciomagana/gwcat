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

import json
import warnings

import numpy as np
import h5py

#: Spin bases each side of the v2 pipeline implements -- read from the
#: builders themselves, so the validator cannot reject a file the builder
#: legitimately wrote (nospin PE files failed here) or accept one it could
#: never have produced.
from .pe_builder import SUPPORTED_SPIN_BASES as _PE_BASES
from .contract import (DL_PRIOR_IMPLS_EXACT, _impl_str,
                       mixed_prior_impl_events)
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


def validate_export_v2(pe_path, selection_path=None, strict=False, *,
                       spin_prior_allow_list=None, require_exact_priors=False):
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
      * a boolean ``mock_data`` attr -- ``True`` (a synthetic campaign, from the
        store's provenance) and ``False`` are both valid;
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
      * ``mock_data``, when present, is boolean (it is written only when True;
        absent means real data);
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
    spin_prior_allow_list : optional
        The events a ``chieff`` / ``chieff_reference`` pair may carry although
        their spin prior is NOT the label's own analytic one (GW-40d; see
        :func:`load_spin_prior_allow_list` for the accepted forms).  Without it
        every such event is refused.
    require_exact_priors : bool, default False
        Refuse (always raise) unless every analytic prior factor was evaluated
        EXACTLY (GW-40i): the PE's ``dL_prior_impl_per_event`` all
        ``"exact"``/``"analytic"``, and ``chi_eff_prior_impl == "exact"`` on
        every side that carries a chi_eff factor.  A file that does not record
        its implementation is refused too.  Without it, a legacy or a mixed
        product is only warned about (and refused under ``strict``).

    Returns
    -------
    dict
        ``{check_name: passed_bool}``.
    """
    results = {}
    spin_prior_allow_list = load_spin_prior_allow_list(spin_prior_allow_list)

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
        pe_path, _PE_LEGACY + _PE_COMPONENT_EXTRA + ["in_support"])

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

    # One event, one block (review 2026-10-20).  waveform_policy='all' writes
    # an event once per sample set (homogeneous_sample_sets=False), and the
    # event-list digest de-duplicates by design, so such a file matched the
    # whitelist it was cut from while a consumer multiplied that event's
    # likelihood once per copy.
    if pe_attrs.get("event_names") is not None:
        _names = [_impl_str(x) for x in np.atleast_1d(pe_attrs["event_names"])]
        _check("pe_event_names_length", len(_names) == nobs,
               f"{len(_names)} event_names but nobs={nobs}")
        _seen, _dup = set(), set()
        for _n in _names:
            (_dup if _n in _seen else _seen).add(_n)
        _check("pe_event_names_unique", not _dup,
               f"{len(_dup)} event(s) appear more than once in event_names "
               f"(e.g. {sorted(_dup)[:5]}); a consumer counts each copy as a "
               f"separate detection. Export one sample set per event "
               f"(waveform_policy other than 'all').")
    if pe_attrs.get("homogeneous_sample_sets") is not None:
        _check("pe_homogeneous_sample_sets",
               bool(pe_attrs["homogeneous_sample_sets"]),
               "homogeneous_sample_sets=False: the file holds more than one "
               "sample set for some event (waveform_policy='all'), which is "
               "not a population-analysis input.")

    # p_pe finite & STRICTLY positive.  An exact zero used to be documented as
    # legal ("distance-prior tails"), but that state only ever arose from the
    # truncated distance prior GW-01 removed, and darksirens turns a zero weight
    # into a -inf log-weight that still counts in n for the per-event MC
    # variance.  It is now a hard failure, reported with a count.
    if "p_pe" in pe_cols and pe_cols["p_pe"].size:
        p = pe_cols["p_pe"]
        _check("pe_p_pe_finite", np.all(np.isfinite(p)))
        # Samples outside the spin prior's support carry a RECORDED zero when
        # the builder was told allow_out_of_support (in_support=0, counted in
        # n_samples_out_of_support).  Only an in-support zero is the truncated
        # distance prior; failing the recorded ones blamed the distance prior
        # for a state the builder wrote on purpose (review 2026-10-20).
        ins = pe_cols.get("in_support")
        has_ins = ins is not None
        if has_ins:
            _check("pe_in_support_length", ins.shape == p.shape,
                   f"{ins.shape[0]} in_support rows != {p.shape[0]} p_pe rows")
        ins = (np.asarray(ins).astype(bool)
               if has_ins and ins.shape == p.shape
               else np.ones(p.shape, dtype=bool))
        n_zero = int(np.sum((p == 0.0) & ins))
        n_neg = int(np.sum(p < 0.0))
        _check("pe_p_pe_positive", n_zero == 0 and n_neg == 0,
               f"{n_zero} zero ({100 * n_zero / p.size:.2f}%) at in-support "
               f"samples and {n_neg} negative of {p.size} samples; "
               f"min={np.nanmin(p):.3e}. A zero p_pe at an in-support sample "
               f"means the store's p_dL_pe was truncated at the recorded "
               f"distance-prior bounds -- re-ingest so the distance prior is "
               f"evaluated over the full sample range.")
        n_out = int(np.sum(~ins))
        if n_out:
            n_rec = pe_attrs.get("n_samples_out_of_support")
            _check("pe_out_of_support_recorded",
                   bool(pe_attrs.get("out_of_support_allowed", False))
                   and n_rec is not None and int(n_rec) == n_out,
                   f"{n_out} samples have in_support=0, but the file records "
                   f"out_of_support_allowed="
                   f"{pe_attrs.get('out_of_support_allowed')!r} and "
                   f"n_samples_out_of_support={n_rec!r}")
            _check("pe_out_of_support_p_pe_zero", np.all(p[~ins] == 0.0),
                   f"{int(np.sum(p[~ins] != 0.0))} out-of-support samples "
                   f"carry a non-zero p_pe; outside the spin prior's support "
                   f"the density is zero")
            print(f"  NOTE: {n_out} of {p.size} samples "
                  f"({100 * n_out / p.size:.3f}%) are outside the spin prior's "
                  f"support and carry a recorded zero p_pe "
                  f"(allow_out_of_support); the consumer drops them but "
                  f"counts them in each event's n.")
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

    # Mock-data provenance: a boolean, True or False.  True is a legitimate
    # export (a synthetic campaign run through these builders), not a defect.
    _check("pe_mock_data_attr",
           "mock_data" in pe_attrs and _is_flag(pe_attrs["mock_data"]),
           f"mock_data={pe_attrs.get('mock_data', '<absent>')!r}; every gwcat "
           f"PE export records a boolean mock_data attr.")
    if _is_flag(pe_attrs.get("mock_data")) and bool(pe_attrs["mock_data"]):
        print("  NOTE: PE export is flagged mock_data=True (synthetic data).")

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

    # How the PE's analytic prior factors were evaluated (GW-40i).
    _check_pe_prior_impl(_fail, results, pe_attrs, strict=strict,
                         require_exact=require_exact_priors)

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
        # GW-39: a sky-marginalised product omits ra/dec by construction, and
        # must then really omit them (a file that says marginalised and ships
        # a sky would be ambiguous about which it is).
        _sky_marg = bool(sel_attrs.get("sky_marginalized", False))
        if _sky_marg:
            sel_required = [c for c in sel_required if c not in ("ra", "dec")]
            _check("sel_sky_marginalized_has_no_radec",
                   "ra" not in sel_present and "dec" not in sel_present,
                   "sky_marginalized=True but the file carries ra/dec")
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

        # The reference basis's zero-weight rows must be exactly the rows
        # outside the reference support, and their sentinel must really weigh
        # nothing.  a1/a2 were required above for this purpose but never
        # compared, so a sentinel on in-support rows (beta biased low) or a
        # finite weight on out-of-reference rows (biased high) passed.
        if sel_basis == "chieff_reference":
            _reference_sentinel_checks(_check, sel_attrs, sel_cols)

        # Every name a 2.1 file declares in its OWN fit_columns must exist as a
        # dataset (GW-37).  A generic check beats extending two more hardcoded
        # lists: the selection file declared `q` and shipped no `q`, and nothing
        # compared the declaration against the file, so 70/70 checks passed on a
        # product whose contract described coordinates it did not contain.
        _fit_columns_present(_check, sel_attrs, sel_present, prefix="sel")

        # Mock-data provenance: written only when True, so absence is valid.
        _check("sel_mock_data_attr",
               "mock_data" not in sel_attrs or _is_flag(sel_attrs["mock_data"]),
               f"mock_data={sel_attrs.get('mock_data')!r} is not boolean.")

        # Per-campaign injected-spin provenance attrs must be present.
        for a in ("injected_spin_format", "injected_spin_amax_detected",
                  "injected_spin_uniform_isotropic"):
            _check(f"sel_has_{a}", a in sel_attrs, f"attr {a!r} missing")

        # GW-39: a cumulative-mixture product must say how each run was
        # detected and how its exposure is defined, and its per-run written
        # counts must add up to the file.
        if bool(sel_attrs.get("cumulative_mixture", False)):
            for a in ("run_labels", "n_detected_per_run",
                      "detection_rule_per_run", "T_definition_per_run",
                      "component_of_run", "mixture_components",
                      "N_per_component", "T_per_component_s",
                      "T_definition_per_component"):
                _check(f"sel_mixture_has_{a}", a in sel_attrs,
                       f"cumulative_mixture=True but attr {a!r} is missing")
            # Every *_per_run attr is aligned with run_labels and every
            # *_per_component attr with mixture_components, so a consumer can
            # zip them without guessing the granularity.
            if "run_labels" in sel_attrs:
                _nr = len(np.atleast_1d(sel_attrs["run_labels"]))
                for a in sorted(k for k in sel_attrs
                                if k.endswith("_per_run")
                                and k != "mixture_weights_per_run"):
                    _v = sel_attrs[a]
                    _len = (np.asarray(_v).shape[0]
                            if np.ndim(_v) >= 1 else 1)
                    _check(f"sel_mixture_{a}_aligned_with_run_labels",
                           _len == _nr,
                           f"len({a})={_len} != len(run_labels)={_nr}")
            if "mixture_components" in sel_attrs:
                _nc = len(np.atleast_1d(sel_attrs["mixture_components"]))
                for a in ("N_per_component", "T_per_component_s",
                          "T_definition_per_component"):
                    if a in sel_attrs:
                        _len = len(np.atleast_1d(sel_attrs[a]))
                        _check(f"sel_mixture_{a}_aligned_with_components",
                               _len == _nc,
                               f"len({a})={_len} != "
                               f"len(mixture_components)={_nc}")
            if "n_detected_per_run" in sel_attrs:
                # The per-run counts are the MIXTURE campaign's own rows; a
                # product may add a campaign disjoint from it in time, so
                # compare with the mixture's own written count, which can
                # never exceed the product's.
                _nper = np.asarray(sel_attrs["n_detected_per_run"]).ravel()
                _n_camp = len(np.atleast_1d(
                    sel_attrs.get("campaign_kind_per_campaign", [""])))
                _n_mix = sel_attrs.get("n_detected_mixture")
                if _n_mix is None and _n_camp == 1:
                    _n_mix = n_det
                _check("sel_mixture_has_n_detected_mixture",
                       _n_mix is not None,
                       "a multi-campaign cumulative-mixture product must "
                       "record n_detected_mixture")
                if _n_mix is not None:
                    _check("sel_mixture_n_detected_per_run_sum",
                           int(np.sum(_nper)) == int(_n_mix)
                           and int(_n_mix) <= n_det
                           and (_n_camp > 1 or int(_n_mix) == n_det),
                           f"sum(n_detected_per_run)={int(np.sum(_nper))}, "
                           f"n_detected_mixture={int(_n_mix)}, "
                           f"n_detected={n_det}, n_campaigns={_n_camp}")

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
            _xcheck_reference_pair(_check, _fail, results, pe_attrs, sel_attrs,
                                   spin_prior_allow_list=spin_prior_allow_list,
                                   strict=strict)
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
                # A caller-forced ceiling (--amax) REPLACES the campaign's own
                # in the swap; where the campaign's ceiling is known and
                # differs, pdraw is the wrong density, chi_eff-dependently
                # (review 2026-10-20: 0.99 forced on endo3's 0.998 is 0.7% off
                # in the bulk and 29% above chi_eff 0.9).
                forced = [(k, used[k].tolist(), det[k].tolist())
                          for k, s_ in enumerate(sel_src)
                          if s_ == "caller" and np.all(np.isfinite(det[k]))
                          and not np.allclose(used[k], det[k], rtol=1e-9,
                                              atol=1e-12)]
                if forced:
                    msg = (f"chieff: {len(forced)} campaign(s) swapped at a "
                           f"caller-forced ceiling that is not the campaign's "
                           f"own detected one (campaign, used, detected): "
                           f"{forced}. The swap replaces the injected spin "
                           f"density with the marginal at the ceiling it is "
                           f"given, so pdraw is off chi_eff-dependently; "
                           f"export without --amax.")
                    if strict:
                        _fail("xcheck_chieff_sel_amax_caller_matches_detected",
                              msg)
                    warnings.warn(msg)
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

            # The swap's violation record (GW-06): a campaign whose draw was
            # MEASURED not uniform-isotropic, or ships a joint spin density
            # the builder refuses by default, reached the file only through
            # the builder's strict=False opt-out, and its pdraw is the wrong
            # density.  The flags were written and never read.
            try:
                _viol = json.loads(str(sel_attrs.get(
                    "spin_basis_assumption_violations", "[]")) or "[]")
            except ValueError:
                _viol = []
            _refused = [v for v in _viol if isinstance(v, dict)
                        and (v.get("verified") or v.get("reason"))]
            if (bool(sel_attrs.get("spin_basis_assumption_violated", False))
                    or _refused):
                _fail("xcheck_chieff_spin_basis_assumption",
                      f"the selection file records that the chi_eff swap was "
                      f"applied to campaign(s) whose spin draw is not (or "
                      f"cannot be shown to be) uniform-magnitude isotropic: "
                      f"{[v.get('path') for v in _refused] or _viol}. It was "
                      f"exported with strict=False; its pdraw is the wrong "
                      f"density by a chi_eff-dependent factor. Use "
                      f"spin_basis='chieff_reference' or 'component'.")
            if bool(sel_attrs.get("spin_basis_assumption_unverified", False)):
                warnings.warn(
                    "chieff: the selection file records campaign(s) with no "
                    "injected spin draw densities, so the uniform-isotropic "
                    "assumption behind the chi_eff swap was not checked "
                    "(spin_basis_assumption_unverified=True).")
            results["xcheck_chieff_spin_basis_assumption"] = True

            pe_u = sorted({round(float(x), 6) for x in pe_amax_arr.ravel()
                           if np.isfinite(x)})
            sel_u = sorted({round(float(x), 6) for x in sel_amax_arr.ravel()
                            if np.isfinite(x)})
            det_u = sorted({round(float(x), 6) for x in det.ravel()
                            if np.isfinite(x)})
            print(f"  NOTE: chieff amax -- PE prior amax {pe_u} vs selection "
                  f"injected (detected) amax {det_u or 'undetectable'}; the "
                  f"selection's swap used {sel_u}. The PE and selection "
                  f"ceilings are DIFFERENT quantities (the PE sampling prior's "
                  f"ceiling vs the injected draw's) and are NOT required to "
                  f"match.")
            if set(pe_u) != set(sel_u):
                warnings.warn(
                    f"chieff: PE prior amax {pe_u} != selection swap amax "
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

        # (c) Cosmology agreement (the v1 tolerances), against the cosmology
        # each campaign's pdraw actually used -- not the selection's scalar,
        # which is the constructor default (review 2026-10-20).
        _xcheck_cosmology(_fail, results, pe_attrs, sel_attrs)

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

        # (e1) The same z_max is the same truncation only at the same distance.
        _xcheck_z_max_distance(_fail, results, pe_attrs, sel_attrs)

        # (e2) Per-campaign selection cosmology (GW-09).
        _xcheck_campaign_cosmology(_check, results, sel_attrs)

        # (e2b) Mock-data provenance on the two sides.
        _xcheck_mock_data(results, pe_attrs, sel_attrs)

        # (e2c) A cumulative mixture must detect something in every PE
        # event's run.
        if bool(sel_attrs.get("cumulative_mixture", False)):
            _xcheck_mixture_event_runs(_fail, results, pe_attrs, sel_attrs)

        # (e3) The same chi_eff prior implementation on both sides (GW-40i).
        _xcheck_chi_eff_prior_impl(_fail, results, pe_attrs, sel_attrs,
                                   strict=strict,
                                   require_exact=require_exact_priors)

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


def validate_export_any(pe_path, selection_path=None, strict=False, *,
                        spin_prior_allow_list=None, require_exact_priors=False):
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
    require_exact_priors : bool, default False
        See :func:`validate_export_v2`; a v1 pair is refused.

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
        return validate_export_v2(pe_path, selection_path, strict=strict,
                                  spin_prior_allow_list=spin_prior_allow_list,
                                  require_exact_priors=require_exact_priors)
    from ..catalog import validate_export as _validate_export_v1
    if spin_prior_allow_list is not None:
        warnings.warn("spin_prior_allow_list applies only to the gwcat-2 "
                      "validator; ignored for this v1 pair.")
    if require_exact_priors:
        raise ValueError(
            "require_exact_priors: a v1 (gwcat-1.0) pair predates the exact "
            "prior evaluations (GW-40i) and records no implementation; it "
            "cannot satisfy the requirement.")
    return _validate_export_v1(pe_path, selection_path, strict=strict)


def _is_flag(v) -> bool:
    """A boolean HDF5 attr as h5py reads it back (or a 0/1 integer)."""
    if isinstance(v, (bool, np.bool_)):
        return True
    return isinstance(v, (int, np.integer)) and int(v) in (0, 1)


def _xcheck_mock_data(results, pe_attrs, sel_attrs):
    """Warn when exactly one side of a pair is flagged mock_data.

    Not a failure: synthetic events paired with a real campaign's injections is
    a legitimate design (events drawn through the real sensitivity), and a mock
    campaign with real events is at least conceivable.  It is still worth
    saying, because the usual cause is a mock export paired with the wrong
    file.  An absent selection attr means False (the builder writes it only
    when True).
    """
    pe_mock = bool(pe_attrs.get("mock_data", False))
    sel_mock = bool(sel_attrs.get("mock_data", False))
    if pe_mock != sel_mock:
        warnings.warn(
            f"PE mock_data={pe_mock} but selection mock_data={sel_mock}: one "
            f"side of this pair is synthetic and the other is not. Intended "
            f"for a mock-events-through-real-injections design; otherwise the "
            f"pair is mismatched.")
    results["xcheck_mock_data"] = True


#: Leap seconds GPS - UTC from each UTC date on (none since 2017-01-01).
_GPS_LEAP_SECONDS = ((2012, 7, 1, 16), (2015, 7, 1, 17), (2017, 1, 1, 18))


def _event_gps_span(name):
    """The GPS time of an event NAME, as a ``(start, end)`` span, or ``None``.

    ``GWyymmdd_hhmmss`` is a UTC time and gives a zero-width span; a bare
    ``GWyymmdd`` gives its whole UTC day.  Anything else (synthetic names) is
    ``None``: the files record no event times, so the name is the only clock.
    """
    import datetime as _dt
    import re as _re

    m = _re.fullmatch(r"GW(\d{2})(\d{2})(\d{2})(?:_(\d{2})(\d{2})(\d{2}))?",
                      str(name).strip())
    if m is None:
        return None
    y, mo, d = (int(x) for x in m.group(1, 2, 3))
    try:
        t0 = _dt.datetime(2000 + y, mo, d)
        if m.group(4) is not None:
            t0 = t0.replace(hour=int(m.group(4)), minute=int(m.group(5)),
                            second=int(m.group(6)))
    except ValueError:
        return None

    def _gps(t):
        leap = 0
        for yy, mm, dd, n in _GPS_LEAP_SECONDS:
            if t >= _dt.datetime(yy, mm, dd):
                leap = n
        return (t - _dt.datetime(1980, 1, 6)).total_seconds() + leap

    if m.group(4) is not None:
        g = _gps(t0)
        return g, g
    return _gps(t0), _gps(t0 + _dt.timedelta(days=1)) - 1.0


def _event_run(name):
    """The observing run of an event name (official exposure windows), or
    ``None`` when the name gives no time, or a time in no single window."""
    from ..observing_runs import EXPOSURE_WINDOWS_GPS, RunWindowError, run_of_gps

    span = _event_gps_span(name)
    if span is None:
        return None
    try:
        lo = run_of_gps(span[0], EXPOSURE_WINDOWS_GPS)
        hi = run_of_gps(span[1], EXPOSURE_WINDOWS_GPS)
    except RunWindowError:
        return None
    return lo if lo == hi else None


def _xcheck_mixture_event_runs(_fail, results, pe_attrs, sel_attrs):
    """Every PE event's run must have detectable rows in the mixture.

    A cumulative-mixture selection records, per run, how many rows it wrote
    (``n_detected_per_run``) and which detection rule applied
    (``detection_rule_per_run``).  An acknowledged FAR-only export detects
    nothing in O1/O2 (rule ``none(semianalytic_rows_excluded)``): it is correct
    only for an event list with no O1/O2 events, and nothing compared the two.
    Each PE event is placed in a run by the time in its name (official
    exposure windows); an event in a run the mixture does not cover, or covers
    with no detectable rows, enters the likelihood with no term for its run in
    mu, which is refused.

    With other campaigns in the product, an event outside the mixture's runs
    may be covered by one of them (the files record no campaign time ranges),
    so that case warns.  Names that give no time (synthetic events) warn.
    """
    names = pe_attrs.get("event_names")
    runs = sel_attrs.get("run_labels")
    if names is None or runs is None:
        # The sel_mixture_has_* checks report a missing run_labels; a PE file
        # without event_names has nothing to place.
        results["xcheck_mixture_event_runs"] = True
        return
    runs = [_impl_str(r) for r in np.atleast_1d(runs)]
    n_per = np.atleast_1d(np.asarray(
        sel_attrs.get("n_detected_per_run", [0] * len(runs)))).astype(int)
    rules = [_impl_str(r) for r in np.atleast_1d(
        sel_attrs.get("detection_rule_per_run", [""] * len(runs)))]
    if not (len(runs) == n_per.size == len(rules)):
        # The alignment checks above have failed already.
        results["xcheck_mixture_event_runs"] = True
        return
    n_camp = len(np.atleast_1d(
        sel_attrs.get("campaign_kind_per_campaign", [""])))
    detectable = {r: (int(n) > 0 and not rule.startswith("none"))
                  for r, n, rule in zip(runs, n_per, rules)}
    blind, outside, untimed = [], [], []
    for n in (_impl_str(x) for x in np.atleast_1d(names)):
        if _event_gps_span(n) is None:
            untimed.append(n)
            continue
        r = _event_run(n)
        if r is None or r not in detectable:
            outside.append((n, r))
        elif not detectable[r]:
            blind.append((n, r))
    if blind:
        by_run = {}
        for n, r in blind:
            by_run.setdefault(r, []).append(n)
        _fail("xcheck_mixture_event_runs",
              f"{len(blind)} PE event(s) lie in run(s) where the cumulative-"
              f"mixture selection detects nothing: "
              + "; ".join(f"{r} [{len(v)}]: {v[:5]}"
                          for r, v in sorted(by_run.items()))
              + f" (n_detected_per_run={dict(zip(runs, n_per.tolist()))}, "
                f"detection_rule_per_run={dict(zip(runs, rules))}). Those "
                f"events enter the likelihood with no term for their run in "
                f"mu. Drop them from the PE export (--event-list), or build "
                f"the selection with --snr-threshold so the semianalytic rows "
                f"are detected.")
    if outside:
        msg = (f"{len(outside)} PE event(s) lie outside every run of the "
               f"cumulative mixture ({runs}): "
               + ", ".join(f"{n} ({r or 'no official run window'})"
                           for n, r in outside[:10])
               + ". Their exposure is not in this mixture.")
        if n_camp > 1:
            warnings.warn(msg + " Another campaign of this product may cover "
                                "them; the files cannot show which.")
        else:
            _fail("xcheck_mixture_event_runs", msg)
    if untimed:
        warnings.warn(
            f"{len(untimed)} PE event name(s) carry no GPS time (e.g. "
            f"{untimed[:3]}), so whether the cumulative mixture detects "
            f"anything in their runs cannot be checked.")
    results["xcheck_mixture_event_runs"] = True


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


#: Bases whose p_pe (PE) / pdraw (selection) carries an isotropic chi_eff factor.
_PE_CHI_EFF_BASES = ("chieff", "chieff_chip")
_SEL_CHI_EFF_BASES = ("chieff", "chieff_reference", "chieff_chip")


def _check_pe_prior_impl(_fail, results, pe_attrs, *, strict=False,
                         require_exact=False):
    """The PE file's two analytic prior factors: same family, and (on request)
    exact (GW-40i).

    * **Mixed** -- ``chi_eff_prior_impl="exact"`` over a store whose
      ``p_dL_pe`` came from a legacy interpolation, or the legacy ``"grid"``
      over an exact store.  Each factor is still its declared prior to its own
      accuracy, so this is not a cancellation error, but the file is neither
      the exact product nor a legacy regression: WARN, and FAIL under
      ``strict``.
    * ``require_exact`` -- every ``dL_prior_impl_per_event`` must be
      ``"exact"`` or ``"analytic"`` and, for a basis with a chi_eff factor,
      ``chi_eff_prior_impl`` must be ``"exact"``; an unrecorded implementation
      is refused as well.  Always raises.
    """
    basis = _impl_str(pe_attrs.get("spin_basis"))
    chi = pe_attrs.get("chi_eff_prior_impl")
    chi = None if chi is None else _impl_str(chi)
    dl = pe_attrs.get("dL_prior_impl_per_event")
    dl = None if dl is None else [_impl_str(x) for x in np.atleast_1d(dl)]
    names = pe_attrs.get("event_names")
    names = ([_impl_str(x) for x in np.atleast_1d(names)]
             if names is not None else [f"#{i}" for i in range(len(dl or []))])

    if chi is not None and dl is not None:
        mixed = mixed_prior_impl_events(chi, dl, names)
        if mixed:
            msg = (f"PE chi_eff_prior_impl={chi!r} but p_dL_pe of "
                   f"{len(mixed)} event(s) was evaluated by the other family "
                   f"(e.g. {mixed[:3]}; dL_prior_impl_per_event). The file is "
                   f"neither the exact product nor a legacy regression: "
                   f"re-ingest and re-export with --legacy-grid-priors on both "
                   f"steps or on neither.")
            if strict:
                _fail("pe_prior_impl_consistent", msg)
            warnings.warn(msg)
    results["pe_prior_impl_consistent"] = True

    if not require_exact:
        return
    if dl is None or any(x == "" for x in dl):
        _fail("pe_prior_impl_exact",
              "require_exact_priors: the PE file does not record "
              "dL_prior_impl_per_event for every event (written before GW-40i, "
              "or from a store that predates the record), so an exact "
              "distance prior cannot be verified. Re-ingest and re-export.")
    bad = [(n, x) for n, x in zip(names, dl) if x not in DL_PRIOR_IMPLS_EXACT]
    if bad:
        _fail("pe_prior_impl_exact",
              f"require_exact_priors: p_dL_pe of {len(bad)} event(s) was not "
              f"evaluated exactly (e.g. {bad[:3]}); expected every "
              f"dL_prior_impl_per_event in {DL_PRIOR_IMPLS_EXACT}. Re-ingest "
              f"without --legacy-grid-priors.")
    if basis in _PE_CHI_EFF_BASES and chi != "exact":
        _fail("pe_prior_impl_exact",
              f"require_exact_priors: PE spin_basis={basis!r} carries a "
              f"chi_eff prior factor but chi_eff_prior_impl={chi!r} (expected "
              f"'exact'). Re-export without --legacy-grid-priors.")
    results["pe_prior_impl_exact"] = True


def _xcheck_chi_eff_prior_impl(_fail, results, pe_attrs, sel_attrs, *,
                               strict=False, require_exact=False):
    """PE ``p_pe`` and selection ``pdraw`` must carry the SAME chi_eff density.

    The chi_eff factor of a chieff-type pair is the same analytic prior on both
    sides; GW-40i made it exact and kept the interpolated table only as a
    legacy option.  A pair built with different implementations does not
    divide out one density: the exact and grid values differ by up to 4e-3
    relative, chi_eff-dependently, so the mismatch does not cancel.

    Both sides recording ``chi_eff_prior_impl`` and disagreeing -> FAIL.  One
    side recording it and the other not (a file from before the record) ->
    WARN (unknown is not evidence of a mismatch, but the older file almost
    certainly used the grid), and FAIL under ``strict``.  ``require_exact``
    additionally refuses a selection whose chieff-type pdraw is not
    ``chi_eff_prior_impl == "exact"``.
    """
    pe_impl = pe_attrs.get("chi_eff_prior_impl")
    sel_impl = sel_attrs.get("chi_eff_prior_impl")
    pe_impl = None if pe_impl is None else _impl_str(pe_impl)
    sel_impl = None if sel_impl is None else _impl_str(sel_impl)
    if pe_impl is not None and sel_impl is not None:
        if pe_impl != sel_impl:
            _fail("xcheck_chi_eff_prior_impl",
                  f"PE chi_eff_prior_impl={pe_impl!r} but selection "
                  f"chi_eff_prior_impl={sel_impl!r}: p_pe and pdraw carry "
                  f"different chi_eff densities (the legacy grid is up to "
                  f"4e-3 off the exact prior, chi_eff-dependently, so the "
                  f"difference does not cancel). Rebuild both with the same "
                  f"implementation (--legacy-grid-priors on both or neither).")
    elif (pe_impl is None) != (sel_impl is None):
        msg = (f"chi_eff_prior_impl recorded on one side only (PE={pe_impl!r}, "
               f"selection={sel_impl!r}); the unrecorded file predates GW-40i "
               f"and almost certainly used the legacy interpolated grid.")
        if strict:
            _fail("xcheck_chi_eff_prior_impl_recorded",
                  msg + " A strict validation refuses the pair; re-export the "
                        "older file with this gwcat.")
        warnings.warn(msg)
    if require_exact:
        sel_basis = _impl_str(sel_attrs.get("spin_basis"))
        if sel_basis in _SEL_CHI_EFF_BASES and sel_impl != "exact":
            _fail("xcheck_chi_eff_prior_impl_exact",
                  f"require_exact_priors: selection spin_basis={sel_basis!r} "
                  f"carries a chi_eff prior factor in pdraw but "
                  f"chi_eff_prior_impl={sel_impl!r} (expected 'exact'). "
                  f"Re-export without --legacy-grid-priors.")
    results["xcheck_chi_eff_prior_impl"] = True


def _pe_cosmologies(_fail, pe_attrs):
    """The ``(H0, Om0)`` pairs the PE file's source-frame columns were computed
    under: one per event when ``cosmology_per_event_varies``, else the file's
    single cosmology (``[]`` when it records none).

    A varying-cosmology file without per-event arrays of length ``nobs`` is
    refused: the cross-check used to run over empty arrays and pass.
    """
    nobs = int(pe_attrs.get("nobs", 0))
    if bool(pe_attrs.get("cosmology_per_event_varies", False)):
        H = np.asarray(pe_attrs.get("cosmology_H0_per_event", []),
                       dtype=float).ravel()
        O = np.asarray(pe_attrs.get("cosmology_Om0_per_event", []),
                       dtype=float).ravel()
        if not (H.size == O.size == nobs and nobs > 0
                and np.all(np.isfinite(H)) and np.all(np.isfinite(O))):
            _fail("xcheck_cosmology",
                  f"PE cosmology_per_event_varies=True but "
                  f"cosmology_H0_per_event / cosmology_Om0_per_event hold "
                  f"{H.size} / {O.size} entries (finite: "
                  f"{int(np.isfinite(H).sum())} / {int(np.isfinite(O).sum())}) "
                  f"for nobs={nobs}, so the cosmology of the source-frame "
                  f"columns cannot be compared with the selection's.")
        return list(zip(H.tolist(), O.tolist()))
    H0, Om0 = pe_attrs.get("pe_cosmology_H0"), pe_attrs.get("pe_cosmology_Om0")
    if H0 is None and Om0 is None:
        return []
    return [(float("nan") if H0 is None else float(H0),
             float("nan") if Om0 is None else float(Om0))]


def _sel_cosmologies(sel_attrs):
    """``(pairs, n_without, source)``: the cosmologies the selection's pdraw
    used, one per campaign that used one; how many campaigns used none (an
    ``events``-format campaign reads its own ddL/dz); and where the answer came
    from (``"per_campaign"``, or ``"scalar"`` on a file written before the
    per-campaign record, whose scalar is all there is)."""
    src = sel_attrs.get("cosmology_source_per_campaign")
    H = np.atleast_1d(np.asarray(
        sel_attrs.get("cosmology_H0_per_campaign", []), dtype=float)).ravel()
    O = np.atleast_1d(np.asarray(
        sel_attrs.get("cosmology_Om0_per_campaign", []), dtype=float)).ravel()
    if src is not None and len(np.atleast_1d(src)) == H.size == O.size:
        ok = np.isfinite(H) & np.isfinite(O)
        return (list(zip(H[ok].tolist(), O[ok].tolist())),
                int(np.sum(~ok)), "per_campaign")
    H0, Om0 = sel_attrs.get("cosmology_H0"), sel_attrs.get("cosmology_Om0")
    if H0 is None or Om0 is None:
        return [], 0, "scalar"
    return [(float(H0), float(Om0))], 0, "scalar"


def _xcheck_cosmology(_fail, results, pe_attrs, sel_attrs):
    """The PE cosmology against the cosmology each selection campaign used.

    The tolerances are the v1 ones (``|dH0| >= 1.0`` or ``|dOm0| >= 0.05``
    fails).  The selection side is ``cosmology_*_per_campaign`` -- the
    cosmology that entered each campaign's pdraw, NaN for a campaign that used
    none -- not the scalar ``cosmology_H0``/``Om0``, which is the constructor
    default and the cosmology of no campaign (review 2026-10-20); the scalar is
    used only for a file written before the per-campaign record.
    """
    pe_cos = _pe_cosmologies(_fail, pe_attrs)
    sel_cos, _, sel_src = _sel_cosmologies(sel_attrs)
    bad = [(ph, po, sh, so) for ph, po in pe_cos for sh, so in sel_cos
           if (np.isfinite(ph) and abs(ph - sh) >= 1.0)
           or (np.isfinite(po) and abs(po - so) >= 0.05)]
    if bad:
        pe_h = sorted({round(b[0], 6) for b in bad})
        pe_o = sorted({round(b[1], 6) for b in bad})
        sel_u = sorted({(round(b[2], 6), round(b[3], 6)) for b in bad})
        where = ("per-campaign cosmology (cosmology_H0/Om0_per_campaign)"
                 if sel_src == "per_campaign"
                 else "scalar cosmology_H0/Om0 (a file without the "
                      "per-campaign record)")
        _fail("xcheck_cosmology",
              f"PE cosmology H0 {pe_h[:5]} / Om0 {pe_o[:5]} disagrees with the "
              f"selection's {where} (H0, Om0) = {sel_u} (|dH0| >= 1.0 or "
              f"|dOm0| >= 0.05). A campaign's cosmology is the one its "
              f"injections were generated under, so re-exporting the selection "
              f"under another cosmology is not a fix; export the PE file under "
              f"a cosmology within tolerance, or check that the pair is the "
              f"intended one.")
    results["xcheck_cosmology"] = True


#: Relative agreement required of the two sides' z_max truncation distances.
_ZMAX_DL_RTOL = 1e-6


def _xcheck_z_max_distance(_fail, results, pe_attrs, sel_attrs):
    """Equal ``z_max`` must be an equal truncation in distance.

    The PE export drops samples with ``z(dL; event cosmology) > z_max``, and
    the selection drops injections with injected ``z > z_max``, i.e. with
    ``dL > dL(z_max; campaign cosmology)``.  Compared as numbers, z_max = 1
    matched while the O3 events were truncated at dL = 6798.0 Mpc and the
    injections at 6785.9 Mpc.  So when both sides state the same z_max, every
    cosmology defining it -- each PE event's and each campaign's -- must give
    the same luminosity distance at z_max (relative ``_ZMAX_DL_RTOL``).  A
    campaign that records no cosmology (an ``events``-format file with its own
    redshifts) warns: its distance edge is not in the files.
    """
    def _f(x):
        try:
            return float(np.asarray(x).ravel()[0])
        except (TypeError, ValueError, IndexError):
            return float("nan")

    pe_z, sel_z = _f(pe_attrs.get("z_max")), _f(sel_attrs.get("z_max"))
    if not (np.isfinite(pe_z) and np.isfinite(sel_z)
            and np.isclose(pe_z, sel_z, rtol=1e-12, atol=0.0)):
        # Unequal or unstated truncations are judged by _xcheck_detection_cut.
        results["xcheck_z_max_distance"] = True
        return
    from ..cosmology import make_cosmology

    pe_cos = _pe_cosmologies(_fail, pe_attrs)
    sel_cos, n_without, sel_src = _sel_cosmologies(sel_attrs)
    if sel_src == "scalar":
        # The scalar is no campaign's cosmology; nothing to compare it with.
        sel_cos, n_without = [], max(1, int(sel_attrs.get("n_campaigns", 1)))
    edges = []
    for side, pairs in (("PE", pe_cos), ("selection", sel_cos)):
        for h, o in sorted({(round(h, 8), round(o, 8)) for h, o in pairs
                            if np.isfinite(h) and np.isfinite(o)}):
            d = float(make_cosmology(h, o).luminosity_distance(pe_z).value)
            edges.append((side, h, o, d))
    if edges:
        ds = np.array([e[3] for e in edges])
        if ds.max() - ds.min() > _ZMAX_DL_RTOL * ds.max():
            _fail("xcheck_z_max_distance",
                  f"both sides state z_max={pe_z}, but it is not one "
                  f"truncation: "
                  + "; ".join(f"{side} (H0={h}, Om0={o}) cuts at dL = "
                              f"{d:.1f} Mpc" for side, h, o, d in edges)
                  + ". The PE side converts each sample's dL to z under the "
                    "event's cosmology and the selection cuts the injected z "
                    "of a campaign generated under its own, so the two "
                    "integrate over different distance ranges. Export both "
                    "under the cosmology of the injections, or drop z_max.")
    if n_without:
        warnings.warn(
            f"z_max={pe_z}: {n_without} selection campaign(s) record no "
            f"cosmology, so the distance at which their injections are "
            f"truncated cannot be compared with the PE side's.")
    results["xcheck_z_max_distance"] = True


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


#: The only spin-prior source a reference pair accepts without an allow-list.
_SPIN_PRIOR_SOURCE_OK = "own_analytic"


def load_spin_prior_allow_list(spec):
    """Normalise a spin-prior allow-list to ``{event_name: kind_or_None}``.

    Accepted forms (GW-40d):

    * ``None`` -- no allow-list (every non-``own_analytic`` event is refused);
    * a mapping ``{event: kind}`` (or ``{event: {"kind": kind, ...}}``) -- the
      event is allowed ONLY with that exact ``prior_source_kind``;
    * an iterable of names -- allowed with any kind;
    * a path: ``.json`` holding either of the above, or a text file with one
      ``NAME`` or ``NAME KIND`` per line (``#`` comments allowed).

    Being explicit about the kind is recommended: the operator-approved list
    (OD-7) says WHY each event is an exception, and an event whose provenance
    changed underneath it should then fail rather than pass silently.
    """
    import json
    import os

    if spec is None:
        return None
    if isinstance(spec, (str, os.PathLike)):
        path = os.fspath(spec)
        with open(path) as f:
            text = f.read()
        if path.lower().endswith(".json"):
            return load_spin_prior_allow_list(json.loads(text))
        out = {}
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            bits = line.split()
            out[bits[0]] = bits[1] if len(bits) > 1 else None
        return out
    if isinstance(spec, dict):
        out = {}
        for k, v in spec.items():
            if isinstance(v, dict):
                v = v.get("kind")
            out[str(k)] = None if v in (None, "", "*") else str(v)
        return out
    return {str(k): None for k in spec}


def _xcheck_reference_spin_prior_source(_fail, results, pe_attrs,
                                        allow_list, strict=False):
    """Refuse non-own-analytic spin priors in a reference pair (GW-40d).

    The reference basis divides the SAME U(0, a_ref) isotropic prior out of
    both files.  That is exact for an event whose spin prior is its label's own
    analytic declaration.  It is an ASSUMPTION for an event whose prior was
    inherited from a sibling (a ``Mixed`` set), read from a LALInference config,
    or not declared at all -- so each such event must be named on an explicit,
    operator-approved allow-list (OD-7), or the pair is refused.
    """
    kinds = pe_attrs.get("prior_source_kind_spin_per_event")
    names = pe_attrs.get("event_names")
    if (kinds is None or names is None) and strict:
        # Under --strict an older PE file must not bypass the refusal by
        # simply not recording the provenance it would be refused on.
        _fail("xcheck_reference_spin_prior_source_recorded",
              "chieff_reference: the PE file records no "
              "prior_source_kind_spin_per_event (written before GW-40c), so "
              "whether each event's spin prior is its own declared "
              "U(0, a_ref) cannot be checked, and a strict validation refuses "
              "the pair. Re-export the PE file with this gwcat.")
    if kinds is None or names is None:
        warnings.warn(
            "chieff_reference: the PE file records no "
            "prior_source_kind_spin_per_event (written before GW-40c), so "
            "whether each event's spin prior is its own declared "
            "U(0, a_ref) cannot be checked. Re-export the PE file to have it "
            "checked.")
        return
    dec = [x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
           for x in np.atleast_1d(kinds)]
    nms = [x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
           for x in np.atleast_1d(names)]
    allow = allow_list or {}
    offenders, wrong_kind = [], []
    for n, k in zip(nms, dec):
        if k == _SPIN_PRIOR_SOURCE_OK:
            continue
        if n not in allow:
            offenders.append((n, k))
        elif allow[n] is not None and allow[n] != k:
            wrong_kind.append((n, k, allow[n]))
    if offenders or wrong_kind:
        by_kind = {}
        for n, k in offenders:
            by_kind.setdefault(k, []).append(n)
        parts = [f"{k} [{len(v)}]: {v[:10]}"
                 + ("" if len(v) <= 10 else f" (+{len(v) - 10} more)")
                 for k, v in sorted(by_kind.items())]
        if wrong_kind:
            parts.append("allow-listed with a DIFFERENT kind: "
                         + ", ".join(f"{n} is {k}, list says {w}"
                                     for n, k, w in wrong_kind[:10]))
        _fail("xcheck_reference_spin_prior_source",
              f"{len(offenders) + len(wrong_kind)} PE event(s) divide out a "
              f"spin prior that is NOT their label's own analytic declaration, "
              f"and are not on the spin-prior allow-list: "
              + "; ".join(parts)
              + ". The chieff_reference pair assumes each event's spin prior "
                "IS the declared U(0, a_ref) reference; for these that is an "
                "assumption (sibling-inherited, config-declared or assumed "
                "default), so it must be approved explicitly -- pass "
                "spin_prior_allow_list= (CLI: --spin-prior-allow-list FILE) "
                "naming each event (and its kind), or drop the events.")
    unused = sorted(set(allow) - set(nms))
    if unused:
        warnings.warn(f"spin_prior_allow_list names {len(unused)} event(s) "
                      f"not in the PE file: {unused[:10]}")
    results["xcheck_reference_spin_prior_source"] = True


def _xcheck_reference_pair(_check, _fail, results, pe_attrs, sel_attrs,
                           spin_prior_allow_list=None, strict=False):
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

    # WHERE each event's ceiling came from, not only its value (GW-40d).
    _xcheck_reference_spin_prior_source(_fail, results, pe_attrs,
                                        spin_prior_allow_list, strict=strict)

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


#: How much larger than every in-support pdraw the out-of-reference sentinel
#: must be: its inverse weight is then at least this factor below any real
#: row's, i.e. zero to every precision a selection integral is computed at.
_SENTINEL_MIN_RATIO = 1e100


def _reference_sentinel_checks(_check, attrs, cols):
    """The ``chieff_reference`` sentinel rows against ``a1``/``a2``.

    The builder writes ``spin_reference_excluded_pdraw`` (default 1e300) at
    exactly the rows with ``a1 > a_ref`` or ``a2 > a_ref`` (or no magnitude),
    and counts them in ``spin_reference_excluded_rows``.  Checked here: the two
    row sets are identical, the count matches, and the sentinel is so large
    that its inverse weight is negligible against every in-support row.
    """
    if not all(k in cols for k in ("pdraw", "a1", "a2")):
        return  # the sel_has_* checks above have already failed
    pd = np.asarray(cols["pdraw"], dtype=float)
    a1 = np.asarray(cols["a1"], dtype=float)
    a2 = np.asarray(cols["a2"], dtype=float)
    if not (pd.shape == a1.shape == a2.shape):
        return  # the length checks above have already failed
    names = ("spin_reference_amax", "spin_reference_excluded_pdraw",
             "spin_reference_excluded_rows")
    missing = [k for k in names if attrs.get(k) is None]
    if not _check("sel_reference_sentinel_recorded", not missing,
                  f"spin_basis='chieff_reference' but attr(s) {missing} are "
                  f"missing, so which rows carry the zero-weight sentinel "
                  f"cannot be checked"):
        return
    a_ref = float(attrs["spin_reference_amax"])
    sentinel = float(attrs["spin_reference_excluded_pdraw"])
    n_rec = int(attrs["spin_reference_excluded_rows"])
    outside = ~((a1 <= a_ref) & (a2 <= a_ref))
    is_sent = pd == sentinel
    n_in_sent = int(np.sum(is_sent & ~outside))
    n_out_real = int(np.sum(outside & ~is_sent))
    _check("sel_reference_sentinel_rows", n_in_sent == 0 and n_out_real == 0,
           f"{n_in_sent} row(s) with a1, a2 <= a_ref={a_ref} carry the "
           f"out-of-reference sentinel pdraw={sentinel:.3e} (their weight is "
           f"zeroed: beta biased low), and {n_out_real} row(s) with a1 or a2 "
           f"> a_ref carry a real pdraw (the reference prior is zero there: "
           f"beta biased high)")
    _check("sel_reference_excluded_rows_count", int(np.sum(is_sent)) == n_rec,
           f"{int(np.sum(is_sent))} rows carry the sentinel but "
           f"spin_reference_excluded_rows={n_rec}")
    real = pd[~is_sent]
    if outside.any() and real.size and np.all(np.isfinite(real)):
        top = float(np.max(real))
        _check("sel_reference_sentinel_negligible",
               np.isfinite(sentinel) and sentinel >= _SENTINEL_MIN_RATIO * top,
               f"spin_reference_excluded_pdraw={sentinel:.3e} is not far above "
               f"the largest in-support pdraw {top:.3e} (required: at least "
               f"{_SENTINEL_MIN_RATIO:.0e} times larger), so the "
               f"out-of-reference rows the attrs call zero-weight carry a "
               f"real weight")


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
