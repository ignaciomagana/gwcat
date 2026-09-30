"""Unified gwcat command-line interface (PR 10).

::

    gwcat fetch ...              download PE / injection releases (Zenodo)
    gwcat ingest ...             raw PESummary files -> store.h5
    gwcat inspect store.h5       events, sample sets, params, availability,
                                  source classes
    gwcat export-darksirens ...  GWCatalog.to_darksirens
    gwcat selection ...          SelectionSet/CombinedSelectionSet.to_darksirens
    gwcat validate ...           gwcat.export.validate.validate_export_any
                                  (the format dispatcher behind
                                  ``gwcat.validate_export``)

This module intentionally contains no scientific logic of its own: every
subcommand either (a) delegates argument parsing AND execution wholesale to
an existing function (``fetch``/``ingest``, whose own ``_cli`` stays the
single source of truth for those flags), or (b) is a thin argparse ->
keyword-argument translation over an existing public API
(``GWCatalog.to_darksirens``, ``SelectionSet``/``CombinedSelectionSet``,
``validate_export``).

``fetch``/``ingest`` dispatch: :func:`main` intercepts ``argv[0] in
("fetch", "ingest")`` BEFORE handing anything to this module's own
``argparse`` parser, and passes the rest of argv straight to
``gwcat.fetch._cli`` / ``gwcat.ingest._cli``. This is deliberate, not just
stylistic: ``argparse.REMAINDER`` on a subparser's first positional does not
reliably capture tokens starting with ``-``/``--`` (a long-standing argparse
limitation -- see https://bugs.python.org/issue17050), so composing two
independent ``ArgumentParser`` instances as subparser + delegate does not
actually work for flag-heavy subcommands. Manual argv[0] dispatch sidesteps
the bug entirely and still lets ``gwcat fetch --help`` show fetch's own full
help text (the ``fetch``/``ingest`` entries in :func:`build_parser` exist
only so ``gwcat --help`` lists them).

Rename-friendliness
--------------------
The eventual rename to ``gwrangler`` (see the forward-handoff doc) should only
ever require adding a new ``[project.scripts]`` entry -- nothing in this
module's logic is keyed on the literal string "gwcat".  :data:`PROG` is
derived from ``sys.argv[0]`` (the console-script name actually invoked), so
help/usage text adapts automatically under a renamed entry point.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, Sequence

#: Re-exported for the CLI's help text and for callers that used to read it
#: here.  The single source of truth is :data:`gwcat.params.
#: DEFAULT_PARAMETER_SPACE`, so the CLI and the Python API cannot drift.


def _default_parameter_space() -> str:
    from .params import DEFAULT_PARAMETER_SPACE as _d
    return _d


DEFAULT_PARAMETER_SPACE = _default_parameter_space()


def _add_space_argument(parser, kind) -> None:
    """Add ``--parameter-space`` (with ``--spin-basis`` as the legacy alias).

    The choices are the registry spaces the *builder for this export kind*
    actually implements (its ``SUPPORTED_SPIN_BASES``), in registry order.
    Offering the full registry here advertised spaces (`component_6d`,
    `cartesian`, `aligned`) that then crashed in the builder with a raw
    traceback -- and `nospin`, which only the PE side can build.  A registered
    but not-yet-buildable space stays visible via 'export list-spaces'.
    """
    from .params import list_spaces
    if kind == "pe":
        from .export.pe_builder import SUPPORTED_SPIN_BASES as supported
    elif kind == "selection":
        from .export.selection_builder import (
            SUPPORTED_SPIN_BASES as supported)
    else:
        raise ValueError(f"unknown export kind {kind!r}")

    choices = [s for s in list_spaces() if s in supported]
    parser.add_argument("--parameter-space", "--spin-basis",
                        dest="spin_basis",
                        default=DEFAULT_PARAMETER_SPACE,
                        choices=choices,
                        help=f"Parameter space to export "
                             f"(default: {DEFAULT_PARAMETER_SPACE}). "
                             f"--spin-basis is an accepted alias. "
                             f"Run 'export list-spaces' for every declared "
                             f"space, including ones no builder ships yet.")


def _invoked_program_name() -> str:
    """Return the console entry-point name, with a stable library-call default.

    Normal console scripts still derive their identity from ``sys.argv[0]`` so
    a future renamed entry point remains automatic.  Test runners and
    ``python -m gwcat.cli`` are implementation details, not useful CLI names;
    direct calls to :func:`main` from those surfaces use ``gwcat``.
    """
    name = os.path.basename(sys.argv[0])
    if (not name or name in {"pytest", "py.test", "cli.py", "__main__.py"}
            or name.startswith("python")):
        return "gwcat"
    return name


#: Program name for argparse usage/help text -- derived from how the script
#: was actually invoked (``gwcat``, or a future renamed entry point).
PROG = _invoked_program_name()


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="gwcat: CBC posterior-sample and selection-product "
                    "release wrangler.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- fetch / ingest: listed here ONLY so `gwcat --help` shows them. `main`
    #    intercepts these two subcommands by argv[0] before parsing ever
    #    reaches this parser and hands the rest of argv straight to
    #    gwcat.fetch._cli / gwcat.ingest._cli (see the module docstring for
    #    why: argparse.REMAINDER cannot carry flag-style tokens through a
    #    subparser reliably). Every flag those modules define is therefore
    #    automatically available under `gwcat fetch`/`gwcat ingest` without
    #    being duplicated or reimplemented here.
    sub.add_parser(
        "fetch",
        help="Download GWTC PE/injection releases from Zenodo "
             "(see `%s fetch --help`)." % PROG)
    sub.add_parser(
        "ingest",
        help="Ingest raw PESummary cosmo files into a store.h5 "
             "(see `%s ingest --help`)." % PROG)

    # -- inspect --------------------------------------------------------
    p_inspect = sub.add_parser(
        "inspect",
        help="Inspect a built store.h5: events, sample sets, params, "
             "availability, source classes.")
    p_inspect.add_argument("store", help="Path to a store.h5 written by "
                                        "`%s ingest`." % PROG)
    p_inspect.add_argument("--json", action="store_true",
                           help="Print machine-readable JSON instead of a "
                                "human-readable table.")

    # -- export-darksirens ----------------------------------------------
    from .waveform_policy import WAVEFORM_POLICIES
    p_export = sub.add_parser(
        "export-darksirens",
        help="Export a darksirens-format PE file from a store.h5 "
             "(GWCatalog.to_darksirens).")
    p_export.add_argument("store", help="Path to a store.h5.")
    p_export.add_argument("--out", required=True, metavar="OUT.h5")
    p_export.add_argument("--source-class", default=None,
                          help="bbh / nsbh / bns / massgap / cbc, or a "
                               "canonical class name.")
    p_export.add_argument("--spin-prior-mode", default="include",
                          choices=["include", "exclude"])
    p_export.add_argument("--waveform-policy", default="preferred",
                          choices=list(WAVEFORM_POLICIES))
    p_export.add_argument("--approximant", default=None,
                          help="Required with "
                               "--waveform-policy=strict-approximant.")
    p_export.add_argument("--cosmology", default=None, metavar="H0,Om0",
                          help="Override cosmology applied to every exported "
                               "event. Omit (default) to use each event's own "
                               "stored PE cosmology.")
    p_export.add_argument("--event-list", default=None, metavar="FILE",
                          help="Restrict to a user event-list file (one name "
                               "per line, '#' comments allowed).")
    p_export.add_argument("--far-max", type=float, default=None,
                          metavar="FAR_YR")
    far_group = p_export.add_mutually_exclusive_group()
    far_group.add_argument("--allow-missing-far", action="store_true")
    far_group.add_argument("--require-far", action="store_true")
    p_export.add_argument("--nsamp", type=int, default=4096)
    p_export.add_argument("--seed", type=int, default=0)
    p_export.add_argument("--z-max", type=float, default=None)
    # "auto" like the v2 `export pe` sibling (GW-37): one hardcoded ceiling
    # cannot describe a store that mixes spin priors, and the chi_eff marginal
    # depends on the ceiling in a chi_eff-DEPENDENT way that does not cancel.
    p_export.add_argument("--amax", default="auto", metavar="AMAX",
                          help="Spin-prior ceiling: 'auto' (default) reads each "
                               "event's own spin_amax_1/2 from the store, a "
                               "number forces one ceiling on every event.")
    p_export.add_argument("--amax-fallback", type=float, default=0.99,
                          help="Fallback spin amax for events whose store meta "
                               "lacks spin_amax_1/2 (NaN); default 0.99.")
    p_export.add_argument("--no-summary", action="store_true",
                          help="Skip writing validation_summary.json/.md "
                               "next to --out.")

    # -- export (versioned registry: gwcat2 PE format) -------------------
    p_export2 = sub.add_parser(
        "export",
        help="Export products via the versioned export registry "
             "(gwcat2 'gwcat-pe-2.0' PE format).")
    xsub = p_export2.add_subparsers(dest="export_command", required=True)

    p_pe = xsub.add_parser(
        "pe",
        help="Export a gwcat2 PE file from a store.h5 "
             "(build_pe_product + registered writer).")
    p_pe.add_argument("store", help="Path to a store.h5.")
    p_pe.add_argument("--out", required=True, metavar="OUT.h5")
    p_pe.add_argument("--format", default="gwcat2",
                      help="Registered export format (default: gwcat2).")
    _add_space_argument(p_pe, "pe")
    p_pe.add_argument("--source-class", default=None,
                      help="bbh / nsbh / bns / massgap / cbc, or a "
                           "canonical class name.")
    p_pe.add_argument("--waveform-policy", default="preferred",
                      choices=list(WAVEFORM_POLICIES))
    p_pe.add_argument("--approximant", default=None,
                      help="Required with "
                           "--waveform-policy=strict-approximant.")
    p_pe.add_argument("--cosmology", default=None, metavar="H0,Om0",
                      help="Override cosmology applied to every exported "
                           "event. Omit (default) to use each event's own "
                           "stored PE cosmology.")
    p_pe.add_argument("--event-list", default=None, metavar="FILE",
                      help="Restrict to a user event-list file (one name "
                           "per line, '#' comments allowed).")
    p_pe.add_argument("--far-max", type=float, default=None, metavar="FAR_YR")
    pe_far_group = p_pe.add_mutually_exclusive_group()
    pe_far_group.add_argument("--allow-missing-far", action="store_true")
    pe_far_group.add_argument("--require-far", action="store_true")
    p_pe.add_argument("--pastro-min", type=float, default=None)
    p_pe.add_argument("--nsamp", type=int, default=4096)
    p_pe.add_argument("--seed", type=int, default=0)
    p_pe.add_argument("--z-max", type=float, default=None)
    p_pe.add_argument("--amax", default="auto", metavar="AMAX",
                      help="Spin-prior ceiling: 'auto' (default) reads each "
                           "event's own spin_amax_1/2 from the store, a number "
                           "forces one ceiling on every event.")
    p_pe.add_argument("--amax-fallback", type=float, default=0.99,
                      help="Fallback spin amax for events whose store meta "
                           "lacks spin_amax_1/2 (NaN); default 0.99.")
    p_pe.add_argument("--no-summary", action="store_true",
                      help="Skip writing validation_summary.json/.md "
                           "next to --out.")

    p_xsel = xsub.add_parser(
        "selection",
        help="Export a gwcat2 selection file (gwcat-selection-2.0) from one or "
             "more injection files (build_selection_product + writer).")
    p_xsel.add_argument("injections", nargs="+", metavar="INJ",
                        help="One or more LVK injection HDF5 files. More than "
                             "one is combined (Essick et al. fractions).")
    p_xsel.add_argument("--out", required=True, metavar="OUT.h5")
    p_xsel.add_argument("--format", default="gwcat2",
                        help="Registered export format (default: gwcat2).")
    _add_space_argument(p_xsel, "selection")
    p_xsel.add_argument("--far-threshold", type=float, default=1.0,
                        metavar="FAR_YR")
    p_xsel.add_argument("--source-class", default=None,
                        help="bbh / nsbh / bns / massgap / cbc, or a canonical "
                             "class name.")
    p_xsel.add_argument("--amax", default="auto", metavar="AMAX",
                        help="chieff-basis chi_eff-prior spin ceiling: 'auto' "
                             "(default) uses each campaign's own DETECTED "
                             "injected amax, a number forces one ceiling. "
                             "Ignored by 'component'; 'chieff_chip' always "
                             "uses the detected amax.")
    p_xsel.add_argument("--spin-reference-amax", type=float, default=None,
                        metavar="A_REF",
                        help="REQUIRED by (and only accepted by) "
                             "--parameter-space chieff_reference: the ceiling "
                             "of the isotropic uniform-magnitude reference "
                             "spin prior the exported pdraw is expressed "
                             "against. It must equal the ceiling the paired PE "
                             "export divides out (0.99 for the GWTC sampling "
                             "priors); there is no default because it is a "
                             "declaration about that pairing, not a property "
                             "of the campaign.")
    p_xsel.add_argument("--z-max", type=float, default=None, metavar="Z",
                        help="Subset injections to z <= Z, matching the PE "
                             "export's --z-max. Subsetting only (ndraw is "
                             "unchanged). Omit it against a truncated PE "
                             "export and mu covers a redshift range the "
                             "events do not; the validator refuses the pair.")
    p_xsel.add_argument("--snr-threshold", type=float, default=None,
                        metavar="SNR",
                        help="Semianalytic SNR threshold. On a cumulative "
                             "multi-run mixture it is applied to the O1/O2 "
                             "rows only (their FAR is +inf), with the FAR cut "
                             "on the O3/O4 rows only; on any other file it is "
                             "an OR-branch, detection = far-detected OR "
                             "(snr > SNR). Default: FAR cut only.")
    p_xsel.add_argument("--detection-policy", default="far",
                        choices=["far", "lvk-cumulative"],
                        help="'far' (default): the FAR cut (OR-ed with "
                             "--snr-threshold when given). 'lvk-cumulative': "
                             "the per-run rule of the LVK cumulative O1-O4b "
                             "mixture (Zenodo 19500052) -- semianalytic SNR > "
                             "--snr-threshold on O1/O2 rows, min search FAR < "
                             "--far-threshold over each run's own searches on "
                             "O3/O4 rows; requires --snr-threshold and a "
                             "mixture file. A FAR-only cut on such a file is "
                             "refused: it detects no O1/O2 row while their "
                             "draws and exposure stay in the normalisation.")
    p_xsel.add_argument("--acknowledge-semianalytic-excluded",
                        action="store_true",
                        help="Allow a FAR-only cut on a cumulative mixture "
                             "with semianalytic O1/O2 rows, so none of them "
                             "is detected. Correct ONLY for a deliberately "
                             "O3+O4-only analysis whose PE file has no O1/O2 "
                             "events.")
    p_xsel.add_argument("--sky-marginal", action="store_true",
                        help="Omit ra/dec and record sky_marginalized=True: "
                             "pdraw carries no sky density and the population "
                             "is isotropic, so the sky is marginalised. The "
                             "cumulative mixtures ship no sky position at "
                             "all. gwcat2 (2.0) format only.")
    p_xsel.add_argument("--H0", type=float, default=None,
                        help="Reference cosmology (default: Planck15) applied "
                             "to every injection file.")
    p_xsel.add_argument("--Om0", type=float, default=None)
    p_xsel.add_argument("--no-summary", action="store_true",
                        help="Skip writing validation_summary.json/.md "
                             "next to --out.")

    xsub.add_parser(
        "list-formats",
        help="List the registered (format, kind) export writers.")

    xsub.add_parser(
        "list-spaces",
        help="List the registered parameter spaces and what each declares "
             "(columns, projection vs bijection, exactness).")

    # -- selection -------------------------------------------------------
    p_sel = sub.add_parser(
        "selection",
        help="Build a darksirens selection-function export from one or more "
             "injection files (SelectionSet / CombinedSelectionSet).")
    p_sel.add_argument("--injections", nargs="+", required=True, metavar="FILE",
                       help="One or more LVK injection HDF5 files. More than "
                            "one is combined via CombinedSelectionSet.")
    p_sel.add_argument("--out", required=True, metavar="OUT.h5")
    p_sel.add_argument("--far-threshold", type=float, default=1.0,
                       metavar="FAR_YR")
    # "auto" like the v2 `export selection` sibling (GW-37): the swap replaces
    # a campaign's OWN injected spin density, so the ceiling must be that
    # campaign's -- endo3 injects 0.998, not 0.99.
    p_sel.add_argument("--amax", default="auto", metavar="AMAX",
                       help="chi_eff-prior spin ceiling: 'auto' (default) uses "
                            "each campaign's own DETECTED injected amax, a "
                            "number forces one ceiling on every campaign.")
    p_sel.add_argument("--source-class", default=None,
                       help="bbh / nsbh / bns / massgap / cbc, or a canonical "
                            "class name.")
    p_sel.add_argument("--H0", type=float, default=None,
                       help="Reference cosmology (default: Planck15) applied "
                            "to every --injections file.")
    p_sel.add_argument("--Om0", type=float, default=None)
    p_sel.add_argument("--no-summary", action="store_true",
                       help="Skip writing validation_summary.json/.md next "
                            "to --out.")

    # -- validate ---------------------------------------------------------
    p_val = sub.add_parser(
        "validate",
        help="Validate a darksirens PE export (and optionally a selection "
             "export) for internal + cross-file consistency.")
    p_val.add_argument("pe_path", metavar="PE.h5")
    p_val.add_argument("selection_path", nargs="?", default=None,
                       metavar="SELECTION.h5")
    p_val.add_argument("--strict", action="store_true",
                       help="Raise on the first internal-consistency failure "
                            "(cross-file contract checks always raise).")

    return parser


# ---------------------------------------------------------------------------
# Subcommand implementations (thin argparse -> library-call translation)
# ---------------------------------------------------------------------------
def _parse_cosmology(spec: Optional[str]):
    if not spec:
        return None
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 2:
        raise SystemExit(
            f"--cosmology must be 'H0,Om0' (e.g. 67.74,0.3089), got {spec!r}")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as e:
        raise SystemExit(f"--cosmology: {e}")


def _cmd_inspect(args) -> int:
    from .catalog import GWCatalog
    from .validation_summary import summarize_catalog, _json_default

    cat = GWCatalog(args.store)
    info = summarize_catalog(cat)
    info["store_path"] = args.store

    if args.json:
        print(json.dumps(info, indent=2, default=_json_default))
        return 0

    cat.summary()
    print()
    print(f"schema_version: {info['schema_version']}")
    print(f"package_version: {info['package_version']}")
    print(f"stored_parameters ({len(info['stored_parameters'])}): "
          f"{', '.join(info['stored_parameters'])}")
    if info["missing_required_parameters"]:
        print(f"missing_required_parameters: "
              f"{info['missing_required_parameters']}")
    if info["missing_optional_parameters"]:
        print(f"missing_optional_parameters: "
              f"{info['missing_optional_parameters']}")
    print(f"source_class_counts: {info['source_class_counts']}")
    if info["waveform_counts"]:
        print(f"waveform_counts: {info['waveform_counts']}")
    if info["approximant_counts"]:
        print(f"approximant_counts: {info['approximant_counts']}")
    if info["n_events_with_multiple_sample_sets"]:
        print(f"events with >1 sample set: "
              f"{info['n_events_with_multiple_sample_sets']}")
    print(f"far_missing_count: {info['far_missing_count']} / {info['n_events']}")
    print(f"p_astro_available_count: {info['p_astro_available_count']} / "
          f"{info['n_events']}")
    if info["per_event_cosmology_present"]:
        print(f"per_event_cosmology_varies: "
              f"{info['per_event_cosmology_varies']}")
    return 0


def _parse_source_class(spec: Optional[str]):
    """A bare CLI string can't express "an iterable of classes" the way the
    Python API does, so accept a comma-separated list as a convenience (e.g.
    ``--source-class nsbh,bns``); a single class/keyword passes through
    unchanged to ``resolve_filter_classes``."""
    if spec is None or "," not in spec:
        return spec
    return [s.strip() for s in spec.split(",") if s.strip()]


def _cmd_export_darksirens(args) -> int:
    from .catalog import GWCatalog

    cosmology = _parse_cosmology(args.cosmology)
    cat = GWCatalog(args.store)
    cat.to_darksirens(
        args.out,
        source_class=_parse_source_class(args.source_class),
        spin_prior_mode=args.spin_prior_mode,
        waveform_policy=args.waveform_policy,
        approximant=args.approximant,
        cosmology=cosmology,
        event_list=args.event_list,
        far_max=args.far_max,
        allow_missing_far=args.allow_missing_far,
        require_far=args.require_far,
        nsamp=args.nsamp,
        seed=args.seed,
        z_max=args.z_max,
        amax=args.amax,
        amax_fallback=args.amax_fallback,
        write_summary=not args.no_summary,
    )
    return 0


def _cmd_export(args) -> int:
    if args.export_command == "pe":
        return _cmd_export_pe(args)
    if args.export_command == "selection":
        return _cmd_export_selection(args)
    if args.export_command == "list-formats":
        return _cmd_export_list_formats(args)
    if args.export_command == "list-spaces":
        return _cmd_export_list_spaces(args)
    return 2  # pragma: no cover -- argparse requires a valid subcommand


def _cmd_export_list_spaces(args) -> int:
    """Print each registered space and the declaration that governs its use.

    ``kind`` is the load-bearing column: a *projection* is definable only
    against a uniform-magnitude/isotropic parent draw and must refuse a campaign
    that is not one (R1), while a *bijection* is always definable and cancels
    exactly in the consumer's per-event renormalisation.
    """
    from .params import get_space, list_spaces

    rows = []
    for name in list_spaces():
        s = get_space(name)
        rows.append((name, s.spin_block.map_kind,
                     "yes" if s.is_exact else "no",
                     ",".join(s.fit_columns),
                     ",".join(s.advisory_columns) or "-"))

    head = ("space", "kind", "exact", "fit_columns", "advisory")
    w = [max(len(r[i]) for r in rows + [head]) for i in range(len(head))]
    print("  ".join(h.ljust(w[i]) for i, h in enumerate(head)))
    print("  ".join("-" * w[i] for i in range(len(head))))
    for r in rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
    print(f"\ndefault for both 'export pe' and 'export selection': "
          f"{DEFAULT_PARAMETER_SPACE}")
    print("advisory columns are written but NOT covered by the density -- "
          "do not fit them.")
    return 0


def _cmd_export_pe(args) -> int:
    from .catalog import GWCatalog

    cosmology = _parse_cosmology(args.cosmology)
    cat = GWCatalog(args.store)
    cat.export(
        args.out,
        format=args.format,
        spin_basis=args.spin_basis,
        write_summary=not args.no_summary,
        source_class=_parse_source_class(args.source_class),
        waveform_policy=args.waveform_policy,
        approximant=args.approximant,
        cosmology=cosmology,
        event_list=args.event_list,
        far_max=args.far_max,
        allow_missing_far=args.allow_missing_far,
        require_far=args.require_far,
        pastro_min=args.pastro_min,
        nsamp=args.nsamp,
        seed=args.seed,
        z_max=args.z_max,
        amax=args.amax,
        amax_fallback=args.amax_fallback,
    )
    return 0


def _cmd_export_selection(args) -> int:
    from .selection import SelectionSet, CombinedSelectionSet

    if args.detection_policy == "lvk-cumulative" and args.snr_threshold is None:
        raise ValueError(
            "--detection-policy lvk-cumulative requires --snr-threshold (10 in "
            "the LVK analyses). The O1/O2 rows of a cumulative mixture are "
            "semianalytic and carry no search FAR, so without an SNR "
            "threshold none of them is detected -- while total_generated and "
            "total_analysis_time still count their draws and their O1/O2 "
            "exposure, which biases the detection probability low and the "
            "inferred rate high.")

    kwargs = {}
    if args.H0 is not None:
        kwargs["H0"] = args.H0
    if args.Om0 is not None:
        kwargs["Om0"] = args.Om0

    sets = [SelectionSet(path, **kwargs) for path in args.injections]
    target = sets[0] if len(sets) == 1 else CombinedSelectionSet(sets)
    target.export(
        args.out,
        format=args.format,
        spin_basis=args.spin_basis,
        write_summary=not args.no_summary,
        far_threshold=args.far_threshold,
        source_class=_parse_source_class(args.source_class),
        amax=args.amax,
        spin_reference_amax=args.spin_reference_amax,
        snr_threshold=args.snr_threshold,
        z_max=args.z_max,
        detection_policy=args.detection_policy,
        acknowledge_semianalytic_excluded=(
            args.acknowledge_semianalytic_excluded),
        sky_marginal=args.sky_marginal,
    )
    return 0


def _cmd_export_list_formats(args) -> int:
    from .export import list_formats

    formats = list_formats()
    if not formats:
        print("(no export formats registered)")
        return 0
    for name, kind in formats:
        print(f"{name}\t{kind}")
    return 0


def _cmd_selection(args) -> int:
    from .selection import SelectionSet, CombinedSelectionSet

    kwargs = {}
    if args.H0 is not None:
        kwargs["H0"] = args.H0
    if args.Om0 is not None:
        kwargs["Om0"] = args.Om0

    sets = [SelectionSet(path, **kwargs) for path in args.injections]
    target = sets[0] if len(sets) == 1 else CombinedSelectionSet(sets)
    target.to_darksirens(
        args.out,
        far_threshold=args.far_threshold,
        amax=args.amax,
        source_class=_parse_source_class(args.source_class),
        write_summary=not args.no_summary,
    )
    return 0


def _cmd_validate(args) -> int:
    # One dispatcher, shared with the public `gwcat.validate_export`: v1 files
    # (gwcat-1.0 / gwcat-selection-1.0) go to the frozen
    # gwcat.catalog.validate_export, v2 files (gwcat-pe-2.x /
    # gwcat-selection-2.x) to gwcat.export.validate_export_v2, and a mixed or
    # unrecognised pair is refused. The routing used to live HERE only, so the
    # library entry point validated every file against the v1 contract.
    from .export.validate import validate_export_any

    try:
        results = validate_export_any(args.pe_path, args.selection_path,
                                      strict=args.strict)
    except (ValueError, AssertionError) as e:
        print(f"validate: FAILED: {e}", file=sys.stderr)
        return 1
    return 0 if all(results.values()) else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(argv) if argv is not None else list(sys.argv[1:])

    # Intercept fetch/ingest by argv[0] BEFORE this module's own argparse
    # parser ever sees them -- see the module docstring for why (argparse's
    # REMAINDER does not reliably carry flag-style tokens through a
    # subparser). Everything after "fetch"/"ingest" goes straight to the
    # existing, fully-featured CLI in gwcat.fetch/gwcat.ingest unmodified.
    if argv and argv[0] == "fetch":
        from .fetch import _cli as fetch_cli
        return fetch_cli(argv=argv[1:], _deprecated=False,
                         default_write_summary=True,
                         prog=f"{PROG} fetch") or 0
    if argv and argv[0] == "ingest":
        from .ingest import _cli as ingest_cli
        return ingest_cli(argv=argv[1:], _deprecated=False,
                          default_write_summary=True,
                          prog=f"{PROG} ingest") or 0

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect":
        return _cmd_inspect(args)
    # A ValueError out of these commands is a *diagnosed refusal* -- an
    # unrecognised --source-class token, a partial cosmology override, a
    # zero-density injection -- whose message already says what to fix.  The
    # user asked on the command line, so answer there, not with a traceback
    # (validate already did; the export commands surfaced raw tracebacks).
    try:
        if args.command == "export-darksirens":
            return _cmd_export_darksirens(args)
        if args.command == "export":
            return _cmd_export(args)
        if args.command == "selection":
            return _cmd_selection(args)
        if args.command == "validate":
            return _cmd_validate(args)
    except ValueError as e:
        print(f"{PROG} {args.command}: error: {e}", file=sys.stderr)
        return 1

    parser.error(f"unknown command {args.command!r}")  # pragma: no cover
    return 2


if __name__ == "__main__":
    sys.exit(main())
