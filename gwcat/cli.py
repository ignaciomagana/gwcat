"""Unified gwcat command-line interface (PR 10).

::

    gwcat fetch ...              download PE / injection releases (Zenodo)
    gwcat ingest ...             raw PESummary files -> store.h5
    gwcat inspect store.h5       events, sample sets, params, availability,
                                  source classes
    gwcat export-darksirens ...  GWCatalog.to_darksirens
    gwcat selection ...          SelectionSet/CombinedSelectionSet.to_darksirens
    gwcat validate ...           gwcat.catalog.validate_export

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
    p_export.add_argument("--amax", type=float, default=0.99)
    p_export.add_argument("--no-summary", action="store_true",
                          help="Skip writing validation_summary.json/.md "
                               "next to --out.")

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
    p_sel.add_argument("--amax", type=float, default=0.99)
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
        write_summary=not args.no_summary,
    )
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
    from .catalog import validate_export

    try:
        results = validate_export(args.pe_path, args.selection_path,
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
    if args.command == "export-darksirens":
        return _cmd_export_darksirens(args)
    if args.command == "selection":
        return _cmd_selection(args)
    if args.command == "validate":
        return _cmd_validate(args)

    parser.error(f"unknown command {args.command!r}")  # pragma: no cover
    return 2


if __name__ == "__main__":
    sys.exit(main())
