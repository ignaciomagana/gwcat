"""Parameter schema contract for gwcat (PR 5).

The posterior store must NOT shrink to the intersection of parameters present in
all events.  Instead it stores the *union* of parameters, NaN-filling event
slices where a parameter is absent and recording a per-event x per-parameter
availability mask.  This module is the single, declarative source of truth for:

  * the canonical parameter *groups* (``PARAMETER_GROUPS``), and
  * what each *export* REQUIRES (``EXPORT_REQUIREMENTS``), so that a requested
    export fails loudly -- naming the missing parameter(s) and event(s) -- when a
    required column is absent or unavailable, rather than silently dropping it.

Nothing here needs network access or heavy dependencies; it is plain data plus a
couple of small helpers used by :mod:`gwcat.ingest` and :mod:`gwcat.catalog`.

Parameter groups (see the handoff "Parameter Schema Contract")::

    core_intrinsic:  mass_1, mass_2, mass_ratio, chirp_mass
    core_extrinsic:  luminosity_distance, redshift, ra, dec, theta_jn, psi
    spin:            a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, chi_eff, chi_p,
                     cos_tilt_1, cos_tilt_2
    bns_nsbh:        lambda_1, lambda_2, lambda_tilde, delta_lambda_tilde
    diagnostic:      log_likelihood, log_prior, weights

Groups are advisory metadata (used for classification/diagnostics).  A store may
hold any subset/superset of these; the availability mask is what makes
"present for some events, absent for others" a first-class, non-lossy state.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

# ── Declarative parameter groups ─────────────────────────────────────────────
#: Canonical parameter groups, in a stable order.  Values are ordered tuples.
PARAMETER_GROUPS = {
    "core_intrinsic": ("mass_1", "mass_2", "mass_ratio", "chirp_mass"),
    "core_extrinsic": ("luminosity_distance", "redshift", "ra", "dec",
                       "theta_jn", "psi"),
    "spin": ("a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl",
             "chi_eff", "chi_p", "cos_tilt_1", "cos_tilt_2"),
    "bns_nsbh": ("lambda_1", "lambda_2", "lambda_tilde", "delta_lambda_tilde"),
    "diagnostic": ("log_likelihood", "log_prior", "weights"),
}

#: Every parameter that belongs to a declared group, in group/tuple order.
ALL_GROUP_PARAMS = tuple(p for grp in PARAMETER_GROUPS.values() for p in grp)

# param -> group name (first group that lists it)
_PARAM_TO_GROUP = {}
for _grp, _params in PARAMETER_GROUPS.items():
    for _p in _params:
        _PARAM_TO_GROUP.setdefault(_p, _grp)


def group_of(param: str) -> str | None:
    """Return the group name a parameter belongs to, or ``None`` if ungrouped."""
    return _PARAM_TO_GROUP.get(param)


def params_in_groups(groups: Iterable[str]) -> List[str]:
    """Flatten a set of group names into their ordered parameter list."""
    out: List[str] = []
    for g in groups:
        if g not in PARAMETER_GROUPS:
            raise KeyError(f"unknown parameter group {g!r}; "
                           f"known groups: {list(PARAMETER_GROUPS)}")
        out.extend(PARAMETER_GROUPS[g])
    return out


# ── Export requirements ──────────────────────────────────────────────────────
# The darksirens PE export reads exactly these columns.  ``p_dL_pe`` is the
# stored, mass-prior-agnostic distance prior (not a group parameter); the rest
# are group parameters.  Keeping the list here makes GWCatalog.to_darksirens's
# ``need`` list declarative and lets the required-vs-optional check live in one
# place.
DARKSIRENS_REQUIRED = ("mass_1", "mass_2", "luminosity_distance", "ra", "dec",
                       "chi_eff", "p_dL_pe")

# ── Spin-basis PE exports (PR 6) ─────────────────────────────────────────────
# The versioned gwcat2 PE export supports three spin bases; each declares its
# required parameters here (keyed ``"gwcat2_pe:<basis>"``).
#
# ``component`` needs the component spins ``a_1``/``a_2`` on top of the
# darksirens set; ``chi_eff`` is KEPT required (unlike the bare ingredient tuple
# in the handoff) so the exported ``chieff`` column is always present -- a
# deliberate simplification documented in :mod:`gwcat.export.pe_builder`.  Its
# tilt requirement (``cos_tilt_i`` OR ``tilt_i``) is an *alternative* group,
# expressed via :func:`check_required_alternatives` rather than a flat required
# list (either member satisfies it, per event).
COMPONENT_REQUIRED = ("mass_1", "mass_2", "luminosity_distance", "ra", "dec",
                      "chi_eff", "a_1", "a_2", "p_dL_pe")

#: Component-basis tilt alternatives: each group is satisfied per event if ANY
#: member (a stored *and available* parameter) is present.
COMPONENT_TILT_ALTERNATIVES = (
    ("cos_tilt_1", "tilt_1"),
    ("cos_tilt_2", "tilt_2"),
)

#: chieff_chip-basis chi_p requirement: ONE group whose alternatives are either
#: the stored ``chi_p`` itself, or a full ingredient set from which
#: ``chi_p_from_components`` can derive it at export.  An alternative that is a
#: tuple names params that must ALL be present+available together.
CHIEFF_CHIP_CHIP_ALTERNATIVES = (
    (
        "chi_p",
        ("a_1", "a_2", "cos_tilt_1", "cos_tilt_2"),
        ("a_1", "a_2", "tilt_1", "tilt_2"),
    ),
)

#: export name -> ordered tuple of REQUIRED parameters.  (Alternative groups --
#: the tilt/chi_p "any-of" requirements -- are enforced separately by the
#: builder via :func:`check_required_alternatives`.)
EXPORT_REQUIREMENTS = {
    "darksirens": DARKSIRENS_REQUIRED,
    "gwcat2_pe:chieff": DARKSIRENS_REQUIRED,
    "gwcat2_pe:component": COMPONENT_REQUIRED,
    "gwcat2_pe:chieff_chip": DARKSIRENS_REQUIRED,
}


def required_params(export: str) -> List[str]:
    """Return the ordered list of parameters a named export requires."""
    if export not in EXPORT_REQUIREMENTS:
        raise KeyError(f"unknown export {export!r}; "
                       f"known exports: {list(EXPORT_REQUIREMENTS)}")
    return list(EXPORT_REQUIREMENTS[export])


class MissingParameterError(KeyError):
    """A required posterior parameter is absent from the store or unavailable
    (NaN-filled) for one or more requested events.

    Subclasses :class:`KeyError` so legacy ``except KeyError`` handlers still
    catch it, but overrides ``__str__`` so the message is shown verbatim (a bare
    ``KeyError`` would wrap it in quotes).
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


def check_required(required: Sequence[str], store_params: Sequence[str],
                   avail, event_names, sel_idx, param_index,
                   export: str = "export") -> None:
    """Raise :class:`MissingParameterError` if any required parameter is absent
    from the store, or is unavailable (NaN-filled) for any selected event.

    Parameters
    ----------
    required : sequence of str
        Parameters the export needs.
    store_params : sequence of str
        Parameter names present in the store (columns of ``avail``).
    avail : 2-D bool array, shape (n_events_total, n_store_params)
        Per-event x per-parameter availability mask.
    event_names : 1-D array of str
        Event names aligned with the rows of ``avail``.
    sel_idx : 1-D int array
        Row indices of the currently selected events.
    param_index : dict
        Mapping ``param -> column index`` into ``avail``.
    export : str
        Human-readable export name for the error message.
    """
    import numpy as np

    absent = [p for p in required if p not in param_index]
    if absent:
        raise MissingParameterError(
            f"{export} requires parameter(s) {absent} which are not in the "
            f"store; stored parameters are {list(store_params)}.")

    sel = np.asarray(sel_idx)
    problems = []
    for p in required:
        if sel.size == 0:
            continue
        col = avail[sel, param_index[p]]
        if not col.all():
            bad = sorted(np.asarray(event_names)[sel][~col].tolist())
            problems.append(f"{p!r} for event(s) {bad}")
    if problems:
        raise MissingParameterError(
            f"{export} requires parameter(s) that are present in the store but "
            f"unavailable (NaN-filled) for some selected events: "
            + "; ".join(problems)
            + ". Those parameters were not available for those events at "
              "ingest; drop the events, choose a different export, or re-ingest "
              "with the parameter present.")


def _describe_alternative(alt) -> str:
    """Human-readable description of a single alternative (str or tuple)."""
    if isinstance(alt, str):
        return alt
    alt = tuple(alt)
    if len(alt) == 1:
        return alt[0]
    return "(" + " and ".join(alt) + ")"


def check_required_alternatives(alternative_groups, store_params, avail,
                                event_names, sel_idx, param_index,
                                export: str = "export") -> None:
    """Raise :class:`MissingParameterError` if, for any *alternative group*, no
    alternative is satisfied for some selected event.

    This is the "any-of" companion to :func:`check_required` (which it does NOT
    modify): use it for requirements like "``cos_tilt_1`` OR ``tilt_1``" or
    "``chi_p`` OR its ingredient set".

    Parameters
    ----------
    alternative_groups : sequence of groups
        Each *group* is a tuple of *alternatives*; the group is satisfied for an
        event when ANY of its alternatives is.  An alternative is either a
        single parameter name (``str``) or a tuple of names that must ALL be
        present+available together (an "and" bundle).  Example::

            (("cos_tilt_1", "tilt_1"), ("cos_tilt_2", "tilt_2"))     # two groups
            (("chi_p", ("a_1", "a_2", "cos_tilt_1", "cos_tilt_2")),) # one group

    store_params, avail, event_names, sel_idx, param_index, export
        As in :func:`check_required`.
    """
    import numpy as np

    sel = np.asarray(sel_idx)
    if sel.size == 0:
        return

    problems = []
    for group in alternative_groups:
        alts = list(group)
        satisfied = np.zeros(sel.size, dtype=bool)
        for alt in alts:
            names = (alt,) if isinstance(alt, str) else tuple(alt)
            if all(p in param_index for p in names):
                cols = [avail[sel, param_index[p]] for p in names]
                alt_ok = (np.logical_and.reduce(cols) if cols
                          else np.ones(sel.size, dtype=bool))
            else:
                alt_ok = np.zeros(sel.size, dtype=bool)
            satisfied |= alt_ok
        if not satisfied.all():
            bad = sorted(np.asarray(event_names)[sel][~satisfied].tolist())
            desc = " or ".join(_describe_alternative(a) for a in alts)
            problems.append(f"[{desc}] for event(s) {bad}")
    if problems:
        raise MissingParameterError(
            f"{export} requires at least one of each alternative group, but no "
            f"alternative is present+available for some selected events: "
            + "; ".join(problems)
            + ". Provide one alternative per group (or drop the events).")
