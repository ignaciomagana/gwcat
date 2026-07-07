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
    spin:            a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, chi_eff, chi_p
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
             "chi_eff", "chi_p"),
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

#: export name -> ordered tuple of REQUIRED parameters.
EXPORT_REQUIREMENTS = {
    "darksirens": DARKSIRENS_REQUIRED,
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
