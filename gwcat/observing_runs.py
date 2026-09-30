"""Observing-run time windows, and the run a GPS time belongs to (GW-39).

Two tables, deliberately kept apart because they answer different questions.

``EXPOSURE_WINDOWS_GPS`` -- the OFFICIAL run definitions.  Events are assigned
to runs with this table, and an injection campaign's exposure is a statement
about these windows.

  * O1 ``[1126051217, 1137254417]`` = 2015-09-12 00:00 UTC to 2016-01-19 16:00
    UTC, and O2 ``[1164556817, 1187733618]`` = 2016-11-30 16:00 UTC to
    2017-08-25 22:00 UTC: the GWOSC run definitions (the O1 and O2 strain
    data-release documentation at gwosc.org; "Open data from the first and
    second observing runs of Advanced LIGO and Advanced Virgo", SoftwareX 13
    (2021) 100658).
  * O3a ``[1238166018, 1253977218]``, O3b ``[1256655618, 1269363618]``, O4a
    ``[1368975618, 1389456018]`` and O4b ``[1396969218, 1422118818]``: the
    "officially defined time ranges" of the GWTC-5.0 cumulative sensitivity
    release README (Zenodo 19500052, ``gwtc-5_o1234ab_sensitivity-estimates.md``,
    md5 ``5ffdec765b6b3b4935f88d88ac11a3cc``), which states that O3/O4
    injections outside them were removed.  The GPS numbers are authoritative;
    two of the README's human-readable UTC labels disagree with them (O3b ends
    at GPS 1269363618 = 2020-03-27 17:00 UTC, labelled "17 Mar 2020"; O4b starts
    at GPS 1396969218 = 2024-04-12 15:00 UTC, labelled "10 Apr 2024").

``INJECTION_SUPPORT_GPS`` -- where a campaign's injections actually LIE in
time.  Rows are assigned to runs with this table.  For O3a-O4b it is the same
README window (the release cut the rows to it).  For O1 and O2 it is the
support of the semianalytic rows of that release, which is NARROWER than the
official run: O1 ``[1126627425, 1136649608]`` (2015-09-18 16:03 to 2016-01-12
16:00 UTC) and O2 ``[1164556874, 1187733519]``.  The loader re-verifies every
row against this table at runtime (:func:`run_of_gps` raises on a row outside
it), so a table that stopped describing the file cannot pass silently.

Why two: an event can be inside its run's exposure window and still outside
the injection support.  GW150914 (GPS 1126259462.4, 2015-09-14, ER8) is the
live case -- inside O1's exposure window, before the first O1 semianalytic
injection.  Whether the O1 exposure nevertheless covers it is a question about
the O1 livetime, not about the timestamps, and it is answered elsewhere; this
module only makes the distinction impossible to lose.
"""
from __future__ import annotations

import numpy as np

#: The run labels, in time order.
RUN_LABELS = ("O1", "O2", "O3a", "O3b", "O4a", "O4b")

#: Runs whose injections are semianalytic (Essick 2023, PRD 108 043011) in the
#: GWTC-5.0 cumulative mixture: detection there is the simulated observed SNR,
#: and no search FAR exists.
SEMIANALYTIC_RUNS = ("O1", "O2")

#: Runs whose injections are real (processed by the searches): detection there
#: is the minimum search FAR over that run's own searches.
REAL_RUNS = ("O3a", "O3b", "O4a", "O4b")

#: The prefix of a run's search names in the mixture's ``searches`` attr (and
#: of its ``<search>_far`` columns): O3 searches are shared by O3a and O3b.
RUN_SEARCH_PREFIX = {"O3a": "o3_", "O3b": "o3_", "O4a": "o4a_", "O4b": "o4b_"}

#: Official run definitions (closed intervals, GPS seconds).  See the module
#: docstring for the sources.
EXPOSURE_WINDOWS_GPS = {
    "O1": (1126051217.0, 1137254417.0),
    "O2": (1164556817.0, 1187733618.0),
    "O3a": (1238166018.0, 1253977218.0),
    "O3b": (1256655618.0, 1269363618.0),
    "O4a": (1368975618.0, 1389456018.0),
    "O4b": (1396969218.0, 1422118818.0),
}

#: Where the GWTC-5.0 cumulative mixture's injections lie (closed intervals,
#: GPS seconds).  See the module docstring.
INJECTION_SUPPORT_GPS = {
    "O1": (1126627425.0, 1136649608.0),
    "O2": (1164556874.0, 1187733519.0),
    "O3a": EXPOSURE_WINDOWS_GPS["O3a"],
    "O3b": EXPOSURE_WINDOWS_GPS["O3b"],
    "O4a": EXPOSURE_WINDOWS_GPS["O4a"],
    "O4b": EXPOSURE_WINDOWS_GPS["O4b"],
}

#: The exposure components of the cumulative mixture's per-row ``weights``,
#: the runs each covers, and what its duration T_k MEANS.  The weights are
#: ``(T_k/T)/(N_k/N)`` per component (Essick 2021, RNAAS 5 220), and the three
#: kinds of T_k are not the same physical quantity, so each is labelled:
#:
#:   * O1, O2 -- coincident livetime: the semianalytic estimates do not model
#:     the detector duty cycle (README), so T is the coincident time itself;
#:   * O3 -- the analysis time of the endo3 multi-population campaign whose
#:     rows these are;
#:   * O4 -- monthly wall-clock time ("constructed based on wall-time within
#:     each month", README), which includes breaks and is NOT an observing time.
#: Largest |N_k - round(N_k)| accepted for the solved O1/O2 draw counts.
N_INTEGRALITY_TOL = 1e-5

MIXTURE_COMPONENTS = (
    ("O1", ("O1",), "coincident_livetime_semianalytic"),
    ("O2", ("O2",), "coincident_livetime_semianalytic"),
    ("O3", ("O3a", "O3b"), "endo3_analysis_time"),
    ("O4", ("O4a", "O4b"), "monthly_wall_clock"),
)

#: The two component campaigns of a released cumulative mixture whose (T, N)
#: are published separately, keyed by the mixture's own
#: ``(total_generated, total_analysis_time)``.  With them the per-row weights
#: determine the O1 and O2 (T, N) exactly -- see
#: :func:`derive_mixture_bookkeeping`, which checks that the O3 anchor
#: reproduces the file's O3 weight and that the solved N_O1, N_O2 are integers,
#: so a wrong anchor cannot pass unnoticed.
#:
#: GWTC-5.0 cumulative O1-O4b (Zenodo 19500052, both spin flavours):
#:   * O3 = ``endo3_mixture-LIGO-T2100113-v12.hdf5`` (Zenodo 7890437) attrs
#:     ``analysis_time_s`` = 28749576, ``total_generated`` = 86318296;
#:   * O4 = ``samples-rpo4ab-1366933504-55469568-clipped.hdf`` (Zenodo
#:     19500064) attrs ``total_analysis_time`` = 52962816,
#:     ``total_generated`` = 870454872.
MIXTURE_RELEASE_ANCHORS = {
    (1568035640, 96114016): {
        "release": "zenodo:19500052",
        "O3": {"T_s": 28749576.0, "N": 86318296,
               "source": "zenodo:7890437 endo3_mixture-LIGO-T2100113-v12.hdf5 "
                         "attrs analysis_time_s, total_generated"},
        "O4": {"T_s": 52962816.0, "N": 870454872,
               "source": "zenodo:19500064 samples-rpo4ab-1366933504-55469568-"
                         "clipped.hdf attrs total_analysis_time, "
                         "total_generated"},
    },
}


class RunWindowError(ValueError):
    """A GPS time lies outside every window of the table it was assigned with."""


def _check_table(table):
    """Refuse a window table whose intervals are malformed or overlap."""
    items = sorted(table.items(), key=lambda kv: kv[1][0])
    for label, (lo, hi) in items:
        if not (np.isfinite(lo) and np.isfinite(hi) and lo <= hi):
            raise ValueError(f"window {label!r} = ({lo}, {hi}) is malformed")
    for (la, (_, ha)), (lb, (lob, _)) in zip(items, items[1:]):
        if lob <= ha:
            raise ValueError(f"windows {la!r} and {lb!r} overlap")


_check_table(EXPOSURE_WINDOWS_GPS)
_check_table(INJECTION_SUPPORT_GPS)
for _run, (_lo, _hi) in INJECTION_SUPPORT_GPS.items():
    _elo, _ehi = EXPOSURE_WINDOWS_GPS[_run]
    if not (_elo <= _lo and _hi <= _ehi):     # pragma: no cover - table typo
        raise ValueError(f"{_run}: injection support ({_lo}, {_hi}) is not "
                         f"inside its exposure window ({_elo}, {_ehi})")


def run_of_gps(t, table=INJECTION_SUPPORT_GPS):
    """The run label of each GPS time ``t`` under ``table``.

    ``table`` maps a run label to a closed ``(start, end)`` GPS interval --
    :data:`INJECTION_SUPPORT_GPS` (default; for injection rows) or
    :data:`EXPOSURE_WINDOWS_GPS` (for events).  Returns a numpy array of labels
    with the shape of ``t`` (a plain ``str`` for a scalar).

    Raises
    ------
    RunWindowError
        When ANY time is outside every window of ``table`` (or not finite).
        There is no "unassigned" label: a row whose run is unknown cannot be
        given a detection rule, and an event whose run is unknown cannot be
        paired with an exposure.
    """
    _check_table(table)
    arr = np.asarray(t, dtype=float)
    flat = np.atleast_1d(arr).ravel()
    width = max(len(k) for k in table)
    out = np.full(flat.shape, "", dtype=f"<U{width}")
    assigned = np.zeros(flat.shape, dtype=bool)
    for label, (lo, hi) in table.items():
        m = (flat >= lo) & (flat <= hi)
        out[m] = label
        assigned |= m
    if not assigned.all():
        bad = np.flatnonzero(~assigned)
        which = ("INJECTION_SUPPORT_GPS" if table is INJECTION_SUPPORT_GPS
                 else "EXPOSURE_WINDOWS_GPS" if table is EXPOSURE_WINDOWS_GPS
                 else "the given table")
        raise RunWindowError(
            f"{bad.size} of {flat.size} GPS time(s) lie outside every window "
            f"of {which} (first at index {int(bad[0])}: "
            f"{flat[bad[0]]!r}); windows = "
            + ", ".join(f"{k} [{lo:.0f}, {hi:.0f}]"
                        for k, (lo, hi) in table.items()))
    if arr.ndim == 0:
        return str(out[0])
    return out.reshape(arr.shape)


def derive_mixture_bookkeeping(run, weights, total_generated,
                               total_analysis_time, anchors):
    """Per-component (N_k, T_k) of a cumulative mixture, from its weights.

    The mixture weight of a row in component ``k`` is
    ``w_k = (T_k/T) / (N_k/N)`` with ``T`` = ``total_analysis_time`` and ``N`` =
    ``total_generated``.  Given the separately published (T, N) of the O3 and O4
    component campaigns (``anchors``, see :data:`MIXTURE_RELEASE_ANCHORS`), the
    O1 and O2 components follow from two linear equations::

        N_O1 + N_O2         = N - N_O3 - N_O4
        w_O1 N_O1 + w_O2 N_O2 = (T - T_O3 - T_O4) N / T

    and ``T_k = w_k N_k T / N``.  Two checks keep the anchors honest: the O3
    anchor must reproduce the file's (single) O3 weight to 1e-10 relative, and
    the solved N_O1, N_O2 must be integers to 1e-3 -- a wrong anchor lands them
    far from any integer.

    Returns a dict with ``components`` (labels), ``N`` and ``T_s`` (floats, in
    component order), ``T_definition``, ``component_runs`` (JSON-able), the
    integrality residuals and the anchor sources.

    Raises
    ------
    ValueError
        When a semianalytic run or O3 does not carry a single weight, or either
        check fails.
    """
    run = np.asarray(run)
    weights = np.asarray(weights, dtype=float)
    N = float(total_generated)
    T = float(total_analysis_time)

    def _single_weight(runs):
        m = np.isin(run, runs)
        vals = np.unique(weights[m])
        if vals.size != 1:
            raise ValueError(
                f"mixture component {runs}: expected ONE weight value, found "
                f"{vals.size} ({vals[:5].tolist()}...)")
        return float(vals[0])

    w1 = _single_weight(["O1"])
    w2 = _single_weight(["O2"])
    w3 = _single_weight(["O3a", "O3b"])
    T3, N3 = float(anchors["O3"]["T_s"]), float(anchors["O3"]["N"])
    T4, N4 = float(anchors["O4"]["T_s"]), float(anchors["O4"]["N"])

    w3_pred = (T3 / T) / (N3 / N)
    if not np.isclose(w3_pred, w3, rtol=1e-10, atol=0.0):
        raise ValueError(
            f"the O3 anchor (T={T3}, N={N3}) predicts an O3 mixture weight "
            f"{w3_pred!r}, but the file's O3 rows carry {w3!r}: the anchor "
            f"does not describe this file.")

    if np.isclose(w1, w2, rtol=1e-12, atol=0.0):
        raise ValueError(
            f"the O1 and O2 mixture weights are equal ({w1!r}), so the two "
            f"bookkeeping equations are degenerate and N_O1, N_O2 cannot be "
            f"separated.")
    T12, N12 = T - T3 - T4, N - N3 - N4
    A = np.array([[1.0, 1.0], [w1, w2]])
    b = np.array([N12, T12 * N / T])
    N1, N2 = np.linalg.solve(A, b)
    resid = (float(N1 - np.round(N1)), float(N2 - np.round(N2)))
    # The release's own residual is 7.2e-7 (float64 weights); 1e-5 leaves a
    # margin for rounding while still catching any anchor that is off by one
    # draw (the O4 anchor is checked only through this test).
    if max(abs(r) for r in resid) > N_INTEGRALITY_TOL or min(N1, N2) <= 0:
        raise ValueError(
            f"solving the mixture weights with the O3/O4 anchors gives "
            f"N_O1={N1!r}, N_O2={N2!r}, which are not positive integers "
            f"(residuals {resid}): the anchors do not describe this file.")
    T1, T2 = w1 * N1 * T / N, w2 * N2 * T / N

    comps = [c for c, _, _ in MIXTURE_COMPONENTS]
    return {
        "components": comps,
        "component_runs": {c: list(r) for c, r, _ in MIXTURE_COMPONENTS},
        "T_definition": [d for _, _, d in MIXTURE_COMPONENTS],
        "N": [float(N1), float(N2), N3, N4],
        "T_s": [float(T1), float(T2), T3, T4],
        "N_integrality_residual": list(resid),
        "anchor_sources": {"O3": anchors["O3"]["source"],
                           "O4": anchors["O4"]["source"]},
    }
