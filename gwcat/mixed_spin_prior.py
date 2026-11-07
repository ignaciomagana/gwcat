"""Each sample of a combined ("Mixed") set divided by its OWN analysis's spin prior.

GWTC-2.1 and GWTC-3 ``C01:Mixed`` concatenate the posterior samples of two
analyses: ``C01:IMRPhenomXPHM`` (bilby; ``a_i ~ U(0, 0.99)``, isotropic
tilts, declared in its ``priors/analytic`` group) and ``C01:SEOBNRv4PHM``
(RIFT; no analytic group, and a config that declares
``a_spin{1,2}-max = 0.99``).  The SEOBNRv4PHM samples do not stop at 0.99:
over the 72 releases that carry this constituent every set has samples above
0.99 (the lowest maximum is 0.9936, the median 0.9998), and the pooled
magnitude density is flat through the declared edge (1.03 and 1.01 times its
value in [0.97, 0.975) in the two 0.005-wide bins above 0.99) with the same
near-uniform shape below it as the IMRPhenomXPHM half.  Its prior is uniform in
magnitude to the Kerr bound, 1, and isotropic.

The store used to give the whole Mixed set ONE box -- XPHM's 0.99, widened to
the largest drawn magnitude (``spin_amax_source = samples_bound``) -- which
describes the SEOBNRv4PHM half and misdescribes the XPHM half by the ratio of
the two boxes, ``(0.99 / 1)^2``: about 2% in each XPHM row's weight, most of it
a per-event constant (review 2026-10-20, Low 77 / high-2).  This module gives
each row the ceiling of the analysis it came from, so the component-basis
``p_pe`` divides out ``1 / (4 amax_1 amax_2)`` of that analysis (the
``own_prior`` reading of :mod:`gwcat.constituent_mixture`, applied to the spin
prior only; the shapes are the same, uniform magnitude and isotropic tilt).

Which analysis a Mixed row came from is not recorded in the release, so it is
established row by row and verified (:func:`attribute_mixed_rows`):

1. **exact** -- the row equals a released row of one constituent on
   ``(mass_1, mass_2, luminosity_distance)``;
2. **time** -- a constituent whose released rows ALL share one
   ``geocent_time`` (RIFT fixes the event time; every one of the 72
   SEOBNRv4PHM sets does) owns every Mixed row with that time;
3. **remainder** -- the rows left belong to the only other constituent that
   contributes exactly matched rows.

and refused (:class:`MixedSpinPriorError`) where the rules disagree: an
exactly matched row carrying another constituent's time signature, a row
exactly matched to a time-signature constituent without its time, rows left
over with no unique owner, or a row whose magnitude lies above its own
analysis's ceiling.  On the 72 releases no rule disagrees and no row outside
the SEOBNRv4PHM half exceeds 0.99 (its largest magnitude is 0.989999).

The ceilings (:func:`constituent_ceilings`): a constituent's analytic prior,
else its config's ``a_spin{1,2}-max``, when its samples lie inside it; the
Kerr bound when it has no analytic prior and its samples exceed the configured
ceiling (an analytic prior its own samples exceed is refused).

The per-sample ceilings travel as two store sample columns,
:data:`PER_SAMPLE_AMAX_PARAMS`, written by the ingest for a combined row whose
constituents' ceilings differ, or added to an existing store by
:func:`annotate_store` (``gwcat annotate-spin-priors``), which leaves every
other dataset of the store bitwise as it was.  The component-basis PE export
reads them (``gwcat.export.pe_builder``); the projection bases do not.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .constituent_mixture import MATCH_KEYS, _row_keys

#: The store sample columns that carry each sample's own spin-magnitude
#: ceilings (NaN, and unavailable, outside the annotated rows).
PER_SAMPLE_AMAX_PARAMS = ("spin_prior_amax_1", "spin_prior_amax_2")

#: Per-row provenance of the per-sample ceilings (JSON, "" for a row that is
#: not a combined set).
PER_SAMPLE_META_FIELD = "spin_prior_per_sample"

#: No spin magnitude exceeds it.
KERR_BOUND = 1.0

#: The time column a time-signature constituent is recognised by.
TIME_KEY = "geocent_time"

SPIN_KEYS = ("a_1", "a_2")


class MixedSpinPriorError(ValueError):
    """A combined set's rows cannot be given their own analysis's spin prior."""


@dataclass
class MixedSpinPrior:
    """Per-sample spin ceilings of one combined set (see the module docstring)."""

    amax_1: np.ndarray
    amax_2: np.ndarray
    labels: List[str]
    rows: np.ndarray
    ceilings: Dict[str, Tuple[float, float]]
    sources: Dict[str, str]
    counts: Dict[str, int]
    methods: Dict[str, int]
    time_signature: Dict[str, float] = field(default_factory=dict)

    def provenance(self) -> dict:
        return {"status": "per_sample",
                "constituents": list(self.labels),
                "ceilings": {k: [float(a), float(b)]
                             for k, (a, b) in self.ceilings.items()},
                "sources": dict(self.sources),
                "rows": {k: int(v) for k, v in self.counts.items()},
                "attribution": {k: int(v) for k, v in self.methods.items()},
                "time_signature": {k: float(v)
                                   for k, v in self.time_signature.items()}}

    def to_json(self) -> str:
        return json.dumps(self.provenance(), sort_keys=True)


def _maxabs(samples: dict, key: str) -> Optional[float]:
    if samples is None or key not in samples:
        return None
    v = np.asarray(samples[key], dtype=float)
    if v.size == 0:
        return None
    return float(np.nanmax(np.abs(v)))


def attribute_mixed_rows(mixed: dict, constituents: Dict[str, dict]):
    """Which constituent each Mixed row came from (module docstring, rules 1-3).

    Returns ``(labels, rows, methods, time_signature)``: the contributing
    constituents (in ``constituents`` order), the index into ``labels`` of each
    row's constituent, the number of rows each rule attributed, and the time of
    every time-signature constituent.  Raises :class:`MixedSpinPriorError`
    when the rules disagree or leave a row without a unique owner.
    """
    missing = [k for k in MATCH_KEYS if k not in mixed]
    if missing:
        raise MixedSpinPriorError(
            f"cannot attribute the Mixed rows: columns {missing} are absent.")
    names = list(constituents)
    n = len(np.asarray(mixed[MATCH_KEYS[0]]))
    owner = np.full(n, -1, dtype=np.int64)
    method = np.zeros(n, dtype=np.int64)          # 1 exact, 2 time, 3 remainder
    mk = _row_keys(mixed)
    seen: Dict[bytes, int] = {}
    for j, lab in enumerate(names):
        smp = constituents[lab]
        if smp is None or any(k not in smp for k in MATCH_KEYS):
            continue
        for kb in _row_keys(smp):
            prev = seen.get(kb)
            if prev is not None and prev != j:
                raise MixedSpinPriorError(
                    f"a released row of {names[prev]} equals one of {lab}; "
                    f"the Mixed rows cannot be attributed by row equality.")
            seen[kb] = j
    for i, kb in enumerate(mk):
        j = seen.get(kb)
        if j is not None:
            owner[i] = j
            method[i] = 1

    # Rule 2: time signatures.
    tsig: Dict[int, float] = {}
    for j, lab in enumerate(names):
        smp = constituents[lab]
        if smp is None or TIME_KEY not in smp:
            continue
        t = np.asarray(smp[TIME_KEY], dtype=float)
        if t.size and np.all(t == t[0]):
            tsig[j] = float(t[0])
    if len({v for v in tsig.values()}) != len(tsig):
        raise MixedSpinPriorError(
            "two constituents share one time signature; it does not identify "
            "either.")
    if tsig:
        if TIME_KEY not in mixed:
            raise MixedSpinPriorError(
                f"constituents {[names[j] for j in tsig]} carry a time "
                f"signature but the Mixed set has no {TIME_KEY} column.")
        tm = np.asarray(mixed[TIME_KEY], dtype=float)
        for j, t0 in tsig.items():
            has_t = tm == t0
            if np.any((owner == j) & ~has_t):
                raise MixedSpinPriorError(
                    f"{int(np.sum((owner == j) & ~has_t))} Mixed row(s) equal "
                    f"released rows of {names[j]} but lack its time signature "
                    f"{t0!r}.")
            clash = has_t & (owner >= 0) & (owner != j)
            if np.any(clash):
                raise MixedSpinPriorError(
                    f"{int(clash.sum())} Mixed row(s) equal released rows of "
                    f"another constituent but carry the time signature of "
                    f"{names[j]}.")
            take = has_t & (owner < 0)
            owner[take] = j
            method[take] = 2

    # Rule 3: the remainder.
    left = owner < 0
    if np.any(left):
        exact_owners = {int(j) for j in np.unique(owner[method == 1])}
        cand = sorted(exact_owners - set(tsig))
        if not tsig or len(cand) != 1:
            raise MixedSpinPriorError(
                f"{int(left.sum())} of {n} Mixed row(s) equal no released "
                f"constituent row and carry no time signature, and no single "
                f"other constituent can own them (candidates "
                f"{[names[j] for j in cand]}, time signatures "
                f"{[names[j] for j in tsig]}).")
        owner[left] = cand[0]
        method[left] = 3

    used = [j for j in range(len(names)) if np.any(owner == j)]
    pos = {j: k for k, j in enumerate(used)}
    rows = np.fromiter((pos[int(j)] for j in owner), dtype=np.int64, count=n)
    labels = [names[j] for j in used]
    methods = {"exact": int(np.sum(method == 1)),
               "time_signature": int(np.sum(method == 2)),
               "remainder": int(np.sum(method == 3))}
    return labels, rows, methods, {names[j]: t for j, t in tsig.items()}


def constituent_ceilings(label, analytic, config, sample_max):
    """``((amax_1, amax_2), source)`` of one constituent's spin-magnitude prior.

    ``analytic`` / ``config``: the ``(amax_1, amax_2)`` its analytic prior / its
    config declares, or None; ``sample_max``: the largest ``(|a_1|, |a_2|)``
    of its samples (released and attributed).  See the module docstring.
    """
    s1, s2 = sample_max
    if analytic is not None:
        a1, a2 = float(analytic[0]), float(analytic[1])
        if s1 > a1 or s2 > a2:
            raise MixedSpinPriorError(
                f"{label}: its samples reach ({s1:.6g}, {s2:.6g}), above the "
                f"ceilings its own analytic prior declares ({a1:g}, {a2:g}).")
        return (a1, a2), "analytic"
    if config is not None:
        c1, c2 = float(config[0]), float(config[1])
        if s1 <= c1 and s2 <= c2:
            return (c1, c2), "config_file"
        return ((KERR_BOUND, KERR_BOUND),
                f"kerr_bound(samples {max(s1, s2):.6g} > config "
                f"{max(c1, c2):g})")
    raise MixedSpinPriorError(
        f"{label}: declares no spin prior (no analytic group, no config "
        f"ceilings), so its rows cannot be given their own.")


def resolve_mixed_spin_prior(mixed: dict, constituents: Dict[str, dict],
                             analytic: Dict[str, Optional[tuple]],
                             config: Dict[str, Optional[tuple]]):
    """The per-sample spin ceilings of a combined set, or None.

    None when the contributing constituents share one ceiling (the event's
    single box is then exact, as before).  Raises :class:`MixedSpinPriorError`
    when the rows cannot be attributed or a row lies above its own ceiling.
    """
    labels, rows, methods, tsig = attribute_mixed_rows(mixed, constituents)
    a1m = np.abs(np.asarray(mixed["a_1"], dtype=float))
    a2m = np.abs(np.asarray(mixed["a_2"], dtype=float))
    ceilings, sources, counts = {}, {}, {}
    for k, lab in enumerate(labels):
        mine = rows == k
        smp = constituents.get(lab)
        s1 = max(x for x in (_maxabs(smp, "a_1"),
                             float(a1m[mine].max()) if mine.any() else None,
                             0.0) if x is not None)
        s2 = max(x for x in (_maxabs(smp, "a_2"),
                             float(a2m[mine].max()) if mine.any() else None,
                             0.0) if x is not None)
        ceilings[lab], sources[lab] = constituent_ceilings(
            lab, analytic.get(lab), config.get(lab), (s1, s2))
        counts[lab] = int(mine.sum())
    if len(set(ceilings.values())) <= 1:
        return None
    c1 = np.array([ceilings[lab][0] for lab in labels], dtype=float)
    c2 = np.array([ceilings[lab][1] for lab in labels], dtype=float)
    amax_1, amax_2 = c1[rows], c2[rows]
    bad = (a1m > amax_1) | (a2m > amax_2)
    if bad.any():
        raise MixedSpinPriorError(
            f"{int(bad.sum())} Mixed row(s) lie above their own analysis's "
            f"spin ceiling.")
    return MixedSpinPrior(amax_1=amax_1, amax_2=amax_2, labels=labels,
                          rows=rows, ceilings=ceilings, sources=sources,
                          counts=counts, methods=methods, time_signature=tsig)


# ---------------------------------------------------------------------------
# Reading a release file directly (the store annotation)
# ---------------------------------------------------------------------------
_READ_COLS = MATCH_KEYS + (TIME_KEY,) + SPIN_KEYS


def _label_parts(label):
    bits = str(label).split(":")
    return (bits[0] if bits else "", bits[1] if len(bits) > 1 else "",
            ":".join(bits[2:]) if len(bits) > 2 else "")


def _is_combined(label) -> bool:
    return _label_parts(label)[1].startswith("Mixed")


def _read_cols(group, cols=_READ_COLS) -> dict:
    ps = group["posterior_samples"]
    have = ps.dtype.names or ()
    return {c: np.asarray(ps[c][:], dtype=np.float64) for c in cols
            if c in have}


def _decode(x):
    if isinstance(x, (bytes, bytearray)):
        return x.decode()
    if isinstance(x, np.ndarray):
        x = x.ravel()
        return _decode(x[0]) if x.size else ""
    return str(x)


def _analytic_ceilings(group) -> Optional[tuple]:
    from .ingest import _parse_analytic_spin
    try:
        an = group["priors"]["analytic"]
    except KeyError:
        return None
    out = []
    for k in SPIN_KEYS:
        if k not in an:
            return None
        p = _parse_analytic_spin(_decode(an[k][()]))
        if p is None or p[0] != "Uniform" or p[2] is None:
            return None
        out.append(float(p[2]))
    return tuple(out)


def _config_ceilings(group) -> Optional[tuple]:
    try:
        eng = group["config_file"]["engine"]
    except KeyError:
        return None
    out = []
    for k in ("a_spin1-max", "a_spin2-max"):
        if k not in eng:
            return None
        try:
            out.append(float(_decode(eng[k][()]).strip().strip("'\"")))
        except ValueError:
            return None
    return tuple(out)


def release_mixed_spin_prior(path: str, mixed_label: str,
                             constituents: Sequence[str] = ()):
    """:func:`resolve_mixed_spin_prior` for ``mixed_label`` of a release file.

    The candidate constituents are ``constituents`` when given, else every
    same-prefix analysis of the file that is not itself a combined set (the
    GWTC-2.1/3 records name run tags, not labels).  Returns
    ``(MixedSpinPrior or None, mixed columns)``.
    """
    import h5py
    with h5py.File(path, "r") as f:
        if mixed_label not in f:
            raise MixedSpinPriorError(f"{path}: no analysis {mixed_label!r}.")
        mixed = _read_cols(f[mixed_label])
        pfx = _label_parts(mixed_label)[0]
        cands = list(constituents) or [
            g for g in f.keys() if g != mixed_label and _label_parts(g)[0] == pfx
            and not _is_combined(g) and isinstance(f[g], h5py.Group)
            and "posterior_samples" in f[g]]
        cons = {g: _read_cols(f[g]) for g in cands}
        analytic = {g: _analytic_ceilings(f[g]) for g in cands}
        config = {g: _config_ceilings(f[g]) for g in cands}
    return resolve_mixed_spin_prior(mixed, cons, analytic, config), mixed


def _sha256(path, chunk=16 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _find_release(pe_dirs: Iterable[str], file_name: str) -> str:
    hits = []
    for d in pe_dirs:
        hits += glob.glob(os.path.join(d, "**", file_name), recursive=True)
    hits = sorted(set(os.path.realpath(h) for h in hits))
    if len(hits) != 1:
        raise MixedSpinPriorError(
            f"{file_name}: found {len(hits)} release files under {list(pe_dirs)}"
            f" ({hits[:3]}); need exactly one.")
    return hits[0]


def annotate_store(store_in: str, store_out: str, pe_dirs: Sequence[str], *,
                   names: Optional[Sequence[str]] = None,
                   verify_checksum: bool = True) -> dict:
    """Copy ``store_in`` to ``store_out`` with each combined row's per-sample
    spin ceilings added (:data:`PER_SAMPLE_AMAX_PARAMS`).

    Every combined (``Mixed``) row of the store -- of the events in ``names``
    when given -- is read back from its release file (found by
    ``meta/file_name`` under ``pe_dirs`` and, with ``verify_checksum``, matched
    to ``meta/file_checksum``), its rows checked to be the store's rows in the
    store's order, and given :func:`release_mixed_spin_prior`'s ceilings.  A
    row whose constituents share one ceiling, or that cannot be attributed,
    keeps no per-sample ceilings (its status says which, in
    ``meta/spin_prior_per_sample``) and exports exactly as before.  Every
    dataset ``store_in`` has is copied unchanged; ``store_out`` must not exist.
    Returns a summary dict.
    """
    import h5py
    if os.path.exists(store_out):
        raise FileExistsError(f"refusing to overwrite {store_out}")
    if os.path.realpath(store_in) == os.path.realpath(store_out):
        raise ValueError("store_out must differ from store_in")
    with h5py.File(store_in, "r") as f:
        params = [_decode(p) for p in f.attrs["param_names"]]
        if any(p in params for p in PER_SAMPLE_AMAX_PARAMS):
            raise MixedSpinPriorError(
                f"{store_in} already carries {PER_SAMPLE_AMAX_PARAMS}.")
        ev = [_decode(x) for x in f["index/event_names"][:]]
        off = f["index/offsets"][:]
        an = [_decode(x) for x in f["meta/analysis_used"][:]]
        fn = [_decode(x) for x in f["meta/file_name"][:]]
        ck = ([_decode(x) for x in f["meta/file_checksum"][:]]
              if "meta/file_checksum" in f else [""] * len(ev))
        n_tot = int(off[-1])
        a1 = np.full(n_tot, np.nan)
        a2 = np.full(n_tot, np.nan)
        status = [""] * len(ev)
        summary = {"store_in": os.path.abspath(store_in),
                   "store_in_sha256": _sha256(store_in), "rows": {}}
        want = None if names is None else set(names)
        for i, (name, lab) in enumerate(zip(ev, an)):
            if not _is_combined(lab):
                continue
            if want is not None and name not in want:
                status[i] = json.dumps({"status": "not_requested"})
                continue
            try:
                path = _find_release(pe_dirs, fn[i])
                got = _sha256(path)
                if verify_checksum and ck[i]:
                    if got != ck[i]:
                        raise MixedSpinPriorError(
                            f"{path}: sha256 {got[:12]} is not the ingested "
                            f"file's {ck[i][:12]}.")
                msp, mixed = release_mixed_spin_prior(path, lab)
                lo, hi = int(off[i]), int(off[i + 1])
                for c in ("mass_1", "luminosity_distance", "a_1", "a_2"):
                    sv = f[f"samples/{c}"][lo:hi]
                    if sv.shape != mixed[c].shape or not np.array_equal(
                            sv, mixed[c]):
                        raise MixedSpinPriorError(
                            f"the store's {c} of {name} [{lab}] is not the "
                            f"release's {lab} column in order.")
                if msp is None:
                    status[i] = json.dumps({"status": "single_ceiling"})
                else:
                    a1[lo:hi], a2[lo:hi] = msp.amax_1, msp.amax_2
                    prov = msp.provenance()
                    prov["release_file"] = fn[i]
                    prov["release_sha256"] = got
                    status[i] = json.dumps(prov, sort_keys=True)
            except (MixedSpinPriorError, KeyError, OSError) as exc:
                status[i] = json.dumps({"status": "unresolved",
                                        "reason": str(exc)})
            summary["rows"][f"{name} [{lab}]"] = json.loads(status[i])
        ds_ref = f["samples/luminosity_distance"]
        layout = dict(chunks=ds_ref.chunks, compression=ds_ref.compression,
                      compression_opts=ds_ref.compression_opts,
                      shuffle=ds_ref.shuffle)
        n_ev = len(ev)
        avail_old = (np.asarray(f["avail/mask"][:], dtype=bool)
                     if "avail" in f and "mask" in f["avail"] else None)

    tmp = store_out + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    shutil.copyfile(store_in, tmp)
    with h5py.File(tmp, "r+") as g:
        g.create_dataset(f"samples/{PER_SAMPLE_AMAX_PARAMS[0]}", data=a1,
                         **layout)
        g.create_dataset(f"samples/{PER_SAMPLE_AMAX_PARAMS[1]}", data=a2,
                         **layout)
        row_has = np.array([np.isfinite(a1[int(off[i]):int(off[i + 1])]).all()
                            and off[i + 1] > off[i] for i in range(n_ev)])
        if avail_old is not None:
            mask = np.concatenate([avail_old, np.stack([row_has, row_has], 1)],
                                  axis=1)
            del g["avail/mask"]
            g.create_dataset("avail/mask", data=mask)
        new_params = params + list(PER_SAMPLE_AMAX_PARAMS)
        g.attrs["param_names"] = np.array(new_params,
                                          dtype=h5py.string_dtype())
        if PER_SAMPLE_META_FIELD in g["meta"]:
            # A store ingested with per_sample_spin_prior off carries the
            # field empty for every row.
            del g[f"meta/{PER_SAMPLE_META_FIELD}"]
        g.create_dataset(f"meta/{PER_SAMPLE_META_FIELD}",
                         data=np.array(status, dtype=object),
                         dtype=h5py.string_dtype())
        try:
            from . import __version__ as _ver
        except Exception:                            # pragma: no cover
            _ver = ""
        g.attrs["spin_prior_per_sample_annotation"] = json.dumps({
            "source_store": summary["store_in"],
            "source_store_sha256": summary["store_in_sha256"],
            "pe_dirs": [os.path.abspath(d) for d in pe_dirs],
            "gwcat_version": str(_ver),
            "n_rows_annotated": int(row_has.sum())}, sort_keys=True)
    os.replace(tmp, store_out)
    summary["store_out"] = os.path.abspath(store_out)
    summary["n_rows_annotated"] = int(row_has.sum())
    summary["n_rows_combined"] = int(sum(1 for s in status if s))
    return summary
