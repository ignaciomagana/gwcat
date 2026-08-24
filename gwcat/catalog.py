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

import contextlib
import json
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import h5py

from .cosmology import make_cosmology, z_of_dL
from .source_class import (normalize_source_class, resolve_filter_classes,
                           format_source_class_filter, load_event_list,
                           SOURCE_CLASSES, CUT_ESTIMATOR_ATTR,
                           pe_cut_estimator, assess_cut_estimator_pair,
                           PE_MEDIAN_CUT_WARNING)


class UnpairableSelectionCut(ValueError):
    """An event cut for which the paired injection campaign has no equivalent.

    Raised by the exporters, not by :meth:`GWCatalog.select`: cutting a view is
    always legal, and the cut only becomes a defect when the resulting event
    list is handed to a selection function that cannot reproduce it.
    """


def _tighter_upper(a, b):
    """The tighter of two upper bounds (``None`` = no bound)."""
    if a is None:
        return None if b is None else float(b)
    return float(a) if b is None else float(min(float(a), float(b)))


def _tighter_lower(a, b):
    """The tighter of two lower bounds (``None`` = no bound)."""
    if a is None:
        return None if b is None else float(b)
    return float(a) if b is None else float(max(float(a), float(b)))


def _intersect_range(a, b):
    """The intersection of two ``(lo, hi)`` windows (``None`` = no window)."""
    if a is None:
        return None if b is None else (float(b[0]), float(b[1]))
    if b is None:
        return (float(a[0]), float(a[1]))
    return (max(float(a[0]), float(b[0])), min(float(a[1]), float(b[1])))


def _round(x):
    """A float rounded to the digest's precision, or ``None``."""
    return None if x is None else round(float(x), 10)


@dataclass(frozen=True)
class SelectionSpec:
    """The EFFECTIVE event selection behind a view: every cut, accumulated.

    :meth:`GWCatalog.select` intersects rows correctly -- it always ANDs its
    mask with the view it was called on -- but the provenance it attached used
    to describe only the LAST call's arguments.  Export then calls ``select()``
    once more with its own (defaulted) arguments, so a product built from
    ``cat.select(source_class="bbh")`` kept all 273 BBH rows while recording an
    empty ``source_class_filter``, no cut estimator and no event list: a
    filtered file advertising itself as unfiltered, which a paired selection
    function has no way to contradict.

    The spec is immutable and composable.  :meth:`refine` returns a NEW spec --
    this one intersected with one further ``select()`` call -- so the effective
    cut survives any chain of views:

      * thresholds compose to the TIGHTER value, because that is the cut the
        surviving rows actually reflect (two chained ``far_max`` cuts leave the
        smaller one standing);
      * ``source_class`` composes to the intersection of the resolved class
        sets, and an EMPTY intersection raises rather than serialising to the
        "" that every reader takes as "no restriction";
      * every name filter -- ``allowed_names``, the legacy ``names``, and
        ``event_list`` -- normalises into ONE set of names, also intersected, so
        the three spellings cannot describe three different things.

    Nothing here is a measurement of the resulting rows; it is the declaration
    of what was asked for.  ``n_missing_far`` is the one exception, kept
    alongside the FAR cut it belongs to.
    """

    #: Every ``compact_type`` gate actually applied, in call order.
    compact_type: tuple = ()
    #: Canonical class labels the composed request resolves to, or ``None``.
    source_class: Optional[tuple] = None
    #: Whether a class filter was requested at all (``"cbc"`` resolves to every
    #: class, which is not the same as never having asked).
    source_class_requested: bool = False
    far_max: Optional[float] = None
    snr_min: Optional[float] = None
    pastro_min: Optional[float] = None
    sky_area_max: Optional[float] = None
    m1_src_range: Optional[tuple] = None
    m2_src_range: Optional[tuple] = None
    #: The intersection of every name whitelist, sorted; ``None`` = none given.
    allowed_names: Optional[tuple] = None
    #: Labels of the name filters applied, in call order: ``"allowed_names"``,
    #: ``"names"``, or ``"event_list:<path-or-custom_sequence>"``.
    name_filters: tuple = ()
    allowed_names_authoritative: bool = True
    far_policy: str = "none"
    n_missing_far: int = 0
    allow_missing_far: bool = False
    require_far: bool = False
    waveform_policy: str = "preferred"
    approximant: Optional[str] = None

    # ---- composition ----------------------------------------------------
    def refine(self, *, compact_type=None, source_class=None, far_max=None,
               snr_min=None, pastro_min=None, sky_area_max=None,
               m1_src_range=None, m2_src_range=None, name_filters=(),
               allowed_names_authoritative=True, far_policy="none",
               n_missing_far=0, allow_missing_far=False, require_far=False,
               waveform_policy="preferred", approximant=None):
        """This spec intersected with one further ``select()`` call.

        ``name_filters`` is a sequence of ``(label, names)`` pairs -- one per
        name filter the call applied, already resolved to names.
        """
        classes, requested = self.source_class, self.source_class_requested
        if source_class is not None:
            resolved = resolve_filter_classes(source_class)
            new = tuple(c for c in SOURCE_CLASSES if c in resolved)
            if classes is None:
                classes = new
            else:
                merged = tuple(c for c in classes if c in set(new))
                if not merged:
                    raise ValueError(
                        f"source_class={source_class!r} selects "
                        f"{list(new)}, but this view is already restricted to "
                        f"{list(classes)}: the two share no class, so the "
                        f"selection is empty. An empty class restriction "
                        f"serialises identically to 'no restriction', so it "
                        f"would be exported as an UNFILTERED file holding zero "
                        f"events -- refusing instead. Select the classes you "
                        f"mean in one call.")
                classes = merged
            requested = True

        names, labels = self.allowed_names, self.name_filters
        authoritative = self.allowed_names_authoritative
        for label, seq in name_filters:
            new_names = tuple(sorted({str(x) for x in seq}))
            names = (new_names if names is None
                     else tuple(n for n in names if n in set(new_names)))
            labels = labels + (str(label),)
            authoritative = bool(allowed_names_authoritative)

        compact = self.compact_type
        if compact_type is not None and str(compact_type) not in compact:
            compact = compact + (str(compact_type),)

        # The FAR policy, the missing-FAR count and the two policy flags belong
        # to the call that actually applied a FAR cut; a later, FAR-less
        # select() must not overwrite them with its own defaults.
        if far_max is not None:
            far, policy = _tighter_upper(self.far_max, far_max), str(far_policy)
            n_miss = int(n_missing_far)
            allow_mf, req_far = bool(allow_missing_far), bool(require_far)
        elif self.far_max is not None:
            far, policy, n_miss = self.far_max, self.far_policy, self.n_missing_far
            allow_mf, req_far = self.allow_missing_far, self.require_far
        else:
            far, policy, n_miss = None, "none", 0
            allow_mf, req_far = bool(allow_missing_far), bool(require_far)

        return SelectionSpec(
            compact_type=compact,
            source_class=classes,
            source_class_requested=requested,
            far_max=far,
            snr_min=_tighter_lower(self.snr_min, snr_min),
            pastro_min=_tighter_lower(self.pastro_min, pastro_min),
            sky_area_max=_tighter_upper(self.sky_area_max, sky_area_max),
            m1_src_range=_intersect_range(self.m1_src_range, m1_src_range),
            m2_src_range=_intersect_range(self.m2_src_range, m2_src_range),
            allowed_names=names,
            name_filters=labels,
            allowed_names_authoritative=authoritative,
            far_policy=policy,
            n_missing_far=n_miss,
            allow_missing_far=allow_mf,
            require_far=req_far,
            waveform_policy=str(waveform_policy),
            approximant=None if approximant is None else str(approximant),
        )

    # ---- derived views --------------------------------------------------
    @property
    def is_filtered(self) -> bool:
        """Whether ANY event cut is in effect (policies alone do not count)."""
        return bool(self.compact_type or self.source_class_requested
                    or self.name_filters
                    or self.far_max is not None or self.snr_min is not None
                    or self.pastro_min is not None
                    or self.sky_area_max is not None
                    or self.m1_src_range is not None
                    or self.m2_src_range is not None)

    @property
    def source_class_filter(self) -> str:
        """The composed class request, in the round-trippable attr spelling."""
        if not self.source_class_requested or not self.source_class:
            return ""
        return format_source_class_filter(list(self.source_class))

    @property
    def event_list_filter(self) -> str:
        """The ``event_list=`` sources, in the legacy attr spelling."""
        return ";".join(lab.split(":", 1)[1] for lab in self.name_filters
                        if lab.startswith("event_list:"))

    @property
    def allowed_names_filter(self) -> str:
        """Which direct name-whitelist spellings were used (``""`` if none)."""
        return ";".join(lab for lab in self.name_filters
                        if not lab.startswith("event_list:"))

    @property
    def allowed_names_digest(self) -> str:
        """A deterministic digest of the composed name whitelist."""
        from .export.contract import event_list_digest
        return event_list_digest(self.allowed_names)

    @property
    def cut_estimator(self) -> str:
        """WHICH quantity the class restriction was applied to (GW-12).

        Every name filter counts as a whitelist, not only ``event_list=``: an
        ``allowed_names=`` selection is the same "membership in a fixed list"
        the pairing check treats as reproducible, and recording it as ``"none"``
        described a filtered file as unfiltered.
        """
        return pe_cut_estimator(
            self.source_class if self.source_class_requested else None,
            self.allowed_names if self.name_filters else None)

    # ---- serialisation --------------------------------------------------
    def to_dict(self) -> dict:
        """A JSON-able, deterministic record of the effective selection.

        The whitelist itself is summarised by count + digest rather than
        written out: the events actually exported are already listed in
        ``event_names``, and a 259-name list in an HDF5 attr helps nobody.
        """
        return {
            "compact_type": list(self.compact_type),
            "source_class": (list(self.source_class)
                             if self.source_class_requested else None),
            "far_max": _round(self.far_max),
            "snr_min": _round(self.snr_min),
            "pastro_min": _round(self.pastro_min),
            "sky_area_max": _round(self.sky_area_max),
            "m1_src_range": (None if self.m1_src_range is None
                             else [_round(v) for v in self.m1_src_range]),
            "m2_src_range": (None if self.m2_src_range is None
                             else [_round(v) for v in self.m2_src_range]),
            "name_filters": list(self.name_filters),
            "n_allowed_names": (-1 if self.allowed_names is None
                                else len(self.allowed_names)),
            "allowed_names_digest": self.allowed_names_digest,
            "allowed_names_authoritative": bool(
                self.allowed_names_authoritative),
            "far_policy": str(self.far_policy),
            "n_events_missing_far": int(self.n_missing_far),
            "allow_missing_far": bool(self.allow_missing_far),
            "require_far": bool(self.require_far),
            "waveform_policy": str(self.waveform_policy),
            "approximant": (None if self.approximant is None
                            else str(self.approximant)),
            "cut_estimator": self.cut_estimator,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True,
                          separators=(",", ":"))

    def digest(self) -> str:
        """A short digest of the whole effective spec."""
        from .export.contract import stable_digest
        return stable_digest(self.to_json())

    def to_attrs(self) -> dict:
        """The provenance attrs both writers stamp on an export.

        Every exporter writes these from the EFFECTIVE spec, never from its own
        keyword arguments, so the file states the selection it actually holds.
        A cut with no value is NaN rather than a missing attr: HDF5 has no null,
        and "no cut on this statistic" has to be distinguishable from "this file
        predates the record".
        """
        def _cut(x):
            return float("nan") if x is None else float(x)

        return {
            "selection_spec": self.to_json(),
            "selection_spec_digest": self.digest(),
            "selection_filtered": bool(self.is_filtered),
            "compact_type": ",".join(self.compact_type),
            "source_class_filter": self.source_class_filter,
            CUT_ESTIMATOR_ATTR: self.cut_estimator,
            "event_list_filter": self.event_list_filter,
            "allowed_names_filter": self.allowed_names_filter,
            "n_allowed_names": (-1 if self.allowed_names is None
                                else len(self.allowed_names)),
            "allowed_names_digest": self.allowed_names_digest,
            "far_max": _cut(self.far_max),
            "snr_min": _cut(self.snr_min),
            "pastro_min": _cut(self.pastro_min),
            "sky_area_max": _cut(self.sky_area_max),
            "far_policy": str(self.far_policy),
            "n_events_missing_far": int(self.n_missing_far),
            "allow_missing_far": bool(self.allow_missing_far),
            "require_far": bool(self.require_far),
            "waveform_policy": str(self.waveform_policy),
            "approximant": "" if self.approximant is None else str(
                self.approximant),
        }

    # ---- pairing ---------------------------------------------------------
    def check_pairable(self, *, allow_unpaired_pastro_min: bool = False):
        """Refuse the cuts no injection campaign can reproduce.

        ``pastro_min`` is the one that exists today.  Every selection export
        records ``p_astro_available=False``, because the campaigns carry no
        per-injection p_astro at all: there is no injection-side threshold that
        reproduces an event-side p_astro cut, so beta would describe a
        different selection than the one that produced the event list -- the
        Essick & Fishbach failure, and invisible in the files until the cut
        started being recorded.
        """
        if self.pastro_min is None or allow_unpaired_pastro_min:
            return
        raise UnpairableSelectionCut(
            f"pastro_min={self.pastro_min} was applied to the events, but the "
            f"injection side has no equivalent: every selection export records "
            f"p_astro_available=False because the campaigns carry no "
            f"per-injection p_astro, so no injection cut reproduces this one "
            f"and the exported beta would describe a different selection than "
            f"the event list it is paired with (Essick & Fishbach; GW-33).\n"
            f"\n"
            f"Cut on FAR instead (far_max=, CLI --far-max: the statistic the "
            f"campaigns threshold on directly), or select the events by name "
            f"(event_list= / allowed_names=, which is membership rather than a "
            f"cut on a statistic). Pass allow_unpaired_pastro_min=True to write "
            f"the file anyway -- it is then NOT usable with a selection "
            f"function.")


class _SampleReader:
    """Per-event, per-row posterior reads against ONE open handle on a store.

    Why this exists (GW-32).  :meth:`GWCatalog.get` is a whole-selection read:
    it returns every row of every selected event, so an exporter that keeps
    ``nsamp`` rows per event had to materialise the entire posterior first --
    6.88M rows read to emit 1.16M on the shipped store.  A reader inverts the
    order: :meth:`counts` answers "how many rows does event ``e`` have?" with no
    sample read at all (the offsets are already in memory), the caller draws its
    indices, and :meth:`read` fetches only those.

    Why a SPAN read and not fancy indexing.  ``d[a + rows]`` (an h5py point
    selection) is the obvious way to read exactly the drawn rows, and it is
    catastrophically slow: on the shipped store, drawing 5000 of ~24k rows for
    30 events took 6.18 s as a point selection versus 0.019 s reading the
    enclosing span and indexing in memory -- HDF5 issues per-point I/O, and the
    span costs no extra chunks anyway (a uniform draw touches them all).  So the
    span is read, indexed, and dropped; what is RETAINED is only the drawn rows,
    which is where the memory goes.
    """

    def __init__(self, cat, f):
        self._cat = cat
        self._f = f
        self._sl = cat._slices()
        # Dataset handles, opened once each.  A per-event loop asks for the same
        # column hundreds of times, and re-resolving "samples/<p>" every time
        # costs more than the reads do.
        self._ds = {}

    def _dataset(self, p):
        d = self._ds.get(p)
        if d is None:
            d = self._ds[p] = self._f[f"samples/{p}"]
        return d

    def counts(self):
        """Rows behind each selected event, without touching the sample data."""
        return np.array([b - a for (a, b) in self._sl], dtype=np.int64)

    def read(self, e, params, rows=None, required=True, fill_value=np.nan):
        """Columns for selected event ``e``, restricted to ``rows``.

        ``rows`` are indices INTO THE EVENT's own slice (as
        :meth:`GWCatalog.get` ``per_event=True`` would return it); they may
        repeat and need not be sorted, and the result is exactly
        ``get(params, per_event=True)[p][e][rows]`` -- negative indices count
        from the end and an out-of-range one raises, as numpy's would.
        ``rows=None`` reads the whole slice.  ``required``/``fill_value``
        follow :meth:`GWCatalog.get`.
        """
        from .schema import MissingParameterError
        cat = self._cat
        params = [params] if isinstance(params, str) else list(params)
        a, b = self._sl[e]
        if rows is None:
            take = None
            n_out = b - a
        else:
            rows = np.asarray(rows, dtype=np.intp)
            n_out = rows.size
            if n_out:
                n_slice = b - a
                if (rows < 0).any():
                    rows = np.where(rows < 0, rows + n_slice, rows)
                lo = int(rows.min())
                hi = int(rows.max())
                if lo < 0 or hi >= n_slice:
                    raise IndexError(
                        f"row index out of range for selected event {e}, "
                        f"which has {n_slice} sample(s)")
                take = (lo, hi, rows - lo)
            else:
                take = (0, -1, rows)
        out = {}
        for p in params:
            if p not in cat._param_index:
                if required:
                    raise MissingParameterError(
                        f"required parameter {p!r} is not in the store; "
                        f"stored parameters are {cat.params}. Pass "
                        f"required=False for a {fill_value}-filled column.")
                out[p] = np.full(n_out, fill_value)
                continue
            d = self._dataset(p)
            if take is None:
                out[p] = d[a:b]
            elif take[1] < take[0]:      # rows is empty
                out[p] = np.empty(0, dtype=d.dtype)
            else:
                lo, hi, local = take
                out[p] = d[a + lo:a + hi + 1][local]
        return out


class GWCatalog:
    def __init__(self, store_path, _sel=None):
        self.path = store_path
        with h5py.File(store_path, "r") as f:
            self.params = [p.decode() if isinstance(p, bytes) else str(p)
                           for p in f.attrs["param_names"]]
            self.offsets = f["index/offsets"][:]
            self.names = np.array([n for n in f["index/event_names"][:]])
            self.meta = {k: f[f"meta/{k}"][:] for k in f["meta"].keys()}
            # Per-event x per-parameter availability mask (PR 5).  Legacy stores
            # (schema < 1.1) have no mask; treat every stored column as available
            # for every event, which is exact because the old intersection
            # ingest guaranteed it.
            if "avail" in f and "mask" in f["avail"]:
                avail = np.asarray(f["avail/mask"][:], dtype=bool)
            else:
                avail = None
        # decode bytes -> str for string meta
        for k, v in self.meta.items():
            if v.dtype.kind in ("S", "O"):
                self.meta[k] = np.array([x.decode() if isinstance(x, bytes) else x
                                         for x in v])
        self.names = np.array([n.decode() if isinstance(n, bytes) else n
                               for n in self.names])
        #: param name -> column index into ``self.avail`` / ``self.params``.
        self._param_index = {p: j for j, p in enumerate(self.params)}
        self.avail = (avail if avail is not None
                      else np.ones((len(self.names), len(self.params)),
                                   dtype=bool))
        self._sel = np.arange(len(self.names)) if _sel is None else np.asarray(_sel)
        self._source_class_cache = None
        # Provenance of the most recent select() call (defaults for a fresh
        # catalog with no filters applied yet).
        self._far_policy = "none"
        self._n_missing_far = 0
        self._selection_source_class = None
        self._selection_event_list = None
        # The EFFECTIVE selection: every cut this view accumulated, across
        # however many select() calls produced it.  The per-call attributes
        # above describe only the last one, which is what let a filtered export
        # advertise itself as unfiltered (GW-33).
        self._selection_spec = SelectionSpec()
        # Waveform / sample-set policy provenance (PR 6).  Defaults describe a
        # fresh view with no policy applied yet.  NOTE: _homogeneous_sample_sets
        # = True here is a placeholder, not a measurement -- a fresh multi-set
        # store IS inhomogeneous, and the real value is computed by select(),
        # which both exporters go through.  Read it only from a select()ed view.
        self._waveform_policy = "preferred"
        self._waveform_approximant = None
        self._selection_reasons = None
        self._homogeneous_sample_sets = True

    # ---- selection (operates on metadata only; cheap) --------------------
    @property
    def selection_spec(self) -> SelectionSpec:
        """The EFFECTIVE :class:`SelectionSpec` behind this view.

        Composed across every :meth:`select` call that produced it, so it
        describes the events the view actually holds -- not the arguments of
        whichever call came last.  This is what the exporters record.
        """
        return self._selection_spec

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

    def _pastro_column(self):
        """Per-row astrophysical probability, resolved across both spellings.

        ``p_astro`` and ``pastro`` are the same quantity (see
        :data:`gwcat.event_metadata.PASTRO_KEYS`); which column a store
        populated depends on which key its event table used and on when it was
        ingested.  Every consumer of a p_astro reads this, so none of them can
        report "absent" about a value the store holds under the other name.
        """
        from .event_metadata import resolve_pastro_column
        return resolve_pastro_column(self.meta, len(self.names))

    def select(self, compact_type=None, far_max=None, pastro_min=None,
               snr_min=None, z_max=None, m1_src_range=None, m2_src_range=None,
               sky_area_max=None, names=None, allowed_names=None,
               allowed_names_authoritative=True, source_class=None,
               event_list=None, allow_missing_far=False, require_far=False,
               waveform_policy="preferred", approximant=None):
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
        pastro_min : float or None
            Threshold on the event's astrophysical probability, read from
            whichever of the ``p_astro`` / ``pastro`` meta columns carries a
            finite value for that row (they are one quantity under two
            spellings).  Events with neither are dropped, with a warning naming
            them; a store where NO candidate event has either raises, because a
            threshold crossed with an all-absent column is a configuration
            error, not a selection (GW-14).

        Waveform / sample-set policy (PR 6)
        -----------------------------------
        waveform_policy : {"preferred", "mixed-first", "strict-approximant", \
"all"}
            Resolve which sample set represents each event AFTER the metadata
            cuts above.  Guarantees one sample set per event unless
            ``"all"``.  ``"preferred"`` (default) uses the ``is_preferred`` /
            ``priority_rank`` meta columns; ``"mixed-first"`` prefers an
            ``is_mixed`` set; ``"strict-approximant"`` requires ``approximant``
            for every event (failing loudly otherwise); ``"all"`` keeps every
            sample set.  For a single-sample-set store (one row per event, no
            sample-set columns) every policy is a no-op.  See
            :mod:`gwcat.waveform_policy`.
        approximant : str or None
            Required approximant for ``waveform_policy="strict-approximant"``;
            matched against each row's ``approximant`` or ``waveform`` family.

        Effective selection provenance (GW-33)
        --------------------------------------
        The returned view carries :attr:`selection_spec`, the EFFECTIVE
        :class:`SelectionSpec`: this call composed with the spec of the view it
        was called on, exactly as the rows are composed.  The exporters record
        that spec rather than their own arguments, so a product built from an
        already-filtered view states the filter it actually holds.
        """
        import warnings
        if require_far and allow_missing_far:
            raise ValueError(
                "require_far and allow_missing_far are mutually exclusive")

        if z_max is not None:
            raise NotImplementedError(
                "select(z_max=...) is not a per-sample redshift cut and never "
                "was -- the argument was accepted and silently ignored, so an "
                "analysis asking for it shipped uncut. The per-sample cut lives "
                "on the exporters: to_darksirens(z_max=...) / "
                "export(..., z_max=...) / build_pe_product(..., z_max=...), "
                "which drop samples above z_max before resampling.\n"
                "\n"
                "An event-level z_max is deliberately NOT offered in its place. "
                "select()'s other cuts act on posterior medians, and a hard cut "
                "on a noisy point estimate is not reproduced by the same cut on "
                "the injections' true values, so it biases the selection "
                "function exactly at the boundary (Essick & Fishbach; see "
                "GW-12). Cut samples, or cut on something the injection "
                "campaign can reproduce.")

        m = np.ones(len(self.names), dtype=bool)
        # What this call actually applies, for the effective spec below: a
        # compact_type that allowed_names overrules is NOT a cut this view made.
        applied_compact_type = None
        name_filters = []
        if compact_type is not None:
            apply_compact_type = (allowed_names is None or
                                  not allowed_names_authoritative)
            if apply_compact_type:
                m &= (self.meta["compact_type"] == compact_type)
                applied_compact_type = compact_type
        if source_class is not None:
            allowed_classes = resolve_filter_classes(source_class)
            m &= np.isin(self.source_class, list(allowed_classes))
        if pastro_min is not None:
            # Read the RESOLVED value, not one spelling of it: a store whose
            # event table used `p_astro` (the documented override key) holds a
            # NaN `pastro` column, and reading only that turned a populated
            # catalog into an empty selection (GW-14).
            pa = self._pastro_column()
            in_scope = m & np.isin(np.arange(len(self.names)), self._sel)
            nan_pa = np.isnan(pa)
            if in_scope.any() and nan_pa[in_scope].all():
                raise ValueError(
                    f"select(pastro_min={pastro_min}) would exclude all "
                    f"{int(in_scope.sum())} candidate event(s) because this "
                    f"store carries no p_astro at all: both the 'p_astro' and "
                    f"the 'pastro' meta columns are NaN for every one of them. "
                    f"A threshold crossed with an all-absent column is a "
                    f"configuration error, not a selection -- populate "
                    f"p_astro via the event_table/override file at ingest (see "
                    f"gwcat.event_metadata.assemble_event_metadata), or drop "
                    f"pastro_min.")
            n_nan = int(nan_pa[in_scope].sum())
            if n_nan:
                warnings.warn(
                    f"pastro_min={pastro_min} drops {n_nan} event(s) with no "
                    f"p_astro under either spelling: "
                    f"{sorted(self.names[in_scope & nan_pa].tolist())}. "
                    f"Populate p_astro via event_table at ingest to use this "
                    f"cut on them.")
            m &= np.where(nan_pa, False, pa >= pastro_min)
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
            # Both spellings are the same filter -- membership in a fixed list
            # of names -- and normalise into the one whitelist the spec keeps.
            name_filters.append(
                ("allowed_names" if allowed_names is not None else "names",
                 [str(x) for x in _whitelist_arr.tolist()]))

        # User event-list file / sequence (additional intersection gate).
        if event_list is not None:
            listed = load_event_list(event_list)
            listed_arr = np.asarray(listed)
            name_filters.append(
                ("event_list:" + (str(event_list)
                                  if isinstance(event_list, (str, bytes))
                                  else "custom_sequence"),
                 [str(x) for x in listed]))
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

        # (The p_astro-absence warning used to live here, after the cut it was
        # meant to describe had already removed every NaN row from `m` -- so it
        # could never fire.  It is issued at the cut itself now.)
        sel = np.nonzero(m & np.isin(np.arange(len(self.names)), self._sel))[0]

        # ── Waveform / sample-set policy resolution (PR 6) ────────────────────
        # Collapse the metadata-selected rows to one sample set per event
        # (unless waveform_policy="all").  A no-op for single-sample-set stores.
        from .waveform_policy import resolve_policy
        kept, reasons, homogeneous = resolve_policy(
            self.names, sel, self.meta, policy=waveform_policy,
            approximant=approximant)

        # ── The EFFECTIVE selection (GW-33) ───────────────────────────────────
        # Composed with the spec of the view this call was made on, because the
        # ROWS are composed the same way (`m` is ANDed with self._sel above).
        # Recording only this call's arguments is what let an export from
        # cat.select(source_class="bbh") keep every BBH row while writing an
        # empty source_class_filter.
        spec = self._selection_spec.refine(
            compact_type=applied_compact_type,
            source_class=source_class,
            far_max=far_max, snr_min=snr_min, pastro_min=pastro_min,
            sky_area_max=sky_area_max,
            m1_src_range=m1_src_range, m2_src_range=m2_src_range,
            name_filters=name_filters,
            allowed_names_authoritative=allowed_names_authoritative,
            far_policy=far_policy, n_missing_far=n_missing_far,
            allow_missing_far=allow_missing_far, require_far=require_far,
            waveform_policy=waveform_policy, approximant=approximant)

        result = GWCatalog(self.path, _sel=kept)
        result._selection_spec = spec
        result._far_policy = far_policy
        result._n_missing_far = n_missing_far
        # The detection cut, numerically.  Essick & Fishbach require the EVENT
        # cut and the INJECTION detection cut to be the same statistic at the
        # same threshold; the exports recorded only the qualitative far_policy,
        # so there was nothing for the paired-file check to compare against.
        result._far_max = far_max
        result._snr_min = snr_min
        result._selection_source_class = source_class
        result._selection_event_list = event_list
        result._waveform_policy = waveform_policy
        result._waveform_approximant = approximant
        result._selection_reasons = np.asarray(reasons, dtype=object)
        result._homogeneous_sample_sets = bool(homogeneous)
        return result

    # ---- sample access ---------------------------------------------------
    def _slices(self):
        return [(self.offsets[i], self.offsets[i + 1]) for i in self._sel]

    def get(self, params, per_event=False, required=True, fill_value=np.nan):
        """Read columns for the current selection.

        per_event=False -> dict of flat concatenated arrays.
        per_event=True  -> dict of lists (one array per event).

        Required vs optional access (PR 5)
        ----------------------------------
        required : bool, default True
            When True (the default, preserving the historical contract), a
            requested parameter that is not in the store raises a clear
            :class:`gwcat.schema.MissingParameterError` (a ``KeyError`` subclass)
            naming the parameter and the stored set -- never a bare ``KeyError``.
        fill_value : float, default NaN
            When ``required=False``, a parameter absent from the store is
            returned as a ``fill_value``-filled column shaped like the current
            selection instead of raising.  Parameters present in the store but
            NaN-filled for some events at ingest return their stored values
            as-is; use :meth:`param_available` to learn which events had them.
        """
        from .schema import MissingParameterError
        params = [params] if isinstance(params, str) else list(params)
        out = {p: [] for p in params}
        sl = self._slices()
        with h5py.File(self.path, "r") as f:
            for p in params:
                if p not in self._param_index:
                    if required:
                        raise MissingParameterError(
                            f"required parameter {p!r} is not in the store; "
                            f"stored parameters are {self.params}. Pass "
                            f"required=False for a {fill_value}-filled column.")
                    out[p] = [np.full(b - a, fill_value) for (a, b) in sl]
                    continue
                d = f[f"samples/{p}"]
                out[p] = [d[a:b] for (a, b) in sl]
        if per_event:
            return out
        return {p: np.concatenate(v) if v else np.array([]) for p, v in out.items()}

    @contextlib.contextmanager
    def sample_reader(self):
        """One open handle on the store, for per-event / per-row sample reads.

        The counterpart to :meth:`get` for a consumer that decides WHICH rows it
        will keep before it wants their values -- the downsampling exporters.
        ``get`` must read every selected slice in full, so the PE export
        materialised 6.88M rows to emit 1.16M and peaked at 1.20 GiB; a reader
        streams one event at a time and retains only the drawn rows.

        Usage::

            with cat.sample_reader() as rd:
                n = rd.counts()                      # no sample read at all
                rows = rng.choice(n[0], size=5000)
                d = rd.read(0, ["mass_1"], rows=rows)
        """
        with h5py.File(self.path, "r") as f:
            yield _SampleReader(self, f)

    def param_available(self, param):
        """Boolean per-event availability of ``param`` for the current selection.

        True where the event actually provided the parameter at ingest, False
        where its slice is NaN-filled.  A parameter not in the store returns an
        all-False array (its column is entirely absent).
        """
        sel = np.asarray(self._sel)
        if param not in self._param_index:
            return np.zeros(sel.size, dtype=bool)
        return self.avail[sel, self._param_index[param]]

    def _require_params(self, need, export="export"):
        """Fail loudly if any ``need`` parameter is absent from the store or
        unavailable (NaN-filled) for a selected event, naming param + events."""
        from .schema import check_required
        check_required(need, self.params, self.avail, self.names,
                       np.asarray(self._sel), self._param_index, export=export)

    def _require_alternatives(self, alternative_groups, export="export"):
        """Fail loudly if, for any alternative group, no alternative is
        present+available for a selected event (the "any-of" companion to
        :meth:`_require_params`); naming the alternatives + events."""
        from .schema import check_required_alternatives
        check_required_alternatives(alternative_groups, self.params, self.avail,
                                    self.names, np.asarray(self._sel),
                                    self._param_index, export=export)

    def event(self, name, params=None):
        """Posterior samples for one event, as this view sees it.

        Respects the current selection and waveform policy, so inspecting an
        event agrees with what an export from the same view wrote.  It used to
        index the *whole* store and ignore both, which meant a per-event check
        could silently disagree with the file it was checking.

        Raises :class:`KeyError` for a name this view does not contain, naming
        whether the event exists in the store but was filtered out -- the two
        cases need different fixes, and a bare ``IndexError`` distinguished
        neither.
        """
        rows = np.nonzero(np.asarray(self.names) == name)[0]
        sel = np.asarray(self._sel)
        visible = [int(i) for i in rows if i in set(sel.tolist())]

        if not visible:
            if rows.size:
                raise KeyError(
                    f"event {name!r} is in the store but not in this view: it "
                    f"was removed by select() or by the waveform policy "
                    f"({getattr(self, '_waveform_policy', 'preferred')!r}). "
                    f"Call .event() on the unfiltered catalog to inspect it.")
            raise KeyError(
                f"event {name!r} is not in {self.path}. This view has "
                f"{len(sel)} event row(s).")
        if len(visible) > 1:
            raise KeyError(
                f"event {name!r} matches {len(visible)} sample-set rows in this "
                f"view (waveform_policy="
                f"{getattr(self, '_waveform_policy', 'preferred')!r}); "
                f".event() returns one row. Re-select with a policy that "
                f"resolves to a single sample set.")

        i = visible[0]
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
                      spin_prior_mode="include",
                      allowed_names=None,
                      allowed_names_authoritative=True,
                      source_class=None, event_list=None,
                      allow_missing_far=False, require_far=False,
                      waveform_policy="preferred", approximant=None,
                      write_summary: bool = False,
                      summary_context: Optional[dict] = None,
                      allow_zero_p_pe: bool = False,
                      allow_unpaired_pastro_min: bool = False):
        """Write an HDF5 consumable by darksirens.gw.utils.load_gw_samples.

        p_pe convention (spin-prior contract)
        -------------------------------------
        p_pe is the PE prior in the (m1det, q, dL[, chi_eff]) basis that
        darksirens.gw.utils.load_gw_samples divides out per event.

        For a uniform detector-frame component-mass prior the (m1det, q)-basis
        density carries a Jacobian |dm2det/dq| = m1det, so the mass-Jacobian
        contribution is:

            p_pe = m1det * p_dL_pe

        This is the ONLY place the mass Jacobian is applied. The store keeps the
        mass-prior-agnostic p_dL_pe, and the distance prior p_dL_pe remains a
        FACTOR of p_pe (it is not divided out).

        The 1-D chi_eff prior is governed by ``spin_prior_mode`` (Mode A is the
        default):

        * ``"include"`` (default): the 1-D isotropic chi_eff prior is multiplied
          into p_pe here, so the exported p_pe already contains it.  Downstream
          (darksirens) MUST NOT multiply the chi_eff prior again — doing so
          double-counts it.  Recorded as ``chi_eff_prior_applied_to_p_pe=True``
          and the legacy ``chi_eff_in_p_pe=True``.
        * ``"exclude"``: p_pe carries NO chi_eff prior factor (only the mass
          Jacobian and the distance prior).  Downstream MUST apply the 1-D
          chi_eff prior itself.  Recorded as
          ``chi_eff_prior_applied_to_p_pe=False``.

        ``"passthrough"`` is intentionally NOT offered: the store never bakes a
        spin prior into p_dL_pe, so "no spin-prior manipulation" is byte-for-byte
        identical to ``"exclude"`` (there is nothing distinct to pass through).

        Parameters
        ----------
        spin_prior_mode : {"include", "exclude"}
            Whether the exported p_pe contains the 1-D chi_eff prior factor.
            Default ``"include"`` (Mode A) is byte-identical to prior behavior.
            Any other value raises ``ValueError``.
        cosmology : tuple (H0, Om0) or None
            Cosmology used for the z_max cut, the dL→z inversion, and the
            stored source masses / redshift.

            * ``None`` (default, "per-event" mode): EACH event independently
              uses its OWN stored PE cosmology (``meta/dL_prior_H0`` /
              ``meta/dL_prior_Om0``).  This is the scientifically correct
              behavior for mixed-release selections whose events were analysed
              under different cosmologies.  If any selected event has a missing
              (NaN) or absent per-event cosmology, the export fails loudly and
              names the offending events; pass an explicit override instead.
            * ``(H0, Om0)`` ("override" mode): that single cosmology is applied
              to ALL events, ``cosmology_override_used=True`` is recorded, and
              the override parameters are written into the output attrs.

            .. note:: Migration.  Earlier versions took the FIRST selected
               event's cosmology and applied it to every event's z_of_dL and
               source-frame masses.  For selections whose events share one
               cosmology (the common case, including all bundled tests) the
               output is byte-identical.  For mixed-cosmology selections the
               numerical output now differs -- that difference is the bug fix.
               Provenance is recorded in the output attrs ``cosmology_mode``,
               ``cosmology_per_event_varies``, ``cosmology_H0_per_event`` and
               ``cosmology_Om0_per_event``.
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
        waveform_policy : {"preferred", "mixed-first", "strict-approximant", \
"all"}
            Which sample set represents each event (PR 6).  Guarantees one
            sample set per event unless ``"all"``.  Default ``"preferred"`` is a
            no-op for single-sample-set stores, so existing exports are
            unchanged.  ``"strict-approximant"`` fails loudly (naming the events)
            when ``approximant`` is unavailable for one.  See :meth:`select` and
            :mod:`gwcat.waveform_policy`.  The chosen policy, the per-event
            chosen ``sample_set_name`` / ``approximant`` arrays, and a
            ``homogeneous_sample_sets`` boolean are written to the output attrs
            so a multi-waveform (``"all"``) file is never presented as
            homogeneous.
        approximant : str or None
            Required approximant for ``waveform_policy="strict-approximant"``.
        write_summary : bool, default False
            (PR 10) When True, write ``<out_path>.validation_summary.json`` and
            ``.md`` next to ``out_path`` (see :mod:`gwcat.validation_summary`).
            Opt-in at the library level; the unified ``gwcat export-darksirens``
            CLI turns this on by default (``--no-summary`` to disable).
        summary_context : dict, optional
            Extra fields merged into the written summary. Never populated
            automatically.
        allow_unpaired_pastro_min : bool, default False
            Write the file even though a ``pastro_min`` cut is in effect.  The
            campaigns carry no per-injection p_astro (every selection export
            records ``p_astro_available=False``), so no injection-side cut
            reproduces it and the pair is not usable; see
            :meth:`SelectionSpec.check_pairable`.

        Provenance (GW-33)
        ------------------
        The selection attrs are written from the EFFECTIVE
        :attr:`selection_spec` of the exported view -- this call's filters
        composed with those of the view it was called on -- so exporting from
        ``cat.select(source_class="bbh")`` records the BBH filter instead of
        this call's defaults.  ``event_list_digest`` identifies the events
        written, independently of how they were spelled.
        """
        valid_spin_modes = ("include", "exclude")
        if spin_prior_mode not in valid_spin_modes:
            raise ValueError(
                f"spin_prior_mode={spin_prior_mode!r} is invalid; choose one "
                f"of {valid_spin_modes}. 'passthrough' is not offered because "
                f"the store keeps a spin-prior-agnostic p_dL_pe, so 'exclude' "
                f"already means 'no chi_eff prior applied'.")
        sub = self.select(compact_type=compact_type, far_max=far_max,
                          pastro_min=pastro_min, allowed_names=allowed_names,
                          allowed_names_authoritative=allowed_names_authoritative,
                          source_class=source_class, event_list=event_list,
                          allow_missing_far=allow_missing_far,
                          require_far=require_far,
                          waveform_policy=waveform_policy,
                          approximant=approximant)
        # The EFFECTIVE selection: this call's filters composed with those the
        # exported view already carried (GW-33).  Everything this exporter
        # records about the selection comes from here, and the cut no injection
        # campaign can reproduce is refused before any samples are read.
        spec = sub.selection_spec
        spec.check_pairable(
            allow_unpaired_pastro_min=allow_unpaired_pastro_min)
        # ── The class cut the paired selection file cannot reproduce (GW-12) ─
        # Warned here, at export time, rather than only by the validator that
        # will refuse the finished pair: the cheap alternative (event_list=) is
        # a choice about how to build THIS product.  Asked of the EFFECTIVE
        # spec, so a view that was already class-filtered warns too -- that is
        # the case where nothing else would have said it (GW-33).
        if spec.source_class_requested:
            warnings.warn(PE_MEDIAN_CUT_WARNING)
        from .schema import DARKSIRENS_REQUIRED
        need = list(DARKSIRENS_REQUIRED)
        # Required-parameter contract (PR 5): fail loudly -- naming the missing
        # parameter(s) and event(s) -- if a required export column is absent
        # from the store or NaN-filled for any selected event, rather than
        # producing a silently-wrong export.
        sub._require_params(need, export="darksirens export")
        per = sub.get(need, per_event=True)
        rng = np.random.default_rng(seed)

        # ── Resolve cosmology: per-event (default) or a single override ─────
        # cosmology=None  -> each event uses ITS OWN stored PE cosmology
        #                    (meta/dL_prior_H0, meta/dL_prior_Om0).
        # cosmology=(H0,Om0) -> that single override is applied to EVERY event.
        # The first selected event's cosmology is kept as the scalar
        # pe_cosmology_H0/Om0 for backward compatibility, but the authoritative
        # record in per-event mode is the per-event array written to attrs.
        sel_idx = np.asarray(sub._sel)
        have_cosmo_cols = ("dL_prior_H0" in sub.meta
                           and "dL_prior_Om0" in sub.meta)
        if cosmology is not None:
            cosmology_mode = "override"
            override_H0, override_Om0 = float(cosmology[0]), float(cosmology[1])
            per_event_H0 = np.full(sub.n_events, override_H0, dtype=float)
            per_event_Om0 = np.full(sub.n_events, override_Om0, dtype=float)
            pe_H0, pe_Om0 = override_H0, override_Om0
        else:
            cosmology_mode = "per-event"
            if not have_cosmo_cols:
                raise ValueError(
                    "cosmology=None requires a per-event PE cosmology in the "
                    "store (meta/dL_prior_H0 and meta/dL_prior_Om0), but those "
                    "columns are absent. Pass an explicit cosmology=(H0, Om0) "
                    "override to apply one cosmology to all events.")
            per_event_H0 = np.asarray(sub.meta["dL_prior_H0"],
                                      dtype=float)[sel_idx]
            per_event_Om0 = np.asarray(sub.meta["dL_prior_Om0"],
                                       dtype=float)[sel_idx]
            bad = ~(np.isfinite(per_event_H0) & np.isfinite(per_event_Om0))
            if bad.any():
                bad_names = sorted(np.asarray(sub.event_names)[bad].tolist())
                raise ValueError(
                    f"cosmology=None but {int(bad.sum())} selected event(s) "
                    f"have no stored PE cosmology (dL_prior_H0/dL_prior_Om0 is "
                    f"NaN): {bad_names}. Pass an explicit cosmology=(H0, Om0) "
                    f"override to apply one cosmology to all events.")
            pe_H0 = float(per_event_H0[0]) if sub.n_events else float("nan")
            pe_Om0 = float(per_event_Om0[0]) if sub.n_events else float("nan")

        # Per-event cosmology objects, built once per unique (H0, Om0) pair.
        _cosmo_cache: dict = {}

        def _cosmo_for(e):
            key = (per_event_H0[e], per_event_Om0[e])
            c = _cosmo_cache.get(key)
            if c is None:
                c = make_cosmology(*key)
                _cosmo_cache[key] = c
            return c

        cols = {k: [] for k in ["m1det", "m2det", "dL", "ra", "dec",
                                "chieff", "p_pe", "redshift", "m1src", "m2src"]}
        kept = []
        kept_H0, kept_Om0 = [], []
        # Per-written-row sample-set provenance (PR 6), aligned with ``kept``.
        kept_ss_name, kept_ss_approx, kept_ss_reason = [], [], []
        # Resampling provenance (GW-13), aligned with ``kept``.
        n_unique_per_event, upsampled_events = [], []

        def _ss_meta(row, field):
            v = sub.meta.get(field)
            if v is None:
                return ""
            x = v[int(row)]
            return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)

        sel_rows = np.asarray(sub._sel)
        reasons_arr = getattr(sub, "_selection_reasons", None)
        for e in range(sub.n_events):
            n = len(per["luminosity_distance"][e])
            if n == 0:
                continue

            # This event's own PE cosmology (or the single override).
            cosmo_e = _cosmo_for(e)

            dL_e = per["luminosity_distance"][e]
            m1_e = per["mass_1"][e]
            m2_e = per["mass_2"][e]

            # Per-sample z_max cut
            if z_max is not None:
                z_e = z_of_dL(dL_e, cosmo_e)
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
                warnings.warn(f"Event {sub.event_names[e]}: only {n_kept} samples "
                              f"after z_max cut, but replace=False and nsamp={nsamp}. "
                              f"Skipping.")
                continue
            idx_local = rng.choice(n_kept, size=nsamp, replace=rep)
            idx_orig = idx_map[idx_local]
            # How many DISTINCT posterior samples back this event's nsamp rows.
            # Bootstrapping an under-sampled event up to nsamp used to leave no
            # trace, and every downstream per-event MC ESS is then overestimated
            # by the duplication factor -- which is precisely the diagnostic
            # meant to flag the event as unusable.
            n_unique_per_event.append(int(np.unique(idx_orig).size))
            if rep and n_kept < nsamp:
                upsampled_events.append(str(sub.event_names[e]))

            m1 = per["mass_1"][e][idx_orig]
            m2 = per["mass_2"][e][idx_orig]
            dL = per["luminosity_distance"][e][idx_orig]
            p_dL = per["p_dL_pe"][e][idx_orig]

            # Jacobian: uniform detector-frame component-mass prior
            p_pe = m1 * p_dL

            # Redshift and source masses under THIS event's PE cosmology
            z = z_of_dL(dL, cosmo_e)

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
            kept_H0.append(float(per_event_H0[e]))
            kept_Om0.append(float(per_event_Om0[e]))
            row = sel_rows[e]
            kept_ss_name.append(_ss_meta(row, "sample_set_name"))
            kept_ss_approx.append(_ss_meta(row, "approximant"))
            kept_ss_reason.append(
                str(reasons_arr[e]) if reasons_arr is not None
                and e < len(reasons_arr) else "")

        nobs = len(kept)
        data = {k: np.concatenate(v) if v else np.array([])
                for k, v in cols.items()}

        # Whether the events actually written span more than one cosmology.
        kept_H0_arr = np.asarray(kept_H0, dtype=float)
        kept_Om0_arr = np.asarray(kept_Om0, dtype=float)
        cosmology_per_event_varies = bool(
            nobs > 1 and (np.ptp(kept_H0_arr) > 0 or np.ptp(kept_Om0_arr) > 0))

        # Apply the 1-D chi_eff prior to p_pe (Mode A default: "include").
        # In "exclude" mode the exported p_pe carries no chi_eff prior factor
        # and darksirens must apply it downstream.
        # Everything is in support unless the chi_eff prior says otherwise, which
        # only the "include" branch evaluates (GW-03).
        in_support_v1 = np.ones(np.shape(data["p_pe"]), dtype=bool)
        if spin_prior_mode == "include" and data["chieff"].size > 0:
            from .spin import chi_eff_prior_logprob
            logp_chi = chi_eff_prior_logprob(data["chieff"], data["m1src"],
                                             data["m2src"], amax=amax)
            # No -50 floor (GW-03): out of support is zero density, not 2e-22
            # in a denominator.  The v1 exporter keeps the zeros (the GW-01
            # positivity check below exempts them) and reports the count.
            logp_chi = np.asarray(logp_chi, dtype=float)
            n_unsupported = int(np.sum(~np.isfinite(logp_chi)))
            if n_unsupported:
                warnings.warn(
                    f"to_darksirens: {n_unsupported} of {logp_chi.size} samples "
                    f"fall outside the chi_eff prior's support (amax={amax}); "
                    f"their p_pe is exactly zero, which the consumer masks "
                    f"while still counting them in n for the per-event MC "
                    f"variance.")
            with np.errstate(over="ignore"):
                data["p_pe"] = data["p_pe"] * np.exp(logp_chi)
            in_support_v1 = np.isfinite(logp_chi)

        # ── Exported-weight support contract (GW-01) ────────────────────────
        from .schema import check_p_pe_positive
        check_p_pe_positive(
            data["p_pe"], event_names=kept, nsamp=nsamp,
            allow_zero=allow_zero_p_pe, expected_zero=~in_support_v1,
            context="gwcat-1.0 export (to_darksirens)",
            remedy=("A zero p_pe comes from a store whose p_dL_pe was truncated "
                    "at the recorded distance-prior bounds; re-ingest the store "
                    "so the distance prior is evaluated over the full sample "
                    "range, or pass allow_zero_p_pe=True to write it anyway."))

        # Rectangularity.  A raise, not an assert: `python -O` strips asserts,
        # and this one guards the invariant every consumer reshapes on.
        expected = nobs * nsamp
        if data["m1det"].size != expected:
            raise RuntimeError(
                f"internal error: assembled {data['m1det'].size} rows but "
                f"nobs*nsamp = {nobs}*{nsamp} = {expected}. The export would "
                f"not be reshapeable by any consumer; refusing to write it.")

        if upsampled_events:
            warnings.warn(
                f"{len(upsampled_events)} event(s) had fewer than nsamp="
                f"{nsamp} samples and were bootstrapped WITH replacement, so "
                f"their rows repeat: {upsampled_events[:5]}"
                + (" ..." if len(upsampled_events) > 5 else "")
                + ". Any per-event effective-sample-size computed from these "
                  "rows is overestimated by the duplication factor -- read "
                  "n_unique_samples_per_event, which records how many distinct "
                  "posterior samples actually back each event.")

        with h5py.File(out_path, "w") as f:
            # --- Attributes darksirens reads ---
            f.attrs["nsamp"] = int(nsamp)
            f.attrs["nobs"] = int(nobs)
            f.attrs["mock_data"] = False
            # --- Provenance (gwcat-specific) ---
            f.attrs["format_version"] = "gwcat-1.0"

            # Which gwcat wrote this file (DS-10 provenance; also on the v2
            # writers).  Commit, not version: an editable install moves per
            # commit while the version string stands still.
            from .validation_summary import gwcat_commit, package_version as _pkg_version
            f.attrs["writer_commit"] = gwcat_commit()
            f.attrs["writer_version"] = _pkg_version()
            f.attrs["mass_prior_basis"] = "uniform_detector_frame"
            # ── Spin-prior contract provenance (PR 3) ──────────────────────
            chi_eff_included = (spin_prior_mode == "include")
            f.attrs["spin_prior_mode"] = spin_prior_mode
            f.attrs["chi_eff_prior_applied_to_p_pe"] = bool(chi_eff_included)
            f.attrs["mass_jacobian_applied"] = True
            # The distance prior p_dL_pe is a FACTOR of p_pe, not removed.
            f.attrs["distance_prior_removed"] = False
            # ── Cosmology contract provenance (PR 4) ───────────────────────
            # cosmology_mode: "per-event" (each event's own stored PE cosmology)
            #                 or "override" (one user cosmology applied to all).
            f.attrs["cosmology_mode"] = cosmology_mode
            f.attrs["cosmology_override_used"] = bool(cosmology is not None)
            # Source-frame masses / redshift were computed under the cosmology
            # recorded here (per-event array below, or the override scalars).
            f.attrs["source_frame_under_recorded_cosmology"] = True
            f.attrs["cosmology_per_event_varies"] = bool(
                cosmology_per_event_varies)
            # Per-event cosmology actually used, aligned with event_names.
            f.attrs["cosmology_H0_per_event"] = kept_H0_arr
            f.attrs["cosmology_Om0_per_event"] = kept_Om0_arr
            # Legacy flag, kept for backward compat; consistent with the mode.
            f.attrs["chi_eff_in_p_pe"] = bool(chi_eff_included)
            f.attrs["chi_eff_amax"] = float(amax)
            # Scalar PE cosmology: the override, or the first kept event's
            # cosmology in per-event mode (authoritative record is the
            # per-event array above when cosmology_per_event_varies=True).
            f.attrs["pe_cosmology_H0"] = pe_H0
            f.attrs["pe_cosmology_Om0"] = pe_Om0
            # --- Effective event selection (PR 2, GW-12, GW-33) ---
            # From the EFFECTIVE spec of the exported view, never from this
            # call's arguments: the rows are the intersection of every select()
            # that produced `sub`, so the provenance has to be too. Includes the
            # composed source_class_filter, every numeric cut (NaN = no cut on
            # that statistic), the name-filter digest, the FAR policy, and
            # CUT_ESTIMATOR_ATTR -- WHICH masses the class threshold was applied
            # to (GW-12): the store's source_class column is classify_by_mass()
            # of the POSTERIOR MEDIAN source-frame masses, while the paired
            # selection file cuts its injections on injected truth. Recording it
            # is what lets the paired-file validator refuse that combination
            # instead of shipping a beta that describes a cut nobody applied.
            for _k, _v in spec.to_attrs().items():
                f.attrs[_k] = _v
            # WHICH events this file holds, independent of how they were
            # spelled: a sorted, de-duplicated digest of the written names.
            from .export.contract import event_list_digest
            f.attrs["event_list_digest"] = event_list_digest(kept)
            # --- Waveform / sample-set provenance (PR 6) ---
            # homogeneous_sample_sets describes the SELECTED VIEW: False iff it
            # holds more than one sample set of one event (only possible under
            # waveform_policy="all").  A multi-waveform export is thus never
            # advertised as homogeneous, even if a duplicate row was later
            # skipped by z_max/undersampling and the written file happens to be
            # one-row-per-event.
            f.attrs["waveform_policy"] = str(waveform_policy)
            f.attrs["approximant"] = "" if approximant is None else str(approximant)
            # Derived from the policy resolution, not re-inferred from name
            # uniqueness: `kept` holds store ROW indices, so two sample sets of
            # one event are two distinct rows and the old test called that
            # homogeneous. Under waveform_policy="all" the same physical event
            # is written N times and the hierarchical likelihood counts each as
            # an independent detection, which is exactly what this flag exists
            # to warn about.
            f.attrs["homogeneous_sample_sets"] = bool(
                getattr(sub, "_homogeneous_sample_sets", True))
            # Resampling provenance (GW-13).
            f.attrs["n_unique_samples_per_event"] = np.asarray(
                n_unique_per_event, dtype=np.int64)
            f.attrs["resampled_with_replacement"] = bool(upsampled_events)
            f.attrs["n_events_resampled_with_replacement"] = int(
                len(upsampled_events))
            f.attrs.create("sample_set_name_per_event",
                           np.array([str(x) for x in kept_ss_name],
                                    dtype=h5py.string_dtype()))
            f.attrs.create("sample_set_approximant_per_event",
                           np.array([str(x) for x in kept_ss_approx],
                                    dtype=h5py.string_dtype()))
            f.attrs.create("sample_set_selection_reason",
                           np.array([str(x) for x in kept_ss_reason],
                                    dtype=h5py.string_dtype()))
            f.attrs.create("event_names",
                           np.array([str(k) for k in kept],
                                    dtype=h5py.string_dtype()))
            # --- Datasets ---
            for k in ["ra", "dec", "m1det", "m2det", "chieff", "dL", "p_pe",
                       "redshift", "m1src", "m2src"]:
                f.create_dataset(k, data=data[k], compression="gzip",
                                 shuffle=False)

        if write_summary:
            # Generic diagnostics over the post-select() population (`sub`,
            # i.e. events considered for export after metadata + waveform-policy
            # cuts) via the same honest counting logic `gwcat inspect` and
            # ingest's write_summary use, overlaid with export-specific
            # provenance already computed above. `n_events_exported`/`nobs` can
            # be slightly smaller than `n_events_considered` when the per-event
            # z_max cut or replace=False drops an event entirely inside the
            # sampling loop above.
            from .validation_summary import summarize_catalog, write_validation_summary
            summary = summarize_catalog(sub)
            summary.update({
                "kind": "darksirens_export",
                "output_path": str(out_path),
                "n_events_considered": int(sub.n_events),
                "n_events_exported": int(nobs),
                "n_events_skipped_after_selection": int(sub.n_events - nobs),
                "event_names_exported": [str(k) for k in kept],
                "nsamp_per_event": int(nsamp),
                # The EFFECTIVE selection (GW-33), the same record the file's
                # attrs carry.
                "selection_spec": spec.to_dict(),
                "selection_spec_digest": spec.digest(),
                "event_list_digest": event_list_digest(kept),
                "source_class_filter": spec.source_class_filter or None,
                CUT_ESTIMATOR_ATTR: spec.cut_estimator,
                "event_list_filter": spec.event_list_filter or None,
                "far_policy": spec.far_policy,
                "allow_missing_far": bool(spec.allow_missing_far),
                "require_far": bool(spec.require_far),
                "n_events_missing_far": int(spec.n_missing_far),
                "spin_prior_mode": spin_prior_mode,
                "chi_eff_prior_applied_to_p_pe": bool(chi_eff_included),
                "cosmology_mode": cosmology_mode,
                "cosmology_override_used": bool(cosmology is not None),
                "cosmology_per_event_varies": bool(cosmology_per_event_varies),
                "waveform_policy": str(waveform_policy),
                "approximant": None if approximant is None else str(approximant),
                "homogeneous_sample_sets": bool(
                    len(set(str(k) for k in kept)) == len(kept)),
            })
            if summary_context:
                summary.update(summary_context)
            write_validation_summary(out_path, summary)

        if cosmology_mode == "per-event" and cosmology_per_event_varies:
            cosmo_desc = "cosmology=per-event (varies across events)"
        else:
            cosmo_desc = f"H0={pe_H0}, Om0={pe_Om0} ({cosmology_mode})"
        print(f"Wrote {out_path}: nobs={nobs}, nsamp={nsamp}, "
              f"{cosmo_desc}, compact_type={compact_type}")
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

    # ---- versioned export (PR 3): build a product, look up a writer ----------
    def export(self, out_path, format="gwcat2", spin_basis=None,
               write_summary=False, summary_context=None, **builder_kwargs):
        """Export via the versioned :mod:`gwcat.export` pipeline.

        Thin dispatch: build a PE :class:`~gwcat.export.product.ExportProduct`
        (which owns ALL the physics -- selection, cosmology, resampling, the
        mass Jacobian and spin prior), look up the ``(format, "pe")`` writer,
        and serialize.  For ``format="gwcat2", spin_basis="chieff"`` the arrays
        are byte-identical to :meth:`to_darksirens` with the same kwargs; the
        file carries ``format_version="gwcat-pe-2.0"``.  Unlike
        :meth:`to_darksirens`, the mass Jacobian is applied in the builder, not
        here -- this method never touches sample arrays.

        ``spin_basis=None`` means :data:`gwcat.params.DEFAULT_PARAMETER_SPACE`
        (``component``), the same value the CLI uses, so a no-argument PE export
        and a no-argument selection export pair.  It used to default to
        ``chieff`` here while the selection side defaulted to ``component``, so
        the no-argument Python pair failed the cross-file basis check by
        construction -- the same defect as the CLI's, one layer down.  Pass
        ``spin_basis="chieff"`` for the legacy behaviour; :meth:`to_darksirens`
        (frozen v1) is unaffected either way.

        ``**builder_kwargs`` are forwarded to
        :func:`gwcat.export.build_pe_product` (``nsamp``, ``seed``, ``far_max``,
        ``pastro_min``, ``z_max``, ``replace``, ``cosmology``, ``amax``,
        ``allowed_names``, ``source_class``, ``event_list``, ``far`` policy
        flags, ``waveform_policy``, ``approximant``).
        """
        from .export import build_pe_product, get_exporter
        from .params import DEFAULT_PARAMETER_SPACE
        if spin_basis is None:
            spin_basis = DEFAULT_PARAMETER_SPACE
        product = build_pe_product(self, spin_basis=spin_basis,
                                   **builder_kwargs)
        writer = get_exporter(format, "pe")
        return writer(product, out_path, write_summary=write_summary,
                      summary_context=summary_context)

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
        pastro_col = self._pastro_column()
        for j, i in enumerate(self._sel):
            name = self.names[i]
            cat = self.meta["catalog"][i] if "catalog" in self.meta else "?"
            ana = self.meta["analysis_used"][i] if "analysis_used" in self.meta else "?"
            m1 = self.meta["m1_src_med"][i] if "m1_src_med" in self.meta else np.nan
            m2 = self.meta["m2_src_med"][i] if "m2_src_med" in self.meta else np.nan
            ct = self.meta["compact_type"][i] if "compact_type" in self.meta else "?"
            far = self.meta["far"][i] if "far" in self.meta else np.nan
            pa = pastro_col[i]

            far_s = f"{far:.2e}" if np.isfinite(far) else "NaN"
            pa_s = f"{pa:.3f}" if np.isfinite(pa) else "NaN"
            m1_s = f"{m1:.1f}" if np.isfinite(m1) else "?"
            m2_s = f"{m2:.1f}" if np.isfinite(m2) else "?"

            print(f"{name:<22} {str(cat):<10} {str(ana):<22} {ns[j]:>6} "
                  f"{m1_s:>6} {m2_s:>6} {str(ct):<5} "
                  f"{far_s:>10} {pa_s:>7}")
        print(f"\n{self.n_events} events, "
              f"{int(ns.sum())} total samples")


#: Datasets the ``gwcat-1.0`` PE schema MANDATES -- exactly what
#: :meth:`GWCatalog.to_darksirens` writes and what darksirens' loader reads.
#: Their presence is checked, not assumed: the loop below used to say
#: ``if ds in f`` and skip the check when the dataset was absent, so a file
#: missing p_pe / the masses / the sky produced no failure at all.
V1_PE_REQUIRED = ("ra", "dec", "m1det", "m2det", "chieff", "dL", "p_pe",
                  "redshift", "m1src", "m2src")

#: Datasets the ``gwcat-selection-1.0`` schema MANDATES (what
#: ``SelectionSet``/``CombinedSelectionSet.to_darksirens`` write).
V1_SELECTION_REQUIRED = ("m1det", "m2det", "dL", "chieff", "ra", "dec",
                         "m1src", "m2src", "redshift", "pdraw")


def validate_export(gw_path: str, selection_path: str = None, strict: bool = False):
    """Check a darksirens PE export (and optionally a selection export) for
    internal consistency.

    This is the FROZEN v1 validator, and it validates the ``gwcat-1.0`` /
    ``gwcat-selection-1.0`` contract only.  The public entry point is
    :func:`gwcat.validate_export` (i.e.
    :func:`gwcat.export.validate.validate_export_any`), which dispatches on
    each file's ``format_version`` and sends v2 files to
    :func:`gwcat.export.validate_export_v2`.

    Checks:
      * every dataset the gwcat-1.0 schema mandates is PRESENT
        (:data:`V1_PE_REQUIRED`, :data:`V1_SELECTION_REQUIRED`) -- an absent
        one is a failed check, not a skipped one
      * array lengths == nobs * nsamp
      * p_pe finite and STRICTLY positive -- an exact zero is a hard failure
        (GW-01), not a documented "distance-prior tail"
      * source masses <= detector masses
      * redshift non-negative
      * format_version present and a v1 one
      * if selection_path: cosmology consistent, pdraw finite/positive,
        ndraw > n_detected, chi_eff_swap_applied flag present

    Cross-validation (PR 9).  When ``selection_path`` is given, the PE export and
    the selection export are checked for *contract* agreement and any mismatch
    raises ``ValueError`` with a specific message (independent of ``strict`` --
    a contract mismatch is always a hard error, since the whole point is to stop
    a file that looks valid while carrying the wrong prior/cosmology/source
    class):

      (a) spin-prior contract: ``spin_prior_mode`` must match and the
          ``chi_eff_prior_applied_to_p_pe`` / ``chi_eff_prior_applied_to_pdraw``
          flags must agree (else the chi_eff prior is double-counted / omitted);
      (b) cosmology: the selection cosmology must match the PE cosmology.  When
          the PE export carries per-event cosmologies
          (``cosmology_per_event_varies=True``, PR 4), every per-event value is
          compared against the single selection cosmology rather than the legacy
          scalar;
      (c) source-class compatibility: the PE ``source_class_filter`` and the
          selection ``source_class_filter`` must resolve to the same canonical
          class set (the injections must cover the same source class(es) as the
          PE events), and their ``source_class_cut_estimator`` must agree when
          both sides really do carry a class filter -- a median-based event cut
          paired with a truth-based injection cut is the same threshold on a
          different quantity, and biases beta at the class boundary (GW-12).

    Returns a dict of {check_name: passed_bool}.  If strict=True, raises on
    the first internal-consistency failure; the cross-validation contract
    checks above always raise on mismatch.
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

        pe_version = f.attrs.get("format_version")
        pe_version = (pe_version.decode()
                      if isinstance(pe_version, (bytes, bytearray))
                      else pe_version)
        _check("pe_format_version", pe_version == "gwcat-1.0",
               f"format_version={pe_version!r} is not 'gwcat-1.0'; this is the "
               f"frozen v1 validator. Call gwcat.validate_export, which "
               f"dispatches on format_version.")
        _check("pe_mock_data_attr", "mock_data" in f.attrs)
        _check("pe_cosmology_H0", "pe_cosmology_H0" in f.attrs)

        # Presence FIRST: a mandated dataset that is absent is a failure, not a
        # check that quietly does not run.
        for ds in V1_PE_REQUIRED:
            _check(f"pe_has_{ds}", ds in f,
                   f"dataset {ds!r} missing; the gwcat-1.0 PE schema mandates "
                   f"{list(V1_PE_REQUIRED)}")

        for ds in V1_PE_REQUIRED:
            if ds in f:
                _check(f"pe_{ds}_length", f[ds].shape[0] == expected,
                       f"{f[ds].shape[0]} != {expected}")

        if "p_pe" in f:
            p = np.array(f["p_pe"])
            if p.size > 0:
                _check("pe_p_pe_finite", np.all(np.isfinite(p)))
                # Strictly positive (GW-01).  An exact zero was previously
                # tolerated as a "distance-prior tail", but that state only ever
                # came from the truncated distance prior, and darksirens turns a
                # zero weight into a -inf log-weight that still counts in n.
                n_zero = int(np.sum(p == 0))
                n_neg = int(np.sum(p < 0))
                _check("pe_p_pe_positive", n_zero == 0 and n_neg == 0,
                       f"{n_zero} zero ({100 * n_zero / p.size:.2f}%) and "
                       f"{n_neg} negative of {p.size} samples; "
                       f"min={np.nanmin(p):.3e}. A zero p_pe means the store's "
                       f"p_dL_pe was truncated at the recorded distance-prior "
                       f"bounds -- re-ingest so the distance prior is evaluated "
                       f"over the full sample range.")
                # Check no event is entirely zero-weight.  A ragged file is the
                # exact condition this function exists to report, so record it
                # as a failed check -- reshape would raise and take the whole
                # report down with it, on the one file that most needs a report.
                rect = (nobs > 0 and nsamp > 0 and p.size == nobs * nsamp)
                _check("pe_rectangular", rect,
                       f"p_pe has {p.size} samples but nobs*nsamp = {nobs}*"
                       f"{nsamp} = {nobs * nsamp}; the file is not reshapeable "
                       f"and no consumer can read it")
                p_ev = p.reshape(nobs, nsamp) if rect else p
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
            sel_version = f.attrs.get("format_version")
            sel_version = (sel_version.decode()
                           if isinstance(sel_version, (bytes, bytearray))
                           else sel_version)
            _check("sel_format_version",
                   sel_version == "gwcat-selection-1.0",
                   f"format_version={sel_version!r} is not "
                   f"'gwcat-selection-1.0'; this is the frozen v1 validator. "
                   f"Call gwcat.validate_export, which dispatches on "
                   f"format_version.")
            _check("sel_chi_eff_swap_flag", "chi_eff_swap_applied" in f.attrs)

            for ds in V1_SELECTION_REQUIRED:
                _check(f"sel_has_{ds}", ds in f,
                       f"dataset {ds!r} missing; the gwcat-selection-1.0 "
                       f"schema mandates {list(V1_SELECTION_REQUIRED)}")

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

        # ── Cross-validation: PE export <-> selection export contract ──────
        # Contract mismatches ALWAYS raise (a "looks-valid" file with the wrong
        # prior/cosmology/source-class is exactly what we must stop).
        with h5py.File(gw_path, "r") as fg, h5py.File(selection_path, "r") as fs:
            def _sattr(fobj, name, default=None):
                v = fobj.attrs.get(name, default)
                return v.decode() if isinstance(v, bytes) else v

            def _fail(name, msg):
                results[name] = False
                raise ValueError(f"validate_export FAILED: {name}. {msg}")

            # (a) Spin-prior contract agreement -----------------------------
            pe_mode = _sattr(fg, "spin_prior_mode")
            sel_mode = _sattr(fs, "spin_prior_mode")
            if pe_mode is not None and sel_mode is not None:
                if pe_mode != sel_mode:
                    _fail("xcheck_spin_prior_mode",
                          f"PE spin_prior_mode={pe_mode!r} but selection "
                          f"spin_prior_mode={sel_mode!r}. Both must match or the "
                          f"chi_eff prior is double-counted / omitted.")
                results["xcheck_spin_prior_mode"] = True
            if ("chi_eff_prior_applied_to_p_pe" in fg.attrs
                    and "chi_eff_prior_applied_to_pdraw" in fs.attrs):
                pe_chi = bool(fg.attrs["chi_eff_prior_applied_to_p_pe"])
                sel_chi = bool(fs.attrs["chi_eff_prior_applied_to_pdraw"])
                if pe_chi != sel_chi:
                    _fail("xcheck_chi_eff_flag",
                          f"PE chi_eff_prior_applied_to_p_pe={pe_chi} but "
                          f"selection chi_eff_prior_applied_to_pdraw={sel_chi}. "
                          f"The chi_eff prior must be applied to both or neither.")
                results["xcheck_chi_eff_flag"] = True

            # (b) Cosmology agreement (PR 4 per-event arrays) ---------------
            sel_H0 = _sattr(fs, "cosmology_H0")
            sel_Om = _sattr(fs, "cosmology_Om0")
            if sel_H0 is not None and sel_Om is not None:
                varies = bool(fg.attrs.get("cosmology_per_event_varies", False))
                if varies:
                    pe_H0_arr = np.asarray(
                        fg.attrs.get("cosmology_H0_per_event", []), dtype=float)
                    pe_Om_arr = np.asarray(
                        fg.attrs.get("cosmology_Om0_per_event", []), dtype=float)
                    bad = ((np.abs(pe_H0_arr - float(sel_H0)) >= 1.0).any()
                           or (np.abs(pe_Om_arr - float(sel_Om)) >= 0.05).any())
                    if bad:
                        h0rng = ((pe_H0_arr.min(), pe_H0_arr.max())
                                 if pe_H0_arr.size else ("?", "?"))
                        omrng = ((pe_Om_arr.min(), pe_Om_arr.max())
                                 if pe_Om_arr.size else ("?", "?"))
                        _fail("xcheck_cosmology",
                              f"PE export uses per-event cosmologies "
                              f"(cosmology_per_event_varies=True) with H0 in "
                              f"{h0rng} and Om0 in {omrng} that do not all match "
                              f"the single selection cosmology (H0={sel_H0}, "
                              f"Om0={sel_Om}). Re-export the selection under a "
                              f"matching cosmology, or use a single-cosmology PE "
                              f"export.")
                    results["xcheck_cosmology"] = True
                else:
                    pe_H0 = fg.attrs.get("pe_cosmology_H0")
                    pe_Om = fg.attrs.get("pe_cosmology_Om0")
                    if pe_H0 is not None and abs(float(pe_H0) - float(sel_H0)) >= 1.0:
                        _fail("xcheck_cosmology",
                              f"PE cosmology H0={pe_H0} disagrees with selection "
                              f"H0={sel_H0} (|Δ| >= 1.0).")
                    if pe_Om is not None and abs(float(pe_Om) - float(sel_Om)) >= 0.05:
                        _fail("xcheck_cosmology",
                              f"PE cosmology Om0={pe_Om} disagrees with selection "
                              f"Om0={sel_Om} (|Δ| >= 0.05).")
                    results["xcheck_cosmology"] = True

            # (c) Source-class compatibility --------------------------------
            def _sc_classes(raw):
                s = "" if raw is None else str(raw)
                if s == "":
                    return set(SOURCE_CLASSES)
                from .source_class import parse_source_class_filter
                return set(parse_source_class_filter(s))

            pe_scf = _sattr(fg, "source_class_filter", "")
            sel_scf = _sattr(fs, "source_class_filter", "")
            try:
                pe_classes = _sc_classes(pe_scf)
                sel_classes = _sc_classes(sel_scf)
            except ValueError as exc:
                # Mirrors the v2 validator: a pre-canonical-form file recorded
                # a Python repr that was never checkable.  Report it as this
                # check's failure instead of raising an unexplained parse
                # error out of the middle of validation.
                _fail("xcheck_source_class",
                      f"unparseable source_class_filter (PE={pe_scf!r}, "
                      f"selection={sel_scf!r}): {exc} Files written before "
                      f"the canonical form record a Python repr that was "
                      f"never checkable; re-export to make the pairing "
                      f"verifiable.")  # _fail raises; nothing runs below
            if pe_classes != sel_classes:
                _fail("xcheck_source_class",
                      f"PE source_class_filter={pe_scf!r} -> {sorted(pe_classes)} "
                      f"but selection source_class_filter={sel_scf!r} -> "
                      f"{sorted(sel_classes)}. The selection injections must "
                      f"cover the same source class(es) as the PE events.")
            results["xcheck_source_class"] = True

            # (c2) The class cut's ESTIMATOR (GW-12).  The check above proves
            # the two sides asked for the same CLASSES; it cannot see that they
            # asked different questions to get them -- PE events are classified
            # from posterior median source-frame masses, injections from
            # injected truth.  Same threshold, different quantity, and the
            # exported selection function then describes a cut nobody applied.
            # Kept in lockstep with gwcat.export.validate's v2 twin: fixing
            # only the v2 path is the recurring failure mode here (GW-25).
            def _filtered(raw):
                from .source_class import parse_source_class_filter
                try:
                    return bool(parse_source_class_filter(raw))
                except ValueError:
                    return False

            verdict, msg = assess_cut_estimator_pair(
                _sattr(fg, CUT_ESTIMATOR_ATTR, None),
                _sattr(fs, CUT_ESTIMATOR_ATTR, None),
                _filtered(pe_scf), _filtered(sel_scf))
            if verdict == "fail":
                _fail("xcheck_source_class_estimator", msg)
            if verdict == "warn":
                warnings.warn(msg)
            results["xcheck_source_class_estimator"] = True

    n_pass = sum(results.values())
    n_total = len(results)
    status = "ALL PASSED" if n_pass == n_total else f"{n_total - n_pass} FAILED"
    print(f"  {n_pass}/{n_total} checks: {status}")
    return results