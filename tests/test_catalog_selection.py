"""Catalog selection-layer tests (PR 2).

Covers user event-list filtering and the explicit missing-FAR policy on the
``far_max`` cut:

  * user event-list file / in-memory sequence filtering,
  * require_far=True fails loudly when a selected event has no FAR,
  * allow_missing_far=True keeps missing-FAR events (records + warns),
  * the default drops missing-FAR events (legacy behavior) with a warning,
  * to_darksirens writes the FAR policy into output provenance attributes.

Reuses the tiny synthetic mixed-catalog builder from
test_source_class_filters.py.
"""
import warnings

import numpy as np
import h5py
import pytest

from gwcat.catalog import GWCatalog
from gwcat.source_class import load_event_list

from test_source_class_filters import build_mixed_store, MIXED_EVENTS


# Two BBH with FAR, one BBH with MISSING FAR (np.nan), one NSBH with FAR.
FAR_EVENTS = [
    {"name": "GW910001_000001", "source_class": "BBH", "far": 1e-3},
    {"name": "GW910002_000002", "source_class": "BBH", "far": 5e-2},
    {"name": "GW910003_000003", "source_class": "BBH", "far": np.nan},  # missing
    {"name": "GW910004_000004", "source_class": "NSBH", "far": 2e-3},
]


# ── user event-list filtering ────────────────────────────────────────────────
def test_event_list_file_filter(tmp_path):
    cat = GWCatalog(build_mixed_store(tmp_path, FAR_EVENTS))
    list_path = tmp_path / "my_events.txt"
    list_path.write_text(
        "# my analysis subset\n"
        "GW910001_000001\n"
        "GW910004_000004  # keep the NSBH\n"
        "\n"
    )
    sub = cat.select(event_list=str(list_path))
    assert sub.n_events == 2
    assert set(sub.event_names) == {"GW910001_000001", "GW910004_000004"}


def test_event_list_sequence_filter(tmp_path):
    cat = GWCatalog(build_mixed_store(tmp_path, FAR_EVENTS))
    sub = cat.select(event_list=["GW910002_000002"])
    assert sub.n_events == 1
    assert list(sub.event_names) == ["GW910002_000002"]


def test_event_list_warns_on_unknown_names(tmp_path):
    cat = GWCatalog(build_mixed_store(tmp_path, FAR_EVENTS))
    with pytest.warns(UserWarning, match="not found in store"):
        sub = cat.select(event_list=["GW910001_000001", "GW000000_999999"])
    assert sub.n_events == 1


def test_event_list_combines_with_source_class(tmp_path):
    cat = GWCatalog(build_mixed_store(tmp_path, FAR_EVENTS))
    sub = cat.select(
        source_class="bbh",
        event_list=["GW910001_000001", "GW910004_000004"],  # 1 BBH + 1 NSBH
    )
    # intersection of BBH and the list -> only the BBH survives
    assert list(sub.event_names) == ["GW910001_000001"]


def test_load_event_list_helper(tmp_path):
    p = tmp_path / "events.txt"
    p.write_text("GWa\nGWb  # note\n\n# comment line\nGWa\n")
    # blanks/comments dropped, duplicates de-duped, order preserved
    assert load_event_list(str(p)) == ["GWa", "GWb"]
    assert load_event_list(["GWx", "GWy"]) == ["GWx", "GWy"]


# ── missing-FAR policy ───────────────────────────────────────────────────────
def test_require_far_fails_when_missing(tmp_path):
    cat = GWCatalog(build_mixed_store(tmp_path, FAR_EVENTS))
    with pytest.raises(ValueError, match="require_far=True"):
        cat.select(far_max=1.0, require_far=True)


def test_require_far_passes_when_all_present(tmp_path):
    # restrict to the events that DO have FAR, then require_far succeeds
    cat = GWCatalog(build_mixed_store(tmp_path, FAR_EVENTS))
    sub = cat.select(
        event_list=["GW910001_000001", "GW910002_000002", "GW910004_000004"],
        far_max=1.0, require_far=True,
    )
    assert sub.n_events == 3
    assert sub._far_policy == "require"
    assert sub._n_missing_far == 0


def test_allow_missing_far_keeps_and_records(tmp_path):
    cat = GWCatalog(build_mixed_store(tmp_path, FAR_EVENTS))
    with pytest.warns(UserWarning, match="allow_missing_far=True"):
        sub = cat.select(far_max=1.0, allow_missing_far=True)
    # far<=1.0 keeps 3 events with FAR, PLUS the 1 missing-FAR event = 4
    assert sub.n_events == 4
    assert "GW910003_000003" in set(sub.event_names)  # missing-FAR kept
    assert sub._far_policy == "allow_missing"
    assert sub._n_missing_far == 1


def test_default_drops_missing_far(tmp_path):
    cat = GWCatalog(build_mixed_store(tmp_path, FAR_EVENTS))
    with pytest.warns(UserWarning, match="missing FAR"):
        sub = cat.select(far_max=1.0)  # default: drop missing FAR
    assert sub.n_events == 3
    assert "GW910003_000003" not in set(sub.event_names)
    assert sub._far_policy == "drop_missing"
    assert sub._n_missing_far == 1


def test_require_and_allow_missing_conflict(tmp_path):
    cat = GWCatalog(build_mixed_store(tmp_path, FAR_EVENTS))
    with pytest.raises(ValueError, match="mutually exclusive"):
        cat.select(far_max=1.0, require_far=True, allow_missing_far=True)


def test_far_available_fallback_without_column(tmp_path):
    """A store with no far_available column derives availability from far NaN."""
    path = build_mixed_store(tmp_path, FAR_EVENTS, include_far_available=False)
    with h5py.File(path, "r") as f:
        assert "far_available" not in f["meta"]
    cat = GWCatalog(path)
    with pytest.warns(UserWarning, match="missing FAR"):
        sub = cat.select(far_max=1.0)
    assert sub.n_events == 3
    assert sub._n_missing_far == 1


# ── provenance in the darksirens export ──────────────────────────────────────
def test_to_darksirens_records_far_policy_allow_missing(tmp_path):
    cat = GWCatalog(build_mixed_store(tmp_path, FAR_EVENTS))
    out = tmp_path / "allow_far.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cat.to_darksirens(str(out), far_max=1.0, allow_missing_far=True,
                          nsamp=8, seed=0, cosmology=(67.74, 0.3089))
    with h5py.File(out, "r") as f:
        assert f.attrs["far_policy"] == "allow_missing"
        assert bool(f.attrs["allow_missing_far"]) is True
        assert bool(f.attrs["require_far"]) is False
        assert int(f.attrs["n_events_missing_far"]) == 1
        assert f.attrs["nobs"] == 4


def test_to_darksirens_require_far_raises(tmp_path):
    cat = GWCatalog(build_mixed_store(tmp_path, FAR_EVENTS))
    out = tmp_path / "wont_exist.h5"
    with pytest.raises(ValueError, match="require_far=True"):
        cat.to_darksirens(str(out), far_max=1.0, require_far=True,
                          nsamp=8, seed=0, cosmology=(67.74, 0.3089))
    assert not out.exists()


def test_to_darksirens_default_far_policy_attr(tmp_path):
    cat = GWCatalog(build_mixed_store(tmp_path, FAR_EVENTS))
    out = tmp_path / "default_far.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cat.to_darksirens(str(out), far_max=1.0, nsamp=8, seed=0,
                          cosmology=(67.74, 0.3089))
    with h5py.File(out, "r") as f:
        assert f.attrs["far_policy"] == "drop_missing"
        assert f.attrs["n_events_missing_far"] == 1
        assert f.attrs["nobs"] == 3


# ==========================================================================
# GW-13: four silent-wrong-answer defects in the catalog view
# ==========================================================================
def test_z_max_on_select_raises_instead_of_being_ignored(tmp_path):
    """`select(z_max=...)` was accepted and silently did nothing.

    The same keyword IS implemented on the exporters as a per-sample cut, so an
    analysis that asked select() for it shipped uncut with no warning.
    """
    cat = GWCatalog(build_mixed_store(tmp_path, MIXED_EVENTS))
    with pytest.raises(NotImplementedError) as ei:
        cat.select(z_max=0.5)
    msg = str(ei.value)
    # Must point at where the cut actually lives...
    assert "to_darksirens" in msg
    # ...and say why an event-level median cut is not offered instead.
    assert "median" in msg or "point estimate" in msg


def test_select_without_z_max_is_unaffected(tmp_path):
    cat = GWCatalog(build_mixed_store(tmp_path, MIXED_EVENTS))
    assert cat.select(source_class="bbh").n_events == 2


def test_event_respects_selection_and_policy(tmp_path):
    """`event()` indexed the whole store, ignoring the current view.

    Per-event inspection could therefore disagree with the file exported from
    that same view.
    """
    cat = GWCatalog(build_mixed_store(tmp_path, MIXED_EVENTS))
    bbh = cat.select(source_class="bbh")

    # An event inside the view still reads.
    d = bbh.event("GW900001_000001")
    assert len(d["mass_1"]) > 0

    # One filtered OUT must raise, and say it exists but was filtered -- the
    # two failure modes need different fixes.
    with pytest.raises(KeyError) as ei:
        bbh.event("GW900004_000004")          # the BNS
    msg = str(ei.value)
    assert "in the store but not in this view" in msg

    # A name that does not exist at all is a different message.
    with pytest.raises(KeyError) as ei2:
        bbh.event("GW999999_999999")
    assert "not in" in str(ei2.value)


def test_event_returns_the_same_samples_the_view_would_export(tmp_path):
    cat = GWCatalog(build_mixed_store(tmp_path, MIXED_EVENTS))
    bbh = cat.select(source_class="bbh")
    ev = bbh.event("GW900002_000002")
    per = bbh.get(["mass_1"], per_event=True)["mass_1"]
    idx = list(bbh.event_names).index("GW900002_000002")
    np.testing.assert_array_equal(ev["mass_1"], per[idx])


def test_upsampling_recorded(tmp_path):
    """Bootstrapping an under-sampled event used to leave no trace at all."""
    few = [dict(e, n=6) for e in MIXED_EVENTS[:2]]
    cat = GWCatalog(build_mixed_store(tmp_path, few, name="few.h5"))
    out = tmp_path / "up.h5"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cat.to_darksirens(str(out), nsamp=64, seed=0,
                          cosmology=(67.74, 0.3089))
    assert any("bootstrapped WITH replacement" in str(x.message) for x in w)

    with h5py.File(out, "r") as f:
        assert bool(f.attrs["resampled_with_replacement"]) is True
        assert int(f.attrs["n_events_resampled_with_replacement"]) == 2
        nu = np.asarray(f.attrs["n_unique_samples_per_event"])
        assert nu.size == int(f.attrs["nobs"])
        # 64 rows drawn from 6 distinct samples: the ESS a naive reader would
        # compute is inflated by ~10x, which is what the attr exists to expose.
        assert np.all(nu <= 6)


def test_no_upsampling_flag_when_samples_are_plentiful(tmp_path):
    cat = GWCatalog(build_mixed_store(tmp_path, MIXED_EVENTS))
    out = tmp_path / "noup.h5"
    cat.to_darksirens(str(out), nsamp=8, seed=0, cosmology=(67.74, 0.3089))
    with h5py.File(out, "r") as f:
        assert bool(f.attrs["resampled_with_replacement"]) is False
        assert np.all(np.asarray(f.attrs["n_unique_samples_per_event"]) == 8)


def test_homogeneity_comes_from_the_policy_not_name_uniqueness(tmp_path):
    """`kept` holds store ROW indices, so two sample sets of one event are two
    distinct rows and the old uniqueness test called that homogeneous."""
    cat = GWCatalog(build_mixed_store(tmp_path, MIXED_EVENTS))
    out = tmp_path / "hom.h5"
    cat.to_darksirens(str(out), nsamp=8, seed=0, cosmology=(67.74, 0.3089))
    sub = cat.select()
    with h5py.File(out, "r") as f:
        assert (bool(f.attrs["homogeneous_sample_sets"])
                is bool(sub._homogeneous_sample_sets))


# ── p_astro / pastro: one quantity, two spellings (GW-14) ───────────────────
PASTRO_EVENTS = [
    # populated ONLY under the `p_astro` spelling -- what a manifest or
    # override file following this package's own documented example produces.
    {"name": "GW930001_000001", "source_class": "BBH", "far": 1e-3,
     "p_astro": 0.999},
    {"name": "GW930002_000002", "source_class": "BBH", "far": 5e-4,
     "p_astro": 0.4},
]


def test_pastro_min_reads_the_p_astro_spelling(tmp_path):
    """A store whose event table used `p_astro` is still selectable.

    The cut used to read `meta/pastro` alone, so this store -- populated by the
    documented override key -- returned zero events with no error at all.
    """
    cat = GWCatalog(build_mixed_store(tmp_path, PASTRO_EVENTS,
                                      name="pastro_only.h5"))
    assert np.all(np.isnan(np.asarray(cat.meta["pastro"], dtype=float)))
    sub = cat.select(pastro_min=0.9)
    assert list(sub.event_names) == ["GW930001_000001"]


def test_pastro_min_prefers_a_finite_value_row_by_row(tmp_path):
    """Resolution is per row: one event under each spelling, both selectable."""
    events = [dict(PASTRO_EVENTS[0]),
              {"name": "GW930003_000003", "source_class": "BBH", "far": 1e-3,
               "pastro": 0.95}]
    cat = GWCatalog(build_mixed_store(tmp_path, events, name="both.h5"))
    sub = cat.select(pastro_min=0.9)
    assert set(sub.event_names) == {"GW930001_000001", "GW930003_000003"}


def test_pastro_min_still_cuts_and_warns_on_partial_absence(tmp_path):
    events = [dict(PASTRO_EVENTS[0]),
              {"name": "GW930004_000004", "source_class": "BBH", "far": 1e-3}]
    cat = GWCatalog(build_mixed_store(tmp_path, events, name="partial.h5"))
    with pytest.warns(UserWarning, match="no p_astro under either spelling"):
        sub = cat.select(pastro_min=0.9)
    assert list(sub.event_names) == ["GW930001_000001"]


def test_all_nan_pastro_raises_not_warns(tmp_path):
    """A threshold crossed with an all-absent column is a config error.

    Every event NaN under both spellings used to yield an empty catalog and a
    warning that could not even fire (it was issued after the cut that removed
    the NaN rows it described).
    """
    cat = GWCatalog(build_mixed_store(tmp_path, FAR_EVENTS, name="nopa.h5"))
    with pytest.raises(ValueError, match="carries no p_astro at all"):
        cat.select(pastro_min=0.5)


def test_pastro_summary_shows_the_resolved_value(tmp_path, capsys):
    cat = GWCatalog(build_mixed_store(tmp_path, PASTRO_EVENTS,
                                      name="pastro_summary.h5"))
    cat.summary()
    assert "0.999" in capsys.readouterr().out


# ==========================================================================
# GW-33: the EFFECTIVE selection survives a chain of views
# ==========================================================================
def _spec(cat, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cat.select(**kw).selection_spec


def test_select_composes_the_filters_of_the_view_it_was_called_on(tmp_path):
    """The reviewer's reproduction, in miniature.

    ``select()`` has always ANDed its mask with the view it was called on, so
    the ROWS were right; the provenance it attached described only the LAST
    call.  Exporting calls ``select()`` once more with its own defaults, and the
    filtered file then recorded no class filter at all.
    """
    cat = GWCatalog(build_mixed_store(tmp_path, MIXED_EVENTS))
    bbh = cat.select(source_class="bbh")
    again = bbh.select()                    # what an exporter does

    assert again.n_events == bbh.n_events == 2      # rows were never the bug
    spec = again.selection_spec
    assert spec.source_class == ("BBH",)
    assert spec.source_class_filter == "BBH"
    assert spec.cut_estimator == "posterior_median_mass"
    assert spec.is_filtered is True


def test_chained_thresholds_compose_to_the_tighter_cut(tmp_path):
    """Two chained cuts leave the tighter one standing -- which is the one the
    surviving rows actually reflect."""
    events = [dict(e, pastro=0.99) for e in MIXED_EVENTS]
    cat = GWCatalog(build_mixed_store(tmp_path, events, name="tight.h5"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sub = (cat.select(far_max=1e-2, pastro_min=0.5)
                  .select(far_max=1e-3, pastro_min=0.9, snr_min=None))
    spec = sub.selection_spec
    assert spec.far_max == 1e-3          # upper bound -> the smaller
    assert spec.pastro_min == 0.9        # lower bound -> the larger
    assert spec.far_policy in ("drop_missing", "require", "allow_missing")


def test_every_name_filter_normalises_into_one_whitelist(tmp_path):
    """``allowed_names``, ``names`` and ``event_list`` are one filter --
    membership in a fixed list -- so they compose into one set and one digest."""
    cat = GWCatalog(build_mixed_store(tmp_path, MIXED_EVENTS))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sub = (cat.select(allowed_names=["GW900001_000001", "GW900002_000002"])
                  .select(event_list=["GW900002_000002", "GW900003_000003"]))
    spec = sub.selection_spec
    assert list(sub.event_names) == ["GW900002_000002"]
    assert spec.allowed_names == ("GW900002_000002",)      # the intersection
    assert spec.allowed_names_filter == "allowed_names"
    assert spec.event_list_filter == "custom_sequence"
    # A direct allowed_names= selection is a whitelist too; recording it as
    # "none" described a filtered file as unfiltered.
    assert spec.cut_estimator == "name_whitelist"
    assert _spec(cat, allowed_names=["GW900001_000001"]).cut_estimator == \
        "name_whitelist"


def test_the_whitelist_digest_ignores_order_and_duplicates(tmp_path):
    cat = GWCatalog(build_mixed_store(tmp_path, MIXED_EVENTS))
    a = _spec(cat, allowed_names=["GW900001_000001", "GW900002_000002"])
    b = _spec(cat, allowed_names=["GW900002_000002", "GW900001_000001",
                                  "GW900001_000001"])
    c = _spec(cat, allowed_names=["GW900001_000001"])
    assert a.allowed_names_digest == b.allowed_names_digest
    assert a.allowed_names_digest != c.allowed_names_digest
    assert _spec(cat).allowed_names_digest == ""      # no whitelist at all


def test_contradictory_class_filters_raise_rather_than_read_as_unfiltered(
        tmp_path):
    """An empty class restriction serialises exactly like "no restriction"."""
    cat = GWCatalog(build_mixed_store(tmp_path, MIXED_EVENTS))
    with pytest.raises(ValueError, match="share no class"):
        cat.select(source_class="bbh").select(source_class="bns")


def test_the_spec_digest_moves_with_the_cuts(tmp_path):
    cat = GWCatalog(build_mixed_store(tmp_path, MIXED_EVENTS))
    base = _spec(cat).digest()
    assert _spec(cat, source_class="bbh").digest() != base
    assert _spec(cat, source_class="bbh").digest() == \
        _spec(cat, source_class="BBH").digest()          # spelling-independent


# ── the v1 exporter records the view, not its own arguments ─────────────────
def test_to_darksirens_records_the_filter_of_the_view_it_exported(tmp_path):
    cat = GWCatalog(build_mixed_store(tmp_path, MIXED_EVENTS))
    out = tmp_path / "from_view.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bbh = cat.select(source_class="bbh")
        bbh.to_darksirens(str(out), nsamp=8, seed=0,
                          cosmology=(67.74, 0.3089))
    with h5py.File(out, "r") as f:
        assert int(f.attrs["nobs"]) == 2
        assert f.attrs["source_class_filter"] == "BBH"
        assert f.attrs["source_class_cut_estimator"] == "posterior_median_mass"
        assert bool(f.attrs["selection_filtered"]) is True
        assert f.attrs["selection_spec_digest"]


def test_to_darksirens_warns_about_a_class_cut_inherited_from_the_view(tmp_path):
    """The GW-12 warning asked the export call's arguments, so a class cut made
    one line earlier produced a file with no warning and (before GW-33) no
    record either."""
    cat = GWCatalog(build_mixed_store(tmp_path, MIXED_EVENTS))
    out = tmp_path / "inherited_warn.h5"
    bbh = cat.select(source_class="bbh")
    with pytest.warns(UserWarning, match="POSTERIOR MEDIAN"):
        bbh.to_darksirens(str(out), nsamp=8, seed=0,
                          cosmology=(67.74, 0.3089))


def test_to_darksirens_digests_the_events_it_wrote(tmp_path):
    from gwcat.export.contract import event_list_digest

    cat = GWCatalog(build_mixed_store(tmp_path, MIXED_EVENTS))
    out = tmp_path / "digest.h5"
    cat.to_darksirens(str(out), nsamp=8, seed=0, cosmology=(67.74, 0.3089))
    with h5py.File(out, "r") as f:
        names = [n.decode() if isinstance(n, bytes) else n
                 for n in f.attrs["event_names"]]
        assert f.attrs["event_list_digest"] == event_list_digest(names)
        # order- and duplicate-independent: it identifies WHICH events
        assert event_list_digest(names[::-1]) == event_list_digest(names)


# ── the cut no injection campaign can reproduce ─────────────────────────────
def test_pastro_min_is_refused_for_a_paired_export(tmp_path):
    from gwcat.catalog import UnpairableSelectionCut

    events = [dict(e, pastro=0.99) for e in MIXED_EVENTS]
    cat = GWCatalog(build_mixed_store(tmp_path, events, name="pa_refuse.h5"))
    out = tmp_path / "never.h5"
    with pytest.raises(UnpairableSelectionCut) as ei:
        cat.to_darksirens(str(out), pastro_min=0.5, nsamp=8, seed=0,
                          cosmology=(67.74, 0.3089))
    msg = str(ei.value)
    assert "p_astro_available=False" in msg      # why there is no equivalent
    assert "far_max" in msg                      # names the reproducible cut
    assert "allow_unpaired_pastro_min" in msg    # names the escape hatch
    assert not out.exists()

    # ...and it is refused just as loudly when the cut came from the view.
    with pytest.raises(UnpairableSelectionCut):
        cat.select(pastro_min=0.5).to_darksirens(
            str(out), nsamp=8, seed=0, cosmology=(67.74, 0.3089))


def test_pastro_min_writes_when_explicitly_unpaired_and_says_so(tmp_path):
    events = [dict(e, pastro=0.99) for e in MIXED_EVENTS]
    cat = GWCatalog(build_mixed_store(tmp_path, events, name="pa_allow.h5"))
    out = tmp_path / "unpaired.h5"
    cat.to_darksirens(str(out), pastro_min=0.5, nsamp=8, seed=0,
                      cosmology=(67.74, 0.3089),
                      allow_unpaired_pastro_min=True)
    with h5py.File(out, "r") as f:
        assert float(f.attrs["pastro_min"]) == 0.5
        assert bool(f.attrs["selection_filtered"]) is True


# ==========================================================================
# GW-32 (#8): the sample reader -- per-event, per-row posterior access
# ==========================================================================
def test_sample_reader_rows_equal_a_full_read_then_index(tmp_path):
    """``read(e, p, rows=...)`` IS ``get(p, per_event=True)[p][e][rows]``.

    The export downsamples by drawing row indices and reading only those; the
    values it writes may not change because of that, so the two access paths
    are pinned equal here -- for sorted, unsorted, repeated (bootstrap) and
    empty draws alike.
    """
    events = [dict(e, n=n) for e, n in zip(MIXED_EVENTS, (23, 40, 11, 7))]
    cat = GWCatalog(build_mixed_store(tmp_path, events, name="reader.h5"))
    params = ["mass_1", "luminosity_distance", "p_dL_pe"]
    full = cat.get(params, per_event=True)

    rng = np.random.default_rng(3)
    with cat.sample_reader() as rd:
        counts = rd.counts()
        assert counts.tolist() == [len(x) for x in full["mass_1"]]
        for e, n in enumerate(counts):
            draws = [
                np.sort(rng.choice(n, size=5, replace=False)),   # sorted
                rng.choice(n, size=7, replace=True),             # bootstrap
                np.array([n - 1, 0, n // 2, 0]),                 # unsorted, dup
                np.array([], dtype=int),                         # empty
                None,                                            # whole slice
            ]
            for rows in draws:
                got = rd.read(e, params, rows=rows)
                for p in params:
                    want = (full[p][e] if rows is None
                            else full[p][e][np.asarray(rows, dtype=int)])
                    np.testing.assert_array_equal(got[p], want)


def test_sample_reader_honours_the_required_contract(tmp_path):
    """Absent parameters behave exactly as they do through :meth:`get`."""
    from gwcat.schema import MissingParameterError

    cat = GWCatalog(build_mixed_store(tmp_path, MIXED_EVENTS, name="req.h5"))
    with cat.sample_reader() as rd:
        with pytest.raises(MissingParameterError, match="not_a_param"):
            rd.read(0, ["not_a_param"], rows=np.arange(3))
        got = rd.read(0, ["not_a_param"], rows=np.arange(3),
                      required=False, fill_value=-1.0)
        assert got["not_a_param"].tolist() == [-1.0, -1.0, -1.0]
        whole = rd.read(0, ["not_a_param"], required=False)
        assert whole["not_a_param"].size == int(rd.counts()[0])


def test_sample_reader_matches_numpy_indexing_at_the_edges(tmp_path):
    """Negative indices count from the end; out of range raises, as numpy does."""
    cat = GWCatalog(build_mixed_store(tmp_path, MIXED_EVENTS, name="edges.h5"))
    full = cat.get("mass_1", per_event=True)["mass_1"]
    with cat.sample_reader() as rd:
        n = int(rd.counts()[0])
        rows = np.array([-1, -n, 0, n - 1])
        np.testing.assert_array_equal(
            rd.read(0, "mass_1", rows=rows)["mass_1"], full[0][rows])
        with pytest.raises(IndexError):
            rd.read(0, "mass_1", rows=np.array([n]))
        with pytest.raises(IndexError):
            rd.read(0, "mass_1", rows=np.array([-n - 1]))
