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

from test_source_class_filters import build_mixed_store


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
