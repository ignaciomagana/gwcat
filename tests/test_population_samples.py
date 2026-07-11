from __future__ import annotations

import re

import pytest

from gwcat.bbh_allowed_names import (
    BBH_ALL,
    BBH_O1O2,
    NON_BBH_EXCLUSIONS,
    validate_bbh_allowed_names,
)
from gwcat.population_samples import (
    resolve_event_name_aliases,
    resolve_gwtc5_bbh_population_names,
    select_gwtc5_bbh_population,
)


O1O2_CANONICAL = [
    "GW150914_095045",
    "GW151012_095443",
    "GW151226_033853",
    "GW170104_101158",
    "GW170608_020116",
    "GW170729_185629",
    "GW170809_082821",
    "GW170814_103043",
    "GW170818_022509",
    "GW170823_131358",
]


def test_bundled_population_names_are_canonical_and_stable():
    validate_bbh_allowed_names(expected_total=259, expected_o4b_count=103)
    assert BBH_O1O2 == O1O2_CANONICAL
    assert len(BBH_ALL) == 259
    assert all(re.fullmatch(r"GW\d{6}_\d{6}", name) for name in BBH_ALL)


def test_gw190814_is_intentionally_retained_by_population_membership():
    assert "GW190814_211039" in BBH_ALL
    assert "GW190814" not in NON_BBH_EXCLUSIONS
    assert "GW190814_211039" not in NON_BBH_EXCLUSIONS


def test_historical_short_aliases_resolve_to_timestamped_names():
    requested = ["GW150914", "GW190814_211039"]
    available = ["GW150914_095045", "GW190814_211039", "GW200105_162426"]
    resolved, aliases = resolve_event_name_aliases(requested, available)
    assert resolved == ["GW150914_095045", "GW190814_211039"]
    assert aliases == {"GW150914": "GW150914_095045"}


def test_short_alias_must_be_unique():
    with pytest.raises(ValueError, match="ambiguous aliases"):
        resolve_event_name_aliases(
            ["GW150914"],
            ["GW150914_095045", "GW150914_123456"],
        )


def test_full_name_is_not_fuzzily_shortened():
    with pytest.raises(ValueError, match="canonical timestamped name not found"):
        resolve_event_name_aliases(
            ["GW150914_095045"],
            ["GW150914_123456"],
        )


def test_resolve_full_bundled_population_against_store_names():
    resolved, aliases = resolve_gwtc5_bbh_population_names(BBH_ALL)
    assert resolved == BBH_ALL
    assert aliases == {}


class _DummyView:
    def __init__(self, names):
        self.n_events = len(names)


class _DummyCatalog:
    def __init__(self, names):
        self.names = names
        self.call = None

    def select(self, **kwargs):
        self.call = kwargs
        return _DummyView(kwargs["allowed_names"])


def test_population_selector_does_not_reapply_source_class():
    catalog = _DummyCatalog(BBH_ALL)
    view, aliases = select_gwtc5_bbh_population(catalog)
    assert view.n_events == 259
    assert aliases == {}
    assert catalog.call["allowed_names_authoritative"] is True
    assert catalog.call["source_class"] is None
    assert catalog.call["waveform_policy"] == "mixed-first"
