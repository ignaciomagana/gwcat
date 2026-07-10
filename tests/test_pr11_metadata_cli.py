"""No-network coverage for PR11 metadata-override CLI plumbing."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import gwcat.fetch as fetch
import gwcat.ingest as ingest
from gwcat.catalog import GWCatalog
from gwcat.cli import main


class _FakeData:
    """Minimal PESummary-like object; no config means f_ref is unavailable."""


def _fake_pe_reader(path):
    rng = np.random.default_rng(11)
    samples = {
        "luminosity_distance": rng.uniform(300.0, 700.0, 12),
        "mass_1": rng.uniform(20.0, 40.0, 12),
        "mass_2": rng.uniform(8.0, 15.0, 12),
        "mass_1_source": rng.uniform(18.0, 35.0, 12),
        "mass_2_source": rng.uniform(7.0, 13.0, 12),
        "ra": rng.uniform(0.0, 2.0 * np.pi, 12),
        "dec": rng.uniform(-np.pi / 2.0, np.pi / 2.0, 12),
    }
    samples_dict = {"C00:Mixed": samples}
    return _FakeData(), samples_dict, list(samples_dict), {}


def _stage_pe_file(tmp_path, event_name="GW240101_000000"):
    path = tmp_path / f"GWTC-5_{event_name}_cosmo.h5"
    path.write_bytes(b"")
    return path


def test_ingest_metadata_overrides_write_store_diagnostics_and_summary(
    tmp_path, monkeypatch,
):
    event_name = "GW240101_000000"
    pe_path = _stage_pe_file(tmp_path, event_name)
    out = tmp_path / "store.h5"
    overrides_path = tmp_path / "metadata_overrides.yaml"
    overrides_path.write_text(
        f"{event_name}:\n"
        "  source_class: BNS\n"
        "  far: 1.0e-9\n"
        "  p_astro: 0.999\n"
        "  observing_run: O4-override\n"
    )

    monkeypatch.setattr(ingest, "_read_event_pesummary", _fake_pe_reader)
    monkeypatch.setattr(
        fetch,
        "fetch_event_table_gwosc",
        lambda **kwargs: {
            event_name: {"far": 1.0e-5, "p_astro": 0.5, "p_bbh": 0.8}
        },
    )

    rc = main([
        "ingest",
        "--glob", str(pe_path),
        "--out", str(out),
        "--metadata-overrides", str(overrides_path),
    ])

    assert rc == 0
    cat = GWCatalog(str(out))
    assert cat.meta["source_class"][0] == "BNS"
    assert cat.meta["far"][0] == pytest.approx(1.0e-9)
    assert cat.meta["p_astro"][0] == pytest.approx(0.999)
    assert cat.meta["observing_run"][0] == "O4-override"
    assert cat.meta["metadata_source"][0] == "online+user_override"

    diagnostics_path = Path(str(out) + ".metadata_diagnostics.json")
    diagnostics = json.loads(diagnostics_path.read_text())
    assert diagnostics[event_name]["far"]["source"] == "user_override"
    assert diagnostics[event_name]["p_bbh"]["source"] == "online"

    summary_path = Path(str(out) + ".validation_summary.json")
    summary = json.loads(summary_path.read_text())
    assert summary["metadata_overrides_path"] == str(overrides_path)
    assert summary["metadata_diagnostics_path"] == str(diagnostics_path)
    assert summary["n_metadata_overrides"] == 1


def test_ingest_no_event_table_with_overrides_never_calls_gwosc(
    tmp_path, monkeypatch,
):
    event_name = "GW240102_000000"
    pe_path = _stage_pe_file(tmp_path, event_name)
    out = tmp_path / "offline_store.h5"
    overrides_path = tmp_path / "offline_overrides.csv"
    overrides_path.write_text(
        f"event_name,far,source_class\n{event_name},2e-9,BBH\n")

    monkeypatch.setattr(ingest, "_read_event_pesummary", _fake_pe_reader)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("GWOSC must not be called with --no-event-table")

    monkeypatch.setattr(fetch, "fetch_event_table_gwosc", fail_if_called)

    rc = main([
        "ingest",
        "--glob", str(pe_path),
        "--out", str(out),
        "--no-event-table",
        "--metadata-overrides", str(overrides_path),
        "--no-summary",
    ])

    assert rc == 0
    cat = GWCatalog(str(out))
    assert cat.meta["far"][0] == pytest.approx(2.0e-9)
    assert cat.meta["metadata_source"][0] == "user_override"


def test_fetch_metadata_overrides_are_assembled_once_and_passed_to_build(
    tmp_path, monkeypatch,
):
    event_name = "GW240103_000000"
    pe_path = tmp_path / f"GWTC-5_{event_name}_cosmo.h5"
    out = tmp_path / "fetch_store.h5"
    overrides_path = tmp_path / "fetch_overrides.yaml"
    overrides_path.write_text(f"{event_name}:\n  far: 3.0e-9\n")

    monkeypatch.setattr(
        fetch,
        "fetch_catalog",
        lambda catalog, **kwargs: [str(pe_path)],
    )
    online_calls = []

    def fake_event_table(**kwargs):
        online_calls.append(kwargs)
        return {event_name: {"p_astro": 0.98}}

    monkeypatch.setattr(fetch, "fetch_event_table_gwosc", fake_event_table)
    captured = {}

    def fake_build_store(paths, out_path, **kwargs):
        captured["paths"] = paths
        captured["out_path"] = out_path
        captured.update(kwargs)

    monkeypatch.setattr(ingest, "build_store", fake_build_store)

    rc = main([
        "fetch",
        "--catalog", "GWTC-5",
        "--out", str(out),
        "--metadata-overrides", str(overrides_path),
    ])

    assert rc == 0
    assert len(online_calls) == 1
    assert captured["event_table"][event_name]["far"] == pytest.approx(3.0e-9)
    assert captured["event_table"][event_name]["p_astro"] == pytest.approx(0.98)
    assert captured["event_table"][event_name]["metadata_source"] == (
        "online+user_override")
    assert captured["summary_context"]["n_metadata_overrides"] == 1
    diagnostics_path = Path(str(out) + ".metadata_diagnostics.json")
    assert diagnostics_path.exists()
