"""Focused no-network coverage for PR11 CLI hardening."""
from __future__ import annotations

import json
from pathlib import Path

import gwcat.catalog as catalog_module
import gwcat.fetch as fetch
import gwcat.ingest as ingest
from gwcat.cli import main


def _install_fake_fetch_and_build(tmp_path, monkeypatch):
    pe_path = tmp_path / "GWTC-5_GW240104_000000_cosmo.h5"
    captured = {"fetch_kwargs": [], "build_calls": []}

    def fake_fetch_catalog(catalog, **kwargs):
        captured["fetch_kwargs"].append(kwargs)
        provenance = kwargs.get("provenance")
        if provenance is not None:
            provenance[pe_path.name] = {
                "record_id": "12345",
                "file_checksum": "abc123",
            }
        return [str(pe_path)]

    def fake_build_store(paths, out_path, **kwargs):
        captured["build_calls"].append((paths, out_path, kwargs))

    monkeypatch.setattr(fetch, "fetch_catalog", fake_fetch_catalog)
    monkeypatch.setattr(ingest, "build_store", fake_build_store)
    return pe_path, captured


def test_fetch_write_file_provenance_passes_and_writes_default_path(
    tmp_path, monkeypatch,
):
    pe_path, captured = _install_fake_fetch_and_build(tmp_path, monkeypatch)
    out = tmp_path / "store.h5"

    rc = main([
        "fetch",
        "--catalog", "GWTC-5",
        "--out", str(out),
        "--no-event-table",
        "--write-file-provenance",
    ])

    assert rc == 0
    expected = {
        pe_path.name: {"record_id": "12345", "file_checksum": "abc123"}
    }
    assert captured["build_calls"][0][2]["file_provenance"] == expected
    provenance_path = Path(str(out) + ".file_provenance.json")
    assert json.loads(provenance_path.read_text()) == expected


def test_fetch_file_provenance_path_implies_collection(tmp_path, monkeypatch):
    pe_path, captured = _install_fake_fetch_and_build(tmp_path, monkeypatch)
    out = tmp_path / "store.h5"
    provenance_path = tmp_path / "custom_provenance.json"

    rc = main([
        "fetch",
        "--catalog", "GWTC-5",
        "--out", str(out),
        "--no-event-table",
        "--file-provenance", str(provenance_path),
    ])

    assert rc == 0
    assert "provenance" in captured["fetch_kwargs"][0]
    assert json.loads(provenance_path.read_text())[pe_path.name]["record_id"] == (
        "12345")


def test_fetch_does_not_collect_provenance_by_default(tmp_path, monkeypatch):
    _pe_path, captured = _install_fake_fetch_and_build(tmp_path, monkeypatch)
    out = tmp_path / "store.h5"

    rc = main([
        "fetch",
        "--catalog", "GWTC-5",
        "--out", str(out),
        "--no-event-table",
    ])

    assert rc == 0
    assert "provenance" not in captured["fetch_kwargs"][0]
    assert "file_provenance" not in captured["build_calls"][0][2]
    assert not Path(str(out) + ".file_provenance.json").exists()


def test_fetch_dry_run_never_writes_provenance_or_builds(tmp_path, monkeypatch):
    _pe_path, captured = _install_fake_fetch_and_build(tmp_path, monkeypatch)
    out = tmp_path / "store.h5"
    provenance_path = tmp_path / "dry_run_provenance.json"

    rc = main([
        "fetch",
        "--catalog", "GWTC-5",
        "--out", str(out),
        "--dry-run",
        "--file-provenance", str(provenance_path),
    ])

    assert rc == 0
    assert "provenance" in captured["fetch_kwargs"][0]
    assert not provenance_path.exists()
    assert captured["build_calls"] == []


def test_validate_json_prints_only_results_json(monkeypatch, capsys):
    def fake_validate(*args, **kwargs):
        print("human trace that must be suppressed")
        return {"pe_format_version": True, "pe_p_pe_finite": True}

    monkeypatch.setattr(catalog_module, "validate_export", fake_validate)

    assert main(["validate", "pe.h5", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "pe_format_version": True,
        "pe_p_pe_finite": True,
    }


def test_validate_json_contract_exception_is_structured(monkeypatch, capsys):
    def fake_validate(*args, **kwargs):
        raise ValueError("source-class contract mismatch")

    monkeypatch.setattr(catalog_module, "validate_export", fake_validate)

    assert main(["validate", "pe.h5", "selection.h5", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"] == "source-class contract mismatch"
    assert payload["checks"] == {}
