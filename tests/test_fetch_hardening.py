"""Fetch / online-metadata hardening tests (PR 8).

Covers the handoff's "PR 8: Fetch / Online Metadata Hardening" acceptance
criteria:

  * raw metadata response caching -- Zenodo file listings (``list_files``) and
    GWOSC event tables (``fetch_event_table_gwosc``) write their raw parsed
    JSON response, with a fetch timestamp, under ``<cache_dir>/metadata/``;
  * offline mode (``offline=True`` and/or ``GWCAT_OFFLINE``) reads ONLY that
    cache and never touches the network, raising a clear error naming the
    missing cache file when it was never populated;
  * ``metadata_diagnostics`` / ``assemble_event_metadata`` (gwcat.event_metadata):
    per-event, per-field provenance ("online" / "manifest" / "user_override" /
    "absent"), with user overrides taking precedence over online metadata;
  * download-provenance wiring: ``fetch_catalog(provenance=...)`` records
    sha256 + record_id per downloaded file, and ``build_store(file_provenance=...)``
    threads that into the store's per-row ``record_id`` / ``file_checksum``
    meta columns;
  * the end-to-end missing-FAR chain: assembled metadata with FAR genuinely
    absent -> event_table -> build_store -> far_available=False in the
    written store (not a crash, not a fabricated value).

Every test here is offline: all HTTP entry points (``gwcat.fetch._zenodo_get``,
``gwcat.fetch.urlopen``, ``gwcat.fetch.list_files``, ``download_file``) are
monkeypatched or bypassed via pre-staged local files.  ``build_store`` is
driven through the real ingest path with a fake PESummary reader (the same
pattern used in tests/test_waveform_policy.py), so the real PR-8 code -- not a
reimplementation -- is under test.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

import gwcat.fetch as fetch
import gwcat.fetch_cache as fetch_cache
import gwcat.ingest as ing
from gwcat.catalog import GWCatalog
from gwcat.event_metadata import (assemble_event_metadata, metadata_diagnostics,
                                  load_user_overrides, resolve_pastro)
from gwcat.ingest import build_store, IngestConfig


# ==========================================================================
# shared helpers
# ==========================================================================
class _FakeHTTPResponse:
    """Minimal stand-in for the context manager ``urlopen()`` returns."""

    def __init__(self, payload: dict):
        self._data = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._data


class _FakeData:
    """Minimal stand-in for a pesummary read() result (no config -> f_ref NaN)."""


def _fake_reader_factory(analyses_params):
    """A fake ``_read_event_pesummary`` returning the given analyses, no network."""
    def _fake(path):
        return _FakeData(), analyses_params, list(analyses_params.keys()), {}
    return _fake


def _rand_analysis(rng, n=20):
    return {
        "luminosity_distance": rng.uniform(300, 800, n),
        "mass_1": rng.uniform(25, 50, n),
        "mass_2": rng.uniform(10, 25, n),
        "ra": rng.uniform(0, 2 * np.pi, n),
        "dec": rng.uniform(-np.pi / 2, np.pi / 2, n),
        "chi_eff": rng.uniform(-0.4, 0.4, n),
    }


def _one_event_store(tmp_path, monkeypatch, event_name, seed=0):
    """Stage a fake single-event PE file + fake reader; return its path."""
    rng = np.random.default_rng(seed)
    analyses = {"C00:Mixed": _rand_analysis(rng)}
    monkeypatch.setattr(ing, "_read_event_pesummary", _fake_reader_factory(analyses))
    path = tmp_path / f"GWTC-5_{event_name}_cosmo.h5"
    path.write_bytes(b"")
    return path


# ==========================================================================
# 1. Zenodo file-listing cache (FILE discovery side)
# ==========================================================================
def test_list_files_writes_and_reads_cache(tmp_path, monkeypatch):
    fake_payload = {"files": [{"key": "a.h5", "size": 1, "checksum": "md5:abc",
                               "links": {"self": "http://example/a.h5"}}]}
    calls = []

    def fake_zenodo_get(url, timeout=30):
        calls.append(url)
        return fake_payload

    monkeypatch.setattr(fetch, "_zenodo_get", fake_zenodo_get)
    cache_dir = tmp_path / "cache"

    files = fetch.list_files(999, cache_dir=cache_dir)
    assert files == fake_payload["files"]
    assert len(calls) == 1

    cache_file = cache_dir / "metadata" / "zenodo_999.json"
    assert cache_file.exists()
    record = json.loads(cache_file.read_text())
    assert record["payload"] == fake_payload
    assert record["key"] == "zenodo_999"
    assert "fetched_at" in record and "fetched_at_iso" in record

    # Offline replay must reproduce the identical result with NO network call.
    def _boom(*a, **k):
        raise AssertionError("network should not be called in offline mode")
    monkeypatch.setattr(fetch, "_zenodo_get", _boom)
    files_offline = fetch.list_files(999, cache_dir=cache_dir, offline=True)
    assert files_offline == fake_payload["files"]
    assert len(calls) == 1  # unchanged: no additional live fetch happened


def test_list_files_offline_missing_cache_raises_clear_error(tmp_path):
    with pytest.raises(fetch_cache.OfflineCacheMissError, match="cache_dir"):
        fetch.list_files(999, offline=True)

    empty_cache = tmp_path / "empty_cache"
    with pytest.raises(fetch_cache.OfflineCacheMissError) as ei:
        fetch.list_files(999, cache_dir=empty_cache, offline=True)
    msg = str(ei.value)
    assert str(empty_cache) in msg
    assert "zenodo_999.json" in msg


def test_gwcat_offline_env_var_forces_offline_mode(tmp_path, monkeypatch):
    fake_payload = {"files": [{"key": "a.h5", "size": 1, "checksum": "",
                               "links": {"self": "http://example/a.h5"}}]}
    monkeypatch.setattr(fetch, "_zenodo_get", lambda *a, **k: fake_payload)
    cache_dir = tmp_path / "cache"
    fetch.list_files(42, cache_dir=cache_dir)  # populate the cache while "online"

    monkeypatch.setenv("GWCAT_OFFLINE", "1")

    def _boom(*a, **k):
        raise AssertionError("network should not be called when GWCAT_OFFLINE=1")
    monkeypatch.setattr(fetch, "_zenodo_get", _boom)
    files = fetch.list_files(42, cache_dir=cache_dir)  # offline not passed explicitly
    assert files == fake_payload["files"]


def test_fetch_catalog_default_behavior_unchanged_without_cache_args(tmp_path, monkeypatch):
    """No cache_dir/offline passed -> identical to pre-PR8 behavior: list_files
    is called with just the record id (a monkeypatched single-arg fake still
    works), and no cache file is ever written."""
    fake_files = [{"key": "IGWN-GWTC2p1-v2-GW150914_095045_cosmo.h5", "size": 1,
                   "checksum": "md5:abc", "links": {"self": "http://example/x"}}]

    def fake_list_files(record_id):
        assert record_id == 6513631
        return fake_files

    monkeypatch.setattr(fetch, "list_files", fake_list_files)
    paths = fetch.fetch_catalog("GWTC-2.1", data_dir=str(tmp_path),
                                resolve=False, dry_run=True)
    assert paths == []
    assert not (tmp_path / ".cache").exists()
    assert not (tmp_path / "metadata").exists()


# ==========================================================================
# 2. GWOSC event-table cache + offline (EVENT-METADATA discovery side)
# ==========================================================================
def test_fetch_event_table_gwosc_cache_and_offline(tmp_path, monkeypatch):
    page = {
        "events": {
            "GW150914-v3": {"parameters": {
                "GW150914": {"far": 1e-7, "p_astro": 0.999}}},
            "GW999999-v1": {"parameters": {
                "GW999999": {"far": None, "p_astro": None}}},
        },
        "links": {},
    }
    calls = []

    def fake_urlopen(req, timeout=30):
        calls.append(req.full_url)
        return _FakeHTTPResponse(page)

    monkeypatch.setattr(fetch, "urlopen", fake_urlopen)
    cache_dir = tmp_path / "cache"

    table = fetch.fetch_event_table_gwosc(cache_dir=cache_dir)
    assert table["GW150914"]["far"] == pytest.approx(1e-7)
    assert np.isnan(table["GW999999"]["far"])
    assert len(calls) == 1

    cache_file = cache_dir / "metadata" / "gwosc_event_table_GWTC.json"
    assert cache_file.exists()
    record = json.loads(cache_file.read_text())
    assert record["payload"]["catalog_tag"] == "GWTC"
    assert "fetched_at" in record

    def _boom(*a, **k):
        raise AssertionError("network should not be called offline")
    monkeypatch.setattr(fetch, "urlopen", _boom)
    table_offline = fetch.fetch_event_table_gwosc(cache_dir=cache_dir, offline=True)
    assert table_offline["GW150914"]["far"] == pytest.approx(1e-7)
    assert np.isnan(table_offline["GW999999"]["far"])


def test_fetch_event_table_gwosc_offline_missing_cache_raises(tmp_path):
    with pytest.raises(fetch_cache.OfflineCacheMissError):
        fetch.fetch_event_table_gwosc(cache_dir=tmp_path / "nope", offline=True)


def test_fetch_bbh_names_gwosc_cache_and_offline(tmp_path, monkeypatch):
    # Every entry carries its catalog, as the live event-versions API does:
    # the LVK-catalog restriction (GW-15) is applied to it.
    page = {
        "results": [
            {"name": "GW150914-v3", "catalog": "GWTC-2.1-confident",
             "default_parameters": [{"name": "mass_2_source", "best": 30.0}]},
            {"name": "GW170817-v2", "catalog": "GWTC-2.1-confident",
             "default_parameters": [{"name": "mass_2_source", "best": 1.3}]},
        ],
        "next": None,
    }
    calls = []

    def fake_gwosc_json(url, timeout):
        calls.append(url)
        return page

    monkeypatch.setattr(fetch, "_gwosc_json", fake_gwosc_json)
    cache_dir = tmp_path / "cache"

    # The tiny fake page is well below the real GWTC-5 BBH count, so the
    # function's own "index looks incomplete" sanity warning is expected here.
    with pytest.warns(RuntimeWarning, match="GWOSC returned only"):
        names = fetch.fetch_bbh_names_gwosc(verbose=False, cache_dir=cache_dir)
    assert names == ["GW150914"]
    assert len(calls) == 1

    def _boom(*a, **k):
        raise AssertionError("network should not be called offline")
    monkeypatch.setattr(fetch, "_gwosc_json", _boom)
    with pytest.warns(RuntimeWarning, match="GWOSC returned only"):
        names_offline = fetch.fetch_bbh_names_gwosc(
            verbose=False, cache_dir=cache_dir, offline=True)
    assert names_offline == ["GW150914"]


# ==========================================================================
# 3. metadata_diagnostics / assemble_event_metadata
# ==========================================================================
def test_metadata_diagnostics_sources_and_absence():
    online = {"GW150914": {"far": 1e-7, "p_astro": 0.99}}
    overrides = {"GW150914": {"far": 1e-9}, "GW190425": {"source_class": "BNS"}}
    names = ["GW150914", "GW190425", "GW999999"]

    diag = metadata_diagnostics(names, online_table=online, user_overrides=overrides)

    # user override wins over online for the same field
    assert diag["GW150914"]["far"] == {"value": 1e-9, "source": "user_override"}
    # online value kept for a field the override doesn't touch
    assert diag["GW150914"]["p_astro"] == {"value": 0.99, "source": "online"}
    # override-only event/field
    assert diag["GW190425"]["source_class"] == {"value": "BNS",
                                                 "source": "user_override"}
    # nothing known anywhere -> explicit absence, not a crash
    assert diag["GW999999"]["far"] == {"value": None, "source": "absent"}
    assert diag["GW999999"]["source_class"] == {"value": None, "source": "absent"}


def test_metadata_diagnostics_manifest_fallback():
    diag = metadata_diagnostics(
        ["GW150914"], online_table={}, user_overrides={},
        manifest_defaults={"p_astro": 0.5}, fields=("far", "p_astro"))
    assert diag["GW150914"]["p_astro"] == {"value": 0.5, "source": "manifest"}
    assert diag["GW150914"]["far"] == {"value": None, "source": "absent"}


def test_metadata_diagnostics_is_json_serializable():
    diag = metadata_diagnostics(["GW150914"], online_table={"GW150914": {"far": 1e-8}})
    json.dumps(diag)  # must not raise


def test_assemble_event_metadata_merges_and_labels_source():
    online = {"GW150914": {"far": 1e-7, "p_astro": 0.99}}
    overrides = {"GW150914": {"far": 1e-9}}
    table, diag = assemble_event_metadata(
        ["GW150914", "GW999999"], online_table=online, user_overrides=overrides)

    assert table["GW150914"]["far"] == pytest.approx(1e-9)
    assert table["GW150914"]["p_astro"] == pytest.approx(0.99)
    assert table["GW150914"]["metadata_source"] == "online+user_override"

    # nothing known for GW999999 -> no fabricated fields, explicit "absent"
    assert "far" not in table["GW999999"]
    assert table["GW999999"]["metadata_source"] == "absent"
    assert diag["GW999999"]["far"]["source"] == "absent"


# ==========================================================================
# 4. User override file loading (YAML / CSV)
# ==========================================================================
def test_load_user_overrides_yaml_mapping(tmp_path):
    path = tmp_path / "overrides.yaml"
    path.write_text(
        "GW150914:\n  far: 1.0e-8\n  p_astro: 0.999\n"
        "GW190425:\n  source_class: BNS\n"
    )
    out = load_user_overrides(path)
    assert out["GW150914"]["far"] == pytest.approx(1e-8)
    assert out["GW150914"]["p_astro"] == pytest.approx(0.999)
    assert out["GW190425"]["source_class"] == "BNS"


def test_load_user_overrides_yaml_list_form(tmp_path):
    path = tmp_path / "overrides_list.yaml"
    path.write_text(
        "- event_name: GW150914\n  far: 2.0e-8\n"
        "- name: GW190425\n  source_class: BNS\n"
    )
    out = load_user_overrides(path)
    assert out["GW150914"]["far"] == pytest.approx(2e-8)
    assert out["GW190425"]["source_class"] == "BNS"


def test_load_user_overrides_csv(tmp_path):
    path = tmp_path / "overrides.csv"
    path.write_text("event_name,far,source_class\nGW150914,1e-8,\nGW190425,,BNS\n")
    out = load_user_overrides(path)
    assert out["GW150914"]["far"] == pytest.approx(1e-8)
    assert "source_class" not in out["GW150914"]  # blank cell dropped, not ""
    assert out["GW190425"]["source_class"] == "BNS"
    assert "far" not in out["GW190425"]


def test_load_user_overrides_rejects_unknown_extension(tmp_path):
    path = tmp_path / "overrides.txt"
    path.write_text("whatever")
    with pytest.raises(ValueError, match="unsupported"):
        load_user_overrides(path)


# ==========================================================================
# 5. Download-provenance wiring (fetch_catalog -> build_store)
# ==========================================================================
def test_fetch_catalog_provenance_mapping(tmp_path, monkeypatch):
    content = b"pretend PE data, just needs to hash consistently"
    md5 = hashlib.md5(content).hexdigest()
    sha256 = hashlib.sha256(content).hexdigest()

    fname = "IGWN-GWTC2p1-v2-GW150914_095045_cosmo.h5"
    dest_dir = tmp_path / "GWTC" / "GWTC-2p1"
    dest_dir.mkdir(parents=True)
    (dest_dir / fname).write_bytes(content)

    fake_files = [{"key": fname, "size": len(content), "checksum": f"md5:{md5}",
                   "links": {"self": "http://example/x"}}]
    monkeypatch.setattr(fetch, "list_files", lambda record_id, **kw: fake_files)

    provenance: dict = {}
    paths = fetch.fetch_catalog("GWTC-2.1", data_dir=str(tmp_path / "GWTC"),
                                resolve=False, provenance=provenance)

    assert paths == [str(dest_dir / fname)]
    assert provenance[fname]["record_id"] == "6513631"  # pinned GWTC-2.1 record
    assert provenance[fname]["file_checksum"] == sha256


def test_build_store_file_provenance_populates_meta(tmp_path, monkeypatch):
    path = _one_event_store(tmp_path, monkeypatch, "GW240101_000000", seed=0)
    out = tmp_path / "store.h5"

    provenance = {path.name: {"record_id": "20348005",
                              "file_checksum": "deadbeef" * 8}}
    build_store([str(path)], str(out), event_table={},
                cfg=IngestConfig(validate_prior=False),
                file_provenance=provenance)

    cat = GWCatalog(str(out))
    assert list(cat.meta["record_id"]) == ["20348005"]
    assert list(cat.meta["file_checksum"]) == ["deadbeef" * 8]


def test_build_store_without_provenance_defaults_to_empty_strings(tmp_path, monkeypatch):
    """Byte-identical default: omitting file_provenance leaves the columns ""."""
    path = _one_event_store(tmp_path, monkeypatch, "GW240101_000001", seed=1)
    out = tmp_path / "store.h5"
    build_store([str(path)], str(out), event_table={},
                cfg=IngestConfig(validate_prior=False))
    cat = GWCatalog(str(out))
    assert list(cat.meta["record_id"]) == [""]
    assert list(cat.meta["file_checksum"]) == [""]


# ==========================================================================
# 6. The missing-FAR end-to-end chain: metadata -> event_table -> ingest
# ==========================================================================
def test_missing_far_chain_end_to_end(tmp_path, monkeypatch):
    """FAR genuinely absent from online metadata for this event -> diagnostics
    say so explicitly -> the assembled event_table carries no 'far' key for it
    -> build_store writes far_available=False (not NaN-as-a-crash, not a
    fabricated FAR)."""
    path = _one_event_store(tmp_path, monkeypatch, "GW990001_000001", seed=2)

    # Online metadata exists, but only for a different event.
    online_table = {"GW150914": {"far": 1e-8, "p_astro": 0.99}}
    event_table, diagnostics = assemble_event_metadata(
        ["GW990001_000001"], online_table=online_table)

    assert diagnostics["GW990001_000001"]["far"] == {"value": None, "source": "absent"}
    assert "far" not in event_table["GW990001_000001"]

    out = tmp_path / "store.h5"
    build_store([str(path)], str(out), event_table=event_table,
                cfg=IngestConfig(validate_prior=False))

    cat = GWCatalog(str(out))
    assert cat.meta["far_available"][0] == 0.0
    assert np.isnan(cat.meta["far"][0])
    assert cat.meta["metadata_source"][0] == "absent"


def test_user_override_precedence_flows_into_store(tmp_path, monkeypatch):
    path = _one_event_store(tmp_path, monkeypatch, "GW991111_000001", seed=3)

    online_table = {"GW991111_000001": {"far": 1e-6, "p_astro": 0.9}}
    overrides = {"GW991111_000001": {"far": 1e-9, "source_class": "BNS"}}
    event_table, diagnostics = assemble_event_metadata(
        ["GW991111_000001"], online_table=online_table, user_overrides=overrides)

    assert diagnostics["GW991111_000001"]["far"]["source"] == "user_override"
    assert diagnostics["GW991111_000001"]["p_astro"]["source"] == "online"

    out = tmp_path / "store.h5"
    build_store([str(path)], str(out), event_table=event_table,
                cfg=IngestConfig(validate_prior=False))

    cat = GWCatalog(str(out))
    assert cat.meta["far"][0] == pytest.approx(1e-9)       # override wins
    assert cat.meta["far_available"][0] == 1.0
    assert cat.meta["p_astro"][0] == pytest.approx(0.9)    # online kept, untouched
    assert cat.meta["source_class"][0] == "BNS"            # user override applied
    assert cat.meta["source_class_method"][0] == "user_override"
    assert cat.meta["metadata_source"][0] == "online+user_override"


def test_build_store_offline_event_table_uses_cache(tmp_path, monkeypatch):
    """event_table=None (auto-fetch) + offline=True must use the metadata
    cache instead of GWOSC, with no network call."""
    path = _one_event_store(tmp_path, monkeypatch, "GW992222_000001", seed=4)

    cache_dir = tmp_path / "cache"
    fetch_cache.write_metadata_cache(
        cache_dir, "gwosc_event_table_GWTC",
        {"catalog_tag": "GWTC", "pages": [{
            "events": {"GW992222_000001-v1": {"parameters": {
                "GW992222_000001": {"far": 5e-9, "p_astro": 0.8}}}},
            "links": {},
        }]},
    )

    def _boom(*a, **k):
        raise AssertionError("network should not be called in offline mode")
    monkeypatch.setattr(fetch, "urlopen", _boom)

    out = tmp_path / "store.h5"
    build_store([str(path)], str(out), event_table=None,
                cfg=IngestConfig(validate_prior=False),
                cache_dir=cache_dir, offline=True)

    cat = GWCatalog(str(out))
    assert cat.meta["far"][0] == pytest.approx(5e-9)
    assert cat.meta["far_available"][0] == 1.0


def test_build_store_offline_missing_cache_raises_not_silently_swallowed(tmp_path,
                                                                          monkeypatch):
    """Unlike a live-fetch failure (which warns and falls back to {}), an
    offline cache-miss during the event_table auto-fetch must raise -- it is
    not silently swallowed into an empty table."""
    path = _one_event_store(tmp_path, monkeypatch, "GW993333_000001", seed=5)
    out = tmp_path / "store.h5"
    with pytest.raises(fetch_cache.OfflineCacheMissError):
        build_store([str(path)], str(out), event_table=None,
                    cfg=IngestConfig(validate_prior=False),
                    cache_dir=tmp_path / "empty_cache", offline=True)


# ==========================================================================
# 7. p_astro / pastro: one quantity, both columns (GW-14)
# ==========================================================================
def test_p_astro_only_table_populates_both_columns(tmp_path, monkeypatch):
    """A manifest/override supplying `p_astro` -- the key this package's own
    documentation uses -- must reach the legacy `pastro` column too.

    It used to be stored under `p_astro` alone, leaving `pastro` NaN, and
    ``select(pastro_min=...)`` read only `pastro`: the whole event set dropped
    out of the selection.
    """
    path = _one_event_store(tmp_path, monkeypatch, "GW992222_000001", seed=4)
    event_table, _ = assemble_event_metadata(
        ["GW992222_000001"],
        manifest_defaults={"p_astro": 0.999})

    out = tmp_path / "store.h5"
    build_store([str(path)], str(out), event_table=event_table,
                cfg=IngestConfig(validate_prior=False))

    cat = GWCatalog(str(out))
    assert cat.meta["p_astro"][0] == pytest.approx(0.999)
    assert cat.meta["pastro"][0] == pytest.approx(0.999)
    assert cat.select(pastro_min=0.9).n_events == 1


def test_pastro_only_table_populates_both_columns(tmp_path, monkeypatch):
    """Symmetrically: the GWOSC fetch layer's `pastro` key fills `p_astro`."""
    path = _one_event_store(tmp_path, monkeypatch, "GW993333_000001", seed=5)
    event_table, _ = assemble_event_metadata(
        ["GW993333_000001"],
        online_table={"GW993333_000001": {"far": 1e-8, "pastro": 0.88}})

    out = tmp_path / "store.h5"
    build_store([str(path)], str(out), event_table=event_table,
                cfg=IngestConfig(validate_prior=False))

    cat = GWCatalog(str(out))
    assert cat.meta["p_astro"][0] == pytest.approx(0.88)
    assert cat.meta["pastro"][0] == pytest.approx(0.88)


def test_override_beats_online_across_spellings(tmp_path):
    """Precedence is over layers, not over spellings: an override written as
    `p_astro` must beat an online table written as `pastro`, and vice versa."""
    ev = ["GW994444_000001"]
    table, diag = assemble_event_metadata(
        ev, online_table={ev[0]: {"pastro": 0.5}},
        user_overrides={ev[0]: {"p_astro": 0.99}})
    assert resolve_pastro(table[ev[0]]) == pytest.approx(0.99)
    assert diag[ev[0]]["pastro"]["source"] == "user_override"

    table2, _ = assemble_event_metadata(
        ev, online_table={ev[0]: {"p_astro": 0.5}},
        user_overrides={ev[0]: {"pastro": 0.99}})
    assert resolve_pastro(table2[ev[0]]) == pytest.approx(0.99)


# ==========================================================================
# 8. Offline replays the record the ONLINE run resolved (GW-15)
# ==========================================================================
_GWTC21_PINNED = 6513631
_GWTC21_LATEST = 99999999      # what the concept DOI resolves to "today"


def _staged_listing(monkeypatch, seen):
    """Fake list_files that records which record id it was asked for."""
    files = [{"key": "IGWN-GWTC2p1-v2-GW150914_095045_cosmo.h5", "size": 1,
              "checksum": "md5:abc", "links": {"self": "http://example/x"}}]

    def fake_list_files(record_id, **kw):
        seen.append(record_id)
        return files

    monkeypatch.setattr(fetch, "list_files", fake_list_files)


def test_offline_uses_the_resolved_record(tmp_path, monkeypatch):
    """Online, resolve=True follows the concept DOI to whatever is latest.
    Offline used to fall back to the PINNED record, so as soon as a release was
    versioned past its pin the replay looked for a different Zenodo record than
    the online run had cached: a spurious cache miss, or -- with an older
    --no-resolve download on disk -- a silent replay of an older file set."""
    cache_dir = tmp_path / "cache"
    seen: list = []
    _staged_listing(monkeypatch, seen)
    monkeypatch.setattr(fetch, "resolve_latest", lambda cid: _GWTC21_LATEST)

    fetch.fetch_catalog("GWTC-2.1", data_dir=str(tmp_path), dry_run=True,
                        cache_dir=cache_dir)
    assert seen == [_GWTC21_LATEST]

    # the resolution itself is cached, not just the file listing
    resolution = json.loads(
        (cache_dir / "metadata" / "resolution_GWTC-2.1.json").read_text())
    assert resolution["payload"]["record_ids"] == [_GWTC21_LATEST]
    assert resolution["payload"]["pinned_record_ids"] == [_GWTC21_PINNED]

    def _boom(cid):
        raise AssertionError("resolve_latest must not be called offline")
    monkeypatch.setattr(fetch, "resolve_latest", _boom)

    seen.clear()
    fetch.fetch_catalog("GWTC-2.1", data_dir=str(tmp_path), dry_run=True,
                        cache_dir=cache_dir, offline=True)
    assert seen == [_GWTC21_LATEST], "offline replayed a different record"


def test_offline_without_a_cached_resolution_raises(tmp_path, monkeypatch):
    """A cache that never recorded a resolution is a cache miss, named as one
    -- never a silent fallback to the pinned record."""
    seen: list = []
    _staged_listing(monkeypatch, seen)
    with pytest.raises(fetch_cache.OfflineCacheMissError) as ei:
        fetch.fetch_catalog("GWTC-2.1", data_dir=str(tmp_path), dry_run=True,
                            cache_dir=tmp_path / "empty", offline=True)
    assert "resolution_GWTC-2.1.json" in str(ei.value)
    assert seen == []


def test_offline_no_resolve_is_the_deliberate_opt_in_to_the_pins(tmp_path,
                                                                 monkeypatch):
    seen: list = []
    _staged_listing(monkeypatch, seen)
    fetch.fetch_catalog("GWTC-2.1", data_dir=str(tmp_path), dry_run=True,
                        cache_dir=tmp_path / "empty", offline=True,
                        resolve=False)
    assert seen == [_GWTC21_PINNED]


def test_a_truncated_cache_is_a_cache_miss_not_a_json_error(tmp_path):
    cache_dir = tmp_path / "cache"
    fetch_cache.write_metadata_cache(cache_dir, "zenodo_777", {"files": []})
    path = cache_dir / "metadata" / "zenodo_777.json"
    text = path.read_text()
    path.write_text(text[:len(text) // 2])          # interrupted write

    with pytest.raises(fetch_cache.OfflineCacheMissError, match="truncated"):
        fetch_cache.read_metadata_cache(cache_dir, "zenodo_777")

    path.write_text('{"key": "zenodo_777"}')        # no payload
    with pytest.raises(fetch_cache.OfflineCacheMissError, match="payload"):
        fetch_cache.read_metadata_cache(cache_dir, "zenodo_777")


def test_metadata_cache_writes_are_atomic(tmp_path):
    """The record is renamed into place, so a reader never sees a partial
    file and no temporary file is left behind."""
    cache_dir = tmp_path / "cache"
    path = fetch_cache.write_metadata_cache(cache_dir, "zenodo_778", {"a": 1})
    assert json.loads(path.read_text())["payload"] == {"a": 1}
    assert sorted(p.name for p in path.parent.iterdir()) == ["zenodo_778.json"]

    fetch_cache.write_metadata_cache(cache_dir, "zenodo_778", {"a": 2})
    assert json.loads(path.read_text())["payload"] == {"a": 2}
    assert sorted(p.name for p in path.parent.iterdir()) == ["zenodo_778.json"]


# ==========================================================================
# 9. BBH whitelist provenance: a failed live query is never "live" (GW-15)
# ==========================================================================
def _fail_gwosc(monkeypatch):
    def _boom(*a, **k):
        raise OSError("no route to host")
    monkeypatch.setattr(fetch, "fetch_bbh_names_gwosc", _boom)


def test_network_failure_is_not_recorded_as_a_gwosc_result(tmp_path, monkeypatch):
    """fetch_bbh_list swallowed its own network error and returned the static
    BBH_ALL; get_bbh_allowed_names then reported GWOSC_used=True -- the static
    list masquerading as live in the one diagnostic meant to tell them apart."""
    from gwcat import bbh_allowed_names as ban

    _fail_gwosc(monkeypatch)
    with pytest.warns(UserWarning) as records:
        names = ban.get_bbh_allowed_names(
            data_dir=str(tmp_path), prefer_gwosc=True, expected=0)

    messages = [str(r.message) for r in records]
    assert any("GWOSC_used=False" in m for m in messages), messages
    assert not any("GWOSC_used=True" in m for m in messages), messages
    assert names == sorted(ban.BBH_ALL)


def test_fetch_bbh_list_says_which_list_it_returned(monkeypatch):
    from gwcat import bbh_allowed_names as ban

    _fail_gwosc(monkeypatch)
    with pytest.warns(UserWarning, match="GWOSC query failed"):
        names, source = ban.fetch_bbh_list_with_source(verbose=False)
    assert source == "static" and names == list(ban.BBH_ALL)

    monkeypatch.setattr(fetch, "fetch_bbh_names_gwosc",
                        lambda **kw: ["GW150914_095045", "GW170817"])
    names, source = ban.fetch_bbh_list_with_source(verbose=False)
    assert source == "gwosc"
    assert names == ["GW150914_095045"]     # exclusions still applied


def test_a_caller_that_demanded_live_data_gets_an_error(monkeypatch):
    from gwcat import bbh_allowed_names as ban

    _fail_gwosc(monkeypatch)
    with pytest.raises(OSError):
        ban.fetch_bbh_list(verbose=False, require_live=True)


def test_an_equally_long_live_list_does_not_replace_the_curated_one(tmp_path,
                                                                    monkeypatch):
    """The condition was ``len(live) >= len(curated)``: a live list that had
    swapped k curated events for k others replaced the authoritative GWTC-5
    population sample on a coincidence of length."""
    from gwcat import bbh_allowed_names as ban

    swapped = sorted(set(ban.BBH_ALL) - {ban.BBH_ALL[0]} | {"GW260101_000000"})
    assert len(swapped) == len(ban.BBH_ALL)
    monkeypatch.setattr(fetch, "fetch_bbh_names_gwosc", lambda **kw: swapped)

    with pytest.warns(UserWarning, match="would redefine the sample"):
        names = ban.get_bbh_allowed_names(data_dir=str(tmp_path),
                                          prefer_gwosc=True)
    assert names == sorted(ban.BBH_ALL)
    assert "GW260101_000000" not in names


def test_a_live_list_that_extends_the_sample_is_adopted(tmp_path, monkeypatch):
    from gwcat import bbh_allowed_names as ban

    extended = sorted(set(ban.BBH_ALL) | {"GW260101_000000"})
    monkeypatch.setattr(fetch, "fetch_bbh_names_gwosc", lambda **kw: extended)

    with pytest.warns(UserWarning, match="GWOSC_used=True"):
        names = ban.get_bbh_allowed_names(data_dir=str(tmp_path),
                                          prefer_gwosc=True)
    assert names == extended


def test_expected_bbh_total_matches_the_data_files_own_provenance():
    """One number, in one place: the count the code checks and the count the
    bundled event lists declare must not drift apart."""
    import yaml

    from gwcat import bbh_allowed_names as ban

    provenance = yaml.safe_load(
        (ban._event_list_path("provenance.yaml")).read_text())
    assert provenance["expected"]["bbh_all_total"] == ban.EXPECTED_BBH_TOTAL
    assert provenance["expected"]["o4b_non_excluded"] == ban.EXPECTED_O4B_COUNT
