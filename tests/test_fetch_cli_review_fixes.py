"""Fixes from the 2026-10-20 review of the fetch / CLI / summary layer.

  1. The GWOSC BBH query asks for every event version and keeps the highest
     LVK version of each event: under ``lastver=true`` the IAS-O3a version
     superseded ~30 genuine LVK O3a events, which the LVK-catalog filter then
     dropped.  A GWTC release's non-confident list takes part in the
     ordering, so an event LVK demoted in a later release stays out.  A cache
     written by the old query is not replayed.
  2. Cached GWOSC pages are checked before they are indexed.
  3. ``writer_commit`` names gwcat's own commit or says "unknown": an install
     inside another repository's checkout was stamped with that repository's
     HEAD.
  4. CLI help and argument checks: the --H0 help says what the code does (it
     is not applied to every campaign), --strict no longer claims every
     cross-file check raises, and a NaN/out-of-range cosmology or a zero
     --nsamp is refused at parse time.
  5. A date-only event name resolves only through the table of historical
     GWTC names, never to whichever same-day event the store holds.
  6. ``--event-list`` refuses names the store does not hold instead of
     dropping them with a Python warning; ``--allow-missing-events`` skips
     them and records them in the validation summary.
  7. ``inspect`` reports the requirements of the space ``export pe`` would
     use; a MissingParameterError is a clean CLI error; ``--parameter-space
     chieff_chip`` and ``--pastro-min`` have the opt-in flags they need.
  8. validation_summary.json is strict JSON: a NaN at any depth is null.
  9. Zenodo fetch: the destination check is lexical (a per-file symlink into
     a shared copy is accepted); sha256 and absent checksums are handled; the
     cached record resolution is written only by a run that resolved, never
     clobbered by a dry run, records a failed resolution, and is checked
     offline; superseded files are reported and every run writes a fetch
     manifest; an alias is its release (one directory, fetched once); the CLI
     fills the store's record_id / file_checksum; the GWOSC event table also
     answers to the full names of the O1/O2 events.
 10. The BBH whitelist's cache-discovery and refresh helpers, which nothing
     exercised.

Every test is offline: GWOSC and Zenodo are monkeypatched.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import warnings
from urllib.parse import parse_qs, urlparse

import pytest

import gwcat.fetch as fetch
import gwcat.fetch_cache as fetch_cache
from gwcat.bbh_allowed_names import BBH_O3A, BBH_O3B


# ==========================================================================
# 1. The BBH query keeps the highest LVK version of every event
# ==========================================================================
def _fake_gwosc_server(entries):
    """A stand-in for the event-versions endpoint over ``entries``.

    ``entries`` are ``(catalog, name, version, m2_source)``.  The fake honours
    the two query parameters whose server-side semantics matter: ``lastver``
    (one version per event, the highest, whatever its catalog) and
    ``min-mass-2-source`` (applied to each version on its own).
    """
    def _json(url, timeout):
        query = parse_qs(urlparse(url).query)
        rows = list(entries)
        if query.get("lastver", ["false"])[0] == "true":
            latest = {}
            for row in rows:
                if row[1] not in latest or row[2] > latest[row[1]][2]:
                    latest[row[1]] = row
            rows = list(latest.values())
        if "min-mass-2-source" in query:
            cut = float(query["min-mass-2-source"][0])
            rows = [r for r in rows if r[3] is not None and r[3] > cut]
        return {
            "results": [
                {"name": name, "version": version, "catalog": catalog,
                 "default_parameters": [{"name": "mass_2_source", "best": m2}]}
                for catalog, name, version, m2 in rows
            ],
            "next": None,
        }
    return _json


def _query(monkeypatch, entries, **kw):
    monkeypatch.setattr(fetch, "_gwosc_json", _fake_gwosc_server(entries))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # the count guard
        return fetch.fetch_bbh_names_gwosc(verbose=False, **kw)


def test_lvk_o3a_events_superseded_by_ias_are_kept(monkeypatch):
    """GW190408_181802 and the other O3a events GWOSC also lists under IAS-O3a:
    their latest version is the IAS one, so a ``lastver`` response held no LVK
    version of them at all, and the LVK-only filter removed the event."""
    o3a = BBH_O3A[:10]
    entries = (
        [("GWTC-2.1-confident", n, 2, 30.0) for n in o3a]
        + [("IAS-O3a", n, 3, 30.0) for n in o3a]
        + [("GWTC-3-confident", n, 1, 30.0) for n in BBH_O3B[:5]]
    )
    names = _query(monkeypatch, entries)
    assert set(o3a) <= set(names), sorted(set(o3a) - set(names))
    assert sorted(names) == sorted(o3a + BBH_O3B[:5])


def test_an_event_known_only_to_a_third_party_catalog_stays_out(monkeypatch):
    names = _query(monkeypatch, [("IAS-O3a", "GW190101_000000", 1, 30.0),
                                 ("GWTC-3-confident", BBH_O3B[0], 1, 30.0)])
    assert names == [BBH_O3B[0]]


@pytest.mark.parametrize("m2_by_version,kept", [
    ((3.5, 2.8), False),      # the current LVK version is below the cut
    ((2.8, 3.5), True),       # the current LVK version is above it
])
def test_the_highest_lvk_version_decides_the_mass_cut(monkeypatch,
                                                      m2_by_version, kept):
    """A server-side mass cut is applied to each version on its own, so it can
    hide an event's current version and return an older one; the cut belongs
    on the version that decides."""
    old, new = m2_by_version
    entries = [("GWTC-2", "GW190521_030229", 1, old),
               ("GWTC-2.1-confident", "GW190521_030229", 2, new),
               ("IAS-O3a", "GW190521_030229", 3, 40.0)]
    names = _query(monkeypatch, entries)
    assert ("GW190521_030229" in names) is kept


@pytest.mark.parametrize("demoting_list", ["GWTC-2.1-marginal",
                                           "GWTC-2.1-auxiliary"])
def test_an_event_demoted_in_a_later_release_stays_out(monkeypatch,
                                                       demoting_list):
    """GW190424_180648 and GW190909_114149 were confident in GWTC-2 and are
    not in GWTC-2.1-confident.  With the non-confident lists ignored, the
    GWTC-2 version decided and both came back: neither is in the cumulative
    GWTC table the expected count was measured on, nor in any curated list."""
    entries = [("GWTC-2", "GW190909_114149", 1, 10.0),
               (demoting_list, "GW190909_114149", 2, 10.0),
               ("GWTC-2", "GW190424_180648", 1, 30.0),
               (demoting_list, "GW190424_180648", 2, None),
               ("IAS-O3a", "GW190424_180648", 3, 30.0),
               ("GWTC-3-confident", "GW200129_065458", 1, 29.0)]
    assert _query(monkeypatch, entries) == ["GW200129_065458"]


def test_an_event_only_in_gwtc2_is_kept(monkeypatch):
    """No later LVK version: the confident GWTC-2 one is current, so the
    demotion rule does not touch it."""
    entries = [("GWTC-2", "GW190909_114149", 1, 10.0),
               ("GWTC-3-confident", "GW200129_065458", 1, 29.0)]
    assert _query(monkeypatch, entries) == ["GW190909_114149",
                                            "GW200129_065458"]


def test_a_later_confident_version_outranks_an_older_marginal_one(monkeypatch):
    entries = [("GWTC-3-marginal", "GW200129_065458", 1, None),
               ("GWTC-4.0", "GW200129_065458", 2, 29.0)]
    assert _query(monkeypatch, entries) == ["GW200129_065458"]


def test_a_marginal_and_a_confident_entry_of_one_version_are_an_error(
        monkeypatch):
    entries = [("GWTC-2", "GW190909_114149", 1, 10.0),
               ("GWTC-2.1-marginal", "GW190909_114149", 1, 10.0)]
    monkeypatch.setattr(fetch, "_gwosc_json", _fake_gwosc_server(entries))
    with pytest.raises(RuntimeError, match="cannot be ordered"):
        fetch.fetch_bbh_names_gwosc(verbose=False)


@pytest.mark.parametrize("label,expected", [
    ("GWTC-2.1-marginal", True),
    ("GWTC-3-marginal", True),
    ("GWTC-2.1-auxiliary", True),
    ("GWTC-1-marginal", True),
    ("GWTC-2.1-confident", False),
    ("GWTC-2", False),
    ("GWTC-5.0", False),
    ("O3_IMBH_marginal", False),
    ("IAS-O3a", False),
    (None, False),
])
def test_non_confident_gwtc_list_predicate(label, expected):
    assert fetch._is_lvk_gwtc_nonconfident_catalog(label) is expected


def test_the_version_is_read_from_the_name_suffix_too(monkeypatch):
    pages = {"results": [
        {"name": "GW190521_030229-v1", "catalog": "GWTC-2",
         "default_parameters": [{"name": "mass_2_source", "best": 3.5}]},
        {"name": "GW190521_030229-v2", "catalog": "GWTC-2.1-confident",
         "default_parameters": [{"name": "mass_2_source", "best": 2.8}]},
    ], "next": None}
    monkeypatch.setattr(fetch, "_gwosc_json", lambda url, timeout: pages)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert fetch.fetch_bbh_names_gwosc(verbose=False) == []


def test_two_unordered_lvk_entries_that_disagree_are_an_error(monkeypatch):
    pages = {"results": [
        {"name": "GW190521_030229", "catalog": "GWTC-2",
         "default_parameters": [{"name": "mass_2_source", "best": 3.5}]},
        {"name": "GW190521_030229", "catalog": "GWTC-2.1-confident",
         "default_parameters": [{"name": "mass_2_source", "best": 2.8}]},
    ], "next": None}
    monkeypatch.setattr(fetch, "_gwosc_json", lambda url, timeout: pages)
    with pytest.raises(RuntimeError, match="cannot be ordered"):
        fetch.fetch_bbh_names_gwosc(verbose=False)


def test_the_query_asks_for_every_version_with_no_server_side_cut(monkeypatch):
    urls = []

    def _json(url, timeout):
        urls.append(url)
        return {"results": [{"name": "GW150914", "version": 3,
                             "catalog": "GWTC-2.1-confident",
                             "default_parameters": [
                                 {"name": "mass_2_source", "best": 30.0}]}],
                "next": None}

    monkeypatch.setattr(fetch, "_gwosc_json", _json)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fetch.fetch_bbh_names_gwosc(verbose=False)
    query = parse_qs(urlparse(urls[0]).query)
    assert "lastver" not in query
    assert "min-mass-2-source" not in query


def test_a_cache_written_by_the_lastver_query_is_not_replayed(tmp_path):
    """The old cache holds one version per event; replaying it through the
    all-versions logic would reproduce the short list without a word."""
    cache_dir = tmp_path / "cache"
    old_key = fetch_cache.gwosc_cache_key("bbh_names_m2min_3.0")
    fetch_cache.write_metadata_cache(cache_dir, old_key, {"m2_min": 3.0, "pages": [
        {"results": [{"name": "GW150914-v3", "catalog": "GWTC-2.1-confident",
                      "default_parameters": [
                          {"name": "mass_2_source", "best": 30.0}]}],
         "next": None}]})
    with pytest.raises(fetch_cache.OfflineCacheMissError, match="allversions"):
        fetch.fetch_bbh_names_gwosc(verbose=False, cache_dir=cache_dir,
                                    offline=True)


def test_the_expected_count_is_the_per_catalog_baseline():
    # GWTC-5.0 104 + GWTC-4.1 86 + GWTC-2.1-confident 50 + GWTC-3-confident 32
    assert fetch._GWOSC_BBH_EXPECTED_NAMES == 104 + 86 + 50 + 32


# ==========================================================================
# 2. Cached GWOSC pages are checked before they are indexed
# ==========================================================================
@pytest.mark.parametrize("payload", [{"m2_min": 3.0}, {"pages": "oops"},
                                     ["not", "a", "dict"]])
def test_a_bbh_cache_without_pages_is_a_cache_miss(tmp_path, payload):
    cache_dir = tmp_path / "cache"
    # under both the all-versions key and the old lastver one
    for stem in ("bbh_names_allversions_m2min", "bbh_names_m2min"):
        fetch_cache.write_metadata_cache(
            cache_dir, fetch_cache.gwosc_cache_key(f"{stem}_3.0"), payload)
    with pytest.raises(fetch_cache.OfflineCacheMissError, match="pages"):
        fetch.fetch_bbh_names_gwosc(verbose=False, cache_dir=cache_dir,
                                    offline=True)


@pytest.mark.parametrize("payload", [{"catalog_tag": "GWTC"},
                                     {"pages": [None]}])
def test_an_event_table_cache_without_pages_is_a_cache_miss(tmp_path, payload):
    cache_dir = tmp_path / "cache"
    fetch_cache.write_metadata_cache(
        cache_dir, fetch_cache.gwosc_cache_key("event_table_GWTC"), payload)
    with pytest.raises(fetch_cache.OfflineCacheMissError, match="pages"):
        fetch.fetch_event_table_gwosc(cache_dir=cache_dir, offline=True)


# ==========================================================================
# 3. writer_commit is gwcat's own commit, or "unknown"
# ==========================================================================
def _git(cwd, *args):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(cwd),
         *args], check=True, capture_output=True, text=True).stdout.strip()


def _fake_module_file(monkeypatch, path):
    import gwcat.validation_summary as vs
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# stand-in\n")
    monkeypatch.setattr(vs, "__file__", str(path))
    monkeypatch.setattr(vs, "_GWCAT_COMMIT_CACHE", None)
    return vs


def test_an_install_inside_another_checkout_is_not_stamped_with_its_head(
        tmp_path, monkeypatch):
    host = tmp_path / "host"
    host.mkdir()
    _git(host, "init", "-q")
    (host / ".gitignore").write_text(".venv/\n")
    (host / "README").write_text("host\n")
    _git(host, "add", ".")
    _git(host, "commit", "-q", "-m", "host")
    vs = _fake_module_file(
        monkeypatch,
        host / ".venv" / "lib" / "site-packages" / "gwcat" /
        "validation_summary.py")
    assert vs.gwcat_commit() == "unknown"


def test_gwcats_own_checkout_is_stamped_with_its_commit(tmp_path,
                                                        monkeypatch):
    repo = tmp_path / "gwcat-src"
    repo.mkdir()
    _git(repo, "init", "-q")
    vs = _fake_module_file(monkeypatch,
                           repo / "gwcat" / "validation_summary.py")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "gwcat")
    assert vs.gwcat_commit() == _git(repo, "rev-parse", "HEAD")


# ==========================================================================
# 4. CLI help says what the code does; bad numbers are refused at parse time
# ==========================================================================
def _help(capsys, *argv):
    from gwcat.cli import main
    with pytest.raises(SystemExit) as ei:
        main([*argv, "--help"])
    assert ei.value.code == 0
    return " ".join(capsys.readouterr().out.split())


@pytest.mark.parametrize("argv", [("export", "selection"), ("selection",)])
def test_the_h0_help_no_longer_claims_it_reaches_every_campaign(capsys, argv):
    """Every documented O3+O4ab command passed --H0 on that promise and exited
    1: the override cannot reach a campaign that ships its own ddL/dz, and an
    override that reaches only some campaigns is refused."""
    text = _help(capsys, *argv)
    assert "applied to every" not in text
    assert "refused" in text
    assert "dluminosity_distance_dredshift" in text


def test_the_v1_selection_help_names_the_route_that_builds_o3_plus_o4ab(capsys):
    from gwcat.cli import main
    with pytest.raises(SystemExit):
        main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "chieff_reference" in text and "--spin-reference-amax" in text


def test_the_strict_help_no_longer_claims_cross_file_checks_always_raise(
        capsys):
    text = _help(capsys, "validate")
    assert "always raise" not in text
    assert "only warn" in text


@pytest.mark.parametrize("spec", ["nan,0.3", "inf,0.3", "-70,0.3", "0,0.3",
                                  "70,nan", "70,1.5", "70,0", "70,-0.1"])
def test_a_bad_cosmology_is_refused_at_parse_time(spec):
    from gwcat.cli import _parse_cosmology
    with pytest.raises(SystemExit, match="must be finite"):
        _parse_cosmology(spec)


def test_a_good_cosmology_still_parses():
    from gwcat.cli import _parse_cosmology
    assert _parse_cosmology("67.74,0.3089") == (67.74, 0.3089)
    assert _parse_cosmology("70,1") == (70.0, 1.0)
    assert _parse_cosmology(None) is None


@pytest.mark.parametrize("flags", [("--H0", "nan"), ("--Om0", "1.5"),
                                   ("--H0", "-1")])
def test_a_bad_selection_override_is_refused_before_any_file_is_read(
        tmp_path, flags):
    from gwcat.cli import main
    missing = str(tmp_path / "no_such_injections.hdf")
    out = str(tmp_path / "x.h5")
    for argv in (["export", "selection", missing, "--out", out],
                 ["selection", "--injections", missing, "--out", out]):
        with pytest.raises(SystemExit, match="must be finite"):
            main(argv + list(flags))


@pytest.mark.parametrize("sub", [("export", "pe"), ("export-darksirens",)])
def test_nsamp_zero_is_refused(sub, capsys):
    from gwcat.cli import build_parser
    with pytest.raises(SystemExit) as ei:
        build_parser().parse_args([*sub, "store.h5", "--out", "x.h5",
                                   "--nsamp", "0"])
    assert ei.value.code == 2
    assert "must be >= 1" in capsys.readouterr().err


# ==========================================================================
# 5. Date-only names resolve through the historical table only
# ==========================================================================
def test_a_date_only_name_never_resolves_to_the_other_event_of_the_day():
    """GW200208 is not a GWTC name; with only the non-member GW200208_222617
    in the store it used to resolve to it."""
    from gwcat.population_samples import resolve_event_name_aliases
    with pytest.raises(ValueError, match="not a historical date-only"):
        resolve_event_name_aliases(["GW200208"], ["GW200208_222617"])


def test_a_store_listing_an_event_twice_is_not_ambiguous():
    from gwcat.population_samples import resolve_event_name_aliases
    resolved, aliases = resolve_event_name_aliases(
        ["GW150914"], ["GW150914_095045", "GW150914_095045"])
    assert resolved == ["GW150914_095045"]


def test_gwtc2s_gw190521_is_the_030229_event():
    from gwcat.population_samples import resolve_event_name_aliases
    resolved, aliases = resolve_event_name_aliases(
        ["GW190521", "GW190521_074359"],
        ["GW190521_030229", "GW190521_074359"])
    assert resolved == ["GW190521_030229", "GW190521_074359"]
    assert aliases == {"GW190521": "GW190521_030229"}


def test_the_historical_table_agrees_with_the_bundled_lists():
    from gwcat.bbh_allowed_names import BBH_ALL, NON_BBH_EXCLUSIONS
    from gwcat.population_samples import HISTORICAL_DATE_ONLY_NAMES
    for short, full in HISTORICAL_DATE_ONLY_NAMES.items():
        assert full.startswith(short + "_"), (short, full)
    members = {f for f in HISTORICAL_DATE_ONLY_NAMES.values() if f in BBH_ALL}
    # GWTC-1's ten BBH plus GW190412 and GW190521; the BNS and the mass-gap
    # system are not members
    assert len(members) == 12
    assert "GW190814_211039" in NON_BBH_EXCLUSIONS


# ==========================================================================
# 6. --event-list names the store lacks are an error, not a silent drop
# ==========================================================================
def _store_and_list(tmp_path, listed):
    from test_cli import MIXED_EVENTS, _build_mixed_store
    events = [dict(MIXED_EVENTS[0], name="GW150914_095045"),
              dict(MIXED_EVENTS[0], name="GW910002_000002")]
    store = _build_mixed_store(tmp_path, events)
    path = tmp_path / "events.txt"
    path.write_text("\n".join(listed) + "\n")
    return store, str(path)


_V1 = ["--cosmology", "67.74,0.3089", "--nsamp", "8", "--no-summary"]


def test_a_typo_in_the_event_list_fails_the_export(tmp_path, capsys):
    from gwcat.cli import main
    store, listed = _store_and_list(
        tmp_path, ["GW150914_095045", "GW910002_00002"])
    out = tmp_path / "gw.h5"
    rc = main(["export-darksirens", store, "--out", str(out),
               "--event-list", listed] + _V1)
    assert rc == 1
    assert "GW910002_00002" in capsys.readouterr().err
    assert not out.exists()


def test_a_short_alias_fails_and_names_the_stores_spelling(tmp_path, capsys):
    """The README example passed date-only names, which select() dropped."""
    from gwcat.cli import main
    store, listed = _store_and_list(tmp_path, ["GW150914", "GW910002_000002"])
    for cmd in (["export-darksirens"], ["export", "pe"]):
        out = tmp_path / "gw.h5"
        rc = main(cmd + [store, "--out", str(out), "--event-list", listed,
                         "--nsamp", "8", "--no-summary"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "'GW150914' (the store has 'GW150914_095045')" in err
        assert not out.exists()


def test_allow_missing_events_skips_and_records_them(tmp_path):
    from gwcat.cli import main
    store, listed = _store_and_list(
        tmp_path, ["GW150914_095045", "GW910002_000002", "GW999999_999999"])
    out = tmp_path / "gw.h5"
    with pytest.warns(UserWarning, match="not found in store"):
        rc = main(["export-darksirens", store, "--out", str(out),
                   "--event-list", listed, "--allow-missing-events",
                   "--cosmology", "67.74,0.3089", "--nsamp", "8",
                   "--allow-missing-far"])
    assert rc == 0
    summary = json.loads(
        open(str(out) + ".validation_summary.json").read())
    assert summary["event_list_missing_names"] == ["GW999999_999999"]
    assert summary["event_list_allow_missing_events"] is True


def test_a_complete_event_list_exports_as_before(tmp_path):
    from gwcat.cli import main
    store, listed = _store_and_list(
        tmp_path, ["GW150914_095045", "GW910002_000002"])
    out = tmp_path / "gw.h5"
    assert main(["export-darksirens", store, "--out", str(out),
                 "--event-list", listed, "--allow-missing-far"] + _V1) == 0
    assert out.exists()


# ==========================================================================
# 7. inspect requirements, MissingParameterError, and the opt-in flags
# ==========================================================================
def test_inspect_reports_what_the_default_export_needs(tmp_path, capsys):
    """The default `export pe` space is component; inspect checked the
    chi_eff-only list and reported nothing missing on a store without the
    component spins."""
    from gwcat.cli import main
    from test_cli import MIXED_EVENTS, _build_mixed_store
    store = _build_mixed_store(tmp_path, MIXED_EVENTS)
    assert main(["inspect", store, "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["required_for_parameter_space"] == "component"
    assert "a_1" in info["missing_required_parameters"]


def test_a_missing_parameter_is_a_clean_cli_error(tmp_path, capsys):
    from gwcat.cli import main
    from test_cli import MIXED_EVENTS, _build_mixed_store
    store = _build_mixed_store(tmp_path, MIXED_EVENTS)
    rc = main(["export", "pe", store, "--out", str(tmp_path / "x.h5"),
               "--nsamp", "8", "--no-summary"])
    assert rc == 1
    assert "a_1" in capsys.readouterr().err


def _spin_store(tmp_path):
    from test_export_v2_pe import _build_spin_store
    store, _ = _build_spin_store(
        tmp_path, [{"name": "GW910001_000001"}, {"name": "GW910002_000002"}])
    return str(store)


def test_chieff_chip_is_reachable_with_its_opt_in(tmp_path):
    import h5py
    from gwcat.cli import main
    store = _spin_store(tmp_path)
    base = ["export", "pe", store, "--nsamp", "8", "--no-summary",
            "--parameter-space", "chieff_chip"]
    # still refused by default ...
    assert main(base + ["--out", str(tmp_path / "a.h5")]) == 1
    # ... and buildable with the flag the Python API takes
    out = tmp_path / "b.h5"
    assert main(base + ["--out", str(out), "--allow-projection-basis"]) == 0
    with h5py.File(out, "r") as f:
        assert f.attrs["spin_basis"] == "chieff_chip"


def test_pastro_min_is_reachable_with_its_opt_in(tmp_path):
    from gwcat.cli import main
    store = _spin_store(tmp_path)
    base = ["export", "pe", store, "--nsamp", "8", "--no-summary",
            "--parameter-space", "chieff", "--pastro-min", "0.5"]
    assert main(base + ["--out", str(tmp_path / "a.h5")]) == 1
    out = tmp_path / "b.h5"
    assert main(base + ["--out", str(out),
                        "--allow-unpaired-pastro-min"]) == 0
    assert out.exists()


# ==========================================================================
# 8. The validation summary is strict JSON
# ==========================================================================
def _strict_load(text):
    def _refuse(token):
        raise ValueError(f"non-JSON token {token}")
    return json.loads(text, parse_constant=_refuse)


def test_validation_summary_json_has_no_nan_tokens(tmp_path):
    import numpy as np
    from gwcat.validation_summary import write_validation_summary
    out = tmp_path / "sel.h5"
    json_path, _ = write_validation_summary(out, {
        "kind": "selection_export",
        "H0": float("nan"),
        "detected_frac": np.float64("inf"),
        "per_campaign": [1.0, float("nan")],
        "arr": np.array([0.5, np.nan]),
        "nested": {"x": (float("-inf"), 2)},
        "count": np.int64(3),
    }, write_md=False)
    data = _strict_load(open(json_path).read())
    assert data["H0"] is None and data["detected_frac"] is None
    assert data["per_campaign"] == [1.0, None]
    assert data["arr"] == [0.5, None]
    assert data["nested"] == {"x": [None, 2]}
    assert data["count"] == 3


# ==========================================================================
# 9. Zenodo fetch: destination, checksums, resolution, manifest, aliases
# ==========================================================================
_GWTC21_FILE = "IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_cosmo.h5"
_GWTC21_PINNED = 6513631


def _entry(key, content, checksum="md5"):
    if checksum == "md5":
        cs = "md5:" + hashlib.md5(content).hexdigest()
    elif checksum == "sha256":
        cs = "sha256:" + hashlib.sha256(content).hexdigest()
    else:
        cs = checksum
    return {"key": key, "size": len(content), "checksum": cs,
            "links": {"self": "https://example.invalid/" + key}}


def _listing(monkeypatch, entries, seen=None):
    def fake_list_files(record_id, **kw):
        if seen is not None:
            seen.append(record_id)
        return entries
    monkeypatch.setattr(fetch, "list_files", fake_list_files)


def _no_download(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("nothing should be downloaded")
    monkeypatch.setattr(fetch, "download_file", _boom)


@pytest.mark.parametrize("key", ["../evil.h5", "a/../../evil.h5", "/etc/x.h5",
                                 "", "a//b.h5", "./a.h5", "a\\b.h5"])
def test_a_remote_name_that_leaves_the_data_directory_is_refused(tmp_path,
                                                                 key):
    with pytest.raises(RuntimeError, match="refusing to download"):
        fetch._contained_dest(tmp_path, key)


def test_a_per_file_symlink_into_a_shared_copy_is_accepted_offline(
        tmp_path, monkeypatch):
    """The guard resolved the joined path, which follows the link, and refused
    a data directory whose files link into a group share -- even offline."""
    content = b"shared PE bytes"
    shared = tmp_path / "share" / "GWTC-2p1"
    shared.mkdir(parents=True)
    (shared / _GWTC21_FILE).write_bytes(content)
    data = tmp_path / "data"
    (data / "GWTC-2p1").mkdir(parents=True)
    os.symlink(shared / _GWTC21_FILE, data / "GWTC-2p1" / _GWTC21_FILE)
    _listing(monkeypatch, [_entry(_GWTC21_FILE, content)])
    _no_download(monkeypatch)
    paths = fetch.fetch_catalog("GWTC-2.1", data_dir=str(data),
                                cache_dir=str(tmp_path / "cache"),
                                offline=True, resolve=False,
                                show_progress=False)
    assert paths == [str((data / "GWTC-2p1").resolve() / _GWTC21_FILE)]


def test_a_sha256_checksum_verifies_a_cached_file(tmp_path, monkeypatch):
    """A sha256 entry was compared with the file's md5: never a match."""
    content = b"sha256-listed bytes"
    (tmp_path / "GWTC-2p1").mkdir()
    (tmp_path / "GWTC-2p1" / _GWTC21_FILE).write_bytes(content)
    _listing(monkeypatch, [_entry(_GWTC21_FILE, content, "sha256")])
    _no_download(monkeypatch)
    paths = fetch.fetch_catalog("GWTC-2.1", data_dir=str(tmp_path),
                                resolve=False, show_progress=False)
    assert len(paths) == 1


def test_download_file_verifies_a_sha256_checksum(tmp_path, monkeypatch):
    import sys

    class _Resp:
        headers = {"Content-Length": "4"}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=0):
            yield b"data"

    fake_tqdm = type(sys)("tqdm")
    fake_tqdm.tqdm = lambda *a, **kw: None
    fake_requests = type(sys)("requests")
    fake_requests.get = lambda *a, **kw: _Resp()
    monkeypatch.setitem(sys.modules, "tqdm", fake_tqdm)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    good = ("sha256", hashlib.sha256(b"data").hexdigest())
    dest = tmp_path / "x.h5"
    fetch.download_file("https://example.invalid/x", str(dest),
                        show_progress=False, expected_checksum=good)
    assert dest.read_bytes() == b"data"
    dest.unlink()
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        fetch.download_file("https://example.invalid/x", str(dest),
                            show_progress=False,
                            expected_checksum=("sha256", "0" * 64))
    assert not dest.exists()


def test_a_file_without_a_published_checksum_is_reused_not_refetched(
        tmp_path, monkeypatch):
    """With no checksum and none required, an existing file was downloaded
    again on every call, and refused offline."""
    content = b"no checksum published"
    (tmp_path / "GWTC-2p1").mkdir()
    (tmp_path / "GWTC-2p1" / _GWTC21_FILE).write_bytes(content)
    entry = _entry(_GWTC21_FILE, content, checksum="")
    monkeypatch.setattr(fetch, "_require_checksums", lambda info: False)
    _listing(monkeypatch, [entry])
    _no_download(monkeypatch)
    for kw in ({}, {"offline": True, "cache_dir": str(tmp_path / "c")}):
        assert len(fetch.fetch_catalog("GWTC-2.1", data_dir=str(tmp_path),
                                       resolve=False, show_progress=False,
                                       **kw)) == 1
    # ... but a copy of the wrong size is not a finished download
    (tmp_path / "GWTC-2p1" / _GWTC21_FILE).write_bytes(content[:-1])
    with pytest.raises(fetch_cache.OfflineCacheMissError):
        fetch.fetch_catalog("GWTC-2.1", data_dir=str(tmp_path), resolve=False,
                            show_progress=False, offline=True,
                            cache_dir=str(tmp_path / "c"))


def _resolution(cache_dir):
    return json.loads((cache_dir / "metadata" /
                       "resolution_GWTC-2.1.json").read_text())["payload"]


def test_a_no_resolve_run_does_not_overwrite_the_resolved_record(
        tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    _listing(monkeypatch, [_entry(_GWTC21_FILE, b"x")])
    monkeypatch.setattr(fetch, "resolve_latest", lambda cid: 99999999)
    fetch.fetch_catalog("GWTC-2.1", data_dir=str(tmp_path), dry_run=True,
                        cache_dir=cache_dir)
    fetch.fetch_catalog("GWTC-2.1", data_dir=str(tmp_path), dry_run=True,
                        cache_dir=cache_dir, resolve=False)
    assert _resolution(cache_dir)["record_ids"] == [99999999]


def test_a_dry_run_does_not_overwrite_a_real_runs_record(tmp_path,
                                                         monkeypatch):
    content = b"pe"
    (tmp_path / "GWTC-2p1").mkdir()
    (tmp_path / "GWTC-2p1" / _GWTC21_FILE).write_bytes(content)
    cache_dir = tmp_path / "cache"
    _listing(monkeypatch, [_entry(_GWTC21_FILE, content)])
    _no_download(monkeypatch)
    monkeypatch.setattr(fetch, "resolve_latest", lambda cid: 11111111)
    fetch.fetch_catalog("GWTC-2.1", data_dir=str(tmp_path),
                        cache_dir=cache_dir, show_progress=False)
    monkeypatch.setattr(fetch, "resolve_latest", lambda cid: 22222222)
    fetch.fetch_catalog("GWTC-2.1", data_dir=str(tmp_path), dry_run=True,
                        cache_dir=cache_dir)
    assert _resolution(cache_dir)["record_ids"] == [11111111]


def test_a_failed_resolution_is_recorded_and_not_replayed_as_resolved(
        tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    seen = []
    _listing(monkeypatch, [_entry(_GWTC21_FILE, b"x")], seen)

    def _down(cid):
        raise OSError("no route to host")
    monkeypatch.setattr(fetch, "resolve_latest", _down)
    with pytest.warns(UserWarning, match="Could not resolve concept"):
        fetch.fetch_catalog("GWTC-2.1", data_dir=str(tmp_path), dry_run=True,
                            cache_dir=cache_dir)
    record = _resolution(cache_dir)
    assert record["resolved_latest"] is False
    assert record["failed_concept_ids"]
    seen.clear()
    with pytest.raises(fetch_cache.OfflineCacheMissError,
                       match="could not resolve"):
        fetch.fetch_catalog("GWTC-2.1", data_dir=str(tmp_path), dry_run=True,
                            cache_dir=cache_dir, offline=True)
    assert seen == []


def test_an_unresolved_record_is_not_replayed_as_a_resolution(tmp_path):
    """A record a --no-resolve run wrote (resolved_latest=False) was read back
    by an offline resolve=True replay; the flag was never consulted."""
    cache_dir = tmp_path / "cache"
    fetch_cache.write_metadata_cache(
        cache_dir, fetch_cache.resolution_cache_key("GWTC-2.1"),
        {"catalog": "GWTC-2.1", "record_ids": [_GWTC21_PINNED],
         "resolved_latest": False, "pinned_record_ids": [_GWTC21_PINNED],
         "concept_ids": [None]})
    with pytest.raises(fetch_cache.OfflineCacheMissError,
                       match="did not resolve"):
        fetch.fetch_catalog("GWTC-2.1", data_dir=str(tmp_path), dry_run=True,
                            cache_dir=cache_dir, offline=True)


def test_superseded_files_are_reported_and_a_manifest_is_written(
        tmp_path, monkeypatch):
    content = b"current release"
    old_name = _GWTC21_FILE.replace("-v2-", "-v1-")
    rel = tmp_path / "GWTC-2p1"
    rel.mkdir()
    (rel / _GWTC21_FILE).write_bytes(content)
    (rel / old_name).write_bytes(b"older release")
    _listing(monkeypatch, [_entry(_GWTC21_FILE, content)])
    _no_download(monkeypatch)
    with pytest.warns(UserWarning, match="in none of the record"):
        fetch.fetch_catalog("GWTC-2.1", data_dir=str(tmp_path),
                            resolve=False, show_progress=False)
    manifest = json.loads((rel / fetch.FETCH_MANIFEST_NAME).read_text())
    assert manifest["record_ids"] == [_GWTC21_PINNED]
    assert manifest["record_source"] == "pinned"
    assert [f["key"] for f in manifest["files"]] == [_GWTC21_FILE]
    assert manifest["files"][0]["checksum"].startswith("md5:")
    assert manifest["superseded_files"] == [old_name]
    assert (rel / old_name).exists()          # reported, never deleted


def test_naming_an_alias_and_its_release_fetches_it_once(tmp_path,
                                                        monkeypatch):
    """GWTC-4 and GWTC-4.1 had two directories and two cache keys, so naming
    both fetched the release twice and ingested every O4a event twice."""
    calls = []
    monkeypatch.setattr(fetch, "fetch_catalog",
                        lambda cat, **kw: calls.append(cat) or [])
    fetch._cli(["--catalog", "GWTC-4", "GWTC-4.1", "GWTC-3", "--dry-run",
                "--data-dir", str(tmp_path)], _deprecated=False)
    assert calls == ["GWTC-4.1", "GWTC-3"]


def test_an_alias_shares_its_releases_cache_key(tmp_path, monkeypatch):
    _listing(monkeypatch, [_entry("GW230529_181500.hdf5", b"x")])
    monkeypatch.setattr(fetch, "resolve_latest", lambda cid: 33333333)
    cache_dir = tmp_path / "cache"
    fetch.fetch_catalog("GWTC-4", data_dir=str(tmp_path), dry_run=True,
                        cache_dir=cache_dir)
    assert sorted(p.name for p in (cache_dir / "metadata").iterdir()
                  if p.name.startswith("resolution_")) == [
        "resolution_GWTC-4.1.json"]


def test_an_alias_is_its_release(tmp_path, monkeypatch):
    key = "IGWN-GWTC4p0-1a206db3d_721-GW230529_181500-combined_PEDataRelease.hdf5"
    assert fetch.RELEASES["GWTC-4.1"].file_filter(key)
    content = b"o4a"
    legacy = tmp_path / "GWTC-4"
    legacy.mkdir()
    (legacy / key).write_bytes(content)
    _listing(monkeypatch, [_entry(key, content)])
    _no_download(monkeypatch)
    paths = fetch.fetch_catalog("GWTC-4", data_dir=str(tmp_path),
                                resolve=False, show_progress=False)
    # one release directory, and the copy the alias fetched earlier is reused
    # in place instead of being downloaded again
    assert paths == [str(legacy.resolve() / key)]
    assert (tmp_path / "GWTC-4p1" / fetch.FETCH_MANIFEST_NAME).exists()


def test_the_cli_fills_the_stores_record_id_and_checksum(tmp_path,
                                                         monkeypatch):
    content = b"pe for the store"
    (tmp_path / "GWTC-2p1").mkdir()
    (tmp_path / "GWTC-2p1" / _GWTC21_FILE).write_bytes(content)
    _listing(monkeypatch, [_entry(_GWTC21_FILE, content)])
    _no_download(monkeypatch)
    import gwcat.ingest as ing
    seen = {}
    monkeypatch.setattr(ing, "build_store",
                        lambda paths, out, **kw: seen.update(kw))
    fetch._cli(["--catalog", "GWTC-2.1", "--data-dir", str(tmp_path),
                "--no-resolve", "--no-progress", "--no-event-table",
                "--out", str(tmp_path / "store.h5")], _deprecated=False)
    assert seen["file_provenance"] == {_GWTC21_FILE: {
        "record_id": str(_GWTC21_PINNED),
        "file_checksum": hashlib.sha256(content).hexdigest()}}


def test_the_event_table_answers_to_the_full_o1o2_names(monkeypatch):
    page = {"events": {
        "GW150914-v4": {"far": 1e-7, "p_astro": 1.0},
        "GW170817-v3": {"far": 1e-7, "p_astro": 1.0},
        "GW190412_053044-v4": {"far": 1e-5, "p_astro": 1.0},
        "GW170104-v3": {"far": None, "p_astro": None},
        "GW170104_101158-v9": {"far": 3.0, "p_astro": 0.5},
    }, "links": {}}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(page).encode()

    monkeypatch.setattr(fetch, "urlopen", lambda req, timeout=30: _Resp())
    table = fetch.fetch_event_table_gwosc()
    assert table["GW150914_095045"]["far"] == pytest.approx(1e-7)
    assert table["GW170817_124104"]["pastro"] == pytest.approx(1.0)
    assert table["GW150914"]["far"] == pytest.approx(1e-7)   # still there
    # a full-name entry GWOSC publishes itself is never overwritten
    assert table["GW170104_101158"]["far"] == pytest.approx(3.0)


# ==========================================================================
# 10. The BBH whitelist's cache and refresh helpers
# ==========================================================================
def test_cached_gwtc5_names_are_discovered_from_local_file_names(tmp_path):
    from gwcat import bbh_allowed_names as ban
    # (not tmp_path itself: its name, from this test's, contains "gwtc5")
    tmp_path = tmp_path / "data"
    (tmp_path / "GWTC-5" / "sub").mkdir(parents=True)
    (tmp_path / "GWTC-5" / "IGWN-GWTC5p0-GW240101_000001-combined.hdf5"
     ).write_bytes(b"")
    (tmp_path / "GWTC-5" / "sub" / "GW240102_000002.hdf5").write_bytes(b"")
    (tmp_path / "GWTC-5" / "GW230529_181500.hdf5").write_bytes(b"")  # NSBH
    (tmp_path / "GWTC-5" / fetch.FETCH_MANIFEST_NAME).write_text("{}")
    (tmp_path / "my_o4b_copy").mkdir()          # a custom spelling
    (tmp_path / "my_o4b_copy" / "GW240103_000003.hdf5").write_bytes(b"")
    (tmp_path / "GWTC-3").mkdir()               # not GWTC-5: ignored
    (tmp_path / "GWTC-3" / "GW200101_000004_cosmo.h5").write_bytes(b"")
    assert ban._discover_cached_gwtc5_names(tmp_path) == {
        "GW240101_000001", "GW240102_000002", "GW240103_000003"}
    assert ban._discover_cached_gwtc5_names(tmp_path / "missing") == set()


def test_gwtc5_names_are_read_from_a_store(tmp_path):
    import h5py
    from gwcat import bbh_allowed_names as ban
    path = tmp_path / "store.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("index/event_names", data=[
            b"GW240101_000001", b"GW200101_000004", b"GW230529_181500"])
        f.create_dataset("meta/catalog", data=[b"GWTC-5", b"GWTC-3",
                                               b"GWTC-5"])
    assert ban._read_store_gwtc5_names(path) == {"GW240101_000001"}


def test_refresh_returns_the_live_list_and_never_the_static_one(monkeypatch,
                                                                 capsys):
    from gwcat import bbh_allowed_names as ban
    live = sorted(ban.BBH_ALL[:3] + ["GW230529_181500", "GW250101_000000"])
    monkeypatch.setattr(fetch, "fetch_bbh_names_gwosc", lambda **kw: live)
    names = ban.refresh_bbh_list()
    assert names == sorted(set(live) - ban.NON_BBH_EXCLUSIONS)
    assert "GW250101_000000" in capsys.readouterr().out

    def _down(**kw):
        raise OSError("no route to host")
    monkeypatch.setattr(fetch, "fetch_bbh_names_gwosc", _down)
    with pytest.raises(OSError):
        ban.refresh_bbh_list()
