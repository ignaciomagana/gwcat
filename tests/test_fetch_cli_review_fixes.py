"""Fixes from the 2026-10-20 review of the fetch / CLI / summary layer.

  1. The GWOSC BBH query asks for every event version and keeps the highest
     LVK version of each event: under ``lastver=true`` the IAS-O3a version
     superseded ~30 genuine LVK O3a events, which the LVK-catalog filter then
     dropped.  A cache written by the old query is not replayed.
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

Every test is offline: GWOSC and Zenodo are monkeypatched.
"""
from __future__ import annotations

import json
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
