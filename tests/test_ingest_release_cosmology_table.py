"""GW-40a: the release-reweight cosmology comes from an explicit, cited table.

A GWTC-2.1/3 ``_cosmo`` release was reweighted by the LVK to UniformSourceFrame
at a cosmology the file does not reliably declare.  gwcat used to evaluate it at
``IngestConfig.o3_default_cosmo`` (astropy Planck15, 67.74/0.3075), chosen by
the label PREFIX.  It now takes (H0, Om0) from
``gwcat/data/release_reweight_cosmology.yaml`` keyed by CATALOG, records the
row's name/source and the table's sha256, never reads ``meta_data``, and
refuses a reweighted catalog the table does not list.
"""
import hashlib
import os

import h5py
import numpy as np
import pytest

from gwcat.cosmology import (LAL_PLANCK15, NAMED_COSMOLOGIES, PLANCK15,
                             dL_prior_prob, make_cosmology)
from gwcat.ingest import (IngestConfig, build_store, rank_analyses,
                          resolve_dL_prior, _catalog_era)
from gwcat.release_cosmology import (LEGACY_TABLE_FILENAME,
                                     ReleaseReweightCosmologyError,
                                     bundled_table_path,
                                     load_release_cosmology_table)

REPR_POWERLAW = ("PowerLaw(alpha=2, minimum=10, maximum=10000, "
                 "name='luminosity_distance', latex_label='$d_L$', unit='Mpc', "
                 "boundary=None)")
_ANALYSIS = "C01:IMRPhenomXPHM"


class _FakeData:
    """A pesummary read() stand-in whose meta_data DECLARES astropy Planck15.

    If gwcat consulted ``meta_data`` the row would come out at 67.74; the table
    says 67.90.  That is the whole point of the test.
    """
    config = {}
    extra_kwargs = [{"meta_data": {"cosmology": "Planck15",
                                   "H0": 67.74, "Om0": 0.3075}},
                    {"meta_data": {"cosmology": "Planck15"}}]


def _samples(rng, n):
    return {
        "mass_1": rng.uniform(25, 50, n),
        "mass_2": rng.uniform(10, 25, n),
        "luminosity_distance": rng.uniform(300, 800, n),
        "ra": rng.uniform(0, 2 * np.pi, n),
        "dec": rng.uniform(-np.pi / 2, np.pi / 2, n),
        "chi_eff": rng.uniform(-0.4, 0.4, n),
    }


def _ingest(tmp_path, monkeypatch, filename, cfg=None, n=300,
            analyses_labels=("C01:Mixed", _ANALYSIS), dl_repr=REPR_POWERLAW):
    import gwcat.ingest as ing
    rng = np.random.default_rng(3)
    analyses = {lab: _samples(rng, n) for lab in analyses_labels}
    # Only the sibling carries an analytic prior (as in the real releases,
    # whose C01:Mixed groups are empty).
    priors = {"analytic": {_ANALYSIS: {"luminosity_distance": dl_repr}}}
    monkeypatch.setattr(ing, "_read_event_pesummary",
                        lambda path: (_FakeData(), analyses, list(analyses),
                                      priors))
    path = tmp_path / filename
    path.write_bytes(b"")
    out = tmp_path / f"store_{filename}.h5"
    build_store([str(path)], str(out), event_table={},
                cfg=cfg or IngestConfig(validate_prior=False))
    return str(out)


def _meta(store, col):
    with h5py.File(store, "r") as f:
        v = f[f"meta/{col}"][0]
    return v.decode() if isinstance(v, (bytes, bytearray)) else v


def _col(store, col):
    with h5py.File(store, "r") as f:
        return np.asarray(f[f"samples/{col}"])


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# --------------------------------------------------------------------------
# The bundled table itself
# --------------------------------------------------------------------------
def test_bundled_table_is_od2_lal_planck15_with_citations():
    t = load_release_cosmology_table()
    assert set(t.rows) == {"GWTC-2.1", "GWTC-3"}
    for cat, arxiv in (("GWTC-2.1", "2108.01045"), ("GWTC-3", "2111.03606")):
        r = t.rows[cat]
        assert (r.H0, r.Om0) == (67.90, 0.3065)
        assert (r.H0, r.Om0) == (LAL_PLANCK15.H0.value, LAL_PLANCK15.Om0)
        assert r.name == "Planck15_LAL"
        assert r.source in ("documented", "inferred_from_z(dL)")
        assert arxiv in r.citation and "0.3065" in r.citation
        assert "Planck 2015" in r.citation
    assert t.sha256 == _sha(bundled_table_path())


def test_legacy_table_reproduces_the_pre_gw40_default_bitwise():
    """R2 regression support: the legacy rows are astropy Planck15 EXACTLY."""
    t = load_release_cosmology_table(bundled_table_path(LEGACY_TABLE_FILENAME))
    for r in t.rows.values():
        assert r.H0 == PLANCK15.H0.value and r.Om0 == PLANCK15.Om0
        assert r.source == "legacy_gwcat_default"
    assert IngestConfig().o3_default_cosmo == (t.rows["GWTC-3"].H0,
                                               t.rows["GWTC-3"].Om0)


@pytest.mark.parametrize("bad, match", [
    ("catalogs: {}\n", "no 'catalogs'"),
    ("catalogs:\n  GWTC-3: {name: X, H0: 67.9, Om0: 0.3065, source: documented}\n",
     "lacks"),
    ("catalogs:\n  GWTC-3: {name: X, H0: 67.9, Om0: 0.3065, source: guessed,"
     " citation: c}\n", "source="),
    ("catalogs:\n  GWTC-3: {name: X, H0: -1, Om0: 0.3065, source: documented,"
     " citation: c}\n", "unphysical"),
])
def test_malformed_tables_are_refused(tmp_path, bad, match):
    p = tmp_path / "bad.yaml"
    p.write_text(bad)
    with pytest.raises(ReleaseReweightCosmologyError, match=match):
        load_release_cosmology_table(str(p))


# --------------------------------------------------------------------------
# build_store: a _cosmo fixture gives the TABLE's H0; meta_data is ignored
# --------------------------------------------------------------------------
@pytest.mark.parametrize("filename, catalog", [
    ("IGWN-GWTC2p1-v2-GW950101_000101_PEDataRelease_mixed_cosmo.h5",
     "GWTC-2.1"),
    ("IGWN-GWTC3p0-v2-GW950101_000101_PEDataRelease_mixed_cosmo.h5",
     "GWTC-3"),
])
def test_cosmo_release_takes_the_table_cosmology_not_meta_data(
        tmp_path, monkeypatch, filename, catalog):
    store = _ingest(tmp_path, monkeypatch, filename)
    assert _meta(store, "catalog") == catalog
    assert _meta(store, "analysis_used") == "C01:Mixed"
    assert _meta(store, "dL_prior_basis") == "release_reweighted"
    # The table's value -- NOT the 67.74 the fake meta_data declares, and NOT
    # the pre-GW-40 astropy default.
    assert float(_meta(store, "dL_prior_H0")) == 67.90
    assert float(_meta(store, "dL_prior_Om0")) == 0.3065
    assert _meta(store, "dL_prior_cosmology_name") == "Planck15_LAL"
    assert _meta(store, "dL_prior_cosmology_source") == "documented"
    assert (_meta(store, "dL_prior_cosmology_table_sha256")
            == _sha(bundled_table_path()))

    # ...and the table's cosmology is what actually reached the density.
    dL = _col(store, "luminosity_distance")
    expected = dL_prior_prob(dL, kind="UniformSourceFrame",
                             cosmology=make_cosmology(67.90, 0.3065),
                             dmin=10.0, dmax=10000.0, alpha=None,
                             impl=_meta(store, "dL_prior_impl"))
    np.testing.assert_allclose(_col(store, "p_dL_pe"), expected, rtol=1e-12,
                               atol=0)
    astro = dL_prior_prob(dL, kind="UniformSourceFrame",
                          cosmology=make_cosmology(67.74, 0.3075),
                          dmin=10.0, dmax=10000.0, alpha=None,
                          impl=_meta(store, "dL_prior_impl"))
    assert not np.allclose(_col(store, "p_dL_pe"), astro, rtol=1e-8, atol=0)


def test_legacy_table_path_gives_the_old_astropy_density(tmp_path,
                                                          monkeypatch):
    legacy = bundled_table_path(LEGACY_TABLE_FILENAME)
    cfg = IngestConfig(validate_prior=False,
                       release_reweight_cosmology_table=legacy)
    store = _ingest(tmp_path, monkeypatch,
                    "IGWN-GWTC3p0-v2-GW950101_000101_PEDataRelease_mixed_cosmo.h5",
                    cfg=cfg)
    assert float(_meta(store, "dL_prior_H0")) == PLANCK15.H0.value
    assert float(_meta(store, "dL_prior_Om0")) == PLANCK15.Om0
    assert _meta(store, "dL_prior_cosmology_source") == "legacy_gwcat_default"
    assert _meta(store, "dL_prior_cosmology_table_sha256") == _sha(legacy)


def test_reweighted_catalog_missing_from_the_table_is_refused(tmp_path,
                                                               monkeypatch):
    p = tmp_path / "only_gwtc21.yaml"
    p.write_text("catalogs:\n  GWTC-2.1: {name: Planck15_LAL, H0: 67.9, "
                 "Om0: 0.3065, source: documented, citation: c}\n")
    cfg = IngestConfig(validate_prior=False,
                       release_reweight_cosmology_table=str(p))
    with pytest.raises(ReleaseReweightCosmologyError, match="GWTC-3"):
        _ingest(tmp_path, monkeypatch,
                "IGWN-GWTC3p0-v2-GW950101_000101_PEDataRelease_mixed_cosmo.h5",
                cfg=cfg)


def test_nocosmo_release_does_not_consult_the_table(tmp_path, monkeypatch):
    """A native/nocosmo row's declared class IS its prior; no table involved."""
    p = tmp_path / "empty_row_table.yaml"
    p.write_text("catalogs:\n  GWTC-2.1: {name: X, H0: 67.9, Om0: 0.3065, "
                 "source: documented, citation: c}\n")
    cfg = IngestConfig(validate_prior=False,
                       release_reweight_cosmology_table=str(p))
    store = _ingest(tmp_path, monkeypatch,
                    "IGWN-GWTC3p0-v2-GW950101_000101_PEDataRelease_nocosmo.h5",
                    cfg=cfg)
    assert _meta(store, "dL_prior_basis") == "analytic_declared"
    assert _meta(store, "dL_prior_kind") == "PowerLaw"
    assert _meta(store, "dL_prior_cosmology_table_sha256") == ""


def test_resolve_dL_prior_unit_no_analytic_anywhere_uses_the_table():
    res = resolve_dL_prior("GWTC-2.1", "C01:Mixed", ["C01:Mixed"], {},
                           np.linspace(100, 900, 50),
                           IngestConfig(), flavour="cosmo")
    assert (res.H0, res.Om0) == (67.90, 0.3065)
    assert res.basis == "release_reweighted"
    assert res.prior_source_kind == "release_reweighted"
    assert res.cosmology_source == "documented"


# --------------------------------------------------------------------------
# Defaults by CATALOG, not label prefix (GW240925_005809)
# --------------------------------------------------------------------------
def test_catalog_era_is_by_catalog_not_prefix():
    assert _catalog_era("GWTC-5", "C01") == "O4"
    assert _catalog_era("GWTC-3", "C01") == "O1-O3"
    assert _catalog_era("GWTC-4.1", "C00") == "O4"
    # an unplaceable file keeps the historical prefix rule
    assert _catalog_era("unknown", "C01") == "O1-O3"
    assert _catalog_era("unknown", "C00") == "O4"


def test_o4_file_with_c01_labels_uses_o4_defaults():
    labels = ["C01:SEOBNRv5PHM", "C01:IMRPhenomXPNR", "C01:Mixed",
              "C01:IMRPhenomXPHM-SpinTaylor"]
    cfg = IngestConfig()
    ranked = rank_analyses(labels, "C01", cfg, catalog="GWTC-5")
    assert ranked == ["C01:Mixed", "C01:IMRPhenomXPHM-SpinTaylor",
                      "C01:SEOBNRv5PHM", "C01:IMRPhenomXPNR"]
    # A prior declaring no cosmology gets the O4 fallback (LAL), not the O3
    # astropy default the C01 prefix used to select.
    usf_nocosmo = ("UniformSourceFrame(minimum=10.0, maximum=4000.0, "
                   "name='luminosity_distance')")
    pri = {"analytic": {"C01:IMRPhenomXPHM-SpinTaylor":
                        {"luminosity_distance": usf_nocosmo}}}
    res = resolve_dL_prior("GWTC-5", "C01:Mixed", labels, pri,
                           np.linspace(100, 900, 50), cfg, flavour="native")
    assert (res.H0, res.Om0) == cfg.o4_fallback_cosmo
    assert res.cosmology_source == "catalog_default"
    assert res.prior_source_kind == "sibling_inherited"
    assert res.prior_source_label == "C01:IMRPhenomXPHM-SpinTaylor"
    # Same labels in an O3 catalog -> the O3 default.
    res3 = resolve_dL_prior("GWTC-3", "C01:Mixed", labels, pri,
                            np.linspace(100, 900, 50), cfg, flavour="native")
    assert (res3.H0, res3.Om0) == cfg.o3_default_cosmo


# --------------------------------------------------------------------------
# The exact 'Planck15_lal' alias
# --------------------------------------------------------------------------
def test_planck15_lal_alias_is_exact_without_case_folding():
    from gwcat.ingest import _parse_analytic_dL
    assert NAMED_COSMOLOGIES["Planck15_lal"] is LAL_PLANCK15
    got = _parse_analytic_dL("UniformSourceFrame(minimum=1, maximum=10, "
                             "cosmology='Planck15_lal')")
    assert got.cosmology_recognized is True
    assert (got.H0, got.Om0) == (67.90, 0.3065)
    for spelling in ("planck15_lal", "PLANCK15_LAL", "Planck15_Lal"):
        bad = _parse_analytic_dL("UniformSourceFrame(minimum=1, maximum=10, "
                                 f"cosmology='{spelling}')")
        assert bad.cosmology_recognized is False
        assert bad.H0 is None


def test_pesummary_version_is_recorded(tmp_path, monkeypatch):
    import gwcat.ingest as ing
    name = "IGWN-GWTC3p0-v2-GW950101_000101_PEDataRelease_mixed_cosmo.h5"
    rng = np.random.default_rng(0)
    analyses = {"C01:Mixed": _samples(rng, 200), _ANALYSIS: _samples(rng, 200)}
    priors = {"analytic": {_ANALYSIS: {"luminosity_distance": REPR_POWERLAW}}}
    monkeypatch.setattr(ing, "_read_event_pesummary",
                        lambda path: (_FakeData(), analyses, list(analyses),
                                      priors))
    path = tmp_path / name
    with h5py.File(path, "w") as f:
        f.create_dataset("version/pesummary", data=[b"0.13.0"])
    out = tmp_path / "s.h5"
    build_store([str(path)], str(out), event_table={},
                cfg=IngestConfig(validate_prior=False))
    assert _meta(str(out), "release_pesummary_version") == "0.13.0"
