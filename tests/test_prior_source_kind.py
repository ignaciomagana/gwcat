"""GW-40b: per-prior source provenance (mass, spin, distance) at ingest.

A combined ``Mixed`` sample set carries no priors group of its own, so every
analytic prior gwcat resolves for it is a SIBLING's.  Until GW-40b nothing on the
row said so.  Each row now records, for each of the mass, spin and dL priors,
``prior_source_label_*`` (whose declaration) and ``prior_source_kind_*`` (one of
own_analytic / sibling_inherited / config_file_declared / assumed_default /
release_reweighted / constituent_mixture).
"""
import warnings

import h5py
import numpy as np
import pytest

from gwcat.catalog import GWCatalog
from gwcat.ingest import (IngestConfig, PRIOR_SOURCE_KINDS, build_store,
                          resolve_mass_prior, resolve_spin_prior,
                          resolve_spin_prior_full, _spin_amax_from_config)

_XPHM = "C01:IMRPhenomXPHM"
_SEOB = "C01:SEOBNRv4PHM"
_MIXED = "C01:Mixed"
_O3_NAME = "IGWN-GWTC2p1-v2-GW950101_000101_PEDataRelease_mixed_cosmo.h5"
_O4_NAME = "IGWN-GWTC5p0-29ebe06b7_25-GW950102_000102-combined_PEDataRelease.hdf5"

_POWERLAW = ("PowerLaw(alpha=2, minimum=10, maximum=10000, "
             "name='luminosity_distance')")
_USF_LAL = ("bilby.gw.prior.UniformSourceFrame(minimum=10.0, maximum=8000.0, "
            "cosmology='Planck15_LAL', name='luminosity_distance')")


def _full_analytic(dl_repr, amax=0.99):
    return {
        "a_1": f"Uniform(minimum=0.0, maximum={amax}, name='a_1')",
        "a_2": f"Uniform(minimum=0.0, maximum={amax}, name='a_2')",
        "tilt_1": "Sine(name='tilt_1', minimum=0.0, maximum=3.141592653589793)",
        "tilt_2": "Sine(name='tilt_2', minimum=0.0, maximum=3.141592653589793)",
        "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass("
                      "minimum=10.0, maximum=100.0, name='chirp_mass')",
        "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio("
                      "minimum=0.05, maximum=1.0, name='mass_ratio')",
        "luminosity_distance": dl_repr,
    }


class _Data:
    def __init__(self, config=None):
        self.config = config or {}


def _samples(rng, n, amax=0.9):
    return {
        "mass_1": rng.uniform(25, 50, n), "mass_2": rng.uniform(10, 25, n),
        "luminosity_distance": rng.uniform(300, 800, n),
        "ra": rng.uniform(0, 2 * np.pi, n),
        "dec": rng.uniform(-np.pi / 2, np.pi / 2, n),
        "chi_eff": rng.uniform(-0.4, 0.4, n),
        "a_1": rng.uniform(0, amax, n), "a_2": rng.uniform(0, amax, n),
        "tilt_1": rng.uniform(0, np.pi, n), "tilt_2": rng.uniform(0, np.pi, n),
    }


def _ingest(tmp_path, monkeypatch, labels, priors, *, name=_O3_NAME,
            data=None, sample_sets="all"):
    import gwcat.ingest as ing
    rng = np.random.default_rng(5)
    analyses = {lab: _samples(rng, 120) for lab in labels}
    monkeypatch.setattr(ing, "_read_event_pesummary",
                        lambda path: (data or _Data(), analyses,
                                      list(analyses), priors))
    path = tmp_path / name
    path.write_bytes(b"")
    out = tmp_path / "store.h5"
    build_store([str(path)], str(out), event_table={}, sample_sets=sample_sets,
                cfg=IngestConfig(validate_prior=False))
    return GWCatalog(str(out))


def _row(cat, label):
    i = list(cat.meta["sample_set_name"]).index(label)
    return {k: cat.meta[k][i] for k in cat.meta}


def test_kind_vocabulary():
    assert PRIOR_SOURCE_KINDS == ("own_analytic", "sibling_inherited",
                                  "config_file_declared", "assumed_default",
                                  "release_reweighted", "constituent_mixture")


def test_o3_mixed_label_is_sibling_inherited(tmp_path, monkeypatch):
    """The C01:Mixed row of a GWTC-2.1 _cosmo file: mass and spin come from the
    sibling XPHM analytic group; the distance prior is release_reweighted."""
    priors = {"analytic": {_XPHM: _full_analytic(_POWERLAW)}}
    cat = _ingest(tmp_path, monkeypatch, [_MIXED, _XPHM], priors)
    mixed = _row(cat, _MIXED)
    assert mixed["prior_source_kind_spin"] == "sibling_inherited"
    assert mixed["prior_source_label_spin"] == _XPHM
    assert mixed["prior_source_kind_mass"] == "sibling_inherited"
    assert mixed["prior_source_label_mass"] == _XPHM
    assert mixed["prior_source_kind_dL"] == "release_reweighted"
    # the legacy class columns are unchanged by the provenance
    assert mixed["spin_prior_kind"] == "uniform_magnitude_isotropic"
    assert mixed["mass_prior_kind"] == "uniform_detector_frame"

    own = _row(cat, _XPHM)
    assert own["prior_source_kind_spin"] == "own_analytic"
    assert own["prior_source_label_spin"] == _XPHM
    assert own["prior_source_kind_mass"] == "own_analytic"
    assert own["prior_source_kind_dL"] == "release_reweighted"
    for r in (mixed, own):
        for p in ("mass", "spin", "dL"):
            assert r[f"prior_source_kind_{p}"] in PRIOR_SOURCE_KINDS


def test_o4_own_analytic_and_c01_mixed_in_gwtc5(tmp_path, monkeypatch):
    """GW240925_005809-like: an O4 file released with C01 labels.  Its Mixed set
    inherits the sibling's LAL UniformSourceFrame; the sibling's own row is
    own_analytic for all three priors."""
    sib = "C01:IMRPhenomXPHM-SpinTaylor"
    priors = {"analytic": {sib: _full_analytic(_USF_LAL)}}
    cat = _ingest(tmp_path, monkeypatch, [_MIXED, sib], priors, name=_O4_NAME)
    mixed = _row(cat, _MIXED)
    assert mixed["catalog"] == "GWTC-5"
    assert mixed["prior_source_kind_dL"] == "sibling_inherited"
    assert mixed["prior_source_label_dL"] == sib
    assert mixed["prior_source_kind_spin"] == "sibling_inherited"
    assert (float(mixed["dL_prior_H0"]), float(mixed["dL_prior_Om0"])) == (
        67.90, 0.3065)
    own = _row(cat, sib)
    assert [own[f"prior_source_kind_{p}"] for p in ("mass", "spin", "dL")] \
        == ["own_analytic"] * 3


def test_no_priors_anywhere_is_assumed_default(tmp_path, monkeypatch):
    cat = _ingest(tmp_path, monkeypatch, [_MIXED, _XPHM], {},
                  name=_O4_NAME.replace("GW950102", "GW950103"))
    r = _row(cat, _MIXED)
    assert r["prior_source_kind_spin"] == "assumed_default"
    assert r["prior_source_label_spin"] == ""
    assert r["prior_source_kind_mass"] == "assumed_default"
    assert r["prior_source_kind_dL"] == "assumed_default"


# --------------------------------------------------------------------------
# LALInference events: config_file/engine/a_spin{1,2}-max
# --------------------------------------------------------------------------
def _lalinf_config(a1="0.99", a2="0.99", a1_seob=None):
    return {
        _XPHM: {"engine": {"a_spin1-max": a1, "a_spin2-max": a2,
                           "q-min": "0.05"}},
        _MIXED: {},
        _SEOB: {"engine": {"a_spin1-max": a1_seob or a1, "a_spin2-max": a2}},
    }


def test_lalinference_spin_prior_is_config_file_declared(tmp_path,
                                                         monkeypatch):
    """GW170608-like: no analytic priors anywhere, both constituent configs
    declare a_spin{1,2}-max = 0.99.  Spin provenance is upgraded to
    config_file_declared; mass stays assumed_default (the config gives bounds,
    not the class); dL stays release_reweighted."""
    cat = _ingest(tmp_path, monkeypatch, [_XPHM, _MIXED, _SEOB], {},
                  data=_Data(_lalinf_config()), sample_sets="preferred")
    r = _row(cat, _MIXED)
    assert r["prior_source_kind_spin"] == "config_file_declared"
    assert r["prior_source_label_spin"] == f"{_XPHM},{_SEOB}"
    assert float(r["spin_amax_1"]) == 0.99 and float(r["spin_amax_2"]) == 0.99
    assert "a_spin1-max=0.99" in r["spin_prior_source"]
    # the legacy class stays: the config declares the ceiling, not the class
    assert r["spin_prior_kind"] == "assumed_default"
    assert r["prior_source_kind_mass"] == "assumed_default"
    assert r["prior_source_kind_dL"] == "release_reweighted"


def test_config_ceiling_is_the_declared_value_not_the_fallback():
    got = resolve_spin_prior_full(
        _MIXED, [_XPHM, _MIXED, _SEOB], {}, None, None, fallback_amax=0.99,
        data=_Data(_lalinf_config(a1="0.95", a2="0.95")))
    assert (got.amax_1, got.amax_2) == (0.95, 0.95)
    assert got.source_kind == "config_file_declared"
    # the legacy 4-tuple wrapper is unchanged (no data -> no config path)
    assert resolve_spin_prior(_MIXED, [_XPHM, _MIXED, _SEOB], {}, None,
                              None)[2:] == ("assumed_default",
                                            "default(no_analytic_prior)")


def test_disagreeing_constituent_configs_do_not_upgrade():
    with pytest.warns(UserWarning, match="DIFFERENT spin ceilings"):
        got = resolve_spin_prior_full(
            _MIXED, [_XPHM, _MIXED, _SEOB], {}, None, None,
            data=_Data(_lalinf_config(a1="0.99", a1_seob="0.8")))
    assert got.source_kind == "assumed_default"
    assert (got.amax_1, got.amax_2) == (0.99, 0.99)
    assert "config_file_inconsistent" in got.source


def test_own_config_wins_and_values_are_parsed_robustly():
    data = _Data({_XPHM: {"engine": {"a_spin1-max": [b"0.9"],
                                      "a_spin2-max": "'0.8'"}}})
    assert _spin_amax_from_config(data, _XPHM, [_XPHM]) == (0.9, 0.8, [_XPHM],
                                                           True)
    assert _spin_amax_from_config(_Data({}), _XPHM, [_XPHM]) is None
    assert _spin_amax_from_config(object(), _XPHM, [_XPHM]) is None


def test_analytic_prior_beats_config():
    """Config is consulted ONLY when no analytic spin prior exists."""
    priors = {"analytic": {_XPHM: _full_analytic(_POWERLAW, amax=0.99)}}
    got = resolve_spin_prior_full(_MIXED, [_XPHM, _MIXED, _SEOB], priors,
                                  None, None,
                                  data=_Data(_lalinf_config(a1="0.5")))
    assert got.source_kind == "sibling_inherited"
    assert got.amax_1 == 0.99


def test_mass_prior_constraints_and_own_only_search():
    node = _full_analytic(_POWERLAW)
    node["mass_1"] = "Constraint(minimum=1, maximum=1000, name='mass_1')"
    node["mass_2"] = "Constraint(minimum=1, maximum=1000, name='mass_2')"
    priors = {"analytic": {_XPHM: node}}
    mp = resolve_mass_prior(_MIXED, [_MIXED, _XPHM], priors)
    assert mp.source_kind == "sibling_inherited"
    assert (mp.m1_min, mp.m1_max, mp.m2_min, mp.m2_max) == (1, 1000, 1, 1000)
    own_only = resolve_mass_prior(_MIXED, [_MIXED, _XPHM], priors,
                                  siblings=False)
    assert own_only.kind == "assumed_default"
    assert own_only.source_kind == "assumed_default"
