"""GW-40c/d: where each event's spin ceiling came from, and the reference-pair
refusal of spin priors that are not the label's own declaration.

The six GWTC-2.1 LALInference events carry no analytic priors anywhere; their
spin ceiling 0.99 is declared only by the constituent runs'
``config_file/engine/a_spin{1,2}-max``.  A chieff PE file paired with a
chieff_reference selection assumes every event's spin prior IS the declared
U(0, a_ref) reference, so :func:`_xcheck_reference_pair` now refuses such
events unless an explicit allow-list names them.
"""
import json
import warnings

import h5py
import numpy as np
import pytest

from gwcat.bbh_allowed_names import BBH_ALL
from gwcat.catalog import GWCatalog
from gwcat.cli import main
from gwcat.export import validate_export_v2
from gwcat.export.contract import event_list_digest
from gwcat.export.pe_builder import POPULATION_RESOLVER
from gwcat.export.validate import load_spin_prior_allow_list

from test_export_v2_pe import _build_spin_store, _read
from test_validate_v2 import _sel_reference

_COSMO = (67.74, 0.3089)
SIX = ["GW170608_020116", "GW190707_093326", "GW190720_000836",
       "GW190725_174728", "GW190728_064510", "GW190924_021846"]
OWN = ["GW150914_095045", "GW200129_065458"]


def _pe_with_six(tmp_path, name="pe_six.h5"):
    events = ([{"name": n, "amax1": 0.99, "amax2": 0.99,
                "kind": "assumed_default",
                "prior_source_kind_spin": "config_file_declared",
                "prior_source_label_spin":
                    "C01:IMRPhenomXPHM,C01:SEOBNRv4PHM"} for n in SIX]
              + [{"name": n, "amax1": 0.99, "amax2": 0.99} for n in OWN])
    store, _ = _build_spin_store(tmp_path, events, n_per_event=80,
                                 name=f"store_{name}")
    out = tmp_path / name
    GWCatalog(store).export(str(out), format="gwcat2", spin_basis="chieff",
                            nsamp=32, seed=0, cosmology=_COSMO)
    return out


def _s(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def test_amax_source_is_the_spin_prior_source_kind(tmp_path):
    pe = _pe_with_six(tmp_path)
    _, attrs = _read(pe)
    names = [_s(x) for x in attrs["event_names"]]
    src = dict(zip(names, (_s(x) for x in
                           attrs["chi_eff_amax_source_per_event"])))
    kinds = dict(zip(names, (_s(x) for x in
                             attrs["prior_source_kind_spin_per_event"])))
    for n in SIX:
        assert src[n] == kinds[n] == "config_file_declared"
    for n in OWN:
        assert src[n] == kinds[n] == "own_analytic"
    assert sorted(_s(x) for x in attrs["spin_prior_non_own_analytic_events"]) \
        == sorted(SIX)
    # config-declared is not "assumed": the ceiling IS declared somewhere
    assert list(attrs["spin_prior_assumed_events"]) == []
    np.testing.assert_array_equal(attrs["chi_eff_amax_1_per_event"], 0.99)


def test_the_six_fail_the_reference_pair_without_the_allow_list(tmp_path):
    pe = _pe_with_six(tmp_path)
    sel = _sel_reference(tmp_path, 0.99)
    with pytest.raises(ValueError) as ei:
        validate_export_v2(str(pe), str(sel))
    msg = str(ei.value)
    assert "xcheck_reference_spin_prior_source" in msg
    assert "config_file_declared [6]" in msg
    for n in SIX:
        assert n in msg
    for n in OWN:
        assert n not in msg


def test_the_six_pass_with_the_allow_list(tmp_path):
    pe = _pe_with_six(tmp_path)
    sel = _sel_reference(tmp_path, 0.99)
    for allow in (SIX, {n: "config_file_declared" for n in SIX},
                  {n: {"kind": "config_file_declared", "reason": "OD-7"}
                   for n in SIX}):
        results = validate_export_v2(str(pe), str(sel),
                                     spin_prior_allow_list=allow)
        assert results["xcheck_reference_spin_prior_source"] is True
        assert all(results.values()), \
            [k for k, v in results.items() if not v]


def test_allow_list_with_the_wrong_kind_fails(tmp_path):
    pe = _pe_with_six(tmp_path)
    sel = _sel_reference(tmp_path, 0.99)
    allow = {n: "sibling_inherited" for n in SIX}
    with pytest.raises(ValueError, match="DIFFERENT kind"):
        validate_export_v2(str(pe), str(sel), spin_prior_allow_list=allow)


def test_partial_allow_list_names_only_the_rest(tmp_path):
    pe = _pe_with_six(tmp_path)
    sel = _sel_reference(tmp_path, 0.99)
    with pytest.raises(ValueError) as ei:
        validate_export_v2(str(pe), str(sel), spin_prior_allow_list=SIX[:4])
    msg = str(ei.value)
    assert "config_file_declared [2]" in msg
    assert SIX[4] in msg and SIX[5] in msg and SIX[0] not in msg


def test_sibling_inherited_and_assumed_default_are_refused_too(tmp_path):
    events = [{"name": "GWx_000001", "prior_source_kind_spin":
               "sibling_inherited"},
              {"name": "GWx_000002", "kind": "assumed_default",
               "prior_source_kind_spin": "assumed_default"},
              {"name": "GWx_000003"}]
    store, _ = _build_spin_store(tmp_path, events, name="mixed_kinds.h5")
    pe = tmp_path / "pe_mk.h5"
    GWCatalog(store).export(str(pe), format="gwcat2", spin_basis="chieff",
                            nsamp=32, seed=0, cosmology=_COSMO)
    _, attrs = _read(pe)
    assert [_s(x) for x in attrs["spin_prior_assumed_events"]] == ["GWx_000002"]
    sel = _sel_reference(tmp_path, 0.99)
    with pytest.raises(ValueError) as ei:
        validate_export_v2(str(pe), str(sel))
    assert "sibling_inherited [1]" in str(ei.value)
    assert "assumed_default [1]" in str(ei.value)


def test_pre_gw40_pe_file_warns_instead_of_failing(tmp_path):
    pe = _pe_with_six(tmp_path)
    with h5py.File(pe, "r+") as f:
        del f.attrs["prior_source_kind_spin_per_event"]
    sel = _sel_reference(tmp_path, 0.99)
    with pytest.warns(UserWarning, match="prior_source_kind_spin_per_event"):
        results = validate_export_v2(str(pe), str(sel))
    assert "xcheck_reference_spin_prior_source" not in results


def test_allow_list_loader_forms(tmp_path):
    txt = tmp_path / "allow.txt"
    txt.write_text("# OD-7\nGWa_1 config_file_declared\nGWb_2\n\n")
    assert load_spin_prior_allow_list(str(txt)) == {
        "GWa_1": "config_file_declared", "GWb_2": None}
    js = tmp_path / "allow.json"
    js.write_text(json.dumps({"GWa_1": {"kind": "sibling_inherited"},
                              "GWb_2": "*"}))
    assert load_spin_prior_allow_list(str(js)) == {
        "GWa_1": "sibling_inherited", "GWb_2": None}
    assert load_spin_prior_allow_list(None) is None
    assert load_spin_prior_allow_list(["a", "b"]) == {"a": None, "b": None}


def test_cli_allow_list(tmp_path):
    pe = _pe_with_six(tmp_path)
    sel = _sel_reference(tmp_path, 0.99)
    assert main(["validate", str(pe), str(sel)]) == 1
    allow = tmp_path / "allow.txt"
    allow.write_text("\n".join(f"{n} config_file_declared" for n in SIX))
    assert main(["validate", str(pe), str(sel), "--spin-prior-allow-list",
                 str(allow)]) == 0


# --------------------------------------------------------------------------
# End to end: a LALInference-like ingest carries config_file_declared through
# --------------------------------------------------------------------------
def test_ingest_to_export_carries_config_file_declared(tmp_path, monkeypatch):
    import gwcat.ingest as ing
    from gwcat.ingest import IngestConfig, build_store

    class _Data:
        config = {lab: {"engine": {"a_spin1-max": "0.99",
                                   "a_spin2-max": "0.99"}}
                  for lab in ("C01:IMRPhenomXPHM", "C01:SEOBNRv4PHM")}
        config["C01:Mixed"] = {}

    rng = np.random.default_rng(1)

    def _smp(n=200):
        return {"mass_1": rng.uniform(25, 50, n),
                "mass_2": rng.uniform(10, 25, n),
                "luminosity_distance": rng.uniform(300, 800, n),
                "ra": rng.uniform(0, 2 * np.pi, n),
                "dec": rng.uniform(-np.pi / 2, np.pi / 2, n),
                "chi_eff": rng.uniform(-0.3, 0.3, n),
                "a_1": rng.uniform(0, 0.99, n), "a_2": rng.uniform(0, 0.99, n),
                "tilt_1": rng.uniform(0, np.pi, n),
                "tilt_2": rng.uniform(0, np.pi, n)}

    analyses = {"C01:IMRPhenomXPHM": _smp(), "C01:Mixed": _smp(),
                "C01:SEOBNRv4PHM": _smp()}
    monkeypatch.setattr(ing, "_read_event_pesummary",
                        lambda path: (_Data(), analyses, list(analyses), {}))
    raw = tmp_path / ("IGWN-GWTC2p1-v2-GW170608_020116_PEDataRelease_"
                      "mixed_cosmo.h5")
    raw.write_bytes(b"")
    store = tmp_path / "store.h5"
    build_store([str(raw)], str(store), event_table={},
                cfg=IngestConfig(validate_prior=False))
    out = tmp_path / "pe.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        GWCatalog(str(store)).export(str(out), format="gwcat2",
                                     spin_basis="chieff", nsamp=32, seed=0)
    _, attrs = _read(out)
    assert [_s(x) for x in attrs["prior_source_kind_spin_per_event"]] == [
        "config_file_declared"]
    assert [_s(x) for x in attrs["chi_eff_amax_source_per_event"]] == [
        "config_file_declared"]
    assert [_s(x) for x in attrs["prior_source_kind_mass_per_event"]] == [
        "assumed_default"]
    assert [_s(x) for x in attrs["prior_source_kind_dL_per_event"]] == [
        "release_reweighted"]
    # the table's cosmology, per event
    assert float(attrs["cosmology_H0_per_event"][0]) == 67.90


# --------------------------------------------------------------------------
# population_resolver
# --------------------------------------------------------------------------
def test_population_resolver_recorded_only_for_the_bundled_population(
        tmp_path):
    events = [{"name": n, "n": 12} for n in BBH_ALL]
    store, _ = _build_spin_store(tmp_path, events, name="pop259.h5")
    cat = GWCatalog(store)
    out = tmp_path / "pop.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cat.export(str(out), format="gwcat2", spin_basis="chieff", nsamp=4,
                   seed=0, cosmology=_COSMO)
    _, attrs = _read(out)
    assert len(BBH_ALL) == 259
    assert _s(attrs["population_resolver"]) == POPULATION_RESOLVER
    assert bool(attrs["population_resolver_match"]) is True
    inputs = json.loads(_s(attrs["population_resolver_inputs"]))
    assert inputs["allowed_names_digest"] == event_list_digest(BBH_ALL)
    assert inputs["n_resolved"] == 259 and inputs["n_aliases"] == 0
    assert set(inputs["bundled_list_sha256"]) >= {
        "bbh_o1o2.txt", "bbh_o3a.txt", "bbh_o3b.txt", "bbh_o4a.txt",
        "bbh_o4b.txt"}
    assert all(len(v) == 64 for v in inputs["bundled_list_sha256"].values())

    sub = tmp_path / "sub.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cat.export(str(sub), format="gwcat2", spin_basis="chieff", nsamp=4,
                   seed=0, cosmology=_COSMO, event_list=list(BBH_ALL[:10]))
    _, attrs = _read(sub)
    assert _s(attrs["population_resolver"]) == ""
    assert bool(attrs["population_resolver_match"]) is False
