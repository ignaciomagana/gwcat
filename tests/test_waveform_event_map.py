"""GW-40e: ``waveform_policy="event-map"`` -- an explicit {event: label} map.

* a selected event missing from the map, or mapped to a label the store lacks
  with no declared substitute, FAILS loudly (naming the events);
* a declared substitute is used and recorded with its reason;
* the optional NRSur q-rule refuses an NRSur7dq4 label whose
  C..:IMRPhenomXPHM-SpinTaylor posterior has more than ``frac`` of its mass at
  q < 1/6, substitutes it when asked, and verifies declared q-rule substitutes.
"""
import hashlib
import json

import h5py
import numpy as np
import pytest

from gwcat.catalog import GWCatalog
from gwcat.cli import main
from gwcat.waveform_policy import (MissingSampleSetError, NRSurQRuleError,
                                   SampleSetMapError, UnmappedEventError,
                                   WAVEFORM_POLICIES, load_sample_set_map,
                                   resolve_policy, sample_set_map_from_mapping)

from test_export_v2_pe import _build_spin_store, _read

_COSMO = (67.74, 0.3089)
NR, XP, MX = "NRSur7dq4", "IMRPhenomXPHM-SpinTaylor", "Mixed"
E1, E2, E3 = "GWm_000001", "GWm_000002", "GWm_000003"


def _store(tmp_path, q_frac=None, n=400, name="map_store.h5"):
    """Three events: E1 {NR, XP}, E2 {NR, XP}, E3 {XP, Mixed}.  ``q_frac``
    {event: fraction} forces that fraction of the event's XP samples below
    q = 1/6."""
    events = []
    for ev, wfs in ((E1, (NR, XP)), (E2, (NR, XP)), (E3, (XP, MX))):
        for wf in wfs:
            events.append({"name": ev, "waveform": wf, "n": n})
    store, _ = _build_spin_store(tmp_path, events, name=name)
    q_frac = q_frac or {}
    with h5py.File(store, "r+") as f:
        off = f["index/offsets"][:]
        m1 = f["samples/mass_1"][:]
        m2 = f["samples/mass_2"][:]
        ss = [x.decode() if isinstance(x, bytes) else x
              for x in f["meta/sample_set_name"][:]]
        nm = [x.decode() if isinstance(x, bytes) else x
              for x in f["index/event_names"][:]]
        for i, (ev, lab) in enumerate(zip(nm, ss)):
            if lab == f"C01:{XP}" and ev in q_frac:
                k = int(round(q_frac[ev] * n))
                lo = int(off[i])
                m2[lo:lo + k] = 0.1 * m1[lo:lo + k]
        f["samples/mass_2"][...] = m2
    return store


def _export(store, out, **kw):
    GWCatalog(store).export(str(out), format="gwcat2", spin_basis="chieff",
                            nsamp=32, seed=0, cosmology=_COSMO,
                            waveform_policy="event-map", **kw)
    return _read(out)[1]


def _s(xs):
    return [x.decode() if isinstance(x, bytes) else str(x)
            for x in np.atleast_1d(xs)]


def test_event_map_is_a_registered_policy():
    assert "event-map" in WAVEFORM_POLICIES


def test_mapped_labels_are_used_exactly(tmp_path):
    store = _store(tmp_path)
    smap = {E1: f"C01:{NR}", E2: f"C01:{XP}", E3: f"C01:{MX}"}
    attrs = _export(store, tmp_path / "pe.h5", sample_set_map=smap)
    got = dict(zip(_s(attrs["event_names"]),
                   _s(attrs["sample_set_name_per_event"])))
    assert got == smap
    assert all(r.startswith("event-map:mapped:")
               for r in _s(attrs["sample_set_selection_reason"]))
    assert _s(attrs["sample_set_substitute_events"]) == []
    assert np.isnan(float(attrs["nrsur_q_rule"]))
    assert len(_s(attrs["sample_set_map_sha256"])[0]) == 64
    assert bool(attrs["homogeneous_sample_sets"]) is True


def test_unmapped_event_fails_loudly(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(UnmappedEventError, match=E3):
        _export(store, tmp_path / "pe.h5",
                sample_set_map={E1: f"C01:{NR}", E2: f"C01:{XP}"})


def test_missing_label_fails_unless_a_substitute_is_declared(tmp_path):
    store = _store(tmp_path)
    smap = {E1: f"C01:{NR}", E2: f"C01:{XP}", E3: f"C01:{NR}"}
    with pytest.raises(MissingSampleSetError) as ei:
        _export(store, tmp_path / "a.h5", sample_set_map=smap)
    assert E3 in str(ei.value) and f"C01:{NR}" in str(ei.value)

    doc = {"events": smap,
           "substitutes": {E3: {"label": f"C01:{XP}",
                                "reason": "not_in_release"}}}
    attrs = _export(store, tmp_path / "b.h5", sample_set_map=doc)
    got = dict(zip(_s(attrs["event_names"]),
                   _s(attrs["sample_set_name_per_event"])))
    assert got[E3] == f"C01:{XP}"
    assert _s(attrs["sample_set_substitute_events"]) == [E3]
    assert _s(attrs["sample_set_substitute_reasons"]) == ["not_in_release"]
    assert _s(attrs["sample_set_substitute_original_labels"]) == [
        f"C01:{NR}"]
    reasons = dict(zip(_s(attrs["event_names"]),
                       _s(attrs["sample_set_selection_reason"])))
    assert reasons[E3] == f"event-map:substitute(not_in_release):C01:{XP}"


def test_substitute_without_a_reason_is_refused():
    with pytest.raises(SampleSetMapError, match="reason"):
        sample_set_map_from_mapping({"events": {E1: "C01:X"},
                                     "substitutes": {E1: {"label": "C01:Y"}}})


def test_map_entries_must_be_labels():
    with pytest.raises(SampleSetMapError, match="not a sample-set label"):
        sample_set_map_from_mapping({E1: "NRSur7dq4"})


# --------------------------------------------------------------------------
# NRSur q-rule
# --------------------------------------------------------------------------
def test_q_rule_refuses_a_violating_nrsur_label(tmp_path):
    store = _store(tmp_path, q_frac={E1: 0.03, E2: 0.005})
    smap = {E1: f"C01:{NR}", E2: f"C01:{NR}", E3: f"C01:{MX}"}
    with pytest.raises(NRSurQRuleError) as ei:
        _export(store, tmp_path / "pe.h5", sample_set_map=smap,
                nrsur_q_rule=0.01)
    msg = str(ei.value)
    assert E1 in msg and "3.00%" in msg
    assert E2 not in msg          # 0.5% passes the 1% rule


def test_q_rule_substitute_path(tmp_path):
    store = _store(tmp_path, q_frac={E1: 0.03, E2: 0.005})
    smap = {E1: f"C01:{NR}", E2: f"C01:{NR}", E3: f"C01:{MX}"}
    attrs = _export(store, tmp_path / "pe.h5", sample_set_map=smap,
                    nrsur_q_rule=0.01, nrsur_q_rule_substitute=True)
    got = dict(zip(_s(attrs["event_names"]),
                   _s(attrs["sample_set_name_per_event"])))
    assert got == {E1: f"C01:{XP}", E2: f"C01:{NR}", E3: f"C01:{MX}"}
    assert _s(attrs["sample_set_substitute_events"]) == [E1]
    assert _s(attrs["sample_set_substitute_reasons"]) == ["nrsur_q_rule"]
    assert _s(attrs["sample_set_substitute_original_labels"]) == [
        f"C01:{NR}"]
    assert float(attrs["nrsur_q_rule"]) == 0.01
    qf = dict(zip(_s(attrs["nrsur_q_frac_events"]),
                  attrs["nrsur_q_frac_below_floor"]))
    assert qf[E1] == pytest.approx(0.03) and qf[E2] == pytest.approx(0.005)


def test_declared_q_rule_substitutes_are_verified(tmp_path):
    """The OD-12 map already carries the fallback as the final label, with
    substitutes {event: 'nrsur_q_rule'}.  With the rule set, gwcat verifies it."""
    store = _store(tmp_path, q_frac={E1: 0.03, E2: 0.005})
    ok = {E1: f"C01:{XP}", E2: f"C01:{NR}", E3: f"C01:{MX}",
          "substitutes": {E1: "nrsur_q_rule"}}
    attrs = _export(store, tmp_path / "ok.h5", sample_set_map=ok,
                    nrsur_q_rule=0.01)
    assert _s(attrs["sample_set_substitute_events"]) == [E1]
    assert _s(attrs["sample_set_substitute_reasons"]) == ["nrsur_q_rule"]

    bad = {E1: f"C01:{NR}", E2: f"C01:{XP}", E3: f"C01:{MX}",
           "substitutes": {E2: "nrsur_q_rule"}}
    with pytest.raises(NRSurQRuleError, match="NOT justified"):
        _export(store, tmp_path / "bad.h5", sample_set_map=bad,
                nrsur_q_rule=0.03)


def test_q_rule_off_does_not_touch_nrsur_labels(tmp_path):
    store = _store(tmp_path, q_frac={E1: 0.5})
    smap = {E1: f"C01:{NR}", E2: f"C01:{NR}", E3: f"C01:{MX}"}
    attrs = _export(store, tmp_path / "pe.h5", sample_set_map=smap)
    assert _s(attrs["sample_set_name_per_event"])[0] == f"C01:{NR}"
    assert _s(attrs["nrsur_q_frac_events"]) == []


# --------------------------------------------------------------------------
# Loaders and CLI
# --------------------------------------------------------------------------
def test_popsummary_map_loader(tmp_path):
    p = tmp_path / "x_popsummary.h5"
    with h5py.File(p, "w") as f:
        f.attrs["events"] = np.array([E1, E2], dtype="S")
        f.attrs["event_sample_IDs"] = np.array([f"C01:{NR}", f"C01:{XP}"],
                                               dtype="S")
    m = load_sample_set_map(str(p))
    assert m.labels == {E1: f"C01:{NR}", E2: f"C01:{XP}"}
    assert m.substitutes == {}
    assert m.sha256 == hashlib.sha256(p.read_bytes()).hexdigest()
    q = tmp_path / "bad.h5"
    with h5py.File(q, "w") as f:
        f.attrs["events"] = np.array([E1], dtype="S")
    with pytest.raises(SampleSetMapError, match="event_sample_IDs"):
        load_sample_set_map(str(q))


def test_json_map_loader_both_layouts(tmp_path):
    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps({E1: "C01:A", "policy": "may26",
                                "substitutes": {E1: "nrsur_q_rule"}}))
    m = load_sample_set_map(str(flat))
    assert m.labels == {E1: "C01:A"} and m.meta == {"policy": "may26"}
    assert m.substitutes[E1]["reason"] == "nrsur_q_rule"
    assert m.sha256 == hashlib.sha256(flat.read_bytes()).hexdigest()
    nested = tmp_path / "nested.json"
    nested.write_text(json.dumps({"events": {E1: "C01:A"}}))
    assert load_sample_set_map(str(nested)).labels == {E1: "C01:A"}


def test_map_arguments_refused_for_other_policies():
    with pytest.raises(ValueError, match="event-map"):
        resolve_policy(np.array([E1]), [0], {}, policy="preferred",
                       sample_set_map={E1: "C01:A"})
    with pytest.raises(SampleSetMapError, match="requires sample_set_map"):
        resolve_policy(np.array([E1]), [0], {"sample_set_name": ["C01:A"]},
                       policy="event-map")


def test_cli_event_map(tmp_path):
    store = _store(tmp_path, q_frac={E1: 0.03})
    mp = tmp_path / "sample_set_map.json"
    mp.write_text(json.dumps({E1: f"C01:{XP}", E2: f"C01:{NR}",
                              E3: f"C01:{MX}",
                              "substitutes": {E1: "nrsur_q_rule"}}))
    out = tmp_path / "cli.h5"
    rc = main(["export", "pe", store, "--out", str(out), "--parameter-space",
               "chieff", "--nsamp", "16", "--seed", "0", "--cosmology",
               "67.74,0.3089", "--waveform-policy", "event-map",
               "--sample-set-map", str(mp), "--nrsur-q-rule", "0.01",
               "--no-summary"])
    assert rc == 0
    attrs = _read(out)[1]
    assert _s(attrs["sample_set_map_sha256"])[0] == hashlib.sha256(
        mp.read_bytes()).hexdigest()
    # an unmapped event through the CLI is a clean error, not a traceback
    mp.write_text(json.dumps({E1: f"C01:{XP}"}))
    assert main(["export", "pe", store, "--out", str(tmp_path / "c2.h5"),
                 "--parameter-space", "chieff", "--nsamp", "16",
                 "--cosmology", "67.74,0.3089", "--waveform-policy",
                 "event-map", "--sample-set-map", str(mp),
                 "--no-summary"]) == 1
