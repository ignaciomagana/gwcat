"""GW-40: the PE provenance, spin cut and event map on four released PE files.

Data-gated: skipped unless ``GWCAT_REAL_DATA=1`` and ``GWCAT_PE_ROOT`` names the
GWTC PE release root (the directory holding ``GWTC-2p1/``, ``GWTC-4p1/`` and
``GWTC-5/``).  Four events cover the four provenance paths:

* GW150914_095045 (GWTC-2.1 ``C01:Mixed``): spin prior inherited from the
  ``C01:IMRPhenomXPHM`` sibling, distance prior release-reweighted, 6 raw
  samples above a_i = 0.99;
* GW170608_020116 (GWTC-2.1, LALInference): spin ceiling declared only in the
  constituent configs (``config_file_declared``), 6 raw samples above 0.99;
* GW230702_185453 (GWTC-4.1): the OD-12 NRSur q-rule fallback to
  ``C00:IMRPhenomXPHM-SpinTaylor`` (2.90% of its mass at q < 1/6);
* GW240925_005809 (GWTC-5, released with the C01 prefix): ``C01:Mixed``
  inheriting its sibling's spin prior.

The raw inputs are read, never written; the store and exports go to pytest's
``tmp_path``.  No network: ingest runs with an empty event table.
"""
import json
import os
import warnings

import h5py
import numpy as np
import pytest

_REAL = os.environ.get("GWCAT_REAL_DATA") == "1"

pytestmark = pytest.mark.skipif(
    not _REAL, reason="set GWCAT_REAL_DATA=1 and GWCAT_PE_ROOT to run on the "
    "released PE files")

_FILES = {
    "GW150914_095045": "GWTC-2p1/IGWN-GWTC2p1-v2-GW150914_095045_"
                       "PEDataRelease_mixed_cosmo.h5",
    "GW170608_020116": "GWTC-2p1/IGWN-GWTC2p1-v2-GW170608_020116_"
                       "PEDataRelease_mixed_cosmo.h5",
    "GW230702_185453": "GWTC-4p1/IGWN-GWTC4p1-18965dda8_5-GW230702_185453-"
                       "combined_PEDataRelease.hdf5",
    "GW240925_005809": "GWTC-5/IGWN-GWTC5p0-29ebe06b7_25-GW240925_005809-"
                       "combined_PEDataRelease.hdf5",
}
_MAP = {"GW150914_095045": "C01:Mixed",
        "GW170608_020116": "C01:Mixed",
        "GW230702_185453": "C00:IMRPhenomXPHM-SpinTaylor",
        "GW240925_005809": "C01:Mixed",
        "substitutes": {"GW230702_185453": {
            "reason": "nrsur_q_rule", "original_label": "C00:NRSur7dq4"}}}


def _s(xs):
    return [x.decode() if isinstance(x, bytes) else str(x)
            for x in np.atleast_1d(xs)]


def _raw_path(ev):
    root = os.environ.get("GWCAT_PE_ROOT")
    if not root:
        pytest.fail("GWCAT_REAL_DATA=1 but $GWCAT_PE_ROOT is unset.")
    p = os.path.join(root, _FILES[ev])
    if not os.path.exists(p):
        pytest.fail(f"GWCAT_REAL_DATA=1 but {p} does not exist.")
    return p


def _raw_count_above(ev, label, amax=0.99):
    """Independent count: samples of ``label`` with max(a_1, a_2) > amax."""
    with h5py.File(_raw_path(ev), "r") as f:
        ps = f[label]["posterior_samples"]
        a1, a2 = ps["a_1"][()], ps["a_2"][()]
    return int(np.sum((a1 > amax) | (a2 > amax)))


@pytest.fixture(scope="module")
def real_export(tmp_path_factory):
    from gwcat.catalog import GWCatalog
    from gwcat.ingest import IngestConfig, build_store

    d = tmp_path_factory.mktemp("real_pe")
    store = str(d / "store4.h5")
    paths = [_raw_path(ev) for ev in _FILES]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        build_store(paths, store, cfg=IngestConfig(), event_table={},
                    sample_sets="all")
    mp = d / "map.json"
    mp.write_text(json.dumps(_MAP))
    out = d / "pe.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        GWCatalog(store).export(
            str(out), format="gwcat2", spin_basis="chieff", nsamp=64, seed=0,
            waveform_policy="event-map", sample_set_map=str(mp),
            nrsur_q_rule=0.01, drop_spin_above_ceiling=True,
            event_list=list(_FILES))
    with h5py.File(out, "r") as f:
        attrs = dict(f.attrs)
    return store, out, attrs


def test_labels_and_q_rule(real_export):
    _, _, a = real_export
    names = _s(a["event_names"])
    assert sorted(names) == sorted(_FILES)
    got = dict(zip(names, _s(a["sample_set_name_per_event"])))
    want = {k: v for k, v in _MAP.items() if k != "substitutes"}
    assert got == want
    assert _s(a["sample_set_substitute_events"]) == ["GW230702_185453"]
    assert _s(a["sample_set_substitute_reasons"]) == ["nrsur_q_rule"]
    qf = dict(zip(_s(a["nrsur_q_frac_events"]),
                  np.asarray(a["nrsur_q_frac_below_floor"], float)))
    # Research value: 2.90% (BUILD_PLAN section 2.4), above the 1% rule.
    assert qf["GW230702_185453"] == pytest.approx(0.02903, abs=5e-5)


def test_spin_cut_counts_equal_independent_raw_counts(real_export):
    _, _, a = real_export
    names = _s(a["event_names"])
    labels = _s(a["sample_set_name_per_event"])
    drop = dict(zip(names, a["n_dropped_spin_above_ceiling_per_event"]))
    raw = dict(zip(names, a["n_above_spin_ceiling_raw_per_event"]))
    for ev, lab in zip(names, labels):
        want = _raw_count_above(ev, lab)
        assert int(raw[ev]) == want, ev
        assert int(drop[ev]) == want, ev          # no z_max: the two agree
    assert int(drop["GW150914_095045"]) == 6
    assert int(drop["GW170608_020116"]) == 6
    np.testing.assert_array_equal(a["chi_eff_amax_1_per_event"], 0.99)
    np.testing.assert_array_equal(a["chi_eff_amax_2_per_event"], 0.99)


def test_prior_source_kinds_and_cosmology(real_export):
    _, _, a = real_export
    names = _s(a["event_names"])
    spin = dict(zip(names, _s(a["prior_source_kind_spin_per_event"])))
    dl = dict(zip(names, _s(a["prior_source_kind_dL_per_event"])))
    assert spin == {"GW150914_095045": "sibling_inherited",
                    "GW170608_020116": "config_file_declared",
                    "GW230702_185453": "own_analytic",
                    "GW240925_005809": "sibling_inherited"}
    assert dl["GW150914_095045"] == dl["GW170608_020116"] == \
        "release_reweighted"
    assert dl["GW230702_185453"] == "own_analytic"
    # OD-2: LAL Planck15 for the O1-O3 reweighted rows; O4 declares it.
    np.testing.assert_array_equal(a["cosmology_H0_per_event"], 67.90)
    # A5: the Mixed rows record what each constituent's config declares.
    cfg = dict(zip(names,
                   _s(a["spin_amax_config_per_constituent_per_event"])))
    g150914 = json.loads(cfg["GW150914_095045"])
    assert g150914["C01:SEOBNRv4PHM"] == [0.99, 0.99]
    assert cfg["GW230702_185453"] == ""


def test_config_declared_event_refused_without_allow_list(real_export,
                                                          tmp_path):
    from gwcat.export import validate_export_v2
    from test_validate_v2 import _sel_reference
    _, pe, _ = real_export
    sel = _sel_reference(tmp_path, 0.99)
    with pytest.raises(ValueError) as ei:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            validate_export_v2(str(pe), str(sel))
    msg = str(ei.value)
    assert "xcheck_reference_spin_prior_source" in msg
    assert "config_file_declared [1]" in msg and "GW170608_020116" in msg
    assert "sibling_inherited [2]" in msg
