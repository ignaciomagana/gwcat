"""Tests for the ingest-correctness cluster (GW-07).

Four coupled defects that between them made the downstream spin gates
meaningless, all reproduced against the real GWTC-4.1 label set for
GW230529_181500:

  a) ``select_analysis`` exact-matched ``f"{prefix}:Mixed"``, but real releases
     label the combined sets with a spin-variant suffix
     (``C00:Mixed:HighSpin``).  With no exact match it fell through the
     waveform-priority list (also exact-matched, so also missed) to a
     ``startswith`` last resort returning whatever h5py yielded first --
     ``C00:IMRPhenomNSBH``, an aligned-spin NSBH approximant, chosen over three
     available Mixed sets.  ``selection_reason`` then recorded
     ``"preferred_priority"``, which was simply untrue.
  b) ``_waveform_family`` split on the first ``'-'``, mapping
     ``IMRPhenomPv2-NRTidalv2`` (a tidal waveform) to ``IMRPhenomPv2`` (a BBH
     one).
  c) ``resolve_spin_prior`` borrowed the first sibling carrying ``a_1``/``a_2``
     in h5py key order, with no check that it shared the ingested run's spin
     parameterisation -- across HighSpin/LowSpin variants that is a ~20x error
     in the secondary's prior support.
  d) ``_read_f_ref`` looked only in ``data.config``, which is EMPTY for the
     combined Mixed sets, so 175 of 282 rows stored ``f_ref = NaN`` even though a
     sibling's ``meta_data['f_ref']`` had it.  Spins, tilts and chi_p are all
     defined *at* ``f_ref``.

Plus (e) the mass prior is now parsed rather than stamped, and the ``0.0``
truthiness bug on ``f_ref`` is gone.

Fixtures are the real label sets and the real repr shapes, driven through the
repo's fake-PESummary pattern -- no pesummary, bilby or network.
"""
import numpy as np
import pytest

import gwcat.ingest as ing
from gwcat.catalog import GWCatalog
from gwcat.ingest import (IngestConfig, SpinPriorMismatchError, _label_parts,
                          _read_f_ref, _sample_set_meta, _waveform_family,
                          build_store, find_mixed_analyses, rank_analyses,
                          resolve_mass_prior, resolve_spin_prior,
                          select_analysis)

# The real analysis labels in
# IGWN-GWTC4p1-18965dda8_5-GW230529_181500-combined_PEDataRelease.hdf5
GW230529_LABELS = [
    "C00:IMRPhenomNSBH",
    "C00:IMRPhenomPv2_NRTidalv2",
    "C00:IMRPhenomXAS:HighSpin",
    "C00:IMRPhenomXAS:LowSpinSecondary",
    "C00:IMRPhenomXHM",
    "C00:IMRPhenomXP",
    "C00:IMRPhenomXPHM:HighSpin",
    "C00:IMRPhenomXPHM:LowSpin",
    "C00:IMRPhenomXPHM:LowSpinSecondary",
    "C00:Mixed:HighSpin",
    "C00:Mixed:LowSpinSecondary",
    "C00:SEOBNRv4_ROM_NRTidalv2_NSBH",
    "C00:SEOBNRv5PHM:HighSpin",
    "C00:SEOBNRv5PHM:LowSpinSecondary",
]

CFG = IngestConfig()


# --------------------------------------------------------------------------
# (a) analysis selection
# --------------------------------------------------------------------------
def test_label_parts():
    assert _label_parts("C00:Mixed:HighSpin") == ("C00", "Mixed", "HighSpin")
    assert _label_parts("C01:Mixed") == ("C01", "Mixed", "")
    assert _label_parts("C00:IMRPhenomXPHM-SpinTaylor") == (
        "C00", "IMRPhenomXPHM-SpinTaylor", "")


def test_finds_suffixed_mixed_sets_best_variant_first():
    assert find_mixed_analyses(GW230529_LABELS, "C00") == [
        "C00:Mixed:HighSpin", "C00:Mixed:LowSpinSecondary"]


def test_gw230529_selects_a_mixed_set_not_the_aligned_spin_nsbh_run():
    """The headline regression: an aligned-spin NSBH approximant was being
    ingested over three available Mixed sets."""
    picked = select_analysis(GW230529_LABELS, "C00", CFG)
    assert picked == "C00:Mixed:HighSpin"
    assert picked != "C00:IMRPhenomNSBH"


def test_unsuffixed_mixed_still_wins():
    """O3-style labels are unaffected."""
    labels = ["C01:IMRPhenomXPHM", "C01:Mixed", "C01:SEOBNRv4PHM"]
    assert select_analysis(labels, "C01", CFG) == "C01:Mixed"


def test_ranking_puts_mixed_then_priority_then_the_rest():
    ranked = rank_analyses(GW230529_LABELS, "C00", CFG)
    assert ranked[:2] == ["C00:Mixed:HighSpin", "C00:Mixed:LowSpinSecondary"]
    # the priority list entry "C00:SEOBNRv5PHM" matches its suffixed spellings
    assert ranked[2:4] == ["C00:SEOBNRv5PHM:HighSpin",
                           "C00:SEOBNRv5PHM:LowSpinSecondary"]
    assert set(ranked) == set(GW230529_LABELS)
    assert len(ranked) == len(GW230529_LABELS)


def test_priority_list_tolerates_a_spin_variant_suffix():
    """Only suffixed spellings present: the priority list must still match."""
    labels = ["C00:IMRPhenomNSBH", "C00:SEOBNRv5PHM:HighSpin"]
    assert select_analysis(labels, "C00", CFG) == "C00:SEOBNRv5PHM:HighSpin"


def test_last_resort_is_labelled_honestly_not_as_a_priority_hit():
    """When neither a Mixed set nor the priority list matches, say so."""
    labels = ["C00:IMRPhenomXHM", "C00:IMRPhenomXP"]
    picked = select_analysis(labels, "C00", CFG)
    ss = _sample_set_meta(picked, picked, rank_analyses(labels, "C00", CFG),
                          "f.h5", "preferred", priority_labels=())
    assert ss["selection_reason"] == "preferred_last_resort"


@pytest.mark.parametrize("label,reason", [
    ("C00:Mixed:HighSpin", "preferred_mixed"),
    ("C01:Mixed", "preferred_mixed"),
])
def test_mixed_sets_report_preferred_mixed(label, reason):
    ss = _sample_set_meta(label, label, [label], "f.h5", "preferred")
    assert ss["selection_reason"] == reason
    assert ss["is_mixed"] == 1.0


def test_sample_set_meta_splits_the_variant_off_the_approximant():
    ss = _sample_set_meta("C00:Mixed:HighSpin", "C00:Mixed:HighSpin",
                          ["C00:Mixed:HighSpin"], "f.h5", "preferred")
    assert ss["approximant"] == "Mixed"        # not "Mixed:HighSpin"
    assert ss["spin_variant"] == "HighSpin"
    assert ss["sample_set_name"] == "C00:Mixed:HighSpin"


# --------------------------------------------------------------------------
# (b) waveform family
# --------------------------------------------------------------------------
@pytest.mark.parametrize("approximant,family", [
    # a genuine configuration suffix is stripped ...
    ("IMRPhenomXPHM-SpinTaylor", "IMRPhenomXPHM"),
    # ... but a hyphen that is part of the waveform NAME is not
    ("IMRPhenomPv2-NRTidalv2", "IMRPhenomPv2-NRTidalv2"),
    ("SEOBNRv4_ROM_NRTidalv2_NSBH", "SEOBNRv4_ROM_NRTidalv2_NSBH"),
    ("SEOBNRv5PHM", "SEOBNRv5PHM"),
    ("Mixed", "Mixed"),
    ("", ""),
])
def test_waveform_family(approximant, family):
    assert _waveform_family(approximant) == family


# --------------------------------------------------------------------------
# (c) the spin prior may not be borrowed across spin variants
# --------------------------------------------------------------------------
def _spin_priors(mapping):
    """{analysis: amax} -> a bilby-style analytic priors group."""
    return {"analytic": {
        an: {
            "a_1": f"Uniform(minimum=0.0, maximum={amax}, name='a_1')",
            "a_2": f"Uniform(minimum=0.0, maximum={amax}, name='a_2')",
            "tilt_1": "Sine(name='tilt_1', minimum=0.0, maximum=3.14159)",
            "tilt_2": "Sine(name='tilt_2', minimum=0.0, maximum=3.14159)",
        } for an, amax in mapping.items()}}


def test_spin_prior_prefers_the_matching_variant_sibling():
    """A HighSpin Mixed set must borrow from a HighSpin sibling, not the
    LowSpinSecondary one that happens to sort first."""
    analyses = ["C00:IMRPhenomXPHM:LowSpinSecondary",
                "C00:Mixed:HighSpin",
                "C00:SEOBNRv5PHM:HighSpin"]
    priors = _spin_priors({"C00:IMRPhenomXPHM:LowSpinSecondary": 0.05,
                           "C00:SEOBNRv5PHM:HighSpin": 0.99})
    amax_1, amax_2, kind, src = resolve_spin_prior(
        "C00:Mixed:HighSpin", analyses, priors, None, None)
    assert (amax_1, amax_2) == pytest.approx((0.99, 0.99))
    assert kind == "uniform_magnitude_isotropic"
    assert "C00:SEOBNRv5PHM:HighSpin" in src


def test_spin_prior_refuses_to_borrow_across_variants():
    """Only a differently-parameterised sibling exists -> raise, do not borrow."""
    analyses = ["C00:Mixed:HighSpin", "C00:IMRPhenomXPHM:LowSpinSecondary"]
    priors = _spin_priors({"C00:IMRPhenomXPHM:LowSpinSecondary": 0.05})
    with pytest.raises(SpinPriorMismatchError, match="spin variant"):
        resolve_spin_prior("C00:Mixed:HighSpin", analyses, priors, None, None)


def test_spin_prior_variant_mismatch_can_be_allowed_with_a_warning():
    analyses = ["C00:Mixed:HighSpin", "C00:IMRPhenomXPHM:LowSpinSecondary"]
    priors = _spin_priors({"C00:IMRPhenomXPHM:LowSpinSecondary": 0.05})
    with pytest.warns(UserWarning, match="spin variant"):
        amax_1, amax_2, _kind, src = resolve_spin_prior(
            "C00:Mixed:HighSpin", analyses, priors, None, None,
            allow_variant_mismatch=True)
    assert (amax_1, amax_2) == pytest.approx((0.05, 0.05))
    assert "variant_mismatch" in src


def test_unsuffixed_mixed_assumes_the_best_variant_rather_than_refusing():
    """The real GWTC-3 NSBH case: a plain ``C01:Mixed`` set whose only
    prior-carrying siblings are variant-suffixed.

    Which variant the combined set corresponds to is not stated in the file, so
    this is an assumption, not a contradiction -- refusing here would block
    GW191219_163120, GW200105_162426 and GW200115_042309, which ingest fine
    today.  HighSpin is taken deterministically (and is what h5py key order
    happened to give before GW-07, so no current number moves).
    """
    analyses = ["C01:IMRPhenomXPHM:HighSpin", "C01:IMRPhenomXPHM:LowSpin",
                "C01:Mixed"]
    priors = _spin_priors({"C01:IMRPhenomXPHM:HighSpin": 0.99,
                           "C01:IMRPhenomXPHM:LowSpin": 0.05})
    with pytest.warns(UserWarning, match="declares no spin variant"):
        amax_1, amax_2, kind, src = resolve_spin_prior(
            "C01:Mixed", analyses, priors, None, None)
    assert (amax_1, amax_2) == pytest.approx((0.99, 0.99))
    assert kind == "uniform_magnitude_isotropic"
    assert "variant_assumed(HighSpin)" in src


def test_compound_variants_match_on_the_spin_token():
    """``C01:Mixed:NSBH:HighSpin`` is a HighSpin prior; the ``NSBH`` component
    must not make it look like a different spin restriction."""
    analyses = ["C01:Mixed:NSBH:HighSpin", "C01:IMRPhenomXPHM:HighSpin"]
    priors = _spin_priors({"C01:IMRPhenomXPHM:HighSpin": 0.99})
    amax_1, _a2, _kind, src = resolve_spin_prior(
        "C01:Mixed:NSBH:HighSpin", analyses, priors, None, None)
    assert amax_1 == pytest.approx(0.99)
    assert "C01:IMRPhenomXPHM:HighSpin" in src
    assert "variant_assumed" not in src and "mismatch" not in src


def test_compound_variant_still_refuses_a_genuine_contradiction():
    analyses = ["C01:Mixed:NSBH:HighSpin", "C01:IMRPhenomXPHM:LowSpin"]
    priors = _spin_priors({"C01:IMRPhenomXPHM:LowSpin": 0.05})
    with pytest.raises(SpinPriorMismatchError, match="HighSpin"):
        resolve_spin_prior("C01:Mixed:NSBH:HighSpin", analyses, priors,
                           None, None)


def test_unsuffixed_labels_are_unaffected_by_the_variant_rule():
    """O3-style: every label has variant "", so they all match each other."""
    analyses = ["C01:Mixed", "C01:IMRPhenomXPHM"]
    priors = _spin_priors({"C01:IMRPhenomXPHM": 0.99})
    amax_1, amax_2, kind, src = resolve_spin_prior(
        "C01:Mixed", analyses, priors, None, None)
    assert (amax_1, amax_2) == pytest.approx((0.99, 0.99))
    assert "C01:IMRPhenomXPHM" in src


def test_aligned_spin_siblings_are_skipped_not_borrowed_from():
    """An aligned-spin run records chi_1/chi_2, not a_1/a_2, so it is simply not
    a candidate -- the precessing sibling is used instead."""
    analyses = ["C00:Mixed:HighSpin", "C00:IMRPhenomNSBH",
                "C00:IMRPhenomXPHM:HighSpin"]
    priors = _spin_priors({"C00:IMRPhenomXPHM:HighSpin": 0.99})
    priors["analytic"]["C00:IMRPhenomNSBH"] = {
        "chi_1": "Uniform(minimum=-0.5, maximum=0.5, name='chi_1')",
        "chi_2": "Uniform(minimum=-0.05, maximum=0.05, name='chi_2')",
    }
    amax_1, _amax_2, _kind, src = resolve_spin_prior(
        "C00:Mixed:HighSpin", analyses, priors, None, None)
    assert amax_1 == pytest.approx(0.99)
    assert "C00:IMRPhenomXPHM:HighSpin" in src


# --------------------------------------------------------------------------
# (d) f_ref
# --------------------------------------------------------------------------
class _Data:
    """Stand-in for a pesummary read() result with the real layout."""

    def __init__(self, labels, config=None, extra=None):
        self.samples_dict = {lbl: {} for lbl in labels}
        self.config = config or {}
        # pesummary exposes extra_kwargs as a LIST aligned with samples_dict
        self.extra_kwargs = [(extra or {}).get(lbl, {}) for lbl in labels]


def test_f_ref_from_config():
    d = _Data(["C01:IMRPhenomXPHM"],
              config={"C01:IMRPhenomXPHM": {"config": {
                  "reference-frequency": "20"}}})
    val, src = _read_f_ref(d, "C01:IMRPhenomXPHM", ["C01:IMRPhenomXPHM"])
    assert val == pytest.approx(20.0)
    assert src == "config[C01:IMRPhenomXPHM]"


def test_f_ref_from_meta_data_block():
    d = _Data(["C00:SEOBNRv5PHM"],
              extra={"C00:SEOBNRv5PHM": {"meta_data": {"f_ref": 20.0}}})
    val, src = _read_f_ref(d, "C00:SEOBNRv5PHM", ["C00:SEOBNRv5PHM"])
    assert val == pytest.approx(20.0)
    assert src.startswith("meta_data[")


def test_f_ref_falls_back_to_a_sibling_for_a_mixed_set():
    """The 175/282 NaN case: the Mixed set has an EMPTY config and no f_ref of
    its own, while its constituent sibling has both."""
    labels = ["C01:IMRPhenomXPHM", "C01:Mixed", "C01:SEOBNRv4PHM"]
    d = _Data(labels,
              config={"C01:IMRPhenomXPHM": {"config": {
                  "reference-frequency": "20"}},
                  "C01:Mixed": {}},                      # empty, as in reality
              extra={"C01:IMRPhenomXPHM": {"meta_data": {"f_ref": 20.0}},
                     "C01:Mixed": {"meta_data": {"f_low": 20.0}}})
    val, src = _read_f_ref(d, "C01:Mixed", labels)
    assert val == pytest.approx(20.0)
    assert src.endswith(":sibling")
    assert "C01:IMRPhenomXPHM" in src


def test_f_ref_absent_everywhere_returns_none():
    labels = ["C01:Mixed"]
    d = _Data(labels, config={"C01:Mixed": {}},
              extra={"C01:Mixed": {"meta_data": {"f_low": 20.0}}})
    assert _read_f_ref(d, "C01:Mixed", labels) == (None, "")


@pytest.mark.parametrize("bad", [0.0, -20.0, np.nan, np.inf])
def test_f_ref_rejects_non_physical_values(bad):
    """A non-positive or non-finite f_ref is absence, not a value.  (0.0 also
    used to be silently converted to NaN by an `if f_ref` truthiness test.)"""
    labels = ["C00:X"]
    d = _Data(labels, extra={"C00:X": {"meta_data": {"f_ref": bad}}})
    assert _read_f_ref(d, "C00:X", labels) == (None, "")


def test_f_ref_accepts_a_one_element_array():
    """h5py hands back 1-element arrays, which is how the real files store it."""
    labels = ["C00:X"]
    d = _Data(labels, extra={"C00:X": {"meta_data": {"f_ref": np.array([20.0])}}})
    val, _src = _read_f_ref(d, "C00:X", labels)
    assert val == pytest.approx(20.0)


# --------------------------------------------------------------------------
# (e) the mass prior is parsed, not stamped
# --------------------------------------------------------------------------
_MC = ("bilby.gw.prior.UniformInComponentsChirpMass(minimum=21.418182160215295, "
       "maximum=41.97447913941358, name='chirp_mass', unit=None)")
_Q = ("bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.05, maximum=1.0, "
      "name='mass_ratio', unit=None, boundary=None)")


def test_mass_prior_recognised_as_uniform_in_detector_frame_components():
    """UniformInComponents{ChirpMass,MassRatio} is exactly what makes the prior
    flat in (m1det, m2det), which is what the |dm2det/dq| = m1det Jacobian
    assumes."""
    priors = {"analytic": {"C01:IMRPhenomXPHM": {"chirp_mass": _MC,
                                                 "mass_ratio": _Q}}}
    mp = resolve_mass_prior("C01:Mixed",
                            ["C01:Mixed", "C01:IMRPhenomXPHM"], priors)
    assert mp.kind == "uniform_detector_frame"
    assert mp.chirp_min == pytest.approx(21.418182160215295)
    assert mp.chirp_max == pytest.approx(41.97447913941358)
    assert (mp.q_min, mp.q_max) == pytest.approx((0.05, 1.0))
    assert "C01:IMRPhenomXPHM" in mp.source


def test_mass_prior_other_classes_are_flagged_unrecognised():
    """A flat-in-chirp-mass prior is NOT flat in components, so the Jacobian
    would be wrong -- gwcat must say so rather than stamp the constant."""
    priors = {"analytic": {"C01:X": {
        "chirp_mass": "Uniform(minimum=5, maximum=50, name='chirp_mass')",
        "mass_ratio": _Q}}}
    mp = resolve_mass_prior("C01:X", ["C01:X"], priors)
    assert mp.kind == "unrecognized"
    assert "unrecognized" in mp.source


def test_mass_prior_absent_is_assumed_default():
    mp = resolve_mass_prior("C01:X", ["C01:X"], {})
    assert mp.kind == "assumed_default"
    assert mp.source == "default(no_analytic_prior)"


# --------------------------------------------------------------------------
# End to end through build_store
# --------------------------------------------------------------------------
class _FakeData:
    def __init__(self, labels, config=None, extra=None):
        self.samples_dict = {lbl: {} for lbl in labels}
        self.config = config or {}
        self.extra_kwargs = [(extra or {}).get(lbl, {}) for lbl in labels]


def _core(rng, n):
    return {
        "mass_1": rng.uniform(25, 50, n),
        "mass_2": rng.uniform(10, 25, n),
        "luminosity_distance": rng.uniform(300, 800, n),
        "ra": rng.uniform(0, 2 * np.pi, n),
        "dec": rng.uniform(-np.pi / 2, np.pi / 2, n),
        "chi_eff": rng.uniform(-0.4, 0.4, n),
        "a_1": rng.uniform(0, 0.9, n),
        "a_2": rng.uniform(0, 0.9, n),
        "tilt_1": rng.uniform(0, np.pi, n),
        "tilt_2": rng.uniform(0, np.pi, n),
    }


def test_build_store_ingests_the_mixed_variant_and_records_the_provenance(
        tmp_path, monkeypatch):
    """The whole cluster, end to end on the real GW230529 label shape."""
    rng = np.random.default_rng(0)
    labels = ["C00:IMRPhenomNSBH", "C00:IMRPhenomXPHM:HighSpin",
              "C00:Mixed:HighSpin"]
    analyses = {lbl: _core(rng, 40) for lbl in labels}
    priors = _spin_priors({"C00:IMRPhenomXPHM:HighSpin": 0.99})
    priors["analytic"]["C00:IMRPhenomXPHM:HighSpin"].update(
        {"chirp_mass": _MC, "mass_ratio": _Q})
    data = _FakeData(
        labels,
        config={"C00:Mixed:HighSpin": {}},
        extra={"C00:IMRPhenomXPHM:HighSpin": {"meta_data": {"f_ref": 20.0}},
               "C00:Mixed:HighSpin": {"meta_data": {"f_low": 20.0}}})

    monkeypatch.setattr(ing, "_read_event_pesummary",
                        lambda path: (data, analyses, labels, priors))
    src = tmp_path / "IGWN-GWTC4p1-x-GW230529_181500-combined_PEDataRelease.hdf5"
    src.write_bytes(b"")
    out = tmp_path / "store.h5"
    build_store([str(src)], str(out), event_table={},
                cfg=IngestConfig(validate_prior=False))

    cat = GWCatalog(str(out))
    assert cat.n_events == 1
    # (a) the Mixed set was ingested, not the aligned-spin NSBH run
    assert cat.meta["analysis_used"][0] == "C00:Mixed:HighSpin"
    assert cat.meta["selection_reason"][0] == "preferred_mixed"
    assert cat.meta["spin_variant"][0] == "HighSpin"
    assert cat.meta["approximant"][0] == "Mixed"
    # (d) f_ref came from the constituent sibling, and says so
    assert float(cat.meta["f_ref"][0]) == pytest.approx(20.0)
    assert "sibling" in cat.meta["f_ref_source"][0]
    # (c) the spin prior came from the HighSpin sibling
    assert float(cat.meta["spin_amax_1"][0]) == pytest.approx(0.99)
    assert "HighSpin" in cat.meta["spin_prior_source"][0]
    # (e) the mass prior was parsed and recognised
    assert cat.meta["mass_prior_kind"][0] == "uniform_detector_frame"
    assert float(cat.meta["mass_prior_q_min"][0]) == pytest.approx(0.05)
    assert "C00:IMRPhenomXPHM:HighSpin" in cat.meta["mass_prior_source"][0]


def test_build_store_warns_when_the_mass_prior_is_not_uniform_in_components(
        tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    labels = ["C01:Mixed", "C01:IMRPhenomXPHM"]
    analyses = {lbl: _core(rng, 30) for lbl in labels}
    priors = _spin_priors({"C01:IMRPhenomXPHM": 0.99})
    priors["analytic"]["C01:IMRPhenomXPHM"].update({
        "chirp_mass": "Uniform(minimum=5, maximum=50, name='chirp_mass')",
        "mass_ratio": _Q})
    data = _FakeData(labels)
    monkeypatch.setattr(ing, "_read_event_pesummary",
                        lambda path: (data, analyses, labels, priors))
    src = tmp_path / "IGWN-GWTC3p0-x-GW190101_000000_mixed_cosmo.h5"
    src.write_bytes(b"")
    with pytest.warns(UserWarning, match="not uniform in detector-frame"):
        build_store([str(src)], str(tmp_path / "s.h5"), event_table={},
                    cfg=IngestConfig(validate_prior=False))
