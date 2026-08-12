import warnings

import numpy as np
import h5py
import pytest

from gwcat.selection import SelectionSet

# The endo3/O4 fixture writers live alongside this file.
from test_selection_spin import write_o4_full, write_endo3_full


def _base_o4_columns(n=3):
    fields = [
        ("mass1_source", "f8"),
        ("mass2_source", "f8"),
        ("mass1_detector", "f8"),
        ("mass2_detector", "f8"),
        ("luminosity_distance", "f8"),
        ("z", "f8"),
        ("dluminosity_distance_dredshift", "f8"),
        ("right_ascension", "f8"),
        ("declination", "f8"),
        ("spin1x", "f8"),
        ("spin1y", "f8"),
        ("spin1z", "f8"),
        ("spin2x", "f8"),
        ("spin2y", "f8"),
        ("spin2z", "f8"),
        ("chi_eff", "f8"),
        ("weights", "f8"),
        ("lnpdraw_mass1_source", "f8"),
        ("lnpdraw_mass2_source_GIVEN_mass1_source", "f8"),
        ("lnpdraw_z", "f8"),
        ("pycbc_far", "f8"),
        ("cwb-bbh_far", "f8"),
    ]
    events = np.zeros(n, dtype=fields)
    events["mass1_source"] = 30.0
    events["mass2_source"] = 20.0
    events["z"] = 0.1
    events["mass1_detector"] = 33.0
    events["mass2_detector"] = 22.0
    events["luminosity_distance"] = 450.0
    events["dluminosity_distance_dredshift"] = 4500.0
    events["right_ascension"] = 1.0
    events["declination"] = 0.5
    events["spin1z"] = 0.2
    events["spin2z"] = -0.1
    events["chi_eff"] = 0.08
    events["weights"] = 2.0
    events["lnpdraw_mass1_source"] = -1.0
    events["lnpdraw_mass2_source_GIVEN_mass1_source"] = -2.0
    events["lnpdraw_z"] = -3.0
    events["pycbc_far"] = [0.5, 2.0, 0.2]
    events["cwb-bbh_far"] = [2.0, 2.0, 2.0]
    return events


def test_selection_reads_public_o4_compound_factored_lnpdraw(tmp_path):
    path = tmp_path / "o4_public.hdf"
    with h5py.File(path, "w") as f:
        f.attrs["total_analysis_time"] = 365.25 * 24 * 3600
        f.attrs["total_generated"] = 100
        f.attrs["searches"] = np.array([b"pycbc", b"cwb-bbh"])
        f.create_dataset("events", data=_base_o4_columns())

    selection = SelectionSet(str(path))
    selection._load()

    assert selection._pdraw.shape == (3,)
    assert np.all(np.isfinite(selection._pdraw))
    assert selection.detected_mask(1.0).tolist() == [True, False, True]


def test_selection_reads_events_group_factored_lnpdraw(tmp_path):
    path = tmp_path / "o4_group.hdf"
    events = _base_o4_columns()
    with h5py.File(path, "w") as f:
        f.attrs["total_analysis_time"] = 365.25 * 24 * 3600
        f.attrs["total_generated"] = 100
        f.attrs["searches"] = np.array(["pycbc"], dtype=h5py.string_dtype())
        group = f.create_group("events")
        for name in events.dtype.names:
            group.create_dataset(name, data=events[name])

    selection = SelectionSet(str(path))
    selection._load()

    assert selection._pdraw.shape == (3,)
    assert np.all(np.isfinite(selection._pdraw))
    assert selection.detected_mask(1.0).tolist() == [True, False, True]


# ==========================================================================
# GW-09: cosmology honesty, and a fatal missing ndraw
# ==========================================================================
def test_detect_generation_cosmology_recovers_a_known_one():
    """The detector must identify the cosmology that made the pairs, tightly."""
    from gwcat.selection import detect_generation_cosmology
    from gwcat.cosmology import make_cosmology

    z = np.linspace(0.01, 2.0, 500)
    dL = make_cosmology(67.90, 0.3065).luminosity_distance(z).value
    got = detect_generation_cosmology(z, dL)
    assert got is not None
    H0, Om0, resid = got
    assert abs(H0 - 67.90) < 1e-3
    assert abs(Om0 - 0.3065) < 1e-3
    assert resid < 1e-5


def test_detect_generation_cosmology_refuses_non_lcdm_pairs():
    """A miss must return None, not a nearest-grid-point answer.

    The whole value of the detector is the gap between "identified" and
    "guessed": the real endo3 campaign matches at 1.6e-7 while a wrong-but-
    nearby cosmology sits at 2.3e-3, four orders of magnitude away.
    """
    from gwcat.selection import detect_generation_cosmology

    rng = np.random.default_rng(0)
    z = np.linspace(0.01, 2.0, 500)
    dL = 3000.0 * z * (1.0 + 0.4 * z ** 1.7) * (1 + 0.05 * rng.normal(size=z.size))
    assert detect_generation_cosmology(z, dL) is None


def test_missing_total_generated_raises(tmp_path):
    """ndraw=0 zeroed a campaign's Essick fraction and sent it to Lambda/0."""
    from gwcat.selection import SelectionSet

    p = write_endo3_full(tmp_path / "no_ndraw.hdf", n=200)
    with h5py.File(p, "r+") as f:
        if "total_generated" in f.attrs:
            del f.attrs["total_generated"]
        if "total_generated" in f["injections"].attrs:
            del f["injections"].attrs["total_generated"]

    s = SelectionSet(str(p))
    with pytest.raises(RuntimeError) as ei:
        s._load()
    msg = str(ei.value)
    assert "total_generated" in msg
    # The message has to say what goes wrong, not just what is missing.
    assert "infinite" in msg or "inf" in msg


def test_events_format_records_that_no_cosmology_was_used(tmp_path):
    """An events-format campaign reads its own ddL/dz -- no cosmology at all."""
    from gwcat.selection import SelectionSet

    s = SelectionSet(str(write_o4_full(tmp_path / "o4.hdf", n=200)))
    s._load()
    assert s._cosmology_source == "file"
    assert s._cosmology_used_H0 is None and s._cosmology_used_Om0 is None


def test_override_on_an_events_campaign_warns_that_it_is_inert(tmp_path):
    from gwcat.selection import SelectionSet

    s = SelectionSet(str(write_o4_full(tmp_path / "o4w.hdf", n=200)),
                     H0=70.0, Om0=0.30)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s._load()
    assert s._cosmology_source == "file"
    assert any("INERT" in str(x.message) for x in w)


def test_override_rejected_when_it_reaches_only_some_campaigns(tmp_path):
    """The corrupting case: one product, two cosmologies, one reported number."""
    from gwcat.selection import SelectionSet
    from gwcat.export.selection_builder import build_selection_product

    o4 = write_o4_full(tmp_path / "mo4.hdf", n=200)
    e3 = write_endo3_full(tmp_path / "me3.hdf", n=200)
    sets = [SelectionSet(str(e3), H0=70.0, Om0=0.30),
            SelectionSet(str(o4), H0=70.0, Om0=0.30)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError) as ei:
            build_selection_product(sets, spin_basis="component")
    msg = str(ei.value)
    assert "only SOME campaigns" in msg
    assert "cosmology_source='file'" in msg


def test_campaigns_using_their_own_cosmologies_are_not_refused(tmp_path):
    """Different-per-campaign is CORRECT, not mixed-and-wrong.

    endo3 should use its own generation cosmology while an events campaign uses
    its stored derivative. Refusing that would refuse the right answer.
    """
    from gwcat.selection import SelectionSet
    from gwcat.export.selection_builder import build_selection_product

    o4 = write_o4_full(tmp_path / "co4.hdf", n=200)
    e3 = write_endo3_full(tmp_path / "ce3.hdf", n=200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = build_selection_product(
            [SelectionSet(str(e3)), SelectionSet(str(o4))],
            spin_basis="component")
    src = [s.decode() if isinstance(s, bytes) else str(s)
           for s in p.attrs["cosmology_source_per_campaign"]]
    assert len(src) == 2 and "file" in src
    # The 'file' campaign carries NaN, which is how "no cosmology" is spelled.
    H0 = np.asarray(p.attrs["cosmology_H0_per_campaign"], dtype=float)
    assert not np.isfinite(H0[src.index("file")])


def test_combined_records_per_campaign_cosmology(tmp_path):
    from gwcat.selection import SelectionSet
    from gwcat.export.selection_builder import build_selection_product

    o4 = write_o4_full(tmp_path / "po4.hdf", n=200)
    e3 = write_endo3_full(tmp_path / "pe3.hdf", n=200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = build_selection_product(
            [SelectionSet(str(e3)), SelectionSet(str(o4))],
            spin_basis="component")
    a = p.attrs
    assert int(a["n_campaigns"]) == 2
    for key in ("cosmology_source_per_campaign", "cosmology_H0_per_campaign",
                "cosmology_Om0_per_campaign"):
        assert len(np.atleast_1d(a[key])) == 2
    assert bool(a["cosmology_mixed_across_campaigns"]) is True
