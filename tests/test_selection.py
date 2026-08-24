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


def test_detect_generation_cosmology_finds_off_grid_om0():
    """A campaign generated at an Om0 between grid points must still detect.

    The coarse grid alone left a ~2e-4 residual for a half-step Om0 error --
    20x over rtol -- so Planck18's 0.30966 (or any future off-grid campaign)
    silently fell back to Planck15: the exact wrong-Jacobian bug the detector
    exists to prevent.  Local refinement closes it.
    """
    from gwcat.selection import detect_generation_cosmology
    from gwcat.cosmology import make_cosmology

    z = np.linspace(0.01, 2.0, 500)
    dL = make_cosmology(67.66, 0.30966).luminosity_distance(z).value
    got = detect_generation_cosmology(z, dL)
    assert got is not None
    H0, Om0, resid = got
    assert abs(H0 - 67.66) < 1e-3
    assert abs(Om0 - 0.30966) < 1e-4
    assert resid < 1e-5


def test_detect_generation_cosmology_rejects_unit_errors():
    """dL scales exactly as 1/H0, so dL-in-Gpc fits PERFECTLY at H0~68000.

    A confident detection of an absurd H0 is a unit mistake, not a cosmology;
    accepting it would build that campaign's Jacobian off by the unit factor
    relative to every other campaign in a combined product.
    """
    from gwcat.selection import detect_generation_cosmology
    from gwcat.cosmology import make_cosmology

    z = np.linspace(0.01, 2.0, 500)
    dL_gpc = make_cosmology(67.90, 0.3065).luminosity_distance(z).value / 1e3
    assert detect_generation_cosmology(z, dL_gpc) is None


def test_v1_combined_export_refuses_partial_override(tmp_path):
    """The v1/CLI path hits the identical corruption the v2 builder refuses.

    GW-09 added the refusal only to build_selection_product; `gwcat selection
    --H0 ... --Om0 ...` dispatches to CombinedSelectionSet.to_darksirens,
    which also has every campaign in view and must refuse identically.
    """
    from gwcat.selection import SelectionSet, CombinedSelectionSet

    o4 = write_o4_full(tmp_path / "v1o4.hdf", n=200)
    e3 = write_endo3_full(tmp_path / "v1e3.hdf", n=200)
    comb = CombinedSelectionSet([SelectionSet(str(e3), H0=70.0, Om0=0.30),
                                 SelectionSet(str(o4), H0=70.0, Om0=0.30)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError) as ei:
            comb.to_darksirens(str(tmp_path / "v1out.h5"))
    assert "only SOME campaigns" in str(ei.value)


def test_v1_writers_stamp_the_cosmology_pdraw_used(tmp_path):
    """v1 attrs must describe the file's own Jacobian, not Planck15 by rote.

    After GW-09 an endo3-style campaign's pdraw uses its DETECTED generation
    cosmology, and an events-format campaign's uses none at all; stamping
    self.H0 regardless (the pre-GW-09 stamp) made every v1 file mis-describe
    its own pdraw.
    """
    from gwcat.selection import SelectionSet, CombinedSelectionSet

    o4 = write_o4_full(tmp_path / "h1o4.hdf", n=200)
    e3 = write_endo3_full(tmp_path / "h1e3.hdf", n=200)

    # Single events-format campaign: no cosmology entered pdraw -> NaN + source.
    out1 = tmp_path / "h1_single.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        SelectionSet(str(o4)).to_darksirens(str(out1))
    with h5py.File(out1, "r") as f:
        assert not np.isfinite(f.attrs["cosmology_H0"])
        src = f.attrs["cosmology_source"]
        assert (src.decode() if isinstance(src, bytes) else str(src)) == "file"

    # Combined campaigns that legitimately differ: scalar NaN, per-campaign
    # arrays authoritative, mixed flag set.
    out2 = tmp_path / "h1_comb.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        CombinedSelectionSet([SelectionSet(str(e3)), SelectionSet(str(o4))]
                             ).to_darksirens(str(out2))
    with h5py.File(out2, "r") as f:
        assert not np.isfinite(f.attrs["cosmology_H0"])
        assert bool(f.attrs["cosmology_mixed_across_campaigns"]) is True
        src = [s.decode() if isinstance(s, bytes) else str(s)
               for s in f.attrs["cosmology_source_per_campaign"]]
        assert len(src) == 2 and "file" in src
        H0pc = np.asarray(f.attrs["cosmology_H0_per_campaign"], dtype=float)
        assert not np.isfinite(H0pc[src.index("file")])


def _strip_sky(path):
    """Remove ra/dec columns from an events-format file (semianalytic O1/O2)."""
    with h5py.File(path, "r+") as f:
        ev = f["events"]
        if isinstance(ev, h5py.Dataset):
            keep = [n for n in ev.dtype.names
                    if n not in ("right_ascension", "declination")]
            sub = np.zeros(ev.shape, dtype=[(n, ev.dtype[n]) for n in keep])
            for n in keep:
                sub[n] = ev[n]
            del f["events"]
            f.create_dataset("events", data=sub)
        else:
            for n in ("right_ascension", "declination"):
                if n in ev:
                    del ev[n]
    return path


def test_v1_writers_record_sky_availability(tmp_path):
    """GW-10, v1 half: the writers must say when ra/dec are declared-NaN.

    The v2 builder has recorded sky_position_available since GW-20; the v1
    exporters wrote the NaN sky columns with no provenance flag at all, so a
    consumer could not tell declared-absent from corrupt.
    """
    from gwcat.selection import SelectionSet, CombinedSelectionSet

    nosky = _strip_sky(write_o4_full(tmp_path / "sky0.hdf", n=200))
    sky = write_o4_full(tmp_path / "sky1.hdf", n=200)

    out1 = tmp_path / "sky_single.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        SelectionSet(str(nosky)).to_darksirens(str(out1))
    with h5py.File(out1, "r") as f:
        assert bool(f.attrs["sky_position_available"]) is False

    out2 = tmp_path / "sky_comb.h5"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        CombinedSelectionSet([SelectionSet(str(nosky)), SelectionSet(str(sky))]
                             ).to_darksirens(str(out2))
    assert any("WITHOUT drawn sky positions" in str(x.message) for x in w)
    with h5py.File(out2, "r") as f:
        got = np.asarray(f.attrs["sky_position_available"], dtype=bool)
        assert got.tolist() == [False, True]


def test_v2_builder_warns_on_mixed_sky_availability(tmp_path):
    from gwcat.selection import SelectionSet
    from gwcat.export.selection_builder import build_selection_product

    nosky = _strip_sky(write_o4_full(tmp_path / "v2sky0.hdf", n=200))
    sky = write_o4_full(tmp_path / "v2sky1.hdf", n=200)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        p = build_selection_product(
            [SelectionSet(str(nosky)), SelectionSet(str(sky))],
            spin_basis="component")
    assert any("WITHOUT drawn sky positions" in str(x.message) for x in w)
    got = np.asarray(p.attrs["sky_position_available"], dtype=bool)
    assert got.tolist() == [False, True]


# ==========================================================================
# GW-28 (#3): the efficiency denominator is the GENERATED draw count
# ==========================================================================
def _write_o4(path, events=None, total_generated=100,
              total_analysis_time=365.25 * 24 * 3600):
    """Write a minimal compound-dataset O4 file from ``_base_o4_columns``."""
    events = _base_o4_columns() if events is None else events
    with h5py.File(path, "w") as f:
        f.attrs["total_analysis_time"] = total_analysis_time
        f.attrs["total_generated"] = total_generated
        f.attrs["searches"] = np.array([b"pycbc", b"cwb-bbh"])
        f.create_dataset("events", data=events)
    return str(path)


def test_detection_efficiency_denominator_is_total_generated(tmp_path):
    """A clipped file's retained rows are NOT the campaign's draw count.

    The O4ab clipped release keeps 2,959,534 of 870,454,872 generated draws, so
    dividing the 986,829 detections by the retained rows reported 0.33344 for a
    true generated-draw efficiency of 0.0011337 -- high by 294x.  Here the same
    ratio in miniature: 2 detected, 3 retained rows, 1000 generated.
    """
    from gwcat.selection import SelectionSet

    s = SelectionSet(_write_o4(tmp_path / "eff.hdf", total_generated=1000))

    assert s.n_retained == 3
    assert s.n_generated == 1000
    assert int(s.detected_mask(1.0).sum()) == 2
    assert s.detection_efficiency(1.0) == pytest.approx(2 / 1000)
    # The old denominator, explicitly refused.
    assert s.detection_efficiency(1.0) != pytest.approx(2 / 3)


def test_n_injections_is_the_retained_row_count(tmp_path):
    """The legacy name keeps its value; the two counts are now distinct."""
    from gwcat.selection import SelectionSet

    s = SelectionSet(_write_o4(tmp_path / "counts.hdf", total_generated=1000))
    assert s.n_injections == s.n_retained == 3
    assert s.n_generated == 1000
    assert s.n_injections != s.n_generated


def test_combined_detection_efficiency_denominator_is_total_generated(tmp_path):
    """Combined: sum of detections over the summed ndraw, not summed rows."""
    from gwcat.selection import SelectionSet, CombinedSelectionSet

    a = SelectionSet(_write_o4(tmp_path / "ca.hdf", total_generated=1000))
    b = SelectionSet(_write_o4(tmp_path / "cb.hdf", total_generated=4000))
    comb = CombinedSelectionSet([a, b])

    assert comb.n_retained == 6
    assert comb.n_generated == 5000
    assert comb.detection_efficiency(1.0) == pytest.approx(4 / 5000)
    assert comb.detection_efficiency(1.0) != pytest.approx(4 / 6)


# ==========================================================================
# GW-28 (#5): malformed PDFs / weights / times are refused, never floored
# ==========================================================================
def _load_error(path, H0=None, Om0=None):
    """Load ``path`` expecting a ValueError; return its message."""
    from gwcat.selection import SelectionSet

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError) as ei:
            SelectionSet(str(path), H0=H0, Om0=Om0)._load()
    return str(ei.value)


def _endo3(tmp_path, name, mutate):
    """Write an endo3 fixture and mutate one field/attr in place."""
    p = write_endo3_full(tmp_path / name, n=64)
    with h5py.File(p, "r+") as f:
        mutate(f)
    return p


def test_o3_zero_mass_sampling_pdf_is_refused_not_floored(tmp_path):
    """It used to become exp(-690.8) -- a density that was never drawn."""
    def zero_it(f):
        f["injections/mass1_source_mass2_source_sampling_pdf"][3] = 0.0

    msg = _load_error(_endo3(tmp_path, "e3_pmass.hdf", zero_it))
    assert "mass1_source_mass2_source_sampling_pdf" in msg
    assert "1 of 64" in msg and "index 3" in msg


def test_o3_negative_redshift_sampling_pdf_is_refused(tmp_path):
    def negate_it(f):
        f["injections/redshift_sampling_pdf"][7] = -0.5

    msg = _load_error(_endo3(tmp_path, "e3_pz.hdf", negate_it))
    assert "redshift_sampling_pdf" in msg and "index 7" in msg


def test_o3_zero_mixture_weight_is_refused(tmp_path):
    """pdraw /= 0 used to write inf straight into the selection file."""
    def add_bad_weight(f):
        w = np.ones(64)
        w[5] = 0.0
        f["injections"].create_dataset("mixture_weight", data=w)

    msg = _load_error(_endo3(tmp_path, "e3_w.hdf", add_bad_weight))
    assert "mixture_weight" in msg and "index 5" in msg


def test_o3_nonpositive_analysis_time_is_refused(tmp_path):
    def zero_time(f):
        f["injections"].attrs["analysis_time_s"] = 0.0

    msg = _load_error(_endo3(tmp_path, "e3_t.hdf", zero_time))
    assert "analysis_time_s" in msg


def test_o3_zero_total_generated_is_refused(tmp_path):
    """Present-but-zero was as fatal as absent, and used to load silently."""
    def zero_ndraw(f):
        f.attrs["total_generated"] = 0

    msg = _load_error(_endo3(tmp_path, "e3_n.hdf", zero_ndraw))
    assert "total_generated" in msg and "ndraw" in msg


def test_o3_zero_joint_sampling_pdf_is_refused(tmp_path):
    """The component-spin factor floored this one at 1e-300 too."""
    def zero_joint(f):
        f["injections/sampling_pdf"][11] = 0.0

    msg = _load_error(_endo3(tmp_path, "e3_joint.hdf", zero_joint))
    assert "sampling_pdf" in msg and "index 11" in msg


def test_o4_nonpositive_weight_is_refused(tmp_path):
    ev = _base_o4_columns()
    ev["weights"][1] = 0.0
    msg = _load_error(_write_o4(tmp_path / "o4_w.hdf", ev))
    assert "weights" in msg and "index 1" in msg


def test_o4_nonfinite_analysis_time_is_refused(tmp_path):
    msg = _load_error(_write_o4(tmp_path / "o4_t.hdf",
                                total_analysis_time=np.nan))
    assert "total_analysis_time" in msg


def test_o4_zero_total_generated_is_refused(tmp_path):
    msg = _load_error(_write_o4(tmp_path / "o4_n.hdf", total_generated=0))
    assert "total_generated" in msg and "ndraw" in msg


def test_final_pdraw_is_refused_when_non_finite(tmp_path):
    """A zero Jacobian makes pdraw inf; nothing used to notice."""
    ev = _base_o4_columns()
    ev["dluminosity_distance_dredshift"][2] = 0.0
    with np.errstate(divide="ignore"):
        msg = _load_error(_write_o4(tmp_path / "o4_inf.hdf", ev))
    assert "pdraw" in msg and "index 2" in msg
    assert "infinite weight" in msg


def test_final_pdraw_is_refused_when_it_underflows_to_zero(tmp_path):
    """Underflow to exactly zero is an infinite weight in mu, not a small one."""
    ev = _base_o4_columns()
    ev["lnpdraw_mass1_source"][0] = -800.0
    msg = _load_error(_write_o4(tmp_path / "o4_zero.hdf", ev))
    assert "pdraw" in msg and "index 0" in msg


def test_exported_pdraw_is_validated_before_it_is_written(tmp_path):
    """The chi_eff swap touches pdraw after load; the writers re-check it."""
    from gwcat.selection import SelectionSet

    s = SelectionSet(_write_o4(tmp_path / "o4_ok.hdf", total_generated=1000))
    s._load()
    # A corrupt row that survives the load-time check only because it is
    # injected afterwards -- exactly what the swap could do.
    s._pdraw[0] = np.inf
    out = tmp_path / "sel.h5"
    with pytest.raises(ValueError) as ei:
        s.to_darksirens(str(out))
    assert "pdraw" in str(ei.value)
    assert not out.exists()


def test_a_clean_file_still_loads_and_exports(tmp_path):
    """The validation must not reject well-formed campaigns."""
    from gwcat.selection import SelectionSet, CombinedSelectionSet

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o4 = SelectionSet(str(write_o4_full(tmp_path / "ok_o4.hdf", n=200)))
        e3 = SelectionSet(str(write_endo3_full(tmp_path / "ok_e3.hdf", n=200)))
        o4._load()
        e3._load()
        assert np.all(np.isfinite(o4._pdraw)) and np.all(o4._pdraw > 0)
        assert np.all(np.isfinite(e3._pdraw)) and np.all(e3._pdraw > 0)
        out = tmp_path / "ok_comb.h5"
        CombinedSelectionSet([e3, o4]).to_darksirens(str(out))
    with h5py.File(out, "r") as f:
        pdraw = f["pdraw"][:]
    assert np.all(np.isfinite(pdraw)) and np.all(pdraw > 0)


# ==========================================================================
# GW-32 (#8): the compound events dataset is read in ONE pass
# ==========================================================================
def _to_compound_events(path):
    """Rewrite an ``events/`` GROUP as the ONE compound dataset O4 ships.

    The layout is what makes the field plan necessary: h5py serves
    ``dset["column"]`` on a compound dataset by reading the whole record, so a
    field-by-field load re-reads the file once per column.
    """
    with h5py.File(path, "r+") as f:
        g = f["events"]
        names = list(g.keys())
        n = g[names[0]].shape[0]
        arr = np.zeros(n, dtype=[(k, g[k].dtype) for k in names])
        for k in names:
            arr[k] = g[k][:]
        del f["events"]
        f.create_dataset("events", data=arr)
    return str(path)


def _load_arrays(s):
    """Everything a load produces, for a bit-for-bit comparison."""
    keys = ["_m1det", "_m2det", "_dL", "_chieff", "_ra", "_dec", "_m1src",
            "_m2src", "_z", "_pdraw", "_weights", "_a1", "_a2", "_cost1",
            "_cost2", "_chi_p", "_ln_spin_component", "_fars"]
    return {k: getattr(s, k) for k in keys}


def test_compound_events_planned_read_is_identical_to_field_by_field(tmp_path,
                                                                     monkeypatch):
    """The field plan may make the load faster; it may not change a number.

    Load the same compound file twice -- through :class:`_PlannedFields` and
    with the planning disabled, i.e. the pre-GW-32 column-at-a-time read -- and
    require every stored array, the FAR columns and the spin provenance to be
    bit-identical.
    """
    import gwcat.selection as S

    path = _to_compound_events(write_o4_full(tmp_path / "plan.hdf", n=400))

    planned = SelectionSet(path)
    planned._load()

    monkeypatch.setattr(S, "_planned_events_table", lambda table, f: table)
    direct = SelectionSet(path)
    direct._load()

    a, b = _load_arrays(planned), _load_arrays(direct)
    for k in a:
        if a[k] is None or b[k] is None:
            assert a[k] is None and b[k] is None, k
            continue
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]),
                                      err_msg=f"{k} moved under the field plan")
    assert planned._far_columns == direct._far_columns
    assert planned.spin_meta == direct.spin_meta
    assert planned._ndraw == direct._ndraw
    assert planned._T_yr == direct._T_yr


def test_compound_events_are_read_in_one_pass(tmp_path):
    """One sweep for the whole plan, not one whole-record read per column.

    Counting the string-keyed ``Dataset.__getitem__`` calls counts exactly the
    reads that cost a full pass over the 994-byte records; the plan must leave
    none of them, which is also the check that it names every column the loader
    asks for.
    """
    import gwcat.selection as S

    path = _to_compound_events(write_o4_full(tmp_path / "onepass.hdf", n=400))

    field_reads = []
    orig = h5py.Dataset.__getitem__

    def spy(self, args, *a, **kw):
        if isinstance(args, str):
            field_reads.append(args)
        return orig(self, args, *a, **kw)

    h5py.Dataset.__getitem__ = spy
    try:
        s = SelectionSet(path)
        s._load()
    finally:
        h5py.Dataset.__getitem__ = orig

    assert field_reads == [], (
        f"{len(field_reads)} column(s) fell outside the field plan and cost a "
        f"whole-record pass each: {sorted(set(field_reads))}")
    # ... and the plan named nothing the file does not have.
    with h5py.File(path, "r") as f:
        have = set(f["events"].dtype.names)
        plan = S._events_field_plan(f["events"], f)
    assert set(plan) <= have
    assert "pycbc_far" in plan          # the FAR column the search declares
    assert "lnpdraw_spin1_magnitude" in plan   # the component-spin provenance


def test_events_group_layout_is_left_alone(tmp_path):
    """An ``events/`` GROUP already reads one column per read: no plan, no wrap."""
    import gwcat.selection as S

    path = write_o4_full(tmp_path / "group.hdf", n=100)
    with h5py.File(path, "r") as f:
        ev = f["events"]
        assert S._planned_events_table(ev, f) is ev
