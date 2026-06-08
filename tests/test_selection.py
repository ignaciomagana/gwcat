import numpy as np
import h5py

from gwcat.selection import SelectionSet


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
