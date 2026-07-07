#!/usr/bin/env python
"""Build a tiny, fully synthetic gwcat store + injection file for the
fake-data tutorial (see ``examples/tutorial_fake_data.md``).

No network access, no PESummary/bilby dependency: everything is written
directly with h5py, matching the on-disk schema ``GWCatalog``/``SelectionSet``
read (the same approach the test suite's fixtures use -- see e.g.
``tests/test_source_class_filters.py::build_mixed_store`` and
``tests/test_selection_cbc.py::write_o4``).

This intentionally bypasses ``gwcat ingest`` (which needs real PESummary-format
PE files): the fake PE "store.h5" here stands in for what ``gwcat ingest``
would have produced, so the tutorial can exercise the rest of the chain
(``inspect`` / ``export-darksirens`` / ``selection`` / ``validate``) fully
offline.

Usage
-----
    python examples/make_fake_store.py --outdir ./fake_data

Produces, under ``--outdir``:
    fake_store.h5        a tiny mixed BBH/NSBH/BNS posterior-sample store
    fake_injections.hdf  a tiny O4-format injection file (for `gwcat selection`)
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import h5py

#: Reference cosmology shared by the fake store and the fake injection file,
#: so the tutorial's `gwcat validate` step (which cross-checks cosmology
#: agreement between a PE export and a selection export) passes cleanly.
H0 = 67.66
OM0 = 0.3096

DARKSIRENS_PARAMS = ["mass_1", "mass_2", "luminosity_distance", "ra", "dec",
                     "chi_eff", "p_dL_pe"]

#: Four fake events: two BBH (one with FAR, one WITHOUT -- to exercise the
#: missing-FAR contract), one NSBH, one BNS. Names use a GW9xxxxx block so
#: they cannot be mistaken for a real GWTC event.
FAKE_EVENTS = [
    {"name": "GW900101_000001", "source_class": "BBH", "far": 5e-4,
     "pastro": 0.99, "waveform": "IMRPhenomXPHM"},
    {"name": "GW900102_000002", "source_class": "BBH", "far": float("nan"),
     "pastro": float("nan"), "waveform": "IMRPhenomXPHM"},  # FAR unavailable
    {"name": "GW900103_000003", "source_class": "NSBH", "far": 2e-3,
     "pastro": 0.95, "waveform": "SEOBNRv5PHM"},
    {"name": "GW900104_000004", "source_class": "BNS", "far": 1e-6,
     "pastro": 0.999, "waveform": "IMRPhenomXPHM"},
]


def build_fake_pe_store(out_path: str, n_per_event: int = 500,
                        seed: int = 0) -> str:
    """Write a tiny synthetic store.h5 with source-class + FAR + waveform
    metadata (schema GWCatalog reads)."""
    rng = np.random.default_rng(seed)
    class_masses = {
        "BBH": (35.0, 28.0), "NSBH": (12.0, 1.4), "BNS": (1.6, 1.3),
    }

    cols = {p: [] for p in DARKSIRENS_PARAMS}
    meta = {k: [] for k in [
        "source_class", "compact_type", "far", "far_available", "pastro",
        "p_astro", "dL_prior_H0", "dL_prior_Om0", "waveform", "approximant",
        "sample_set_name",
    ]}
    names, offsets = [], [0]

    for ev in FAKE_EVENTS:
        n = n_per_event
        m1c, m2c = class_masses[ev["source_class"]]
        cols["mass_1"].append(rng.normal(m1c, 1.5, n))
        cols["mass_2"].append(rng.normal(m2c, 0.5, n))
        cols["luminosity_distance"].append(rng.uniform(300, 900, n))
        cols["ra"].append(rng.uniform(0, 2 * np.pi, n))
        cols["dec"].append(rng.uniform(-np.pi / 2, np.pi / 2, n))
        cols["chi_eff"].append(rng.uniform(-0.3, 0.3, n))
        cols["p_dL_pe"].append(rng.uniform(0.1, 1.0, n))
        offsets.append(offsets[-1] + n)

        names.append(ev["name"])
        far = float(ev["far"])
        meta["source_class"].append(ev["source_class"])
        meta["compact_type"].append(ev["source_class"])
        meta["far"].append(far)
        meta["far_available"].append(1.0 if np.isfinite(far) else 0.0)
        meta["pastro"].append(float(ev["pastro"]))
        meta["p_astro"].append(float(ev["pastro"]))
        meta["dL_prior_H0"].append(H0)
        meta["dL_prior_Om0"].append(OM0)
        meta["waveform"].append(ev["waveform"])
        meta["approximant"].append(ev["waveform"])
        meta["sample_set_name"].append(f"C01:{ev['waveform']}")

    with h5py.File(out_path, "w") as f:
        f.attrs["schema_version"] = "1.2"
        f.attrs["param_names"] = np.array(DARKSIRENS_PARAMS,
                                          dtype=h5py.string_dtype())
        f.attrs["n_events"] = len(names)
        idx = f.create_group("index")
        idx.create_dataset("offsets", data=np.array(offsets, dtype="i8"))
        idx.create_dataset("event_names",
                           data=np.array(names, dtype=h5py.string_dtype()))
        ag = f.create_group("avail")
        ag.create_dataset("mask", data=np.ones((len(names), len(DARKSIRENS_PARAMS)),
                                               dtype=bool))
        mg = f.create_group("meta")
        for k in ["source_class", "compact_type", "waveform", "approximant",
                  "sample_set_name"]:
            mg.create_dataset(k, data=np.array(meta[k], dtype=h5py.string_dtype()))
        for k in ["far", "far_available", "pastro", "p_astro",
                  "dL_prior_H0", "dL_prior_Om0"]:
            mg.create_dataset(k, data=np.asarray(meta[k], dtype="f8"))
        sg = f.create_group("samples")
        for p in DARKSIRENS_PARAMS:
            sg.create_dataset(p, data=np.concatenate(cols[p]))

    print(f"Wrote {out_path}: {len(names)} fake events "
          f"({sum(1 for e in FAKE_EVENTS if e['source_class'] == 'BBH')} BBH, "
          "1 NSBH, 1 BNS), 1 event with far_available=False.")
    return out_path


_O4_FIELDS = [
    ("mass1_source", "f8"), ("mass2_source", "f8"),
    ("mass1_detector", "f8"), ("mass2_detector", "f8"),
    ("luminosity_distance", "f8"), ("z", "f8"),
    ("dluminosity_distance_dredshift", "f8"),
    ("right_ascension", "f8"), ("declination", "f8"),
    ("spin1x", "f8"), ("spin1y", "f8"), ("spin1z", "f8"),
    ("spin2x", "f8"), ("spin2y", "f8"), ("spin2z", "f8"),
    ("chi_eff", "f8"), ("weights", "f8"),
    ("lnpdraw_mass1_source", "f8"),
    ("lnpdraw_mass2_source_GIVEN_mass1_source", "f8"),
    ("lnpdraw_z", "f8"),
    ("pycbc_far", "f8"),
]


def build_fake_injection_file(out_path: str, n_bbh: int = 400,
                              n_other: int = 200,
                              total_generated: int = 100_000) -> str:
    """Write a tiny O4-'events'-format injection file with a BBH + NSBH/BNS
    mix, so `gwcat selection --source-class bbh` has something to filter."""
    rows = ([("BBH", 32.0, 26.0)] * n_bbh
           + [("NSBH", 11.0, 1.4)] * n_other
           + [("BNS", 1.5, 1.3)] * n_other)
    n = len(rows)
    ev = np.zeros(n, dtype=_O4_FIELDS)
    z = 0.15
    for i, (_cls, m1s, m2s) in enumerate(rows):
        ev["mass1_source"][i] = m1s
        ev["mass2_source"][i] = m2s
        ev["z"][i] = z
        ev["mass1_detector"][i] = m1s * (1 + z)
        ev["mass2_detector"][i] = m2s * (1 + z)
        ev["luminosity_distance"][i] = 700.0
        ev["dluminosity_distance_dredshift"][i] = 4200.0
        ev["right_ascension"][i] = 1.2
        ev["declination"][i] = 0.3
        ev["spin1z"][i] = 0.15
        ev["spin2z"][i] = -0.05
        ev["chi_eff"][i] = (m1s * 0.15 + m2s * -0.05) / (m1s + m2s)
        ev["weights"][i] = 1.0
        ev["lnpdraw_mass1_source"][i] = -1.0
        ev["lnpdraw_mass2_source_GIVEN_mass1_source"][i] = -2.0
        ev["lnpdraw_z"][i] = -3.0
    ev["pycbc_far"] = 0.1  # every injection "detected" (FAR < 1/yr)

    with h5py.File(out_path, "w") as f:
        f.attrs["total_analysis_time"] = 365.25 * 24 * 3600
        f.attrs["total_generated"] = total_generated
        f.attrs["searches"] = np.array([b"pycbc"])
        f.create_dataset("events", data=ev)

    print(f"Wrote {out_path}: {n} fake injections "
          f"({n_bbh} BBH, {n_other} NSBH, {n_other} BNS).")
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="./fake_data",
                    help="Directory to write fake_store.h5 / "
                         "fake_injections.hdf into (default: ./fake_data)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    build_fake_pe_store(os.path.join(args.outdir, "fake_store.h5"),
                       seed=args.seed)
    build_fake_injection_file(os.path.join(args.outdir, "fake_injections.hdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
