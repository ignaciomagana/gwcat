# Full-data workflow: mixed BBH/NSBH/BNS ("all-CBC") export

**These commands download real GWTC/injection data from Zenodo and GWOSC and
are documented here for reference; they are NOT executed in the test suite
(no network access in tests).** For an offline, fully synthetic run of the
same command shapes, see `examples/tutorial_fake_data.md`.

The package does not build BBH/NSBH/BNS event lists from a static whitelist:
`source_class` is per-event metadata (mass-threshold classified at ingest, or
overridden via an `event_table`/`user_overrides` file -- see
`gwcat.event_metadata`), and `--source-class cbc` selects every compact-binary
class (BBH + NSBH + BNS + MassGap) at once.

```bash
# 1. Download PE files + the combined O3+O4 CBC injection set
gwcat fetch --catalog GWTC-2.1 GWTC-3 GWTC-4.1 GWTC-5 --data-dir ./GWTC
gwcat fetch --catalog injections-O3-BBH --data-dir ./injections
gwcat fetch --catalog injections-O4ab   --data-dir ./injections

# 2. Ingest ALL events (BBH/NSBH/BNS coexist in one store; tidal/BNS-only
#    parameters are NaN-filled -- not dropped -- for BBH events)
gwcat ingest --glob "./GWTC/GWTC2p1/*.h5" --glob "./GWTC/GWTC3/*.h5" \
             --glob "./GWTC/GWTC4p1/*.h5" --glob "./GWTC/GWTC5/*.h5" \
             --out store_cbc.h5

# 3. Inspect: per-class counts (BBH/NSBH/BNS/MassGap/Unknown), missing-FAR
#    count, and which optional (e.g. tidal) parameters are only partially
#    available.
gwcat inspect store_cbc.h5

# 4. Export every compact-binary class in one file
gwcat export-darksirens store_cbc.h5 \
    --out gw_cbc.h5 \
    --source-class cbc \
    --far-max 1.0 \
    --allow-missing-far \
    --waveform-policy mixed-first \
    --spin-prior-mode include \
    --cosmology 67.74,0.3089 \
    --nsamp 4096 --seed 0

# 5. Build the matching combined CBC selection export (no --source-class:
#    keeps every injected class, matching the PE export's "cbc" filter)
gwcat selection \
    --injections ./injections/injections-O3-BBH/endo3_bbhpop-LIGO-T2100113-v12.hdf5 \
                 ./injections/injections-O4ab/mixture-real_o4a_o4b-cartesian_spins.hdf \
    --out selection_cbc.h5 \
    --far-threshold 1.0 \
    --H0 67.74 --Om0 0.3089

# 6. Validate: the cross-check confirms the PE and selection files cover the
#    SAME source-class set (both "cbc" here) as well as matching spin-prior
#    mode and cosmology.
gwcat validate gw_cbc.h5 selection_cbc.h5
```

To restrict to a specific mix, e.g. NSBH + BNS only, pass
`--source-class nsbh,bns` (or run separate `nsbh`/`bns` exports); a
user-defined event list (`--event-list events.txt`) composes with
`--source-class` as an additional filter.

See `gwcat --help`, `gwcat <subcommand> --help` for the full flag reference.
