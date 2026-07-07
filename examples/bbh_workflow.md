# Full-data workflow: BBH-only dark-siren export

**These commands download real GWTC/injection data from Zenodo and GWOSC and
are documented here for reference; they are NOT executed in the test suite
(no network access in tests).** For an offline, fully synthetic run of the
same command shapes, see `examples/tutorial_fake_data.md`.

```bash
# 1. Download PE files + injection sets
gwcat fetch --catalog GWTC-2.1 GWTC-3 GWTC-4.1 GWTC-5 --data-dir ./GWTC
gwcat fetch --catalog injections-O3-BBH --data-dir ./injections
gwcat fetch --catalog injections-O4ab   --data-dir ./injections

# 2. Ingest PE files into one store (FAR/p_astro auto-fetched from GWOSC)
gwcat ingest --glob "./GWTC/GWTC2p1/*.h5" --glob "./GWTC/GWTC3/*.h5" \
             --glob "./GWTC/GWTC4p1/*.h5" --glob "./GWTC/GWTC5/*.h5" \
             --out store.h5

# 3. Inspect the store: event/source-class/waveform counts, missing FAR, ...
gwcat inspect store.h5

# 4. Export BBH-only darksirens PE samples.
#    allow-missing-far is the practical default for public releases (some
#    events lack a machine-readable FAR); require-far instead fails loudly.
gwcat export-darksirens store.h5 \
    --out gw_bbh.h5 \
    --source-class bbh \
    --far-max 1.0 \
    --allow-missing-far \
    --waveform-policy mixed-first \
    --spin-prior-mode include \
    --cosmology 67.74,0.3089 \
    --nsamp 4096 --seed 0

# 5. Build the matching (combined O3+O4) BBH selection export
gwcat selection \
    --injections ./injections/injections-O3-BBH/endo3_bbhpop-LIGO-T2100113-v12.hdf5 \
                 ./injections/injections-O4ab/mixture-real_o4a_o4b-cartesian_spins.hdf \
    --out selection_bbh.h5 \
    --source-class bbh \
    --far-threshold 1.0 \
    --H0 67.74 --Om0 0.3089

# 6. Validate before running darksirens
gwcat validate gw_bbh.h5 selection_bbh.h5
```

Every `ingest`/`export-darksirens`/`selection` step above also writes
`<out>.validation_summary.json` / `.md` next to its output by default (pass
`--no-summary` to any of them to skip) -- see the README's "Validation
summaries" section for the full field list.

See `gwcat --help`, `gwcat <subcommand> --help` for the full flag reference,
and `README.md`'s "End-to-end" section for the equivalent Python API calls.
