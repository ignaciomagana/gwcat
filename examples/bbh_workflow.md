# Full-data workflow: GWTC-5 BBH/mass-gap population export

**These commands use real GWTC PE and injection products and are documented
for reference; they are not executed in the no-network test suite.** For an
offline synthetic workflow, see `examples/tutorial_fake_data.md`.

The public PE directories contain more files than the 259-event GWTC-5
BBH/mass-gap population sample. Population membership is defined by the bundled
canonical event list, not by counting raw files and not by reapplying the
generic source-mass classifier after the name selection. In particular,
`GW190814_211039` is intentionally retained by the population list.

```bash
# 1. Download PE files and injection sets.
gwcat fetch --catalog GWTC-2.1 GWTC-3 GWTC-4.1 GWTC-5 --data-dir ./GWTC
gwcat fetch --catalog injections-O3-BBH --data-dir ./injections
gwcat fetch --catalog injections-O4ab   --data-dir ./injections

# 2. Ingest all released PE files into one store.
gwcat ingest \
    --glob "./GWTC/GWTC-2p1/*.h5" \
    --glob "./GWTC/GWTC-3/*.h5" \
    --glob "./GWTC/GWTC-4p1/*.h5" \
    --glob "./GWTC/GWTC-5/*.h5" \
    --out store.h5

# 3. Inspect raw event/source-class/waveform counts.
gwcat inspect store.h5

# 4. Resolve and write the authoritative 259-event population list against
#    the local store. Historical date-only O1/O2 aliases are handled safely.
python - <<'PY'
from pathlib import Path
from gwcat import GWCatalog, resolve_gwtc5_bbh_population_names

cat = GWCatalog("store.h5")
names, aliases = resolve_gwtc5_bbh_population_names(cat.names)
Path("gwtc5_bbh_population.txt").write_text("\n".join(names) + "\n")
print(f"resolved {len(names)} population events")
for old, new in aliases.items():
    print(f"{old} -> {new}")
PY

# 5. Export PE samples using authoritative event membership. Do not add
#    --source-class bbh here: the generic mass classifier would remove
#    GW190814_211039 from the intended population sample.
gwcat export-darksirens store.h5 \
    --out gw_gwtc5_population.h5 \
    --event-list gwtc5_bbh_population.txt \
    --waveform-policy mixed-first \
    --spin-prior-mode include \
    --cosmology 67.74,0.3089 \
    --nsamp 4096 --seed 0

# 6. Build the matching O3+O4ab selection export. Retain all detected CBC
#    injection support; the downstream population density assigns zero weight
#    outside its modeled BBH/mass-gap support.
gwcat selection \
    --injections ./injections/injections-O3-BBH/endo3_bbhpop-LIGO-T2100113-v12.hdf5 \
                 ./injections/injections-O4ab/samples-rpo4ab-1366933504-55469568-clipped.hdf \
    --out selection_o3o4ab_allsky.h5 \
    --far-threshold 1.0 \
    --H0 67.74 --Om0 0.3089

# 7. Validate before population inference.
gwcat validate gw_gwtc5_population.h5 selection_o3o4ab_allsky.h5
```

For a strict mass-classified BBH sample instead, use `--source-class bbh` on
both PE and selection exports. That is a different sample and should not be
called the authoritative 259-event GWTC-5 population list.

Every `ingest`, `export-darksirens`, and `selection` step writes
`<out>.validation_summary.json` and `.md` by default. Pass `--no-summary` to
skip those sidecars.
