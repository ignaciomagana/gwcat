# Tutorial: the full `gwcat` CLI chain on fake data

This walkthrough runs entirely offline on tiny, synthetic data -- no Zenodo/
GWOSC access and no PESummary/bilby dependency required. It exercises the same
CLI surface as a real GWTC analysis (see `examples/bbh_workflow.md` and
`examples/all_cbc_workflow.md` for the full-data commands), just on a
hand-built fixture instead of downloaded PE files.

It is also exactly what `tests/test_cli.py` drives end-to-end (build the fake
store in-process, then `gwcat export-darksirens` -> `gwcat validate`), so this
tutorial and the test suite can never drift apart silently.

## 0. Make the fake data

`gwcat ingest` needs real PESummary-format PE files (it calls into
`pesummary.io.read`), which we don't have here. Instead,
`examples/make_fake_store.py` writes a tiny store.h5 directly in the schema
`GWCatalog` reads -- 4 fake events (2 BBH, 1 NSBH, 1 BNS), one of the BBH
events with **no FAR** (`far_available=False`), plus a tiny O4-format
injection file for the `selection` step.

```bash
python examples/make_fake_store.py --outdir ./fake_data
# Wrote ./fake_data/fake_store.h5: 4 fake events (2 BBH, 1 NSBH, 1 BNS), 1
#   event with far_available=False.
# Wrote ./fake_data/fake_injections.hdf: 800 fake injections (400 BBH, 200
#   NSBH, 200 BNS).
```

## 1. Inspect the store

```bash
gwcat inspect ./fake_data/fake_store.h5
```

Prints the compact event table (`GWCatalog.summary()`) plus diagnostics:
stored parameters, per-class event counts, waveform/approximant counts,
`far_missing_count` (1 of 4 here), and `p_astro_available_count`. Add
`--json` for the machine-readable form (the same dict `write_summary=True`
would have written to `validation_summary.json`).

## 2. Export darksirens PE samples (BBH only)

The two BBH events are selected by name. A `--source-class bbh` cut would
classify the events by their posterior-median masses and the injections by
their true masses, a pair `gwcat validate` refuses; a `--far-max` cut would
have to drop the BBH event with no FAR (`--allow-missing-far` keeps it, and
`gwcat validate` then refuses the pair, since the injections cannot model an
event that was never tested against the cut).

```bash
printf 'GW900101_000001\nGW900102_000002\n' > ./fake_data/bbh_events.txt
gwcat export-darksirens ./fake_data/fake_store.h5 \
    --out ./fake_data/fake_gw_bbh.h5 \
    --event-list ./fake_data/bbh_events.txt \
    --cosmology 67.66,0.3096 \
    --nsamp 256 --seed 0
```

Writes `fake_gw_bbh.h5` plus, by default, `fake_gw_bbh.h5.validation_summary.json`
/ `.md` next to it (pass `--no-summary` to skip). The summary records
`n_events_exported`, `source_class_counts_exported`, `far_policy`,
`n_events_missing_far`, `spin_prior_mode`, `cosmology_mode`, and more -- see
`README.md`'s "Validation summaries" section for the full key list.

## 3. Build the matching selection export

```bash
gwcat selection \
    --injections ./fake_data/fake_injections.hdf \
    --out ./fake_data/fake_selection_bbh.h5
```

No class cut (the population model handles the classes) and no `--H0`/`--Om0`:
the campaign's Jacobian uses its own cosmology. `gwcat validate` warns that a
name list meets a selection with no class cut, which is legal when the list's
membership does not come from the events' measured masses.

## 4. Validate

```bash
gwcat validate ./fake_data/fake_gw_bbh.h5 ./fake_data/fake_selection_bbh.h5
```

Runs `gwcat.catalog.validate_export`'s internal checks (array lengths,
`p_pe`/`pdraw` finite and positive, source masses <= detector masses) plus the
PE <-> selection cross-validation (spin-prior mode, cosmology, source-class
agreement). Exits 0 when every check passes and 1 otherwise. Most
cross-validation contract mismatches raise (printed to stderr, exit 1), since
a "looks valid but wrong prior/cosmology/source-class" file is exactly what
this step exists to catch; a few only warn unless `--strict` is given, so use
`--strict` when the pair gates an analysis.

## One-liner

```bash
python examples/make_fake_store.py --outdir ./fake_data \
  && gwcat inspect ./fake_data/fake_store.h5 \
  && printf 'GW900101_000001\nGW900102_000002\n' > ./fake_data/bbh_events.txt \
  && gwcat export-darksirens ./fake_data/fake_store.h5 --out ./fake_data/fake_gw_bbh.h5 \
       --event-list ./fake_data/bbh_events.txt \
       --cosmology 67.66,0.3096 --nsamp 256 --seed 0 \
  && gwcat selection --injections ./fake_data/fake_injections.hdf \
       --out ./fake_data/fake_selection_bbh.h5 \
  && gwcat validate --strict ./fake_data/fake_gw_bbh.h5 ./fake_data/fake_selection_bbh.h5
```
