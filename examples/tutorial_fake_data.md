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

The fake store has one BBH event with no FAR. `--allow-missing-far` keeps it
(with a warning and explicit provenance); the default would silently drop it,
and `--require-far` would fail loudly instead.

```bash
gwcat export-darksirens ./fake_data/fake_store.h5 \
    --out ./fake_data/fake_gw_bbh.h5 \
    --source-class bbh \
    --far-max 1.0 \
    --allow-missing-far \
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
    --out ./fake_data/fake_selection_bbh.h5 \
    --source-class bbh \
    --H0 67.66 --Om0 0.3096
```

Uses the SAME cosmology and source-class filter as step 2, which is what lets
step 4's cross-validation pass.

## 4. Validate

```bash
gwcat validate ./fake_data/fake_gw_bbh.h5 ./fake_data/fake_selection_bbh.h5
```

Runs `gwcat.catalog.validate_export`'s internal checks (array lengths,
`p_pe`/`pdraw` finite and positive, source masses <= detector masses) plus the
PE <-> selection cross-validation (spin-prior mode, cosmology, source-class
agreement). Exits 0 when every check passes; the CLI's exit code is 1 if any
non-strict internal check fails, and a cross-validation contract mismatch
always raises (printed to stderr, exit 1) since a "looks valid but wrong
prior/cosmology/source-class" file is exactly what this step exists to catch.

## One-liner

```bash
python examples/make_fake_store.py --outdir ./fake_data \
  && gwcat inspect ./fake_data/fake_store.h5 \
  && gwcat export-darksirens ./fake_data/fake_store.h5 --out ./fake_data/fake_gw_bbh.h5 \
       --source-class bbh --far-max 1.0 --allow-missing-far \
       --cosmology 67.66,0.3096 --nsamp 256 --seed 0 \
  && gwcat selection --injections ./fake_data/fake_injections.hdf \
       --out ./fake_data/fake_selection_bbh.h5 --source-class bbh --H0 67.66 --Om0 0.3096 \
  && gwcat validate ./fake_data/fake_gw_bbh.h5 ./fake_data/fake_selection_bbh.h5
```
