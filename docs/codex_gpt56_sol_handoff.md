# `gwcat` handoff after PR11

## Package identity

`gwcat` is a CBC posterior-sample and selection-product release wrangler. It
downloads public release products, ingests heterogeneous PESummary posterior
files into a schema-preserving store, filters and exports PE samples, builds
selection products from injection campaigns, and validates the cross-file
scientific contract.

`gwrangler` remains a possible future package/console-script name. Renaming is
not part of PR11; the unified CLI keeps its delegated program name derived from
the invoked entry point so a later alias or rename is localized.

## Current architecture

- `gwcat.fetch`: release-manifest lookup, Zenodo discovery/download, GWOSC
  event-metadata discovery, cache/offline behavior, and opt-in file provenance.
- `gwcat.ingest`: PESummary reading, waveform/sample-set selection, union-schema
  store construction, PE-prior handling, metadata ingestion, and store merging.
- `gwcat.catalog`: `GWCatalog` views, event/sample filtering, darksirens export,
  and PE/selection export validation.
- `gwcat.selection`: single- and multi-campaign injection selection products.
- `gwcat.schema`: canonical required/optional export parameter contracts.
- `gwcat.manifests`: declarative release and injection manifest loading and
  validation.
- `gwcat.event_metadata`: YAML/CSV overrides, layered metadata assembly, and
  per-event/per-field diagnostics.
- `gwcat.validation_summary`: JSON/Markdown catalog and export summaries.
- `gwcat.cli`: the unified `gwcat` command dispatcher and thin translations to
  library APIs. Scientific logic remains in the modules above.

## Scientific contracts already implemented

- Source class: canonical BBH, NSBH, BNS, MassGap, and CBC handling is shared
  across PE and selection filtering. Ingest records class, method, reference,
  and component probabilities where available.
- Missing FAR: absence is explicit through `far_available`; callers must choose
  whether FAR-filtered exports allow or reject events without FAR.
- Waveform/sample set: each row records sample-set name, approximant/family,
  mixed/preferred flags, priority, selection reason, and source-file identity.
  Export policies make the chosen sample set explicit.
- Prior/spin prior: distance-PE prior provenance and the include/exclude spin
  prior mode are recorded and cross-validated, preventing silent double
  counting or omission.
- Per-event cosmology: each event stores the cosmology used for its PE distance
  prior; export can preserve it or apply an explicit override.
- Union parameter schema: parameters present in only some events are retained,
  NaN-filled elsewhere, and accompanied by an availability mask.
- Selection/PE cross-validation: cosmology, source-class, and spin-prior
  contracts are checked across exports; contract mismatches are hard failures.
- Validation summaries: ingest and export commands write matching JSON and
  Markdown summaries by default, with `--no-summary` as an explicit opt-out.

## What PR11 adds

- Delegated help uses the unified identity: `gwcat fetch --help` reports
  `usage: gwcat fetch`, and `gwcat ingest --help` reports
  `usage: gwcat ingest`. The standalone scripts retain their old identities
  and deprecation warnings.
- `gwcat ingest` and `gwcat fetch --out` accept YAML/CSV metadata overrides via
  `--metadata-overrides`. They can write diagnostics to an explicit
  `--metadata-diagnostics` path; with overrides, the default is
  `<out>.metadata_diagnostics.json`.
- Override-related validation summaries record `metadata_overrides_path`,
  `metadata_diagnostics_path`, and the number of loaded event override records.
- Metadata overrides can supply source class, FAR, p_astro/component
  probabilities, release, and observing run. `--no-event-table` applies only
  the user overrides and never calls GWOSC.
- `gwcat fetch` can opt into sha256/record provenance with
  `--write-file-provenance`. `--file-provenance PATH` selects the JSON path and
  implies collection; otherwise a build writes `<out>.file_provenance.json`.
  The same mapping is passed into the store builder. Dry runs never write it.
- `gwcat validate ... --json` produces machine-readable results and returns 0
  only when every check passes. Cross-file exceptions become structured JSON
  failures with exit status 1.

## Recommended real-data commands

```bash
python -m pip install -e ".[dev]"

gwcat fetch --catalog GWTC-2.1 GWTC-3 GWTC-4.1 GWTC-5 \
  --data-dir ./GWTC \
  --cache-dir ./GWTC/.cache

gwcat ingest \
  --glob "./GWTC/GWTC2p1/*.h5" \
  --glob "./GWTC/GWTC3/*.h5" \
  --glob "./GWTC/GWTC4p1/*.h5" \
  --glob "./GWTC/GWTC5/*.h5" \
  --out store.h5 \
  --cache-dir ./GWTC/.cache \
  --metadata-overrides metadata_overrides.yaml

gwcat inspect store.h5

gwcat export-darksirens store.h5 \
  --out gw_bbh.h5 \
  --source-class bbh \
  --far-max 1.0 \
  --allow-missing-far \
  --waveform-policy mixed-first \
  --spin-prior-mode include \
  --cosmology 67.74,0.3089 \
  --nsamp 4096 \
  --seed 0

gwcat selection \
  --injections ./injections/injections-O3-BBH/endo3_bbhpop-LIGO-T2100113-v12.hdf5 \
               ./injections/injections-O4ab/mixture-real_o4a_o4b-cartesian_spins.hdf \
  --out selection_bbh.h5 \
  --source-class bbh \
  --far-threshold 1.0 \
  --H0 67.74 \
  --Om0 0.3089

gwcat validate gw_bbh.h5 selection_bbh.h5
```

For a one-step download/build with both new audit artifacts enabled:

```bash
gwcat fetch --catalog GWTC-2.1 GWTC-3 GWTC-4.1 GWTC-5 \
  --data-dir ./GWTC \
  --cache-dir ./GWTC/.cache \
  --out store.h5 \
  --metadata-overrides metadata_overrides.yaml \
  --write-file-provenance
```

## Operational notes and future work

- Default Python API behavior is unchanged: metadata auto-fetch remains driven
  by `event_table=None`, validation summaries remain library-level opt-in, and
  file hashing remains opt-in.
- `gwcat-fetch` and `gwcat-ingest` remain available for compatibility and still
  print their deprecation warnings.
- The CLI tests are synthetic and no-network. PR11 does not claim that live,
  full GWTC downloads or full public PESummary files were exercised.
- A future rename can add a `gwrangler` entry point and packaging migration;
  it should not be mixed into scientific or storage changes.
