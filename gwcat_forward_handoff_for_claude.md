# `gwcat` / GW Release Wrangler Handoff for Claude

## Executive Summary

We want to develop `ignaciomagana/gwcat` into a robust, extensible gravitational-wave release wrangling package.

The current name `gwcat` is too vague. The package is not merely a catalog. It should become a **CBC posterior-sample and selection-product release engine**: download public LVK/GWOSC/Zenodo release products, ingest heterogeneous PE files, normalize metadata and priors, preserve waveform provenance, handle BBH/NSBH/BNS samples, and export validated HDF5 products for downstream population inference and dark-siren analyses.

The immediate goal is **not** to add a large feature pile. The immediate goal is to define and test the scientific contracts so future GWTC releases can be added safely.

Best name candidate: **`gwrangler`**.

Other acceptable names:

| Name | Comment |
|---|---|
| `gwrangler` | Best overall. Says what the package does: wrangle annoying GW release products. |
| `gwtcflow` | Good if the package remains GWTC-release-specific. |
| `peprep` | Good if the identity is PE sample normalization. |
| `sirenprep` | Good if the main target is dark-siren workflows. |
| `gwrelease` | Accurate but boring and broad. |

Avoid names that imply this is only a source catalog. The useful abstraction is release-product normalization, not a table of events.

## Desired Package Identity

The package should eventually answer:

> Given a public GW release or event list, fetch all available posterior samples and selection products, preserve the release and waveform provenance, normalize priors and metadata, and export science-ready files with explicit contracts.

It should support:

- BBH posterior samples
- NSBH posterior samples
- BNS posterior samples
- mixed CBC catalogs
- user-defined event lists
- release-defined event lists
- public online metadata when available
- local release archives when online metadata are incomplete
- multiple waveform/sample sets per event
- prior provenance and prior reweighting state
- darksirens-compatible exports
- generic HDF5 exports for population inference

Do **not** build the package around a BBH whitelist. BBH-only should be one mode, not the package identity.

## Important Context from the Audit

The repo already has the right rough shape:

| Module | Current Role |
|---|---|
| `fetch.py` | Release registry/downloader for PE and injection products. |
| `ingest.py` | Converts PESummary-style HDF5 into a ragged posterior-sample store. |
| `catalog.py` | Query/export layer with metadata cuts and darksirens export. |
| `selection.py` | Normalizes O3/O4 injection products into a selection file contract. |
| tests | Currently too thin; mostly covers one O4 injection bug. |

The architecture is promising, but the scientific contracts are still unstable:

- spin-prior handling is inconsistent between docs and code
- per-event PE cosmology can be lost during export
- ingest/merge can silently drop parameters by intersection
- source-type handling is too BBH-centered
- waveform/sample-set consistency is not explicit enough
- online event metadata cannot be trusted to expose everything needed, especially FAR
- the public API still exposes private methods
- release definitions are hardcoded in Python rather than declarative manifests

## Core Scientific Contracts to Settle First

### 1. Source Class Contract

The package must support at least these source classes:

```text
BBH
NSBH
BNS
MassGap / ambiguous
Unknown / unclassified
```

Do not infer source class from a static event-name list alone.

Preferred source-class metadata model:

```text
event_name
release
observing_run
source_class
source_class_method
source_class_reference
p_astro
p_bbh
p_nsbh
p_bns
p_terr
far
far_available
metadata_source
```

The key point: `far_available=False` must be a valid state. Some online/public interfaces may expose `p_astro`, preferred sample links, or event pages without exposing FAR in a machine-readable way. The package should degrade gracefully and record that absence rather than pretending FAR exists.

Selection modes should be explicit:

```text
--source-class bbh
--source-class nsbh
--source-class bns
--source-class cbc
--event-list events.txt
--min-p-astro 0.5
--max-far 1/yr
--allow-missing-far
--require-far
```

For public release workflows, `--allow-missing-far` should often be the practical default, with a warning and provenance attribute.

### 2. Posterior Sample-Set / Waveform Contract

Each event can have multiple PE sample sets. The package must not collapse them without recording what happened.

Minimum metadata per sample set:

```text
event_name
sample_set_name
waveform
approximant
calibration_model
release
record_id
file_name
file_checksum
is_mixed
is_preferred
priority_rank
selection_reason
available_parameters
sample_count
```

Waveform selection should be configurable:

```text
--waveform-policy preferred
--waveform-policy mixed-first
--waveform-policy strict-approximant
--waveform-policy all
--approximant IMRPhenomXPHM
--approximant SEOBNRv5PHM
```

Important behavior:

- `preferred` uses release metadata or a manifest priority list.
- `mixed-first` prefers mixed posterior samples when available.
- `strict-approximant` requires the same approximant across selected events and fails loudly when unavailable.
- `all` ingests all sample sets and lets downstream code decide.

For waveform consistency studies, the package should be able to produce:

```text
one event x many waveform sample sets
many events x one chosen waveform policy
many events x all available sample sets
```

Do not silently mix waveform families and call the result homogeneous.

### 3. Prior Contract

The package must explicitly track what prior is already included in every exported weight.

Minimum prior metadata:

```text
distance_prior
mass_prior_basis
spin_prior
redshift_prior
cosmology_for_source_frame_masses
jacobian_state
spin_prior_state
p_pe_state
pdraw_state
```

The most dangerous current issue is `chi_eff` prior handling. The docs suggest one contract while the code appears to do another.

Choose one of these:

```text
Mode A:
  gwcat/gwrangler includes the chi_eff prior in exported p_pe/pdraw.
  darksirens must not multiply it again.

Mode B:
  gwcat/gwrangler exports spin-prior-free p_pe/pdraw.
  darksirens applies the chi_eff prior.
```

The better long-term design is to make this a required explicit option:

```text
--spin-prior-mode include
--spin-prior-mode exclude
--spin-prior-mode passthrough
```

Every output file must contain attributes making this impossible to miss:

```text
spin_prior_mode
chi_eff_prior_applied_to_p_pe
chi_eff_prior_applied_to_pdraw
mass_jacobian_applied
distance_prior_removed
cosmology_override_used
```

Tests must verify that priors are included or excluded exactly once.

### 4. Cosmology Contract

For PE samples, the cosmology used to infer source-frame masses and redshifts can differ by release or sample set.

When `cosmology=None`, export must use each event's stored PE cosmology independently.

When the user passes a cosmology override, export must:

- use that override consistently
- write the override into output attributes
- mark that source-frame quantities were recomputed or interpreted under the override

Do not take the first selected event's cosmology and apply it to all events.

### 5. Parameter Schema Contract

The posterior store should not shrink to the intersection of parameters present in all events.

Instead:

```text
required parameters:
  fail loudly if missing for a requested export

optional parameters:
  store if present
  fill missing event slices with NaN
  keep availability masks
```

This matters for future releases and for BNS/NSBH, where tidal parameters, spins, waveform choices, and EOS-related columns may differ.

Suggested parameter groups:

```text
core_intrinsic:
  mass_1, mass_2, mass_ratio, chirp_mass

core_extrinsic:
  luminosity_distance, redshift, ra, dec, theta_jn, psi

spin:
  a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, chi_eff, chi_p

bns_nsbh:
  lambda_1, lambda_2, lambda_tilde, delta_lambda_tilde

diagnostic:
  log_likelihood, log_prior, weights
```

Exports should declare which groups they require.

## Online Data Strategy

The package should fetch what it can online, but it must not assume online metadata are complete.

Recommended model:

```text
Release manifest = source of expected products and policy
Online services   = source of downloadable files and supplemental metadata
Local cache       = source of reproducibility
Output HDF5 attrs = source of provenance for downstream science
```

Important online limitations:

- FAR may not be available from the easiest public endpoint.
- Some event pages expose different metadata than machine-readable APIs.
- Preferred waveform/sample-set choices may live in release notes or paper tables, not clean API fields.
- Zenodo records can contain multiple files with naming conventions rather than structured metadata.
- GWOSC metadata may lag behind release products or omit fields needed for population analyses.

Therefore, build a layered metadata system:

```text
1. Declarative release manifest bundled with the package.
2. Online discovery/fetch of files and event metadata.
3. User override files for event lists, source class, waveform policy, or FAR cuts.
4. Validation summary showing missing or conflicting fields.
```

Future release support should be mostly:

```text
add manifests/releases/gwtc-6.yaml
add tests/fixtures/gwtc6_tiny_manifest.yaml
run manifest validation
```

not:

```text
edit downloader internals
edit BBH list in Python
edit hardcoded file filters
patch darksirens export by hand
```

## Declarative Manifest Design

Move release metadata out of Python and into files such as:

```text
src/gwrangler/manifests/releases/gwtc-2p1.yaml
src/gwrangler/manifests/releases/gwtc-3.yaml
src/gwrangler/manifests/releases/gwtc-4p1.yaml
src/gwrangler/manifests/releases/gwtc-5.yaml
src/gwrangler/manifests/injections/o3-bbh.yaml
src/gwrangler/manifests/injections/o4ab-cbc.yaml
```

Each release manifest should include:

```yaml
release: GWTC-5
observing_runs: [O4b]
description: Public GWTC-5 posterior samples
records:
  - provider: zenodo
    record_id: "..."
    concept_id: "..."
products:
  pe_samples:
    file_patterns:
      - "*.h5"
    expected_event_count: null
    sample_set_policy: mixed-first
metadata:
  source_class_reference: release_table
  far_available_online: false
  preferred_waveform_source: manifest
validation:
  require_checksums: true
  allow_missing_far: true
```

Event lists should be data files, not Python literals:

```text
src/gwrangler/data/event_lists/gwtc5_bbh.txt
src/gwrangler/data/event_lists/gwtc5_nsbh.txt
src/gwrangler/data/event_lists/gwtc5_bns.txt
src/gwrangler/data/event_lists/gwtc5_cbc_all.txt
src/gwrangler/data/event_lists/non_bbh_exclusions.txt
src/gwrangler/data/event_lists/provenance.yaml
```

## Proposed Public API

The README should not ask users to call private methods.

Replace:

```python
cat._to_darksirens_format(...)
```

with:

```python
cat.to_darksirens(...)
```

Suggested Python API:

```python
from gwrangler import Release, PosteriorStore, SelectionSet

release = Release.from_manifest("gwtc-5")
release.fetch(cache_dir="data/raw")

store = PosteriorStore.ingest(
    release,
    waveform_policy="mixed-first",
    source_class="cbc",
)

store.validate()
store.to_hdf5("posterior_store.h5")

cat = PosteriorStore.open("posterior_store.h5")
cat.to_darksirens(
    out="gw_cbc_darksirens.h5",
    source_class="bbh",
    spin_prior_mode="include",
    cosmology="per-event",
)
```

Suggested CLI:

```bash
gwrangle fetch --release gwtc-5 --cache data/raw
gwrangle ingest --release gwtc-5 --cache data/raw --out posterior_store.h5 --waveform-policy mixed-first --source-class cbc
gwrangle inspect posterior_store.h5
gwrangle export-darksirens posterior_store.h5 --source-class bbh --out gw_bbh.h5 --spin-prior-mode include
gwrangle selection --manifest injections/o4ab-cbc --out selection_o4ab.h5
gwrangle validate gw_bbh.h5 selection_o4ab.h5
```

## Validation Outputs

Every ingest/export should produce a machine-readable and human-readable validation summary:

```text
validation_summary.json
validation_summary.md
```

The summary should include:

- number of events discovered
- number of events ingested
- number skipped and why
- source-class counts
- sample-set counts per event
- waveform/approximant counts
- missing required parameters
- missing optional parameters
- missing FAR status
- p_astro availability
- prior mode used
- cosmology mode used
- output schema version
- package version
- release manifest version
- source file checksums

This is especially important because public online metadata are imperfect.

## Test Suite That Should Exist

Current tests are not enough. Add contract tests before large feature work.

Minimum test files:

```text
tests/test_manifest_loading.py
tests/test_fetch_registry.py
tests/test_posterior_schema.py
tests/test_ingest_pesummary_tiny.py
tests/test_catalog_selection.py
tests/test_waveform_policy.py
tests/test_source_class_filters.py
tests/test_export_darksirens.py
tests/test_selection_o3.py
tests/test_selection_o4.py
tests/test_combined_selection.py
tests/test_cosmology.py
tests/test_prior_contract.py
tests/test_validation_summary.py
```

High-priority test cases:

```text
1. Missing FAR is represented explicitly and does not crash source-class selection when allow_missing_far=True.
2. require_far=True fails when FAR is missing.
3. BBH/NSBH/BNS event classes can coexist in one posterior store.
4. waveform-policy=strict-approximant fails if one selected event lacks the requested approximant.
5. waveform-policy=all stores multiple sample sets for one event.
6. Optional BNS tidal parameters are NaN-filled for BBH events, not dropped globally.
7. Required darksirens export columns fail loudly if missing.
8. chi_eff prior is applied exactly once in include mode.
9. chi_eff prior is not applied in exclude mode.
10. per-event cosmology is respected when cosmology=None.
11. cosmology override writes explicit output provenance.
12. merging stores preserves schema and does not silently drop columns.
13. z_of_dL does not silently clip samples beyond interpolation range.
14. validation_summary.md and validation_summary.json record skipped events and missing metadata.
```

Use tiny synthetic HDF5 fixtures. Do not require downloading full GWTC products in unit tests.

## Recommended PR Sequence

### PR 1: Rename Decision and Public API Stabilization

Goal:

Decide whether to rename to `gwrangler`. Regardless of final name, create public API wrappers and deprecate private API usage.

Tasks:

- Add public `to_darksirens()` method.
- Keep `_to_darksirens_format()` as a deprecated alias.
- Add CLI skeleton if not already present.
- Update README examples away from private methods.
- Decide package name or create a transition plan.

Acceptance:

- Existing workflows still run.
- README uses only public API.
- Tests cover the public export method.

### PR 2: Source-Class Generalization

Goal:

Make BBH, NSBH, BNS, and mixed CBC catalogs first-class.

Tasks:

- Add source-class metadata fields.
- Move hardcoded BBH lists into data files.
- Add source-class filters.
- Add user event-list support.
- Support missing FAR with explicit provenance.

Acceptance:

- A toy mixed catalog can select BBH only, NSBH only, BNS only, or all CBC.
- `require_far=True` fails when FAR is missing.
- `allow_missing_far=True` records missing FAR and continues.

### PR 3: Prior Contract

Goal:

Make prior handling impossible to double-count silently.

Tasks:

- Add explicit `spin_prior_mode`.
- Add output attrs for prior state.
- Align PE export and selection export behavior.
- Update docs to match code.

Acceptance:

- Tests prove `chi_eff` prior is included/excluded exactly once.
- Output HDF5 records the prior mode.

### PR 4: Per-Event Cosmology

Goal:

Respect PE cosmology per event unless user overrides it.

Tasks:

- Fix export to use event-level stored cosmology when `cosmology=None`.
- Add explicit override mode.
- Write cosmology provenance into output.

Acceptance:

- Two-event fixture with different cosmologies exports correctly.
- Override test proves global override is intentional.

### PR 5: Schema-Preserving Ingest and Merge

Goal:

Stop silently dropping parameters by intersection.

Tasks:

- Define required and optional parameter groups.
- Store optional missing columns as NaN.
- Add availability masks.
- Fix merge behavior to preserve schema.

Acceptance:

- Mixed BBH/BNS fixture preserves tidal columns with NaN for BBH.
- Required export columns fail loudly if absent.

### PR 6: Waveform and Sample-Set Policy

Goal:

Make waveform consistency explicit and configurable.

Tasks:

- Store multiple sample sets per event.
- Add waveform policy machinery.
- Add strict approximant mode.
- Record chosen sample set and reason.

Acceptance:

- One event with two waveform sample sets can be ingested as `all`.
- `mixed-first` picks mixed samples when present.
- `strict-approximant` fails if consistency cannot be satisfied.

### PR 7: Declarative Release Manifests

Goal:

Move release metadata out of Python internals.

Tasks:

- Add YAML/JSON manifest loader.
- Port existing release registry into manifests.
- Add validation for manifests.
- Include expected products and metadata limitations.

Acceptance:

- Existing supported releases load from manifests.
- Adding a tiny fake release manifest requires no downloader code changes.

### PR 8: Fetch / Online Metadata Hardening

Goal:

Make online fetching useful but honest about missing metadata.

Tasks:

- Separate file discovery from event metadata discovery.
- Cache raw metadata responses.
- Add offline mode from local cache.
- Add missing-field diagnostics.

Acceptance:

- Fetch works when FAR is missing.
- Validation summary records which fields came from online metadata, manifest, or user override.

### PR 9: Selection Products for All CBC

Goal:

Normalize injections for BBH/NSBH/BNS/CBC workflows.

Tasks:

- Generalize selection products beyond BBH.
- Track pdraw/prior state.
- Track search/significance metadata when available.
- Keep O3/O4 format handling tested.

Acceptance:

- Toy O3/O4 selection fixtures export with explicit prior and source-class metadata.
- Combined selection file validates against posterior export.

### PR 10: End-to-End Validated Workflows

Goal:

Provide real user-facing workflows.

Tasks:

- Add `gwrangle fetch/ingest/inspect/export/selection/validate`.
- Add example workflows for BBH-only and all-CBC.
- Add validation reports.
- Add small fake data tutorial.

Acceptance:

- One command sequence goes from manifest to validated export on fixture data.
- Full data workflows are documented but not required in unit tests.

## Detailed First Codex Prompt

Use this prompt for the next coding agent if the goal is PR 1:

```text
You are working in ignaciomagana/gwcat. First inspect the repository structure, tests, README, and package metadata. Do not download public GW data. Make the smallest PR that stabilizes the public API for darksirens export.

Goals:
1. Add a public `to_darksirens(...)` method that calls the existing private exporter.
2. Keep `_to_darksirens_format(...)` as a deprecated compatibility alias.
3. Update README examples to use the public method.
4. Add or update tests to cover the public method on tiny synthetic data.
5. Do not change scientific behavior in this PR.

Report:
- changed files
- test command run
- any behavior intentionally left unchanged
- follow-up issues discovered
```

Use this prompt if the goal is PR 2:

```text
You are working in ignaciomagana/gwcat. Implement source-class generalization without downloading public data.

Goals:
1. Introduce source-class metadata supporting BBH, NSBH, BNS, MassGap/ambiguous, and Unknown.
2. Move hardcoded BBH event lists into data files if they are currently embedded in Python.
3. Add filtering by source class and user event-list file.
4. Represent missing FAR explicitly with `far_available=False`.
5. Add `allow_missing_far` and `require_far` behavior where selection/filtering uses FAR.
6. Add tests using tiny fixtures for mixed BBH/NSBH/BNS catalogs.

Do not make online FAR access a required dependency. Public online metadata may not expose FAR reliably. The package should record missing FAR and continue when configured.
```

Use this prompt if the goal is PR 3:

```text
You are working in ignaciomagana/gwcat. Fix the prior contract around chi_eff/spin priors.

Goals:
1. Audit where PE export and selection export include or exclude spin priors.
2. Add an explicit `spin_prior_mode` option with at least `include`, `exclude`, and `passthrough` if passthrough is useful.
3. Align code, README, and HDF5 attrs.
4. Add tests proving the chi_eff prior is applied exactly once in include mode and not applied in exclude mode.
5. Do not silently change defaults without documenting the migration.

The central risk is double-counting or omitting spin priors in darksirens workflows.
```

## Non-Negotiable Rules for Claude

1. Do not silently change scientific weights.
2. Do not weaken tests to make a refactor pass.
3. Do not treat online metadata as complete.
4. Do not make BBH the only supported source class.
5. Do not collapse waveform/sample-set distinctions without provenance.
6. Do not drop parameters globally because one event lacks a column.
7. Do not export files without prior/cosmology/schema provenance.
8. Keep PRs small and reviewable.

## Working Philosophy

This package should become boring in the best way: explicit contracts, loud validation, clean manifests, and reproducible exports.

The scientific danger is not an exception. The scientific danger is a file that looks valid while having the wrong prior, wrong cosmology, wrong waveform mixture, missing FAR assumption, or silently dropped parameter.

Build the package so those states are visible.

