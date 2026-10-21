# Changes

## Review fixes (2026-10-20)

A review of gwcat at 3ba1dbf on 2026-10-20 confirmed 144 findings. They were
fixed in nine packages, merged on branch `fix/review-2026-10-20`: ingest,
selection export, catalog and selection view, selection, PE export, export
writers, validator and 2.1 contract, fetch and CLI, priors and spin. The owner's
modelling choices keep their defaults and are listed at the end.

### What changes for the production products

The production pair is the gwcat-2.0 BBH PE export in the `chieff` basis
(`gwcat export pe --format gwcat2 --parameter-space chieff --event-list
<259 names> --waveform-policy mixed-first --seed 0 --nsamp 4096`) and the
O3 + O4ab selection export in the `chieff_reference` basis
(`gwcat export selection endo3 rpo4ab --format gwcat2 --parameter-space
chieff_reference --spin-reference-amax 0.99 --far-threshold 1.0`).

- **Datasets: byte-identical.** Rebuilt from the existing 8263ae9 store with
  default options on this branch, every dataset of both files matches the
  trusted products bit for bit (12 of 12 in the PE file, 16 of 16 in the
  selection file).
- **Attributes.** The PE file gains 29 attributes and the selection file 29.
  Values that change:
  - PE `n_above_spin_ceiling_raw_per_event`: the true count with the cut off
    (it was written as zeros).
  - PE `population_resolver_inputs`: the corrected exclusion list, which now
    names GW190425_081805.
  - PE `selection_spec_digest`: no longer depends on how a whitelist was
    spelled (`1839ec1fedbb8808` to `45d5449be453016d`).
  - Selection `cosmology_H0` / `cosmology_Om0`: NaN, because the two campaigns'
    `pdraw` used different cosmologies. Read `cosmology_*_per_campaign`.
  - `writer_commit` on both.
- **Validation of a rebuild.** The trusted files still validate as before
  (they predate the new records). A rebuild now meets three refusals, each an
  open decision below:
  - non-strict, as at 8263ae9, needs a spin-prior allow-list for the 154
    Mixed-label events that divide out a borrowed spin prior;
  - even with that list, `xcheck_reference_spin_support` fails (and
    `--strict` raises): the PE file keeps 1,168 rows in 54 events above the
    0.99 reference ceiling, because `--drop-spin-above-ceiling` is off;
  - `--require-exact-priors` refuses the 69 events whose distance prior was
    evaluated with the legacy release-reweight table, which the production
    store was ingested with.

### Ingest (`gwcat/ingest.py`)

Store numbers change only in `sky_area_90` and `observing_run`, plus one
recorded `mass_prior_q_min`. None of these is exported.

New `IngestConfig` options, all off or unchanged by default:
- `mixed_prior_mismatch = "warn" | "raise"` (CLI `--refuse-mixed-prior-mismatch`)
- `spin_exceeds_amax = "warn" | "raise"` (CLI `--refuse-spin-above-amax`)
- `defaulted_cosmology_fatal` (CLI `--refuse-defaulted-cosmology`)
- `prior_ks_alpha = 1e-3`: the KS threshold is now `max(prior_ks_max, ` the
  Kolmogorov critical value at this significance for N`)`
- `prior_bounds_pvalue_min = 1e-3`

New per-row store columns: `mixed_constituent_labels`,
`mixed_constituent_priors`; `spin_sample_max_1/2`,
`spin_sample_max_per_constituent`; `dL_prior_ks_threshold`,
`dL_prior_n_prior_samples`, `dL_prior_frac_mass_uncovered`,
`dL_prior_frac_samples_outside`, `dL_prior_bounds_pvalue`;
`sky_area_90_nside`, `observing_run_source`.

New store attributes: `STORE_PROVENANCE_ATTRS` (writer version and commit,
the `IngestConfig` as JSON, the release table's sha256, the inputs), read
with `read_store_provenance()`.

Behaviour changes:
- Combined sets borrow priors from their `meta_data/combined` constituents
  first. `C00:Mixed+XO4a` is a combined set ranked after `C00:Mixed`.
- Aligned-spin labels use their own AlignedSpin ceilings (kind `aligned_spin`).
- A mass prior in any other coordinates is `unrecognized`.
- A repeated (event, sample set) in one build raises `DuplicateSampleSetError`,
  and the CLI de-duplicates glob matches.
- A merge treats an event's preferred row as one key whatever its label.
- `merge_store` takes `mock_data` (`None` inherits); a merge is mock only if a
  written row came from a mock store.
- A non-reweighted O1-O3 row with no analytic distance prior takes
  LALInference's dL² prior from its config, or is refused (it used to divide
  out `UniformSourceFrame`).
- A NaN KS with prior samples present fails under `prior_ks_fatal`.
- `sky_area_90` uses an adaptive nside (at least 20 samples per pixel), else NaN.
- `observing_run` comes from the median GPS time and gwcat's run windows.
- `--inspect` uses the build's flags.

### Selection export (`gwcat/export/selection_builder.py`)

- New provenance per campaign: the GPS span and its source, the observing runs
  covered, `T_obs_yr`; the injection file's path, size and sha256; the p_astro
  columns carried, with `p_astro_used_for_detection=False`;
  `chip_column_emitted`.
- The scalar `cosmology_H0` / `cosmology_Om0` is the cosmology every campaign's
  `pdraw` used, NaN when they differ or none was used (the v1 convention).
- New refusals:
  - the same campaign twice (by path or by content), or two campaigns over
    overlapping GPS time;
  - non-finite or non-positive FAR or SNR thresholds;
  - a non-finite or non-positive finished `pdraw` on any basis;
  - non-finite or non-positive masses or distance, or a non-finite or negative
    redshift, on written rows;
  - a NaN chi_eff or spin magnitude on a `chieff_reference` row;
  - `out_of_reference_pdraw` below 1e100.
- `chieff_reference` coverage: the drawn tilts must cover the isotropic
  reference, and a magnitude shortfall below `a_ref` that sampling noise cannot
  explain is refused under `strict`. Both use a 1e-9 false-alarm test. Holes
  under `strict=False` are recorded in the new tilt attributes and in
  `spin_reference_coverage_ok`.
- `chieff` basis: a forced `amax` that contradicts the detected ceiling now
  warns. The `*_per_campaign` spin attributes keep campaigns with no
  detections (NaN, `no_detections`).
- Docs: what `strict` governs, and what `spin_basis_assumption_violations` holds.
- Tests: an independent astropy check of the endo3 ddL/dz Jacobian, and
  multi-campaign combination tests.

### Catalog and selection view (`gwcat/catalog.py` and neighbours)

1. `select()` refuses event cuts on posterior-median masses and sky area at
   export unless declared (`allow_unpaired_point_estimate_cuts=True`); a
   declared cut is recorded as `point_estimate_cuts`.
2. `snr_min` warns about, and records as `n_events_missing_snr`, the events
   dropped only because their SNR is unknown, counted among the rows every
   other cut keeps. It raises when no candidate has an SNR.
3. Per-sample-set cuts apply to the sample set the waveform policy chose. A
   chained view keeps the policy, reasons and event map that chose it, and the
   event-map identity enters the selection spec.
4. `compact_type` is one canonical, case-insensitive class, and the class
   filter is the intersection of `compact_type` and `source_class`.
5. The FAR cut records `far_cut_operator='<='` and names events that sit on the
   rounded threshold.
6. The format 1.0 exporter records `z_max` truncation, out-of-support counts
   (now refused by default), the spin-prior source per event, the samples
   available to bootstrapped events, and the effective waveform policy. It
   takes an event map, refuses zero-event files, and writes a single
   `chi_eff_amax` only for one symmetric ceiling. The v1 validator refuses a
   truncated PE file against a v1 selection, a declared point-estimate cut, a
   file holding one event several times, and an exclude-mode file without one
   spin ceiling.
7. A name whitelist paired with a selection that has no class cut now warns
   instead of passing silently.
8. The live GWOSC BBH list no longer replaces the curated FAR < 1/yr list
   unless `adopt_live_list=True`.
9. Event lists with a UTF-8 BOM keep their first event; `refresh_bbh_list`
   assigns names to runs by their run window.
10. Metadata overrides warn on keys that match nothing, treat `'nan'` as
    absent, and refuse unknown classes.

### Selection (`gwcat/selection.py`, `selection_spin.py`, `observing_runs.py`)

Under default options every dataset of a selection export is bit-identical;
only attributes change. New attributes on every export, v1 and 2.x:
`injection_draw_domain_per_campaign` (with `_mismatch`, `_mismatch_detail`,
`_policy`, `_applied`), `campaign_T_obs_yr`, `pdraw_normalisation`,
`significance_far_discovery_per_campaign`,
`cosmology_generation_H0/Om0_per_campaign`, `cosmology_model_per_campaign`,
`spin_removal_amax_reference/factor_per_campaign`. v1 writers also gain
`injected_spin_uniform_isotropic` and
`spin_basis_assumption_violations/_violated/_unverified`.

New option: `build_selection_product` / `export(draw_domain="intersection")`
keeps only the detected injections inside the draw domain every campaign
declares. `ndraw` is unchanged, so the population model and the PE samples
must be truncated to the same domain. Python API only; there is no CLI flag yet.

Behaviour changes:
- A campaign drawn past another campaign's declared domain (O4ab against
  endo3's mass and redshift bounds) warns.
- The spin-ceiling detectors return no ceiling unless the draw is uniform on
  [0, amax] with amax <= 1.
- A declared search with no FAR column, or an unused FAR column, warns; FAR
  read errors are raised. The fallback FAR scan runs in file order.
- An H0/Om0 override that differs from a detectable generation cosmology warns.
- A campaign generated with a named astropy cosmology (Planck18 and similar,
  with radiation) gets that object's Jacobian.
- The spin-removal check fires on joint-format files whose magnitudes exceed 0.99.
- A zero per-spin cartesian pdf is refused.
- v1 combined per-campaign ceilings have one entry per campaign.

### PE export (`gwcat/export/pe_builder.py`, gwcat-pe-2.0 and 2.1)

Under default options no dataset changes.
- Provenance added: spin-ceiling counts with the cut off
  (`n_above_spin_ceiling_raw_per_event` is now the true count, -1 when not
  countable; new `n_exported_above_spin_ceiling_per_event` and
  `spin_ceiling_counted`); `prior_reweight_ess_distinct_per_event`;
  `pe_cosmology_scalar_source` / `_n_events`, with `pe_cosmology_H0/Om0` now
  taken from the first exported event; `f_ref_per_event` /
  `f_ref_source_per_event`; `events_dropped_by_z_max`,
  `n_samples_cut_by_z_max_dropped_events`, `z_max_dL_per_event` and
  `z_max_cut_coordinate`; `resample_seed`, `resample_replace_policy`,
  `resample_rng_scheme` and `n_kept_samples_per_event`;
  `events_resampled_with_replacement` (the replacement flags now include a
  forced `replace=True`); `component_columns_emitted` /
  `component_columns_missing_events`.
- New warnings: chieff rows kept above the spin ceiling without
  `--drop-spin-above-ceiling`; draws with replacement; a tilt fit across mixed
  `f_ref`; events wholly above `z_max`; omitted component columns.
- New refusals: an event with no in-support sample under any pooled
  `max_out_of_support_frac` (only `allow_out_of_support=True` exports it); a
  `chieff_chip` event with `amax_1 != amax_2`, unless `amax=` states one
  ceiling or the API opt-in `allow_chieff_chip_amax_mismatch=True` keeps the
  old `amax_1` evaluation (warned and recorded).
- Validator: a `chieff` / `chieff_reference` pair whose PE file keeps rows above
  the reference ceiling without the cut records a failure, and raises under
  `--strict` (`xcheck_reference_spin_support`).
- Docs: the `samples_bound` "exact" claim is scoped to sample sets drawn under
  one box; `population_resolver_match` is documented as a name-set equality.

### Export writers (`writers_gwcat2.py`, `atomic.py`, summaries)

- The 2.1 PE contract names the event cut `select()` applied: `far_and_snr_med`
  for `far_max` with `snr_min`, `snr_med` for `snr_min` alone (before:
  `far_or_snr` and `snr`). `gwcat validate` refuses any PE event cut on the
  posterior-median SNR, because no injection detection mask reproduces it.
- PE exports record which distance prior was divided out:
  `dL_prior_kind_per_event`, `dL_prior_alpha_per_event`,
  `dL_prior_basis_per_event`, and a label `dL_prior_kind` (for example
  `UniformSourceFrame` or `PowerLaw(alpha=2)`, `+`-joined when events differ).
  The 2.1 contract's `dL_prior_kind` is set from it. For the production BBH
  product it is `UniformSourceFrame`.
- In the 2.1 contract `sky_prior_in_density` is now `False`, because neither
  `p_pe` nor `pdraw` contains the `-ln 4π` factor. Every 2.1 `contract_hash`
  changes as a result.
- Every export records `writer_environment`: python, machine,
  numpy/scipy/astropy/h5py/HDF5 versions, numpy's BLAS, its SIMD baseline and
  the active dispatch targets.
- Validation summaries: a re-export removes the previous summary before the new
  file is published; summaries are written atomically and carry the
  `output_sha256` of the file they describe.
- `chieff_chip` PE summaries give `chi_eff_prior_applied_to_p_pe` and
  `spin_prior_mode` as the file does (`False`, `chieff_chip_joint`).
- An output path that is a symlink is written through to its target, and the
  link is kept.
- Writing a sky-marginal or NaN-sky selection product warns that
  darksirens-core cannot load it (this includes the cumulative O1-O4b recipe).
- Known limitation: `gwcat2` still means format 2.0, which carries no
  `contract_hash`, so darksirens-core's pairing gate does not check a 2.0
  pair, the production pair included.

### Validator and 2.1 contract (`validate.py`, `contract.py`)

- New PE checks: event names unique and `nobs` long; `homogeneous_sample_sets`
  not False (a `waveform_policy='all'` file now fails); masses, dL and redshift
  finite and positive, with m2 <= m1 and q in (0, 1]. A zero `p_pe` fails only
  at in-support samples, so recorded out-of-support zeros under
  `allow_out_of_support` validate.
- New selection checks: `ndraw` equals `sum(campaign_ndraws)`, and `T_obs_yr`
  is positive. `chieff_reference` sentinel rows must be exactly the rows above
  `a_ref`, with the recorded count and a negligible sentinel. With mixed sky
  availability, NaN-sky rows must follow the per-campaign flags.
- Cross-file checks: cosmology is compared with each campaign's recorded
  cosmology, not the scalar, and a varying PE cosmology without per-event arrays
  is refused. An equal `z_max` must be the same truncation in dL. A cumulative
  mixture must detect something in every PE event's run (placed from the event
  name). A `chieff` caller-forced ceiling that differs from the detected one
  warns (fails under `--strict`), and the swap's violation flags are refused.
- 2.1 contract: the hash also covers `source_class_cut_estimator`,
  `source_class_mass_threshold` (on a side with a class filter) and `z_max`,
  each only when set; a file with no class filter and no truncation keeps its
  previous `contract_hash`. The `contract` JSON gains these three keys.
- Tests: set `GWCAT_CONTRACT_STORE` to run the 11 store-backed 2.1 contract
  tests. A requested store that is missing fails instead of skipping.

### Fetch, CLI and summaries (`fetch.py`, `cli.py`, `population_samples.py`)

- GWOSC BBH list: `fetch_bbh_names_gwosc` / `fetch_bbh_list()` query every event
  version, with no `lastver` and no server-side mass filter. The highest LVK
  GWTC version of each event decides; a release's non-confident lists
  (`GWTC-*-marginal`, `GWTC-2.1-auxiliary`, ...) count in that ordering, so an
  event LVK demoted later (GW190424_180648, GW190909_114149) stays out. The
  m2 > 3 cut is applied to the deciding confident version. About 272 names are
  expected (the 2026-09-24 cumulative table), not the curated 259. An old
  `lastver` cache is no longer read offline.
- `--event-list` (`export pe`, `export-darksirens`) refuses names the store does
  not hold and suggests the store's spelling for date-only names.
  `--allow-missing-events` restores the skip and records
  `event_list_missing_names` in the validation summary; it is refused with
  `--no-summary`.
- Date-only names resolve only through
  `gwcat.population_samples.HISTORICAL_DATE_ONLY_NAMES` (23 names: GWTC-1's
  eleven plus the discovery-paper events). Any other date-only name is refused.
- New CLI flags: `inspect --parameter-space` (default component, the export
  default); `export pe --allow-projection-basis` (for `chieff_chip`) and
  `--allow-unpaired-pastro-min` (for `--pastro-min`). `--cosmology` / `--H0` /
  `--Om0` must be finite, with H0 > 0 and 0 < Om0 <= 1; `--nsamp` must be >= 1.
  A `MissingParameterError` is a one-line error with exit code 1.
- `validate --strict` help corrected. `validation_summary.json` is strict JSON
  (NaN becomes null).
- `writer_commit` is `unknown` when gwcat is installed inside another
  repository's checkout.
- `gwcat fetch`: each release directory gets `gwcat_fetch_manifest.json`. A run
  that chose its records itself warns about files from superseded releases and
  lists them in `superseded_files`; nothing is deleted. An explicit
  `--record-ids` run lists them as `files_outside_records_used` and leaves
  `superseded_files` null. GWTC-4 is an alias of GWTC-4.1 (one directory,
  fetched once). sha256 checksums are supported, and files with no published
  checksum are reused. The cached record resolution is written only by resolving
  runs, records failed resolutions, and is checked offline. `gwcat fetch --out`
  fills the store's `record_id` / `file_checksum` columns. The GWOSC event table
  also answers to the full O1/O2 names.

### Priors and spin (`spin.py`, `constituent_mixture.py`, `params/`, `cosmology.py`)

Every dataset of the production exports is unchanged; the selection file gains
one attribute and the PE file eight.
- Constituent-mixture prior (opt-in): every `Constraint` enters the support
  (NRSur7dq4's 60-400 total-mass cut was being dropped), and a `Constraint` the
  support does not implement refuses the mixture. New
  `--constituent-mixture-convention {equal_weight (default), own_prior}`,
  recorded in each row's prior sources.
- PE provenance: `dL_prior_cosmology_source_per_event`,
  `dL_prior_cosmology_table_sha256_per_event`,
  `release_reweight_cosmology_table_ids`, `mass_prior_source_per_event`,
  `sample_set_ingest_selection_reason_per_event`,
  `sample_set_priority_rank_per_event`, `allow_zero_p_pe` and
  `n_zero_p_pe_allowed`. `validate --require-exact-priors` refuses events
  evaluated with the legacy release-reweight table, and warns on files that
  predate the record.
- Spin priors: a NaN chi_eff or mass is an error (`NonFiniteSpinInputError` /
  `NonFiniteSampleError`), no longer an out-of-support sample, and so is a NaN
  density at a finite, supported point. `support()` gives the true
  mass-dependent edge for unequal ceilings. Spin ceilings (`amax`,
  `amax_fallback`) must be finite and in (0, 1]. `chieff_chip` refuses events or
  campaigns whose two ceilings differ.
- Selection spin provenance: `injected_spin_azimuths_per_campaign` (`factored`,
  `checked_isotropic`, `assumed_uniform`, `none`) records whether the 4-D
  component density's azimuth treatment is exact, verified or assumed. Nothing
  is refused on it.
- Event map: duplicate (event, label) rows are refused. An exported label is
  counted over the selection; a label the NRSur q-rule rules on or measures is
  counted over the whole store. A q-rule substitute flag without a rule or
  outside `event-map`, and duplicate JSON keys, are refused. A rule
  substitution keeps the chain of declared substitutes.
- Cosmology tokens: any spelling of `planck15_lal` is accepted, as in bilby.
  Unknown tokens, and named cosmologies with radiation (astropy Planck15), are
  refused at ingest instead of falling back to a default.
- Parameter registry and 2.1 writers: a 2.1 selection with any non-finite row
  in a fit column (ra/dec of a campaign without a sky draw) is refused; use
  `--sky-marginal`. `m1src`/`m2src` and the extra spin columns are declared
  advisory, and the writers refuse undeclared datasets. **Pairing break:** this
  changes every space's 2.1 `advisory_columns` and so its 2.1 contract hash;
  re-export 2.1 PE and selection files together. Format 2.0 is unaffected.
- `allow_zero_p_pe` now covers exact zeros only. The missing-parameter hint
  offers only spaces the builder can build.
- Packaging: `scipy>=1.8` is a declared dependency.

### Docs, examples and CI (integration)

- README: the end-to-end recipe, the BBH lists, the quick start, point-estimate
  cuts, merge keys, mock pairs, the 1.0 formats, the 2.1 hash, the
  `chieff_reference` basis, the nocosmo distance prior, `o3_default_cosmo`, the
  mass Jacobian, the sky area, the new ingest guards and selection provenance.
- Examples: the O3 + O4ab workflows use the gwcat-2.0 commands that run, with no
  `--H0`/`--Om0`, and the fake-data tutorial now passes `validate --strict`.
- Validator docstrings say which cross-file checks only warn without `strict`.
- `.github/workflows/tests.yml`: a clean-install import (Python 3.9, 3.12) and
  the test suite (3.9, 3.11). It has not run on GitHub yet.

### Integration notes

- `pe_builder.py`: the PE export and priors/spin packages both refused
  `chieff_chip` events with two ceilings. One refusal is kept, with both
  escapes (`amax=<value>`, and `allow_chieff_chip_amax_mismatch=True` with its
  warning).
- `validate.py`: the writers and validator packages both added the refusal of a
  posterior-median SNR event cut; the writers version (refuse any `snr_min`) is
  kept.
- `ingest.py`: the unknown-cosmology fallback warning is gone, because the token
  is now refused; `defaulted_cosmology_fatal` still refuses a prior that names
  no cosmology.
- `validation_summary.py`: the atomic write and the strict-JSON dump are both kept.
- `catalog.py`: the restructured `select()` passes `nrsur_q_rule_substitute` to
  `resolve_policy` in its non-event-map branch.

### Options awaiting the owner's decision

Every default below is unchanged.

1. **Mixed-set prior convention.** Today a C00:Mixed row borrows one
   constituent's prior (now taken from its constituents first); 44 population
   rows are flagged `differ:mass_ratio,total_mass`. Options: refuse such rows
   (`mixed_prior_mismatch='raise'`), the opt-in constituent mixture with
   `equal_weight` or `own_prior` weights, or inverse-evidence weights (not
   implemented: they need evidences under one likelihood). Choose before any
   rebuild of the C00:Mixed rows.
2. **Spin samples above the 0.99 ceiling.** 73 of 282 store rows have such
   samples (the C01:Mixed SEOBNRv4PHM halves); the production PE file keeps
   1,168 rows in 54 events above it. Options: keep them (a rebuild then fails
   `validate --strict`, or an explicit allow flag could be added), adopt
   `--drop-spin-above-ceiling` for the next rebuild (measured effect about
   -0.0007 spectral-only standard deviations), widen the ceiling, or refuse at
   ingest (`spin_exceeds_amax='raise'`).
3. **Release-reweight cosmology table.** The production store used the legacy
   table, so 69 events record `legacy_gwcat_default`, and
   `validate --require-exact-priors` now fails any new export from that store,
   although the phase12t recipe records that gate as passing (the existing
   files still pass, with a warning). Options: (a) keep the legacy table and
   drop `--require-exact-priors` from the recipe, (b) re-ingest with the
   production table (this changes `p_pe` for those 69 events), (c) make the
   legacy-table check a warning.
4. **O3 injection mass domain.** Now recorded, with the opt-in
   `draw_domain='intersection'`. As-is the effect was about -0.01 spectral-only
   standard deviations; restricting to the intersection is a different model
   (about +0.14, driven by GW231123) and needs the PE samples and population
   truncated to the same domain.
5. **Whitelist m2 cut versus the injections.** The 259-event list coincides with
   a posterior-median m2 >= 3 cut that the selection does not apply. An opt-in,
   recorded injected-truth m2 >= 3 cut on the selection side is **not
   implemented** on any branch; for now a warning is the only guard.
6. **Time coverage of the 10 O1/O2 events.** The production pair puts 10 O1/O2
   events against O3 + O4ab injections with no O1/O2 exposure. The selection
   side now records GPS spans and runs; refusing the pair needs the PE side to
   record event times and validate/core to compare them.
7. **Audited builds.** Whether to turn on `mixed_prior_mismatch='raise'`,
   `spin_exceeds_amax='raise'`, `defaulted_cosmology_fatal` and `prior_ks_fatal`.
8. **Format and pairing gate.** Once the 2.1 hash accepts the `chieff` +
   `chieff_reference` pair (a contract-format decision: a shared pairing
   identity or a separate ceiling field, with a core loader check), whether to
   make 2.1 the default and rebuild the production pair; whether core should
   refuse a pair with no contract hash; and whether core should run
   `gwcat validate`, since the hash leaves out the detection cut, cosmology,
   spin ceilings and `chi_eff_prior_impl`. Existing 2.1 files must be rebuilt
   in pairs.
9. **Spin-prior allow-list.** The production pair validates only with the
   self-derived allow-list for the 154 borrowed-prior events, which is still
   unapproved.
10. **Cumulative O1-O4b mixtures.** Whether to agree a sky-marginal contract
    with core so a spectral-only fit can load one; for now the writer warns.
11. **PE FAR operator.** The cut stays inclusive (`<=`) and is recorded; whether
    to make it strict, or have the validator compare operators.
12. **Selection-coverage constants and core.** The coverage tests use a 1e-9
    false-alarm rate and a minimum density ratio of 0.1 against the reference
    (production margins are 15 to 25 times). Whether core should refuse
    `spin_reference_coverage_ok=False`; whether to split unverified spin-less
    campaigns out of `spin_basis_assumption_violations` (it would change what
    core refuses); the overlap policy (a pair whose overlap cannot be judged
    only warns, since legacy files carry no times).
13. **Smaller choices.** A CLI flag for `allow_chieff_chip_amax_mismatch`;
    exact per-campaign NaN-sky membership (needs per-campaign row counts in the
    file, which changes 2.0 export bytes); `T_obs_yr` of the acknowledged
    O1/O2-blind mixture still includes O1/O2 time; deprecating the v1
    `gwcat selection` subcommand or giving it a basis option; an opt-in
    size-and-mtime cache to skip re-hashing about 75 GB on every fetch;
    embedding the gwcat commit at build time so non-editable installs stamp a
    real `writer_commit`; re-measuring the live GWOSC BBH count (expected 272)
    when there is network access.

### Follow-ups not done in this round

- Copy the new store provenance and per-row flags into PE products, and have
  `validate --strict` and core read them.
- The validator's acceptance of `constituent_mixture` as a prior source.
- A lower-edge test in `_check_reference_coverage`.
- A CLI `--draw-domain` flag; a 2.1 contract field for the draw domain, and a
  validator check of PE samples beyond a campaign's domain.
- Recording `spin_ceiling_cut_applied`, `f_ref`, `events_dropped_by_z_max` and
  `point_estimate_cuts` in the 2.1 contract; `f_ref` on the injection side and
  one shared `z_max` distance threshold; the validator reading `in_support`.
- Comparing `nobs` with `n_allowed_names` in the validator.
- Core: read the per-event spin ceilings of v1 files (or refuse a NaN or
  asymmetric `chi_eff_amax`), apply the bootstrap variance correction, and read
  `homogeneous_sample_sets`.
- The 2.0 builder writing `n_samples_available_per_event`; an event-map flag
  (or its removal) for `export-darksirens`; filling `snr_med` from the
  matched-filter SNR (it changes store values).
- `examples/make_fake_store.py` draws its NSBH and BNS fake events with noise
  that makes some component masses negative, so exporting them fails at the
  chi_eff prior; the tutorial exports the BBH events only.
