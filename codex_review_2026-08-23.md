# `gwcat` Code Review — 2026-08-23

Overall, the package shows strong domain awareness and unusually thorough regression coverage, but I would not rely on the current export metadata alone to establish PE/selection consistency. Several contracts can describe different physics than the arrays actually contain.

## Physics and correctness

1. **High — Effective event-selection provenance is lost.**

   `select()` intersects rows correctly, but constructs a new catalog whose provenance contains only the latest call’s arguments ([catalog.py:329](/hildafs/projects/phy230014p/magana/src/gwcat/gwcat/catalog.py:329)). Export then calls `select()` again with defaults ([pe_builder.py:292](/hildafs/projects/phy230014p/magana/src/gwcat/gwcat/export/pe_builder.py:292)) and writes those defaults into the product.

   Reproduction on the working store: exporting `cat.select(source_class="bbh")` retained all 273 BBH rows, but recorded an empty `source_class_filter`, no cut estimator, and no event-list filter.

   Direct `allowed_names` and `pastro_min` filters are also absent from the pairing contract ([contract.py:58](/hildafs/projects/phy230014p/magana/src/gwcat/gwcat/export/contract.py:58)). Thus a filtered PE file can advertise itself as unfiltered and pass checks against an incompatible selection function.

   Fix: introduce an immutable, composable `SelectionSpec`; normalize all name filters into it; include every effective cut and a deterministic event-list digest in the contract. `pastro_min` should be refused for paired export unless an injection-side equivalent is defined.

2. **High — The declarative parameter blocks do not drive the numerical physics.**

   Blocks declare prior callbacks and safety gates, including rejecting unsupported mass priors ([mass.py:18](/hildafs/projects/phy230014p/magana/src/gwcat/gwcat/params/blocks/mass.py:18)), but production builders never invoke these callbacks. Instead, PE export hard-codes `p_pe = m1 * p_dL` ([pe_builder.py:473](/hildafs/projects/phy230014p/magana/src/gwcat/gwcat/export/pe_builder.py:473)) and stamps `mass_prior_basis="uniform_detector_frame"`.

   The current store contains 273 rows marked `uniform_detector_frame` and 9 marked `assumed_default`; all are nevertheless exported as verified uniform priors.

   The 2.1 contract is also coordinate-inconsistent: the mass block publishes `(m1det,m2det)` as fit columns ([mass.py:38](/hildafs/projects/phy230014p/magana/src/gwcat/gwcat/params/blocks/mass.py:38)), while the `m1` factor is explicitly the Jacobian for `(m1det,q)`. A generic contract consumer could therefore apply the wrong measure.

   Fix: make builders compose the registered block callbacks and contexts. Publish `q` as the density coordinate, with `m2det` derived/advisory.

3. **High — `detection_efficiency()` uses retained rows instead of total generated draws.**

   `n_injections` returns `len(_m1det)` and becomes the denominator ([selection.py:1269](/hildafs/projects/phy230014p/magana/src/gwcat/gwcat/selection.py:1269)), even though clipped injection files carry the authoritative `total_generated`.

   On the O4 artifact:

   - Current result: `986,829 / 2,959,534 = 0.33344`
   - Correct generated-draw efficiency: `986,829 / 870,454,872 = 0.0011337`

   The public diagnostic is approximately 294× too high. Use `_ndraw`; expose retained row count separately.

4. **High for `spin_basis="chieff"` — Wrong spin ceilings and broken support enforcement.**

   This does not affect the default component basis.

   The χeff paths use one caller-provided `amax=0.99` for both PE and injections, rather than the PE event’s prior ceiling and each campaign’s detected ceiling. For example, end-O3 uses approximately 0.998. The validator then incorrectly requires PE and injection ceilings to match, although their proposal priors may legitimately differ.

   Separately, `ChiEffPrior.logprob()` intentionally does not mask support ([spin.py:205](/hildafs/projects/phy230014p/magana/src/gwcat/gwcat/spin.py:205)), but the builders equate “finite log probability” with “in support” ([pe_builder.py:1004](/hildafs/projects/phy230014p/magana/src/gwcat/gwcat/export/pe_builder.py:1004), [selection_builder.py:477](/hildafs/projects/phy230014p/magana/src/gwcat/gwcat/export/selection_builder.py:477)). For `amax=.99`, `χeff=.995`, `m1=50`, `m2=25`, the code returns a finite density `2.73e-12` even though `support()` is false.

   Fix: use proposal-specific ceilings, explicitly apply `support()`, and compare each ceiling with its own provenance rather than with the other file.

5. **High robustness — Invalid injection PDFs and weights are not rejected before arithmetic.**

   O3 floors zero or negative mass/redshift densities to `1e-300`, while O3/O4 divide by unchecked weights and observing times ([selection.py:1068](/hildafs/projects/phy230014p/magana/src/gwcat/gwcat/selection.py:1068), [selection.py:804](/hildafs/projects/phy230014p/magana/src/gwcat/gwcat/selection.py:804)). Final `pdraw` is not validated before writing.

   This can turn malformed support into fabricated tiny densities or propagate `inf`/`NaN`. Require finite, strictly positive PDFs, weights, observing time, `ndraw`, and final `pdraw`.

6. **Medium — The Astropy UniformSourceFrame fallback is inaccurate near zero distance.**

   The fallback uses a 4,000-point log-redshift grid ([cosmology.py:240](/hildafs/projects/phy230014p/magana/src/gwcat/gwcat/cosmology.py:240)). Compared with the same formula on a 400,000-point grid, its relative shape error was approximately +165% at 1 Mpc, +32.7% at 2 Mpc, +2.9% at 5 Mpc, and +1.3% at 10 Mpc. Current tests start much farther away.

   Use an adaptive or hybrid linear/log grid near `z=0`, and test the actual supported prior-bound range.

## Performance

7. **High — Ingest and merge retain several complete catalogs simultaneously.**

   Merge eagerly reads both stores and then allocates concatenated copies ([ingest.py:2044](/hildafs/projects/phy230014p/magana/src/gwcat/gwcat/ingest.py:2044)); the in-place path reads the existing store once for inspection and again during merge.

   A production `_read_store` of the 1.7 GB catalog took 7.6 seconds and peaked at 2.03 GiB RSS. Appending one small event can exceed 6 GiB. Initial ingest similarly retains per-analysis records plus a complete assembled union.

   Replace this with a metadata pass, preallocated/extendible HDF5 datasets, and bounded event-slice copying.

8. **High — Export loads far more data than it emits.**

   `GWCatalog.get()` reads every selected posterior slice ([catalog.py:360](/hildafs/projects/phy230014p/magana/src/gwcat/gwcat/catalog.py:360)); downsampling happens afterward. The measured PE export read 6.88 million rows to emit 1.16 million, taking 5.6 seconds and 1.20 GiB peak RSS.

   Selection loading reads a 2.8 GB compound HDF5 dataset field-by-field ([selection.py:121](/hildafs/projects/phy230014p/magana/src/gwcat/gwcat/selection.py:121)) and eagerly derives all spin representations. Loading O4 took 16.5 seconds/1.01 GiB; the combined selection build took 38.1 seconds/1.07 GiB.

   Stream event-by-event, select indices before loading every parameter, and use a basis-specific field plan with chunked `Dataset.fields(...)` reads.

## Package structure and cleanliness

9. **High — The public validator can falsely bless incomplete v2 files.**

   `gwcat.validate_export` exposes the legacy validator ([__init__.py:17](/hildafs/projects/phy230014p/magana/src/gwcat/gwcat/__init__.py:17)), while only the CLI dispatches by format. The legacy validator checks required datasets only if they happen to exist ([catalog.py:1181](/hildafs/projects/phy230014p/magana/src/gwcat/gwcat/catalog.py:1181)); missing masses, sky, or `p_pe` may therefore produce no failure.

   Use one public format dispatcher and require every dataset mandated by the selected schema.

10. **Medium — Parallel pipelines are causing drift.**

   Physics is duplicated across legacy `catalog.py`/`selection.py` and the v2 builders. This is already visible in the provenance, support, and validation discrepancies above. Additionally, 2.1 writers generate summaries while the file still identifies as 2.0, then mutate its attributes afterward ([writers_gwcat2.py:279](/hildafs/projects/phy230014p/magana/src/gwcat/gwcat/export/writers_gwcat2.py:279)).

   Consolidate around canonical builders with thin versioned writers. Also address non-atomic scientific-output writes, duplicate static version sources, unused `setuptools-scm`, and the missing tracked LICENSE file.

## Verification

A broad test run produced **546 passed, 2 skipped**. The only two failures were caused by the available Python 3.8.3 lacking `importlib.resources.files`; the package correctly declares Python ≥3.9, so I did not count those as code defects.

The review was read-only, included production-artifact benchmarks and focused numerical reproductions, and left the worktree clean.
