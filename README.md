# gwcat

Fast preprocessing of GWTC-2.1 / 3 / 4.1 / 5 PE files and LVK injection sets
into the formats consumed by [darksirens](https://github.com/ignaciomagana/darksirens),
with automated Zenodo fetching.

Covers O1 through O4b (GWTC-5.0, May 2026): **259 BBH** events with detailed
parameter estimation, all available through the same pipeline.

## Installation

```bash
pip install .                # core: query, export, selection processing
pip install ".[fetch]"       # + Zenodo downloader (requests, tqdm)
pip install ".[ingest]"      # + PESummary/bilby for raw PE file ingest
pip install ".[all]"         # everything
pip install -e ".[dev]"      # editable + pytest
```

For sky-area computation at ingest (optional): `pip install healpy`

## End-to-end: PE samples + selection function → darksirens

The full pipeline for a GWTC-5 dark-siren BBH analysis. Steps 1–3 are
one-time setup; steps 4–8 are the working loop.

```python
from gwcat import GWCatalog, SelectionSet, CombinedSelectionSet
from gwcat import build_store, validate_export
from gwcat.fetch import fetch_catalog
from gwcat.bbh_allowed_names import fetch_bbh_list, BBH_ALL

# ── 1. Download PE files from Zenodo ─────────────────────────
paths_21 = fetch_catalog("GWTC-2.1")
paths_3  = fetch_catalog("GWTC-3")
paths_41 = fetch_catalog("GWTC-4.1")
paths_5  = fetch_catalog("GWTC-5")       # fetches Part 1 + Part 2

# ── 2. Download injection files ──────────────────────────────
fetch_catalog("injections-O3-BBH", data_dir="./injections")   # O1+O2+O3 BBH
fetch_catalog("injections-O4ab",   data_dir="./injections")   # O4a+O4b

# ── 3. Ingest PE files into the store ────────────────────────
all_pe = paths_21 + paths_3 + paths_41 + paths_5
build_store(all_pe, "store.h5")           # FAR/p_astro auto-fetched from GWOSC

# ── 4. Build the BBH event list ──────────────────────────────
# Queries GWOSC for all events with m2_source > 3 Msun (= BBH threshold).
# Falls back to a static list of 126 confirmed O1–O4a events if offline.
# Once GWOSC indexes GWTC-5.0, this returns all 259 BBH automatically.
bbh_names = fetch_bbh_list()              # live from GWOSC
# bbh_names = BBH_ALL                    # offline / reproducible fallback

# ── 5. Explore the catalog ───────────────────────────────────
cat = GWCatalog("store.h5")
cat.summary()                             # compact event table
bbh = cat.select(
    allowed_names=bbh_names,              # authoritative GWTC-5 BBH list
)
print(f"Selected {bbh.n_events} events")

# ── 6. Export PE samples for darksirens ──────────────────────
cat.to_darksirens(
    "gw_bbh.h5",
    compact_type=None,                    # no derived gate; whitelist is authoritative
    allowed_names=bbh_names,
    nsamp=4096,
    z_max=10.0,                           # per-sample redshift cut
    cosmology=(67.74, 0.3089),
    seed=0,
)

# ── 7. Export combined selection function ────────────────────
sel_o3 = SelectionSet("injections/injections-O3-BBH/endo3_bbhpop-LIGO-T2100113-v12.hdf5")
sel_o4 = SelectionSet("injections/injections-O4ab/mixture-real_o4a_o4b-cartesian_spins.hdf")
combined = CombinedSelectionSet([sel_o3, sel_o4])
combined.to_darksirens("selection_bbh.h5", far_threshold=1.0)

# ── 8. Validate before running darksirens ────────────────────
validate_export("gw_bbh.h5", "selection_bbh.h5")

# ── 9. Run darksirens ────────────────────────────────────────
# from darksirens.gw.utils import load_gw_samples, load_selection_samples
# m1, m2, dL, chieff, ra, dec, p_pe, nEvents, nsamp = load_gw_samples("gw_bbh.h5")
# m1s, m2s, dLs, chis, ras, decs, pdraw, ndraw = load_selection_samples("selection_bbh.h5")
```

## BBH event selection

gwcat uses `allowed_names` to gate which events enter the export, bypassing
the need for reliable FAR/p_astro values (which are not in per-event PE files
and can be hard to fetch in bulk).

```python
from gwcat.bbh_allowed_names import fetch_bbh_list, refresh_bbh_list, BBH_ALL

# Live query — returns 259 events once GWOSC indexes GWTC-5.0:
bbh_names = fetch_bbh_list()

# Offline / reproducible — 126 confirmed O1–O4a events:
bbh_names = BBH_ALL

# One-time refresh: run this when GWOSC has indexed GWTC-5.0 to print
# the O4b event list you can paste into BBH_O4B in bbh_allowed_names.py:
refresh_bbh_list()
```

`fetch_bbh_list()` queries `gwosc.org/api/v2/event-versions` with a
`min-mass-2-source=3.0` filter, which is the exact threshold the LVK uses to
classify events as BBH in the GWTC-5 populations paper. It follows pagination
automatically and excludes NSBH/BNS by construction. No manual exclusion list
needed.

`allowed_names` can be combined with any other `select()` cuts:

```python
# Allowed list + additional mass cut
bbh = cat.select(
    allowed_names=bbh_names,
    m1_src_range=(5, 100),
    snr_min=8.0,
)

# Or pass directly to the exporter (skips a separate select() call).
# compact_type=None avoids adding a derived compact-type gate on top of the
# authoritative whitelist.
cat.to_darksirens(
    "gw_bbh.h5",
    compact_type=None,
    allowed_names=bbh_names,
    nsamp=4096,
    cosmology=(67.74, 0.3089),
)
```

## Incremental updates with `merge_store`

When a new catalog drops, append without re-ingesting the full set:

```python
from gwcat import merge_store
from gwcat.fetch import fetch_catalog

new_paths = fetch_catalog("GWTC-5", data_dir="./GWTC")
merge_store("store.h5", new_paths)   # duplicates auto-skipped, FAR auto-fetched
```

## Quick start (CLI)

One console script, `gwcat`, with a subcommand per pipeline stage:

```bash
gwcat fetch --out store.h5                            # download all PE + build (FAR auto-fetched)
gwcat fetch --out store.h5 --no-event-table           # skip GWOSC FAR/p_astro fetch
gwcat fetch --catalog injections-O3-BBH               # O3 BBH injection set
gwcat fetch --catalog injections-O4ab                 # O4a+b injection set
gwcat fetch --catalog all                             # PE + all injection sets
gwcat fetch --catalog GWTC-5 --dry-run                # preview files
gwcat fetch --no-resolve                              # skip concept DOI resolution
gwcat fetch --out store.h5 --metadata-overrides metadata_overrides.yaml
gwcat fetch --out store.h5 --write-file-provenance    # opt-in sha256 + JSON

gwcat ingest --glob "./GWTC/GWTC5/*.h5" --out store.h5              # raw PE files -> store.h5
gwcat ingest --glob "./GWTC/GWTC5/*.h5" --out store.h5 \
             --sample-sets all --cache-dir ./cache                   # PR6/PR8 options
gwcat ingest --glob "./GWTC/GWTC5/*.h5" --out store.h5 \
             --metadata-overrides metadata_overrides.yaml

gwcat inspect store.h5                                # events, sample sets, params,
                                                       # availability, source classes
gwcat inspect store.h5 --json                         # same, machine-readable

gwcat export-darksirens store.h5 --out gw_bbh.h5 \
    --source-class bbh --spin-prior-mode include \
    --waveform-policy mixed-first --cosmology 67.74,0.3089 \
    --far-max 1.0 --allow-missing-far --nsamp 4096 --seed 0

gwcat selection --injections inj_o3.hdf inj_o4.hdf \
    --out selection_bbh.h5 --source-class bbh --far-threshold 1.0 \
    --H0 67.74 --Om0 0.3089

gwcat validate gw_bbh.h5 selection_bbh.h5             # exit 0 iff every check passes
gwcat validate gw_bbh.h5 selection_bbh.h5 --json      # machine-readable result
```

Run `gwcat --help` / `gwcat <subcommand> --help` for the full flag reference.
In particular, `gwcat fetch --help` and `gwcat ingest --help` identify their
usage as the unified commands (not the deprecated standalone script names).
See `examples/bbh_workflow.md` and `examples/all_cbc_workflow.md` for the
complete full-data command sequences, and `examples/tutorial_fake_data.md` +
`examples/make_fake_store.py` for a fully offline walkthrough on synthetic
data (`make → inspect → export-darksirens → selection → validate`) that
mirrors what `tests/test_cli.py` drives end-to-end.

For the current architecture, scientific contracts, and recommended real-data
commands, see [`docs/codex_gpt56_sol_handoff.md`](docs/codex_gpt56_sol_handoff.md).

The standalone `gwcat-fetch` / `gwcat-ingest` scripts still work unchanged
(same flags, same behavior) but are deprecated: they print a one-line pointer
to `gwcat fetch` / `gwcat ingest` on stderr before delegating.

### Validation summaries

Every `gwcat ingest` / `gwcat export-darksirens` / `gwcat selection` run
writes `<out>.validation_summary.json` and `<out>.validation_summary.md`
next to its output by default (pass `--no-summary` to skip). At the library
level this is opt-in via `write_summary=True` on `build_store`,
`GWCatalog.to_darksirens`, and `SelectionSet`/`CombinedSelectionSet.to_darksirens`
(default `False`, so existing calls are unaffected). The `.md` is a
human-rendering of the exact same dict written to `.json` -- never a separate
computation.

Fields actually reported (never fabricated -- a value not available from the
store/export is simply omitted): `package_version`, `schema_version`,
`n_events`, `event_names`, `stored_parameters`, `missing_required_parameters`,
`missing_optional_parameters`, `partial_availability` (per-parameter count of
events where it is NaN-filled), `source_class_counts`, `waveform_counts`,
`approximant_counts`, `sample_set_counts_per_event`, `far_missing_count`,
`far_available_count`, `p_astro_available_count`, `per_event_cosmology_present`
/ `per_event_cosmology_varies`, plus per-`kind` fields:

* `kind="ingest"` (from `build_store`): `n_files_provided`,
  `n_unique_events_ingested`, `n_rows_ingested`, `sample_sets_mode`,
  `source_file_checksums` (when `file_provenance` was passed).
* `kind="darksirens_export"` (from `GWCatalog.to_darksirens`):
  `n_events_considered`, `n_events_exported`,
  `n_events_skipped_after_selection`, `event_names_exported`,
  `source_class_filter`, `event_list_filter`, `far_policy`,
  `allow_missing_far`, `require_far`, `n_events_missing_far`,
  `spin_prior_mode`, `chi_eff_prior_applied_to_p_pe`, `cosmology_mode`,
  `cosmology_override_used`, `cosmology_per_event_varies`, `waveform_policy`,
  `approximant`, `homogeneous_sample_sets`.
* `kind="selection_export"` (from `SelectionSet`/`CombinedSelectionSet`):
  `n_campaigns`, `n_injections_total`, `n_injections_before_filter`,
  `n_injections_after_filter`, `n_detected`, `ndraw`, `far_threshold`,
  `significance_columns`, `source_class_counts_detected`, `cosmology_H0`,
  `cosmology_Om0`, `p_astro_available` (always `False` -- selection files
  never carry per-injection `p_astro`).

## Modules

### `GWCatalog` — query + explore + export

```python
cat = GWCatalog("store.h5")
cat.summary()                                     # compact event table
cat.event_names                                   # array of GW names

# Metadata-based selection (no I/O, all cuts composable)
bbh = cat.select(
    allowed_names=bbh_names,                      # authoritative BBH list (recommended)
    snr_min=8.0,
    m1_src_range=(5, 100),
    sky_area_max=500,                             # requires healpy at ingest
)

# FAR/pastro cuts still available when event_table was populated at ingest
bbh = cat.select(compact_type="BBH", far_max=1.0, pastro_min=0.9)

# --- Source-class selection (BBH / NSBH / BNS / all-CBC) ---
# Filters on per-event source_class metadata (falls back to compact_type),
# NOT on a static event-name list.  "cbc" = all compact-binary classes.
bbh  = cat.select(source_class="bbh")
nsbh = cat.select(source_class="nsbh")
bns  = cat.select(source_class="bns")
cbc  = cat.select(source_class="cbc")             # everything

# --- User event-list filtering ---
sub = cat.select(event_list="my_events.txt")      # one name per line, # comments ok
sub = cat.select(event_list=["GW150914", "GW170817"])

# --- Missing-FAR policy on far_max cuts ---
# Public metadata may not expose FAR; far_available=False is a valid state.
sub = cat.select(far_max=1.0, allow_missing_far=True)  # keep, warn, record
sub = cat.select(far_max=1.0, require_far=True)        # fail loudly if any FAR missing
# default: drop missing-FAR events (with a warning).  to_darksirens records the
# choice in output attrs: far_policy, allow_missing_far, require_far,
# n_events_missing_far, source_class_filter, event_list_filter.

# Read posterior samples
d = bbh.get(["mass_1", "luminosity_distance"])    # flat concatenated
d = bbh.get(["mass_1"], per_event=True)           # list of arrays per event

# Derived quantities (computed on demand, not stored)
q   = bbh.mass_ratio()
Mc  = bbh.chirp_mass(frame="source")
chi = bbh.chi_eff()
m1s, m2s = bbh.source_masses(cosmology=(67.74, 0.3089))
```

### `SelectionSet` — injection processing (all CBC)

Reads both LVK injection formats:
- **`events/` format** (O4 sets, Zenodo 19500064): modern format with log-joint draw PDF
- **`injections/` format** (O3 BBH, Zenodo 7890437): legacy format with factored draw PDF components

Selection products are **not BBH-only**.  Pass `source_class` to subset the
injections by source class — BBH / NSBH / BNS / MassGap / `cbc` (all
compact-binary classes).  Injections are classified by their *injected*
source-frame component masses using the **same shared mass-threshold classifier**
(`gwcat.source_class.classify_by_mass`, NS/BH split at 3 M⊙) that labels PE
events, so a `bbh` selection of injections is consistent with a `bbh` selection
of PE events by construction.  `source_class=None` (the default) applies no
restriction and is byte-identical to the pre-existing export.

```python
sel = SelectionSet("injection_file.hdf", H0=67.74, Om0=0.3089)
sel.n_injections                                  # total in file
sel.detection_efficiency(far_threshold=1.0)       # fraction detected
sel.source_class_mask("nsbh")                     # boolean mask by injected mass
sel.to_darksirens("selection_bbh.h5", far_threshold=1.0, source_class="bbh")
```

**Source-class filtering is subsetting, not reweighting.** `ndraw`
(`total_generated`) is left unchanged, following the Essick et al. (2023)
multi-campaign estimator; the analyst **must** pair a class-filtered selection
file with a PE export filtered to the *same* class(es).  Every export records
explicit provenance in its attrs:

- `pdraw_state` — what the exported `pdraw` represents after all manipulations
  (draw density in the `(m1det, q, dL)` basis with the 1-D chi_eff prior swapped
  in, normalised by `T_obs` and injection weights);
- `source_class_filter`, `source_class_method`, `nsbh_mass_threshold`,
  `n_injections_before_filter`, `n_injections_after_filter`, and a
  `source_class_filter_note` when a filter was applied;
- search/significance provenance: `significance_columns` (which FAR pipeline
  columns were thresholded), `significance_type`, `significance_far_threshold`,
  `significance_available`, and `p_astro_available=False` (explicit absence —
  no per-injection `p_astro` is used for thresholding), mirroring the FAR
  contract.

### `CombinedSelectionSet` — multi-campaign combination

Combines injection sets from different observing runs following
Essick et al. (2023), with proper ndraw reweighting.  `source_class` applies
per campaign (again subsetting only — the `N_k / N_total` fractions are
unchanged):

```python
sel_o3 = SelectionSet("endo3_bbhpop-LIGO-T2100113-v12.hdf5")
sel_o4 = SelectionSet("injections-O4ab/...-cartesian_spins_*.hdf")
combined = CombinedSelectionSet([sel_o3, sel_o4])
combined.to_darksirens("selection_bbh.h5", far_threshold=1.0, source_class="bbh")
```

### `validate_export` — pre-flight checks & cross-validation

```python
from gwcat import validate_export
results = validate_export("gw_bbh.h5", "selection_bbh.h5")
# Internal checks: array lengths, p_pe/pdraw finite+positive, source≤detector
# masses, format versions.
```

When a selection file is supplied, `validate_export` **cross-validates the PE
export against the selection export** and raises a clear `ValueError` on any
contract mismatch (the point is to stop a file that looks valid while carrying
the wrong prior/cosmology/source class):

- **spin-prior contract** — `spin_prior_mode` must match and the
  `chi_eff_prior_applied_to_p_pe` / `chi_eff_prior_applied_to_pdraw` flags must
  agree, so the chi_eff prior is applied exactly once end-to-end;
- **cosmology** — the selection cosmology must match the PE cosmology; when the
  PE export carries per-event cosmologies (`cosmology_per_event_varies=True`),
  every per-event value is compared against the single selection cosmology
  rather than a legacy scalar;
- **source-class compatibility** — the PE and selection `source_class_filter`
  attrs must resolve to the same canonical class set (the injections must cover
  the same source class(es) as the PE events).

### `merge_store` / `merge_stores` — incremental ingest

```python
from gwcat import merge_store, merge_stores
merge_store("store.h5", new_pe_paths)             # appends PE files in place
merge_store("store.h5", new_paths, out_path="store_v2.h5")  # or to a new file
merge_stores("a.h5", "b.h5", "merged.h5")         # merge two existing stores
```

Skips duplicate event names automatically.  Both merges are
**schema-preserving**: the output holds the *union* of parameters across the
inputs.  A parameter present in only some events (e.g. BNS tidal columns absent
from BBH events) is kept as a full column, NaN-filled for the events that lack
it and marked unavailable in the availability mask — never silently dropped by
intersection.  Meta fields merge as a union too, with explicit-absence defaults
(NaN for floats, `""` for strings).

### `fetch_catalog` — Zenodo downloader

Resolves concept DOIs to the latest version by default.

```python
from gwcat.fetch import fetch_catalog, RELEASES, INJECTION_RELEASES

# PE samples
fetch_catalog("GWTC-2.1")                        # → ./GWTC/GWTC-2p1/
fetch_catalog("GWTC-5")                           # both Part 1 + Part 2

# Injection sets
fetch_catalog("injections-O3-BBH")                # O1+O2+O3 BBH (Zenodo 7890437)
fetch_catalog("injections-O4ab")                  # O4a+b (Zenodo 19500064)
fetch_catalog("injections-O1O2O3O4")              # cumulative O1–O4b (Zenodo 19500052)
```

### Release manifests (`gwcat.manifests`)

Everything `fetch_catalog` needs to know about a release — Zenodo record/concept
IDs, the per-release file-name filter, description, observing run(s) — is
declarative data, not Python: it lives in YAML manifests bundled under
`gwcat/manifests/releases/*.yaml` (PE data releases) and
`gwcat/manifests/injections/*.yaml` (injection/selection sets). `fetch.py`
builds `RELEASES`/`INJECTION_RELEASES` from these manifests at import time; it
contains no per-release data of its own.

```python
from gwcat.manifests import list_releases, get_manifest

list_releases()                    # every bundled release + injection manifest name
manifest = get_manifest("GWTC-5")  # -> ReleaseManifest
manifest.record_ids                # [20348005, 20348006]
manifest.observing_run             # "O4b"
manifest.products["pe_samples"].matches("IGWN-GWTC5-v1-GW240601_000000.hdf5")  # True
```

#### Adding a new release

Adding a new release requires **only a new manifest file** — no changes to
`fetch.py` or any other downloader code:

1. Drop a new YAML file into `gwcat/manifests/releases/` (or `injections/`
   for a selection-function product), following the schema documented in
   `gwcat/manifests/__init__.py` (release, observing_runs, description,
   records, products, metadata, validation).
2. `fetch_catalog("your-new-release")` will pick it up automatically.

`get_manifest()` also accepts a filesystem path to a YAML file directly, so a
manifest can be tried out (or used permanently) without adding it to the
bundled package at all:

```python
get_manifest("/path/to/my_release_manifest.yaml")
```

Manifests are validated on load; `validate_manifest()` raises a
`ManifestValidationError` naming the offending file and field for anything
missing or malformed (required top-level fields, record IDs, product file
filters, metadata/validation blocks).

### `fetch_bbh_list` / `BBH_ALL` — BBH event list

```python
from gwcat.bbh_allowed_names import fetch_bbh_list, refresh_bbh_list, BBH_ALL

fetch_bbh_list()           # live GWOSC query; returns 259 events once GWTC-5.0 indexed
BBH_ALL                    # static fallback: 126 confirmed O1–O4a BBH
refresh_bbh_list()         # print O4b event names to populate BBH_O4B
```

`fetch_bbh_list()` uses the GWOSC v2 API filter `min-mass-2-source=3.0`, which
is the LVK's standard BBH classification boundary. NSBH/BNS events are
excluded automatically without any manual exclusion list.

### Offline mode & metadata provenance

`gwcat.fetch` keeps two concerns separate: **file discovery/download** from
Zenodo (`list_files`, `fetch_catalog`) and **event-metadata discovery** from
GWOSC (`fetch_event_table_gwosc`, `fetch_bbh_names_gwosc`) — see the
`gwcat/fetch.py` module docstring. Both support the same local-cache /
offline-mode contract:

```python
from gwcat.fetch import fetch_catalog, fetch_event_table_gwosc

# Populate the cache while online (writes raw JSON responses, with a fetch
# timestamp, under <cache_dir>/metadata/).
fetch_catalog("GWTC-5", data_dir="./GWTC", cache_dir="./GWTC/.cache")
fetch_event_table_gwosc(cache_dir="./GWTC/.cache")

# Later, fully offline: read ONLY the cache, never touch the network.
fetch_catalog("GWTC-5", data_dir="./GWTC", cache_dir="./GWTC/.cache", offline=True)
fetch_event_table_gwosc(cache_dir="./GWTC/.cache", offline=True)
```

Set `GWCAT_OFFLINE=1` (or `gwcat fetch --offline --cache-dir ...`) to force
offline mode without threading `offline=True` through every call. A cache
miss in offline mode raises `gwcat.fetch_cache.OfflineCacheMissError` naming
the exact missing file — it never silently falls back to a network call.
Omitting `cache_dir`/`offline` entirely (the default) is byte-identical to
pre-existing behavior: no cache is written, nothing changes.

**Metadata assembly + diagnostics** (`gwcat.event_metadata`) layers online
GWOSC metadata, an optional user-override file, and manifest defaults into the
`event_table` dict `build_store` consumes, and records where every field came
from:

```python
from gwcat.event_metadata import (assemble_event_metadata, load_user_overrides,
                                  metadata_diagnostics)
from gwcat.fetch import fetch_event_table_gwosc

online = fetch_event_table_gwosc()
overrides = load_user_overrides("my_overrides.yaml")   # or .csv

event_table, diagnostics = assemble_event_metadata(
    event_names, online_table=online, user_overrides=overrides)
build_store(paths, "store.h5", event_table=event_table)
```

`diagnostics` is a plain, JSON-serializable
`{event_name: {field: {"value": ..., "source": ...}}}` dict — `source` is one
of `"user_override"`, `"online"`, `"manifest"`, or `"absent"` — the raw
provenance record written by the CLI. Pass `--metadata-overrides FILE` to
`gwcat ingest` or to `gwcat fetch --out ...`; YAML and CSV are accepted. The
CLI writes `<out>.metadata_diagnostics.json` by default (override with
`--metadata-diagnostics PATH`) and records the override/diagnostics paths plus
the loaded override count in the validation summary. Override files map
`event_name -> {far, p_astro, source_class, observing_run, ...}`; their values
win over online metadata, and the store's `metadata_source` column reflects
the mix (e.g. `"online+user_override"`, or `"absent"` when nothing was found at
all). FAR genuinely missing for an event is a fully supported, non-crashing
path end to end: `assemble_event_metadata` records `far: absent`, and the
resulting store has `far_available=False` for that event.

`fetch_catalog(provenance={})` populates a `{file_name: {record_id,
file_checksum}}` mapping (sha256, computed once per downloaded/verified file)
that `build_store(file_provenance=...)` uses to fill the per-row `record_id` /
`file_checksum` meta columns — both opt-in and both `""` by default, as
before. `gwcat fetch --write-file-provenance --out store.h5` wires this through
and writes `store.h5.file_provenance.json`; `--file-provenance PATH` selects a
different path and implies collection. Dry runs never write provenance files.

## Zenodo records

These tables mirror the bundled manifests under `gwcat/manifests/releases/`
and `gwcat/manifests/injections/` (see "Release manifests" above) — the
manifests are the source of truth; update them first when a record changes.

### PE samples

| Catalog   | Record(s) | Concept | Run |
|-----------|-----------|---------|-----|
| GWTC-2.1  | [6513631](https://zenodo.org/records/6513631)   | 5117702  | O1+O2+O3a |
| GWTC-3    | [8177023](https://zenodo.org/records/8177023)   | 5546662  | O3b |
| GWTC-4.1  | [20275769](https://zenodo.org/records/20275769) | 20275768 | O4a |
| GWTC-5 P1 | [20348005](https://zenodo.org/records/20348005) | 20276105 | O4b |
| GWTC-5 P2 | [20348006](https://zenodo.org/records/20348006) | 20291739 | O4b |

### Injection / selection sets

| Name | Record | Run |
|------|--------|-----|
| injections-O3-BBH    | [7890437](https://zenodo.org/records/7890437)   | O1+O2+O3 BBH only |
| injections-O4ab      | [19500064](https://zenodo.org/records/19500064) | O4a+b only |
| injections-O1O2O3O4  | [19500052](https://zenodo.org/records/19500052) | O1–O4b cumulative |

For dark-siren cosmology, use `injections-O3-BBH` + `injections-O4ab` combined
via `CombinedSelectionSet`.  The cumulative record (O1O2O3O4) mixes
semi-analytical O1+O2 estimates (no sky location / FAR) with proper O3+O4
injections, making it harder to combine consistently.

## darksirens integration

```
gwcat                              darksirens
─────                              ──────────
 store.h5                              │
   │                                   │
   ├─ to_darksirens()           ──→  load_gw_samples()
   │   allowed_names filter           reads p_pe as-is (do NOT × p_chieff)
   │   p_pe = m1det × p_dL_pe         normalise per event
   │           × p_chieff  ← Mode A   ↓ use PE cosmology
   │   + redshift, m1src, m2src        │
   │   + PE cosmology metadata         │
   │                                   │
   ├─ CombinedSelectionSet        ─→  load_selection_samples()
   │   .to_darksirens()               reads pdraw as-is (do NOT × p_chieff)
   │   O3 + O4 campaigns             no format branching
   │   Essick et al. reweighting      │
   │   6D spin removed,               │
   │   then × p_chieff  ← Mode A      │
   │   Jacobian + time applied        │
   │   FAR cut applied                │
   │                                   ▼
   ├─ validate_export() ───────────  catch mismatches before MCMC
   │                                   │
   └───────────────────────────────  H₀ posterior
```

Source-frame masses and the PE cosmology travel with both export files, so
darksirens never hardcodes a cosmology for the dL→z conversion.

### Prior convention (spin-prior contract)

**Mode A is the default.** gwcat includes the 1-D isotropic chi_eff prior in
*both* exported weights: `p_pe` (PE export) and `pdraw` (selection export).
This is the historical behavior and is byte-for-byte unchanged.

- **What downstream (darksirens) must do in the default (`include`) mode:**
  read `p_pe` / `pdraw` as-is. **Do NOT** multiply the chi_eff prior again —
  it is already baked in. Doing so double-counts the spin prior.
- **How to get the chi_eff prior OUT of `p_pe`:** pass
  `spin_prior_mode="exclude"` to `GWCatalog.to_darksirens(...)`. The exported
  `p_pe` then carries only the mass Jacobian and distance prior, and downstream
  **must** apply the 1-D chi_eff prior itself (exactly once).
- **`passthrough` is intentionally not offered:** the store keeps a
  spin-prior-agnostic `p_dL_pe`, so "no spin-prior manipulation" is identical to
  `exclude` — there is nothing distinct to pass through.

Every export records the contract in HDF5 attributes so it is impossible to
miss and so PE and selection files can be cross-checked:

| Attribute | PE export | Selection export |
|---|---|---|
| `spin_prior_mode` | `include` / `exclude` | `include` (fixed) |
| `chi_eff_prior_applied_to_p_pe` | ✓ | — |
| `chi_eff_prior_applied_to_pdraw` | — | ✓ |
| `mass_jacobian_applied` | ✓ (`True`) | ✓ (`True`) |
| `distance_prior_removed` | ✓ (`False`) | ✓ (`False`) |
| `cosmology_override_used` | ✓ | ✓ |
| `chi_eff_in_p_pe` (legacy) | ✓ | — |
| `chi_eff_swap_applied` (legacy) | — | ✓ (`True`) |

A downstream consumer verifies the two files agree by checking
`spin_prior_mode` matches and that `chi_eff_prior_applied_to_p_pe` equals
`chi_eff_prior_applied_to_pdraw`.

### Cosmology convention (per-event cosmology contract)

The cosmology used to infer source-frame masses and redshifts can differ by
release or sample set, so `GWCatalog.to_darksirens` handles it **per event**:

- **`cosmology=None` (default, per-event mode):** each event independently uses
  **its own** stored PE cosmology (`meta/dL_prior_H0` / `meta/dL_prior_Om0`) for
  the dL→z inversion and source-frame masses. This is correct for
  mixed-release selections whose events were analysed under different
  cosmologies. If any selected event has a missing (`NaN`) or absent stored
  cosmology, the export **fails loudly and names the events** — pass an explicit
  override instead.
- **`cosmology=(H0, Om0)` (override mode):** that single cosmology is applied to
  **all** events, `cosmology_override_used=True` is recorded, and the override
  H0/Om0 are written to the output attrs.

```python
cat.to_darksirens("gw.h5")                        # per-event PE cosmology (default)
cat.to_darksirens("gw.h5", cosmology=(67.74, 0.3089))   # one override for all events
```

Cosmology provenance written to every PE export:

| Attribute | Meaning |
|---|---|
| `cosmology_mode` | `per-event` or `override` |
| `cosmology_override_used` | `True` iff a `cosmology=(H0,Om0)` override was passed |
| `cosmology_per_event_varies` | `True` iff the exported events span >1 cosmology |
| `cosmology_H0_per_event` / `cosmology_Om0_per_event` | per-event arrays aligned with `event_names` |
| `pe_cosmology_H0` / `pe_cosmology_Om0` | scalar (override, or first event) for legacy consumers |
| `source_frame_under_recorded_cosmology` | `True`: source masses/redshift use the recorded cosmology |

The dL→z inverter (`gwcat.cosmology.z_of_dL`) also **never silently clips**
samples beyond its interpolation range: distances past `dL(z=zmax)` extend the
grid (with a warning) instead of being pinned to `zmax`.

> **Migration note.** Earlier versions took the **first** selected event's
> cosmology and applied it to every event. For selections whose events share
> one cosmology (the common case, including all bundled tests) the output is
> byte-identical. For mixed-cosmology selections the numerical output now
> differs — that difference is the bug fix, and it is recorded in the provenance
> attrs above.

### Waveform / sample-set policy (schema 1.2)

A single event can have **multiple posterior sample sets** — e.g. an
`IMRPhenomXPHM` analysis, a `SEOBNRv5PHM` analysis, and a combined `Mixed` set.
gwcat never collapses them without recording what happened.

**Ingest.** `build_store` picks one sample set per event by default
(`sample_sets="preferred"`: the `Mixed`/priority heuristic, unchanged from
before). Pass `sample_sets="all"` to ingest **every** analysis of each PE file
as a separate row, or a list of analysis labels to ingest specific ones. Event
identity stays `event_name`; uniqueness is `(event_name, sample_set_name)`. Each
row records its sample-set provenance in `meta/` columns: `sample_set_name`,
`waveform` (family), `approximant`, `calibration_model`, `is_mixed`,
`is_preferred`, `priority_rank`, `selection_reason`, `file_name`,
`file_checksum`, `record_id`. (`available_parameters`/`sample_count` are **not**
stored — they are already derivable from the availability mask and the offsets
index.)

```python
build_store(paths, "store.h5")                      # one sample set per event
build_store(paths, "store.h5", sample_sets="all")   # every analysis, one row each
```

**Selection / export.** `GWCatalog.select` and `to_darksirens` take a
`waveform_policy=` (and `approximant=`) argument that resolves **which** sample
set represents each event. Except for `all`, the result is guaranteed to hold
exactly one sample set per event.

| `waveform_policy` | Behavior |
|---|---|
| `preferred` (default) | Pick by `is_preferred`, then smallest `priority_rank`. |
| `mixed-first` | Prefer an `is_mixed` set; else fall back to `preferred`. |
| `strict-approximant` | Require `approximant=...` for **every** event; **fail loudly** naming events that lack it. |
| `all` | Keep every sample set; the output is explicitly **not** homogeneous. |

```python
cat.to_darksirens("gw.h5")                                   # preferred (default)
cat.to_darksirens("gw.h5", waveform_policy="mixed-first")
cat.to_darksirens("gw.h5", waveform_policy="strict-approximant",
                  approximant="IMRPhenomXPHM")               # fails if any event lacks it
cat.to_darksirens("gw.h5", waveform_policy="all")            # one event × many sample sets
```

The default `preferred` is a **no-op** for a single-sample-set store (one row
per event, no sample-set columns), so existing stores and exports are unchanged.
Every resolution records a per-row `selection_reason`, and every export writes
the policy provenance:

| Attribute | Meaning |
|---|---|
| `waveform_policy` | The policy applied. |
| `approximant` | The requested approximant (`""` if none). |
| `homogeneous_sample_sets` | `False` iff any event contributes >1 written sample set (only under `all`). |
| `sample_set_name_per_event` / `sample_set_approximant_per_event` | Per-written-row chosen sets, aligned with `event_names`. |
| `sample_set_selection_reason` | Why each row was kept under the active policy. |

> **Schema 1.2.** A store written with sample-set metadata advertises
> `schema_version = "1.2"`. Older stores (1.0/1.1, no sample-set columns) load as
> single-sample-set-per-event and any waveform policy is a no-op.

## Design

- **Store layout**: concatenated 1-D columns + integer `offsets` index. Each
  row is one `(event, sample_set)` pair; a single-sample-set event is one row,
  exactly as before. Schema versions: 1.0 (no availability mask), 1.1 (adds the
  `avail/mask`), 1.2 (adds the per-row sample-set/waveform meta columns).
- **Union parameter schema** (schema 1.1): ingest and merge store the *union*
  of parameters across events, not the intersection. A parameter present for
  only some events is kept as a full column, NaN-filled for the events that lack
  it, and a per-event × per-parameter boolean **availability mask** is stored at
  `avail/mask` (rows aligned with `index/event_names`, columns with
  `attrs/param_names`). Legacy stores (schema 1.0, no mask) still load: every
  stored column is treated as available for every event, which is exact because
  the old intersection ingest guaranteed it. See `gwcat.schema` for the
  parameter groups (`core_intrinsic`, `core_extrinsic`, `spin`, `bns_nsbh`,
  `diagnostic`) and the per-export required-parameter lists.
- **Required vs optional parameters**: `GWCatalog.get(params)` raises a clear,
  named `MissingParameterError` (a `KeyError` subclass) when a required
  parameter is absent — pass `required=False` for a NaN-filled column plus
  `param_available(param)` availability. Exports declare their required columns
  (`gwcat.schema.EXPORT_REQUIREMENTS`); `to_darksirens` fails loudly, naming the
  parameter **and** the offending event(s), if a required column is absent from
  the store or NaN-filled for a selected event.
- **Mass-prior-agnostic store**: keeps `p_dL_pe` + its cosmology.
  The mass Jacobian is applied only in `to_darksirens`
  (`_to_darksirens_format` is a deprecated alias kept for compatibility).
- **BBH selection via `allowed_names`**: the recommended way to select BBH
  events is via `fetch_bbh_list()` (GWOSC live) or the static `BBH_ALL`,
  rather than FAR/p_astro cuts that require a separately fetched event table.
  `allowed_names` is treated as authoritative by default, so a derived
  `compact_type="BBH"` metadata cut is not required. If both are supplied,
  gwcat warns about whitelist events that the compact-type cut would exclude.
- **Sky area at ingest** (optional): 90% credible area computed via HEALPix
  and stored as `sky_area_90` metadata, enabling `select(sky_area_max=...)`.
  Requires `healpy`; NaN if absent.
- **Incremental updates**: `merge_store` appends new events without
  re-ingesting the full catalog.

## Configuration (`IngestConfig`)

- `o4_waveform_priority`: fallback order when a GWTC-5 event has no `C00:Mixed`.
- `o3_default_cosmo`: assumed for GWTC-2.1 (no analytic prior); KS-validated.
- `nsbh_mass_threshold`: source-frame mass cut for BBH/NSBH/BNS classification.

## Known limitations

- **FAR / p_astro** are not in per-event PE files.  `build_store` auto-fetches
  them from GWOSC (requires network).  Pass `event_table={}` or `--no-event-table`
  to skip.  Use `allowed_names` / `fetch_bbh_list()` instead of FAR cuts for
  BBH selection — this is more robust and does not depend on the event table.
- **GWTC-5.0 GWOSC indexing**: the GWOSC v2 API was not yet updated to include
  GWTC-5.0 events at the time of the May 2026 paper release. `fetch_bbh_list()`
  will return the full 259-event list once the API is updated (expected within
  weeks). Until then, `BBH_ALL` covers the 126 confirmed O1–O4a BBH.
- **`mass_prior_kind`** is assumed `uniform_detector_frame`.
- **Ingest requires `pesummary`** (`pip install gwcat[ingest]`).
- **healpy** is optional; without it `sky_area_90` is NaN.
