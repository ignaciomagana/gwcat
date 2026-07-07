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

```bash
gwcat-fetch --out store.h5                            # download all PE + build (FAR auto-fetched)
gwcat-fetch --out store.h5 --no-event-table           # skip GWOSC FAR/p_astro fetch
gwcat-fetch --catalog injections-O3-BBH               # O3 BBH injection set
gwcat-fetch --catalog injections-O4ab                 # O4a+b injection set
gwcat-fetch --catalog all                             # PE + all injection sets
gwcat-fetch --catalog GWTC-5 --dry-run                # preview files
gwcat-fetch --no-resolve                              # skip concept DOI resolution
```

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

### `SelectionSet` — injection processing

Reads both LVK injection formats:
- **`events/` format** (O4 sets, Zenodo 19500064): modern format with log-joint draw PDF
- **`injections/` format** (O3 BBH, Zenodo 7890437): legacy format with factored draw PDF components

```python
sel = SelectionSet("injection_file.hdf", H0=67.74, Om0=0.3089)
sel.n_injections                                  # total in file
sel.detection_efficiency(far_threshold=1.0)       # fraction detected
sel.to_darksirens("selection.h5", far_threshold=1.0)
```

### `CombinedSelectionSet` — multi-campaign combination

Combines injection sets from different observing runs following
Essick et al. (2023), with proper ndraw reweighting:

```python
sel_o3 = SelectionSet("endo3_bbhpop-LIGO-T2100113-v12.hdf5")
sel_o4 = SelectionSet("injections-O4ab/...-cartesian_spins_*.hdf")
combined = CombinedSelectionSet([sel_o3, sel_o4])
combined.to_darksirens("selection_bbh.h5", far_threshold=1.0)
```

### `validate_export` — pre-flight checks

```python
from gwcat import validate_export
results = validate_export("gw_bbh.h5", "selection_bbh.h5")
# Checks: array lengths, p_pe/pdraw finite+positive, source≤detector masses,
# format versions, cosmology consistency between PE and selection files.
```

### `merge_store` — incremental ingest

```python
from gwcat import merge_store
merge_store("store.h5", new_pe_paths)             # appends in place
merge_store("store.h5", new_paths, out_path="store_v2.h5")  # or to a new file
```

Skips duplicate event names automatically.

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

## Zenodo records

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
gwcat                             darksirens
─────                             ──────────
 store.h5                             │
   │                                  │
   ├─ to_darksirens()          ──→  load_gw_samples()
   │   allowed_names filter          × p_chieff (gwdistributions)
   │   p_pe = m1det × p_dL_pe        normalise per event
   │   + redshift, m1src, m2src      ↓ use PE cosmology
   │   + PE cosmology metadata        │
   │                                  │
   ├─ CombinedSelectionSet       ─→  load_selection_samples()
   │   .to_darksirens()              × p_chieff (gwdistributions)
   │   O3 + O4 campaigns            no format branching
   │   Essick et al. reweighting     │
   │   6D spin removed               │
   │   Jacobian + time applied       │
   │   FAR cut applied               │
   │                                  ▼
   ├─ validate_export() ──────────  catch mismatches before MCMC
   │                                  │
   └──────────────────────────────  H₀ posterior
```

The 1-D chi_eff spin-prior swap lives exclusively in darksirens (via
gwdistributions).  gwcat stays spin-prior-agnostic.

Source-frame masses and the PE cosmology travel with both export files, so
darksirens never hardcodes a cosmology for the dL→z conversion.

## Design

- **Store layout**: concatenated 1-D columns + integer `offsets` index.
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
