# gwcat

Fast preprocessing of GWTC-2.1 / 3 / 4.1 / 5 PE files and LVK injection sets
into the formats consumed by [darksirens](https://github.com/ignaciomagana/darksirens),
with automated Zenodo fetching.

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

```python
from gwcat import GWCatalog, SelectionSet, CombinedSelectionSet
from gwcat import build_store, validate_export
from gwcat.fetch import fetch_catalog

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

# ── 4. Explore the catalog ───────────────────────────────────
cat = GWCatalog("store.h5")
cat.summary()                             # compact event table
bbh = cat.select(
    compact_type="BBH",
    far_max=1.0,                          # match GWTC-5 populations paper
)
print("Selected " + str(bbh.n_events) + " events")

# ── 5. Export PE samples for darksirens ──────────────────────
bbh._to_darksirens_format(
    "gw_bbh.h5",
    nsamp=4096,
    z_max=10.0,                            # per-sample redshift cut
    cosmology=(67.74, 0.3089),
    seed=0,
)

# ── 6. Export combined selection function ─────────────────────
sel_o3 = SelectionSet("injections/injections-O3-BBH/endo3_bbhpop-LIGO-T2100113-v12.hdf5")
sel_o4 = SelectionSet("injections/injections-O4ab/mixture-real_o4a_o4b-cartesian_spins.hdf")
combined = CombinedSelectionSet([sel_o3, sel_o4])
combined.to_darksirens("selection_bbh.h5", far_threshold=1.0)

# ── 7. Validate before running darksirens ────────────────────
validate_export("gw_bbh.h5", "selection_bbh.h5")

# ── 8. Run darksirens ────────────────────────────────────────
# from darksirens.gw.utils import load_gw_samples, load_selection_samples
# m1, m2, dL, chieff, ra, dec, p_pe, nEvents, nsamp = load_gw_samples("gw_bbh.h5")
# m1s, m2s, dLs, chis, ras, decs, pdraw, ndraw = load_selection_samples("selection_bbh.h5")
```

## Incremental updates with `merge_store`

When a new catalog drops, append without re-ingesting the full set:

```python
from gwcat import merge_store

new_paths = fetch_catalog("GWTC-5", data_dir="./GWTC")
merge_store("store.h5", new_paths)   # duplicates auto-skipped, FAR auto-fetched
```

## Quick start (CLI)

```bash
gwcat-fetch --out store.h5                            # download all PE + build (FAR auto-fetched)
gwcat-fetch --out store.h5 --no-event-table           # skip GWOSC FAR/p_astro fetch
gwcat-fetch --catalog injections-O3-BBH               # O3 BBH injection set
gwcat-fetch --catalog injections-O4ab                  # O4a+b injection set
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
    compact_type="BBH",
    pastro_min=0.9,
    far_max=1.0,
    snr_min=8.0,
    m1_src_range=(5, 100),
    sky_area_max=500,                             # requires healpy at ingest
)

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
sel.detection_efficiency(far_threshold=1.0)        # fraction detected
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
fetch_catalog("injections-O4ab")                   # O4a+b (Zenodo 19500064)
fetch_catalog("injections-O1O2O3O4")              # cumulative O1–O4b (Zenodo 19500052)
```

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
   ├─ _to_darksirens_format()  ──→  load_gw_samples()
   │   p_pe = m1det × p_dL_pe       × p_chieff (gwdistributions)
   │   + redshift, m1src, m2src      normalise per event
   │   + PE cosmology metadata       ↓ use PE cosmology
   │                                  │
   ├─ CombinedSelectionSet       ─→  load_selection_samples()
   │   .to_darksirens()              × p_chieff (gwdistributions)
   │   O3 + O4 campaigns            no format branching
   │   Essick et al. reweighting     │
   │   6D spin removed               │
   │   Jacobian + time applied        │
   │   FAR cut applied                │
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
  The mass Jacobian is applied only in `_to_darksirens_format`.
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
  to skip.
- **`mass_prior_kind`** is assumed `uniform_detector_frame`.
- **Ingest requires `pesummary`** (`pip install gwcat[ingest]`).
- **healpy** is optional; without it `sky_area_90` is NaN.