# S2S Atmospheric River Forecasting

UGA-side AI/data-driven modeling workflow for the NASA ROSES-2025 Water Resources project:

> **Improving Subseasonal-to-Seasonal Forecasts of Atmospheric Rivers in the Western U.S. through Temporal Clustering Analysis, Orientation Prediction, and Hybrid Modeling Approaches**

The current implementation focuses on reproducible GraphCast Operational experiments on UGA GACRC Sapelo2. Pangu-Weather and SFNO environments are also retained in `pixi.toml` for later model intercomparison work.

## Current GraphCast experiment

The experiment is defined in `config/graphcast_operational.yaml`.

- Production initialization period: 2024-10-01 through 2025-04-30
- Development config: `config/graphcast_test10.yaml` (10 initializations, 2024-10-01 through 2024-11-01)
- Initialization days: Tuesday and Friday
- Forecast lead: 42 days
- Temporal resolution: 6 hours
- Lead times per forecast: 169, including lead time 0
- Variables: 70
  - Surface: `msl`, `t2m`, `tp06`, `u10m`, `v10m`
  - Pressure levels: 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa
  - Pressure-level fields: temperature (`t`), zonal wind (`u`), meridional wind (`v`), specific humidity (`q`), geopotential (`z`)

The 42-day rollout is an experimental S2S evaluation setup. Later-lead skill must be evaluated rather than assumed from GraphCast's conventional medium-range use.

## Pipeline

GPU inference and NetCDF conversion are separated so compression does not keep the L40S GPUs idle.

```text
hu_p GPU job (2 x L40S)
  producer 0 ─ GraphCast ─ Zarr ─┐
  producer 1 ─ GraphCast ─ Zarr ─┼─> /scratch staging + .ready marker
                                  │
                                  v
batch CPU job
  converter 0/1 ─ read staged Zarr directly from /scratch
                ─ synchronous NetCDF4 write
                ─ atomic .partial -> .nc publish on /scratch
                ─ remove staged Zarr
```

The producer keeps the GPU allocation focused on inference. The converter runs independently on regular CPU nodes. Sapelo2 benchmarking showed that the tested batch node could read the staged Zarr directly from shared `/scratch` faster than copying it to node-local `/lscratch`, so the converter now reads staging in place. Compression level 0 is used by the 10-initialization development config because full-file uncompressed conversion completed in about 1.5 minutes, whereas zlib level 1 was much slower.

## Environment

Nix provides Pixi; Pixi manages Python and the weather-model environments.

```bash
nix develop
pixi install -e graphcast
```

On Sapelo2, the batch scripts assume the rootless Nix bootstrap is available at:

```text
~/rootless-nix-bootstrap
```

with the wrapper at:

```text
~/.local/bin/nix
```

The Nix shell keeps Pixi environments and transient caches on `/lscratch`, while the Earth2Studio model cache is kept under `/work/whlab/$USER`.

## Run on Sapelo2

Submit from the repository root:

```bash
bash submit_graphcast_pipeline.sh

# 10-initialization development run
CONFIG=config/graphcast_test10.yaml bash submit_graphcast_pipeline.sh
```

This submits:

1. `slurm/run_graphcast_forecast.slurm`: two GraphCast GPU producers on `hu_p`.
2. `slurm/run_graphcast_convert.slurm`: two NetCDF CPU converters on `batch`.

The converter job uses Slurm's `after` dependency so it becomes eligible after the producer job starts rather than waiting for all GPU inference to finish.

Manual submission:

```bash
producer_job=$(sbatch --parsable slurm/run_graphcast_forecast.slurm)
producer_job=${producer_job%%;*}

sbatch \
  --dependency="after:${producer_job}" \
  slurm/run_graphcast_convert.slurm
```

## Storage and restart behavior

Default paths:

```text
/lscratch/$USER/graphcast-operational/
    node-local producer workspace and Nix/Pixi caches

/scratch/$USER/weather-ai/graphcast-operational/.staging/
    shared Zarr handoff

/scratch/$USER/weather-ai/graphcast-operational/
    final NetCDF files
```

A Zarr forecast is eligible for conversion only after its shared copy finishes and the corresponding `.ready` marker is created. NetCDF publication uses a `.partial` file followed by an atomic rename.

On restart:

- Existing final NetCDF files are skipped.
- A staged Zarr with a `.ready` marker is reused rather than recomputed.
- Incomplete local or partial products are cleaned and regenerated.

`/scratch` is temporary cluster storage. Outputs needed for long-term retention should be transferred to persistent or archival storage.

## Configuration

Research settings live in YAML rather than in the Python scripts:

```yaml
experiment:
  start_date: "2024-10-01"
  end_date: "2025-04-30"
  init_weekdays: [Tuesday, Friday]
  forecast_days: 42
  hours_per_step: 6

output:
  compression_level: 1
```

The YAML controls dates, forecast length, variables, paths, and NetCDF compression. Slurm resources remain in the `.slurm` files because they describe scheduler resources rather than the experiment itself.

## Repository layout

```text
.
├── config/
│   ├── graphcast_operational.yaml
│   └── graphcast_test10.yaml
├── scripts/
│   ├── graphcast_config.py
│   ├── run_graphcast_forecast.py
│   └── convert_graphcast_netcdf.py
├── slurm/
│   ├── run_graphcast_forecast.slurm
│   └── run_graphcast_convert.slurm
├── submit_graphcast_pipeline.sh
├── flake.nix
├── flake.lock
├── pixi.toml
└── pixi.lock
```

`flake.lock` and `pixi.lock` are committed for reproducibility. Generated Pixi environments, Slurm logs, Zarr stores, NetCDF files, partial files, and ready markers are ignored.
