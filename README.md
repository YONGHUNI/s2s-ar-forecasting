# S2S Atmospheric River Forecasting

UGA-side AI/data-driven modeling workflow for the NASA ROSES-2025 Water Resources project:

> **Improving Subseasonal-to-Seasonal Forecasts of Atmospheric Rivers in the Western U.S. through Temporal Clustering Analysis, Orientation Prediction, and Hybrid Modeling Approaches**

The current implementation focuses on reproducible GraphCast Operational experiments on UGA GACRC Sapelo2. Pangu-Weather and SFNO environments are also retained in `pixi.toml` for later model intercomparison work.

## Current GraphCast experiment

The experiment is defined in `config/graphcast_operational.yaml`.

- Production initialization period: 2024-10-01 through 2025-04-30
- Baseline development config: `config/graphcast_test10.yaml` (10 initializations, synchronous Zarr)
- Async-I/O development config: `config/graphcast_test10_async.yaml` (same 10 initializations, AsyncZarrBackend)
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
  converter manager
    ├─ scans shared .ready queue on /scratch
    ├─ process 1 ─ Zarr -> NetCDF
    └─ process 2 ─ Zarr -> NetCDF

  each worker:
    ─ reads staged Zarr directly from /scratch
    ─ writes NetCDF synchronously
    ─ atomically publishes .partial -> .nc
    ─ manager removes staged Zarr + .ready after success
```

The producer can use either the synchronous `ZarrBackend` or Earth2Studio's `AsyncZarrBackend`. The async test configuration uses non-blocking writes with a four-thread I/O pool so per-step Zarr writes can overlap model execution. It explicitly calls `close()` before staging so all pending writes are drained before the Zarr is published. Lead-time sharding is currently disabled (`shard_lead_times: 1`) so the first A/B test changes only the synchronous-versus-asynchronous write behavior.

The producer keeps the GPU allocation focused on inference. The converter runs independently on a regular CPU node as one manager process with a bounded process pool. The manager alone scans `.ready` markers and submits each ready Zarr to at most one worker, which avoids duplicate work inside the job while keeping two independent conversions in flight when data are available. If both workers are busy, additional ready Zarr stores remain queued on the shared filesystem until a slot opens.

Sapelo2 benchmarking showed that the tested batch node could read staged Zarr directly from shared `/scratch` faster than copying it to node-local `/lscratch`, so converter workers read staging in place. The reusable conversion logic is isolated in `scripts/netcdf_conversion.py`; the GraphCast-specific manager only handles discovery, queueing, validation expectations, cleanup, and restart behavior. Compression level 0 is used by the 10-initialization development config because full-file uncompressed conversion was dramatically faster than zlib level 1 in testing.

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

# 10-initialization synchronous baseline
CONFIG=config/graphcast_test10.yaml bash submit_graphcast_pipeline.sh

# same 10 initializations with asynchronous Zarr writes
CONFIG=config/graphcast_test10_async.yaml bash submit_graphcast_pipeline.sh
```

This submits:

1. `slurm/run_graphcast_forecast.slurm`: two GraphCast GPU producers on `hu_p`.
2. `slurm/run_graphcast_convert.slurm`: one CPU-side converter manager on `batch`, with a configurable local process pool.

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
  zarr_backend: sync
  async_pool_size: 4
  shard_lead_times: 1

converter:
  poll_seconds: 5
  workers: 2
```

The YAML controls dates, forecast length, variables, paths, and NetCDF compression. Slurm resources remain in the `.slurm` files because they describe scheduler resources rather than the experiment itself.

## Repository layout

```text
.
├── config/
│   ├── graphcast_operational.yaml
│   ├── graphcast_test10.yaml
│   └── graphcast_test10_async.yaml
├── scripts/
│   ├── graphcast_config.py
│   ├── netcdf_conversion.py
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
