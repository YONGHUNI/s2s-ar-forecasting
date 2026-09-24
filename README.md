# S2S Atmospheric River Forecasting

UGA-side AI/data-driven modeling workflow for the NASA ROSES-2025 Water Resources project:

> **Improving Subseasonal-to-Seasonal Forecasts of Atmospheric Rivers in the Western U.S. through Temporal Clustering Analysis, Orientation Prediction, and Hybrid Modeling Approaches**

The current implementation focuses on reproducible GraphCast Operational experiments on UGA GACRC Sapelo2. Pangu-Weather and SFNO environments are also retained in `pixi.toml` for later model intercomparison work.

## Current GraphCast experiment

The production experiment is defined in `config/graphcast_operational.yaml`.

- Production period: 2015-10-01 through 2025-04-30
- Seasonal filter: October through April only
- Initialization days: Tuesday and Friday
- Total initializations: 607
- Seasons: 2015-16 through 2024-25
- Forecast lead: 42 days
- Temporal resolution: 6 hours
- Lead times per forecast: 169, including lead time 0
- Variables: 70
  - Surface: `msl`, `t2m`, `tp06`, `u10m`, `v10m`
  - Pressure levels: 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa
  - Pressure-level fields: temperature (`t`), zonal wind (`u`), meridional wind (`v`), specific humidity (`q`), geopotential (`z`)

The YAML includes `validation.expected_initializations: 607`. Configuration loading fails before any forecast is launched if the date, weekday, or month filters no longer produce exactly 607 initialization dates.

The 42-day rollout is an experimental S2S evaluation setup. Later-lead skill must be evaluated rather than assumed from GraphCast's conventional medium-range use.

## Optimized production pipeline

GPU inference and NetCDF conversion are separated so compression does not keep the L40S GPUs idle.

```text
hu_p GPU job (2 x L40S)
  producer 0 ─ GraphCast ─┐
                           ├─> async Zarr writes directly to /scratch staging
  producer 1 ─ GraphCast ─┘                         + .ready marker
                                                    |
                                                    v
batch CPU job
  converter manager
    └─ bounded ProcessPoolExecutor (12 workers)
         ├─ worker 1  ─ Zarr -> NetCDF (zlib level 2 + shuffle)
         ├─ worker 2  ─ Zarr -> NetCDF
         ├─ ...
         └─ worker 12 ─ Zarr -> NetCDF

  each worker:
    ─ reads staged Zarr directly from /scratch
    ─ writes NetCDF synchronously
    ─ atomically publishes .partial -> .nc
    ─ manager removes staged Zarr + .ready after success
```

The production producer uses Earth2Studio's `AsyncZarrBackend` with a four-thread I/O pool. Writes go directly to a temporary Zarr under shared `/scratch`, pending writes are drained before the backend's per-initialization event loops are stopped and closed, and the completed store is atomically renamed before the `.ready` marker is created.

Lead-time sharding remains disabled (`shard_lead_times: 1`). Benchmarking showed that four-lead-time sharding did not improve wall-clock time for this workload.

The converter runs independently on a CPU node as one manager process with a bounded process pool. The manager alone scans `.ready` markers and submits each ready Zarr to at most one worker. Production is configured for 12 converter workers to provide throughput headroom relative to the two GPU producers. Actual aggregate conversion throughput depends on the CPU node assigned by Slurm and on concurrent Lustre I/O.

Sapelo2 benchmarking showed that reading staged Zarr directly from shared `/scratch` is preferable to first copying it to node-local `/lscratch` for this conversion path. The reusable conversion logic is isolated in `scripts/netcdf_conversion.py`.

NetCDF production uses zlib compression level 2 with shuffle. A single compression worker on an AMD EPYC TurinDense node converted one approximately 45.8 GiB uncompressed NetCDF in about 15 minutes 11 seconds and peaked at about 1.6 GiB resident memory. Production therefore uses process-level parallelism rather than trying to multithread a single netCDF4/zlib encoding task.

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
```

Each submission creates `logs/YYYYMMDD_HHMMSS/` under the repository root. Forecast and converter stdout/stderr are combined into `graphcast-forecast-<jobid>.log` and `graphcast-convert-<jobid>.log`; Python status lines also include wall-clock timestamps and the Slurm job ID.

This submits:

1. `slurm/run_graphcast_forecast.slurm`: two GraphCast GPU producers on `hu_p`.
2. `slurm/run_graphcast_convert.slurm`: one CPU-side converter manager on `batch`, with 12 converter worker processes.

The converter job uses Slurm's `after` dependency so it becomes eligible after the producer job starts rather than waiting for all GPU inference to finish. The submission wrapper also passes the producer job ID to the converter; once the producer has left Slurm's active queue and the converter has no active or ready work remaining, the converter exits instead of polling indefinitely.

Manual submission:

```bash
producer_job=$(sbatch --parsable slurm/run_graphcast_forecast.slurm)
producer_job=${producer_job%%;*}

sbatch \
  --dependency="after:${producer_job}" \
  --export="ALL,CONFIG=config/graphcast_operational.yaml,PRODUCER_JOB_ID=${producer_job}" \
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

A Zarr forecast is eligible for conversion only after its direct-staging write finishes and the corresponding `.ready` marker is created. NetCDF publication uses a `.partial` file followed by an atomic rename.

On restart:

- Existing final NetCDF files are skipped.
- A staged Zarr with a `.ready` marker is reused rather than recomputed.
- Incomplete local or partial products are cleaned and regenerated.

`/scratch` is temporary cluster storage. Outputs needed for long-term retention should be transferred to persistent or archival storage.

## Configuration

Research settings live in YAML rather than in the Python scripts:

```yaml
experiment:
  start_date: "2015-10-01"
  end_date: "2025-04-30"
  init_weekdays: [Tuesday, Friday]
  init_months: [10, 11, 12, 1, 2, 3, 4]
  forecast_days: 42
  hours_per_step: 6

output:
  compression_level: 2
  zarr_backend: async
  write_mode: direct_staging
  async_pool_size: 4
  shard_lead_times: 1

converter:
  poll_seconds: 5
  workers: 12

validation:
  expected_variable_count: 70
  expected_initializations: 607
```

Slurm resources remain in the `.slurm` files because they describe scheduler resources rather than the experiment itself.

Forecast job:

```text
2 x L40S
2 tasks
8 CPUs per task
128 GiB RAM
30 hours walltime
```

Converter job:

```text
12 CPUs
48 GiB RAM
36 hours walltime
```

The converter memory and walltime include deliberate headroom to reduce the risk that a long 607-initialization production run is lost to an OOM or modestly slower-than-benchmarked conversion throughput.

## Repository layout

```text
.
├── config/
│   └── graphcast_operational.yaml
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
