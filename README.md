# S2S Atmospheric River Forecasting

UGA-side AI/data-driven modeling workflow for the NASA ROSES-2025 Water Resources project:

> **Improving Subseasonal-to-Seasonal Forecasts of Atmospheric Rivers in the Western U.S. through Temporal Clustering Analysis, Orientation Prediction, and Hybrid Modeling Approaches**

The repository supports GraphCast Operational and a Pangu6 pipeline for UGA GACRC Sapelo2. SFNO remains available as a Pixi feature for later evaluation.

## Current GraphCast experiment

The production experiment is defined in `config/graphcast_operational_deterministic.yaml`.

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

GPU inference and NetCDF conversion are separated so compression does not keep H100 GPUs idle.

```text
gpu_p producer array: 20 fixed logical shards, at most 8 concurrent tasks
  array task 0  ─ 1 x H100 ─ GraphCast ─┐
  array task 1  ─ 1 x H100 ─ GraphCast ─┤
  ...                                    ├─> async Zarr writes directly to /scratch staging
  array task 19 ─ 1 x H100 ─ GraphCast ─┘                         + .ready marker
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

The converter runs independently on a CPU node as one manager process with a bounded process pool. The manager alone scans `.ready` markers and submits each ready Zarr to at most one worker. Production is configured for 12 converter workers; deterministic H100 inference is slow enough that this provides throughput headroom even when up to eight producer array tasks run concurrently. Actual aggregate conversion throughput still depends on the CPU node assigned by Slurm and on concurrent Lustre I/O.

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

The Nix shell keeps Pixi environments and node-local transient caches on `/lscratch`, while the Earth2Studio model cache is kept under `/work/whlab/$USER`. GraphCast Slurm jobs override the default JAX cache location with a shared `/scratch/$USER/weather-ai/jax-cache/<gpu-model>/` directory so persistent XLA compilation artifacts can be reused across jobs on different compute nodes. The GPU-model suffix is derived automatically from `nvidia-smi`, so new GPU types receive their own cache namespace without hard-coded model lists.

## Run on Sapelo2

Submit from the repository root:

```bash
bash submit_graphcast_pipeline.sh
```

Each submission creates `logs/YYYYMMDD_HHMMSS/` under the repository root. Forecast array stdout/stderr are combined per task in `graphcast-forecast-<array_jobid>_<taskid>.log`, while converter output is written to `graphcast-convert-<jobid>.log`; Python status lines also include wall-clock timestamps and the Slurm job ID.

This submits:

1. `slurm/run_graphcast_forecast.slurm`: a `gpu_p` job array with 20 fixed logical shards (`0-19%8`), one H100 per array task, and at most eight concurrent producer tasks.
2. `slurm/run_graphcast_convert.slurm`: one CPU-side converter manager on `batch`, with 12 converter worker processes.

The converter is submitted immediately after the producer array rather than depending on the whole array's `after` condition. This lets NetCDF conversion begin as soon as `.ready` Zarr stores appear instead of waiting for every array element to start. The submission wrapper passes the producer array's base job ID to the converter. The converter checks Slurm's array base-job field and exits only after the producer array has left the active queue and no active or ready conversion work remains. Because the CPU allocation can start long before scarce H100 producer tasks, the converter also has a lifecycle requeue guard: after 46 hours of a 48-hour allocation it stops accepting new conversions, drains any active workers, and requests a Slurm requeue if the producer is still active or ready conversion work remains. A normal completed lifecycle exits without requeue.

Manual submission:

```bash
producer_job=$(sbatch --parsable slurm/run_graphcast_forecast.slurm)
producer_job=${producer_job%%;*}

sbatch \
  --export="ALL,CONFIG=config/graphcast_operational_deterministic.yaml,PRODUCER_JOB_ID=${producer_job}" \
  slurm/run_graphcast_convert.slurm
```

## Storage and restart behavior

Default paths:

```text
/lscratch/$USER/graphcast-operational-deterministic/
    node-local producer workspace

/scratch/$USER/weather-ai/graphcast-operational-deterministic/.staging/
    shared Zarr handoff

/scratch/$USER/weather-ai/graphcast-operational-deterministic/
    final NetCDF files
```

A Zarr forecast is eligible for conversion only after its direct-staging write finishes and the corresponding `.ready` marker is created. NetCDF publication uses a `.partial` file followed by an atomic rename.

On restart:

- Existing final NetCDF files are skipped.
- A staged Zarr with a `.ready` marker is reused rather than recomputed.
- Incomplete local or partial products are cleaned and regenerated.
- The logical shard count remains fixed at 20 even when only a subset of array indices is resubmitted. This preserves the original date-to-shard mapping.

If one array task fails while the original array is still recoverable, requeue that element with:

```bash
scontrol requeue <array_job_id>_<task_id>
```

The producer is submitted with `--requeue`, which also makes it eligible for Slurm requeue after supported infrastructure events such as node failure or preemption. This does not automatically retry arbitrary application failures.

If the original array is no longer active, a missing shard can be submitted by itself without changing its logical assignment:

```bash
sbatch --array=<task_id> slurm/run_graphcast_forecast.slurm
```

Command-line `--array` overrides the script's default `0-19%8` submission range, while the Python worker still receives `--num-workers 20`. If the previous converter has already exited, start a converter for the replacement producer as well.

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
  poll_seconds: 30
  workers: 12

validation:
  expected_variable_count: 70
  expected_initializations: 607
```

Slurm resources remain in the `.slurm` files because they describe scheduler resources rather than the experiment itself.

Forecast array task:

```text
1 x H100
1 task
8 CPUs
128 GiB RAM
10 hours walltime
20 fixed logical shards
maximum 8 concurrent array tasks
```

Converter job:

```text
12 CPUs
48 GiB RAM
48 hours walltime
46-hour graceful lifecycle requeue threshold
```

The producer shard count is intentionally fixed at 20 so restart/resume runs preserve the same date-to-worker mapping. The converter memory and walltime include deliberate headroom for queueing variation and slower-than-benchmarked conversion throughput.

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

## Pangu6 pipeline (pilot validation before production)

Pangu follows the same architecture as GraphCast: one Nix flake and one root
Pixi workspace with independent `-e pangu` and `-e graphcast` environments;
per-node Nix/Pixi state under `/lscratch`; GPU producer array and concurrent CPU
NetCDF converter; shared Zarr staging with `.ready` markers; atomic final output;
and restart-safe skipping of finished NetCDF files.

The validated single-initialization Pangu6 rollout uses 168 six-hour forecast
steps (169 lead times), with 69 variables and 721 x 1440 grid points. Pangu6
does not produce GraphCast's `tp06` precipitation variable.

The only root submission entrypoint is:

```bash
# Initial production-architecture test: 2024-10-01 only, array task 0
bash submit_pangu_pipeline.sh

# Full 607-initialization production after pilot validation
CONFIG=config/pangu_operational.yaml bash submit_pangu_pipeline.sh
```

**Do not start the full 607-initialization run until the pilot producer and CPU
converter both finish successfully.** The default pilot configuration stages
data in a separate `pangu-pilot` output tree; no GraphCast output paths are
modified. Default Slurm scheduling is `hu_p` L40S for forecast and `batch`
for CPU conversion. Production uses 20 fixed logical shards with a maximum
of 4 concurrent GPU array tasks. Experiment variables, dates, and output roots
are managed by the model-specific YAML files.

The root `pixi.lock` currently contains Earth2Studio 0.18.0 in `-e pangu`
but not `netcdf4` in that environment. Therefore the **model-independent**
CPU NetCDF converter temporarily uses the existing root `-e graphcast` Pixi
environment (which already includes NetCDF4). GPU forecasts use `-e pangu`.
A future common I/O feature can replace this without maintaining an independent
Pangu `pixi.toml` or `pixi.lock`. The existing NetCDF conversion utility
adds metadata for variables shared by GraphCast and Pangu; scientific attribute
validation is still needed before publishing all 607 products.

Slurm initialization uses the existing rootless-Nix bootstrap. A batch CPU
node whose DNS cannot resolve `releases.nixos.org` may still fail before the
conversion starts; this is an unresolved infrastructure preflight and must be
checked with the pilot job.

After validation, the default can be switched from pilot to operational.

## Shared implementation (October 2026)

Both GraphCast and Pangu now use the common Python entrypoints `scripts/run_forecast.py`, `scripts/forecast_config.py`, and `scripts/convert_netcdf.py`. The model is selected using each configuration file's `model.name`; Slurm jobs continue to use their own model-specific Pixi environments and resource requests. The model-specific submit scripts remain separate. NetCDF I/O logic stays in `scripts/netcdf_conversion.py`. The GraphCast QC module remains model-specific.

The refactor was checked for Slurm references and model dispatch statically; a post-refactor compute-node smoke test has not yet been run. Do not launch the 607-case operation until that smoke test passes.
