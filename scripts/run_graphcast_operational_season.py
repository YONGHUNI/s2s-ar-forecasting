#!/usr/bin/env python3

import argparse
import os
import shutil
import subprocess
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

from earth2studio import run
from earth2studio.data import ARCO
from earth2studio.io import ZarrBackend
from earth2studio.models.px import GraphCastOperational


# ============================================================
# Command-line arguments
# ============================================================

parser = argparse.ArgumentParser(
    description="Run 42-day GraphCast Operational forecasts."
)

parser.add_argument(
    "--worker",
    type=int,
    required=True,
    help="Worker index, starting from 0.",
)

parser.add_argument(
    "--num-workers",
    type=int,
    default=1,
    help="Total number of parallel workers.",
)

args = parser.parse_args()

WORKER = args.worker
NUM_WORKERS = args.num_workers

if NUM_WORKERS < 1:
    raise ValueError("--num-workers must be >= 1")

if not 0 <= WORKER < NUM_WORKERS:
    raise ValueError(
        f"--worker must satisfy 0 <= worker < {NUM_WORKERS}"
    )


# ============================================================
# Experiment configuration
# ============================================================

START_DATE = date(2024, 10, 1)
END_DATE = date(2025, 4, 30)

FORECAST_DAYS = 42
HOURS_PER_STEP = 6
STEPS = FORECAST_DAYS * 24 // HOURS_PER_STEP

# Earth2Studio also writes the initial state (lead_time = 0),
# therefore the expected output length is 168 + 1 = 169.
EXPECTED_LEAD_TIMES = STEPS + 1

COMPRESSION_LEVEL = 2

USER = os.environ["USER"]

# Each Slurm worker gets 8 CPUs in the proposed job script.
NC_WORKERS = int(os.environ.get("NC_WORKERS", "8"))

# Fast, node-local temporary storage
WORK_DIR = Path(
    f"/lscratch/{USER}/graphcast-operational/worker-{WORKER}"
)

# Shared temporary/persistent output area
FINAL_DIR = Path(
    f"/scratch/{USER}/weather-ai/graphcast-operational"
)

WORK_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Variables
# ============================================================

PRESSURE_LEVELS = [
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
]

VARIABLES = [
    "msl",
    "t2m",
    "tp06",
    "u10m",
    "v10m",
]

for level in PRESSURE_LEVELS:
    VARIABLES.extend(
        [
            f"t{level}",
            f"u{level}",
            f"v{level}",
            f"q{level}",
            f"z{level}",
        ]
    )

VARIABLES = np.asarray(VARIABLES)

EXPECTED_VARIABLES = len(VARIABLES)

assert EXPECTED_VARIABLES == 70


# ============================================================
# Initialization dates
# Tuesday + Friday
# ============================================================

all_dates = []

current_date = START_DATE

while current_date <= END_DATE:
    # Monday = 0
    # Tuesday = 1
    # Friday = 4
    if current_date.weekday() in (1, 4):
        all_dates.append(current_date)

    current_date += timedelta(days=1)

# Divide independent initialization dates among workers.
my_dates = all_dates[WORKER::NUM_WORKERS]


# ============================================================
# Utility functions
# ============================================================

def log(message=""):
    """Print immediately so redirected Slurm logs stay current."""
    print(message, flush=True)


def size_gib(path: Path) -> float:
    """Return disk usage in GiB, including Zarr directories."""

    result = subprocess.check_output(
        [
            "du",
            "-s",
            "-B1",
            str(path),
        ],
        text=True,
    )

    size_bytes = int(result.split()[0])

    return size_bytes / (1024 ** 3)


def format_duration(seconds: float) -> str:
    """Human-readable elapsed time."""

    seconds = int(seconds)

    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"

    return f"{minutes:d}m {seconds:02d}s"


def make_netcdf_encoding(ds: xr.Dataset) -> dict:
    """
    NetCDF4 compression settings.

    Let the netCDF4/HDF5 backend choose chunk sizes rather than
    copying Zarr chunk geometry directly.
    """

    return {
        name: {
            "zlib": True,
            "complevel": COMPRESSION_LEVEL,
            "shuffle": True,
        }
        for name in ds.data_vars
    }


def validate_netcdf(path: Path):
    """
    Verify the generated NetCDF before publishing it to /scratch.
    """

    if not path.exists():
        raise RuntimeError(
            f"NetCDF output does not exist: {path}"
        )

    if path.stat().st_size == 0:
        raise RuntimeError(
            f"NetCDF output is empty: {path}"
        )

    with xr.open_dataset(
        path,
        engine="netcdf4",
    ) as ds:

        lead_times = ds.sizes.get("lead_time")

        if lead_times != EXPECTED_LEAD_TIMES:
            raise RuntimeError(
                "Invalid lead_time dimension: "
                f"expected {EXPECTED_LEAD_TIMES}, "
                f"got {lead_times}"
            )

        actual_variables = len(ds.data_vars)

        if actual_variables != EXPECTED_VARIABLES:
            raise RuntimeError(
                "Invalid variable count: "
                f"expected {EXPECTED_VARIABLES}, "
                f"got {actual_variables}"
            )


def convert_zarr_to_netcdf(
    zarr_path: Path,
    local_nc_path: Path,
):
    """
    Convert Zarr to compressed NetCDF on node-local /lscratch.

    Conversion happens locally first so that the expensive read/write
    workload does not operate directly against shared /scratch.
    """

    local_tmp = Path(
        str(local_nc_path) + ".tmp"
    )

    if local_tmp.exists():
        local_tmp.unlink()

    if local_nc_path.exists():
        local_nc_path.unlink()

    log("Opening Zarr for NetCDF conversion...")

    ds = xr.open_zarr(
        zarr_path,
        chunks={},
    )

    encoding = make_netcdf_encoding(ds)

    t0 = time.perf_counter()

    try:
        delayed = ds.to_netcdf(
            local_tmp,
            engine="netcdf4",
            encoding=encoding,
            compute=False,
        )

        with ProgressBar():
            delayed.compute(
                scheduler="threads",
                num_workers=NC_WORKERS,
            )

    finally:
        ds.close()

    elapsed = time.perf_counter() - t0

    log(
        "NetCDF encoding finished in "
        f"{format_duration(elapsed)}"
    )

    # Validate while still on local storage.
    validate_netcdf(local_tmp)

    # Atomic publication inside /lscratch.
    local_tmp.replace(local_nc_path)

    return elapsed


def publish_netcdf(
    local_nc_path: Path,
    final_nc_path: Path,
):
    """
    Copy a validated local NetCDF to /scratch safely.

    Data first goes to *.partial. Only after the complete copy succeeds
    is it renamed to the final filename.
    """

    final_partial = Path(
        str(final_nc_path) + ".partial"
    )

    if final_partial.exists():
        final_partial.unlink()

    log("Copying NetCDF to shared /scratch...")

    t0 = time.perf_counter()

    shutil.copy2(
        local_nc_path,
        final_partial,
    )

    local_size = local_nc_path.stat().st_size
    copied_size = final_partial.stat().st_size

    if local_size != copied_size:
        raise RuntimeError(
            "NetCDF copy size mismatch: "
            f"local={local_size}, "
            f"scratch={copied_size}"
        )

    # Both files are on /scratch, so rename is atomic.
    final_partial.replace(final_nc_path)

    elapsed = time.perf_counter() - t0

    log(
        "Copy to /scratch finished in "
        f"{format_duration(elapsed)}"
    )

    return elapsed


# ============================================================
# Startup information
# ============================================================

log("=" * 80)
log("GraphCast Operational 42-day forecast")
log("=" * 80)

log(f"Worker:               {WORKER}")
log(f"Number of workers:    {NUM_WORKERS}")
log(f"Total initializations:{len(all_dates):>5}")
log(f"This worker:          {len(my_dates):>5}")
log(f"Forecast steps:       {STEPS:>5}")
log(f"Output lead times:    {EXPECTED_LEAD_TIMES:>5}")
log(f"Variables:            {EXPECTED_VARIABLES:>5}")
log(f"NetCDF CPU workers:   {NC_WORKERS:>5}")

log()
log(f"Local work directory: {WORK_DIR}")
log(f"Final directory:      {FINAL_DIR}")

log()
log("Initialization dates assigned to this worker:")

for d in my_dates:
    log(f"  {d.isoformat()}")

log()


# ============================================================
# Load model ONCE per GPU worker
# ============================================================

log("=" * 80)
log(f"[worker {WORKER}] Loading GraphCastOperational")
log("=" * 80)

model_load_start = time.perf_counter()

package = GraphCastOperational.load_default_package()
model = GraphCastOperational.load_model(package)

data = ARCO()

model_load_elapsed = time.perf_counter() - model_load_start

log(
    f"[worker {WORKER}] Model loaded in "
    f"{format_duration(model_load_elapsed)}"
)

log()


# ============================================================
# Main forecast loop
# ============================================================

worker_start = time.perf_counter()

successful_run_times = []

for sequence, init_date in enumerate(
    my_dates,
    start=1,
):

    stamp = init_date.strftime("%Y%m%d")

    init_time = datetime(
        init_date.year,
        init_date.month,
        init_date.day,
        0,
        0,
        0,
    )

    basename = (
        f"graphcast_operational_"
        f"{stamp}_42day_subset"
    )

    zarr_path = (
        WORK_DIR /
        f"{basename}.zarr"
    )

    local_nc_path = (
        WORK_DIR /
        f"{basename}.nc"
    )

    final_nc_path = (
        FINAL_DIR /
        f"{basename}.nc"
    )

    final_partial_path = Path(
        str(final_nc_path) + ".partial"
    )

    log()
    log("=" * 80)
    log(
        f"[worker {WORKER}] "
        f"{sequence}/{len(my_dates)} "
        f"| initialization {init_time.isoformat()}"
    )
    log("=" * 80)

    # --------------------------------------------------------
    # Resume support
    # --------------------------------------------------------

    if final_nc_path.exists():

        log(
            f"Already complete: {final_nc_path}"
        )

        # Clean up leftovers from an interrupted previous run.
        if zarr_path.exists():
            log(
                f"Removing leftover Zarr: {zarr_path}"
            )
            shutil.rmtree(zarr_path)

        if local_nc_path.exists():
            log(
                f"Removing leftover local NetCDF: "
                f"{local_nc_path}"
            )
            local_nc_path.unlink()

        if final_partial_path.exists():
            final_partial_path.unlink()

        continue

    # An existing Zarr without the final NC is considered incomplete.
    if zarr_path.exists():
        log(
            f"Removing incomplete Zarr: {zarr_path}"
        )
        shutil.rmtree(zarr_path)

    if local_nc_path.exists():
        log(
            f"Removing incomplete local NetCDF: "
            f"{local_nc_path}"
        )
        local_nc_path.unlink()

    if final_partial_path.exists():
        log(
            f"Removing incomplete /scratch copy: "
            f"{final_partial_path}"
        )
        final_partial_path.unlink()

    initialization_start = time.perf_counter()

    # --------------------------------------------------------
    # GraphCast inference
    # --------------------------------------------------------

    log()
    log("Starting GraphCast inference...")

    io = ZarrBackend(
        file_name=str(zarr_path),
        backend_kwargs={
            "overwrite": True,
        },
    )

    inference_start = time.perf_counter()

    run.deterministic(
        [init_time],
        STEPS,
        model,
        data,
        io,
        output_coords={
            "variable": VARIABLES,
        },
        device="cuda",
    )

    inference_elapsed = (
        time.perf_counter() -
        inference_start
    )

    log(
        "Inference completed in "
        f"{format_duration(inference_elapsed)}"
    )

    zarr_size = size_gib(zarr_path)

    log(
        f"Zarr size: {zarr_size:.2f} GiB"
    )

    # --------------------------------------------------------
    # Zarr -> compressed NetCDF on /lscratch
    # --------------------------------------------------------

    log()
    log("Starting NetCDF conversion...")

    conversion_elapsed = convert_zarr_to_netcdf(
        zarr_path,
        local_nc_path,
    )

    local_nc_size = size_gib(local_nc_path)

    log(
        f"Local NetCDF size: "
        f"{local_nc_size:.2f} GiB"
    )

    # --------------------------------------------------------
    # Copy final NetCDF to /scratch
    # --------------------------------------------------------

    log()

    copy_elapsed = publish_netcdf(
        local_nc_path,
        final_nc_path,
    )

    final_size = size_gib(final_nc_path)

    log(
        f"Final NetCDF: {final_nc_path}"
    )

    log(
        f"Final size: {final_size:.2f} GiB"
    )

    # --------------------------------------------------------
    # Clean local temporary products only after success
    # --------------------------------------------------------

    log()
    log("Cleaning temporary local products...")

    if zarr_path.exists():
        shutil.rmtree(zarr_path)

    if local_nc_path.exists():
        local_nc_path.unlink()

    initialization_elapsed = (
        time.perf_counter() -
        initialization_start
    )

    successful_run_times.append(
        initialization_elapsed
    )

    # --------------------------------------------------------
    # Per-initialization summary + ETA
    # --------------------------------------------------------

    average_run_time = (
        sum(successful_run_times) /
        len(successful_run_times)
    )

    remaining = (
        len(my_dates) -
        sequence
    )

    eta_seconds = (
        average_run_time *
        remaining
    )

    log()
    log("-" * 80)
    log(
        f"Initialization complete: "
        f"{init_date.isoformat()}"
    )

    log(
        f"  Inference: "
        f"{format_duration(inference_elapsed)}"
    )

    log(
        f"  NetCDF conversion: "
        f"{format_duration(conversion_elapsed)}"
    )

    log(
        f"  Copy to /scratch: "
        f"{format_duration(copy_elapsed)}"
    )

    log(
        f"  Total: "
        f"{format_duration(initialization_elapsed)}"
    )

    log(
        f"  Average completed run: "
        f"{format_duration(average_run_time)}"
    )

    log(
        f"  Approx. worker ETA: "
        f"{format_duration(eta_seconds)}"
    )

    log("-" * 80)


# ============================================================
# Final summary
# ============================================================

worker_elapsed = (
    time.perf_counter() -
    worker_start
)

log()
log("=" * 80)
log(f"WORKER {WORKER} COMPLETE")
log("=" * 80)

log(
    f"Assigned initializations: "
    f"{len(my_dates)}"
)

log(
    f"Newly completed this run: "
    f"{len(successful_run_times)}"
)

log(
    f"Worker elapsed time: "
    f"{format_duration(worker_elapsed)}"
)

log(
    f"Final output directory: "
    f"{FINAL_DIR}"
)
