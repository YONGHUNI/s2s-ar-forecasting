#!/usr/bin/env python3
"""Convert published NeuralGCM 42-day Zarr forecasts to independent NetCDF4 files.

This script intentionally never deletes a Zarr source. Run on a CPU Slurm job.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path

import numpy as np

from run_neuralgcm_batch import SEED, initialization_dates, output_path

REQUEUE_EXIT_CODE = 75
REQUIRED = frozenset({"specific_humidity", "u_component_of_wind",
                       "v_component_of_wind", "temperature", "geopotential"})


def stamp(message: str) -> None:
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"{now} | job={os.getenv('SLURM_JOB_ID','local')} | {message}", flush=True)


def producer_active(job_id: str | None) -> bool | None:
    if not job_id:
        return None
    try:
        result = subprocess.run(["squeue", "-h", "-j", job_id, "-r",
                                 "-o", "%A"], text=True, capture_output=True,
                                check=False)
        if result.returncode != 0:
            return None
        return job_id in result.stdout.split()
    except OSError:
        return None


def validate_ds(ds, date_string: str) -> None:
    """Metadata-level checks without loading the full payload."""
    if ds.sizes.get("time") != 168 or ds.sizes.get("level") != 37:
        raise ValueError(f"Wrong time/level dimensions: {dict(ds.sizes)}")
    if not REQUIRED.issubset(ds.data_vars):
        raise ValueError(f"Missing atmospheric variables: {REQUIRED-set(ds.data_vars)}")
    if ds.attrs.get("seed") != SEED:
        raise ValueError("Unexpected RNG seed")
    if not np.array_equal(ds.time.values, np.arange(6, 1009, 6)):
        raise ValueError("Forecast lead times differ from +6..+1008 by 6 hours")
    if str(ds.attrs.get("initialization_utc", ""))[:10] != date_string:
        raise ValueError(f"Wrong initialization: expected {date_string}")


def convert_one(zarr_path: str, nc_path: str, date_string: str, level: int) -> tuple[int, float]:
    import xarray as xr

    source, destination = Path(zarr_path), Path(nc_path)
    partial = Path(str(destination) + ".partial")
    if destination.exists():
        raise FileExistsError(destination)
    if partial.exists():
        # Never silently overwrite an earlier failed writer's output.
        raise FileExistsError(partial)
    started = time.monotonic()
    with xr.open_zarr(source, chunks=None, consolidated=False) as ds:
        validate_ds(ds, date_string)
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Write coordinates/metadata once, then stream each variable separately.
        # Avoids making a ~5.5 GiB in-memory Dataset or large Dask task graph.
        coordinates = xr.Dataset(coords={
            name: ds.coords[name] for name in ds.coords
        }, attrs=dict(ds.attrs))
        coordinates.to_netcdf(partial, mode="w", engine="netcdf4")
        try:
            for name in ds.data_vars:
                variable = ds[name].variable.copy(deep=False)
                chunk = xr.Dataset({str(name): variable})
                chunk.to_netcdf(partial, mode="a", engine="netcdf4", encoding={
                    str(name): {"zlib": level > 0, "complevel": level,
                                "shuffle": level > 0}
                })
            with xr.open_dataset(partial, engine="netcdf4") as nc:
                validate_ds(nc, date_string)
                if set(nc.data_vars) != set(ds.data_vars):
                    raise ValueError("Variables differ after conversion")
                for name in ds.data_vars:
                    if nc[name].dims != ds[name].dims or nc[name].shape != ds[name].shape:
                        raise ValueError(f"Shape mismatch: {name}")
            if destination.exists():
                raise FileExistsError(destination)
            partial.rename(destination)
        except Exception:
            stamp(f"Conversion failed; retained partial for review: {partial}")
            raise
    return destination.stat().st_size, time.monotonic() - started


def valid_nc(path: Path, date_string: str) -> bool:
    import xarray as xr
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with xr.open_dataset(path, engine="netcdf4") as ds:
            validate_ds(ds, date_string)
        return True
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-job-id", default=os.getenv("PRODUCER_JOB_ID"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--compression-level", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--requeue-after-seconds", type=int, default=int(os.getenv(
        "CONVERTER_REQUEUE_AFTER_SECONDS", "165600")))
    parser.add_argument("--limit", type=int, help="Optional first N dates for a pilot")
    args = parser.parse_args()
    if args.workers < 1 or args.poll_seconds < 1 or not 0 <= args.compression_level <= 9:
        parser.error("Invalid worker/poll/compression configuration")
    user = os.getenv("USER")
    if not user:
        parser.error("USER is required")

    root = Path(f"/scratch/{user}/weather-ai/neuralgcm-operational")
    dates = initialization_dates()
    if args.limit is not None:
        if args.limit < 1:
            parser.error("limit must be positive")
        dates = dates[:args.limit]
    items = [(day.isoformat(), output_path(root, day),
              root / "netcdf" / f"neuralgcm_{day:%Y%m%d}_42day_seed{SEED}.nc")
             for day in dates]
    (root / "netcdf").mkdir(parents=True, exist_ok=True)
    active = {}
    started = time.monotonic()
    failed = set()
    verified = set()
    stamp(f"NeuralGCM converter started: {len(items)} files, workers={args.workers}, "
          f"compression={args.compression_level}, producer={args.producer_job_id}")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while True:
            for future in [f for f in active if f.done()]:
                date_str, source, dest = active.pop(future)
                try:
                    size, seconds = future.result()
                    if not valid_nc(dest, date_str):
                        raise ValueError(f"Post-publication validation failed: {dest}")
                    verified.add(date_str)
                    stamp(f"Published {dest.name}: {size/1024**3:.2f} GiB in {seconds:.1f}s")
                except Exception as exc:
                    failed.add(date_str)
                    stamp(f"FAILED {date_str}: {exc!r}; preserved source {source}")

            # Completed files are immutable. Cache validation instead of opening
            # hundreds of multi-GiB NetCDF files every polling interval.
            for day, _, dest in items:
                if day not in verified and day not in failed and dest.exists():
                    if valid_nc(dest, day):
                        verified.add(day)
                    else:
                        failed.add(day)
                        stamp(f"INVALID existing NetCDF (not overwritten): {dest}")
            completed = len(verified)
            if completed == len(items) and not active:
                stamp(f"All {completed} NetCDF outputs completed.")
                return 0
            due = (args.requeue_after_seconds > 0 and
                   time.monotonic() - started >= args.requeue_after_seconds)

            if not due:
                for day, source, dest in items:
                    if len(active) >= args.workers:
                        break
                    if day in failed or any(d == day for d, _, _ in active.values()):
                        continue
                    if day in verified or dest.exists():
                        continue
                    if Path(str(dest) + ".partial").exists():
                        failed.add(day)
                        stamp(f"INCOMPLETE NetCDF (not overwritten): {dest}.partial")
                        continue
                    if source.is_dir():
                        active[pool.submit(convert_one, str(source), str(dest), day,
                                           args.compression_level)] = (day, source, dest)

            if active:
                wait(active, timeout=args.poll_seconds, return_when=FIRST_COMPLETED)
                continue

            remaining_ready = any(source.is_dir() and not dest.exists() and day not in failed
                                  for day, source, dest in items)
            state = producer_active(args.producer_job_id)
            if due:
                if state is False and not remaining_ready:
                    stamp(f"Producer ended, {completed}/{len(items)} converted, "
                          f"{len(failed)} failed; no requeue")
                    return 1 if completed != len(items) else 0
                stamp(f"Requeue requested ({completed}/{len(items)} converted, "
                      f"producer_active={state})")
                return REQUEUE_EXIT_CODE
            if state is False and not remaining_ready:
                stamp(f"Producer ended, {completed}/{len(items)} converted, "
                      f"{len(failed)} failed or missing")
                return 1 if completed != len(items) else 0
            if not args.producer_job_id and not remaining_ready:
                stamp("No producer job specified; no pending Zarr to convert")
                return 1 if completed != len(items) else 0
            stamp(f"Waiting for Zarr: {completed}/{len(items)} NetCDF, "
                  f"producer_active={state}")
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
