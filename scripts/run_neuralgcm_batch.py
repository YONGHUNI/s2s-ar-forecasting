#!/usr/bin/env python3
"""Schedule existing NeuralGCM pilot across the 607 legacy-model initializations.

Keeps inference unchanged: one stochastic realization per initialization, seed
42, 42 days, 6-hour outputs. The driver uses separate subprocesses to avoid
retaining JAX compiled executables across initializations.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

START = date(2015, 10, 1)
END = date(2025, 4, 30)
MONTHS = frozenset((10, 11, 12, 1, 2, 3, 4))
WEEKDAYS = frozenset((1, 4))  # Tuesday, Friday
EXPECTED = 607
FORECAST_HOURS = 1008
OUTPUT_INTERVAL_HOURS = 6
SEED = 42
OUTPUT_DIR = "neuralgcm-operational"


def initialization_dates() -> list[date]:
    dates = []
    day = START
    while day <= END:
        if day.month in MONTHS and day.weekday() in WEEKDAYS:
            dates.append(day)
        day += timedelta(days=1)
    if len(dates) != EXPECTED:
        raise RuntimeError(f"Expected {EXPECTED} initializations; got {len(dates)}")
    return dates


def assigned_dates(worker: int, workers: int) -> list[date]:
    if workers < 1 or worker < 0 or worker >= workers:
        raise ValueError("workers must be positive; worker in [0, workers)")
    return initialization_dates()[worker::workers]


def output_path(root: Path, day: date) -> Path:
    return root / f"neuralgcm_{day:%Y%m%d}_42day_seed{SEED}.zarr"


def valid_published_zarr(path: Path) -> bool:
    """Verify time and variables before skipping a previously completed case."""
    if not path.is_dir():
        return False
    import numpy as np
    import xarray as xr

    required = {"specific_humidity", "u_component_of_wind", "v_component_of_wind",
                "temperature", "geopotential"}
    with xr.open_zarr(path, chunks=None, consolidated=False) as ds:
        if ds.sizes.get("time") != 168 or ds.sizes.get("level") != 37:
            return False
        if not required.issubset(ds.data_vars):
            return False
        if ds.attrs.get("seed") != SEED:
            return False
        return np.array_equal(ds.time.values, np.arange(6, 1009, 6))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None,
                        help="Debug: run only first N assigned initializations")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print exact assigned dates; no model or GPU required")
    args = parser.parse_args()
    mine = assigned_dates(args.worker, args.num_workers)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be >= 1")
        mine = mine[:args.limit]
    user = os.environ.get("USER")
    if not user:
        parser.error("USER is not set")
    root = Path(f"/scratch/{user}/weather-ai/{OUTPUT_DIR}")
    print(json.dumps({
        "worker": args.worker, "workers": args.num_workers,
        "count": len(mine), "dates": [str(day) for day in mine],
        "model": "NeuralGCM v1 stochastic 1.4deg",
        "checkpoint": "gs://neuralgcm/models/v1/stochastic_1_4_deg.pkl",
        "forecast_hours": FORECAST_HOURS, "interval_hours": OUTPUT_INTERVAL_HOURS,
        "seed": SEED, "realizations_per_initialization": 1,
        "forcing": "initial_SST_and_sea_ice_persistence",
        "output_dir": str(root),
    }, indent=2), flush=True)
    if args.dry_run:
        return 0

    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    runner = Path(__file__).resolve().with_name("run_neuralgcm_pilot.py")

    errors = []
    for idx, day in enumerate(mine, 1):
        output = output_path(root, day)
        final_nc = root / "netcdf" / f"neuralgcm_{day:%Y%m%d}_42day_seed{SEED}.nc"
        # Match legacy producers: a validated final NetCDF is the terminal
        # artifact, even when its transient Zarr was already reclaimed.
        if final_nc.exists():
            from convert_neuralgcm_netcdf import valid_nc
            if valid_nc(final_nc, day.isoformat()):
                print(f"[{idx}/{len(mine)}] SKIP validated NetCDF {day}", flush=True)
                continue
            print(f"INVALID existing NetCDF (no overwrite): {final_nc}", flush=True)
            errors.append(str(day))
            continue
        partial = Path(str(output) + ".partial.zarr")
        if output.exists():
            try:
                if valid_published_zarr(output):
                    print(f"[{idx}/{len(mine)}] SKIP validated {day}", flush=True)
                    continue
            except Exception as exc:
                print(f"Validation error for {output}: {exc}", flush=True)
            print(f"INVALID pre-existing output (no overwrite): {output}", flush=True)
            errors.append(str(day))
            continue
        if partial.exists():
            print(f"INCOMPLETE prior attempt (no overwrite): {partial}", flush=True)
            errors.append(str(day))
            continue

        command = [
            sys.executable, "-u", str(runner),
            "--initialization", f"{day.isoformat()}T00:00:00",
            "--forecast-hours", str(FORECAST_HOURS),
            "--output-every-hours", str(OUTPUT_INTERVAL_HOURS),
            "--write-chunk-steps", "8",
            "--max-pending-writes", "2",
            "--seed", str(SEED),
            "--output", str(output),
        ]
        logfile = logs / f"neuralgcm_{day:%Y%m%d}_42day.log"
        print(f"[{idx}/{len(mine)}] RUN {day} -> {logfile}", flush=True)
        started = time.perf_counter()
        with logfile.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                       check=False)
        elapsed = time.perf_counter() - started
        if completed.returncode != 0:
            print(f"FAILED {day}: exit={completed.returncode} after {elapsed:.1f}s; {logfile}",
                  flush=True)
            errors.append(str(day))
            continue
        try:
            complete = valid_published_zarr(output)
        except Exception as exc:
            complete = False
            print(f"Output validation failed: {exc}", flush=True)
        if not complete:
            errors.append(str(day))
            print(f"INVALID output after inference: {output}", flush=True)
            continue
        print(f"FINISHED {day} in {elapsed:.1f}s: {output}", flush=True)

    if errors:
        print(f"FAILED initializations: {', '.join(errors)}", flush=True)
        return 1
    print(f"Worker {args.worker} complete: {len(mine)} assignments examined", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
