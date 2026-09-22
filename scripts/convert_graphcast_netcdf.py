#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import xarray as xr
from dask.diagnostics import ProgressBar

from graphcast_config import initialization_dates, load_graphcast_config, output_basename


def log(msg=""):
    print(msg, flush=True)


def duration(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def remove(path: Path):
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def validate(path: Path, cfg):
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Invalid NetCDF: {path}")
    with xr.open_dataset(path, engine="netcdf4") as ds:
        if ds.sizes.get("lead_time") != cfg.expected_lead_times:
            raise RuntimeError("Unexpected lead_time dimension")
        if len(ds.data_vars) != cfg.expected_variable_count:
            raise RuntimeError("Unexpected variable count")


def encoding(ds, level):
    return {
        name: {"zlib": True, "complevel": level, "shuffle": True}
        for name in ds.data_vars
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/graphcast_operational.yaml")
    p.add_argument("--worker", type=int, required=True)
    p.add_argument("--num-workers", type=int, default=1)
    args = p.parse_args()

    if args.num_workers < 1 or not 0 <= args.worker < args.num_workers:
        raise ValueError("invalid worker/num-workers")

    cfg = load_graphcast_config(args.config)
    threads = int(os.environ.get("NC_WORKERS", "8"))
    dates = initialization_dates(cfg)[args.worker::args.num_workers]
    work = cfg.local_root / f"converter-{args.worker}"
    work.mkdir(parents=True, exist_ok=True)
    cfg.staging_root.mkdir(parents=True, exist_ok=True)
    cfg.final_root.mkdir(parents=True, exist_ok=True)

    for n, init_date in enumerate(dates, 1):
        base = output_basename(cfg, init_date)
        staged = cfg.staging_root / f"{base}.zarr"
        ready = cfg.staging_root / f"{base}.ready"
        local_zarr = work / f"{base}.zarr"
        tmp = work / f"{base}.tmp.nc"
        local_nc = work / f"{base}.nc"
        final = cfg.final_root / f"{base}.nc"
        partial = Path(str(final) + ".partial")

        log(f"[{n}/{len(dates)}] {init_date}")
        if final.exists():
            remove(staged); remove(ready)
            continue

        waited = 0
        while not (ready.exists() and staged.exists()):
            if final.exists():
                break
            if waited % 60 == 0:
                log(f"Waiting for producer ({waited}s)...")
            time.sleep(cfg.poll_seconds)
            waited += cfg.poll_seconds
        if final.exists():
            continue

        for path in (local_zarr, tmp, local_nc, partial):
            remove(path)

        log("Copying staged Zarr to node-local /lscratch...")
        shutil.copytree(staged, local_zarr)

        ds = xr.open_zarr(local_zarr, chunks={})
        try:
            start = time.perf_counter()
            task = ds.to_netcdf(
                tmp,
                engine="netcdf4",
                encoding=encoding(ds, cfg.compression_level),
                compute=False,
            )
            with ProgressBar():
                task.compute(scheduler="threads", num_workers=threads)
            log(f"NetCDF encoding: {duration(time.perf_counter()-start)}")
        finally:
            ds.close()

        validate(tmp, cfg)
        tmp.replace(local_nc)
        shutil.copy2(local_nc, partial)
        if local_nc.stat().st_size != partial.stat().st_size:
            raise RuntimeError("NetCDF publication size mismatch")
        partial.replace(final)
        validate(final, cfg)

        remove(staged); remove(ready); remove(local_zarr); remove(local_nc)
        log(f"Published: {final}")


if __name__ == "__main__":
    main()
