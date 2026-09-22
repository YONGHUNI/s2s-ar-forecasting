#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import xarray as xr

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
    if level == 0:
        return {
            name: {"zlib": False}
            for name in ds.data_vars
        }

    return {
        name: {
            "zlib": True,
            "complevel": level,
            "shuffle": True,
        }
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
    dates = initialization_dates(cfg)[args.worker::args.num_workers]

    cfg.staging_root.mkdir(parents=True, exist_ok=True)
    cfg.final_root.mkdir(parents=True, exist_ok=True)

    log(
        f"Converter {args.worker}: {len(dates)} initializations; "
        f"compression level={cfg.compression_level}"
    )

    for n, init_date in enumerate(dates, 1):
        base = output_basename(cfg, init_date)
        staged = cfg.staging_root / f"{base}.zarr"
        ready = cfg.staging_root / f"{base}.ready"
        final = cfg.final_root / f"{base}.nc"
        partial = Path(str(final) + ".partial")

        log(f"[{n}/{len(dates)}] {init_date}")

        if final.exists():
            log("Final NetCDF exists; skipping.")
            remove(staged)
            remove(ready)
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

        remove(partial)

        # Sapelo2 benchmark: shared /scratch reads the staged GraphCast Zarr
        # faster than copying it to node-local /lscratch on the tested batch node.
        # Write directly to a .partial file on the same shared filesystem so the
        # final rename is atomic.
        log("Opening staged Zarr directly from /scratch...")
        ds = xr.open_zarr(
            staged,
            chunks={},
            consolidated=False,
        )

        try:
            start = time.perf_counter()

            ds.to_netcdf(
                partial,
                engine="netcdf4",
                encoding=encoding(ds, cfg.compression_level),
            )

            log(f"NetCDF encoding: {duration(time.perf_counter() - start)}")
        finally:
            ds.close()

        validate(partial, cfg)
        partial.replace(final)
        validate(final, cfg)

        remove(staged)
        remove(ready)

        log(f"Published: {final}")


if __name__ == "__main__":
    main()
