#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
from earth2studio import run
from earth2studio.data import ARCO
from earth2studio.io import AsyncZarrBackend, ZarrBackend
from earth2studio.models.px import GraphCastOperational

from graphcast_config import initialization_dates, load_graphcast_config, output_basename


def log(msg=""):
    print(msg, flush=True)


def duration(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def size_gib(path: Path):
    out = subprocess.check_output(["du", "-s", "-B1", str(path)], text=True)
    return int(out.split()[0]) / 1024**3


def remove(path: Path):
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def make_io(cfg, local: Path, init_date):
    if cfg.zarr_backend == "sync":
        return ZarrBackend(
            file_name=str(local),
            backend_kwargs={"overwrite": True},
        )

    time_coord = np.asarray(
        [np.datetime64(datetime.combine(init_date, datetime.min.time()))]
    )
    lead_time_coord = np.asarray(
        [
            np.timedelta64(cfg.hours_per_step * step, "h")
            for step in range(cfg.expected_lead_times)
        ]
    )
    parallel_coords = OrderedDict(
        {
            "time": time_coord,
            "lead_time": lead_time_coord,
        }
    )

    shard_coords = {}
    if cfg.shard_lead_times > 1:
        shard_coords["lead_time"] = cfg.shard_lead_times

    return AsyncZarrBackend(
        file_name=str(local),
        parallel_coords=parallel_coords,
        blocking=False,
        pool_size=cfg.async_pool_size,
        shard_coords=shard_coords,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/graphcast_operational.yaml")
    p.add_argument("--worker", type=int, required=True)
    p.add_argument("--num-workers", type=int, default=1)
    args = p.parse_args()

    if args.num_workers < 1 or not 0 <= args.worker < args.num_workers:
        raise ValueError("invalid worker/num-workers")

    cfg = load_graphcast_config(args.config)
    dates = initialization_dates(cfg)
    mine = dates[args.worker::args.num_workers]
    work = cfg.local_root / f"producer-{args.worker}"
    work.mkdir(parents=True, exist_ok=True)
    cfg.staging_root.mkdir(parents=True, exist_ok=True)
    cfg.final_root.mkdir(parents=True, exist_ok=True)

    log(
        f"Producer {args.worker}: {len(mine)} of {len(dates)} initializations; "
        f"zarr_backend={cfg.zarr_backend}; "
        f"async_pool={cfg.async_pool_size}; "
        f"lead_time_shard={cfg.shard_lead_times}"
    )

    t0 = time.perf_counter()
    pkg = GraphCastOperational.load_default_package()
    model = GraphCastOperational.load_model(pkg)
    data = ARCO()
    log(f"Model loaded in {duration(time.perf_counter()-t0)}")

    for n, init_date in enumerate(mine, 1):
        base = output_basename(cfg, init_date)
        local = work / f"{base}.zarr"
        staged = cfg.staging_root / f"{base}.zarr"
        partial = cfg.staging_root / f"{base}.partial.zarr"
        ready = cfg.staging_root / f"{base}.ready"
        final = cfg.final_root / f"{base}.nc"

        log(f"[{n}/{len(mine)}] {init_date}")

        if final.exists():
            log("Final NetCDF exists; skipping.")
            remove(local)
            remove(partial)
            continue

        if staged.exists() and ready.exists():
            log("Staged Zarr already ready; skipping.")
            remove(local)
            remove(partial)
            continue

        remove(local)
        remove(partial)
        remove(staged)
        remove(ready)

        io = make_io(cfg, local, init_date)

        start = time.perf_counter()
        run.deterministic(
            [datetime.combine(init_date, datetime.min.time())],
            cfg.steps,
            model,
            data,
            io,
            output_coords={"variable": np.asarray(cfg.variables)},
            device=cfg.device,
        )
        inference_loop = time.perf_counter() - start

        drain = 0.0
        if cfg.zarr_backend == "async":
            start = time.perf_counter()
            io.close()
            drain = time.perf_counter() - start

        log(
            f"Inference loop: {duration(inference_loop)}; "
            f"Zarr drain: {duration(drain)}; "
            f"Zarr: {size_gib(local):.2f} GiB"
        )

        start = time.perf_counter()
        shutil.copytree(local, partial)
        partial.replace(staged)
        ready.write_text(
            f"ready {datetime.now().isoformat()}\n",
            encoding="utf-8",
        )
        stage = time.perf_counter() - start
        remove(local)

        log(f"Staged to /scratch in {duration(stage)}")


if __name__ == "__main__":
    main()
