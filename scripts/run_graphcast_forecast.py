#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
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


def log(msg="", *, producer=None):
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    job = os.environ.get("SLURM_JOB_ID", "local")
    context = f"{stamp} | job={job}"
    if producer is not None:
        context += f" | producer={producer}"
    print(f"{context} | {msg}", flush=True)


def duration(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def remove(path: Path):
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def process_status():
    values = {}

    try:
        with Path("/proc/self/status").open(encoding="utf-8") as stream:
            for line in stream:
                key, _, value = line.partition(":")
                if key in {"VmRSS", "VmSize", "Threads"}:
                    values[key] = value.strip()
    except OSError:
        return "RSS=?; VMS=?; threads=?"

    def gib(key):
        value = values.get(key)
        if value is None:
            return None
        return int(value.split()[0]) / 1024**2

    rss = gib("VmRSS")
    vms = gib("VmSize")
    threads = values.get("Threads", "?")
    rss_text = "?" if rss is None else f"{rss:.2f} GiB"
    vms_text = "?" if vms is None else f"{vms:.2f} GiB"
    return f"RSS={rss_text}; VMS={vms_text}; threads={threads}"


def close_async_backend(io):
    """Drain AsyncZarr writes and stop its per-instance event-loop threads."""

    start = time.perf_counter()
    close_error = None

    try:
        io.close()
    except BaseException as exc:
        close_error = exc
    finally:
        drain_elapsed = time.perf_counter() - start
        shutdown_start = time.perf_counter()
        loops = tuple(getattr(io, "loop_pool", ()))

        for loop in loops:
            if loop.is_running() and not loop.is_closed():
                try:
                    loop.call_soon_threadsafe(loop.stop)
                except RuntimeError:
                    pass

        deadline = time.monotonic() + 5.0
        while any(loop.is_running() for loop in loops):
            if time.monotonic() >= deadline:
                if close_error is None:
                    close_error = RuntimeError(
                        "AsyncZarr event-loop threads did not stop within 5 seconds"
                    )
                break
            time.sleep(0.01)

        for loop in loops:
            if not loop.is_running() and not loop.is_closed():
                loop.close()

    shutdown_elapsed = time.perf_counter() - shutdown_start

    if close_error is not None:
        raise close_error

    return drain_elapsed, shutdown_elapsed


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
    if cfg.write_mode == "local_then_stage":
        work.mkdir(parents=True, exist_ok=True)
    cfg.staging_root.mkdir(parents=True, exist_ok=True)
    cfg.final_root.mkdir(parents=True, exist_ok=True)

    log(
        f"Producer {args.worker}: {len(mine)} of {len(dates)} initializations; "
        f"zarr_backend={cfg.zarr_backend}; "
        f"write_mode={cfg.write_mode}; "
        f"async_pool={cfg.async_pool_size}; "
        f"lead_time_shard={cfg.shard_lead_times}",
        producer=args.worker,
    )

    t0 = time.perf_counter()
    pkg = GraphCastOperational.load_default_package()
    model = GraphCastOperational.load_model(pkg)
    data = ARCO()
    log(f"Model loaded in {duration(time.perf_counter()-t0)}", producer=args.worker)
    log(f"Process after model load: {process_status()}", producer=args.worker)

    for n, init_date in enumerate(mine, 1):
        base = output_basename(cfg, init_date)
        local = work / f"{base}.zarr"
        staged = cfg.staging_root / f"{base}.zarr"
        partial = cfg.staging_root / f"{base}.partial.zarr"
        ready = cfg.staging_root / f"{base}.ready"
        final = cfg.final_root / f"{base}.nc"

        log(f"[{n}/{len(mine)}] {init_date}", producer=args.worker)

        if final.exists():
            log("Final NetCDF exists; skipping.", producer=args.worker)
            remove(local)
            remove(partial)
            continue

        if staged.exists() and ready.exists():
            log("Staged Zarr already ready; skipping.", producer=args.worker)
            remove(local)
            remove(partial)
            continue

        path_cleanup_start = time.perf_counter()
        remove(local)
        remove(partial)
        remove(staged)
        remove(ready)
        path_cleanup = time.perf_counter() - path_cleanup_start

        write_target = local if cfg.write_mode == "local_then_stage" else partial
        backend_setup_start = time.perf_counter()
        io = make_io(cfg, write_target, init_date)
        backend_setup = time.perf_counter() - backend_setup_start

        start = time.perf_counter()
        drain = 0.0
        loop_shutdown = 0.0
        inference_error = None

        try:
            run.deterministic(
                [datetime.combine(init_date, datetime.min.time())],
                cfg.steps,
                model,
                data,
                io,
                output_coords={"variable": np.asarray(cfg.variables)},
                device=cfg.device,
            )
        except BaseException as exc:
            inference_error = exc
            raise
        finally:
            inference_loop = time.perf_counter() - start

            if cfg.zarr_backend == "async":
                try:
                    drain, loop_shutdown = close_async_backend(io)
                except BaseException as cleanup_exc:
                    if inference_error is None:
                        raise
                    log(
                        f"Async Zarr cleanup also failed: {cleanup_exc!r}",
                        producer=args.worker,
                    )

        log(
            f"Inference loop: {duration(inference_loop)}; "
            f"Zarr drain: {duration(drain)}",
            producer=args.worker,
        )
        log(f"Process after I/O cleanup: {process_status()}", producer=args.worker)

        start = time.perf_counter()
        if cfg.write_mode == "local_then_stage":
            shutil.copytree(local, partial)
        partial.replace(staged)
        ready.write_text(
            f"ready {datetime.now().isoformat()}\n",
            encoding="utf-8",
        )
        stage = time.perf_counter() - start
        remove(local)

        if cfg.write_mode == "local_then_stage":
            log(f"Staged to /scratch in {duration(stage)}", producer=args.worker)
        else:
            log(
                f"Direct /scratch finalize in {duration(stage)}",
                producer=args.worker,
            )

        log(
            "Timing detail: "
            f"path_cleanup={path_cleanup:.3f}s; "
            f"backend_setup={backend_setup:.3f}s; "
            f"inference={inference_loop:.3f}s; "
            f"zarr_drain={drain:.3f}s; "
            f"loop_shutdown={loop_shutdown:.3f}s; "
            f"finalize={stage:.3f}s",
            producer=args.worker,
        )


if __name__ == "__main__":
    main()
