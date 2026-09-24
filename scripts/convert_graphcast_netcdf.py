#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from graphcast_config import initialization_dates, load_graphcast_config, output_basename
from netcdf_conversion import ConversionResult, convert_zarr_to_netcdf


@dataclass(frozen=True)
class ConversionJob:
    base: str
    staged: Path
    ready: Path
    final: Path


def log(msg=""):
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    job = os.environ.get("SLURM_JOB_ID", "local")
    print(f"{stamp} | job={job} | {msg}", flush=True)


def duration(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def remove(path: Path):
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def producer_is_active(job_id: str | None) -> bool | None:
    """Return whether the producer is still present in Slurm's active queue."""

    if not job_id:
        return None

    try:
        result = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%T"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        log(f"Could not query producer job {job_id}: {exc}")
        return None

    if result.returncode != 0:
        log(
            f"Could not query producer job {job_id}: "
            f"squeue exited {result.returncode}"
        )
        return None

    return bool(result.stdout.strip())


def build_jobs(cfg) -> list[ConversionJob]:
    jobs = []

    for init_date in initialization_dates(cfg):
        base = output_basename(cfg, init_date)
        jobs.append(
            ConversionJob(
                base=base,
                staged=cfg.staging_root / f"{base}.zarr",
                ready=cfg.staging_root / f"{base}.ready",
                final=cfg.final_root / f"{base}.nc",
            )
        )

    return jobs


def convert_one(job: ConversionJob, cfg) -> tuple[int, ConversionResult]:
    result = convert_zarr_to_netcdf(
        job.staged,
        job.final,
        compression_level=cfg.compression_level,
        consolidated=False,
        expected_sizes={"lead_time": cfg.expected_lead_times},
        expected_variable_count=cfg.expected_variable_count,
    )
    return os.getpid(), result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/graphcast_operational.yaml")
    args = parser.parse_args()

    cfg = load_graphcast_config(args.config)
    cfg.staging_root.mkdir(parents=True, exist_ok=True)
    cfg.final_root.mkdir(parents=True, exist_ok=True)

    producer_job_id = os.environ.get("PRODUCER_JOB_ID")
    jobs = build_jobs(cfg)
    active = {}

    log(
        f"Converter manager: {len(jobs)} initializations; "
        f"workers={cfg.converter_workers}; "
        f"poll={cfg.poll_seconds}s; "
        f"compression level={cfg.compression_level}; "
        f"producer_job={producer_job_id or 'untracked'}"
    )

    # One manager process owns discovery/submission. Worker processes never
    # scan the staging directory, so one ready item is submitted at most once.
    with ProcessPoolExecutor(max_workers=cfg.converter_workers) as pool:
        while True:
            # Collect finished workers first so newly freed slots can be filled
            # in the same polling iteration.
            done = [future for future in active if future.done()]

            for future in done:
                job = active.pop(future)

                try:
                    worker_pid, result = future.result()
                except Exception:
                    log(f"FAILED: {job.base}")
                    raise

                remove(job.staged)
                remove(job.ready)

                log(
                    f"Published: {result.destination} | worker_pid={worker_pid} | "
                    f"NetCDF encoding: {duration(result.encoding_seconds)} | "
                    f"size={result.output_size_bytes / 1024**3:.2f} GiB"
                )

            completed = sum(job.final.exists() for job in jobs)

            if completed == len(jobs) and not active:
                for job in jobs:
                    remove(job.staged)
                    remove(job.ready)

                log(f"All {completed} NetCDF files are complete.")
                break

            slots = cfg.converter_workers - len(active)

            if slots > 0:
                active_bases = {job.base for job in active.values()}

                for job in jobs:
                    if slots == 0:
                        break

                    if job.final.exists():
                        # Safe restart cleanup after an already-published file.
                        remove(job.staged)
                        remove(job.ready)
                        continue

                    if job.base in active_bases:
                        continue

                    if not (job.ready.exists() and job.staged.exists()):
                        continue

                    log(f"Queue -> worker: {job.base}")
                    future = pool.submit(convert_one, job, cfg)
                    active[future] = job
                    active_bases.add(job.base)
                    slots -= 1

            if active:
                # Wake quickly when a worker finishes, but still rescan
                # periodically for newly-created .ready markers.
                wait(
                    active,
                    timeout=cfg.poll_seconds,
                    return_when=FIRST_COMPLETED,
                )
            else:
                producer_active = producer_is_active(producer_job_id)

                if producer_active is False:
                    log(
                        f"Producer job {producer_job_id} is no longer active and "
                        f"no conversion work remains; exiting "
                        f"({completed}/{len(jobs)} complete)."
                    )
                    break

                log(
                    f"Waiting for ready Zarr files "
                    f"({completed}/{len(jobs)} complete)..."
                )
                time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    main()
