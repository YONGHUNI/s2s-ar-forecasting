#!/usr/bin/env python3
"""Shared, read-only NetCDF QC for GraphCastOperational and Pangu6."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import random
import statistics

import numpy as np
from netCDF4 import Dataset, num2date

from forecast_config import load_forecast_config, initialization_dates, output_basename
from qc_graphcast import sampled_variable_stats, full_variable_stats

UNITS = {
    "msl": "Pa", "t2m": "K", "tp06": "m",
    "u10m": "m s**-1", "v10m": "m s**-1",
    "t": "K", "u": "m s**-1", "v": "m s**-1",
    "q": "kg kg**-1", "z": "m**2 s**-2",
}


def units_for(name):
    return UNITS.get(name, UNITS.get(name[0]))


def _check_metadata_one(path, cfg):
    errors = []
    shapes = None
    try:
            with Dataset(path) as ds:
                expected_dims = {"time": 1, "lead_time": cfg.expected_lead_times,
                                 "lat": 721, "lon": 1440}
                for name, value in expected_dims.items():
                    if name not in ds.dimensions or len(ds.dimensions[name]) != value:
                        errors.append((path.name, "dimension " + name))
                missing = set(cfg.variables) - set(ds.variables)
                if missing:
                    errors.append((path.name, "missing vars: " + ",".join(sorted(missing))))
                    continue
                extra = {name for name, var in ds.variables.items()
                         if var.dimensions == ("time", "lead_time", "lat", "lon")} - set(cfg.variables)
                if extra:
                    errors.append((path.name, "extra vars: " + ",".join(sorted(extra))))
                if "lead_time" not in ds.variables:
                    errors.append((path.name, "lead_time coordinate missing"))
                else:
                    lead = ds.variables["lead_time"]
                    expected_leads = np.arange(cfg.expected_lead_times) * cfg.hours_per_step
                    if (getattr(lead, "units", "") != "hours"
                            or not np.array_equal(lead[:], expected_leads)):
                        errors.append((path.name, "lead_time values/units"))
                if "time" not in ds.variables:
                    errors.append((path.name, "time coordinate missing"))
                else:
                    var = ds.variables["time"]
                    dt = num2date(var[0], var.units,
                                  calendar=getattr(var, "calendar", "standard"))
                    init = datetime.strptime(path.name.split("_")[-3], "%Y%m%d")
                    if dt.strftime("%Y%m%d%H%M%S") != init.strftime("%Y%m%d000000"):
                        errors.append((path.name, "initialization time mismatch"))
                shapes = {}
                for name in cfg.variables:
                    var = ds.variables[name]
                    shapes[name] = var.shape
                    if var.dimensions != ("time", "lead_time", "lat", "lon"):
                        errors.append((path.name, name + " dimension order"))
                    if getattr(var, "units", None) != units_for(name):
                        errors.append((path.name, name + " units"))
                    if name[0] in "tuvqz" and name[1:].isdigit():
                        if (getattr(var, "pressure_level", None) != int(name[1:])
                                or getattr(var, "pressure_level_units", None) != "hPa"):
                            errors.append((path.name, name + " pressure metadata"))
    except Exception as exc:
        errors.append((path.name, "open/metadata: " + repr(exc)))
    return errors, shapes


def lightweight(files, cfg, dates, metadata_workers):
    errors = []
    expected = {output_basename(cfg, d) + ".nc" for d in dates}
    actual = {p.name for p in files}
    errors.extend((name, "missing") for name in sorted(expected - actual))
    errors.extend((name, "unexpected") for name in sorted(actual - expected))
    sizes = [p.stat().st_size / 1024**3 for p in files]
    median = statistics.median(sizes) if sizes else 0
    for path, size in zip(files, sizes):
        if median and size < median * 0.8:
            errors.append((path.name, "size below 80% of median"))

    reference = None
    with ProcessPoolExecutor(max_workers=metadata_workers) as pool:
        futures = {pool.submit(_check_metadata_one, path, cfg): path for path in files}
        for i, future in enumerate(as_completed(futures), 1):
            path = futures[future]
            try:
                file_errors, shapes = future.result()
                errors.extend(file_errors)
                if shapes is not None:
                    if reference is None:
                        reference = shapes
                    elif shapes != reference:
                        errors.append((path.name, "shape mismatch"))
            except Exception as exc:
                errors.append((path.name, "worker failure: " + repr(exc)))
            if i % 100 == 0 or i == len(files):
                print(f"[metadata {i}/{len(files)}]", flush=True)

    print(f"Files: {len(files)} / {len(dates)}; size GiB "
          f"min/median/max: {min(sizes) if sizes else 0:.2f}/"
          f"{median:.2f}/{max(sizes) if sizes else 0:.2f}")
    print(f"Metadata errors: {len(errors)}", flush=True)
    for name, reason in sorted(errors)[:100]:
        print(f"  {name}: {reason}", flush=True)
    return not errors and len(files) == len(dates)


def scan_file(path, variables, lead_indices, stride, mode, chunk):
    rows = []
    date = path.name.split("_")[-3]
    month = int(date[4:6])
    season = int(date[:4]) if month >= 10 else int(date[:4]) - 1
    with Dataset(path) as ds:
        for name in variables:
            var = ds.variables[name]
            stats = (sampled_variable_stats(var, lead_indices, stride)
                     if mode == "sampled" else full_variable_stats(var, chunk))
            # Additional diagnostics: frequency and magnitude of negative humidity.
            # Read the exact same sample/chunks; do not modify original NetCDF.
            neg = 0
            finite = 0
            if name.startswith("q") and name[1:].isdigit():
                if mode == "sampled":
                    for lead in lead_indices:
                        arr = np.ma.asarray(var[:, lead, ::stride, ::stride])
                        values = arr.compressed()
                        values = values[np.isfinite(values)]
                        neg += int(np.count_nonzero(values < 0))
                        finite += values.size
                else:
                    for start in range(0, var.shape[1], chunk):
                        arr = np.ma.asarray(var[:, start:start + chunk])
                        values = arr.compressed()
                        values = values[np.isfinite(values)]
                        neg += int(np.count_nonzero(values < 0))
                        finite += values.size
            rows.append(dict(file=path.name, init_date=date, season=season,
                             variable=name, units=getattr(var, "units", ""),
                             mode=mode, spatial_stride=stride if mode == "sampled" else 1,
                             negative_count=neg if finite else "",
                             negative_fraction=neg / finite if finite else "",
                             **stats))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--root")
    p.add_argument("--outdir", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--metadata-workers", type=int, default=4)
    p.add_argument("--samples-per-season", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-lightweight", action="store_true")
    p.add_argument("--deep-mode", choices=["sampled", "full"], default="sampled")
    p.add_argument("--lead-days", default="0,7,14,21,28,35,42")
    p.add_argument("--spatial-stride", type=int, default=4)
    p.add_argument("--lead-chunk", type=int, default=16)
    args = p.parse_args()
    cfg = load_forecast_config(args.config)
    dates = initialization_dates(cfg)
    root = Path(args.root) if args.root else cfg.final_root
    files = sorted(root.glob("*_42day_subset.nc"))
    if len(files) != len(dates):
        raise SystemExit(f"File count mismatch: {len(files)} != {len(dates)}")
    if not args.skip_lightweight and not lightweight(files, cfg, dates, args.metadata_workers):
        raise SystemExit("Lightweight QC FAILED; deep QC not started")
    if args.workers < 1 or args.metadata_workers < 1 or args.samples_per_season < 1 or args.spatial_stride < 1 or args.lead_chunk < 1:
        raise SystemExit("workers, samples-per-season, stride, and chunk must be positive")
    lead_days = [int(v) for v in args.lead_days.split(",")]
    if any(d < 0 or d > cfg.forecast_days or d * 24 % cfg.hours_per_step for d in lead_days):
        raise SystemExit("Invalid lead days")
    leads = [d * 24 // cfg.hours_per_step for d in lead_days]
    by_season = defaultdict(list)
    for path in files:
        date = path.name.split("_")[-3]
        year, month = int(date[:4]), int(date[4:6])
        by_season[year if month >= 10 else year - 1].append(path)
    rng = random.Random(args.seed)
    sample = []
    for season, group in sorted(by_season.items()):
        if len(group) < args.samples_per_season:
            raise SystemExit(f"Too few files for season {season}")
        sample.extend(rng.sample(sorted(group), args.samples_per_season))
    sample.sort()
    print(f"{cfg.model_name} deep QC: {len(sample)} files, {args.deep_mode}", flush=True)
    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_map = {
            pool.submit(scan_file, path, cfg.variables, leads,
                        args.spatial_stride, args.deep_mode, args.lead_chunk): path
            for path in sample
        }
        for i, future in enumerate(as_completed(future_map), 1):
            path = future_map[future]
            try:
                rows.extend(future.result())
                print(f"[{i}/{len(sample)}] {path.name}", flush=True)
            except Exception as exc:
                errors.append((path.name, repr(exc)))
                print(f"ERROR {path.name}: {exc!r}", flush=True)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{cfg.model_name.lower()}_qc_{len(sample)}_{args.deep_mode}"
    detailed = out / f"{tag}_detailed.csv"
    fields = list(rows[0]) if rows else []
    if fields:
        with detailed.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    summary = out / f"{tag}_summary.csv"
    with summary.open("w", newline="") as fp:
        cols = ["variable", "files_sampled", "overall_min", "overall_max",
                "mean_of_file_means", "max_missing_fraction", "total_inf_count",
                "total_negative_count", "max_negative_fraction"]
        writer = csv.DictWriter(fp, fieldnames=cols)
        writer.writeheader()
        for variable in cfg.variables:
            group = [r for r in rows if r["variable"] == variable]
            writer.writerow({
                "variable": variable,
                "files_sampled": len(group),
                "overall_min": min((r["min"] for r in group), default=float("nan")),
                "overall_max": max((r["max"] for r in group), default=float("nan")),
                "mean_of_file_means": statistics.fmean(r["mean"] for r in group) if group else float("nan"),
                "max_missing_fraction": max((r["missing_fraction"] for r in group), default=float("nan")),
                "total_inf_count": sum(r["n_inf"] for r in group),
                "total_negative_count": sum(r["negative_count"] for r in group if r["negative_count"] != ""),
                "max_negative_fraction": max((r["negative_fraction"] for r in group if r["negative_fraction"] != ""), default=""),
            })
    print(f"Detailed: {detailed}\nSummary: {summary}\nErrors: {errors}", flush=True)
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
