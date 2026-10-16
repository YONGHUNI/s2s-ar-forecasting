#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import statistics
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


LEVELS = [50, 100, 150, 200, 250, 300, 400,
          500, 600, 700, 850, 925, 1000]

SURFACE_VARS = ["msl", "t2m", "tp06", "u10m", "v10m"]

PRESSURE_VARS = [
    f"{v}{level}"
    for v in ["t", "u", "v", "q", "z"]
    for level in LEVELS
]

EXPECTED_VARS = SURFACE_VARS + PRESSURE_VARS
DATE_RE = re.compile(r"_(\d{8})_42day_subset\.nc$")


def get_init_date(path: Path) -> str:
    m = DATE_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse initialization date: {path.name}")
    return m.group(1)


def get_season(path: Path) -> int:
    date = get_init_date(path)
    year = int(date[:4])
    month = int(date[4:6])
    return year if month >= 10 else year - 1


def lightweight_qc(files: list[Path], expected_files: int) -> bool:
    print("=" * 72, flush=True)
    print("GraphCast lightweight QC", flush=True)
    print("=" * 72, flush=True)

    print(f"Files found:          {len(files)}", flush=True)
    print(f"Expected files:       {expected_files}", flush=True)
    print(f"Expected variables:   {len(EXPECTED_VARS)}", flush=True)
    print("Expected lead times:  169", flush=True)

    sizes = [f.stat().st_size for f in files]
    size_gib = [x / 1024**3 for x in sizes]

    med = statistics.median(size_gib) if size_gib else 0.0

    if size_gib:
        print(
            f"File size [GiB]:      "
            f"min={min(size_gib):.2f}, "
            f"median={med:.2f}, "
            f"max={max(size_gib):.2f}",
            flush=True,
        )

    open_errors = []
    missing_errors = []
    lead_errors = []
    shape_errors = []
    size_errors = []

    ref_shapes = None
    ref_file = None

    for i, path in enumerate(files, 1):
        try:
            with Dataset(path, "r") as ds:
                missing = [v for v in EXPECTED_VARS if v not in ds.variables]

                if missing:
                    missing_errors.append((path.name, ",".join(missing)))
                    continue

                if "lead_time" not in ds.dimensions:
                    lead_errors.append((path.name, "missing"))
                elif len(ds.dimensions["lead_time"]) != 169:
                    lead_errors.append(
                        (path.name, len(ds.dimensions["lead_time"]))
                    )

                shapes = {
                    v: tuple(ds.variables[v].shape)
                    for v in EXPECTED_VARS
                }

                if ref_shapes is None:
                    ref_shapes = shapes
                    ref_file = path.name
                else:
                    bad = [
                        v for v in EXPECTED_VARS
                        if shapes[v] != ref_shapes[v]
                    ]
                    if bad:
                        shape_errors.append((path.name, ",".join(bad)))

        except Exception as exc:
            open_errors.append((path.name, repr(exc)))

        if i % 100 == 0:
            print(
                f"[lightweight {i}/{len(files)}]",
                flush=True,
            )

    if med:
        for path, size in zip(files, size_gib):
            if size < med * 0.80:
                size_errors.append((path.name, size))

    print(flush=True)
    print(f"Reference file:       {ref_file}", flush=True)
    print(f"Open failures:        {len(open_errors)}", flush=True)
    print(f"Missing-var files:    {len(missing_errors)}", flush=True)
    print(f"Lead-time mismatches: {len(lead_errors)}", flush=True)
    print(f"Shape mismatches:     {len(shape_errors)}", flush=True)
    print(f"Small-size outliers:  {len(size_errors)}", flush=True)

    for title, errors in (
        ("OPEN ERRORS", open_errors),
        ("MISSING VARIABLES", missing_errors),
        ("LEAD-TIME ERRORS", lead_errors),
        ("SHAPE ERRORS", shape_errors),
        ("SIZE OUTLIERS", size_errors),
    ):
        if errors:
            print(f"\n[{title}]", flush=True)
            for item in errors:
                print("  ", *item, flush=True)

    ok = (
        len(files) == expected_files
        and not open_errors
        and not missing_errors
        and not lead_errors
        and not shape_errors
        and not size_errors
    )

    print(flush=True)
    print(
        "Lightweight QC:",
        "PASS" if ok else "WARNING/FAIL",
        flush=True,
    )
    print("=" * 72, flush=True)

    return ok


def empty_state():
    # finite_count, mean, M2, min, max, missing, inf, total
    return [0, 0.0, 0.0, math.inf, -math.inf, 0, 0, 0]


def merge_array(state, arr):
    if np.ma.isMaskedArray(arr):
        mask = np.ma.getmaskarray(arr)
        data = np.asarray(arr.data)
        masked_count = int(np.count_nonzero(mask))

        if masked_count:
            values = data[~mask]
        else:
            values = data.reshape(-1)
    else:
        data = np.asarray(arr)
        masked_count = 0
        values = data.reshape(-1)

    total = data.size
    missing = masked_count
    inf_count = 0

    if np.issubdtype(values.dtype, np.inexact):
        finite_mask = np.isfinite(values)
        nan_count = int(np.count_nonzero(np.isnan(values)))
        inf_count = int(np.count_nonzero(np.isinf(values)))

        missing += nan_count
        values = values[finite_mask]

    state[5] += missing
    state[6] += inf_count
    state[7] += total

    if values.size == 0:
        return

    values = values.astype(np.float64, copy=False)

    n2 = int(values.size)
    mean2 = float(values.mean(dtype=np.float64))
    var2 = float(values.var(dtype=np.float64))
    m2_2 = var2 * n2

    min2 = float(values.min())
    max2 = float(values.max())

    n1, mean1, m2_1 = state[0], state[1], state[2]

    if n1 == 0:
        state[0] = n2
        state[1] = mean2
        state[2] = m2_2
    else:
        n = n1 + n2
        delta = mean2 - mean1

        state[1] = mean1 + delta * n2 / n
        state[2] = (
            m2_1
            + m2_2
            + delta * delta * n1 * n2 / n
        )
        state[0] = n

    state[3] = min(state[3], min2)
    state[4] = max(state[4], max2)


def finish_state(state):
    n, mean, m2, minval, maxval, missing, inf_count, total = state

    if n:
        std = math.sqrt(m2 / n)
    else:
        mean = math.nan
        std = math.nan
        minval = math.nan
        maxval = math.nan

    return {
        "n_total": total,
        "n_finite": n,
        "n_missing": missing,
        "missing_fraction": missing / total if total else math.nan,
        "n_inf": inf_count,
        "min": minval,
        "max": maxval,
        "mean": mean,
        "std": std,
    }


def sampled_variable_stats(
    var,
    lead_indices: list[int],
    spatial_stride: int,
):
    """
    Read selected lead times and a regular spatial subsample.

    This is intended as a production sanity check, not an exact global
    extrema calculation.
    """
    slices = [slice(None)] * var.ndim

    for axis, dim in enumerate(var.dimensions):
        lower = dim.lower()

        if lower == "lead_time":
            valid = [i for i in lead_indices if i < var.shape[axis]]
            slices[axis] = valid

        elif lower in {"lat", "latitude", "lon", "longitude"}:
            slices[axis] = slice(None, None, spatial_stride)

    arr = var[tuple(slices)]

    state = empty_state()
    merge_array(state, arr)
    return finish_state(state)


def full_variable_stats(var, lead_chunk: int):
    """
    Exact full-field scan. This reads essentially the complete variable.
    """
    state = empty_state()

    if "lead_time" not in var.dimensions:
        merge_array(state, var[:])
        return finish_state(state)

    axis = var.dimensions.index("lead_time")
    length = var.shape[axis]

    for start in range(0, length, lead_chunk):
        stop = min(start + lead_chunk, length)

        slices = [slice(None)] * var.ndim
        slices[axis] = slice(start, stop)

        merge_array(state, var[tuple(slices)])

    return finish_state(state)


def scan_one_file(
    path_string: str,
    mode: str,
    lead_indices: list[int],
    spatial_stride: int,
    lead_chunk: int,
):
    path = Path(path_string)
    init_date = get_init_date(path)
    season = get_season(path)

    rows = []

    with Dataset(path, "r") as ds:
        for variable in EXPECTED_VARS:
            var = ds.variables[variable]

            if mode == "sampled":
                stats = sampled_variable_stats(
                    var,
                    lead_indices=lead_indices,
                    spatial_stride=spatial_stride,
                )
            else:
                stats = full_variable_stats(
                    var,
                    lead_chunk=lead_chunk,
                )

            rows.append({
                "file": path.name,
                "init_date": init_date,
                "season": season,
                "variable": variable,
                "units": getattr(var, "units", ""),
                "pressure_level": getattr(var, "pressure_level", ""),
                "shape": "x".join(str(x) for x in var.shape),
                "mode": mode,
                "spatial_stride":
                    spatial_stride if mode == "sampled" else 1,
                **stats,
            })

    return rows


def choose_samples(files, seed, samples_per_season):
    groups = defaultdict(list)

    for f in files:
        groups[get_season(f)].append(f)

    rng = random.Random(seed)
    selected = []

    for season in sorted(groups):
        group = sorted(groups[season])

        if len(group) < samples_per_season:
            raise RuntimeError(
                f"Season {season}: only {len(group)} files"
            )

        selected.extend(
            rng.sample(group, samples_per_season)
        )

    selected.sort(key=get_init_date)
    return selected


def write_detailed(rows, path):
    fields = [
        "file",
        "init_date",
        "season",
        "variable",
        "units",
        "pressure_level",
        "shape",
        "mode",
        "spatial_stride",
        "n_total",
        "n_finite",
        "n_missing",
        "missing_fraction",
        "n_inf",
        "min",
        "max",
        "mean",
        "std",
    ]

    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows, path):
    groups = defaultdict(list)

    for row in rows:
        groups[row["variable"]].append(row)

    fields = [
        "variable",
        "files_sampled",
        "mode",
        "overall_min",
        "overall_max",
        "mean_of_file_means",
        "median_file_mean",
        "min_file_mean",
        "max_file_mean",
        "median_file_std",
        "max_missing_fraction",
        "total_inf_count",
    ]

    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()

        for variable in EXPECTED_VARS:
            rs = groups[variable]

            mins = [r["min"] for r in rs if math.isfinite(r["min"])]
            maxs = [r["max"] for r in rs if math.isfinite(r["max"])]
            means = [r["mean"] for r in rs if math.isfinite(r["mean"])]
            stds = [r["std"] for r in rs if math.isfinite(r["std"])]

            writer.writerow({
                "variable": variable,
                "files_sampled": len(rs),
                "mode": rs[0]["mode"] if rs else "",
                "overall_min": min(mins) if mins else math.nan,
                "overall_max": max(maxs) if maxs else math.nan,
                "mean_of_file_means":
                    statistics.fmean(means) if means else math.nan,
                "median_file_mean":
                    statistics.median(means) if means else math.nan,
                "min_file_mean":
                    min(means) if means else math.nan,
                "max_file_mean":
                    max(means) if means else math.nan,
                "median_file_std":
                    statistics.median(stds) if stds else math.nan,
                "max_missing_fraction":
                    max(r["missing_fraction"] for r in rs),
                "total_inf_count":
                    sum(r["n_inf"] for r in rs),
            })


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", required=True)
    parser.add_argument("--outdir", required=True)

    parser.add_argument("--expected-files", type=int, default=607)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--samples-per-season", type=int, default=3)

    parser.add_argument(
        "--skip-lightweight",
        action="store_true",
    )

    parser.add_argument(
        "--deep-mode",
        choices=["sampled", "full"],
        default="sampled",
    )

    parser.add_argument(
        "--lead-days",
        default="0,7,14,21,28,35,42",
        help="Comma-separated forecast lead days for sampled mode",
    )

    parser.add_argument(
        "--spatial-stride",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lead-chunk",
        type=int,
        default=16,
        help="Lead-time chunk size for exact full mode",
    )

    args = parser.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(root.glob("*.nc"))

    if not args.skip_lightweight:
        lightweight_qc(files, args.expected_files)
    else:
        print(
            f"Lightweight QC skipped "
            f"(found {len(files)} NetCDF files).",
            flush=True,
        )

    if len(files) != args.expected_files:
        raise RuntimeError(
            f"Expected {args.expected_files} files, found {len(files)}"
        )

    lead_days = [
        int(x)
        for x in args.lead_days.split(",")
        if x.strip()
    ]

    # Forecast interval is 6 h:
    # 0, 7, 14, ..., 42 days -> indices 0, 28, 56, ..., 168.
    lead_indices = [
        day * 4
        for day in lead_days
    ]

    samples = choose_samples(
        files,
        args.seed,
        args.samples_per_season,
    )

    print(flush=True)
    print("=" * 72, flush=True)
    print("GraphCast deep QC", flush=True)
    print("=" * 72, flush=True)
    print(f"Mode:              {args.deep_mode}", flush=True)
    print(f"Workers:           {args.workers}", flush=True)
    print(f"Files selected:    {len(samples)}", flush=True)
    print(
        f"Samples/season:    {args.samples_per_season}",
        flush=True,
    )

    if args.deep_mode == "sampled":
        print(
            f"Lead days:         {lead_days}",
            flush=True,
        )
        print(
            f"Lead indices:      {lead_indices}",
            flush=True,
        )
        print(
            f"Spatial stride:    {args.spatial_stride}",
            flush=True,
        )
        print(
            "NOTE: sampled min/max are not exact global extrema.",
            flush=True,
        )
    else:
        print(
            f"Full scan chunk:   {args.lead_chunk} leads",
            flush=True,
        )

    for season in sorted(set(get_season(f) for f in samples)):
        print(f"  {season}-{season + 1}:", flush=True)

        for f in samples:
            if get_season(f) == season:
                print(f"    {f.name}", flush=True)

    all_rows = []
    errors = []

    with ProcessPoolExecutor(
        max_workers=args.workers
    ) as executor:

        futures = {
            executor.submit(
                scan_one_file,
                str(path),
                args.deep_mode,
                lead_indices,
                args.spatial_stride,
                args.lead_chunk,
            ): path
            for path in samples
        }

        completed = 0

        for future in as_completed(futures):
            path = futures[future]

            try:
                rows = future.result()
                all_rows.extend(rows)

                completed += 1
                print(
                    f"[deep QC {completed:02d}/{len(samples):02d}] "
                    f"{path.name}",
                    flush=True,
                )

            except Exception as exc:
                errors.append((path.name, repr(exc)))
                print(
                    f"[deep QC ERROR] {path.name}: {exc!r}",
                    flush=True,
                )

    mode_tag = args.deep_mode

    detailed = (
        outdir /
        f"graphcast_qc_{len(samples)}_{mode_tag}_detailed.csv"
    )

    summary = (
        outdir /
        f"graphcast_qc_{len(samples)}_{mode_tag}_summary.csv"
    )

    write_detailed(all_rows, detailed)
    write_summary(all_rows, summary)

    print(flush=True)
    print("=" * 72, flush=True)
    print("Deep QC finished", flush=True)
    print(f"Detailed CSV: {detailed}", flush=True)
    print(f"Summary CSV:  {summary}", flush=True)
    print(f"Errors:       {len(errors)}", flush=True)
    print("=" * 72, flush=True)

    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
