#!/usr/bin/env python3
"""Benchmark NetCDF QC sampling strategies on an actual output file.

Execute on a Slurm CPU compute node, not a login node.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

from netCDF4 import Dataset

from qc_sampling import variable_stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--variables", default="msl,q500")
    p.add_argument("--lead-days", default="0,7,14,21,28,35,42")
    p.add_argument("--spatial-stride", type=int, default=4)
    args = p.parse_args()
    leads = [int(day) * 4 for day in args.lead_days.split(",")]
    if args.spatial_stride < 1:
        p.error("--spatial-stride must be positive")

    with Dataset(args.file) as ds:
        for name in args.variables.split(","):
            var = ds.variables[name]
            print(
                f"{name}: shape={var.shape}; chunks={var.chunking()}; "
                f"filters={var.filters()}",
                flush=True,
            )
            outputs = {}
            for mode in ("legacy", "contiguous"):
                start = time.perf_counter()
                outputs[mode] = variable_stats(
                    var, mode="sampled", lead_indices=leads,
                    spatial_stride=args.spatial_stride, lead_chunk=16,
                    sample_read_mode=mode,
                    count_negative=name.startswith("q") and name[1:].isdigit(),
                )
                print(
                    f"  {mode}: {time.perf_counter()-start:.2f} s",
                    flush=True,
                )
            old, new = outputs["legacy"], outputs["contiguous"]
            old_stats, old_neg, old_finite = old
            new_stats, new_neg, new_finite = new
            if (old_neg, old_finite) != (new_neg, new_finite):
                raise SystemExit(f"{name}: negative/finite counts differ")
            for key, lhs in old_stats.items():
                rhs = new_stats[key]
                if isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)):
                    if not math.isclose(lhs, rhs, rel_tol=1e-6, abs_tol=1e-6):
                        raise SystemExit(f"{name}: {key} differs: {lhs} vs {rhs}")
                elif lhs != rhs:
                    raise SystemExit(f"{name}: {key} differs: {lhs} vs {rhs}")
            print("  QC statistics: MATCH", flush=True)


if __name__ == "__main__":
    main()
