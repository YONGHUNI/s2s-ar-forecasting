#!/usr/bin/env python3
"""Single-initialization 42-day Pangu6 rollout; never touch GraphCast outputs."""
from __future__ import annotations
import argparse
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from earth2studio import run
from earth2studio.data import ARCO
from earth2studio.io import ZarrBackend
from earth2studio.models.px import Pangu6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/pangu_42day_test.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    exp = cfg["experiment"]
    init = datetime.fromisoformat(exp["initialization"])
    days, hours = int(exp["forecast_days"]), int(exp["hours_per_step"])
    if hours != 6 or days != 42:
        raise ValueError("This test is intentionally restricted to the 42-day / 6h experiment")
    steps = days * 24 // hours
    expected = int(exp["expected_lead_times"])
    assert steps + 1 == expected
    root = Path(cfg["output"]["root"].format(user=os.environ["USER"]))
    root.mkdir(parents=True, exist_ok=True)
    tag = init.strftime("%Y%m%d_%H%M%S")
    target = root / f"pangu6_{tag}_42day.zarr"
    if target.exists():
        raise FileExistsError(f"Existing output retained (possibly incomplete): {target}")
    print(f"init={init.isoformat()} steps={steps} lead_count={expected}", flush=True)
    print(f"output={target}", flush=True)
    print(f"slurm_job={os.environ.get('SLURM_JOB_ID')} git_commit="
          f"{subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}", flush=True)
    started = time.monotonic()
    package = Pangu6.load_default_package()
    model = Pangu6.load_model(package)
    print(f"model_load_seconds={time.monotonic()-started:.1f}", flush=True)
    backend = ZarrBackend(file_name=str(target))
    run.deterministic([init], steps, model, ARCO(), backend, device="cuda")
    print(f"rollout_seconds={time.monotonic()-started:.1f}", flush=True)
    ds = xr.open_zarr(str(target))
    try:
        actual = ds.lead_time.values.astype("timedelta64[h]")
        desired = np.arange(expected).astype("timedelta64[h]") * hours
        if not np.array_equal(actual, desired):
            raise RuntimeError(f"Lead-time mismatch: shape={actual.shape}, final={actual[-1]}")
        if int(ds.sizes.get("time", -1)) != 1:
            raise RuntimeError("Unexpected time dimension")
        for dim, val in (("lat", 721), ("lon", 1440)):
            if int(ds.sizes.get(dim, -1)) != val:
                raise RuntimeError(f"Unexpected {dim}: {ds.sizes.get(dim)}")
        if len(ds.data_vars) != int(cfg["validation"]["expected_variable_count"]):
            raise RuntimeError(f"Unexpected number of variables: {len(ds.data_vars)}")
        print(f"QC passed: variables={len(ds.data_vars)}, lead_times={len(actual)}, "
              f"last_lead={actual[-1]}, dims={dict(ds.sizes)}", flush=True)
    finally:
        ds.close()
    (root / f"pangu6_{tag}_42day.complete").write_text(
        f"output={target}\ncompleted={datetime.now().astimezone().isoformat()}\n")
    print("Pangu6 42-day test completed.", flush=True)


if __name__ == "__main__":
    main()
