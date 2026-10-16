#!/usr/bin/env python3
"""One-step Pangu6 smoke test. Writes outside GraphCast output roots."""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from earth2studio import run
from earth2studio.data import ARCO
from earth2studio.io import ZarrBackend
from earth2studio.models.px import Pangu6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", default="2024-10-01T00:00:00")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    root = Path(args.output_root or f"/scratch/{os.environ['USER']}/weather-ai/pangu-operational/smoke")
    root.mkdir(parents=True, exist_ok=True)
    tag = args.init.replace("-", "").replace(":", "").replace("T", "_")
    dest = root / f"pangu6_{tag}_one_step.zarr"
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite existing forecast: {dest}")

    print(f"initialization={args.init}", flush=True)
    print(f"output={dest}", flush=True)
    package = Pangu6.load_default_package()
    model = Pangu6.load_model(package)
    data = ARCO()
    io = ZarrBackend(file_name=str(dest))
    run.deterministic([datetime.fromisoformat(args.init)], 1, model, data, io, device="cuda")

    import xarray as xr
    ds = xr.open_zarr(str(dest))
    expected = [np.timedelta64(0, "h"), np.timedelta64(6, "h")]
    actual = np.asarray(ds.lead_time.values).astype("timedelta64[h]")
    assert actual.shape == (2,), f"Unexpected lead coordinates: {actual}"
    assert np.array_equal(actual, expected), f"Unexpected lead coordinates: {actual}"
    print(f"variables={len(ds.data_vars)}", flush=True)
    print(f"lead_times={actual}", flush=True)
    print(f"dims={dict(ds.sizes)}", flush=True)
    print("Pangu6 one-step test passed.", flush=True)
    ds.close()


if __name__ == "__main__":
    main()
