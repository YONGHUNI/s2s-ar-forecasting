#!/usr/bin/env python3
"""Convert the existing Pangu6 42-day Zarr to NetCDF on a CPU node.

Keep the source and publish the destination only after structure and sampled
value checks. Never import GraphCast-specific variable metadata.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import xarray as xr


def validate(ds, *, check_samples: bool):
    expected = {"time": 1, "lead_time": 169, "lat": 721, "lon": 1440}
    for dim, size in expected.items():
        if ds.sizes.get(dim) != size:
            raise ValueError(f"{dim}: expected {size}, got {ds.sizes.get(dim)}")
    if len(ds.data_vars) != 69:
        raise ValueError(f"Expected 69 variables, got {len(ds.data_vars)}")
    actual = ds.lead_time.values.astype("timedelta64[h]")
    wanted = np.arange(169).astype("timedelta64[h]") * 6
    if not np.array_equal(actual, wanted):
        raise ValueError("Lead-time coordinate differs from 0..1008 hours / 6h")
    if check_samples:
        for name in ds.data_vars:
            da = ds[name]
            sample = da.isel(lead_time=[0, 84, 168], lat=[0, 180, 360, 540, 720],
                             lon=[0, 360, 720, 1080]).values
            if not np.isfinite(sample).all():
                raise ValueError(f"Non-finite sampled values: {name}")
    return list(ds.data_vars)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=None)
    p.add_argument("--destination", default=None)
    p.add_argument("--compression", type=int, default=2)
    args = p.parse_args()
    if not 0 <= args.compression <= 9:
        p.error("--compression must be in 0..9")
    user = os.environ["USER"]
    src = Path(args.source or f"/scratch/{user}/weather-ai/pangu-operational/long-test/pangu6_20241001_000000_42day.zarr")
    dst = Path(args.destination or f"/scratch/{user}/weather-ai/pangu-operational/netcdf/pangu6_20241001_000000_42day.nc")
    tmp = Path(str(dst) + ".partial")
    if not src.is_dir():
        raise FileNotFoundError(src)
    if dst.exists() or tmp.exists():
        raise FileExistsError(f"Refusing to overwrite destination or partial: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"source={src}\ndestination={dst}", flush=True)
    started = time.monotonic()
    with xr.open_zarr(src, chunks=None, consolidated=False) as ds:
        vars_ = validate(ds, check_samples=True)
        coords = xr.Dataset(coords={k: v for k, v in ds.coords.items()}, attrs=dict(ds.attrs))
        coords.to_netcdf(tmp, mode="w", engine="netcdf4")
        for i, name in enumerate(vars_, 1):
            one = xr.Dataset({name: ds[name].variable.copy(deep=False)})
            one.to_netcdf(tmp, mode="a", engine="netcdf4",
                          encoding={name: {"zlib": args.compression > 0,
                                           "complevel": args.compression,
                                           "shuffle": args.compression > 0}})
            print(f"encoded {i}/{len(vars_)}: {name}", flush=True)

    with xr.open_dataset(tmp, engine="netcdf4") as nc:
        names = validate(nc, check_samples=True)
    tmp.replace(dst)
    print(f"NETCDF_OK variables={len(names)} bytes={dst.stat().st_size} "
          f"elapsed_seconds={time.monotonic()-started:.1f}", flush=True)


if __name__ == "__main__":
    main()
