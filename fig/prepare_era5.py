#!/usr/bin/env python3
"""Prefetch required ARCO ERA5 valid times once, in restartable monthly NetCDFs."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta
import os
from pathlib import Path
import re

import numpy as np
import yaml
from netCDF4 import Dataset, date2num

EPOCH_UNITS = "hours since 1970-01-01 00:00:00"
FILE_RE = re.compile(r"graphcast_operational_(\d{8})_42day_subset\.nc$")


def config(path):
    with open(path, encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    for group in ("forecast", "reference", "output"):
        settings[group]["root"] = Path(settings[group]["root"].replace("{user}", os.environ["USER"]))
    return settings


def inventory(cfg):
    files = sorted(cfg["forecast"]["root"].glob(cfg["forecast"]["pattern"]))
    expected = int(cfg["forecast"]["expected_initializations"])
    if len(files) != expected:
        raise RuntimeError(f"Expected {expected} forecast files, got {len(files)}: {cfg['forecast']['root']}")
    dates = []
    for p in files:
        match = FILE_RE.fullmatch(p.name)
        if not match:
            raise ValueError(f"Unexpected forecast filename: {p.name}")
        dates.append(datetime.strptime(match.group(1), "%Y%m%d"))
    if len(set(dates)) != len(dates):
        raise ValueError("Duplicate forecast initialization dates")
    return files, dates


def monthly_valid_times(cfg):
    _, dates = inventory(cfg)
    steps = int(cfg["forecast"]["steps"])
    hours = int(cfg["forecast"]["hours_per_step"])
    grouped = defaultdict(set)
    for init in dates:
        for k in range(steps):
            t = init + timedelta(hours=hours * k)
            grouped[t.strftime("%Y%m")].add(t)
    return {month: sorted(times) for month, times in sorted(grouped.items())}


def extract_one(data_source, valid_time, stride):
    # Earth2Studio ARCO returns (time, variable, lat, lon) with ERA5 units.
    data = data_source([valid_time], ["t2m", "tp06"])
    if not all(x in data.dims for x in ("time", "variable", "lat", "lon")):
        raise RuntimeError(f"Unexpected ARCO dimensions: {data.dims}")
    if data.sizes["time"] != 1:
        raise RuntimeError("ARCO returned multiple times")
    lat = np.asarray(data.lat.values)[::stride].astype("f4")
    lon = np.asarray(data.lon.values)[::stride].astype("f4")
    fields = {}
    for var in ("t2m", "tp06"):
        field = np.asarray(data.sel(variable=var).isel(time=0).transpose("lat", "lon").values)
        fields[var] = np.asarray(field[::stride, ::stride], dtype="f4")
    return lat, lon, fields


def prepare_one_month(cfg, month, timestamps):
    root = cfg["reference"]["root"]
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"era5_{month}.nc"
    if path.is_file():
        print(f"SKIP existing {path}", flush=True)
        return
    partial = root / f"era5_{month}.nc.partial"
    stride = int(cfg["reference"]["stride"])
    if stride < 1:
        raise ValueError("Reference stride must be positive")
    # Only import Earth2Studio for actual remote access, not for --list-months.
    from earth2studio.data import ARCO
    source = ARCO()
    start = 0
    if partial.exists():
        with Dataset(partial, "r") as ds:
            if len(ds.dimensions["time"]) != len(timestamps) or int(ds.getncattr("stride")) != stride:
                raise ValueError(f"Stale reference partial with incompatible settings: {partial}")
            actual = np.asarray(ds.variables["time"][:])
            expected = np.asarray(date2num(timestamps, units=EPOCH_UNITS), dtype="f8")
            if not np.allclose(actual, expected, rtol=0, atol=1e-7):
                raise ValueError(f"Stale reference timestamps: {partial}")
            start = int(ds.getncattr("completed_count"))
    if not partial.exists():
        lat, lon, _ = extract_one(source, timestamps[0], stride)
        with Dataset(partial, "w", format="NETCDF4") as dst:
            dst.createDimension("time", len(timestamps))
            dst.createDimension("lat", len(lat))
            dst.createDimension("lon", len(lon))
            tv = dst.createVariable("time", "f8", ("time",))
            tv.units = EPOCH_UNITS
            tv.calendar = "standard"
            tv[:] = date2num(timestamps, units=EPOCH_UNITS)
            dst.createVariable("lat", "f4", ("lat",))[:] = lat
            dst.createVariable("lon", "f4", ("lon",))[:] = lon
            for name, units in (("t2m", "K"), ("tp06", "m")):
                variable = dst.createVariable(name, "f4", ("time", "lat", "lon"),
                    zlib=True, complevel=2, shuffle=True,
                    chunksizes=(1, len(lat), len(lon)), fill_value=np.float32(-9999))
                variable.units = units
            dst.setncattr("source", "Earth2Studio ARCO ERA5")
            dst.setncattr("stride", stride)
            dst.setncattr("completed_count", 0)
    with Dataset(partial, "a") as dst:
        original_lat = np.asarray(dst.variables["lat"][:])
        original_lon = np.asarray(dst.variables["lon"][:])
        for j in range(start, len(timestamps)):
            lat, lon, fields = extract_one(source, timestamps[j], stride)
            if not np.allclose(lat, original_lat, atol=0.0001, rtol=0) or not np.allclose(lon, original_lon, atol=0.0001, rtol=0):
                raise RuntimeError("Inconsistent ARCO coordinates")
            for name, field in fields.items():
                dst.variables[name][j, :, :] = np.ma.masked_invalid(field)
            dst.setncattr("completed_count", j + 1)
            dst.sync()
            if j % 12 == 0 or j == len(timestamps) - 1:
                print(month, j + 1, "/", len(timestamps), timestamps[j].isoformat(), flush=True)
    partial.replace(path)
    print("COMPLETE", path, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="fig/config.yaml")
    p.add_argument("--list-months", action="store_true")
    p.add_argument("--count-months", action="store_true")
    p.add_argument("--month-index", type=int)
    args = p.parse_args()
    cfg = config(args.config)
    months = monthly_valid_times(cfg)
    if args.count_months:
        print(len(months))
        return
    if args.list_months:
        print("\n".join(f"{i:3d} {month} ({len(times)} valid times)" for i, (month, times) in enumerate(months.items())))
        return
    if args.month_index is None or not 0 <= args.month_index < len(months):
        p.error(f"--month-index must be in 0..{len(months)-1}")
    month, times = list(months.items())[args.month_index]
    prepare_one_month(cfg, month, times)


if __name__ == "__main__":
    main()
