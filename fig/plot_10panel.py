#!/usr/bin/env python3
"""Render one 10-panel 169-frame GraphCast vs ARCO ERA5 comparison GIF."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from contextlib import ExitStack
import os
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import yaml
from netCDF4 import Dataset, date2num

from fig.prepare_era5 import config, inventory, EPOCH_UNITS, FILE_RE


def percentiles_full_field(values, levels=(90, 95, 99)):
    """Exact quantiles over all valid lead times and grid cells of one variable."""
    a = np.ma.filled(np.ma.asarray(values), np.nan)
    a = np.asarray(a, dtype="f4")
    if not np.isfinite(a).any():
        raise ValueError("Variable contains no finite values")
    return np.nanpercentile(a, levels)


def normalize_precip(values, negative_tolerance_m):
    """Clip only tiny numerical negative amounts, otherwise fail explicitly."""
    a = np.asarray(np.ma.filled(np.ma.asarray(values), np.nan), dtype="f4")
    finite = a[np.isfinite(a)]
    if finite.size and np.min(finite) < -float(negative_tolerance_m):
        print(f"QC WARNING: negative precipitation minimum {float(np.min(finite))*1000:.4f} mm; clipping to zero", flush=True)
    if finite.size and np.any(finite < 0):
        print(f"Clipping {np.count_nonzero(finite < 0)} tiny negative precipitation cells", flush=True)
    return np.maximum(a, 0.0)


def read_forecast(path, expected_steps, hours_per_step, neg_tolerance):
    match = FILE_RE.fullmatch(Path(path).name)
    if not match:
        raise ValueError(f"Unrecognized forecast filename: {path}")
    init = datetime.strptime(match.group(1), "%Y%m%d")
    with Dataset(path) as ds:
        if len(ds.dimensions["time"]) != 1:
            raise ValueError("Expected one initialization")
        if len(ds.dimensions["lead_time"]) != expected_steps:
            raise ValueError(f"Expected {expected_steps} lead times")
        lead = ds.variables["lead_time"]
        if getattr(lead, "units", "") != "hours":
            raise ValueError("Expected lead_time in hours")
        offsets = np.asarray(lead[:], dtype="f8")
        if not np.array_equal(offsets, np.arange(expected_steps) * hours_per_step):
            raise ValueError("Lead times are not the expected six-hour sequence")
        for name, unit in (("t2m", "K"), ("tp06", "m")):
            v = ds.variables[name]
            if v.dimensions != ("time", "lead_time", "lat", "lon"):
                raise ValueError(f"Unexpected {name} dimensions: {v.dimensions}")
            if getattr(v, "units", "") != unit:
                raise ValueError(f"Unexpected {name} units: {getattr(v, 'units', '?')}")
        lat = np.asarray(ds.variables["lat"][:], dtype="f4")
        lon = np.asarray(ds.variables["lon"][:], dtype="f4")
        # Only two payload variables (~0.7 GiB each); the full 31+GB dataset
        # is never loaded. Full resolution is needed for exact percentiles.
        t2m = np.asarray(np.ma.filled(ds.variables["t2m"][0, :, :, :], np.nan), dtype="f4")
        tp06 = normalize_precip(ds.variables["tp06"][0, :, :, :], neg_tolerance)
    return init, lat, lon, t2m, tp06


def setup_reference(stack, cfg, valid_times, lat, lon, stride):
    cache = {}
    wanted = sorted(set(t.strftime("%Y%m") for t in valid_times))
    for month in wanted:
        path = cfg["reference"]["root"] / f"era5_{month}.nc"
        if not path.is_file():
            raise FileNotFoundError(f"Missing ERA5 monthly cache: {path}. Run fig/submit.sh prepare first.")
        ds = stack.enter_context(Dataset(path))
        if int(ds.getncattr("completed_count")) != len(ds.dimensions["time"]):
            raise RuntimeError(f"Incomplete ERA5 cache: {path}")
        for coord, expected in (("lat", lat[::stride]), ("lon", lon[::stride])):
            got = np.asarray(ds.variables[coord][:])
            if not np.allclose(got, expected, rtol=0, atol=0.0002):
                raise ValueError(f"ERA5/forecast coordinate mismatch in {path}: {coord}")
        times = np.asarray(ds.variables["time"][:], dtype="f8")
        cache[month] = (ds, {int(round(x)): j for j, x in enumerate(times)})
    for valid in valid_times:
        key = valid.strftime("%Y%m")
        h = int(round(date2num(valid, units=EPOCH_UNITS)))
        if h not in cache[key][1]:
            raise KeyError(f"ERA5 reference missing valid time {valid}")
    return cache


def coastline_segments(longitudes):
    """Coastline outlines from bundled Natural Earth 1:110m land polygons.

    Split geometry crossing the lon=0/360 seam instead of drawing
    spurious lines across the map. Drawn once, never per GIF frame.
    """
    source = Path(__file__).with_name("ne_110m_land.geojson")
    with source.open(encoding="utf-8") as stream:
        features = json.load(stream)["features"]
    use_360 = float(np.nanmin(longitudes)) >= 0
    segments = []
    for feature in features:
        geom = feature["geometry"]
        polygons = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for polygon in polygons:
            for ring in polygon:
                coords = np.asarray(ring, dtype="f8")
                if coords.ndim != 2 or coords.shape[0] < 2:
                    continue
                if use_360:
                    coords[:, 0] = np.mod(coords[:, 0], 360.0)
                cuts = np.where(np.abs(np.diff(coords[:, 0])) > 180.0)[0] + 1
                for part in np.split(coords, cuts):
                    if len(part) >= 2:
                        segments.append(part)
    return segments


def plot_one(cfg, index, *, max_frames=None):
    files, _ = inventory(cfg)
    if not 0 <= index < len(files):
        raise IndexError(f"Forecast index {index} outside 0..{len(files)-1}")
    forecast = files[index]
    steps = int(cfg["forecast"]["steps"])
    hours = int(cfg["forecast"]["hours_per_step"])
    stride = int(cfg["plot"]["stride"])
    if stride < 1 or stride != int(cfg["reference"]["stride"]):
        raise ValueError("Plot and ERA5 reference strides must agree and be >= 1")
    if tuple(cfg["plot"]["percentiles"]) != (90, 95, 99):
        raise ValueError("This 10-panel layout requires [90,95,99] percentiles")
    init, lat, lon, t2m, tp06 = read_forecast(
        forecast, steps, hours, cfg["plot"]["precipitation_negative_tolerance_m"]
    )
    q_t = percentiles_full_field(t2m)
    q_p = percentiles_full_field(tp06)
    print(f"{forecast.name}: T2m K P90/P95/P99={q_t}; tp06 m={q_p}", flush=True)

    # The plot samples spatially only AFTER full-file quantile calculation.
    t2m = t2m[:, ::stride, ::stride]
    tp06 = tp06[:, ::stride, ::stride]
    nframes = steps if max_frames is None else min(steps, max_frames)
    if nframes <= 0:
        raise ValueError("max_frames must be positive")
    valid_times = [init + timedelta(hours=hours*k) for k in range(nframes)]
    outdir = cfg["output"]["root"]
    outdir.mkdir(parents=True, exist_ok=True)
    suffix = "" if nframes == steps else f"_preview_{nframes}"
    outfile = outdir / f"graphcast_10panel_wide_{init:%Y%m%d}{suffix}.gif"
    if outfile.exists() and outfile.stat().st_size:
        print(f"SKIP {outfile}", flush=True)
        return
    partial = outdir / (outfile.name + ".partial.gif")
    if partial.exists():
        raise FileExistsError(f"Stale incomplete GIF: {partial}")

    with ExitStack() as stack:
        reference = setup_reference(stack, cfg, valid_times, lat, lon, stride)
        a_t = np.ma.masked_invalid(t2m)
        a_p = np.ma.masked_invalid(tp06)
        temp_low = float(cfg["plot"]["t2m_min_c"])
        temp_high = float(cfg["plot"]["t2m_max_c"])
        precip_high = float(cfg["plot"]["pr_max_mm"])
        if not temp_low < temp_high or precip_high <= 0:
            raise ValueError("Invalid fixed colour limits in fig/config.yaml")
        temp_cmap = plt.get_cmap("RdYlBu_r").copy()
        precip_cmap = plt.get_cmap("turbo").copy()
        temp_cmap.set_bad((0, 0, 0, 0))
        precip_cmap.set_bad((0, 0, 0, 0))
        fig, axes = plt.subplots(2, 5, figsize=(19.2, 10.8))
        fig.subplots_adjust(left=.025, right=.985, top=.89, bottom=.145,
                            wspace=.07, hspace=.18)
        origin = "upper" if lat[0] > lat[-1] else "lower"
        extent = (float(np.min(lon)), float(np.max(lon)), float(np.min(lat)), float(np.max(lat)))
        titles = [
            ("GraphCast T2m",
             f"T2m ≥ P90 ({q_t[0]-273.15:.1f} °C)",
             f"T2m ≥ P95 ({q_t[1]-273.15:.1f} °C)",
             f"T2m ≥ P99 ({q_t[2]-273.15:.1f} °C)",
             "ERA5 T2m reference"),
            ("GraphCast PR (6h)",
             f"PR ≥ P90 ({q_p[0]*1000:.1f} mm)",
             f"PR ≥ P95 ({q_p[1]*1000:.1f} mm)",
             f"PR ≥ P99 ({q_p[2]*1000:.1f} mm)",
             "ERA5 PR (6h) reference"),
        ]
        coastlines = coastline_segments(lon)
        images = []
        for row in range(2):
            pair = []
            for col in range(5):
                ax = axes[row, col]
                ax.set_facecolor("#f1f3f5")
                ax.set_title(titles[row][col], fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                im = ax.imshow(np.ma.masked_all(t2m.shape[1:]), origin=origin,
                    extent=extent, interpolation="nearest",
                    cmap=temp_cmap if row == 0 else precip_cmap,
                    vmin=temp_low if row == 0 else 0.0,
                    vmax=temp_high if row == 0 else precip_high,
                    aspect="auto", zorder=1)
                # Draw fixed geographic context under sparse extreme maps.
                # Coastline is on top of the raster to stay visible.
                if col in (1, 2, 3):
                    ax.add_collection(LineCollection(
                        coastlines, colors="#697783", linewidths=.35,
                        alpha=.82, zorder=2))
                pair.append(im)
            images.append(pair)
        cax_t = fig.add_axes([.10, .064, .34, .020])
        cax_p = fig.add_axes([.57, .064, .34, .020])
        fig.colorbar(images[0][0], cax=cax_t, orientation="horizontal",
                     extend="both").set_label("T2m (°C)")
        fig.colorbar(images[1][0], cax=cax_p, orientation="horizontal",
                     extend="max").set_label("PR (mm / 6h)")
        n = len(valid_times)

        def update(k):
            valid = valid_times[k]
            month = valid.strftime("%Y%m")
            ds, lookup = reference[month]
            timestamp = int(round(date2num(valid, units=EPOCH_UNITS)))
            j = lookup[timestamp]
            observed_t = np.ma.filled(ds.variables["t2m"][j, :, :], np.nan) - 273.15
            observed_p = normalize_precip(ds.variables["tp06"][j, :, :],
                cfg["plot"]["precipitation_negative_tolerance_m"]) * 1000
            temp = np.asarray(a_t[k]) - 273.15
            precip = np.asarray(a_p[k]) * 1000
            images[0][0].set_data(np.ma.masked_invalid(temp))
            images[1][0].set_data(np.ma.masked_invalid(precip))
            for i, (t_cut, p_cut) in enumerate(zip(q_t, q_p), start=1):
                images[0][i].set_data(np.ma.masked_where(
                    ~np.isfinite(t2m[k]) | (t2m[k] < t_cut), temp))
                images[1][i].set_data(np.ma.masked_where(
                    ~np.isfinite(tp06[k]) | (tp06[k] <= 0) | (tp06[k] < p_cut), precip))
            images[0][4].set_data(np.ma.masked_invalid(observed_t))
            images[1][4].set_data(np.ma.masked_invalid(observed_p))
            fig.suptitle(f"GraphCast vs ERA5 | Init {init:%Y-%m-%d} | Lead {k*hours}h ({k*hours/24:.2f} d) | Valid {valid:%Y-%m-%d %H} UTC",
                         fontsize=12, weight="bold")
            if k % 24 == 0 or k == n-1:
                print(f"frame {k+1}/{n}", flush=True)
            return [im for row in images for im in row]

        ani = animation.FuncAnimation(fig, update, frames=nframes, interval=200, blit=False)
        try:
            ani.save(partial, writer=animation.PillowWriter(fps=int(cfg["plot"]["fps"])),
                     dpi=int(cfg["plot"]["dpi"]))
            os.replace(partial, outfile)
        finally:
            plt.close(fig)
    print(f"WROTE {outfile} ({outfile.stat().st_size / 1024**2:.1f} MiB)", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="fig/config.yaml")
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--max-frames", type=int, help="Short smoke test (writes preview suffix)")
    a = p.parse_args()
    plot_one(config(a.config), a.index, max_frames=a.max_frames)


if __name__ == "__main__":
    main()
