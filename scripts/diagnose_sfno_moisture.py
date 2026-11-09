from pathlib import Path
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset

source = Path(
    "/scratch/ys01849/weather-ai/sfno-pilot/"
    "sfno_20241001_42day_subset.nc"
)
out = Path("results/qc/sfno_moisture")
out.mkdir(parents=True, exist_ok=True)

variables = ["tcwv", "q500", "q700", "q850", "q925", "q1000"]
map_variables = {"tcwv", "q850", "q925"}
map_leads = {4, 28}  # Day 1 and Day 7

rows = []

with Dataset(source) as ds:
    lat = np.asarray(ds.variables["lat"][:])
    lon = np.asarray(ds.variables["lon"][:])
    lon180 = (lon + 180) % 360 - 180
    lon_order = np.argsort(lon180)

    for name in variables:
        var = ds.variables[name]

        # Read 0-7 days in one contiguous block.
        block = np.ma.asarray(var[0, 0:29, :, :])

        first_negative = None

        for lead in range(29):
            plane = np.ma.asarray(block[lead])
            data = np.ma.getdata(plane)

            valid = (~np.ma.getmaskarray(plane)) & np.isfinite(data)
            negative = valid & (data < 0)

            n_valid = int(valid.sum())
            n_negative = int(negative.sum())
            fraction = 100 * n_negative / n_valid if n_valid else np.nan
            minimum = float(data[valid].min()) if n_valid else np.nan

            rows.append({
                "variable": name,
                "lead_hours": lead * 6,
                "lead_days": lead / 4,
                "negative_count": n_negative,
                "negative_pct": fraction,
                "minimum": minimum,
            })

            if n_negative and first_negative is None:
                first_negative = lead

            if name in map_variables and lead in map_leads:
                magnitude = np.where(negative, -data, np.nan)
                magnitude = magnitude[:, lon_order]
                plot_data = np.ma.masked_invalid(magnitude)

                fig, ax = plt.subplots(figsize=(12, 5))
                cmap = plt.colormaps["magma"].copy()
                cmap.set_bad("white")

                im = ax.imshow(
                    plot_data,
                    extent=[
                        lon180.min(), lon180.max(),
                        lat.min(), lat.max()
                    ],
                    origin="upper" if lat[0] > lat[-1] else "lower",
                    aspect="auto",
                    cmap=cmap,
                )
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                ax.set_title(
                    f"SFNO {name} negative magnitude "
                    f"| Day {lead / 4:g} | {fraction:.2f}% negative"
                )
                ax.grid(alpha=0.2)
                fig.colorbar(im, ax=ax, label="Magnitude of negative value")
                fig.tight_layout()
                fig.savefig(
                    out / f"{name}_day{lead // 4}_map.png",
                    dpi=160
                )
                plt.close(fig)

        print(
            f"{name}: first negative = "
            + (
                f"{first_negative * 6} h (Day {first_negative / 4:g})"
                if first_negative is not None else "none"
            ),
            flush=True,
        )
        del block

csv_path = out / "negative_moisture_0to7d_6h.csv"

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

fig, ax = plt.subplots(figsize=(11, 5))

for name in variables:
    selected = [r for r in rows if r["variable"] == name]
    ax.plot(
        [r["lead_days"] for r in selected],
        [r["negative_pct"] for r in selected],
        label=name
    )

ax.set(
    xlabel="Forecast lead time (days)",
    ylabel="Negative grid cells (%)",
    title="SFNO negative moisture values: first 7 days",
    xlim=(0, 7)
)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(out / "negative_fraction_0to7d.png", dpi=180)

print(f"Results: {out.resolve()}")
