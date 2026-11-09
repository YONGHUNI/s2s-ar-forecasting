#!/usr/bin/env python3
"""Isolated NeuralGCM stochastic GPU smoke pilot, independent of Earth2Studio."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np

CHECKPOINT = "gs://neuralgcm/models/v1/stochastic_1_4_deg.pkl"
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"


def forecast_schedule(hours: int, interval: int = 6) -> list[int]:
    if type(hours) is not int or type(interval) is not int or hours < 1 or interval < 1 or hours % interval:
        raise ValueError("forecast hours must be a positive integer multiple of interval")
    return list(range(interval, hours + 1, interval))


def validate_output_path(path: Path, username: str) -> Path:
    """Do not overwrite any pre-existing forecast output."""
    root = Path(f"/scratch/{username}/weather-ai/neuralgcm-pilot").resolve()
    resolved = path.expanduser().resolve()
    if root not in resolved.parents:
        raise ValueError(f"output must be a child of {root}")
    if resolved.exists():
        raise FileExistsError(f"output already exists: {resolved}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialization", default="2024-10-01T00:00:00")
    parser.add_argument("--forecast-hours", type=int, default=6)
    parser.add_argument("--output-every-hours", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    leads = forecast_schedule(args.forecast_hours, args.output_every_hours)
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    user = os.environ.get("USER")
    if not user:
        parser.error("USER not set")
    init = np.datetime64(args.initialization, "h")
    if init != np.datetime64(args.initialization):
        parser.error("initialization must be exactly on an hour")
    name = f"neuralgcm_{str(init).replace(':', '').replace('-', '')}_seed{args.seed}_{args.forecast_hours}h.zarr"
    output = validate_output_path(
        args.output or Path(f"/scratch/{user}/weather-ai/neuralgcm-pilot") / name, user
    )
    print(f"Initial time {init} UTC; lead hours {leads}; output {output}", flush=True)

    import gcsfs
    import jax
    import neuralgcm
    import xarray as xr
    from dinosaur import horizontal_interpolation, spherical_harmonic, xarray_utils

    if not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError(f"CUDA JAX backend required; found {jax.devices()}")
    fs = gcsfs.GCSFileSystem(token="anon")
    with fs.open(CHECKPOINT, "rb") as stream:
        raw = stream.read()
    digest = hashlib.sha256(raw).hexdigest()
    # Only deserialize the official, trusted checkpoint; pickle is not safe for arbitrary files.
    model = neuralgcm.PressureLevelModel.from_checkpoint(pickle.loads(raw))
    print("Checkpoint loaded (SHA256):", digest, flush=True)

    # Follow official quick start. Shift forcing timestamps BEFORE selecting the start time.
    full = xr.open_zarr(ARCO, chunks=None, storage_options={"token": "anon"})
    variables = list(dict.fromkeys(model.input_variables + model.forcing_variables))
    missing = sorted(set(variables) - set(full.data_vars))
    if missing:
        raise KeyError(f"ERA5 is missing required variables: {missing}")
    selected = (
        full[variables]
        .pipe(xarray_utils.selective_temporal_shift,
              variables=model.forcing_variables, time_shift="24 hours")
        .sel(time=[init])
        .compute()
    )
    grid = spherical_harmonic.Grid(
        latitude_nodes=full.sizes["latitude"],
        longitude_nodes=full.sizes["longitude"],
        latitude_spacing=xarray_utils.infer_latitude_spacing(full.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(full.longitude),
    )
    regridder = horizontal_interpolation.ConservativeRegridder(
        grid, model.data_coords.horizontal, skipna=True
    )
    initial = xarray_utils.fill_nan_with_nearest(
        xarray_utils.regrid(selected, regridder)
    )
    print("ERA5 initial state regridded and NaNs filled", flush=True)

    inputs = model.inputs_from_xarray(initial.isel(time=0))
    initial_forcings = model.forcings_from_xarray(initial.isel(time=0))
    state = model.encode(inputs, initial_forcings, jax.random.key(args.seed))
    temporal_forcings = model.forcings_from_xarray(initial.head(time=1))
    # Unlike start_with_input=True, this advances exactly to the final saved lead.
    _, output_data = model.unroll(
        state, temporal_forcings, steps=len(leads),
        timedelta=np.timedelta64(args.output_every_hours, "h"),
        start_with_input=False,
    )
    predictions = model.data_to_xarray(jax.device_get(output_data), times=np.array(leads))
    predictions.attrs.update({
        "model": "NeuralGCM v1 stochastic 1.4deg",
        "initialization_utc": str(init),
        "checkpoint_uri": CHECKPOINT,
        "checkpoint_sha256": digest,
        "seed": args.seed,
        "forcing_policy": "initial_SST_and_sea_ice_persistence",
        "input_dataset": ARCO,
        "lead_time_units": "hours",
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"output appeared during inference: {output}")
    predictions.to_zarr(output, mode="w")
    print(json.dumps({"output": str(output), "lead_hours": leads,
                      "variables": list(predictions.data_vars),
                      "sha256": digest}, indent=2), flush=True)


if __name__ == "__main__":
    main()
