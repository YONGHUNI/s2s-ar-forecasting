#!/usr/bin/env python3
"""Isolated NeuralGCM stochastic GPU smoke pilot, independent of Earth2Studio."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

from neuralgcm_async_io import AsyncNeuralGCMZarrWriter

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
    parser.add_argument("--write-chunk-steps", type=int, default=8,
                        help="Number of 6-hour output steps per GPU/async-I/O chunk")
    parser.add_argument("--max-pending-writes", type=int, default=2,
                        help="Bounded number of Zarr chunks pending in the writer")
    args = parser.parse_args()
    total_start = time.perf_counter()
    timings = {}

    def stage(name: str, started: float) -> None:
        elapsed = time.perf_counter() - started
        timings[name] = round(elapsed, 4)
        print(f"[PROFILE] {name}: {elapsed:.3f} s", flush=True)
    leads = forecast_schedule(args.forecast_hours, args.output_every_hours)
    if args.write_chunk_steps < 1 or args.max_pending_writes < 1:
        parser.error("write-chunk-steps and max-pending-writes must be positive")
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

    started = time.perf_counter()
    import gcsfs
    import jax
    import neuralgcm
    import xarray as xr
    from dinosaur import horizontal_interpolation, spherical_harmonic, xarray_utils

    stage("imports_and_jax_backend_setup", started)
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError(f"CUDA JAX backend required; found {jax.devices()}")
    started = time.perf_counter()
    fs = gcsfs.GCSFileSystem(token="anon")
    with fs.open(CHECKPOINT, "rb") as stream:
        raw = stream.read()
    stage("checkpoint_download", started)
    started = time.perf_counter()
    digest = hashlib.sha256(raw).hexdigest()
    # Only deserialize the official, trusted checkpoint; pickle is not safe for arbitrary files.
    model = neuralgcm.PressureLevelModel.from_checkpoint(pickle.loads(raw))
    stage("checkpoint_deserialization_and_model_load", started)
    print("Checkpoint loaded (SHA256):", digest, flush=True)

    # Follow official quick start. Shift forcing timestamps BEFORE selecting the start time.
    started = time.perf_counter()
    full = xr.open_zarr(ARCO, chunks=None, storage_options={"token": "anon"})
    stage("arco_metadata_open", started)
    variables = list(dict.fromkeys(model.input_variables + model.forcing_variables))
    missing = sorted(set(variables) - set(full.data_vars))
    if missing:
        raise KeyError(f"ERA5 is missing required variables: {missing}")
    started = time.perf_counter()
    selected = (
        full[variables]
        .pipe(xarray_utils.selective_temporal_shift,
              variables=model.forcing_variables, time_shift="24 hours")
        .sel(time=[init])
        .compute()
    )
    stage("era5_initial_data_fetch", started)
    started = time.perf_counter()
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
    stage("conservative_regrid_and_nan_fill", started)
    print("ERA5 initial state regridded and NaNs filled", flush=True)

    started = time.perf_counter()
    inputs = model.inputs_from_xarray(initial.isel(time=0))
    initial_forcings = model.forcings_from_xarray(initial.isel(time=0))
    state = model.encode(inputs, initial_forcings, jax.random.key(args.seed))
    jax.block_until_ready(state)  # JAX dispatch is asynchronous; include actual encode work.
    temporal_forcings = model.forcings_from_xarray(initial.head(time=1))
    stage("encode_and_initial_state_gpu_sync", started)
    # Each unroll advances only to the last emitted lead for this chunk.
    # The CPU writer overlaps one Zarr append with later GPU inference.
    writer = AsyncNeuralGCMZarrWriter(
        output, max_pending=args.max_pending_writes
    )
    variable_names = None
    chunk_profiles = []
    try:
        for offset in range(0, len(leads), args.write_chunk_steps):
            chunk_leads = leads[offset:offset + args.write_chunk_steps]
            chunk_index = len(chunk_profiles) + 1
            started = time.perf_counter()
            state, output_data = model.unroll(
                state, temporal_forcings, steps=len(chunk_leads),
                timedelta=np.timedelta64(args.output_every_hours, "h"),
                start_with_input=False,
            )
            # Synchronization and host transfer are explicit. The worker then
            # writes immutable host arrays while the next chunk is computed.
            jax.block_until_ready((state, output_data))
            gpu_seconds = time.perf_counter() - started
            started = time.perf_counter()
            predictions = model.data_to_xarray(
                jax.device_get(output_data), times=np.asarray(chunk_leads)
            )
            transfer_seconds = time.perf_counter() - started
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
            variable_names = list(predictions.data_vars)
            started = time.perf_counter()
            writer.submit(predictions)
            queue_seconds = time.perf_counter() - started
            chunk_profiles.append({
                "chunk": chunk_index,
                "first_lead_hours": chunk_leads[0],
                "last_lead_hours": chunk_leads[-1],
                "outputs": len(chunk_leads),
                "gpu_unroll_and_sync_seconds": round(gpu_seconds, 4),
                "host_transfer_and_decode_seconds": round(transfer_seconds, 4),
                "async_queue_and_backpressure_seconds": round(queue_seconds, 4),
            })
            print(f"[PROFILE] chunk={chunk_index} GPU={gpu_seconds:.3f}s "
                  f"host/decode={transfer_seconds:.3f}s queue={queue_seconds:.3f}s", flush=True)
            print(
                f"Submitted async Zarr chunk: lead {chunk_leads[0]}-"
                f"{chunk_leads[-1]}h ({len(chunk_leads)} outputs)", flush=True
            )
        started = time.perf_counter()
        published = writer.finish()  # Wait for all writes, validate, then rename.
        stage("async_finish_wait_validate_and_publish", started)
    finally:
        writer.close()

    timings["zarr_worker_write_seconds_sum"] = round(sum(writer.write_seconds), 4)
    timings["zarr_backpressure_wait_seconds"] = round(writer.backpressure_seconds, 4)
    timings["zarr_finish_drain_wait_seconds"] = round(writer.drain_seconds, 4)
    timings["zarr_validation_seconds"] = round(writer.validation_seconds, 4)
    timings["zarr_atomic_publish_seconds"] = round(writer.publication_seconds, 4)
    timings["total_wall_seconds"] = round(time.perf_counter() - total_start, 4)
    print("[PROFILE] wall clock and async I/O summary:", json.dumps(timings, indent=2), flush=True)
    print(json.dumps({
        "output": str(published),
        "lead_hours": leads,
        "variables": variable_names,
        "checkpoint_sha256": digest,
        "io_mode": "async_chunked_zarr",
        "write_chunk_steps": args.write_chunk_steps,
        "max_pending_writes": args.max_pending_writes,
        "profile_seconds": timings,
        "chunk_profiles": chunk_profiles,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
