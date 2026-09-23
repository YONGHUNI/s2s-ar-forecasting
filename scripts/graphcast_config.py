#!/usr/bin/env python3

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import yaml


WEEKDAY_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


@dataclass(frozen=True)
class GraphCastConfig:
    start_date: date
    end_date: date
    init_weekdays: tuple[int, ...]
    forecast_days: int
    hours_per_step: int
    steps: int
    expected_lead_times: int
    device: str
    variables: tuple[str, ...]
    expected_variable_count: int
    compression_level: int
    zarr_backend: str
    write_mode: str
    async_pool_size: int
    shard_lead_times: int
    local_root: Path
    staging_root: Path
    final_root: Path
    poll_seconds: int
    converter_workers: int


def _expand_path(value: str) -> Path:
    user = os.environ["USER"]
    return Path(value.format(user=user)).expanduser()


def _weekday_indices(names: list[str]) -> tuple[int, ...]:
    indices = []

    for name in names:
        key = str(name).strip().lower()

        if key not in WEEKDAY_INDEX:
            valid = ", ".join(name.title() for name in WEEKDAY_INDEX)
            raise ValueError(
                f"Unknown weekday {name!r}. Expected one of: {valid}"
            )

        indices.append(WEEKDAY_INDEX[key])

    if not indices:
        raise ValueError("experiment.init_weekdays must not be empty")

    return tuple(indices)


def load_graphcast_config(path: str | Path) -> GraphCastConfig:
    path = Path(path)

    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    experiment = raw["experiment"]
    model = raw["model"]
    variable_config = raw["variables"]
    output = raw["output"]
    converter = raw.get("converter", {})
    validation = raw.get("validation", {})

    if model.get("name") != "GraphCastOperational":
        raise ValueError(
            "This workflow currently supports model.name=GraphCastOperational only"
        )

    start_date = date.fromisoformat(str(experiment["start_date"]))
    end_date = date.fromisoformat(str(experiment["end_date"]))

    if end_date < start_date:
        raise ValueError("experiment.end_date must be on or after start_date")

    forecast_days = int(experiment["forecast_days"])
    hours_per_step = int(experiment["hours_per_step"])

    total_hours = forecast_days * 24

    if total_hours % hours_per_step != 0:
        raise ValueError(
            "forecast_days * 24 must be divisible by hours_per_step"
        )

    steps = total_hours // hours_per_step

    surface_variables = [
        str(value) for value in variable_config["surface"]
    ]
    pressure_levels = [
        int(value) for value in variable_config["pressure_levels"]
    ]
    pressure_variables = [
        str(value) for value in variable_config["pressure_variables"]
    ]

    variables = list(surface_variables)

    for level in pressure_levels:
        variables.extend(
            f"{variable}{level}"
            for variable in pressure_variables
        )

    expected_variable_count = int(
        validation.get("expected_variable_count", len(variables))
    )

    if len(variables) != expected_variable_count:
        raise ValueError(
            "Configured variable count does not match validation expectation: "
            f"{len(variables)} != {expected_variable_count}"
        )

    compression_level = int(output["compression_level"])

    if not 0 <= compression_level <= 9:
        raise ValueError("output.compression_level must be between 0 and 9")

    zarr_backend = str(output.get("zarr_backend", "sync")).strip().lower()

    if zarr_backend not in {"sync", "async"}:
        raise ValueError("output.zarr_backend must be 'sync' or 'async'")

    write_mode = str(output.get("write_mode", "local_then_stage")).strip().lower()

    if write_mode not in {"local_then_stage", "direct_staging"}:
        raise ValueError(
            "output.write_mode must be 'local_then_stage' or 'direct_staging'"
        )

    async_pool_size = int(output.get("async_pool_size", 4))

    if async_pool_size < 1:
        raise ValueError("output.async_pool_size must be >= 1")

    shard_lead_times = int(output.get("shard_lead_times", 1))

    if shard_lead_times < 1:
        raise ValueError("output.shard_lead_times must be >= 1")

    if shard_lead_times > steps + 1:
        raise ValueError(
            "output.shard_lead_times cannot exceed the number of lead times"
        )

    converter_workers = int(converter.get("workers", 2))

    if converter_workers < 1:
        raise ValueError("converter.workers must be >= 1")

    return GraphCastConfig(
        start_date=start_date,
        end_date=end_date,
        init_weekdays=_weekday_indices(experiment["init_weekdays"]),
        forecast_days=forecast_days,
        hours_per_step=hours_per_step,
        steps=steps,
        expected_lead_times=steps + 1,
        device=str(model.get("device", "cuda")),
        variables=tuple(variables),
        expected_variable_count=expected_variable_count,
        compression_level=compression_level,
        zarr_backend=zarr_backend,
        write_mode=write_mode,
        async_pool_size=async_pool_size,
        shard_lead_times=shard_lead_times,
        local_root=_expand_path(output["local_root"]),
        staging_root=_expand_path(output["staging_root"]),
        final_root=_expand_path(output["final_root"]),
        poll_seconds=int(converter.get("poll_seconds", 30)),
        converter_workers=converter_workers,
    )


def initialization_dates(config: GraphCastConfig) -> list[date]:
    dates = []
    current = config.start_date

    while current <= config.end_date:
        if current.weekday() in config.init_weekdays:
            dates.append(current)

        current += timedelta(days=1)

    return dates


def output_basename(config: GraphCastConfig, init_date: date) -> str:
    stamp = init_date.strftime("%Y%m%d")

    return (
        f"graphcast_operational_{stamp}_"
        f"{config.forecast_days}day_subset"
    )
