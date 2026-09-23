#!/usr/bin/env python3
"""Reusable Zarr-to-NetCDF conversion utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import xarray as xr


@dataclass(frozen=True)
class ConversionResult:
    source: Path
    destination: Path
    encoding_seconds: float
    output_size_bytes: int


def build_netcdf_encoding(
    data_vars,
    compression_level: int = 0,
) -> dict[str, dict[str, object]]:
    """Build per-variable NetCDF4 encoding settings."""

    if not 0 <= compression_level <= 9:
        raise ValueError("compression_level must be between 0 and 9")

    if compression_level == 0:
        return {
            str(name): {"zlib": False}
            for name in data_vars
        }

    return {
        str(name): {
            "zlib": True,
            "complevel": compression_level,
            "shuffle": True,
        }
        for name in data_vars
    }


def validate_netcdf(
    path: str | Path,
    *,
    expected_sizes: Mapping[str, int] | None = None,
    expected_variable_count: int | None = None,
) -> None:
    """Validate a NetCDF file without loading its full data payload."""

    path = Path(path)

    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Invalid NetCDF: {path}")

    with xr.open_dataset(path, engine="netcdf4") as ds:
        if expected_sizes is not None:
            for dimension, expected in expected_sizes.items():
                actual = ds.sizes.get(dimension)
                if actual != expected:
                    raise RuntimeError(
                        f"Unexpected {dimension} dimension: "
                        f"expected {expected}, got {actual}"
                    )

        if (
            expected_variable_count is not None
            and len(ds.data_vars) != expected_variable_count
        ):
            raise RuntimeError(
                "Unexpected variable count: "
                f"expected {expected_variable_count}, "
                f"got {len(ds.data_vars)}"
            )


def _write_netcdf_variable_stream(
    ds: xr.Dataset,
    partial: Path,
    *,
    compression_level: int,
) -> None:
    """Write a dataset one data variable at a time without Dask.

    Opening Zarr with chunks=None keeps xarray on its native lazy backend
    instead of constructing a Dask graph. Writing the complete Dataset in one
    call would cause xarray to materialize too much data at once for the
    GraphCast workload. Instead, write global metadata/coordinates first and
    then append one data variable per call. This bounds memory to roughly one
    variable while avoiding the high Dask overhead observed for the native
    GraphCast Zarr chunk layout.
    """

    encoding = build_netcdf_encoding(
        ds.data_vars,
        compression_level=compression_level,
    )

    # Coordinates are small relative to the GraphCast payload, and letting
    # xarray write them preserves datetime/timedelta serialization metadata.
    coordinate_ds = xr.Dataset(
        coords={name: ds.coords[name] for name in ds.coords},
        attrs=dict(ds.attrs),
    )
    coordinate_ds.to_netcdf(
        partial,
        mode="w",
        engine="netcdf4",
    )

    # Append only the payload variable on each pass. Using .variable keeps
    # its dimensions/attributes but does not repeatedly attach all coordinates
    # to every temporary Dataset.
    for name in ds.data_vars:
        variable_ds = xr.Dataset({str(name): ds[name].variable})
        variable_ds.to_netcdf(
            partial,
            mode="a",
            engine="netcdf4",
            encoding={str(name): encoding[str(name)]},
        )


def convert_zarr_to_netcdf(
    source: str | Path,
    destination: str | Path,
    *,
    compression_level: int = 0,
    consolidated: bool = False,
    expected_sizes: Mapping[str, int] | None = None,
    expected_variable_count: int | None = None,
) -> ConversionResult:
    """Convert one Zarr store to NetCDF and publish it atomically.

    The Zarr store is opened without Dask and the NetCDF payload is appended
    one data variable at a time. This avoids both failure modes observed during
    GraphCast conversion benchmarking:

    * chunks={} created a large Dask task graph and was much slower.
    * a single no-Dask Dataset.to_netcdf() attempted to materialize the
      full dataset and exceeded the worker memory limit.

    Variable-by-variable writing keeps memory bounded while retaining the fast
    no-Dask Zarr read path.

    The NetCDF is first written to a .partial path on the same filesystem,
    validated, and then atomically renamed to the destination. The source
    Zarr is never removed by this function.
    """

    source = Path(source)
    destination = Path(destination)
    partial = Path(str(destination) + ".partial")

    if not source.is_dir():
        raise FileNotFoundError(f"Zarr source does not exist: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    if partial.exists():
        partial.unlink()

    # chunks=None is intentional. It bypasses Dask while preserving lazy reads
    # through xarray's Zarr backend.
    ds = xr.open_zarr(
        source,
        chunks=None,
        consolidated=consolidated,
    )

    try:
        start = time.perf_counter()

        _write_netcdf_variable_stream(
            ds,
            partial,
            compression_level=compression_level,
        )

        encoding_seconds = time.perf_counter() - start
    finally:
        ds.close()

    validate_netcdf(
        partial,
        expected_sizes=expected_sizes,
        expected_variable_count=expected_variable_count,
    )

    partial.replace(destination)

    validate_netcdf(
        destination,
        expected_sizes=expected_sizes,
        expected_variable_count=expected_variable_count,
    )

    return ConversionResult(
        source=source,
        destination=destination,
        encoding_seconds=encoding_seconds,
        output_size_bytes=destination.stat().st_size,
    )
