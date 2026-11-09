"""Bounded background Zarr writer for NeuralGCM output chunks.

One writer thread serializes appends to a single Zarr store while the
producer can continue GPU inference. Publication happens only after
all writes complete and the full time coordinate is validated.
"""
from __future__ import annotations

import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import xarray as xr


def write_one_chunk(dataset: xr.Dataset, path: Path, first: bool) -> None:
    if "time" not in dataset.dims or dataset.sizes["time"] < 1:
        raise ValueError("NeuralGCM output chunk must have nonempty time dimension")
    if first:
        dataset.to_zarr(path, mode="w", consolidated=False)
    else:
        dataset.to_zarr(path, mode="a", append_dim="time", consolidated=False)


class AsyncNeuralGCMZarrWriter:
    """A single serialized background writer, bounded to 2 queued chunks.

    No final output appears until `finish` validates and atomically renames
    the partial store. A failing write is propagated; partial data remains
    available for debugging and is never represented as published output.
    """

    def __init__(self, destination: Path, *, max_pending: int = 2):
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self.destination = Path(destination)
        self.partial = Path(str(destination) + ".partial.zarr")
        if self.destination.exists() or self.partial.exists():
            raise FileExistsError(f"output or partial store already exists: {destination}")
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="neuralgcm-zarr")
        self._pending: list[Future] = []
        self._max_pending = max_pending
        self._submitted = 0
        self._expected_time_count = 0
        self._variables: set[str] | None = None
        self._closed = False

    def submit(self, dataset: xr.Dataset) -> None:
        if self._closed:
            raise RuntimeError("writer is already closed")
        if "time" not in dataset.dims or dataset.sizes["time"] < 1:
            raise ValueError("NeuralGCM output must have a nonempty time axis")
        variables = set(dataset.data_vars)
        if self._variables is None:
            self._variables = variables
        elif variables != self._variables:
            raise ValueError("variable set changed between forecast chunks")
        # Bound memory consumption and detect background write failures.
        if len(self._pending) >= self._max_pending:
            self._pending.pop(0).result()
        self._pending.append(
            self._pool.submit(write_one_chunk, dataset, self.partial, self._submitted == 0)
        )
        self._submitted += 1
        self._expected_time_count += dataset.sizes["time"]

    def finish(self) -> Path:
        if self._closed:
            raise RuntimeError("writer has already closed")
        try:
            for future in self._pending:
                future.result()
            if not self._submitted:
                raise RuntimeError("no Zarr chunks were written")
        finally:
            self._pool.shutdown(wait=True)
            self._closed = True
        with xr.open_zarr(self.partial, consolidated=False, chunks=None) as result:
            if result.sizes.get("time") != self._expected_time_count:
                raise RuntimeError(
                    f"Zarr time count mismatch: expected {self._expected_time_count}, "
                    f"got {result.sizes.get('time')}"
                )
            if set(result.data_vars) != self._variables:
                raise RuntimeError("Zarr variable set does not match submitted chunks")
        if self.destination.exists():
            raise FileExistsError(f"output appeared during write: {self.destination}")
        self.partial.rename(self.destination)
        return self.destination

    def close(self) -> None:
        if not self._closed:
            self._pool.shutdown(wait=True)
            self._closed = True
