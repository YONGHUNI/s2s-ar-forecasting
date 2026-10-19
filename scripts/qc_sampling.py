"""Chunk-friendly, single-pass NetCDF sampling for shared forecast QC.

The fast sampled path reads a complete spatial slice for each selected lead
before subsampling in NumPy. This avoids costly strided hyperslab accesses to
compressed HDF5 chunks. The legacy mode preserves the previous QC read path
for comparisons. No forecast data are modified.
"""
from __future__ import annotations

import numpy as np

from qc_graphcast import (
    empty_state,
    finish_state,
    full_variable_stats,
    merge_array,
    sampled_variable_stats,
)


def _negative_counts(data):
    """Count finite and negative unmasked values in the same data already read."""
    values = np.ma.asarray(data).compressed()
    values = values[np.isfinite(values)]
    return int(np.count_nonzero(values < 0)), int(values.size)


def variable_stats(
    var,
    *,
    mode: str,
    lead_indices: list[int],
    spatial_stride: int,
    lead_chunk: int,
    sample_read_mode: str = "contiguous",
    count_negative: bool = False,
):
    """Return (standard QC statistics, negative count, finite count).

    Counts are computed from the *same* sample as the statistics. In full
    mode, one bounded lead-time block is read at a time.
    """
    if mode == "sampled" and sample_read_mode == "legacy":
        # Exact previous data selection, kept for reproducibility comparisons.
        stats = sampled_variable_stats(var, lead_indices, spatial_stride)
        negative = finite = 0
        if count_negative:
            for lead in lead_indices:
                dn, fn = _negative_counts(
                    var[:, lead, ::spatial_stride, ::spatial_stride]
                )
                negative += dn
                finite += fn
        return stats, negative, finite

    if mode not in ("sampled", "full"):
        raise ValueError(f"Unknown QC mode: {mode}")
    if mode == "sampled" and sample_read_mode != "contiguous":
        raise ValueError(f"Unknown sample read mode: {sample_read_mode}")
    if var.dimensions != ("time", "lead_time", "lat", "lon"):
        raise ValueError(f"Unexpected variable dimensions: {var.dimensions}")

    state = empty_state()
    negative = finite = 0

    if mode == "sampled":
        for lead in lead_indices:
            # Full 2-D spatial hyperslab allows HDF5 to decompress chunks
            # sequentially. Sample *after* loading the contiguous spatial
            # slice; data values and sampled grid points remain unchanged.
            plane = var[:, lead, :, :]
            sample = plane[..., ::spatial_stride, ::spatial_stride]
            merge_array(state, sample)
            if count_negative:
                dn, fn = _negative_counts(sample)
                negative += dn
                finite += fn
    else:
        for start in range(0, var.shape[1], lead_chunk):
            data = var[:, start : start + lead_chunk, :, :]
            merge_array(state, data)
            if count_negative:
                dn, fn = _negative_counts(data)
                negative += dn
                finite += fn

    return finish_state(state), negative, finite
