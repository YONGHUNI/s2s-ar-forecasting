"""Regression tests for chunk-friendly forecast QC reads.

Run: pixi run -e postprocess --locked python -m unittest discover -s tests -p 'test_qc_sampling.py' -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from qc_sampling import variable_stats


class TrackingVariable:
    """Delegate NetCDF reads while recording the hyperslabs requested."""

    def __init__(self, variable):
        self.variable = variable
        self.dimensions = variable.dimensions
        self.shape = variable.shape
        self.reads = []

    def __getitem__(self, key):
        self.reads.append(key)
        return self.variable[key]


class TestSamplingQC(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "tiny.nc"
        data = np.arange(1 * 9 * 8 * 12, dtype=np.float32).reshape(1, 9, 8, 12) / 300
        data -= 1.0
        data[0, 0, 0, 4] = np.inf
        data[0, 3, 4, 8] = -9999.0
        data[0, 8, 0, 0] = np.nan
        with Dataset(self.path, "w") as ds:
            for dim, size in (("time", 1), ("lead_time", 9), ("lat", 8), ("lon", 12)):
                ds.createDimension(dim, size)
            v = ds.createVariable(
                "q500", "f4", ("time", "lead_time", "lat", "lon"),
                zlib=True, complevel=2, chunksizes=(1, 2, 4, 6),
                fill_value=-9999.0,
            )
            v[:] = data

    def tearDown(self):
        self.tmp.cleanup()

    def test_sampled_contiguous_matches_legacy(self):
        with Dataset(self.path) as ds:
            var = ds.variables["q500"]
            params = dict(mode="sampled", lead_indices=[0, 3, 8],
                          spatial_stride=2, lead_chunk=4, count_negative=True)
            legacy, old_negative, old_finite = variable_stats(
                var, sample_read_mode="legacy", **params)
            improved, new_negative, new_finite = variable_stats(
                var, sample_read_mode="contiguous", **params)
            self.assertEqual((old_negative, old_finite), (new_negative, new_finite))
            for key in ("n_total", "n_finite", "n_missing", "n_inf"):
                self.assertEqual(legacy[key], improved[key], key)
            for key in ("min", "max", "mean", "std", "missing_fraction"):
                self.assertAlmostEqual(legacy[key], improved[key], places=6, msg=key)

    def test_contiguous_reads_do_not_stride_netcdf_spatial_axes(self):
        with Dataset(self.path) as ds:
            spy = TrackingVariable(ds.variables["q500"])
            variable_stats(
                spy, mode="sampled", lead_indices=[0, 3, 8],
                spatial_stride=2, lead_chunk=4, count_negative=True)
            self.assertEqual(len(spy.reads), 3)
            for key in spy.reads:
                self.assertEqual(key[2], slice(None))
                self.assertEqual(key[3], slice(None))
            self.assertEqual([key[1] for key in spy.reads], [0, 3, 8])

    def test_full_mode_counts_negative_from_same_read(self):
        with Dataset(self.path) as ds:
            spy = TrackingVariable(ds.variables["q500"])
            result, negative, finite = variable_stats(
                spy, mode="full", lead_indices=[], spatial_stride=2,
                lead_chunk=4, count_negative=True)
            self.assertEqual(len(spy.reads), 3)
            self.assertEqual(result["n_total"], 1 * 9 * 8 * 12)
            self.assertEqual(result["n_inf"], 1)
            self.assertGreater(result["n_missing"], 0)
            self.assertGreater(negative, 0)
            self.assertGreater(finite, 0)


if __name__ == "__main__":
    unittest.main()
