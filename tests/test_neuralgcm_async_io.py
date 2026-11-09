"""Tests for async NeuralGCM Zarr writer. No GPU or network required."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from neuralgcm_async_io import AsyncNeuralGCMZarrWriter


def chunk(leads):
    leads = np.asarray(leads, dtype=np.int64)
    return xr.Dataset(
        {
            "temperature": (("time", "latitude"), np.repeat(leads[:, None], 2, axis=1)),
            "specific_humidity": (("time", "latitude"), np.full((len(leads), 2), 0.01)),
        },
        coords={"time": leads, "latitude": [-10.0, 10.0]},
        attrs={"model": "NeuralGCM"},
    )


class NeuralGCMAsyncIOTests(unittest.TestCase):
    def test_async_append_and_atomic_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "forecast.zarr"
            writer = AsyncNeuralGCMZarrWriter(target, max_pending=2)
            try:
                writer.submit(chunk([6, 12]))
                self.assertFalse(target.exists())
                writer.submit(chunk([18]))
                writer.submit(chunk([24, 30]))
                writer.finish()
            finally:
                writer.close()
            self.assertEqual(len(writer.write_seconds), 3)
            self.assertTrue(all(seconds >= 0 for seconds in writer.write_seconds))
            self.assertGreaterEqual(writer.backpressure_seconds, 0)
            self.assertGreaterEqual(writer.drain_seconds, 0)
            self.assertGreaterEqual(writer.validation_seconds, 0)
            self.assertGreaterEqual(writer.publication_seconds, 0)
            self.assertTrue(target.is_dir())
            self.assertFalse(Path(str(target) + ".partial.zarr").exists())
            with xr.open_zarr(target, consolidated=False, chunks=None) as ds:
                self.assertEqual(ds.sizes["time"], 5)
                self.assertEqual(ds["time"].values.tolist(), [6, 12, 18, 24, 30])
                self.assertEqual(ds["temperature"].isel(latitude=0).values.tolist(), [6, 12, 18, 24, 30])
                self.assertEqual(set(ds.data_vars), {"temperature", "specific_humidity"})

    def test_reject_existing_output_and_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "forecast.zarr"
            target.mkdir()
            with self.assertRaises(FileExistsError):
                AsyncNeuralGCMZarrWriter(target)
            target.rmdir()
            Path(str(target) + ".partial.zarr").mkdir()
            with self.assertRaises(FileExistsError):
                AsyncNeuralGCMZarrWriter(target)

    def test_failed_background_write_never_publishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "forecast.zarr"
            with patch("neuralgcm_async_io.write_one_chunk", side_effect=RuntimeError("write failed")):
                writer = AsyncNeuralGCMZarrWriter(target)
                try:
                    writer.submit(chunk([6]))
                    with self.assertRaisesRegex(RuntimeError, "write failed"):
                        writer.finish()
                finally:
                    writer.close()
            self.assertFalse(target.exists())

    def test_reject_changed_variable_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "forecast.zarr"
            writer = AsyncNeuralGCMZarrWriter(target)
            try:
                writer.submit(chunk([6]))
                with self.assertRaises(ValueError):
                    writer.submit(chunk([12]).drop_vars("temperature"))
            finally:
                writer.close()
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
