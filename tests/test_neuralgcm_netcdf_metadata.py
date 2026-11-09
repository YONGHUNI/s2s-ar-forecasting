"""NetCDF attribute sanitization regression tests for NeuralGCM output."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from convert_neuralgcm_netcdf import netcdf_safe_attrs


class NeuralGCMNetCDFMetadataTests(unittest.TestCase):
    def test_neuralgcm_null_and_structured_attrs(self):
        original = {
            "basis_as_jax_arrays": None,
            "seed": np.int64(42),
            "has_sst": True,
            "description": "stochastic forecast",
            "nested": {"x": None, "y": [1, 2]},
            "pressure_levels": np.array([100, 200, 300]),
        }
        attrs = netcdf_safe_attrs(original)
        self.assertNotIn("basis_as_jax_arrays", attrs)
        self.assertEqual(attrs["seed"], 42)
        self.assertEqual(attrs["has_sst"], 1)
        self.assertEqual(attrs["nested"], '{"x": null, "y": [1, 2]}')
        self.assertEqual(attrs["pressure_levels"].tolist(), [100, 200, 300])
        self.assertIsNone(original["basis_as_jax_arrays"])

    def test_netcdf4_accepts_sanitized_dataset_and_coordinate_attrs(self):
        import xarray as xr
        source = xr.Dataset(
            {"temperature": ("time", np.array([280., 281.], dtype="float32"))},
            coords={"time": [6, 12]},
            attrs={"basis_as_jax_arrays": None, "seed": 42},
        )
        source.time.attrs["unexpected"] = None
        source.temperature.attrs["flag"] = False
        prepared = source.copy(deep=False)
        prepared.attrs = netcdf_safe_attrs(source.attrs)
        for key in prepared.coords:
            prepared.coords[key].attrs = netcdf_safe_attrs(source.coords[key].attrs)
        for key in prepared.data_vars:
            prepared[key].attrs = netcdf_safe_attrs(source[key].attrs)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sample.nc"
            prepared.to_netcdf(p, engine="netcdf4")
            with xr.open_dataset(p, engine="netcdf4") as ds:
                self.assertEqual(ds.attrs["seed"], 42)
                self.assertNotIn("basis_as_jax_arrays", ds.attrs)
                np.testing.assert_array_equal(ds.temperature.values, [280., 281.])


if __name__ == "__main__":
    unittest.main()
