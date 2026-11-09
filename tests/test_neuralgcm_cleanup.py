"""Regression tests for NeuralGCM NetCDF-terminal restart and Zarr cleanup."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_neuralgcm_batch import final_netcdf_verified
from convert_neuralgcm_netcdf import delete_verified_zarr


class TestNeuralGCMCleanup(unittest.TestCase):
    def test_restart_receipt_needs_matching_date_and_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            nc = Path(tmp) / "neuralgcm_20151002_42day_seed42.nc"
            nc.write_bytes(b"example")
            receipt = Path(str(nc) + ".verified.json")
            receipt.write_text(json.dumps({
                "schema": "neuralgcm-netcdf-v1",
                "initialization": "2015-10-02",
                "seed": 42,
                "size_bytes": 7,
            }))
            self.assertTrue(final_netcdf_verified(nc, date(2015, 10, 2)))
            self.assertFalse(final_netcdf_verified(nc, date(2015, 10, 6)))
            nc.write_bytes(b"changed")
            self.assertFalse(final_netcdf_verified(nc, date(2015, 10, 2)))

    def test_reject_wrong_directory_even_with_valid_nc(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            outside = Path(tmp) / "outside"
            outside.mkdir()
            source = outside / "neuralgcm_20151002_42day_seed42.zarr"
            source.mkdir()
            nc = root / "netcdf" / "neuralgcm_20151002_42day_seed42.nc"
            nc.parent.mkdir(parents=True)
            nc.write_bytes(b"sample")
            with patch("convert_neuralgcm_netcdf.valid_nc", return_value=True):
                with self.assertRaises(ValueError):
                    delete_verified_zarr(source, nc, "2015-10-02", root=root)
            self.assertTrue(source.exists())

    def test_dry_run_does_not_remove_source_or_create_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "neuralgcm_20151002_42day_seed42.zarr"
            source.mkdir()
            (source / "chunk").write_bytes(b"data")
            nc = root / "netcdf" / "neuralgcm_20151002_42day_seed42.nc"
            nc.parent.mkdir()
            nc.write_bytes(b"sample")
            with patch("convert_neuralgcm_netcdf.valid_nc", return_value=True):
                self.assertTrue(delete_verified_zarr(source, nc, "2015-10-02",
                                                    root=root, dry_run=True))
            self.assertTrue(source.exists())
            self.assertFalse(Path(str(nc) + ".verified.json").exists())


if __name__ == "__main__":
    unittest.main()
