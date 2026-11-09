"""Pure-Python tests for the 607-date NeuralGCM production schedule."""
from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_neuralgcm_batch import (
    EXPECTED, FORECAST_HOURS, OUTPUT_INTERVAL_HOURS,
    SEED, assigned_dates, initialization_dates, output_path
)


class NeuralGCMBatchTests(unittest.TestCase):
    def test_exact_same_calendar_as_existing_models(self):
        dates = initialization_dates()
        self.assertEqual(len(dates), EXPECTED)
        self.assertEqual(dates[0], date(2015, 10, 2))
        self.assertEqual(dates[-1], date(2025, 4, 29))
        self.assertEqual(len(dates), len(set(dates)))
        for day in dates:
            self.assertIn(day.weekday(), (1, 4))
            self.assertIn(day.month, (1, 2, 3, 4, 10, 11, 12))

    def test_20_shard_coverage_without_overlap(self):
        original = initialization_dates()
        shards = [assigned_dates(i, 20) for i in range(20)]
        combined = [d for shard in shards for d in shard]
        self.assertEqual(set(combined), set(original))
        self.assertEqual(len(combined), len(original))
        self.assertTrue(all(30 <= len(s) <= 31 for s in shards))

    def test_invalid_shards(self):
        for worker, count in [(0, 0), (-1, 20), (20, 20)]:
            with self.subTest(worker=worker, count=count):
                with self.assertRaises(ValueError):
                    assigned_dates(worker, count)

    def test_run_settings_and_names(self):
        self.assertEqual((FORECAST_HOURS, OUTPUT_INTERVAL_HOURS, SEED), (1008, 6, 42))
        p = output_path(Path("/scratch/test/weather-ai/neuralgcm-operational"),
                        date(2024, 10, 1))
        self.assertEqual(p.name, "neuralgcm_20241001_42day_seed42.zarr")


if __name__ == "__main__":
    unittest.main()
