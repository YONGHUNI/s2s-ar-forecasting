"""Regression checks for existing GraphCast/Pangu configuration contracts.

These tests do not run inference, touch forecast outputs, or require a GPU.
Run from repository root:
    pixi run -e graphcast python -m unittest discover -s tests -v
"""
from __future__ import annotations

import copy
import os
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from forecast_config import initialization_dates, load_forecast_config, output_basename


class ForecastConfigurationRegressionTests(unittest.TestCase):
    def load(self, name: str):
        return load_forecast_config(ROOT / "config" / name)

    def test_operational_calendar_and_steps(self):
        for filename in (
            "graphcast_operational.yaml",
            "graphcast_operational_deterministic.yaml",
            "pangu_operational.yaml",
        ):
            with self.subTest(filename=filename):
                cfg = self.load(filename)
                dates = initialization_dates(cfg)
                self.assertEqual(len(dates), 607)
                self.assertEqual(cfg.forecast_days, 42)
                self.assertEqual(cfg.hours_per_step, 6)
                self.assertEqual(cfg.steps, 168)
                self.assertEqual(cfg.expected_lead_times, 169)
                self.assertTrue(all(d.weekday() in (1, 4) for d in dates))
                self.assertTrue(all(d.month in (10, 11, 12, 1, 2, 3, 4) for d in dates))
                self.assertEqual(dates, sorted(set(dates)))

    def test_existing_model_variable_contracts(self):
        for filename, model, count in (
            ("graphcast_operational.yaml", "GraphCastOperational", 70),
            ("graphcast_operational_deterministic.yaml", "GraphCastOperational", 70),
            ("pangu_operational.yaml", "Pangu6", 69),
        ):
            with self.subTest(filename=filename):
                cfg = self.load(filename)
                self.assertEqual(cfg.model_name, model)
                self.assertEqual(cfg.expected_variable_count, count)
                self.assertEqual(len(cfg.variables), count)
                self.assertEqual(len(set(cfg.variables)), count)
                self.assertIn("q850", cfg.variables)
                self.assertIn("u850", cfg.variables)
                self.assertIn("v850", cfg.variables)
                self.assertEqual("tp06" in cfg.variables, model == "GraphCastOperational")

    def test_output_paths_and_filenames_are_separate(self):
        cases = [
            self.load("graphcast_operational.yaml"),
            self.load("graphcast_operational_deterministic.yaml"),
            self.load("pangu_operational.yaml"),
            self.load("pangu_pilot.yaml"),
        ]
        self.assertEqual(len({str(c.final_root) for c in cases}), len(cases))
        self.assertEqual(len({str(c.staging_root) for c in cases}), len(cases))
        for cfg in cases:
            self.assertNotEqual(cfg.staging_root, cfg.final_root)
        stamp = date(2024, 10, 1)
        self.assertEqual(
            output_basename(cases[0], stamp),
            "graphcast_operational_20241001_42day_subset",
        )
        self.assertEqual(
            output_basename(cases[2], stamp),
            "pangu6_20241001_42day_subset",
        )

    def test_pangu_pilot_is_one_initialization(self):
        cfg = self.load("pangu_pilot.yaml")
        self.assertEqual(initialization_dates(cfg), [date(2024, 10, 1)])
        self.assertEqual(cfg.expected_variable_count, 69)

    def test_invalid_model_and_calendar_are_rejected(self):
        source = ROOT / "config" / "graphcast_operational.yaml"
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        mutations = (
            (("model", "name"), "NeuralGCM"),
            (("validation", "expected_initializations"), 606),
            (("validation", "expected_variable_count"), 69),
            (("experiment", "init_weekdays"), ["NotAWeekday"]),
            (("output", "zarr_backend"), "unknown"),
            (("output", "compression_level"), 10),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.yaml"
            for keys, replacement in mutations:
                with self.subTest(keys=keys):
                    changed = copy.deepcopy(raw)
                    changed[keys[0]][keys[1]] = replacement
                    path.write_text(yaml.safe_dump(changed), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_forecast_config(path)

    def test_username_expansion_does_not_change_template(self):
        with patch.dict(os.environ, {"USER": "regression_test_user"}):
            cfg = self.load("pangu_pilot.yaml")
        self.assertIn("regression_test_user", str(cfg.final_root))
        self.assertNotIn("{user}", str(cfg.final_root))


if __name__ == "__main__":
    unittest.main()
