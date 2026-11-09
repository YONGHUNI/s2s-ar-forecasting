"""Unit tests for the NeuralGCM pilot preflight; no GPU/dependencies required."""
import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from neuralgcm_preflight import validate_config


class NeuralGCMPilotPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "config/neuralgcm_pilot.json").read_text())

    def test_expected_lead_times_and_seeds(self):
        summary = validate_config(self.config)
        self.assertEqual(summary["lead_time_count"], 169)
        self.assertEqual(summary["ensemble_seeds"], [42])
        self.assertFalse(summary["inference_executed"])

    def test_fails_if_output_path_targets_existing_model(self):
        data = copy.deepcopy(self.config)
        data["output_root_template"] = "/scratch/{user}/weather-ai/sfno-operational"
        with self.assertRaises(ValueError):
            validate_config(data)

    def test_fails_if_forcing_policy_is_missing_or_changed(self):
        data = copy.deepcopy(self.config)
        data["forcing_policy"] = "future_era5_truth"
        with self.assertRaises(ValueError):
            validate_config(data)

    def test_fails_if_duplicate_seeds(self):
        data = copy.deepcopy(self.config)
        data["ensemble_seeds"] = [42, 42]
        with self.assertRaises(ValueError):
            validate_config(data)


if __name__ == "__main__":
    unittest.main()
