"""Pure NeuralGCM pilot schedule and isolation checks (no GPU/network)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_neuralgcm_pilot import forecast_schedule, validate_output_path


class NeuralGCMSmokeTests(unittest.TestCase):
    def test_smoke_forecast(self):
        self.assertEqual(forecast_schedule(6), [6])

    def test_42_day_forecast(self):
        leads = forecast_schedule(42 * 24)
        self.assertEqual(len(leads), 168)
        self.assertEqual(leads[0], 6)
        self.assertEqual(leads[-1], 1008)
        self.assertEqual(len([0] + leads), 169)

    def test_invalid_schedules(self):
        for hours in (-1, 0, 7, 1.5):
            with self.subTest(hours=hours), self.assertRaises(ValueError):
                forecast_schedule(hours)

    def test_existing_model_paths_rejected(self):
        with self.assertRaises(ValueError):
            validate_output_path(
                Path("/scratch/ys01849/weather-ai/sfno-operational/test.zarr"),
                "ys01849",
            )

    def test_neuralgcm_path_accepted(self):
        path = Path("/scratch/ys01849/weather-ai/neuralgcm-pilot/smoke.zarr")
        self.assertEqual(validate_output_path(path, "ys01849"), path)


if __name__ == "__main__":
    unittest.main()
