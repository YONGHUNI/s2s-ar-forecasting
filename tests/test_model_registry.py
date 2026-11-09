"""Guard against accidentally executing GraphCast for a new model name."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from model_registry import get_model_type


class ModelRegistryTests(unittest.TestCase):
    def test_dispatches_known_models(self):
        sentinel_graphcast = object()
        sentinel_pangu = object()
        fake_module = type("FakeModels", (), {
            "GraphCastOperational": sentinel_graphcast,
            "Pangu6": sentinel_pangu,
        })()
        with patch("model_registry.import_module", return_value=fake_module) as importer:
            self.assertIs(get_model_type("GraphCastOperational"), sentinel_graphcast)
            self.assertIs(get_model_type("Pangu6"), sentinel_pangu)
        importer.assert_called_with("earth2studio.models.px")

    def test_unknown_model_fails_before_importing_earth2studio(self):
        with patch("model_registry.import_module") as importer:
            for name in ("NeuralGCM", "UnknownModel", ""):
                with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, "Unsupported deterministic model"
                ):
                    get_model_type(name)
            importer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
