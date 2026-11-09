"""Pure scheduling regression tests for NeuralGCM 169-lead reruns."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_neuralgcm_batch import expected_leads, output_directory


class TestNeuralGCMLeadZero(unittest.TestCase):
    def test_new_dataset_has_169_leads(self):
        with patch.dict(os.environ, {"NGCM_OUTPUT_DIR": "neuralgcm-operational-lead0"}):
            self.assertEqual(expected_leads(), list(range(0, 1009, 6)))
            self.assertEqual(len(expected_leads()), 169)

    def test_existing_dataset_stays_168_leads(self):
        with patch.dict(os.environ, {"NGCM_OUTPUT_DIR": "neuralgcm-operational"}):
            self.assertEqual(expected_leads(), list(range(6, 1009, 6)))
            self.assertEqual(len(expected_leads()), 168)

    def test_reject_unrecognized_output_root(self):
        with patch.dict(os.environ, {"NGCM_OUTPUT_DIR": "../other"}):
            with self.assertRaises(ValueError):
                output_directory()

    def test_all_chunk_time_labels_contiguous(self):
        leads = list(range(0, 1009, 6))
        chunks = [leads[i:i + 8] for i in range(0, len(leads), 8)]
        # The model's start_with_input=True unroll decodes the input time
        # and returns a state advanced by len(chunk) * 6 hours.
        current_model_time = 0
        for chunk in chunks:
            self.assertEqual(chunk[0], current_model_time)
            self.assertEqual(chunk, list(range(
                current_model_time, current_model_time + len(chunk) * 6, 6
            )))
            current_model_time += len(chunk) * 6
        self.assertEqual(chunks[-1], [1008])
        self.assertEqual(current_model_time, 1014)


if __name__ == "__main__":
    unittest.main()
