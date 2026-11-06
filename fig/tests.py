"""Small validation tests, runnable with the postprocess Pixi environment.

Run: pixi run -e postprocess python -m unittest fig/tests.py -v
"""
import unittest
import numpy as np

from fig.plot_10panel import percentiles_full_field, normalize_precip


class FigureTests(unittest.TestCase):
    def test_quantiles_across_all_times_and_cells(self):
        a = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, np.nan]]])
        q = percentiles_full_field(a)
        np.testing.assert_allclose(q, np.nanpercentile(a, [90, 95, 99]))

    def test_zero_rain_is_included(self):
        a = np.array([0., 0., 0., 2., 3.])
        q = percentiles_full_field(a)
        self.assertAlmostEqual(q[0], 2.6)

    def test_tiny_negative_rain_is_clipped(self):
        x = np.array([-1e-7, 0, 0.002])
        np.testing.assert_allclose(normalize_precip(x, 1e-5), [0, 0, 0.002])

    def test_large_negative_rain_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_precip(np.array([-0.01]), 1e-5)


if __name__ == "__main__":
    unittest.main()
