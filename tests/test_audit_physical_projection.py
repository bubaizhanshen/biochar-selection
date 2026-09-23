import unittest

import numpy as np
import pandas as pd

from audit_physical_projection import nominal_upper_bound, score_conditions, series_balanced


class PhysicalProjectionTest(unittest.TestCase):
    def test_nominal_bound_uses_shared_condition_precision(self):
        frame = pd.DataFrame({
            "C0_mg_L": [10.0, 10.00000000001],
            "dose_g_L": [2.0, 2.00000000001],
        })
        np.testing.assert_array_equal(nominal_upper_bound(frame), [5.0, 5.0])

    def test_out_of_range_scores_can_become_a_candidate_tie(self):
        frame = pd.DataFrame({
            "material_group": ["a", "b", "c"],
            "series": ["one"] * 3,
            "C0_mg_L": [10.0] * 3,
            "dose_g_L": [1.0] * 3,
            "response_mg_g": [9.0, 3.0, 8.0],
        })
        raw = score_conditions(frame, np.array([11.0, 12.0, 13.0]),
                               ["series", "C0_mg_L", "dose_g_L"])
        projected = score_conditions(frame, np.array([10.0, 10.0, 10.0]),
                                     ["series", "C0_mg_L", "dose_g_L"])
        self.assertEqual(len(raw), 1)
        self.assertEqual(int(raw.selected_tie_count.iat[0]), 1)
        self.assertEqual(int(projected.selected_tie_count.iat[0]), 3)
        self.assertAlmostEqual(series_balanced(raw, "selection_loss"), 1.0)
        self.assertAlmostEqual(series_balanced(projected, "selection_loss"), 9 - 20 / 3)


if __name__ == "__main__":
    unittest.main()
