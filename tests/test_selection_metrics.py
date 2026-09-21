import unittest

import numpy as np

from selection_metrics import selection_metrics


class SelectionMetricTests(unittest.TestCase):
    def test_correct_selection_and_random_gain(self):
        m = selection_metrics([1, 3, 2], [11, 13, 12])
        self.assertEqual(m['selection_loss'], 0)
        self.assertEqual(m['gain_over_random'], 1)
        self.assertEqual(m['pairwise_accuracy'], 1)
        self.assertEqual(m['common_bias_squared'], 100)
        self.assertEqual(m['relative_error_mse'], 0)

    def test_constant_prediction_is_random_not_first_row(self):
        m = selection_metrics([1, 3, 2], [0, 0, 0])
        self.assertEqual(m['selection_loss'], 1)
        self.assertEqual(m['gain_over_random'], 0)
        self.assertEqual(m['pairwise_accuracy'], 0.5)
        self.assertAlmostEqual(m['observed_best_selected_probability'], 1 / 3)

    def test_partial_tie_uses_expected_loss(self):
        m = selection_metrics([1, 3, 2], [5, 5, 0])
        self.assertEqual(m['selection_loss'], 1)
        self.assertEqual(m['selected_tie_count'], 2)

    def test_worse_than_random_is_negative_gain(self):
        self.assertEqual(selection_metrics([1, 3, 2], [3, 1, 2])['gain_over_random'], -1)

    def test_constant_observation_has_no_normalized_loss(self):
        m = selection_metrics([2, 2], [0, 1])
        self.assertEqual(m['selection_loss'], 0)
        self.assertTrue(np.isnan(m['normalized_loss']))
        self.assertTrue(np.isnan(m['pairwise_accuracy']))

    def test_mse_decomposition(self):
        m = selection_metrics([1, 2, 8], [-1, 2, 3])
        self.assertAlmostEqual(m['mse'], m['common_bias_squared'] + m['relative_error_mse'])

    def test_row_order_does_not_change_tie_loss(self):
        a = selection_metrics([1, 3, 2], [4, 4, 1])
        b = selection_metrics([3, 2, 1], [4, 1, 4])
        self.assertEqual(a, b)

    def test_invalid_inputs_fail(self):
        for y, p in [([1], [1]), ([1, 2], [1]), ([1, np.nan], [1, 2])]:
            with self.assertRaises(ValueError):
                selection_metrics(y, p)


if __name__ == '__main__':
    unittest.main()
