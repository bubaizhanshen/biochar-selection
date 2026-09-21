import unittest

import numpy as np
import pandas as pd

from run_selection_benchmark import evaluate_conditions


class SelectionBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.config = {'condition_features': ['concentration'], 'categorical_condition_features': ['matrix']}
        self.data = pd.DataFrame({
            'material_group': ['A', 'B', 'C', 'A', 'B'],
            'series': ['pH'] * 5, 'concentration': [1, 1, 1, 2, 2],
            'matrix': ['buffer'] * 5, 'response_mg_g': [1, 4, 2, 3, 5],
        })

    def test_incomplete_grid_is_logged_not_ranked(self):
        scores, coverage = evaluate_conditions(self.data, np.ones(5), {'A', 'B', 'C'}, self.config)
        self.assertEqual(len(scores), 1)
        self.assertEqual([r['status'] for r in coverage], ['complete', 'incomplete_candidate_grid'])
        self.assertEqual(scores[0]['gain_over_random'], 0)

    def test_different_matrices_never_match(self):
        self.data.loc[2, 'matrix'] = 'lake'
        scores, coverage = evaluate_conditions(self.data, np.ones(5), {'A', 'B', 'C'}, self.config)
        self.assertFalse(scores)
        self.assertEqual(len(coverage), 3)

    def test_different_series_never_match(self):
        self.data.loc[2, 'series'] = 'isotherm'
        scores, _ = evaluate_conditions(self.data, np.ones(5), {'A', 'B', 'C'}, self.config)
        self.assertFalse(scores)

    def test_duplicate_cell_requires_explicit_policy(self):
        data = pd.concat([self.data, self.data.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(RuntimeError, 'aggregation policy'):
            evaluate_conditions(data, np.ones(6), {'A', 'B', 'C'}, self.config)


if __name__ == '__main__':
    unittest.main()
