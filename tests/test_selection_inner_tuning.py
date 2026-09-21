import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from run_selection_benchmark import select_parameters


class InnerTuningTest(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'source_study_id': ['a', 'a', 'a', 'b'],
            'source_table_row_id': [1, 2, 3, 4],
            'response_mg_g': [0., 0., 0., 10.],
        })
        self.config = {'ridge': {'alpha': 1.},
                       'inner_parameter_candidates': {'ridge': [{'alpha': 0.}, {'alpha': 10.}]}}

    def test_source_isolation_equal_source_weight_and_tie_order(self):
        fitted = []

        class Constant:
            def __init__(self, value):
                self.value = value

            def fit(self, x, y):
                self.sources = set(x.source_study_id)
                fitted.append(set(x.source_table_row_id))

            def predict(self, x):
                assert not self.sources.intersection(x.source_study_id)
                return np.full(len(x), self.value)

        with patch('run_selection_benchmark.estimator',
                   side_effect=lambda name, full, cfg: Constant(cfg['ridge']['alpha'])):
            cfg, trials, status = select_parameters(self.data, 'ridge', True, self.config)
        self.assertEqual(cfg['ridge']['alpha'], 0.)
        self.assertEqual(self.config['ridge']['alpha'], 1.)
        self.assertEqual(status, 'leave_one_training_source_out')
        self.assertEqual(len(trials), 4)
        self.assertTrue(all(t['source_balanced_inner_mae'] == 5. for t in trials))
        self.assertEqual(fitted, [{4}, {1, 2, 3}, {4}, {1, 2, 3}])

    def test_single_source_uses_fixed_parameters(self):
        with patch('run_selection_benchmark.estimator') as model:
            cfg, trials, status = select_parameters(self.data.iloc[:3], 'ridge', True, self.config)
        model.assert_not_called()
        self.assertEqual(cfg['ridge']['alpha'], 1.)
        self.assertFalse(trials)
        self.assertEqual(status, 'fixed_insufficient_training_sources')


if __name__ == '__main__':
    unittest.main()
