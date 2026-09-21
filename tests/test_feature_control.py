import copy
import unittest

import numpy as np
import pandas as pd

from run_feature_control import feature_estimator
from run_selection_benchmark import evaluate_conditions


class FeatureControlTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(material_features=['SA'], condition_features=['concentration'],
                           categorical_condition_features=['matrix'], ridge={'alpha': 1},
                           svr={}, random_forest={'n_estimators': 3, 'random_state': 1})
        self.data = pd.DataFrame(dict(SA=[1, 2, 1, 2], concentration=[1, 1, 2, 2],
                                     matrix=['water'] * 4, series=['isotherm'] * 4,
                                     material_group=['A', 'B'] * 2,
                                     response_mg_g=[1, 3, 2, 5]))

    def test_material_only_preserves_scoring_conditions_and_input_config(self):
        original = copy.deepcopy(self.config)
        model = feature_estimator('ridge', 'material_only', self.config)
        model.fit(self.data, self.data.response_mg_g)
        pred = model.predict(self.data)
        self.assertEqual(self.config, original)
        np.testing.assert_allclose(pred[:2], pred[2:])
        scores, _ = evaluate_conditions(self.data, pred, {'A', 'B'}, self.config)
        self.assertEqual(len(scores), 2)

    def test_condition_only_ties_candidates_at_each_condition(self):
        for name in ('ridge', 'svr', 'random_forest'):
            model = feature_estimator(name, 'condition_only', self.config)
            model.fit(self.data, self.data.response_mg_g)
            scores, _ = evaluate_conditions(self.data, model.predict(self.data), {'A', 'B'}, self.config)
            self.assertEqual(len(scores), 2)
            for score in scores:
                self.assertAlmostEqual(score['selection_loss'], score['random_selection_loss'])

    def test_unknown_variant_rejected(self):
        with self.assertRaises(ValueError):
            feature_estimator('ridge', 'unknown', self.config)


if __name__ == '__main__':
    unittest.main()
