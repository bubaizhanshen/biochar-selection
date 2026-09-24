"""Checks for sensitivity-only operations; no scientific output is selected here."""
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
import numpy as np
import pandas as pd
from run_numerical_sensitivity import score, training_diagnostic, candidate_probabilities, support
from summarize_numerical_sensitivity import input_filenames


class SensitivityTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(material_features=['SA'], condition_features=['C0'], categorical_condition_features=['matrix'])
        self.data = pd.DataFrame(dict(source_study_id=['a', 'a', 'b', 'b'],
            material_group=['a1', 'a2', 'b1', 'b2'], series=['s1', 's1', 's2', 's2'],
            C0=[1., 1., 1., 1.], matrix=['water']*4, SA=[1., 2., 3., 4.],
            response_mg_g=[2., 4., 6., 8.], source_table_row_id=[1, 2, 3, 4]))

    def test_training_groups_remain_separate(self):
        got = training_diagnostic(self.data, [4., 2., 6., 8.], self.config)
        self.assertEqual(got['loss'], 1.)
        self.assertEqual(got['mae'], 1.)

    def test_candidate_tie_probability(self):
        probs = candidate_probabilities(self.data.iloc[:2], [3., 3.], self.config)
        self.assertEqual([r['probability'] for r in probs], [.5, .5])
        self.assertEqual(score(self.data.iloc[:2], [3., 3.], self.config)[0]['loss'], 1.)

    def test_rank_and_unknown_category(self):
        test = self.data.iloc[2:].copy()
        test['matrix'] = 'unknown'
        got = support(self.data.iloc[:2], test, self.config)
        self.assertEqual(got['material_rank'], 1)
        self.assertEqual(got['unknown_matrix_cells'], 2)

    def test_identical_material_vectors_have_zero_rank(self):
        train = self.data.iloc[:2].copy()
        train['SA'] = 1e12
        self.assertEqual(support(train, self.data.iloc[2:], self.config)['material_rank'], 0)

    def test_incomplete_condition_is_not_given_a_probability(self):
        data = self.data.iloc[:2].copy()
        extra = data.iloc[[0]].copy()
        extra['C0'] = 2.
        extra['source_table_row_id'] = 5
        data = pd.concat([data, extra], ignore_index=True)
        probs = candidate_probabilities(data, [2., 4., 100.], self.config)
        self.assertEqual({r['row_id'] for r in probs}, {1, 2})

    def test_rule_has_no_response_mae(self):
        from run_strategy_selection import evaluate
        _, metrics, _, scores = evaluate('surface_area', self.data.iloc[2:],
                                         self.data.iloc[:2], self.config)
        self.assertTrue(np.isnan(scores['mae']))
        self.assertTrue(all(np.isnan(m['mae']) for m in metrics))
        self.assertEqual(scores['loss'], 0.)

    def test_input_filenames_accept_new_and_legacy_contracts(self):
        self.assertEqual(input_filenames({'input_files': ['data.csv', 'run.py']}),
                         ['data.csv', 'run.py'])
        self.assertEqual(input_filenames({'inputs': {'run.py': 'old-value', 'data.csv': 'old-value'}}),
                         ['data.csv', 'run.py'])


if __name__ == '__main__':
    unittest.main()
