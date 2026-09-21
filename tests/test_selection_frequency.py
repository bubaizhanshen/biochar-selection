import unittest

import numpy as np
import pandas as pd

from run_selection_benchmark import training_fit_rows


class FrequencyTest(unittest.TestCase):
    def test_original_responses_require_frequency_configuration(self):
        with self.assertRaisesRegex(ValueError, 'cell-frequency'):
            training_fit_rows(pd.DataFrame({'response_mg_g': [2.], 'values': ['[1,3]']}),
                              {'training_raw_responses_json_column': 'values'})

    def test_expansion_keeps_means_and_original_input(self):
        d = pd.DataFrame({'response_mg_g': [2., 8.], 'count': [1, 3],
                          'source_table_row_id': [12, 19]})
        expanded = training_fit_rows(d, {'training_cell_frequency_column': 'count'})
        self.assertEqual(expanded.response_mg_g.tolist(), [2., 8., 8., 8.])
        self.assertEqual(expanded.source_table_row_id.tolist(), [12, 19, 19, 19])
        self.assertEqual(len(d), 2)

    def test_bad_counts_rejected(self):
        for count in [0, -1, 1.5, np.nan, np.inf]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                training_fit_rows(pd.DataFrame({'count': [count]}),
                                  {'training_cell_frequency_column': 'count'})

    def test_original_responses_replace_repeated_mean(self):
        d = pd.DataFrame({'response_mg_g': [2.], 'count': [2], 'values': ['[1,3]']})
        cfg = {'training_cell_frequency_column': 'count', 'training_raw_responses_json_column': 'values'}
        self.assertEqual(training_fit_rows(d, cfg).response_mg_g.tolist(), [1., 3.])
        d['values'] = '[1,4]'
        with self.assertRaises(ValueError):
            training_fit_rows(d, cfg)

    def test_unspecified_inner_weighting_rejected(self):
        with self.assertRaises(ValueError):
            training_fit_rows(pd.DataFrame({'count': [2]}), {
                'training_cell_frequency_column': 'count', 'inner_parameter_candidates': {}})
