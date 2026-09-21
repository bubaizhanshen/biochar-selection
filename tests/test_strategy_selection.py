import unittest
import pandas as pd
from run_strategy_selection import choose, MODELS, validate_inputs


class StrategySelectionTest(unittest.TestCase):
    def inputs(self):
        data = pd.DataFrame([dict(SA=1., T=25., response_mg_g=2., matrix='water',
            source_study_id='s', material_group='m', source_table_row_id='r',
            series='a', pollutant='IBU')])
        manifest = pd.DataFrame([{'panel_id':'p'}])
        config = dict(material_features=['SA'], condition_features=['T'],
                      categorical_condition_features=['matrix'])
        return data, manifest, config

    def test_complete_inputs_pass(self):
        validate_inputs(*self.inputs())

    def test_empty_manifest_is_rejected(self):
        data, manifest, config = self.inputs()
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            validate_inputs(data, manifest.iloc[:0], config)

    def test_missing_category_is_rejected(self):
        data, manifest, config = self.inputs()
        data.loc[0,'matrix'] = None
        with self.assertRaisesRegex(ValueError, 'Missing input values'):
            validate_inputs(data, manifest, config)

    def test_nonfinite_predictor_is_rejected(self):
        data, manifest, config = self.inputs()
        data.loc[0,'SA'] = float('inf')
        with self.assertRaisesRegex(ValueError, 'finite'):
            validate_inputs(data, manifest, config)

    def test_objectives_can_select_different_models(self):
        scores = {'ridge': {'mae': 1., 'loss': 2.}, 'svr': {'mae': 2., 'loss': 0.},
                  'random_forest': {'mae': 3., 'loss': 1.}}
        self.assertEqual(choose(scores, 'mae', MODELS), 'ridge')
        self.assertEqual(choose(scores, 'loss', MODELS), 'svr')

    def test_tie_follows_declared_order(self):
        scores = {s: {'loss': 0.} for s in reversed(MODELS)}
        self.assertEqual(choose(scores, 'loss', MODELS), 'ridge')

    def test_nonfinite_is_rejected(self):
        with self.assertRaises(ValueError):
            choose({'ridge': {'loss': float('nan')}}, 'loss', ('ridge',))
