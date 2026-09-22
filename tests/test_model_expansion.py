import json
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from model_registry import configured_models, DEFAULT_MODELS
from expand_model_protocol import expand
from run_selection_benchmark import estimator
from run_strategy_selection import choose

ROOT=Path(__file__).resolve().parents[1]


class ModelExpansionTests(unittest.TestCase):
    def config(self):
        base=json.loads((ROOT/'config/selection.json').read_text())
        return expand(base,json.loads((ROOT/'config/model_expansion.json').read_text()))

    def test_order_and_parameters(self):
        config=self.config()
        self.assertEqual(configured_models(config)[:3],DEFAULT_MODELS)
        self.assertEqual(len(configured_models(config)),9)
        base=json.loads((ROOT/'config/selection.json').read_text())
        for model in DEFAULT_MODELS:self.assertEqual(config[model],base[model])

    def test_duplicate_unknown_and_missing_config_rejected(self):
        for names in [('ridge','ridge'),('unknown',),()]:
            with self.assertRaises(ValueError):configured_models({'models':names,'ridge':{}})
        with self.assertRaises(ValueError):configured_models({'models':['xgboost']})

    def test_nine_models_fit_without_test_preprocessing(self):
        config=self.config()
        rng=np.random.default_rng(83)
        numeric=config['condition_features']+config['material_features']
        data=pd.DataFrame(rng.normal(size=(24,len(numeric))),columns=numeric)
        for column in config['categorical_condition_features']:data[column]='training'
        target=rng.normal(size=24)
        test=data.iloc[:4].copy()
        test[numeric]+=50
        for column in config['categorical_condition_features']:test[column]='unseen'
        for name in configured_models(config):
            with self.subTest(name=name):
                if name in ('xgboost','lightgbm','catboost') and importlib.util.find_spec(name) is None:
                    self.skipTest('Optional dependency not installed: '+name)
                model=estimator(name,True,config)
                model.fit(data,target)
                prediction=model.predict(test)
                self.assertEqual(prediction.shape,(4,))
                self.assertTrue(np.isfinite(prediction).all())
                scaler=model.regressor_.named_steps['inputs'].named_transformers_['numeric']
                np.testing.assert_allclose(scaler.mean_,data[numeric].mean())

    def test_objectives_use_all_models_and_declared_ties(self):
        names=configured_models(self.config())
        scores={name:{'mae':10.,'loss':10.} for name in names}
        scores['catboost']['mae']=0
        scores['knn']['loss']=0
        self.assertEqual(choose(scores,'mae',names),'catboost')
        self.assertEqual(choose(scores,'loss',names),'knn')
        scores['ridge']['loss']=0
        self.assertEqual(choose(scores,'loss',names),'ridge')
