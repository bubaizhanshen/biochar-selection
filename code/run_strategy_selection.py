"""Source-held-out selection of fixed models and simple rules."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from panel_input import validate_split
from run_selection_benchmark import estimator, evaluate_conditions, training_fit_rows

MODELS = ('ridge', 'svr', 'random_forest')
STRATEGIES = ('surface_area', 'random', *MODELS)


def validate_inputs(data, manifest, config):
    if data.empty or manifest.empty:
        raise ValueError('Data and split manifest must both be nonempty')
    numeric = [*config['material_features'], *config['condition_features'], 'response_mg_g']
    required = [*numeric, *config['categorical_condition_features'],
                'source_study_id', 'material_group', 'source_table_row_id', 'series', 'pollutant']
    missing = set(required).difference(data.columns)
    if missing:
        raise ValueError(f'Missing input columns: {sorted(missing)}')
    if data[required].isna().any().any():
        raise ValueError('Missing input values; implicit imputation is not permitted')
    if not np.isfinite(data[numeric].to_numpy(dtype=float)).all():
        raise ValueError('Numeric predictors and responses must be finite')


def choose(scores, objective, choices):
    values = [scores[s][objective] for s in choices]
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite inner objective')
    return min(choices, key=lambda s: (scores[s][objective], choices.index(s)))


def predict(strategy, train, test, config):
    if strategy == 'random': return np.zeros(len(test))
    if strategy == 'surface_area': return test.SA.to_numpy(float)
    fit = training_fit_rows(train, config)
    model = estimator(strategy, True, config)
    model.fit(fit, fit.response_mg_g)
    return model.predict(test)


def evaluate(strategy, train, test, config):
    pred = predict(strategy, train, test, config)
    if not np.isfinite(pred).all(): raise ValueError('Nonfinite predictions')
    metrics, coverage = evaluate_conditions(test, pred, set(test.material_group), config)
    if not metrics: raise ValueError('No complete candidate conditions')
    frame = pd.DataFrame(metrics)
    scores = frame.groupby('series')[['selection_loss', 'mae']].mean().mean()
    return pred, metrics, coverage, dict(loss=float(scores.selection_loss), mae=float(scores.mae))


def run(data_path, manifest_path, protocol_path, out):
    if out.exists() and any(out.iterdir()): raise ValueError('Output must be empty')
    data, manifest = pd.read_csv(data_path), pd.read_csv(manifest_path)
    config = json.loads(protocol_path.read_text())
    validate_inputs(data, manifest, config)
    if 'inner_parameter_candidates' in config:
        raise ValueError('This comparison requires fixed model parameters')
    if 'SA' not in config['material_features']:
        raise ValueError('Surface-area comparator requires shared SA input')
    if manifest.panel_id.duplicated().any(): raise ValueError('Duplicate panels')
    out.mkdir(parents=True, exist_ok=True)
    data.to_csv(out/'executed_input.csv', index=False)
    manifest.to_csv(out/'executed_manifest.csv', index=False)
    (out/'protocol.json').write_text(json.dumps(dict(model_config=config,
        strategy_order=STRATEGIES, scope='Retrospective fixed-model development comparison'), indent=2)+'\n')
    trials, predictions, conditions, decisions = [], [], [], []
    for _, panel in manifest.iterrows():
        if panel.holdout_unit != 'study_block': raise ValueError('Source holdout required')
        task = data[data.pollutant.eq(panel.contaminant)].reset_index(drop=True)
        tr, te = validate_split(task, panel)
        train, test = task.iloc[tr], task.iloc[te]
        sources = sorted(train.source_study_id.unique())
        if len(sources) < 2: raise ValueError('At least two inner sources required')
        scores = {}
        for strategy in STRATEGIES:
            inner = []
            for source in sources:
                fit = train[train.source_study_id.ne(source)]
                val = train[train.source_study_id.eq(source)]
                pred, metrics, _, score = evaluate(strategy, fit, val, config)
                inner.append(score)
                meta = dict(panel_id=panel.panel_id, strategy=strategy, inner_source=source)
                trials.append(dict(**meta, **score,
                    train_ids=json.dumps(fit.source_table_row_id.tolist()),
                    validation_ids=json.dumps(val.source_table_row_id.tolist())))
                for (_, r), value in zip(val.iterrows(), pred):
                    predictions.append(dict(**meta, stage='inner', row_id=r.source_table_row_id,
                                            observed=r.response_mg_g, predicted=value))
                conditions.extend(dict(**meta, **m) for m in metrics)
            scores[strategy] = {key: float(np.mean([r[key] for r in inner])) for key in ('mae', 'loss')}
        selected = {
            'fixed_models_mae': choose(scores, 'mae', MODELS),
            'fixed_models_selection_loss': choose(scores, 'loss', MODELS),
            'models_and_rules_selection_loss': choose(scores, 'loss', STRATEGIES),
        }
        outer = {}
        for strategy in STRATEGIES:
            pred, _, _, score = evaluate(strategy, train, test, config)
            outer[strategy] = score['loss']
            for (_, r), value in zip(test.iterrows(), pred):
                predictions.append(dict(panel_id=panel.panel_id, strategy=strategy,
                    inner_source='', stage='outer', row_id=r.source_table_row_id,
                    observed=r.response_mg_g, predicted=value))
        for selector, strategy in selected.items():
            decisions.append(dict(panel_id=panel.panel_id, source=panel.test_source,
                pollutant=panel.contaminant, selector=selector, selected_strategy=strategy,
                outer_loss=outer[strategy]))
        for name, rows in [('inner_trials', trials), ('predictions', predictions),
                           ('inner_conditions', conditions), ('decisions', decisions)]:
            pd.DataFrame(rows).to_csv(out/f'{name}.csv', index=False)
        print(panel.panel_id, selected, flush=True)
    (out/'completed.json').write_text(json.dumps(dict(outer_folds=len(manifest)))+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'manifest', 'protocol', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.manifest, args.protocol, args.out)
