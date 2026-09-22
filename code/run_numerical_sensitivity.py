"""Full source-grid numerical sensitivity; evaluation responses remain unchanged."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from model_registry import configured_models
from panel_input import validate_split
from run_selection_benchmark import estimator, evaluate_conditions, training_fit_rows
from run_strategy_selection import choose, validate_inputs

PRECISIONS = ('released', '12', '8', '4')
SEEDS = tuple(range(1729, 1734))


def score(test, prediction, config):
    metrics, coverage = evaluate_conditions(test, prediction, set(test.material_group), config)
    frame = pd.DataFrame(metrics)
    if frame.empty:
        raise ValueError('No complete conditions')
    mean = frame.groupby('series')[['selection_loss', 'mae']].mean().mean()
    return dict(loss=float(mean.selection_loss), mae=float(mean.mae)), metrics, coverage


def training_diagnostic(train, prediction, config):
    frame = train.copy()
    frame['_prediction'] = prediction
    values = [score(group, group['_prediction'].to_numpy(), config)[0]
              for _, group in frame.groupby('source_study_id', sort=True)]
    return {key: float(np.mean([v[key] for v in values])) for key in ('mae', 'loss')}


def support(train, test, config):
    columns = config['material_features']
    vectors = train[['material_group', *columns]].drop_duplicates()
    if vectors.material_group.duplicated().any():
        raise ValueError('Material descriptors vary within material group')
    # Remove a reference row before centering to avoid a spurious rank direction
    # from cancellation when only two materials share large descriptor offsets.
    x = vectors[columns].to_numpy(float)
    x = x - x[0]
    x = x - x.mean(axis=0)
    scale = np.sqrt(np.mean(x*x, axis=0))
    x = x / np.where(scale > 0, scale, 1.)
    singular = np.linalg.svd(x, compute_uv=False)
    tolerance = float(max(x.shape) * np.finfo(float).eps * singular.max())
    return dict(train_sources=';'.join(sorted(train.source_study_id.unique())),
        train_materials=int(train.material_group.nunique()), train_cells=len(train),
        distinct_material_vectors=len(vectors[columns].drop_duplicates()),
        material_rank=int((singular > tolerance).sum()), rank_tolerance=tolerance,
        test_materials=int(test.material_group.nunique()), test_cells=len(test),
        **{f'unknown_{c}_cells': int((~test[c].isin(train[c])).sum())
           for c in config['categorical_condition_features']})


def candidate_probabilities(test, prediction, config):
    frame = test.copy()
    frame['prediction'] = prediction
    keys = ['series', *config['condition_features'], *config['categorical_condition_features']]
    for c in config['condition_features']:
        frame[c] = frame[c].round(8)
    result = []
    candidates = set(test.material_group)
    for key, group in frame.groupby(keys, sort=True, dropna=False):
        if set(group.material_group) != candidates:
            continue
        selected = np.isclose(group.prediction, group.prediction.max(), atol=1e-12, rtol=0)
        for (_, row), prob in zip(group.iterrows(), selected / selected.sum()):
            result.append(dict(row_id=row.source_table_row_id, material=row.material_group,
                series=row.series, probability=float(prob), ties=int(selected.sum())))
    return result


def run(data_path, manifest_path, protocol_path, out, policy, precision,
        seeds=SEEDS, reference=None):
    if out.exists():
        raise FileExistsError(out)
    if precision not in PRECISIONS:
        raise ValueError('Unsupported fitting precision')
    seeds = tuple(seeds)
    if not seeds or len(seeds) != len(set(seeds)) or any(s < 0 for s in seeds):
        raise ValueError('Seeds must be unique nonnegative integers')
    data = pd.read_csv(data_path)
    manifest = pd.read_csv(manifest_path)
    config = json.loads(protocol_path.read_text())
    validate_inputs(data, manifest, config)
    models = configured_models(config)
    if models[:3] != ('ridge', 'svr', 'random_forest') or len(models) not in (3, 9):
        raise ValueError('Use the declared three- or nine-model protocol')
    if 'inner_parameter_candidates' in config:
        raise ValueError('Fixed parameters required')
    if 'SA' not in config['material_features'] or manifest.panel_id.duplicated().any():
        raise ValueError('SA and unique panels required')
    if not manifest.holdout_unit.eq('study_block').all():
        raise ValueError('Only complete source holdouts are supported')
    out.mkdir(parents=True, exist_ok=False)
    receipt = dict(policy=policy, precision=precision, seeds=seeds, model_config=config,
        rounding='Fitting responses only, after original-response expansion',
        inference='Computational sensitivity, not confidence intervals',
        inputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [data_path, manifest_path, protocol_path, Path(__file__)]})
    (out / 'contract.json').write_text(json.dumps(receipt, indent=2) + '\n')
    strategies = ('surface_area', 'random', *models)
    cache = {}
    trials, outer, decisions, probabilities, conditions, supports = [], [], [], [], [], []

    def evaluate(name, train, test, seed):
        conf = copy.deepcopy(config)
        randomized = name in models and ('random_state' in conf[name] or 'random_seed' in conf[name])
        key = (name, tuple(train.source_table_row_id), tuple(test.source_table_row_id),
               seed if randomized else None)
        if key in cache:
            return cache[key]
        if name in ('surface_area', 'random'):
            pred = test.SA.to_numpy(float) if name == 'surface_area' else np.zeros(len(test))
            train_score = dict(train_mae=np.nan, train_loss=np.nan)
        else:
            if randomized:
                conf[name]['random_seed' if name == 'catboost' else 'random_state'] = seed
            fit = training_fit_rows(train, conf).copy()
            original_y = fit.response_mg_g.to_numpy(copy=True)
            if precision != 'released':
                fit['response_mg_g'] = np.round(original_y, int(precision))
            model = estimator(name, True, conf)
            model.fit(fit, fit.response_mg_g)
            pred = model.predict(test)
            training_score = training_diagnostic(train, model.predict(train), conf)
            train_score = dict(train_mae=training_score['mae'], train_loss=training_score['loss'])
        if not np.isfinite(pred).all():
            raise ValueError('Nonfinite prediction')
        values, metrics, coverage = score(test, pred, conf)
        if name not in models:
            values['mae'] = np.nan
        cache[key] = values, metrics, coverage, pred, train_score
        return cache[key]

    for seed in seeds:
        for _, panel in manifest.iterrows():
            task = data[data.pollutant.eq(panel.contaminant)].reset_index(drop=True)
            tr, te = validate_split(task, panel)
            train, test = task.iloc[tr], task.iloc[te]
            if train.source_study_id.nunique() < 2:
                raise ValueError('At least two inner sources required')
            meta = dict(policy=policy, precision=precision, seed=seed,
                        panel_id=panel.panel_id, source=panel.test_source, pollutant=panel.contaminant)
            scores, outer_by_name = {}, {}
            for name in strategies:
                inner = []
                for source in sorted(train.source_study_id.unique()):
                    fitting = train[train.source_study_id.ne(source)]
                    val = train[train.source_study_id.eq(source)]
                    values, _, _, _, training = evaluate(name, fitting, val, seed)
                    trials.append(dict(**meta, strategy=name, inner_source=source, **values, **training))
                    inner.append(values)
                    if seed == seeds[0] and name == 'ridge':
                        supports.append(dict(**meta, stage='inner', validation_source=source,
                                             **support(fitting, val, config)))
                scores[name] = {c: float(np.mean([v[c] for v in inner])) for c in ('mae', 'loss')}
                values, metrics, coverage, pred, training = evaluate(name, train, test, seed)
                outer_by_name[name] = values
                outer.append(dict(**meta, strategy=name, **values, **training,
                                  conditions=len(metrics), scored_cells=sum(m['candidate_count'] for m in metrics)))
                conditions.extend(dict(**meta, strategy=name, **m) for m in metrics)
                probabilities.extend(dict(**meta, strategy=name, **p)
                                     for p in candidate_probabilities(test, pred, config))
            if seed == seeds[0]:
                supports.append(dict(**meta, stage='outer', validation_source=panel.test_source,
                                     **support(train, test, config)))
            for size in sorted({3, len(models)}):
                for selector, objective, available in (
                    ('mae', 'mae', models[:size]), ('loss', 'loss', models[:size]),
                    ('models_rules', 'loss', ('surface_area', 'random', *models[:size]))):
                    winner = choose(scores, objective, available)
                    tied = [m for m in available if scores[m][objective] == scores[winner][objective]]
                    losses = [outer_by_name[m]['loss'] for m in tied]
                    decisions.append(dict(**meta, model_count=size, selector=selector, selected=winner,
                        inner_score=scores[winner][objective], outer_loss=outer_by_name[winner]['loss'],
                        outer_mae=outer_by_name[winner]['mae'],
                        gain_over_surface_area=outer_by_name['surface_area']['loss']-outer_by_name[winner]['loss'],
                        tied_strategies=';'.join(tied), tied_outer_min=min(losses), tied_outer_max=max(losses),
                        tied_outer_mean=float(np.mean(losses))))
            print(policy, precision, seed, panel.test_source, flush=True)
    for name, rows in [('inner_scores', trials), ('outer_scores', outer), ('decisions', decisions),
                       ('candidate_probabilities', probabilities), ('conditions', conditions), ('support', supports)]:
        pd.DataFrame(rows).to_csv(out / f'{name}.csv', index=False)
    baseline_verified = False
    if reference is not None:
        if precision != 'released' or 1729 not in seeds:
            raise ValueError('Reference comparison requires released precision and seed 1729')
        got = pd.DataFrame(outer).query('seed == 1729').set_index(['panel_id', 'strategy'])
        expected = pd.read_csv(reference / 'outer_results.csv').set_index(['panel_id', 'strategy'])
        if set(got.index) != set(expected.index):
            raise ValueError('Reference strategy/fold coverage differs')
        np.testing.assert_allclose(got.loc[expected.index, 'loss'], expected.loss, rtol=1e-12, atol=1e-12)
        mask = ~expected.index.get_level_values('strategy').isin(['surface_area', 'random'])
        np.testing.assert_allclose(got.loc[expected.index[mask], 'mae'],
                                   expected.loc[expected.index[mask], 'mae'], rtol=1e-12, atol=1e-12)
        baseline_verified = True
    (out / 'completed.json').write_text(json.dumps(dict(seeds=len(seeds),
        folds=len(seeds)*len(manifest), outer_rows=len(outer),
        unique_evaluations=len(cache), baseline_verified=baseline_verified)) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'manifest', 'protocol', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--training-policy', choices=('cell_mean', 'original_response'), required=True)
    parser.add_argument('--precision', choices=PRECISIONS, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    warnings.filterwarnings('ignore', message='.*valid feature names.*')
    run(args.data, args.manifest, args.protocol, args.out,
        args.training_policy, args.precision, args.seeds, args.reference)
