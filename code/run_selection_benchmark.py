"""Fixed-model selection pilot with explicit train/test membership.

This runner never loads a historical task implicitly or selects a model from
outer-test performance. Inputs are a harmonized table, manifest and protocol.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from model_registry import configured_models, make_regressor

from panel_input import validate_split
from selection_metrics import selection_metrics


def estimator(name, full, config):
    numeric = config['condition_features'] + (config['material_features'] if full else [])
    columns = ColumnTransformer([
        ('numeric', StandardScaler(), numeric),
        ('matrix', OneHotEncoder(handle_unknown='ignore', sparse_output=False), config['categorical_condition_features']),
    ])
    pipe = Pipeline([('inputs', columns), ('model', make_regressor(name, config[name]))])
    return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())


def evaluate_conditions(test, prediction, candidates, config):
    frame = test.copy()
    frame['prediction_mg_g'] = prediction
    keys = ['series', *config['condition_features'], *config['categorical_condition_features']]
    for key in config['condition_features']:
        frame[key] = frame[key].round(8)
    metrics, coverage = [], []
    for condition, group in frame.groupby(keys, sort=True, dropna=False):
        meta = dict(zip(keys, condition))
        present = set(group.material_group)
        reason = 'complete' if present == candidates else 'incomplete_candidate_grid'
        coverage.append({**meta, 'status': reason, 'rows': len(group), 'candidates_present': len(present)})
        if reason != 'complete':
            continue
        if group.material_group.duplicated().any():
            raise RuntimeError('Repeated candidate-condition cells require an explicit aggregation policy')
        group = group.sort_values('material_group')
        m = selection_metrics(group.response_mg_g, group.prediction_mg_g)
        metrics.append({**meta, **m})
    return metrics, coverage


def select_parameters(train, name, full, config):
    """Select within outer training sources; each validation source has equal weight."""
    grids = config.get('inner_parameter_candidates')
    if grids is None:
        return config, [], 'fixed_parameters'
    sources = sorted(train.source_study_id.unique())
    if len(sources) < 2:
        return config, [], 'fixed_insufficient_training_sources'
    if name not in grids or not grids[name]:
        raise ValueError('Explicit nonempty parameter candidates required for each family')
    trials = []
    configs = []
    for index, parameters in enumerate(grids[name]):
        candidate = copy.deepcopy(config)
        candidate[name].update(parameters)
        configs.append(candidate)
        for source in sources:
            fitting = train[train.source_study_id.ne(source)]
            validation = train[train.source_study_id.eq(source)]
            model = estimator(name, full, candidate)
            model.fit(fitting, fitting.response_mg_g)
            prediction = model.predict(validation)
            if not np.isfinite(prediction).all():
                raise ValueError('Nonfinite inner prediction')
            trials.append({
                'candidate_index': index, 'parameters_json': json.dumps(candidate[name], sort_keys=True),
                'inner_validation_source': source,
                'inner_train_row_ids_json': json.dumps(fitting.source_table_row_id.tolist()),
                'inner_validation_row_ids_json': json.dumps(validation.source_table_row_id.tolist()),
                'inner_mae': float(np.abs(prediction - validation.response_mg_g).mean()),
                'inner_train_sources': fitting.source_study_id.nunique(),
            })
    means = pd.DataFrame(trials).groupby('candidate_index', sort=True).inner_mae.mean()
    winner = int(means.idxmin())  # Exact ties use the declared candidate order.
    for trial in trials:
        trial['selected'] = trial['candidate_index'] == winner
        trial['source_balanced_inner_mae'] = float(means.loc[trial['candidate_index']])
    return configs[winner], trials, 'leave_one_training_source_out'


def training_fit_rows(train, config):
    column = config.get('training_cell_frequency_column')
    if column is None:
        if config.get('training_raw_responses_json_column'):
            raise ValueError('Original-response training requires a cell-frequency column')
        return train
    if 'inner_parameter_candidates' in config:
        raise ValueError('Frequency sensitivity is fixed-parameter only; inner weighting not specified')
    counts = train[column].to_numpy(dtype=float)
    if not np.isfinite(counts).all() or (counts < 1).any() or (counts != np.floor(counts)).any():
        raise ValueError('Training cell frequencies must be finite positive integers')
    expanded = train.iloc[np.repeat(np.arange(len(train)), counts.astype(int))].reset_index(drop=True)
    response_column = config.get('training_raw_responses_json_column')
    if response_column:
        responses = []
        for (_, row), count in zip(train.iterrows(), counts):
            values = np.asarray(json.loads(row[response_column]), dtype=float)
            if values.ndim != 1 or len(values) != int(count) or not np.isfinite(values).all():
                raise ValueError('Invalid original response list')
            if not np.isclose(values.mean(), row.response_mg_g, rtol=1e-10, atol=1e-10):
                raise ValueError('Original responses do not reproduce cell mean')
            responses.extend(values)
        expanded['response_mg_g'] = responses
    return expanded


def run(data_path, manifest_path, protocol_path, out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Output directory must be empty; do not overwrite a previous run')
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(data_path)
    manifest = pd.read_csv(manifest_path)
    config = json.loads(Path(protocol_path).read_text())
    features = [*config['material_features'], *config['condition_features'], *config['categorical_condition_features']]
    if data[features + ['response_mg_g']].isna().any().any():
        raise RuntimeError('Missing features/response: no implicit test-time imputation')
    if manifest.panel_id.duplicated().any():
        raise RuntimeError('Duplicate manifest experiment IDs')
    manifest.to_csv(out / 'executed_manifest.csv', index=False)
    Path(out / 'executed_protocol.json').write_text(json.dumps(config, indent=2) + '\n')
    data.to_csv(out / 'executed_input.csv', index=False)
    predictions, condition_metrics, all_coverage, timing, splits = [], [], [], [], []
    tuning_records, selected_records = [], []
    started = time.monotonic()
    for _, row in manifest.iterrows():
        task = data[data.pollutant.eq(row.contaminant)].copy().reset_index(drop=True)
        train_pos, test_pos = validate_split(task, row)
        train, test = task.iloc[train_pos].copy(), task.iloc[test_pos].copy()
        if not train.train_eligible.eq(True).all() or not test.test_eligible.eq(True).all():
            raise RuntimeError('Manifest violates training/test roles')
        allowed = config.get('allowed_provenance_by_training_tier', {
            'verified_compilation': ['verified_compilation'],
            'expanded_compilation': ['verified_compilation', 'local_compilation_condition_supported'],
        })
        if row.training_tier not in allowed:
            raise RuntimeError('Unknown training tier')
        expected_tiers = set(allowed[row.training_tier])
        if not set(train.provenance_tier).issubset(expected_tiers):
            raise RuntimeError('Training provenance inconsistent with manifest')
        for role, subset in [('train', train), ('test', test)]:
            for item in subset[['source_table_row_id', 'source_study_id', 'material_group', 'original_row_locator']].to_dict('records'):
                splits.append({'panel_id': row.panel_id, 'role': role, **item})
        candidates = set(json.loads(row.candidate_materials_json))
        fitting_rows = training_fit_rows(train, config)
        for name in configured_models(config):
            for full in (False, True):
                model_id = name + ('_full' if full else '_condition_only')
                begin = time.monotonic()
                chosen_config, trials, tuning_status = select_parameters(train, name, full, config)
                tuning_records.extend([{'panel_id': row.panel_id, 'model': model_id, **t} for t in trials])
                selected_records.append({'panel_id': row.panel_id, 'model': model_id,
                                         'status': tuning_status,
                                         'parameters_json': json.dumps(chosen_config[name], sort_keys=True)})
                model = estimator(name, full, chosen_config)
                model.fit(fitting_rows[features], fitting_rows.response_mg_g)
                pred = model.predict(test[features])
                if not np.isfinite(pred).all():
                    raise RuntimeError('Nonfinite prediction')
                meta = {
                    'panel_id': row.panel_id, 'pollutant': row.contaminant,
                    'source_study_id': row.test_source, 'training_tier': row.training_tier,
                    'model': model_id, 'n_train_rows': len(train),
                    'n_fitting_rows_after_frequency_expansion': len(fitting_rows),
                    'n_train_materials': train.material_group.nunique(),
                    'n_train_sources': train.source_study_id.nunique(),
                    'support_note': row.support_note,
                }
                p = test[['source_table_row_id', 'material_group', 'series', 'response_mg_g']].copy()
                p['prediction_mg_g'] = pred
                p['training_mean_mg_g'] = fitting_rows.response_mg_g.mean()
                for key, value in meta.items():
                    p[key] = value
                predictions.extend(p.to_dict('records'))
                values, coverage = evaluate_conditions(test, pred, candidates, config)
                condition_metrics.extend([{**meta, **value} for value in values])
                all_coverage.extend([{**meta, **value} for value in coverage])
                timing.append({**meta, 'fit_and_score_seconds': time.monotonic() - begin})
        print(row.panel_id, 'completed', flush=True)

    pd.DataFrame(predictions).to_csv(out / 'predictions.csv', index=False)
    result = pd.DataFrame(condition_metrics)
    result.to_csv(out / 'condition_metrics.csv', index=False)
    pd.DataFrame(all_coverage).to_csv(out / 'condition_coverage.csv', index=False)
    pd.DataFrame(splits).to_csv(out / 'split_membership.csv', index=False)
    pd.DataFrame(timing).to_csv(out / 'timing.csv', index=False)
    pd.DataFrame(selected_records).to_csv(out / 'selected_parameters.csv', index=False)
    if tuning_records:
        pd.DataFrame(tuning_records).to_csv(out / 'inner_tuning.csv', index=False)
    ids = ['panel_id', 'pollutant', 'source_study_id', 'training_tier', 'model', 'support_note']
    scores = ['selection_loss', 'random_selection_loss', 'gain_over_random', 'normalized_loss',
              'pairwise_accuracy', 'mae', 'mse', 'common_bias_squared', 'relative_error_mse',
              'observed_best_selected_probability', 'observed_range']
    if len(result):
        by_series = result.groupby(ids + ['series'], as_index=False)[scores].mean()
        by_series.to_csv(out / 'series_metrics.csv', index=False)
        by_source = by_series.groupby(ids, as_index=False)[scores].mean()
        by_source.to_csv(out / 'source_metrics.csv', index=False)
        condition_counts = result.groupby(ids).size().rename('n_conditions').reset_index()
        by_source = by_source.merge(condition_counts, on=ids, validate='one_to_one')
        by_source.to_csv(out / 'source_metrics.csv', index=False)
    summary = {'status': 'completed_development_pilot', 'experiments': len(manifest),
               'fitted_models': len(timing), 'prediction_rows': len(predictions),
               'elapsed_seconds': time.monotonic() - started,
               'scope': 'Retrospective development analysis; optional training-source-only tuning. Not evidence of environmental deployment',
               'inner_tuning_enabled': 'inner_parameter_candidates' in config}
    Path(out / 'run_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.manifest, args.protocol, args.out)
