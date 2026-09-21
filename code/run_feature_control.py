"""Compare fixed models with full, material-only, and condition-only inputs."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from panel_input import validate_split
from run_selection_benchmark import estimator, evaluate_conditions


def feature_estimator(name, variant, config):
    if variant not in ('full', 'material_only', 'condition_only'):
        raise ValueError(f'Unknown feature variant: {variant}')
    selected = copy.deepcopy(config)
    if variant == 'material_only':
        selected['condition_features'] = []
        selected['categorical_condition_features'] = []
    return estimator(name, variant != 'condition_only', selected)


def run(data_path, manifest_path, protocol_path, out):
    config = json.loads(Path(protocol_path).read_text())
    if ('inner_parameter_candidates' in config or config.get('training_cell_frequency_column')
            or config.get('training_raw_responses_json_column')):
        raise ValueError('Feature controls require fixed parameters and cell-mean training')
    data = pd.read_csv(data_path)
    manifest = pd.read_csv(manifest_path)
    if manifest.empty or manifest.panel_id.duplicated().any():
        raise ValueError('Manifest must contain unique, nonempty experiments')
    if manifest.duplicated(['contaminant', 'test_source']).any():
        raise ValueError('Each task-source must have exactly one feature-control fold')
    features = config['material_features'] + config['condition_features'] + config['categorical_condition_features']
    if data[features + ['response_mg_g']].isna().any().any():
        raise ValueError('Missing predictors or response; no implicit imputation')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    results, predictions = [], []
    for row in manifest.itertuples(index=False):
        task = data[data.pollutant.eq(row.contaminant)].reset_index(drop=True)
        train_idx, test_idx = validate_split(task, pd.Series(row._asdict()))
        train, test = task.iloc[train_idx], task.iloc[test_idx]
        if not train.train_eligible.eq(True).all() or not test.test_eligible.eq(True).all():
            raise ValueError('Manifest violates training/test roles')
        allowed = config.get('allowed_provenance_by_training_tier', {})
        if row.training_tier not in allowed or not set(train.provenance_tier).issubset(allowed[row.training_tier]):
            raise ValueError('Training provenance inconsistent with protocol')
        candidates = set(json.loads(row.candidate_materials_json))
        for variant in ('full', 'material_only', 'condition_only'):
            for name in ('ridge', 'svr', 'random_forest'):
                model = feature_estimator(name, variant, config)
                model.fit(train, train.response_mg_g)
                pred = model.predict(test)
                if not np.isfinite(pred).all():
                    raise ValueError('Nonfinite prediction')
                # Scoring always uses the complete condition vector, not the reduced inputs.
                metrics, _ = evaluate_conditions(test, pred, candidates, config)
                if not metrics:
                    raise ValueError(f'No complete conditions: {row.test_source}')
                values = pd.DataFrame(metrics)
                loss = values.groupby('series').selection_loss.mean().mean()
                if variant == 'condition_only' and not np.allclose(
                    values.selection_loss, values.random_selection_loss, atol=1e-10, rtol=1e-10
                ):
                    raise ValueError('Condition-only selection differs from uniform candidate choice')
                results.append(dict(source=row.test_source, pollutant=row.contaminant,
                                    model=name, features=variant, loss_mg_g=loss,
                                    conditions=len(metrics)))
                frame = test[['source_table_row_id', 'material_group', 'response_mg_g']].copy()
                frame['prediction_mg_g'] = pred
                frame['source'] = row.test_source
                frame['model'] = name
                frame['features'] = variant
                predictions.append(frame)
        print(row.test_source, 'complete', flush=True)
    result = pd.DataFrame(results)
    result.to_csv(out / 'source_results.csv', index=False)
    pd.concat(predictions).to_csv(out / 'predictions.csv', index=False)
    summary = result.groupby(['pollutant', 'model', 'features']).loss_mg_g.mean().unstack('features')
    summary['gain_from_conditions_mg_g'] = summary.material_only - summary.full
    summary.to_csv(out / 'task_summary.csv')
    (out / 'executed_protocol.json').write_text(json.dumps(config, indent=2) + '\n')
    manifest.to_csv(out / 'executed_manifest.csv', index=False)
    (out / 'run_summary.json').write_text(json.dumps(dict(
        completed=True, fits=len(result), source_folds=len(manifest),
        status='retrospective_development_feature_control',
        aggregation='equal conditions within series; equal series within source; equal sources within task',
        training='cell means; fixed models; no outer-outcome model selection'), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'manifest', 'protocol', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.manifest, args.protocol, args.out)
