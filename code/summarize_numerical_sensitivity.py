"""Summarize complete precision/seed runs without selecting favorable settings."""
import argparse
import json
from pathlib import Path

import pandas as pd

from run_numerical_sensitivity import PRECISIONS, SEEDS


def input_filenames(contract):
    if 'input_files' in contract:
        return contract['input_files']
    legacy_inputs = contract.get('inputs', {})
    return sorted(legacy_inputs) if isinstance(legacy_inputs, dict) else legacy_inputs


def summarize(paths, out):
    if out.exists():
        raise FileExistsError(out)
    frames, runs = [], []
    for path in paths:
        contract = json.loads((path/'contract.json').read_text())
        complete = json.loads((path/'completed.json').read_text())
        frame = pd.read_csv(path/'decisions.csv', dtype={'precision': str})
        if not frame.policy.eq(contract['policy']).all() or not frame.precision.eq(contract['precision']).all():
            raise ValueError('Run labels disagree with contract')
        if set(frame.seed) != set(contract['seeds']) or complete['seeds'] != len(contract['seeds']):
            raise ValueError('Incomplete seed coverage')
        if len(frame[['source', 'seed']].drop_duplicates()) != complete['folds']:
            raise ValueError('Incomplete source coverage')
        for key, group in frame.groupby(['source', 'seed']):
            sizes = sorted({3, len(contract['model_config']['models'])})
            expected = {(size, selector) for size in sizes for selector in ('mae', 'loss', 'models_rules')}
            if set(zip(group.model_count, group.selector)) != expected or len(group) != len(expected):
                raise ValueError(f'Incomplete or duplicate selector coverage: {key}')
        frames.append(frame)
        runs.append(contract)
    data = pd.concat(frames, ignore_index=True)
    keys = ['policy', 'precision', 'seed', 'source', 'model_count', 'selector']
    if data.duplicated(keys).any():
        raise ValueError('Duplicate runs/settings')
    expected_grid = {(p, s) for p in PRECISIONS for s in SEEDS}
    for key, group in data.groupby(['policy', 'source', 'model_count', 'selector']):
        if set(zip(group.precision, group.seed)) != expected_grid:
            raise ValueError(f'Full four-precision/five-seed grid required: {key}')
    # Require the same input filenames and protocol within each training policy.
    for policy in data.policy.unique():
        contracts = [r for r in runs if r['policy'] == policy]
        reference = contracts[0]
        if any(input_filenames(r) != input_filenames(reference)
               or r['model_config'] != reference['model_config']
               for r in contracts[1:]):
            raise ValueError(f'Input or code versions differ within {policy}')
    source = data.groupby(['policy', 'source', 'pollutant', 'model_count', 'selector']).agg(
        loss_min=('outer_loss', 'min'), loss_max=('outer_loss', 'max'),
        distinct_choices=('selected', 'nunique'),
        gain_min=('gain_over_surface_area', 'min'), gain_max=('gain_over_surface_area', 'max')).reset_index()
    task = data.groupby(['policy', 'precision', 'seed', 'pollutant', 'model_count', 'selector'])[
        ['outer_loss', 'gain_over_surface_area']].mean().reset_index()
    paired = task.pivot(index=['policy', 'precision', 'seed', 'pollutant', 'model_count'],
                        columns='selector', values='outer_loss').reset_index()
    paired['improvement'] = paired.mae-paired.loss
    ranges = paired.groupby(['policy', 'pollutant', 'model_count']).agg(
        improvement_min=('improvement', 'min'), improvement_max=('improvement', 'max'),
        positive_settings=('improvement', lambda v: int((v > 1e-12).sum())),
        settings=('improvement', 'size')).reset_index()
    frequency = data.groupby(['policy', 'source', 'model_count', 'selector', 'selected']).size().rename('settings').reset_index()
    out.mkdir(parents=True)
    for name, frame in [('source_stability', source), ('task_grid', task),
                        ('task_stability', ranges), ('selection_frequency', frequency)]:
        frame.to_csv(out/f'{name}.csv', index=False)
    (out/'scope.json').write_text(json.dumps(dict(runs=len(runs), decisions=len(data),
        interpretation='Computational sensitivity ranges, not confidence intervals or independent experiments'), indent=2)+'\n')
    print(ranges.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.runs, args.out)
