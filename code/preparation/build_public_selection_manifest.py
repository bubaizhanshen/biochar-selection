"""Prepare a source-held-out pilot on harmonized public experimental records."""

from pathlib import Path
import argparse
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from panel_input import validate_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells', type=Path, required=True)
    parser.add_argument('--base-protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error('Output must be an empty directory or a new path.')
    data = pd.read_csv(args.cells)
    config = json.loads(args.base_protocol.read_text())
    data['source_table_row_id'] = range(2_000_000, 2_000_000 + len(data))
    data['original_row_locator'] = data.raw_record_ids_json
    data['task_row_id'] = data.groupby('pollutant').cumcount()
    data['provenance_tier'] = 'author_released_experimental_data_protocol_harmonized'
    data['train_eligible'] = True
    data['test_eligible'] = True
    data['nominal_mass_balance_excess_mg_g'] = (data.response_mg_g - data.C0_mg_L / data.dose_g_L).clip(lower=0)
    data['mass_balance_excess_mg_g'] = data.nominal_mass_balance_excess_mg_g
    manifests, skipped = [], []
    for pollutant, task in data.groupby('pollutant', sort=True):
        for source, test in task.groupby('source_study_id', sort=True):
            train = task[task.source_study_id.ne(source)]
            label = f'{pollutant.split()[0]}__{source}__public_experimental_cells'
            if train.empty:
                skipped.append({'panel_id': label, 'reason': 'no_other_reviewed_training_source'})
                continue
            row = {
                'panel_id': label, 'contaminant': pollutant, 'test_source': source,
                'training_tier': 'public_experimental_cells', 'holdout_unit': 'study_block',
                'candidate_materials_json': json.dumps(sorted(test.material_group.unique())),
                'train_source_row_ids_json': json.dumps(train.source_table_row_id.astype(int).tolist()),
                'test_source_row_ids_json': json.dumps(test.source_table_row_id.astype(int).tolist()),
                'n_train_rows': len(train), 'n_candidate_rows': len(test),
                'n_train_materials': train.material_group.nunique(), 'n_train_sources': train.source_study_id.nunique(),
                'n_candidate_materials': test.material_group.nunique(),
                'support_note': 'paired_only' if test.material_group.nunique() == 2 else 'multi_candidate',
            }
            validate_split(task, pd.Series(row))
            manifests.append(row)
    config.update({
        'purpose': 'Development comparison on author-released experimental records after source-specific unit and condition harmonization; not independent-laboratory validation.',
        'training_tiers': ['public_experimental_cells'],
        'training_policy': 'Use reviewed released experimental rows after source-supported mappings. Exclude 15 pH=1 rows outside source protocol; retain signed measured response estimates, including negatives. Average rows with identical source/material/series/complete condition vector; do not call them verified replicates.',
        'allowed_provenance_by_training_tier': {'public_experimental_cells': ['author_released_experimental_data_protocol_harmonized']},
        'external_sources': 'None. These blocks are drawn from two existing public datasets from related research groups.',
        'material_features': ['C', 'H', 'N', 'O', 'Ash', 'SA'],
        'condition_features': ['T', 'pH_solution', 'C0_mg_L', 'dose_g_L', 'contact_time_h', 'ionic_strength_M', 'DOM_mg_L'],
        'categorical_condition_features': ['matrix', 'adsorption_mode', 'co_solute_set'],
        'feature_policy': 'Six directly reported shared descriptors. Source-coded nondetection values such as N=0 remain source encodings, not established true zeros. No final concentration, source identity, or predictor computed by our pipeline from test responses enters the model. Initial concentrations remain as released; no nominal-grid snapping. Their experimental acquisition logs were not independently recovered.',
        'response': 'Mass-normalized reported adsorption amount at recorded contact time (mg/g); kinetic qt remains annotated, not relabeled equilibrium qe. No qmax. Competitive IBU source converted from umol/g using factor 0.20628.',
        'scaling': 'Training-only predictor and response standardization; equal cell fitting weights. Cell spread is not treated as a sampling standard error.',
        'known_limits': 'Source-related experimental releases, unresolved physical batch independence, uncertain repeated-row identities and absent original replicate logs. Corrected-label/time/unit mappings and aggregation need sensitivity analysis before manuscript use. CBZ currently has one reviewed source and is not model-evaluable.',
    })
    out.mkdir(parents=True, exist_ok=True)
    data.to_csv(out / 'public_pilot_input.csv', index=False)
    pd.DataFrame(manifests).to_csv(out / 'public_pilot_manifest.csv', index=False)
    pd.DataFrame(skipped).to_csv(out / 'public_pilot_not_evaluable.csv', index=False)
    (out / 'public_pilot_protocol.json').write_text(json.dumps(config, indent=2) + '\n')
    print(pd.DataFrame(manifests)[['panel_id', 'n_train_rows', 'n_train_materials', 'n_train_sources', 'n_candidate_materials']].to_string(index=False))
    print('Skipped:', skipped)


if __name__ == '__main__':
    main()
