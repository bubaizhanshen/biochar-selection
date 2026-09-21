"""Attach recorded responses using exact existing cell membership, without new matching."""

from pathlib import Path
import argparse
import csv
import json
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cells', type=Path, required=True)
parser.add_argument('--source-rows', type=Path, required=True)
parser.add_argument('--base-protocol', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
OUT = args.out.resolve()
if OUT.exists() and (not OUT.is_dir() or any(OUT.iterdir())):
    parser.error('Output must be an empty directory or a new path.')
cells = pd.read_csv(args.cells)
raw = pd.read_csv(args.source_rows).set_index('record_id', verify_integrity=True)
config = json.loads(args.base_protocol.read_text())
features = config['material_features'] + config['condition_features'] + config['categorical_condition_features']
numeric = config['material_features'] + config['condition_features']
lists = []
seen = set()
for _, cell in cells.iterrows():
    ids = json.loads(cell.raw_record_ids_json)
    if len(ids) != int(cell.n_compilation_rows) or len(set(ids)) != len(ids):
        raise ValueError('Cell membership disagrees with the recorded count or contains duplicate IDs')
    if seen.intersection(ids):
        raise ValueError('A released record is assigned to more than one cell')
    seen.update(ids)
    records = raw.loc[ids]
    if not records.row_disposition.eq('admit_to_public_data_analysis').all():
        raise ValueError('Cell contains records outside the admitted source data')
    for column in ['source_study_id', 'material_group', 'series', 'pollutant', *features]:
        if column in numeric:
            np.testing.assert_allclose(records[column].to_numpy(dtype=float), float(cell[column]), rtol=1e-10, atol=1e-8)
        else:
            if not records[column].eq(cell[column]).all():
                raise ValueError(f'Raw-to-cell mismatch: {column}')
    values = records.response_mg_g.to_numpy()
    np.testing.assert_allclose(values.mean(), cell.response_mg_g, rtol=1e-10, atol=1e-10)
    lists.append(json.dumps(values.tolist()))
cells['raw_responses_mg_g_json'] = lists
OUT.mkdir(parents=True, exist_ok=True)
with args.cells.open(newline='') as handle:
    original = list(csv.DictReader(handle))
if len(original) != len(lists):
    raise ValueError('CSV row count changed during response attachment')
with (OUT / 'public_raw_response_input.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=[*original[0], 'raw_responses_mg_g_json'])
    writer.writeheader()
    for row, values in zip(original, lists):
        writer.writerow({**row, 'raw_responses_mg_g_json': values})
config['training_cell_frequency_column'] = 'n_compilation_rows'
config['training_raw_responses_json_column'] = 'raw_responses_mg_g_json'
config['purpose'] = 'Retrospective sensitivity using original harmonized released response rows in training; unchanged cell-mean outer tests.'
config['training_policy'] = 'Expand training cells to original response lists with exact raw-row membership. Do not claim these rows are independent experimental replicates.'
config['scaling'] = 'Training-only standardization over original harmonized released training records.'
(OUT / 'public_raw_response_protocol.json').write_text(json.dumps(config, indent=2) + '\n')
print('Validated cells:', len(cells), 'original records:', len(seen))
