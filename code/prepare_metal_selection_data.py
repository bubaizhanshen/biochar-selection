"""Rebuild additional metal comparisons from HM2 and user-supplied source XML."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from lxml import etree

from panel_input import validate_split

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['pH_biochar', 'C', 'H', 'N', 'O', 'Ash', 'SA']
MASS = {'Cd (II)': 112.414, 'Cu (II)': 63.546, 'Pb (II)': 207.2}


def table_rows(root, identifier):
    tables = root.xpath('//*[local-name()="table" and @id=$id]', id=identifier)
    if len(tables) != 1:
        raise ValueError(f'Expected one source table: {identifier}')
    return [[' '.join(''.join(cell.itertext()).split()) for cell in row]
            for row in tables[0].xpath('.//*[local-name()="row"]')]


def lee_records(path):
    root = etree.parse(str(path))
    descriptors = {}
    for cells in table_rows(root, 'tbl1'):
        for name in ('GB', 'PB', 'MB'):
            if name in cells:
                values = cells[cells.index(name)+1:]
                if len(values) != 9 or name in descriptors:
                    raise ValueError('Unexpected source descriptor table layout')
                descriptors[name] = dict(zip(['SA','C','H','N','S','O','Ash','P','pH_biochar'], map(float,values)))
    rows = table_rows(root, 'tbl2')
    if 'qe (exp.)' not in ' '.join(rows[0]):
        raise ValueError('Experimental capacity column not identified')
    responses = {}
    task = None
    for cells in rows[2:]:
        if cells[0] in ('Pb', 'Cu'):
            task, cells = cells[0], cells[1:]
        if cells[0] in ('GB', 'PB', 'MB'):
            key = task, cells[0]
            if task is None or key in responses:
                raise ValueError('Ambiguous source response row')
            responses[key] = float(cells[1])
    if len(descriptors) != 3 or len(responses) != 6:
        raise ValueError('Expected three materials and six metal/material responses')
    result = []
    for task in ('Pb', 'Cu'):
        for candidate in ('GB', 'PB', 'MB'):
            result.append(dict(source_table_row_id=1_000_000+len(result),
                source_study_id='Kim2019', material_group='Kim2019::'+candidate,
                pollutant=task+' (II)', series='reported_equilibrium',
                response_mg_g=responses[task,candidate], C0_mg_L=50., dose_g_L=1.,
                contact_time_h=48., T=25., pH_solution=5.,
                matrix='0.07 M sodium acetate + 0.03 M acetic acid buffer',
                provenance_tier='direct_original_table', train_eligible=False,
                test_eligible=task=='Cu', **{k:descriptors[candidate][k] for k in FEATURES}))
    return result


def reconstruct(workbook, annotations):
    raw = pd.read_excel(workbook)
    records = []
    for _, annotation in annotations.iterrows():
        index = int(annotation.source_table_row_id)
        if index < 0 or index >= len(raw):
            raise ValueError(f'Workbook row is missing: {index}')
        row = raw.iloc[index]
        if str(row.Adsorbent).strip() != str(annotation.expected_material_label).strip():
            raise ValueError(f'Workbook material label mismatch at row {index}')
        task = annotation.pollutant
        if not str(row.HM).strip().startswith(task.split()[0]):
            raise ValueError(f'Workbook pollutant mismatch at row {index}')
        # Preserve the documented concentration mapping separately from response conversion.
        concentration_mass = 112.411 if task=='Cd (II)' and annotation.source_study_id!='Wang2021_invasive' else MASS[task]
        record = annotation.drop(labels=['expected_material_label','metal_order','lead_order']).to_dict()
        record.update({k:float(row[k]) for k in FEATURES+['T','pH_solution']})
        record['C0_mg_L'] = float(row.C0)*float(annotation.dose_g_L)*concentration_mass
        record['response_mg_g'] = float(row.Eta)*MASS[task]
        records.append(record)
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbook', type=Path, required=True)
    parser.add_argument('--lee-source-xml', type=Path, required=True,
                        help='Lawfully obtained XML for DOI 10.1016/j.jenvman.2019.01.100; not included here')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        parser.error('Output must be an empty directory or a new path.')
    annotation_dir = ROOT / 'data/metal_annotations'
    annotations = pd.read_csv(annotation_dir / 'rows.csv')
    if len(annotations)!=252 or annotations.source_table_row_id.duplicated().any():
        raise ValueError('Unexpected or duplicate source annotation membership')
    records = reconstruct(args.workbook, annotations)
    outputs = {}
    for pool, expected in (('metal',244), ('lead',62)):
        indices = annotations[annotations[pool+'_order'].notna()].sort_values(pool+'_order').index
        data = records.loc[indices].copy()
        if pool=='metal':
            data = pd.concat([data,pd.DataFrame(lee_records(args.lee_source_xml))],ignore_index=True)
        if len(data)!=expected:
            raise ValueError(f'Unexpected {pool} row count')
        data['task_row_id'] = data.groupby('pollutant').cumcount()
        data['mass_balance_excess_mg_g'] = (data.response_mg_g-data.C0_mg_L/data.dose_g_L).clip(lower=0)
        numeric = FEATURES+['T','pH_solution','C0_mg_L','response_mg_g','dose_g_L','contact_time_h']
        if not np.isfinite(data[numeric].to_numpy(float)).all():
            raise ValueError('Nonfinite model input')
        for kind in ([pool,'family'] if pool=='metal' else [pool]):
            manifest = pd.read_csv(annotation_dir / (kind+'_manifest.csv'))
            for _, row in manifest.iterrows():
                validate_split(data[data.pollutant.eq(row.contaminant)],row)
        outputs[pool] = data
    args.out.mkdir(parents=True,exist_ok=True)
    for pool,data in outputs.items():
        data.to_csv(args.out / (pool+'_input.csv'),index=False)
    for name in ('metal_manifest.csv','family_manifest.csv','lead_manifest.csv','metal_protocol.json','lead_protocol.json'):
        shutil.copyfile(annotation_dir/name,args.out/name)
    print('Prepared 244 metal rows, 62 Pb rows, and original source/family manifests.')
    print('These are separate cases, not extra folds in the six-source IBU/Sr comparison.')


if __name__ == '__main__':
    main()
