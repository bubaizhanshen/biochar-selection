"""Harmonize reviewed author-released sources and preserve every raw row.

Compilation-cell means are descriptive aggregates, not claimed experimental
replicate means. No response-based filtering or approximate condition matching.
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'work/source_copies'
OUT = ROOT / 'work/selection_cells'
UNIT_CORRECTIONS = OUT / 'shin2021_competitive_ibu_unit_harmonized.csv'

SOURCES = {
    'shin2021_alkaline': ('Shin2021_Alkaline', '10.1016/j.envres.2021.111346', 'Dataset II'),
    'shin2021_magnetic': ('Shin2021_Magnetic', '10.1016/j.jece.2021.105119', 'Dataset II'),
    'shin2022_postmodification': ('Shin2022_Postmodification', '10.1016/j.jhazmat.2022.129081', 'Dataset II'),
    'shin2020_micropollutants': ('Shin2020_Micropollutants', '10.1016/j.jhazmat.2020.123102', 'Dataset III'),
    'shin2021_competitive': ('Shin2021_Competitive', '10.1016/j.envpol.2020.116244', 'Dataset III'),
    'shin2022_ibuprofen': ('Shin2022_Ibuprofen', '10.1016/j.jece.2022.107914', 'Dataset III'),
}
CONDITIONS = ['C0_mg_L', 'dose_g_L', 'contact_time_h', 'T', 'pH_solution', 'ionic_strength_M', 'DOM_mg_L', 'matrix', 'adsorption_mode', 'co_solute_set']
FEATURES = ['C', 'H', 'N', 'O', 'Ash', 'SA']


def build_source(stem, source, doi, dataset):
    if stem == 'shin2021_competitive':
        d = pd.read_csv(UNIT_CORRECTIONS)
    else:
        d = pd.read_csv(BASE / f'{stem}_analysis_copy_candidate.csv')
    p = pd.DataFrame(index=d.index)
    p['source_study_id'], p['doi'], p['dataset'] = source, doi, dataset
    p['source_row_id'] = d.source_row_id.astype(int)
    p['record_id'] = dataset + ':' + p.source_row_id.astype(str)
    source_path = BASE / f'{stem}_analysis_copy_candidate.csv'
    p['source_file'] = str(source_path.relative_to(ROOT)) if source_path.is_relative_to(ROOT) else source_path.name
    p['material_group'] = source + '::' + d.analysis_material_label
    p['series'] = d.ordered_source_series if stem == 'shin2021_competitive' else d.analysis_series
    p['endpoint_annotation'] = d.analysis_endpoint
    p['pollutant'] = d.analysis_task.replace({'IBF': 'IBU', 'Sr(II)': 'Sr (II)'})
    for name in ['C', 'H', 'N', 'O', 'Ash']:
        p[name] = d[name]
    p['SA'] = d['Surface area']
    p['contact_time_h'] = d.analysis_time_min / 60
    p['time_raw_min'] = d['Adsorption_time (min)'] if dataset == 'Dataset II' else d['Adsorption time']
    p['material_label_raw'] = d.Adsorbent
    p['transformation_log'] = d.analysis_transformations
    p['provenance_tier'] = 'author_released_experimental_data_protocol_harmonized'
    if dataset == 'Dataset II':
        p['C0_mg_L'], p['response_mg_g'] = d.Ci, d.qe
        p['response_raw'], p['concentration_raw'] = d.qe, d.Ci
        p['dose_g_L'] = d['Dosage(g/L)']
        p['T'], p['pH_solution'] = d.adsorption_temp, d['solution pH']
        p['ionic_strength_M'], p['DOM_mg_L'] = d.analysis_ionic_strength_M, d.DOM
        p['matrix'] = 'laboratory_solution_matrix_not_recorded'
        p['adsorption_mode'], p['co_solute_set'] = 'Single', 'none_reported'
        p['source_response_unit'], p['source_concentration_unit'] = 'mg/g', 'mg/L'
    else:
        p['C0_mg_L'], p['response_mg_g'] = d['Initial concentration'], d.Capacity
        p['response_raw'], p['concentration_raw'] = d.Capacity, d['Initial concentration']
        p['dose_g_L'], p['T'] = d['Adsorbent dosage'], d['Adsorption temperature']
        p['pH_solution'], p['ionic_strength_M'] = d['Solution pH'], d['Ion concentration']
        p['DOM_mg_L'], p['matrix'] = d['Humic acid'], d['Wastewater type']
        mode = d.analysis_adsorption_type if 'analysis_adsorption_type' in d else d['Adsorption type']
        p['adsorption_mode'] = mode.replace({'Competative': 'Competitive'})
        p['co_solute_set'] = 'none_reported'
        p['source_response_unit'], p['source_concentration_unit'] = 'mg/g', 'mg/L'
        competitive = p.adsorption_mode.eq('Competitive')
        if stem == 'shin2020_micropollutants':
            p.loc[competitive & p.pollutant.eq('IBU'), 'co_solute_set'] = 'CBZ+EE2'
            p.loc[competitive & p.pollutant.eq('CBZ'), 'co_solute_set'] = 'IBU+EE2'
            p.loc[competitive & p.pollutant.eq('EE2'), 'co_solute_set'] = 'IBU+CBZ'
        if stem == 'shin2021_competitive':
            p['C0_mg_L'], p['response_mg_g'] = d.analysis_initial_concentration_mg_L, d.analysis_capacity_mg_g
            p['source_response_unit'], p['source_concentration_unit'] = 'umol/g', 'umol/L'
            p['co_solute_set'] = 'NPX+DCF'
            p['transformation_log'] += '; concentration and capacity multiplied by 0.20628'
    p['negative_response_flag'] = p.response_mg_g < 0
    p['nominal_mass_balance_excess_mg_g'] = np.maximum(p.response_mg_g - p.C0_mg_L / p.dose_g_L, 0)
    p['row_disposition'] = 'admit_to_public_data_analysis'
    p['disposition_reason'] = 'Public experimental release; source-supported units, material labels and recorded conditions; independent replicate identities not asserted.'
    if dataset == 'Dataset II':
        lower, upper = (3, 11) if stem == 'shin2022_postmodification' else (3, 9)
        bad = ~p.pH_solution.between(lower, upper)
        p.loc[bad, 'row_disposition'] = 'exclude_unresolved_input_conflict'
        p.loc[bad, 'disposition_reason'] = 'Recorded pH falls outside the source protocol; no inferred replacement.'
    outside = ~p.pollutant.isin(['Sr (II)', 'IBU', 'CBZ'])
    p.loc[outside, 'row_disposition'] = 'outside_current_task_scope'
    p.loc[outside, 'disposition_reason'] = 'Retained in raw release but not a current Sr/IBU/CBZ task.'
    if p[FEATURES + CONDITIONS + ['response_mg_g']].isna().any().any():
        raise RuntimeError(f'Missing fields in {source}')
    return p


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError('Output must be empty; do not overwrite archived analysis files')
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.concat([build_source(stem, *spec) for stem, spec in SOURCES.items()], ignore_index=True)
    assert not data.record_id.duplicated().any()
    data.to_csv(OUT / 'public_source_rows.csv', index=False)
    admitted = data[data.row_disposition.eq('admit_to_public_data_analysis')]
    keys = ['dataset', 'source_study_id', 'pollutant', 'material_group', 'series', *CONDITIONS]
    cells = []
    for group_key, g in admitted.groupby(keys, sort=True, dropna=False):
        if (g[FEATURES].nunique() != 1).any():
            raise RuntimeError('Material descriptors vary within an exact candidate-condition cell')
        cells.append({
            **dict(zip(keys, group_key)), **g[FEATURES].iloc[0].to_dict(),
            'response_mg_g': g.response_mg_g.mean(),
            'recorded_min_mg_g': g.response_mg_g.min(), 'recorded_max_mg_g': g.response_mg_g.max(),
            'recorded_sd_mg_g': g.response_mg_g.std(ddof=1) if len(g) > 1 else np.nan,
            'n_compilation_rows': len(g), 'raw_record_ids_json': json.dumps(g.record_id.tolist()),
            'negative_response_rows': int(g.negative_response_flag.sum()),
            'aggregation': 'mean of reported rows sharing exact recorded condition and series; not verified replicate mean',
        })
    cells = pd.DataFrame(cells)
    cells.to_csv(OUT / 'public_source_cells.csv', index=False)
    support = []
    for (source, task, series), g in cells.groupby(['source_study_id', 'pollutant', 'series']):
        candidates = set(g.material_group)
        for condition, h in g.groupby(CONDITIONS, sort=True, dropna=False):
            support.append({
                'source_study_id': source, 'pollutant': task, 'series': series,
                **dict(zip(CONDITIONS, condition)),
                'candidates_in_series': len(candidates), 'candidates_present': h.material_group.nunique(),
                'complete_common_condition': set(h.material_group) == candidates and len(candidates) >= 2,
                'candidate_count_policy': 'two-candidate cells are paired comparisons, not multi-candidate panels',
            })
    support = pd.DataFrame(support)
    support.to_csv(OUT / 'public_source_condition_support.csv', index=False)
    summary = data.groupby(['dataset', 'source_study_id', 'pollutant', 'row_disposition']).size().rename('rows').reset_index()
    summary.to_csv(OUT / 'public_source_admission_summary.csv', index=False)
    print(summary.to_string(index=False))
    print('Exact cells:', len(cells))
    print(support.groupby(['source_study_id', 'pollutant']).complete_common_condition.agg(['size', 'sum']).to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-copies', type=Path, required=True)
    parser.add_argument('--competitive-unit-copy', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    BASE = args.source_copies.resolve()
    UNIT_CORRECTIONS = args.competitive_unit_copy.resolve()
    OUT = args.out.resolve()
    main()
