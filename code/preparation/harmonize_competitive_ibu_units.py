"""Retain source-unit values and add mass-unit fields for the public IBU block."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

IBU_MOLAR_MASS_G_MOL = 206.28


def main(source, source_copy, series_mapping, out):
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Output must be empty; preserve previous unit-conversion records')
    if source is not None:
        from lxml import etree
        root = etree.parse(str(source))
        para = root.xpath('//*[@id="p0080"]')[0]
        text = ' '.join(' '.join(para.itertext()).split())
        if 'μmol/g' not in text or 'μM' not in text:
            raise RuntimeError('Source unit evidence is absent')
    raw = pd.read_csv(source_copy)
    mapping = pd.read_csv(series_mapping)
    mapping = mapping[mapping.analysis_pollutant_label.eq('IBU')]
    data = raw.merge(mapping[['source_row_id', 'ordered_source_series', 'analysis_endpoint_candidate']], on='source_row_id', validate='one_to_one')
    assert len(data) == 309 and data.Pollutant.eq('IBU').all()
    factor = IBU_MOLAR_MASS_G_MOL / 1000.0
    data['source_concentration_unit'] = 'umol/L'
    data['source_capacity_unit'] = 'umol/g'
    data['analysis_initial_concentration_mg_L'] = data['Initial concentration'] * factor
    data['analysis_final_concentration_mg_L'] = data['Final concentration'] * factor
    data['analysis_capacity_mg_g'] = data.Capacity * factor
    data['unit_conversion_factor'] = factor
    data['unit_evidence'] = '10.1016/j.envpol.2020.116244 Methods p0080-p0090; Table 2 qe_exp unit'
    data['analysis_endpoint'] = data.analysis_endpoint_candidate
    data['analysis_adsorption_type'] = 'Competitive'
    reconstructed = (data.analysis_initial_concentration_mg_L - data.analysis_final_concentration_mg_L) / data['Adsorbent dosage']
    if not np.allclose(reconstructed, data.analysis_capacity_mg_g, rtol=1e-8, atol=1e-8):
        raise RuntimeError('Converted mass balance does not reproduce the public response')
    # Correct units alone do not verify replicate identities or missing co-solute fields.
    data['model_admission'] = 'unit_harmonized_pending_repeat_and_cosolute_review'
    out.mkdir(parents=True, exist_ok=True)
    data.to_csv(out / 'shin2021_competitive_ibu_unit_harmonized.csv', index=False)
    k = data[data.ordered_source_series.eq('kinetics') & data['Adsorption time'].eq(1440)]
    comparison = k.groupby(['analysis_material_label', 'Wastewater type']).agg(
        compilation_rows=('Capacity', 'size'),
        compilation_endpoint_mean_umol_g=('Capacity', 'mean'),
        compilation_endpoint_mean_mg_g=('analysis_capacity_mg_g', 'mean'),
    ).reset_index()
    reference = {
        ('Pristine SCW Biochar', 'Lake water'): 16.20,
        ('Pristine SCW Biochar', 'Secondary effluent'): 20.65,
        ('NaOH-activated SCW biochars', 'Lake water'): 80.02,
        ('NaOH-activated SCW biochars', 'Secondary effluent'): 61.25,
    }
    comparison['source_Table2_qe_exp_umol_g'] = [reference[key] for key in comparison[['analysis_material_label', 'Wastewater type']].itertuples(index=False, name=None)]
    comparison['difference_umol_g'] = comparison.compilation_endpoint_mean_umol_g - comparison.source_Table2_qe_exp_umol_g
    comparison.to_csv(out / 'shin2021_competitive_endpoint_scale_check.csv', index=False)
    print(comparison.to_string(index=False))
    print('Preserved 309 source-unit rows; added mg/L and mg/g conversion factor', factor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    evidence = parser.add_mutually_exclusive_group(required=True)
    evidence.add_argument('--source-xml', type=Path)
    evidence.add_argument('--use-reviewed-source-units', action='store_true',
                          help='Use documented umol/L and umol/g units from DOI 10.1016/j.envpol.2020.116244; no article XML is redistributed.')
    parser.add_argument('--source-copy', type=Path, required=True)
    parser.add_argument('--series-mapping', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    main(args.source_xml, args.source_copy, args.series_mapping, args.out)
