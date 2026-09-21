import unittest
from unittest.mock import patch

import pandas as pd

from prepare_metal_selection_data import reconstruct, FEATURES


class MetalPreparationTest(unittest.TestCase):
    def fixture(self, source='Cui2016_Canna'):
        raw = pd.DataFrame([dict(Adsorbent='BC', HM='Cd2+', C0=2., Eta=.5,
                                T=25., pH_solution=5., **{k:1. for k in FEATURES})])
        annotations = pd.DataFrame([dict(source_table_row_id=0,
            source_study_id=source, material_group=source+'::BC',
            expected_material_label='BC', pollutant='Cd (II)', series='pH_series',
            dose_g_L=1., contact_time_h=24., matrix='recorded',
            provenance_tier='verified_compilation', train_eligible=True,
            test_eligible=True, metal_order=0, lead_order=float('nan'))])
        return raw, annotations

    def test_documented_concentration_and_response_factors(self):
        raw, annotations = self.fixture()
        with patch('prepare_metal_selection_data.pd.read_excel', return_value=raw):
            result = reconstruct('unused', annotations).iloc[0]
        self.assertAlmostEqual(result.C0_mg_L,2.*112.411)
        self.assertAlmostEqual(result.response_mg_g,.5*112.414)

    def test_wang_conversion_preserves_executed_policy(self):
        raw, annotations = self.fixture('Wang2021_invasive')
        annotations.loc[0,'dose_g_L']=.5
        with patch('prepare_metal_selection_data.pd.read_excel', return_value=raw):
            result = reconstruct('unused', annotations).iloc[0]
        self.assertAlmostEqual(result.C0_mg_L,112.414)

    def test_wrong_material_is_rejected(self):
        raw, annotations = self.fixture()
        raw.loc[0,'Adsorbent']='different'
        with patch('prepare_metal_selection_data.pd.read_excel', return_value=raw):
            with self.assertRaisesRegex(ValueError,'material label mismatch'):
                reconstruct('unused', annotations)

    def test_wrong_pollutant_is_rejected(self):
        raw, annotations = self.fixture()
        raw.loc[0,'HM']='Pb2+'
        with patch('prepare_metal_selection_data.pd.read_excel', return_value=raw):
            with self.assertRaisesRegex(ValueError,'pollutant mismatch'):
                reconstruct('unused', annotations)


if __name__ == '__main__':
    unittest.main()
