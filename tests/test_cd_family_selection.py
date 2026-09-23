import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from run_cd_family_selection import FAMILY, MODEL_SETS, choose, score_family


class CdFamilySelectionTests(unittest.TestCase):
    def test_related_cui_records_share_one_outer_family(self):
        self.assertEqual(FAMILY["Cui2016_Canna"], FAMILY["Cui2016_Wetland"])
        self.assertEqual(len(MODEL_SETS["nine_models"]), 9)

    def test_exact_validation_tie_uses_declared_order(self):
        scores = {"ridge": {"loss": 2.0}, "svr": {"loss": 2.0}}
        self.assertEqual(choose(scores, "loss", ("svr", "ridge")), "svr")
        self.assertEqual(choose(scores, "loss", ("ridge", "svr")), "ridge")

    def test_source_and_series_are_equally_weighted(self):
        target = pd.DataFrame({
            "source_study_id": ["Canna"] * 3 + ["Wetland"],
            "material_group": ["a", "a", "a", "b"],
            "source_table_row_id": ["a1", "a2", "a3", "b1"],
            "response_mg_g": [1.0] * 4,
            "C0_mg_L": [10.0] * 4,
            "dose_g_L": [1.0] * 4,
        })

        def fake_evaluation(panel, predictions, candidates, config):
            if panel.source_study_id.iloc[0] == "Canna":
                return ([
                    {"series": "pH", "selection_loss": 0.0, "mae": 0.0},
                    {"series": "pH", "selection_loss": 2.0, "mae": 2.0},
                    {"series": "dose", "selection_loss": 5.0, "mae": 5.0},
                ], [])
            return ([{"series": "pH", "selection_loss": 9.0, "mae": 9.0}], [])

        with patch("run_cd_family_selection.evaluate_conditions", side_effect=fake_evaluation):
            score, conditions, predictions = score_family(
                "Cui2016", "ridge", target, np.zeros(4),
                "outer", "verified_compilation", "Cui2016", "",
            )
        # Canna: mean of pH mean 1 and dose 5 = 3; family mean of 3 and 9 = 6.
        self.assertEqual(score["loss"], 6.0)
        self.assertEqual(score["mae"], 6.0)
        self.assertEqual(score["n_sources"], 2)
        self.assertEqual(score["complete_conditions"], 4)
        self.assertEqual(len(conditions), 4)
        self.assertEqual(len(predictions), 4)


if __name__ == "__main__":
    unittest.main()
