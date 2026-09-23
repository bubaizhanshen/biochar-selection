import unittest

import pandas as pd

from audit_response_bounds import audit


class ResponseBoundsAuditTest(unittest.TestCase):
    def inputs(self):
        cells = pd.DataFrame([
            dict(source_table_row_id=1, source_study_id="s1", pollutant="IBU", C0_mg_L=10.0,
                 dose_g_L=1.0, response_mg_g=3.0),
            dict(source_table_row_id=2, source_study_id="s1", pollutant="IBU", C0_mg_L=10.0,
                 dose_g_L=1.0, response_mg_g=4.0),
            dict(source_table_row_id=3, source_study_id="s2", pollutant="IBU", C0_mg_L=5.0,
                 dose_g_L=1.0, response_mg_g=-0.1),
        ])
        rows = []
        for strategy, values in (("ridge", (12.0, -1.0, 3.0)), ("random_forest", (5.0, 5.0, 4.0))):
            for row_id, predicted in enumerate(values, 1):
                rows.append(dict(stage="outer", strategy=strategy, row_id=row_id,
                                 observed=cells.loc[row_id - 1, "response_mg_g"], predicted=predicted))
        rows.append(dict(stage="outer", strategy="surface_area", row_id=1, observed=3.0, predicted=10.0))
        rows.append(dict(stage="inner", strategy="ridge", row_id=1, observed=3.0, predicted=99.0))
        return pd.DataFrame(rows), cells

    def test_model_and_observation_violations_are_separate(self):
        by_model, by_source, observed = audit(*self.inputs())
        ridge = by_model.set_index("strategy").loc["ridge"]
        self.assertEqual(int(ridge.prediction_cells), 3)
        self.assertEqual(int(ridge.above_mass_balance_count), 1)
        self.assertEqual(int(ridge.below_zero_count), 1)
        self.assertEqual(int(ridge.outside_physical_range_count), 2)
        self.assertAlmostEqual(ridge.outside_physical_range_fraction, 2 / 3)
        self.assertEqual(int(by_model.set_index("strategy").loc["random_forest", "outside_physical_range_count"]), 0)
        self.assertEqual(int(by_source.loc[by_source.source_study_id.eq("s1") &
                                           by_source.strategy.eq("ridge"), "above_mass_balance_count"].iat[0]), 1)
        self.assertEqual(int(observed.observed_below_zero.sum()), 1)
        self.assertEqual(int(observed.observed_above_limit.sum()), 0)

    def test_invalid_dose_is_rejected(self):
        predictions, cells = self.inputs()
        cells.loc[0, "dose_g_L"] = 0
        with self.assertRaisesRegex(ValueError, "Nonpositive"):
            audit(predictions, cells)

    def test_unmatched_prediction_is_rejected(self):
        predictions, cells = self.inputs()
        predictions.loc[0, "row_id"] = 99
        with self.assertRaisesRegex(ValueError, "do not match"):
            audit(predictions, cells)


if __name__ == "__main__":
    unittest.main()
