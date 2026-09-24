import unittest

import pandas as pd

from run_asv_source_selection import (
    CONDITION_FEATURES,
    FULL_MATERIAL_FEATURES,
    panel_support,
    validate_data,
)


class AsvSourceSelectionTest(unittest.TestCase):
    def make_frame(self, conditions=((5.0, ("a", "b", "c")),)):
        rows = []
        for pH, materials in conditions:
            for index, material in enumerate(materials):
                row = {
                    "reference": "test",
                    "material_group": material,
                    "capacity_mg_g": float(index + 1),
                    "As_initial_mg_L": 10.0,
                    "temperature_C": 25.0,
                    "pH": pH,
                    "dose_g_L": 2.0,
                }
                row.update({feature: float(index + 1) for feature in FULL_MATERIAL_FEATURES})
                rows.append(row)
        return pd.DataFrame(rows)

    def config(self):
        return {
            "source_column": "reference",
            "material_column": "material_group",
            "response_column": "capacity_mg_g",
            "test_sources": ["test"],
            "training_only_sources": [],
        }

    def test_panel_support_counts_conditions_with_three_biochars(self):
        frame = self.make_frame(
            ((5.0, ("a", "b", "c")), (7.0, ("a", "b", "c")), (9.0, ("a", "b")))
        )

        support = panel_support(frame, "test", self.config())

        self.assertEqual(support["n_panel_biochars"], 3)
        self.assertEqual(support["n_matched_conditions"], 3)
        self.assertEqual(support["n_conditions_with_at_least_3_biochars"], 2)
        self.assertFalse(support["three_biochar_three_condition_subset"])

    def test_complete_three_by_three_panel_enters_descriptive_subset(self):
        frame = self.make_frame(
            ((5.0, ("a", "b", "c")), (7.0, ("a", "b", "c")), (9.0, ("a", "b", "c")))
        )

        support = panel_support(frame, "test", self.config())

        self.assertTrue(support["three_biochar_three_condition_subset"])

    def test_validation_rejects_duplicate_material_condition_rows(self):
        frame = self.make_frame().iloc[[0, 0]].copy()

        with self.assertRaisesRegex(ValueError, "Repeated material-condition"):
            validate_data(frame, self.config())

    def test_validation_rejects_unresolved_missing_values(self):
        frame = self.make_frame()
        frame.loc[0, CONDITION_FEATURES[0]] = float("nan")

        with self.assertRaisesRegex(ValueError, "Missing required values"):
            validate_data(frame, self.config())


if __name__ == "__main__":
    unittest.main()
