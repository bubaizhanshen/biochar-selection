import json
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from panel_input import load_panel_input, validate_split


class PanelTrainingMembershipTests(unittest.TestCase):
    def setUp(self):
        self.task = pd.DataFrame({
            "source_table_row_id": [10, 11, 20, 21, 30, 40],
            "material_group": ["A", "A", "B", "B", "C", "D"],
            "source_study_id": ["s1", "s1", "s2", "s2", "s3", "s1"],
        })
        self.row = pd.Series({
            "dataset": "Dataset I", "contaminant": "Cd (II)",
            "input_contract": "public_compilation", "holdout_unit": "study_block",
            "candidate_materials_json": '["A"]',
            "train_source_row_ids_json": '[20, 21, 30]',
            "test_source_row_ids_json": '[10, 11]',
            "n_train_rows": 3, "n_candidate_rows": 2,
            "n_train_materials": 2, "n_candidate_materials": 1,
        })

    def test_only_listed_training_rows_are_used(self):
        train, test = validate_split(self.task, self.row)
        self.assertEqual(self.task.iloc[train].source_table_row_id.tolist(), [20, 21, 30])
        self.assertEqual(self.task.iloc[test].source_table_row_id.tolist(), [10, 11])

    def test_missing_input_contract_never_loads_old_task(self):
        with patch("panel_input.load_task") as loader:
            with self.assertRaisesRegex(RuntimeError, "explicit input_contract"):
                load_panel_input(self.row.drop("input_contract"), Path("manifest.csv"))
            loader.assert_not_called()

    def test_missing_training_list_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "train_source_row_ids_json"):
            validate_split(self.task, self.row.drop("train_source_row_ids_json"))

    def test_partial_candidate_test_is_rejected(self):
        self.row["test_source_row_ids_json"] = '[10]'
        with self.assertRaisesRegex(RuntimeError, "all admitted rows"):
            validate_split(self.task, self.row)

    def test_candidate_row_cannot_enter_training(self):
        self.row["train_source_row_ids_json"] = '[10, 20, 21, 30]'
        with self.assertRaisesRegex(RuntimeError, "overlap"):
            validate_split(self.task, self.row)

    def test_same_source_other_material_cannot_leak(self):
        self.row["train_source_row_ids_json"] = '[20, 21, 30, 40]'
        with self.assertRaisesRegex(RuntimeError, "study block"):
            validate_split(self.task, self.row)

    def test_declared_source_family_cannot_enter_training(self):
        self.row['source_family_exclusions_json'] = '["s1", "s2"]'
        with self.assertRaisesRegex(RuntimeError, "source family"):
            validate_split(self.task, self.row)

    def test_malformed_source_family_is_rejected(self):
        self.row['source_family_exclusions_json'] = '"s2"'
        with self.assertRaisesRegex(RuntimeError, "JSON string list"):
            validate_split(self.task, self.row)

    def test_missing_source_row_is_rejected(self):
        self.row["train_source_row_ids_json"] = '[20, 21, 999]'
        with self.assertRaisesRegex(RuntimeError, "absent"):
            validate_split(self.task, self.row)

    def test_stale_count_is_rejected(self):
        self.row["n_train_rows"] = 330
        with self.assertRaisesRegex(RuntimeError, "n_train_rows"):
            validate_split(self.task, self.row)

    def test_no_training_data_does_not_fall_back(self):
        self.row["train_source_row_ids_json"] = '[]'
        with self.assertRaisesRegex(RuntimeError, "nonempty"):
            validate_split(self.task, self.row)

    def test_audited_copy_path_and_roles_are_forwarded(self):
        self.row["input_contract"] = "source_audited_analysis_copy_v1"
        self.row["analysis_copy_path"] = "input.csv"
        self.row["analysis_roles_json"] = json.dumps(["primary"])
        with patch("panel_input.load_task", return_value=(self.task, ["feature"])) as loader:
            load_panel_input(self.row, Path("data/manifest.csv"))
        loader.assert_called_once_with("Dataset I", "Cd (II)", Path("data/input.csv"), {"primary"})


if __name__ == "__main__":
    unittest.main()
