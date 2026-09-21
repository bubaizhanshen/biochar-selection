"""Resolve candidate-panel input and enforce explicit train/test membership."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

def load_task(*args, **kwargs):
    """Load historical inputs only when that separate interface is requested."""
    from run_biochar_holdout import load_task as historical_load_task

    return historical_load_task(*args, **kwargs)


def _ids(row: pd.Series, column: str) -> list[int]:
    try:
        values = json.loads(str(row[column]))
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"Panel manifest requires {column} as a JSON integer list.") from exc
    if not isinstance(values, list) or any(type(v) is not int for v in values):
        raise RuntimeError(f"{column} must contain integer row IDs.")
    if len(values) != len(set(values)):
        raise RuntimeError(f"{column} contains duplicate row IDs.")
    return values


def validate_split(task: pd.DataFrame, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Use source-table IDs; task-local IDs can change after admission filtering."""
    if task["source_table_row_id"].duplicated().any():
        raise RuntimeError("Panel input has duplicate source-table row IDs.")
    train_ids = set(_ids(row, "train_source_row_ids_json"))
    test_ids = set(_ids(row, "test_source_row_ids_json"))
    if not train_ids or not test_ids:
        raise RuntimeError("Panel requires nonempty explicit train and test row lists.")
    if train_ids & test_ids:
        raise RuntimeError("Panel train and test row IDs overlap.")
    available = set(task["source_table_row_id"].astype(int))
    if (train_ids | test_ids) - available:
        raise RuntimeError("Manifest row IDs are absent from the selected input.")
    candidates = set(json.loads(str(row["candidate_materials_json"])))
    candidate_rows = task[task["material_group"].isin(candidates)]
    if set(candidate_rows["material_group"]) != candidates:
        raise RuntimeError("Candidate materials are absent from the selected input.")
    if set(candidate_rows["source_table_row_id"].astype(int)) != test_ids:
        raise RuntimeError("Test rows must cover all admitted rows of every candidate material.")
    train_index = np.flatnonzero(task["source_table_row_id"].isin(train_ids).to_numpy())
    test_index = np.flatnonzero(task["source_table_row_id"].isin(test_ids).to_numpy())
    train = task.iloc[train_index]
    test = task.iloc[test_index]
    if set(train["material_group"]) & candidates:
        raise RuntimeError("Candidate material occurs in the training input.")
    holdout_unit = str(row.get("holdout_unit", ""))
    if holdout_unit not in {"candidate_material", "study_block"}:
        raise RuntimeError("Manifest must declare candidate_material or study_block holdout.")
    if holdout_unit == "study_block" and set(train["source_study_id"]) & set(test["source_study_id"]):
        raise RuntimeError("Held-out study block occurs in the training input.")
    if "source_family_exclusions_json" in row and pd.notna(row["source_family_exclusions_json"]):
        try:
            family = json.loads(str(row["source_family_exclusions_json"]))
        except ValueError as exc:
            raise RuntimeError("Source-family exclusions must be a JSON string list.") from exc
        if not isinstance(family, list) or not all(isinstance(v, str) and v for v in family):
            raise RuntimeError("Source-family exclusions must be a JSON string list.")
        if set(train["source_study_id"]) & set(family):
            raise RuntimeError("Excluded source family occurs in the training input.")
    expected = {
        "n_train_rows": len(train),
        "n_candidate_rows": len(test),
        "n_train_materials": train["material_group"].nunique(),
        "n_candidate_materials": len(candidates),
    }
    for name, actual in expected.items():
        if name not in row or pd.isna(row[name]) or float(row[name]) != actual:
            raise RuntimeError(f"Manifest {name} does not match selected input ({actual}).")
    return train_index, test_index


def load_panel_input(
    row: pd.Series, manifest_path: Path, *, allow_legacy_input: bool = False
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, str]:
    contract = str(row.get("input_contract", ""))
    dataset, contaminant = str(row["dataset"]), str(row["contaminant"])
    if contract == "source_audited_analysis_copy_v1":
        if pd.isna(row.get("analysis_copy_path")) or not str(row.get("analysis_copy_path", "")).strip():
            raise RuntimeError("Audited panel manifest requires an analysis-copy path.")
        path = Path(str(row["analysis_copy_path"]))
        if not path.is_absolute():
            path = manifest_path.parent / path
        try:
            roles = json.loads(str(row["analysis_roles_json"]))
        except (KeyError, ValueError) as exc:
            raise RuntimeError("Audited panel manifest requires explicit analysis roles.") from exc
        if not isinstance(roles, list) or not roles or not all(isinstance(v, str) and v for v in roles):
            raise RuntimeError("Analysis roles must be a nonempty JSON string list.")
        task, features = load_task(dataset, contaminant, path, set(roles))
    elif contract == "public_compilation":
        task, features = load_task(dataset, contaminant)
    elif allow_legacy_input and contract in {"", "nan"}:
        task, features = load_task(dataset, contaminant)
        mask = task["material_group"].isin(json.loads(str(row["candidate_materials_json"])))
        return task, features, np.flatnonzero(~mask.to_numpy()), np.flatnonzero(mask.to_numpy()), "legacy_public_compilation"
    else:
        raise RuntimeError(
            "Panel manifest requires an explicit input_contract and train/test row lists. "
            "Use --allow-legacy-input only to reproduce historical unrestricted runs."
        )
    train, test = validate_split(task, row)
    return task, features, train, test, contract


def split_records(task: pd.DataFrame, train: np.ndarray, test: np.ndarray, panel_id: int) -> pd.DataFrame:
    columns = ["source_table_row_id", "task_row_id", "material_group", "source_study_id"]
    frames = []
    for role, indices in [("train", train), ("test", test)]:
        part = task.iloc[indices][columns].copy()
        part.insert(0, "split", role)
        part.insert(0, "panel_id", panel_id)
        frames.append(part)
    return pd.concat(frames, ignore_index=True)
