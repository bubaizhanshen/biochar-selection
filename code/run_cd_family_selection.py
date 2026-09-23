"""Retrospective family-blocked Cd strategy selection with fixed model settings."""

from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


from expand_model_protocol import expand
from run_selection_benchmark import estimator, evaluate_conditions


FAMILY = {
    "Cui2016_Canna": "Cui2016",
    "Cui2016_Wetland": "Cui2016",
    "Gao2019": "Gao2019",
    "Wang2021_invasive": "Wang2021",
    "Zama2017": "Zama2017",
}
TEST_FAMILIES = ("Cui2016", "Gao2019", "Zama2017")
MODEL_SETS = {
    "three_models": ("ridge", "svr", "random_forest"),
    "nine_models": (
        "ridge", "svr", "random_forest", "extra_trees", "gradient_boosting",
        "xgboost", "lightgbm", "catboost", "knn",
    ),
}
RULES = ("surface_area", "biochar_pH", "random")
ALL_STRATEGIES = (*RULES, *MODEL_SETS["nine_models"])


def predict(strategy: str, fit: pd.DataFrame, target: pd.DataFrame, config: dict) -> np.ndarray:
    if strategy == "surface_area":
        return target.SA.to_numpy(float)
    if strategy == "biochar_pH":
        return target.pH_biochar.to_numpy(float)
    if strategy == "random":
        return np.zeros(len(target), dtype=float)
    features = [*config["material_features"], *config["condition_features"],
                *config["categorical_condition_features"]]
    model = estimator(strategy, True, config)
    model.fit(fit[features], fit.response_mg_g)
    result = model.predict(target[features])
    if not np.isfinite(result).all():
        raise ValueError(f"Nonfinite {strategy} prediction")
    return result


def score_family(
    family: str, strategy: str, target: pd.DataFrame, pred: np.ndarray,
    stage: str, tier: str, outer: str, inner: str,
) -> tuple[dict, list[dict], list[dict]]:
    source_scores = []
    condition_rows = []
    prediction_rows = []
    target = target.copy()
    target["prediction"] = pred
    for source, panel in target.groupby("source_study_id", sort=True):
        candidates = set(panel.material_group)
        metrics, coverage = evaluate_conditions(panel, panel.prediction.to_numpy(), candidates, CONFIG)
        if not metrics:
            raise ValueError(f"No complete condition for {stage}, {source}")
        frame = pd.DataFrame(metrics)
        by_series = frame.groupby("series", sort=True)[["selection_loss", "mae"]].mean()
        source_scores.append({
            "source": source,
            "loss": float(by_series.selection_loss.mean()),
            "mae": float(by_series.mae.mean()) if strategy not in RULES else np.nan,
            "complete_conditions": len(metrics),
            "incomplete_conditions": sum(row["status"] != "complete" for row in coverage),
        })
        for metric in metrics:
            if strategy in RULES:
                for diagnostic in ("mae", "mse", "common_bias_squared", "relative_error_mse"):
                    metric[diagnostic] = np.nan
            condition_rows.append({
                "tier": tier, "stage": stage, "outer_family": outer,
                "inner_family": inner, "family": family, "source": source,
                "strategy": strategy, **metric,
            })
        for _, row in panel.iterrows():
            prediction_rows.append({
                "tier": tier, "stage": stage, "outer_family": outer,
                "inner_family": inner, "family": family, "source": source,
                "strategy": strategy, "row_id": row.source_table_row_id,
                "material": row.material_group, "observed": row.response_mg_g,
                "predicted_or_rule_score": row.prediction,
                "above_mass_balance_bound": (
                    bool(row.prediction > row.C0_mg_L / row.dose_g_L + 1e-9)
                    if strategy not in RULES else False
                ),
            })
    result = {
        "tier": tier, "stage": stage, "outer_family": outer,
        "inner_family": inner, "family": family, "strategy": strategy,
        "loss": float(np.mean([s["loss"] for s in source_scores])),
        "mae": float(np.mean([s["mae"] for s in source_scores])) if strategy not in RULES else np.nan,
        "n_sources": len(source_scores),
        "complete_conditions": sum(s["complete_conditions"] for s in source_scores),
        "incomplete_conditions": sum(s["incomplete_conditions"] for s in source_scores),
    }
    return result, condition_rows, prediction_rows


def choose(scores: dict[str, dict], objective: str, strategies: tuple[str, ...]) -> str:
    return min(strategies, key=lambda name: (scores[name][objective], strategies.index(name)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Prepared metal_input.csv")
    parser.add_argument("--protocol", type=Path, required=True, help="Prepared metal_protocol.json")
    parser.add_argument("--extension", type=Path, default=Path(__file__).resolve().parents[1] / "config/model_expansion.json")
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Output directory already exists: {args.out}")
    base_path = args.protocol
    input_path = args.data
    CONFIG.update(expand(json.loads(base_path.read_text()), json.loads(args.extension.read_text())))
    if tuple(CONFIG["models"]) != MODEL_SETS["nine_models"]:
        raise ValueError("Model order differs from the existing nine-model protocol")
    data = pd.read_csv(input_path)
    data = data[data.pollutant.eq("Cd (II)")].copy()
    data["family"] = data.source_study_id.map(FAMILY)
    if data.family.isna().any() or len(data) != 154:
        raise ValueError("Unexpected Cd source inventory")
    if data.source_table_row_id.duplicated().any():
        raise ValueError("Nonunique source-table row IDs")
    needed = [*CONFIG["material_features"], *CONFIG["condition_features"],
              *CONFIG["categorical_condition_features"], "response_mg_g"]
    if data[needed].isna().any().any():
        raise ValueError("Missing model input or response")
    all_scores, all_conditions, all_predictions, decisions, split_rows = [], [], [], [], []
    for tier, allowed in [
        ("verified_compilation", {"verified_compilation"}),
        ("expanded_compilation", {"verified_compilation", "local_compilation_condition_supported"}),
    ]:
        permitted = data[data.provenance_tier.isin(allowed)].copy()
        for outer in TEST_FAMILIES:
            train = permitted[permitted.family.ne(outer) & permitted.train_eligible.eq(True)].copy()
            test = data[data.family.eq(outer) & data.test_eligible.eq(True)].copy()
            if train.empty or test.empty or set(train.family) & set(test.family):
                raise ValueError("Invalid outer family separation")
            if set(train.source_table_row_id) & set(test.source_table_row_id):
                raise ValueError("Outer row overlap")
            inner_families = tuple(f for f in TEST_FAMILIES if f != outer)
            for row in train.itertuples():
                split_rows.append({"tier": tier, "outer_family": outer, "role": "train",
                                   "source": row.source_study_id, "family": row.family,
                                   "row_id": row.source_table_row_id})
            for row in test.itertuples():
                split_rows.append({"tier": tier, "outer_family": outer, "role": "test",
                                   "source": row.source_study_id, "family": row.family,
                                   "row_id": row.source_table_row_id})
            inner_by_strategy = {name: [] for name in ALL_STRATEGIES}
            for inner in inner_families:
                fit = train[train.family.ne(inner)]
                validation = train[train.family.eq(inner)]
                if fit.empty or validation.empty or inner in set(fit.family):
                    raise ValueError("Invalid inner family separation")
                for strategy in ALL_STRATEGIES:
                    pred = predict(strategy, fit, validation, CONFIG)
                    score, conditions, predictions = score_family(
                        inner, strategy, validation, pred, "inner", tier, outer, inner
                    )
                    score["fit_families"] = ";".join(sorted(fit.family.unique()))
                    score["fit_rows"] = len(fit)
                    all_scores.append(score)
                    all_conditions.extend(conditions)
                    all_predictions.extend(predictions)
                    inner_by_strategy[strategy].append(score)
            aggregate = {
                name: {
                    "loss": float(np.mean([row["loss"] for row in rows])),
                    "mae": float(np.mean([row["mae"] for row in rows])) if name not in RULES else np.nan,
                }
                for name, rows in inner_by_strategy.items()
            }
            selections = {}
            for set_name, models in MODEL_SETS.items():
                selections[f"{set_name}_by_mae"] = choose(aggregate, "mae", models)
                selections[f"{set_name}_by_loss"] = choose(aggregate, "loss", models)
                selections[f"{set_name}_with_rules_by_loss"] = choose(
                    aggregate, "loss", (*RULES, *models)
                )
            outer_scores = {}
            for strategy in ALL_STRATEGIES:
                pred = predict(strategy, train, test, CONFIG)
                score, conditions, predictions = score_family(
                    outer, strategy, test, pred, "outer", tier, outer, ""
                )
                score["fit_families"] = ";".join(sorted(train.family.unique()))
                score["fit_rows"] = len(train)
                outer_scores[strategy] = score
                all_scores.append(score)
                all_conditions.extend(conditions)
                all_predictions.extend(predictions)
            for selection_name, strategy in selections.items():
                decisions.append({
                    "tier": tier, "outer_family": outer, "selector": selection_name,
                    "selected_strategy": strategy,
                    "inner_mae": aggregate[strategy]["mae"],
                    "inner_loss": aggregate[strategy]["loss"],
                    "outer_loss": outer_scores[strategy]["loss"],
                    "outer_mae": outer_scores[strategy]["mae"],
                    "complete_conditions": outer_scores[strategy]["complete_conditions"],
                    "test_sources": outer_scores[strategy]["n_sources"],
                })
            print(tier, outer, selections, flush=True)
    outputs = {
        "family_scores.csv": all_scores,
        "condition_scores.csv": all_conditions,
        "predictions.csv": all_predictions,
        "decisions.csv": decisions,
        "outer_splits.csv": split_rows,
    }
    args.out.mkdir(parents=True)
    for name, records in outputs.items():
        pd.DataFrame(records).to_csv(args.out / name, index=False)
    provenance = {
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "base_protocol_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "model_extension_sha256": hashlib.sha256(args.extension.read_bytes()).hexdigest(),
        "test_families": TEST_FAMILIES,
        "model_sets": MODEL_SETS,
    }
    (args.out / "run_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


CONFIG: dict = {}

if __name__ == "__main__":
    main()
