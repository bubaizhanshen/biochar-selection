"""Re-score saved inner and outer predictions under one nominal physical bound.

This is a post hoc sensitivity: no models are refit, and each outer strategy is
chosen using only projected inner validation predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_physical_projection import nominal_upper_bound, score_conditions, series_balanced


SELECTORS = {
    "fixed_models_mae": "mae",
    "fixed_models_selection_loss": "loss",
    "models_and_rules_selection_loss": "loss",
}


def score(frame: pd.DataFrame, predicted: np.ndarray, keys: list[str]) -> dict[str, float]:
    conditions = score_conditions(frame, predicted, keys)
    if conditions.empty:
        raise ValueError("No complete inner conditions")
    return {
        "loss": series_balanced(conditions, "selection_loss"),
        "mae": series_balanced(conditions, "mae"),
    }


def choose(scores: dict[str, dict[str, float]], strategy_order: list[str],
           selector: str) -> str:
    choices = (strategy_order if selector == "models_and_rules_selection_loss" else
               [name for name in strategy_order if name not in {"surface_area", "random"}])
    objective = SELECTORS[selector]
    return min(choices, key=lambda name: (scores[name][objective], choices.index(name)))


def calculate(mode: str, runs_root: Path, bounds_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    run = runs_root / mode
    data = pd.read_csv(run / "executed_input.csv")
    manifest = pd.read_csv(run / "executed_manifest.csv")
    predictions = pd.read_csv(run / "predictions.csv")
    trials = pd.read_csv(run / "inner_trials.csv")
    saved_decisions = pd.read_csv(run / "decisions.csv")
    bound_scores = pd.read_csv(bounds_dir / "fixed_strategy_by_source.csv")
    bound_scores = bound_scores.loc[bound_scores.training_representation.eq(mode)]
    metadata = json.loads((run / "protocol.json").read_text())
    config = metadata["model_config"]
    strategy_order = metadata["strategy_order"]
    keys = ["series", *config["condition_features"],
            *config["categorical_condition_features"]]
    inner = predictions.loc[predictions.stage.eq("inner")]
    if inner.duplicated(["panel_id", "strategy", "inner_source", "row_id"]).any():
        raise ValueError("Duplicate saved inner prediction")

    inner_rows = []
    selector_rows = []
    for panel in manifest.itertuples(index=False):
        task = data.loc[data.pollutant.eq(panel.contaminant)]
        sources = sorted(task.loc[task.source_study_id.ne(panel.test_source),
                                  "source_study_id"].unique())
        if len(sources) != 2:
            raise ValueError("Expected two inner validation sources")
        scores_raw, scores_projected = {}, {}
        for strategy in strategy_order:
            pairs = []
            for source in sources:
                val = task.loc[task.source_study_id.eq(source)].copy()
                saved = inner.loc[
                    inner.panel_id.eq(panel.panel_id)
                    & inner.strategy.eq(strategy)
                    & inner.inner_source.eq(source),
                    ["row_id", "observed", "predicted"],
                ]
                val = val.merge(saved, left_on="source_table_row_id", right_on="row_id",
                                validate="one_to_one")
                if len(val) != len(saved):
                    raise ValueError("Saved inner predictions lack source rows")
                if not np.allclose(val.response_mg_g, val.observed, rtol=0, atol=1e-8):
                    raise ValueError("Inner response mismatch")
                original = val.predicted.to_numpy(float)
                projected = (np.clip(original, 0, nominal_upper_bound(val))
                             if strategy not in {"surface_area", "random"}
                             else original.copy())
                raw_score = score(val, original, keys)
                projected_score = score(val, projected, keys)
                recorded = trials.loc[
                    trials.panel_id.eq(panel.panel_id)
                    & trials.strategy.eq(strategy)
                    & trials.inner_source.eq(source),
                ]
                if len(recorded) != 1:
                    raise ValueError("Missing saved inner trial")
                if not np.isclose(raw_score["loss"], recorded.iloc[0]["loss"], rtol=0,
                                  atol=1e-8):
                    raise ValueError("Original inner loss did not reproduce")
                if strategy not in {"surface_area", "random"} and not np.isclose(
                    raw_score["mae"], recorded.iloc[0]["mae"], rtol=0, atol=1e-8
                ):
                    raise ValueError("Original inner MAE did not reproduce")
                pairs.append((raw_score, projected_score))
                inner_rows.append({
                    "training_representation": mode,
                    "panel_id": panel.panel_id,
                    "validation_source": source,
                    "strategy": strategy,
                    "raw_loss_mg_g": raw_score["loss"],
                    "projected_loss_mg_g": projected_score["loss"],
                    "raw_mae_mg_g": raw_score["mae"] if strategy not in {"surface_area", "random"} else np.nan,
                    "projected_mae_mg_g": projected_score["mae"] if strategy not in {"surface_area", "random"} else np.nan,
                    "out_of_range_predictions": (int(((original < -1e-9) |
                        (original > nominal_upper_bound(val) + 1e-9)).sum())
                        if strategy not in {"surface_area", "random"} else 0),
                })
            scores_raw[strategy] = {key: float(np.mean([pair[0][key] for pair in pairs]))
                                    for key in ("loss", "mae")}
            scores_projected[strategy] = {
                key: float(np.mean([pair[1][key] for pair in pairs]))
                for key in ("loss", "mae")
            }

        for selector in SELECTORS:
            raw_choice = choose(scores_raw, strategy_order, selector)
            projected_choice = choose(scores_projected, strategy_order, selector)
            saved = saved_decisions.loc[
                saved_decisions.panel_id.eq(panel.panel_id)
                & saved_decisions.selector.eq(selector)
            ]
            if len(saved) != 1 or raw_choice != saved.iloc[0].selected_strategy:
                raise ValueError("Original training-side choice did not reproduce")
            outer = bound_scores.loc[
                bound_scores.panel_id.eq(panel.panel_id)
                & bound_scores.strategy.eq(projected_choice)
            ]
            if len(outer) != 1:
                raise ValueError("Projected outer score missing")
            old_outer = bound_scores.loc[
                bound_scores.panel_id.eq(panel.panel_id)
                & bound_scores.strategy.eq(raw_choice)
            ]
            if len(old_outer) != 1 or not np.isclose(
                old_outer.iloc[0].raw_loss_mg_g, saved.iloc[0].outer_loss,
                rtol=0, atol=1e-8,
            ):
                raise ValueError("Original outer loss did not reproduce")
            selector_rows.append({
                "training_representation": mode,
                "panel_id": panel.panel_id,
                "source": panel.test_source,
                "pollutant": panel.contaminant,
                "selector": selector,
                "raw_selected_strategy": raw_choice,
                "projected_inner_selected_strategy": projected_choice,
                "strategy_changed": raw_choice != projected_choice,
                "raw_outer_loss_mg_g": saved.iloc[0].outer_loss,
                "outer_projection_only_loss_mg_g": old_outer.iloc[0].projected_loss_mg_g,
                "projected_inner_and_outer_loss_mg_g": outer.iloc[0].projected_loss_mg_g,
            })
    return pd.DataFrame(inner_rows), pd.DataFrame(selector_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--bounds-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError("Output directory must be empty")
    inner, decisions = zip(*(
        calculate(mode, args.runs_root, args.bounds_dir)
        for mode in ("cell_mean", "original_response")
    ))
    inner = pd.concat(inner, ignore_index=True)
    decisions = pd.concat(decisions, ignore_index=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    inner.to_csv(args.out_dir / "inner_fold_scores.csv", index=False)
    decisions.to_csv(args.out_dir / "selector_source_results.csv", index=False)
    tasks = decisions.groupby(
        ["training_representation", "pollutant", "selector"], sort=True
    ).agg(
        changed_sources=("strategy_changed", "sum"),
        raw_mean_loss_mg_g=("raw_outer_loss_mg_g", "mean"),
        outer_projection_only_mean_loss_mg_g=("outer_projection_only_loss_mg_g", "mean"),
        projected_inner_and_outer_mean_loss_mg_g=(
            "projected_inner_and_outer_loss_mg_g", "mean"
        ),
    ).reset_index()
    tasks.to_csv(args.out_dir / "selector_task_results.csv", index=False)
    print(tasks.to_string(index=False))


if __name__ == "__main__":
    main()
