"""Post hoc physical-range sensitivity for the existing nine-model source holdouts.

This changes only outer-test predictions for scoring. It does not refit models or
use outer responses to choose a strategy. Source observations remain unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


from selection_metrics import selection_metrics


def score_conditions(test: pd.DataFrame, prediction: np.ndarray, keys: list[str]) -> pd.DataFrame:
    frame = test.copy()
    frame["prediction_mg_g"] = prediction
    for key in keys:
        if pd.api.types.is_numeric_dtype(frame[key]):
            frame[key] = frame[key].round(8)
    candidates = set(frame.material_group)
    rows = []
    for condition, group in frame.groupby(keys, sort=True, dropna=False):
        if set(group.material_group) != candidates:
            continue
        if group.material_group.duplicated().any():
            raise ValueError("Repeated candidate within a scored condition")
        group = group.sort_values("material_group")
        rows.append({**dict(zip(keys, condition)), **selection_metrics(
            group.response_mg_g.to_numpy(float), group.prediction_mg_g.to_numpy(float)
        )})
    return pd.DataFrame(rows)


def series_balanced(frame: pd.DataFrame, metric: str) -> float:
    return float(frame.groupby("series", sort=True)[metric].mean().mean())


def nominal_upper_bound(test: pd.DataFrame) -> np.ndarray:
    # Match the precision used to define a shared candidate condition.
    concentration = test.C0_mg_L.round(8).to_numpy(float)
    dose = test.dose_g_L.round(8).to_numpy(float)
    if (concentration <= 0).any() or (dose <= 0).any():
        raise ValueError("Nonpositive nominal concentration or dose")
    return concentration / dose


def audit_run(run: Path, mode: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(run / "executed_input.csv")
    predictions = pd.read_csv(run / "predictions.csv")
    published = pd.read_csv(run / "outer_results.csv")
    decisions = pd.read_csv(run / "decisions.csv")
    protocol = json.loads((run / "protocol.json").read_text())["model_config"]
    keys = ["series", *protocol["condition_features"], *protocol["categorical_condition_features"]]
    outer = predictions.loc[predictions.stage.eq("outer")].copy()
    if outer.duplicated(["panel_id", "strategy", "row_id"]).any():
        raise ValueError("Repeated outer prediction")
    if data.source_table_row_id.duplicated().any():
        raise ValueError("Repeated input row identifier")

    source_rows = []
    condition_rows = []
    for record in published.itertuples(index=False):
        match = outer.loc[
            outer.panel_id.eq(record.panel_id) & outer.strategy.eq(record.strategy)
        ]
        test = data.loc[
            data.source_study_id.eq(record.source)
            & data.pollutant.eq(record.pollutant)
        ].copy()
        if len(match) != len(test):
            raise ValueError(f"Prediction coverage differs from test source: {record.panel_id}")
        test = test.merge(match[["row_id", "observed", "predicted"]],
                          left_on="source_table_row_id", right_on="row_id", validate="one_to_one")
        if not np.allclose(test.response_mg_g, test.observed, rtol=0, atol=1e-8):
            raise ValueError("Saved predictions disagree with source observations")
        bound = nominal_upper_bound(test)
        original = test.predicted.to_numpy(float)
        is_model = record.strategy not in {"surface_area", "random"}
        projected = np.clip(original, 0, bound) if is_model else original.copy()
        raw = score_conditions(test, original, keys)
        bounded = score_conditions(test, projected, keys)
        if len(raw) != record.n_conditions or len(raw) != len(bounded):
            raise ValueError("Complete-condition count changed")
        raw_loss = series_balanced(raw, "selection_loss")
        if not np.isclose(raw_loss, record.loss, rtol=0, atol=1e-8):
            raise ValueError(f"Cannot reproduce original loss: {record.panel_id}/{record.strategy}")
        bounded_loss = series_balanced(bounded, "selection_loss")
        source_rows.append({
            "training_representation": mode,
            "panel_id": record.panel_id,
            "source": record.source,
            "pollutant": record.pollutant,
            "strategy": record.strategy,
            "complete_conditions": len(raw),
            "prediction_cells": len(test),
            "predictions_above_limit": int((original > bound + 1e-9).sum()) if is_model else 0,
            "predictions_below_zero": int((original < -1e-9).sum()) if is_model else 0,
            "observations_above_limit": int((test.response_mg_g.to_numpy(float) > bound + 1e-9).sum()),
            "observations_below_zero": int((test.response_mg_g.to_numpy(float) < -1e-9).sum()),
            "raw_loss_mg_g": raw_loss,
            "projected_loss_mg_g": bounded_loss,
            "loss_change_mg_g": bounded_loss - raw_loss,
            "raw_mae_mg_g": series_balanced(raw, "mae") if is_model else np.nan,
            "projected_mae_mg_g": series_balanced(bounded, "mae") if is_model else np.nan,
            "changed_condition_losses": int((~np.isclose(
                raw.selection_loss.to_numpy(float), bounded.selection_loss.to_numpy(float),
                rtol=0, atol=1e-9,
            )).sum()),
            "changed_selected_tie_counts": int((
                raw.selected_tie_count.to_numpy(int) != bounded.selected_tie_count.to_numpy(int)
            ).sum()),
        })
        if is_model and ((original > bound + 1e-9).any() or (original < -1e-9).any()):
            condition_rows.extend({
                "training_representation": mode,
                "panel_id": record.panel_id,
                "source": record.source,
                "strategy": record.strategy,
                "series": row.series,
                "raw_loss_mg_g": row.selection_loss,
                "projected_loss_mg_g": bounded.iloc[i].selection_loss,
                "raw_ties": row.selected_tie_count,
                "projected_ties": int(bounded.iloc[i].selected_tie_count),
            } for i, row in enumerate(raw.itertuples(index=False)))

    sources = pd.DataFrame(source_rows)
    selected = decisions.merge(
        sources,
        left_on=["panel_id", "source", "pollutant", "selected_strategy"],
        right_on=["panel_id", "source", "pollutant", "strategy"],
        validate="many_to_one",
    )
    if len(selected) != len(decisions):
        raise ValueError("A training-selected strategy is missing an outer score")
    if not np.allclose(selected.outer_loss, selected.raw_loss_mg_g, rtol=0, atol=1e-8):
        raise ValueError("Selected-strategy scores disagree with saved decisions")
    return sources, selected, pd.DataFrame(condition_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    source_tables, decision_tables, condition_tables = zip(*(
        audit_run(args.runs_root / mode, mode)
        for mode in ("cell_mean", "original_response")
    ))
    sources = pd.concat(source_tables, ignore_index=True)
    decisions = pd.concat(decision_tables, ignore_index=True)
    conditions = pd.concat(condition_tables, ignore_index=True)
    summary = decisions.groupby(
        ["training_representation", "pollutant", "selector"], sort=True
    )[["raw_loss_mg_g", "projected_loss_mg_g"]].mean().reset_index()
    summary["change_mg_g"] = summary.projected_loss_mg_g - summary.raw_loss_mg_g
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sources.to_csv(args.out_dir / "fixed_strategy_by_source.csv", index=False)
    decisions.to_csv(args.out_dir / "training_selected_by_source.csv", index=False)
    conditions.to_csv(args.out_dir / "affected_strategy_conditions.csv", index=False)
    summary.to_csv(args.out_dir / "training_selected_task_means.csv", index=False)
    print(summary.to_string(index=False))
    print("\nLargest source-level changes:")
    print(sources.loc[sources.loss_change_mg_g.abs().sort_values(ascending=False).index]
          .head(15)[["training_representation", "source", "strategy", "predictions_above_limit",
                      "predictions_below_zero", "raw_loss_mg_g", "projected_loss_mg_g",
                      "changed_selected_tie_counts"]].to_string(index=False))


if __name__ == "__main__":
    main()
