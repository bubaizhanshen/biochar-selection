"""Describe physical-range violations in existing source-held-out predictions."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


NON_RESPONSE_RULES = {"surface_area", "random"}


def summarize(data, keys):
    rows = []
    for names, group in data.groupby(keys, sort=True):
        if not isinstance(names, tuple):
            names = (names,)
        rows.append({
            **dict(zip(keys, names)),
            "prediction_cells": len(group),
            "above_mass_balance_count": int(group.above_mass_balance.sum()),
            "below_zero_count": int(group.below_zero.sum()),
            "outside_physical_range_count": int(group.outside_physical_range.sum()),
            "outside_physical_range_fraction": float(group.outside_physical_range.mean()),
            "largest_upper_excess_mg_g": float(group.upper_excess_mg_g.max()),
            "largest_prediction_to_limit_ratio": float(group.prediction_to_limit_ratio.max()),
            "lowest_predicted_mg_g": float(group.predicted.min()),
        })
    return pd.DataFrame(rows)


def audit(predictions, cells):
    required_predictions = {"stage", "strategy", "row_id", "observed", "predicted"}
    required_cells = {
        "source_table_row_id", "source_study_id", "pollutant", "C0_mg_L", "dose_g_L", "response_mg_g"
    }
    if required_predictions - set(predictions) or required_cells - set(cells):
        raise ValueError("Missing prediction or executed-input columns")
    if not cells.source_table_row_id.is_unique:
        raise ValueError("Source row identifiers must be unique")
    test = predictions.loc[
        predictions.stage.eq("outer") & ~predictions.strategy.isin(NON_RESPONSE_RULES)
    ].copy()
    if test.empty or test.duplicated(["strategy", "row_id"]).any():
        raise ValueError("Missing or repeated outer model predictions")
    joined = test.merge(cells[list(required_cells)], left_on="row_id", right_on="source_table_row_id",
                        validate="many_to_one")
    if len(joined) != len(test):
        raise ValueError("Predictions do not match executed input rows")
    values = joined[["observed", "predicted", "response_mg_g", "C0_mg_L", "dose_g_L"]]
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite prediction, response, concentration, or dose")
    if joined.C0_mg_L.le(0).any() or joined.dose_g_L.le(0).any():
        raise ValueError("Nonpositive initial concentration or adsorbent dose")
    if not np.allclose(joined.observed, joined.response_mg_g, rtol=0, atol=1e-8):
        raise ValueError("Prediction observations disagree with executed inputs")
    joined["upper_bound_mg_g"] = joined.C0_mg_L / joined.dose_g_L
    joined["above_mass_balance"] = joined.predicted.gt(joined.upper_bound_mg_g + 1e-9)
    joined["below_zero"] = joined.predicted.lt(-1e-9)
    joined["outside_physical_range"] = joined.above_mass_balance | joined.below_zero
    joined["upper_excess_mg_g"] = (joined.predicted - joined.upper_bound_mg_g).clip(lower=0)
    joined["prediction_to_limit_ratio"] = joined.predicted / joined.upper_bound_mg_g
    by_model = summarize(joined, ["strategy"])
    by_source = summarize(joined, ["pollutant", "source_study_id", "strategy"])
    observed = joined.drop_duplicates("row_id").copy()
    observed["observed_above_limit"] = observed.response_mg_g.gt(observed.upper_bound_mg_g + 1e-9)
    observed["observed_below_zero"] = observed.response_mg_g.lt(-1e-9)
    observed_by_source = observed.groupby(["pollutant", "source_study_id"], sort=True).agg(
        cells=("row_id", "size"),
        observed_above_limit=("observed_above_limit", "sum"),
        observed_below_zero=("observed_below_zero", "sum"),
        lowest_observed_mg_g=("response_mg_g", "min"),
    ).reset_index()
    return by_model, by_source, observed_by_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    predictions = pd.read_csv(args.run_dir / "predictions.csv")
    cells = pd.read_csv(args.run_dir / "executed_input.csv")
    by_model, by_source, observed = audit(predictions, cells)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    by_model.to_csv(args.out_dir / "prediction_bounds_by_model.csv", index=False)
    by_source.to_csv(args.out_dir / "prediction_bounds_by_source.csv", index=False)
    observed.to_csv(args.out_dir / "observed_bounds_by_source.csv", index=False)


if __name__ == "__main__":
    main()
