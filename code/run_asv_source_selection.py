"""Run the exploratory arsenic(V) source-held-out selection comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model_registry import configured_models, make_regressor


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "config" / "selection.json"
EXPANSION_CONFIG = ROOT / "config" / "model_expansion.json"
ASV_CONFIG = ROOT / "config" / "asv_source_selection.json"

FULL_MATERIAL_FEATURES = (
    "C_pct", "H_pct", "O_pct", "Fe_pct", "H_C", "O_C", "ON_C", "area_m2_g"
)
SHARED_MATERIAL_FEATURES = ("C_pct", "H_pct", "O_pct", "area_m2_g")
CONDITION_FEATURES = ("As_initial_mg_L", "temperature_C", "pH", "dose_g_L")
RULE_FEATURES = {
    "surface_area": ("area_m2_g", "max"),
    "oxygen": ("O_pct", "max"),
    "O_C": ("O_C", "max"),
    "Fe_10_20": ("Fe_pct", "band"),
    "Fe_max": ("Fe_pct", "max"),
    "low_H": ("H_pct", "min"),
    "low_H_C": ("H_C", "min"),
}


def load_config(path: Path = ASV_CONFIG) -> tuple[dict, dict, dict]:
    asv = json.loads(path.read_text(encoding="utf-8"))
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    expansion = json.loads(EXPANSION_CONFIG.read_text(encoding="utf-8"))
    base.update({key: value for key, value in expansion.items() if key != "models"})
    base["models"] = expansion["models"]
    return asv, base, expansion


def model_features(predictor_set: str) -> tuple[str, ...]:
    material = (
        FULL_MATERIAL_FEATURES
        if predictor_set == "asv_specific"
        else SHARED_MATERIAL_FEATURES
    )
    return (*material, *CONDITION_FEATURES)


def validate_data(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    source_col = config["source_column"]
    material_col = config["material_column"]
    response_col = config["response_column"]
    numeric = sorted(set((*FULL_MATERIAL_FEATURES, *CONDITION_FEATURES, response_col)))
    required = {source_col, material_col, *numeric}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Missing required values; implicit imputation is not permitted")
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Predictors and responses must be finite")
    key = [source_col, material_col, *CONDITION_FEATURES]
    if frame.duplicated(key).any():
        raise ValueError("Repeated material-condition rows require source resolution")
    expected = set(config["test_sources"]) | set(config["training_only_sources"])
    observed = set(frame[source_col].astype(str))
    if observed != expected:
        raise ValueError(
            f"Source IDs differ from the declared protocol; missing={sorted(expected-observed)}, "
            f"unexpected={sorted(observed-expected)}"
        )
    return frame.sort_values(
        [source_col, *FULL_MATERIAL_FEATURES, *CONDITION_FEATURES],
        kind="mergesort",
    ).reset_index(drop=True)


def panel_rows(frame: pd.DataFrame, source: str, config: dict) -> pd.DataFrame:
    source_col = config["source_column"]
    material_col = config["material_column"]
    block = frame[frame[source_col].astype(str).eq(source)]
    keep = [
        group
        for _, group in block.groupby(list(CONDITION_FEATURES), sort=True, dropna=False)
        if group[material_col].nunique() >= 2
    ]
    if not keep:
        return frame.iloc[0:0].copy()
    return pd.concat(keep, ignore_index=True)


def panel_support(frame: pd.DataFrame, source: str, config: dict) -> dict:
    """Describe observed support without treating a row count as replication."""
    material_col = config["material_column"]
    panel = panel_rows(frame, source, config)
    counts = panel.groupby(list(CONDITION_FEATURES), sort=True)[material_col].nunique()
    return {
        config["source_column"]: source,
        "n_panel_biochars": int(panel[material_col].nunique()),
        "n_matched_conditions": int(len(counts)),
        "n_conditions_with_at_least_3_biochars": int((counts >= 3).sum()),
        "min_biochars_per_condition": int(counts.min()) if len(counts) else 0,
        "max_biochars_per_condition": int(counts.max()) if len(counts) else 0,
        "three_biochar_three_condition_subset": bool(
            panel[material_col].nunique() >= 3 and (counts >= 3).sum() >= 3
        ),
    }


def outer_training_rows(
    frame: pd.DataFrame,
    source_column: str,
    test_source: str,
    excluded_sources: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Remove the held-out source and any prespecified sources from fitting."""
    excluded = {str(test_source), *(str(source) for source in excluded_sources)}
    return frame.loc[~frame[source_column].astype(str).isin(excluded)].copy()


def make_model(name: str, parameters: dict):
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), make_regressor(name, parameters)),
        transformer=StandardScaler(),
    )


def rule_scores(name: str, frame: pd.DataFrame) -> np.ndarray:
    feature, direction = RULE_FEATURES[name]
    values = frame[feature].to_numpy(dtype=float)
    if direction == "band":
        return -np.maximum(10.0 - values, 0.0) - np.maximum(values - 20.0, 0.0)
    return values if direction == "max" else -values


def score_source(
    source: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...],
    strategies: tuple[str, ...],
    model_config: dict,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_col = config["source_column"]
    material_col = config["material_column"]
    response_col = config["response_column"]
    predictions: dict[str, np.ndarray | None] = {}
    for strategy in strategies:
        if strategy == "random":
            predictions[strategy] = None
        elif strategy in RULE_FEATURES:
            predictions[strategy] = rule_scores(strategy, test)
        else:
            estimator = make_model(strategy, model_config[strategy])
            estimator.fit(train.loc[:, features], train[response_col].to_numpy(float))
            values = estimator.predict(test.loc[:, features])
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite prediction for {source}/{strategy}")
            predictions[strategy] = values

    condition_results = []
    for strategy in strategies:
        predicted = predictions[strategy]
        block = test.copy()
        block["_score"] = np.nan if predicted is None else predicted
        for condition, group in block.groupby(list(CONDITION_FEATURES), sort=True, dropna=False):
            observed = group[response_col].to_numpy(float)
            best = np.isclose(observed, np.max(observed), rtol=0.0, atol=1e-12)
            span = float(np.ptp(observed))
            if strategy == "random":
                loss = float(np.max(observed) - np.mean(observed))
                hit = float(np.mean(best))
                mae = np.nan
                selected = np.zeros(len(group), dtype=bool)
            else:
                score = group["_score"].to_numpy(float)
                selected = np.isclose(score, np.max(score), rtol=0.0, atol=1e-12)
                loss = float(np.max(observed) - np.mean(observed[selected]))
                hit = float(np.mean(best[selected]))
                mae = (
                    float(np.mean(np.abs(score - observed)))
                    if strategy not in RULE_FEATURES
                    else np.nan
                )
            condition_results.append(
                {
                    source_col: source,
                    "strategy": strategy,
                    **dict(zip(CONDITION_FEATURES, condition)),
                    "n_candidates": int(group[material_col].nunique()),
                    "selection_loss_mg_g": loss,
                    "normalized_loss": loss / span if span else np.nan,
                    "top_hit_fraction": hit,
                    "mae_mg_g": mae,
                    "selected_material_groups": (
                        "uniform-random expectation"
                        if strategy == "random"
                        else ";".join(group.loc[selected, material_col].astype(str))
                    ),
                }
            )
    detail = pd.DataFrame(condition_results)
    summary_rows = []
    for strategy, group in detail.groupby("strategy", sort=False):
        summary_rows.append(
            {
                source_col: source,
                "strategy": strategy,
                "selection_loss_mg_g": float(group.selection_loss_mg_g.mean()),
                "normalized_loss": float(group.normalized_loss.mean()),
                "top_hit_fraction": float(group.top_hit_fraction.mean()),
                "mae_mg_g": float(group.mae_mg_g.mean()),
                "n_conditions": int(len(group)),
                "n_candidates": int(test[material_col].nunique()),
                "n_training_rows": int(len(train)),
                "n_training_sources": int(train[source_col].nunique()),
            }
        )
    return pd.DataFrame(summary_rows), detail


def choose_strategy(scores: dict[str, dict[str, float]], objective: str, order: tuple[str, ...]) -> str:
    values = [scores[name][objective] for name in order]
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite inner-validation scores for {objective}")
    return min(order, key=lambda name: (scores[name][objective], order.index(name)))


def run(
    data_path: Path,
    config_path: Path,
    out: Path,
    predictor_set: str,
    exclude_training_only: bool,
    exclude_training_sources: tuple[str, ...] = (),
) -> None:
    if out.exists():
        if not out.is_dir():
            raise ValueError("Output path must be a directory")
        if any(out.iterdir()):
            raise ValueError("Output directory must be empty")
    asv, model_config, _expansion = load_config(config_path)
    data = validate_data(pd.read_csv(data_path), asv)
    source_col = asv["source_column"]
    excluded_sources = tuple(dict.fromkeys(map(str, exclude_training_sources)))
    declared_sources = set(asv["test_sources"]) | set(asv["training_only_sources"])
    unknown_sources = set(excluded_sources).difference(declared_sources)
    if unknown_sources:
        raise ValueError(f"Unknown training sources to exclude: {sorted(unknown_sources)}")
    if exclude_training_only:
        data = data[~data[source_col].astype(str).isin(asv["training_only_sources"])].reset_index(drop=True)
    test_sources = tuple(asv["test_sources"])
    features = model_features(predictor_set)
    models = tuple(configured_models(model_config))
    rules = tuple(RULE_FEATURES)
    strategies = (*rules, "random", *models)
    selectors = {
        "model_by_MAE": (models, "mae_mg_g"),
        "model_by_loss": (models, "selection_loss_mg_g"),
        "model_by_normalized_loss": (models, "normalized_loss"),
        "rule_by_loss": (rules, "selection_loss_mg_g"),
        "rule_by_normalized_loss": (rules, "normalized_loss"),
        "model_or_rule_by_loss": ((*rules, "random", *models), "selection_loss_mg_g"),
        "model_or_rule_by_normalized_loss": ((*rules, "random", *models), "normalized_loss"),
    }
    out.mkdir(parents=True, exist_ok=True)
    inner_rows, choice_rows, outer_rows, condition_rows = [], [], [], []
    support_rows = [panel_support(data, source, asv) for source in test_sources]
    support = {
        row[source_col]: row["three_biochar_three_condition_subset"]
        for row in support_rows
    }

    for test_source in test_sources:
        test_panel = panel_rows(data, test_source, asv)
        if test_panel.empty:
            raise ValueError(f"Declared test source has no shared-condition panel: {test_source}")
        outer_train = outer_training_rows(data, source_col, test_source, excluded_sources)
        inner_scores = {strategy: {"selection_loss_mg_g": [], "normalized_loss": [], "mae_mg_g": []} for strategy in strategies}
        inner_sources = []
        for validation_source in sorted(outer_train[source_col].astype(str).unique()):
            validation = panel_rows(data, validation_source, asv)
            if validation.empty:
                continue
            inner_train = outer_train[outer_train[source_col].astype(str).ne(validation_source)].copy()
            if inner_train[source_col].nunique() < 2:
                continue
            validation_scores, _ = score_source(
                validation_source,
                inner_train,
                validation,
                features,
                strategies,
                model_config,
                asv,
            )
            by_strategy = validation_scores.set_index("strategy")
            inner_sources.append(validation_source)
            for strategy in strategies:
                for metric in ("selection_loss_mg_g", "normalized_loss"):
                    inner_scores[strategy][metric].append(float(by_strategy.loc[strategy, metric]))
                if strategy in models:
                    inner_scores[strategy]["mae_mg_g"].append(float(by_strategy.loc[strategy, "mae_mg_g"]))
                inner_rows.append(
                    {
                        "outer_test_source": test_source,
                        "inner_validation_source": validation_source,
                        "strategy": strategy,
                        **by_strategy.loc[strategy].to_dict(),
                    }
                )
        means = {
            strategy: {
                metric: float(np.mean(values)) if values else np.nan
                for metric, values in metric_values.items()
            }
            for strategy, metric_values in inner_scores.items()
        }
        selected = {
            selector: choose_strategy(means, objective, order)
            for selector, (order, objective) in selectors.items()
        }
        for selector, strategy in selected.items():
            choice_rows.append(
                {
                    "outer_test_source": test_source,
                    "selector": selector,
                    "selected_strategy": strategy,
                    "n_inner_sources": len(inner_sources),
                    "inner_mean_selection_loss_mg_g": means[strategy]["selection_loss_mg_g"],
                    "inner_mean_MAE_mg_g": means[strategy]["mae_mg_g"],
                }
            )

        outer_scores, detail = score_source(
            test_source,
            outer_train,
            test_panel,
            features,
            strategies,
            model_config,
            asv,
        )
        outer_scores = outer_scores.set_index("strategy")
        for strategy in strategies:
            outer_rows.append(
                outer_scores.loc[strategy].to_dict()
                | {
                    source_col: test_source,
                    "strategy": strategy,
                    "selector": np.nan,
                    "three_biochar_three_condition_subset": support[test_source],
                }
            )
        for selector, strategy in selected.items():
            outer_rows.append(
                outer_scores.loc[strategy].to_dict()
                | {
                    source_col: test_source,
                    "strategy": strategy,
                    "selector": selector,
                    "three_biochar_three_condition_subset": support[test_source],
                }
            )
        choices_for_source = {
            row["selector"]: row
            for row in choice_rows
            if row["outer_test_source"] == test_source
        }
        for selector, strategy in selected.items():
            choices_for_source[selector]["outer_selection_loss_mg_g"] = float(
                outer_scores.loc[strategy, "selection_loss_mg_g"]
            )
        condition_rows.extend(detail.to_dict("records"))

    outer = pd.DataFrame(outer_rows)
    selected_outer = outer[outer["selector"].notna()].copy()
    fixed_outer = outer[outer["selector"].isna()].copy()
    summaries = []
    tier_rows = []
    panel_sets = (
        ("all_test_panels", pd.Series(True, index=outer.index)),
        (
            "three_biochar_three_condition_subset",
            outer["three_biochar_three_condition_subset"].astype(bool),
        ),
    )
    for panel_set, mask in panel_sets:
        selected_subset = selected_outer.loc[mask.reindex(selected_outer.index).fillna(False)]
        for selector, group in selected_subset.groupby("selector", sort=False):
            summaries.append(
                {
                    "panel_set": panel_set,
                    "selector": selector,
                    "n_sources": int(group[source_col].nunique()),
                    "mean_source_loss_mg_g": float(group.selection_loss_mg_g.mean()),
                    "mean_source_normalized_loss": float(group.normalized_loss.mean()),
                    "mean_source_top_hit_fraction": float(group.top_hit_fraction.mean()),
                    "mean_source_MAE_mg_g": float(group.mae_mg_g.mean()),
                    "selected_strategies": ";".join(
                        f"{source}:{strategy}"
                        for source, strategy in zip(group[source_col], group.strategy)
                    ),
                }
            )
        fixed_subset = fixed_outer.loc[mask.reindex(fixed_outer.index).fillna(False)]
        for strategy, group in fixed_subset.groupby("strategy", sort=False):
            tier_rows.append(
                {
                    "panel_set": panel_set,
                    "strategy": strategy,
                    "mean_source_loss_mg_g": float(group.selection_loss_mg_g.mean()),
                    "mean_source_normalized_loss": float(group.normalized_loss.mean()),
                    "mean_source_top_hit_fraction": float(group.top_hit_fraction.mean()),
                    "mean_source_MAE_mg_g": float(group.mae_mg_g.mean()),
                    "n_sources": int(group[source_col].nunique()),
                }
            )
    outer.to_csv(out / "outer_source_results.csv", index=False)
    pd.DataFrame(condition_rows).to_csv(out / "outer_condition_results.csv", index=False)
    pd.DataFrame(inner_rows).to_csv(out / "inner_validation_scores.csv", index=False)
    pd.DataFrame(choice_rows).to_csv(out / "selected_strategies.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "selector_summary.csv", index=False)
    pd.DataFrame(tier_rows).to_csv(out / "fixed_strategy_summary.csv", index=False)
    pd.DataFrame(support_rows).to_csv(out / "panel_support.csv", index=False)
    protocol = {
        "status": "retrospective_exploratory_source_held_out_analysis",
        "input_file": data_path.name,
        "input_values_copied_to_output": False,
        "predictor_set": predictor_set,
        "features": list(features),
        "test_sources": list(test_sources),
        "rows_by_source": {
            str(source): int(count)
            for source, count in data[source_col].astype(str).value_counts().sort_index().items()
        },
        "panel_subset_rule": (
            "At least three unique biochars and at least three recorded conditions "
            "with at least three biochars represented at each condition; descriptive only."
        ),
        "training_only_sources_excluded": bool(exclude_training_only),
        "additional_training_sources_excluded": list(excluded_sources),
        "inner_selection": "equal-source mean of condition-averaged loss; model MAE is also reported",
        "outer_test_rows_excluded_from_all_fitting_and_selection": True,
        "strategies": list(strategies),
        "selectors": {key: {"candidates": list(value[0]), "objective": value[1]} for key, value in selectors.items()},
        "model_parameters": {name: model_config[name] for name in models},
        "analysis_notes": asv["analysis_notes"],
        "n_rows": int(len(data)),
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ASV_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--predictor-set", choices=("asv_specific", "shared"), required=True)
    parser.add_argument("--exclude-training-only", action="store_true")
    parser.add_argument(
        "--exclude-training-source",
        action="append",
        default=[],
        metavar="SOURCE_ID",
        help="Exclude this source from all outer training pools; may be repeated.",
    )
    args = parser.parse_args()
    run(
        args.data,
        args.config,
        args.out,
        args.predictor_set,
        args.exclude_training_only,
        tuple(args.exclude_training_source),
    )


if __name__ == "__main__":
    main()
