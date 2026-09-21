"""Materialize source-specific analysis-copy candidates without admitting rows.

The raw HMI/EC workbooks remain untouched.  This script joins the source-order
review tables to raw rows, records proposed analysis fields, and marks rows
that are outside a reported source range or have unresolved response issues.
The resulting files are audit artifacts, not model manifests.
"""

from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT
WORKBOOKS = {
    "Dataset II": REPO / "data/benchmark/HMI_data.xlsx",
    "Dataset III": REPO / "data/benchmark/EC.xlsx",
}
AUDIT = ROOT / "data/selection_annotations"
REPORT_DIR = AUDIT
OUT_DIR = ROOT / "work/source_copies"


POLICIES = {
    "Shin2021_Alkaline": {
        "dataset": "Dataset II",
        "mapping": "shin2021_alkaline_series_mapping_review.csv",
        "task_filter": {"Metal type": "Sr(II)"},
        "time_column": "Adsorption_time (min)",
        "pollutant_column": "Metal type",
        "material_column": "original_material_label",
        "source_range_column": "solution pH",
        "source_range_min": 3,
        "source_range_max": 9,
    },
    "Shin2021_Magnetic": {
        "dataset": "Dataset II",
        "mapping": "shin2021_magnetic_series_mapping_review.csv",
        "task_filter": {"Metal type": "Sr(II)"},
        "time_column": "Adsorption_time (min)",
        "pollutant_column": "Metal type",
        "material_column": "original_material_label",
        "source_range_column": "source_pH_status",
        "source_range_token": "outside_source",
    },
    "Shin2022_Postmodification": {
        "dataset": "Dataset II",
        "mapping": "shin2022_postmodification_series_mapping_review.csv",
        "task_filter": {"Metal type": "Sr(II)"},
        "time_column": "Adsorption_time (min)",
        "pollutant_column": "Metal type",
        "material_column": "original_material_label",
        "source_range_column": "source_pH_status",
        "source_range_token": "outside_source",
    },
    "Shin2020_Micropollutants": {
        "dataset": "Dataset III",
        "mapping": "shin2020_micropollutants_series_mapping_review.csv",
        "task_filter": {},
        "time_column": "Adsorption time",
        "pollutant_column": "Pollutant",
        "material_column": "original_material_label",
    },
    "Shin2021_Competitive": {
        "dataset": "Dataset III",
        "mapping": "shin2021_competitive_series_mapping_review.csv",
        "task_filter": {"Pollutant": "IBU"},
        "time_column": "Adsorption time",
        "pollutant_column": "analysis_pollutant_label",
        "material_column": "original_material_label",
    },
    "Shin2022_Ibuprofen": {
        "dataset": "Dataset III",
        "mapping": "shin2022_ibuprofen_series_mapping_review.csv",
        "task_filter": {},
        "time_column": "Adsorption time",
        "pollutant_column": "Pollutant",
        "material_column": "candidate_material_label",
    },
}


def _raw_frame(dataset: str) -> pd.DataFrame:
    frame = pd.read_excel(WORKBOOKS[dataset])
    frame.columns = [str(column).strip() for column in frame.columns]
    if dataset == "Dataset II":
        frame = frame.rename(columns={
            "(O+N/C)": "(O+N)/C",
            "inorganics": "Metal type",
            "g/L": "Dosage(g/L)",
        })
    if frame.columns.duplicated().any():
        raise ValueError(f"{dataset}: ambiguous duplicate columns after header normalization")
    return frame.reset_index(names="source_row_id")


def _apply_task_filter(frame: pd.DataFrame, task_filter: dict[str, str]) -> pd.DataFrame:
    for column, value in task_filter.items():
        frame = frame[frame[column].astype(str).eq(value)]
    return frame


def build_source(source_id: str, policy: dict[str, object]) -> pd.DataFrame:
    mapping = pd.read_csv(AUDIT / str(policy["mapping"]))
    raw = _raw_frame(str(policy["dataset"]))
    raw_columns = list(raw.columns)
    merged = mapping.merge(raw, on="source_row_id", how="left", validate="one_to_one", suffixes=("", "_raw"), indicator="_workbook_match")
    raw_value_columns = [
        f"{column}_raw" if column in mapping.columns and column != "source_row_id" else column
        for column in raw_columns
    ]
    if merged['_workbook_match'].ne('both').any():
        missing = merged.loc[merged['_workbook_match'].ne('both'), "source_row_id"].tolist()
        raise RuntimeError(f"{source_id}: source rows missing from raw workbook: {missing[:10]}")

    merged = _apply_task_filter(merged, policy["task_filter"])

    output = merged[raw_value_columns].copy()
    output.columns = raw_columns
    output.insert(0, "analysis_source_study_id", source_id)
    output.insert(1, "analysis_dataset", str(policy["dataset"]))
    output.insert(2, "analysis_source_row_id", merged["source_row_id"].astype(int))
    output["analysis_task"] = merged[policy["pollutant_column"]].astype(str)
    output["analysis_series"] = merged["candidate_series"].astype(str)
    if "response_endpoint_candidate" in merged:
        output["analysis_endpoint"] = merged["response_endpoint_candidate"].astype(str)
    else:
        output["analysis_endpoint"] = "source_endpoint_pending_series_review"
    output["analysis_material_label"] = merged[policy["material_column"]].astype(str)
    output["analysis_time_min"] = merged[policy["time_column"]]
    output["analysis_transformations"] = "none"
    output["analysis_row_status"] = "candidate_pending_source_admission"
    output["analysis_exclusion_reason"] = ""

    if source_id == "Shin2020_Micropollutants":
        output["analysis_adsorption_type"] = merged["analysis_adsorption_type"].astype(str)
        output["analysis_dom_status"] = merged["dom_status"].astype(str)
        output.loc[:, "analysis_transformations"] = merged["adsorption_type_mapping_status"].astype(str)
        negative = merged["response_quality_status"].astype(str).eq(
            "negative_capacity_requires_source_response_review"
        )
        output.loc[negative, "analysis_row_status"] = "candidate_exclude_pending_negative_response_review"
        output.loc[negative, "analysis_exclusion_reason"] = "negative capacity requires source-response review"
    elif source_id == "Shin2021_Competitive":
        output["analysis_adsorption_type"] = merged["analysis_adsorption_type"].astype(str)
        output["analysis_pollutant"] = merged["analysis_pollutant_label"].astype(str)
        output["analysis_transformations"] = (
            merged["adsorption_type_mapping_status"].astype(str)
            + ";"
            + merged["pollutant_label_mapping_status"].astype(str)
        )
    elif source_id == "Shin2022_Ibuprofen":
        output["analysis_transformations"] = (
            merged["material_label_mapping_status"].astype(str)
            + ";time_candidate="
            + merged["candidate_time_min"].astype(str)
        )
        output["analysis_material_label"] = merged["candidate_material_label"].astype(str)
        output["analysis_time_min"] = merged["candidate_time_min"]
        negative = merged["response_quality_status"].astype(str).eq(
            "negative_capacity_requires_source_response_review"
        )
        output.loc[negative, "analysis_row_status"] = "candidate_exclude_pending_negative_response_review"
        output.loc[negative, "analysis_exclusion_reason"] = "negative capacity requires source-response review"
    elif source_id == "Shin2022_Postmodification":
        output["analysis_ionic_strength_M"] = merged["Ion Concentration (M)"]
        ionic = (
            merged["candidate_series"].astype(str).eq("ionic_strength_pattern")
            & pd.to_numeric(merged["Ion Concentration (M)"], errors="coerce").ne(0)
        )
        output.loc[ionic, "analysis_ionic_strength_M"] = merged.loc[ionic, "Ion Concentration (M)"] / 1000.0
        output.loc[ionic, "analysis_transformations"] = "source ionic strength mM -> analysis M"
    else:
        output["analysis_ionic_strength_M"] = merged["Ion Concentration (M)"]

    range_column = policy.get("source_range_column")
    range_token = policy.get("source_range_token")
    range_min = policy.get("source_range_min")
    range_max = policy.get("source_range_max")
    if range_column and range_token:
        outside = merged[str(range_column)].astype(str).str.contains(str(range_token), na=False)
    elif range_column and (range_min is not None or range_max is not None):
        numeric_range = pd.to_numeric(merged[str(range_column)], errors="coerce")
        outside = pd.Series(False, index=merged.index)
        if range_min is not None:
            outside |= numeric_range.lt(float(range_min))
        if range_max is not None:
            outside |= numeric_range.gt(float(range_max))
    else:
        outside = pd.Series(False, index=merged.index)
    if range_column:
        output.loc[outside, "analysis_row_status"] = "candidate_exclude_pending_source_range_review"
        output.loc[outside, "analysis_exclusion_reason"] = "condition lies outside the source-reported range"

    return output.sort_values("analysis_source_row_id").reset_index(drop=True)


def main() -> None:
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise RuntimeError('Output must be empty; do not overwrite reviewed source copies')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    combined: list[pd.DataFrame] = []
    for source_id, policy in POLICIES.items():
        frame = build_source(source_id, policy)
        output_path = OUT_DIR / f"{source_id.lower()}_analysis_copy_candidate.csv"
        frame.to_csv(output_path, index=False)
        combined.append(frame)
        summaries.append(
            {
                "source_study_id": source_id,
                "dataset": policy["dataset"],
                "rows": len(frame),
                "candidate_rows": int(frame["analysis_row_status"].eq("candidate_pending_source_admission").sum()),
                "range_review_rows": int(frame["analysis_row_status"].eq("candidate_exclude_pending_source_range_review").sum()),
                "negative_review_rows": int(frame["analysis_row_status"].eq("candidate_exclude_pending_negative_response_review").sum()),
                "raw_values_changed": "no",
                "manifest_admission": "not_admitted",
                "output": str(output_path.relative_to(ROOT)) if output_path.is_relative_to(ROOT) else output_path.name,
            }
        )

    summary = pd.DataFrame(summaries)
    summary.to_csv(REPORT_DIR / "source_analysis_copy_candidate_summary.csv", index=False)
    pd.concat(combined, ignore_index=True).to_csv(
        REPORT_DIR / "source_analysis_copy_candidate_rows.csv", index=False
    )
    report = [
        "# Source analysis-copy candidate review",
        "",
        "Generated by `preparation/scripts/build_source_analysis_copies.py`.",
        "",
        "These files are analysis-copy candidates only. Raw HMI/EC workbooks are unchanged, and no row is admitted to a primary or sensitivity manifest by this script.",
        "",
        f"- Source blocks materialized: {len(summary)}.",
        f"- Rows materialized: {int(summary['rows'].sum())}.",
        f"- Rows still marked candidate-pending: {int(summary['candidate_rows'].sum())}.",
        f"- Range-review rows: {int(summary['range_review_rows'].sum())}.",
        f"- Negative-response review rows: {int(summary['negative_review_rows'].sum())}.",
        "- Raw values changed: no.",
        "- Manifest blocks admitted: none.",
        "",
        "## Transform boundaries",
        "",
        "- Shin2021_Alkaline, Shin2021_Magnetic, and Shin2022_Postmodification mark source-range conflicts; no synthetic zero-time row is created.",
        "- Shin2022_Postmodification records the source mM-to-analysis-M conversion only in the analysis copy.",
        "- Shin2020_Micropollutants records `Competative` to `Competitive` as a candidate field and does not infer the missing 5 mg/L DOM series.",
        "- Shin2021_Competitive is restricted to the canonical IBU rows; it records `Single` to `Competitive` as a candidate field only. DCF and NPX/NXP rows from the same article remain in the source audit and are not materialized in the IBU analysis copy.",
        "- Shin2022_Ibuprofen records `AMCB` to `MACB` and the source-supported 1440 min candidate time only in the analysis copy; negative rows remain flagged.",
    ]
    (REPORT_DIR / "source_analysis_copy_candidate_review.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"Saved: {REPORT_DIR / 'source_analysis_copy_candidate_summary.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hmi-workbook', type=Path, default=WORKBOOKS['Dataset II'])
    parser.add_argument('--ec-workbook', type=Path, default=WORKBOOKS['Dataset III'])
    parser.add_argument('--mappings', type=Path, default=AUDIT)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    WORKBOOKS = {'Dataset II': args.hmi_workbook.resolve(), 'Dataset III': args.ec_workbook.resolve()}
    AUDIT = args.mappings.resolve()
    OUT_DIR = args.out.resolve()
    REPORT_DIR = OUT_DIR / 'reports'
    main()
