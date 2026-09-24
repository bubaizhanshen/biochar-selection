# Data dictionary

## Prepared Selection Inputs

The exploratory As(V) input is distributed separately at
[`data/asv/analysis_input.csv`](data/asv/analysis_input.csv). Its complete
column dictionary, source crosswalk, screening rules, and grouping caveats are
in [`data/asv/README.md`](data/asv/README.md).

The portable strategy runner accepts an externally prepared cell table and an
explicit manifest. This interface does not by itself certify source eligibility.

| Field | Meaning |
| --- | --- |
| `source_table_row_id` | Stable integer cell identifier used by the manifest; not an experimental replicate ID |
| `source_study_id` | Reported/reconstructed source block, not proof of independent laboratory or batch |
| `material_group` | Source-specific reported material group |
| `series` | Experimental series; conditions are averaged within this unit before source aggregation |
| `response_mg_g` | Mean recorded amount per mass at the recorded contact time; not necessarily equilibrium capacity |
| `raw_record_ids_json` | Exact released-record membership of the cell |
| `n_compilation_rows` | Number of linked released records, not a certified replicate count |
| `raw_responses_mg_g_json` | Harmonized linked response values used only when original-response training is requested |
| `recorded_min_mg_g`, `recorded_max_mg_g`, `recorded_sd_mg_g` | Descriptive within-cell statistics; not automatically experimental confidence limits |
| `matrix` | Recorded water-matrix category; an unknown category does not establish identical chemistry |

`C0_mg_L`, `dose_g_L`, and `contact_time_h` use mg/L, g/L, and hours.
`ionic_strength_M` uses mol/L. The feature list and remaining units must be
checked against the supplied protocol and source records. The complete condition
vector, including categorical variables, defines comparable candidate cells.
Incomplete grids are excluded from decision scoring, not silently completed.

The manifest declares `holdout_unit=study_block`, explicit training and test
cell IDs, the complete candidate set, and expected support counts. At least two
training sources are required for the inner study-block holdout comparison. All
inner choices precede evaluation on the outer source. Exact objective ties use
the declared strategy order; prediction ties instead use the numerical tolerance
specified by the metric implementation.
