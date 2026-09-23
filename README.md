# Biochar Selection

Compare model-guided candidate selection with material-property rules and random
choice under common recorded adsorption conditions. The main experiment compares
training-only model selection by response MAE and by selection loss, using the
same fixed models, inner validation conditions, and weights.

This package contains code, configuration, source annotations, and tests. It
does not contain original workbooks, experimental response tables, article full
text, manuscripts, figures, or full historical analysis outputs. A small aggregate
reference fixture is included for regression testing. Obtain source files
separately under their original terms; see [data preparation](docs/selection_data.md).

## Install and Test

Python 3.13.2 was used for the verified local runs.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-models.txt
make test-full
```

This installs the pinned XGBoost, LightGBM, and CatBoost versions needed for the
manuscript's nine-model comparison. For the three-model reference alone,
`requirements.txt` and `make test` are sufficient. `test-full` fails if an
extra-model dependency is absent rather than silently skipping its tests.

## Prepare and Run

```bash
python code/prepare_selection_data.py \
  --hmi-workbook /path/to/HMI_data.xlsx \
  --ec-workbook /path/to/Raw_data.xlsx \
  --out work/selection_inputs

python code/expand_model_protocol.py \
  --base work/selection_inputs/bundle/cell_mean_protocol.json \
  --out work/nine_model_protocol.json

python code/run_strategy_selection.py \
  --data work/selection_inputs/bundle/cell_mean.csv \
  --manifest work/selection_inputs/bundle/manifest.csv \
  --protocol work/nine_model_protocol.json \
  --out work/nine_model_cell_mean
```

Use the author-released workbooks linked in the preparation guide, without
re-saving them. Very small response-value changes altered some model choices
in the numerical reproducibility check; matching record counts alone is not
sufficient to verify the inputs.

For original-response training, expand `original_response_protocol.json` into
its own nine-model protocol, then run with `original_response.csv` and a new
output directory. Use the unexpanded protocol to reproduce the three-model
reference. Fixed-model sensitivities and separate Cd/Cu/Pb cases are described
in the preparation guide.
Additional metal cases require HM2.xlsx and lawfully obtained Lee article XML.
The [Cd source-family selection check](docs/selection_data.md#separate-metal-cases)
uses those prepared inputs and withholds both Cui records together; it is a
retrospective sensitivity rather than a fourth main task.

The main inputs comprise 595 cells, 1,704 released records, and six source groups
across IBU and Sr. These are retrospective development data, not independent
experiment counts or a prospective test set. Models are selected within training
sources, never using the outer source's outcomes. Ties use the declared strategy
order; all source-specific outcomes, including failures, remain in the output.

## Model Sets and Sensitivities

The manuscript's [nine-model comparison](docs/model_expansion.md) adds Extra Trees,
gradient boosting, XGBoost, LightGBM, CatBoost and k-nearest neighbors, retaining
the original source splits and training-only model selection. The three-model
configuration remains available for reproducing the earlier comparison.

The [numerical sensitivity check](docs/numerical_sensitivity.md) crosses fitting
precision and seeds without changing evaluation observations. It reports all
settings, per-fold support, and both model-selection criteria; it does not choose
a favorable seed or treat numerical ranges as confidence intervals.

For an optional response-plausibility check after a nine-model run:

```bash
python code/audit_response_bounds.py \
  --run-dir work/nine_model_results \
  --out-dir work/response_bounds_cell_mean
```

The command counts predictions below zero or above the initial-concentration /
adsorbent-dose mass-balance limit, separately from observed-value exceptions.
It does not clip predictions, change candidate choices, or interpret cell counts
as independent-study failure rates. Run it separately for each training
representation.

To score a separate post hoc physical-bound sensitivity, place the two
nine-model outputs in `cell_mean/` and `original_response/` under one directory:

```bash
python code/audit_physical_projection.py \
  --runs-root work/nine_model_runs \
  --out-dir work/physical_projection
```

This projects saved outer model predictions to zero through the nominal
initial-concentration / dose limit, then rescores the unchanged candidate
grids. It reproduces the original outer losses before projection and keeps
training-selected strategies fixed. The result is a diagnostic, not a newly
trained constrained model.

To apply the same bound to inner validation before selecting a strategy, then
to outer predictions, use the saved runs and the previous projection output:

```bash
python code/audit_bounded_selection.py \
  --runs-root work/nine_model_runs \
  --bounds-dir work/physical_projection \
  --out-dir work/bounded_selection
```

This check reproduces the original inner scores and choices first. It may
change the training-selected strategy, but never uses held-out responses for
that choice. Clipping is post hoc and can create artificial ties; neither
command fits a physically constrained model.

## Feature Controls

To compare full, material-only, and condition-only inputs for all three fixed
models, use the same cell-mean inputs and source manifest:

```bash
python code/run_feature_control.py \
  --data work/selection_inputs/bundle/cell_mean.csv \
  --manifest work/selection_inputs/bundle/manifest.csv \
  --protocol work/selection_inputs/bundle/cell_mean_protocol.json \
  --out work/feature_control
```

All variants retain the complete condition vector for scoring and the same
training/test membership. The command does not choose a feature set using test
outcomes. Output includes source losses, predictions, equal-source task means,
and the executed protocol and manifest. Use a new output directory for each run.

## Contents

- `code/`: data preparation, metrics, and experiment runners.
- `config/`: fixed experiment and sensitivity settings.
- `data/`: source-row annotations and split definitions, without responses.
- `tests/`: input, split, model-selection, and metric tests.
- `docs/`: source provenance and execution instructions.

The MIT license applies to software, not the original scientific observations.
See [DATA_USE.md](DATA_USE.md) and [DATA_DICTIONARY.md](DATA_DICTIONARY.md).
