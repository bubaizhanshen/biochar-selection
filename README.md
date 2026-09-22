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
pip install -r requirements.txt
make test
```

## Prepare and Run

```bash
python code/prepare_selection_data.py \
  --hmi-workbook /path/to/HMI_data.xlsx \
  --ec-workbook /path/to/Raw_data.xlsx \
  --out work/selection_inputs

python code/run_strategy_selection.py \
  --data work/selection_inputs/bundle/cell_mean.csv \
  --manifest work/selection_inputs/bundle/manifest.csv \
  --protocol work/selection_inputs/bundle/cell_mean_protocol.json \
  --out work/selection_cell_mean
```

Use the author-released workbooks linked in the preparation guide, without
re-saving them. Very small response-value changes altered some model choices
in the numerical reproducibility check; matching record counts alone is not
sufficient to verify the inputs.

For original-response training, substitute `original_response.csv` and
`original_response_protocol.json` and use a new output directory. Fixed-model
sensitivities and separate Cd/Cu/Pb cases are described in the preparation guide.
Additional metal cases require HM2.xlsx and lawfully obtained Lee article XML.

The main inputs comprise 595 cells, 1,704 released records, and six source groups
across IBU and Sr. These are retrospective development data, not independent
experiment counts or a prospective test set. Models are selected within training
sources, never using the outer source's outcomes. Ties use the declared strategy
order; all source-specific outcomes, including failures, remain in the output.

## Additional Models

The [nine-model comparison](docs/model_expansion.md) adds Extra Trees,
gradient boosting, XGBoost, LightGBM, CatBoost and k-nearest neighbors, retaining
the original source splits and training-only model selection. The three-model
configuration remains available for reproducing the earlier comparison.

The [numerical sensitivity check](docs/numerical_sensitivity.md) crosses fitting
precision and seeds without changing evaluation observations. It reports all
settings, per-fold support, and both model-selection criteria; it does not choose
a favorable seed or treat numerical ranges as confidence intervals.

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
