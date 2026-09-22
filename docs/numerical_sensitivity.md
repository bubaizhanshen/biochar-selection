# Numerical Sensitivity

This check changes only fitting responses and model seeds. It keeps released
validation/test observations, source splits, predictors, candidate conditions,
and aggregation unchanged. The four precisions are released values and values
rounded to 12, 8, or 4 decimal places, crossed with seeds 1729 through 1733.
These are numerical perturbations, not measurement-error estimates.

Prepare the bundle and nine-model protocol using the README and model-extension
instructions. Then run each precision for each training representation:

```bash
for policy in cell_mean original_response; do
  python code/expand_model_protocol.py \
    --base work/selection_inputs/bundle/${policy}_protocol.json \
    --out work/${policy}_nine_models.json
  for precision in released 12 8 4; do
    python code/run_numerical_sensitivity.py \
      --data work/selection_inputs/bundle/${policy}.csv \
      --manifest work/selection_inputs/bundle/manifest.csv \
      --protocol work/${policy}_nine_models.json \
      --training-policy "$policy" --precision "$precision" \
      --out work/stability/${policy}_${precision}
  done
done
python code/summarize_numerical_sensitivity.py \
  --runs work/stability/cell_mean_* work/stability/original_response_* \
  --out work/stability_summary
python code/check_numerical_reference.py \
  --summary work/stability_summary/task_stability.csv
```

Use new output paths; completed or partial runs are never overwritten. Each
setting needs one CPU; independent precision/representation runs can be scheduled
separately. Install `requirements-models.txt` for the nine-model configuration.
Use `--seeds` for a small smoke test, but the summary command requires the full
declared grid and rejects missing or duplicate settings. An optional
`--reference /path/to/run_strategy_selection_output` checks released precision
and seed 1729 against a previous run; it is not an external-validation option.

## Outputs

- `inner_scores.csv`: per-source fitting and validation errors.
- `outer_scores.csv`: all fixed-strategy outer losses and response MAE.
- `decisions.csv`: training-selected strategies, exact score ties, and outer
  results. Tied strategies' outer losses are diagnostic, not used for selection.
- `candidate_probabilities.csv`: uniform probabilities among predicted maxima,
  using an absolute prediction tie tolerance of 1e-12.
- `conditions.csv`: complete matched-condition metrics; incomplete conditions
  remain excluded from scoring without being removed from training. For rules,
  the legacy condition-level `mae` field compares unlike score scales and must
  not be interpreted as prediction error; the outer/inner score tables mask it.
- `support.csv`: material counts, centered descriptor rank, and unseen categories.
- `contract.json` and `completed.json`: settings, input identities, and completion.

The summary gives source-level ranges, selected-model frequencies, and task means
with equal source weights. All settings are included. A three-model reference is
extracted from the same predictions as the nine-model comparison. Ranges are not
confidence intervals, and 20 settings are not 20 independent experiments.

Both cell-mean and original-response checks showed changes in the sign of the
MAE-versus-selection-loss comparison. Do not select a seed or precision to report
the largest apparent improvement. Surface-area and random rules have no response
MAE; their score scale is not mg/g.

`support.csv` computes descriptor rank after a reference-row shift and centering
to avoid floating-point cancellation. This diagnostic does not change model
preprocessing. Original-response expansion precedes rounding; validation and test
cell means are never rounded by this check.

The small reference CSV contains eight aggregate computational ranges, not raw
experimental responses. Matching it checks numerical reproduction in the stated
environment, not scientific generalization. A mismatch must be investigated;
do not change input precision or seeds to force a favorable match.
