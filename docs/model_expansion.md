# Nine-model comparison

The extension adds Extra Trees, gradient boosting regression, XGBoost, LightGBM,
CatBoost and distance-weighted k-nearest neighbors to Ridge, SVR and Random Forest.
These families have been used in biochar adsorption modeling, including Jaffari
et al., Chemical Engineering Journal 466 (2023), 143073
(https://doi.org/10.1016/j.cej.2023.143073).

This is a retrospective extension on previously examined data, not independent
validation. All families use the same source splits, predictors, training-only
preprocessing, candidate sets and condition-level scoring. CatBoost receives the
same one-hot encoded inputs as the other models; native categorical processing
is not used. There is one fixed configuration per family, specified in
`config/model_expansion.json`, with no test-based early stopping or tuning.
This compares the declared configurations, not the best attainable performance
of each algorithm. Original three-model parameters and tie priorities are retained.

Install the optional dependencies and prepare a data bundle as described in the
main README. Then, for either `cell_mean` or `original_response`:

```bash
python -m pip install -r requirements-models.txt
python code/expand_model_protocol.py \
  --base work/selection_inputs/bundle/cell_mean_protocol.json \
  --out work/nine_model_protocol.json
python code/run_strategy_selection.py \
  --data work/selection_inputs/bundle/cell_mean.csv \
  --manifest work/selection_inputs/bundle/manifest.csv \
  --protocol work/nine_model_protocol.json \
  --out work/nine_model_results
```

Every source produces predictions for all nine models and both simple rules.
Inner validation chooses a model by MAE or selection loss, or chooses among
models and rules by selection loss. Declared-order ties are resolved without
outer responses. `outer_results.csv` and `outer_conditions.csv` retain all
strategies, including those not selected. MAE is left missing for random-choice
and property rules because their scores are not predicted adsorption amounts.
Use their selection losses only. Rule scores retained in `predictions.csv` are
selection scores, not response predictions.

For the full fitting-precision and random-seed check, see
[numerical sensitivity](numerical_sensitivity.md).
