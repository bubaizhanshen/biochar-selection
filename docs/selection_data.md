# Preparing the Selection Inputs

The current six-source comparison uses two published compilation workbooks:

- `HMI_data.xlsx`: Jaffari et al., *Transformer-based deep learning models for
  adsorption capacity prediction of heavy metal ions toward biochar-based
  adsorbents*, Journal of Hazardous Materials 462 (2024), 132773.
  DOI: [10.1016/j.jhazmat.2023.132773](https://doi.org/10.1016/j.jhazmat.2023.132773).
- `Raw_data.xlsx` (named `EC.xlsx` in the local analysis): Jaffari et al., *Machine-learning-based prediction and optimization
  of emerging contaminants' adsorption capacity on biochar materials*, Chemical
  Engineering Journal 466 (2023), 143073.
  DOI: [10.1016/j.cej.2023.143073](https://doi.org/10.1016/j.cej.2023.143073).

Obtain the released data tables associated with these articles separately.
The author repositories are
[Transformer_models_prediction](https://github.com/ZeeshanHJ/Transformer_models_prediction)
(`HMI_data.xlsx`) and
[Adsorption-capacity-prediction-for-ECs](https://github.com/ZeeshanHJ/Adsorption-capacity-prediction-for-ECs)
(`Raw_data.xlsx`). The `--ec-workbook` argument accepts either filename.
The reader strips header whitespace and maps the released HMI headers
`inorganics`, `g/L`, and `(O+N/C)` to the analysis headers `Metal type`,
`Dosage(g/L)`, and `(O+N)/C`. This changes column names only, not values or
row order; ambiguous duplicate headers are rejected.
No source workbooks are included in this package; original source terms apply. Do not use a rearranged, filtered, or independently
reformatted workbook with the row-based mappings. The mapping locators depend
on the original sheet and row order. No publisher credentials or downloaded
article full text are needed by the preparation command.

From the repository root:

```bash
python code/prepare_selection_data.py \
  --hmi-workbook /path/to/HMI_data.xlsx \
  --ec-workbook /path/to/Raw_data.xlsx \
  --out work/selection_inputs
```

Use a new or empty output directory. The command reads the six source
annotations, reconstructs source copies, applies documented unit and condition
mappings, aggregates cells, builds source holdouts, and attaches the original
response lists. Its `bundle/` directory is the input to the model runners:

```bash
python code/run_strategy_selection.py \
  --data work/selection_inputs/bundle/cell_mean.csv \
  --manifest work/selection_inputs/bundle/manifest.csv \
  --protocol work/selection_inputs/bundle/cell_mean_protocol.json \
  --out work/selection_cell_mean
```

For original-response training, use `original_response.csv` and
`original_response_protocol.json` from the same bundle. The evaluation cells
and holdout membership remain unchanged. No test responses are expanded into
training data.

The final inputs contain 595 cells linked to 1,704 released records and six
source holdouts. Preparation also retains intermediate CBZ and excluded rows;
the 90 CBZ cells do not enter the experiment because no second reviewed CBZ
source is available for training. Record counts do not establish independent
replicate counts. Model results remain retrospective development evidence.

Generated intermediate files contain source-derived observations. Keep them
local unless their redistribution is permitted under the source terms. This
command covers the six-source IBU/Sr experiment, not the separate metal-case
analyses or historical benchmarks.

## Numerical Reproducibility

Use the author-released workbooks, not locally re-saved copies. In a controlled
comparison, response differences below 1e-12 between released and re-saved files
changed some Random Forest predictions and the model selected by inner validation.
Agreement of input values within a tolerance therefore did not guarantee identical
material recommendations. Fixed seeds alone did not remove this sensitivity.

Preserve the released numeric values, record the source version, and use the
documented software versions. Do not round responses to recover a preferred
outcome. Compare source-level model choices as well as aggregate scores when
checking a reproduction. These differences concern numerical computation, not
experimental measurement uncertainty.

For an exact-byte check of the released copies used here, the SHA-256 digests
are:

```text
2074a592c184aa680f321d5d28f4c8371dba1c249c915a67352cc7e2e4fc725d  HMI_data.xlsx
bbcb3e6b89b5186770a25cc89479a819680200895412f52694bc1de13f19a115  Raw_data.xlsx
9b7a19fa115075448a7497187c991cd7035e971a2ec6e31930dd7d96d014ff2b  bundle/cell_mean.csv
```

The last digest is for the prepared file under the pinned Python dependencies.
These fingerprints identify the input version; they are not evidence of
experimental independence or measurement accuracy.

## Separate Metal Cases

The additional Cd/Cu/Pb cases use `HM2.xlsx` from Zhu et al., *The application
of machine learning methods for prediction of metal sorption onto biochars*,
Journal of Hazardous Materials 378 (2019), 120727, DOI
[10.1016/j.jhazmat.2019.06.004](https://doi.org/10.1016/j.jhazmat.2019.06.004).
They also require user-supplied article XML for Lee et al., DOI
[10.1016/j.jenvman.2019.01.100](https://doi.org/10.1016/j.jenvman.2019.01.100),
to extract the experimental response and characterization tables. The article
file is not bundled or downloaded automatically.

```bash
python code/prepare_metal_selection_data.py \
  --workbook /path/to/HM2.xlsx \
  --lee-source-xml /path/to/lee_article.xml \
  --out work/metal_inputs

python code/run_selection_benchmark.py \
  --data work/metal_inputs/metal_input.csv \
  --manifest work/metal_inputs/metal_manifest.csv \
  --protocol work/metal_inputs/metal_protocol.json \
  --out work/metal_results
```

Use `family_manifest.csv` with the same metal input/protocol for the Cui-family
exclusion sensitivity. For the two-source Pb comparison, use `lead_input.csv`,
`lead_manifest.csv`, and `lead_protocol.json`. Each run needs its own empty
output directory. These comparisons have different training pools and descriptors
from the six-source IBU/Sr experiment and must not be pooled with its folds.
See [metal annotations](../data/metal_annotations/README.md) for conversion
constants, provenance tiers, row ordering, and the Lee/Kim legacy-key mapping.

The exploratory Cd source-family selection check uses the same prepared metal
input. It withholds both Cui records together and leaves out Gao or Zama as the
other test families. The Wang rows enter only expanded-tier fitting, never
validation or testing. It compares three- and nine-model choices made by inner
MAE or selection loss, with an additional model-plus-property-rule comparison:

```bash
python code/run_cd_family_selection.py \
  --data work/metal_inputs/metal_input.csv \
  --protocol work/metal_inputs/metal_protocol.json \
  --out work/cd_family_selection
```

This is a retrospective sensitivity, not an independent test or a new main
analysis fold. The output directory contains executed split membership,
condition and family scores, model predictions, decisions, and input hashes.

## Fixed-Model Sensitivities

The four training policies for the six-source fixed-model comparison use
`code/run_selection_benchmark.py`, not the model-choice runner:

| Policy | Bundle input | Protocol |
|---|---|---|
| Cell means | `cell_mean.csv` | `bundle/cell_mean_protocol.json` |
| Repeated cell means | `cell_mean.csv` | `config/selection_repeated_means.json` |
| Original responses | `original_response.csv` | `bundle/original_response_protocol.json` |
| Within-model tuning | `cell_mean.csv` | `config/selection_tuned.json` |

Use the same `bundle/manifest.csv` for all four policies and a new output
directory for each run. For example:

```bash
python code/run_selection_benchmark.py \
  --data work/selection_inputs/bundle/cell_mean.csv \
  --manifest work/selection_inputs/bundle/manifest.csv \
  --protocol config/selection_repeated_means.json \
  --out work/selection_repeated_means
```

Repeated means change training frequency weights but do not restore the original
within-cell response variation. Original-response training restores the linked
responses for training only. Within-model tuning selects parameters separately
for each algorithm using mean validation-source MAE over all validation cells;
it does not choose the algorithm with the lowest outer loss. Its inner scoring
population differs from the complete-condition, series-balanced population of
the paired model-selection-objective experiment. Do not interpret these four
policies as an objective-only comparison.
