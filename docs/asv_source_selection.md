# As(V) Study-Block Holdout Comparison

This exploratory analysis is separate from the primary IBU, Sr, and Cd
comparisons. It compares nested selection among nine fixed model families with
selection among seven single-property rules. Each test study block is omitted
from fitting; model and rule choices are made by inner
leave-one-study-block-out validation using the remaining eligible blocks.

The analysis input is included at
[`data/asv/analysis_input.csv`](../data/asv/analysis_input.csv). The screening
steps, source crosswalk, digitization details, column definitions, and limits on
the material grouping are documented in
[`data/asv/README.md`](../data/asv/README.md). In brief, the file contains 138
source-derived records across nine study blocks, including figure-digitized
values. Its descriptor-profile groups are analytical keys, not verified
physical-material or batch identifiers. The branch is retrospective and does
not constitute prospective validation.

## Run

Install the pinned model dependencies with `pip install -r requirements-models.txt`,
then run from the repository root:

```bash
python code/run_asv_source_selection.py \
  --data data/asv/analysis_input.csv \
  --predictor-set asv_specific \
  --out work/asv_full_features

python code/run_asv_source_selection.py \
  --data data/asv/analysis_input.csv \
  --predictor-set shared \
  --out work/asv_shared_features

python code/run_asv_source_selection.py \
  --data data/asv/analysis_input.csv \
  --predictor-set asv_specific \
  --exclude-training-only \
  --out work/asv_without_training_only_sources

python code/run_asv_source_selection.py \
  --data data/asv/analysis_input.csv \
  --predictor-set shared \
  --exclude-training-only \
  --out work/asv_without_training_only_shared_features

# Exclude Sun [40] from training when evaluating Alchouron [39].
python code/run_asv_source_selection.py \
  --data data/asv/analysis_input.csv \
  --predictor-set asv_specific \
  --exclude-training-source "[40]" \
  --out work/asv_without_sun_training

# Exclude Alchouron [39] from training when evaluating Sun [40].
python code/run_asv_source_selection.py \
  --data data/asv/analysis_input.csv \
  --predictor-set asv_specific \
  --exclude-training-source "[39]" \
  --out work/asv_without_alchouron_training
```

The `shared` predictor set uses C, H, O, surface area, and the four condition
variables used in the primary IBU/Sr/Cd models. The `asv_specific` set also uses
Fe and the reported elemental ratios. Both are compared with the same seven
property rules. `--exclude-training-only` removes the Lata and Jin blocks from
model fitting while keeping the seven test panels unchanged. The final two
commands reproduce the paired Table S8 sensitivity: use the Alchouron [39]
outer-test rows from the first run and the Sun [40] outer-test rows from the
second; each run excludes the other digitized source from training.

Each run writes study-block-level and condition-level results, inner-validation
scores, selected strategies, support counts, summary tables, and a protocol
record. The reported means weight study blocks equally after averaging
conditions within each source. A condition with fewer than two candidate
profiles is not scored. The three-profile/three-condition subset is
descriptive, not an inferential sufficiency threshold. This analysis is small,
retrospective, and post-screen; it is not a prospective or population-level
estimate.
