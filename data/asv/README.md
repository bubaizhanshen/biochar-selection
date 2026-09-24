# As(V) Analysis Input

`analysis_input.csv` is the screened input used for the exploratory study-block
holdout As(V) comparison. It contains 138 records from nine study blocks. The
table is source-derived; it is not a set of 138 independent experiments. The
software license in this repository does not grant a new license to the
original source material. Cite the original sources listed below when reusing the data.

## Source and Screening

Su et al. compiled 477 records from 26 studies in Supporting Information Table
S2. Direct extraction yielded 476 numerical rows; the one-row discrepancy was
not reconciled. Candidate study blocks were screened against the cited primary
reports and the Alchouron thesis. Retention required a source-supported As(V)
endpoint, reported biochar descriptors, and all fields needed for exact
condition matching: initial As concentration, temperature, pH, and dose. The
compilation did not provide contact time, solution matrix, or replicate
identifiers, so these fields could not be matched.

The screened table contains 108 records from eight source studies. Twelve
Alchouron responses in that set are replaced by means digitized from Figure
3.2 of the doctoral thesis. The compiled Sun records were excluded because
they did not distinguish As(III) from As(V); 30 As(V) means were instead
digitized from Figure 3 of the original Sun article. Two activated-carbon
profiles in the Tan panel were removed, leaving its three biochars. Su Table
S2 labels 38 and 44 refer to the same study and were not counted as separate
blocks. Label 34 was excluded because its concentration basis could not be
reconciled; label 41 was excluded because its pH grid conflicted with the
primary report. These Su Table S2 labels are source identifiers, not this
manuscript's reference numbers.

| Su Table S2 label | Source and source record | Role in this analysis |
| --- | --- | --- |
| `[24]` | Tan et al., DOI: [10.1016/j.cherd.2020.05.011](https://doi.org/10.1016/j.cherd.2020.05.011) | Test panel; activated-carbon profiles excluded |
| `[25]` | Cha et al., DOI: [10.1016/j.chemosphere.2021.130521](https://doi.org/10.1016/j.chemosphere.2021.130521) | Test panel |
| `[27]` | Lata et al., DOI: [10.1007/s11356-019-06300-w](https://doi.org/10.1007/s11356-019-06300-w) | Training only |
| `[30]` | Fan et al., DOI: [10.1155/2018/5137694](https://doi.org/10.1155/2018/5137694) | Test panel |
| `[33]` | He et al., DOI: [10.1016/j.scitotenv.2017.09.016](https://doi.org/10.1016/j.scitotenv.2017.09.016) | Test panel |
| `[35]` | Jin et al., DOI: [10.1016/j.biortech.2014.06.103](https://doi.org/10.1016/j.biortech.2014.06.103) | Training only |
| `[39]` | Alchouron et al., DOI: [10.1016/j.scitotenv.2019.135943](https://doi.org/10.1016/j.scitotenv.2019.135943); thesis Figure 3.2, [repository record](http://ri.agro.uba.ar/greenstone3/library/collection/tesis/document/2022alchouronjacinta) | Test panel; 12 responses digitized from thesis |
| `[40]` | Sun et al., DOI: [10.1016/j.seppur.2022.120836](https://doi.org/10.1016/j.seppur.2022.120836) | Test panel; 30 As(V) responses digitized from Figure 3 |
| `[46]` | Lee et al., DOI: [10.1016/j.chemosphere.2021.132179](https://doi.org/10.1016/j.chemosphere.2021.132179) | Test panel |

Su et al.'s compilation: [10.1021/acsestwater.5c00339](https://doi.org/10.1021/acsestwater.5c00339).
The source-level screening procedure and study-block analysis are also described
in Text S7 of the manuscript Supporting Information.

## Figure-Derived Responses

For Alchouron, bar heights in thesis Figure 3.2 (PDF page 75) were calibrated
against the labeled removal-percentage ticks. Capacity was calculated as
`q (mg/g) = C0 (mg/L) / dose (g/L) * removal fraction`; the reported conditions
were C0 = 10 mg/L, dose = 2 g/L, and pH 5, 7, or 9. The nominal coordinate
reading uncertainty is +/-1 pixel. For Sun, As(V) marker centers in Figure 3a-c
were calibrated against the labeled capacity axis; nominal coordinate-reading
uncertainty is +/-3 pixels. Plotted error bars were not digitized. These
uncertainties do not include replicate-level or other figure-extraction error.

## Material Grouping

`material_group` is an analytical, source-local key created by exact matching
the eight reported descriptors `C_pct`, `H_pct`, `O_pct`, `Fe_pct`, `H_C`,
`O_C`, `ON_C`, and `area_m2_g` within each study block. Where the original
reports provided a material label, it is retained in `source_material_label`;
the source labels for Tan, Alchouron, and Sun were checked against their source
reports. The literal value `nan` in `source_material_label` indicates that no
label was available in the compiled table. Descriptor-profile matches do not
verify physical material identity, production batch, or independence. The
analysis uses these groups only to define candidate profiles within the
available source tables.

## Column Dictionary

| Column | Meaning and unit |
| --- | --- |
| `reference` | Source identifier from Su et al.'s Table S2; use the crosswalk above |
| `source_material_label` | Reported source label where available; literal `nan` means not supplied in the compilation |
| `source_rows_origin` | `Su2025_SI_Table_S2`, `Alchouron_thesis_Figure_3.2_digitized`, or `Sun2022_Figure_3_digitized_AsV` |
| `C_pct`, `H_pct`, `O_pct`, `Fe_pct` | Reported elemental contents, percent |
| `H_C`, `O_C`, `ON_C` | Reported elemental ratios; `ON_C` is `(O+N)/C` |
| `area_m2_g` | Specific surface area, m²/g |
| `As_initial_mg_L` | Initial As(V) concentration, mg/L |
| `temperature_C` | Reaction temperature, °C |
| `pH` | Initial pH as reported |
| `dose_g_L` | Biochar dose, g/L |
| `capacity_mg_g` | Recorded or figure-derived As(V) capacity, mg/g |
| `material_group` | Source-local exact descriptor-profile key; not a verified physical-material ID |

## Reproduction

Install the model dependencies with `pip install -r requirements-models.txt`.
From the repository root, run:

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

The configuration identifies seven test panels and two training-only source
blocks. Outputs are written to the requested directories; the input file is not
copied into them. This is a small, retrospective, post-screen analysis, not a
prospective validation or a population-level estimate of arsenic-model
performance. The last two runs reproduce the paired Table S8 sensitivity; only
the Alchouron outer-test rows from the first run and the Sun outer-test rows
from the second are used for that comparison.
