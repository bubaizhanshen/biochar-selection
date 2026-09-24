# As(V) source-held-out comparison

This is a separate exploratory branch. It does not alter the primary IBU, Sr,
or Cd results. The analysis compares nested selection among nine fixed model
families with selection among seven single-property rules. A complete source
block is withheld for testing; the remaining eligible source blocks are used
for inner selection and fitting.

## Reconstruct the input

The runner accepts a CSV with these columns:

| Column | Meaning |
| --- | --- |
| `reference` | Source identifier as cited in Su et al.'s Table S2, including brackets |
| `material_group` | Source-specific biochar label; labels must not be merged across sources |
| `C_pct`, `H_pct`, `O_pct`, `Fe_pct` | Reported elemental contents (%) |
| `H_C`, `O_C`, `ON_C` | Reported elemental ratios; `ON_C` denotes (O+N)/C |
| `area_m2_g` | Specific surface area (m²/g) |
| `As_initial_mg_L` | Initial arsenic concentration (mg/L) |
| `temperature_C` | Reaction temperature (°C) |
| `pH` | Initial pH as reported |
| `dose_g_L` | Biochar dose (g/L) |
| `capacity_mg_g` | Recorded or explicitly figure-digitized As(V) capacity (mg/g) |

The source IDs in the configuration map to the SI source list as follows:

| Su et al. Table S2 | Source used here | Role |
| --- | --- | --- |
| `[24]` | Tan et al. | Test panel |
| `[25]` | Cha et al. | Test panel |
| `[27]` | Lata et al. | Training only |
| `[30]` | Fan et al. | Test panel |
| `[33]` | He et al. | Test panel |
| `[35]` | Jin et al. | Training only |
| `[39]` | Alchouron et al. | Test panel; 12 compiled responses are replaced with thesis-figure means |
| `[40]` | Sun et al. | Test panel; compiled rows are replaced with 30 As(V)-specific figure means |
| `[46]` | Lee et al. | Test panel |

Use the source references in Text S7 for full bibliographic details. The source
compilation reports 477 records; a direct extraction of Table S2 yielded 476
numeric rows, and the discrepancy was not resolved. The table does not identify
arsenic species, contact time, solution matrix, or replicate IDs. Apply the
source-screening and exact-condition rules in Text S7 before constructing the
input. Do not infer species or material identity from a property fingerprint.

The code repository intentionally does not redistribute the 138 source-derived
response values. Reconstruct the table from the cited source SI and papers, and
retain the source and figure provenance for every value. Figure-derived means
do not include digitization or replicate-level uncertainty.

## Run

Install the pinned model dependencies with `pip install -r requirements-models.txt`.

```bash
python code/run_asv_source_selection.py \
  --data /path/to/asv_source_screened.csv \
  --predictor-set asv_specific \
  --out work/asv_full_features

python code/run_asv_source_selection.py \
  --data /path/to/asv_source_screened.csv \
  --predictor-set shared \
  --out work/asv_shared_features

python code/run_asv_source_selection.py \
  --data /path/to/asv_source_screened.csv \
  --predictor-set asv_specific \
  --exclude-training-only \
  --out work/asv_without_training_only_sources
```

The `shared` predictor set uses C, H, O, surface area, and the four condition
variables used in the primary IBU/Sr/Cd models. The `asv_specific` set also uses
Fe and the reported elemental ratios. Both sets are compared with the same
seven property rules. `--exclude-training-only` removes the Lata and Jin blocks
from model fitting while keeping the seven test panels unchanged.

Each run writes source-level and condition-level derived results, inner
validation scores, selected strategies, panel-support counts, summary tables,
and a protocol record. It does not copy the supplied input table into the
output directory. The reported means weight source blocks equally after
averaging conditions within each source. A condition with fewer than two
biochars is not scored. The three-biochar/three-condition subset is descriptive,
not an inferential sufficiency threshold. This analysis is retrospective,
small-source, and post-screen; it is not a prospective or population-level
estimate.
