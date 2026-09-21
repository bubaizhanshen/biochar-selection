# Source Annotations

The six CSVs identify workbook rows and record reviewed series, material-label,
time, and protocol mappings. Measured capacities, concentration columns, and
material-descriptor columns are not copied into these files. They must be read
from the original workbooks. These annotations do not establish independent
experimental batches or replicate identities.

`source_row_id` is the zero-based data-row index from `pandas.read_excel` with
its default first-sheet and header settings. It is not the printed Excel row
number. Preserve workbook row order, sheet layout, and headers. The alkaline
mapping includes other source-review pollutants; the preparation code selects
Sr from the workbook rather than using those rows in the Sr experiment.

Intermediate flags retain their original review wording. In particular, a
negative-response review flag is not a final exclusion rule. Final preparation
retains signed responses and applies the documented source-protocol exclusions.

Sources for the mappings:

- Shin2021_Alkaline: 10.1016/j.envres.2021.111346
- Shin2021_Magnetic: 10.1016/j.jece.2021.105119
- Shin2022_Postmodification: 10.1016/j.jhazmat.2022.129081
- Shin2020_Micropollutants: 10.1016/j.jhazmat.2020.123102
- Shin2021_Competitive: 10.1016/j.envpol.2020.116244
- Shin2022_Ibuprofen: 10.1016/j.jece.2022.107914

For the competitive IBU source, Methods and Table 2 identify micromolar solution
concentrations and micromoles per gram. Multiplication by 0.20628 converts them
to mg/L and mg/g. This reviewed conversion does not require distributing article
XML. It preserves differences between released 24-hour means and the article's
Table 2 equilibrium values; it does not adjust observations to force agreement.

See [data preparation](../../docs/selection_data.md) for workbook provenance and
the reconstruction command. Source-data terms remain separate from software
licensing.
