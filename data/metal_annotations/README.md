# Additional Metal Comparisons

`rows.csv` records HM2 row membership, source-specific labels, experimental
series, dose, contact time, matrix, provenance tier, and training/test roles.
It does not contain measured capacities or the fitted material descriptors.
`metal_order` and `lead_order` preserve the input order used in the separate
executed comparisons, including shared Zama Pb rows.

The manifests contain explicit training/test IDs, including exclusion of both
Cui sources in the family sensitivity. They do not choose models from their
outer performance. The protocols specify fixed models and eligible training
tiers. The expanded tier includes condition-supported Wang records whose
figure-derived responses were not independently digitized.

The concentration reconstruction multiplies the original amount-per-mass field
by the source-specific dose and molar-mass factor. To reproduce the executed
inputs, the existing Cd concentration mappings retain 112.411 mg/mmol, whereas
Wang Cd and the Cd response conversion use 112.414 mg/mmol. Cu and Pb use 63.546
and 207.2 mg/mmol. No values are snapped to nominal concentration grids.

Lee et al. (DOI 10.1016/j.jenvman.2019.01.100) retains the legacy key `Kim2019`.
Its descriptors and experimental capacities are read from Tables 1 and 2 of
user-supplied article XML. The adjacent fitted kinetic capacities are not used.
The three Pb rows are preserved but not evaluated because the GB value exceeds
the nominal mass-balance bound; the Cu rows are test-only.

Reconstruction requires HM2.xlsx from Zhu et al., DOI
10.1016/j.jhazmat.2019.06.004, and lawful access to the Lee source XML. No article
full text is distributed. Original data terms still apply to generated outputs.
