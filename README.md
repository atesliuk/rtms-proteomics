# rtms-proteomics
Analysis pipeline for pre-treatment serum proteomic prediction of rTMS/iTBS response in treatment-resistant depression (Tesliuk et al.).
# Pre-treatment serum proteomics predicts rTMS response in treatment-resistant depression

Analysis code for the manuscript:

> **Pre-treatment serum proteomic markers of response and remission to intermittent theta-burst stimulation in treatment-resistant depression: an exploratory study.**
> *Tesliuk A, Valiulis V, Kaupinis A, Germanavičius A, Navakauskienė R, Valiulienė G. Journal, year, and DOI to be added.*

This repository contains the complete, single-file Python pipeline that reproduces
every statistical result, figure, and supplementary table in the paper.

---

## What the pipeline does

Starting from a label-free serum proteomics abundance matrix (Waters PLGS/ISOQuant
export) and a clinical metadata table, `Pre_TMS_proteomics_pipeline.py` runs the
full analysis end to end:

1. **Quality control & imputation** — filters proteins by confidence and unique-peptide
   count (≥ 2), removes high-missingness proteins, and applies min-probability
   imputation.
2. **Covariate correction** — per-protein ordinary-least-squares residual correction
   for six medication-class indicators and body weight. *Season is not a covariate*;
   it is retained only for a PCA confounder check.
3. **PCA** — structure/confounder checks before and after correction.
4. **Group and continuous analyses** — Wilcoxon rank-sum (responders vs non-responders)
   and Spearman correlations with percent symptom reduction, plus a
   concordance-tightened Scenario A/B/C/D classification.
5. **Sex comparison.**
6. **UniProt annotation** of retained proteins (REST API, cached).
7. **Prediction models** — dual LASSO panels (HAMD- and MADRS-native), leave-one-out
   cross-validated comparison of clinical scores vs protein panel vs combined,
   single-marker ROC with bootstrap CIs, and a **nested cross-validation** that repeats
   the whole selection procedure inside each fold (optimism-corrected AUC).
8. **Outputs** — publication-quality TIFF figures and an Excel results workbook.

Response is evaluated in parallel on the **HAMD-17** and **MADRS** scales, with
remission and ≥ 50 %-reduction endpoints as sensitivity analyses. All findings are
exploratory: no protein survives Benjamini–Hochberg correction at this sample size.

---

## Repository structure

```
.
├── Pre_TMS_proteomics_pipeline.py   # the complete analysis (run this)
├── requirements.txt                 # Python dependencies
├── data/                            # place input files here (not distributed)
│   └── README.md                    # expected input format
├── output/                          # created on run (figures, results, reference, cache)
├── LICENSE
└── README.md
```

---

## Requirements

- Python ≥ 3.11 (developed under 3.14)
- Packages in `requirements.txt`:

```bash
python -m venv venv && source venv/bin/activate      # optional
pip install -r requirements.txt
```

Core libraries: numpy, pandas, scipy, statsmodels, scikit-learn, matplotlib,
seaborn, requests, openpyxl (reads the proteomics `.xlsx`), xlsxwriter (writes the
results workbook), adjustText.

---

## Input data

Two files are expected under `data/` (paths are set in the `CONFIGURATION` block
at the top of the script):

**1. `proteomics_export.xlsx`** — the PLGS/ISOQuant protein × sample matrix.
A two-row header is expected: annotation columns (`Accession`, `Description`,
`Confidence score`, `Unique peptides`, …) plus per-sample columns grouped under a
`Normalized abundance` super-header. Sample IDs must match `patient_id` in the
metadata.

**2. `metadata.csv`** (`;`-separated, `,` decimal) — one row per participant,
indexed by `patient_id`. Expected columns:

| Column | Meaning |
|---|---|
| `patient_id` | sample ID, matches the proteomics columns |
| `hamd_before`, `hamd_middle`, `hamd_after` | HAMD-17 at baseline/mid/post |
| `madrs_before`, `madrs_middle`, `madrs_after` | MADRS at baseline/mid/post |
| `sex` | `F` / `M` |
| `weight_kg` | body weight (kg) |
| `antidepressant`, `antipsychotic`, `mood_stabilizer`, `anxiolytic`, `lithium` | 0/1 medication-class indicators |
| `cardiovascular_med` | 0/1 (defaults to 0 if the column is absent) |
| `collection_month` | month of blood draw (1–12; used only for the PCA season check) |

Responder, remission and ≥ 50 %-reduction labels are derived automatically from the
score columns.

> **Data availability.** The patient-level clinical and proteomics data are not
> included here to protect participant privacy. They are available from the
> corresponding author on reasonable request, subject to the study's ethics
> approval (Vilnius Regional Biomedical Research Ethics Committee, 2019/11-1161-653).

---

## Running

```bash
python Pre_TMS_proteomics_pipeline.py
```

Outputs are written under `output/`:
- `output/figures/` — main and supplementary figures (300-dpi TIFF, plus a vector PDF copy of each)
- `output/results/complete_results.xlsx` — all statistics and per-protein sheets
- `output/results/nested_cv_auc.csv` — nested cross-validation AUCs
- `output/reference/uniprot_data.json` — UniProt annotation cache (allows offline re-runs)
- `output/cache/` — intermediate pickle/JSON caches (config, matrices, patient-number map)

The UniProt annotation step requires internet access on the first run; thereafter
the cached `output/reference/uniprot_data.json` is reused.

---

## Reproducibility

- All stochastic steps use a fixed seed (`seed = 42`; LASSO/logistic `random_state = 42`).
- Season is never used as a covariate in the OLS correction (six medication classes
  + body weight only), matching the manuscript.
- Re-running on the same inputs reproduces the published numbers exactly.

---

## Citing

If you use this code, please cite the manuscript above. You may also cite this
repository directly (see `CITATION` / the repository DOI once archived, e.g. via
Zenodo).

## License

Released under the terms in [LICENSE](LICENSE).

## Contact

Code and analysis: Anastasiia Tesliuk, Life Sciences Center, Vilnius University
anastasiia.tesliuk@bchi.stud.vu.lt. For data requests, contact the corresponding
author of the manuscript.
