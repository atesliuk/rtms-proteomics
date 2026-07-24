"""
Pre-treatment serum proteomics pipeline for predicting rTMS response in
treatment-resistant depression.

Analysis code accompanying the manuscript "Pre-treatment serum proteomic markers
of response and remission to intermittent theta-burst stimulation in
treatment-resistant depression: an exploratory study."

Given a label-free proteomics abundance matrix (Waters PLGS/ISOQuant export) and
a clinical metadata table, the script runs the full analysis:
  1. Quality filtering and min-probability imputation of the protein matrix.
  2. Per-protein OLS covariate correction (six medication-class indicators and
     body weight). Season is NOT a covariate; it is retained only for a PCA
     confounder check.
  3. PCA structure/confounder checks before and after correction.
  4. Group comparisons (Wilcoxon rank-sum) and continuous-outcome correlations
     (Spearman), with a concordance-tightened Scenario A/B/C/D classification.
  5. Sex comparison.
  6. UniProt REST annotation of the retained proteins (cached).
  7. Dual LASSO panel selection (HAMD- and MADRS-native), leave-one-out
     cross-validated model comparison (clinical scores vs panel vs combined),
     single-marker ROC analysis, and a nested cross-validation that repeats the
     whole feature-selection procedure inside each fold (optimism-corrected AUC).
  8. Publication-quality figures and an Excel results workbook.

Response is defined in parallel on the HAMD-17 and MADRS scales, with remission
and >=50%-reduction endpoints as sensitivity analyses. All findings are
exploratory (no protein survives BH-FDR at this sample size).

Inputs (see the CONFIGURATION block and README): MS_FILE (proteomics export,
.xlsx) and META_FILE (clinical metadata, .csv). Outputs are written under
output/. Random seeds are fixed (seed=42). The UniProt annotation step requires
internet access; its results are cached under output/reference/uniprot_data.json.
"""

import os, re, json, time, warnings, subprocess, sys
import numpy  as np
import pandas as pd
import matplotlib
import matplotlib.pyplot   as plt
import matplotlib.patches  as mpatches
import matplotlib.ticker   as mticker
import matplotlib.gridspec as gridspec
import seaborn             as sns
import statsmodels.api     as sm
import requests

from scipy                               import stats
from scipy.stats                         import pearsonr
from statsmodels.stats.multitest         import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition               import PCA
from sklearn.linear_model               import LassoCV, LogisticRegression
from sklearn.preprocessing              import StandardScaler
from sklearn.metrics                    import roc_curve, roc_auc_score
from sklearn.model_selection            import LeaveOneOut

warnings.filterwarnings("ignore")
matplotlib.rcParams["pdf.fonttype"] = 42

try:
    import adjustText
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "adjustText", "--break-system-packages", "-q"], check=False)
    try:
        import adjustText
    except ImportError:
        adjustText = None

pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", "{:.4f}".format)

# ============================================================================
# CONFIGURATION
# ============================================================================

MS_FILE   = "data/proteomics_export.xlsx"  # Waters PLGS/ISOQuant label-free export (.xlsx)
MS_SHEET  = "Protein_data"                 # worksheet holding the protein x sample matrix
META_FILE = "data/metadata.csv"
META_SEP  = ";"
META_DEC  = ","

# Adjust the lables accordingly
NORM_LABEL      = "Normalized abundance"
RAW_LABEL       = "Raw abundance"
CONFIDENCE_COL  = "Confidence score"
PEPTIDE_COL     = "Unique peptides"
ACCESSION_COL   = "Accession"
DESCRIPTION_COL = "Description"

MIN_PEPTIDES     = 2
MAX_MISSING_FRAC = 0.50
SAMPLE_MISS_WARN = 0.30

# Primary responder definitions (absolute drop in points)
HAMD_RESPONDER_THRESHOLD  = -7
MADRS_RESPONDER_THRESHOLD = -10

# Remission cut-offs (post-treatment score)
HAMD_REMISSION_CUTOFF  = 7
MADRS_REMISSION_CUTOFF = 10

# Standard >=50% reduction for sensitivity analysis
PCT_RESPONSE_THRESHOLD = 50.0

# Target sensitivity for operating-point reporting
TARGET_SENSITIVITY = 0.80

NOMINAL_P          = 0.05
BORDERLINE_P       = 0.20
FOLD_CHANGE_THRESH = 1.5
MIN_GROUP_SIZE     = 3
MIN_PAIRS_SPEARMAN = 5

VIF_THRESHOLD            = 5
R2_SUBSAMPLE_N           = 200
NL_WEIGHT_R_THRESHOLD    = 0.15
SENSITIVITY_DELTA_THRESH = 0.10

LASSO_TOP_N        = 50
LASSO_N_ALPHAS     = 100
LASSO_MAX_ITER     = 10000
ROC_TOP_N_FALLBACK = 10
BOOTSTRAP_N        = 1000
AUC_TIER1          = 0.75
AUC_TIER2          = 0.70

TOP_N_VOLCANO_LABELS = 20
TOP_N_TABLE          = 15
TOP_N_CANDIDATES     = 15
FIG_DPI              = 300

for d in ["output/cache","output/results","output/figures","output/reference"]:
    os.makedirs(d, exist_ok=True)

_cfg = dict(
    HAMD_RESPONDER_THRESHOLD=HAMD_RESPONDER_THRESHOLD,
    MADRS_RESPONDER_THRESHOLD=MADRS_RESPONDER_THRESHOLD,
    HAMD_REMISSION_CUTOFF=HAMD_REMISSION_CUTOFF,
    MADRS_REMISSION_CUTOFF=MADRS_REMISSION_CUTOFF,
    PCT_RESPONSE_THRESHOLD=PCT_RESPONSE_THRESHOLD,
    TARGET_SENSITIVITY=TARGET_SENSITIVITY,
    NOMINAL_P=NOMINAL_P, BORDERLINE_P=BORDERLINE_P,
    FOLD_CHANGE_THRESH=FOLD_CHANGE_THRESH,
    MIN_GROUP_SIZE=MIN_GROUP_SIZE, MIN_PAIRS_SPEARMAN=MIN_PAIRS_SPEARMAN,
    MIN_PEPTIDES=MIN_PEPTIDES, MAX_MISSING_FRAC=MAX_MISSING_FRAC,
    SAMPLE_MISS_WARN=SAMPLE_MISS_WARN,
    VIF_THRESHOLD=VIF_THRESHOLD, R2_SUBSAMPLE_N=R2_SUBSAMPLE_N,
    NL_WEIGHT_R_THRESHOLD=NL_WEIGHT_R_THRESHOLD,
    SENSITIVITY_DELTA_THRESH=SENSITIVITY_DELTA_THRESH,
    LASSO_TOP_N=LASSO_TOP_N, LASSO_N_ALPHAS=LASSO_N_ALPHAS,
    LASSO_MAX_ITER=LASSO_MAX_ITER, ROC_TOP_N_FALLBACK=ROC_TOP_N_FALLBACK,
    BOOTSTRAP_N=BOOTSTRAP_N, AUC_TIER1=AUC_TIER1, AUC_TIER2=AUC_TIER2,
    TOP_N_VOLCANO_LABELS=TOP_N_VOLCANO_LABELS, TOP_N_TABLE=TOP_N_TABLE,
    TOP_N_CANDIDATES=TOP_N_CANDIDATES, FIG_DPI=FIG_DPI,
)
with open("output/cache/config.json", "w") as f:
    json.dump(_cfg, f, indent=2)

PRISM_STYLE = {
    "axes.spines.top":   False, "axes.spines.right":  False,
    "axes.linewidth":    1.2,   "xtick.direction":    "out",
    "ytick.direction":  "out",  "xtick.major.width":  1.2,
    "ytick.major.width": 1.2,   "xtick.major.size":   5,
    "ytick.major.size":  5,     "font.family":        "Arial",
    "axes.labelweight": "bold", "axes.titleweight":   "bold",
    "figure.facecolor": "white","axes.facecolor":     "white",
}
SEX_COLORS  = {"F": "#C0392B", "M": "#2980B9"}
RESP_COLORS = {"R": "#27AE60", "NR": "#E74C3C"}
ALT = "EBF3FA"

def save_fig(fig, name):
    """Save a matplotlib figure as a 300-dpi LZW-compressed TIFF plus a vector PDF copy (the caller closes the figure)."""
    fig.savefig(f"output/figures/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"output/figures/{name}.tiff", dpi=FIG_DPI,
                bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    print(f"  Saved: output/figures/{name}.pdf + .tiff")

print("Configuration saved.\n")

# ============================================================================
# NB01 - LOAD AND PRE-PROCESS
# ============================================================================
print("=" * 60)
print("NB01 - Load data and compute derived variables")
print("=" * 60)

print(f"Loading: {MS_FILE}")
raw_multi = pd.read_excel(MS_FILE, sheet_name=MS_SHEET, header=[0, 1])
print(f"  Raw shape: {raw_multi.shape}")

flat = []
for (l0, l1) in raw_multi.columns:
    l0, l1 = str(l0).strip(), str(l1).strip()
    flat.append(l0 if l1 in ("", "nan", "NaN") else f"{l0}_{l1}")
raw_multi.columns = flat

norm_cols = [c for c in raw_multi.columns if c.startswith(NORM_LABEL + "_")]
print(f"  Normalised columns found: {len(norm_cols)}")
if not norm_cols:
    raise ValueError(
        f"No columns starting with '{NORM_LABEL}_' found. "
        f"Check NORM_LABEL in configuration. "
        f"First 10 flat column names: {flat[:10]}")

meta_raw    = pd.read_csv(META_FILE, sep=META_SEP, decimal=META_DEC)
meta        = meta_raw.set_index("patient_id")
patient_ids = meta.index.tolist()
n_enrolled  = len(patient_ids)
print(f"  Patients in metadata: {n_enrolled}")
print(f"  Metadata columns: {meta.columns.tolist()}")

patient_number = {pid: i + 1 for i, pid in enumerate(patient_ids)}
number_patient = {v: k for k, v in patient_number.items()}
meta["patient_number"] = meta.index.map(patient_number)
with open("output/cache/patient_number_map.json", "w") as f:
    json.dump(patient_number, f)
print(f"  Patient numbers assigned: 1-{n_enrolled}")

# --- Delta scores AND percent reduction (positive = improvement) ------------
print("\nComputing delta scores and percent reductions...")
for scale in ["hamd", "madrs"]:
    bc, mc, ac = f"{scale}_before", f"{scale}_middle", f"{scale}_after"
    if any(c not in meta.columns for c in [bc, mc, ac]):
        if f"{scale}_delta" in meta.columns:
            meta[f"{scale}_delta_full"] = meta[f"{scale}_delta"]
            print(f"  {scale}: using existing '{scale}_delta' column "
                  f"as '{scale}_delta_full'")
            meta[f"{scale}_delta_early"]   = np.nan
            meta[f"{scale}_delta_late"]    = np.nan
            meta[f"{scale}_pct_reduction"] = np.nan
            meta[f"{scale}_pct_early"]     = np.nan
        else:
            print(f"  WARNING: {scale} timepoint columns and delta column missing")
        continue
    meta[f"{scale}_delta_full"]  = meta[ac]  - meta[bc]
    meta[f"{scale}_delta_early"] = meta[mc]  - meta[bc]
    meta[f"{scale}_delta_late"]  = meta[ac]  - meta[mc]
    base = meta[bc].replace(0, np.nan).astype(float)
    meta[f"{scale}_pct_reduction"] = ((meta[bc] - meta[ac]) / base) * 100
    meta[f"{scale}_pct_early"]     = ((meta[bc] - meta[mc]) / base) * 100
    n = meta[f"{scale}_delta_full"].notna().sum()
    mn, mx = (meta[f"{scale}_delta_full"].min(), meta[f"{scale}_delta_full"].max())
    pct_mn, pct_mx = (meta[f"{scale}_pct_reduction"].min(),
                      meta[f"{scale}_pct_reduction"].max())
    print(f"  {scale}: n={n}/{n_enrolled}  delta range=[{mn:.1f}, {mx:.1f}]  "
          f"%-reduction range=[{pct_mn:.1f}, {pct_mx:.1f}]")

# --- Responder classification (response + 50% + remission) ------------------
print("\nResponder and remission classification:")
for scale, abs_thr, rem_cut, label in [
    ("hamd",  HAMD_RESPONDER_THRESHOLD,  HAMD_REMISSION_CUTOFF,  "HAMD-17"),
    ("madrs", MADRS_RESPONDER_THRESHOLD, MADRS_REMISSION_CUTOFF, "MADRS"),
]:
    if f"{scale}_delta_full" in meta.columns:
        meta[f"{scale}_responder"] = np.where(
            meta[f"{scale}_delta_full"] <= abs_thr, "R", "NR")
        r = (meta[f"{scale}_responder"] == "R").sum()
        print(f"  {label} responder (delta <= {abs_thr}): "
              f"R={r}, NR={n_enrolled - r}")
    else:
        meta[f"{scale}_responder"] = "NR"
    if f"{scale}_pct_reduction" in meta.columns:
        meta[f"{scale}_responder_50pct"] = np.where(
            meta[f"{scale}_pct_reduction"] >= PCT_RESPONSE_THRESHOLD, "R", "NR")
        r50 = (meta[f"{scale}_responder_50pct"] == "R").sum()
        print(f"  {label} >={PCT_RESPONSE_THRESHOLD:.0f}% reduction: "
              f"R={r50}, NR={n_enrolled - r50}")
    else:
        meta[f"{scale}_responder_50pct"] = "NR"
    after_col = f"{scale}_after"
    if after_col in meta.columns:
        meta[f"{scale}_remission"] = np.where(
            meta[after_col] <= rem_cut, "R", "NR")
        rem = (meta[f"{scale}_remission"] == "R").sum()
        print(f"  {label} remission (post <= {rem_cut}): "
              f"R={rem}, NR={n_enrolled - rem}")
    else:
        meta[f"{scale}_remission"] = "NR"

print("\n  HAMD response x MADRS response:")
print(pd.crosstab(meta["hamd_responder"], meta["madrs_responder"],
                  rownames=["HAMD-R"], colnames=["MADRS-R"]).to_string())
print("\n  Response (MADRS) x Remission (MADRS):")
print(pd.crosstab(meta["madrs_responder"], meta["madrs_remission"],
                  rownames=["Resp-MADRS"], colnames=["Rem-MADRS"]).to_string())

# --- Season, medications, weight --------------------------------------------
def parse_months(val):
    """Parse a collection-month cell into a list of integer month numbers (1-12)."""
    if pd.isna(val): return []
    parts = re.split(r"[,/;\s]+", str(val).strip())
    return [int(float(p)) for p in parts
            if p.strip() and 1 <= float(p) <= 12]

def month_to_season(m):
    """Map a month number (1-12) to a meteorological season label."""
    try: m = int(m)
    except: return "Unknown"
    if   m in [12, 1, 2]:  return "Winter"
    elif m in [3, 4, 5]:   return "Spring"
    elif m in [6, 7, 8]:   return "Summer"
    elif m in [9, 10, 11]: return "Autumn"
    return "Unknown"

if "collection_month" in meta.columns:
    mf = {}
    for pid in patient_ids:
        raw    = meta.loc[pid, "collection_month"]
        parsed = parse_months(raw)
        mf[pid] = parsed[0] if parsed else np.nan
    meta["collection_month_clean"] = pd.Series(mf)
    meta["season"]           = meta["collection_month_clean"].apply(month_to_season)
    # NOTE: season is deliberately NOT used as a covariate in the OLS correction
    #       (different serum proteins peak in different seasons, so a single
    #       seasonal term is inappropriate; acknowledged in the manuscript
    #       limitations). "season" is kept only for the PCA confounder check.
    print(f"\n  Season: {meta['season'].value_counts().to_dict()}")
else:
    meta["season"]                 = "Unknown"
    meta["collection_month_clean"] = np.nan

MED_COLS = ["antidepressant","antipsychotic","mood_stabilizer","anxiolytic","lithium"]
for col in MED_COLS:
    if col in meta.columns:
        meta[col] = pd.to_numeric(meta[col], errors="coerce").fillna(0).astype(int)
    else:
        meta[col] = 0

if "weight_kg" in meta.columns:
    meta["weight_kg"] = pd.to_numeric(meta["weight_kg"], errors="coerce")
else:
    meta["weight_kg"] = np.nan

# Cardiovascular-medication indicator (0/1), read from metadata.csv. Provide
# this column in the metadata to include it in the covariate correction; if it
# is absent it defaults to 0 for all participants.
if "cardiovascular_med" not in meta.columns:
    meta["cardiovascular_med"] = 0
meta["cardiovascular_med"] = (pd.to_numeric(meta["cardiovascular_med"],
                               errors="coerce").fillna(0).astype(int))

ids_in_file  = [c.replace(NORM_LABEL + "_", "") for c in norm_cols]
matched_pids = [p for p in patient_ids if p in ids_in_file]
missing_pids = [p for p in patient_ids if p not in ids_in_file]
if missing_pids:
    print(f"\n  WARNING: {missing_pids} in metadata but not in proteomics export")
print(f"  Matched patients: {len(matched_pids)}/{n_enrolled}")

ANNOT_KEYS = ["accession","description","mass","peptide","unique","confidence"]
annot_cols  = [c for c in raw_multi.columns
               if any(k in c.lower() for k in ANNOT_KEYS)
               and not c.startswith(NORM_LABEL + "_")
               and not c.startswith(RAW_LABEL  + "_")]
annot_df = raw_multi[annot_cols].copy()

rename_map = {}
for c in annot_df.columns:
    cl = c.lower()
    if   "accession"    in cl: rename_map[c] = "Accession"
    elif "description"  in cl: rename_map[c] = "Description"
    elif "mass"         in cl and "mass_da" not in rename_map.values():
        rename_map[c] = "mass_da"
    elif "unique"       in cl: rename_map[c] = "unique_peptides"
    elif "confidence"   in cl: rename_map[c] = "confidence"
annot_df = annot_df.rename(columns=rename_map)

print(f"\n  Annotation columns retained: {list(annot_df.columns)}")

norm_matched = [f"{NORM_LABEL}_{p}" for p in matched_pids]
abund        = raw_multi[norm_matched].copy()
abund.columns= [c.replace(NORM_LABEL + "_", "") for c in abund.columns]

df = pd.concat([annot_df, abund], axis=1)
df[matched_pids] = df[matched_pids].replace(0, np.nan)
print(f"  Full table before QC: {len(df)} proteins x {len(matched_pids)} patients")

# --- QC filtering ------------------------------------------------------------
n_start = len(df)
if "confidence" in df.columns:
    conf_vals = df["confidence"]
    if conf_vals.dtype == object:
        unique_conf = conf_vals.dropna().unique()
        print(f"  Confidence column values (text): {unique_conf[:10]}")
        df_qc = df[conf_vals.astype(str).str.strip().str.capitalize() == "High"].copy()
        print(f"  Filter 1 (High confidence, text): {n_start} -> {len(df_qc)} proteins")
    else:
        unique_conf = conf_vals.dropna().unique()
        print(f"  Confidence column values (numeric): min={conf_vals.min():.3f}, "
              f"max={conf_vals.max():.3f}")
        df_qc = df[pd.to_numeric(conf_vals, errors="coerce") >= 0.99].copy()
        print(f"  Filter 1 (confidence >= 0.99, numeric): {n_start} -> {len(df_qc)} proteins")
    if len(df_qc) == 0:
        print("  WARNING: confidence filter removed all proteins.")
        df_qc = df.copy()
else:
    print("  Confidence column not found - skipping Filter 1")
    df_qc = df.copy()

if "unique_peptides" in df_qc.columns:
    n_before = len(df_qc)
    df_qc = df_qc[
        pd.to_numeric(df_qc["unique_peptides"], errors="coerce") >= MIN_PEPTIDES
    ].copy()
    print(f"  Filter 2 (>= {MIN_PEPTIDES} unique peptides): "
          f"{n_before} -> {len(df_qc)} proteins")

if "Accession" not in df_qc.columns:
    raise KeyError("Accession column not found in QC table.")

mat = df_qc.set_index("Accession")[matched_pids].astype(float)
n_before  = len(mat)
miss_frac = mat.isna().mean(axis=1)
mat       = mat[miss_frac <= MAX_MISSING_FRAC]
print(f"  Filter 3 (<= {MAX_MISSING_FRAC:.0%} missing): "
      f"{n_before} -> {len(mat)} proteins")

if len(mat) == 0:
    raise ValueError("Zero proteins retained after QC.")

n_prot = len(mat)
print(f"\n  Proteins retained: {n_prot}")

keep      = [c for c in ["Accession","Description","mass_da","unique_peptides"]
             if c in df_qc.columns]
df_annot  = df_qc[df_qc["Accession"].isin(mat.index)][keep].set_index("Accession")
if "mass_da" in df_annot.columns:
    df_annot["mass_kda"] = pd.to_numeric(df_annot["mass_da"], errors="coerce") / 1000

meta_matched = meta.loc[matched_pids].copy()

mat.to_pickle("output/cache/mat_raw.pkl")
df_annot.to_pickle("output/cache/df_annot.pkl")
meta_matched.to_pickle("output/cache/meta.pkl")
with open("output/cache/patient_ids.json", "w") as f:
    json.dump(matched_pids, f)
with open("output/cache/patient_number_map.json", "w") as f:
    json.dump({p: patient_number[p] for p in matched_pids}, f)

DELTA_COLS = [c for c in [
    "hamd_delta_full","hamd_delta_early","hamd_delta_late",
    "madrs_delta_full","madrs_delta_early","madrs_delta_late",
] if c in meta_matched.columns]
PCT_COLS = [c for c in [
    "hamd_pct_reduction","hamd_pct_early",
    "madrs_pct_reduction","madrs_pct_early",
] if c in meta_matched.columns]
with open("output/cache/delta_cols.json", "w") as f:
    json.dump(DELTA_COLS, f)
with open("output/cache/pct_cols.json", "w") as f:
    json.dump(PCT_COLS, f)

pids    = matched_pids
n_total = len(pids)
pm      = patient_number
n_F     = (meta_matched["sex"] == "F").sum()
n_M     = (meta_matched["sex"] == "M").sum()

print(f"\nNB01 complete: {n_prot} proteins x {n_total} patients")

# ============================================================================
# NB02 - NORMALIZATION AND IMPUTATION
# ============================================================================
print("\n" + "=" * 60)
print("NB02 - Normalization and imputation")
print("=" * 60)

mat_log  = np.log2(mat).replace([np.inf, -np.inf], np.nan)
medians  = mat_log.median(axis=0)
global_m = medians.median()
spread   = medians.max() - medians.min()
print(f"  Median spread: {spread:.3f} log2 units  (threshold: 1.0)")

if spread > 1.0:
    mat_norm = mat_log.subtract(medians - global_m, axis=1)
    print("  Cross-sample normalization applied.")
else:
    mat_norm = mat_log.copy()
    print("  ISOQuant normalization sufficient - no additional cross-sample step.")

def minprob_impute(mat, downshift=1.8, width=0.3, seed=42):
    """Impute missing values from a down-shifted Gaussian (Perseus-style min-probability imputation)."""
    rng     = np.random.default_rng(seed)
    mat_imp = mat.copy()
    for col in mat_imp.columns:
        detected = mat_imp[col].dropna()
        n_miss   = mat_imp[col].isna().sum()
        if n_miss > 0 and len(detected) > 1:
            mat_imp.loc[mat_imp[col].isna(), col] = rng.normal(
                loc=detected.mean() - downshift * detected.std(),
                scale=max(detected.std() * width, 1e-6),
                size=n_miss)
    assert mat_imp.isna().sum().sum() == 0
    return mat_imp

mat_imp = minprob_impute(mat_norm)
print(f"  Imputation complete. Missing after: {mat_imp.isna().sum().sum()}")

mat_imp.to_pickle("output/cache/mat_imp.pkl")
mat_norm.to_pickle("output/cache/mat_norm.pkl")

# Figure 1 - data quality (unchanged from v2)
print("\nGenerating Figure 1 (data quality)...")
fig_w = max(18, n_total * 0.55 + 6)
fig   = plt.figure(figsize=(fig_w, 11), facecolor="white")
gs_out   = gridspec.GridSpec(1, 2, figure=fig,
    left=0.05, right=0.97, top=0.93, bottom=0.08, wspace=0.12)
gs_left  = gridspec.GridSpecFromSubplotSpec(2, 1,
    subplot_spec=gs_out[0], height_ratios=[2.2, 1], hspace=0.40)
gs_right = gridspec.GridSpecFromSubplotSpec(2, 1,
    subplot_spec=gs_out[1], hspace=0.40)
ax_A = fig.add_subplot(gs_left[0])
ax_B = fig.add_subplot(gs_left[1])
ax_C = fig.add_subplot(gs_right[0])
ax_D = fig.add_subplot(gs_right[1])

miss_data = mat.isna().astype(float)
n_vals    = miss_data.size
n_miss    = int(miss_data.values.sum())
pct_str   = f"{n_miss / n_vals:.1%}" if n_vals > 0 else "0.0%"

im = ax_A.imshow(miss_data.values.T, aspect="auto", cmap="RdYlGn_r",
                 interpolation="none", vmin=0, vmax=1, origin="upper")
ax_A.set_yticks(range(n_total))
ax_A.set_yticklabels([str(pm[p]) for p in pids], fontsize=8)
ax_A.set_ylabel("Patient number", fontsize=9, fontweight="bold")
ax_A.set_xlabel(f"Proteins (n = {n_prot})", fontsize=10, fontweight="bold")
ax_A.set_title("A  Missing value map", fontsize=11, fontweight="bold", loc="left", pad=6)
ax_A.text(0.98, 0.02, f"Missing: {n_miss}/{n_vals} ({pct_str})",
          transform=ax_A.transAxes, fontsize=8, ha="right", va="bottom",
          color="white",
          bbox=dict(boxstyle="round,pad=0.2", fc="#1A6B3C", ec="none", alpha=0.85))
cbar = fig.colorbar(im, ax=ax_A, fraction=0.025, pad=0.02)
cbar.set_label("Missing / Detected", fontsize=8)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(["Detected", "Missing"], fontsize=7.5)

sample_miss = miss_data.mean(axis=0)
bar_colors  = [SEX_COLORS.get(meta_matched.loc[p, "sex"], "grey") for p in pids]
ax_B.barh(range(n_total), [sample_miss[p] for p in pids],
          color=bar_colors, alpha=0.85, edgecolor="white",
          linewidth=0.4, height=0.7)
ax_B.set_yticks(range(n_total))
ax_B.set_yticklabels([str(pm[p]) for p in pids], fontsize=8)
ax_B.set_ylabel("Patient number", fontsize=9, fontweight="bold")
ax_B.axvline(SAMPLE_MISS_WARN, color="#C0392B", ls="--", lw=1.5)
ax_B.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax_B.set_xlabel("Fraction of proteins missing", fontsize=10, fontweight="bold")
ax_B.set_title("B  Per-patient missing rate", fontsize=11, fontweight="bold",
               loc="left", pad=6)
ax_B.set_xlim(0, max(SAMPLE_MISS_WARN * 1.5, sample_miss.max() * 1.2 + 0.01))
ax_B.legend(handles=[
    mpatches.Patch(color="#C0392B", alpha=0.85, label=f"Female (n={n_F})"),
    mpatches.Patch(color="#2980B9", alpha=0.85, label=f"Male (n={n_M})"),
    plt.Line2D([0], [0], color="#C0392B", ls="--", lw=1.5,
               label=f"Warning ({SAMPLE_MISS_WARN:.0%})"),
], fontsize=10, loc="lower right", frameon=True, framealpha=0.95, edgecolor="#888888")

def draw_boxplots(ax, matrix, title, ylabel):
    """Draw per-sample log2-abundance boxplots on the given axis."""
    colors = [SEX_COLORS.get(meta_matched.loc[p, "sex"], "grey") for p in pids]
    bp = ax.boxplot([matrix[p].dropna().values for p in pids],
                    patch_artist=True,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(color="#555555", linewidth=1),
                    capprops=dict(color="#555555", linewidth=1),
                    flierprops=dict(marker="o", markersize=1.8, alpha=0.35,
                                    markerfacecolor="#888888",
                                    markeredgecolor="none"),
                    widths=0.65)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.78)
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=6)
    ax.set_ylabel(ylabel, fontsize=9, fontweight="bold")
    ax.set_xlabel("Patient number", fontsize=9, fontweight="bold")
    ax.set_xticks(range(1, n_total + 1))
    ax.set_xticklabels([str(pm[p]) for p in pids], rotation=90, fontsize=7)

draw_boxplots(ax_C, mat_log,  "C  Before cross-sample normalization", "log2 abundance (ISOQuant output)")
ax_C.legend(handles=[
    mpatches.Patch(color="#C0392B", alpha=0.78, label=f"Female (n={n_F})"),
    mpatches.Patch(color="#2980B9", alpha=0.78, label=f"Male (n={n_M})"),
], fontsize=10, loc="upper right", frameon=True, framealpha=0.95, edgecolor="#888888")
draw_boxplots(ax_D, mat_norm, "D  After normalization (used for analysis)", "log Abundance (analysis-ready)")
save_fig(fig, "Figure_1_data_quality")
plt.close(fig)

# ============================================================================
# NB03 - COVARIATE CORRECTION AND PCA
# ============================================================================
print("\n" + "=" * 60)
print("NB03 - Covariate correction and PCA")
print("=" * 60)
# Covariates for the per-protein OLS residual correction: five psychotropic
# medication-class indicators (MED_COLS) + cardiovascular_med + weight_kg
# (= six medication-class indicators + body weight, matching the manuscript).
# Season is intentionally NOT included as a covariate.

EXTRA_COLS = []
for col in ["weight_kg", "cardiovascular_med"]:
    if col in meta_matched.columns:
        v = pd.to_numeric(meta_matched[col], errors="coerce")
        if v.notna().sum() > 0 and v.std() > 1e-9:
            EXTRA_COLS.append(col)

all_cands = [c for c in MED_COLS + EXTRA_COLS
             if c in meta_matched.columns
             and pd.to_numeric(meta_matched[c], errors="coerce").std() > 1e-9]
print(f"  Candidate covariates: {all_cands}")

def compute_vif(df, cols):
    """Return variance inflation factors for the given covariate columns."""
    data = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    X    = sm.add_constant(data.astype(float))
    return pd.DataFrame({
        "covariate": cols,
        "VIF": [variance_inflation_factor(X.values, i + 1)
                for i in range(len(cols))]
    }).sort_values("VIF", ascending=False)

remaining = all_cands.copy()
for step in range(10):
    if len(remaining) <= 1: break
    vif_df  = compute_vif(meta_matched, remaining)
    max_vif = vif_df["VIF"].max()
    if max_vif <= VIF_THRESHOLD:
        print(f"  VIF step {step}: all <= {VIF_THRESHOLD}. Done.")
        break
    worst = vif_df.iloc[0]["covariate"]
    print(f"  VIF step {step}: removing '{worst}' (VIF={max_vif:.2f})")
    remaining.remove(worst)

FINAL_COVARIATES = remaining
print(f"  Final covariates: {FINAL_COVARIATES}")
with open("output/cache/final_covariates.txt", "w") as f:
    f.write("\n".join(FINAL_COVARIATES))

# Covariate correlation heatmap with tighter layout and bigger legend
cov_data = meta_matched[FINAL_COVARIATES].apply(pd.to_numeric, errors="coerce")
corr     = cov_data.corr()
with plt.rc_context(PRISM_STYLE):
    sz = max(7, len(FINAL_COVARIATES) + 1)
    fig, ax = plt.subplots(figsize=(sz, sz - 0.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=ax, square=True, linewidths=0.5,
                mask=np.triu(np.ones_like(corr, dtype=bool)),
                annot_kws={"size": 11},
                cbar_kws={"label": "Pearson r", "shrink": 0.7})
    ax.set_title(
        f"Covariate correlation matrix\n"
        f"|r| > {1/VIF_THRESHOLD**0.5:.2f} may indicate collinearity "
        f"(VIF threshold = {VIF_THRESHOLD})",
        fontsize=12, fontweight="bold")
    ax.tick_params(axis="both", labelsize=10)
    plt.tight_layout()
    save_fig(fig, "NB03_covariate_correlation")
    plt.close(fig)

def remove_covariate_effects(mat, meta_df, covariates, label=""):
    """Per-protein OLS covariate correction; returns residuals with the protein grand mean re-added."""
    cov_data   = meta_df[covariates].apply(pd.to_numeric, errors="coerce").fillna(0)
    cov_matrix = sm.add_constant(cov_data.astype(float))
    mat_corr   = mat.copy()
    for i, prot in enumerate(mat.index):
        if (i + 1) % 200 == 0:
            print(f"  [{label}] {i+1}/{len(mat)}...", end="\r")
        y   = mat.loc[prot].values.astype(float)
        fit = sm.OLS(y, cov_matrix).fit()
        mat_corr.loc[prot] = fit.resid + y.mean()
    print(f"  [{label}] Corrected {len(mat)} proteins.          ")
    return mat_corr

mat_corr = remove_covariate_effects(mat_imp, meta_matched, FINAL_COVARIATES, "primary")

covs_no_weight = [c for c in FINAL_COVARIATES if c != "weight_kg"]
if covs_no_weight != FINAL_COVARIATES:
    mat_corr_no_weight = remove_covariate_effects(
        mat_imp, meta_matched, covs_no_weight, "no-weight")
    mat_corr_no_weight.to_pickle("output/cache/mat_corr_no_weight.pkl")
    SENS_AVAILABLE = True
    mat_corr_nw    = mat_corr_no_weight
else:
    SENS_AVAILABLE = False
    mat_corr_nw    = mat_corr.copy()

print("  Post-correction residual check:")
cov_num  = meta_matched[FINAL_COVARIATES].apply(pd.to_numeric, errors="coerce").fillna(0)
nl_prots = []
for col in FINAL_COVARIATES:
    x = cov_num[col].values.astype(float)
    rs = []
    for prot in mat_corr.index:
        y     = mat_corr.loc[prot].values.astype(float)
        valid = ~np.isnan(x) & ~np.isnan(y)
        if valid.sum() >= 5:
            r, _ = pearsonr(y[valid], x[valid])
            rs.append(r)
    r_arr = np.array(rs)
    if r_arr.size == 0: continue
    print(f"    {col:25s}: mean|r|={np.mean(np.abs(r_arr)):.4f}  "
          f"max|r|={np.max(np.abs(r_arr)):.4f}")
    if col == "weight_kg":
        nl_prots = mat_corr.index[np.abs(r_arr) > NL_WEIGHT_R_THRESHOLD].tolist()

with open("output/cache/nonlinear_weight_proteins.json", "w") as f:
    json.dump(nl_prots, f)

mat_corr.to_pickle("output/cache/mat_corr.pkl")
meta_matched.to_pickle("output/cache/meta.pkl")

# PCA
print("\n  Generating PCA figure...")
def pca_panel(ax, matrix, colour_col, palette, title):
    """Draw a 2-component PCA scatter coloured by a metadata column and labelled with patient numbers."""
    cols_ok = [p for p in pids if p in matrix.columns]
    if len(cols_ok) < 2:
        ax.set_visible(False); return
    mat_sub = matrix[cols_ok].copy()
    row_medians = mat_sub.median(axis=1)
    for col in mat_sub.columns:
        mask = mat_sub[col].isna()
        if mask.any():
            mat_sub.loc[mask, col] = row_medians[mask]
    arr = mat_sub.values.T
    if arr.shape[1] < 1 or np.all(arr == 0):
        ax.set_visible(False); return
    pca    = PCA(n_components=min(2, arr.shape[1], arr.shape[0]))
    coords = pca.fit_transform(arr)
    var    = pca.explained_variance_ratio_ * 100
    for g in sorted(meta_matched.loc[cols_ok, colour_col].unique().astype(str)):
        idx = [i for i, p in enumerate(cols_ok)
               if str(meta_matched.loc[p, colour_col]) == g]
        if not idx: continue
        ax.scatter(coords[idx, 0], coords[idx, 1],
                   color=palette.get(g, "grey"), s=72,
                   alpha=0.85, edgecolors="white", linewidths=0.6,
                   zorder=3, label=g)
        for i in idx:
            ax.annotate(str(patient_number[cols_ok[i]]),
                        (coords[i, 0] + 0.05, coords[i, 1] + 0.05),
                        fontsize=10, fontweight="bold", alpha=0.9)
    ax.set_xlabel(f"PC1 ({var[0]:.1f}%)" if len(var) > 0 else "PC1", fontsize=11)
    ax.set_ylabel(f"PC2 ({var[1]:.1f}%)" if len(var) > 1 else "PC2", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6, loc="left")
    ax.legend(title=colour_col.replace("_", " "),
              fontsize=10, title_fontsize=10, frameon=True,
              framealpha=0.95, edgecolor="#888888", loc="lower right",
              borderpad=0.6)
    ax.axhline(0, color="grey", lw=0.5, ls="--", alpha=0.5)
    ax.axvline(0, color="grey", lw=0.5, ls="--", alpha=0.5)

season_pal = {"Spring":"#E74C3C","Autumn":"#E67E22",
              "Summer":"#27AE60","Winter":"#3498DB","Unknown":"#95A5A6"}
with plt.rc_context(PRISM_STYLE):
    fig, axes = plt.subplots(2, 2, figsize=(13, 12), constrained_layout=True)
    pca_panel(axes[0, 0], mat_imp,  "sex", SEX_COLORS,
              "A  Before correction \u2014 sex")
    pca_panel(axes[0, 1], mat_corr, "sex", SEX_COLORS,
              "B  After correction \u2014 sex")
    pca_panel(axes[1, 0], mat_corr, "madrs_responder", RESP_COLORS,
              "C  After correction \u2014 MADRS response")
    if ("season" in meta_matched.columns and meta_matched["season"].nunique() > 1):
        pca_panel(axes[1, 1], mat_corr, "season", season_pal,
                  "D  After correction \u2014 season")
    else:
        axes[1, 1].set_visible(False)
    fig.suptitle(
        "PCA of serum proteomes \u2014 before and after covariate correction\n"
        f"Covariates: {', '.join(FINAL_COVARIATES)}",
        fontsize=12, fontweight="bold")
    save_fig(fig, "NB03_pca")
    plt.close(fig)

# ============================================================================
# NB04 - STATISTICS
# ============================================================================
print("\n" + "=" * 60)
print("NB04 - Statistical analysis")
print("=" * 60)

df_annot_l = pd.read_pickle("output/cache/df_annot.pkl")
def extract_gene(desc):
    """Extract the gene symbol from a UniProt protein-description string."""
    m = re.search(r"GN=([^\s]+)", str(desc))
    return m.group(1) if m else None
if "Description" in df_annot_l.columns:
    df_annot_l["gene_name"] = df_annot_l["Description"].apply(extract_gene)
    df_annot_l["gene_name"] = df_annot_l["gene_name"].fillna(
        df_annot_l.index.to_series())

with open("output/cache/nonlinear_weight_proteins.json") as f:
    nl_weight_prots = set(json.load(f))

def ids_for(label_col):
    """Return (responder_ids, non_responder_ids) for a responder-definition column."""
    R  = meta_matched.index[meta_matched[label_col] == "R"].tolist()
    NR = meta_matched.index[meta_matched[label_col] == "NR"].tolist()
    return R, NR

OUTCOME_DEFS = {
    "hamd_response":   ("hamd_responder",          "HAMD-17 response (>=7-pt drop)"),
    "madrs_response":  ("madrs_responder",         "MADRS response (>=10-pt drop)"),
    "hamd_50pct":      ("hamd_responder_50pct",    "HAMD-17 >=50% reduction"),
    "madrs_50pct":     ("madrs_responder_50pct",   "MADRS >=50% reduction"),
    "hamd_remission":  ("hamd_remission",          "HAMD-17 remission (post <=7)"),
    "madrs_remission": ("madrs_remission",         "MADRS remission (post <=10)"),
}
outcome_ids = {key: ids_for(col) for key, (col, _) in OUTCOME_DEFS.items()}

print("\n  Outcome group sizes:")
for key, (col, label) in OUTCOME_DEFS.items():
    R, NR = outcome_ids[key]
    print(f"    {label:36s}: R={len(R):2d}, NR={len(NR):2d}")

F_ids = meta_matched.index[meta_matched["sex"] == "F"].tolist()
M_ids = meta_matched.index[meta_matched["sex"] == "M"].tolist()
n_F = len(F_ids); n_M = len(M_ids)
print(f"    Sex: F={n_F}, M={n_M}")

HAMD_R_ids,  HAMD_NR_ids  = outcome_ids["hamd_response"]
MADRS_R_ids, MADRS_NR_ids = outcome_ids["madrs_response"]
n_HAMD_R  = len(HAMD_R_ids);  n_HAMD_NR  = len(HAMD_NR_ids)
n_MADRS_R = len(MADRS_R_ids); n_MADRS_NR = len(MADRS_NR_ids)

def run_wilcoxon(mat, R_ids, NR_ids, suffix):
    """Per-protein Wilcoxon rank-sum test (responders vs non-responders) with log2 fold change and BH-FDR."""
    if len(mat) == 0 or len(R_ids) < MIN_GROUP_SIZE or len(NR_ids) < MIN_GROUP_SIZE:
        print(f"  Wilcoxon ({suffix}) skipped - insufficient group size "
              f"(R={len(R_ids)}, NR={len(NR_ids)})")
        return pd.DataFrame(columns=["Accession",
            f"log2FC_{suffix}", f"mean_R_{suffix}",
            f"mean_NR_{suffix}", f"p_wilcox_{suffix}",
            f"adj_p_wilcox_{suffix}"])
    rows = []
    for prot in mat.index:
        rv  = mat.loc[prot, R_ids].values.astype(float)
        nrv = mat.loc[prot, NR_ids].values.astype(float)
        _, p = stats.mannwhitneyu(rv, nrv, alternative="two-sided")
        rows.append({"Accession":          prot,
                     f"log2FC_{suffix}":   np.mean(rv) - np.mean(nrv),
                     f"mean_R_{suffix}":   np.mean(rv),
                     f"mean_NR_{suffix}":  np.mean(nrv),
                     f"p_wilcox_{suffix}": p})
    df = pd.DataFrame(rows)
    _, df[f"adj_p_wilcox_{suffix}"], _, _ = multipletests(
        df[f"p_wilcox_{suffix}"], method="fdr_bh")
    return df

wilcoxon_tables = {}
for key, (col, label) in OUTCOME_DEFS.items():
    print(f"\n  Wilcoxon: {label}")
    R, NR = outcome_ids[key]
    wdf = run_wilcoxon(mat_corr, R, NR, key)
    wilcoxon_tables[key] = wdf
    if len(wdf) > 0:
        n_nom = (wdf[f"p_wilcox_{key}"] < NOMINAL_P).sum()
        n_fdr = (wdf[f"adj_p_wilcox_{key}"] < NOMINAL_P).sum()
        print(f"    nominally sig p<{NOMINAL_P}: {n_nom},  FDR sig: {n_fdr}")

wilcox_hamd  = wilcoxon_tables["hamd_response"]
wilcox_madrs = wilcoxon_tables["madrs_response"]

BASELINE_COLS = [c for c in ["hamd_before","madrs_before"]
                 if c in meta_matched.columns]
ALL_CORR_COLS = PCT_COLS + DELTA_COLS + BASELINE_COLS
if "weight_kg" in meta_matched.columns:
    ALL_CORR_COLS = ALL_CORR_COLS + ["weight_kg"]

def run_spearman(mat, label=""):
    """Spearman correlations of each protein with the continuous outcomes (percent reduction, early trajectory, baseline severity)."""
    if len(mat) == 0:
        print(f"  Spearman [{label}] skipped - empty matrix")
        return pd.DataFrame(columns=["Accession"])
    rows = []
    for prot in mat.index:
        vals = mat.loc[prot].values.astype(float)
        row  = {"Accession": prot}
        for col in ALL_CORR_COLS:
            if col not in meta_matched.columns: continue
            outcome = meta_matched[col].values.astype(float)
            mask    = ~np.isnan(outcome) & ~np.isnan(vals)
            if mask.sum() >= MIN_PAIRS_SPEARMAN:
                rho, p = stats.spearmanr(vals[mask], outcome[mask])
            else:
                rho, p = np.nan, np.nan
            if col in PCT_COLS:
                short = (col.replace("_pct_reduction", "_pct_full")
                            .replace("_pct_early",     "_pct_early"))
                row[f"rho_{short}"] = rho
                row[f"p_{short}"]   = p
            elif col in DELTA_COLS:
                short = (col.replace("_delta_full",  "_full")
                            .replace("_delta_early", "_early")
                            .replace("_delta_late",  "_late"))
                row[f"rho_{short}"] = rho
                row[f"p_{short}"]   = p
            elif col in BASELINE_COLS:
                short = col.replace("_before", "")
                row[f"rho_{short}_baseline"] = rho
                row[f"p_{short}_baseline"]   = p
            elif col == "weight_kg":
                row["rho_weight"] = rho
                row["p_weight"]   = p
        rows.append(row)
    df = pd.DataFrame(rows)
    for pc in [c for c in df.columns if c.startswith("p_") and "adj" not in c]:
        adj_c = "adj_" + pc
        valid = df[pc].notna()
        df[adj_c] = np.nan
        if valid.sum() > 0:
            _, adj, _, _ = multipletests(df.loc[valid, pc], method="fdr_bh")
            df.loc[valid, adj_c] = adj
    print(f"  Spearman [{label}] - nominal sig (p<{NOMINAL_P}):")
    for pc in [c for c in df.columns
               if c.startswith("p_") and "adj" not in c
               and "baseline" not in c and "weight" not in c]:
        n = (df[pc] < NOMINAL_P).sum()
        if n > 0:
            print(f"    {pc}: {n}")
    return df

print("\n  Running Spearman correlations (with percent reduction as primary)...")
sp_df = run_spearman(mat_corr, "primary")

weight_sensitive = []
if SENS_AVAILABLE and len(sp_df) > 0:
    sp_nw = run_spearman(mat_corr_nw, "no-weight")
    if ("rho_madrs_pct_full" in sp_df.columns and
            "rho_madrs_pct_full" in sp_nw.columns):
        prim  = sp_df.set_index("Accession")["rho_madrs_pct_full"]
        sens  = sp_nw.set_index("Accession")["rho_madrs_pct_full"]
        delta = (prim - sens).abs()
        weight_sensitive = delta[delta > SENSITIVITY_DELTA_THRESH].index.tolist()
        print(f"  Weight-sensitive proteins (MADRS %-reduction): {len(weight_sensitive)}")

if len(sp_df) > 0:
    sp_df["weight_sensitive"]         = sp_df["Accession"].isin(weight_sensitive)
    sp_df["weight_nonlinear_residual"] = sp_df["Accession"].isin(nl_weight_prots)

def get_gene_name(acc):
    """Return the gene symbol for a UniProt accession from the cached annotation."""
    if acc in df_annot_l.index and "gene_name" in df_annot_l.columns:
        g = df_annot_l.loc[acc, "gene_name"]
        if isinstance(g, str): return g
    return acc

res_df = wilcox_hamd.merge(wilcox_madrs, on="Accession", how="left")
for key in ["hamd_50pct","madrs_50pct","hamd_remission","madrs_remission"]:
    sub = wilcoxon_tables[key]
    if len(sub) > 0:
        cols_keep = [c for c in sub.columns if c.startswith(("log2FC_","p_wilcox_","adj_p_wilcox_"))]
        res_df = res_df.merge(sub[["Accession"] + cols_keep], on="Accession", how="left")

if len(sp_df) > 0:
    res_df = res_df.merge(sp_df, on="Accession", how="left")
if "mass_kda" in df_annot_l.columns:
    res_df = res_df.merge(df_annot_l[["mass_kda"]].reset_index(), on="Accession", how="left")
res_df["gene_name"] = res_df["Accession"].apply(get_gene_name)
sort_cols = [c for c in ["p_wilcox_madrs_response","p_wilcox_hamd_response"] if c in res_df.columns]
if sort_cols:
    res_df = res_df.sort_values(sort_cols)

log2_fc_thresh = np.log2(FOLD_CHANGE_THRESH)
for sfx, p_col, fc_col in [
    ("hamd",  "p_wilcox_hamd_response",  "log2FC_hamd_response"),
    ("madrs", "p_wilcox_madrs_response", "log2FC_madrs_response"),
]:
    if p_col not in res_df.columns: continue
    res_df[f"label_{sfx}"] = "Not significant"
    res_df.loc[(res_df[p_col] < NOMINAL_P) & (res_df[fc_col] >  log2_fc_thresh),
               f"label_{sfx}"] = "Higher in future responders"
    res_df.loc[(res_df[p_col] < NOMINAL_P) & (res_df[fc_col] < -log2_fc_thresh),
               f"label_{sfx}"] = "Lower in future responders"

def classify_scenario_generic(row, p_delta_col, p_baseline_col,
                              p_wilcox_col=None, concordance_alpha=BORDERLINE_P):
    """Scenario classification.
    A = significant continuous correlation (p_delta_col < NOMINAL_P) AND
        non-significant baseline correlation (p_baseline_col >= NOMINAL_P)
        AND (when concordance is required) the dichotomized Wilcoxon also
        reaches at least borderline significance (p_wilcox_col < BORDERLINE_P).
    The Wilcoxon concordance requirement filters out proteins where the
    continuous Spearman correlation is being driven by a few outliers in the
    percent-reduction outcome but the R vs NR group distributions are
    visually indistinguishable.
    """
    sd = row.get(p_delta_col, 1) < NOMINAL_P
    pb = row.get(p_baseline_col, np.nan)
    try:   pb_f = float(pb)
    except: pb_f = np.nan
    sb = (not np.isnan(pb_f)) and (pb_f < NOMINAL_P)
    if p_wilcox_col is not None:
        pw = row.get(p_wilcox_col, 1)
        concordant = pw < concordance_alpha
    else:
        concordant = True
    if   sd and not sb and concordant: return "A"
    elif sd and sb:                    return "B"
    elif not sd and sb:                return "C"
    return "D"

# Strict (default) Scenario A requires concordance with the dichotomized test.
res_df["scenario_hamd"]  = res_df.apply(
    lambda r: classify_scenario_generic(r, "p_hamd_pct_full",  "p_hamd_baseline",
                                        "p_wilcox_hamd_response"),  axis=1)
res_df["scenario_madrs"] = res_df.apply(
    lambda r: classify_scenario_generic(r, "p_madrs_pct_full", "p_madrs_baseline",
                                        "p_wilcox_madrs_response"), axis=1)
res_df["scenario"] = res_df["scenario_madrs"]

# Loose Scenario A (no concordance) kept for transparency. Proteins that are
# A under the loose criterion but not A under the strict one are flagged as
# "A (discordant)" - significant continuous correlation but no group separation.
res_df["scenario_hamd_loose"]  = res_df.apply(
    lambda r: classify_scenario_generic(r, "p_hamd_pct_full",  "p_hamd_baseline"),  axis=1)
res_df["scenario_madrs_loose"] = res_df.apply(
    lambda r: classify_scenario_generic(r, "p_madrs_pct_full", "p_madrs_baseline"), axis=1)
res_df["A_discordant_madrs"] = (
    (res_df["scenario_madrs_loose"] == "A") & (res_df["scenario_madrs"] != "A"))
res_df["A_discordant_hamd"]  = (
    (res_df["scenario_hamd_loose"]  == "A") & (res_df["scenario_hamd"]  != "A"))

print(f"\n  Scenario distribution (MADRS-primary, concordance required):\n"
      f"{res_df['scenario'].value_counts().to_string()}")
print(f"\n  Scenario distribution (HAMD, concordance required):\n"
      f"{res_df['scenario_hamd'].value_counts().to_string()}")
n_disc_m = res_df["A_discordant_madrs"].sum()
n_disc_h = res_df["A_discordant_hamd"].sum()
if n_disc_m > 0:
    print(f"\n  MADRS: {n_disc_m} protein(s) drop from Scenario A under concordance "
          f"requirement: "
          f"{res_df.loc[res_df['A_discordant_madrs'], 'gene_name'].tolist()}")
if n_disc_h > 0:
    print(f"  HAMD:  {n_disc_h} protein(s) drop from Scenario A under concordance "
          f"requirement: "
          f"{res_df.loc[res_df['A_discordant_hamd'], 'gene_name'].tolist()}")

# Also flag BH-FDR survivors (for transparency in the workbook)
if "adj_p_madrs_pct_full" in res_df.columns:
    n_fdr_m = (res_df["adj_p_madrs_pct_full"] < 0.05).sum()
    print(f"\n  MADRS continuous correlations surviving BH-FDR (q<0.05): {n_fdr_m}")
if "adj_p_hamd_pct_full" in res_df.columns:
    n_fdr_h = (res_df["adj_p_hamd_pct_full"] < 0.05).sum()
    print(f"  HAMD  continuous correlations surviving BH-FDR (q<0.05): {n_fdr_h}")

res_df.to_csv("output/results/statistical_results.csv", index=False)
res_df.to_pickle("output/cache/stats_results.pkl")

# ============================================================================
# VOLCANO PLOTS - tight axes, visible legend lower-right, bigger fonts (v3)
# ============================================================================
def make_volcano(res_df, p_col, fc_col, n_R, n_NR, response_def, outname):
    """Render a volcano plot (log2FC vs -log10 p) with significance tiers and labelled proteins."""
    if p_col not in res_df.columns or fc_col not in res_df.columns:
        print(f"  Volcano '{outname}' skipped: columns missing"); return
    df = res_df.copy()
    df["-log10p"] = -np.log10(df[p_col].clip(lower=1e-300))
    sig_mask   = df[p_col] <  NOMINAL_P
    bord_mask  = (df[p_col] >= NOMINAL_P) & (df[p_col] < BORDERLINE_P)
    top_idx    = df.sort_values(p_col).head(TOP_N_VOLCANO_LABELS).index
    top_mask   = df.index.isin(top_idx)
    df_plot    = df[sig_mask | bord_mask | top_mask].copy()

    def tier(row):
        p, fc = row[p_col], row[fc_col]
        if   p < NOMINAL_P:    c="#D62728" if fc>0 else "#1F77B4"; s=110; z=5; a=0.92
        elif p < BORDERLINE_P: c="#F08030" if fc>0 else "#6BAED6"; s=75;  z=4; a=0.85
        else:                  c="#9E9E9E"; s=35; z=3; a=0.70
        return pd.Series({"dot_c":c, "dot_s":s, "dot_z":z, "dot_a":a})
    df_plot[["dot_c","dot_s","dot_z","dot_a"]] = df_plot.apply(tier, axis=1)

    # TIGHT axis limits - zoom to plotted data range, no blank borders
    fc_min = df_plot[fc_col].min()
    fc_max = df_plot[fc_col].max()
    fc_pad = max(0.04, (fc_max - fc_min) * 0.10)
    y_min  = max(0, df_plot["-log10p"].min() - 0.05)
    y_max  = df_plot["-log10p"].max() + 0.18
    x_lim  = (fc_min - fc_pad, fc_max + fc_pad)
    y_lim  = (y_min, y_max)

    with plt.rc_context(PRISM_STYLE):
        # Wider figure to make room for the legend on the right side
        fig, ax = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)
        for mask in [top_mask & ~sig_mask & ~bord_mask, bord_mask, sig_mask]:
            sub = df_plot[df_plot.index.isin(df[mask].index)]
            if sub.empty: continue
            ax.scatter(sub[fc_col].values, sub["-log10p"].values,
                       c=list(sub["dot_c"]), s=list(sub["dot_s"]),
                       alpha=float(sub["dot_a"].iloc[0]),
                       zorder=int(sub["dot_z"].iloc[0]),
                       edgecolors="white", linewidths=0.6)
        ax.axhline(-np.log10(NOMINAL_P),    color="#888888", lw=1.0, ls=":", alpha=0.75)
        ax.axhline(-np.log10(BORDERLINE_P), color="#BBBBBB", lw=0.8, ls=":", alpha=0.65)
        ax.axvline( log2_fc_thresh, color="#BBBBBB", lw=0.8, ls="--", alpha=0.5)
        ax.axvline(-log2_fc_thresh, color="#BBBBBB", lw=0.8, ls="--", alpha=0.5)
        ax.axvline(0, color="#DDDDDD", lw=0.6, alpha=0.5)

        ax.annotate(f"p = {NOMINAL_P}", xy=(x_lim[1] - 0.005, -np.log10(NOMINAL_P) + 0.02),
                    fontsize=10, color="#666666", ha="right")
        ax.annotate(f"p = {BORDERLINE_P}", xy=(x_lim[1] - 0.005, -np.log10(BORDERLINE_P) + 0.02),
                    fontsize=10, color="#888888", ha="right")

        texts = []
        for _, row in df_plot.iterrows():
            gene = row.get("gene_name", row["Accession"])
            is_sig = row[p_col] < NOMINAL_P
            texts.append(ax.text(row[fc_col], row["-log10p"], str(gene),
                fontsize=11 if is_sig else 9.5,
                fontweight="bold" if is_sig else "normal",
                color="#111111", zorder=6))
        if adjustText is not None:
            try:
                adjustText.adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6),
                    expand_points=(1.3, 1.5), force_text=(0.5, 0.8))
            except Exception:
                pass

        # Legend OUTSIDE the axes, on the right side, vertically centered
        legend_handles = [
            mpatches.Patch(fc="#D62728", ec="white", lw=0.5,
                           label=f"Higher in responders, p < {NOMINAL_P}"),
            mpatches.Patch(fc="#1F77B4", ec="white", lw=0.5,
                           label=f"Lower in responders, p < {NOMINAL_P}"),
            mpatches.Patch(fc="#F08030", ec="white", lw=0.5,
                           label=f"Higher, {NOMINAL_P} \u2264 p < {BORDERLINE_P}"),
            mpatches.Patch(fc="#6BAED6", ec="white", lw=0.5,
                           label=f"Lower, {NOMINAL_P} \u2264 p < {BORDERLINE_P}"),
            mpatches.Patch(fc="#9E9E9E", ec="none",
                           label=f"Top-ranked, p \u2265 {BORDERLINE_P}"),
        ]
        ax.legend(handles=legend_handles, fontsize=11, frameon=True,
                  framealpha=0.95, edgecolor="#888888",
                  bbox_to_anchor=(1.02, 0.5), loc="center left",
                  borderpad=0.8, labelspacing=0.6)

        ax.set_xlim(x_lim); ax.set_ylim(y_lim)
        ax.set_xlabel("log fold change (responders / non-responders)",
                      fontsize=13, fontweight="bold")
        ax.set_ylabel("\u2212log (Wilcoxon p)",
                      fontsize=13, fontweight="bold")
        ax.set_title(f"Pre-treatment serum proteome \u2014 future {response_def} responders "
                     f"(n={n_R}) vs non-responders (n={n_NR})  [total n={n_total}]",
                     fontsize=12, fontweight="bold")
        ax.tick_params(axis="both", labelsize=11)
        save_fig(fig, outname)
        plt.close(fig)

print("\n  Generating volcano plots...")
make_volcano(res_df, "p_wilcox_hamd_response",  "log2FC_hamd_response",
             n_HAMD_R,  n_HAMD_NR,  "HAMD-17", "NB04_volcano_response_HAMD")
make_volcano(res_df, "p_wilcox_madrs_response", "log2FC_madrs_response",
             n_MADRS_R, n_MADRS_NR, "MADRS",   "NB04_volcano_response_MADRS")

# ============================================================================
# DOTPLOTS - Scenario A only, tight axes, sex legend restored (v3)
# ============================================================================
def make_scenario_A_dotplot(res_df, scenario_col, p_rank_col, p_corr_col,
                            rho_corr_col, R_ids, NR_ids, outname, title_text):
    """Dotplot of every Scenario A protein under the chosen scale.
    Panels ordered by p of the continuous criterion. Per-panel y-axis is
    auto-zoomed to that protein's range. Sex legend at lower right.
    """
    sel = res_df[res_df[scenario_col] == "A"].copy()
    if len(sel) == 0:
        print(f"  Dotplot '{outname}' skipped: no Scenario A proteins"); return
    sel = sel.sort_values(p_corr_col)
    top_prots = [p for p in sel["Accession"].tolist() if p in mat_corr.index]
    if not top_prots:
        print(f"  Dotplot '{outname}' skipped: no Scenario A proteins in mat_corr")
        return

    # Layout: explicit legend slot inside the grid. The slot is chosen so
    # that the visual reading order is natural and the legend sits in row 1
    # next to the first plot (matching the supervisor's requested layout).
    # For n=3 the result is:
    #   Row 1: Plot 1  |  Sex legend
    #   Row 2: Plot 2  |  Plot 3
    n = len(top_prots)
    if   n == 1:  n_cols, n_rows, legend_slot = 2, 1, 1            # P L
    elif n == 2:  n_cols, n_rows, legend_slot = 3, 1, 2            # P P L
    elif n == 3:  n_cols, n_rows, legend_slot = 2, 2, 1            # P L / P P
    elif n == 4:  n_cols, n_rows, legend_slot = 2, 2, None         # P P / P P, overlay
    elif n <= 5:  n_cols, n_rows, legend_slot = 3, 2, 2            # P P L / P P P
    elif n == 6:  n_cols, n_rows, legend_slot = 3, 2, None
    elif n <= 8:  n_cols, n_rows, legend_slot = 3, 3, 2            # P P L / P P P / P P P (truncate at n)
    elif n == 9:  n_cols, n_rows, legend_slot = 3, 3, None
    elif n <= 11: n_cols, n_rows, legend_slot = 4, 3, 3            # P P P L / ...
    elif n == 12: n_cols, n_rows, legend_slot = 4, 3, None
    else:         n_cols, n_rows, legend_slot = 4, (n + 3) // 4, None

    # Slots reserved for plots vs the legend
    if legend_slot is not None:
        plot_slots = [i for i in range(n_cols * n_rows) if i != legend_slot]
        plot_slots = plot_slots[:n]
    else:
        plot_slots = list(range(n))

    with plt.rc_context(PRISM_STYLE):
        # Reserve a fixed top band for the suptitle so it never overlaps the
        # panel titles. With matplotlib's default va='top', suptitle y is the
        # TOP of the text - we anchor it ~0.10 inch below figure top.
        suptitle_band_inches = 1.2
        plot_inches_per_row  = 4.3
        fig_height_inches    = n_rows * plot_inches_per_row + suptitle_band_inches
        top_frac     = 1.0 - (suptitle_band_inches / fig_height_inches)
        suptitle_y   = 1.0 - (0.10 / fig_height_inches)

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 3.7, fig_height_inches))
        plt.subplots_adjust(left=0.08, right=0.97,
                            top=top_frac, bottom=0.07,
                            hspace=0.55, wspace=0.30)
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        rng = np.random.default_rng(42)

        for slot_idx, acc in zip(plot_slots, top_prots):
            ax = axes[slot_idx]
            row    = sel[sel["Accession"] == acc].iloc[0]
            gene   = row.get("gene_name", acc)
            rho    = row[rho_corr_col]
            p_corr = row[p_corr_col]
            p_wilc = row[p_rank_col]
            abund  = mat_corr.loc[acc]

            for pid in pids:
                resp   = "R" if pid in R_ids else "NR"
                sex    = meta_matched.loc[pid, "sex"]
                x_pos  = 0 if resp == "R" else 1
                jitter = rng.uniform(-0.12, 0.12)
                ax.scatter(x_pos + jitter, abund[pid],
                           color=SEX_COLORS.get(sex, "grey"),
                           s=65, alpha=0.88, zorder=3,
                           edgecolors="white", linewidths=0.55)

            for xi, gids in enumerate([R_ids, NR_ids]):
                gv = abund[gids].values.astype(float)
                ax.errorbar(xi, gv.mean(), yerr=gv.std(),
                            fmt="k_", capsize=7, linewidth=2.4,
                            zorder=4, capthick=2.4)

            # Tight y-axis per panel
            y_vals = abund.values.astype(float)
            y_min, y_max = float(np.nanmin(y_vals)), float(np.nanmax(y_vals))
            y_pad = (y_max - y_min) * 0.10
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.set_xlim(-0.55, 1.55)

            sign = "+" if rho >= 0 else "\u2212"
            # Bigger fonts: title 13, xlabel/ylabel 11.5-12, ticks 11.5
            ax.set_title(f"{gene}\n\u03c1 = {sign}{abs(rho):.2f}, p = {p_corr:.3f}",
                         fontsize=13, fontweight="bold", pad=8)
            wilc_txt = (f"Wilcoxon p = {p_wilc:.3f}" if p_wilc >= 0.001
                        else f"Wilcoxon p = {p_wilc:.2e}")
            ax.set_xlabel(wilc_txt, fontsize=11.5)
            ax.set_ylabel("log abundance", fontsize=12)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([f"Responders\n(n = {len(R_ids)})",
                                f"Non-responders\n(n = {len(NR_ids)})"],
                               fontsize=11.5)
            ax.tick_params(axis="y", labelsize=10.5)

        # Place legend - either in its dedicated slot or overlay on last panel
        if legend_slot is not None:
            ax_leg = axes[legend_slot]
            ax_leg.set_visible(True)
            ax_leg.axis("off")
            ax_leg.legend(handles=[
                mpatches.Patch(color=SEX_COLORS["F"], alpha=0.88, label=f"Female (n = {n_F})"),
                mpatches.Patch(color=SEX_COLORS["M"], alpha=0.88, label=f"Male (n = {n_M})"),
            ], fontsize=14, frameon=True, framealpha=0.95, edgecolor="#888888",
               loc="center", title="Sex", title_fontsize=14)
            # Hide any other unused slots
            used = set(plot_slots) | {legend_slot}
            for i in range(n_cols * n_rows):
                if i not in used:
                    axes[i].set_visible(False)
        else:
            # No dedicated slot - overlay on last plot panel at lower right
            axes[plot_slots[-1]].legend(handles=[
                mpatches.Patch(color=SEX_COLORS["F"], alpha=0.88, label=f"F (n={n_F})"),
                mpatches.Patch(color=SEX_COLORS["M"], alpha=0.88, label=f"M (n={n_M})"),
            ], fontsize=11, frameon=True, framealpha=0.92, edgecolor="#888888",
               loc="lower right", title="Sex", title_fontsize=11)
            # Hide any unused slots beyond the plots
            for i in range(n_cols * n_rows):
                if i not in plot_slots:
                    axes[i].set_visible(False)

        fig.suptitle(title_text, fontsize=14, fontweight="bold", y=suptitle_y)
        save_fig(fig, outname)
        plt.close(fig)

print("\n  Generating Scenario A dotplots...")
make_scenario_A_dotplot(
    res_df, scenario_col="scenario_madrs",
    p_rank_col="p_wilcox_madrs_response",
    p_corr_col="p_madrs_pct_full", rho_corr_col="rho_madrs_pct_full",
    R_ids=MADRS_R_ids, NR_ids=MADRS_NR_ids,
    outname="NB04_dotplots_top_MADRS",
    title_text=("Scenario A markers under MADRS \u2014 pre-treatment levels\n"
                f"MADRS responders vs non-responders "
                f"(\u2265 {abs(MADRS_RESPONDER_THRESHOLD)}-point reduction; "
                f"sorted by \u03c1 with MADRS percent reduction)"))

make_scenario_A_dotplot(
    res_df, scenario_col="scenario_hamd",
    p_rank_col="p_wilcox_hamd_response",
    p_corr_col="p_hamd_pct_full", rho_corr_col="rho_hamd_pct_full",
    R_ids=HAMD_R_ids, NR_ids=HAMD_NR_ids,
    outname="NB04_dotplots_top_HAMD",
    title_text=("Scenario A markers under HAMD-17 \u2014 pre-treatment levels\n"
                f"HAMD-17 responders vs non-responders "
                f"(\u2265 {abs(HAMD_RESPONDER_THRESHOLD)}-point reduction; "
                f"sorted by \u03c1 with HAMD-17 percent reduction)"))

# ============================================================================
# CLINICAL TRAJECTORIES - tight axes, bigger legend lower-right (v3)
# ============================================================================
print("\n  Generating clinical trajectory figure...")
TIMEPOINT_COLS = {
    "hamd":  ["hamd_before",  "hamd_middle",  "hamd_after"],
    "madrs": ["madrs_before", "madrs_middle", "madrs_after"],
}
if all(c in meta_matched.columns
       for cols in TIMEPOINT_COLS.values() for c in cols):
    tp_labels = ["Baseline","Mid-treatment","Post-treatment"]
    x_pos     = [0, 1, 2]
    with plt.rc_context(PRISM_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), constrained_layout=True)
        for ax, (scale, label) in zip(axes, [("hamd","HAMD-17"),("madrs","MADRS")]):
            cols   = TIMEPOINT_COLS[scale]
            r_ids  = HAMD_R_ids  if scale == "hamd" else MADRS_R_ids
            nr_ids = HAMD_NR_ids if scale == "hamd" else MADRS_NR_ids
            data_R  = meta_matched.loc[r_ids,  cols].astype(float)
            data_NR = meta_matched.loc[nr_ids, cols].astype(float)
            for _, row in data_R.iterrows():
                ax.plot(x_pos, row.values, color="#27AE60", lw=0.9, alpha=0.35)
            for _, row in data_NR.iterrows():
                ax.plot(x_pos, row.values, color="#E74C3C", lw=0.9, alpha=0.35)
            for data, color, light, lbl in [
                (data_R,  "#27AE60","#A8D8A8", f"Responders (n = {len(r_ids)})"),
                (data_NR, "#E74C3C","#F4AAAA", f"Non-responders (n = {len(nr_ids)})"),
            ]:
                means = data.mean(axis=0).values
                sds   = data.std(axis=0).values
                ax.fill_between(x_pos, means - sds, means + sds,
                                color=light, alpha=0.35)
                ax.plot(x_pos, means, color=color, lw=2.8, marker="o",
                        markersize=8, markeredgecolor="white",
                        markeredgewidth=1.2, label=lbl)
            thresh = (abs(HAMD_RESPONDER_THRESHOLD) if scale == "hamd"
                      else abs(MADRS_RESPONDER_THRESHOLD))
            # Tight y limits to data range
            all_data = np.concatenate([data_R.values.flatten(), data_NR.values.flatten()])
            all_data = all_data[~np.isnan(all_data)]
            y_min, y_max = float(all_data.min()), float(all_data.max())
            y_pad = (y_max - y_min) * 0.04
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.set_xlim(-0.15, 2.15)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(tp_labels, fontsize=11)
            ax.set_ylabel(f"{label} score", fontsize=12, fontweight="bold")
            ax.set_xlabel("Assessment timepoint", fontsize=11)
            letter = "A" if scale == "hamd" else "B"
            ax.set_title(f"{letter}  {label} trajectories "
                         f"(response \u2265 {thresh}-point reduction)",
                         fontsize=12, fontweight="bold", loc="left", pad=6)
            ax.legend(fontsize=11, frameon=True, framealpha=0.95,
                      edgecolor="#888888", loc="lower right",
                      borderpad=0.7, labelspacing=0.5)
            ax.axvline(1, color="#CCCCCC", lw=0.7, ls="--", alpha=0.7)
        fig.suptitle("Clinical symptom trajectories across the rTMS treatment course",
                     fontsize=13, fontweight="bold")
        save_fig(fig, "NB04_clinical_trajectories")
        plt.close(fig)

# Sex comparison
def wilcoxon_two_groups(mat, ids_A, ids_B, sfx_A="F", sfx_B="M"):
    """Per-protein Wilcoxon rank-sum test between two groups (e.g. female vs male) with BH-FDR."""
    rows = []
    for prot in mat.index:
        av  = mat.loc[prot, ids_A].values.astype(float)
        bv  = mat.loc[prot, ids_B].values.astype(float)
        _, p = stats.mannwhitneyu(av, bv, alternative="two-sided")
        rows.append({"Accession":                  prot,
                     f"log2FC_{sfx_A}_vs_{sfx_B}": np.mean(av) - np.mean(bv),
                     "p_value":                    p})
    df = pd.DataFrame(rows)
    if len(df) > 0:
        _, df["adj_p"], _, _ = multipletests(df["p_value"], method="fdr_bh")
    return df.sort_values("p_value")

print("\n  Running sex comparisons...")
sex_all = wilcoxon_two_groups(mat_corr, F_ids, M_ids)
sex_all = sex_all.merge(res_df[["Accession","gene_name"]].drop_duplicates(),
                        on="Accession", how="left")
sex_all.to_pickle("output/cache/sex_all.pkl")

# ============================================================================
# NB05 - UniProt annotation
# ============================================================================
print("\n" + "=" * 60)
print("NB05 - UniProt annotation and deviation analysis")
print("=" * 60)
UNIPROT_CACHE = "output/reference/uniprot_data.json"

def clean_accessions(acc_list):
    """Normalise UniProt accessions (take the primary accession and drop duplicates)."""
    cleaned = []
    for acc in acc_list:
        for part in str(acc).split(";"):
            part = part.strip()
            if re.match(r"^[A-Z][0-9][A-Z0-9]{3}[0-9]", part):
                cleaned.append(part)
    return list(dict.fromkeys(cleaned))

def download_uniprot(accessions, batch_size=50):
    """Fetch gene name, subcellular location and blood-detection flags from the UniProt REST API (batched, cached)."""
    all_data = {}
    clean_acc = clean_accessions(accessions)
    n_batches = (len(clean_acc) + batch_size - 1) // batch_size
    for i in range(n_batches):
        batch = clean_acc[i * batch_size:(i + 1) * batch_size]
        query = " OR ".join([f"accession:{a}" for a in batch])
        try:
            resp = requests.get(
                "https://rest.uniprot.org/uniprotkb/search",
                params={"query": query, "format": "json",
                        "fields": "accession,gene_names,protein_name,"
                                  "cc_subcellular_location,cc_tissue_specificity",
                        "size": batch_size}, timeout=30)
            resp.raise_for_status()
            for entry in resp.json().get("results", []):
                acc    = entry.get("primaryAccession", "")
                genes  = entry.get("genes", [])
                gene   = (genes[0].get("geneName", {}).get("value", "")
                          if genes else "")
                pname  = (entry.get("proteinDescription", {})
                          .get("recommendedName", {})
                          .get("fullName", {}).get("value", ""))
                comments = entry.get("comments", [])
                locs = []; tissues = []
                for c in comments:
                    if c.get("commentType") == "SUBCELLULAR LOCATION":
                        for sl in c.get("subcellularLocations", []):
                            v = sl.get("location", {}).get("value", "")
                            if v: locs.append(v)
                    if c.get("commentType") == "TISSUE SPECIFICITY":
                        for t in c.get("texts", []):
                            tissues.append(t.get("value", ""))
                tissue_text = " ".join(tissues)[:600]
                in_blood    = any(w in tissue_text.lower()
                                  for w in ["blood","serum","circulating","secreted"])
                all_data[acc] = {"gene_name": gene, "protein_name": pname,
                                 "locations": "; ".join(locs[:3]),
                                 "in_blood":  in_blood}
        except Exception as e:
            print(f"  UniProt batch {i} error: {e}")
        time.sleep(0.3)
    return all_data

accs_clean = clean_accessions(mat_corr.index.tolist())
if os.path.exists(UNIPROT_CACHE):
    with open(UNIPROT_CACHE) as f:
        uniprot_data = json.load(f)
    missing = [a for a in accs_clean if a not in uniprot_data]
    if missing:
        print(f"  Downloading {len(missing)} missing proteins from UniProt...")
        uniprot_data.update(download_uniprot(missing))
        with open(UNIPROT_CACHE, "w") as f:
            json.dump(uniprot_data, f)
else:
    print(f"  Downloading {len(accs_clean)} proteins from UniProt...")
    uniprot_data = download_uniprot(accs_clean)
    with open(UNIPROT_CACHE, "w") as f:
        json.dump(uniprot_data, f)
print(f"  UniProt: {len(uniprot_data)} proteins annotated")

ref_rows = []
for acc in mat_corr.index:
    clean = str(acc).split(";")[0].strip()
    u     = uniprot_data.get(clean, {})
    ref_rows.append({
        "Accession":    acc,
        "gene_name_ref":u.get("gene_name", clean),
        "in_blood":     u.get("in_blood", False),
        "mean_R_hamd":   mat_corr.loc[acc, HAMD_R_ids].mean(),
        "mean_NR_hamd":  mat_corr.loc[acc, HAMD_NR_ids].mean(),
        "mean_R_madrs":  mat_corr.loc[acc, MADRS_R_ids].mean(),
        "mean_NR_madrs": mat_corr.loc[acc, MADRS_NR_ids].mean(),
        "mean_F":       mat_corr.loc[acc, F_ids].mean(),
        "mean_M":       mat_corr.loc[acc, M_ids].mean(),
    })
ref_df = pd.DataFrame(ref_rows).set_index("Accession")

combined = res_df.merge(ref_df.reset_index(), on="Accession", how="left")
if "gene_name_x" in combined.columns:
    combined["gene_name"] = combined["gene_name_x"].fillna(combined.get("gene_name_y",""))
    combined.drop(columns=["gene_name_x","gene_name_y","gene_name_ref"],
                  errors="ignore", inplace=True)
elif "gene_name_ref" in combined.columns:
    combined["gene_name"] = combined["gene_name"].fillna(combined["gene_name_ref"])
    combined.drop(columns=["gene_name_ref"], errors="ignore", inplace=True)

combined.to_csv("output/results/combined_results.csv", index=False)
combined.to_pickle("output/cache/combined_results.pkl")

# ============================================================================
# NB06 - LASSO, ROC (with direction handling), model comparison
# ============================================================================
print("\n" + "=" * 60)
print("NB06 - Predictive modelling")
print("=" * 60)

y_vectors = {}
for key, (col, _) in OUTCOME_DEFS.items():
    y_vectors[key] = (meta_matched[col] == "R").astype(int).values
y_hamd  = y_vectors["hamd_response"]
y_madrs = y_vectors["madrs_response"]

sexF = (meta_matched["sex"] == "F").astype(int).values.reshape(-1, 1)
loo  = LeaveOneOut()
scaler_loc = StandardScaler()

def lasso_select(y_labels, rank_p_col, label):
    """LASSO-select a protein panel from the top-N Wilcoxon-ranked proteins (plus sex) for one responder definition."""
    input_prots = (combined.sort_values(rank_p_col)
                   .head(LASSO_TOP_N)["Accession"].tolist())
    input_prots = [p for p in input_prots if p in mat_corr.index]
    X    = mat_corr.loc[input_prots].T.values
    X    = np.hstack([X, sexF])
    feat_names = input_prots + ["sex_F"]
    X_sc = scaler_loc.fit_transform(X)
    lasso = LassoCV(cv=loo, max_iter=LASSO_MAX_ITER,
                    n_alphas=LASSO_N_ALPHAS, random_state=42)
    lasso.fit(X_sc, y_labels)
    sel_idx  = np.where(lasso.coef_ != 0)[0]
    sel_df = pd.DataFrame({
        "feature":   [feat_names[i] for i in sel_idx],
        "coef":      lasso.coef_[sel_idx],
        "abs_coef":  np.abs(lasso.coef_[sel_idx]),
        "selected_for": label,
    }).sort_values("abs_coef", ascending=False)
    print(f"  LASSO ({label}): {len(sel_df)} features  "
          f"(alpha={lasso.alpha_:.4f}, candidates={len(input_prots)})")
    return sel_df

print("\n  Fitting dual LASSO panels...")
lasso_df_hamd  = lasso_select(y_hamd,  "p_wilcox_hamd_response",  "HAMD-native")
lasso_df_madrs = lasso_select(y_madrs, "p_wilcox_madrs_response", "MADRS-native")
lasso_panels = {
    "HAMD-native":  [f for f in lasso_df_hamd["feature"].tolist()
                     if f != "sex_F" and f in mat_corr.index],
    "MADRS-native": [f for f in lasso_df_madrs["feature"].tolist()
                     if f != "sex_F" and f in mat_corr.index],
}
print(f"  HAMD-native panel  ({len(lasso_panels['HAMD-native'])}): "
      f"{lasso_panels['HAMD-native']}")
print(f"  MADRS-native panel ({len(lasso_panels['MADRS-native'])}): "
      f"{lasso_panels['MADRS-native']}")

for k in lasso_panels:
    if not lasso_panels[k]:
        rank_col = ("p_wilcox_hamd_response" if "HAMD" in k
                    else "p_wilcox_madrs_response")
        lasso_panels[k] = (combined.sort_values(rank_col)
                           .head(5)["Accession"].tolist())
        print(f"  {k} empty - falling back to top-5 Wilcoxon: {lasso_panels[k]}")

pd.concat([lasso_df_hamd, lasso_df_madrs], ignore_index=True).to_csv(
    "output/results/lasso_selected_panels.csv", index=False)

BASELINE_SCORE_COLS = [c for c in ["hamd_before","madrs_before"]
                       if c in meta_matched.columns]

def loo_predict(X, y_labels):
    """Leave-one-out cross-validated predicted probabilities from an L2 logistic model."""
    y_sc = np.zeros(len(y_labels))
    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y_labels[train_idx]
        if len(np.unique(y_tr)) < 2:
            y_sc[test_idx] = np.nan; continue
        try:
            clf = LogisticRegression(C=1.0, max_iter=5000,
                                     solver="lbfgs", random_state=42)
            clf.fit(X_tr, y_tr)
            y_sc[test_idx] = clf.predict_proba(X_te)[0, 1]
        except Exception:
            y_sc[test_idx] = np.nan
    return y_sc

def metrics_at_sensitivity(y_true, y_score, target_sens=TARGET_SENSITIVITY):
    """Operating-point metrics (sensitivity, specificity, PPV, NPV) at a target sensitivity."""
    if len(np.unique(y_true)) < 2: return {}
    fpr, tpr, thr = roc_curve(y_true, y_score)
    valid = tpr >= target_sens
    if not valid.any(): return {}
    idx = np.argmax(valid)
    sens = tpr[idx]; spec = 1 - fpr[idx]; threshold = thr[idx]
    pred = (y_score >= threshold).astype(int)
    TP = int(((pred == 1) & (y_true == 1)).sum())
    FP = int(((pred == 1) & (y_true == 0)).sum())
    TN = int(((pred == 0) & (y_true == 0)).sum())
    FN = int(((pred == 0) & (y_true == 1)).sum())
    ppv = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    npv = TN / (TN + FN) if (TN + FN) > 0 else np.nan
    return {"threshold": float(threshold),
            "sensitivity": float(sens), "specificity": float(spec),
            "PPV": ppv, "NPV": npv,
            "TP": TP, "FP": FP, "TN": TN, "FN": FN}

def loo_block(panel_prots, y_labels, panel_name, outcome_name):
    """LOO-CV AUC and operating points for Models A (clinical), B (panel) and C (combined) for one outcome."""
    out = []
    if BASELINE_SCORE_COLS:
        Xa = meta_matched[BASELINE_SCORE_COLS].values.astype(float)
        Xa = StandardScaler().fit_transform(Xa)
        ya = loo_predict(Xa, y_labels)
        valid_a = ~np.isnan(ya)
        if valid_a.sum() >= 5 and len(np.unique(y_labels[valid_a])) >= 2:
            auc_a = roc_auc_score(y_labels[valid_a], ya[valid_a])
            m_a   = metrics_at_sensitivity(y_labels[valid_a], ya[valid_a])
        else:
            auc_a, m_a = np.nan, {}
    else:
        auc_a, m_a = np.nan, {}
    Xb = mat_corr.loc[panel_prots].T.values.astype(float)
    Xb = StandardScaler().fit_transform(Xb)
    yb = loo_predict(Xb, y_labels)
    valid_b = ~np.isnan(yb)
    if valid_b.sum() >= 5 and len(np.unique(y_labels[valid_b])) >= 2:
        auc_b = roc_auc_score(y_labels[valid_b], yb[valid_b])
        m_b   = metrics_at_sensitivity(y_labels[valid_b], yb[valid_b])
    else:
        auc_b, m_b = np.nan, {}
    if BASELINE_SCORE_COLS:
        Xc = np.hstack([
            StandardScaler().fit_transform(
                meta_matched[BASELINE_SCORE_COLS].values.astype(float)),
            StandardScaler().fit_transform(
                mat_corr.loc[panel_prots].T.values.astype(float))])
        yc = loo_predict(Xc, y_labels)
        valid_c = ~np.isnan(yc)
        if valid_c.sum() >= 5 and len(np.unique(y_labels[valid_c])) >= 2:
            auc_c = roc_auc_score(y_labels[valid_c], yc[valid_c])
            m_c   = metrics_at_sensitivity(y_labels[valid_c], yc[valid_c])
        else:
            auc_c, m_c = np.nan, {}
    else:
        auc_c, m_c = np.nan, {}
    for model, auc_v, m, n_feat in [
        ("A_scores",   auc_a, m_a, len(BASELINE_SCORE_COLS)),
        ("B_proteins", auc_b, m_b, len(panel_prots)),
        ("C_combined", auc_c, m_c, len(BASELINE_SCORE_COLS) + len(panel_prots)),
    ]:
        out.append({"panel": panel_name, "outcome": outcome_name,
                    "model": model, "n_features": n_feat,
                    "loo_cv_auc": auc_v, **m})
    return out

print("\n  Evaluating LASSO panels on all outcomes (LOO-CV)...")
all_model_rows = []
panel_for_outcome = {
    "hamd_response":   "HAMD-native",
    "madrs_response":  "MADRS-native",
    "hamd_50pct":      "HAMD-native",
    "madrs_50pct":     "MADRS-native",
    "hamd_remission":  "HAMD-native",
    "madrs_remission": "MADRS-native",
}
for outcome_key, (col, label) in OUTCOME_DEFS.items():
    panel_name = panel_for_outcome[outcome_key]
    R, NR = outcome_ids[outcome_key]
    if len(R) < MIN_GROUP_SIZE or len(NR) < MIN_GROUP_SIZE:
        print(f"    {label}: skipped (R={len(R)}, NR={len(NR)})")
        continue
    rows = loo_block(lasso_panels[panel_name], y_vectors[outcome_key],
                     panel_name, label)
    all_model_rows.extend(rows)
    for r in rows:
        if not np.isnan(r["loo_cv_auc"]):
            extra = (f"  spec={r.get('specificity', float('nan')):.2f}, "
                     f"PPV={r.get('PPV', float('nan')):.2f}, NPV={r.get('NPV', float('nan')):.2f}"
                     if 'specificity' in r else "")
            print(f"    {label:30s} | panel={panel_name:12s} | "
                  f"{r['model']:11s}: AUC={r['loo_cv_auc']:.3f}{extra}")

print("\n  Cross-panel check (panel under the OPPOSITE outcome) ...")
for outcome_key, (col, label), cross_panel in [
    ("madrs_response",  OUTCOME_DEFS["madrs_response"],  "HAMD-native"),
    ("hamd_response",   OUTCOME_DEFS["hamd_response"],   "MADRS-native"),
]:
    R, NR = outcome_ids[outcome_key]
    if len(R) < MIN_GROUP_SIZE or len(NR) < MIN_GROUP_SIZE: continue
    rows = loo_block(lasso_panels[cross_panel], y_vectors[outcome_key],
                     cross_panel + " (cross)", label)
    all_model_rows.extend(rows)
    for r in rows:
        if r["model"] == "B_proteins" and not np.isnan(r["loo_cv_auc"]):
            print(f"    {label} under {cross_panel}: AUC={r['loo_cv_auc']:.3f}")

model_comparison_df = pd.DataFrame(all_model_rows)
model_comparison_df.to_csv("output/results/model_comparison.csv", index=False)

# ============================================================================
# NB06b - NESTED LEAVE-ONE-OUT CROSS-VALIDATION (honest panel AUC)
# ============================================================================
# Model B above is LASSO-selected on all patients and then scored by LOO-CV, so
# selection and evaluation share the same patients (optimistic). Here the FULL
# selection (Wilcoxon top-N ranking -> LassoCV) is repeated INSIDE each LOO fold
# using only the training patients; the held-out patient is scored by an L2
# logistic model on the fold-selected panel. This removes the selection leak and
# gives an honest (usually lower) AUC to report alongside the in-sample one.
# (To speed this up, replace cv=loo inside LassoCV with cv=5.)
print("\n" + "=" * 60)
print("NB06b - Nested LOO-CV (feature selection inside each fold)")
print("=" * 60)

def nested_loo_auc(y_labels, top_n=LASSO_TOP_N):
    """Optimism-corrected panel AUC: the Wilcoxon ranking and LASSO selection are repeated inside every LOO fold."""
    y_labels = np.asarray(y_labels)
    Xall = mat_corr.T.values.astype(float)          # patients x proteins
    preds = np.full(len(y_labels), np.nan)
    for tr, te in loo.split(Xall):
        ytr = y_labels[tr]
        if len(np.unique(ytr)) < 2:
            continue
        # 1) rank proteins by Wilcoxon on TRAINING patients only
        rpos = np.where(ytr == 1)[0]; rneg = np.where(ytr == 0)[0]
        pv = np.ones(Xall.shape[1])
        for j in range(Xall.shape[1]):
            try:
                pv[j] = stats.mannwhitneyu(Xall[tr][rpos, j], Xall[tr][rneg, j],
                                           alternative="two-sided")[1]
            except ValueError:
                pv[j] = 1.0
        top = np.argsort(pv)[:top_n]
        # 2) candidate design = top proteins + sex_F, standardized on training
        Xtr = np.hstack([Xall[tr][:, top], sexF[tr]])
        Xte = np.hstack([Xall[te][:, top], sexF[te]])
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        # 3) LASSO selects the panel on training only (same settings as pipeline)
        las = LassoCV(cv=loo, max_iter=LASSO_MAX_ITER,
                      n_alphas=LASSO_N_ALPHAS, random_state=42).fit(Xtr_s, ytr)
        sel = np.where(las.coef_ != 0)[0]
        if sel.size == 0:                       # fall back to top-5 proteins
            sel = np.arange(min(5, Xtr_s.shape[1]))
        # 4) score the held-out patient with an L2 logistic on the selected panel
        clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs",
                                 random_state=42).fit(Xtr_s[:, sel], ytr)
        preds[te] = clf.predict_proba(Xte_s[:, sel])[0, 1]
    ok = ~np.isnan(preds)
    if ok.sum() < 5 or len(np.unique(y_labels[ok])) < 2:
        return np.nan
    return roc_auc_score(y_labels[ok], preds[ok])

# Use the native-panel Model B AUC for comparison (exclude cross-panel rows,
# which otherwise overwrite the native value for the response endpoints).
# NOTE: for remission/>=50% endpoints the nested loop re-selects on that
# endpoint's own labels, whereas the paper's Model B reuses the response
# panel; the clean apples-to-apples nested comparison is the response rows.
insample_B = {r["outcome"]: r["loo_cv_auc"]
              for r in all_model_rows
              if r["model"] == "B_proteins" and "(cross)" not in str(r["panel"])}

nested_rows = []
print(f"\n  {'outcome':32} {'in-sample B':>12} {'nested B':>10}")
for outcome_key, (col, label) in OUTCOME_DEFS.items():
    R, NR = outcome_ids[outcome_key]
    if len(R) < MIN_GROUP_SIZE or len(NR) < MIN_GROUP_SIZE:
        continue
    na = nested_loo_auc(y_vectors[outcome_key])
    ib = insample_B.get(label, float("nan"))
    nested_rows.append({"outcome": label,
                        "in_sample_panel_auc": ib,
                        "nested_panel_auc": na})
    print(f"  {label:32} {ib:>12.3f} {na:>10.3f}")

nested_df = pd.DataFrame(nested_rows)
nested_df.to_csv("output/results/nested_cv_auc.csv", index=False)
print("\n  Saved: output/results/nested_cv_auc.csv")


# ============================================================================
# Single-marker ROC with DIRECTION HANDLING (v3 fix for inverse predictors)
# ============================================================================
def compute_roc_data(prots, y_labels):
    """Single-marker ROC that picks the BETTER of (raw score) vs (-raw score),
    so an inverse predictor (e.g. SERPINA3) reports its true AUC (~0.75)
    instead of its mirrored value (~0.25). The chosen direction is recorded.
    """
    roc_data = {}; auc_rows = []
    rng = np.random.default_rng(42)
    for acc in prots:
        scores = mat_corr.loc[acc].values.astype(float)
        auc_pos = roc_auc_score(y_labels, scores)
        auc_neg = roc_auc_score(y_labels, -scores)
        if auc_neg > auc_pos:
            scores_used = -scores
            auc_val     = auc_neg
            direction   = "lower = R"
        else:
            scores_used = scores
            auc_val     = auc_pos
            direction   = "higher = R"
        fpr, tpr, _ = roc_curve(y_labels, scores_used)
        boot = []
        for _ in range(BOOTSTRAP_N):
            idx = rng.integers(0, len(y_labels), len(y_labels))
            if len(np.unique(y_labels[idx])) < 2: continue
            try: boot.append(roc_auc_score(y_labels[idx], scores_used[idx]))
            except Exception: pass
        ci_lo = np.percentile(boot, 2.5)  if boot else np.nan
        ci_hi = np.percentile(boot, 97.5) if boot else np.nan
        m = metrics_at_sensitivity(y_labels, scores_used, TARGET_SENSITIVITY)
        gene = combined.loc[combined["Accession"] == acc, "gene_name"].values
        gene = gene[0] if len(gene) > 0 else acc
        roc_data[acc] = {"fpr": fpr, "tpr": tpr, "auc": auc_val, "gene": gene}
        auc_rows.append({"Accession": acc, "gene_name": gene,
                         "AUC": auc_val, "CI_95_low": ci_lo, "CI_95_high": ci_hi,
                         "direction": direction, **m})
    return roc_data, pd.DataFrame(auc_rows).sort_values("AUC", ascending=False)

roc_data_madrs, auc_df_m = compute_roc_data(
    lasso_panels["MADRS-native"][:ROC_TOP_N_FALLBACK], y_madrs)
roc_data_hamd,  auc_df_h = compute_roc_data(
    lasso_panels["HAMD-native"][:ROC_TOP_N_FALLBACK], y_hamd)
auc_df_m.to_csv("output/results/auc_results_MADRS.csv", index=False)
auc_df_h.to_csv("output/results/auc_results_HAMD.csv",  index=False)
auc_df_h.to_pickle("output/cache/auc_results.pkl")

# ============================================================================
# ROC PLOTS - bigger legend lower-right with direction arrows (v3)
# ============================================================================
def plot_roc(roc_data, auc_df, title, outname, n_R, n_NR):
    """Plot single-marker ROC curves with bootstrap AUC confidence intervals."""
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(roc_data), 1)))
    with plt.rc_context(PRISM_STYLE):
        fig, ax = plt.subplots(figsize=(7.5, 7), constrained_layout=True)
        for (acc, d), col in zip(roc_data.items(), colors):
            row = auc_df[auc_df["Accession"] == acc].iloc[0]
            direction = row.get("direction", "higher = R")
            arrow     = "\u2191" if direction == "higher = R" else "\u2193"
            lbl = (f"{d['gene']} {arrow}  AUC = {row.AUC:.2f} "
                   f"[{row.CI_95_low:.2f}\u2013{row.CI_95_high:.2f}]")
            ax.plot(d["fpr"], d["tpr"], label=lbl, linewidth=2.3, color=col)
        ax.plot([0, 1], [0, 1], "k--", lw=0.9, alpha=0.5, label="Chance (AUC = 0.50)")
        ax.axhline(TARGET_SENSITIVITY, color="#888888", lw=0.9, ls=":", alpha=0.75, zorder=1)
        ax.annotate(f"sensitivity = {TARGET_SENSITIVITY:.2f}",
                    xy=(0.03, TARGET_SENSITIVITY + 0.012),
                    fontsize=10, color="#666666")
        ax.set_xlabel("1 \u2212 specificity", fontsize=12, fontweight="bold")
        ax.set_ylabel("Sensitivity",         fontsize=12, fontweight="bold")
        ax.set_title(f"{title}\n(n = {n_total}; R = {n_R}, NR = {n_NR})",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=11, frameon=True, framealpha=0.95,
                  edgecolor="#888888", loc="lower right",
                  borderpad=0.7, labelspacing=0.5)
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
        save_fig(fig, outname)
        plt.close(fig)

plot_roc(roc_data_madrs, auc_df_m,
         "ROC curves \u2014 MADRS response (MADRS-native panel)",
         "NB06_roc_curves_MADRS", n_MADRS_R, n_MADRS_NR)
plot_roc(roc_data_hamd, auc_df_h,
         "ROC curves \u2014 HAMD-17 response (HAMD-native panel)",
         "NB06_roc_curves_HAMD", n_HAMD_R, n_HAMD_NR)

# Tier assignment - MADRS primary
def assign_tier(row):
    """Assign a confidence tier to a candidate-marker row."""
    primary    = row.get("p_wilcox_madrs_response", 1) < NOMINAL_P
    hamd_corr  = row.get("p_hamd_pct_full",  1) < NOMINAL_P
    madrs_corr = row.get("p_madrs_pct_full", 1) < NOMINAL_P
    acc_match  = auc_df_m[auc_df_m["Accession"] == row["Accession"]]
    good_auc   = (acc_match["AUC"].values[0] >= AUC_TIER2
                  if not acc_match.empty else False)
    great_auc  = (acc_match["AUC"].values[0] >= AUC_TIER1
                  if not acc_match.empty else False)
    n_supp     = sum([hamd_corr, madrs_corr, good_auc])
    if primary and great_auc and n_supp >= 2: return "Tier 1 \u2014 Strong"
    elif primary and (good_auc or n_supp >= 1): return "Tier 2 \u2014 Good"
    elif primary: return "Tier 3 \u2014 Exploratory"
    return "Not reported"

combined["confidence_tier"] = combined.apply(assign_tier, axis=1)
final = combined[combined["confidence_tier"] != "Not reported"].copy()
final.to_csv("output/results/biomarker_candidates.csv", index=False)
print("\n  Tier distribution (MADRS-primary):")
print(combined["confidence_tier"].value_counts().to_string())

# ============================================================================
# EXCEL EXPORT - per-protein sheets now driven by Scenario A union (v3)
# ============================================================================
print("\n  Generating Excel workbook...")
output_xlsx = "output/results/complete_results.xlsx"
writer = pd.ExcelWriter(output_xlsx, engine="xlsxwriter")
wb     = writer.book

hdr_fmt  = wb.add_format({"bold":True,"bg_color":"#1F4E79","font_color":"white",
                          "border":1,"text_wrap":True})
norm_fmt = wb.add_format({"border":1})
pval_fmt = wb.add_format({"border":1,"num_format":"0.0000"})
fc_fmt   = wb.add_format({"border":1,"num_format":"0.000"})
auc_fmt  = wb.add_format({"border":1,"num_format":"0.000"})
alt_fmt  = wb.add_format({"border":1,"bg_color":ALT})

def write_sheet(df, sheet_name, col_fmts=None):
    """Write a dataframe to the results workbook with optional per-column number formats."""
    df = df.reset_index(drop=True)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
    sheet_name = sheet_name[:31]
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    ws = writer.sheets[sheet_name]
    for ci, cn in enumerate(df.columns):
        ws.write(0, ci, cn, hdr_fmt)
        fmt = col_fmts.get(cn, norm_fmt) if col_fmts else norm_fmt
        for ri in range(len(df)):
            val = df.iloc[ri, ci]
            use_fmt = alt_fmt if ri % 2 == 1 else fmt
            ws.write(ri + 1, ci, "" if pd.isna(val) else val, use_fmt)
        max_w = max(len(str(cn)),
                    df[cn].astype(str).str.len().max() if len(df) > 0 else 0)
        ws.set_column(ci, ci, min(max_w + 2, 40))

def pv_fc_fmts(df):
    """Return default Excel number formats for p-value and fold-change columns."""
    pf = {c: pval_fmt for c in df.columns
          if c.startswith("p_") or c.startswith("adj_p")}
    ff = {c: fc_fmt for c in df.columns
          if "log2FC" in c or "rho_" in c}
    return {**pf, **ff}

# README
ws_r = wb.add_worksheet("README")
ws_r.set_column(0, 0, 35); ws_r.set_column(1, 1, 90)
ws_r.write(0, 0, "Pre-TMS Serum Proteomics \u2014 v3.3 (clinical orientation, concordance-tightened Scenario A)",
           wb.add_format({"bold":True,"font_size":14,"font_color":"#1F4E79"}))
ws_r.write(1, 0,
    f"Cohort n={n_total} | HAMD response R={n_HAMD_R}/NR={n_HAMD_NR} | "
    f"MADRS response R={n_MADRS_R}/NR={n_MADRS_NR} | F={n_F}, M={n_M}")
ws_r.write(2, 0,
    f"Primary continuous outcome: percent reduction (positive = improvement). "
    f"Response: HAMD delta <= {HAMD_RESPONDER_THRESHOLD}, MADRS delta <= {MADRS_RESPONDER_THRESHOLD}. "
    f"Remission: HAMD <= {HAMD_REMISSION_CUTOFF}, MADRS <= {MADRS_REMISSION_CUTOFF}. "
    f"Sensitivity: >= {PCT_RESPONSE_THRESHOLD:.0f}% reduction. "
    f"Operating point: sensitivity = {TARGET_SENSITIVITY:.2f}.")
ws_r.write(3, 0,
    "v3.3 Scenario A is now CONCORDANCE-TIGHTENED. A protein is classified Scenario A "
    f"only if BOTH (a) its continuous Spearman correlation with percent reduction is "
    f"nominally significant (p < {NOMINAL_P}) with a non-significant baseline "
    f"correlation, AND (b) its dichotomized Wilcoxon test reaches at least "
    f"borderline significance (p < {BORDERLINE_P}). The 'scenario_*_loose' columns "
    "report the v3.2 unconcordance-checked classification. The 'A_discordant_*' "
    "columns flag proteins that were A under the loose criterion but dropped under "
    "strict. With n=27 patients and 112 proteins, none of the continuous "
    "correlations survive BH-FDR (q<0.05); all Scenario A findings are nominal-level "
    "exploratory and should be interpreted as such.")
ws_r.write(4, 0,
    "v3 ROC fix: Single-marker ROC AUC in sheets 6a/6b is computed in the better of "
    "the two score directions. The 'direction' column flags 'higher = R' or "
    "'lower = R'. This corrected the v2 issue where SERPINA3 reported AUC = 0.25 "
    "(mirror image of its true ~0.75).")
readme = [
    ("1. Statistical results (primary)",
     "Wilcoxon for HAMD and MADRS response. Spearman with %-reduction (primary), "
     "absolute deltas, and baseline scores. Scenario A/B/C/D under both scales. "
     "Includes 'scenario_*' (strict, concordance-tightened), 'scenario_*_loose' "
     "(unconcordance-checked), and 'A_discordant_*' (proteins dropped by tightening)."),
    ("2. Sensitivity - 50% reduction",
     "Wilcoxon under >=50% reduction criterion for both scales."),
    ("3. Remission outcomes",
     "Wilcoxon under remission criterion (HAMD post <=7, MADRS post <=10)."),
    ("4. Sex comparison", "All F vs M proteome contrast."),
    ("5. Candidate biomarkers",
     "Tier-ranked candidates with MADRS as primary endpoint."),
    ("6a. AUC - MADRS markers",
     "MADRS-native panel single-marker ROC AUCs with bootstrap CI and "
     f"operating-point metrics at sensitivity = {TARGET_SENSITIVITY:.2f}. "
     "Direction column shows which way the predictor points."),
    ("6b. AUC - HAMD markers",
     "HAMD-native panel single-marker ROC AUCs (parallel to 6a)."),
    ("7. LASSO panels",
     "Two LASSO refits: HAMD-native and MADRS-native."),
    ("8. Model comparison",
     "LOO-CV AUC + operating-point metrics for Models A/B/C. Each panel "
     "reported under its native outcome AND under remission and 50% outcomes. "
     "Cross-panel check included."),
    ("9. Healthy reference",
     "UniProt annotation + per-protein means under both response definitions."),
    ("Dot/Scatter sheets",
     "Per-protein sheets for every strict Scenario A protein (under MADRS or HAMD), "
     "matching the dotplot figures."),
]
ws_r.write(5, 0, "SHEET", wb.add_format({"bold":True}))
ws_r.write(5, 1, "CONTENTS", wb.add_format({"bold":True}))
for i, (s, d) in enumerate(readme):
    ws_r.write(6 + i, 0, s); ws_r.write(6 + i, 1, d)

# Sheet 1 - primary statistics
sheet1_cols = ["gene_name","Accession",
               "log2FC_madrs_response","p_wilcox_madrs_response","adj_p_wilcox_madrs_response",
               "log2FC_hamd_response", "p_wilcox_hamd_response", "adj_p_wilcox_hamd_response",
               "scenario","scenario_madrs","scenario_hamd",
               "scenario_madrs_loose","scenario_hamd_loose",
               "A_discordant_madrs","A_discordant_hamd",
               "mass_kda"]
sp_cols = [c for c in combined.columns
           if c.startswith("rho_") or c.startswith("p_hamd_pct")
           or c.startswith("p_madrs_pct") or c.startswith("p_hamd_full")
           or c.startswith("p_madrs_full") or c.startswith("p_hamd_baseline")
           or c.startswith("p_madrs_baseline")
           or c.startswith("adj_p_hamd") or c.startswith("adj_p_madrs")]
out1 = [c for c in sheet1_cols + sp_cols if c in combined.columns]
out1 = list(dict.fromkeys(out1))
sort_col = "p_wilcox_madrs_response" if "p_wilcox_madrs_response" in combined.columns else "p_wilcox_hamd_response"
write_sheet(combined[out1].sort_values(sort_col),
            "1. Statistical results", pv_fc_fmts(combined[out1]))
print("  Sheet 1: Statistical results")

# Sheet 2 - sensitivity (50%)
s2_cols = ["gene_name","Accession",
           "log2FC_madrs_50pct","p_wilcox_madrs_50pct","adj_p_wilcox_madrs_50pct",
           "log2FC_hamd_50pct", "p_wilcox_hamd_50pct", "adj_p_wilcox_hamd_50pct"]
s2_cols = [c for c in s2_cols if c in combined.columns]
if len(s2_cols) > 2:
    write_sheet(combined[s2_cols].sort_values(
        "p_wilcox_madrs_50pct" if "p_wilcox_madrs_50pct" in combined.columns else s2_cols[2]),
        "2. Sensitivity 50pct", pv_fc_fmts(combined[s2_cols]))
    print("  Sheet 2: Sensitivity 50%")

# Sheet 3 - remission
s3_cols = ["gene_name","Accession",
           "log2FC_madrs_remission","p_wilcox_madrs_remission","adj_p_wilcox_madrs_remission",
           "log2FC_hamd_remission", "p_wilcox_hamd_remission", "adj_p_wilcox_hamd_remission"]
s3_cols = [c for c in s3_cols if c in combined.columns]
if len(s3_cols) > 2:
    write_sheet(combined[s3_cols].sort_values(
        "p_wilcox_madrs_remission" if "p_wilcox_madrs_remission" in combined.columns else s3_cols[2]),
        "3. Remission", pv_fc_fmts(combined[s3_cols]))
    print("  Sheet 3: Remission")

# Sheet 4 - sex
if len(sex_all) > 0:
    s4c = [c for c in ["gene_name","Accession","log2FC_F_vs_M","p_value","adj_p"]
           if c in sex_all.columns]
    write_sheet(sex_all[s4c], "4. Sex comparison",
                {"p_value":pval_fmt,"adj_p":pval_fmt,"log2FC_F_vs_M":fc_fmt})
    print("  Sheet 4: Sex comparison")

# Sheet 5 - candidates
if len(final) > 0:
    s5c = [c for c in ["gene_name","Accession","confidence_tier",
                       "log2FC_madrs_response","p_wilcox_madrs_response",
                       "log2FC_hamd_response", "p_wilcox_hamd_response",
                       "scenario","scenario_madrs","scenario_hamd",
                       "mass_kda","in_blood"]
           if c in final.columns]
    write_sheet(final[s5c], "5. Candidate biomarkers", pv_fc_fmts(final[s5c]))
    print("  Sheet 5: Candidate biomarkers")

# Sheets 6a / 6b - AUC with direction column
op_fmt_cols = {"AUC":auc_fmt,"CI_95_low":auc_fmt,"CI_95_high":auc_fmt,
               "threshold":fc_fmt,"sensitivity":auc_fmt,"specificity":auc_fmt,
               "PPV":auc_fmt,"NPV":auc_fmt}
for df_auc, sname in [(auc_df_m, "6a. AUC - MADRS"),
                       (auc_df_h, "6b. AUC - HAMD")]:
    if len(df_auc) > 0:
        keep = [c for c in ["gene_name","Accession","direction",
                            "AUC","CI_95_low","CI_95_high",
                            "threshold","sensitivity","specificity","PPV","NPV",
                            "TP","FP","TN","FN"] if c in df_auc.columns]
        write_sheet(df_auc[keep], sname, op_fmt_cols)
print("  Sheets 6a-6b: AUC + operating point metrics (with direction column)")

# Sheet 7 - LASSO panels
lasso_combined = pd.concat([lasso_df_hamd, lasso_df_madrs], ignore_index=True)
if len(lasso_combined) > 0:
    write_sheet(lasso_combined[["selected_for","feature","coef","abs_coef"]],
                "7. LASSO panels", {"coef":fc_fmt,"abs_coef":fc_fmt})
    print("  Sheet 7: LASSO panels")

# Sheet 8 - model comparison
if len(model_comparison_df) > 0:
    write_sheet(model_comparison_df, "8. Model comparison", op_fmt_cols)
    print("  Sheet 8: Model comparison")

# Sheet 9 - healthy reference
href_c = [c for c in ["gene_name","Accession","in_blood",
                       "mean_R_madrs","mean_NR_madrs","mean_R_hamd","mean_NR_hamd",
                       "mean_F","mean_M"]
          if c in combined.columns]
sort_h = "p_wilcox_madrs_response" if "p_wilcox_madrs_response" in combined.columns else "Accession"
write_sheet(combined[href_c + [sort_h]].sort_values(sort_h).drop(columns=[sort_h]),
            "9. Healthy reference")
print("  Sheet 9: Healthy reference")

# Per-protein dot/scatter sheets - driven by Scenario A union (v3)
print("\n  Writing per-protein sheets for Scenario A proteins...")
scenario_A_accs = combined[
    (combined["scenario_madrs"] == "A") | (combined["scenario_hamd"] == "A")
].copy()
scenario_A_accs["best_pct_p"] = scenario_A_accs[
    ["p_madrs_pct_full", "p_hamd_pct_full"]
].min(axis=1)
scenario_A_iter = scenario_A_accs.sort_values("best_pct_p")
print(f"    Scenario A union: {len(scenario_A_iter)} proteins")

dot_count = scatter_count = 0
for _, prow in scenario_A_iter.iterrows():
    acc  = prow["Accession"]
    gene = str(prow.get("gene_name", acc))
    safe = re.sub(r'[/\\?*\[\]:\'"]', "_", gene[:12])
    if acc not in mat_corr.index: continue
    abund = mat_corr.loc[acc]
    dot = {}
    for resp_def, r_list, nr_list in [
        ("MADRS_R", MADRS_R_ids, MADRS_NR_ids),
        ("HAMD_R",  HAMD_R_ids,  HAMD_NR_ids),
    ]:
        for sex_k in ["F", "M"]:
            r_sex  = [p for p in r_list  if meta_matched.loc[p,"sex"]==sex_k]
            nr_sex = [p for p in nr_list if meta_matched.loc[p,"sex"]==sex_k]
            if r_sex:  dot[f"{resp_def}_{sex_k}"]    = pd.Series(abund[r_sex].values)
            if nr_sex: dot[f"{resp_def}_NR_{sex_k}"] = pd.Series(abund[nr_sex].values)
    if dot:
        write_sheet(pd.DataFrame(dot), f"Dot {safe}"[:31]); dot_count += 1
    scat = {
        "patient_number": [patient_number[p] for p in pids],
        "sex":            meta_matched["sex"].values,
        "hamd_responder": meta_matched["hamd_responder"].values,
        "madrs_responder":meta_matched["madrs_responder"].values,
        "madrs_remission":meta_matched["madrs_remission"].values,
        "abundance":      abund.values,
        "madrs_pct_reduction":(meta_matched["madrs_pct_reduction"].values
                               if "madrs_pct_reduction" in meta_matched.columns
                               else np.full(n_total, np.nan)),
        "hamd_pct_reduction": (meta_matched["hamd_pct_reduction"].values
                               if "hamd_pct_reduction" in meta_matched.columns
                               else np.full(n_total, np.nan)),
    }
    write_sheet(pd.DataFrame(scat), f"Scatter {safe}"[:31],
                {"abundance":fc_fmt,"madrs_pct_reduction":fc_fmt,
                 "hamd_pct_reduction":fc_fmt})
    scatter_count += 1

writer.close()
print(f"\n  Excel saved: {output_xlsx}")
print(f"  Sheets: README + 9 results + {dot_count} dot + {scatter_count} scatter")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("PIPELINE COMPLETE (v3)")
print("=" * 60)
print(f"  Cohort:                 n={n_total}  (F={n_F}, M={n_M})")
print(f"  Proteins retained:      {n_prot}")
print(f"  HAMD response:  R={n_HAMD_R}, NR={n_HAMD_NR}  (delta <= {HAMD_RESPONDER_THRESHOLD})")
print(f"  MADRS response: R={n_MADRS_R}, NR={n_MADRS_NR} (delta <= {MADRS_RESPONDER_THRESHOLD})")
print(f"  HAMD remission: R={(meta_matched['hamd_remission']=='R').sum()}, "
      f"NR={(meta_matched['hamd_remission']=='NR').sum()}  (post <= {HAMD_REMISSION_CUTOFF})")
print(f"  MADRS remission:R={(meta_matched['madrs_remission']=='R').sum()}, "
      f"NR={(meta_matched['madrs_remission']=='NR').sum()}  (post <= {MADRS_REMISSION_CUTOFF})")
print(f"  HAMD 50%:       R={(meta_matched['hamd_responder_50pct']=='R').sum()}, "
      f"NR={(meta_matched['hamd_responder_50pct']=='NR').sum()}")
print(f"  MADRS 50%:      R={(meta_matched['madrs_responder_50pct']=='R').sum()}, "
      f"NR={(meta_matched['madrs_responder_50pct']=='NR').sum()}")
print(f"  Scenario A (MADRS):  {(res_df['scenario_madrs']=='A').sum()} proteins")
print(f"  Scenario A (HAMD):   {(res_df['scenario_hamd']=='A').sum()} proteins")
print(f"  LASSO panels: HAMD-native ({len(lasso_panels['HAMD-native'])}), "
      f"MADRS-native ({len(lasso_panels['MADRS-native'])})")
for d in ["output/figures","output/results"]:
    n_files = len(os.listdir(d))
    print(f"  {d}/  ({n_files} files)")
