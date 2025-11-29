# ============================
# LMM + Variance Components Pipeline
# (for acoustic_features_24 + metadata_full_clips)
# ============================

!pip install statsmodels --quiet

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt

from google.colab import files

# ----------------------------
# 1) Upload both files (CSV or XLSX)
# ----------------------------
print("Please upload acoustic_features_24 and metadata_full_clips files (CSV or XLSX)...")
uploaded = files.upload()

acoustic_path = None
meta_path = None
for fname in uploaded.keys():
    if "acoustic_features_24" in fname:
        acoustic_path = fname
    if "metadata_full_clips" in fname:
        meta_path = fname

if acoustic_path is None or meta_path is None:
    raise ValueError(
        "Could not find both 'acoustic_features_24*' and 'metadata_full_clips*' "
        "among uploaded files. Check file names."
    )

print(f"Using acoustic file: {acoustic_path}")
print(f"Using metadata file: {meta_path}")

# ----------------------------
# 2) Robust file readers
# ----------------------------

def safe_read_table(path):
    """Read CSV or Excel with fallback encodings for CSV."""
    lower = path.lower()
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        print(f"Reading {path} as Excel...")
        return pd.read_excel(path)
    else:
        for enc in ["utf-8", "latin1", "cp1252"]:
            try:
                print(f"Trying to read {path} as CSV with encoding='{enc}'...")
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError as e:
                print(f"  Failed with encoding='{enc}': {e}")
                continue
        raise UnicodeDecodeError(
            "file", b"", 0, 1,
            "Could not decode file with utf-8, latin1, or cp1252."
        )

# ----------------------------
# 3) Load, clean, and merge
# ----------------------------

acoustic_df = safe_read_table(acoustic_path)
meta_df     = safe_read_table(meta_path)

print("Acoustic features shape:", acoustic_df.shape)
print("Metadata shape:", meta_df.shape)

def clean_cols(df):
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace(r"[^0-9a-zA-Z_]", "", regex=True)
    )
    return df

acoustic_df = clean_cols(acoustic_df)
meta_df     = clean_cols(meta_df)

print("Columns in acoustic_df:", acoustic_df.columns.tolist())
print("Columns in meta_df:", meta_df.columns.tolist())

if "file_name" not in acoustic_df.columns or "file_name" not in meta_df.columns:
    raise ValueError(
        "'file_name' column not found in one of the dataframes after cleaning."
    )

df = pd.merge(meta_df, acoustic_df, on="file_name", how="inner")
print("Merged shape:", df.shape)

print("\nAll columns after merge:")
print(df.columns.tolist())

# ----------------------------
# 4) Clean categorical variables
# ----------------------------

df = df.copy()

cat_cols = [
    "Farm",
    "Barn_Zone",
    "Microphone",
    "Mic_Placement_Context",
    "Main_Category_final",
    "Sub_Category_final",
]

def clean_text_series(s):
    # convert to string, remove non-breaking spaces, strip whitespace
    return (
        s.astype(str)
         .str.replace("\xa0", " ", regex=False)
         .str.strip()
         .fillna("Unknown")
    )

for col in cat_cols:
    if col in df.columns:
        df[col] = clean_text_series(df[col])
    else:
        print(f"WARNING: expected categorical column '{col}' not found in df.")

if "Farm" in df.columns:
    df["Farm"] = df["Farm"].fillna("Unknown").astype(str)

# ----------------------------
# 5) Select numeric features
# ----------------------------

candidate_numeric_features = [
    "duration_s",
    "f0_mean_hz",
    "spectral_centroid_mean",
    "rms_energy_mean",
]

numeric_features = [f for f in candidate_numeric_features if f in df.columns]
if not numeric_features:
    raise ValueError(
        "None of the candidate numeric features were found in df. "
        "Check column names and update 'candidate_numeric_features'."
    )

print("\nNumeric features selected for modelling:", numeric_features)

if "Sub_Category_final" not in df.columns:
    raise ValueError("'Sub_Category_final' not found in df. Check column names.")

# Filter out very rare subcategories
min_n = 5
class_counts = df["Sub_Category_final"].value_counts()
keep_classes = class_counts[class_counts >= min_n].index
df_lmm = df[df["Sub_Category_final"].isin(keep_classes)].copy()
df_lmm = df_lmm.reset_index(drop=True)

print("\nNumber of clips after filtering rare classes:", df_lmm.shape[0])
print("Subcategories kept:", list(keep_classes))

# ----------------------------
# 6) Helper functions: LMM + variance components
# ----------------------------

def prep_data_for_model(df_data, feature, include_class=True):
    """
    Create a clean dataframe for modelling:
    - keep only needed columns
    - drop NA
    - reset index
    """
    cols = [feature, "Farm"]
    if include_class and "Sub_Category_final" in df_data.columns:
        cols.append("Sub_Category_final")
    if "Barn_Zone" in df_data.columns:
        cols.append("Barn_Zone")
    if "Microphone" in df_data.columns:
        cols.append("Microphone")

    dfm = df_data[cols].dropna().copy()
    dfm = dfm.reset_index(drop=True)
    return dfm

def fit_lmm_for_feature(df_data, feature):
    """
    Linear Mixed Model:
      feature ~ C(Sub_Category_final)
      random intercepts: Farm, Barn_Zone, Microphone
    Returns: full_fit, null_fit, lr_stat, df_diff, p_value, classes_used
    """
    dfm = prep_data_for_model(df_data, feature, include_class=True)

    if "Farm" not in dfm.columns:
        raise ValueError("'Farm' column not in dfm. Needed for random groups.")
    if "Sub_Category_final" not in dfm.columns:
        raise ValueError("'Sub_Category_final' column not in dfm.")

    vc_formula = {}
    if "Barn_Zone" in dfm.columns:
        vc_formula["BarnZone"] = "0 + C(Barn_Zone)"
    if "Microphone" in dfm.columns:
        vc_formula["Mic"] = "0 + C(Microphone)"

    # Full model with class effect
    formula_full = f"{feature} ~ C(Sub_Category_final)"
    md_full = smf.mixedlm(
        formula_full,
        data=dfm,
        groups=dfm["Farm"],
        vc_formula=vc_formula
    )
    full_fit = md_full.fit(
        reml=False, method="lbfgs", maxiter=500, full_output=True
    )

    # Null model without class
    formula_null = f"{feature} ~ 1"
    md_null = smf.mixedlm(
        formula_null,
        data=dfm,
        groups=dfm["Farm"],
        vc_formula=vc_formula
    )
    null_fit = md_null.fit(
        reml=False, method="lbfgs", maxiter=500, full_output=True
    )

    # Likelihood ratio test
    lr_stat = 2 * (full_fit.llf - null_fit.llf)
    df_diff = len(full_fit.fe_params) - len(null_fit.fe_params)
    p_value = stats.chi2.sf(lr_stat, df_diff)

    classes_used = sorted(dfm["Sub_Category_final"].unique())
    return full_fit, null_fit, lr_stat, df_diff, p_value, classes_used

def fit_vc_for_feature(df_data, feature):
    """
    Variance components model:
      feature ~ 1
      random: Farm, Barn_Zone, Microphone, Sub_Category_final
    """
    dfm = prep_data_for_model(df_data, feature, include_class=True)

    if "Farm" not in dfm.columns:
        raise ValueError("'Farm' column not in dfm. Needed for random groups.")

    vc_formula = {}
    if "Barn_Zone" in dfm.columns:
        vc_formula["BarnZone"] = "0 + C(Barn_Zone)"
    if "Microphone" in dfm.columns:
        vc_formula["Mic"] = "0 + C(Microphone)"
    if "Sub_Category_final" in dfm.columns:
        vc_formula["Class"] = "0 + C(Sub_Category_final)"

    md = smf.mixedlm(
        f"{feature} ~ 1",
        data=dfm,
        groups=dfm["Farm"],
        vc_formula=vc_formula
    )
    fit = md.fit(
        reml=True, method="lbfgs", maxiter=500, full_output=True
    )
    return fit

def extract_vc_table(fit, feature_name):
    """
    Extract variance components into a tidy table for one feature.
    Assumes vc_formula order in summary is:
      BarnZone Var, Class Var, Mic Var
    so vcomp = [BarnZone, Class, Mic].
    Farm variance (group intercept) is in cov_re if non-empty.
    """
    # Farm (group) variance
    var_farm = 0.0
    if hasattr(fit, "cov_re") and fit.cov_re is not None:
        try:
            if isinstance(fit.cov_re, pd.DataFrame) and fit.cov_re.size > 0:
                var_farm = float(np.diag(fit.cov_re)[0])
        except Exception:
            var_farm = 0.0

    # Other components from vcomp (NumPy array)
    var_zone  = 0.0
    var_class = 0.0
    var_mic   = 0.0

    if hasattr(fit, "vcomp") and fit.vcomp is not None:
        v_arr = np.asarray(fit.vcomp).ravel()
        # Map by position: [BarnZone Var, Class Var, Mic Var]
        if v_arr.size >= 1:
            var_zone = float(v_arr[0])
        if v_arr.size >= 2:
            var_class = float(v_arr[1])
        if v_arr.size >= 3:
            var_mic = float(v_arr[2])

    # Residual variance
    var_resid = float(fit.scale)

    data = {
        "feature":   [feature_name] * 5,
        "component": [
            "Vocalization class",
            "Farm",
            "Barn zone",
            "Microphone",
            "Residual"
        ],
        "variance": [
            var_class,
            var_farm,
            var_zone,
            var_mic,
            var_resid
        ],
    }

    vc_df = pd.DataFrame(data)
    total = vc_df["variance"].sum()
    if total > 0:
        vc_df["perc_variance"] = 100 * vc_df["variance"] / total
    else:
        vc_df["perc_variance"] = np.nan
    return vc_df

# ----------------------------
# 7) Run LMMs for all numeric features
# ----------------------------

lmm_results = []
class_means_lmm = {}  # feature -> DataFrame(Sub_Category_final, lmm_adjusted_mean)

for feat in numeric_features:
    print("\n" + "="*80)
    print(f"LMM for feature: {feat}")
    full_fit, null_fit, lr_stat, df_diff, p_value, classes_used = fit_lmm_for_feature(df_lmm, feat)

    print(full_fit.summary())

    lmm_results.append({
        "feature": feat,
        "lr_stat": lr_stat,
        "df_diff": df_diff,
        "p_value": p_value
    })

    # LMM-adjusted means per class
    means = []
    for cl in classes_used:
        tmp = pd.DataFrame({"Sub_Category_final": [cl]})
        pred = full_fit.predict(exog=tmp)[0]
        means.append(pred)

    class_means_lmm[feat] = pd.DataFrame({
        "Sub_Category_final": classes_used,
        "lmm_adjusted_mean": means
    })

lmm_results_df = pd.DataFrame(lmm_results)
print("\nLMM Likelihood Ratio Test results (effect of Sub_Category_final):")
print(lmm_results_df)

lmm_results_df.to_csv("lmm_results_summary.csv", index=False)
print("Saved: lmm_results_summary.csv")

# ----------------------------
# 8) Variance components for all numeric features
# ----------------------------

vc_all_features = []

for feat in numeric_features:
    print("\n" + "="*80)
    print(f"Variance component model for feature: {feat}")
    vc_fit = fit_vc_for_feature(df_lmm, feat)
    print(vc_fit.summary())

    vc_table = extract_vc_table(vc_fit, feat)
    vc_all_features.append(vc_table)

vc_all_df = pd.concat(vc_all_features, ignore_index=True)

print("\nVariance components summary for all features:")
print(vc_all_df)

vc_all_df.to_csv("variance_components_summary.csv", index=False)
print("Saved: variance_components_summary.csv")

# ----------------------------
# 9) Plots – variance partitioning
# ----------------------------

for feat in numeric_features:
    vc_one = vc_all_df[vc_all_df["feature"] == feat]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(vc_one["component"], vc_one["perc_variance"])
    ax.set_xlabel("Variance explained (%)")
    ax.set_title(f"Variance partitioning for {feat}")
    plt.tight_layout()
    out_name = f"var_partition_{feat}.png"
    fig.savefig(out_name, dpi=300)
    plt.close(fig)
    print(f"Saved variance partition plot: {out_name}")

# ----------------------------
# 10) Plots – LMM-adjusted class means
# ----------------------------

for feat in numeric_features:
    df_means = class_means_lmm[feat].sort_values("lmm_adjusted_mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df_means["Sub_Category_final"], df_means["lmm_adjusted_mean"])
    ax.set_xlabel(f"LMM-adjusted mean {feat}")
    ax.set_ylabel("Vocalization subcategory")
    ax.set_title(f"LMM-adjusted class means for {feat}")
    plt.tight_layout()
    out_name = f"lmm_class_means_{feat}.png"
    fig.savefig(out_name, dpi=300)
    plt.close(fig)
    print(f"Saved LMM class means plot: {out_name}")

print("\nAll done! Download:")
print("- lmm_results_summary.csv")
print("- variance_components_summary.csv")
print("- var_partition_<feature>.png")
print("- lmm_class_means_<feature>.png")

from google.colab import files
import glob
import zipfile
import os

# Download the two CSVs
files.download("lmm_results_summary.csv")
files.download("variance_components_summary.csv")

# Zip all PNGs (variance plots + class means plots) into one file
png_files = glob.glob("var_partition_*.png") + glob.glob("lmm_class_means_*.png")
print("PNG files found:", png_files)

zip_name = "lmm_variance_plots.zip"
with zipfile.ZipFile(zip_name, "w") as zf:
    for f in png_files:
        zf.write(f)

# Download the zip with all figures
files.download(zip_name)