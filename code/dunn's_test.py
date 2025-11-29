"""
Dunn's Post-hoc Analysis from Raw Audio (MP3)
============================================

Pipeline:
1. Upload a ZIP that contains folders with MP3 audio files.
2. Unzip the dataset.
3. Walk all audio files, using the parent folder name as the class label.
4. Extract acoustic features per file:
   - Duration (seconds)
   - Mean F0 (Hz) via YIN
   - Spectral centroid (Hz)
   - RMS energy (dB)
5. Save a feature table (CSV).
6. Run Kruskal–Wallis tests for each feature across classes.
7. For features with significant KW results, run Dunn's post-hoc with
   Bonferroni correction.
8. Save result tables (CSV) and plots (boxplots + Dunn heatmap).

"""

# ============================================================
# Section 0: Install and import dependencies
# ============================================================

!pip install librosa soundfile

import os
import zipfile
import math
import itertools

import numpy as np
import pandas as pd
import librosa

from scipy.stats import kruskal, norm
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.grid"] = True

# ============================================================
# Section 1: Upload and unzip your audio ZIP
# ============================================================


USE_COLAB_UPLOAD = True  # set to False if running locally otherwise TRUE 

AUDIO_ROOT = "/content/audio_dataset"  # where we will unzip

if USE_COLAB_UPLOAD:
    from google.colab import files

    print("Please upload your ZIP file containing the audio dataset...")
    uploaded = files.upload()  # opens a file chooser

    # Take the first uploaded file name
    zip_filename = list(uploaded.keys())[0]
    print("Uploaded ZIP:", zip_filename)

    # Unzip into AUDIO_ROOT
    os.makedirs(AUDIO_ROOT, exist_ok=True)
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(AUDIO_ROOT)

    print("Unzipped to:", AUDIO_ROOT)

else:
    # LOCAL USE EXAMPLE:
    ZIP_PATH = "path/to/your_dataset.zip"
    os.makedirs(AUDIO_ROOT, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(AUDIO_ROOT)
    print("Unzipped to:", AUDIO_ROOT)

# ============================================================
# Section 2: Walk audio files and define how to get class labels
# ============================================================

AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg", ".m4a")

audio_files = []

for root, dirs, files in os.walk(AUDIO_ROOT):
    for fname in files:
        if fname.lower().endswith(AUDIO_EXTENSIONS):
            full_path = os.path.join(root, fname)
            audio_files.append(full_path)

print(f"Found {len(audio_files)} audio files.")

# Sanity check: show a few example paths
for path in audio_files[:5]:
    print("Example file:", path)

# ============================================================
# Section 3: Feature extraction function
# ============================================================

"""
We extract 4 basic features per audio file:

1. Duration (seconds)
2. Mean F0 (Hz) using librosa.yin
3. Spectral centroid (Hz)
4. RMS energy (dB)
"""

def extract_features_for_file(filepath,
                              sr_target=22050,
                              fmin=50.0,
                              fmax=500.0):
    """
    Extracts acoustic features for one audio file.

    Parameters
    ----------
    filepath : str
        Path to the audio file.
    sr_target : int
        Target sampling rate for loading audio.
    fmin, fmax : float
        Min and max frequency (Hz) for F0 estimation (YIN).

    Returns
    -------
    features : dict
        {
          "duration_s": float,
          "F0_mean_Hz": float or np.nan,
          "spec_centroid_Hz": float,
          "RMS_energy_dB": float
        }
    """
    # Load audio as mono
    y, sr = librosa.load(filepath, sr=sr_target, mono=True)

    # Duration
    duration_s = librosa.get_duration(y=y, sr=sr)

    # RMS energy (frame-wise) and convert to dB
    rms = librosa.feature.rms(y=y)[0]  # shape (frames,)
    rms_mean = float(rms.mean())
    # Avoid log of zero
    rms_mean = max(rms_mean, 1e-10)
    rms_db = 20 * np.log10(rms_mean)

    # Spectral centroid (frame-wise)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spec_centroid_mean = float(spec_centroid.mean())

    # F0 estimation with YIN
    try:
        f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr)
        f0 = np.asarray(f0)
        # Keep only finite positive F0 values
        f0_valid = f0[np.isfinite(f0) & (f0 > 0)]
        if len(f0_valid) > 0:
            F0_mean_Hz = float(f0_valid.mean())
        else:
            F0_mean_Hz = np.nan
    except Exception as e:
        print(f"Warning: F0 estimation failed for {filepath}: {e}")
        F0_mean_Hz = np.nan

    return {
        "duration_s": duration_s,
        "F0_mean_Hz": F0_mean_Hz,
        "spec_centroid_Hz": spec_centroid_mean,
        "RMS_energy_dB": rms_db,
    }

# ============================================================
# Section 4: Loop over all audio files and build feature table
# ============================================================

records = []

for idx, filepath in enumerate(audio_files, start=1):
    # Class label = parent folder name
    parent_dir = os.path.basename(os.path.dirname(filepath))
    class_label = parent_dir

    try:
        feats = extract_features_for_file(filepath)
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        continue

    rec = {
        "FilePath": filepath,
        "FileName": os.path.basename(filepath),
        "ClassLabel": class_label,
    }
    rec.update(feats)
    records.append(rec)

    if idx % 50 == 0:
        print(f"Processed {idx} files...")

print(f"Finished feature extraction for {len(records)} files.")

features_df = pd.DataFrame(records)
print(features_df.head())

# Save feature table to CSV
FEATURES_CSV_PATH = "cow_acoustic_features.csv"
features_df.to_csv(FEATURES_CSV_PATH, index=False)
print("Saved feature table to:", FEATURES_CSV_PATH)

# ============================================================
# Section 5: Prepare data for Kruskal–Wallis and Dunn
# ============================================================

MIN_SAMPLES_PER_CLASS = 10     
ALPHA = 0.05                  
OUTPUT_DIR = "dunn_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GROUP_COL = "ClassLabel"
FEATURE_COLS = ["duration_s", "F0_mean_Hz", "spec_centroid_Hz", "RMS_energy_dB"]

# Drop rows with missing class or any of the features
df = features_df[[GROUP_COL] + FEATURE_COLS].dropna()
print("\nRows after dropping NA:", len(df))

# Filter classes with at least MIN_SAMPLES_PER_CLASS
class_counts = df[GROUP_COL].value_counts()
valid_classes = class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].index.tolist()

print("\nClasses with at least", MIN_SAMPLES_PER_CLASS, "samples:")
for c in valid_classes:
    print(f"  {c}: {class_counts[c]} clips")

df = df[df[GROUP_COL].isin(valid_classes)].copy()
CLASS_ORDER = sorted(valid_classes)
print("\nClass order used in analysis:", CLASS_ORDER)

# ============================================================
# Section 6: Define Dunn's post-hoc function
# ============================================================

def dunn_posthoc(data, group_col, value_col, p_adjust='bonferroni'):
    """
    Dunn's post-hoc test for multiple pairwise comparisons
    following a Kruskal–Wallis test.

    Parameters
    ----------
    data : pandas.DataFrame
        Must contain 'group_col' and 'value_col'.
    group_col : str
        Column with group labels (e.g., 'ClassLabel').
    value_col : str
        Column with numeric feature.
    p_adjust : str
        'bonferroni' or 'none'.

    Returns
    -------
    pval_matrix : pandas.DataFrame
        Symmetric matrix of adjusted p-values for all group pairs.
    """
    df_sub = data[[group_col, value_col]].dropna().copy()
    groups = np.sort(df_sub[group_col].unique())

    # 1. Global ranking of all observations for this feature
    df_sub = df_sub.sort_values(value_col)
    df_sub["rank"] = np.arange(1, len(df_sub) + 1)
    # Average ranks for tied values
    df_sub["rank"] = df_sub.groupby(value_col)["rank"].transform("mean")

    N = len(df_sub)
    if N <= 1:
        raise ValueError("Not enough data points for Dunn's test.")

    # 2. Tie correction term for variance
    tie_counts = df_sub.groupby(value_col).size().values
    tie_term = np.sum(tie_counts**3 - tie_counts)
    var_term = (N * (N + 1) / 12.0) - (tie_term / (12.0 * (N - 1)))

    # 3. Mean ranks and sample sizes for each group
    mean_ranks = {}
    ns = {}
    for g in groups:
        r = df_sub.loc[df_sub[group_col] == g, "rank"]
        mean_ranks[g] = r.mean()
        ns[g] = r.size

    # 4. Pairwise z and p for each pair of groups
    pair_pvals = {}
    for (i, j) in itertools.combinations(range(len(groups)), 2):
        gi, gj = groups[i], groups[j]
        Ri, Rj = mean_ranks[gi], mean_ranks[gj]
        ni, nj = ns[gi], ns[gj]

        se = math.sqrt(var_term * (1.0 / ni + 1.0 / nj))
        z = (Ri - Rj) / se

        p = 2 * (1 - norm.cdf(abs(z)))  # two-sided p-value
        pair_pvals[(gi, gj)] = p

    # 5. Bonferroni correction
    m = len(pair_pvals)
    if p_adjust == "bonferroni":
        pair_pvals = {k: min(v * m, 1.0) for k, v in pair_pvals.items()}
    elif p_adjust in (None, "none"):
        pass
    else:
        raise NotImplementedError("Only 'bonferroni' is implemented here.")

    # 6. Build symmetric p-value matrix
    pval_matrix = pd.DataFrame(1.0, index=groups, columns=groups, dtype=float)
    for (gi, gj), p in pair_pvals.items():
        pval_matrix.loc[gi, gj] = p
        pval_matrix.loc[gj, gi] = p

    return pval_matrix

# ============================================================
# Section 7: Kruskal–Wallis omnibus tests
# ============================================================

print("\nRunning Kruskal–Wallis tests for each feature...")

kruskal_results = []

for feat in FEATURE_COLS:
    print(f"\nFeature: {feat}")
    grouped_vals = [
        df.loc[df[GROUP_COL] == g, feat].values
        for g in CLASS_ORDER
    ]
    H, p_kw = kruskal(*grouped_vals)
    print(f"  H = {H:.3f}, p = {p_kw:.3e}")

    kruskal_results.append({
        "feature": feat,
        "H": H,
        "p_value": p_kw
    })

kw_df = pd.DataFrame(kruskal_results)
kw_path = os.path.join(OUTPUT_DIR, "kruskal_wallis_summary.csv")
kw_df.to_csv(kw_path, index=False)
print("\nSaved Kruskal–Wallis summary to:", kw_path)
print(kw_df)

# ============================================================
# Section 8: Dunn's post-hoc tests + summary of significant pairs
# ============================================================

print("\nRunning Dunn's post-hoc tests (Bonferroni) where KW is significant...")

significant_pairs_all = []

for feat in FEATURE_COLS:
    H_val = kw_df.loc[kw_df["feature"] == feat, "H"].values[0]
    p_kw = kw_df.loc[kw_df["feature"] == feat, "p_value"].values[0]

    if p_kw >= ALPHA:
        print(f"\nFeature {feat}: KW p = {p_kw:.3e} (not significant) -> Dunn skipped.")
        continue

    print(f"\nFeature {feat}: KW p = {p_kw:.3e} (significant) -> running Dunn's test...")

    dunn_mat = dunn_posthoc(df, group_col=GROUP_COL,
                            value_col=feat, p_adjust="bonferroni")

    # Save full p-value matrix
    dunn_csv_path = os.path.join(OUTPUT_DIR, f"dunn_pvalues_{feat}.csv")
    dunn_mat.to_csv(dunn_csv_path)
    print("  Saved Dunn p-value matrix to:", dunn_csv_path)

    # Medians per class
    medians = df.groupby(GROUP_COL)[feat].median()
    groups = dunn_mat.index.tolist()

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            gi, gj = groups[i], groups[j]
            p_adj = dunn_mat.loc[gi, gj]
            if p_adj < ALPHA:
                delta_median = medians[gi] - medians[gj]
                significant_pairs_all.append({
                    "feature": feat,
                    "class_1": gi,
                    "class_2": gj,
                    "p_adj": p_adj,
                    "median_1": medians[gi],
                    "median_2": medians[gj],
                    "delta_median": delta_median
                })

# Save summary of significant pairs across all features
sig_pairs_df = pd.DataFrame(significant_pairs_all)
sig_pairs_path = os.path.join(OUTPUT_DIR, "dunn_significant_pairs_summary.csv")
sig_pairs_df.to_csv(sig_pairs_path, index=False)
print("\nSaved summary of significant pairs to:", sig_pairs_path)
print(sig_pairs_df.head())

# ============================================================
# Section 9: Boxplots for each feature across classes
# ============================================================

print("\nGenerating boxplots for each feature...")

for feat in FEATURE_COLS:
    plt.figure()
    df.boxplot(column=feat, by=GROUP_COL)
    plt.title(f"{feat} by class")
    plt.suptitle("")
    plt.xlabel("Class")
    plt.ylabel(feat)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    fig_path = os.path.join(OUTPUT_DIR, f"boxplot_{feat}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print("  Saved boxplot to:", fig_path)

# ============================================================
# Section 10: Dunn heatmap for one key feature (e.g., F0)
# ============================================================

KEY_FEATURE_FOR_HEATMAP = "F0_mean_Hz"
print(f"\nGenerating Dunn p-value heatmap for feature: {KEY_FEATURE_FOR_HEATMAP}")

if KEY_FEATURE_FOR_HEATMAP in kw_df["feature"].values:
    p_kw_key = kw_df.loc[kw_df["feature"] == KEY_FEATURE_FOR_HEATMAP, "p_value"].values[0]
    if p_kw_key < ALPHA:
        # Recompute Dunn matrix (or load from CSV)
        dunn_mat_key = dunn_posthoc(df, group_col=GROUP_COL,
                                    value_col=KEY_FEATURE_FOR_HEATMAP,
                                    p_adjust="bonferroni")

        pvals = dunn_mat_key.values
        classes = dunn_mat_key.index.tolist()

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(pvals, interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Bonferroni-adjusted p-value")

        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes)

        ax.set_title(f"Dunn's test (p-values) for {KEY_FEATURE_FOR_HEATMAP}")
        plt.tight_layout()

        heatmap_path = os.path.join(OUTPUT_DIR,
                                    f"dunn_heatmap_{KEY_FEATURE_FOR_HEATMAP}.png")
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        print("  Saved Dunn heatmap to:", heatmap_path)
    else:
        print("  KW for key feature not significant; heatmap skipped.")
else:
    print("  Key feature not found in KW results; heatmap skipped.")

print("\nAll analyses complete.")