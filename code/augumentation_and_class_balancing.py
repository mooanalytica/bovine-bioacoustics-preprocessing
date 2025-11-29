# This script:
#   1. Indexes the curated dataset (folders = class labels, files = .wav)
#   2. Performs a per-class Train/Val/Test split
#   3. Applies class balancing on the TRAIN split:
#        - Majority classes: undersample down to TARGET_PER_CLASS
#        - Minority classes: augment up to TARGET_PER_CLASS with:
#            * time-stretching (0.8–1.2×)
#            * pitch shifting (±2 semitones)
#            * Gaussian noise (SNR 20–30 dB)
#            * gain adjustment (±6 dB)
#        - Classes with < MIN_TRAIN_CLIPS original clips: excluded from training
#   4. Writes a new balanced dataset + a metadata CSV
#
#
# Expected input structure:
#   DATA_ROOT/
#       <label_1>/*.wav
#       <label_2>/*.wav
#       ...
#
# Output structure:
#   OUT_ROOT/
#       train/<label>/*.wav    (original + augmented)
#       val/<label>/*.wav      (original only)
#       test/<label>/*.wav     (original only)
#       excluded/<label>/*.wav (labels with <3 clips)
#       metadata_balanced.csv
#

# ============================================================
# 0) Install libraries
# ============================================================
try:
    import librosa, soundfile
except ImportError:
    !pip -q install librosa soundfile tqdm

# ============================================================
# 1) Imports & Config
# ============================================================
import os, shutil, random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm.auto import tqdm

# ---- Paths ----
DATA_ROOT = Path("/content/cow_dataset_curated")
OUT_ROOT  = Path("/content/cow_dataset_balanced")

# ---- Random seed for reproducibility ----
SEED = 0
random.seed(SEED)
np.random.seed(SEED)

# ---- Audio & augmentation hyperparameters ----
SR               = 16000          # analysis sampling rate
MIN_TRAIN_CLIPS  = 3              # labels with <3 clips are excluded from training
TARGET_PER_CLASS = 81             # target per-class size for TRAIN split

# augmentation parameter ranges
STRETCH_RANGE    = (0.8, 1.2)     # time-stretching factor
PITCH_STEPS      = (-2.0, 2.0)    # semitones
SNR_RANGE_DB     = (20.0, 30.0)   # target SNR for Gaussian noise
GAIN_DB_RANGE    = (-6.0, 6.0)    # gain adjustment in dB

AUG_PROBS = {                 
    "time_stretch": 0.30,
    "pitch_shift":  0.30,
    "gaussian_noise": 0.40,
    "gain":         0.30,
}

# ============================================================
# 2) Index dataset
# ============================================================

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}

def index_dataset(root: Path) -> pd.DataFrame:
    rows = []
    for label_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        label = label_dir.name
        files = sorted(
            f for f in label_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in AUDIO_EXTS
        )
        if not files:
            rows.append({"filepath": None, "label": label, "exists": False})
            continue
        for f in files:
            rows.append({"filepath": str(f), "label": label, "exists": True})
    return pd.DataFrame(rows)

meta = index_dataset(DATA_ROOT)
print("Total rows (incl. empties):", len(meta))
print("Unique labels (directories):", meta["label"].nunique())
display(meta.head())

# Filter to existing audio files only
meta_existing = meta[meta["exists"] == True].copy()

# Per-class counts in the curated dataset
counts_all = meta_existing.groupby("label").size().sort_values(ascending=False)
print("\nPer-class counts (original curated dataset):")
display(counts_all)

labels_for_training  = counts_all[counts_all >= MIN_TRAIN_CLIPS].index.tolist()
labels_too_rare      = counts_all[counts_all <  MIN_TRAIN_CLIPS].index.tolist()

print(f"\nLabels used for training (≥{MIN_TRAIN_CLIPS} clips): {len(labels_for_training)}")
print("Labels excluded from training (<3 clips):", labels_too_rare)

# ============================================================
# 3) Train / Val / Test split
# ============================================================

def split_per_label(df_label: pd.DataFrame,
                    train_frac: float = 0.7,
                    val_frac: float   = 0.15,
                    seed: int         = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-label split: tries approx 70/15/15 train/val/test."""
    rng = np.random.default_rng(seed)
    n = len(df_label)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_train = max(0, int(round(train_frac * n)))
    n_val   = max(0, int(round(val_frac * n)))
    n_train = min(n_train, n)
    n_val   = min(n_val, n - n_train)
    n_test  = n - n_train - n_val

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    return (
        df_label.iloc[train_idx].copy(),
        df_label.iloc[val_idx].copy(),
        df_label.iloc[test_idx].copy(),
    )

train_df_list, val_df_list, test_df_list, excluded_df_list = [], [], [], []

for label in sorted(counts_all.index):
    df_label = meta_existing[meta_existing["label"] == label].copy()
    n = len(df_label)

    if label in labels_too_rare:
        # Excluded from training, but kept in 'excluded' split
        df_label["split"] = "excluded"
        excluded_df_list.append(df_label)
        continue

    # Labels with ≥ MIN_TRAIN_CLIPS: perform train/val/test split
    tr, va, te = split_per_label(df_label, seed=SEED)

    tr["split"] = "train"
    va["split"] = "val"
    te["split"] = "test"

    train_df_list.append(tr)
    val_df_list.append(va)
    test_df_list.append(te)

train_df = pd.concat(train_df_list, ignore_index=True)
val_df   = pd.concat(val_df_list,   ignore_index=True)
test_df  = pd.concat(test_df_list,  ignore_index=True)
excluded_df = pd.concat(excluded_df_list, ignore_index=True) if excluded_df_list else pd.DataFrame()

print("\nSplit sizes:")
print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df), "Excluded:", len(excluded_df))

print("\nTrain counts BEFORE balancing:")
display(train_df["label"].value_counts().sort_index())

# ============================================================
# 4) Augmentation primitives
# ============================================================

def load_audio(path: str, sr: int = SR) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y

def time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    return librosa.effects.time_stretch(y, rate=rate)

def pitch_shift_samples(y: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def add_gaussian_noise_snr(y: np.ndarray,
                           snr_range_db: Tuple[float, float] = SNR_RANGE_DB) -> np.ndarray:
    """Add white Gaussian noise at a random SNR in snr_range_db."""
    rms_signal = np.sqrt(np.mean(y ** 2) + 1e-12)
    snr_db = np.random.uniform(*snr_range_db)
    rms_noise_target = rms_signal / (10 ** (snr_db / 20.0))

    noise = np.random.normal(0.0, 1.0, size=len(y))
    rms_noise_current = np.sqrt(np.mean(noise ** 2) + 1e-12)
    noise *= (rms_noise_target / rms_noise_current)

    return y + noise

def apply_gain_db(y: np.ndarray,
                  db_range: Tuple[float, float] = GAIN_DB_RANGE) -> np.ndarray:
    gain_db = np.random.uniform(*db_range)
    gain_lin = 10 ** (gain_db / 20.0)
    return y * gain_lin

def match_length(y: np.ndarray, target_len: int) -> np.ndarray:
    """Center-pad or center-trim to exactly target_len samples."""
    n = len(y)
    if n == target_len:
        return y
    if n > target_len:
        start = (n - target_len) // 2
        return y[start:start + target_len]
    pad = target_len - n
    left = pad // 2
    right = pad - left
    return np.pad(y, (left, right))

def augment_clip(y: np.ndarray, sr: int = SR) -> Tuple[np.ndarray, str]:
    """Apply a random combination of augmentation operations."""
    y_aug = y.copy()
    ops = []
    base_len = len(y)

    # time-stretch
    if np.random.rand() < AUG_PROBS["time_stretch"]:
        rate = np.random.uniform(*STRETCH_RANGE)
        y_aug = time_stretch(y_aug, rate)
        y_aug = match_length(y_aug, base_len)
        ops.append(f"time_stretch_{rate:.2f}x")

    # pitch shift
    if np.random.rand() < AUG_PROBS["pitch_shift"]:
        steps = np.random.uniform(*PITCH_STEPS)
        y_aug = pitch_shift_samples(y_aug, sr, steps)
        ops.append(f"pitch_{steps:.2f}st")

    # gaussian noise
    if np.random.rand() < AUG_PROBS["gaussian_noise"]:
        y_aug = add_gaussian_noise_snr(y_aug)
        ops.append("gaussian_noise")

    # gain
    if np.random.rand() < AUG_PROBS["gain"]:
        y_aug = apply_gain_db(y_aug)
        ops.append("gain")

    # normalise to avoid clipping
    max_abs = np.max(np.abs(y_aug))
    if max_abs > 0.99:
        y_aug = y_aug / max_abs * 0.99

    ops_str = ";".join(ops) if ops else "none"
    return y_aug.astype(np.float32), ops_str

# ============================================================
# 5) Create output folders and copy VAL / TEST / EXCLUDED (no augmentation)
# ============================================================

for split in ["train", "val", "test", "excluded"]:
    (OUT_ROOT / split).mkdir(parents=True, exist_ok=True)

def copy_to_split(row: pd.Series, split: str) -> str:
    src = Path(row["filepath"])
    dst = OUT_ROOT / split / row["label"] / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)

val_records, test_records, excluded_records = [], [], []

print("\nCopying VAL originals...")
for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
    out_path = copy_to_split(row, "val")
    val_records.append({
        "filepath": out_path,
        "label": row["label"],
        "split": "val",
        "is_augmented": False,
        "source_file": row["filepath"],
        "augmentation_ops": "",
    })

print("Copying TEST originals...")
for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    out_path = copy_to_split(row, "test")
    test_records.append({
        "filepath": out_path,
        "label": row["label"],
        "split": "test",
        "is_augmented": False,
        "source_file": row["filepath"],
        "augmentation_ops": "",
    })

if not excluded_df.empty:
    print("Copying EXCLUDED originals (labels with <3 clips)...")
    for _, row in tqdm(excluded_df.iterrows(), total=len(excluded_df)):
        out_path = copy_to_split(row, "excluded")
        excluded_records.append({
            "filepath": out_path,
            "label": row["label"],
            "split": "excluded",
            "is_augmented": False,
            "source_file": row["filepath"],
            "augmentation_ops": "",
        })

# ============================================================
# 6) Apply class balancing on TRAIN split
# ============================================================

train_balanced_records = []

train_counts_before = train_df["label"].value_counts().sort_index()
print("\nTrain counts BEFORE balancing:")
display(train_counts_before)

for label, group in train_df.groupby("label"):
    files = list(group["filepath"])
    n_orig = len(files)

    if n_orig >= TARGET_PER_CLASS:
        # Majority class: undersample down to TARGET_PER_CLASS
        selected = np.random.choice(files, size=TARGET_PER_CLASS, replace=False)
        for src in selected:
            src_path = Path(src)
            dst = OUT_ROOT / "train" / label / src_path.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst)
            train_balanced_records.append({
                "filepath": str(dst),
                "label": label,
                "split": "train",
                "is_augmented": False,
                "source_file": src,
                "augmentation_ops": "",
            })
    else:
        # Minority class: keep all originals, then augment until TARGET_PER_CLASS
        for src in files:
            src_path = Path(src)
            dst = OUT_ROOT / "train" / label / src_path.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst)
            train_balanced_records.append({
                "filepath": str(dst),
                "label": label,
                "split": "train",
                "is_augmented": False,
                "source_file": src,
                "augmentation_ops": "",
            })

        n_needed = max(0, TARGET_PER_CLASS - n_orig)
        for k in range(n_needed):
            base_src = random.choice(files)
            y = load_audio(base_src, sr=SR)
            y_aug, ops_str = augment_clip(y, sr=SR)
            base_name = Path(base_src).stem
            dst = OUT_ROOT / "train" / label / f"{base_name}_aug{k+1:03d}.wav"
            dst.parent.mkdir(parents=True, exist_ok=True)
            sf.write(dst, y_aug, SR)

            train_balanced_records.append({
                "filepath": str(dst),
                "label": label,
                "split": "train",
                "is_augmented": True,
                "source_file": base_src,
                "augmentation_ops": ops_str,
            })

print("\nFinished balancing TRAIN split.")
print("Balanced train size:", len(train_balanced_records))

# ============================================================
# 7) Save final metadata and summary
# ============================================================

balanced_meta = pd.DataFrame(
    train_balanced_records + val_records + test_records + excluded_records
)

balanced_meta_path = OUT_ROOT / "metadata_balanced.csv"
balanced_meta.to_csv(balanced_meta_path, index=False)
print("Saved balanced metadata to:", balanced_meta_path)

print("\nFinal per-split counts (balanced dataset):")
print(balanced_meta["split"].value_counts())

print("\nFinal per-class counts (TRAIN only):")
display(
    balanced_meta[balanced_meta["split"] == "train"]["label"].value_counts().sort_index()
)
