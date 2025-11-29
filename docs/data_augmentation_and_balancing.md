# Data Augmentation and Class-Balancing Strategy

This document describes the data augmentation and class-balancing procedure
used to expand the curated bovine vocalization dataset from **569** clips to
approximately **2,900** training clips.

The goals were to:

- Reduce class imbalance across the 48 behavioral call categories
- Preserve biologically realistic variability in cow vocalizations
- Keep validation and test sets strictly **unaugmented**

---

## 1. Starting Point and Motivation

- Curated dataset: **569** manually segmented, denoised clips
- Behavioral granularity: **48** labeled vocal / non-vocal classes
- Class distribution: some classes with **>100** clips, others with very few

This severe imbalance would bias supervised models toward majority classes, so
a hybrid strategy combining **undersampling** and **augmentation** was used.

---

## 2. Hybrid Class-Balancing Strategy

To make the augmentation and balancing procedure explicit, we adopted the following
rules:

1. **Target threshold**

   - A target of **81 clips per class** was chosen based on the **median class size**.

2. **Majority-class undersampling**

   - Classes with more than **81** original clips (e.g., *Estrus Call* with
     117 clips, *Feed Anticipation Call* with 113 clips) were **randomly
     undersampled to 81** clips **for the training set**.
   - This prevents majority classes from dominating the model.

3. **Minority-class oversampling via augmentation**

   - Classes with **3–81 original clips** were **oversampled by augmentation**
     until they reached **approximately 81** training examples.
   - Augmentation operations (see Section 3) were applied to the original clips
     with randomized parameters.

4. **Exclusion of extremely rare classes**

   - Classes with **fewer than 3 original clips** were **excluded from model
     training** (to avoid severe overfitting).
   - These labels remain documented for transparency and future work but do not
     contribute to the current supervised models.

5. **Unaugmented validation and test splits**

   - The **validation** and **test** sets contain **only original, unaugmented
     clips**.
   - All balancing and augmentation is restricted to the **training** split.

After applying these rules, the **training set** expanded from 569 clips to
approximately **2,900** clips (≈48 classes × ≈60 training examples on average
after under/oversampling), while validation and test sets remained unchanged
and unaugmented.

---

## 3. Augmentation Operations and Parameter Ranges

Augmentation was applied **only** to clips in the **training** split and
only for classes with **3–81** original clips. Each augmented clip was
generated from an original clip using one or more of the following
“biologically informed” perturbations:

1. **Time-stretching**

   - Factor range: **0.8–1.2 ×** the original duration
   - Implemented with `librosa.effects.time_stretch`
   - After stretching, clips are center-padded or center-trimmed to match
     the original length (to keep the same number of samples per clip)

2. **Pitch shifting**

   - Range: **±2 semitones**
   - Implemented with `librosa.effects.pitch_shift` at 16 kHz
   - Keeps shifts within realistic bounds for bovine F0 while adding natural
     variation

3. **Gaussian noise addition**

   - Additive white noise with a target signal-to-noise ratio (SNR) in the
     range **20–30 dB**
   - Noise power is scaled so that:
     - `SNR_dB = 20 * log10(RMS_signal / RMS_noise)`
   - This emulates realistic barn background variation without drowning out
     the vocalization.

4. **Gain adjustment**

   - Amplitude scaling in the range **±6 dB**
   - Implemented as a multiplicative gain:
     - `gain_lin = 10^(gain_db / 20)`
   - Models changes in caller distance, vocal effort, or microphone orientation.

For each augmented clip, the script randomly decides whether to apply each
operation (with specified probabilities) and samples a parameter value from
the corresponding range. This generates a diverse but **biologically
plausible** set of variants for minority classes.

