# Barn Audio Denoising Pipeline  
**iZotope RX 11 workflow + Python-based alternative**

This document describes the **noise reduction pipeline** used for the barn audio in this project, and outlines how each step can be approximated with **open-source Python tools**.

Steps explained in the documentation:

1. Noise profiling of barn environments  
2. Band-pass filtering (50–1800 Hz)  
3. Multi-stage denoising in iZotope RX 11  
4. Python-based alternative for iZotope RX denoising

---

## 1. Recording Context

- Multimicrophone recordings from commercial dairy barns (feeding, drinking, milking, resting zones).  
- Audio format: WAV, 44.1/48 kHz, 24-bit.  
- Target signal: cattle vocalizations (F0 roughly 80–300 Hz, harmonics within ~1 kHz).  
- Noise sources: machinery, ventilation, metallic impacts (gates, troughs), footsteps, human speech, electrical hiss.

The aim of this pipeline is to **improve SNR** while preserving the **time–frequency structure** of cow vocalizations.

---

## 2. Noise Profiling

**Goal:** Characterize barn-specific noise to justify filter settings and denoising parameters.

- We extract short **noise-only segments** (no visible vocal harmonics) from each barn zone.  
- We compute **power spectral densities** to identify dominant noise bands (e.g., low-frequency machinery hum, high-frequency hiss).  
- This informs the decision to retain primarily the 50–1800 Hz band where most vocal energy lies.

**Python approximation:**

- Use `scipy.signal.welch` to compute power spectral density.  
- Use `matplotlib` to visualize and inspect dominant noise bands.

---

## 3. Band-Pass Filtering (50–1800 Hz) — Python Step Before RX

**Manuscript step:** Before any RX 11 processing, each file is **band-pass filtered** using a 4th-order Butterworth filter with cut-offs at **50 Hz** and **1800 Hz**.

**Purpose:**

- Preserve the main vocal band (F0 + harmonics).  
- Suppress:
  - Very low-frequency machinery noise (< 50 Hz).  
  - High-frequency hiss and electrical noise (> 1800 Hz).

**Implementation details:**

- This band-pass step is implemented in Python.  
- The **band-passed WAV** (50–1800 Hz) is the input to all subsequent RX 11 steps.

**Python approximation:**

- Use `scipy.signal.butter` to design a 4th-order band-pass filter.  
- Use `scipy.signal.filtfilt` for zero-phase filtering (to avoid phase distortion).

---

## 4. Multi-Stage Noise Reduction in iZotope RX 11

After band-pass filtering, denoising is performed in **iZotope RX 11** using the following five stages:

1. Gain normalization and DC offset removal  
2. Spectral De-noise  
3. Spectral Repair  
4. De-clip and De-crackle (used when necessary)  
5. EQ Match

Each stage below describes:

- The **RX 11 action**, and  
- A **Python-based alternative concept**.

---

### 4.1 Stage 1 — Gain Normalization & DC Offset Removal

**In RX 11:**

- Open the band-passed 50–1800 Hz file.  
- Use **Waveform Statistics** to inspect DC offset and levels.  
- If DC offset exists, apply **Remove DC Offset**.  
- Use **Gain** and/or **Loudness Control** to:
  - Bring peaks into a comfortable range (e.g., around -6 dBFS).  
  - Standardize levels across files without clipping.

**Python alternative:**

- Subtract the **mean** of the waveform to remove DC offset.  
- Apply **peak normalization** or **loudness normalization**:
  - Peak normalization: scale the signal so the maximum absolute value is below a target (e.g., 0.9).  
  - Loudness normalization: use a loudness meter (e.g., `pyloudnorm`) to normalize all files to a target LUFS value.

---

### 4.2 Stage 2 — Spectral De-noise (Broadband Noise Reduction)

**In RX 11:**

**Goal:** Reduce broadband background noise (ventilation, preamp hiss) while preserving vocal structure.

Typical procedure:

1. In the spectrogram, select a **noise-only region** (no vocal harmonics).  
2. Open **Spectral De-noise** and click **Learn** to capture a noise profile.  
3. Use moderate settings:
   - Quality set to **High** or **Best**.  
   - Reduction amount around **6–12 dB**, adjusted per file.  
   - Time and frequency smoothing tuned to avoid musical or “watery” artifacts.  
   - Adaptive mode enabled when background noise changes over time.  
4. Apply to the entire file or long segments.

**Python alternative:**

- Use a **spectral gating** approach for non-stationary noise, e.g.:
  - `noisereduce` (spectral gating library for Python), with:
    - A **noise-only segment** as reference where available, or  
    - Automatic noise estimation for adaptive denoising.  
- Configure reduction strength and smoothing to match the **moderate** reduction used in RX (prioritizing preservation of harmonics and onsets).

---

### 4.3 Stage 3 — Spectral Repair (Transient & Artifact Mitigation)

**In RX 11:**

**Goal:** Reduce localized transient artifacts (metallic hits, clicks, clanks) in the vocal band.

Procedure:

1. Zoom in on problematic regions in the spectrogram where:
   - The noise appears as sharp, vertical stripes or distinct blobs, and  
   - Overlaps the vocal frequency range.  
2. Use time–frequency selection tools (lasso or rectangular) to isolate the artifact.  
3. Open **Spectral Repair** and select:
   - **Attenuate** for moderate reduction, or  
   - **Replace** for stronger interpolation when the artifact is dominant.  
4. Apply with conservative settings (small attenuation, multiple passes).  
5. Audition after each edit to ensure:
   - Vocal harmonics remain continuous and natural.  
   - No obvious “holes” are created in the spectrum.

**Python alternative:**

- Compute a **time–frequency representation** (e.g., STFT using `librosa`).  
- Detect transient/artifact frames based on:
  - Sudden jumps in short-time energy or spectral flux, and/or  
  - Characteristic spectral shapes (very broadband, short-duration events).  
- Apply local operations such as:
  - **Attenuation** of magnitudes in identified time–frequency bins.  
  - **Interpolation** or **median filtering** across neighboring frames to replace short artifacts.

This approximates RX’s Spectral Repair by performing **local, context-aware smoothing** in the spectrogram domain.

---

### 4.4 Stage 4 — De-clip & De-crackle (As Needed)

**In RX 11:**

These modules are used selectively, only when the waveform exhibits obvious recording faults.

- **De-clip**  
  - Identify segments where the waveform has flat-topped peaks (hard clipping).  
  - Apply **De-clip** on those segments to reconstruct the clipped waveform shape.

- **De-crackle**  
  - Used when low-level crackling, intermittent spikes, or fine impulsive noise are present.  
  - Apply **De-crackle** with conservative settings to avoid over-smoothing.

**Python alternative:**

- **De-clip approximation:**
  - Detect samples near the maximum amplitude threshold.  
  - Replace these samples (and a small neighborhood) by interpolation between unclipped neighbors.

- **De-crackle approximation:**
  - Use **median filtering** on the waveform or on high-frequency bands to suppress fine impulsive components.  
  - Alternatively, use a **wavelet-based denoising** framework (e.g., `pywavelets`) focusing on high-frequency detail coefficients.

In many clips this stage will not be necessary and can be skipped.

---

### 4.5 Stage 5 — EQ Match (Microphone Coloration Compensation)

**In RX 11:**

**Goal:** Reduce differences in tonal color caused by different microphones or placements and obtain a more consistent spectral balance across recordings.

Procedure:

1. Choose a **reference clip** (50–1800 Hz band) that sounds neutral and representative.  
2. Open **EQ Match** and capture the spectral profile of this reference.  
3. Apply EQ Match to other clips, using moderate settings to:
   - Match their **average spectral shape** to the reference.  
   - Avoid strong resonances or unnatural tonal shifts.

This helps ensure that differences in microphone coloration do not dominate downstream analysis.

**Python alternative:**

- Compute average magnitude spectra (across time) for:
  - The **reference clip** (band-passed).  
  - The **current clip** to be corrected.
- Compute a **frequency-wise gain curve** (ratio) between reference and current spectra.  
- Implement this as a frequency-domain **shaping filter**:
  - Apply the gain curve to the STFT magnitude, or  
  - Approximate it with a parametric EQ designed using `scipy.signal` or similar tools.

---

## 5. Final Output & Variants

After Stage 5 (EQ Match), the processed file is exported from RX 11 as a **denoised 50–1800 Hz WAV**. These files are used for:

- Manual and semi-automated segmentation.  
- Acoustic feature extraction (e.g., Praat/Parselmouth, `librosa`, openSMILE).  
- Model training and evaluation.

---

## 6. Python Pipeline Summary (No Code)

A Python script to implement an **end-to-end pipeline** that follows the same logic:

1. Load audio, convert to mono if needed.  
2. Remove DC offset and normalize gain.  
3. Apply 50–1800 Hz band-pass filter.  
4. Apply spectral gating noise reduction.  
5. Apply localized spectral repair approximations for transients.  
6. Optionally apply de-clip / de-crackle approximations.  
7. Apply spectral envelope / EQ matching to a neutral reference.  
8. Optionally apply loudness normalization for consistent listening / analysis levels.  
9. Export to WAV for downstream tasks.

This Python implementation does **not** aim to reproduce RX 11 sample-by-sample, but to **closely follow the same processing philosophy and sequence**, enabling fully open-source reproduction of the denoising strategy.
