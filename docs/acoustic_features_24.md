# Acoustic Feature Definitions (24-feature set)

This document describes the 24 acoustic features extracted by the script
`acoustic_feature_extraction.py` for the bovine vocalization dataset.

All features are computed from mono audio resampled to **16 kHz**
using **librosa**, **Praat–Parselmouth**, and related Python libraries.

For each feature we list:

- **Type & units**
- **How it is computed (implementation note)**

---

## 1. `duration_s`

- **Type & units:** Float, seconds  
- **Computation:**  
  `duration_s = len(y) / sr` where `y` is the audio signal and `sr` is the sampling rate.  
- **Meaning:**  
  Captures how long each vocalization event lasts, helping to differentiate brief grunts or short calls from long moos or extended call sequences.

---

## 2. `snr_db`

- **Type & units:** Float, decibels (dB)  
- **Computation:**  
  1. Compute frame-wise RMS energy.  
  2. Define:
     - “Signal” RMS = 95th percentile of frame-wise RMS  
     - “Noise” RMS  = 5th percentile of frame-wise RMS  
  3. Compute: `snr_db = 20 * log10(signal_rms / noise_rms)`.  
- **Meaning:**  
  Describes how clearly the cow’s vocalization stands out from barn/background noise, indicating how “clean” or “noisy” the recording is.

---

## 3–5. `f0_mean_hz`, `f0_min_hz`, `f0_max_hz`

- **Type & units:** Float, Hertz (Hz)  
- **Computation:**  
  1. Estimate fundamental frequency (F0) using `librosa.pyin` with:
     - `fmin = 50 Hz`  
     - `fmax = 1250 Hz`  
  2. Exclude NaN frames.  
  3. Compute:
     - `f0_mean_hz` = mean of valid F0 values  
     - `f0_min_hz`  = minimum of valid F0 values  
     - `f0_max_hz`  = maximum of valid F0 values.  
- **Meaning:**  
  Summarizes the overall pitch and pitch range of the call, which can help separate low-pitched, calm moos from higher-pitched or more aroused vocalizations.

---

## 6–7. `intensity_min_db`, `intensity_max_db`

- **Type & units:** Float, decibels relative to full scale (dBFS)  
- **Computation:**  
  1. Compute frame-wise RMS.  
  2. Convert RMS to decibels using `librosa.amplitude_to_db`.  
  3. Compute:
     - `intensity_min_db` = minimum RMS dB across frames  
     - `intensity_max_db` = maximum RMS dB across frames.  
- **Meaning:**  
  Characterizes the quietest and loudest parts of each clip, indicating how soft or forceful a given vocalization is within the recording.

---

## 8–9. `f1_mean_hz`, `f2_mean_hz`

- **Type & units:** Float, Hertz (Hz)  
- **Computation:**  
  1. Use **Praat–Parselmouth** to estimate formants with Praat’s Burg method.  
  2. Sample first (`F1`) and second (`F2`) formants at multiple time points across the clip.  
  3. Compute:
     - `f1_mean_hz` = mean of sampled F1 values  
     - `f2_mean_hz` = mean of sampled F2 values.  
- **Meaning:**  
  Reflect vocal tract resonances that shape the spectral envelope of the call, aiding in distinguishing more tonal, vowel-like moos from noisier, less harmonic events.

---

## 10–11. `rms_energy_mean`, `rms_energy_std`

- **Type & units:** Float, linear RMS units  
- **Computation:**  
  1. Compute frame-wise RMS using `librosa.feature.rms`.  
  2. Compute:
     - `rms_energy_mean` = mean RMS across frames  
     - `rms_energy_std`  = standard deviation of frame-wise RMS.  
- **Meaning:**  
  `rms_energy_mean` reflects overall loudness of the vocalization, while `rms_energy_std` captures how much the loudness fluctuates over time (e.g., steady vs. highly variable calls).

---

## 12. `spectral_centroid_mean`

- **Type & units:** Float, Hertz (Hz)  
- **Computation:**  
  1. Compute magnitude spectrogram via STFT.  
  2. Compute spectral centroid for each frame using `librosa.feature.spectral_centroid`.  
  3. Average over all frames to obtain `spectral_centroid_mean`.  
- **Meaning:**  
  Indicates whether the call’s energy is concentrated at lower or higher frequencies, helping to distinguish low, booming moos from brighter or noisier sounds.

---

## 13. `spectral_bandwidth_mean`

- **Type & units:** Float, Hertz (Hz)  
- **Computation:**  
  1. Use the same magnitude spectrogram as above.  
  2. Compute spectral bandwidth per frame using `librosa.feature.spectral_bandwidth`.  
  3. Average over frames to obtain `spectral_bandwidth_mean`.  
- **Meaning:**  
  Describes how narrowly or broadly the spectral energy is spread, which helps separate narrow-band harmonic calls from broadband or noisy acoustic events.

---

## 14. `spectral_rolloff_95`

- **Type & units:** Float, Hertz (Hz)  
- **Computation:**  
  1. Compute spectral rolloff per frame using  
     `librosa.feature.spectral_rolloff(roll_percent=0.95)`.  
  2. Average over frames to obtain `spectral_rolloff_95`.  
- **Meaning:**  
  Approximates the upper edge of the main energy band of the call, indicating how far vocal energy extends into higher frequencies.

---

## 15–16. `zcr_mean`, `zcr_std`

- **Type & units:** Float, unitless (proportion per frame)  
- **Computation:**  
  1. Compute zero-crossing rate (ZCR) per frame using  
     `librosa.feature.zero_crossing_rate`.  
  2. Compute:
     - `zcr_mean` = mean ZCR across frames  
     - `zcr_std`  = standard deviation of ZCR across frames.  
- **Meaning:**  
  Higher ZCR values usually correspond to noisier, less periodic sounds (e.g., coughs, slurps), while lower values indicate smoother, more periodic moos.

---

## 17. `time_to_peak_s`

- **Type & units:** Float, seconds  
- **Computation:**  
  1. Compute frame-wise RMS energy.  
  2. Find index of frame with maximum RMS (`peak_frame`).  
  3. Convert to time using:  
     `time_to_peak_s = peak_frame * hop_length / sr`.  
- **Meaning:**  
  Describes how quickly the call reaches its maximum loudness, helping to distinguish sudden, explosive sounds from calls that build up more gradually.

---

## 18–23. `mfcc1_mean`, `mfcc1_std`, `mfcc2_mean`, `mfcc2_std`, `mfcc3_mean`, `mfcc3_std`

- **Type & units:** Float, arbitrary cepstral units  
- **Computation:**  
  1. Compute Mel-frequency cepstral coefficients (MFCCs) using  
     `librosa.feature.mfcc(n_mfcc=3)`.  
  2. Let `mfcc[0,:]`, `mfcc[1,:]`, `mfcc[2,:]` be the time series for the
     first three coefficients.  
  3. For each coefficient `k ∈ {1, 2, 3}` compute:
     - `mfcck_mean` = mean of `mfcc[k-1,:]` over frames  
     - `mfcck_std`  = standard deviation of `mfcc[k-1,:]` over frames.  
- **Meaning:**  
  Compactly describe the spectral “shape” or timbre of the call, enabling discrimination between different call types and acoustic contexts beyond simple pitch and loudness.

---

## 24. `voiced_ratio`

- **Type & units:** Float, proportion between 0 and 1  
- **Computation:**  
  1. Estimate F0 track using `librosa.pyin`.  
  2. Mark all frames with a valid (non-NaN) F0 as “voiced”.  
  3. Compute:  
     `voiced_ratio = (number of voiced frames) / (total number of frames)`.  
- **Meaning:**  
  Indicates what fraction of the clip is dominated by periodic, voiced energy, helping to distinguish fully voiced moos from clips that contain substantial unvoiced or noisy segments.

---

## Implementation Notes

- All features are computed at a sampling rate of **16 kHz**.
- STFT-based features use:
  - `n_fft = 1024`  
  - `hop_length = 512`
- See `acoustic_feature_extraction.py` in this repository for exact
  implementation details.
