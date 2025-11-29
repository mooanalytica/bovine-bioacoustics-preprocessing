# Bovine Bioacoustics Preprocessing Pipeline

This repository contains the complete open-source preprocessing and 
feature extraction pipeline used in the manuscript:

**"Big Data Approaches to Bovine Bioacoustics: A FAIR-Compliant Dataset and 
Scalable ML Framework for Precision Livestock Welfare" (Frontiers in Big Data).**

## Contents

- `preprocess_pipeline.py`  
  Replicates the denoising workflow without proprietary software (iZotope RX11),
  using `noisereduce`, `scipy.signal`, and `librosa`.

- `extract_features.py`  
  Computes the full 24-feature acoustic vector for each vocalization clip.

- `requirements.txt`  
  Python package dependencies.

- Example folder structure for running batch processing.

## Preprocessing Pipeline Overview

1. Load WAV file  
2. Apply Butterworth band-pass filter (50–1800 Hz)  
3. Apply non-stationary spectral gating (noisereduce)  
4. Normalize amplitude  
5. Export processed WAV  

## Feature Extraction

Features include:
- F0, formants (F1–F3)
- HNR, intensity
- Spectral centroid, bandwidth, rolloff
- MFCCs (n=13)
- RMS energy, zero-crossing rate

All extracted using Parselmouth + librosa.

## Data Availability

Raw and processed audio clips, metadata, and feature tables are deposited in 
Zenodo under **restricted access** to comply with farm data-sharing agreements.
The repository DOI is listed in the manuscript.

## License

MIT License (open for academic and industry use).

