# Bovine Bioacoustics Preprocessing & Feature Engineering Pipeline

This repository contains the complete open-source preprocessing, denoising, 
segmentation, feature extraction, augmentation, balancing, and statistical 
analysis pipeline used in the manuscript:

**"Big Data Approaches to Bovine Bioacoustics: A FAIR-Compliant Dataset and 
Scalable ML Framework for Precision Livestock Welfare" (Frontiers in Big Data).**

It includes all steps required to reproduce the acoustic feature dataset, 
statistical analysis, and intermediate processing described in the manuscript.

---

## Repository Files and Their Purpose

### Python Code Files

**acoustic_feature_extraction.py**  
Extracts the full 24-dimensional acoustic feature set (F0, F1–F3, intensity, HNR,
spectral features, MFCCs, RMS energy, ZCR). Uses Parselmouth + Librosa.

**audio_band_pass_filter.py**  
Implements Butterworth band-pass filtering (configurable frequency range).  
Used for preprocessing and noise removal before feature extraction.

**augmentation_and_class_balancing.py**  
Performs data augmentation (time-stretching, pitch shifting, noise addition)  
and creates class-balanced datasets based on target sample counts.

**denoising_pipeline.py**  
Implements the open-source version of the iZotope RX11-style denoising workflow  
using:  
- Spectral gating (noisereduce)  
- Band-pass filtering  
- Amplitude normalization  
- Optional harmonic filtering  
Outputs denoised WAV files for downstream analysis.

**dunn's_test.py**  
Runs Dunn’s post-hoc statistical test to evaluate feature-level differences 
across cow vocalization classes. Produces tables and figures for the manuscript.

**llm_variance_components.py**  
Fits Linear Mixed Models (LMM) to estimate variance components contributed by:  
Farm, Barn Zone, Microphone, Mic Placement Context, Call Category, etc.  
Used for reviewer-requested variability analysis.

**pca_analysis.py**  
Performs Principal Component Analysis (PCA) on the 24 acoustic features, 
generates scree plots, loadings, and class separation visualizations.

---

### Documentation Files (`docs/`)

**acoustic_features_24.md**  
Detailed definitions, units, extraction settings, and formulas for all  
24 acoustic features used in the study.

**cow_call_categories.md**  
Taxonomy of cow vocalization categories and subcategories, including behavioral 
and emotional interpretations.

**data_augmentation_and_balancing.md**  
Complete description of augmentation methods and balancing logic used to 
address dataset imbalance.

**denoising_pipeline_rx11.md**  
Exact steps performed in iZotope RX11 during original denoising, with 
corresponding Python replacements for full reproducibility.

**segmentation_annotation_process.md**  
Workflow describing call segmentation, manual annotation, quality control, 
and metadata structure.

---

## Preprocessing Pipeline Overview

This pipeline replicates the acoustic cleaning steps described in the manuscript:

1. Load WAV file  
2. Apply Butterworth band-pass filter  
3. Apply non-stationary spectral gating  
4. Normalize amplitude  
5. Export processed WAV  
6. (Optional) Perform augmentation & class balancing  
7. Extract 24 acoustic features  

The Python implementation matches the published RX11 steps as closely 
as possible using open-source signal-processing libraries.

---

## Feature Extraction Overview

The feature extraction script computes:

- **F0** (fundamental frequency)  
- **Formants** (F1–F3)  
- **HNR**, harmonics-based metrics  
- **Intensity**, RMS energy  
- **Spectral centroid**, bandwidth, rolloff  
- **MFCCs** (13 coefficients)  
- **Zero-crossing rate (ZCR)**  

The output is a CSV file suitable for machine learning or statistical analysis.

---

## Statistical Analysis Tools

The repository includes the scripts used to generate analysis figures:

- PCA dimensionality reduction  
- Dunn’s post-hoc test  
- Linear mixed-effects models for variance partitioning  

These were used to address reviewer feedback and strengthen the manuscript.

---

## Data Availability

Raw audio, processed audio, metadata tables, spectrograms, and extracted 
features are hosted on **Zenodo** under **restricted access** to comply with 
farm-level agreements.

The DOI is provided in the manuscript.

---

## Installation

```bash
pip install -r requirements.txt
```

## Citation
If you use this repository, please cite:

Kate, M., & Neethirajan, S. (2025).
Big Data Approaches to Bovine Bioacoustics: A FAIR-Compliant Dataset and
Scalable ML Framework for Precision Livestock Welfare.
Frontiers in Big Data.

Dataset DOI (Zenodo): [10.5281/zenodo.17764250]

## Funding 

The researchers thank the Natural Sciences and Engineering Research Council of Canada (NSERC), Nova Scotia Department of Agriculture, Mitacs Canada, and the New Brunswick Department of Agriculture, Aquaculture and Fisheries for funding this study.

