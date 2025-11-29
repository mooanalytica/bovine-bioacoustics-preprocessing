# Segmentation and Annotation Protocol (Raven Lite + Praat)

This document describes the segmentation and annotation procedure used to
curate individual cow vocalization clips from the continuous barn recordings.

The workflow has three main stages:

1. **Segmentation in Raven Lite (v2.0)**
2. **Boundary verification in Praat**
3. **First-level annotation (behaviour + emotional context)**

---

## 1. Segmentation in Raven Lite

All continuous recordings were first denoised (band-pass filtering + iZotope RX)
and then segmented into individual vocalizations using **Raven Lite 2.0**.

### 1.1 Spectrogram configuration

In Raven Lite, each audio file was opened and displayed as a waveform +
spectrogram:

- Spectrogram type: short-time Fourier transform (STFT)
- Window size: **1,024 samples**
- Overlap: **50%**
- Window function: **Hamming**

This configuration was chosen to balance time and frequency resolution for
cattle vocalizations.

### 1.2 Identifying candidate events

Candidate vocal events were identified by **visually scanning** the spectrogram
and confirming with audio playback:

- Look for:
  - **Harmonic stacks** (for moos and tonal calls)
  - **Broadband bursts** (for coughs, sneezes, slurps, etc.)
- Each potential event was:
  - Highlighted in the spectrogram using a **selection box**
  - Audibly verified via playback to confirm it was a genuine vocal or
    non-vocal animal sound rather than pure barn noise or mechanical artefact

### 1.3 Marking onset and offset

For each selected event, start and end times were defined using consistent
criteria:

- **Onset**:
  - Point where amplitude rises clearly above the noise floor
  - First harmonic band becomes visible in the spectrogram
- **Offset**:
  - Point where energy returns to baseline and harmonic structure disappears

To preserve context, each clip included **2–3 seconds of padding** before onset
and after offset (e.g., inhalation or resonance tails).

### 1.4 Exporting segmented clips

Each selection was exported from Raven Lite as an individual WAV file:

- Export format: **WAV**
- Each clip was saved using a **structured file naming convention**
  encoding:
  - Farm ID
  - Barn zone (feeding, drinking, resting, milking, etc.)
  - Date and time
  - Microphone ID
  - A provisional class placeholder (to be refined during annotation)

Using this procedure, **569 clips** were extracted from approximately **90 hours**
of raw barn recordings.

### 1.5 Inter-annotator agreement and conservative segmentation

To reduce subjectivity in segmentation and labelling, two annotators
independently checked a subset of clips for:

- Correct placement of onset/offset boundaries  
- Inclusion of the full vocal event  
- Consistency of the assigned call category and context labels  

Inter-annotator agreement on these labels was quantified using **Cohen’s
kappa**, providing an objective measure of consistency between annotators.
For clips where the annotators initially disagreed, the team revisited the
corresponding **audio together with the time-matched video recordings**
(GoPro footage) to inspect the behavioural context at the exact timestamps.
Final labels and boundaries were then resolved by consensus, following a
conservative policy of retaining extra context rather than risking
truncation of the vocalization.

---

## 2. Boundary Verification in Praat

After segmentation in Raven Lite, each selection was visually and acoustically
checked in **Praat** to confirm that it represented a valid vocal event and that
boundaries were accurately placed.

### 2.1 Visual inspection

In Praat, the following displays were inspected for each clip:

- Waveform
- Spectrogram
- **Pitch contour (F0)**
- **Formant tracks (F1, F2)**
- **Intensity envelope**

Checks performed:

- Confirm that:
  - Harmonic structure is **continuous** within the marked segment
  - Formant tracks and F0 contour are consistent with known patterns of
    bovine vocal production
- Verify that:
  - The selection is not dominated by mechanical noise or transient artefacts
  - Onset and offset in Praat align with the Raven Lite boundaries

### 2.2 Boundary refinement

Where necessary, **onset and offset** were fine-tuned in Praat:

- Refine boundaries using:
  - High-precision time axis
  - Alignment of waveform, spectrogram, F0, F1, F2 and intensity
- Ensure that:
  - The full vocal event is included
  - Non-vocal noise before/after the call is kept minimal but contextual

The final, Praat-verified clips were then used for feature extraction and
behavioural annotation.

---

## 3. First-Level Annotation

Each segmented and Praat-verified clip underwent **first-level annotation** by
two researchers. For every clip, multiple labels were recorded to capture call
type, affective state, and behavioural context.

### 3.1 Labels recorded for each clip

For each clip, annotators recorded:

1. **Main category and subcategory**
   - According to the ethology-driven scheme (9 main categories, 48
     sub-types; e.g., *Estrus Call*, *Feed Anticipation Call*, *Mother
     Separation Call*, *Breathing Respiratory Sounds*).
2. **Emotional context**
   - Affective label reflecting the inferred state at the time of vocalization,
     e.g.:
     - Distress
     - Pain
     - Anticipation
     - Hunger
3. **Confidence score**
   - Integer scale from **1 (low)** to **10 (high)** indicating annotator
     certainty in the assigned labels.
4. **Free-text description**
   - Short behavioural description and situational context, for example:
     - “Cow waiting at feed gate”
     - "Cow standing near water trough”
     - “Cow walking past milking parlour”

### 3.2 Annotation guidelines

Annotations were grounded in ethological principles:

- Link each vocalization to **proximate stimuli**:
  - Feeding, separation, handling, barn disturbance, etc.
- Consider possible **ultimate functions**:
  - Contact call, distress signal, solicitation, social coordination, etc.
- Decisions were based on:
  - **Spectro-temporal shape** (harmonic stacks, modulation, call duration)
  - **Amplitude envelope** (sudden vs gradual onset, overall intensity)
  - **Context** from observation notes and video

### 3.3 Handling ambiguity

For clips where behaviour or call type could not be confidently determined:

- The label **“Unknown”** was assigned rather than forcing a classification.
- Ambiguity typically arose for:
  - Overlapping calls from multiple animals
  - Heavily masked events in very noisy segments

### 3.4 Use of video and field notes

To improve label reliability, annotators:

- Consulted:
  - **Co-recorded video** from GoPro cameras
  - **Manual observation logs**
- Verified whether:
  - The cow was feeding vs frustrated at an empty bunk
  - The animal was being handled, separated, or simply walking past a zone

These additional sources provided behavioural context that could not be inferred
from audio alone.

---

## 4. Output of the Segmentation & Annotation Pipeline

The combined Raven Lite + Praat + first-level annotation workflow produced:

- **569 denoised, segmented clips**
- Each clip with:
  - Precisely verified onset/offset boundaries
  - Main and subcategory labels
  - Emotional context
  - Confidence score
  - Behavioural description
- These clips form the input to:
  - The 24-feature acoustic extraction pipeline
  - Subsequent statistical and machine-learning analyses

This protocol ensures that each audio segment in the dataset is:
1. Carefully segmented,
2. Acoustically verified, and
3. Ethologically annotated with transparent, reproducible criteria.
