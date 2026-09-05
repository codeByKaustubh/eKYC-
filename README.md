# KYCShield — Multimodal Anti-eKYC Spoofing & Synthetic-Media Detection Platform

> **Academic Research & Cybersecurity Engineering Prototype**  
> A layered, multimodal verification platform designed to detect presentation attacks, synthetic media (deepfakes), voice cloning, and virtual camera injection in electronic Know Your Customer (eKYC) and Video Customer Identification Processes (V-CIP).

---

## 📑 Table of Contents
- [Project Overview](#project-overview)
- [Architecture & Threat Defense Model](#architecture--threat-defense-model)
- [Dataset Sources & Benchmarks](#dataset-sources--benchmarks)
- [Subsystems & Implementations](#subsystems--implementations)
  - [1. React Enterprise Console (KYCShield Frontend)](#1-react-enterprise-console-kycshield-frontend)
  - [2. Streamlit Detection Engine (`app.py`)](#2-streamlit-detection-engine-apppy)
- [Multimodal Risk Engine Policy](#multimodal-risk-engine-policy)
- [Getting Started](#getting-started)
  - [Running the React Frontend](#running-the-react-frontend)
  - [Running the Python Streamlit Engine](#running-the-python-streamlit-engine)
- [Research Methods & Academic References](#research-methods--academic-references)
- [Research Limitations & Disclaimer](#research-limitations--disclaimer)

---

## Project Overview

eKYC systems rely on remote video and audio streams to verify customer identities. However, generative AI advances have lowered the barrier to orchestrating sophisticated attacks—including 2D/3D presentation attacks, generative face swaps (DeepFaceLab, SimSwap), zero-shot neural voice cloning (HiFi-GAN, XTTS), lip-sync modifications (Wav2Lip), and virtual camera driver injections.

**KYCShield** addresses these vulnerabilities through **multimodal layered defense**. Rather than relying on a single fallible biometric indicator, it cross-references visual, behavioral, spectro-temporal acoustic, and audio-visual synchronization signals into a unified, explainable decision matrix.

---

## Architecture & Threat Defense Model

```
               Applicant Input (Webcam Video / Mic Audio / File Upload)
                                         │
                                         ▼
                            Media Preprocessing & Decode
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
  Face Verification             Liveness / Behavioral            Deepfake Detection
  (RetinaFace + ArcFace)        (Blinks, Head Rotation,          (Capsule-Forensics,
                                Micro-Saccade Dynamics)          Frequency Domain SRM)
        │                                │                                │
        └────────────────────────────────┼────────────────────────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
 Voice Anti-Spoofing           Lip-Sync Alignment              Camera Integrity
 (AASIST / RawNet3)            (SyncNet / BioLip)              (Sensor Noise / Virtual Driver)
        │                                │                                │
        └────────────────────────────────┴────────────────────────────────┘
                                         │
                                         ▼
                             Multimodal Risk Engine
              (Linear Weighted Policy + Multi-Signal Escalation Rules)
                                         │
                 ┌───────────────────────┼───────────────────────┐
                 ▼                       ▼                       ▼
            VERIFIED                  REVIEW                  REJECT
          (Risk < 30%)            (Risk 30%–59%)           (Risk ≥ 60%)
```

---

## Dataset Sources & Benchmarks

The project is designed to evaluate and benchmark against the following dataset sources:

### 1. AUDIO DATASET
- **Source**: [Real vs Fake Human Voice Deepfake Audio (Kaggle)](https://www.kaggle.com/datasets/unidpro/real-vs-fake-human-voice-deepfake-audio)
- **URL**: `https://www.kaggle.com/datasets/unidpro/real-vs-fake-human-voice-deepfake-audio`
- **Description**: Features paired genuine human speech recordings alongside diverse AI-synthesized, cloned, and vocoder-manipulated speech samples for evaluating acoustic voice anti-spoofing (AASIST / RawNet3).

### 2. VIDEO DATASET
- **Source**: [SDFVD - Synthetic & DeepFake Video Dataset (Hugging Face)](https://huggingface.co/datasets/Hemgg/SDFVD-video-dataset)
- **URL**: `https://huggingface.co/datasets/Hemgg/SDFVD-video-dataset`
- **Description**: High-resolution video dataset covering facial manipulation techniques (face-swap, reenactment, and generative facial synthesis) used to evaluate spatial and temporal continuity detectors.

### Benchmark Comparison Matrix

| Modality | Dataset / Source | Direct Link |
| :--- | :--- | :--- |
| **Audio Forensics** | Real vs Fake Human Voice Deepfake Audio | https://www.kaggle.com/datasets/unidpro/real-vs-fake-human-voice-deepfake-audio |
| **Video Forensics** | SDFVD Video Dataset | https://huggingface.co/datasets/Hemgg/SDFVD-video-dataset |
| **Domain Benchmark** | eKYC-DF & HAV-DF | Mobile-centric eKYC & Hindi Audio-Visual Deepfakes |
| **General Research** | FaceForensics++ (FF++) & DFDC | Standard academic benchmarks (DeepFakes, Face2Face, FaceSwap) |


---

## Subsystems & Implementations

The repository contains two complementary prototype implementations:

### 1. React Enterprise Console (KYCShield Frontend)
A high-density, SOC-styled cybersecurity dashboard located in `src/`:
- **Dashboard**: Real-time KPI cards, operational module statuses, and recent verification audit feed.
- **Live KYC**: Full webcam stream integration with an animated **Demo Camera simulator**, face framing guide rectangle, and interactive 3-stage challenge execution (head rotation, natural blinking, speech phrase reading).
- **Upload & Analyze**: Drag-and-drop forensic evaluation for images, videos, and audio tracks with an 8-stage progress pipeline and a **suspicious temporal anomaly timeline**.
- **Attack Lab**: 5 controlled demonstration scenarios (Genuine Media, Replay Attack, Synthetic Video, Lip-Sync Anomaly, Neural Voice Clone) with 1-click live scenario injection.
- **Analysis Reports**: Comprehensive forensic report page with visual **Prototype Risk Aggregation** weight breakdowns, latency metrics (ms), and JSON report export.
- **Audit Logs**: Filterable verification log (by Decision, Media Type, Risk Level, and ID search) with JSON export.
- **Research & Methodology**: Detailed threat vector breakdown, research methods, and academic limitations notice.
- **API Boundary (`src/services/api.ts`)**: Structured service layer pre-configured to connect to Python FastAPI endpoints when `VITE_API_URL` is set.

### 2. Streamlit Detection Engine (`app.py`)
A fast, lightweight Python demonstration interface:
- **Tab 1 — Video Verification Pipeline**: Processes MP4/MOV videos, visualizes real-time bounding box tracking and scanning lines with OpenCV, and computes biometric risk weights.
- **Tab 2 — Standalone Audio Anti-Spoofing (AASIST3)**: Analyzes MP3/WAV speech waveforms, extracts spectro-temporal coefficients, and evaluates voice clone probability.

---

## Multimodal Risk Engine Policy

The composite risk score is evaluated using the prototype weighted fusion policy:

$$\text{Overall Risk} = 0.25 \times \text{Face} + 0.20 \times \text{Liveness} + 0.25 \times \text{Deepfake} + 0.10 \times \text{Voice} + 0.10 \times \text{LipSync} + 0.10 \times \text{Camera}$$

### Decision Thresholds
- **`0% – 29%` → `VERIFIED`**: All biometrics and behavioral responses within authentic human tolerances.
- **`30% – 59%` → `REVIEW`**: Borderline composite risk or a single anomalous detector; escalated for secondary human analyst review.
- **`60% – 100%` → `REJECT`**: Multiple independent suspicious indicators or critical deepfake/replay flags detected.

---

## Getting Started

### Prerequisites
- **Node.js** (v18 or higher) & **npm**
- **Python** (3.10+ recommended for Streamlit and OpenCV)

---

### Running the React Frontend

```bash
# 1. Install Node dependencies
npm install

# 2. Launch the Vite development server
npm run dev

# 3. Access in your browser:
# http://localhost:5173
```

> [!TIP]
> If running on Windows PowerShell and script execution is disabled, use:
> ```powershell
> npm.cmd run dev
> ```

To create a production build:
```bash
npm run build
npm run preview
```

---

### Running the Python Streamlit Engine

```bash
# 1. Install required Python packages
pip install streamlit opencv-python numpy

# 2. Run the Streamlit application
streamlit run app.py

# 3. Access in your browser:
# http://localhost:8501
```

---

## Research Methods & Academic References

1. **Capsule-Forensics (Nguyen et al.)**: Uses capsule networks to evaluate spatial hierarchical relationships within facial regions to detect blending seams.
2. **SyncNet, BioLip & SAVe (Chung et al. / Agarwal et al.)**: Evaluates cross-correlation between spoken phoneme audio envelopes and video viseme mouth apertures.
3. **AASIST / AASIST3 (Jung et al.)**: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks applied directly to raw waveforms.
4. **ISO/IEC 30107-3**: Standard for Biometric Presentation Attack Detection (PAD).

---

## Research Limitations & Disclaimer

> [!IMPORTANT]
> **Academic Research Prototype Notice:**  
> KYCShield is an academic research prototype built to demonstrate the architecture and workflow of multimodal identity security.  
> - **Not Production Banking Software**: This prototype has not undergone regulatory financial audits.
> - **No Regulatory Certification**: Thresholds are illustrative and not formally certified under ISO/IEC 30107-3 or FIDO biometrics.
> - **No Guaranteed 100% Detection**: Detection performance varies across generative models, compression codecs, and ambient lighting conditions.
