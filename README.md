# KYCShield — Multimodal Anti-eKYC Spoofing & Synthetic-Media Detection

A multimodal identity verification prototype designed to evaluate presentation attacks, synthetic media (deepfakes), voice cloning, and audio-visual synchronization anomalies for eKYC and V-CIP workflows.

---

## 🔗 Dataset Sources

### 1. AUDIO DATASET
- **Source**: [Real vs Fake Human Voice Deepfake Audio (Kaggle)](https://www.kaggle.com/datasets/unidpro/real-vs-fake-human-voice-deepfake-audio)
- **URL**: `https://www.kaggle.com/datasets/unidpro/real-vs-fake-human-voice-deepfake-audio`
- **Application**: Evaluation of speech synthesis, voice cloning, and neural vocoder artifacts.

### 2. VIDEO DATASET
- **Source**: [SDFVD - Synthetic & DeepFake Video Dataset (Hugging Face)](https://huggingface.co/datasets/Hemgg/SDFVD-video-dataset)
- **URL**: `https://huggingface.co/datasets/Hemgg/SDFVD-video-dataset`
- **Application**: Evaluation of facial replacement, boundary blending, and temporal frame continuity.

---

## 🛠️ Implemented Project Features

The repository contains two working implementations:

### 1. React Web Application (`src/`)

- **Dashboard**:
  - 4 security KPIs: Active Sessions, Today's Verifications, Flagged Sessions, and Average Risk.
  - Recent Verification Sessions table with session IDs, timestamps, verification types, risk scores, status badges, and direct report view links.
  - Operational health grid for all 6 detection modules.

- **Live KYC**:
  - Live webcam preview via WebRTC with an interactive toggle to switch to an animated **Demo Camera simulator**.
  - Face framing guide rectangle with landmark crosshairs and stream telemetry (FPS, resolution, stream integrity).
  - 3-stage interactive challenge pipeline:
    1. Head rotation challenge (`Yaw angle < -18°`)
    2. Spontaneous blink challenge (`EAR < 0.2`)
    3. Verbal confirmation challenge (`"KYC Verification 7-0-4-2"`)
  - Live AI Analysis Panel displaying individual scores, risk percentages, model names, latency, and overall risk verdict.

- **Upload & Analyze**:
  - Drag-and-drop file uploader supporting Images (`JPG`, `PNG`), Videos (`MP4`, `MOV`), and Audio (`WAV`, `MP3`).
  - 1-click test sample loaders (`verification_sample.mp4`, `live_capture_0012.mp4`, `id_selfie_scan.png`, `verbal_phrase_tts.wav`).
  - Media preview player with file metadata (duration, format, size, status).
  - 8-stage simulated forensic pipeline with progress tracking.
  - Media Analysis Report displaying overall risk, reasons flagged, video/image/audio metrics, and an interactive **Suspicious Anomaly Timeline** highlighting flagged intervals.

- **Attack Lab**:
  - 5 research attack scenarios:
    1. Genuine Media Baseline
    2. Synthetic Video (Deepfake)
    3. Replay Presentation Attack
    4. Lip-Sync Manipulation
    5. Synthetic Voice (Voice Clone)
  - Detailed expected signal breakdowns and scenario simulation modal with an **Inject Scenario into Live KYC** trigger.

- **Analysis Reports**:
  - In-depth report viewer for completed sessions (`KYC-2026-0012`, `KYC-2026-0011`, `KYC-2026-0010`).
  - Visual Prototype Risk Aggregation diagram showing the linear weighted formula:
    $$\text{Overall Risk} = 0.25 \times \text{Face} + 0.20 \times \text{Liveness} + 0.25 \times \text{Deepfake} + 0.10 \times \text{Voice} + 0.10 \times \text{LipSync} + 0.10 \times \text{Camera}$$
  - 6 individual modality cards with diagnostic explanations, processing times, and technical evidence.
  - Chronological verification event log and JSON report download.

- **Audit Logs**:
  - Filterable audit table displaying Session ID, Timestamp, Type, Risk, Decision, Model Version, and Latency.
  - Multi-criteria filters: Decision (`VERIFIED`, `REVIEW`, `REJECT`), Verification Type, Risk Range, and search query.
  - JSON export of filtered audit logs.

- **Research & Methodology**:
  - Threat vector overview, multimodal pipeline data flow diagram, literature citations (Capsule-Forensics, SyncNet, BioLip, AASIST), and research limitations notice.

- **Global Demo Mode Bar**:
  - Fast scenario switcher accessible across views to trigger genuine or attack simulations.

- **API Service Layer (`src/services/api.ts`)**:
  - TypeScript service layer with mock responses and support for `VITE_API_URL` to route requests to a Python/FastAPI backend.

---

### 2. Python Streamlit Application (`app.py`)

- **Tab 1: Video Verification Pipeline**:
  - Upload verification video (`MP4` / `MOV`).
  - Real-time OpenCV video stream with dynamic facial tracking box, animated scan line, and fraud detection status tag.
  - Video threat index score calculation and localized diagnostic metrics (Facial Frame Artifact Weight, BioLip Sync Discrepancy Matrix).
  - Automated `VERIFIED` / `REJECTED` decision verdict.

- **Tab 2: Standalone Audio Anti-Spoofing (AASIST3)**:
  - Upload audio speech sample (`MP3` / `WAV`).
  - Audio playback interface.
  - Spectral parsing progress indicator.
  - Voice clone probability score, spectro-temporal diagnostics, and V-CIP action determination.

---

## 🚀 How to Run

### 1. React Web Application

```bash
# Install dependencies
npm install

# Start development server
npm run dev
# (or 'npm.cmd run dev' on Windows PowerShell)

# Open browser at:
# http://localhost:5173
```

To build for production:
```bash
npm run build
npm run preview
```

---

### 2. Python Streamlit Application

```bash
# Install Python dependencies
pip install streamlit opencv-python numpy

# Run the Streamlit app
streamlit run app.py

# Open browser at:
# http://localhost:8501
```

---

## ⚠️ Prototype Disclaimer

This software is an academic research prototype built to demonstrate multimodal anti-spoofing and synthetic-media detection workflows. Detection scores are experimental and do not constitute formal banking or regulatory certification.
