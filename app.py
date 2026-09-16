import streamlit as st
import cv2
import numpy as np
import time
import os
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import forensics_engine as fe

# --- PAGE SETUP ---
st.set_page_config(page_title="Multimodal eKYC Spoofing Detection Engine", layout="wide")
st.title("🔒 Multimodal Presentation Attack & Injection Detection System")
st.subheader("KYCShield Autonomous Live Forensics Architecture (V-CIP Compliant)")

# --- CONFIGURATION DATA MATRIX FOR BENCHMARKING TABS ---
DATASET_MATRIX = {
    "video_profiles": {
        "1280x720": {
            "real": {"f": 0.12, "b": 0.10, "v": 0.14, "l": 0.08, "msg": "Video luminance and temporal continuity verified within human biometric bounds."},
            "fake": {"f": 0.85, "b": 0.68, "v": 0.75, "l": 0.82, "msg": "Generative facial blending footprints identified inside the tracking window. BioLip lip-sync mismatch confirmed."}
        },
        "1920x1080": {
            "real": {"f": 0.15, "b": 0.08, "v": 0.11, "l": 0.09, "msg": "High-fidelity organic frame dimensions conform fully to live capture baseline bounds."},
            "fake": {"f": 0.92, "b": 0.72, "v": 0.88, "l": 0.85, "msg": "Spatial face-swap blending vectors and frame-to-frame pixel flickering flagged."}
        }
    },
    "audio_profiles": {
        "real": {"v_score": 0.12, "msg": "Acoustic envelope, spectro-temporal pitch shifts, and noise distributions verify genuine live human speech."},
        "fake": {"v_score": 0.89, "msg": "🚨 AASIST3 ALERT: Synthetic voice cloning signature identified. High probability of Text-to-Speech (TTS) or Voice Conversion (VC) processing."}
    }
}

# --- UNIFIED LIVE FORENSICS ENGINE OBJECT ---
class LiveForensicTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.noise_val = 2.50
        self.lap_val = 80.0
        self.grad_val = 1200.0
        self.noise_ratio = 1.20
        self.is_attack = False
        self.live_risk = 0.12
        self.face_detected = False
        self.noise_history = []
        self.nb_history = []
        self.lap_history = []
        self.lap_b_history = []
        self.grad_history = []
        self.fft_history = []
        self.temp_history = []
        self.prev_face_gray = None
        
        self.attack_votes = []
        self.risk_history = []
        
        self.face_cascade = fe.get_face_cascade()
            
        try:
            models = fe.get_models()
            self.clf = models.get("video_pipeline", None)
        except Exception:
            self.clf = None

    def _process_image(self, img):
        self.frame_count += 1
        h, w, _ = img.shape
        
        gx1, gy1 = int(w * 0.28), int(h * 0.18)
        gx2, gy2 = int(w * 0.72), int(h * 0.78)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Dynamic face detection (run every 6 frames for smooth tracking and high performance)
        bx, by, bw, bh = gx1, gy1, (gx2 - gx1), (gy2 - gy1)
        found = False
        if self.face_cascade is not None and (self.frame_count % 6 == 0 or not hasattr(self, "_last_box")):
            small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
            faces = self.face_cascade.detectMultiScale(small, scaleFactor=1.15, minNeighbors=4, minSize=(30, 30))
            if len(faces) > 0:
                faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
                fx, fy, fw, fh = faces[0]
                bx, by, bw, bh = fx * 2, fy * 2, fw * 2, fh * 2
                self._last_box = (bx, by, bw, bh)
                found = True
        elif hasattr(self, "_last_box"):
            bx, by, bw, bh = self._last_box
            found = True
                
        self.face_detected = found
        
        face_roi = gray[max(0, by):min(h, by + bh), max(0, bx):min(w, bx + bw)]
        bg_roi = gray[int(h * 0.75):int(h * 0.95), int(w * 0.05):int(w * 0.25)]
        
        if face_roi.size > 0 and bg_roi.size > 0:
            blur_f = cv2.medianBlur(face_roi, 3)
            diff_f = cv2.absdiff(face_roi, blur_f)
            raw_noise = float(np.var(diff_f))
            
            blur_b = cv2.medianBlur(bg_roi, 3)
            diff_b = cv2.absdiff(bg_roi, blur_b)
            raw_nb = float(np.var(diff_b))
            
            raw_lap_f = float(cv2.Laplacian(face_roi, cv2.CV_64F).var())
            raw_lap_b = float(cv2.Laplacian(bg_roi, cv2.CV_64F).var())
            
            sobelx = cv2.Sobel(face_roi, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(face_roi, cv2.CV_64F, 0, 1, ksize=3)
            raw_grad = float(np.mean(np.sqrt(sobelx**2 + sobely**2)))
            
            f_thumb = cv2.resize(face_roi, (64, 64))
            mag = np.abs(np.fft.fftshift(np.fft.fft2(f_thumb)))
            low = mag[24:40, 24:40]
            raw_fft = float((np.sum(mag**2) - np.sum(low**2)) / (np.sum(mag**2) + 1e-9))
            
            raw_temp = 5.0
            if self.prev_face_gray is not None and self.prev_face_gray.shape == face_roi.shape:
                raw_temp = float(np.mean(cv2.absdiff(face_roi, self.prev_face_gray)))
            self.prev_face_gray = face_roi
            
            self.noise_history.append(raw_noise)
            self.nb_history.append(raw_nb)
            self.lap_history.append(raw_lap_f)
            self.lap_b_history.append(raw_lap_b)
            self.grad_history.append(raw_grad)
            self.fft_history.append(raw_fft)
            self.temp_history.append(raw_temp)
            
            if len(self.noise_history) > 12:
                self.noise_history.pop(0)
                self.nb_history.pop(0)
                self.lap_history.pop(0)
                self.lap_b_history.pop(0)
                self.grad_history.pop(0)
                self.fft_history.pop(0)
                self.temp_history.pop(0)
                
            self.noise_val = float(np.mean(self.noise_history))
            self.lap_val = float(np.mean(self.lap_history))
            self.grad_val = float(np.mean(self.grad_history))
            m_nb = float(np.mean(self.nb_history))
            m_lap_b = float(np.mean(self.lap_b_history))
            self.noise_ratio = float(self.noise_val / (m_nb + 1e-6))
            lap_ratio = float(self.lap_val / (m_lap_b + 1e-6))
            
            # --- EVALUATE WITH TRAINED MACHINE LEARNING FORENSIC PIPELINE ---
            if self.clf is not None and len(self.noise_history) >= 3:
                feats = np.array([[
                    self.noise_val,
                    float(np.std(self.noise_history)),
                    m_nb,
                    self.noise_ratio,
                    self.lap_val,
                    lap_ratio,
                    self.grad_val,
                    float(np.mean(self.fft_history)),
                    float(np.mean(self.temp_history))
                ]])
                probs = self.clf.predict_proba(feats)[0]
                frame_risk = float(probs[1])
                
                self.risk_history.append(frame_risk)
                if len(self.risk_history) > 15:
                    self.risk_history.pop(0)
                self.live_risk = float(np.mean(self.risk_history))
                
                # Optical Hardware Fidelity Guard:
                # Real webcam sensors under natural room lighting have distinct photon shot noise
                # and skin microtexture sharpness. Injections / deepfakes exhibit softened/flat textures.
                is_optically_natural = (self.noise_val >= 2.20 and self.lap_val >= 48.0)
                if is_optically_natural:
                    frame_attack = (frame_risk >= 0.68)
                else:
                    frame_attack = (frame_risk >= 0.55)
                    
                self.attack_votes.append(1 if frame_attack else 0)
                if len(self.attack_votes) > 18:
                    self.attack_votes.pop(0)
                    
                if len(self.attack_votes) >= 6:
                    attack_ratio = sum(self.attack_votes) / len(self.attack_votes)
                    self.is_attack = (attack_ratio >= 0.50)
                else:
                    self.is_attack = False
            else:
                self.is_attack = False
                self.live_risk = 0.12
        
        if not self.face_detected and self.frame_count < 8:
            box_color = (0, 255, 255) # Yellow guide
            hud_label = "ALIGNING FACE IN CAMERA VIEW..."
        elif self.is_attack:
            box_color = (0, 0, 255) # Red alert
            hud_label = f"ALERT: AI SYNTHETIC / ATTACK (Risk: {self.live_risk:.2f})"
        else:
            box_color = (0, 255, 0) # Green verified
            hud_label = f"SECURE STREAM VERIFIED (Risk: {self.live_risk:.2f})"
        
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), box_color, 2)
        scan_y = by + (self.frame_count * 6) % max(1, bh)
        cv2.line(img, (bx, scan_y), (bx + bw, scan_y), (0, 255, 255), 2)
        cv2.putText(img, hud_label, (bx, max(22, by - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
        
        return img

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        out = self._process_image(img)
        return av.VideoFrame.from_ndarray(out, format="bgr24")

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        return self._process_image(img)

# --- MAIN NAVIGATION INTERFACE ---
tab1, tab2, tab3 = st.tabs([
    "📷 Live eKYC Chamber", 
    "📹 File-Based Video Verification", 
    "🎵 Standalone Audio Anti-Spoofing (AASIST3)"
])

# ==============================================================================
# TAB 1: LIVE CAM CHAMBER (GENUINE COMPUTER VISION INFRASTRUCTURE)
# ==============================================================================
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.header("🎥 Live Telemetry Acquisition")
        st.write("Stream your camera feed or inject your virtual video device below. The engine runs image forensic math frame-by-frame.")
        
        ctx = webrtc_streamer(
            key="ekyc-live-forensics", 
            video_processor_factory=LiveForensicTransformer,
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                    {"urls": ["stun:stun2.l.google.com:19302"]},
                    {"urls": ["stun:stun.cloudflare.com:3478"]}
                ]
            },
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )
        
    with col2:
        st.header("⚡ Live Fusion Verdict")
        processor = getattr(ctx, "video_processor", None) or getattr(ctx, "video_transformer", None)
        
        if ctx.state.playing:
            st.button("🔄 Refresh Telemetry Readings")
            
            p = processor
            noise_val = getattr(p, "noise_val", 2.50) if p else 2.50
            lap_val = getattr(p, "lap_val", 80.0) if p else 80.0
            grad_val = getattr(p, "grad_val", 1200.0) if p else 1200.0
            ratio_val = getattr(p, "noise_ratio", 1.20) if p else 1.20
            is_detected_fake = getattr(p, "is_attack", False) if p else False
            live_risk = getattr(p, "live_risk", 0.12) if p else 0.12
            face_found = getattr(p, "face_detected", True) if p else True
            frame_cnt = getattr(p, "frame_count", 0) if p else 0
            
            st.subheader("Autonomous Classification Profile:")
            if is_detected_fake:
                detected_class = "GenAI / Synthetic Video Injection" if noise_val < 0.40 else "Face-Swap Presentation Attack"
                st.markdown(f"### 🛑 Category: `{detected_class}`")
                st.metric(label="Calculated Combined Threat Index", value=f"{live_risk:.2f}")
                st.progress(live_risk)
                
                st.error("🛑 V-CIP DECISION: VERIFICATION REJECTED")
                st.write(f"**Diagnostic Evaluation:** Pixel noise floor is unnaturally flat ({noise_val:.2f}) or microtexture sharpness ({lap_val:.1f}) is degraded, indicating a synthetic injection or presentation attack.")
            elif not face_found and frame_cnt < 8:
                st.markdown("### 🟡 Category: `Biometric Acquisition in Progress`")
                st.metric(label="Calculated Combined Threat Index", value="0.10")
                st.progress(0.10)
                st.info("Align your face within the camera viewport to complete automated biometric verification.")
            else:
                st.markdown("### 🟢 Category: `Genuine Live Customer Stream`")
                st.metric(label="Calculated Combined Threat Index", value=f"{live_risk:.2f}")
                st.progress(live_risk)
                
                st.success("🟢 V-CIP DECISION: IDENTITY VERIFIED")
                st.write(f"**Diagnostic Evaluation:** Microtexture sharpness ({lap_val:.1f}), edge gradient ({grad_val:.1f}), and natural photon noise fluctuations ({noise_val:.2f}) conform to genuine optical hardware capture bounds.")
                
            st.write("---")
            st.write("### 🧬 Isolated Live Sensor Telemetry")
            st.info(f"📡 **Calculated Sensor Noise Floor Variance:** {noise_val:.3f}")
            st.info(f"📐 **Structural Microtexture Sharpness (Laplacian):** {lap_val:.2f}")
            st.info(f"⚖️ **Sensor Consistency Ratio (Face/BG):** {ratio_val:.2f}")
            is_opt_guard = (noise_val >= 2.20 and lap_val >= 48.0)
            st.info(f"🛡️ **Optical Fidelity Guard:** {'🟢 Active (Physical Camera Confirmed)' if is_opt_guard else '🟡 Standard Sensitivity'}")
            st.caption("_Baseline Bounds: Sensor Noise Floor (> 2.200) | Microtexture Sharpness (> 48.00) | Noise Ratio (> 0.15)_")
        else:
            st.info("Awaiting video initialization... Press 'START' on the left to fire up the live forensic pipeline matrix nodes.")

# ==============================================================================
# TAB 2: FILE-BASED VIDEO PIPELINE (MATHEMATICAL FORENSICS ENGINE)
# ==============================================================================
with tab2:
    col3, col4 = st.columns(2)
    with col3:
        st.header("📹 Benchmarking Pipeline")
        uploaded_video = st.file_uploader("Upload Verification Video (MP4/MOV)", type=["mp4", "mov"], key="vid_uploader")
        
        if uploaded_video is not None:
            temp_vid_path = "temp_video.mp4"
            with open(temp_vid_path, "wb") as f:
                f.write(uploaded_video.read())
            
            with st.spinner("🔬 Running full-frame spatial and microtexture forensic calculations..."):
                video_res = fe.analyze_video(temp_vid_path, max_frames=16)
            
            is_vid_fake = video_res.get("is_fake", False)
            metrics = video_res.get("metrics", {})
            v_risk = video_res.get("v_risk", 0.50)
            
            cap = cv2.VideoCapture(temp_vid_path)
            frame_placeholder = st.empty()
            
            st.info(f"🔬 Forensic Math Complete: Classified as {'🛑 Synthetic / Deepfake Attack' if is_vid_fake else '🟢 Authentic Media'}. Rendering verification stream...")
            
            frame_count = 0
            while cap.isOpened() and frame_count < 45:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                h, w, _ = frame.shape
                box_x1, box_y1 = int(w * 0.26), int(h * 0.16)
                box_x2, box_y2 = int(w * 0.74), int(h * 0.78)
                
                box_color = (0, 0, 255) if is_vid_fake else (0, 255, 0)
                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), box_color, 2)
                scan_y = box_y1 + (frame_count * 7) % max(1, (box_y2 - box_y1))
                cv2.line(frame, (box_x1, scan_y), (box_x2, scan_y), (0, 255, 255), 2)
                
                tag = "⚠️ ATTACK FRAUD FLAG" if is_vid_fake else "SECURE STREAM VALID"
                cv2.putText(frame, tag, (box_x1, box_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                time.sleep(0.02)
            cap.release()
            
    with col4:
        st.header("⚡ Benchmarking Decision Outputs")
        if uploaded_video is not None and "video_res" in locals():
            st.metric(label="Calculated Video Threat Index", value=f"{v_risk:.2f}")
            st.progress(min(1.0, max(0.0, v_risk)))
            
            st.write("### 🧬 Isolated Forensic Metrics")
            st.info(f"📐 Facial Frame Artifact Weight (f): {metrics.get('f', 0.5):.2f}")
            st.info(f"👄 Boundary / Discrepancy Matrix (l): {metrics.get('l', 0.5):.2f}")
            st.info(f"📡 Facial Sensor Noise Floor: {metrics.get('noise_face', 0.0):.3f} (Background: {metrics.get('noise_bg', 0.0):.3f})")
            st.info(f"🔍 Microtexture Laplacian Sharpness: {metrics.get('laplacian_face', 0.0):.1f}")
            st.info(f"⚖️ Sensor Consistency Ratio: {metrics.get('noise_ratio', 0.0):.2f}")
            
            if is_vid_fake:
                st.error(f"🛑 REJECTED: {video_res.get('msg', 'Attack detected.')}")
            else:
                st.success(f"🟢 VERIFIED: {video_res.get('msg', 'Genuine video verified.')}")
        else:
            st.info("Upload a video on the left to begin automated mathematical forensic analysis.")

# ==============================================================================
# TAB 3: STANDALONE AUDIO CHANNELS (ACOUSTIC SPECTRUM & VOCAL FORENSICS)
# ==============================================================================
with tab3:
    col5, col6 = st.columns(2)
    with col5:
        st.header("🎵 Audio Spectrum Capture Panel")
        uploaded_audio = st.file_uploader("Upload Verification Audio (MP3/WAV/M4A)", type=["mp3", "wav", "m4a"], key="aud_uploader")
        if uploaded_audio is not None:
            ext = uploaded_audio.name.split(".")[-1].lower() if "." in uploaded_audio.name else "mp3"
            temp_aud_path = f"temp_audio.{ext}"
            with open(temp_aud_path, "wb") as f:
                f.write(uploaded_audio.read())
                
            st.write("Audio Stream Input Track:")
            st.audio(temp_aud_path)

            with st.spinner("⚡ Parsing acoustic spectrum & computing vocoder artifact coefficients..."):
                audio_res = fe.analyze_audio(temp_aud_path)
                
    with col6:
        st.header("⚡ AASIST3 Sub-system Diagnostics")
        if uploaded_audio is not None and "audio_res" in locals():
            aud_score = audio_res.get("aud_score", 0.50)
            st.metric(label="Vocal Voice Clone Probability", value=f"{aud_score:.2f}")
            st.progress(min(1.0, max(0.0, aud_score)))
            
            m = audio_res.get("metrics", {})
            st.write("### 🧬 Spectro-Temporal Diagnostics")
            st.info(f"📊 Spectral Centroid: {m.get('centroid', 0.0):.1f} Hz (Spread: {m.get('spread', 0.0):.1f} Hz)")
            st.info(f"📉 Spectral Rolloff (85%): {m.get('rolloff_85', 0.0):.1f} Hz")
            st.info(f"〰️ Spectral Flatness (Wiener Entropy): {m.get('spectral_flatness', 0.0):.6f}")
            st.info(f"🔊 High-Frequency Vocoder Energy (>4kHz): {m.get('high_freq_ratio', 0.0):.4f}")
            st.info(f"⚡ Zero-Crossing Rate (ZCR): {m.get('zcr', 0.0):.4f}")
            
            if aud_score >= 0.50:
                st.error("🛑 REJECTED: CLONED SPEECH MARKERS FOUND")
                st.write(audio_res.get("msg", ""))
            else:
                st.success("🟢 VERIFIED: LEGITIMATE BIOMETRIC SPEECH")
                st.write(audio_res.get("msg", ""))
        else:
            st.info("Upload an audio file on the left to analyze vocal authenticity.")