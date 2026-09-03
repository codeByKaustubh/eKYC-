import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
import sys
import tempfile
import json
import time

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from modules.sample_data_generator import SampleDataGenerator
from modules.preprocessor import VideoPreprocessor
from modules.face_analyzer import FaceAnalyzer
from modules.liveness_detector import LivenessDetector
from modules.deepfake_detector import DeepfakeDetector
from modules.audio_analyzer import AudioAnalyzer
from modules.lipsync_analyzer import LipSyncAnalyzer
from modules.risk_engine import MultimodalRiskEngine
from modules.evaluator import BenchmarkEvaluator

# Page Configuration
st.set_page_config(
    page_title="eKYC Multimodal Spoof Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphism & Modern Cybersecurity Aesthetics)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Header Container */
    .header-card {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.9) 0%, rgba(13, 17, 23, 0.95) 100%);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    
    .header-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #58a6ff 0%, #38d430 50%, #00e5ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    
    .header-subtitle {
        color: #8b949e;
        font-size: 0.95rem;
        font-weight: 400;
    }
    
    /* Decision Hero Badges */
    .decision-badge-accept {
        background: linear-gradient(135deg, rgba(46, 125, 50, 0.25) 0%, rgba(27, 94, 32, 0.4) 100%);
        border: 2px solid #2e7d32;
        color: #81c784;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .decision-badge-review {
        background: linear-gradient(135deg, rgba(237, 108, 2, 0.25) 0%, rgba(230, 81, 0, 0.4) 100%);
        border: 2px solid #ed6c02;
        color: #ffb74d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .decision-badge-reject {
        background: linear-gradient(135deg, rgba(211, 47, 47, 0.25) 0%, rgba(183, 28, 28, 0.4) 100%);
        border: 2px solid #d32f2f;
        color: #e57373;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }

    .decision-text {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: 2px;
        margin: 0;
    }
    
    /* Risk Metric Cards */
    .metric-card {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid rgba(48, 54, 61, 0.6);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #58a6ff;
    }
    
    .metric-label {
        font-size: 0.82rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Risk Progress Bar */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #2e7d32 , #ed6c02, #d32f2f);
    }
""", unsafe_allow_html=True)

# Initialize Demo Samples & Directories
@st.cache_resource
def init_sample_data():
    gen = SampleDataGenerator(data_dir="data")
    sample_paths = gen.generate_all_samples()
    return gen, sample_paths

data_gen, sample_media = init_sample_data()

# App Header
st.markdown("""
<div class="header-card">
    <div class="header-title">🛡️ Multimodal eKYC Presentation Attack & Injection Detection</div>
    <div class="header-subtitle">Layered Anti-Spoofing Architecture | Video, Facial, Behavioural, Deepfake, Speech & Lip-Sync Analysis Engine</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Setup
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/fingerprint-scan.png", width=64)
    st.header("⚙️ Media & Engine Config")
    
    input_mode = st.radio(
        "Select Verification Media:",
        ["Built-in Demo Samples", "Upload Custom Video & Audio"]
    )
    
    selected_video_path = None
    selected_audio_path = None
    
    if input_mode == "Built-in Demo Samples":
        sample_choice = st.selectbox(
            "Choose Test Case:",
            [
                "Genuine eKYC Verification (Bona Fide)",
                "Screen Replay Attack (Moire Grid Artifacts)",
                "Deepfake Face Swap Attack (Spatial Jitter)",
                "Voice Clone Speech Attack (Monotone Pitch)"
            ]
        )
        
        if sample_choice.startswith("Genuine"):
            selected_video_path = sample_media["genuine_video"]
            selected_audio_path = sample_media["genuine_audio"]
        elif sample_choice.startswith("Screen Replay"):
            selected_video_path = sample_media["replay_video"]
            selected_audio_path = sample_media["genuine_audio"]
        elif sample_choice.startswith("Deepfake"):
            selected_video_path = sample_media["deepfake_video"]
            selected_audio_path = sample_media["genuine_audio"]
        else: # Voice Clone
            selected_video_path = sample_media["genuine_video"]
            selected_audio_path = sample_media["clone_audio"]
            
    else: # Custom Upload
        uploaded_video = st.file_uploader("Upload eKYC Video (MP4 / AVI / MOV)", type=["mp4", "avi", "mov"])
        uploaded_audio = st.file_uploader("Upload Paired Audio (WAV / MP3)", type=["wav", "mp3"])
        
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            selected_video_path = tfile.name
            
        if uploaded_audio is not None:
            afile = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            afile.write(uploaded_audio.read())
            selected_audio_path = afile.name

    st.markdown("---")
    st.subheader("🎛️ Analysis Parameters")
    max_frames_sample = st.slider("Max Video Frames Sampled", 15, 90, 30, 5)
    
    st.markdown("---")
    st.caption("Patkar-Varde College | M.Sc. Cyber Security Project")
    st.caption("Student: Kaustubh Anant Rane")

# Tabs Navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 eKYC Verification",
    "📊 Multimodal Signal Deep-Dive",
    "🧪 Dataset Benchmark & Metrics",
    "⚙️ Risk Engine Tuner",
    "📜 Audit Log & Export"
])

# Shared session state for analysis results
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# TAB 1: eKYC MEDIA VERIFICATION
with tab1:
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("📽️ Input Media Preview")
        if selected_video_path and os.path.exists(selected_video_path):
            st.video(selected_video_path)
        else:
            st.info("Please select or upload a video file to proceed.")
            
        if selected_audio_path and os.path.exists(selected_audio_path):
            st.audio(selected_audio_path)
            
        run_analysis = st.button("🚀 Run Multimodal Verification Pipeline", type="primary", use_container_width=True)

    if run_analysis and selected_video_path and os.path.exists(selected_video_path):
        with st.spinner("Processing video frames, facial landmarks, deepfake artifacts & speech anti-spoofing..."):
            progress_bar = st.progress(0)
            
            # Step 1: Preprocessor
            progress_bar.progress(15)
            prep = VideoPreprocessor(max_frames=max_frames_sample)
            frames, v_meta = prep.process_video(selected_video_path)
            sr, signal, a_meta = prep.process_audio(selected_audio_path)
            
            # Step 2: Face Analyzer
            progress_bar.progress(35)
            face_mod = FaceAnalyzer()
            ann_frames, face_obs = face_mod.analyze_frames(frames, v_meta["width"], v_meta["height"])
            
            # Step 3: Liveness Detector
            progress_bar.progress(55)
            live_mod = LivenessDetector()
            liveness_res = live_mod.analyze_liveness(ann_frames)
            
            # Step 4: Deepfake Detector
            progress_bar.progress(70)
            df_mod = DeepfakeDetector()
            df_res = df_mod.analyze_deepfake(ann_frames)
            
            # Step 5: Audio & Speech Analyzer
            progress_bar.progress(85)
            aud_mod = AudioAnalyzer()
            aud_res = aud_mod.analyze_audio(sr, signal)
            
            # Step 6: Lip-Sync Analyzer
            progress_bar.progress(95)
            sync_mod = LipSyncAnalyzer()
            sync_res = sync_mod.analyze_lipsync(ann_frames, aud_res, fps=v_meta["fps"])
            
            # Step 7: Risk Engine
            progress_bar.progress(100)
            engine = MultimodalRiskEngine()
            eval_res = engine.evaluate_risk(face_obs, liveness_res, df_res, aud_res, sync_res)
            
            # Store in session state
            st.session_state.v_meta = v_meta
            st.session_state.a_meta = a_meta
            st.session_state.ann_frames = ann_frames
            st.session_state.face_obs = face_obs
            st.session_state.liveness_res = liveness_res
            st.session_state.df_res = df_res
            st.session_state.aud_res = aud_res
            st.session_state.sync_res = sync_res
            st.session_state.eval_res = eval_res
            st.session_state.analysis_done = True
            
            time.sleep(0.3)
            progress_bar.empty()

    with col_right:
        st.subheader("🛡️ Verification Decision & Risk Assessment")
        
        if st.session_state.analysis_done:
            eval_res = st.session_state.eval_res
            dec = eval_res["decision"]
            risk_score = eval_res["overall_risk_score"]
            
            # Decision Badge Render
            badge_class = f"decision-badge-{dec.lower()}"
            icon = "✅" if dec == "ACCEPT" else ("⚠️" if dec == "REVIEW" else "⛔")
            
            st.markdown(f"""
            <div class="{badge_class}">
                <div style="font-size: 1.1rem; text-transform: uppercase; letter-spacing: 2px;">Overall Decision</div>
                <div class="decision-text">{icon} {dec}</div>
                <div style="font-size: 1.2rem; font-weight: 600; margin-top: 8px;">
                    Weighted Risk Score: {risk_score}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.progress(float(risk_score) / 100.0)
            st.caption(f"**Security Summary:** {eval_res['summary']}")
            
            st.markdown("---")
            st.markdown("#### 📊 Modality Risk Breakdown")
            
            mb = eval_res["modality_breakdown"]
            for mod_name, info in mb.items():
                s = info["score"]
                w = info["weight"]
                color = "#2e7d32" if s < 35 else ("#ed6c02" if s < 65 else "#d32f2f")
                st.markdown(f"**{mod_name}** *(Weight: {w*100:.0f}%)*: `{s}%`")
                st.progress(float(s) / 100.0)
                
        else:
            st.info("Click 'Run Multimodal Verification Pipeline' to perform automated security analysis.")

    if st.session_state.analysis_done:
        st.markdown("---")
        st.subheader("🔍 Annotated Facial Tracking & Frame Inspector")
        
        ann_frames = st.session_state.ann_frames
        if ann_frames:
            slider_idx = st.slider("Select Frame Index:", 0, len(ann_frames) - 1, 0)
            f_idx, img, bbox = ann_frames[slider_idx]
            
            col_f1, col_f2 = st.columns([1.5, 1])
            with col_f1:
                st.image(img, caption=f"Sampled Frame #{f_idx} (BBox: {bbox})", use_column_width=True)
            with col_f2:
                st.markdown("#### Frame Observations")
                st.json({
                    "frame_index": f_idx,
                    "bounding_box": bbox,
                    "face_detected": bbox is not None,
                    "face_coverage": st.session_state.face_obs.get("mean_coverage_ratio", 0.0)
                })
                
                warnings = st.session_state.eval_res["warning_flags"]
                st.markdown("#### ⚠️ Security Warnings & Anomaly Flags")
                if warnings:
                    for w in warnings:
                        st.warning(f"• {w}")
                else:
                    st.success("No anomaly flags triggered. Clean presentation.")

# TAB 2: MULTIMODAL SIGNAL DEEP-DIVE
with tab2:
    if not st.session_state.analysis_done:
        st.info("Please run the verification pipeline in Tab 1 to unlock detailed signal analytics.")
    else:
        st.subheader("📈 Multimodal Biometric & Digital Signal Analytics")
        
        row1_c1, row1_c2 = st.columns(2)
        
        with row1_c1:
            st.markdown("#### 👁️ Eye Aspect Ratio (EAR) & Blink Dynamics")
            ear_data = st.session_state.liveness_res.get("ear_series", [])
            if ear_data:
                df_ear = pd.DataFrame({"Frame": range(len(ear_data)), "EAR_Proxy": ear_data})
                fig_ear = px.line(df_ear, x="Frame", y="EAR_Proxy", title="Eye Region Aspect / Contrast Curve (Blink Dips)",
                                  line_shape="spline", color_discrete_sequence=["#00e5ff"])
                fig_ear.update_layout(template="plotly_dark", height=320)
                st.plotly_chart(fig_ear, use_container_width=True)
                st.caption(f"**Blinks Detected:** {st.session_state.liveness_res.get('blinks_detected', 0)} | **Laplacian Sharpness:** {st.session_state.liveness_res.get('mean_laplacian_sharpness', 0)}")
            else:
                st.write("No EAR series available.")
                
        with row1_c2:
            st.markdown("#### 🎞️ Inter-Frame Structural Similarity (SSIM)")
            ssim_data = st.session_state.df_res.get("ssim_series", [])
            if ssim_data:
                df_ssim = pd.DataFrame({"Frame_Pair": range(len(ssim_data)), "SSIM": ssim_data})
                fig_ssim = px.area(df_ssim, x="Frame_Pair", y="SSIM", title="Temporal SSIM Consistency (Deepfake Jitter Check)",
                                  color_discrete_sequence=["#38d430"])
                fig_ssim.update_layout(template="plotly_dark", height=320)
                st.plotly_chart(fig_ssim, use_container_width=True)
                st.caption(f"**SSIM Variance:** {st.session_state.df_res.get('ssim_variance', 0)} | **Boundary Mismatch:** {st.session_state.df_res.get('boundary_mismatch_score', 0)}")
            else:
                st.write("No SSIM series available.")
                
        st.markdown("---")
        row2_c1, row2_c2 = st.columns(2)
        
        with row2_c1:
            st.markdown("#### 🎙️ Speech MFCC Spectrogram Matrix")
            mfcc_mat = st.session_state.aud_res.get("mfcc_matrix", np.zeros((1, 13)))
            if mfcc_mat.shape[0] > 1:
                fig_mfcc = px.imshow(mfcc_mat.T, labels=dict(x="Audio Frame", y="MFCC Coeff", color="Energy"),
                                     title="Mel-Frequency Cepstral Coefficients (13-Trom)",
                                     color_continuous_scale="Viridis")
                fig_mfcc.update_layout(template="plotly_dark", height=320)
                st.plotly_chart(fig_mfcc, use_container_width=True)
                st.caption(f"**Pitch Variance:** {st.session_state.aud_res.get('pitch_variance_hz', 0)} Hz | **Spectral Flatness:** {st.session_state.aud_res.get('mean_spectral_flatness', 0)}")
            else:
                st.write("No audio track loaded.")
                
        with row2_c2:
            st.markdown("#### 👄 Audio-Visual Lip-Sync Cross-Correlation")
            m_aligned = st.session_state.sync_res.get("aligned_mouth_series", [])
            a_aligned = st.session_state.sync_res.get("aligned_audio_rms", [])
            
            if m_aligned and a_aligned:
                df_sync = pd.DataFrame({
                    "Sample": range(len(m_aligned)),
                    "Mouth_Opening": m_aligned,
                    "Audio_RMS_Energy": a_aligned
                })
                fig_sync = px.line(df_sync, x="Sample", y=["Mouth_Opening", "Audio_RMS_Energy"],
                                   title="Mouth Motion vs Audio RMS Energy",
                                   color_discrete_sequence=["#ffab00", "#58a6ff"])
                fig_sync.update_layout(template="plotly_dark", height=320)
                st.plotly_chart(fig_sync, use_container_width=True)
                st.caption(f"**Pearson Correlation r:** {st.session_state.sync_res.get('correlation_coefficient', 0)} | **Lag Offset:** {st.session_state.sync_res.get('best_lag_seconds', 0)}s")
            else:
                st.write("Lip-sync series unavailable.")

# TAB 3: DATASET BENCHMARK & METRICS
with tab3:
    st.subheader("🧪 Quantitative Benchmark Evaluation Engine")
    st.markdown("Evaluate eKYC anti-spoofing performance metrics on secondary benchmark dataset samples (SDFVD-style labelled suite).")
    
    col_b1, col_b2 = st.columns([1, 2])
    
    with col_b1:
        num_bench_samples = st.slider("Benchmark Sample Count", 10, 100, 30, 10)
        risk_th = st.slider("Decision Threshold (Risk %)", 20.0, 80.0, 35.0, 5.0)
        run_bench = st.button("📊 Run Benchmark Evaluation", type="primary", use_container_width=True)
        
    if run_bench or "bench_metrics" not in st.session_state:
        bench_data = data_gen.generate_benchmark_suite(num_samples=num_bench_samples)
        y_true = [item["label"] for item in bench_data]
        y_scores = [item["risk_score"] for item in bench_data]
        
        evaluator = BenchmarkEvaluator()
        metrics = evaluator.evaluate_predictions(y_true, y_scores, threshold=risk_th)
        st.session_state.bench_metrics = metrics
        st.session_state.bench_data = bench_data

    metrics = st.session_state.bench_metrics
    
    with col_b2:
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        m_c1.metric("APCER (False Accept)", f"{metrics['apcer']*100:.1f}%")
        m_c2.metric("BPCER (False Reject)", f"{metrics['bpcer']*100:.1f}%")
        m_c3.metric("EER (Equal Error Rate)", f"{metrics['eer']*100:.1f}%")
        m_c4.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
        
    st.markdown("---")
    row_g1, row_g2 = st.columns(2)
    
    with row_g1:
        st.markdown("#### 🧩 Confusion Matrix")
        cm_dict = metrics["confusion_matrix"]
        cm_arr = np.array([
            [cm_dict["TN_Genuine_Correct"], cm_dict["FP_Genuine_as_Attack"]],
            [cm_dict["FN_Attack_as_Genuine"], cm_dict["TP_Attack_Correct"]]
        ])
        
        fig_cm = px.imshow(cm_arr, x=["Pred: Genuine", "Pred: Attack"], y=["True: Genuine", "True: Attack"],
                           text_auto=True, color_continuous_scale="Blues",
                           title="Biometric Confusion Matrix (ISO/IEC 30107-3 Standard)")
        fig_cm.update_layout(template="plotly_dark", height=320)
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with row_g2:
        st.markdown("#### 📉 APCER vs BPCER Threshold Sweep (EER Finding)")
        thresholds_sweep = np.linspace(0, 100, 50)
        apcer_list = []
        bpcer_list = []
        
        bench_data = st.session_state.bench_data
        yt = [item["label"] for item in bench_data]
        ys = [item["risk_score"] for item in bench_data]
        
        for th in thresholds_sweep:
            res_th = BenchmarkEvaluator().evaluate_predictions(yt, ys, threshold=th)
            apcer_list.append(res_th["apcer"])
            bpcer_list.append(res_th["bpcer"])
            
        df_eer = pd.DataFrame({"Threshold": thresholds_sweep, "APCER (False Accept)": apcer_list, "BPCER (False Reject)": bpcer_list})
        fig_eer = px.line(df_eer, x="Threshold", y=["APCER (False Accept)", "BPCER (False Reject)"],
                          title="Error Rates vs Decision Threshold", color_discrete_sequence=["#d32f2f", "#2e7d32"])
        fig_eer.update_layout(template="plotly_dark", height=320)
        st.plotly_chart(fig_eer, use_container_width=True)

# TAB 4: RISK ENGINE TUNER
with tab4:
    st.subheader("⚙️ Multimodal Risk Engine Weight & Threshold Tuner")
    st.markdown("Security Administrator Portal for dynamic weight allocation across modalities.")
    
    col_w1, col_w2 = st.columns(2)
    
    with col_w1:
        st.markdown("#### Modality Weights ($w_i$)")
        w_face = st.slider("Face Presence & Tracking Weight", 0.0, 0.5, 0.15, 0.05)
        w_live = st.slider("Liveness & Screen Replay Weight", 0.0, 0.5, 0.25, 0.05)
        w_df = st.slider("Deepfake & SSIM Weight", 0.0, 0.5, 0.25, 0.05)
        w_audio = st.slider("Speech Anti-Spoofing Weight", 0.0, 0.5, 0.20, 0.05)
        w_sync = st.slider("Lip-Sync Consistency Weight", 0.0, 0.5, 0.15, 0.05)
        
        total_w = w_face + w_live + w_df + w_audio + w_sync
        if abs(total_w - 1.0) > 0.01:
            st.warning(f"Total Weight Sum = {total_w:.2f} (Weights will be normalized to 1.0 automatically)")
        else:
            st.success("Weights perfectly balanced (Sum = 1.0)")
            
    with col_w2:
        st.markdown("#### Categorical Decision Boundaries")
        accept_th = st.slider("Accept Max Risk Threshold (%)", 10.0, 50.0, 35.0, 5.0)
        review_th = st.slider("Review Max Risk Threshold (%)", 50.0, 85.0, 65.0, 5.0)
        
        st.markdown("---")
        st.markdown("#### Live Weight Simulator")
        sim_scores = {
            "Face": 10.0,
            "Liveness": 75.0,
            "Deepfake": 80.0,
            "Audio": 20.0,
            "Lip-Sync": 60.0
        }
        
        norm_w = [w_face/total_w, w_live/total_w, w_df/total_w, w_audio/total_w, w_sync/total_w]
        sim_total = sum([s * w for s, w in zip(sim_scores.values(), norm_w)])
        
        sim_dec = "ACCEPT" if sim_total < accept_th else ("REVIEW" if sim_total < review_th else "REJECT")
        st.markdown(f"**Simulated Total Risk:** `{sim_total:.1f}%` -> Decision: **{sim_dec}**")

# TAB 5: AUDIT LOG & EXPORT
with tab5:
    st.subheader("📜 Security Audit Logs & Forensic Report Export")
    
    if st.session_state.analysis_done:
        eval_res = st.session_state.eval_res
        v_meta = st.session_state.v_meta
        
        report_dict = {
            "system": "eKYC Multimodal Spoof Detection Architecture",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "video_metadata": v_meta,
            "decision": eval_res["decision"],
            "overall_risk_score": eval_res["overall_risk_score"],
            "modality_scores": eval_res["modality_breakdown"],
            "security_warning_flags": eval_res["warning_flags"]
        }
        
        st.markdown("#### Executive Summary Json Report")
        st.json(report_dict)
        
        json_bytes = json.dumps(report_dict, indent=4).encode('utf-8')
        st.download_button(
            label="📥 Download JSON Security Audit Report",
            data=json_bytes,
            file_name=f"eKYC_Audit_Report_{int(time.time())}.json",
            mime="application/json",
            use_container_width=True
        )
    else:
        st.info("Run verification in Tab 1 to generate timestamped audit logs.")
        
    st.markdown("---")
    st.markdown("#### 📚 Academic References & Standards")
    st.markdown("""
    - **Felouat et al. (2024)**: *eKYC-DF: A Large-Scale Deepfake Dataset for Developing and Evaluating eKYC Systems*, IEEE Access.
    - **ISO/IEC 30107-3:2023**: *Biometric presentation attack detection — Part 3: Testing and reporting*.
    - **RBI V-CIP Directions**: *Reserve Bank of India Master Direction – Know Your Customer (KYC) Direction (Video-based Customer Identification Process)*.
    - **AASIST**: *Audio anti-spoofing and speech deepfake detection architecture*.
    """)
