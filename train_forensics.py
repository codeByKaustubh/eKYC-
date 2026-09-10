import os
import glob
import pickle
import numpy as np
import cv2
import av
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

os.makedirs("models", exist_ok=True)

# ==========================================
# 1. AUDIO FORENSIC FEATURE EXTRACTION
# ==========================================
def load_audio_any(file_path_or_bytes):
    try:
        container = av.open(file_path_or_bytes)
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=16000)
        frames = []
        for frame in container.decode(audio=0):
            frame.pts = None
            for rf in resampler.resample(frame):
                frames.append(rf.to_ndarray())
        container.close()
        if not frames:
            return np.array([], dtype=np.float32), 16000
        y = np.concatenate(frames, axis=1).squeeze()
        return y.astype(np.float32), 16000
    except Exception as e:
        print(f"Error loading audio {file_path_or_bytes}: {e}")
        return np.array([], dtype=np.float32), 16000

def extract_audio_features(y, sr=16000):
    if len(y) == 0:
        return None
    # DC offset removal & peak normalization
    y = y - np.mean(y)
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    # Trim leading/trailing silence
    energy = y ** 2
    thresh = 0.005 * np.max(energy)
    active = np.where(energy > thresh)[0]
    if len(active) > sr:
        y = y[active[0]:active[-1]]
    
    # 1. Zero Crossing Rate
    zcr = float(np.mean(np.abs(np.diff(np.signbit(y)))))

    # 2. Welch Power Spectral Density (PSD)
    f, psd = welch(y, fs=sr, nperseg=1024)
    total_power = np.sum(psd) + 1e-12
    norm_psd = psd / total_power

    # 3. Spectral Moments (Centroid, Spread, Skew, Kurtosis)
    centroid = float(np.sum(f * norm_psd))
    spread = float(np.sqrt(np.sum(((f - centroid) ** 2) * norm_psd)))
    skewness = float(np.sum(((f - centroid) ** 3) * norm_psd) / (spread ** 3 + 1e-9))
    kurt = float(np.sum(((f - centroid) ** 4) * norm_psd) / (spread ** 4 + 1e-9))

    # 4. Energy Bands (Formant & Vocoder Bandwidths)
    b_low = float(np.sum(norm_psd[f < 800]))             # 0 - 800 Hz (f0 and fundamental vowels)
    b_mid1 = float(np.sum(norm_psd[(f >= 800) & (f < 2000)])) # 800 - 2000 Hz (formants)
    b_mid2 = float(np.sum(norm_psd[(f >= 2000) & (f < 4000)]))# 2000 - 4000 Hz (consonants)
    b_high = float(np.sum(norm_psd[f >= 4000]))          # > 4000 Hz (vocoder cut-off / synthetic noise)

    # 5. Spectral Flatness (Wiener entropy)
    geom_mean = np.exp(np.mean(np.log(psd + 1e-12)))
    flatness = float(geom_mean / (np.mean(psd) + 1e-12))

    # 6. Spectral Rolloff (85% and 95%)
    cumsum = np.cumsum(norm_psd)
    r85_arr = np.where(cumsum >= 0.85)[0]
    r85 = float(f[r85_arr[0]]) if len(r85_arr) > 0 else float(f[-1])
    r95_arr = np.where(cumsum >= 0.95)[0]
    r95 = float(f[r95_arr[0]]) if len(r95_arr) > 0 else float(f[-1])

    # 7. Energy Envelope Dynamics (25ms window, 10ms hop)
    frame_size = int(0.025 * sr)
    hop = int(0.010 * sr)
    frame_energies = [float(np.sum(y[i:i+frame_size]**2)) for i in range(0, len(y)-frame_size, hop)]
    env_std = float(np.std(frame_energies)) if frame_energies else 0.0

    features = [
        zcr, centroid, spread, skewness, kurt,
        b_low, b_mid1, b_mid2, b_high,
        flatness, r85, r95, env_std
    ]
    meta = {
        "zcr": zcr,
        "centroid": centroid,
        "spread": spread,
        "b_high": b_high,
        "flatness": flatness,
        "r85": r85
    }
    return features, meta

# ==========================================
# 2. VIDEO FORENSIC FEATURE EXTRACTION
# ==========================================
def extract_video_features(video_path, max_frames=20):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None, {}

    step = max(1, total_frames // max_frames)
    noise_face_list = []
    noise_bg_list = []
    lap_face_list = []
    lap_bg_list = []
    grad_face_list = []
    fft_hf_list = []
    temporal_diffs = []
    prev_face_gray = None

    frame_idx = 0
    sampled_count = 0
    while cap.isOpened() and sampled_count < max_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue
        frame_idx += 1
        sampled_count += 1

        h, w, _ = frame.shape
        # Face ROI (center 40% width, 15% to 65% height)
        fy1, fy2 = int(h * 0.15), int(h * 0.65)
        fx1, fx2 = int(w * 0.30), int(w * 0.70)

        # Background reference ROI (lower corner, outside face swap region)
        by1, by2 = int(h * 0.70), int(h * 0.95)
        bx1, bx2 = int(w * 0.05), int(w * 0.30)

        face_crop = frame[fy1:fy2, fx1:fx2]
        bg_crop = frame[by1:by2, bx1:bx2]
        if face_crop.size == 0 or bg_crop.size == 0:
            continue

        gf = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(bg_crop, cv2.COLOR_BGR2GRAY)

        # 1. High frequency sensor noise floor variance
        blur_f = cv2.medianBlur(gf, 3)
        diff_f = cv2.absdiff(gf, blur_f)
        nf = float(np.var(diff_f))

        blur_b = cv2.medianBlur(gb, 3)
        diff_b = cv2.absdiff(gb, blur_b)
        nb = float(np.var(diff_b))

        # 2. Laplacian micro-texture sharpness
        lap_f = float(cv2.Laplacian(gf, cv2.CV_64F).var())
        lap_b = float(cv2.Laplacian(gb, cv2.CV_64F).var())

        # 3. Sobel edge gradient magnitude
        gx = cv2.Sobel(gf, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gf, cv2.CV_64F, 0, 1, ksize=3)
        grad_f = float(np.mean(np.sqrt(gx**2 + gy**2)))

        # 4. 2D FFT High-Frequency Spectral Ratio
        gf_thumb = cv2.resize(gf, (64, 64))
        f_shift = np.fft.fftshift(np.fft.fft2(gf_thumb))
        mag = np.abs(f_shift)
        low_mag = mag[24:40, 24:40] # center low freqs
        hf_ratio = float((np.sum(mag**2) - np.sum(low_mag**2)) / (np.sum(mag**2) + 1e-9))

        # 5. Temporal consistency
        if prev_face_gray is not None and prev_face_gray.shape == gf.shape:
            t_diff = float(np.mean(cv2.absdiff(gf, prev_face_gray)))
            temporal_diffs.append(t_diff)
        prev_face_gray = gf

        noise_face_list.append(nf)
        noise_bg_list.append(nb)
        lap_face_list.append(lap_f)
        lap_bg_list.append(lap_b)
        grad_face_list.append(grad_f)
        fft_hf_list.append(hf_ratio)

    cap.release()
    if not noise_face_list:
        return None, {}

    m_nf = float(np.mean(noise_face_list))
    m_nb = float(np.mean(noise_bg_list))
    m_lap_f = float(np.mean(lap_face_list))
    m_lap_b = float(np.mean(lap_bg_list))
    m_grad_f = float(np.mean(grad_face_list))
    m_fft_hf = float(np.mean(fft_hf_list))
    m_temp = float(np.mean(temporal_diffs)) if temporal_diffs else 0.0

    n_ratio = float(m_nf / (m_nb + 1e-6))
    lap_ratio = float(m_lap_f / (m_lap_b + 1e-6))

    features = [
        m_nf,
        float(np.std(noise_face_list)),
        m_nb,
        n_ratio,
        m_lap_f,
        lap_ratio,
        m_grad_f,
        m_fft_hf,
        m_temp
    ]
    meta = {
        "noise_floor": m_nf,
        "laplacian_var": m_lap_f,
        "noise_ratio": n_ratio,
        "edge_gradient": m_grad_f,
        "fft_hf_ratio": m_fft_hf,
        "temporal_delta": m_temp
    }
    return features, meta

# ==========================================
# 3. TRAINING & SERIALIZATION
# ==========================================
def train_and_save():
    print("Training Audio Spoofing Detector...")
    real_audios = glob.glob("Audio/*/*/original.*")
    fake_audios = glob.glob("Audio/*/*/synthetic_*.mp3")
    
    X_aud, y_aud = [], []
    for ra in real_audios:
        aud, sr = load_audio_any(ra)
        feats, _ = extract_audio_features(aud, sr)
        if feats:
            X_aud.append(feats)
            y_aud.append(0) # Real
            
    for fa in fake_audios:
        aud, sr = load_audio_any(fa)
        feats, _ = extract_audio_features(aud, sr)
        if feats:
            X_aud.append(feats)
            y_aud.append(1) # Fake
            
    X_aud = np.array(X_aud)
    y_aud = np.array(y_aud)
    
    audio_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
    ])
    audio_pipeline.fit(X_aud, y_aud)
    print(f"Audio model trained on {len(X_aud)} samples.")
    
    print("Training Video Forensic Detector...")
    real_videos = glob.glob("Real/v*.mp4")
    fake_videos = glob.glob("Fake/vs*.mp4")
    
    X_vid, y_vid = [], []
    for rv in real_videos:
        feats, _ = extract_video_features(rv, max_frames=12)
        if feats:
            X_vid.append(feats)
            y_vid.append(0) # Real
            
    for fv in fake_videos:
        feats, _ = extract_video_features(fv, max_frames=12)
        if feats:
            X_vid.append(feats)
            y_vid.append(1) # Fake
            
    X_vid = np.array(X_vid)
    y_vid = np.array(y_vid)
    
    video_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', ExtraTreesClassifier(n_estimators=100, max_depth=6, random_state=42))
    ])
    video_pipeline.fit(X_vid, y_vid)
    print(f"Video model trained on {len(X_vid)} samples.")
    
    model_payload = {
        "audio_pipeline": audio_pipeline,
        "video_pipeline": video_pipeline,
        "meta": {
            "version": "2.0-mathematical",
            "audio_samples": len(X_aud),
            "video_samples": len(X_vid)
        }
    }
    
    with open("models/forensic_models.pkl", "wb") as f:
        pickle.dump(model_payload, f)
        
    print("Saved models to models/forensic_models.pkl successfully!")

if __name__ == "__main__":
    train_and_save()
