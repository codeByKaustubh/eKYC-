import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

class DeepfakeDetector:
    """
    Video Deepfake & Spatial-Temporal Anomaly Detector.
    Evaluates:
    - Inter-frame SSIM & temporal structural jitter
    - Face boundary blend artifact & edge ratio mismatch
    - Frequency domain generative upsampling artifacts
    - Color channel correlation anomalies
    """

    def analyze_deepfake(self, annotated_frames):
        """
        Analyzes video frames for deepfake & synthetic manipulation traces.
        Returns:
            deepfake_results: dict containing scores, metrics, time-series, and risk factors
        """
        if len(annotated_frames) < 2:
            return self._empty_response("Insufficient frames for deepfake temporal analysis")

        ssim_scores = []
        boundary_mismatch_scores = []
        channel_corr_scores = []
        spectral_roll_offs = []

        prev_face_gray = None

        for f_idx, rgb_frame, bbox in annotated_frames:
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
            
            if bbox is not None:
                x, y, w, h = bbox
                face_crop_rgb = rgb_frame[y:y+h, x:x+w]
                face_crop_gray = gray[y:y+h, x:x+w]
            else:
                face_crop_rgb = rgb_frame
                face_crop_gray = gray

            if face_crop_gray.size == 0 or face_crop_gray.shape[0] < 20 or face_crop_gray.shape[1] < 20:
                continue

            # 1. Temporal Inter-frame SSIM
            resized_gray = cv2.resize(face_crop_gray, (128, 128))
            if prev_face_gray is not None:
                score, _ = ssim(prev_face_gray, resized_gray, full=True)
                ssim_scores.append(float(score))
            prev_face_gray = resized_gray

            # 2. Facial Boundary Edge Mismatch (Blend Artifacts)
            boundary_score = self._analyze_face_boundary_blend(gray, bbox)
            boundary_mismatch_scores.append(boundary_score)

            # 3. Color Channel Correlation (R-G, G-B skin correlation)
            r, g, b = cv2.split(face_crop_rgb)
            rg_corr = np.corrcoef(r.ravel(), g.ravel())[0, 1]
            gb_corr = np.corrcoef(g.ravel(), b.ravel())[0, 1]
            if not np.isnan(rg_corr) and not np.isnan(gb_corr):
                channel_corr_scores.append(float(min(rg_corr, gb_corr)))

            # 4. Spectral Roll-Off / Generative Upsampling Artifacts
            roll_off = self._compute_spectral_roll_off(face_crop_gray)
            spectral_roll_offs.append(roll_off)

        # Aggregate Metrics
        mean_ssim = float(np.mean(ssim_scores)) if ssim_scores else 0.95
        ssim_var = float(np.std(ssim_scores)) if ssim_scores else 0.0

        mean_boundary_mismatch = float(np.mean(boundary_mismatch_scores)) if boundary_mismatch_scores else 0.0
        mean_channel_corr = float(np.mean(channel_corr_scores)) if channel_corr_scores else 0.95
        mean_spectral_rolloff = float(np.mean(spectral_roll_offs)) if spectral_roll_offs else 0.5

        # Risk Computation
        risk_factors = []

        # Risk 1: Temporal Jitter / Unnatural SSIM volatility
        if ssim_var > 0.08:
            jitter_risk = 80.0
            risk_factors.append(f"High inter-frame temporal jitter / frame warping (SSIM variance: {ssim_var:.3f})")
        elif ssim_var > 0.04:
            jitter_risk = 50.0
            risk_factors.append("Moderate frame-to-frame structural instability")
        else:
            jitter_risk = 10.0

        # Risk 2: Face Boundary Blending Artifacts
        if mean_boundary_mismatch > 0.4:
            blend_risk = 85.0
            risk_factors.append(f"Pronounced facial boundary blending artifact (Face-swap boundary score: {mean_boundary_mismatch:.2f})")
        elif mean_boundary_mismatch > 0.25:
            blend_risk = 55.0
            risk_factors.append("Moderate edge discrepancy around facial oval boundary")
        else:
            blend_risk = 10.0

        # Risk 3: Color Channel Decorrelation
        if mean_channel_corr < 0.70:
            color_risk = 75.0
            risk_factors.append(f"Skin color channel decorrelation (Mean channel corr: {mean_channel_corr:.3f})")
        else:
            color_risk = 10.0

        # Risk 4: Generative High-Frequency Anomaly
        if mean_spectral_rolloff < 0.25:
            spectral_risk = 80.0
            risk_factors.append("Missing natural micro-textures / GAN smoothed facial region")
        elif mean_spectral_rolloff > 0.85:
            spectral_risk = 70.0
            risk_factors.append("High-frequency checkerboard up-sampling noise artifact")
        else:
            spectral_risk = 15.0

        overall_deepfake_risk = (0.35 * blend_risk + 0.30 * jitter_risk +
                                 0.20 * spectral_risk + 0.15 * color_risk)

        return {
            "deepfake_risk_score": round(overall_deepfake_risk, 1),
            "mean_ssim": round(mean_ssim, 4),
            "ssim_variance": round(ssim_var, 4),
            "ssim_series": [round(s, 4) for s in ssim_scores],
            "boundary_mismatch_score": round(mean_boundary_mismatch, 3),
            "channel_correlation": round(mean_channel_corr, 3),
            "spectral_rolloff": round(mean_spectral_rolloff, 3),
            "sub_risks": {
                "face_swap_boundary": round(blend_risk, 1),
                "temporal_jitter": round(jitter_risk, 1),
                "generative_spectral": round(spectral_risk, 1),
                "color_decorrelation": round(color_risk, 1)
            },
            "risk_factors": risk_factors
        }

    def _analyze_face_boundary_blend(self, full_gray, bbox):
        """
        Measures gradient mismatch along the face oval boundary vs inner face.
        """
        if bbox is None:
            return 0.0

        x, y, w, h = bbox
        img_h, img_w = full_gray.shape

        # Inner face region
        inner_crop = full_gray[y + int(h*0.2):y + int(h*0.8), x + int(w*0.2):x + int(w*0.8)]
        if inner_crop.size == 0:
            return 0.0

        inner_grad = cv2.Laplacian(inner_crop, cv2.CV_64F).var()

        # Outer boundary ring region
        mask_outer = np.zeros_like(full_gray, dtype=np.uint8)
        cv2.rectangle(mask_outer, (max(0, x-10), max(0, y-10)), (min(img_w, x+w+10), min(img_h, y+h+10)), 255, -1)
        mask_inner = np.zeros_like(full_gray, dtype=np.uint8)
        cv2.rectangle(mask_inner, (x+5, y+5), (x+w-5, y+h-5), 255, -1)
        ring_mask = cv2.bitwise_and(mask_outer, cv2.bitwise_not(mask_inner))

        ring_pixels = full_gray[ring_mask > 0]
        if ring_pixels.size == 0:
            return 0.0

        ring_var = float(np.var(ring_pixels))
        
        # Mismatch ratio
        ratio = abs(inner_grad - ring_var) / (inner_grad + ring_var + 1e-5)
        return float(ratio)

    def _compute_spectral_roll_off(self, img_gray):
        """
        Computes 1D magnitude spectrum roll-off point in facial region.
        """
        f = np.fft.fft2(img_gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift).ravel()
        magnitude = np.sort(magnitude)
        
        cum_energy = np.cumsum(magnitude)
        total_energy = cum_energy[-1] + 1e-8
        
        # 85% energy cutoff index
        cutoff_idx = np.searchsorted(cum_energy, 0.85 * total_energy)
        norm_rolloff = cutoff_idx / len(magnitude)
        return float(norm_rolloff)

    def _empty_response(self, reason):
        return {
            "deepfake_risk_score": 50.0,
            "mean_ssim": 1.0,
            "ssim_variance": 0.0,
            "ssim_series": [],
            "boundary_mismatch_score": 0.0,
            "channel_correlation": 1.0,
            "spectral_rolloff": 0.5,
            "sub_risks": {
                "face_swap_boundary": 50.0,
                "temporal_jitter": 50.0,
                "generative_spectral": 50.0,
                "color_decorrelation": 50.0
            },
            "risk_factors": [reason]
        }
