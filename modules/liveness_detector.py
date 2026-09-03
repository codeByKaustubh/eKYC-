import cv2
import numpy as np

class LivenessDetector:
    """
    Liveness & Behavioural Analysis Engine for eKYC.
    Evaluates:
    - Eye region variance & Eye Aspect Ratio (EAR) proxy (Blink Detection)
    - 2D FFT Moire / Screen Replay Frequency Artifacts
    - Laplacian Sharpness & Blur Analysis
    - Color Space Gamut (YCrCb / HSV) Distribution
    """

    def analyze_liveness(self, annotated_frames):
        """
        Analyzes sampled frames with bounding boxes.
        Returns:
            liveness_metrics: dict containing sub-scores and time-series for UI plotting
        """
        if not annotated_frames:
            return self._empty_response("No annotated frames provided")

        ear_series = []
        laplacian_vars = []
        fft_high_freq_ratios = []
        skin_chroma_vars = []

        for f_idx, rgb_frame, bbox in annotated_frames:
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
            
            if bbox is not None:
                x, y, w, h = bbox
                face_crop_gray = gray[y:y+h, x:x+w]
                face_crop_rgb = rgb_frame[y:y+h, x:x+w]
            else:
                face_crop_gray = gray
                face_crop_rgb = rgb_frame

            if face_crop_gray.size == 0 or face_crop_gray.shape[0] < 10 or face_crop_gray.shape[1] < 10:
                continue

            # 1. Eye Region / Blink proxy
            # Focus on top 50% of face crop for eye region contrast & aspect ratio
            h_crop, w_crop = face_crop_gray.shape[:2]
            upper_face = face_crop_gray[int(h_crop*0.15):int(h_crop*0.50), :]

            if upper_face.size > 0:
                # Vertical vs horizontal gradient ratio in eye region
                v_std = np.std(np.mean(upper_face, axis=1))
                h_std = np.std(np.mean(upper_face, axis=0)) + 1e-5
                ear_proxy = v_std / h_std
            else:
                ear_proxy = 0.5

            ear_series.append(float(ear_proxy))

            # 2. Laplacian Blur & Focus Sharpness
            lap_var = cv2.Laplacian(face_crop_gray, cv2.CV_64F).var()
            laplacian_vars.append(float(lap_var))

            # 3. 2D FFT Frequency Analysis (Screen Moire & Replay Patterns)
            fft_score = self._compute_fft_moire_score(face_crop_gray)
            fft_high_freq_ratios.append(fft_score)

            # 4. Color Space Gamut (YCrCb Skin Chroma Variance)
            ycrcb = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2YCrCb)
            cr_std = np.std(ycrcb[:, :, 1])
            cb_std = np.std(ycrcb[:, :, 2])
            skin_chroma_vars.append(float(cr_std + cb_std))

        # Summarize time series into metrics
        ear_variance = float(np.std(ear_series)) if ear_series else 0.0

        # Estimate blinks from EAR dips
        blinks_detected = 0
        if len(ear_series) > 5:
            threshold = np.mean(ear_series) - 0.4 * np.std(ear_series)
            for i in range(1, len(ear_series) - 1):
                if ear_series[i] < threshold and ear_series[i] < ear_series[i-1] and ear_series[i] < ear_series[i+1]:
                    blinks_detected += 1

        mean_laplacian = float(np.mean(laplacian_vars)) if laplacian_vars else 0.0
        mean_fft_moire = float(np.mean(fft_high_freq_ratios)) if fft_high_freq_ratios else 0.0
        mean_chroma_var = float(np.mean(skin_chroma_vars)) if skin_chroma_vars else 0.0

        # Liveness Risk calculation (0 = Bona fide / Natural, 100 = Spoof)
        liveness_risk_factors = []

        # Risk 1: Static face / zero blink activity
        if blinks_detected == 0 and len(ear_series) >= 15:
            blink_risk = 75.0
            liveness_risk_factors.append("No eye blink activity detected across video (Static media presentation attack)")
        elif ear_variance < 0.01:
            blink_risk = 60.0
            liveness_risk_factors.append("Extremely rigid facial posture (Static photo risk)")
        else:
            blink_risk = max(0.0, 30.0 - blinks_detected * 10.0)

        # Risk 2: Blur or unnatural print focus
        if mean_laplacian < 20.0:
            sharpness_risk = 70.0
            liveness_risk_factors.append(f"Low frame sharpness / blur detected (Laplacian: {mean_laplacian:.1f})")
        elif mean_laplacian > 1800.0:
            sharpness_risk = 50.0
            liveness_risk_factors.append(f"Unnaturally harsh edge gradients (Laplacian: {mean_laplacian:.1f})")
        else:
            sharpness_risk = 10.0

        # Risk 3: Moire / Screen Replay FFT artifacts
        if mean_fft_moire > 0.45:
            moire_risk = 85.0
            liveness_risk_factors.append(f"High 2D FFT periodic frequency peak (Screen replay attack indicator: {mean_fft_moire:.3f})")
        elif mean_fft_moire > 0.35:
            moire_risk = 50.0
            liveness_risk_factors.append("Moderate screen texture / display grid artifact detected")
        else:
            moire_risk = 10.0

        # Risk 4: Compressed Chroma
        if mean_chroma_var < 5.0:
            chroma_risk = 65.0
            liveness_risk_factors.append("Compressed color gamut / uniform digital screen skin tone")
        else:
            chroma_risk = 10.0

        overall_liveness_risk = (0.35 * moire_risk + 0.25 * blink_risk +
                                 0.20 * sharpness_risk + 0.20 * chroma_risk)

        liveness_results = {
            "liveness_risk_score": round(overall_liveness_risk, 1),
            "blinks_detected": blinks_detected,
            "ear_series": [round(x, 4) for x in ear_series],
            "mean_laplacian_sharpness": round(mean_laplacian, 2),
            "mean_fft_moire_score": round(mean_fft_moire, 4),
            "mean_chroma_var": round(mean_chroma_var, 2),
            "sub_risks": {
                "screen_replay_moire": round(moire_risk, 1),
                "blink_static_risk": round(blink_risk, 1),
                "sharpness_blur_risk": round(sharpness_risk, 1),
                "chroma_gamut_risk": round(chroma_risk, 1)
            },
            "risk_factors": liveness_risk_factors
        }

        return liveness_results

    def _compute_fft_moire_score(self, img_gray):
        """
        Computes 2D Fast Fourier Transform high-frequency power spectrum ratio
        to identify screen grid / Moire artifacts.
        """
        h, w = img_gray.shape
        if h < 32 or w < 32:
            return 0.0

        # Compute 2D FFT and shift zero-frequency component to center
        f = np.fft.fft2(img_gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

        # Center coordinates
        cy, cx = h // 2, w // 2

        # Create radial mask for high frequency ring vs low frequency center
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        radius = np.sqrt(x**2 + y**2)

        inner_radius = min(h, w) * 0.15
        outer_radius = min(h, w) * 0.45

        high_freq_mask = (radius >= inner_radius) & (radius <= outer_radius)
        total_mask = radius > 0

        high_freq_energy = np.sum(magnitude_spectrum[high_freq_mask])
        total_energy = np.sum(magnitude_spectrum[total_mask]) + 1e-8

        return float(high_freq_energy / total_energy)

    def _empty_response(self, reason):
        return {
            "liveness_risk_score": 50.0,
            "blinks_detected": 0,
            "ear_series": [],
            "mean_laplacian_sharpness": 0.0,
            "mean_fft_moire_score": 0.0,
            "mean_chroma_var": 0.0,
            "sub_risks": {
                "screen_replay_moire": 50.0,
                "blink_static_risk": 50.0,
                "sharpness_blur_risk": 50.0,
                "chroma_gamut_risk": 50.0
            },
            "risk_factors": [reason]
        }
