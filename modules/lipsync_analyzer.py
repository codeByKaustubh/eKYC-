import cv2
import numpy as np

class LipSyncAnalyzer:
    """
    Audio-Visual Consistency & Lip-Sync Alignment Engine for eKYC.
    Correlates lower face mouth region opening dynamics with audio RMS energy.
    """

    def analyze_lipsync(self, annotated_frames, audio_results, fps=30.0):
        """
        Analyzes audio-visual synchrony.
        Returns:
            lipsync_results: dict containing correlation, lag, sub-score, and risk factors
        """
        rms_series = audio_results.get("rms_series", [])
        if not annotated_frames or len(rms_series) < 5:
            return self._empty_response("Audio track or video frames insufficient for lip-sync analysis")

        mouth_opening_series = []

        for f_idx, rgb_frame, bbox in annotated_frames:
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
            if bbox is not None:
                x, y, w, h = bbox
                # Mouth region is roughly bottom 35% of facial bounding box
                mouth_crop = gray[y + int(h * 0.65):y + h, x + int(w * 0.2):x + int(w * 0.8)]
            else:
                h_img, w_img = gray.shape
                mouth_crop = gray[int(h_img * 0.6):, int(w_img * 0.2):int(w_img * 0.8)]

            if mouth_crop.size == 0:
                mouth_opening_series.append(0.0)
                continue

            # Estimate mouth height / aperture via thresholded vertical intensity profile
            sobel_y = cv2.Sobel(mouth_crop, cv2.CV_64F, 0, 1, ksize=3)
            mouth_opening = float(np.std(sobel_y))
            mouth_opening_series.append(mouth_opening)

        # Interpolate mouth opening series to match audio RMS series length
        if len(mouth_opening_series) > 1 and len(rms_series) > 1:
            x_video = np.linspace(0, 1, len(mouth_opening_series))
            x_audio = np.linspace(0, 1, len(rms_series))
            mouth_interp = np.interp(x_audio, x_video, mouth_opening_series)
        else:
            return self._empty_response("Unable to align video and audio time series")

        # Standardize signals
        m_norm = (mouth_interp - np.mean(mouth_interp)) / (np.std(mouth_interp) + 1e-8)
        a_norm = (rms_series - np.mean(rms_series)) / (np.std(rms_series) + 1e-8)

        # Pearson Correlation
        corr = float(np.corrcoef(m_norm, a_norm)[0, 1])
        if np.isnan(corr):
            corr = 0.0

        # Cross-Correlation Lag Shift Analysis
        cross_corr = np.correlate(m_norm, a_norm, mode='full')
        lags = np.arange(-len(m_norm) + 1, len(m_norm))
        best_lag_idx = np.argmax(cross_corr)
        best_lag_sec = float(lags[best_lag_idx] * 0.01) # approx 10ms frame step

        risk_factors = []

        # Lip-Sync Risk Thresholding
        if corr < 0.0:
            sync_risk = 85.0
            risk_factors.append(f"Negative audio-visual correlation ({corr:.3f}) - Audio track dubbing / injection attack risk")
        elif corr < 0.20:
            sync_risk = 70.0
            risk_factors.append(f"Low lip-sync consistency ({corr:.3f}) - Mouth motion does not match spoken audio")
        elif abs(best_lag_sec) > 0.3:
            sync_risk = 55.0
            risk_factors.append(f"Pronounced audio-video latency lag ({best_lag_sec:.2f}s delay)")
        else:
            sync_risk = 10.0

        return {
            "lipsync_risk_score": round(sync_risk, 1),
            "correlation_coefficient": round(corr, 3),
            "best_lag_seconds": round(best_lag_sec, 2),
            "mouth_opening_series": [round(m, 3) for m in mouth_opening_series],
            "aligned_mouth_series": [round(m, 3) for m in mouth_interp],
            "aligned_audio_rms": [round(a, 3) for a in rms_series],
            "risk_factors": risk_factors
        }

    def _empty_response(self, reason):
        return {
            "lipsync_risk_score": 50.0,
            "correlation_coefficient": 0.0,
            "best_lag_seconds": 0.0,
            "mouth_opening_series": [],
            "aligned_mouth_series": [],
            "aligned_audio_rms": [],
            "risk_factors": [reason]
        }
