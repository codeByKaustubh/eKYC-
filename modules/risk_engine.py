import numpy as np

class MultimodalRiskEngine:
    """
    Multimodal Risk Aggregation and Decision Engine for eKYC.
    Combines Face, Liveness, Deepfake Video, Audio, and Lip-Sync signals.
    Outputs:
    - Overall Risk Score (0 - 100%)
    - Categorical Decision: ACCEPT / REVIEW / REJECT
    - Weighted modality breakdown
    - Natural language forensic report and security audit recommendations
    """

    def __init__(self, weights=None, thresholds=None):
        # Default security weights
        self.weights = weights or {
            "face": 0.15,
            "liveness": 0.25,
            "deepfake": 0.25,
            "audio": 0.20,
            "lipsync": 0.15
        }
        # Decision thresholds
        self.thresholds = thresholds or {
            "accept_max": 35.0,  # Below 35 => ACCEPT
            "review_max": 65.0   # 35 to 65 => REVIEW, Above 65 => REJECT
        }

    def evaluate_risk(self, face_obs, liveness_res, deepfake_res, audio_res, lipsync_res):
        """
        Combines modality results into unified decision.
        """
        # Extract sub-scores
        face_risk = self._compute_face_risk(face_obs)
        liveness_risk = liveness_res.get("liveness_risk_score", 50.0)
        deepfake_risk = deepfake_res.get("deepfake_risk_score", 50.0)
        audio_risk = audio_res.get("audio_risk_score", 50.0)
        lipsync_risk = lipsync_res.get("lipsync_risk_score", 50.0)

        # Normalize weights if audio or video missing
        active_weights = self.weights.copy()
        if audio_res.get("sample_rate", 0) == 0:
            # Re-distribute audio & lipsync weights to video components
            active_weights["face"] += 0.05
            active_weights["liveness"] += 0.15
            active_weights["deepfake"] += 0.15
            active_weights["audio"] = 0.0
            active_weights["lipsync"] = 0.0

        weight_sum = sum(active_weights.values())
        if weight_sum > 0:
            for k in active_weights:
                active_weights[k] /= weight_sum

        # Calculate overall weighted risk score
        overall_risk = (
            active_weights["face"] * face_risk +
            active_weights["liveness"] * liveness_risk +
            active_weights["deepfake"] * deepfake_risk +
            active_weights["audio"] * audio_risk +
            active_weights["lipsync"] * lipsync_risk
        )
        overall_risk = round(float(np.clip(overall_risk, 0.0, 100.0)), 1)

        # Determine Categorical Decision
        if overall_risk < self.thresholds["accept_max"]:
            decision = "ACCEPT"
            decision_color = "#2e7d32" # Green
            summary = "Identity verification passed. All biometric and media authenticity signals indicate a bona fide user."
        elif overall_risk < self.thresholds["review_max"]:
            decision = "REVIEW"
            decision_color = "#ed6c02" # Orange
            summary = "Suspicious or ambiguous signals detected. Manual verification by an eKYC compliance officer is recommended."
        else:
            decision = "REJECT"
            decision_color = "#d32f2f" # Red
            summary = "High risk presentation attack or manipulated media detected. Request blocked for identity spoofing security risk."

        # Compile forensic warning messages
        all_warnings = []
        all_warnings.extend(face_obs.get("anomaly_flags", []))
        all_warnings.extend(liveness_res.get("risk_factors", []))
        all_warnings.extend(deepfake_res.get("risk_factors", []))
        all_warnings.extend(audio_res.get("risk_factors", []))
        all_warnings.extend(lipsync_res.get("risk_factors", []))

        modality_breakdown = {
            "Face Presence & Bounding Box": {"score": round(face_risk, 1), "weight": round(active_weights["face"], 2)},
            "Liveness & Screen Replay": {"score": round(liveness_risk, 1), "weight": round(active_weights["liveness"], 2)},
            "Video Deepfake & SSIM": {"score": round(deepfake_risk, 1), "weight": round(active_weights["deepfake"], 2)},
            "Audio & Speech Anti-Spoofing": {"score": round(audio_risk, 1), "weight": round(active_weights["audio"], 2)},
            "Audio-Visual Lip-Sync": {"score": round(lipsync_risk, 1), "weight": round(active_weights["lipsync"], 2)}
        }

        return {
            "overall_risk_score": overall_risk,
            "decision": decision,
            "decision_color": decision_color,
            "summary": summary,
            "modality_breakdown": modality_breakdown,
            "warning_flags": all_warnings,
            "active_weights": active_weights,
            "thresholds": self.thresholds
        }

    def _compute_face_risk(self, face_obs):
        ratio = face_obs.get("face_detected_ratio", 0.0)
        multi = face_obs.get("multiple_faces_detected", False)
        var = face_obs.get("bbox_center_variance", 0.0)

        risk = 0.0
        if ratio < 0.5:
            risk += 60.0
        elif ratio < 0.8:
            risk += 30.0

        if multi:
            risk += 40.0
        if var > 2500.0:
            risk += 30.0

        return min(100.0, risk)
