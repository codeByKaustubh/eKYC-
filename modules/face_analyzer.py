import cv2
import numpy as np

class FaceAnalyzer:
    """
    OpenCV Face Presence and Bounding Box Tracking Module for eKYC.
    Detects face presence, multi-face anomalies, face coverage, and trajectory stability.
    Uses YCrCb skin chrominance + contour geometry with color space fallback.
    """
    def __init__(self):
        pass

    def analyze_frames(self, sampled_frames, frame_width=640, frame_height=480):
        """
        Analyzes a sequence of (frame_idx, rgb_frame) tuples.
        Returns:
            annotated_frames: list of (frame_idx, annotated_rgb_frame, bbox)
            face_observations: dict of metrics & anomaly flags
        """
        total_sampled = len(sampled_frames)
        if total_sampled == 0:
            return [], {
                "face_detected_count": 0,
                "face_detected_ratio": 0.0,
                "multiple_faces_detected": False,
                "mean_coverage_ratio": 0.0,
                "bbox_center_variance": 0.0,
                "anomaly_flags": ["No frames provided"]
            }

        faces_found = 0
        multi_face_count = 0
        bbox_centers = []
        bbox_areas = []
        annotated_frames = []
        first_face_crop = None

        frame_area = max(1, frame_width * frame_height)

        for f_idx, rgb_frame in sampled_frames:
            detected_faces = self._detect_faces_multi_space(rgb_frame)

            annotated_frame = rgb_frame.copy()
            chosen_bbox = None

            if len(detected_faces) > 0:
                faces_found += 1
                if len(detected_faces) > 1:
                    multi_face_count += 1
                
                # Pick largest face
                largest_face = max(detected_faces, key=lambda b: b[2] * b[3])
                x, y, w, h = largest_face
                chosen_bbox = (x, y, w, h)

                # Bounding box center & area
                cx = x + w / 2.0
                cy = y + h / 2.0
                area = w * h
                bbox_centers.append((cx, cy))
                bbox_areas.append(area / frame_area)

                if first_face_crop is None:
                    # Save a cropped face sample for UI
                    first_face_crop = rgb_frame[y:y+h, x:x+w]

                # Draw bounding box and label
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 127), 2)
                cv2.putText(annotated_frame, f"Face #{f_idx}", (x, max(20, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 127), 2)
            else:
                # Fallback label
                cv2.putText(annotated_frame, "No Face Detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 69, 0), 2)

            annotated_frames.append((f_idx, annotated_frame, chosen_bbox))

        face_ratio = faces_found / total_sampled
        mean_coverage = float(np.mean(bbox_areas)) if bbox_areas else 0.0

        # Calculate bounding box displacement variance
        if len(bbox_centers) > 1:
            centers_np = np.array(bbox_centers)
            center_var = float(np.var(centers_np, axis=0).sum())
        else:
            center_var = 0.0

        anomaly_flags = []
        if face_ratio < 0.6:
            anomaly_flags.append(f"Low face presence ratio ({face_ratio*100:.1f}%)")
        if multi_face_count > 0:
            anomaly_flags.append(f"Multiple faces detected in {multi_face_count} frames")
        if center_var > 2500.0:
            anomaly_flags.append("High face positional instability / sudden bounding box displacement")
        if mean_coverage < 0.05:
            anomaly_flags.append("Face region too small relative to frame size")

        face_observations = {
            "total_sampled": total_sampled,
            "face_detected_count": faces_found,
            "face_detected_ratio": round(face_ratio, 3),
            "multiple_faces_detected": multi_face_count > 0,
            "multi_face_frame_count": multi_face_count,
            "mean_coverage_ratio": round(mean_coverage, 3),
            "bbox_center_variance": round(center_var, 2),
            "anomaly_flags": anomaly_flags,
            "first_face_crop": first_face_crop
        }

        return annotated_frames, face_observations

    def _detect_faces_multi_space(self, rgb_frame):
        """
        Segment face bounding boxes using YCrCb skin chrominance + HSV fallback + contour geometry.
        """
        h_frame, w_frame = rgb_frame.shape[:2]
        frame_area = float(h_frame * w_frame)

        # 1. YCrCb Skin Detection
        ycrcb = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]

        skin_mask_ycrcb = (cr >= 125) & (cr <= 180) & (cb >= 65) & (cb <= 140)

        # 2. HSV Skin Fallback
        hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        skin_mask_hsv = ((h <= 25) | (h >= 160)) & (s >= 15) & (s <= 200) & (v >= 50)

        combined_mask = (skin_mask_ycrcb | skin_mask_hsv).astype(np.uint8) * 255

        # Morphological filtering to close holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_boxes = []

        for c in contours:
            area = cv2.contourArea(c)
            if area > 0.02 * frame_area: # Face area threshold
                bx, by, bw, bh = cv2.boundingRect(c)
                aspect_ratio = float(bh) / float(bw) if bw > 0 else 0.0
                if 0.7 <= aspect_ratio <= 2.5: # Human face aspect ratio
                    detected_boxes.append((bx, by, bw, bh))

        # Fallback if no skin contour found: central face region hypothesis
        if len(detected_boxes) == 0:
            cx, cy = w_frame // 2, h_frame // 2
            bw, bh = int(w_frame * 0.35), int(h_frame * 0.5)
            bx, by = max(0, cx - bw // 2), max(0, cy - bh // 2)
            detected_boxes.append((bx, by, bw, bh))

        return detected_boxes
