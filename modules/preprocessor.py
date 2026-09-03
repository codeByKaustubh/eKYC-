import cv2
import numpy as np
import os
import wave
from scipy.io import wavfile

class VideoPreprocessor:
    """
    Handles video & audio file loading, metadata extraction, frame sampling,
    and conversion into standard structures for multimodal eKYC evaluation.
    """
    def __init__(self, max_frames=60):
        self.max_frames = max_frames

    def process_video(self, video_path):
        """
        Reads video from path, extracts metadata, and samples up to `max_frames`.
        Returns:
            frames: list of numpy arrays (RGB)
            metadata: dict with fps, frame_count, duration, width, height
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if total_frames > 0 else 0.0

        # Uniform frame sampling indices
        if total_frames > self.max_frames and total_frames > 0:
            sample_indices = set(np.linspace(0, total_frames - 1, self.max_frames, dtype=int))
        else:
            sample_indices = None

        frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if sample_indices is None or frame_idx in sample_indices:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((frame_idx, rgb_frame))
            
            frame_idx += 1

        cap.release()

        metadata = {
            "fps": fps,
            "total_frames": total_frames,
            "sampled_frames_count": len(frames),
            "width": width,
            "height": height,
            "duration_sec": round(duration, 2),
            "video_path": video_path
        }

        return frames, metadata

    def process_audio(self, audio_path):
        """
        Loads WAV audio file using scipy.io.wavfile or std wave.
        Returns:
            sample_rate: int
            audio_signal: numpy array (1D float normalized [-1, 1])
            audio_meta: dict
        """
        if not audio_path or not os.path.exists(audio_path):
            return None, None, {"status": "No audio file provided or found"}

        try:
            sr, signal = wavfile.read(audio_path)
            # Handle multi-channel audio
            if signal.ndim > 1:
                signal = signal.mean(axis=1)

            # Convert integer types to float [-1.0, 1.0]
            if signal.dtype == np.int16:
                signal = signal.astype(np.float32) / 32768.0
            elif signal.dtype == np.int32:
                signal = signal.astype(np.float32) / 2147483648.0
            elif signal.dtype == np.uint8:
                signal = (signal.astype(np.float32) - 128.0) / 128.0
            else:
                signal = signal.astype(np.float32)
                max_val = np.max(np.abs(signal))
                if max_val > 0:
                    signal = signal / max_val

            duration = len(signal) / float(sr) if sr > 0 else 0.0

            audio_meta = {
                "sample_rate": sr,
                "num_samples": len(signal),
                "duration_sec": round(duration, 2),
                "audio_path": audio_path,
                "status": "Success"
            }
            return sr, signal, audio_meta

        except Exception as e:
            return None, None, {"status": f"Audio processing error: {str(e)}"}
