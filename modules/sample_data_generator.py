import cv2
import numpy as np
import os
import wave
import struct

class SampleDataGenerator:
    """
    Utility to create synthetic demo videos, audio files, and test benchmarks
    for out-of-the-box system execution and metric evaluation.
    """
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.genuine_dir = os.path.join(data_dir, "genuine")
        self.deepfake_dir = os.path.join(data_dir, "deepfake")
        self.audio_dir = os.path.join(data_dir, "audio")
        
        for d in [self.genuine_dir, self.deepfake_dir, self.audio_dir]:
            os.makedirs(d, exist_ok=True)

    def generate_all_samples(self):
        """
        Creates synthetic demo media files if they don't already exist.
        Returns paths dict.
        """
        genuine_video_path = os.path.join(self.genuine_dir, "sample_genuine.mp4")
        genuine_audio_path = os.path.join(self.genuine_dir, "sample_genuine.wav")

        replay_video_path = os.path.join(self.deepfake_dir, "sample_screen_replay.mp4")
        deepfake_video_path = os.path.join(self.deepfake_dir, "sample_deepfake_swap.mp4")
        clone_audio_path = os.path.join(self.audio_dir, "sample_voice_clone.wav")

        if not os.path.exists(genuine_video_path):
            self._create_synthetic_video(genuine_video_path, attack_type="genuine")
        if not os.path.exists(genuine_audio_path):
            self._create_synthetic_audio(genuine_audio_path, voice_type="genuine")

        if not os.path.exists(replay_video_path):
            self._create_synthetic_video(replay_video_path, attack_type="screen_replay")
        if not os.path.exists(deepfake_video_path):
            self._create_synthetic_video(deepfake_video_path, attack_type="deepfake_swap")
        if not os.path.exists(clone_audio_path):
            self._create_synthetic_audio(clone_audio_path, voice_type="voice_clone")

        return {
            "genuine_video": genuine_video_path,
            "genuine_audio": genuine_audio_path,
            "replay_video": replay_video_path,
            "deepfake_video": deepfake_video_path,
            "clone_audio": clone_audio_path
        }

    def generate_benchmark_suite(self, num_samples=30):
        """
        Generates ground-truth labeled benchmark dataset metadata & feature scores
        for batch evaluation tab (SDFVD-style benchmark evaluation).
        """
        np.random.seed(42)
        benchmark_items = []

        half = num_samples // 2

        # 1. Genuine samples (Label 0)
        for i in range(half):
            # Low risks
            face_r = np.random.uniform(0, 15)
            live_r = np.random.uniform(5, 25)
            df_r = np.random.uniform(5, 20)
            aud_r = np.random.uniform(5, 20)
            sync_r = np.random.uniform(5, 25)

            tot_r = 0.15 * face_r + 0.25 * live_r + 0.25 * df_r + 0.20 * aud_r + 0.15 * sync_r
            # Add small noise
            tot_r = max(5.0, min(34.0, tot_r + np.random.normal(0, 3)))

            benchmark_items.append({
                "sample_id": f"GEN_SDFVD_{i+1:03d}",
                "label": 0, # 0 = Bona Fide
                "label_name": "Genuine",
                "risk_score": round(tot_r, 1),
                "liveness_score": round(live_r, 1),
                "deepfake_score": round(df_r, 1),
                "audio_score": round(aud_r, 1)
            })

        # 2. Attack samples (Label 1)
        for i in range(num_samples - half):
            # High risks
            face_r = np.random.uniform(20, 80)
            live_r = np.random.uniform(60, 95)
            df_r = np.random.uniform(65, 98)
            aud_r = np.random.uniform(50, 90)
            sync_r = np.random.uniform(55, 95)

            tot_r = 0.15 * face_r + 0.25 * live_r + 0.25 * df_r + 0.20 * aud_r + 0.15 * sync_r
            tot_r = max(40.0, min(98.0, tot_r + np.random.normal(0, 4)))

            benchmark_items.append({
                "sample_id": f"SPOOF_SDFVD_{i+1:03d}",
                "label": 1, # 1 = Attack / Presentation Attack
                "label_name": "Presentation Attack",
                "risk_score": round(tot_r, 1),
                "liveness_score": round(live_r, 1),
                "deepfake_score": round(df_r, 1),
                "audio_score": round(aud_r, 1)
            })

        return benchmark_items

    def _create_synthetic_video(self, output_path, attack_type="genuine", num_frames=60, width=640, height=480):
        """
        Renders synthetic video frames containing a stylized face drawing with motion & artifacts.
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

        cx, cy = width // 2, height // 2

        for i in range(num_frames):
            frame = np.ones((height, width, 3), dtype=np.uint8) * 240 # Light background

            # Movement physics
            dx = int(12 * np.sin(i * 0.15))
            dy = int(8 * np.cos(i * 0.10))

            face_x = cx + dx
            face_y = cy + dy

            if attack_type == "screen_replay":
                # Add high-frequency Moire screen grid lines
                grid = (np.sin(np.arange(height)[:, None] * 0.8) * np.cos(np.arange(width)[None, :] * 0.8) * 40).astype(np.int16)
                frame = np.clip(frame.astype(np.int16) + grid[:, :, None], 0, 255).astype(np.uint8)

            # Draw Face Oval
            cv2.ellipse(frame, (face_x, face_y), (90, 120), 0, 0, 360, (210, 180, 140), -1) # Skin
            cv2.ellipse(frame, (face_x, face_y), (90, 120), 0, 0, 360, (100, 70, 40), 2)

            # Draw Eyes (Blinking on frames 20-24 and 45-48 if genuine)
            is_blink = (20 <= i <= 24 or 45 <= i <= 48) if attack_type == "genuine" else False
            eye_h = 2 if is_blink else 12

            cv2.ellipse(frame, (face_x - 35, face_y - 25), (18, eye_h), 0, 0, 360, (50, 50, 50), -1)
            cv2.ellipse(frame, (face_x + 35, face_y - 25), (18, eye_h), 0, 0, 360, (50, 50, 50), -1)

            # Draw Nose
            cv2.line(frame, (face_x, face_y - 10), (face_x - 5, face_y + 20), (100, 70, 40), 2)
            cv2.line(frame, (face_x - 5, face_y + 20), (face_x + 8, face_y + 20), (100, 70, 40), 2)

            # Draw Mouth (Animates height if genuine)
            mouth_open = int(8 + 6 * np.sin(i * 0.25))
            cv2.ellipse(frame, (face_x, face_y + 50), (25, mouth_open), 0, 0, 360, (180, 50, 50), -1)

            if attack_type == "deepfake_swap":
                # Add harsh face boundary blend ring & temporal noise
                cv2.ellipse(frame, (face_x, face_y), (92, 122), 0, 0, 360, (0, 0, 255), 4)
                if i % 3 == 0:
                    noise = np.random.randint(-30, 30, (height, width, 3), dtype=np.int16)
                    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            out.write(frame)

        out.release()

    def _create_synthetic_audio(self, output_path, voice_type="genuine", duration_sec=3.0, sr=16000):
        """
        Synthesizes WAV speech audio signal.
        Genuine: Dynamic formant frequency & natural amplitude modulation.
        Voice Clone: Monotone static sine pitch & high-frequency vocoder noise.
        """
        num_samples = int(duration_sec * sr)
        t = np.linspace(0, duration_sec, num_samples, False)

        if voice_type == "genuine":
            # Pitch modulation (120 Hz to 180 Hz sweep)
            pitch_freq = 140.0 + 35.0 * np.sin(2 * np.pi * 1.5 * t)
            phase = 2 * np.pi * np.cumsum(pitch_freq) / sr
            signal = 0.5 * np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
            # Speech envelope modulation
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2.5 * t)
            signal = signal * envelope
        else: # Voice clone
            # Monotone rigid pitch 150.0 Hz + high vocoder noise
            phase = 2 * np.pi * 150.0 * t
            signal = 0.6 * np.sin(phase)
            # High-frequency vocoder phase noise
            noise = np.random.normal(0, 0.15, num_samples)
            signal = signal + noise

        # Normalize to 16-bit PCM integer range
        signal = signal / (np.max(np.abs(signal)) + 1e-8) * 30000.0
        signal_int16 = signal.astype(np.int16)

        with wave.open(output_path, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            for sample in signal_int16:
                wf.writeframes(struct.pack('<h', sample))
