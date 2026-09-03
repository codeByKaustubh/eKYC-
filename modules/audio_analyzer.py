import numpy as np
from scipy.fft import fft, dct

class AudioAnalyzer:
    """
    Speech Anti-Spoofing & Voice Clone Detection Module.
    Extracts:
    - Mel-Frequency Cepstral Coefficients (MFCCs)
    - Spectral Centroid, Rolloff, Zero Crossing Rate (ZCR)
    - Pitch Variance & Vocoder Spectral Flatness (Synthetic Speech Detection)
    """

    def analyze_audio(self, sample_rate, signal):
        """
        Analyzes audio signal array [-1.0, 1.0].
        Returns:
            audio_results: dict containing risk scores, MFCC matrix for visualization, and indicators
        """
        if signal is None or len(signal) < 1000 or sample_rate <= 0:
            return self._empty_response("No audio signal provided or duration too short")

        # Frame parameters (25ms frame, 10ms step)
        frame_length = int(sample_rate * 0.025)
        frame_step = int(sample_rate * 0.010)

        num_frames = max(1, (len(signal) - frame_length) // frame_step + 1)
        
        mfcc_matrix = []
        zcr_series = []
        rms_series = []
        spectral_centroids = []
        spectral_flatness_series = []

        # Pre-emphasis filter
        pre_emphasized = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
        hamming = np.hamming(frame_length)

        # Create Mel Filter Bank
        n_filt = 26
        nfft = 512
        mel_filters = self._get_mel_filterbank(n_filt, nfft, sample_rate)

        for i in range(num_frames):
            start = i * frame_step
            end = start + frame_length
            if end > len(pre_emphasized):
                break
            
            frame = pre_emphasized[start:end] * hamming
            
            # 1. RMS Energy
            rms = np.sqrt(np.mean(frame**2) + 1e-8)
            rms_series.append(float(rms))

            # 2. Zero Crossing Rate (ZCR)
            zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2.0 * frame_length)
            zcr_series.append(float(zcr))

            # 3. FFT Magnitude Spectrum
            mag_spec = np.abs(fft(frame, nfft)[:nfft // 2 + 1])
            pow_spec = (mag_spec**2) / float(nfft)

            # 4. Spectral Centroid
            freqs = np.linspace(0, sample_rate / 2, len(pow_spec))
            centroid = np.sum(freqs * pow_spec) / (np.sum(pow_spec) + 1e-8)
            spectral_centroids.append(float(centroid))

            # 5. Spectral Flatness (Geometric Mean / Arithmetic Mean)
            geom_mean = np.exp(np.mean(np.log(pow_spec + 1e-8)))
            arith_mean = np.mean(pow_spec) + 1e-8
            flatness = geom_mean / arith_mean
            spectral_flatness_series.append(float(flatness))

            # 6. MFCC calculation
            filter_banks = np.dot(pow_spec, mel_filters.T)
            filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
            filter_banks = 20 * np.log10(filter_banks) # Log mel energy
            
            # DCT to get 13 MFCCs
            mfccs = dct(filter_banks, type=2, axis=-1, norm='ortho')[:13]
            mfcc_matrix.append(mfccs)

        mfcc_matrix = np.array(mfcc_matrix) if mfcc_matrix else np.zeros((1, 13))

        # Pitch contour approximation via autocorrelation peak
        pitch_series = self._estimate_pitch_contour(signal, sample_rate, frame_length, frame_step)
        pitch_variance = float(np.std(pitch_series)) if len(pitch_series) > 0 else 0.0

        mean_zcr = float(np.mean(zcr_series)) if zcr_series else 0.0
        mean_centroid = float(np.mean(spectral_centroids)) if spectral_centroids else 0.0
        mean_flatness = float(np.mean(spectral_flatness_series)) if spectral_flatness_series else 0.0

        # Voice Spoofing / Synthetic Voice Risk Assessment
        risk_factors = []

        # Risk 1: Flat/Robotic Pitch (AI TTS / Voice Cloning)
        if pitch_variance < 5.0 and len(pitch_series) > 10:
            pitch_risk = 80.0
            risk_factors.append(f"Robotic pitch constancy / unnatural pitch variance ({pitch_variance:.2f} Hz) - Voice clone risk")
        elif pitch_variance < 12.0:
            pitch_risk = 50.0
            risk_factors.append("Low pitch dynamic range in vocal track")
        else:
            pitch_risk = 10.0

        # Risk 2: Vocoder Spectral Flatness Anomaly
        if mean_flatness > 0.35:
            flatness_risk = 75.0
            risk_factors.append(f"High spectral noise / vocoder phase artifact (Spectral flatness: {mean_flatness:.3f})")
        elif mean_flatness < 0.005:
            flatness_risk = 60.0
            risk_factors.append("Overly clean / studio-synthesized speech spectrum")
        else:
            flatness_risk = 15.0

        # Risk 3: Abnormal Spectral Centroid Shift
        if mean_centroid > (sample_rate * 0.35) or mean_centroid < 300.0:
            centroid_risk = 65.0
            risk_factors.append(f"Unnatural speech frequency distribution (Spectral centroid: {mean_centroid:.1f} Hz)")
        else:
            centroid_risk = 10.0

        overall_audio_risk = (0.45 * pitch_risk + 0.35 * flatness_risk + 0.20 * centroid_risk)

        return {
            "audio_risk_score": round(overall_audio_risk, 1),
            "sample_rate": sample_rate,
            "duration_sec": round(len(signal) / float(sample_rate), 2),
            "mean_zcr": round(mean_zcr, 4),
            "mean_spectral_centroid": round(mean_centroid, 1),
            "mean_spectral_flatness": round(mean_flatness, 4),
            "pitch_variance_hz": round(pitch_variance, 2),
            "rms_series": [round(r, 4) for r in rms_series],
            "pitch_series": [round(p, 1) for p in pitch_series],
            "mfcc_matrix": mfcc_matrix, # numpy array (frames, 13)
            "sub_risks": {
                "pitch_monotone_clone": round(pitch_risk, 1),
                "vocoder_spectral_flatness": round(flatness_risk, 1),
                "spectral_centroid_shift": round(centroid_risk, 1)
            },
            "risk_factors": risk_factors
        }

    def _get_mel_filterbank(self, n_filt, nfft, sample_rate):
        low_freq_mel = 0
        high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2.0) / 700.0)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filt + 2)
        hz_points = 700 * (10**(mel_points / 2595.0) - 1)
        bin_points = np.floor((nfft + 1) * hz_points / sample_rate).astype(int)

        fbank = np.zeros((n_filt, int(np.floor(nfft / 2 + 1))))
        for m in range(1, n_filt + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin_points[m - 1]) / float(f_m - bin_points[m - 1] + 1e-8)
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin_points[m + 1] - k) / float(bin_points[m + 1] - bin_points[m] + 1e-8)
        return fbank

    def _estimate_pitch_contour(self, signal, sr, frame_length, frame_step):
        pitches = []
        num_frames = max(1, (len(signal) - frame_length) // frame_step + 1)
        min_lag = int(sr / 400.0) # Max pitch 400 Hz
        max_lag = int(sr / 60.0)  # Min pitch 60 Hz

        for i in range(0, num_frames, 2): # Sub-sample frames
            start = i * frame_step
            end = start + frame_length
            if end > len(signal):
                break
            frame = signal[start:end]
            if np.std(frame) < 0.01:
                continue # Unvoiced / Silence
            
            # Autocorrelation
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            if len(autocorr) > max_lag:
                peak_idx = min_lag + np.argmax(autocorr[min_lag:max_lag])
                if autocorr[peak_idx] > 0.3 * autocorr[0]:
                    pitch = sr / float(peak_idx)
                    pitches.append(pitch)
        return pitches

    def _empty_response(self, reason):
        return {
            "audio_risk_score": 50.0,
            "sample_rate": 0,
            "duration_sec": 0.0,
            "mean_zcr": 0.0,
            "mean_spectral_centroid": 0.0,
            "mean_spectral_flatness": 0.0,
            "pitch_variance_hz": 0.0,
            "rms_series": [],
            "pitch_series": [],
            "mfcc_matrix": np.zeros((1, 13)),
            "sub_risks": {
                "pitch_monotone_clone": 50.0,
                "vocoder_spectral_flatness": 50.0,
                "spectral_centroid_shift": 50.0
            },
            "risk_factors": [reason]
        }
