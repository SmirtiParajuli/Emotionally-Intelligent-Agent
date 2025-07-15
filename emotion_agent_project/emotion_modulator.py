import os
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import stft


def extract_dominant_freq(data, sr):
    """
    Computes the most dominant frequency from an audio signal using Short-Time Fourier Transform (STFT).

    Parameters:
        data (np.ndarray): Audio time-series data.
        sr (int): Sampling rate of the audio.

    Returns:
        float: Dominant frequency (Hz) at the temporal midpoint of the audio.
    """
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    f, t, Zxx = stft(data, fs=sr, nperseg=1024, noverlap=512)
    magnitude = np.abs(Zxx)
    top_indices = np.argsort(magnitude, axis=0)[-3:]  # Top 3 freq bands
    dominant_freqs = f[top_indices]
    midpoint = dominant_freqs.shape[1] // 2
    return dominant_freqs[0, midpoint]


def apply_emotion(input_path, output_path, speed_change=1.0, gain_db=0, pitch_shift=0, target_sr=16000):
    """
    Applies pitch shifting, speed modification, and gain adjustment to simulate emotional tone.
    The output is normalized, optionally resampled, and saved to disk.

    Parameters:
        input_path (str): Path to the input neutral .wav file.
        output_path (str): Path to save the modulated emotional .wav file.
        speed_change (float): Factor to change speech tempo (e.g., 1.2 = 20% faster).
        gain_db (float): Gain in decibels (e.g., -5 dB to soften, +6 dB to amplify).
        pitch_shift (int): Number of semitones to shift pitch (positive or negative).
        target_sr (int): Target sampling rate for the output audio (default is 16 kHz).

    Returns:
        None. Saves processed audio file and logs dominant frequency.
    """
    try:
        y, sr = librosa.load(input_path, sr=None)

        # Apply pitch shift
        if pitch_shift != 0:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)

        # Apply time stretch
        if speed_change != 1.0:
            y = librosa.effects.time_stretch(y, rate=speed_change)

        # Apply gain (dB to amplitude scale)
        y *= 10 ** (gain_db / 20)

        # Normalize to [-1, 1]
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val

        # Resample if needed
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, y, sr)

        freq = extract_dominant_freq(y, sr)
        print(f" Saved: {output_path} | 🎯 Dominant Frequency: {freq:.2f} Hz")

    except Exception as e:
        print(f" Failed to process {input_path}: {e}")


def generate_emotional_speech(input_path="assets/neutral.wav"):
    """
    This function creates emotional variants of a neutral speech sample by applying
    pitch shifts, gain adjustments, and tempo modifications corresponding to common
    acoustic features found in human emotional expression.

    Emotion profiles (e.g., "happy", "sad", "angry") are manually defined using
    heuristic values for speed, pitch, and loudness based on psychological and
    linguistic research. For example:
    - "happy" speech is generally faster, louder, and higher pitched,
    - "sad" speech is typically slower, softer, and lower pitched,
    - "angry" tends to be loud and sharp with increased pitch and speed.

    Each emotion is rendered into 3 slightly randomized audio variants to increase
    diversity and robustness for both emotion classification and audio feedback.

    These parameter values were chosen for balance between perceptual clarity and
    realism, allowing the agent to express emotions audibly in a lightweight and
    controllable manner.
    """
    emotions = {
        "happy":     {"speed": 1.2, "gain": 6,  "pitch_shift": 2},
        "sad":       {"speed": 0.9, "gain": -6, "pitch_shift": -2},
        "angry":     {"speed": 1.3, "gain": 10, "pitch_shift": 3},
        "calm":      {"speed": 0.95, "gain": -3, "pitch_shift": -1},
        "surprised": {"speed": 1.5, "gain": 5,  "pitch_shift": 5},
        "fearful":   {"speed": 0.8, "gain": -5, "pitch_shift": -4},
        "disgusted": {"speed": 0.85, "gain": -3, "pitch_shift": -1},
        "nervous":   {"speed": 1.2, "gain": 2,  "pitch_shift": 1}
    }

    print("🎵 Generating emotional speech variants...")

    for i in range(3):  # Generate 3 variants per emotion
        for emotion, params in emotions.items():
            # Slight randomization for variation
            speed = params["speed"] + np.random.uniform(-0.03, 0.03)
            gain = params["gain"] + np.random.uniform(-1, 1)
            pitch = params["pitch_shift"] + np.random.randint(-1, 2)

            output_path = f"assets/{emotion}_{i + 1}.wav"
            apply_emotion(
                input_path=input_path,
                output_path=output_path,
                speed_change=speed,
                gain_db=gain,
                pitch_shift=pitch
            )
