import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Apply a clean and professional style
plt.style.use("seaborn-v0_8-muted")

def plot_waveform(y, sr, emotion, folder):
    """
    Plot and save the time-domain waveform of the audio signal.

    Parameters:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of the audio.
        emotion (str): Emotion label (used for file naming).
        folder (str): Directory to save the plot image.

    Returns:
        None
    """
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"Waveform - {emotion}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{emotion}_waveform.png"))
    plt.close()

def plot_fft(y, sr, emotion, folder):
    """
    Compute and plot the average FFT magnitude spectrum with peak frequency marker.

    Parameters:
        y (np.ndarray): Audio signal.
        sr (int): Sampling rate.
        emotion (str): Emotion label for naming output.
        folder (str): Output directory for saving plots.

    Returns:
        None
    """
    D = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    spectrum = np.mean(D, axis=1)
    peak_idx = np.argmax(spectrum)
    peak_freq = freqs[peak_idx]

    plt.figure(figsize=(10, 4))
    plt.plot(freqs[:len(spectrum)], spectrum, label="FFT Spectrum")
    plt.axvline(x=peak_freq, color='red', linestyle='--', label=f"Peak: {peak_freq:.2f} Hz")
    plt.title(f"FFT Magnitude Spectrum - {emotion}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim([0, 5000])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{emotion}_fft.png"))
    plt.close()

def plot_mfcc(y, sr, emotion, folder):
    """
    Generate and save a heatmap of 13 Mel-frequency cepstral coefficients (MFCCs).

    Parameters:
        y (np.ndarray): Audio waveform.
        sr (int): Sampling rate.
        emotion (str): Emotion label for filename.
        folder (str): Directory to save the MFCC plot.

    Returns:
        None
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    plt.figure(figsize=(10, 4))
    img = librosa.display.specshow(mfccs, x_axis='time', sr=sr)
    plt.title(f"MFCCs - {emotion}")
    plt.xlabel("Time (s)")
    plt.ylabel("MFCC Coefficients")
    plt.colorbar(img)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{emotion}_mfcc.png"))
    plt.close()

def plot_energy_zcr(y, sr, emotion, folder):
    """
    Plot RMS energy and Zero-Crossing Rate (ZCR) over time in a single chart.

    Parameters:
        y (np.ndarray): Audio signal.
        sr (int): Sampling rate.
        emotion (str): Emotion label.
        folder (str): Directory to store the image.

    Returns:
        None
    """
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    frames = range(len(rms))
    times = librosa.frames_to_time(frames, sr=sr)

    plt.figure(figsize=(10, 4))
    plt.plot(times, rms, label="RMS Energy")
    plt.plot(times, zcr, label="Zero-Crossing Rate", alpha=0.75)
    plt.title(f"RMS Energy & ZCR - {emotion}")
    plt.xlabel("Time (s)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{emotion}_energy_zcr.png"))
    plt.close()

def plot_spectrogram(y, sr, emotion, folder):
    """
    Create and save a log-scaled spectrogram (dB) to visualize frequency content over time.

    Parameters:
        y (np.ndarray): Audio waveform.
        sr (int): Sampling rate.
        emotion (str): Emotion name.
        folder (str): Directory to save the image.

    Returns:
        None
    """
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(10, 4))
    img = librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log')
    plt.title(f"Spectrogram (log scale) - {emotion}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(img, format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{emotion}_spectrogram.png"))
    plt.close()

def plot_all_features(audio_path, emotion_name="emotion"):
    """
    Generate and save all key acoustic visualizations for a given .wav file.

    Visuals include:
        - Waveform
        - FFT spectrum
        - MFCCs
        - RMS Energy & ZCR
        - Log-scaled spectrogram

    Parameters:
        audio_path (str): Path to the .wav audio file.
        emotion_name (str): Label used for saving and organizing plots.

    Returns:
        None. All plots are saved in report/<emotion_name>/.
    """
    y, sr = librosa.load(audio_path, sr=None)
    folder = os.path.join("report", emotion_name)
    os.makedirs(folder, exist_ok=True)

    plot_waveform(y, sr, emotion_name, folder)
    plot_fft(y, sr, emotion_name, folder)
    plot_mfcc(y, sr, emotion_name, folder)
    plot_energy_zcr(y, sr, emotion_name, folder)
    plot_spectrogram(y, sr, emotion_name, folder)

    print(f"✅ All plots saved to: {folder}")
