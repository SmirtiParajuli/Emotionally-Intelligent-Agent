import os
import numpy as np
import librosa
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from collections import Counter
from datetime import datetime
from scipy.signal import stft
from emotion_modulator import extract_dominant_freq


def extract_features(wav_path, sr=22050, n_mfcc=13):
    """
    Extracts acoustic features from a .wav file for emotion classification.

    Parameters:
        wav_path (str): Path to the audio file.
        sr (int): Sampling rate for loading the audio (default is 22050 Hz).
        n_mfcc (int): Number of MFCCs to extract (default is 13).

    Returns:
        np.ndarray: A 1D array containing the mean MFCCs, ZCR, spectral centroid,
                    duration, and dominant frequency.
    """

    y, sr = librosa.load(wav_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    zcr = librosa.feature.zero_crossing_rate(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)
    dom_freq = extract_dominant_freq(y, sr)

    return np.concatenate([
        np.mean(mfcc, axis=1),
        [np.mean(zcr)],
        [np.mean(centroid)],
        [duration],
        [dom_freq]
    ])

# Extract from All Files
def collect_audio_descriptors(base_dir="assets", output_csv="report/emotion_features.csv"):
    """
    Iterates over .wav files in the specified directory, extracts features,
    and saves them into a CSV file for training.

    Parameters:
        base_dir (str): Directory containing .wav files named by emotion (e.g., happy_1.wav).
        output_csv (str): Path to save the extracted features as a CSV.

    Returns:
        None. Saves output to CSV and prints status.
    """
        
    if not os.path.exists(base_dir):
        print(f"Base directory does not exist: {base_dir}")
        return
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    allowed_emotions = {
        "happy", "sad", "angry", "neutral", "calm", 
        "surprised", "fearful", "disgusted", "nervous"
    }
    rows = []

    for file in os.listdir(base_dir):
        if file.lower().endswith(".wav"):
            emotion = file.split("_")[0].lower()

            if emotion not in allowed_emotions:
                continue

            full_path = os.path.join(base_dir, file)
            try:
                features = extract_features(full_path)
                row = [emotion, file] + features.tolist()
                rows.append(row)
                print(f"Processed: {file} ({emotion})")
            except Exception as e:
                print(f"Skipped {file} due to error: {e}")

    if rows:
        header = ["emotion", "filename"] + [f"feature_{i+1}" for i in range(len(rows[0]) - 2)]
        df = pd.DataFrame(rows, columns=header)
        df.to_csv(output_csv, index=False)
        print(f"Feature data saved to: {output_csv}")
        print("Emotion counts:", df["emotion"].value_counts().to_dict())
    else:
        print(" No features extracted.")

#  Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels, save_path="report/confusion_matrix.png"):
    """
    Plots and saves a labeled confusion matrix as a PNG file.

    Parameters:
        y_true (array-like): Ground truth emotion labels.
        y_pred (array-like): Predicted emotion labels from classifier.
        labels (list): List of class labels to show in the matrix.
        save_path (str): Path to save the plot image.

    Returns:
        None. Saves the confusion matrix figure to disk.
    """

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved to: {save_path}")

#  Train Model
def train_emotion_classifier(csv_path="report/emotion_features.csv"):
    """
    Trains a RandomForestClassifier on the extracted audio features and saves the model.

    Parameters:
        csv_path (str): Path to the CSV file containing extracted features and emotion labels.

    Returns:
        None. Saves the trained model, scaler, and confusion matrix to disk.
    """

    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    class_counts = df["emotion"].value_counts()
    valid_emotions = class_counts[class_counts >= 2].index.tolist()
    df = df[df["emotion"].isin(valid_emotions)]

    if len(valid_emotions) < 2:
        print("Not enough emotion classes with ≥ 2 samples.")
        return

    X = df.drop(columns=["emotion", "filename"]).values
    y = df["emotion"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    test_size_ratio = min(0.3, max(0.2, len(valid_emotions) / len(y)))

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size_ratio, random_state=42, stratify=y
        )
    except ValueError as e:
        print(f"Split Error: {e}")
        return

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=1))

    print(" Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cm_path = f"report/confusion_matrix_{timestamp}.png"
    plot_confusion_matrix(y_test, y_pred, sorted(set(y)), save_path=cm_path)

    joblib.dump(clf, "report/emotion_model.pkl")
    joblib.dump(scaler, "report/scaler.pkl")
    print("Model and scaler saved to: report/")


def predict_emotion(filepath, top_n=3):
    """
    Predicts the emotion from a given .wav file using a trained classifier.

    Parameters:
        filepath (str): Path to the .wav file to analyze.
        top_n (int): Number of top predictions to return (default is 3).

    Returns:
        tuple:
            - top_emotion (str): The highest-probability emotion label.
            - top_preds (list): List of (emotion, confidence %) tuples for top-N predictions.
    """

    # Load model & scaler
    model = joblib.load("report/emotion_model.pkl")
    scaler = joblib.load("report/scaler.pkl")

    # Extract features
    features = extract_features(filepath)
    features_scaled = scaler.transform([features])

    # Predict
    probs = model.predict_proba(features_scaled)[0]
    classes = model.classes_

    # Get top prediction
    top_index = np.argmax(probs)
    top_emotion = classes[top_index]

    # Top-N predictions
    top_indices = probs.argsort()[-top_n:][::-1]
    top_preds = [(classes[i], round(probs[i] * 100, 1)) for i in top_indices]

    return top_emotion, top_preds

