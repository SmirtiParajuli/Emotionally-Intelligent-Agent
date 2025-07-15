from PyQt5 import QtWidgets, QtGui, QtCore, QtMultimedia
import sys, os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
import noisereduce as nr
from tts_generator import generate_neutral_speech
from emotion_modulator import generate_emotional_speech, extract_dominant_freq
from emotion_classifier import predict_emotion, collect_audio_descriptors, train_emotion_classifier
from agent_response import generate_agent_response
from feature_visualizer import plot_all_features  
import shutil

class EmotionAgentGUI(QtWidgets.QWidget):
    def __init__(self):
        """
        Initialize the Emotion Agent GUI window and configure directories.
        """
        super().__init__()
        self.setWindowTitle("Emotion Agent - Emotionally-Sensitive Interface")
        self.setGeometry(100, 100, 950, 880)
        self.setup_directories()
        self.initUI()

    def setup_directories(self):
        """
        Create necessary directories (assets and report) if they do not exist.
        """
        os.makedirs("report", exist_ok=True)
        os.makedirs("assets", exist_ok=True)

    def initUI(self):
        """
        Initialize the GUI layout and prepare stacked views for main and recording pages.
        """
        self.stack = QtWidgets.QStackedLayout()
        self.main_page = QtWidgets.QWidget()
        self.record_page = QtWidgets.QWidget()

        self.setup_main_page()
        self.setup_record_page()

        self.stack.addWidget(self.main_page)
        self.stack.addWidget(self.record_page)

        wrapper_layout = QtWidgets.QVBoxLayout()
        wrapper_layout.addLayout(self.stack)
        self.setLayout(wrapper_layout)

    def setup_main_page(self):
        """
        Set up the main interaction page with inputs, buttons, dropdowns, and status elements.
        """
        layout = QtWidgets.QVBoxLayout(self.main_page)
        font = QtGui.QFont("Arial", 10)

        self.clear_btn = QtWidgets.QPushButton("🪑 Clear Everything")
        self.clear_btn.setStyleSheet("background-color: #f8d7da; color: #a94442;")
        self.clear_btn.clicked.connect(self.clear_all)

        self.instructions = QtWidgets.QLabel("\ud83c\udfa4 Enter a sentence or record your voice to analyze and explore emotional audio.")
        self.instructions.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Bold))
        self.instructions.setStyleSheet("color: #333;")

        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(self.instructions)
        top_row.addStretch()
        top_row.addWidget(self.clear_btn)
        layout.addLayout(top_row)

        self.text_input = QtWidgets.QLineEdit()
        self.text_input.setPlaceholderText("Type something like 'I have something to tell you.'")
        self.text_input.setFont(font)
        layout.addWidget(self.text_input)

        self.generate_btn = QtWidgets.QPushButton("🚀 Generate & Train from Input")
        self.generate_btn.clicked.connect(self.full_pipeline)
        layout.addWidget(self.generate_btn)

        self.emotion_dropdown = QtWidgets.QComboBox()
        layout.addWidget(self.emotion_dropdown)

        self.play_selected_btn = QtWidgets.QPushButton("🎧 Play Selected Emotion")
        self.play_selected_btn.clicked.connect(self.play_selected_emotion)
        layout.addWidget(self.play_selected_btn)

        file_row = QtWidgets.QHBoxLayout()
        self.file_input = QtWidgets.QLineEdit()
        file_row.addWidget(self.file_input)
        self.browse_btn = QtWidgets.QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_file)
        file_row.addWidget(self.browse_btn)
        layout.addLayout(file_row)

        self.predict_btn = QtWidgets.QPushButton("🔍 Predict Emotion")
        self.predict_btn.clicked.connect(self.predict_emotion_action)
        layout.addWidget(self.predict_btn)

        self.result_label = QtWidgets.QLabel("🎯 Predicted Emotion: ")
        self.freq_label = QtWidgets.QLabel("📊 Dominant Frequency: ")
        self.agent_response = QtWidgets.QLabel("🤖 Agent Response: ")
        self.top_predictions_label = QtWidgets.QLabel("📋 Top Predictions: ")
        self.top_predictions_label.setToolTip("These are the most likely emotions ranked by confidence.")

        for lbl in [self.result_label, self.freq_label, self.agent_response, self.top_predictions_label]:
            lbl.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
            lbl.setStyleSheet("color: #003366; padding: 4px;")
            layout.addWidget(lbl)

        self.play_audio_btn = QtWidgets.QPushButton("🎧 Play Audio")
        self.play_audio_btn.clicked.connect(self.play_audio)
        layout.addWidget(self.play_audio_btn)

        self.canvas = QtWidgets.QLabel()
        layout.addWidget(self.canvas)

        self.voice_btn = QtWidgets.QPushButton("\ud83c\udfa4 Record From Voice")
        self.voice_btn.clicked.connect(self.switch_to_record_page)
        layout.addWidget(self.voice_btn)
       
        self.audio_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.audio_slider.setRange(0, 100)
        self.audio_slider.setEnabled(False)
        layout.addWidget(self.audio_slider)

        # 🖱️ Allow user to seek manually
        self.audio_slider.sliderReleased.connect(
            lambda: self.player.setPosition(self.audio_slider.value())
        )

        # 🎵 Media player
        self.player = QtMultimedia.QMediaPlayer()
        self.player.setVolume(100)
        self.player.positionChanged.connect(self.update_audio_slider)
        self.player.durationChanged.connect(lambda d: self.audio_slider.setRange(0, d))

        self.status_bar = QtWidgets.QLabel("Ready.")
        self.status_bar.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.status_bar)
        
        # 📊 Scrollable area for multi-graph visualizations
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QtWidgets.QWidget()
        self.scroll_layout = QtWidgets.QVBoxLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)
        layout.addWidget(self.scroll_area)

        # 📎 Footer label
        footer = QtWidgets.QLabel("Emotion Agent • NLP303 Assignment • Smriti 2025")
        footer.setStyleSheet("color: gray; font-size: 10px; padding-top: 6px; opacity: 0.7;")
        footer.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(footer)




    def update_audio_slider(self, position):
        """
        Update the slider to match the current playback position of the audio.
        """
        self.audio_slider.setValue(position)


    def update_record_timer(self):
        """
        Display recording duration incrementally during voice capture.
        """
        self.record_seconds += 1
        self.countdown_label.setText(f"🔴 Recording... {self.record_seconds}s")


    def setup_record_page(self):
        """
        Set up the recording interface with start/stop controls and navigation.
        """
        layout = QtWidgets.QVBoxLayout(self.record_page)

        self.countdown_label = QtWidgets.QLabel("🎤 Ready to record")
        self.countdown_label.setFont(QtGui.QFont("Arial", 18, QtGui.QFont.Bold))
        self.countdown_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.countdown_label)

        btn_row = QtWidgets.QHBoxLayout()

        self.record_btn = QtWidgets.QPushButton("⏺️ Start Recording")
        self.record_btn.clicked.connect(self.start_recording)
        btn_row.addWidget(self.record_btn)

        self.stop_btn = QtWidgets.QPushButton("⏹️ Stop Recording & Predict")
        self.stop_btn.setDisabled(True)
        self.stop_btn.clicked.connect(self.stop_recording)
        btn_row.addWidget(self.stop_btn)

        layout.addLayout(btn_row)

        self.back_btn = QtWidgets.QPushButton("🔙 Back to Main Page")
        self.back_btn.clicked.connect(self.switch_to_main_page)
        layout.addWidget(self.back_btn)

    def switch_to_record_page(self):
        """
        Navigate to the recording page and initiate countdown before recording.
        """
        self.stack.setCurrentWidget(self.record_page)
        self.countdown = 3
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_countdown)
        self.timer.start(1000)

    def update_countdown(self):
        """
        Update and display the countdown timer during the transition to recording.
        """
        if self.countdown > 0:
            self.countdown_label.setText(f"⏳ Recording starts in {self.countdown}...")
            self.countdown -= 1
        else:
            self.timer.stop()
            self.countdown_label.setText("\ud83c\udfa4 Recording...")
            QtCore.QTimer.singleShot(100, self.start_recording)

    def switch_to_main_page(self):
        """
        Return from the recording interface to the main GUI page.
        """
        self.stack.setCurrentWidget(self.main_page)

    def start_recording(self):
        """
        Begin voice recording using microphone input.
        """
        self.countdown_label.setText("🔴 Recording...")
        self.status_bar.setText("🔴 Recording in progress")
        self.record_btn.setDisabled(True)
        self.stop_btn.setEnabled(True)

        self.duration = 5
        self.sample_rate = 16000
        self.audio = sd.rec(int(self.duration * self.sample_rate), samplerate=self.sample_rate, channels=1)



    def stop_recording(self):
        """
        Stop recording, denoise and trim audio, perform speech recognition, and predict emotion.
        """
        try:
            sd.stop()
            self.stop_btn.setDisabled(True)
            self.record_btn.setEnabled(True)

            if hasattr(self, 'live_timer'):
                self.live_timer.stop()

            filename = "assets/voice_input.wav"
            sf.write(filename, self.audio, self.sample_rate, subtype='PCM_16')

            #  Load raw audio (avoid overwriting `sr`)
            y, sample_rate = librosa.load(filename, sr=None)

            #  Apply noise reduction
            y_denoised = nr.reduce_noise(y=y, sr=sample_rate)

            #  Trim silence
            y_trimmed, _ = librosa.effects.trim(y_denoised, top_db=25)

            #Save cleaned audio back
            sf.write(filename, y_trimmed, sample_rate, subtype='PCM_16')

            # 🎙️ Speech recognition (no conflict with `sr`)
            recognizer = sr.Recognizer()
            with sr.AudioFile(filename) as source:
                audio_data = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio_data)
                    self.text_input.setText(text)
                except sr.UnknownValueError:
                    self.text_input.setText("⚠️ Could not recognize speech")

            self.countdown_label.setText("✅ Recording Complete. Ready to Predict.")
            self.switch_to_main_page()
            self.status_bar.setText("✅ Voice cleaned and saved. Ready to predict.")

            self.predict_emotion_action()

        except Exception as e:
            self.result_label.setStyleSheet("color: red;")
            self.result_label.setText(f"❌ Error in stop_recording: {e}")




    def full_pipeline(self):
        """
        Generate neutral and emotional speech from text, extract features, and train classifier.
        """
        text = self.text_input.text().strip() or "I have something to tell you."
        try:
            self.result_label.setText("⚙️ Resetting...")
            QtWidgets.QApplication.processEvents()

            for file in os.listdir("assets"):
                if file.endswith(".wav"):
                    os.remove(os.path.join("assets", file))
            for file in ["report/emotion_model.pkl", "report/scaler.pkl", "report/emotion_features.csv"]:
                if os.path.exists(file):
                    os.remove(file)

            self.result_label.setText("🗣️ Generating neutral speech...")
            QtWidgets.QApplication.processEvents()
            generate_neutral_speech(text=text, output_path="assets/neutral.wav")

            self.result_label.setText("🎭 Generating emotional variants...")
            QtWidgets.QApplication.processEvents()
            generate_emotional_speech("assets/neutral.wav")

            self.result_label.setText("📊 Extracting features...")
            QtWidgets.QApplication.processEvents()
            collect_audio_descriptors()

            self.result_label.setText("🧠 Training model...")
            QtWidgets.QApplication.processEvents()
            train_emotion_classifier()

            self.refresh_emotion_dropdown()
            self.result_label.setStyleSheet("color: green;")
            self.result_label.setText("✅ Model trained and ready!")
            self.status_bar.setText("✅ Model trained and ready!")


        except Exception as e:
            self.result_label.setStyleSheet("color: red;")
            self.result_label.setText(f"Error: {str(e)}")

    def clear_all(self):
        """
        Reset the interface by clearing all inputs, outputs, visualizations, and generated files.
        """
        reply = QtWidgets.QMessageBox.question(
            self, "Confirm Reset",
            "This will delete all generated files and reset the interface. Continue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            for folder in ["assets", "report"]:
                for file in os.listdir(folder):
                    if file.endswith((".wav", ".pkl", ".csv", ".png")):
                        os.remove(os.path.join(folder, file))

            self.text_input.clear()
            self.file_input.clear()
            self.result_label.setText("🎯 Predicted Emotion: ")
            self.freq_label.setText("📊 Dominant Frequency: ")
            self.agent_response.setText("🤖 Agent Response: ")
            self.top_predictions_label.setText("📋 Top Predictions: ")

            # Clear scrollable layout
            for i in reversed(range(self.scroll_layout.count())):
                widget = self.scroll_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)

            self.emotion_dropdown.clear()
            self.emotion_dropdown.addItem("Select emotion to play")

            self.result_label.setStyleSheet("color: #003366;")
            self.result_label.setText("🧹 Cleared. Ready for new input.")
            self.status_bar.setText("🧹 Everything cleared.")


    def refresh_emotion_dropdown(self):
        """
        Populate the dropdown with available emotional .wav files in the assets directory.
        """
        self.emotion_dropdown.clear()
        self.emotion_dropdown.addItem("Select emotion to play")
        added = set()
        for file in sorted(os.listdir("assets")):
            if file.endswith(".wav"):
                base = file.split("_")[0]
                if base not in added:
                    self.emotion_dropdown.addItem(base)
                    added.add(base)

    def browse_file(self):
        """
        Open a file dialog to select an input audio (.wav) file for prediction.
        """
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select WAV File", "", "WAV files (*.wav)")
        if path:
            self.file_input.setText(path)

    def predict_emotion_action(self):
        """
        Load the selected file, predict its emotion, update the GUI, and show visualizations.
        """
        try:
            path = self.file_input.text().strip()
            if not os.path.isfile(path):
                self.result_label.setStyleSheet("color: red;")
                self.result_label.setText("⚠️ Invalid path.")
                return

            # Get emotion and top predictions
            self.status_bar.setText("📊 Predicting emotion...")
            emotion, top_preds = predict_emotion(path)
            y, sr = librosa.load(path, sr=None)
            dom_freq = extract_dominant_freq(y, sr)
            sentence = self.text_input.text().strip() or "I have something to tell you."

            # Handle confidence threshold
            confidence = top_preds[0][1]
            if confidence < 50:
                self.result_label.setStyleSheet("color: orange;")
                self.result_label.setText("😶 No strong emotion detected ")
                response = "I'm not sure what you're feeling, but I'm here for you. 😊"
            else:
                self.result_label.setStyleSheet("color: blue;")
                self.result_label.setText(f"🎯 Predicted Emotion: {emotion}")
                response = generate_agent_response(sentence, emotion)

            # Always show top predictions
            pred_text = "\n".join([f"{label}: {score:.1f}%" for label, score in top_preds])
            self.top_predictions_label.setText(f"📋 Top Predictions:\n{pred_text}")

            self.freq_label.setText(f"📊 Dominant Frequency: {dom_freq:.2f} Hz")
            self.agent_response.setText(f"🤖 Agent Response: {response}")

            self.display_visualization(path)

        except Exception as e:
            self.result_label.setStyleSheet("color: red;")
            self.result_label.setText(f"❌ Error: {str(e)}")


    def display_visualization(self, path):
        """
        Load and display feature plots (waveform, FFT, MFCC, etc.) in a scrollable area.
        """
        try:

            emotion_name = os.path.splitext(os.path.basename(path))[0]
            plot_all_features(path, emotion_name=emotion_name)

            folder_path = os.path.join("report", emotion_name)
            for i in reversed(range(self.scroll_layout.count())):
                widget = self.scroll_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)

            for plot_file in sorted(os.listdir(folder_path)):
                if plot_file.endswith(".png"):
                    label = QtWidgets.QLabel()
                    pixmap = QtGui.QPixmap(os.path.join(folder_path, plot_file))
                    label.setPixmap(pixmap.scaledToWidth(850, QtCore.Qt.SmoothTransformation))
                    self.scroll_layout.addWidget(label)

            self.status_bar.setText(f"📊 Visualization loaded from: {folder_path}")
        except Exception as e:
            self.status_bar.setText(f"❌ Visualization error: {e}")


    def play_audio(self):
        """
        Play the currently selected audio file through the media player.
        """
        path = self.file_input.text().strip()
        if os.path.exists(path):
            url = QtCore.QUrl.fromLocalFile(path)
            content = QtMultimedia.QMediaContent(url)
            self.player = QtMultimedia.QMediaPlayer()
            self.player.setMedia(content)
            self.player.setVolume(100)
            self.player.play()
        else:
            self.result_label.setText("⚠️ File not found.")

    def play_selected_emotion(self):
        """
        Load and play the audio file matching the selected emotion from the dropdown.
        """
        selected = self.emotion_dropdown.currentText()
        if selected == "Select emotion to play":
            return
        for file in os.listdir("assets"):
            if file.startswith(selected) and file.endswith(".wav"):
                self.file_input.setText(os.path.join("assets", file))
                self.play_audio()
                break


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = EmotionAgentGUI()
    gui.show()
    sys.exit(app.exec_())
