############################## Emotionally-Intelligent-Agent ##################################
An interactive AI agent that generates emotional speech, classifies audio emotions, and responds with empathy. Built with Python, PyQt5, and machine learning.
    **Author**: Smriti Parajuli  
    **Course**: NLP303 – Speech Processing  
    **Institution**: Media Design School  
    **Submission Date**: June 16, 2025  

---

## 🔍 Overview

This project implements an interactive, speech-based **Emotionally Sensitive AI Agent** that processes user-input text or recorded voice, classifies the emotional state, and responds empathetically via a dynamic GUI.
The system is built entirely from scratch using classical signal processing and supervised machine learning (Random Forest). No external APIs are used — enabling full control and performance evaluation of the core pipeline.

---

## 🧠 Project Goals

    - Explore emotional speech processing using handcrafted logic and ML techniques.
    - Build a functional AI agent that adapts its response to user emotions.
    - Evaluate audio-based emotion detection using engineered features.
    - Set the foundation for future integration of large language models (LLMs) and advanced emotional reasoning.

---

## 📂 Project Structure
    emotionally-sensitive-agent/
    │
    ├── main.py # Main PyQt5 GUI
    ├── tts_generator.py # Neutral TTS generation
    ├── emotion_modulator.py # Modulate speech (happy, sad, angry, etc.)
    ├── emotion_classifier.py # Feature extraction + ML model
    ├── agent_response.py # Agent reply logic (hardcoded for now)
    │
    ├── /assets # Generated audio files
    ├── /report # Trained model, CSVs, and plots
    ├── requirements.txt # Python dependencies
    └── README.md # This file

---

## ⚙️ Installation

    ### 1. Install Python  
    Python 3.8–3.10 is recommended.
    
    ### 2. Install Dependencies  
    Run:
    ```bash
    pip install -r requirements.txt

## Dependencies:
    PyQt5
    
    librosa
    
    scikit-learn
    
    matplotlib
    
    numpy
    
    soundfile
    
    sounddevice
    
    noisereduce
    
    speechrecognition

## Launch the Application:
    python main.py

---

💡 Key Features

    ✅ Generate neutral speech from user-entered sentences
    
    ✅ Apply pitch, speed, and gain changes for emotional modulation
    
    ✅ Extract MFCC, ZCR, spectral centroid, RMS, duration, and dominant frequency
    
    ✅ Train and use RandomForest for emotion classification
    
    ✅ Accept live voice input or uploaded .wav files
    
    ✅ Generate empathetic replies using emoji-enhanced response logic
    
    ✅ Visualize waveform, MFCCs, and spectrogram in PyQt5 GUI
    
    ✅ Includes emotion playback, file browsing, reset/clear, and confidence scores

---

🤖 Agent Logic (Current Status & Future Plan)

    The agent’s response system is currently hardcoded with simple rules based on the predicted emotion — this is intentional for performance and interpretability testing.
    
    🔭 Future Upgrades (Planned for Phase 2):
    🔌 LLM / ChatGPT API Integration: Enable nuanced, adaptive conversations
    
    🧠 Emotion-Conditioned Prompting: Vary prompts based on detected emotion
    
    🎭 Agent Persona Profiles: Custom personalities (friendly, professional, humorous)
    
    🌡️ Emotion Intensity Scaling: Modify tone based on classifier confidence
    
    These enhancements will transform the agent from a rule-based bot to a truly empathetic digital assistant.
---

🖥️ How to Use the GUI

    Enter a sentence and click Generate & Train.
    
    Choose an emotion from the dropdown to hear its modulated version.
    
    Upload a .wav file or use Record Voice to input speech.
    
    Click Predict Emotion.
    
    View the predicted emotion, dominant frequency, waveform, MFCCs, spectrogram, and agent reply.
---

📌 Example Output
    
    Input: Neutral speech modulated as "Happy"
    → Predicted Emotion: Happy
    → Dominant Frequency: 265.62 Hz
    → Agent Response: I have something to tell you. 😊 That really makes me smile!
    
    Top Predictions:
    - Happy: 58%
    - Nervous: 18%
    - Angry: 10%
    
---
🚀 Future Enhancements

    🌐 Multilingual support
    
    🧠 Deep learning models (CNN, RNN) for emotion classification
    
    🎛️ Real-time processing optimization
    
    👤 Avatar-based agent display
    
    🔊 Emotion blending and layered tone shifts

---
 Acknowledgements
 
    Librosa – Audio processing
    
    Scikit-learn – ML algorithms
    
    PyQt5 – GUI framework
    
    SpeechRecognition – Voice input
    
    Noisereduce – Denoising   
    
👩‍💻 Author
Smriti Parajuli
Bachelor of Software Engineering (AI)
Media Design School – 2025
GitHub

---

If you'd like this exported as a downloadable `.md` file or want a shorter version for LinkedIn or your website, just say the word!
