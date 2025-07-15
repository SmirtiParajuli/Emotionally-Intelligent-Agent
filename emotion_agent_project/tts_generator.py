# tts_generator.py
import os
import warnings
from cryptography.utils import CryptographyDeprecationWarning
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

from TTS.api import TTS

def generate_neutral_speech(text=" I have something to tell you", output_path="assets/neutral.wav"):
    """
    Generate a neutral-tone speech waveform from input text using a Tacotron2-based TTS model.

    Parameters:
        text (str): The input sentence to convert into speech.
        output_path (str): Path where the generated .wav file will be saved.

    Notes:
        - Uses the pre-trained 'tacotron2-DDC_ph' model from the Coqui TTS library.
        - Ensures the output directory exists before saving.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC_ph", progress_bar=False, gpu=False)
    tts.tts_to_file(text=text, file_path=output_path)
    print(f" Neutral speech generated at: {output_path}")
