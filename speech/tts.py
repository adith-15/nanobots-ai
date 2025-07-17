from TTS.api import TTS
import sounddevice as sd
import numpy as np

# Load TTS model

def speak_text(text):
    model_name = "tts_models/en/ljspeech/tacotron2-DDC"
    tts = TTS(model_name)
    wav = tts.tts(text)
    wav = np.array(wav)
    sd.play(wav, samplerate=tts.synthesizer.output_sample_rate)
    sd.wait()  

if __name__ == "__main__":
    speak_text("Hi Karthik, how can I help you today?")  # Example usage
