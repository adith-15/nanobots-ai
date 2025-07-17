import argparse
import pyaudio
import wave
import whisper

MODEL_NAME = "base"
AUDIO_FILENAME = "recorded.wav"

def record_audio(filename=AUDIO_FILENAME, record_seconds=5):
    """
    Record audio from the microphone and save as WAV file.
    """
    chunk = 1024  # Record in chunks
    fmt = pyaudio.paInt16
    channels = 1
    rate = 16000  # Whisper works best with 16kHz

    p = pyaudio.PyAudio()

    # print("Recording...")
    stream = p.open(format=fmt,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    frames = []

    for _ in range(0, int(rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)

    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(fmt))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

def transcribe_audio(filename=AUDIO_FILENAME):
    """
    Transcribe audio file using Whisper.
    """
    model = whisper.load_model(MODEL_NAME)
    print("Transcribing...")
    result = model.transcribe(filename)
    text = result["text"].strip()
    print("Transcription complete.")
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record_only", action="store_true")
    args = parser.parse_args()

    if args.record_only:
        record_audio()
    else:
        record_audio()
        transcription = transcribe_audio()
        print(transcription)
