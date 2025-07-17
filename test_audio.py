from speech.stt import record_audio, transcribe_audio
from speech.tts import speak_text

def main():
    print("Recording...")
    record_audio(record_seconds=5)
    text = transcribe_audio()
    print(f"You said: {text}")

    print("Speaking back...")
    speak_text("You said: " + text)

if __name__ == "__main__":
    main()