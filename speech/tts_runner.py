from tts import speak_text

if __name__ == "__main__":
    import sys
    text = sys.argv[1]
    speak_text(text)