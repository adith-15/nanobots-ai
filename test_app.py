import subprocess

def main():
    response = "Hello, this is a minimal test."
    subprocess.run(["python", "speech/tts_runner.py", response])

if __name__ == "__main__":
    main()
