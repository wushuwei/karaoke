import argparse
import whisper
import sys
import torch


# Parse command line arguments
parser = argparse.ArgumentParser(description="Transcribe an audio file using Whisper model")
parser.add_argument("audio_path", type=str, help="Path to the audio file")
args = parser.parse_args()

if len(sys.argv) > 2:
    print("Error: You must provide exactly one audio file path as an argument.")
    sys.exit(1)

# Load the Whisper model
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: " + device )
# llm = "medium"
llm = "base"
#small.en, tiny, medium.en
model = whisper.load_model(llm, device=device)

# Specify the path to your WAV file
if len(sys.argv) != 2:
    audio_path = "/home/steven/Documents/testing.wav"
else:
    audio_path = args.audio_path

# Transcribe the audio
result = model.transcribe(audio_path)
# result = model.transcribe(audio_path, language='en')

# Print the transcription
print(result["text"])
