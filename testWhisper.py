import whisper

# Load the Whisper model
model = whisper.load_model("base", device="cpu")

# Specify the path to your WAV file
audio_path = "/home/steven/Documents/testing.wav"

# Transcribe the audio
result = model.transcribe(audio_path)

# Print the transcription
print(result["text"])
