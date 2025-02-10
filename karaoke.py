import streamlit as st
from pydub import AudioSegment
from io import BytesIO
import whisper
import torch
import numpy as np




# Function to convert audio to WAV
def convert_audio_to_wav(audio_file, file_type):
    audio = AudioSegment.from_file(audio_file, format=file_type)
    wav_io = BytesIO()
    audio.export(wav_io, format="wav")
    return wav_io

# Function to load and process audio data
def load_audio(audio_file):
    audio = AudioSegment.from_file(audio_file, format="wav")
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    return torch.tensor(samples)

# Function to recognize speech using Whisper
def recognize_speech_whisper(audio_file):
    # Check if a GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = whisper.load_model("base")
    model = whisper.load_model("medium", device=device)
    audio = load_audio(audio_file)
    # result = model.transcribe(audio)
    result = model.transcribe(audio, language='en')
    return result['text']

st.title("Audio to Lyrics App")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "aiff", "flac"])

if uploaded_file is not None:
    file_type = uploaded_file.type.split('/')[1]
    st.audio(uploaded_file, format=f'audio/{file_type}')
    
    if st.button("Generate Lyrics"):
        wav_file = convert_audio_to_wav(uploaded_file, file_type)
        lyrics = recognize_speech_whisper(wav_file)
        st.write(lyrics)
