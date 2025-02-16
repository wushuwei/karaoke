import streamlit as st
from pydub import AudioSegment
from io import BytesIO
import whisper
import torch
import numpy as np
import tempfile



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
    # samples = np.array(audio.get_array_of_samples())
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    return torch.tensor(samples)

# Function to recognize speech using Whisper
def recognize_speech_whisper(audio_file):
    # Check if a GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    # model = whisper.load_model("tiny", device=device)
    model = whisper.load_model("base", device=device)
    # model = whisper.load_model("small.en", device=device)
    # model = whisper.load_model("medium", device=device)
    # audio = load_audio(audio_file)
    audio = audio_file

    # Model Synchronization: Ensure your model and data are correctly moved to the GPU or CPU
    # model = model.to(device)
    # audio = audio.to(device)


    result = model.transcribe(audio)
    # result = model.transcribe(audio, language='en')
    return result['text']


def saveToTempFile(audio_file):
    temp_file_path = audio_file
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    return temp_file_path

def transcribeFile(local_file_path, model):
    # Transcribe the audio
    result = model.transcribe(local_file_path)
    return result["text"]

st.title("Audio to Lyrics App")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "aiff", "flac"])

if uploaded_file is not None:
    file_type = uploaded_file.type.split('/')[1]
    st.audio(uploaded_file, format=f'audio/{file_type}')
    
    if st.button("Generate Lyrics"):
        # if torch.cuda.is_available():
            # st.write("use gpu for torch")
            # print("use gpu for torch")
        # wav_file = convert_audio_to_wav(uploaded_file, file_type)
        # wav_file = uploaded_file
        # lyrics = recognize_speech_whisper(wav_file)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("base", device=device)
        lyrics = transcribeFile(local_file_path=saveToTempFile(uploaded_file), model=model)

        st.write(lyrics)
        print(lyrics)
