import tempfile
from pydub import AudioSegment
from io import BytesIO

# Function to convert audio to WAV
def convert_audio_to_wav(audio_file, file_type):
    audio = AudioSegment.from_file(audio_file, format=file_type)
    wav_io = BytesIO()
    audio.export(wav_io, format="wav")
    return wav_io


def saveWavBytesTOToTempFile(wav_audio):
    temp_wav_file_path = ""
    # Saving the WAV audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
        temp_wav_file.write(wav_audio.getvalue())
        temp_wav_file_path = temp_wav_file.name
    return temp_wav_file_path

# Example usage
wav_audio = convert_audio_to_wav("example.flac", "flac")
temp_wav_file_path = saveWavBytesTOToTempFile(wav_audio)

print(f"The WAV file is saved at: {temp_wav_file_path}")
