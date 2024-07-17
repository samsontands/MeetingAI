import streamlit as st
import os
import tempfile
from groq import Groq

# Initialize Groq client with API key from Streamlit secrets
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def transcribe_audio(audio_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        with open(tmp_file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(tmp_file_path), file),
                model="whisper-large-v3",
                response_format="text"
            )
        return transcription.text
    finally:
        os.unlink(tmp_file_path)

st.title("Audio Transcription with Groq Whisper API")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    if st.button("Transcribe"):
        with st.spinner("Transcribing..."):
            transcription = transcribe_audio(uploaded_file)
        st.success("Transcription complete!")
        st.text_area("Transcription", transcription, height=300)
