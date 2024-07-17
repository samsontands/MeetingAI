import streamlit as st
import os
import tempfile
from groq import Groq

# Initialize Groq client with API key from Streamlit secrets
client = Groq(api_key=st.secrets["groq"]["api_key"])

def transcribe_audio(audio_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        with open(tmp_file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(tmp_file_path, file),
                model="whisper-large-v3",
                response_format="text",
                language="en",  # Optional: specify language or let Whisper auto-detect
                temperature=0.0  # Optional: adjust as needed
            )
        return transcription.text
    finally:
        os.unlink(tmp_file_path)

st.title("Audio Transcription with Groq Whisper API")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a", "mp4", "mpeg", "mpga", "webm"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    if st.button("Transcribe"):
        with st.spinner("Transcribing..."):
            try:
                transcription = transcribe_audio(uploaded_file)
                st.success("Transcription complete!")
                st.text_area("Transcription", transcription, height=300)
            except Exception as e:
                st.error(f"An error occurred during transcription: {str(e)}")
