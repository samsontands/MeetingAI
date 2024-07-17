import streamlit as st
import os
import tempfile
import requests

# Groq API endpoint
GROQ_API_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"

def transcribe_audio(audio_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        with open(tmp_file_path, "rb") as file:
            files = {"file": file}
            data = {
                "model": "whisper-large-v3",
                "response_format": "text",
                "language": "en",
                "temperature": 0.0
            }
            headers = {
                "Authorization": f"Bearer {st.secrets['groq']['api_key']}"
            }
            response = requests.post(GROQ_API_ENDPOINT, files=files, data=data, headers=headers)
            response.raise_for_status()
            return response.text
    finally:
        os.unlink(tmp_file_path)

st.title("Audio Transcription with Groq Whisper API")

# Debug: Print first 5 characters of API key
st.sidebar.write(f"API Key: {st.secrets['groq']['api_key'][:5]}...")

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
                st.error(f"Full error: {repr(e)}")  # More detailed error information
