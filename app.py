import streamlit as st
import openai
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO

# Configure OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'

# Function to convert audio to text
def audio_to_text(audio_file):
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_file(audio_file)
    audio.export("temp.wav", format="wav")
    with sr.AudioFile("temp.wav") as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    return text

# Function to summarize text using OpenAI API
def summarize_text(text):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Summarize the following text:\n\n{text}\n\nSummary:",
        max_tokens=150
    )
    summary = response.choices[0].text.strip()
    return summary

# Function to create meeting minutes
def create_meeting_minutes(summary):
    meeting_minutes = f"""
    Meeting Minutes
    ---------------
    Summary:
    {summary}
    """
    return meeting_minutes

# Streamlit UI
st.title("Speech to Text Meeting Minutes")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    # Convert audio to text
    text = audio_to_text(uploaded_file)
    st.subheader("Transcription")
    st.write(text)

    # Summarize text
    summary = summarize_text(text)
    st.subheader("Summary")
    st.write(summary)

    # Create meeting minutes
    meeting_minutes = create_meeting_minutes(summary)
    st.subheader("Meeting Minutes")
    st.write(meeting_minutes)
