import streamlit as st
import openai
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO

# Configure API keys using Streamlit secrets
openai.api_key = st.secrets["openai"]["api_key"]
groq_api_key = st.secrets["groq"]["api_key"]

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
def summarize_text_openai(text):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Summarize the following text:\n\n{text}\n\nSummary:",
        max_tokens=150
    )
    summary = response.choices[0].text.strip()
    return summary

# Function to summarize text using GROQ API (placeholder)
def summarize_text_groq(text):
    # This is a placeholder. You'll need to implement the actual GROQ API call here.
    # For now, we'll just return a message indicating that GROQ summarization is not yet implemented.
    return "GROQ summarization not yet implemented. Please use OpenAI for now."

# Function to create meeting minutes
def create_meeting_minutes(summary):
    meeting_minutes = f"""
Meeting Minutes
---------------
Summary: {summary}
"""
    return meeting_minutes

# Streamlit UI
st.title("Speech to Text Meeting Minutes")
api_choice = st.radio("Choose API for summarization:", ("OpenAI", "GROQ"))
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    # Convert audio to text
    text = audio_to_text(uploaded_file)
    st.subheader("Transcription")
    st.write(text)

    # Summarize text
    if api_choice == "OpenAI":
        summary = summarize_text_openai(text)
    else:  # GROQ
        summary = summarize_text_groq(text)
    
    st.subheader("Summary")
    st.write(summary)

    # Create meeting minutes
    meeting_minutes = create_meeting_minutes(summary)
    st.subheader("Meeting Minutes")
    st.write(meeting_minutes)
