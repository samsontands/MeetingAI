import streamlit as st
import os
import tempfile
import requests
from groq import Groq
import re

# Groq API endpoints
GROQ_API_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_CHAT_API_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

def estimate_duration(transcription):
    # Estimate duration based on word count (assuming 150 words per minute)
    word_count = len(transcription.split())
    estimated_minutes = max(1, round(word_count / 150))
    return estimated_minutes

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

def analyze_meeting(transcription, duration_minutes):
    client = Groq(api_key=st.secrets['groq']['api_key'])
    
    # Adjust summary length based on estimated duration
    if duration_minutes < 10:
        summary_length = "2-3 sentences"
    elif duration_minutes < 30:
        summary_length = "4-5 sentences"
    else:
        summary_length = "6-8 sentences"
    
    prompt = f"""
    Analyze the following meeting transcription for a meeting that lasted approximately {duration_minutes} minutes and provide:

    1. A summary of the meeting ({summary_length})
    2. Sentiment analysis (percentage of positive, negative, and neutral sentiments)
    3. Top 3-5 topic trackers (main themes discussed)
    4. A suggested title for the meeting
    5. Key points discussed (bullet points)
    6. Action items (bullet points)
    7. Next steps (bullet points)

    Meeting Transcription:
    {transcription}

    Format the analysis in a clear and concise manner, using markdown for headers and bullet points.
    """

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes meetings."},
                {"role": "user", "content": prompt}
            ],
            model="mixtral-8x7b-32768",
            max_tokens=1024,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        return None

def extract_meeting_title(analysis):
    title_match = re.search(r"#+\s*Suggested Title[:\n\s]*(.*)", analysis)
    if title_match:
        return title_match.group(1).strip()
    return "Untitled Meeting"

def main():
    st.set_page_config(page_title="Meeting AI Assistant", page_icon="🎙️", layout="wide")
    
    st.title("🎙️ Meeting AI Assistant")
    st.write("Upload your meeting audio file and get instant transcription and analysis!")

    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a", "mp4", "mpeg", "mpga", "webm"])

    if uploaded_file is not None:
        st.audio(uploaded_file)
        if st.button("🚀 Process Meeting", key="process_button"):
            with st.spinner("Transcribing and analyzing... This may take a few minutes."):
                try:
                    transcription = transcribe_audio(uploaded_file)
                    duration_minutes = estimate_duration(transcription)
                    analysis = analyze_meeting(transcription, duration_minutes)

                    if analysis:
                        meeting_title = extract_meeting_title(analysis)
                        new_title = st.text_input("📌 Meeting Title", value=meeting_title)
                        if new_title != meeting_title:
                            st.success("Title updated!")

                        st.subheader(f"📊 Estimated Meeting Duration: {duration_minutes} minutes")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("📝 Transcription")
                            st.text_area("Full Transcription", transcription, height=300)
                            st.download_button(
                                label="📥 Download Transcription",
                                data=transcription,
                                file_name="transcription.txt",
                                mime="text/plain"
                            )

                        with col2:
                            st.subheader("📊 Meeting Analysis")
                            st.markdown(analysis)
                            st.download_button(
                                label="📥 Download Meeting Notes",
                                data=analysis,
                                file_name="meeting_notes.md",
                                mime="text/markdown"
                            )

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error(f"Full error: {repr(e)}")
    else:
        st.info("👆 Upload an audio file to get started!")

if __name__ == "__main__":
    main()
