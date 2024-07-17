import streamlit as st
import os
import tempfile
import requests
from groq import Groq
import re

# Groq API endpoints
GROQ_API_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_CHAT_API_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

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

def analyze_meeting(transcription):
    client = Groq(api_key=st.secrets['groq']['api_key'])
    
    prompt = f"""
    Analyze the following meeting transcription and provide:

    1. A brief summary of the meeting (max 3 sentences)
    2. Total time of the meeting (estimate based on word count, assume 150 words per minute)
    3. Sentiment analysis (percentage of positive, negative, and neutral sentiments)
    4. Top 3 topic trackers (main themes discussed)
    5. A suggested title for the meeting
    6. Key points discussed
    7. Action items
    8. Next steps

    Meeting Transcription:
    {transcription}

    Format the analysis in a clear and concise manner, using markdown for headers.
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
    st.title("Meeting AI with Groq API")
    st.sidebar.write(f"API Key: {st.secrets['groq']['api_key'][:5]}...")

    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a", "mp4", "mpeg", "mpga", "webm"])

    if uploaded_file is not None:
        st.audio(uploaded_file)
        if st.button("Process Meeting"):
            with st.spinner("Transcribing and analyzing..."):
                try:
                    transcription = transcribe_audio(uploaded_file)
                    st.success("Transcription complete!")
                    st.subheader("Transcription")
                    st.text_area("Full Transcription", transcription, height=200)

                    analysis = analyze_meeting(transcription)
                    if analysis:
                        st.success("Analysis complete!")
                        
                        # Extract and display the meeting title
                        meeting_title = extract_meeting_title(analysis)
                        new_title = st.text_input("Meeting Title", value=meeting_title)
                        if new_title != meeting_title:
                            st.info("Title updated!")

                        # Display the full analysis
                        st.subheader("Meeting Analysis")
                        st.markdown(analysis)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error(f"Full error: {repr(e)}")

if __name__ == "__main__":
    main()
