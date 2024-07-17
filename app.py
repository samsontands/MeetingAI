import streamlit as st
import os
import tempfile
import requests
from groq import Groq

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

def summarize_meeting(transcription):
    client = Groq(api_key=st.secrets['groq']['api_key'])
    
    prompt = f"""
    Please summarize the following meeting transcription:

    {transcription}

    Provide the following:
    1. A brief overview of the meeting
    2. Key points discussed
    3. Action items
    4. Next steps

    Format the summary in a clear and concise manner.
    """

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes meetings."},
                {"role": "user", "content": prompt}
            ],
            model="mixtral-8x7b-32768",
            max_tokens=1024,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred during summarization: {str(e)}")
        return None

def main():
    st.title("Meeting AI with Groq API")

    st.sidebar.write(f"API Key: {st.secrets['groq']['api_key'][:5]}...")

    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a", "mp4", "mpeg", "mpga", "webm"])

    if uploaded_file is not None:
        st.audio(uploaded_file)
        if st.button("Process Meeting"):
            with st.spinner("Transcribing and summarizing..."):
                try:
                    transcription = transcribe_audio(uploaded_file)
                    st.success("Transcription complete!")
                    st.subheader("Transcription")
                    st.text_area("Full Transcription", transcription, height=200)

                    summary = summarize_meeting(transcription)
                    if summary:
                        st.success("Summary generated!")
                        st.subheader("Meeting Summary")
                        st.markdown(summary)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error(f"Full error: {repr(e)}")

if __name__ == "__main__":
    main()
