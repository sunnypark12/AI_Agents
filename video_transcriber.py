import streamlit as st
import whisper
import os

def transcribe_video(video_file_path):
    model = whisper.load_model("base")

    try:
        result = model.transcribe(video_file_path)
        transcribed_text = result['text']
        return transcribed_text
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return ""


uploaded_video = st.file_uploader("Upload a video file (.mp4)", type=["mp4"])

if uploaded_video is not None:
    temp_video_path = f"/tmp/{uploaded_video.name}"
    
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())
    
    st.success(f"Video file {uploaded_video.name} uploaded successfully!")

    if st.button("Transcribe Video"):
        with st.spinner("Transcribing..."):
            transcribed_text = transcribe_video(temp_video_path)
            
            if transcribed_text:
                st.write("Transcription completed successfully!")
                st.text_area("Transcribed Video Text:", transcribed_text, height=300)
            else:
                st.error("Transcription failed.")
