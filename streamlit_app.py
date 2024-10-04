import os
import streamlit as st
from dotenv import load_dotenv
from text_summarizer import run_text_summarizer_app
from video_transcriber import transcribe_video
from pdf_translator import translate_pdf

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API Key not found. Please set it in your .env file.")
else:
    from openai import OpenAI
    OpenAI.api_key = openai_api_key

    st.title("🧠 Multi-AI Agent")
    st.write("Upload a document, video, or PDF file and choose the operation (summarization, transcription, or translation) - our agent will help!")

    operation = st.selectbox(
        "Choose an operation",
        ("Text Summarization", "Video Transcription", "PDF Translation")
    )

    # Text Summarization
    if operation == "Text Summarization":
        run_text_summarizer_app(openai_api_key)

    # Video Transcription
    elif operation == "Video Transcription":
        uploaded_video = st.file_uploader("Upload a video file (.mp4)", type="mp4")
        if uploaded_video:
            video_file_path = f"/tmp/{uploaded_video.name}"
            with open(video_file_path, "wb") as f:
                f.write(uploaded_video.read())

            if st.button("Run Video Transcription"):
                transcription = transcribe_video(video_file_path)
                st.write("Transcribed Video Text:")
                st.write(transcription)

    # PDF Translation
    elif operation == "PDF Translation":
        uploaded_pdf = st.file_uploader("Upload a PDF file (.pdf)", type="pdf")
        target_language = st.text_input("Target language (e.g., French, English)")

        if uploaded_pdf and target_language:
            pdf_file_path = f"/tmp/{uploaded_pdf.name}"
            with open(pdf_file_path, "wb") as f:
                f.write(uploaded_pdf.read())

            if st.button("Translate PDF"):
                translated_text = translate_pdf(pdf_file_path, target_language)
                st.write(f"Translated PDF Text to {target_language}:")
                st.write(translated_text)
