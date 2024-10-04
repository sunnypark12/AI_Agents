import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

OpenAI.api_key = os.getenv("OPENAI_API_KEY")

def translate_pdf(pdf_path: str, target_language: str = "English"):
    """Translate the contents of the PDF to the target language."""
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    prompt = f"Translate the following text to {target_language}:\n{text}"

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"You are a translator that translates text to {target_language}."},
            {"role": "user", "content": prompt}
        ]
    )

    translation = response.choices[0].message
    return translation
