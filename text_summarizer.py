import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_qa_chain(text_file_path, openai_api_key):
    loader = TextLoader(text_file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)

    qa_chain = VectorDBQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), vectorstore=vectorstore)
    return qa_chain


def run_text_summarizer_app(openai_api_key):
    uploaded_file = st.file_uploader("Upload a text file (.txt)", type="txt")

    if uploaded_file is not None:
        temp_file_path = f"/tmp/{uploaded_file.name}"

        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"File {uploaded_file.name} uploaded successfully!")

        qa_chain = create_qa_chain(temp_file_path, openai_api_key)

        question = st.text_area("Ask a question about the document", placeholder="Can you give me a short summary?")

        if st.button("Run Text Summarizer") and question:
            summarized_text = qa_chain.run(question)
            st.write(f"Summarized Text: {summarized_text}")
