import streamlit as st
from pdf_processing import process_file, recursive_text_splitter
from chatbot import ChatBot
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from embeddings import LMStudioEmbedding

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "augmented_messages" not in st.session_state:
        st.session_state.augmented_messages = []

    if "llm" not in st.session_state:
        st.session_state["llm"] = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"

    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = []

def main():
    initialize_session_state()

    st.set_page_config("Chatbot", "💬")
    st.title("📝 Upload a file")
    uploaded_file = st.file_uploader("Upload an article", type="pdf", accept_multiple_files=False)

    if uploaded_file:
        book_documents = process_file(uploaded_file, recursive_text_splitter)
        docsearch = FAISS.from_documents(book_documents, LMStudioEmbedding())
        docsearch.save_local(f"db/vector")

        processed_file = FAISS.load_local(f"db/vector", LMStudioEmbedding(), allow_dangerous_deserialization=True)
        ChatBot(processed_file)

if __name__ == "__main__":
    main()
