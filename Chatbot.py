from PyPDF2 import PdfReader
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Any, Dict, List, Optional
from openai import OpenAI

#Initialize model
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
def get_embedding(text, model="nomic-ai/nomic-embed-text-v1.5-GGUF"):
   #text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

class LMStudioEmbedding(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts_embedded = []
        for text in texts:
            chunk_embedded =  get_embedding(text)
            texts_embedded.append(chunk_embedded)
        return texts_embedded

    def embed_query(self, text: str) -> List[float]:
        return get_embedding(text)    


# For reading PDFs and returning text string
def read_pdf(file):
    file_content=""
    # Create a PDF file reader object
    pdf_reader = PdfReader(file)
    # Get the total number of pages in the PDF
    num_pages = len(pdf_reader.pages)
    # Iterate through each page and extract text
    for page_num in range(num_pages):
        # Get the page object
        page = pdf_reader.pages[page_num]
        file_content += page.extract_text()
    return file_content

def process_file(file):
    file_content = read_pdf(file)
    book_documents = recursive_text_splitter.create_documents([file_content])
    # Limit the no of characters, remove \n
    book_documents = [Document(page_content = text.page_content.replace("\n", " ").replace(".", "").replace("-", "")) for text in book_documents]
    docsearch = FAISS.from_documents(book_documents, LMStudioEmbedding())
    return docsearch

                    

recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=20)



def ChatBot(file_processed: FAISS):
    st.title("Ask questions about the PDF")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask questions about the article"):
       # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        match = "Given the following information " + (file_processed.search(prompt, "similarity")[0]).page_content + "answer the following question" +  prompt 
        st.session_state.augmented_messages.append({"role": "user", "content": match})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)     

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model = st.session_state["llm"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.augmented_messages
                ],
                stream=True,
                temperature=0.7,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.augmented_messages.append({"role": "assistant", "content": response})


def initialization(flag=False):
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "augmented_messages" not in st.session_state:
        st.session_state.augmented_messages=[]

    if "llm" not in st.session_state:
        st.session_state["llm"] = "lmstudio-ai/gemma-2b-it-GGUF"

def main():
    initialization(True)

    st.set_page_config("Chatbot","💬")
    st.title("📝 Upload a file")
    uploaded_files = st.file_uploader("Upload an article", type="pdf", accept_multiple_files=False)

    if uploaded_files:
        file_processed = process_file(uploaded_files)
        ChatBot(file_processed)

main()