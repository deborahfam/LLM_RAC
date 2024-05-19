import streamlit as st
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from analize_prompt import analyze_prompt
from embeddings import LMStudioEmbedding

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def submit_to_llm():
    return client.chat.completions.create(
        model=st.session_state["llm"],
        messages=[
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.augmented_messages if m["role"] == "user"
        ][-4:],
        stream=True,
        temperature=0.7,
    )

def refine_prompt(prompt: str, file_processed: FAISS, search_method:str, k_value:int) -> str:
    context = " ".join(d.page_content for d in file_processed.search(prompt, search_type=search_method, k=k_value))
    return f"Given the following information: \n\n{context}\n\n answer the following question: \n\n{prompt}"

def handle_user_input(file_processed: FAISS):
    prompt = st.chat_input("Ask questions about the article")
    if prompt:
        analysis = analyze_prompt(prompt)
        search_method = analysis["search_method"]
        k = analysis["k"]
        st.write(search_method,k)

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            refined_prompt = refine_prompt(prompt, file_processed, search_method, k)
            st.session_state.augmented_messages.append({"role": "user", "content": refined_prompt})

            stream = submit_to_llm()
            response = st.write_stream(stream)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.augmented_messages.append({"role": "assistant", "content": response})

def ChatBot(file_processed: FAISS):
    st.title("Ask questions about the PDF")
    display_chat_history()
    handle_user_input(file_processed)
