import streamlit as st
import PyPDF2 as py2
import random
import time

# Example: reuse your existing OpenAI setup
from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

st.title("Simple chat")
uploaded_files = st.file_uploader("Loaded file about a subject", accept_multiple_files=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "llm" not in st.session_state:
    st.session_state["llm"] = "lmstudio-ai/gemma-2b-it-GGUF"

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model = st.session_state["llm"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
            temperature=0.7,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})