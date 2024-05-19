# LLM_RAG
Implementation in Streamlit of a RAG chatbot using Llms

How to run the project:

1. First we load in the LM-Studio the llm model "TheBloke/Mistral-7B-Instruct-v0.2-GGUF".
2. Then we load "nomic-ai/nomic-embed-text-v1.5-GGUF" model of embedding.
3. After all that we start the server.
4. At this point all we need is to create a python environment, install de dependencies and run it:

```
python -m venv .venv
.venv\Scripts\activate
steamlit run .\main.py
```

And you're free to go
