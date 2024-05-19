from typing import List
from openai import OpenAI
from langchain_core.embeddings import Embeddings

# Initialize model
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def initialize_client(api_key: str, base_url: str = "http://localhost:1234/v1"):
    return OpenAI(base_url=base_url, api_key=api_key)

def get_embedding(text: str, model: str = "nomic-ai/nomic-embed-text-v1.5-GGUF") -> List[float]:
    return client.embeddings.create(input=[text], model=model).data[0].embedding

class LMStudioEmbedding(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return get_embedding(text)
