from pathlib import Path
import faiss
import numpy as np

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM

chunks = []

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

for file in Path("data/synthetic_docs").glob("*.md"):
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()

    chunks.extend(splitter.split_text(text))

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index(
    "backend/vectorstore/faiss_index.bin"
)

query = input("Ask a question: ")

query_embedding = embedding_model.encode([query])

distances, indices = index.search(
    np.array(query_embedding),
    k=3
)

context = "\n\n".join(
    [chunks[i] for i in indices[0]]
)

prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

llm = OllamaLLM(model="qwen2:1.5b")

response = llm.invoke(prompt)

print("\nAnswer:\n")
print(response)