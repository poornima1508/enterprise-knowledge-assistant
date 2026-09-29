from pathlib import Path
import faiss
import numpy as np

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

chunks = []

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

for file in Path("data/synthetic_docs").glob("*.md"):
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()

    file_chunks = splitter.split_text(text)
    chunks.extend(file_chunks)

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index(
    "backend/vectorstore/faiss_index.bin"
)

query = "What is the remote work policy?"

query_embedding = model.encode([query])

distances, indices = index.search(
    np.array(query_embedding),
    k=3
)

print("\nTop Results:\n")

for i in indices[0]:
    print(chunks[i])
    print("\n" + "=" * 80 + "\n")