from pathlib import Path
import numpy as np
import faiss

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

DOCS_PATH = "data/synthetic_docs"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = []

for file in Path(DOCS_PATH).glob("*.md"):
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()

    file_chunks = splitter.split_text(text)

    chunks.extend(file_chunks)

print(f"Total chunks: {len(chunks)}")

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

faiss.write_index(
    index,
    "backend/vectorstore/faiss_index.bin"
)

print("FAISS index created successfully!")