from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_PATH = "data/synthetic_docs"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

all_chunks = []

for file in Path(DOCS_PATH).glob("*.md"):
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = splitter.split_text(text)

    print(f"{file.name}: {len(chunks)} chunks")

    all_chunks.extend(chunks)

print(f"Total Chunks: {len(all_chunks)}")
