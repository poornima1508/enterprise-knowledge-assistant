from pathlib import Path

DOCS_PATH = "data/synthetic_docs"

documents = []

for file in Path(DOCS_PATH).glob("*.md"):
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()

        documents.append({
            "filename": file.name,
            "content": text
        })

print(f"Loaded {len(documents)} documents")

for doc in documents:
    print(f"- {doc['filename']}")
