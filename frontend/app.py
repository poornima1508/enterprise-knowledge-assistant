import streamlit as st
from pathlib import Path
import faiss
import numpy as np

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    layout="wide"
)

st.title("🤖 Enterprise Knowledge Assistant")

st.markdown(
    """
    Ask questions about company policies, employee benefits,
    security guidelines, travel policies, remote work policies,
    and other enterprise knowledge.
    """
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------
# NEW CHAT BUTTON
# --------------------------------------------------

if st.button("🗑️ New Chat"):
    st.session_state.messages = []
    st.rerun()

# --------------------------------------------------
# SUGGESTED QUESTIONS
# --------------------------------------------------

if len(st.session_state.messages) == 0:
    st.info(
        """
        Try asking:

        • What is the VPN access policy?

        • What are the code review standards?

        • How does expense reimbursement work?

        • What is the remote work policy?

        • What are the device security requirements?
        """
    )

# --------------------------------------------------
# LOAD DOCUMENTS
# --------------------------------------------------

chunks = []
sources = []

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

for file in Path("data/synthetic_docs").glob("*.md"):

    with open(file, "r", encoding="utf-8") as f:
        text = f.read()

    file_chunks = splitter.split_text(text)

    for chunk in file_chunks:
        chunks.append(chunk)
        sources.append(file.name)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

index = faiss.read_index(
    "backend/vectorstore/faiss_index.bin"
)

llm = OllamaLLM(
    model="qwen2:1.5b"
)

# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message("user"):
        st.write(message["question"])

    with st.chat_message("assistant"):
        st.write(message["answer"])

        st.caption(
            "📄 Source: "
            + ", ".join(message["sources"])
        )

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about company policies..."
)

# --------------------------------------------------
# RAG PIPELINE
# --------------------------------------------------

if question:

    query_embedding = embedding_model.encode(
        [question]
    )

    distances, indices = index.search(
        np.array(query_embedding),
        k=3
    )

    retrieved_chunks = [
        chunks[i]
        for i in indices[0]
    ]

    retrieved_sources = list(
        set(
            [
                sources[i]
                for i in indices[0]
            ]
        )
    )

    context = "\n\n".join(
        retrieved_chunks
    )

    prompt = f"""
You are an enterprise knowledge assistant.

Answer the user's question using the information from the context below.

Instructions:
- Be concise.
- Use bullet points when appropriate.
- Do not invent information.
- Summarize relevant information from the context.
- If no relevant information exists, respond:
"I could not find that information in the available documents."

Context:
{context}

Question:
{question}

Answer:
"""

    answer = llm.invoke(prompt)

    st.session_state.messages.append(
        {
            "question": question,
            "answer": answer,
            "sources": retrieved_sources
        }
    )

    st.rerun()