"""
Streamlit chat app: ask a question, see the grounded answer plus the
exact source chunks it was built from.

Run with:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from retrieve import retrieve
from generate import generate_answer

st.set_page_config(page_title="RAG Document Q&A", layout="centered")
st.title("RAG Document Q&A")
st.write(
    "Ask a question about the ingested documents. The answer is grounded "
    "only in what's retrieved below — if the documents don't support an "
    "answer, the model is instructed to say so rather than guess."
)

top_k = st.sidebar.slider("Chunks to retrieve (top-k)", min_value=1, max_value=10, value=5)

if "history" not in st.session_state:
    st.session_state.history = []

query = st.chat_input("Ask a question...")

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])

if query:
    with st.chat_message("user"):
        st.write(query)
    st.session_state.history.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        try:
            chunks = retrieve(query, top_k=top_k)
        except Exception as e:
            st.error(
                f"Couldn't reach the vector store — have you run "
                f"`python src/ingest.py` yet? ({e})"
            )
            st.stop()

        answer = generate_answer(query, chunks)
        st.write(answer)
        st.session_state.history.append({"role": "assistant", "content": answer})

        with st.expander(f"Show {len(chunks)} retrieved source chunks"):
            for i, chunk in enumerate(chunks, 1):
                st.markdown(f"**[{i}] {chunk['metadata']['title']}**  "
                            f"(distance: {chunk['distance']:.3f})")
                st.text(chunk["text"][:500] + ("..." if len(chunk["text"]) > 500 else ""))
                st.divider()
