"""
Takes a query + retrieved chunks, builds a grounded prompt that forces
the model to (a) cite which chunk each claim comes from, and (b) say
"not found in the provided documents" rather than guessing if the
chunks don't actually answer the question.

Two backends supported:
  - Anthropic API (set ANTHROPIC_API_KEY) — default
  - Ollama, running a local model (set RAG_BACKEND=ollama) — free,
    no API key needed, useful for a fully-local demo

Switch via the RAG_BACKEND environment variable.
"""

import os

BACKEND = os.environ.get("RAG_BACKEND", "anthropic")

SYSTEM_PROMPT = """You are a research assistant answering questions using \
ONLY the provided document excerpts. Follow these rules strictly:

1. Only use information found in the excerpts below. Do not use outside \
knowledge, even if you're confident it's correct.
2. Every claim in your answer must be followed by a citation to the \
excerpt number it came from, like [1] or [2].
3. If the excerpts do not contain enough information to answer the \
question, say exactly: "The provided documents do not contain enough \
information to answer this question." Do not guess or fill gaps with \
general knowledge.
4. Keep the answer concise — a few sentences, not an essay.
"""


def build_prompt(query: str, chunks: list[dict]) -> str:
    excerpt_block = "\n\n".join(
        f"[{i+1}] (from \"{c['metadata']['title']}\"): {c['text']}"
        for i, c in enumerate(chunks)
    )
    return (
        f"Document excerpts:\n\n{excerpt_block}\n\n"
        f"Question: {query}\n\n"
        f"Answer (with citations to excerpt numbers):"
    )


def _generate_anthropic(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _generate_ollama(prompt: str, model: str = "llama3") -> str:
    import requests

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


def generate_answer(query: str, chunks: list[dict]) -> str:
    if len(chunks) == 0:
        return "The provided documents do not contain enough information to answer this question."

    prompt = build_prompt(query, chunks)

    if BACKEND == "ollama":
        return _generate_ollama(prompt)
    return _generate_anthropic(prompt)
