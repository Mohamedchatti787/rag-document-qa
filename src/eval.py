"""
Evaluates the RAG pipeline against a hand-written test set.

Two things are measured:
  1. Retrieval hit rate: for each test question, does the expected
     paper (by paper_id) actually show up in the top-k retrieved chunks?
  2. Answer groundedness: printed for manual review (or optionally
     scored by an LLM judge — see judge_groundedness() below) — does the
     generated answer actually follow from what was retrieved, or does
     it drift into unsupported claims despite having the right context?

Usage:
    python src/eval.py
"""

import json

from retrieve import retrieve
from generate import generate_answer

TEST_SET_PATH = "eval/qa_test_set.json"


def load_test_set():
    with open(TEST_SET_PATH, "r") as f:
        return json.load(f)


def retrieval_hit(expected_paper_id: str, chunks: list[dict]) -> bool:
    return any(c["metadata"]["paper_id"] == expected_paper_id for c in chunks)


def judge_groundedness(question: str, answer: str, chunks: list[dict]) -> str:
    """
    Optional: use the LLM itself as a judge to flag whether the answer
    is actually supported by the retrieved chunks. This catches the
    failure mode where retrieval succeeds but the model still drifts
    into unsupported claims. Uses the same generate.py backend, so it
    respects RAG_BACKEND too.
    """
    from generate import _generate_anthropic  # reuse the same backend call

    excerpt_block = "\n\n".join(c["text"] for c in chunks)
    judge_prompt = (
        f"Excerpts:\n{excerpt_block}\n\n"
        f"Question: {question}\n"
        f"Answer given: {answer}\n\n"
        f"Is every claim in the answer directly supported by the excerpts? "
        f"Reply with exactly one word: SUPPORTED, PARTIALLY_SUPPORTED, or "
        f"UNSUPPORTED, followed by a one-sentence reason."
    )
    return _generate_anthropic(judge_prompt)


def run_eval(top_k: int = 5, use_llm_judge: bool = False):
    test_set = load_test_set()
    hits = 0

    for i, item in enumerate(test_set, 1):
        question = item["question"]
        expected_paper_id = item["expected_paper_id"]

        chunks = retrieve(question, top_k=top_k)
        hit = retrieval_hit(expected_paper_id, chunks)
        hits += int(hit)

        print(f"\n[{i}] Q: {question}")
        print(f"    retrieval hit: {hit}")

        answer = generate_answer(question, chunks)
        print(f"    answer: {answer}")

        if use_llm_judge:
            verdict = judge_groundedness(question, answer, chunks)
            print(f"    groundedness judge: {verdict}")

    hit_rate = hits / len(test_set)
    print(f"\n=== Retrieval hit rate: {hit_rate:.1%} ({hits}/{len(test_set)}) ===")
    print("Copy this into the README results table, along with a manual")
    print("read of the groundedness column above.")


if __name__ == "__main__":
    run_eval(use_llm_judge=True)
