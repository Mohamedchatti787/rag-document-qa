# RAG Document Q&A — Grounded Chat Over Your Own Documents

Ask questions in natural language, get answers grounded in a specific set
of documents (arXiv paper abstracts by default) — with the exact source
chunks shown alongside every answer, so you can check whether the model
is actually right or just sounds right.

## Why this is more than "call an LLM API"

Anyone can wrap an API call in a chat box. What makes this a real
engineering project is the pieces most tutorials skip:

1. **Retrieval you can actually inspect** — every answer shows which
   chunks it was built from, so wrong answers are debuggable (was it a
   retrieval miss, or did the model ignore good context and hallucinate
   anyway?).
2. **An eval set** — a hand-written set of Q&A pairs with known correct
   answers, scored for both retrieval quality (did we fetch the right
   chunk?) and answer groundedness (does the answer actually follow from
   what was retrieved?). This is the single biggest differentiator
   between a toy RAG demo and something that shows you understand how
   these systems actually fail in production.
3. **A cheap, swappable generation backend** — works with the Anthropic
   API or a fully local model via Ollama, so the whole thing can run
   with zero API cost if you want a truly free-to-clone demo.

## Architecture

```
documents  --chunk-->  text chunks  --embed-->  vectors  -->  ChromaDB
                                                                  |
query  --embed-->  query vector  ---similarity search------------+
                                          |
                                   top-k chunks
                                          |
                                    prompt + chunks --> LLM --> answer
                                                                  |
                                          answer + source chunks shown to user
```

## Dataset

Default: [arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)
(Cornell University, via Kaggle) — using just the `title` + `abstract`
fields for a manageable subset (e.g. filter to `cs.CL` or `cs.LG`
categories, a few thousand papers, so ingestion runs in minutes not
hours).

```bash
kaggle datasets download -d Cornell-University/arxiv -p data/ --unzip
```

`src/ingest.py` includes a filter step so you don't have to embed all
2.7M+ papers — pick a category and a row limit that suits your machine.

## Project structure

```
rag-document-qa/
├── data/                    # raw arxiv-metadata-oai-snapshot.json goes here
├── src/
│   ├── ingest.py            # load, chunk, embed, store in ChromaDB
│   ├── retrieve.py          # similarity search against the vector store
│   ├── generate.py          # grounded-answer prompt + LLM call (Anthropic or Ollama)
│   └── eval.py               # retrieval + groundedness evaluation
├── app/
│   └── app.py                # Streamlit chat UI with source-chunk display
├── eval/
│   └── qa_test_set.json      # hand-written eval questions + expected answers
├── requirements.txt
└── README.md
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. put your Anthropic API key in the environment (or skip this and
#    use Ollama locally — see generate.py for the switch)
export ANTHROPIC_API_KEY=your_key_here

# 2. ingest documents into the vector store
python src/ingest.py --category cs.CL --limit 5000

# 3. run the eval to check retrieval + groundedness before demoing
python src/eval.py

# 4. launch the chat app
streamlit run app/app.py
```



## Suggested next steps once the base version works

- Swap the embedding model for a domain-specific one and compare
  retrieval quality (`all-MiniLM-L6-v2` vs. `multi-qa-mpnet-base-dot-v1`)
- Add a "no answer found" path — if retrieved chunks don't actually
  support any answer, the system should say so instead of forcing a
  response. This is the most-cited real-world RAG failure mode, and
  handling it explicitly is a strong signal.
- Try re-ranking: retrieve top-20 with the fast embedding model, then
  re-rank to top-5 with a cross-encoder before generation
- Deploy the Streamlit app to Streamlit Community Cloud, link the live
  demo here
