"""
Loads arXiv paper metadata, filters to a category, chunks the abstracts,
embeds them, and stores them in a local ChromaDB collection.

Expects the Kaggle arXiv dataset JSON at:
    data/arxiv-metadata-oai-snapshot.json

Each line in that file is one JSON object with fields including:
    id, title, abstract, categories, authors, update_date

Usage:
    python src/ingest.py --category cs.CL --limit 5000
"""

import argparse
import json
import os

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATA_PATH = "data/arxiv-metadata-oai-snapshot.json"
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "arxiv_papers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Simple chunking: title + abstract as one chunk. Abstracts are short
# enough (a few hundred words) that further splitting isn't usually
# necessary — but chunk_text() below is here so you can plug in
# longer documents later without restructuring anything.
CHUNK_SIZE_WORDS = 200
CHUNK_OVERLAP_WORDS = 40


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS,
               overlap: int = CHUNK_OVERLAP_WORDS):
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def load_papers(category: str, limit: int):
    papers = []
    with open(DATA_PATH, "r") as f:
        for line in f:
            if len(papers) >= limit:
                break
            record = json.loads(line)
            if category in record.get("categories", ""):
                papers.append(record)
    return papers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="cs.CL",
                         help="arXiv category to filter to, e.g. cs.CL, cs.LG, cs.CV")
    parser.add_argument("--limit", type=int, default=5000,
                         help="max number of papers to ingest")
    args = parser.parse_args()

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Download it first:\n"
            f"  kaggle datasets download -d Cornell-University/arxiv -p data/ --unzip"
        )

    print(f"Loading papers in category '{args.category}' (limit={args.limit})...")
    papers = load_papers(args.category, args.limit)
    print(f"Loaded {len(papers)} papers.")

    if len(papers) == 0:
        raise RuntimeError(
            f"No papers found for category '{args.category}'. Check the "
            f"category name matches arXiv's taxonomy (e.g. cs.CL, cs.LG, cs.CV, stat.ML)."
        )

    print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    # start clean each run so re-ingesting doesn't duplicate entries
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    ids, documents, metadatas = [], [], []

    for paper in tqdm(papers, desc="Chunking"):
        full_text = f"{paper['title'].strip()}\n\n{paper['abstract'].strip()}"
        chunks = chunk_text(full_text)
        for i, chunk in enumerate(chunks):
            ids.append(f"{paper['id']}_{i}")
            documents.append(chunk)
            metadatas.append({
                "paper_id": paper["id"],
                "title": paper["title"].strip(),
                "categories": paper.get("categories", ""),
                "chunk_index": i,
            })

    print(f"Embedding {len(documents)} chunks...")
    embeddings = embedder.encode(documents, show_progress_bar=True, batch_size=64)

    print("Writing to ChromaDB...")
    batch_size = 500
    for start in tqdm(range(0, len(ids), batch_size), desc="Storing"):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end].tolist(),
        )

    print(f"\nDone. Collection '{COLLECTION_NAME}' has {collection.count()} chunks "
          f"stored at {CHROMA_DIR}.")


if __name__ == "__main__":
    main()
