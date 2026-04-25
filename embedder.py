import chromadb
import hashlib
import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()
## HF failing

HF_TOKEN = os.getenv("HF_TOKEN")
HF_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

if not HF_TOKEN:
    print("WARNING: HF_TOKEN not found in .env")
    print("Get a free token at huggingface.co -> Settings -> Access Tokens")

CHROMA_DIR = "chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)


def get_embeddings(texts: list) -> list:
    """
    Call HuggingFace Inference API to get embeddings for a list of texts.

    CONCEPT: API-based embeddings
    ---------------------------------------------------------------
    Instead of loading a model into RAM, we send texts to HF's
    servers and get back vectors. Same result, zero local memory.

    The API expects:  {"inputs": ["text1", "text2", ...]}
    And returns:      [[0.1, 0.2, ...], [0.3, 0.4, ...]]
    """
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    response = requests.post(
        HF_API_URL,
        headers=headers,
        json={"inputs": texts, "options": {"wait_for_model": True}}
    )

    if response.status_code == 503:
        wait_time = response.json().get("estimated_time", 20)
        print(f"  HF model loading, waiting {wait_time:.0f}s...")
        time.sleep(min(wait_time, 30))
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": texts}
        )

    if response.status_code != 200:
        raise ValueError(f"HuggingFace API error {response.status_code}: {response.text}")

    return response.json()


def get_collection_name(owner, repo):
    """Each repo gets its own ChromaDB collection."""
    return f"{owner}_{repo}".replace("/", "_").replace(".", "_").lower()


def embed_chunks(chunks, owner, repo):
    """
    Embed all chunks and store in ChromaDB.
    Only runs once per repo — results persist to disk.
    """
    collection_name = get_collection_name(owner, repo)

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"owner": owner, "repo": repo}
    )

    existing_count = collection.count()
    if existing_count > 0:
        print(f"Already embedded ({existing_count} chunks). Skipping.")
        return collection

    print(f"Embedding {len(chunks)} chunks via HuggingFace API...")

    batch_size = 16

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        texts = [chunk["content"] for chunk in batch]

        try:
            embeddings = get_embeddings(texts)
        except Exception as e:
            print(f"  Error on batch {i}: {e}")
            continue

        ids = []
        metadatas = []

        for chunk in batch:
            unique_str = f"{chunk['path']}_{chunk['chunk_index']}"
            chunk_id = hashlib.md5(unique_str.encode()).hexdigest()
            ids.append(chunk_id)
            metadatas.append({
                "path":       chunk["path"],
                "start_line": chunk["start_line"],
                "end_line":   chunk["end_line"],
                "type":       chunk.get("type", "lines"),
                "name":       chunk.get("name", ""),
            })

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        done = min(i + batch_size, len(chunks))
        print(f"  Embedded {done}/{len(chunks)} chunks...", end="\r")

    print(f"\nDone! {len(chunks)} chunks stored.")
    return collection


def search(query, owner, repo, top_k=5):
    """
    Find the most relevant chunks for a query using semantic search.
    """
    collection_name = get_collection_name(owner, repo)

    try:
        collection = chroma_client.get_collection(collection_name)
    except Exception:
        raise ValueError("Repo not embedded yet. Call embed_chunks() first.")

    query_embedding = get_embeddings([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "content":    doc,
            "path":       meta["path"],
            "start_line": meta["start_line"],
            "end_line":   meta["end_line"],
            "type":       meta["type"],
            "name":       meta.get("name", ""),
            "distance":   round(dist, 4)
        })

    return chunks


def print_search_results(results, query):
    print(f"\nTop results for: '{query}'")
    print("=" * 50)
    for i, r in enumerate(results):
        name = f" — {r['name']}()" if r["name"] else ""
        print(f"\n[{i+1}] {r['path']}{name}")
        print(f"     Lines {r['start_line']}–{r['end_line']} | distance: {r['distance']}")
        print(f"     {r['content'][:120].strip()}...")
    print("=" * 50)


if __name__ == "__main__":
    from ingest import ingest, parse_github_url
    from chunker import chunk_all

    url = input("Enter a GitHub repo URL: ").strip()
    if not url:
        url = "https://github.com/realpython/reader"
        print(f"Using default: {url}")

    owner, repo = parse_github_url(url)
    files = ingest(url)

    print("\nChunking...")
    chunks = chunk_all(files)
    print(f"Total chunks: {len(chunks)}")

    print("\nEmbedding...")
    embed_chunks(chunks, owner, repo)

    print("\nTest search:")
    while True:
        query = input("\nSearch query (or 'exit'): ").strip()
        if query.lower() == "exit":
            break
        results = search(query, owner, repo)
        print_search_results(results, query)