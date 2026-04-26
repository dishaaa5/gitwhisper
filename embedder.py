# =============================================================
# GitWhisper — embedder.py
# Convert chunks into vectors and store in ChromaDB
# Uses ChromaDB's built-in embeddings — no extra libraries needed
# =============================================================

import chromadb
import hashlib
import os

# ---------------------------------------------------------------
# CONCEPT: ChromaDB default embeddings
# ---------------------------------------------------------------
# ChromaDB has a built-in embedding function that uses
# "all-MiniLM-L6-v2" under the hood via its own lightweight
# implementation. We don't need sentence-transformers or
# any HuggingFace API — ChromaDB handles it all internally.
#
# This means:
#   - No separate embedding library needed
#   - Model is tiny and loads fast (~45MB)
#   - Works offline, completely free
#   - Same quality as sentence-transformers

CHROMA_DIR = "chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)


def get_collection_name(owner, repo):
    """Each repo gets its own ChromaDB collection."""
    return f"{owner}_{repo}".replace("/", "_").replace(".", "_").lower()


def embed_chunks(chunks, owner, repo):
    """
    Embed all chunks and store in ChromaDB.
    ChromaDB handles embeddings automatically.
    Only runs once per repo — results persist to disk.
    """
    collection_name = get_collection_name(owner, repo)

    # ---------------------------------------------------------------
    # CONCEPT: Default embedding function
    # ---------------------------------------------------------------
    # By NOT passing an embedding_function, ChromaDB uses its
    # default: "all-MiniLM-L6-v2" via chromadb.utils.embedding_functions
    # It downloads the model once and caches it locally.
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"owner": owner, "repo": repo}
    )

    # Skip if already embedded
    existing_count = collection.count()
    if existing_count > 0:
        print(f"Already embedded ({existing_count} chunks). Skipping.")
        return collection

    print(f"Embedding {len(chunks)} chunks...")
    print("(Downloading embedding model on first run — one time only)")

    # Process in batches
    batch_size = 32

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        texts = [chunk["content"] for chunk in batch]

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

        # ChromaDB embeds the documents automatically
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )

        done = min(i + batch_size, len(chunks))
        print(f"  Embedded {done}/{len(chunks)} chunks...", end="\r")

    print(f"\nDone! {len(chunks)} chunks stored.")
    return collection


def search(query, owner, repo, top_k=5):
    """
    Find the most relevant chunks for a query.
    ChromaDB embeds the query automatically using the same model.
    """
    collection_name = get_collection_name(owner, repo)

    try:
        collection = chroma_client.get_collection(collection_name)
    except Exception:
        raise ValueError("Repo not embedded yet. Call embed_chunks() first.")

    # ChromaDB embeds the query text automatically
    results = collection.query(
        query_texts=[query],
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