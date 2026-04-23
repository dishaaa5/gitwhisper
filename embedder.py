# =============================================================
# GitWhisper — embedder.py
# Convert chunks into vectors and store in ChromaDB
# =============================================================

import chromadb
from fastembed import TextEmbedding
import hashlib
import os

# ---------------------------------------------------------------
# CONCEPT: Embeddings
# ---------------------------------------------------------------
# An embedding is a list of floating point numbers (a vector)
# that represents the *meaning* of a piece of text.
#
# Text with similar meaning gets similar vectors.
# This lets us do "semantic search" — find text by meaning,
# not just by matching keywords.
#
# We use fastembed — a lightweight embedding library designed
# for production use with minimal RAM usage (~50MB).
# Runs locally, completely free, no API needed.
#
# Model we use: BAAI/bge-small-en-v1.5
#   - Very small (~50MB), fast, free
#   - Produces 384-dimensional vectors
#   - Great for code and text similarity
#   - Downloaded automatically on first run

print("Loading embedding model...")
embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")
print("Embedding model ready.")

# ---------------------------------------------------------------
# CONCEPT: ChromaDB
# ---------------------------------------------------------------
# ChromaDB is a vector database — it stores embeddings and lets
# you search them by similarity.
#
# It runs entirely locally as a folder on your machine.
# No server, no cloud, no cost.
#
# The folder "chroma_db" will be created in your project directory.
# It persists between runs — you only need to embed a repo once.

CHROMA_DIR = "chroma_db"  # folder where vectors are stored

# Create a persistent ChromaDB client
# PersistentClient saves to disk so data survives restarts
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)


def get_collection_name(owner, repo):
    """
    Each repo gets its own "collection" in ChromaDB.
    A collection is like a table — it holds all the embeddings for one repo.

    Collection names must be URL-safe so we format them cleanly.
    Example: "openai/gpt-2" -> "openai_gpt-2"
    """
    return f"{owner}_{repo}".replace("/", "_").replace(".", "_").lower()


def embed_chunks(chunks, owner, repo):
    """
    Take a list of chunks, embed them, and store in ChromaDB.

    This is the most important function in this file.
    It only needs to run ONCE per repo — results are saved to disk.

    Args:
        chunks: list of chunk dicts from chunker.py
        owner:  GitHub repo owner (e.g. "openai")
        repo:   GitHub repo name  (e.g. "gpt-2")
    """
    collection_name = get_collection_name(owner, repo)

    # ---------------------------------------------------------------
    # CONCEPT: Collections
    # ---------------------------------------------------------------
    # get_or_create_collection does what it says:
    #   - If collection exists (from a previous run): returns it
    #   - If not: creates a new empty one
    #
    # This means running embed_chunks twice on the same repo
    # won't duplicate data — it just reuses the existing collection.
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"owner": owner, "repo": repo}
    )

    # Check if already embedded (skip if so)
    existing_count = collection.count()
    if existing_count > 0:
        print(f"Collection '{collection_name}' already has {existing_count} chunks.")
        print("Skipping embedding — delete 'chroma_db/' folder to re-embed.")
        return collection

    print(f"Embedding {len(chunks)} chunks...")
    print("(This may take a minute on first run)")

    # ---------------------------------------------------------------
    # CONCEPT: Batch processing
    # ---------------------------------------------------------------
    # Embedding one chunk at a time would be slow.
    # We batch them — process many at once for speed.
    # batch_size=32 means we embed 32 chunks per call.
    batch_size = 32

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]

        # Extract just the text content from each chunk
        texts = [chunk["content"] for chunk in batch]

        # Generate embeddings — this is where the ML model runs
        # fastembed.embed() returns a generator of numpy arrays
        # we convert to a list so we can index into it
        embeddings = list(embedding_model.embed(texts))

        # ---------------------------------------------------------------
        # CONCEPT: ChromaDB document format
        # ---------------------------------------------------------------
        # ChromaDB needs three things per item:
        #
        # 1. ids        — unique string ID for each chunk
        # 2. embeddings — the actual vectors (lists of floats)
        # 3. documents  — the original text (stored for retrieval)
        # 4. metadatas  — any extra info we want to store alongside
        #
        # We store path, line numbers etc. in metadata so we can
        # show the user WHERE in the code the answer came from.

        ids = []
        metadatas = []

        for chunk in batch:
            # Create a unique ID by hashing path + chunk_index
            # This ensures IDs are always the same for the same chunk
            unique_str = f"{chunk['path']}_{chunk['chunk_index']}"
            chunk_id = hashlib.md5(unique_str.encode()).hexdigest()

            ids.append(chunk_id)
            metadatas.append({
                "path":        chunk["path"],
                "start_line":  chunk["start_line"],
                "end_line":    chunk["end_line"],
                "type":        chunk.get("type", "lines"),
                "name":        chunk.get("name", ""),
            })

        # Add to ChromaDB
        collection.add(
            ids=ids,
            embeddings=[e.tolist() for e in embeddings],  # numpy -> list
            documents=texts,
            metadatas=metadatas
        )

        # Progress indicator
        done = min(i + batch_size, len(chunks))
        print(f"  Embedded {done}/{len(chunks)} chunks...", end="\r")

    print(f"\nDone! {len(chunks)} chunks stored in '{CHROMA_DIR}/{collection_name}'")
    return collection


def search(query, owner, repo, top_k=5):
    """
    Search for the most relevant chunks for a given query.

    This is what runs every time the user asks a question.

    Args:
        query:  the user's question as a string
        owner:  GitHub repo owner
        repo:   GitHub repo name
        top_k:  how many chunks to return (default 5)

    Returns:
        list of dicts, each with 'content' and 'metadata'
    """
    collection_name = get_collection_name(owner, repo)

    # Get the collection
    try:
        collection = chroma_client.get_collection(collection_name)
    except Exception:
        raise ValueError(
            f"Repo '{owner}/{repo}' not embedded yet. "
            "Run embed_chunks() first."
        )

    # ---------------------------------------------------------------
    # CONCEPT: Semantic search
    # ---------------------------------------------------------------
    # 1. Embed the query using the same model
    # 2. Compare query vector to all stored chunk vectors
    # 3. Return the top_k most similar chunks
    #
    # "Similar" = small cosine distance between vectors
    # ChromaDB does all this math internally

    query_embedding = list(embedding_model.embed([query]))[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # Format results into clean dicts
    chunks = []
    documents = results["documents"][0]    # list of chunk texts
    metadatas = results["metadatas"][0]    # list of metadata dicts
    distances = results["distances"][0]    # list of similarity scores

    for doc, meta, dist in zip(documents, metadatas, distances):
        chunks.append({
            "content":    doc,
            "path":       meta["path"],
            "start_line": meta["start_line"],
            "end_line":   meta["end_line"],
            "type":       meta["type"],
            "name":       meta.get("name", ""),
            "distance":   round(dist, 4)   # lower = more similar
        })

    return chunks


def print_search_results(results, query):
    """Pretty print search results."""
    print(f"\nTop results for: '{query}'")
    print("=" * 50)
    for i, r in enumerate(results):
        name = f" — {r['name']}()" if r["name"] else ""
        print(f"\n[{i+1}] {r['path']}{name}")
        print(f"     Lines {r['start_line']}–{r['end_line']} | distance: {r['distance']}")
        print(f"     {r['content'][:120].strip()}...")
    print("=" * 50)


# ---------------------------------------------------------------
# Test it directly
# ---------------------------------------------------------------
if __name__ == "__main__":
    from ingest import ingest, parse_github_url
    from chunker import chunk_all

    url = input("Enter a GitHub repo URL: ").strip()
    if not url:
        url = "https://github.com/realpython/reader"
        print(f"Using default: {url}")

    # Ingest
    owner, repo = parse_github_url(url)
    files = ingest(url)

    # Chunk
    print("\nChunking...")
    chunks = chunk_all(files)
    print(f"Total chunks: {len(chunks)}")

    # Embed and store
    print("\nEmbedding and storing...")
    embed_chunks(chunks, owner, repo)

    # Test search
    print("\n" + "=" * 50)
    print("Now let's test search!")
    print("=" * 50)

    while True:
        query = input("\nSearch query (or 'exit'): ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        results = search(query, owner, repo)
        print_search_results(results, query)