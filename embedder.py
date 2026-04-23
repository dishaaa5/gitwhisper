import chromadb
from fastembed import TextEmbedding
import hashlib
import os


print("Loading embedding model...")
embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")
print("Embedding model ready.")


CHROMA_DIR = "chroma_db"  

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

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"owner": owner, "repo": repo}
    )

    existing_count = collection.count()
    if existing_count > 0:
        print(f"Collection '{collection_name}' already has {existing_count} chunks.")
        print("Skipping embedding — delete 'chroma_db/' folder to re-embed.")
        return collection

    print(f"Embedding {len(chunks)} chunks...")
    print("(This may take a minute on first run)")

    batch_size = 32

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]

        texts = [chunk["content"] for chunk in batch]

        embeddings = list(embedding_model.embed(texts))
     
        ids = []
        metadatas = []

        for chunk in batch:
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

        collection.add(
            ids=ids,
            embeddings=[e.tolist() for e in embeddings], 
            documents=texts,
            metadatas=metadatas
        )

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

    try:
        collection = chroma_client.get_collection(collection_name)
    except Exception:
        raise ValueError(
            f"Repo '{owner}/{repo}' not embedded yet. "
            "Run embed_chunks() first."
        )

    query_embedding = list(embedding_model.embed([query]))[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    documents = results["documents"][0]    
    metadatas = results["metadatas"][0]    
    distances = results["distances"][0]    

    for doc, meta, dist in zip(documents, metadatas, distances):
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
    """Pretty print search results."""
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

    print("\nEmbedding and storing...")
    embed_chunks(chunks, owner, repo)

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