import os
from groq import Groq
from dotenv import load_dotenv

from ingest import ingest, parse_github_url
from chunker import chunk_all
from embedder import embed_chunks, search

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY not found in .env")
    exit(1)

client = Groq(api_key=GROQ_API_KEY)


RAG_SYSTEM_PROMPT = """You are GitWhisper, an AI assistant that helps 
developers understand code repositories.

You will be given relevant code snippets from the repository as context.
Use this context to answer the user's question accurately.

Rules:
- Base your answers on the provided code context
- Always mention which file and line numbers your answer comes from
- If the context doesn't contain enough info, say so honestly
- Keep answers concise and technical
- Use markdown formatting for code blocks

Repository: {repo}"""


def build_context_string(chunks):
    """
    Format retrieved chunks into a readable string for the prompt.

    This is what gets injected into the system prompt as context.
    We format it clearly so the model knows exactly where each
    piece of code comes from.

    Example output:
    --- File: src/auth.py (lines 12-34) ---
    def login(username, password):
        ...
    --------------------------------------------------------
    """
    if not chunks:
        return "No relevant code found for this query."

    parts = []
    for chunk in chunks:
        name = f" [{chunk['name']}]" if chunk.get("name") else ""
        header = f"--- File: {chunk['path']}{name} (lines {chunk['start_line']}–{chunk['end_line']}) ---"
        parts.append(f"{header}\n{chunk['content']}\n{'-' * 56}")

    return "\n\n".join(parts)


def chat_with_context(user_message, history, owner, repo):
    """
    The core RAG function.

    1. Search for relevant chunks
    2. Build context string from chunks
    3. Call Groq with context + history + question
    4. Return reply + the chunks used (for transparency)

    Args:
        user_message: what the user asked
        history:      full conversation history so far
        owner:        repo owner
        repo:         repo name

    Returns:
        (reply_text, chunks_used)
    """
    print("\n  Searching codebase...", end=" ", flush=True)
    chunks = search(user_message, owner, repo, top_k=5)
    print(f"found {len(chunks)} relevant chunks")

    
    context = build_context_string(chunks)

    system_prompt = RAG_SYSTEM_PROMPT.format(repo=f"{owner}/{repo}")
    system_prompt += f"\n\nRELEVANT CODE CONTEXT:\n\n{context}"

    
    history.append({
        "role": "user",
        "content": user_message
    })

    
    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            *history
        ],
        temperature=0.3,   
        max_tokens=2048,   
        stream=True        
    )

    
    print("\nGitWhisper: ", end="", flush=True)
    full_reply = ""

    for chunk in stream:
        
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            full_reply += delta

    print("\n")  

    
    history.append({
        "role": "assistant",
        "content": full_reply
    })

    return full_reply, chunks


def setup_repo(github_url):
    """
    Full setup pipeline for a repo:
    ingest → chunk → embed

    Returns (owner, repo) so we know what to search later.
    Skips embedding if already done (ChromaDB persists to disk).
    """
    print(f"\nSetting up repo: {github_url}")
    print("=" * 50)

    # Parse URL
    owner, repo = parse_github_url(github_url)

    # Ingest files
    print("\n[1/3] Fetching files from GitHub...")
    files = ingest(github_url)
    print(f"      Got {len(files)} files")

    # Chunk files
    print("\n[2/3] Chunking files...")
    chunks = chunk_all(files)
    print(f"      Created {len(chunks)} chunks")

    # Embed and store
    print("\n[3/3] Embedding chunks into vector store...")
    embed_chunks(chunks, owner, repo)

    print(f"\nRepo ready! You can now chat with {owner}/{repo}")
    return owner, repo


def main():
    print()
    print("=" * 50)
    print("  GitWhisper — Chat with any GitHub Repo")
    print("=" * 50)
    print()

    
    github_url = input("Enter GitHub repo URL: ").strip()
    if not github_url:
        github_url = "https://github.com/realpython/reader"
        print(f"Using default: {github_url}")

    
    try:
        owner, repo = setup_repo(github_url)
    except Exception as e:
        print(f"\nSetup failed: {e}")
        return

    history = []

    print()
    print("=" * 50)
    print(f"  Chatting with: {owner}/{repo}")
    print("  Commands: 'exit' to quit | 'clear' to reset | 'sources' to see last chunks")
    print("=" * 50)
    print()

    last_chunks = []

    while True:
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            history.clear()
            print("Conversation cleared.\n")
            continue

        if user_input.lower() == "sources":
            
            if not last_chunks:
                print("No sources yet — ask a question first.\n")
            else:
                print("\nSources used in last answer:")
                for i, c in enumerate(last_chunks):
                    name = f" [{c['name']}]" if c.get("name") else ""
                    print(f"  {i+1}. {c['path']}{name} "
                          f"(lines {c['start_line']}–{c['end_line']}) "
                          f"distance: {c['distance']}")
                print()
            continue

        try:
            reply, last_chunks = chat_with_context(
                user_input, history, owner, repo
            )
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()