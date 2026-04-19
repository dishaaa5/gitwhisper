# =============================================================
# GitWhisper — api.py
# FastAPI backend — exposes GitWhisper as HTTP endpoints
# =============================================================
#
# CONCEPT: Why FastAPI?
# ---------------------------------------------------------------
# Right now GitWhisper only works in the terminal.
# FastAPI wraps our logic in HTTP endpoints so that:
#   - A frontend (React, Next.js) can call it
#   - Any tool or app can use it via REST API
#   - We can deploy it to the cloud
#
# Endpoints we'll build:
#   POST /ingest        → ingest a GitHub repo
#   POST /chat          → ask a question, get streamed answer
#   GET  /repo/status   → check if a repo is already ingested
#   GET  /health        → check if server is running
# =============================================================

import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

from ingest import ingest, parse_github_url
from chunker import chunk_all
from embedder import embed_chunks, search, chroma_client, get_collection_name
from main import build_context_string, RAG_SYSTEM_PROMPT

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in .env")

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------------
# CONCEPT: FastAPI app
# ---------------------------------------------------------------
# This one line creates the entire web application.
# metadata is shown in the auto-generated docs at /docs
app = FastAPI(
    title="GitWhisper API",
    description="Chat with any GitHub repository using RAG",
    version="1.0.0"
)

# ---------------------------------------------------------------
# CONCEPT: CORS (Cross-Origin Resource Sharing)
# ---------------------------------------------------------------
# By default browsers block requests from one domain to another.
# For example: your frontend on vercel.com calling your backend
# on render.com would be blocked.
#
# CORS middleware tells the browser: "yes, this is allowed."
# allow_origins=["*"] means any frontend can call our API.
# In production you'd restrict this to your frontend's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------
# CONCEPT: Pydantic models
# ---------------------------------------------------------------
# These define the shape of data coming IN to our endpoints.
# FastAPI uses them to:
#   1. Validate incoming requests automatically
#   2. Show in the docs what each endpoint expects
#   3. Give us type hints so our editor helps us

class IngestRequest(BaseModel):
    github_url: str         # e.g. "https://github.com/owner/repo"

class ChatRequest(BaseModel):
    github_url: str         # which repo to chat with
    message: str            # the user's question
    history: list = []      # conversation history so far

# ---------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------
# Tracks which repos have been ingested so we don't
# re-ingest on every chat message.
# Key: "owner/repo", Value: True
ingested_repos: dict = {}


# =============================================================
# ENDPOINTS
# =============================================================

@app.get("/health")
def health():
    """
    Simple health check.
    Returns 200 OK if the server is running.
    Useful for deployment platforms to verify the app is alive.
    """
    return {"status": "ok", "service": "GitWhisper API"}


@app.get("/repo/status")
def repo_status(github_url: str):
    """
    Check if a repo has already been ingested.

    The frontend can call this first to decide whether to
    show a loading screen or go straight to chat.

    Query param: ?github_url=https://github.com/owner/repo
    """
    try:
        owner, repo = parse_github_url(github_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    repo_key = f"{owner}/{repo}"
    collection_name = get_collection_name(owner, repo)

    # Check if collection exists in ChromaDB
    try:
        collection = chroma_client.get_collection(collection_name)
        count = collection.count()
        is_ready = count > 0
    except Exception:
        is_ready = False
        count = 0

    return {
        "repo":     repo_key,
        "ingested": is_ready,
        "chunks":   count
    }


@app.post("/ingest")
def ingest_repo(request: IngestRequest):
    """
    Ingest a GitHub repo — fetch, chunk, and embed all files.

    This is the setup step. Call this once per repo before chatting.
    If the repo was already ingested, it returns immediately.

    Request body:
    {
        "github_url": "https://github.com/owner/repo"
    }

    Response:
    {
        "status": "success",
        "repo": "owner/repo",
        "files": 12,
        "chunks": 47,
        "message": "Repo ingested successfully"
    }
    """
    try:
        owner, repo = parse_github_url(request.github_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    repo_key = f"{owner}/{repo}"

    # Check if already ingested — skip if so
    collection_name = get_collection_name(owner, repo)
    try:
        collection = chroma_client.get_collection(collection_name)
        if collection.count() > 0:
            return {
                "status":  "already_ingested",
                "repo":    repo_key,
                "chunks":  collection.count(),
                "message": "Repo already ingested. Ready to chat."
            }
    except Exception:
        pass  # collection doesn't exist yet, proceed with ingestion

    # Run the full pipeline
    try:
        # 1. Fetch files from GitHub
        files = ingest(request.github_url)

        # 2. Chunk files
        chunks = chunk_all(files)

        # 3. Embed and store
        embed_chunks(chunks, owner, repo)

        ingested_repos[repo_key] = True

        return {
            "status":  "success",
            "repo":    repo_key,
            "files":   len(files),
            "chunks":  len(chunks),
            "message": f"Repo ingested successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
def chat(request: ChatRequest):
    """
    Chat with an ingested repo. Returns a STREAMED response.

    CONCEPT: Streaming HTTP responses
    ---------------------------------------------------------------
    Instead of waiting for the full answer then sending it,
    we stream tokens as they arrive from Groq.
    This makes the frontend feel fast and responsive —
    the user sees words appearing immediately.

    We use Server-Sent Events (SSE) format:
    Each chunk is sent as:  data: {"token": "hello"}\n\n
    A special end marker:   data: {"done": true}\n\n

    The frontend reads these events and appends tokens to the UI.

    Request body:
    {
        "github_url": "https://github.com/owner/repo",
        "message": "how does authentication work?",
        "history": [
            {"role": "user",      "content": "hi"},
            {"role": "assistant", "content": "hello!"}
        ]
    }
    """
    try:
        owner, repo = parse_github_url(request.github_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Make sure repo is ingested
    collection_name = get_collection_name(owner, repo)
    try:
        collection = chroma_client.get_collection(collection_name)
        if collection.count() == 0:
            raise HTTPException(
                status_code=400,
                detail="Repo not ingested yet. Call POST /ingest first."
            )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Repo not ingested yet. Call POST /ingest first."
        )

    def generate():
        """
        Generator function that yields SSE events.

        CONCEPT: Python generators
        ---------------------------------------------------------------
        A generator is a function that yields values one at a time
        instead of returning them all at once.
        'yield' pauses the function, sends a value, then resumes.

        FastAPI's StreamingResponse takes a generator and sends
        each yielded value to the client immediately.
        """
        try:
            # 1. Search for relevant chunks
            chunks = search(request.message, owner, repo, top_k=5)

            # Send sources to frontend first
            sources = [
                {
                    "path":       c["path"],
                    "start_line": c["start_line"],
                    "end_line":   c["end_line"],
                    "name":       c.get("name", ""),
                    "distance":   c["distance"]
                }
                for c in chunks
            ]
            yield f"data: {json.dumps({'sources': sources})}\n\n"

            # 2. Build context from chunks
            context = build_context_string(chunks)

            # 3. Build system prompt with context
            system_prompt = RAG_SYSTEM_PROMPT.format(repo=f"{owner}/{repo}")
            system_prompt += f"\n\nRELEVANT CODE CONTEXT:\n\n{context}"

            # 4. Build messages with history
            messages = [
                {"role": "system", "content": system_prompt},
                *request.history,
                {"role": "user", "content": request.message}
            ]

            # 5. Stream from Groq
            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.3,
                max_tokens=2048,
                stream=True
            )

            # 6. Yield each token as an SSE event
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {json.dumps({'token': delta})}\n\n"

            # 7. Send done signal
            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"   
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)