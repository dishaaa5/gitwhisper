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

app = FastAPI(
    title="GitWhisper API",
    description="Chat with any GitHub repository using RAG",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestRequest(BaseModel):
    github_url: str         

class ChatRequest(BaseModel):
    github_url: str         
    message: str            
    history: list = []      

ingested_repos: dict = {}

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
        pass  

    try:
        files = ingest(request.github_url)

        chunks = chunk_all(files)

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
            
            chunks = search(request.message, owner, repo, top_k=5)

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

            context = build_context_string(chunks)

            system_prompt = RAG_SYSTEM_PROMPT.format(repo=f"{owner}/{repo}")
            system_prompt += f"\n\nRELEVANT CODE CONTEXT:\n\n{context}"

            messages = [
                {"role": "system", "content": system_prompt},
                *request.history,
                {"role": "user", "content": request.message}
            ]

            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.3,
                max_tokens=2048,
                stream=True
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {json.dumps({'token': delta})}\n\n"

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