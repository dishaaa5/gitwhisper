import requests  
import os
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  
ALLOWED_EXTENSIONS = {
    ".py",    
    ".js",    
    ".ts",    
    ".jsx",   
    ".tsx",   
    ".java", 
    ".c",     
    ".cpp",   
    ".go",    
    ".rs",    
    ".rb",    
    ".php",  
    ".swift", 
    ".kt",    
    ".md",    
    ".txt",   
    ".env.example",  
    ".json",  
    ".yaml",  
    ".yml",     
    ".toml",    
    ".sh",     
    ".html",  
    ".css",   
}

SKIP_FILES = {
    "package-lock.json",
    "yarn.lock",
    "poetry.lock",
    "Pipfile.lock",
    ".DS_Store",
}


SKIP_FOLDERS = {
    "node_modules",
    ".git",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    "dist",
    "build",
    ".next",
}


MAX_FILE_SIZE = 50_000  # 50KB

def parse_github_url(url):
    """
    Extract owner and repo name from a GitHub URL.

    Examples:
        https://github.com/openai/gpt-2        -> ("openai", "gpt-2")
        https://github.com/torvalds/linux/      -> ("torvalds", "linux")
        github.com/psf/black                   -> ("psf", "black")
    """
    url = url.strip().rstrip("/")
    url = url.replace("https://", "").replace("http://", "")
    url = url.replace("github.com/", "")

    parts = url.split("/")

    if len(parts) < 2:
        raise ValueError(f"Invalid GitHub URL: {url}\nExpected format: https://github.com/owner/repo")

    owner = parts[0]
    repo = parts[1]

    return owner, repo

def get_headers():
    """
    Build request headers.
    If a GitHub token is available, include it for higher rate limits.
    """
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
        print("  Using GitHub token (5000 req/hour limit)")
    else:
        print("  No GitHub token — using public limit (60 req/hour)")
        print("  Tip: Add GITHUB_TOKEN to .env for higher limits")
    return headers


def get_file_tree(owner, repo, headers):
    """
    Fetch the complete file tree of a repo using GitHub's git trees API.

    The ?recursive=1 parameter gets ALL files in ALL folders in one request.
    Without it, you'd have to make a separate request for each folder.

    Returns a list of file objects like:
    [
        {"path": "src/main.py", "type": "blob", "size": 1234, "sha": "abc..."},
        {"path": "README.md",   "type": "blob", "size": 567,  "sha": "def..."},
        ...
    ]
    "blob" = file, "tree" = folder
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD?recursive=1"
    print(f"  Fetching file tree from: {url}")

    response = requests.get(url, headers=headers)

    if response.status_code == 404:
        raise ValueError(f"Repo not found: github.com/{owner}/{repo}\nIs it public?")
    elif response.status_code == 403:
        raise ValueError("Rate limited by GitHub. Add a GITHUB_TOKEN to .env")
    elif response.status_code != 200:
        raise ValueError(f"GitHub API error {response.status_code}: {response.text}")

    data = response.json()
    if data.get("truncated"):
        print("  WARNING: Repo is very large — only partial tree returned")

    return data.get("tree", [])


def should_include_file(file_obj):
    """
    Decide whether a file is worth downloading.

    Returns True if we want it, False if we skip it.
    """
    path = file_obj.get("path", "")
    size = file_obj.get("size", 0)
    file_type = file_obj.get("type", "")

    if file_type != "blob":
        return False

    path_parts = path.split("/")
    for part in path_parts[:-1]:  
        if part in SKIP_FOLDERS:
            return False

    filename = path_parts[-1]

    if filename in SKIP_FILES:
        return False

    _, ext = os.path.splitext(filename)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        return False

    if size > MAX_FILE_SIZE:
        print(f"  Skipping large file: {path} ({size} bytes)")
        return False

    return True

def download_file(owner, repo, path, headers):
    """
    Download the raw content of a single file.

    We use raw.githubusercontent.com which serves file content directly.
    This is faster and simpler than the API endpoint for raw content.
    """

    for branch in ["main", "master"]:
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return response.text  

    return None


def ingest(github_url):
    """
    Main function — takes a GitHub URL, returns a list of file dicts.

    Each dict looks like:
    {
        "path":    "src/utils.py",
        "content": "def helper():\n    ..."
    }

    This list is what we'll feed into the AI in later steps.
    """
    print(f"\nIngesting repo: {github_url}")
    print("-" * 50)

    owner, repo = parse_github_url(github_url)
    print(f"  Owner: {owner}")
    print(f"  Repo:  {repo}")

    headers = get_headers()

    tree = get_file_tree(owner, repo, headers)
    print(f"  Total items in repo: {len(tree)}")

    files_to_fetch = [f for f in tree if should_include_file(f)]
    print(f"  Files to download:   {len(files_to_fetch)}")
    print()

    files = []
    for i, file_obj in enumerate(files_to_fetch):
        path = file_obj["path"]
        print(f"  [{i+1}/{len(files_to_fetch)}] {path}", end=" ")

        content = download_file(owner, repo, path, headers)

        if content:
            files.append({
                "path": path,
                "content": content
            })
            print("✓")
        else:
            print("✗ (skipped)")

    print()
    print(f"Successfully ingested {len(files)} files from {owner}/{repo}")
    return files


def print_summary(files):
    """Print a summary of what was ingested."""
    print("\n" + "=" * 50)
    print("  INGESTION SUMMARY")
    print("=" * 50)

    total_chars = sum(len(f["content"]) for f in files)
    total_lines = sum(f["content"].count("\n") for f in files)

    print(f"  Files:      {len(files)}")
    print(f"  Total lines: {total_lines:,}")
    print(f"  Total chars: {total_chars:,}")
    print()
    print("  Files ingested:")
    for f in files:
        lines = f["content"].count("\n")
        print(f"    {f['path']} ({lines} lines)")
    print("=" * 50)

if __name__ == "__main__":
    test_url = input("Enter a GitHub repo URL to test: ").strip()

    if not test_url:
        test_url = "https://github.com/realpython/reader"
        print(f"Using default: {test_url}")

    files = ingest(test_url)
    print_summary(files)

    if files:
        print(f"\nPreview of first file — {files[0]['path']}:")
        print("-" * 40)
        print(files[0]["content"][:500])
        print("...")