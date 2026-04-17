

import ast 


CHUNK_SIZE = 40       
CHUNK_OVERLAP = 5     


def chunk_by_lines(file_path, content, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Simple chunker — splits any file by line count.
    Used for non-Python files (JS, MD, JSON, etc.)

    Returns a list of chunk dicts:
    [
        {
            "path":       "src/utils.js",
            "chunk_index": 0,
            "start_line":  1,
            "end_line":    40,
            "content":    "function helper() { ... }",
            "type":       "lines"
        },
        ...
    ]
    """
    lines = content.split("\n")
    chunks = []
    chunk_index = 0

   
    step = chunk_size - overlap

    for start in range(0, len(lines), step):
        end = min(start + chunk_size, len(lines))
        chunk_lines = lines[start:end]

        
        non_empty = [l for l in chunk_lines if l.strip()]
        if len(non_empty) < 3:
            continue

        chunks.append({
            "path":        file_path,
            "chunk_index": chunk_index,
            "start_line":  start + 1,      
            "end_line":    end,
            "content":     "\n".join(chunk_lines),
            "type":        "lines"
        })
        chunk_index += 1

     
        if end == len(lines):
            break

    return chunks


def chunk_python_by_functions(file_path, content):
    """
    Smart chunker for Python files.
    Uses Python's AST (Abstract Syntax Tree) to find function
    and class boundaries, then chunks at those boundaries.

    This means each chunk is a complete function or class —
    much more meaningful than arbitrary line splits.

    Falls back to line chunking if the file can't be parsed.
    """
   

    try:
        tree = ast.parse(content)
    except SyntaxError:
        
        return chunk_by_lines(file_path, content)

    lines = content.split("\n")
    chunks = []
    chunk_index = 0

    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1     
            end = node.end_lineno         

            chunk_lines = lines[start:end]
            chunk_content = "\n".join(chunk_lines)

            
            if len(chunk_lines) > CHUNK_SIZE * 2:
                sub_chunks = chunk_by_lines(file_path, chunk_content, CHUNK_SIZE, CHUNK_OVERLAP)
                for sc in sub_chunks:
                    sc["start_line"] += start
                    sc["end_line"] += start
                    sc["chunk_index"] = chunk_index
                    sc["type"] = "function_subchunk"
                    chunks.append(sc)
                    chunk_index += 1
            else:
                
                if isinstance(node, ast.ClassDef):
                    node_type = "class"
                elif isinstance(node, ast.AsyncFunctionDef):
                    node_type = "async_function"
                else:
                    node_type = "function"

                chunks.append({
                    "path":        file_path,
                    "chunk_index": chunk_index,
                    "start_line":  node.lineno,
                    "end_line":    node.end_lineno,
                    "content":     chunk_content,
                    "type":        node_type,
                    "name":        node.name   
                })
                chunk_index += 1

    
    if not chunks:
        return chunk_by_lines(file_path, content)

    return chunks


def chunk_file(file_dict):
    """
    Route a file to the right chunker based on its extension.

    Input: one file dict from ingest.py
    {
        "path": "src/main.py",
        "content": "def foo(): ..."
    }

    Output: list of chunk dicts
    """
    path = file_dict["path"]
    content = file_dict["content"]

    
    if path.endswith(".py"):
        return chunk_python_by_functions(path, content)

    
    return chunk_by_lines(path, content)


def chunk_all(files):
    """
    Chunk every file in the ingested list.

    Input:  list of file dicts from ingest.py
    Output: flat list of all chunks across all files
    """
    all_chunks = []

    for file_dict in files:
        chunks = chunk_file(file_dict)
        all_chunks.extend(chunks)

    return all_chunks


def print_chunk_summary(chunks):
    """Print a readable summary of the chunks."""
    print("\n" + "=" * 50)
    print("  CHUNKING SUMMARY")
    print("=" * 50)
    print(f"  Total chunks: {len(chunks)}")

    
    types = {}
    for c in chunks:
        t = c.get("type", "unknown")
        types[t] = types.get(t, 0) + 1

    print("\n  Chunk types:")
    for t, count in types.items():
        print(f"    {t}: {count}")

    print("\n  Sample chunks:")
    for chunk in chunks[:3]:
        name = chunk.get("name", "")
        name_str = f" — {name}()" if name else ""
        print(f"\n  [{chunk['chunk_index']}] {chunk['path']}{name_str}")
        print(f"  Lines {chunk['start_line']}–{chunk['end_line']} ({chunk['type']})")
        print(f"  Preview: {chunk['content'][:80].strip()}...")

    print("=" * 50)



if __name__ == "__main__":
    
    from ingest import ingest

    url = input("Enter a GitHub repo URL to test: ").strip()
    if not url:
        url = "https://github.com/realpython/reader"
        print(f"Using default: {url}")

    
    print("\nIngesting...")
    files = ingest(url)

    
    print("\nChunking...")
    chunks = chunk_all(files)

    
    print_chunk_summary(chunks)