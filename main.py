import os
import shutil
from collections import defaultdict
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import chromadb
from chromadb.utils import embedding_functions

from ingest import extract_pages_blocks
from utils import normalize_blocks
from chunker import chunk_text

app = FastAPI(title="Enterprise PDF Knowledge Tool")

os.makedirs("uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/pdf_files", StaticFiles(directory="uploads"), name="pdf_files")

vdb_client = chromadb.PersistentClient(path="./chroma_db")
embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def normalize_text(text: str) -> str:
    lines = text.replace("\u2022", " ").split("\n")

    cleaned = []
    for line in lines:
        line = line.strip()
        if len(line) < 4:
            continue
        if line.lower() in ["none", "nil", "n/a"]:
            continue
        cleaned.append(line)

    paragraph = " ".join(cleaned)

    words = paragraph.split()
    final = []
    for w in words:
        if not final or w != final[-1]:
            final.append(w)

    return " ".join(final)


@app.get("/")
async def read_index():
    return FileResponse("templates/index.html")

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join("uploads", file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    pages_data = extract_pages_blocks(file_path)
    clean_pages = normalize_blocks(pages_data)

    collection = vdb_client.get_or_create_collection(
        name="pdf_knowledge",
        embedding_function=embedding_model
    )
    try:
        vdb_client.delete_collection(name="pdf_knowledge")
    except:
        pass 
    collection = vdb_client.get_or_create_collection(
        name="pdf_knowledge", 
        embedding_function=embedding_model
    )
    char_count = 0

    for page in clean_pages:
        texts = []
        for b in page["blocks"]:
            t = b.get("text", "").strip()
            if len(t) > 3:
                texts.append(t)

        text_content = " ".join(texts)
        char_count += len(text_content.strip())

        chunks = chunk_text(text_content)

        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                metadatas=[{
                    "source": file.filename,
                    "page": page["page_num"],
                    "type": "text"
                }],
                ids=[f"{file.filename}_{page['page_num']}_{i}"]
            )

    return {
        "stats": {
            "pages": len(clean_pages),
            "images": 0,
            "characters": char_count
        }
    }

@app.post("/search/semantic")
async def semantic_search(query: str = Form(...)):
    collection = vdb_client.get_or_create_collection(
        name="pdf_knowledge",
        embedding_function=embedding_model
    )

    query_results = collection.query(
        query_texts=[query],
        n_results=10
    )

    documents = query_results["documents"][0]
    metadatas = query_results["metadatas"][0]

    grouped = defaultdict(list)
    for doc, meta in zip(documents, metadatas):
        if meta.get("type") != "text":
            continue
        key = (meta["source"], meta["page"])
        grouped[key].append(doc)

    results = []

    for (source, page_num), texts in grouped.items():
        combined = normalize_text(" ".join(texts))

        results.append({
            "chunk_text": combined,
            "section_id": page_num,
            "subheading": source,
            "type": "text"
        })

    return {"results": results}
