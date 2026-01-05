import os
import shutil
from collections import defaultdict
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import chromadb

from ingest import extract_pages_blocks
from utils import normalize_blocks
from chunker import chunk_text
from openvino_embedder import get_openvino_embedding_function, OPENVINO_AVAILABLE
from rag_engine import SimpleRAGEngine, get_rag_engine

app = FastAPI(title="Enterprise PDF Knowledge Tool - Intel Optimized")

os.makedirs("uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/pdf_files", StaticFiles(directory="uploads"), name="pdf_files")

DB_PATH = "./chroma_db"
BGE_MIGRATION_FLAG = "./chroma_db/.bge_migrated"

# One-time migration: Clear old database to re-index with BGE embeddings
if os.path.exists(DB_PATH) and not os.path.exists(BGE_MIGRATION_FLAG):
    shutil.rmtree(DB_PATH)
    print("🗑️  Cleared old database - will re-index with BGE embeddings")

vdb_client = chromadb.PersistentClient(path=DB_PATH)

# Create migration flag after DB is initialized
os.makedirs(DB_PATH, exist_ok=True)
with open(BGE_MIGRATION_FLAG, "w") as f:
    f.write("migrated to BGE")

# Use Intel OpenVINO optimized BGE embedding model
print("Initializing Intel OpenVINO BGE embedding model...")
embedding_model = get_openvino_embedding_function(
    model_name="bge-small",  
    use_query_prefix=True
)

rag_engine = get_rag_engine(use_llm=False)  
print("RAG engine initialized")

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

@app.get("/engine")
async def engine_info():
    """Returns info about the current search engine."""
    return {
        "engine": "Intel OpenVINO + ChromaDB",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "chunking": True,
        "rag_enabled": True,
        "openvino_available": OPENVINO_AVAILABLE
    }

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    import traceback
    try:
        file_path = os.path.join("uploads", file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"📄 Processing PDF: {file_path}")
        pages_data = extract_pages_blocks(file_path)
        print(f"✅ Extracted {len(pages_data)} pages")
        clean_pages = normalize_blocks(pages_data)
        print(f"✅ Normalized blocks")

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

        print(f"✅ Indexed {char_count} characters")
        return {
            "stats": {
                "pages": len(clean_pages),
                "images": 0,
                "characters": char_count
            }
        }
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        traceback.print_exc()
        return {"error": str(e), "stats": {"pages": 0, "images": 0, "characters": 0}}

@app.post("/search/semantic")
async def semantic_search(query: str = Form(...)):
    import traceback
    try:
        print(f"🔍 Searching for: {query}")
        collection = vdb_client.get_or_create_collection(
            name="pdf_knowledge",
            embedding_function=embedding_model
        )
        
        # Check if collection has any documents
        count = collection.count()
        print(f"📊 Collection has {count} documents")
        if count == 0:
            return {"results": [], "error": "No documents indexed. Please upload a PDF first."}

        # Retrieve more results for better re-ranking
        query_results = collection.query(
            query_texts=[query],
            n_results=min(15, count),  # Get more results for re-ranking
            include=["documents", "metadatas", "distances"]
        )
        
        documents = query_results["documents"][0]
        metadatas = query_results["metadatas"][0]
        distances = query_results["distances"][0]
        
        print(f"✅ Query returned {len(documents)} results")
        for i, (doc, dist) in enumerate(zip(documents[:5], distances[:5])):
            print(f"   Result {i+1}: distance={dist:.3f}, preview={doc[:50]}...")

        # More adaptive threshold based on best result
        best_distance = distances[0] if distances else 1.0
        ADAPTIVE_THRESHOLD = max(best_distance * 3, 1.5)  # Allow results within 3x of best

        filtered_chunks = []
        filtered_metas = []
        similarity_scores = []
        
        for doc, meta, dist in zip(documents, metadatas, distances):
            # Skip non-text results
            if meta.get("type") != "text":
                continue
            
            # Only filter out very dissimilar results
            if dist > ADAPTIVE_THRESHOLD:
                continue
            
            filtered_chunks.append(doc.strip())
            filtered_metas.append(meta)
            # Convert distance to similarity (higher is better)
            similarity_scores.append(round(1 / (1 + dist), 3))
        
        print(f"📋 After filtering (threshold={ADAPTIVE_THRESHOLD:.2f}): {len(filtered_chunks)} chunks")
        
        if not filtered_chunks:
            # If nothing passes filter, return the best match anyway
            if documents:
                print("⚠️ No chunks passed filter, returning best match anyway")
                return {"results": [{
                    "chunk_text": documents[0],
                    "section_id": metadatas[0].get("page", 1),
                    "subheading": metadatas[0].get("source", "Unknown"),
                    "type": "text",
                    "similarity_score": round(1 / (1 + distances[0]), 3),
                    "confidence": 0.3
                }]}
            return {"results": []}
        
        # Use improved RAG engine to extract the best answer
        rag_result = rag_engine.extract_answer(
            query=query,
            context_chunks=filtered_chunks,
            similarity_scores=similarity_scores
        )
        
        best_chunk_idx = rag_result.get("chunk_index", 0)
        best_meta = filtered_metas[best_chunk_idx] if best_chunk_idx < len(filtered_metas) else filtered_metas[0]
        
        results = [{
            "chunk_text": rag_result["answer"],
            "section_id": best_meta["page"],
            "subheading": best_meta["source"],
            "type": "text",
            "similarity_score": similarity_scores[best_chunk_idx] if best_chunk_idx < len(similarity_scores) else similarity_scores[0],
            "rag_source": rag_result.get("source", "extraction"),
            "confidence": rag_result.get("confidence", 0.5)
        }]

        return {"results": results}
    
    except Exception as e:
        print(f"❌ Search error: {e}")
        traceback.print_exc()
        return {"results": [], "error": str(e)}
