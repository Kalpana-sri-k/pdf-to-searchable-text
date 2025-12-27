from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import fitz
import ocrmypdf
import os
import tempfile
import io
import re
from PIL import Image

app = FastAPI(title="PDF → Searchable Knowledge")
# Paths for static files
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount /static for serving images, css, js, etc.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# In-memory "database"
VECTOR_DB = []

# Caption regex 
CAPTION_REGEX = re.compile(r'^(figure|fig\.|table|caption)\b', re.IGNORECASE)

# Helper functions
def find_caption_on_page(page_blocks):
    for b in page_blocks:
        text = b[4].strip()
        if text and CAPTION_REGEX.search(text.splitlines()[0].strip()):
            return text
    for b in page_blocks:
        text = b[4].strip()
        if text and any(x in text.lower() for x in ['figure', 'fig.', 'table']):
            return text
    return None

def extract_chunks_by_heading(text):
    lines = text.split("\n")
    chunks = []
    current_heading = None
    current_content = []

    def is_likely_heading(line):
        line = line.strip()
        if not line:
            return False
        if re.match(r"^\d+(\.\d+)*\s+", line):
            return True
        words = line.split()
        if 1 <= len(words) <= 10 and line.isupper():
            return True
        if 2 <= len(words) <= 10 and line[0].isupper():
            capital_words = sum(1 for word in words if word and word[0].isupper())
            if capital_words >= 0.75 * len(words):
                return True
        if re.search(r"[:\-]$", line) and len(words) > 1 and not line.lower().startswith(("this", "the")):
            return True
        return False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if is_likely_heading(line):
            if current_heading and current_content:
                chunks.append({
                    "subheading": current_heading,
                    "chunk_text": "\n".join(current_content).strip()
                })
            current_heading = line.rstrip(":").rstrip("-")
            current_content = []
        else:
            if current_heading:
                current_content.append(line)

    if current_heading and current_content:
        chunks.append({
            "subheading": current_heading,
            "chunk_text": "\n".join(current_content).strip()
        })

    return chunks

def add_chunks_to_db(doc_id, page_num, chunks):
    for c in chunks:
        VECTOR_DB.append({
            "doc_id": doc_id,
            "section_id": page_num,
            "subheading": c["subheading"],
            "chunk_text": c["chunk_text"],
            "type": "paragraph"
        })

def add_image_entry(doc_id, page_num, img_filename, caption_text):
    VECTOR_DB.append({
        "doc_id": doc_id,
        "section_id": f"img_{page_num}",
        "chunk_text": caption_text.strip() if caption_text else "",
        "subheading": caption_text.strip() if caption_text else "",
        "type": "image",
        "image_url": img_filename,
        "caption": caption_text.strip() if caption_text else ""
    })

def save_image_as_png(img_bytes, img_path):
    image = Image.open(io.BytesIO(img_bytes))
    image.convert("RGB").save(img_path, format="PNG")

# PDF processing
async def process_single_pdf(file: UploadFile):
    global VECTOR_DB
    VECTOR_DB = []  

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    searchable_path = tmp_path.replace(".pdf", "_searchable.pdf")

    # Run OCR to ensure text layer exists
    try:
        ocrmypdf.ocr(tmp_path, searchable_path, deskew=True, force_ocr=True, skip_text=False, language="eng")
    except Exception:
        searchable_path = tmp_path  # fallback

    doc = fitz.open(searchable_path)
    total_chars = 0
    total_images = 0
    toc = doc.get_toc()

    safe_name = os.path.splitext(os.path.basename(file.filename))[0]
    upload_dir = os.path.join(UPLOAD_DIR, safe_name)
    os.makedirs(upload_dir, exist_ok=True)

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1

        blocks = page.get_text("blocks")
        blocks = [b for b in blocks if len(b) >= 5]

        # Extract text
        page_text = page.get_text("text") or ""
        if page_text.strip():
            total_chars += len(page_text)
            chunks = extract_chunks_by_heading(page_text)
            add_chunks_to_db(file.filename, page_num, chunks)

        # Extract inline images
        images = page.get_images(full=True)
        if images:
            caption_candidate = find_caption_on_page(blocks)
            for img_index, img in enumerate(images):
                xref = img[0]
                img_filename = f"{safe_name}_page{page_num}_img{img_index+1}.png"
                img_path = os.path.join(upload_dir, img_filename)

                pix = fitz.Pixmap(doc, xref)
                if pix.n < 5:  
                    pix.save(img_path)
                else:  
                    pix1 = fitz.Pixmap(fitz.csRGB, pix)
                    pix1.save(img_path)
                    pix1 = None
                pix = None

                public_url = f"/static/uploads/{safe_name}/{img_filename}"
                total_images += 1
                add_image_entry(file.filename, page_num, public_url, caption_candidate)

        # Save full page image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        page_img_filename = f"{safe_name}_page{page_num}.png"
        page_img_path = os.path.join(upload_dir, page_img_filename)
        pix.save(page_img_path)
        pix = None
        public_page_url = f"/static/uploads/{safe_name}/{page_img_filename}"
        add_image_entry(file.filename, page_num, public_page_url, page_text)

    stats = {
        "filename": file.filename,
        "pages": len(doc),
        "characters": total_chars,
        "images": total_images + len(doc),  
        "toc": toc
    }
    return stats

# Routes
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    stats = await process_single_pdf(file)
    return JSONResponse({"stats": stats})

@app.post("/search/semantic")
async def semantic_search(query: str = Form(...), doc_id: str = Form(None), top_k: int = Form(10)):
    results = []
    query_l = query.strip().lower()
    if not query_l:
        return {"results": results}

    seen = set()
    for entry in VECTOR_DB:
        if doc_id and entry["doc_id"] != doc_id:
            continue
        subh = (entry.get("subheading") or "").lower()
        text = (entry.get("chunk_text") or "").lower()
        caption = (entry.get("caption") or "").lower() if entry.get("caption") else ""
        if query_l in subh or query_l in text or query_l in caption:
            key = f"{subh}-{text}-{caption}"
            if key not in seen:
                seen.add(key)
                results.append(entry)
                if len(results) >= top_k:
                    break
    return JSONResponse({"results": results})

@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h2>index.html not found in static/</h2>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)