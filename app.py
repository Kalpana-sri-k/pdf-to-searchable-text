from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
documents = []
vectorizer = None
tfidf_matrix = None
pdf_metadata = {}

class SearchRequest(BaseModel):
    query: str

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global documents, vectorizer, tfidf_matrix, pdf_metadata
    documents = []
    pdf_metadata = {
        "pages": 0,
        "characters": 0,
        "images": 0,
        "tables": 0,  # basic heuristic
        "toc": []
    }

    pdf_bytes = await file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    pdf_metadata["pages"] = len(doc)
    pdf_metadata["toc"] = doc.get_toc(simple=True)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()

        if not text.strip():
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            text = pytesseract.image_to_string(img)

        documents.append({"page": page_num + 1, "text": text})
        pdf_metadata["characters"] += len(text)

        # Rough heuristics
        pdf_metadata["images"] += len(page.get_images(full=True))
        if re.search(r"\btable\b", text.lower()):
            pdf_metadata["tables"] += 1

    # Build TF-IDF model
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([d["text"] for d in documents])

    return {"message": "PDF processed", "metadata": pdf_metadata}

@app.post("/search/semantic")
async def search_semantic(req: SearchRequest):
    global documents, vectorizer, tfidf_matrix
    if not documents:
        return {"results": []}

    query_vec = vectorizer.transform([req.query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    ranked = sorted(
        zip(similarities, documents), key=lambda x: x[0], reverse=True
    )

    results = [
        {"page": doc["page"], "text": doc["text"][:300], "score": float(score)}
        for score, doc in ranked if score > 0
    ]

    return {"results": results}
