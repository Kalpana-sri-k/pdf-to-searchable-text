Installation :
    Install the required Python dependencies. This typically involves:

git clone https://github.com/Kalpana-sri-k/pdf-to-searchable-text.git
cd pdf-to-searchable-text
pip install -r requirements.txt

Make sure you also have Tesseract OCR installed on your system:

Structure :

app.py — Main script or API entry point
utils.py — Utility functions for OCR, file handling
ingest.py / ingest_with_embeddings.txt — Document ingestion logic
index.html + main.js + style.css — Web UI components
main.py — Possibly the core execution script
chunker.py — Splits large PDFs into smaller OCR‑friendly chunks
embedder.py — Might embed text for search indices

How It Works

PDF Parsing: Each page is read and rendered as an image
OCR Engine: Text is recognized using Tesseract OCR
Text Overlay: Recognized text is written back into a PDF as invisible (but searchable) text layer
Searchable Output: The resulting PDF can be searched via normal PDF viewers

