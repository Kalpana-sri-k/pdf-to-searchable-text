# app/ingest.py
import fitz  # PyMuPDF
import camelot
import pdfplumber
import os
import tempfile
import ocrmypdf


def run_ocr(pdf_path: str) -> str:
    """Run OCR on a PDF and return the path to the OCRed PDF."""
    tmp_dir = tempfile.gettempdir()
    output_pdf_path = os.path.join(
        tmp_dir, os.path.basename(pdf_path).replace(".pdf", "_ocr.pdf")
    )
    ocrmypdf.ocr(
        pdf_path,
        output_pdf_path,
        force_ocr=True,
        output_type='pdf',
        skip_text=False
    )
    return output_pdf_path


def extract_pages_blocks(pdf_path: str):
    doc = fitz.open(pdf_path)
    all_pages = []
    for page in doc:
        blocks = page.get_text("blocks")
        blocks_data = [{"text": b[4], "bbox": b[:4]} for b in blocks if b[4].strip()]
        all_pages.append({"page_num": page.number + 1, "blocks": blocks_data})
    return all_pages


def extract_sections(pdf_path: str):
    doc = fitz.open(pdf_path)
    sections = []
    for page in doc:
        blocks = page.get_text("blocks")
        for b in blocks:
            text = b[4].strip()
            if len(text) > 5 and text.isupper():
                sections.append({
                    "title": text,
                    "page": page.number + 1,
                    "bbox": b[:4]
                })
    return sections


def extract_tables(pdf_path: str):
    tables_data = []
    try:
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
        for t in tables:
            tables_data.append({"page": t.page, "data": t.df.to_dict(orient='records')})
    except Exception:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                for table in page.extract_tables():
                    tables_data.append({"page": i + 1, "data": [row for row in table]})
    return tables_data


def extract_images(pdf_path: str, output_dir="extracted_images"):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    images_data = []
    for page_num, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img_ext = base_image["ext"]
            img_name = f"page{page_num + 1}_img{img_index + 1}.{img_ext}"
            img_path = os.path.join(output_dir, img_name)
            with open(img_path, "wb") as f:
                f.write(image_bytes)
            images_data.append({"page": page_num + 1, "path": img_path})
    return images_data


def ingest_pdf_json(pdf_path: str):
    ocr_pdf_path = run_ocr(pdf_path)
    return {
        "ocr_pdf_path": ocr_pdf_path,
        "pages": extract_pages_blocks(ocr_pdf_path),
        "sections": extract_sections(ocr_pdf_path),
        "tables": extract_tables(ocr_pdf_path),
        "images": extract_images(ocr_pdf_path)
    }
