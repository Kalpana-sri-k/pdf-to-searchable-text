import hashlib

# Compute SHA256 for file
def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda: f.read(8192), b''):
            h.update(b)
    return h.hexdigest()

# Remove repeated headers/footers & normalize
def normalize_blocks(pages_data):
    clean_pages = []
    seen_texts = set()
    for page in pages_data:
        clean_blocks = []
        for b in page["blocks"]:
            t = b["text"].strip()
            if t not in seen_texts:
                clean_blocks.append(b)
                seen_texts.add(t)
        clean_pages.append({"page_num": page["page_num"], "blocks": clean_blocks})
    return clean_pages
