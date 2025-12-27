# app/chunker.py
import re

def chunk_text(text, max_tokens=300):
    """
    Split text into chunks of roughly max_tokens words (not characters),
    keeping sentence boundaries.
    """
    sentences = re.split(r'(?<=[.!?]) +', text)  # simple sentence splitter
    chunks = []
    current_chunk = ""
    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        if current_len + len(words) <= max_tokens:
            current_chunk += sentence + " "
            current_len += len(words)
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
            current_len = len(words)

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
