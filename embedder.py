# app/embedder.py
from sentence_transformers import SentenceTransformer

# Load model once (all-mpnet-base-v2 is solid for semantic search)
model = SentenceTransformer('all-mpnet-base-v2')

def embed_chunks(chunks):
    """
    Generate embeddings for a list of text chunks.
    Returns a list of numpy arrays.
    """
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    return embeddings
