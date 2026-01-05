from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2')

def embed_chunks(chunks):
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    return embeddings
