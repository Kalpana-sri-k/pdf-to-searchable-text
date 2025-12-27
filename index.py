# app/index.py
from sentence_transformers import SentenceTransformer
import numpy as np
import os


class PDFIndexer:
    def __init__(self, model_name_or_path: str = "sentence-transformers/all-mpnet-base-v2"):
        if os.path.isdir(model_name_or_path):
            self.model = SentenceTransformer(model_name_or_path)
        else:
            self.model = SentenceTransformer(model_name_or_path)

    def embed_texts(self, texts):
        if not texts:
            return []
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return np.asarray(embeddings).tolist()

    def embed_single(self, text: str):
        if not text:
            return []
        return self.model.encode([text])[0].tolist()
