"""
OpenVINO-optimized embedder for ChromaDB.
Uses Intel's optimum-intel library for accelerated inference on Intel hardware.
Supports multiple embedding models including BGE (better for retrieval).
"""

import numpy as np
from typing import List, cast
from pathlib import Path
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

try:
    from optimum.intel import OVModelForFeatureExtraction
    from transformers import AutoTokenizer
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    print("⚠️  OpenVINO not available. Install with: pip install optimum[openvino]")


EMBEDDING_MODELS = {
    "bge-small": "BAAI/bge-small-en-v1.5",      
    "bge-base": "BAAI/bge-base-en-v1.5",        
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",  
    "mpnet": "sentence-transformers/all-mpnet-base-v2",   
}


class OpenVINOEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    Custom embedding function using OpenVINO for optimized inference.
    Properly implements ChromaDB's EmbeddingFunction interface.
    Supports BGE models which are superior for retrieval tasks.
    """
    
    def __init__(
        self, 
        model_name: str = "bge-small",
        use_query_prefix: bool = True
    ):
        self.model_id = EMBEDDING_MODELS.get(model_name, model_name)
        self.model_name = model_name
        self.use_query_prefix = use_query_prefix
        
        self.is_bge = "bge" in self.model_id.lower()
        self.query_prefix = "Represent this sentence for searching relevant passages: " if self.is_bge else ""
        
        self.cache_dir = Path("./openvino_models")
        self.cache_dir.mkdir(exist_ok=True)
        
        self.ov_model_path = self.cache_dir / self.model_id.replace("/", "_")
        
        if OPENVINO_AVAILABLE:
            self._load_openvino_model()
        else:
            self._load_fallback_model()
    
    def _load_openvino_model(self):
        """Load or export the model to OpenVINO IR format."""
        print(f"🔧 Loading OpenVINO model: {self.model_id}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        if self.ov_model_path.exists():
            print(f"✅ Loading cached OpenVINO model from {self.ov_model_path}")
            self.model = OVModelForFeatureExtraction.from_pretrained(
                str(self.ov_model_path),
                device="CPU"  )
        else:
            print(f"📦 Exporting model to OpenVINO format (one-time operation)...")
            self.model = OVModelForFeatureExtraction.from_pretrained(
                self.model_id,
                export=True,  
                device="CPU"
            )
            self.model.save_pretrained(str(self.ov_model_path))
            self.tokenizer.save_pretrained(str(self.ov_model_path))
            print(f"💾 OpenVINO model cached at {self.ov_model_path}")
        
        self.use_openvino = True
        print(f"✅ OpenVINO embedder ready! Model: {self.model_id}")
    
    def _load_fallback_model(self):
        """Fallback to standard SentenceTransformer if OpenVINO is not available."""
        from sentence_transformers import SentenceTransformer
        print(f"⚠️  Using fallback SentenceTransformer model: {self.model_id}")
        self.model = SentenceTransformer(self.model_id)
        self.use_openvino = False
    
    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        token_embeddings = model_output[0]  
        input_mask_expanded = np.broadcast_to(
            np.expand_dims(attention_mask, -1),
            token_embeddings.shape
        ).astype(float)
        
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask
    
    def embed(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        This is the main method called by ChromaDB.
        
        Args:
            texts: List of texts to embed
            is_query: If True and using BGE model, adds query prefix for better retrieval
        """
        if not texts:
            return []
        
        if is_query and self.is_bge and self.use_query_prefix:
            texts = [self.query_prefix + t for t in texts]
        
        if self.use_openvino:
            return self._embed_openvino(texts)
        else:
            return self._embed_fallback(texts)
    
    def _embed_openvino(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenVINO-optimized model."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="np"
        )
        
        outputs = self.model(**encoded)
        
        embeddings = self._mean_pooling(
            [outputs.last_hidden_state],
            encoded["attention_mask"]
        )
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        return embeddings.tolist()
    
    def _embed_fallback(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using standard SentenceTransformer."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def __call__(self, input: Documents) -> Embeddings:
        """
        ChromaDB embedding function interface.
        Called when adding documents or querying.
        NOTE: ChromaDB calls this for BOTH indexing and querying.
        We apply query prefix for BGE since queries benefit from it.
        """
        if not input:
            return cast(Embeddings, [])
        
        texts = list(input)
        
        # For BGE models, add query prefix (helps with retrieval)
        # This is applied to both queries and documents for consistency
        if self.is_bge and self.use_query_prefix:
            # Don't add prefix to documents (they shouldn't have it)
            # But ChromaDB uses same function for both, so we detect by length
            # Short texts are likely queries, long texts are documents
            is_likely_query = len(texts) == 1 and len(texts[0]) < 200
            if is_likely_query:
                texts = [self.query_prefix + t for t in texts]
        
        if self.use_openvino:
            return cast(Embeddings, self._embed_openvino(texts))
        else:
            return cast(Embeddings, self._embed_fallback(texts))
    
    def embed_query(self, input) -> List[List[float]]:
        """Embed a single query with BGE prefix if applicable."""
        # Handle both string and list inputs from ChromaDB
        if isinstance(input, list):
            texts = input
        else:
            texts = [input]
        return self.embed(texts, is_query=True)
    
    def embed_documents(self, input) -> List[List[float]]:
        """Embed documents without query prefix."""
        # Handle both string and list inputs from ChromaDB  
        if isinstance(input, str):
            input = [input]
        return self.embed(input, is_query=False)


def get_openvino_embedding_function(
    model_name: str = "bge-small",
    use_query_prefix: bool = True
):
    """
    Create an OpenVINO embedding function for ChromaDB.
    
    Args:
        model_name: One of 'bge-small', 'bge-base', 'minilm', 'mpnet' or full HF model name
        use_query_prefix: Whether to use BGE query prefix for better retrieval
    """
    return OpenVINOEmbeddingFunction(
        model_name=model_name, 
        use_query_prefix=use_query_prefix
    )


def is_openvino_available() -> bool:
    """Check if OpenVINO is available in the environment."""
    return OPENVINO_AVAILABLE
