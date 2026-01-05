"""
Intel OpenVINO RAG Engine - Combines retrieval with LLM generation.
Uses OpenVINO for optimized inference on Intel hardware.
"""

import os
import re
from typing import List, Optional, Dict, Any
from pathlib import Path

try:
    import openvino_genai as ov_genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("⚠️  OpenVINO GenAI not available. Install with: pip install openvino-genai")

try:
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer, pipeline
    OPTIMUM_AVAILABLE = True
except ImportError:
    OPTIMUM_AVAILABLE = False


class IntelRAGEngine:
    """
    RAG Engine optimized for Intel hardware using OpenVINO.
    Combines document retrieval with LLM-based answer generation.
    """
    
    SUPPORTED_MODELS = {
        "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
        "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "qwen2-0.5b": "Qwen/Qwen2-0.5B-Instruct",
        "gemma-2b": "google/gemma-2b-it",
    }
    
    def __init__(
        self, 
        model_name: str = "tinyllama",
        cache_dir: str = "./openvino_models",
        device: str = "CPU",
        max_new_tokens: int = 256
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.device = device
        self.max_new_tokens = max_new_tokens
        
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """Load the LLM model with OpenVINO optimization."""
        if self.is_loaded:
            return True
            
        model_id = self.SUPPORTED_MODELS.get(self.model_name, self.model_name)
        ov_model_path = self.cache_dir / model_id.replace("/", "_")
        
        print(f"🧠 Loading RAG LLM: {model_id}")
        
        if GENAI_AVAILABLE and ov_model_path.exists():
            try:
                self.pipe = ov_genai.LLMPipeline(str(ov_model_path), self.device)
                self.is_loaded = True
                print(f"✅ Loaded with OpenVINO GenAI (optimized)")
                return True
            except Exception as e:
                print(f"⚠️  GenAI load failed: {e}")
        
        if OPTIMUM_AVAILABLE:
            try:
                if ov_model_path.exists():
                    print(f"📦 Loading cached OpenVINO model from {ov_model_path}")
                    self.model = OVModelForCausalLM.from_pretrained(
                        str(ov_model_path),
                        device=self.device
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained(str(ov_model_path))
                else:
                    print(f"📦 Exporting {model_id} to OpenVINO (one-time)...")
                    self.model = OVModelForCausalLM.from_pretrained(
                        model_id,
                        export=True,
                        device=self.device
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                    
                    self.model.save_pretrained(str(ov_model_path))
                    self.tokenizer.save_pretrained(str(ov_model_path))
                    print(f"💾 Model cached at {ov_model_path}")
                
                self.is_loaded = True
                print(f"✅ RAG LLM ready with Optimum Intel!")
                return True
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
                return False
        
        print("❌ No LLM backend available. Install: pip install optimum[openvino] openvino-genai")
        return False
    
    def generate_answer(
        self, 
        query: str, 
        context_chunks: List[str],
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate an answer using RAG - retrieval-augmented generation.
        
        Args:
            query: User's question
            context_chunks: Retrieved relevant text chunks from the PDF
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with generated answer and metadata
        """
        if not self.is_loaded:
            if not self.load_model():
                return {
                    "answer": context_chunks[0] if context_chunks else "No relevant content found.",
                    "source": "retrieval_only",
                    "error": "LLM not available"
                }
        
        max_tokens = max_tokens or self.max_new_tokens
        
        context = "\n\n".join(context_chunks[:3])  # Use top 3 chunks
        
        prompt = f"""Based on the following document content, answer the question accurately and concisely.
If the answer is not in the provided content, say "The document does not contain this information."

Document Content:
{context}

Question: {query}

Answer:"""
        
        try:
            if self.pipe is not None:
                config = ov_genai.GenerationConfig()
                config.max_new_tokens = max_tokens
                config.temperature = 0.3
                config.do_sample = False
                
                response = self.pipe.generate(prompt, config)
                answer = response.strip()
            
            elif self.model is not None and self.tokenizer is not None:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.3,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                if "Answer:" in answer:
                    answer = answer.split("Answer:")[-1].strip()
            else:
                return {
                    "answer": context_chunks[0] if context_chunks else "No relevant content found.",
                    "source": "retrieval_only",
                    "error": "Model not loaded"
                }
            
            return {
                "answer": answer,
                "source": "rag_llm",
                "model": self.model_name,
                "context_used": len(context_chunks)
            }
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return {
                "answer": context_chunks[0] if context_chunks else "Error generating answer.",
                "source": "retrieval_fallback",
                "error": str(e)
            }


class SimpleRAGEngine:
    """
    Intelligent RAG without LLM - uses advanced extraction from retrieved chunks.
    Uses TF-IDF-like scoring, phrase matching, and query expansion.
    """
    
    # Common synonyms and related terms for query expansion
    SYNONYMS = {
        'array': ['arrays', 'list', 'vector', 'matrix', 'sequence'],
        'function': ['functions', 'method', 'methods', 'procedure', 'routine'],
        'create': ['creating', 'make', 'generate', 'initialize', 'define'],
        'type': ['types', 'datatype', 'datatypes', 'dtype', 'kind'],
        'data': ['information', 'values', 'elements', 'content'],
        'attribute': ['attributes', 'property', 'properties', 'field', 'member'],
        'operation': ['operations', 'method', 'function', 'action'],
        'use': ['using', 'usage', 'utilize', 'apply', 'employ'],
        'define': ['definition', 'declare', 'specify', 'set'],
        'return': ['returns', 'output', 'result', 'give'],
        'parameter': ['parameters', 'argument', 'arguments', 'param', 'arg'],
        'value': ['values', 'number', 'result', 'output'],
        'index': ['indexing', 'indices', 'position', 'location'],
        'shape': ['dimension', 'dimensions', 'size', 'length'],
        'element': ['elements', 'item', 'items', 'member', 'entry'],
    }
    
    # Stop words to ignore in scoring
    STOP_WORDS = {
        'what', 'is', 'the', 'a', 'an', 'of', 'in', 'to', 'for', 'how', 'why', 
        'when', 'where', 'who', 'which', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'this', 'that', 'these', 'those', 'it', 'its', 'and', 'or', 'but',
        'if', 'then', 'else', 'so', 'as', 'by', 'with', 'from', 'on', 'at',
        'about', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'under', 'over', 'out', 'up', 'down', 'off',
        'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'they'
    }
    
    def __init__(self):
        pass
    
    def expand_query(self, query: str) -> set:
        """Expand query with synonyms and related terms."""
        words = query.lower().split()
        expanded = set(words)
        
        for word in words:
            # Add synonyms
            if word in self.SYNONYMS:
                expanded.update(self.SYNONYMS[word])
            # Check if word is a synonym value
            for key, syns in self.SYNONYMS.items():
                if word in syns:
                    expanded.add(key)
                    expanded.update(syns)
        
        return expanded - self.STOP_WORDS
    
    def extract_key_phrases(self, query: str) -> List[str]:
        """Extract potential key phrases from query (2-3 word combinations)."""
        words = [w for w in query.lower().split() if w not in self.STOP_WORDS and len(w) > 2]
        phrases = []
        
        # 3-word phrases
        for i in range(len(words) - 2):
            phrases.append(' '.join(words[i:i+3]))
        
        # 2-word phrases
        for i in range(len(words) - 1):
            phrases.append(' '.join(words[i:i+2]))
        
        return phrases
    
    def score_sentence(
        self, 
        sentence: str, 
        query_keywords: set, 
        key_phrases: List[str],
        position_in_chunk: float = 0.5
    ) -> float:
        """
        Score a sentence based on multiple factors:
        - Keyword overlap (TF-IDF-like)
        - Phrase matching bonus
        - Sentence position (beginning/end of chunks often more important)
        - Keyword density
        """
        sentence_lower = sentence.lower()
        sentence_words = set(sentence_lower.split())
        
        # Remove stop words from sentence
        meaningful_words = sentence_words - self.STOP_WORDS
        
        # 1. Keyword overlap score
        overlap = query_keywords & meaningful_words
        keyword_score = len(overlap) / max(len(query_keywords), 1)
        
        # 2. Phrase matching bonus (exact phrases are highly valuable)
        phrase_bonus = 0
        for phrase in key_phrases:
            if phrase in sentence_lower:
                phrase_bonus += 0.3 * len(phrase.split())  # Longer phrases = higher bonus
        
        # 3. Keyword density (keywords / total words)
        density = len(overlap) / max(len(meaningful_words), 1)
        
        # 4. Position bonus (sentences at start or end get small boost)
        position_bonus = 0.1 if position_in_chunk < 0.2 or position_in_chunk > 0.8 else 0
        
        # 5. Length penalty for very short/long sentences
        word_count = len(sentence.split())
        length_factor = 1.0
        if word_count < 5:
            length_factor = 0.5
        elif word_count > 50:
            length_factor = 0.8
        
        # Combined score
        final_score = (keyword_score * 0.4 + phrase_bonus * 0.3 + density * 0.2 + position_bonus * 0.1) * length_factor
        
        return final_score
    
    def extract_answer(
        self, 
        query: str, 
        context_chunks: List[str],
        similarity_scores: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Extract the most relevant answer from retrieved chunks.
        Uses advanced scoring with query expansion and phrase matching.
        """
        if not context_chunks:
            return {
                "answer": "No relevant content found in the document.",
                "source": "none",
                "confidence": 0.0
            }
        
        # Expand query with synonyms
        query_keywords = self.expand_query(query)
        key_phrases = self.extract_key_phrases(query)
        
        print(f"🔑 Query keywords: {query_keywords}")
        print(f"🔑 Key phrases: {key_phrases}")
        
        best_sentences = []  # Store top candidates
        
        for chunk_idx, chunk in enumerate(context_chunks):
            # Split into sentences more carefully
            sentences = re.split(r'(?<=[.!?])\s+|(?<=:)\s+', chunk.replace('\n', ' '))
            total_sentences = len(sentences)
            
            for sent_idx, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if len(sentence) < 15:  # Skip very short sentences
                    continue
                
                # Calculate position in chunk (0 to 1)
                position = sent_idx / max(total_sentences - 1, 1)
                
                # Score the sentence
                score = self.score_sentence(sentence, query_keywords, key_phrases, position)
                
                # Boost by retrieval similarity score
                if similarity_scores and chunk_idx < len(similarity_scores):
                    score *= (1 + similarity_scores[chunk_idx] * 0.5)
                
                if score > 0.1:  # Only keep decent candidates
                    best_sentences.append({
                        'sentence': sentence,
                        'score': score,
                        'chunk_idx': chunk_idx
                    })
        
        # Sort by score
        best_sentences.sort(key=lambda x: x['score'], reverse=True)
        
        if best_sentences:
            top = best_sentences[0]
            answer = top['sentence']
            
            # Clean up the answer
            answer = answer.strip()
            if not answer.endswith(('.', '!', '?', ':')):
                answer += '.'
            
            print(f"✅ Best answer (score={top['score']:.3f}): {answer[:80]}...")
            
            return {
                "answer": answer,
                "source": "smart_extraction",
                "chunk_index": top['chunk_idx'],
                "confidence": min(top['score'] * 1.5, 1.0)  # Scale confidence
            }
        
        # Fallback: return first chunk's first meaningful sentence
        for chunk in context_chunks:
            sentences = chunk.split('.')
            for sent in sentences:
                if len(sent.strip()) > 20:
                    return {
                        "answer": sent.strip() + ".",
                        "source": "chunk_fallback",
                        "chunk_index": 0,
                        "confidence": similarity_scores[0] if similarity_scores else 0.3
                    }
        
        return {
            "answer": context_chunks[0],
            "source": "raw_chunk",
            "chunk_index": 0,
            "confidence": similarity_scores[0] if similarity_scores else 0.3
        }


def get_rag_engine(use_llm: bool = False, model_name: str = "tinyllama") -> Any:
    """
    Get a RAG engine based on configuration.
    
    Args:
        use_llm: Whether to use LLM for answer generation
        model_name: LLM model to use (if use_llm=True)
        
    Returns:
        RAG engine instance
    """
    if use_llm:
        engine = IntelRAGEngine(model_name=model_name)
        if engine.load_model():
            return engine
        print("⚠️  Falling back to SimpleRAGEngine")
    
    return SimpleRAGEngine()
