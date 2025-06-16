# app/rag/retriever.py

import json
import logging
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DB_DIR = Path("db")
EMBEDDINGS_FILE = DB_DIR / "embeddings.npy"
CHUNKS_FILE = DB_DIR / "chunks.json"
BM25_INDEX_FILE = DB_DIR / "bm25_index.pkl"
EMBEDDING_MODEL = "text-embedding-3-small"

class KnowledgeRetriever:
    """
    Handles retrieving information from the knowledge base using a hybrid approach
    (BM25 for sparse keyword search and dense vector search) followed by a
    re-ranking step with a Cross-Encoder model.
    """
    def __init__(self):
        self.embeddings: Optional[np.ndarray] = None
        self.chunks: List[Dict[str, Any]] = []
        self.bm25_index: Optional[BM25Okapi] = None
        self.cross_encoder: Optional[CrossEncoder] = None
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._load_knowledge_base()

    def _load_knowledge_base(self):
        """Loads all necessary files for the retriever from the db directory."""
        if not all([EMBEDDINGS_FILE.exists(), CHUNKS_FILE.exists(), BM25_INDEX_FILE.exists()]):
            logging.warning("Knowledge base files not found. Please run build_knowledge_base.py.")
            return

        logging.info("Loading knowledge base...")
        self.embeddings = np.load(EMBEDDINGS_FILE)
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        with open(BM25_INDEX_FILE, "rb") as f:
            self.bm25_index = pickle.load(f)
        
        # Load the Cross-Encoder model
        try:
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-minilm-l-6-v2', max_length=512)
            logging.info("Cross-Encoder model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load Cross-Encoder model: {e}", exc_info=True)
            self.cross_encoder = None
        
        logging.info(f"Knowledge base loaded successfully ({len(self.chunks)} chunks).")

    def _get_embedding(self, text: str) -> np.ndarray:
        """Helper to get embedding for a single text query."""
        response = self.openai_client.embeddings.create(input=[text], model="text-embedding-3-small")
        return np.array(response.data[0].embedding)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """Calculates cosine similarity between a vector and a matrix of vectors."""
        vec1 = vec1.reshape(1, -1)
        return (vec1 @ vec2.T) / (np.linalg.norm(vec1) * np.linalg.norm(vec2, axis=1))

    def retrieve(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self.chunks or self.bm25_index is None or self.embeddings is None:
            logging.warning("Knowledge base is not loaded. Cannot retrieve.")
            return []

        # 1. Sparse Search (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # 2. Dense Search (Vector)
        query_embedding = self._get_embedding(query)
        embedding_scores = self._cosine_similarity(query_embedding, self.embeddings).flatten()

        # 3. Hybrid Scoring (Combine and rank initial candidates)
        # Normalize scores to be in a similar range before combining
        norm_bm25_scores = bm25_scores / (np.max(bm25_scores) + 1e-9)
        norm_embedding_scores = embedding_scores / (np.max(embedding_scores) + 1e-9)
        combined_scores = 0.4 * norm_bm25_scores + 0.6 * norm_embedding_scores
        
        # Get a larger pool of candidates for re-ranking
        candidate_pool_size = min(len(self.chunks), 25)
        top_candidate_indices = np.argsort(combined_scores)[-candidate_pool_size:][::-1]

        # 4. Re-ranking with Cross-Encoder
        if self.cross_encoder:
            cross_encoder_pairs = [[query, self.chunks[i]["text"]] for i in top_candidate_indices]
            rerank_scores = self.cross_encoder.predict(cross_encoder_pairs, show_progress_bar=False)
            
            # Sort candidate indices based on the new re-ranking scores
            reranked_indices = [idx for _, idx in sorted(zip(rerank_scores, top_candidate_indices), reverse=True)]
            final_indices_unfiltered = reranked_indices
            logging.info(f"Re-ranked {len(top_candidate_indices)} candidates.")
        else:
            logging.warning("Cross-encoder not available. Falling back to simple hybrid search.")
            final_indices_unfiltered = top_candidate_indices

        # 5. Filter results based on metadata
        final_indices = []
        required_tags = set(filters.get("tags", [])) if filters else set()
        for idx in final_indices_unfiltered:
            if len(final_indices) >= top_k:
                break
            
            chunk_tags = set(self.chunks[idx].get("metadata", {}).get("tags", []))
            if not required_tags or required_tags.issubset(chunk_tags):
                final_indices.append(idx)
        
        # 6. Format and return final results
        results = [{
            "text": self.chunks[i]["text"],
            "source": self.chunks[i]["source"],
            "metadata": self.chunks[i].get("metadata", {}),
        } for i in final_indices]
        
        return results 