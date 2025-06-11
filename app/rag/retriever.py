# app/rag/retriever.py

import json
import logging
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

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
    A class to retrieve relevant knowledge chunks from a database
    using a hybrid search approach (vector search + keyword search).
    """
    def __init__(self):
        """Initializes the retriever, loading all necessary data from disk."""
        self.embeddings: np.ndarray = None
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: BM25Okapi = None
        self._load_knowledge_base()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set for the retriever.")
        self.client = OpenAI(api_key=api_key)

    def _load_knowledge_base(self):
        """Loads embeddings, text chunks, and BM25 index from disk."""
        logging.info("Loading knowledge base...")
        if not all([EMBEDDINGS_FILE.exists(), CHUNKS_FILE.exists(), BM25_INDEX_FILE.exists()]):
            msg = (
                f"Knowledge base file not found. Please run "
                f"'scripts/build_knowledge_base.py' first."
            )
            logging.error(msg)
            raise FileNotFoundError(msg)

        try:
            self.embeddings = np.load(EMBEDDINGS_FILE)
            with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            with open(BM25_INDEX_FILE, "rb") as f:
                self.bm25 = pickle.load(f)
            logging.info(
                f"Knowledge base loaded successfully. "
                f"({len(self.chunks)} chunks, BM25 index, Embeddings)"
            )
        except Exception as e:
            logging.error(f"Failed to load knowledge base: {e}")
            raise

    def _vector_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Performs a pure vector search."""
        response = self.client.embeddings.create(input=[query], model=EMBEDDING_MODEL)
        query_embedding = np.array([response.data[0].embedding])
        similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [(idx, similarities[idx]) for idx in top_k_indices]

    def _keyword_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Performs a pure keyword search using BM25."""
        tokenized_query = query.lower().split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(doc_scores)[-k:][::-1]
        return [(idx, doc_scores[idx]) for idx in top_k_indices]

    def _reciprocal_rank_fusion(self, search_results: List[List[Tuple[int, float]]], k: int = 60) -> Dict[int, float]:
        """Merges search results using Reciprocal Rank Fusion."""
        fused_scores = {}
        for result_list in search_results:
            for rank, (doc_id, _) in enumerate(result_list):
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                fused_scores[doc_id] += 1 / (rank + k)
        return fused_scores

    def retrieve(
        self, query: str, top_k: int = 5, filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the top_k most relevant chunks using hybrid search.
        """
        if self.embeddings is None or not self.chunks or not self.bm25:
            logging.warning("Knowledge base is not fully loaded. Cannot retrieve.")
            return []

        # 1. Get results from both search methods
        candidate_k = top_k * 10
        vector_results = self._vector_search(query, candidate_k)
        keyword_results = self._keyword_search(query, candidate_k)

        # 2. Fuse the results
        fused_scores = self._reciprocal_rank_fusion([vector_results, keyword_results])
        
        # 3. Sort by fused score
        sorted_doc_ids = sorted(fused_scores.keys(), key=lambda id: fused_scores[id], reverse=True)

        # 4. Filter candidates based on metadata
        final_indices = []
        required_tags = set(filters.get("tags", [])) if filters else set()
        
        for doc_id in sorted_doc_ids:
            if len(final_indices) >= top_k:
                break
            
            if required_tags:
                chunk_tags = set(self.chunks[doc_id].get("metadata", {}).get("tags", []))
                if required_tags.issubset(chunk_tags):
                    final_indices.append(doc_id)
            else:
                final_indices.append(doc_id)

        # 5. Format results
        results = []
        for idx in final_indices:
            result = {
                "text": self.chunks[idx]["text"],
                "source": self.chunks[idx]["source"],
                "metadata": self.chunks[idx].get("metadata", {}),
                "score": fused_scores[idx], # Use the fused score
            }
            results.append(result)
            
        if not results:
            logging.warning(f"No documents found for query '{query[:50]}...' with filters {filters}")

        return results 