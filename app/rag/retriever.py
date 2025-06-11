# app/rag/retriever.py

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DB_DIR = Path("db")
EMBEDDINGS_FILE = DB_DIR / "embeddings.npy"
CHUNKS_FILE = DB_DIR / "chunks.json"
EMBEDDING_MODEL = "text-embedding-3-small"

class KnowledgeRetriever:
    """
    A class to retrieve relevant knowledge chunks from a vectorized database.
    """
    def __init__(self):
        """Initializes the retriever, loading the knowledge base and model."""
        self.embeddings: np.ndarray = None
        self.chunks: List[Dict[str, Any]] = []
        self._load_knowledge_base()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set for the retriever.")
        self.client = OpenAI(api_key=api_key)

    def _load_knowledge_base(self):
        """Loads embeddings and text chunks from disk."""
        logging.info("Loading knowledge base...")
        if not EMBEDDINGS_FILE.exists() or not CHUNKS_FILE.exists():
            msg = (
                "Knowledge base not found. Please run "
                "'scripts/build_knowledge_base.py' first."
            )
            logging.error(msg)
            raise FileNotFoundError(msg)

        try:
            self.embeddings = np.load(EMBEDDINGS_FILE)
            with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            logging.info(
                f"Knowledge base loaded successfully. "
                f"({len(self.chunks)} chunks)"
            )
        except Exception as e:
            logging.error(f"Failed to load knowledge base: {e}")
            raise

    def retrieve(
        self, query: str, top_k: int = 3, filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the top_k most relevant chunks for a given query,
        with optional metadata filtering.

        Args:
            query (str): The search query.
            top_k (int): The number of top results to return.
            filters (Dict[str, Any]): A dictionary for filtering based on metadata.
                                      Example: {"tags": ["style-guide"]}

        Returns:
            List[Dict[str, Any]]: A list of the most relevant chunks.
        """
        if self.embeddings is None or not self.chunks:
            logging.warning("Knowledge base is not loaded. Cannot retrieve.")
            return []

        # 1. Create query embedding using OpenAI
        response = self.client.embeddings.create(input=[query], model=EMBEDDING_MODEL)
        query_embedding = np.array([response.data[0].embedding])

        # 2. Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings).flatten()

        # 3. Get a larger number of candidates to filter from
        candidate_k = min(top_k * 5, len(self.chunks)) # Get more to filter
        top_k_indices = np.argpartition(similarities, -candidate_k)[-candidate_k:]
        
        # 4. Filter candidates based on metadata
        filtered_indices = []
        if filters and "tags" in filters:
            required_tags = set(filters["tags"])
            if required_tags: # Only filter if tags are specified
                for idx in top_k_indices:
                    chunk_tags = set(self.chunks[idx].get("metadata", {}).get("tags", []))
                    if required_tags.issubset(chunk_tags):
                        filtered_indices.append(idx)
            else: # If tags filter is empty, don't filter
                filtered_indices = list(top_k_indices)
        else: # No filters
            filtered_indices = list(top_k_indices)
            
        # 5. Sort the filtered indices by similarity score
        sorted_indices = sorted(filtered_indices, key=lambda idx: similarities[idx], reverse=True)
        
        # 6. Get the final top_k results
        final_indices = sorted_indices[:top_k]

        # 7. Format results
        results = []
        for idx in final_indices:
            result = {
                "text": self.chunks[idx]["text"],
                "source": self.chunks[idx]["source"],
                "metadata": self.chunks[idx].get("metadata", {}),
                "score": float(similarities[idx]),
            }
            results.append(result)
            
        if not results:
            logging.warning(f"No documents found for query '{query[:50]}...' with filters {filters}")

        return results 