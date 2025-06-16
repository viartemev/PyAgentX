# scripts/build_knowledge_base.py

import json
import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Any
import pickle

import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from unstructured.partition.md import partition_md
from rank_bm25 import BM25Okapi
from semchunk.chunker import SemanticChunker

# --- Configuration ---
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Constants
KNOWLEDGE_DIR = Path("knowledge")
DB_DIR = Path("db")
EMBEDDINGS_FILE = DB_DIR / "embeddings.npy"
CHUNKS_FILE = DB_DIR / "chunks.json"
BM25_INDEX_FILE = DB_DIR / "bm25_index.pkl"
EMBEDDING_MODEL = "text-embedding-3-small"
TEXT_CHUNK_MAX_TOKENS = 128

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
client = OpenAI(api_key=api_key)

tokenizer = tiktoken.get_encoding("cl100k_base")
# Initialize the semantic chunker
semantic_chunker = SemanticChunker(
    embed_model=client,
    model_name=EMBEDDING_MODEL,
    max_chunk_size=TEXT_CHUNK_MAX_TOKENS,
    # The 'breakpoint_percentile_threshold' is a key parameter to tune.
    # It determines how different two sentences must be to create a split.
    # Lower value = more splits, higher value = fewer splits.
    # Let's start with a value that often works well.
    breakpoint_percentile_threshold=90
)

def load_and_partition_documents(directory: Path) -> List[Dict[str, Any]]:
    """
    Loads documents and extracts metadata from their file path.
    The subdirectories of the knowledge base are used as tags.
    Example: `knowledge/style-guide/python.md` -> tags: ['style-guide']
    """
    documents = []
    logging.info(f"Loading documents from {directory}...")
    for file_path in directory.rglob("*.md"):
        if file_path.name.startswith("."):
            continue
        logging.info(f"Processing file: {file_path.name}")
        try:
            # Extract tags from the path relative to the knowledge directory
            relative_path = file_path.relative_to(directory)
            tags = [part for part in relative_path.parts[:-1] if part]

            elements = partition_md(filename=str(file_path))
            text_content = "\n".join([el.text for el in elements])
            
            documents.append({
                "text": text_content,
                "source": file_path.name,
                "metadata": {
                    "tags": tags,
                    "full_path": str(file_path)
                }
            })
        except Exception as e:
            logging.error(f"Failed to process {file_path}: {e}")
    return documents

def semantic_chunk_document(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Splits a document into semantically coherent chunks using semchunk.
    """
    logging.info(f"Semantically chunking '{doc['source']}'...")
    try:
        # The chunker returns a list of text strings
        chunk_texts = semantic_chunker.chunk(doc["text"])
        
        chunks_with_metadata = []
        for i, chunk_text in enumerate(chunk_texts):
            chunk_metadata = doc["metadata"].copy()
            chunk_metadata["chunk_id"] = f"{doc['source']}_{i}"
            chunks_with_metadata.append({
                "text": chunk_text,
                "source": doc["source"],
                "metadata": chunk_metadata
            })
        
        logging.info(f"Successfully created {len(chunks_with_metadata)} semantic chunks for '{doc['source']}'.")
        return chunks_with_metadata
    except Exception as e:
        logging.error(f"Failed to chunk document {doc['source']}: {e}", exc_info=True)
        return []

def create_embeddings(texts: List[str]) -> np.ndarray:
    """Creates embeddings for a list of texts using OpenAI API."""
    logging.info(f"Creating embeddings for {len(texts)} text chunks with OpenAI...")
    response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings)

def main():
    """Main function to build the knowledge base."""
    logging.info("Starting to build the knowledge base...")
    KNOWLEDGE_DIR.mkdir(exist_ok=True)
    DB_DIR.mkdir(exist_ok=True)
    documents = load_and_partition_documents(KNOWLEDGE_DIR)
    if not documents:
        logging.warning("No documents found in the knowledge directory. Exiting.")
        return
    all_chunks_with_metadata = []
    for doc in documents:
        # Use the new semantic chunking function
        doc_chunks = semantic_chunk_document(doc)
        all_chunks_with_metadata.extend(doc_chunks)
    if not all_chunks_with_metadata:
        logging.warning("Could not create any chunks from the documents. Exiting.")
        return
    chunk_texts = [chunk["text"] for chunk in all_chunks_with_metadata]

    # Create and save vector embeddings
    embeddings = create_embeddings(chunk_texts)
    logging.info(f"Saving {len(embeddings)} embeddings to {EMBEDDINGS_FILE}")
    np.save(EMBEDDINGS_FILE, embeddings)

    # Create and save BM25 index for keyword search
    logging.info("Creating BM25 index...")
    tokenized_corpus = [doc.split(" ") for doc in chunk_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(BM25_INDEX_FILE, "wb") as f:
        pickle.dump(bm25, f)
    logging.info(f"BM25 index saved to {BM25_INDEX_FILE}")

    logging.info(f"Saving {len(all_chunks_with_metadata)} chunks to {CHUNKS_FILE}")
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks_with_metadata, f, ensure_ascii=False, indent=4)
    logging.info("Knowledge base build process completed successfully!")

if __name__ == "__main__":
    main() 