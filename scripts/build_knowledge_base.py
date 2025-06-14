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

def chunk_text(
    doc: Dict[str, Any],
    max_tokens: int = TEXT_CHUNK_MAX_TOKENS,
) -> List[Dict[str, Any]]:
    """Splits a document into smaller, semantically meaningful chunks."""
    chunks = []
    doc_text = doc["text"]
    source_name = doc["source"]
    metadata = doc["metadata"]

    header_splits = re.split(r'(^## .+$|^### .+$)', doc_text, flags=re.MULTILINE)
    texts_to_process = [header_splits[0]]
    if len(header_splits) > 1:
        texts_to_process.extend([h + c for h, c in zip(header_splits[1::2], header_splits[2::2])])

    chunk_id_counter = 0
    for text_block in texts_to_process:
        if not text_block.strip():
            continue
        paragraphs = [p.strip() for p in text_block.split('\n\n') if p.strip()]
        for paragraph in paragraphs:
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            current_chunk = ""
            for sentence in sentences:
                if not sentence:
                    continue
                if len(tokenizer.encode(current_chunk + " " + sentence)) <= max_tokens:
                    current_chunk += " " + sentence
                else:
                    if current_chunk:
                        # Add metadata to each chunk
                        chunk_metadata = metadata.copy()
                        chunk_metadata["chunk_id"] = f"{source_name}_{chunk_id_counter}"
                        chunks.append({"text": current_chunk.strip(), "source": source_name, "metadata": chunk_metadata})
                        chunk_id_counter += 1
                    current_chunk = sentence
            if current_chunk:
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_id"] = f"{source_name}_{chunk_id_counter}"
                chunks.append({"text": current_chunk.strip(), "source": source_name, "metadata": chunk_metadata})
                chunk_id_counter += 1
    return chunks

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
        doc_chunks = chunk_text(doc)
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