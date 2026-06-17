import os
import json
import time
import asyncio
import traceback
import numpy as np
import litellm
from pypdf import PdfReader
from agents import function_tool

# Embeddings model config
COHERE_EMBED_MODEL = "cohere/embed-multilingual-v3.0"
CACHE_FILE = "embeddings_cache.json"
PDF_PATH = os.path.join("rag-content", "Inbound_Sirjami.pdf")

# In-memory storage for chunks and embedding vectors
KNOWLEDGE_CHUNKS = []
KNOWLEDGE_EMBEDDINGS = None  # NumPy array of shape (num_chunks, vector_dim)

# =============================================================================
# 1. PDF Text Extraction
# =============================================================================
def extract_text_from_pdf(filepath: str) -> str:
    """Extracts all text from a PDF file using pypdf."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PDF file not found at: {filepath}")
    
    print(f"[RAG] Reading PDF file: {filepath}...")
    reader = PdfReader(filepath)
    text = ""
    for idx, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    print(f"[RAG] Extracted {len(text)} characters of text from {len(reader.pages)} pages.")
    return text


# =============================================================================
# 2. Text Chunking
# =============================================================================
def chunk_text(text: str, chunk_size: int = 700, overlap: int = 100) -> list[str]:
    """
    Chunks text into sizes of chunk_size with overlap characters.
    Tries to split cleanly on sentence boundaries.
    """
    chunks = []
    # Clean up excessive whitespace/newlines
    text = " ".join(text.split())
    
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Try to find a sentence boundary split point (. followed by space)
        split_idx = text.rfind(". ", start + chunk_size - overlap, end)
        if split_idx == -1:
            # Fallback to space split point
            split_idx = text.rfind(" ", start + chunk_size - overlap, end)
            
        if split_idx != -1:
            end = split_idx + 1
            
        chunks.append(text[start:end].strip())
        start = end - overlap
        
    print(f"[RAG] Split text into {len(chunks)} chunks.")
    return chunks


# =============================================================================
# 3. Create or Load Embeddings Cache
# =============================================================================
def initialize_knowledge_base():
    """
    Initializes the in-memory knowledge chunks and embeddings.
    Loads from cache if valid, otherwise extracts and generates new embeddings.
    """
    global KNOWLEDGE_CHUNKS, KNOWLEDGE_EMBEDDINGS
    
    cohere_key = os.getenv("COHERE_API_KEY")
    if not cohere_key:
        print("[WARNING] COHERE_API_KEY is not set. RAG retrieval will fail.")
        return

    # Check if the PDF file exists
    if not os.path.exists(PDF_PATH):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(PDF_PATH), exist_ok=True)
        print(f"[RAG] [INFO] Please place your PDF file at '{PDF_PATH}' to enable document search.")
        return

    pdf_mtime = os.path.getmtime(PDF_PATH)
    cache_loaded = False

    # Try loading from cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                
            # Verify cache is for the same file, modified time, and embedding model
            if (cache_data.get("file_path") == PDF_PATH and 
                cache_data.get("file_mtime") == pdf_mtime and 
                cache_data.get("model_name") == COHERE_EMBED_MODEL):
                
                KNOWLEDGE_CHUNKS = cache_data["chunks"]
                KNOWLEDGE_EMBEDDINGS = np.array(cache_data["embeddings"], dtype=np.float32)
                print(f"[RAG] Loaded {len(KNOWLEDGE_CHUNKS)} chunks from cache successfully (mtime matches).")
                cache_loaded = True
        except Exception as e:
            print(f"[RAG] Error reading cache file, will re-embed: {e}")

    # Generate embeddings if cache is invalid or missing
    if not cache_loaded:
        print("[RAG] Cache missing or outdated. Indexing PDF document...")
        try:
            text = extract_text_from_pdf(PDF_PATH)
            chunks = chunk_text(text)
            
            if not chunks:
                print("[RAG] No text found in PDF document.")
                return

            print(f"[RAG] Generating embeddings for {len(chunks)} chunks using Cohere...")
            
            # Call LiteLLM embedding API for all chunks
            # Cohere v3/v4 models require input_type="search_document"
            response = litellm.embedding(
                model=COHERE_EMBED_MODEL,
                input=chunks,
                input_type="search_document",
                api_key=cohere_key
            )
            
            # Extract embedding vectors
            embeddings = [item["embedding"] for item in response["data"]]
            
            KNOWLEDGE_CHUNKS = chunks
            KNOWLEDGE_EMBEDDINGS = np.array(embeddings, dtype=np.float32)
            
            # Save to cache file
            cache_data = {
                "file_path": PDF_PATH,
                "file_mtime": pdf_mtime,
                "model_name": COHERE_EMBED_MODEL,
                "chunks": chunks,
                "embeddings": embeddings
            }
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
            print(f"[RAG] Generated and cached {len(chunks)} embeddings successfully.")
            
        except Exception as e:
            print(f"[RAG] Failed to index PDF file: {e}")
            traceback.print_exc()


# =============================================================================
# 4. Search and Retrieval Tool
# =============================================================================
@function_tool
def query_knowledge_base(query: str) -> str:
    """Search the document database for information matching the query.
    Always use this tool if you need details about the 'Inbound Sirjami' document,
    refund policies, contact info, or custom business rules.

    Args:
        query: The search term or question to query the knowledge base for.
    """
    global KNOWLEDGE_CHUNKS, KNOWLEDGE_EMBEDDINGS
    
    print(f"\n[Tool Execution: query_knowledge_base(query={query!r})]")
    
    if not KNOWLEDGE_CHUNKS or KNOWLEDGE_EMBEDDINGS is None:
        print("[RAG Tool Warning] Knowledge base is not initialized or is empty.")
        return "Error: The document database is currently empty or not initialized."

    cohere_key = os.getenv("COHERE_API_KEY")
    if not cohere_key:
        print("[RAG Tool Warning] COHERE_API_KEY is not set.")
        return "Error: Cohere API key is missing. Cannot perform semantic search."

    try:
        # Generate embedding for the query
        # Cohere v3/v4 requires input_type="search_query"
        response = litellm.embedding(
            model=COHERE_EMBED_MODEL,
            input=[query],
            input_type="search_query",
            api_key=cohere_key
        )
        
        query_vector = np.array(response["data"][0]["embedding"], dtype=np.float32)
        
        # Calculate dot products (cosine similarity since vectors are normalized)
        similarities = np.dot(KNOWLEDGE_EMBEDDINGS, query_vector)
        
        # Get indices of top 3 matches
        top_k = min(3, len(KNOWLEDGE_CHUNKS))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Format results
        context_parts = []
        for rank, idx in enumerate(top_indices):
            score = similarities[idx]
            context_parts.append(f"--- MATCH {rank+1} (Relevance Score: {score:.3f}) ---\n{KNOWLEDGE_CHUNKS[idx]}")
            
        context_result = "\n\n".join(context_parts)
        print(f"[RAG Tool] Found {top_k} matches. Highest similarity score: {similarities[top_indices[0]]:.3f}")
        return context_result
        
    except Exception as e:
        print(f"[RAG Tool Error] Similarity retrieval failed: {e}")
        traceback.print_exc()
        return f"Error retrieving context for query: {e}"
