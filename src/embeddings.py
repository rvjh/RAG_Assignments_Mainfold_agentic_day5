"""
Embedding layer for RAG pipeline.

Takes semantic chunks produced by ingestion.py and
adds an embedding vector to every chunk.

Pipeline:
    load_documents()
        ↓
    chunk_documents()
        ↓
    embed_chunks()

"""

from pathlib import Path
import sys
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from src.ingestion import load_documents, chunk_documents


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# 2. CREATE HUGGING FACE EMBEDDING MODEL
# ============================================================

embeddings = HuggingFaceEndpointEmbeddings(model="Octen/Octen-Embedding-0.6B")

# ============================================================
# 3. EMBED CHUNKS
# ============================================================

def embed_chunks(chunks: list) -> list:
    """
    Compute embeddings for each chunk.

    Args:
        chunks:
            List of chunk dictionaries produced by
            chunk_documents().

    Returns:
        Same list, but each chunk contains an
        "embedding" field with a list of floats.
    """

    if not chunks:
        return []

    # Get text from every chunk
    texts = [chunk["text"]for chunk in chunks]

    # Generate embeddings
    embedding_vectors = embeddings.embed_documents(texts)

    # Add embeddings to the existing chunk dictionaries
    for chunk, vector in zip(chunks, embedding_vectors):

        chunk["embedding"] = [
            float(value)
            for value in vector
        ]

    return chunks


# # ============================================================
# # 4. RUN COMPLETE PIPELINE
# # ============================================================

# if __name__ == "__main__":

#     # --------------------------------------------------------
#     # Layer 1: Ingestion
#     # --------------------------------------------------------

#     documents = load_documents()

#     print("=" * 100)
#     print("LAYER 1 - DOCUMENT INGESTION")
#     print("=" * 100)

#     print(f"Loaded {len(documents)} documents")

#     for document in documents:
#         print(
#             f"- {document['id']} "
#             f"({document['source_type']})"
#         )


#     # --------------------------------------------------------
#     # Layer 2: Chunking
#     # --------------------------------------------------------

#     chunks = chunk_documents(documents)

#     print("\n" + "=" * 100)
#     print("LAYER 2 - SEMANTIC CHUNKING")
#     print("=" * 100)

#     print(f"Created {len(chunks)} chunks")


#     # --------------------------------------------------------
#     # Layer 3: Embeddings
#     # --------------------------------------------------------

#     embedded_chunks = embed_chunks(chunks)

#     print("\n" + "=" * 100)
#     print("LAYER 3 - EMBEDDINGS")
#     print("=" * 100)

#     print(
#         f"Generated embeddings for "
#         f"{len(embedded_chunks)} chunks"
#     )


#     # --------------------------------------------------------
#     # Display results
#     # --------------------------------------------------------

#     for chunk in embedded_chunks:

#         print("\n" + "*" * 100)

#         print(f"Document ID: {chunk['doc_id']}")
#         print(f"Chunk ID: {chunk['chunk_id']}")
#         print(
#             f"Source Type: "
#             f"{chunk['metadata']['source_type']}"
#         )
#         print(
#             f"Position: "
#             f"{chunk['metadata']['position']}"
#         )

#         print(
#             f"Embedding Length: "
#             f"{len(chunk['embedding'])}"
#         )

#         print(
#             f"Embedding Type: "
#             f"{type(chunk['embedding']).__name__}"
#         )

#         print(
#             f"First 10 Embedding Values: "
#             f"{chunk['embedding'][:10]}"
#         )

#         print("\nChunk Text:")
#         print(chunk["text"])

#         print("*" * 100)