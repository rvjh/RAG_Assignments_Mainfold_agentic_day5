"""
Vector Store Layer
==================

RAG Pipeline:

    ingestion.py
        |
        +-- load_documents()
        |
        +-- chunk_documents()
        |
        v
    embedding.py
        |
        +-- embed_chunks()
        |
        v
    vector_store.py
        |
        +-- VectorStore.add_documents()
        |
        +-- VectorStore.search()
        |
        v
    PostgreSQL + pgvector
"""


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import sys
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv


# ============================================================
# 2. ADD SRC DIRECTORY TO PYTHON PATH
# ============================================================

SRC_DIR = Path(__file__).parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# 3. IMPORT FROM INGESTION
# ============================================================

from src.ingestion import (
    load_documents,
    chunk_documents,
)


# ============================================================
# 4. IMPORT FROM EMBEDDING
# ============================================================

from src.embeddings import (
    embed_chunks,
    embeddings,
)


# ============================================================
# 5. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# 6. DATABASE CONFIGURATION
# ============================================================

CONNECTION_STRING = os.getenv(
    "DATABASE_URL",
    "postgresql://rag_user:rag_password@localhost:6025/rag_database"
)


# ============================================================
# 7. HELPER FUNCTION
# ============================================================

def convert_to_pgvector(values):
    """
    Convert a Python list of floats into a pgvector string.

    Example:

        [0.1, 0.2, 0.3]

    becomes:

        "[0.1,0.2,0.3]"
    """

    return "[" + ",".join(
        str(float(value))
        for value in values
    ) + "]"


# ============================================================
# 8. VECTOR STORE CLASS
# ============================================================

class VectorStore:

    def __init__(self):
        """
        Connect to PostgreSQL and register pgvector.
        """

        self.conn = psycopg.connect(
            CONNECTION_STRING
        )

        register_vector(
            self.conn
        )


    # ========================================================
    # ADD DOCUMENTS
    # ========================================================

    def add_documents(
        self,
        embedded_chunks: list
    ) -> None:
        """
        Add embedded chunks to PostgreSQL.

        Required fields in each chunk:

            doc_id
            chunk_id
            text
            embedding
            metadata
        """

        if not embedded_chunks:
            return


        with self.conn.cursor() as cur:

            for chunk in embedded_chunks:

                metadata = chunk.get(
                    "metadata",
                    {}
                )


                source_type = metadata.get(
                    "source_type",
                    "unknown"
                )


                position = metadata.get(
                    "position",
                    0
                )


                # ------------------------------------------------
                # Convert embedding to pgvector string
                # ------------------------------------------------

                vector = convert_to_pgvector(
                    chunk["embedding"]
                )


                # ------------------------------------------------
                # Insert document
                # ------------------------------------------------

                cur.execute(
                    """
                    INSERT INTO documents (
                        doc_id,
                        chunk_id,
                        content,
                        embedding,
                        source_type,
                        position,
                        metadata
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s::vector,
                        %s,
                        %s,
                        %s
                    )

                    ON CONFLICT (chunk_id)
                    DO UPDATE SET

                        content = EXCLUDED.content,

                        embedding = EXCLUDED.embedding,

                        source_type = EXCLUDED.source_type,

                        position = EXCLUDED.position,

                        metadata = EXCLUDED.metadata
                    """,

                    (
                        chunk["doc_id"],
                        chunk["chunk_id"],
                        chunk["text"],
                        vector,
                        source_type,
                        position,
                        psycopg.types.json.Jsonb(
                            metadata
                        ),
                    )
                )


        self.conn.commit()


    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query_embedding,
        top_k: int = 5,
        allowed_chunk_ids: list | None = None,
    ) -> list:
        """
        Search the vector store using cosine similarity.

        Returns:

            [
                {
                    "doc_id": "...",
                    "chunk_id": "...",
                    "score": 0.85,
                    "metadata": {...}
                }
            ]
        """

        if query_embedding is None:
            return []


        if len(query_embedding) == 0:
            return []


        if top_k <= 0:
            return []


        # ----------------------------------------------------
        # Convert query embedding to pgvector format
        # ----------------------------------------------------

        query_vector = convert_to_pgvector(
            query_embedding
        )


        # ----------------------------------------------------
        # Search PostgreSQL
        # ----------------------------------------------------

        with self.conn.cursor() as cur:

            cur.execute(
                """
                SELECT
                    doc_id,
                    chunk_id,
                    content,

                    embedding <=> %s::vector
                        AS distance,

                    source_type,
                    position,
                    metadata

                FROM documents

                ORDER BY
                    embedding <=> %s::vector

                LIMIT %s
                """,

                (
                    query_vector,
                    query_vector,
                    top_k,
                )
            )


            rows = cur.fetchall()


        # ----------------------------------------------------
        # Create result list
        # ----------------------------------------------------

        results = []


        for row in rows:

            (
                doc_id,
                chunk_id,
                content,
                distance,
                source_type,
                position,
                metadata,
            ) = row


            # ------------------------------------------------
            # Convert cosine distance to similarity
            # ------------------------------------------------

            score = 1.0 - float(
                distance
            )


            # ------------------------------------------------
            # Build metadata
            # ------------------------------------------------

            result_metadata = {
                **(metadata or {}),
                "source_type": source_type,
                "position": position,
            }


            # ------------------------------------------------
            # Required result structure
            # ------------------------------------------------

            result = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "score": float(score),
                "metadata": result_metadata,
                "text": content,
            }


            results.append(
                result
            )


        return results


    # ========================================================
    # CLOSE CONNECTION
    # ========================================================

    def close(self):

        if self.conn:

            self.conn.close()

            print(
                "PostgreSQL connection closed"
            )


# # ============================================================
# # 9. COMPLETE PIPELINE TEST
# # ============================================================

# if __name__ == "__main__":

#     store = None

#     try:

#         # ====================================================
#         # LAYER 1 - INGESTION
#         # ====================================================

#         print("\n")
#         print("=" * 100)
#         print("LAYER 1 - DOCUMENT INGESTION")
#         print("=" * 100)


#         documents = load_documents()


#         print(
#             f"Loaded {len(documents)} documents"
#         )


#         for document in documents:

#             print(
#                 f"- {document['id']} "
#                 f"({document['source_type']})"
#             )


#         # ====================================================
#         # LAYER 2 - CHUNKING
#         # ====================================================

#         print("\n")
#         print("=" * 100)
#         print("LAYER 2 - SEMANTIC CHUNKING")
#         print("=" * 100)


#         chunks = chunk_documents(
#             documents
#         )


#         print(
#             f"Created {len(chunks)} chunks"
#         )


#         # ====================================================
#         # SHOW FIRST 3 CHUNKS
#         # ====================================================

#         for chunk in chunks[:3]:

#             print("\n")

#             print(
#                 f"Document ID: "
#                 f"{chunk['doc_id']}"
#             )

#             print(
#                 f"Chunk ID: "
#                 f"{chunk['chunk_id']}"
#             )

#             print(
#                 f"Position: "
#                 f"{chunk['metadata']['position']}"
#             )

#             print(
#                 f"Source Type: "
#                 f"{chunk['metadata']['source_type']}"
#             )

#             print(
#                 f"Text: "
#                 f"{chunk['text'][:200]}"
#             )


#         # ====================================================
#         # LAYER 3 - EMBEDDINGS
#         # ====================================================

#         print("\n")
#         print("=" * 100)
#         print("LAYER 3 - EMBEDDINGS")
#         print("=" * 100)


#         embedded_chunks = embed_chunks(
#             chunks
#         )


#         print(
#             f"Created embeddings for "
#             f"{len(embedded_chunks)} chunks"
#         )


#         if embedded_chunks:

#             print(
#                 "Embedding dimension: "
#                 f"{len(embedded_chunks[0]['embedding'])}"
#             )


#         # ====================================================
#         # LAYER 4 - VECTOR STORE
#         # ====================================================

#         print("\n")
#         print("=" * 100)
#         print("LAYER 4 - VECTOR STORE")
#         print("=" * 100)


#         store = VectorStore()


#         print(
#             "Connected to PostgreSQL + pgvector"
#         )


#         # ====================================================
#         # STORE EMBEDDINGS
#         # ====================================================

#         store.add_documents(
#             embedded_chunks
#         )


#         print(
#             f"Stored {len(embedded_chunks)} "
#             f"embedded chunks in PostgreSQL"
#         )


#         # ====================================================
#         # QUERY
#         # ====================================================

#         query = "Liability and Damages"


#         print("\n")
#         print("=" * 100)
#         print("VECTOR SEARCH")
#         print("=" * 100)


#         print(
#             f"Query: {query}"
#         )


#         # ====================================================
#         # QUERY EMBEDDING
#         # ====================================================

#         query_embedding = embeddings.embed_query(
#             query
#         )


#         print(
#             "Query embedding dimension: "
#             f"{len(query_embedding)}"
#         )


#         # ====================================================
#         # SEARCH
#         # ====================================================

#         results = store.search(
#             query_embedding,
#             top_k=5
#         )


#         # ====================================================
#         # DISPLAY RESULTS
#         # ====================================================

#         print("\n")
#         print("=" * 100)
#         print("SEARCH RESULTS")
#         print("=" * 100)


#         print(
#             f"Found {len(results)} results"
#         )


#         for index, result in enumerate(
#             results,
#             start=1
#         ):

#             print("\n")
#             print("*" * 100)


#             print(
#                 f"Rank: {index}"
#             )


#             print(
#                 f"Document ID: "
#                 f"{result['doc_id']}"
#             )


#             print(
#                 f"Chunk ID: "
#                 f"{result['chunk_id']}"
#             )


#             print(
#                 f"Score: "
#                 f"{result['score']:.6f}"
#             )


#             print(
#                 f"Source Type: "
#                 f"{result['metadata']['source_type']}"
#             )


#             print(
#                 f"Position: "
#                 f"{result['metadata']['position']}"
#             )


#             print("\nMetadata:")

#             print(
#                 result["metadata"]
#             )


#             print("\nText:")

#             print(
#                 result["text"]
#             )


#             print("*" * 100)


#     except Exception as e:

#         print("\n")
#         print("=" * 100)
#         print("ERROR")
#         print("=" * 100)


#         print(
#             f"{type(e).__name__}: {e}"
#         )


#         raise


#     finally:

#         if store is not None:

#             store.close()
