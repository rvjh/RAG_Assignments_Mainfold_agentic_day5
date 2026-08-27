"""
============================================================
FULL 7-LAYER PRODUCTION RAG
============================================================

Day 5 Assignment: Full 7-Layer Production RAG System

Layers:

    Layer 1 - Document ingestion
    Layer 2 - Semantic chunking
    Layer 3 - Embeddings
    Layer 4 - PostgreSQL + pgvector
    Layer 5 - Query understanding
    Layer 6 - Access control
    Layer 7 - Hybrid retrieval + RRF

Evaluation:

    Precision@5
    Recall@5
    MRR

    Faithfulness
    Answer Relevancy
    Context Precision
    Context Recall

Run:

    python rag_full_7_layer.py

The root file is intentionally kept as a thin entrypoint.
The actual implementation lives inside src/.
============================================================
"""


# ============================================================
# LAYER 1 + LAYER 2
# ============================================================

from src.ingestion import (
    load_documents,
    chunk_documents,
)


# ============================================================
# LAYER 3
# ============================================================

from src.embeddings import (
    embed_chunks,
)


# ============================================================
# LAYER 4
# ============================================================

from src.vector_store import (
    VectorStore,
)


# ============================================================
# LAYER 6
# ============================================================

from src.access_control import (
    ensure_access_metadata,
)


# ============================================================
# LAYER 5 + LAYER 7
# ============================================================

from src.query_pipeline import (
    configure_pipeline,
    rewrite_query,
    hybrid_search,
)


# ============================================================
# EVALUATION
# ============================================================

from src.evaluation import (
    GOLDEN_QUERIES,
    evaluate_retrieval,
    evaluate_ragas,
)


# ============================================================
# BUILD INDEX
# ============================================================

def build_index():
    """
    Execute Layers 1-4.

    Returns:
        (
            vector_store,
            embedded_chunks
        )
    """

    # ========================================================
    # LAYER 1 — DOCUMENT INGESTION
    # ========================================================

    print()
    print("=" * 80)
    print("LAYER 1 - DOCUMENT INGESTION")
    print("=" * 80)

    documents = load_documents()

    if not isinstance(documents, list):
        raise TypeError(
            "load_documents() must return a list."
        )

    print(
        f"Loaded {len(documents)} documents"
    )

    for document in documents:

        print(
            f"- {document.get('id', 'unknown')} "
            f"({document.get('source_type', 'unknown')})"
        )

    # ========================================================
    # LAYER 2 — SEMANTIC CHUNKING
    # ========================================================

    print()
    print("=" * 80)
    print("LAYER 2 - SEMANTIC CHUNKING")
    print("=" * 80)

    chunks = chunk_documents(
        documents
    )

    if not isinstance(chunks, list):
        raise TypeError(
            "chunk_documents() must return a list."
        )

    # --------------------------------------------------------
    # Add access metadata.
    # --------------------------------------------------------

    chunks = ensure_access_metadata(
        chunks
    )

    if not isinstance(chunks, list):
        raise TypeError(
            "ensure_access_metadata() must return a list."
        )

    print(
        f"Created {len(chunks)} chunks"
    )

    # ========================================================
    # LAYER 3 — EMBEDDINGS
    # ========================================================

    print()
    print("=" * 80)
    print("LAYER 3 - EMBEDDINGS")
    print("=" * 80)

    embedded_chunks = embed_chunks(
        chunks
    )

    if not isinstance(
        embedded_chunks,
        list,
    ):
        raise TypeError(
            "embed_chunks() must return a list."
        )

    print(
        f"Generated embeddings for "
        f"{len(embedded_chunks)} chunks"
    )

    if embedded_chunks:

        first_embedding = embedded_chunks[0].get(
            "embedding",
            [],
        )

        print(
            "Embedding dimension: "
            f"{len(first_embedding)}"
        )

    # ========================================================
    # LAYER 4 — POSTGRESQL + PGVECTOR
    # ========================================================

    print()
    print("=" * 80)
    print("LAYER 4 - POSTGRESQL + PGVECTOR")
    print("=" * 80)

    vector_store = VectorStore()

    print(
        "Connected to PostgreSQL + pgvector"
    )

    vector_store.add_documents(
        embedded_chunks
    )

    print(
        f"Stored {len(embedded_chunks)} "
        f"embedded chunks"
    )

    return (
        vector_store,
        embedded_chunks,
    )


# ============================================================
# CONFIGURE QUERY PIPELINE
# ============================================================

def configure_query_pipeline(
    vector_store,
    chunks,
):
    """
    Configure Layers 5-7.

    IMPORTANT:
        configure_pipeline() expects:

            chunks
            vector_store

        in that exact order.
    """

    if vector_store is None:
        raise ValueError(
            "vector_store cannot be None."
        )

    if not isinstance(chunks, list):
        raise TypeError(
            "chunks must be a list."
        )

    configure_pipeline(
        chunks,
        vector_store,
    )


# ============================================================
# DEMO QUERY
# ============================================================

def run_demo_query():
    """
    Demonstrate Layers 5-7.
    """

    query = (
        "What was the company revenue in Q4 2024?"
    )

    user_role = "employee"

    print()
    print("=" * 80)
    print("LAYER 5-7 - QUERY PIPELINE DEMO")
    print("=" * 80)

    print(
        f"Query: {query}"
    )

    print(
        f"User role: {user_role}"
    )

    # ========================================================
    # LAYER 5 — QUERY UNDERSTANDING
    # ========================================================

    query_info = rewrite_query(
        query
    )

    print()
    print("Query understanding:")

    print(
        f"Original:  "
        f"{query_info.get('original', '')}"
    )

    print(
        f"Expanded:  "
        f"{query_info.get('expanded', '')}"
    )

    print(
        f"Intent:    "
        f"{query_info.get('intent', '')}"
    )

    # ========================================================
    # LAYER 6 + LAYER 7
    # ========================================================

    response = hybrid_search(
        user_query=query,
        user_role=user_role,
        top_k=5,
    )

    if not isinstance(response, dict):
        raise TypeError(
            "hybrid_search() must return a dictionary."
        )

    results = response.get(
        "results",
        [],
    )

    print()
    print("Hybrid search results:")

    if not results:

        print(
            "No accessible results found."
        )

        return

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    for rank, result in enumerate(
        results,
        start=1,
    ):

        metadata = result.get(
            "metadata",
            {},
        )

        print()
        print(
            f"Rank: {rank}"
        )

        print(
            f"Document: "
            f"{result.get('doc_id', '')}"
        )

        print(
            f"Chunk: "
            f"{result.get('chunk_id', '')}"
        )

        print(
            f"Score: "
            f"{float(result.get('score', 0.0)):.6f}"
        )

        print(
            f"Source: "
            f"{metadata.get('source_type', 'unknown')}"
        )

        print(
            f"Access: "
            f"{metadata.get('min_role', 'employee')}"
        )

        text = result.get(
            "text",
            "",
        )

        if text:

            clean_text = " ".join(
                str(text).split()
            )

            print(
                f"Text: "
                f"{clean_text[:300]}"
            )

        else:

            print(
                "Text: "
                "Not returned by VectorStore."
            )


# ============================================================
# RETRIEVAL EVALUATION
# ============================================================

def run_retrieval_evaluation():
    """
    Run retrieval evaluation.

    Metrics:

        Precision@5
        Recall@5
        MRR
    """

    print()
    print("=" * 80)
    print("LAYER 7 - RETRIEVAL EVALUATION")
    print("=" * 80)

    metrics = evaluate_retrieval(
        GOLDEN_QUERIES
    )

    print()
    print(
        "=== RETRIEVAL METRICS (7-LAYER RAG) ==="
    )

    print(
        f"Queries:        "
        f"{int(metrics.get('num_queries', 0))}"
    )

    print(
        f"Precision@5:    "
        f"{float(metrics.get('precision_at_5', 0.0)):.2f}"
    )

    print(
        f"Recall@5:       "
        f"{float(metrics.get('recall_at_5', 0.0)):.2f}"
    )

    print(
        f"MRR:            "
        f"{float(metrics.get('mrr', 0.0)):.2f}"
    )

    return metrics


# ============================================================
# RAGAS EVALUATION
# ============================================================

def run_ragas_evaluation():
    """
    Run RAGAS evaluation.

    Metrics:

        Faithfulness
        Answer Relevancy
        Context Precision
        Context Recall
    """

    print()
    print("=" * 80)
    print("LAYER 7 - RAGAS EVALUATION")
    print("=" * 80)

    metrics = evaluate_ragas(
        GOLDEN_QUERIES
    )

    print()
    print(
        "=== RAGAS METRICS (GENERATION) ==="
    )

    print(
        f"Faithfulness:        "
        f"{float(metrics.get('faithfulness', 0.0)):.2f}"
    )

    print(
        f"AnswerRelevancy:     "
        f"{float(metrics.get('answer_relevancy', 0.0)):.2f}"
    )

    print(
        f"ContextPrecision:    "
        f"{float(metrics.get('context_precision', 0.0)):.2f}"
    )

    print(
        f"ContextRecall:       "
        f"{float(metrics.get('context_recall', 0.0)):.2f}"
    )

    print()
    print(
        f"Lowest metric:       "
        f"{metrics.get('lowest_metric', '')}"
    )

    print(
        f"Root cause layer:    "
        f"{metrics.get('root_cause_layer', 'Layer 7')}"
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Main entrypoint.

    Execution order:

        Layer 1
            |
        Layer 2
            |
        Layer 3
            |
        Layer 4
            |
        Configure Layers 5-7
            |
        Demo query
            |
        Retrieval evaluation
            |
        RAGAS evaluation
    """

    vector_store = None

    try:

        # ====================================================
        # LAYERS 1-4
        # ====================================================

        (
            vector_store,
            chunks,
        ) = build_index()

        # ====================================================
        # CONFIGURE LAYERS 5-7
        # ====================================================

        configure_query_pipeline(
            vector_store,
            chunks,
        )

        # ====================================================
        # DEMO QUERY
        # ====================================================

        run_demo_query()

        # ====================================================
        # RETRIEVAL EVALUATION
        # ====================================================

        run_retrieval_evaluation()

        # ====================================================
        # RAGAS EVALUATION
        # ====================================================

        run_ragas_evaluation()

        # ====================================================
        # COMPLETE
        # ====================================================

        print()
        print("=" * 80)
        print("7-LAYER RAG PIPELINE COMPLETE")
        print("=" * 80)

    finally:

        if vector_store is not None:

            try:

                vector_store.close()

            except Exception as exc:

                print(
                    f"Warning: failed to close "
                    f"VectorStore: {exc}"
                )


# ============================================================
# SCRIPT ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    main()
