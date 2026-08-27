"""
============================================================
LAYER 5 + LAYER 7
QUERY UNDERSTANDING + HYBRID SEARCH
============================================================

Layer 5:
    rewrite_query()

Layer 6:
    Access control

Layer 7:
    lexical_search()
    vector search
    RRF fusion
    hybrid_search()

Pipeline:

    User Query
         |
         v
    Query Rewrite
         |
         v
    Access Control
         |
         +----------------------+
         |                      |
         v                      v
    Vector Search          Lexical Search
         |                      |
         +----------+-----------+
                    |
                    v
               RRF Fusion
                    |
                    v
                Top K

IMPORTANT:
    Access control is applied before final ranking.

This module does not own:
    - document ingestion
    - chunking
    - embedding creation
    - PostgreSQL connection

Those are handled by:
    src.ingestion
    src.embeddings
    src.vector_store
    src.access_control
============================================================
"""

from typing import List, Dict, Any
import math
import re


# ============================================================
# GLOBAL PIPELINE STATE
# ============================================================

_PIPELINE_CHUNKS: List[Dict[str, Any]] = []
_VECTOR_STORE = None


# ============================================================
# CONFIGURE PIPELINE
# ============================================================

def configure_pipeline(
    chunks: list,
    vector_store,
) -> None:
    """
    Configure Layers 5-7.

    Args:
        chunks:
            List of embedded chunk dictionaries.

        vector_store:
            Initialized VectorStore instance.
    """

    global _PIPELINE_CHUNKS
    global _VECTOR_STORE

    if chunks is None:
        chunks = []

    if not isinstance(chunks, list):
        raise TypeError(
            "chunks must be a list of chunk dictionaries."
        )

    if vector_store is None:
        raise ValueError(
            "vector_store cannot be None."
        )

    _PIPELINE_CHUNKS = chunks
    _VECTOR_STORE = vector_store


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def _normalize_text(
    text: str,
) -> str:
    """
    Normalize text for lexical matching.
    """

    if text is None:
        return ""

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9$%.]+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# TOKENIZATION
# ============================================================

def _tokenize(
    text: str,
) -> List[str]:
    """
    Convert text into normalized tokens.
    """

    normalized = _normalize_text(text)

    if not normalized:
        return []

    return normalized.split()


# ============================================================
# LAYER 5 — QUERY UNDERSTANDING
# ============================================================

def rewrite_query(
    user_query: str,
) -> Dict[str, str]:
    """
    Rewrite and expand a user query.

    Returns:

        {
            "original": "...",
            "expanded": "...",
            "intent": "lookup"
        }
    """

    if not isinstance(
        user_query,
        str,
    ):
        user_query = str(user_query)

    original = user_query.strip()

    # --------------------------------------------------------
    # Empty query
    # --------------------------------------------------------

    if not original:
        return {
            "original": "",
            "expanded": "",
            "intent": "lookup",
        }

    query_lower = original.lower()

    # --------------------------------------------------------
    # Intent detection
    # --------------------------------------------------------

    comparison_terms = [
        "compare",
        "comparison",
        "versus",
        " vs ",
        "difference",
        "better",
        "cheaper",
    ]

    policy_terms = [
        "policy",
        "policies",
        "rule",
        "rules",
        "allowed",
        "approval",
        "approved",
        "required",
        "requirement",
        "retention",
        "procedure",
        "process",
        "submit",
        "reimbursement",
        "expense claim",
    ]

    if any(
        term in query_lower
        for term in comparison_terms
    ):
        intent = "compare"

    elif any(
        term in query_lower
        for term in policy_terms
    ):
        intent = "policy"

    else:
        intent = "lookup"

    # --------------------------------------------------------
    # Synonym expansion
    # --------------------------------------------------------

    synonym_map = {

        "revenue": [
            "revenue",
            "sales",
            "income",
            "earnings",
        ],

        "profit": [
            "profit",
            "earnings",
            "margin",
            "net profit",
        ],

        "expense": [
            "expense",
            "expenses",
            "reimbursement",
            "claim",
            "cost",
        ],

        "aws": [
            "aws",
            "amazon",
            "amazon web services",
            "cloud",
            "ec2",
        ],

        "azure": [
            "azure",
            "microsoft",
            "cloud",
        ],

        "pricing": [
            "pricing",
            "price",
            "cost",
            "spend",
        ],

        "cost": [
            "cost",
            "price",
            "pricing",
            "spend",
            "expense",
        ],

        "security": [
            "security",
            "incident",
            "response",
            "cybersecurity",
        ],

        "incident": [
            "incident",
            "security incident",
            "event",
            "response",
        ],

        "api": [
            "api",
            "application programming interface",
        ],

        "limit": [
            "limit",
            "rate limit",
            "throttling",
            "requests",
        ],

        "rate": [
            "rate",
            "rate limit",
            "requests",
            "throttling",
        ],

        "employee": [
            "employee",
            "staff",
            "worker",
        ],

        "contract": [
            "contract",
            "agreement",
            "vendor",
        ],

        "renew": [
            "renew",
            "renewal",
            "expiration",
        ],

        "renewal": [
            "renew",
            "renewal",
            "expiration",
        ],

        "payment": [
            "payment",
            "payment terms",
            "billing",
        ],

        "launch": [
            "launch",
            "release",
            "availability",
        ],

        "retention": [
            "retention",
            "storage",
            "retained",
            "retention period",
        ],

        "hr": [
            "hr",
            "human resources",
        ],
    }

    original_tokens = _tokenize(
        original
    )

    expansion_terms = []

    for token in original_tokens:

        if token in synonym_map:

            expansion_terms.extend(
                synonym_map[token]
            )

    # --------------------------------------------------------
    # Remove duplicate expansion terms
    # --------------------------------------------------------

    unique_terms = []

    for term in expansion_terms:

        if term not in unique_terms:
            unique_terms.append(term)

    expanded = original

    if unique_terms:

        expanded = (
            original
            + " "
            + " ".join(unique_terms)
        )

    return {
        "original": original,
        "expanded": expanded.strip(),
        "intent": intent,
    }


# ============================================================
# LAYER 7 — LEXICAL SEARCH
# ============================================================

def lexical_search(
    query: str,
    chunks: list,
    top_k: int = 5,
) -> list:
    """
    Perform lexical retrieval.

    Uses lightweight TF-style scoring.

    Returns:

        [
            {
                "doc_id": str,
                "chunk_id": str,
                "score": float,
                "metadata": dict,
                "text": str
            }
        ]
    """

    if not chunks:
        return []

    if top_k <= 0:
        return []

    query_tokens = _tokenize(
        query
    )

    if not query_tokens:
        return []

    # --------------------------------------------------------
    # Query term frequency
    # --------------------------------------------------------

    query_counts = {}

    for token in query_tokens:

        query_counts[token] = (
            query_counts.get(token, 0)
            + 1
        )

    scored_results = []

    # --------------------------------------------------------
    # Score every chunk
    # --------------------------------------------------------

    for chunk in chunks:

        if not isinstance(
            chunk,
            dict,
        ):
            continue

        text = str(
            chunk.get(
                "text",
                "",
            )
        )

        if not text.strip():
            continue

        document_tokens = _tokenize(
            text
        )

        if not document_tokens:
            continue

        document_counts = {}

        for token in document_tokens:

            document_counts[token] = (
                document_counts.get(token, 0)
                + 1
            )

        score = 0.0

        for token, query_count in query_counts.items():

            frequency = document_counts.get(
                token,
                0,
            )

            if frequency > 0:

                tf = math.log1p(
                    frequency
                )

                score += (
                    query_count * tf
                )

        if score <= 0:
            continue

        scored_results.append(
            {
                "doc_id": str(
                    chunk.get(
                        "doc_id",
                        "",
                    )
                ),

                "chunk_id": str(
                    chunk.get(
                        "chunk_id",
                        "",
                    )
                ),

                "score": float(
                    score
                ),

                "metadata": dict(
                    chunk.get(
                        "metadata",
                        {},
                    )
                ),

                # IMPORTANT:
                # Preserve the actual chunk text.
                "text": text,
            }
        )

    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    scored_results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return scored_results[:top_k]


# ============================================================
# LAYER 7 — VECTOR SEARCH
# ============================================================

def _vector_search(
    query: str,
    top_k: int = 5,
) -> list:
    """
    Generate query embedding and search VectorStore.
    """

    if _VECTOR_STORE is None:
        return []

    if not query or not query.strip():
        return []

    try:

        from src.embeddings import embeddings

        query_embedding = (
            embeddings.embed_query(
                query
            )
        )

        results = _VECTOR_STORE.search(
            query_embedding,
            top_k=top_k,
        )

        if results is None:
            return []

        return list(results)

    except Exception as exc:

        print(
            "Warning: vector search failed: "
            f"{exc}"
        )

        return []


# ============================================================
# NORMALIZE VECTOR RESULT
# ============================================================

def _normalize_vector_result(
    result: dict,
) -> dict:
    """
    Normalize a vector-store result.

    Different VectorStore implementations may return
    slightly different fields.

    This function makes sure the final pipeline has:

        doc_id
        chunk_id
        score
        metadata
        text
    """

    if not isinstance(
        result,
        dict,
    ):
        return {
            "doc_id": "",
            "chunk_id": "",
            "score": 0.0,
            "metadata": {},
            "text": "",
        }

    doc_id = str(
        result.get(
            "doc_id",
            result.get(
                "document_id",
                "",
            ),
        )
    )

    chunk_id = str(
        result.get(
            "chunk_id",
            result.get(
                "id",
                "",
            ),
        )
    )

    metadata = result.get(
        "metadata",
        {},
    )

    if not isinstance(
        metadata,
        dict,
    ):
        metadata = {}

    text = result.get(
        "text",
        result.get(
            "content",
            "",
        ),
    )

    # --------------------------------------------------------
    # If vector store doesn't return text directly,
    # recover it from the original configured chunks.
    # --------------------------------------------------------

    if not text:

        for chunk in _PIPELINE_CHUNKS:

            if not isinstance(
                chunk,
                dict,
            ):
                continue

            if (
                str(
                    chunk.get(
                        "doc_id",
                        "",
                    )
                )
                == doc_id
                and
                str(
                    chunk.get(
                        "chunk_id",
                        "",
                    )
                )
                == chunk_id
            ):

                text = chunk.get(
                    "text",
                    "",
                )

                # Also recover metadata if needed.
                if not metadata:

                    chunk_metadata = chunk.get(
                        "metadata",
                        {},
                    )

                    if isinstance(
                        chunk_metadata,
                        dict,
                    ):
                        metadata = dict(
                            chunk_metadata
                        )

                break

    # --------------------------------------------------------
    # Some vector stores may return similarity instead
    # of score.
    # --------------------------------------------------------

    score = result.get(
        "score",
        result.get(
            "similarity",
            0.0,
        ),
    )

    try:
        score = float(score)
    except (
        TypeError,
        ValueError,
    ):
        score = 0.0

    return {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "score": score,
        "metadata": dict(metadata),
        "text": str(text or ""),
    }


# ============================================================
# FILTER VECTOR RESULTS BY ACCESS
# ============================================================

def _filter_vector_results_by_access(
    vector_results: list,
    accessible_chunks: list,
) -> list:
    """
    Restrict vector results to chunks allowed for the
    current user role.
    """

    accessible_keys = {
        (
            str(
                chunk.get(
                    "doc_id",
                    "",
                )
            ),
            str(
                chunk.get(
                    "chunk_id",
                    "",
                )
            ),
        )
        for chunk in accessible_chunks
        if isinstance(
            chunk,
            dict,
        )
    }

    filtered = []

    for raw_result in vector_results:

        result = _normalize_vector_result(
            raw_result
        )

        key = (
            result["doc_id"],
            result["chunk_id"],
        )

        if key in accessible_keys:

            filtered.append(
                result
            )

    return filtered


# ============================================================
# RRF FUSION
# ============================================================

def _rrf_fusion(
    vector_results: list,
    lexical_results: list,
    top_k: int = 5,
    rrf_k: int = 60,
) -> list:
    """
    Reciprocal Rank Fusion.

    Formula:

        RRF score = 1 / (rrf_k + rank)

    Results appearing in both vector and lexical rankings
    receive contributions from both.
    """

    if top_k <= 0:
        return []

    fused = {}

    # ========================================================
    # VECTOR RESULTS
    # ========================================================

    for rank, raw_result in enumerate(
        vector_results,
        start=1,
    ):

        result = _normalize_vector_result(
            raw_result
        )

        key = (
            result["doc_id"],
            result["chunk_id"],
        )

        if key not in fused:

            fused[key] = {
                "doc_id": result["doc_id"],
                "chunk_id": result["chunk_id"],
                "score": 0.0,
                "metadata": dict(
                    result["metadata"]
                ),
                "text": result["text"],
            }

        fused[key]["score"] += (
            1.0
            / (
                rrf_k + rank
            )
        )

        # Preserve text.
        if not fused[key]["text"]:

            fused[key]["text"] = (
                result["text"]
            )

        # Preserve metadata.
        if not fused[key]["metadata"]:

            fused[key]["metadata"] = dict(
                result["metadata"]
            )

    # ========================================================
    # LEXICAL RESULTS
    # ========================================================

    for rank, result in enumerate(
        lexical_results,
        start=1,
    ):

        result = _normalize_vector_result(
            result
        )

        key = (
            result["doc_id"],
            result["chunk_id"],
        )

        if key not in fused:

            fused[key] = {
                "doc_id": result["doc_id"],
                "chunk_id": result["chunk_id"],
                "score": 0.0,
                "metadata": dict(
                    result["metadata"]
                ),
                "text": result["text"],
            }

        fused[key]["score"] += (
            1.0
            / (
                rrf_k + rank
            )
        )

        # IMPORTANT:
        # Always preserve lexical text.
        if not fused[key]["text"]:

            fused[key]["text"] = (
                result["text"]
            )

        if not fused[key]["metadata"]:

            fused[key]["metadata"] = dict(
                result["metadata"]
            )

    # ========================================================
    # FINAL SORT
    # ========================================================

    fused_results = list(
        fused.values()
    )

    fused_results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return fused_results[:top_k]


# ============================================================
# LAYER 5-7 — HYBRID SEARCH
# ============================================================

def hybrid_search(
    user_query: str,
    user_role: str,
    top_k: int = 5,
) -> dict:
    """
    Complete Layer 5-7 retrieval pipeline.

    Steps:

        Layer 5:
            Query rewriting

        Layer 6:
            Access control

        Layer 7:
            Vector search
            Lexical search
            RRF fusion

    Returns:

        {
            "query": {
                "original": "...",
                "expanded": "...",
                "intent": "lookup"
            },

            "user_role": "employee",

            "results": [
                {
                    "doc_id": "...",
                    "chunk_id": "...",
                    "score": 0.123,
                    "metadata": {...},
                    "text": "actual chunk text"
                }
            ]
        }
    """

    # --------------------------------------------------------
    # Validate query
    # --------------------------------------------------------

    if not isinstance(
        user_query,
        str,
    ):
        user_query = str(user_query)

    if not isinstance(
        user_role,
        str,
    ):
        user_role = str(user_role)

    if top_k <= 0:

        return {
            "query": rewrite_query(
                user_query
            ),
            "user_role": user_role,
            "results": [],
        }

    # ========================================================
    # LAYER 5 — QUERY UNDERSTANDING
    # ========================================================

    rewritten = rewrite_query(
        user_query
    )

    expanded_query = rewritten.get(
        "expanded",
        "",
    )

    if not expanded_query:

        expanded_query = user_query

    # ========================================================
    # LAYER 6 — ACCESS CONTROL
    # ========================================================

    from src.access_control import (
        filter_chunks_by_access,
    )

    accessible_chunks = (
        filter_chunks_by_access(
            _PIPELINE_CHUNKS,
            user_role,
        )
    )

    if not accessible_chunks:

        return {
            "query": rewritten,
            "user_role": user_role,
            "results": [],
        }

    # ========================================================
    # LAYER 7A — VECTOR SEARCH
    # ========================================================

    vector_candidates = _vector_search(
        expanded_query,
        top_k=max(
            top_k * 10,
            50,
        ),
    )

    vector_results = (
        _filter_vector_results_by_access(
            vector_candidates,
            accessible_chunks,
        )
    )

    # We intentionally keep only top_k vector results
    # before RRF.
    vector_results = vector_results[:top_k]

    # ========================================================
    # LAYER 7B — LEXICAL SEARCH
    # ========================================================

    lexical_results = lexical_search(
        expanded_query,
        accessible_chunks,
        top_k=top_k,
    )

    # ========================================================
    # LAYER 7C — RRF
    # ========================================================

    fused_results = _rrf_fusion(
        vector_results,
        lexical_results,
        top_k=top_k,
    )

    # ========================================================
    # FINAL NORMALIZATION
    # ========================================================

    final_results = []

    for result in fused_results:

        final_results.append(
            {
                "doc_id": str(
                    result.get(
                        "doc_id",
                        "",
                    )
                ),

                "chunk_id": str(
                    result.get(
                        "chunk_id",
                        "",
                    )
                ),

                "score": float(
                    result.get(
                        "score",
                        0.0,
                    )
                ),

                "metadata": dict(
                    result.get(
                        "metadata",
                        {},
                    )
                ),

                # IMPORTANT:
                # Return actual retrieved chunk text.
                "text": str(
                    result.get(
                        "text",
                        "",
                    )
                ),
            }
        )

    # ========================================================
    # RETURN COMPLETE PIPELINE RESPONSE
    # ========================================================

    return {
        "query": rewritten,
        "user_role": user_role,
        "results": final_results,
    }


# ============================================================
# PIPELINE STATUS
# ============================================================

def pipeline_status() -> dict:
    """
    Return basic pipeline state.
    """

    return {
        "configured": (
            _VECTOR_STORE is not None
        ),

        "chunks": len(
            _PIPELINE_CHUNKS
        ),
    }
