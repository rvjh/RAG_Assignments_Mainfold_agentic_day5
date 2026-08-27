"""
============================================================
LAYER 7 - EVALUATION
============================================================

Provides:

    GOLDEN_QUERIES
    validate_golden_queries()
    precision_at_k()
    recall_at_k()
    reciprocal_rank()
    evaluate_retrieval()
    evaluate_ragas()

Retrieval metrics:

    Precision@5
    Recall@5
    MRR

Generation metrics:

    Faithfulness
    AnswerRelevancy
    ContextPrecision
    ContextRecall

The retrieval evaluation uses document-level relevance.

Important:
    The retriever returns CHUNKS, but the golden dataset
    defines relevance at the DOCUMENT level.

Therefore duplicate document IDs are removed before
calculating Precision@5, Recall@5, and MRR.

RAGAS:
    This implementation uses the assignment-approved
    offline deterministic evaluation.

    It does NOT call a real RAGAS/LLM API.
============================================================
"""

from typing import List, Set


# ============================================================
# GOLDEN DATASET
# ============================================================

GOLDEN_QUERIES = [

    # ========================================================
    # FINANCIAL REPORT - 5 QUERIES
    # ========================================================

    {
        "query": "What was the company revenue in Q4 2024?",
        "relevant_doc_ids": [
            "01_financial_report.txt"
        ],
    },

    {
        "query": "What was the Q4 revenue?",
        "relevant_doc_ids": [
            "01_financial_report.txt"
        ],
    },

    {
        "query": "How much revenue did the company generate in Q4?",
        "relevant_doc_ids": [
            "01_financial_report.txt"
        ],
    },

    {
        "query": "What was the annual growth in 2024?",
        "relevant_doc_ids": [
            "01_financial_report.txt"
        ],
    },

    {
        "query": "What percentage of revenue came from enterprise software licenses?",
        "relevant_doc_ids": [
            "01_financial_report.txt"
        ],
    },


    # ========================================================
    # COMPANY POLICY - 5 QUERIES
    # ========================================================

    {
        "query": "What is the company expense policy?",
        "relevant_doc_ids": [
            "11_company_policy.docx"
        ],
    },

    {
        "query": "How do I submit an expense claim?",
        "relevant_doc_ids": [
            "11_company_policy.docx"
        ],
    },

    {
        "query": "What expenses require approval?",
        "relevant_doc_ids": [
            "11_company_policy.docx"
        ],
    },

    {
        "query": "What is the reimbursement process?",
        "relevant_doc_ids": [
            "11_company_policy.docx"
        ],
    },

    {
        "query": "What are the employee expense rules?",
        "relevant_doc_ids": [
            "11_company_policy.docx"
        ],
    },


    # ========================================================
    # BLOG - 5 QUERIES
    # ========================================================

    {
        "query": "What is discussed in the company blog?",
        "relevant_doc_ids": [
            "12_blog_post.html"
        ],
    },

    {
        "query": "What topics are covered in the blog post?",
        "relevant_doc_ids": [
            "12_blog_post.html"
        ],
    },

    {
        "query": "What does the company blog discuss?",
        "relevant_doc_ids": [
            "12_blog_post.html"
        ],
    },

    {
        "query": "What information is provided in the blog?",
        "relevant_doc_ids": [
            "12_blog_post.html"
        ],
    },

    {
        "query": "Summarize the company blog post",
        "relevant_doc_ids": [
            "12_blog_post.html"
        ],
    },


    # ========================================================
    # RESEARCH PAPER - 5 QUERIES
    # ========================================================

    {
        "query": "What does the PDF document discuss?",
        "relevant_doc_ids": [
            "1810.04805v2.pdf"
        ],
    },

    {
        "query": "What is described in the research paper?",
        "relevant_doc_ids": [
            "1810.04805v2.pdf"
        ],
    },

    {
        "query": "What are the main concepts in the PDF?",
        "relevant_doc_ids": [
            "1810.04805v2.pdf"
        ],
    },

    {
        "query": "What is the research paper about?",
        "relevant_doc_ids": [
            "1810.04805v2.pdf"
        ],
    },

    {
        "query": "Summarize the research document",
        "relevant_doc_ids": [
            "1810.04805v2.pdf"
        ],
    },
]


# ============================================================
# VALIDATE GOLDEN DATASET
# ============================================================

def validate_golden_queries(
    golden_queries: list = None,
) -> None:
    """
    Validate the golden dataset.

    Requirements:

        - At least 20 queries
        - Every query must be a dictionary
        - Every query must be a string
        - Every query must be non-empty
        - Every entry must contain at least one
          relevant document ID
        - Every document ID must be a non-empty string
    """

    if golden_queries is None:
        golden_queries = GOLDEN_QUERIES

    if not isinstance(
        golden_queries,
        list,
    ):
        raise AssertionError(
            "GOLDEN_QUERIES must be a list."
        )

    if len(golden_queries) < 20:
        raise AssertionError(
            "GOLDEN_QUERIES must contain at least 20 entries."
        )

    for index, item in enumerate(
        golden_queries,
        start=1,
    ):

        if not isinstance(
            item,
            dict,
        ):
            raise AssertionError(
                f"Golden query #{index} must be a dictionary."
            )

        query = item.get(
            "query"
        )

        if not isinstance(
            query,
            str,
        ):
            raise AssertionError(
                f"Golden query #{index} query must be a string."
            )

        if not query.strip():
            raise AssertionError(
                f"Golden query #{index} cannot be empty."
            )

        relevant = item.get(
            "relevant_doc_ids"
        )

        if not isinstance(
            relevant,
            list,
        ):
            raise AssertionError(
                f"Golden query #{index} relevant_doc_ids "
                "must be a list."
            )

        if not relevant:
            raise AssertionError(
                f"Golden query #{index} must contain "
                "at least one relevant document."
            )

        for doc_id in relevant:

            if not isinstance(
                doc_id,
                str,
            ):
                raise AssertionError(
                    f"Golden query #{index} contains "
                    "a non-string document ID."
                )

            if not doc_id.strip():
                raise AssertionError(
                    f"Golden query #{index} contains "
                    "an empty document ID."
                )


# ============================================================
# HELPER - UNIQUE DOCUMENT IDS
# ============================================================

def _unique_document_ids(
    retrieved_ids: List[str],
) -> List[str]:
    """
    Remove duplicate document IDs while preserving
    their first-ranked position.

    Example:

        financial
        financial
        policy
        financial
        blog

    becomes:

        financial
        policy
        blog

    This is required because retrieval returns chunks,
    while the golden dataset evaluates documents.
    """

    unique_ids = []

    seen: Set[str] = set()

    for doc_id in retrieved_ids:

        if not isinstance(
            doc_id,
            str,
        ):
            continue

        if not doc_id.strip():
            continue

        if doc_id in seen:
            continue

        seen.add(
            doc_id
        )

        unique_ids.append(
            doc_id
        )

    return unique_ids


# ============================================================
# PRECISION@K
# ============================================================

def precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int = 5,
) -> float:
    """
    Calculate document-level Precision@K.

    Precision@K =
        relevant documents in top K
        ----------------------------
        retrieved documents in top K
    """

    if k <= 0:
        return 0.0

    unique_ids = _unique_document_ids(
        retrieved_ids
    )

    top_k = unique_ids[:k]

    if not top_k:
        return 0.0

    relevant_count = sum(
        1
        for doc_id in top_k
        if doc_id in relevant_ids
    )

    score = (
        relevant_count
        / len(top_k)
    )

    return max(
        0.0,
        min(
            1.0,
            float(score),
        ),
    )


# ============================================================
# RECALL@K
# ============================================================

def recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int = 5,
) -> float:
    """
    Calculate document-level Recall@K.

    Recall@K =
        relevant documents retrieved in top K
        -------------------------------------
        total relevant documents
    """

    if k <= 0:
        return 0.0

    if not relevant_ids:
        return 0.0

    unique_ids = _unique_document_ids(
        retrieved_ids
    )

    top_k = unique_ids[:k]

    relevant_count = sum(
        1
        for doc_id in top_k
        if doc_id in relevant_ids
    )

    score = (
        relevant_count
        / len(relevant_ids)
    )

    return max(
        0.0,
        min(
            1.0,
            float(score),
        ),
    )


# ============================================================
# RECIPROCAL RANK
# ============================================================

def reciprocal_rank(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
) -> float:
    """
    Calculate Reciprocal Rank.

    Examples:

        rank 1 -> 1.00
        rank 2 -> 0.50
        rank 3 -> 0.33
        rank 4 -> 0.25
        rank 5 -> 0.20

    No relevant result:

        0.00
    """

    unique_ids = _unique_document_ids(
        retrieved_ids
    )

    for rank, doc_id in enumerate(
        unique_ids,
        start=1,
    ):

        if doc_id in relevant_ids:

            score = (
                1.0 / rank
            )

            return max(
                0.0,
                min(
                    1.0,
                    float(score),
                ),
            )

    return 0.0


# ============================================================
# LAYER 7 - RETRIEVAL EVALUATION
# ============================================================

def evaluate_retrieval(
    golden_queries: list,
) -> dict:
    """
    Evaluate the actual hybrid retrieval pipeline.

    Metrics:

        Precision@5
        Recall@5
        MRR

    Evaluation is performed at the document level.
    """

    # --------------------------------------------------------
    # Validate supplied dataset
    # --------------------------------------------------------

    validate_golden_queries(
        golden_queries
    )

    # --------------------------------------------------------
    # Import here to avoid circular imports
    # --------------------------------------------------------

    from src.query_pipeline import (
        hybrid_search,
    )

    precision_scores = []

    recall_scores = []

    reciprocal_ranks = []

    # --------------------------------------------------------
    # Evaluate each query
    # --------------------------------------------------------

    for item in golden_queries:

        query = str(
            item["query"]
        )

        relevant_ids = set(
            item["relevant_doc_ids"]
        )

        try:

            response = hybrid_search(
                user_query=query,
                user_role="employee",
                top_k=5,
            )

        except Exception as exc:

            print(
                f"Warning: evaluation failed for "
                f"query '{query}': {exc}"
            )

            response = {
                "results": []
            }

        if not isinstance(
            response,
            dict,
        ):
            response = {
                "results": []
            }

        results = response.get(
            "results",
            []
        )

        if not isinstance(
            results,
            list,
        ):
            results = []

        # ----------------------------------------------------
        # Extract document IDs
        # ----------------------------------------------------

        retrieved_ids = []

        for result in results:

            if not isinstance(
                result,
                dict,
            ):
                continue

            doc_id = result.get(
                "doc_id"
            )

            if isinstance(
                doc_id,
                str,
            ):

                retrieved_ids.append(
                    doc_id
                )

        # ----------------------------------------------------
        # Remove duplicate documents
        # ----------------------------------------------------

        unique_retrieved_ids = (
            _unique_document_ids(
                retrieved_ids
            )
        )

        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        precision = precision_at_k(
            unique_retrieved_ids,
            relevant_ids,
            k=5,
        )

        recall = recall_at_k(
            unique_retrieved_ids,
            relevant_ids,
            k=5,
        )

        rr = reciprocal_rank(
            unique_retrieved_ids,
            relevant_ids,
        )

        precision_scores.append(
            precision
        )

        recall_scores.append(
            recall
        )

        reciprocal_ranks.append(
            rr
        )

    # --------------------------------------------------------
    # Number of queries
    # --------------------------------------------------------

    num_queries = len(
        precision_scores
    )

    if num_queries == 0:

        return {
            "num_queries": 0,
            "precision_at_5": 0.0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
        }

    # --------------------------------------------------------
    # Average metrics
    # --------------------------------------------------------

    precision_average = (
        sum(precision_scores)
        / num_queries
    )

    recall_average = (
        sum(recall_scores)
        / num_queries
    )

    mrr_average = (
        sum(reciprocal_ranks)
        / num_queries
    )

    # --------------------------------------------------------
    # Clamp values to [0, 1]
    # --------------------------------------------------------

    precision_average = max(
        0.0,
        min(
            1.0,
            precision_average,
        ),
    )

    recall_average = max(
        0.0,
        min(
            1.0,
            recall_average,
        ),
    )

    mrr_average = max(
        0.0,
        min(
            1.0,
            mrr_average,
        ),
    )

    return {
        "num_queries": int(
            num_queries
        ),

        "precision_at_5": float(
            precision_average
        ),

        "recall_at_5": float(
            recall_average
        ),

        "mrr": float(
            mrr_average
        ),
    }


# ============================================================
# LAYER 7 - RAGAS EVALUATION
# ============================================================

def evaluate_ragas(
    golden_queries: list,
) -> dict:
    """
    Evaluate generation quality.

    The assignment allows two implementations:

        Option 1:
            Real RAGAS evaluation.

        Option 2:
            Offline deterministic stub.

    This implementation uses Option 2.

    Metrics:

        Faithfulness
        AnswerRelevancy
        ContextPrecision
        ContextRecall

    Returns:

        {
            "faithfulness": float,
            "answer_relevancy": float,
            "context_precision": float,
            "context_recall": float,
            "lowest_metric": str,
            "root_cause_layer": str
        }

    IMPORTANT:

        lowest_metric uses the exact display names required
        by the assignment:

            Faithfulness
            AnswerRelevancy
            ContextPrecision
            ContextRecall
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not isinstance(
        golden_queries,
        list,
    ):
        golden_queries = []

    # --------------------------------------------------------
    # Empty dataset
    # --------------------------------------------------------

    if not golden_queries:

        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,

            # Exact assignment spelling
            "lowest_metric": "Faithfulness",

            "root_cause_layer": "Layer 7",
        }

    # ========================================================
    # OFFLINE DETERMINISTIC RAGAS VALUES
    # ========================================================
    #
    # These are assignment-approved offline values.
    #
    # They are NOT real measurements from the RAGAS library.
    # ========================================================

    faithfulness = 0.90

    answer_relevancy = 0.88

    context_precision = 0.84

    context_recall = 0.86

    # --------------------------------------------------------
    # Internal metric values
    #
    # Lowercase keys make calculation convenient.
    # --------------------------------------------------------

    metrics = {

        "Faithfulness": faithfulness,

        "AnswerRelevancy": answer_relevancy,

        "ContextPrecision": context_precision,

        "ContextRecall": context_recall,
    }

    # --------------------------------------------------------
    # Find lowest metric
    #
    # This returns the EXACT required display name.
    # --------------------------------------------------------

    lowest_metric = min(
        metrics,
        key=metrics.get,
    )

    # --------------------------------------------------------
    # Root cause mapping
    # --------------------------------------------------------

    root_cause_mapping = {

        "Faithfulness": "Layer 7",

        "AnswerRelevancy": "Layer 5",

        "ContextPrecision": "Layer 7",

        "ContextRecall": "Layer 2",
    }

    root_cause_layer = (
        root_cause_mapping[
            lowest_metric
        ]
    )

    # --------------------------------------------------------
    # Return required structure
    # --------------------------------------------------------

    return {

        "faithfulness": float(
            max(
                0.0,
                min(
                    1.0,
                    faithfulness,
                ),
            )
        ),

        "answer_relevancy": float(
            max(
                0.0,
                min(
                    1.0,
                    answer_relevancy,
                ),
            )
        ),

        "context_precision": float(
            max(
                0.0,
                min(
                    1.0,
                    context_precision,
                ),
            )
        ),

        "context_recall": float(
            max(
                0.0,
                min(
                    1.0,
                    context_recall,
                ),
            )
        ),

        # EXACT required name:
        "lowest_metric": str(
            lowest_metric
        ),

        "root_cause_layer": str(
            root_cause_layer
        ),
    }
