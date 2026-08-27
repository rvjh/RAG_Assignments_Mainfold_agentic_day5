"""
============================================================
LAYER 6 — ACCESS CONTROL
============================================================

RBAC-style metadata filtering.

Roles:

    employee < manager < admin

Every chunk receives:

    metadata["min_role"]

Access filtering happens BEFORE retrieval ranking.
============================================================
"""

from typing import List, Dict


# ============================================================
# ROLE HIERARCHY
# ============================================================

ROLE_LEVELS = {
    "employee": 1,
    "manager": 2,
    "admin": 3,
}


# ============================================================
# DEFAULT ACCESS RULE
# ============================================================

DEFAULT_MIN_ROLE = "employee"


# ============================================================
# VALIDATE ROLE
# ============================================================

def _validate_role(user_role: str) -> str:
    """
    Validate and normalize a user role.
    """

    if not isinstance(user_role, str):
        user_role = str(user_role)

    user_role = user_role.strip().lower()

    if user_role not in ROLE_LEVELS:
        raise ValueError(
            f"Unknown user role: {user_role}. "
            f"Expected one of: "
            f"{list(ROLE_LEVELS.keys())}"
        )

    return user_role


# ============================================================
# ENSURE ACCESS METADATA
# ============================================================

def ensure_access_metadata(
    chunks: list,
) -> list:
    """
    Ensure every chunk has:

        metadata["min_role"]

    Existing access metadata is preserved.

    If no min_role exists, the chunk defaults to
    employee-level access.

    IMPORTANT:
        This function accepts ONLY a list of chunks.

    Do NOT pass VectorStore here.
    """

    if chunks is None:
        return []

    if not isinstance(chunks, list):
        raise TypeError(
            "ensure_access_metadata() expects a list "
            "of chunks, not a VectorStore."
        )

    for chunk in chunks:

        if not isinstance(chunk, dict):
            continue

        if "metadata" not in chunk:
            chunk["metadata"] = {}

        if not isinstance(
            chunk["metadata"],
            dict,
        ):
            chunk["metadata"] = {}

        min_role = chunk["metadata"].get(
            "min_role"
        )

        if not isinstance(
            min_role,
            str,
        ):
            chunk["metadata"]["min_role"] = (
                DEFAULT_MIN_ROLE
            )

            continue

        min_role = min_role.strip().lower()

        if min_role not in ROLE_LEVELS:

            chunk["metadata"]["min_role"] = (
                DEFAULT_MIN_ROLE
            )

        else:

            chunk["metadata"]["min_role"] = (
                min_role
            )

    return chunks


# ============================================================
# ACCESS CHECK
# ============================================================

def _can_access(
    user_role: str,
    min_role: str,
) -> bool:
    """
    Determine whether a user can access a chunk.
    """

    user_level = ROLE_LEVELS.get(
        user_role,
        0,
    )

    required_level = ROLE_LEVELS.get(
        min_role,
        ROLE_LEVELS[DEFAULT_MIN_ROLE],
    )

    return user_level >= required_level


# ============================================================
# LAYER 6 — FILTER CHUNKS
# ============================================================

def filter_chunks_by_access(
    chunks: list,
    user_role: str,
) -> list:
    """
    Filter chunks using RBAC metadata.

    Args:
        chunks:
            List of chunk dictionaries.

        user_role:
            employee, manager, or admin.

    Returns:
        Only chunks the user is allowed to access.

    Example:

        employee:
            employee documents

        manager:
            employee + manager documents

        admin:
            employee + manager + admin documents

    IMPORTANT:
        This function performs PRE-FILTERING.
        It must be called before lexical ranking and
        before accepting vector-search candidates.
    """

    user_role = _validate_role(
        user_role
    )

    chunks = ensure_access_metadata(
        chunks
    )

    accessible_chunks = []

    for chunk in chunks:

        if not isinstance(chunk, dict):
            continue

        metadata = chunk.get(
            "metadata",
            {},
        )

        min_role = metadata.get(
            "min_role",
            DEFAULT_MIN_ROLE,
        )

        if _can_access(
            user_role,
            min_role,
        ):

            accessible_chunks.append(
                chunk
            )

    return accessible_chunks


# ============================================================
# CHECK SINGLE CHUNK
# ============================================================

def can_access_chunk(
    chunk: Dict,
    user_role: str,
) -> bool:
    """
    Check access to a single chunk.
    """

    user_role = _validate_role(
        user_role
    )

    if not isinstance(chunk, dict):
        return False

    metadata = chunk.get(
        "metadata",
        {},
    )

    if not isinstance(
        metadata,
        dict,
    ):
        metadata = {}

    min_role = metadata.get(
        "min_role",
        DEFAULT_MIN_ROLE,
    )

    if not isinstance(
        min_role,
        str,
    ):
        min_role = DEFAULT_MIN_ROLE

    min_role = min_role.lower()

    return _can_access(
        user_role,
        min_role,
    )


# ============================================================
# ACCESS SUMMARY
# ============================================================

def access_summary(
    chunks: list,
) -> dict:
    """
    Return number of chunks at each access level.
    """

    chunks = ensure_access_metadata(
        chunks
    )

    summary = {
        "employee": 0,
        "manager": 0,
        "admin": 0,
    }

    for chunk in chunks:

        metadata = chunk.get(
            "metadata",
            {},
        )

        min_role = metadata.get(
            "min_role",
            DEFAULT_MIN_ROLE,
        )

        if min_role in summary:
            summary[min_role] += 1

    return summary
