"""The two Qdrant collections, and the sparse (exact-word) vectors stored in one of them."""

import hashlib
import re

from qdrant_client import QdrantClient, models

DENSE_VECTOR_SIZE = 768  # gemini-embedding-001 vectors
SPARSE_VECTOR_NAME = "text"
TOKEN = re.compile(r"[A-Za-z0-9_./:-]+")  # keeps IDs like db-primary:5432 or /usr/src/app in one piece


def ensure_qdrant_collections(client: QdrantClient, dense_collection: str, sparse_collection: str) -> None:
    """Create the collections if they don't exist: dense = meaning vectors (cosine), sparse = word vectors."""
    existing = {collection.name for collection in client.get_collections().collections}
    if dense_collection not in existing:
        client.create_collection(
            collection_name=dense_collection,
            vectors_config=models.VectorParams(size=DENSE_VECTOR_SIZE, distance=models.Distance.COSINE),
        )
    if sparse_collection not in existing:
        client.create_collection(
            collection_name=sparse_collection,
            vectors_config={},
            sparse_vectors_config={SPARSE_VECTOR_NAME: models.SparseVectorParams(index=models.SparseIndexParams())},
        )


def sparse_vector(text: str) -> models.SparseVector:
    """Word counts as a sparse vector: index = hash of the word, value = count / count of the most common word.

    The same word always hashes to the same index, so a question containing ERR_CONN_RESET
    matches chunks containing it, with no vocabulary to maintain.
    """
    counts: dict[str, int] = {}
    for token in TOKEN.findall(text.lower()):
        counts[token] = counts.get(token, 0) + 1
    if not counts:
        return models.SparseVector(indices=[], values=[])

    top = max(counts.values())
    weights = {int.from_bytes(hashlib.sha1(token.encode()).digest()[:4], "big"): count / top
               for token, count in counts.items()}
    indices = sorted(weights)
    return models.SparseVector(indices=indices, values=[weights[i] for i in indices])
