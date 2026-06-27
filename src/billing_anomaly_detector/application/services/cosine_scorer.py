from uuid import UUID

import numpy as np

from billing_anomaly_detector.domain.value_objects import AnomalyScore


def compute_centroid(embeddings: list[list[float]]) -> np.ndarray:
    """
    Compute the centroid (mean vector) of all embeddings.
    The centroid represents the 'average normal invoice' in embedding space.
    Invoices far from this point are anomalous.
    """
    matrix = np.array(embeddings, dtype=np.float32)
    return matrix.mean(axis=0)  # type: ignore[no-any-return]


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine distance between two vectors.
    0.0 = identical direction (very similar).
    1.0 = orthogonal (unrelated).
    Higher = more anomalous.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    similarity = float(np.dot(a, b) / (norm_a * norm_b))
    similarity = float(np.clip(similarity, -1.0, 1.0))
    return 1.0 - similarity


def score_embeddings(
    embeddings: list[tuple[UUID, list[float]]],
    centroid: np.ndarray,
) -> list[tuple[UUID, AnomalyScore]]:
    """
    Score every invoice by its cosine distance from the centroid.
    Returns results sorted descending — most anomalous first.
    Score is capped at 1.0 since text embeddings cluster well
    below the theoretical maximum of 2.0 in practice.
    """
    results = []
    for invoice_id, embedding in embeddings:
        vec = np.array(embedding, dtype=np.float32)
        dist = cosine_distance(vec, centroid)
        score = AnomalyScore(min(dist, 1.0))
        results.append((invoice_id, score))

    return sorted(results, key=lambda pair: pair[1].value, reverse=True)
