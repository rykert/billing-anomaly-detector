import uuid

import numpy as np
import pytest

from billing_anomaly_detector.application.services.cosine_scorer import (
    compute_centroid,
    cosine_distance,
    find_neighbors,
    score_embeddings,
)
from billing_anomaly_detector.domain.value_objects import AnomalyScore


class TestCosineDistance:
    def test_identical_vectors_are_zero_distance(self) -> None:
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_distance(a, a) == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_vectors_are_distance_one(self) -> None:
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert cosine_distance(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_opposite_vectors_are_distance_two(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert cosine_distance(a, b) == pytest.approx(2.0, abs=1e-6)

    def test_zero_vector_returns_zero(self) -> None:
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_distance(a, b) == 0.0


class TestComputeCentroid:
    def test_centroid_of_two_vectors(self) -> None:
        embeddings = [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        centroid = compute_centroid(embeddings)
        assert centroid[0] == pytest.approx(0.5, abs=1e-6)
        assert centroid[1] == pytest.approx(0.5, abs=1e-6)

    def test_centroid_of_identical_vectors_equals_vector(self) -> None:
        embeddings = [[1.0, 2.0, 3.0]] * 5
        centroid = compute_centroid(embeddings)
        assert centroid.tolist() == pytest.approx([1.0, 2.0, 3.0], abs=1e-6)

    def test_centroid_shape(self) -> None:
        embeddings = [[0.1] * 10] * 3
        centroid = compute_centroid(embeddings)
        assert centroid.shape == (10,)


class TestScoreEmbeddings:
    def test_returns_sorted_descending(self) -> None:
        id1 = uuid.uuid4()
        id2 = uuid.uuid4()
        embeddings = [
            (id1, [1.0, 0.0]),   # orthogonal to centroid direction
            (id2, [0.9, 0.1]),   # closer to centroid
        ]
        centroid = np.array([0.5, 0.5], dtype=np.float32)
        results = score_embeddings(embeddings, centroid)

        assert results[0][1].value >= results[1][1].value

    def test_scores_are_valid_anomaly_scores(self) -> None:
        invoice_id = uuid.uuid4()
        embeddings = [(invoice_id, [1.0, 0.0])]
        centroid = np.array([1.0, 0.0], dtype=np.float32)
        results = score_embeddings(embeddings, centroid)

        score = results[0][1]
        assert isinstance(score, AnomalyScore)
        assert 0.0 <= score.value <= 1.0


class TestFindNeighbors:
    def test_excludes_target(self) -> None:
        target_id = uuid.uuid4()
        other_id = uuid.uuid4()
        embeddings = [
            (target_id, [1.0, 0.0]),
            (other_id, [0.9, 0.1]),
        ]
        neighbors = find_neighbors(target_id, [1.0, 0.0], embeddings, limit=3)
        assert target_id not in neighbors

    def test_returns_closest_first(self) -> None:
        target_id = uuid.uuid4()
        close_id = uuid.uuid4()
        far_id = uuid.uuid4()
        embeddings = [
            (target_id, [1.0, 0.0, 0.0]),
            (close_id,  [0.99, 0.01, 0.0]),  # very close
            (far_id,    [0.0, 1.0, 0.0]),    # orthogonal = far
        ]
        neighbors = find_neighbors(target_id, [1.0, 0.0, 0.0], embeddings, limit=2)
        assert close_id == neighbors[0]
        assert far_id == neighbors[1]

    def test_respects_limit(self) -> None:
        target_id = uuid.uuid4()
        embeddings = [(target_id, [1.0, 0.0])] + [
            (uuid.uuid4(), [0.9, 0.1]) for _ in range(10)
        ]
        neighbors = find_neighbors(target_id, [1.0, 0.0], embeddings, limit=3)
        assert len(neighbors) == 3
