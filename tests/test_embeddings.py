"""Offline tests for the chapter 4 similarity helpers (no Ollama needed)."""
from __future__ import annotations

import unittest
from typing import Sequence

from my_rag.embeddings import EmbeddingIndex, cosine_similarity

VOCABULARY = ("truck", "car", "pedestrian", "road")


def _bag_of_words(text: str) -> list[float]:
    words = text.lower().split()
    return [float(words.count(term)) for term in VOCABULARY]


class FakeEmbeddings:
    """Deterministic stand-in for the Ollama embedding function (counts of VOCABULARY words)."""

    def __init__(self) -> None:
        self.calls: list[Sequence[str]] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [_bag_of_words(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class CosineSimilarityTests(unittest.TestCase):
    def test_identical_vectors_score_one(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 2.0], [1.0, 2.0]), 1.0)

    def test_orthogonal_vectors_score_zero(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_opposite_vectors_score_minus_one(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 1.0], [-1.0, -1.0]), -1.0)

    def test_zero_vector_scores_zero_instead_of_dividing_by_zero(self):
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 2.0]), 0.0)


class EmbeddingIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            "a car on the road",
            "a truck on the road",
            "a pedestrian crossing",
        ]
        self.embeddings = FakeEmbeddings()
        self.index = EmbeddingIndex(self.records, self.embeddings, top_k=1)

    def test_records_are_embedded_once_at_construction(self):
        self.assertEqual(self.embeddings.calls, [self.records])
        self.index.search("truck")
        self.index.retrieve("car")
        self.assertEqual(len(self.embeddings.calls), 3)  # only the two queries were added

    def test_search_ranks_the_most_similar_record_first(self):
        (position, score), = self.index.search("how do drones identify a truck")
        self.assertEqual(position, 1)
        self.assertGreater(score, 0.0)

    def test_search_returns_top_k_in_descending_score_order(self):
        hits = self.index.search("truck road", top_k=3)
        self.assertEqual([position for position, _ in hits], [1, 0, 2])
        scores = [score for _, score in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_retrieve_defaults_to_the_constructor_top_k(self):
        self.assertEqual(self.index.retrieve("car"), ["a car on the road"])
        self.assertEqual(len(self.index.retrieve("car", top_k=2)), 2)

    def test_empty_index_is_rejected(self):
        with self.assertRaises(ValueError):
            EmbeddingIndex([], FakeEmbeddings())


if __name__ == "__main__":
    unittest.main()
