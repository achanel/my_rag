"""Offline tests for chapter 6 (scaling a RAG system with Qdrant).

Qdrant runs locally through its embedded engine, so the index round-trip is tested
against the real client with no server. The embeddings and the LLM are replaced by
fakes; no network, key or Ollama is needed.
"""
from __future__ import annotations

import contextlib
import csv
import io
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

# Qdrant's local mode keeps an SQLite connection that can warn on GC; harmless here.
warnings.filterwarnings("ignore", category=ResourceWarning)

from my_rag.cli import build_parser, main
from my_rag.config import DEFAULT_CHURN_CSV, DEFAULT_QDRANT_COLLECTION, Settings
from my_rag.llm import LLM
from my_rag.scaling import (
    DEFAULT_DSIZE,
    DEFAULT_QUERY_RECORD,
    EMAIL_QUERY_PROMPT,
    EMAIL_SYSTEM_PROMPT,
    CustomerRAG,
    build_email_prompt,
    cluster_churn,
    collect_churn,
    exited_overview,
    kmeans,
    load_churn,
    minmax_scale,
    open_qdrant_index,
    query_index,
    score_clusters,
    to_records,
    upsert_records,
)


def _row(customer, credit, age, exited, complain, salary, points=100):
    return {
        "CustomerId": customer,
        "CreditScore": credit,
        "Age": age,
        "Tenure": 2,
        "Balance": 0.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": salary,
        "Exited": exited,
        "Complain": complain,
        "Satisfaction Score": 2,
        "Card Type": "DIAMOND",
        "Point Earned": points,
    }


ROWS = [
    _row("1", 800, 25, 0, 0, 50000),
    _row("2", 810, 26, 0, 0, 55000),
    _row("3", 600, 60, 1, 1, 150000),
    _row("4", 590, 62, 1, 1, 160000),
]


class FakeEmbedding:
    """Deterministic fixed-length vectors keyed by text length; only that matters here."""

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension

    def embed_documents(self, texts):
        return [
            [float((len(text) + index) % 7) / 7 + 0.01 for index in range(self.dimension)]
            for text in texts
        ]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class FakeIndex:
    """Records upserts and replays canned matches for queries."""

    def __init__(self, matches=None, collection: str = "test-index") -> None:
        self.upserts: list[list[dict]] = []
        self.calls: list[tuple] = []
        self.matches = matches or []
        self.collection = collection
        self.closed = False

    def upsert(self, points, **kwargs):
        self.upserts.append(list(points))

    def query(self, vector, *, limit, with_payload=True):
        self.calls.append((vector, limit, with_payload))
        return self.matches[:limit]

    def close(self):
        self.closed = True


def _point(id, score, text):
    return SimpleNamespace(id=id, score=score, payload={"text": text})


class FakeLLM(LLM):
    def __init__(self, reply: str = "Dear Valued Customer, stay with us.") -> None:
        self.reply = reply
        self.calls: list[tuple] = []

    def complete(self, prompt, *, system=None, temperature=None) -> str:
        self.calls.append((prompt, system, temperature))
        return self.reply


class FakeSession:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.urls: list[str] = []

    def get(self, url, **kwargs):
        self.urls.append(url)
        return self

    def raise_for_status(self):
        return None


def _settings(**overrides) -> Settings:
    values = dict(
        provider="ollama",
        model="qwen2.5:3b",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        embedding_model="all-minilm",
        embedding_dimension=4,
        vision_model="gemma3:12b",
        qdrant_collection="test-index",
    )
    values.update(overrides)
    return Settings(**values)


class RecordsTests(unittest.TestCase):
    def test_each_row_becomes_column_value_pairs(self):
        records = to_records(ROWS)
        self.assertEqual(len(records), len(ROWS))
        self.assertTrue(records[0].startswith("CustomerId: 1 CreditScore: 800 Age: 25"))
        self.assertIn("Exited: 0", records[0])

    def test_empty_dataset_is_rejected(self):
        with self.assertRaises(ValueError):
            to_records([])


class EdaTests(unittest.TestCase):
    def test_exited_overview_counts(self):
        overview = exited_overview(ROWS)
        self.assertEqual((overview.sum_exited, overview.sum_complain), (2, 2))
        self.assertAlmostEqual(overview.complain_over_exited, 100.0)

    def test_age_overview(self):
        from my_rag.scaling import age_overview

        overview = age_overview(ROWS, age=50)
        self.assertEqual(overview.aged_and_over, 2)
        self.assertAlmostEqual(overview.aged_over_among_exited, 100.0)

    def test_salary_overview(self):
        from my_rag.scaling import salary_overview

        overview = salary_overview(ROWS, threshold=100000)
        self.assertEqual(overview.salary_over, 2)
        self.assertAlmostEqual(overview.salary_over_among_exited, 100.0)


class ScaleTests(unittest.TestCase):
    def test_minmax_scales_each_column_to_unit_range(self):
        scaled = minmax_scale(np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]]))
        self.assertTrue(np.allclose(scaled[:, 0], [0.0, 0.5, 1.0]))
        self.assertTrue(np.allclose(scaled[:, 1], [0.0, 0.5, 1.0]))

    def test_constant_column_does_not_divide_by_zero(self):
        scaled = minmax_scale(np.array([[7.0, 1.0], [7.0, 2.0]]))
        self.assertTrue(np.allclose(scaled[:, 0], [0.0, 0.0]))


class KMeansTests(unittest.TestCase):
    def test_separates_two_well_separated_blobs(self):
        blobs = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [9.9, 10.0], [10.0, 9.9], [10.0, 10.0]])
        labels, inertia = kmeans(blobs, 2, n_init=5, seed=1)
        self.assertEqual(len(set(labels)), 2)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[3], labels[4])
        self.assertNotEqual(labels[0], labels[3])
        self.assertGreater(inertia, 0)

    def test_out_of_range_cluster_count_is_rejected(self):
        with self.assertRaises(ValueError):
            kmeans(np.zeros((3, 2)), 5)

    def test_book_clustering_isolates_the_complainers(self):
        labels, summary = cluster_churn(ROWS, n_init=5, seed=1)
        self.assertEqual(sum(summary.counts.values()), len(ROWS))
        noisy = max(summary.complaints, key=summary.complaints.get)
        self.assertEqual(summary.complaints[noisy], 2)
        self.assertEqual(summary.exited[noisy], 2)
        self.assertEqual(summary.counts[noisy], 2)

    def test_reported_metrics_differ_by_k(self):
        scores = score_clusters(ROWS, range_=(2, 3), n_init=3, sample=None, seed=2)
        self.assertEqual([score.n_clusters for score in scores], [2, 3])
        for score in scores:
            self.assertGreaterEqual(score.silhouette, -1.0)
            self.assertGreaterEqual(score.davies_bouldin, 0.0)


class UpsertTests(unittest.TestCase):
    def test_duplicates_each_record_and_ids_are_sequential(self):
        index = FakeIndex()
        result = upsert_records(
            ["a", "b"], FakeEmbedding(), index, collection="idx", dsize=3
        )
        self.assertEqual((result.records, result.vectors, result.dsize, result.dimension), (2, 6, 3, 4))
        self.assertEqual(result.collection, "idx")
        points = index.upserts[0]
        self.assertEqual([point["id"] for point in points], list(range(1, 7)))
        self.assertEqual(
            [point["payload"]["text"] for point in points], ["a", "a", "a", "b", "b", "b"]
        )
        self.assertEqual(len(points[0]["vector"]), 4)

    def test_non_positive_dsize_is_rejected(self):
        with self.assertRaises(ValueError):
            upsert_records(["a"], FakeEmbedding(), FakeIndex(), dsize=0)


class QueryTests(unittest.TestCase):
    def test_query_returns_text_and_scores(self):
        index = FakeIndex(matches=[_point(3, 0.91, "record 3")])
        matches = query_index(index, FakeEmbedding(), "target", top_k=1)
        self.assertEqual(len(matches), 1)
        self.assertEqual((matches[0].id, matches[0].text), ("3", "record 3"))
        self.assertAlmostEqual(matches[0].score, 0.91)
        self.assertEqual(index.calls[0][1], 1)


class LocalQdrantTests(unittest.TestCase):
    """The real client, local engine, no server — a genuine end-to-end round-trip."""

    def test_build_query_and_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(qdrant_path=Path(tmp) / "qdrant", qdrant_collection="churn")
            index = open_qdrant_index(settings)
            try:
                upsert_records(["a", "bb"], FakeEmbedding(), index, collection="churn", dsize=2)
                self.assertEqual(index.count(), 4)
                matches = query_index(index, FakeEmbedding(), "a", top_k=1)
            finally:
                index.close()
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0].text, "a")

            reopened = open_qdrant_index(settings, create=False)
            try:
                self.assertEqual(reopened.count(), 4)
            finally:
                reopened.close()

    def test_missing_collection_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(qdrant_path=Path(tmp) / "qdrant", qdrant_collection="absent")
            with self.assertRaises(FileNotFoundError):
                open_qdrant_index(settings, create=False)


class PromptTests(unittest.TestCase):
    def test_prompt_concatenates_query_record_and_context(self):
        prompt = build_email_prompt("RECORD", "CONTEXT")
        self.assertTrue(prompt.startswith(EMAIL_QUERY_PROMPT))
        self.assertIn("RECORD", prompt)
        self.assertTrue(prompt.endswith("CONTEXT"))

    def test_missing_context_is_empty(self):
        prompt = build_email_prompt("RECORD")
        self.assertEqual(prompt, EMAIL_QUERY_PROMPT + "RECORD")


class CustomerRAGTests(unittest.TestCase):
    def test_answer_uses_the_community_manager_system_prompt(self):
        index = FakeIndex(
            matches=[
                _point(1, 0.9, "CustomerId: 1 Exited: 1"),
                _point(2, 0.8, "CustomerId: 2 Exited: 0"),
            ]
        )
        llm = FakeLLM()
        rag = CustomerRAG(_settings(), index, embedding_function=FakeEmbedding(), llm=llm, top_k=2)
        answer = rag.answer("TARGET RECORD")
        self.assertEqual(len(answer.matches), 2)
        self.assertIn("CustomerId: 1 Exited: 1", answer.prompt)
        self.assertEqual(answer.response, llm.reply)
        prompt, system, temperature = llm.calls[0]
        self.assertEqual(system, EMAIL_SYSTEM_PROMPT)
        self.assertEqual(temperature, 0.0)
        self.assertIn("TARGET RECORD", prompt)

    def test_default_record_matches_the_book(self):
        index = FakeIndex()
        rag = CustomerRAG(_settings(), index, embedding_function=FakeEmbedding(), llm=FakeLLM())
        answer = rag.answer()
        self.assertEqual(answer.record, DEFAULT_QUERY_RECORD)
        self.assertIn("Henderson", DEFAULT_QUERY_RECORD)


class CollectTests(unittest.TestCase):
    def test_download_writes_the_dataset(self):
        session = FakeSession(b"CustomerId,CreditScore\n1,619\n")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "churn" / "data1.csv"
            written = collect_churn(path, session=session)
            self.assertEqual(written, path)
            self.assertEqual(path.read_text(), "CustomerId,CreditScore\n1,619\n")
            self.assertEqual(len(session.urls), 1)

    def test_load_missing_dataset_fails_clearly(self):
        with self.assertRaises(FileNotFoundError):
            load_churn("/nonexistent/data1.csv")


class ConfigTests(unittest.TestCase):
    def test_chapter_6_defaults_are_registered(self):
        self.assertEqual(DEFAULT_CHURN_CSV, Path("data/raw/churn/data1.csv"))
        self.assertEqual(DEFAULT_QDRANT_COLLECTION, "bank-index-50000")
        self.assertEqual(DEFAULT_DSIZE, 5)

    def test_settings_read_qdrant_and_dimension(self):
        with mock.patch.dict(
            "os.environ",
            {
                "QDRANT_URL": "http://localhost:6333",
                "QDRANT_API_KEY": "qk",
                "MY_RAG_QDRANT_COLLECTION": "idx",
                "MY_RAG_QDRANT_PATH": "/tmp/qdrant-x",
                "MY_RAG_EMBEDDING_DIMENSION": "768",
            },
            clear=False,
        ):
            settings = Settings.from_env(provider="ollama")
        self.assertEqual(settings.qdrant_url, "http://localhost:6333")
        self.assertEqual(settings.qdrant_api_key, "qk")
        self.assertEqual(settings.qdrant_collection, "idx")
        self.assertEqual(settings.qdrant_path, Path("/tmp/qdrant-x"))
        self.assertEqual(settings.embedding_dimension, 768)

    def test_provider_embedding_dimensions(self):
        with mock.patch.dict(
            "os.environ", {"MY_RAG_EMBEDDING_DIMENSION": "", "MY_RAG_EMBEDDING_MODEL": ""}, clear=False
        ):
            self.assertEqual(Settings.from_env(provider="ollama").embedding_dimension, 384)
            self.assertEqual(Settings.from_env(provider="openai").embedding_dimension, 1536)


class CliTests(unittest.TestCase):
    def test_chapter_6_flags_parse(self):
        args = build_parser().parse_args(
            ["--rag", "qdrant", "--dsize", "3", "--collection", "mine", "--top-k", "1"]
        )
        self.assertEqual((args.rag, args.dsize, args.collection, args.top_k), ("qdrant", 3, "mine", 1))

    def test_churn_collect_option_parses(self):
        args = build_parser().parse_args(["--collect", "--corpus", "churn"])
        self.assertEqual(args.corpus, "churn")

    def test_qdrant_query_runs_offline_with_fakes(self):
        index = FakeIndex(matches=[_point(7, 0.88, "CustomerId: 7 Exited: 1")])
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch("my_rag.scaling.open_qdrant_index", return_value=index), mock.patch(
            "my_rag.scaling.create_embedding_function", return_value=FakeEmbedding()
        ), mock.patch("my_rag.scaling.create_llm", return_value=FakeLLM("Keep your DIAMOND card.")):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                status = main(["--rag", "qdrant", "--show-context", "--top-k", "1", "--collection", "idx"])
        self.assertEqual(status, 0)
        out = stdout.getvalue()
        self.assertIn("Keep your DIAMOND card.", out)
        self.assertIn("CustomerId: 7 Exited: 1", out)
        self.assertIn("Matches: 1", out)
        self.assertTrue(index.closed)

    def test_build_index_runs_offline_with_fakes(self):
        index = FakeIndex()
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "data1.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(ROWS[0]))
                writer.writeheader()
                writer.writerows(ROWS)
            stderr = io.StringIO()
            with mock.patch("my_rag.scaling.open_qdrant_index", return_value=index), mock.patch(
                "my_rag.cli.create_embedding_function", return_value=FakeEmbedding()
            ):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                    status = main(["--build-index", "--output", str(csv_path), "--dsize", "2"])
        self.assertEqual(status, 0)
        self.assertIn("upserted 8 vectors", stderr.getvalue())
        self.assertTrue(index.closed)

    def test_churn_report_runs_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "data1.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(ROWS[0]))
                writer.writeheader()
                writer.writerows(ROWS)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                status = main(["--churn-report", "--output", str(csv_path)])
        self.assertEqual(status, 0)
        out = stdout.getvalue()
        self.assertIn("Loaded 4 customers", out)
        self.assertIn("KMeans k=2 classes:", out)


if __name__ == "__main__":
    unittest.main()
