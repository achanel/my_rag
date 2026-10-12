"""Offline tests for chapter 7 (knowledge-graph RAG over Wikipedia).

Ollama, the embeddings and the network are replaced by fakes: the graph is built
with a typed ``MockLLM`` that returns fixed triplets and an in-memory
``MockEmbedding``, so the whole LlamaIndex pipeline — triplets, persistence,
query engine, re-ranking — runs without a server. The chapter's own helpers
(URL parsing, the Wikipedia walk, the Tree_2_Graph drawing, the metrics) are
covered directly.
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence
from unittest import mock

from my_rag.cli import build_parser, main
from my_rag.collection import Article
from my_rag.config import Settings
from my_rag.graph import (
    DEFAULT_GRAPH_LIMIT,
    DEFAULT_MARKETING_OUTPUT,
    GraphAnswer,
    GraphIndex,
    GraphRAG,
    GraphSource,
    GraphStats,
    RankedSource,
    Ranking,
    TOY_FRIENDS,
    TOY_PAIRS,
    build_networkx_graph,
    comparison_metrics,
    cosine_metrics,
    cosine_scores,
    crawl_summaries,
    draw_networkx_graph,
    graph_triplets,
    judge,
    parse_urls,
    rerank,
    safe_name,
    target_html_path,
    write_report,
)

TRIPLETS = (
    ("Marketing", "is", "activity"),
    ("Marketing", "includes", "advertising"),
    ("Advertising", "is", "communication"),
)

CORPUS = "# Marketing\n\nMarketing is an activity marketing teams run.\n"
URLS_FILE = "\n".join(
    [
        "https://en.wikipedia.org/wiki/Marketing",
        "https://en.wikipedia.org/wiki/Advertising",
        "https://en.wikipedia.org/wiki/Marketing",  # duplicate, dropped
        "# a comment line",
        "",
        "https://example.com/not-a-wiki-page",
    ]
)


def _settings(**overrides) -> Settings:
    return Settings(
        provider="ollama",
        model="qwen2.5:3b",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        embedding_model="all-minilm",
        embedding_dimension=4,
        **overrides,
    )


class FakeEmbedding:
    """Word-count vectors keyed to a small vocabulary (deterministic, offline)."""

    vocabulary = ("marketing", "advertising", "communication", "activity")

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        words = text.lower().split()
        return [float(words.count(term)) + 0.1 for term in self.vocabulary]


def _triplet_llm():
    """A LlamaIndex LLM that answers the triplet-extraction prompt deterministically."""
    from llama_index.core.llms.mock import MockLLM

    class TripletLLM(MockLLM):
        def predict(self, prompt, **kwargs) -> str:
            text = str(prompt)
            if "knowledge triplets" in text.lower():
                return "\n".join(f"({subject}, {relation}, {obj})" for subject, relation, obj in TRIPLETS)
            return "Marketing satisfies customer needs."

    return TripletLLM()


class FakeFetcher:
    """Records fetched URLs and returns canned articles (incl. one failure)."""

    def __init__(self, fail_first: bool = False) -> None:
        self.urls: list[str] = []
        self.fail_first = fail_first

    def __call__(self, url: str) -> Article:
        self.urls.append(url)
        if self.fail_first and len(self.urls) == 1:
            raise ValueError("boom")
        return Article(url=url, title=url.rsplit("/", 1)[-1], text=f"Text of {url} about marketing.")


class FakeSession:
    """Minimal ``requests``-style session replaying one URL list."""

    def __init__(self, text: str) -> None:
        self.text = text

    def get(self, url, **kwargs):
        return SimpleNamespace(text=self.text, raise_for_status=lambda: None)


# --------------------------------------------------------------------------- #
# Corpus preparation
# --------------------------------------------------------------------------- #
class UrlTests(unittest.TestCase):
    def test_parse_urls_keeps_wikipedia_only_and_dedupes(self):
        self.assertEqual(
            parse_urls(URLS_FILE),
            [
                "https://en.wikipedia.org/wiki/Marketing",
                "https://en.wikipedia.org/wiki/Advertising",
            ],
        )

    def test_crawl_walks_the_root_links(self):
        root = {
            "query": {
                "pages": {
                    "1": {
                        "title": "Marketing",
                        "fullurl": "https://en.wikipedia.org/wiki/Marketing",
                        "extract": "Marketing is the act of satisfying customers.",
                        "links": [
                            {"title": "Advertising", "ns": 0},
                            {"title": "Category:Marketing", "ns": 14},
                        ],
                    }
                }
            }
        }
        child = {
            "query": {
                "pages": {
                    "2": {
                        "title": "Advertising",
                        "fullurl": "https://en.wikipedia.org/wiki/Advertising",
                        "extract": "Advertising is communication.",
                    }
                }
            }
        }

        def fake_get(url, params=None, timeout=None):
            payload = root if params["titles"] == "Marketing" else child
            return SimpleNamespace(json=lambda: payload, raise_for_status=lambda: None)

        with mock.patch("my_rag.graph.requests.get", side_effect=fake_get):
            articles = crawl_summaries("Marketing", limit=5)
        self.assertEqual([article.title for article in articles], ["Marketing", "Advertising"])

    def test_crawl_rejects_a_bad_limit(self):
        with self.assertRaises(ValueError):
            crawl_summaries("Marketing", 0)


class CollectTests(unittest.TestCase):
    def _collect(self, limit=2, fail_first=False):
        from my_rag.graph import collect_marketing

        fetcher = FakeFetcher(fail_first=fail_first)
        with tempfile.TemporaryDirectory() as tmp:
            urls_file = Path(tmp) / "urls.txt"
            urls_file.write_text(URLS_FILE, encoding="utf-8")
            with mock.patch("my_rag.graph.collect_urls", return_value=parse_urls(URLS_FILE)):
                path, collected, failed = collect_marketing(
                    Path(tmp) / "marketing.md", limit=limit, urls_file=urls_file, fetcher=fetcher
                )
            return path.read_text(encoding="utf-8"), collected, failed

    def test_writes_one_heading_per_page(self):
        text, collected, failed = self._collect()
        self.assertEqual(len(collected), 2)
        self.assertEqual(failed, [])
        self.assertEqual(text.count("# "), 2)
        self.assertIn("Marketing", text)

    def test_a_failed_page_does_not_discard_the_run(self):
        text, collected, failed = self._collect(fail_first=True)
        self.assertEqual(len(collected), 1)
        self.assertEqual(len(failed), 1)
        self.assertIn("boom", failed[0][1])
        self.assertEqual(text.count("# "), 1)


# --------------------------------------------------------------------------- #
# Tree_2_Graph
# --------------------------------------------------------------------------- #
class TreeTests(unittest.TestCase):
    def test_builds_the_tree_and_finds_the_root(self):
        graph, root = build_networkx_graph(TOY_PAIRS)
        self.assertEqual(root, "a")
        # a b e m p z q — the book's pairs reach seven distinct nodes.
        self.assertEqual(graph.number_of_nodes(), 7)
        self.assertEqual(graph.number_of_edges(), len(TOY_PAIRS))

    def test_several_roots_report_no_single_root(self):
        _, root = build_networkx_graph([("a", "b"), ("c", "d")])
        self.assertIsNone(root)

    def test_draw_writes_an_svg_with_the_layout_in_the_title(self):
        graph, _ = build_networkx_graph(TOY_PAIRS)
        with tempfile.TemporaryDirectory() as tmp:
            path = draw_networkx_graph(
                graph, output=Path(tmp) / "tree.svg", layout="shell", friends=TOY_FRIENDS
            )
            self.assertTrue(path.is_file())
            self.assertIn("svg", path.read_text(encoding="utf-8")[:400].lower())

    def test_unknown_layout_is_rejected(self):
        graph, _ = build_networkx_graph(TOY_PAIRS)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                draw_networkx_graph(graph, output=Path(tmp) / "t.svg", layout="hexagon")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
class MetricTests(unittest.TestCase):
    def test_judge_maps_similarity_to_its_rating(self):
        self.assertEqual([judge(score) for score in (0.95, 0.75, 0.65, 0.55, 0.1)], [5, 4, 3, 2, 1])

    def test_cosine_metrics_match_numpy(self):
        metrics = cosine_metrics([0.9, 0.8, 0.7, 0.6, 0.5])
        self.assertAlmostEqual(metrics.mean, 0.7, places=6)
        self.assertAlmostEqual(metrics.median, 0.7, places=6)
        self.assertAlmostEqual(metrics.range, 0.4, places=6)
        self.assertAlmostEqual(metrics.iqr, 0.2, places=6)

    def test_cosine_metrics_reject_an_empty_sample(self):
        with self.assertRaises(ValueError):
            cosine_metrics([])

    def test_comparison_uses_recorded_human_scores_when_given(self):
        comparison = comparison_metrics([0.9, 0.8], human=[0.75, 0.5])
        self.assertEqual(comparison.source, "recorded")
        self.assertAlmostEqual(comparison.mean_cosine, 0.85, places=6)
        self.assertAlmostEqual(comparison.mean_human, 0.625, places=6)

    def test_comparison_falls_back_to_the_labelled_rubric(self):
        comparison = comparison_metrics([0.9, 0.5])
        self.assertEqual(comparison.source, "rubric")
        self.assertAlmostEqual(comparison.mean_human, (1.0 + 0.4) / 2, places=6)

    def test_comparison_checks_the_human_sample_length(self):
        with self.assertRaises(ValueError):
            comparison_metrics([0.9, 0.8], human=[0.5])


class ReRankTests(unittest.TestCase):
    def test_best_rank_and_score_come_from_the_cosine(self):
        sources = [
            GraphSource("1", 1000.0, "advertising communication"),
            GraphSource("2", 1000.0, "marketing activity"),
        ]
        ranking = rerank("marketing", sources, FakeEmbedding())
        self.assertEqual(ranking.best_rank, 2)
        self.assertGreater(ranking.best_score, 0)
        self.assertEqual([item.rank for item in ranking.scores], [1, 2])

    def test_no_sources_gives_no_ranking(self):
        ranking = rerank("marketing", [], FakeEmbedding())
        self.assertIsNone(ranking.best_rank)
        self.assertEqual(ranking.best_score, 0.0)
        self.assertEqual(ranking.scores, [])

    def test_cosine_scores_have_one_value_per_text(self):
        scores = cosine_scores("marketing", ["marketing", "advertising"], FakeEmbedding())
        self.assertEqual(len(scores), 2)
        self.assertGreater(scores[0], scores[1])

    def test_stats_density_and_string(self):
        stats = GraphStats(triplets=5, nodes=4, edges=3)
        self.assertAlmostEqual(stats.density, 0.25, places=6)
        self.assertIn("5 triplets", str(stats))

    def test_slug_and_html_path(self):
        self.assertEqual(safe_name("What are the 4Ps?"), "what_are_the_4ps")
        self.assertEqual(str(target_html_path("G", "d")), "d/G.html")


# --------------------------------------------------------------------------- #
# The index (real LlamaIndex, fake LLM + embeddings)
# --------------------------------------------------------------------------- #
class GraphIndexTests(unittest.TestCase):
    def _index(self, tmp: Path, **overrides) -> GraphIndex:
        from llama_index.core import Document

        options = dict(
            store_path=tmp,
            llm=_triplet_llm(),
            embed_model=self._mock_embedding(),
            embedding_function=FakeEmbedding(),
        )
        options.update(overrides)
        index = GraphIndex(_settings(), **options)
        index.build([Document(text=CORPUS)])
        return index

    @staticmethod
    def _mock_embedding():
        from llama_index.core.embeddings import MockEmbedding

        return MockEmbedding(embed_dim=4)

    def test_build_extracts_triplets_and_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = self._index(Path(tmp))
            self.assertEqual(
                index.triplets,
                [(s.capitalize(), r.capitalize(), o.capitalize()) for s, r, o in TRIPLETS],
            )
            self.assertTrue(index.exists)
            self.assertEqual(index.stats.nodes, 4)
            self.assertEqual(index.stats.edges, 3)

    def test_graph_triplets_reads_the_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = self._index(Path(tmp))
            self.assertEqual(
                sorted(graph_triplets(index.index)),
                sorted((s.capitalize(), r.capitalize(), o.capitalize()) for s, r, o in TRIPLETS),
            )

    def test_reopened_index_keeps_its_triplets(self):
        from llama_index.core import Settings as LlamaIndexSettings

        with tempfile.TemporaryDirectory() as tmp:
            self._index(Path(tmp))
            reopened = GraphIndex(
                _settings(),
                store_path=tmp,
                llm=_triplet_llm(),
                embed_model=self._mock_embedding(),
                embedding_function=FakeEmbedding(),
            )
            # reload() installs the models into the global settings; restore them after.
            saved = (LlamaIndexSettings.llm, LlamaIndexSettings.embed_model)
            try:
                self.assertEqual(reopened.stats.triplets, len(TRIPLETS))
            finally:
                LlamaIndexSettings.llm, LlamaIndexSettings.embed_model = saved

    def test_query_answers_and_reranks(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = self._index(Path(tmp))
            with mock.patch(
                "my_rag.graph.GraphIndex._resolve_embedding_function", return_value=FakeEmbedding()
            ):
                answer = index.query("What is marketing?", top_k=2)
        self.assertIsInstance(answer, GraphAnswer)
        self.assertTrue(answer.response)
        self.assertEqual(answer.triplets, len(TRIPLETS))
        self.assertIsNotNone(answer.similarity)
        self.assertIsNotNone(answer.ranking.best_rank)

    def test_missing_store_fails_clearly(self):
        index = GraphIndex(_settings(), store_path="/tmp/definitely-not-a-graph-store")
        with self.assertRaises(FileNotFoundError):
            index.load()

    def test_visualize_writes_a_pyvis_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = self._index(Path(tmp) / "store")
            path = index.visualize(Path(tmp) / "graph.html", max_nodes=10)
            self.assertTrue(path.is_file())
            self.assertIn("vis-network", path.read_text(encoding="utf-8").lower())

    def test_evaluate_runs_the_questions_and_reports_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = GraphIndex(
                _settings(),
                store_path=Path(tmp),
                llm=_triplet_llm(),
                embed_model=self._mock_embedding(),
                embedding_function=FakeEmbedding(),
            )
            from llama_index.core import Document

            index.build([Document(text=CORPUS.replace("Marketing", "Marketing advertising communication"))])
            answers, metrics, comparison = GraphRAG(index).evaluate(["What is marketing?"], limit=1)
        self.assertEqual(len(answers), 1)
        self.assertGreaterEqual(metrics.mean, 0.0)
        self.assertEqual(comparison.samples, 1)
        self.assertEqual(comparison.source, "rubric")


class ReportTests(unittest.TestCase):
    def test_write_report_serialises_answers_and_metrics(self):
        answer = GraphAnswer(
            question="What is marketing?",
            response="Marketing satisfies needs.",
            sources=[GraphSource("1", 1000.0, "Marketing")],
            ranking=Ranking(1, 0.5, "Marketing", [RankedSource(1, 0.5, "Marketing")]),
            triplets=3,
            elapsed=1.5,
        )
        metrics = cosine_metrics([0.5])
        comparison = comparison_metrics([0.5])
        with tempfile.TemporaryDirectory() as tmp:
            path = write_report(
                Path(tmp) / "metrics.json", answers=[answer], metrics=metrics, comparison=comparison
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["answers"][0]["question"], "What is marketing?")
        self.assertEqual(payload["comparison"]["source"], "rubric")
        self.assertIn("Mean", payload["metrics"])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
class CliTests(unittest.TestCase):
    def test_chapter_7_flags_parse(self):
        args = build_parser().parse_args(
            ["--rag", "graph", "--show-context", "--metrics", "--top-k", "5", "--graph-store", "/tmp/g"]
        )
        self.assertEqual(
            (args.rag, args.show_context, args.metrics, args.top_k, args.graph_store),
            ("graph", True, True, 5, "/tmp/g"),
        )

    def test_marketing_corpus_and_pairs_parse(self):
        args = build_parser().parse_args(["--collect", "--corpus", "marketing", "--limit", "20"])
        self.assertEqual((args.corpus, args.limit), ("marketing", 20))
        pairs = build_parser().parse_args(["--visualize", "--pairs", "--layout", "spiral"])
        self.assertEqual((pairs.visualize, pairs.pairs, pairs.layout), (True, True, "spiral"))

    def test_graph_query_runs_offline_with_fakes(self):
        answer = GraphAnswer(
            question="What is marketing?",
            response="Marketing satisfies needs.",
            sources=[GraphSource("1", 1000.0, "Marketing")],
            ranking=Ranking(1, 0.5, "Marketing", [RankedSource(1, 0.5, "Marketing")]),
            triplets=3,
            elapsed=1.0,
        )
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch("my_rag.graph.GraphIndex.exists", True), mock.patch(
            "my_rag.graph.GraphIndex.query", return_value=answer
        ):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                status = main(["--rag", "graph", "--show-context", "--graph-store", "/tmp/g"])
        self.assertEqual(status, 0)
        out = stdout.getvalue()
        self.assertIn("Marketing satisfies needs.", out)
        self.assertIn("Triplets: 3", out)
        self.assertIn("Re-ranking", out)

    def test_pairs_visualize_runs_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                status = main(["--pairs", "--layout", "spring", "--graph-html", str(Path(tmp) / "t.svg")])
            self.assertEqual(status, 0)
            self.assertTrue((Path(tmp) / "t.svg").is_file())
        self.assertIn("root: a", stdout.getvalue())

    def test_metrics_flag_runs_offline_with_fakes(self):
        answer = GraphAnswer(
            question="What is marketing?",
            response="Marketing satisfies needs.",
            sources=[],
            ranking=Ranking(None, 0.0, ""),
            triplets=3,
            elapsed=1.0,
        )
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch("my_rag.graph.GraphIndex.exists", True), mock.patch(
            "my_rag.graph.GraphIndex.query", return_value=answer
        ):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                status = main(
                    ["--rag", "graph", "--metrics", "--report", "/tmp/graph-report.json", "--graph-store", "/tmp/g"]
                )
        self.assertEqual(status, 0)
        self.assertIn("Metrics over", stdout.getvalue())
        self.assertIn("Mean:", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
