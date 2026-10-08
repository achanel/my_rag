"""Offline tests for chapter 5 (adaptive RAG with human feedback).

The network fetch, Ollama and the rating input are replaced by fakes, so these
run without network or models.
"""
from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from my_rag.adaptive import (
    ADAPTIVE_SOURCES,
    HUMAN_FEEDBACK,
    AdaptiveRAG,
    NoMatchError,
    RatingState,
    evaluate,
    first_words,
    load_expert_feedback,
    match_keyword,
    retrieve,
    save_expert_feedback,
    strategy_for_ranking,
    tfidf_similarity,
)
from my_rag.cli import DEFAULT_ADAPTIVE_QUERY, build_parser, main
from my_rag.collection import Article
from my_rag.config import Settings
from my_rag.llm import LLM

PAGE = Article(
    url="https://en.wikipedia.org/wiki/Large_language_model",
    title="Large language model",
    text=" ".join(f"word{i}" for i in range(300)),
)


class FakeFetcher:
    """Records the URLs it is asked for and returns one canned article."""

    def __init__(self, article: Article = PAGE) -> None:
        self.article = article
        self.urls: list[str] = []

    def __call__(self, url: str) -> Article:
        self.urls.append(url)
        return self.article


class FakeLLM(LLM):
    def __init__(self, reply: str = "A large language model is a neural network.") -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, prompt, *, system=None, temperature=None) -> str:
        self.prompts.append(prompt)
        return self.reply


def _settings(**overrides) -> Settings:
    values = dict(
        provider="ollama",
        model="qwen2.5:3b",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        embedding_model="all-minilm",
        vision_model="gemma3:12b",
    )
    values.update(overrides)
    return Settings(**values)


class MatchKeywordTests(unittest.TestCase):
    def test_matches_case_insensitively(self):
        self.assertEqual(match_keyword("What is an LLM?"), "llm")

    def test_a_multi_word_keyword_is_matched(self):
        self.assertEqual(match_keyword("explain prompt engineering"), "prompt engineering")

    def test_no_keyword_returns_none(self):
        self.assertIsNone(match_keyword("what is the weather?"))


class StrategyTests(unittest.TestCase):
    def test_rankings_map_to_the_books_branches(self):
        self.assertEqual(strategy_for_ranking(1), "none")
        self.assertEqual(strategy_for_ranking(2), "none")
        self.assertEqual(strategy_for_ranking(3), "feedback")
        self.assertEqual(strategy_for_ranking(4), "feedback")
        self.assertEqual(strategy_for_ranking(5), "rag")

    def test_out_of_range_ranking_is_rejected(self):
        for ranking in (0, 6):
            with self.assertRaises(ValueError):
                strategy_for_ranking(ranking)


class FirstWordsTests(unittest.TestCase):
    def test_limits_to_the_first_words(self):
        self.assertEqual(first_words("a b c d e", 3), "a b c")

    def test_non_positive_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            first_words("a b c", 0)


class RetrieveTests(unittest.TestCase):
    def test_builds_the_excerpt_and_the_book_prompt(self):
        fetcher = FakeFetcher()
        result = retrieve("What is an LLM?", 5, fetcher=fetcher)
        self.assertEqual(fetcher.urls, [ADAPTIVE_SOURCES["llm"]])
        self.assertEqual(result.keyword, "llm")
        self.assertEqual(result.excerpt, "word0 word1 word2 word3 word4")
        self.assertIn(result.excerpt, result.prompt)
        self.assertTrue(result.prompt.startswith("Please summarize or elaborate"))

    def test_unmatched_keyword_returns_none_without_fetching(self):
        fetcher = FakeFetcher()
        self.assertIsNone(retrieve("hello there", fetcher=fetcher))
        self.assertEqual(fetcher.urls, [])


class SimilarityTests(unittest.TestCase):
    def test_identical_texts_score_one(self):
        self.assertAlmostEqual(tfidf_similarity("llm model text", "llm model text"), 1.0)

    def test_disjoint_texts_score_zero(self):
        self.assertEqual(tfidf_similarity("apple banana", "car truck"), 0.0)


class AdaptiveRAGTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fetcher = FakeFetcher()
        self.llm = FakeLLM()
        self.rag = AdaptiveRAG(_settings(), llm=self.llm, fetcher=self.fetcher)

    def test_low_ranking_sends_the_bare_query_and_does_not_fetch(self):
        result = self.rag.answer("What is an LLM?", ranking=1)
        self.assertEqual(result.strategy, "none")
        self.assertEqual(result.text_input, "What is an LLM?")
        self.assertIsNone(result.source)
        self.assertEqual(self.fetcher.urls, [])

    def test_middle_ranking_sends_the_human_feedback_flashcard(self):
        result = self.rag.answer("What is an LLM?", ranking=4)
        self.assertEqual(result.strategy, "feedback")
        self.assertEqual(result.text_input, HUMAN_FEEDBACK)
        self.assertEqual(self.fetcher.urls, [])

    def test_top_ranking_sends_the_retrieved_excerpt(self):
        result = self.rag.answer("What is an LLM?", ranking=5, num_words=4)
        self.assertEqual(result.strategy, "rag")
        self.assertEqual(result.text_input, "word0 word1 word2 word3")
        self.assertEqual(result.source, ADAPTIVE_SOURCES["llm"])
        self.assertEqual(self.fetcher.urls, [ADAPTIVE_SOURCES["llm"]])

    def test_top_ranking_without_a_keyword_raises(self):
        with self.assertRaises(NoMatchError):
            self.rag.answer("nothing relevant", ranking=5)

    def test_generation_uses_the_book_instruction_and_system(self):
        self.rag.answer("What is an LLM?", ranking=1)
        prompt = self.llm.prompts[0]
        self.assertTrue(prompt.startswith("Please summarize or elaborate on the following content:"))
        self.assertIn("What is an LLM?", prompt)
        self.assertEqual(self.rag.settings.model, "qwen2.5:3b")


class EvaluateTests(unittest.TestCase):
    def _result(self, response: str = "llm model text"):
        from my_rag.adaptive import AdaptiveResult

        return AdaptiveResult(
            query="q",
            ranking=5,
            strategy="rag",
            source=None,
            text_input="llm model text",
            response=response,
            elapsed=0.5,
        )

    def test_without_a_rating_only_the_similarity_is_reported(self):
        evaluation = evaluate(self._result())
        self.assertAlmostEqual(evaluation.similarity, 1.0)
        self.assertIsNone(evaluation.rating)
        self.assertIsNone(evaluation.mean_score)
        self.assertFalse(evaluation.expert_review_required)

    def test_rating_updates_the_books_loop(self):
        evaluation = evaluate(self._result(), rating=3)
        self.assertEqual(evaluation.rating, 3)
        self.assertEqual(evaluation.state.counter, 21)
        self.assertEqual(evaluation.mean_score, 3.0)

    def test_low_mean_on_a_large_panel_triggers_expert_review(self):
        state = RatingState(counter=20, score_history=3)
        self.assertTrue(state.expert_review_required(counter_threshold=10, score_threshold=4))
        state.record(5)
        self.assertFalse(state.expert_review_required(counter_threshold=10, score_threshold=4))

    def test_out_of_range_score_is_rejected(self):
        with self.assertRaises(ValueError):
            RatingState().record(9)


class ExpertFeedbackFileTests(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "expert_feedback.txt"
            self.assertIsNone(load_expert_feedback(path))
            saved = save_expert_feedback("make it shorter", path)
            self.assertEqual(saved, path)
            self.assertEqual(load_expert_feedback(path), "make it shorter")


class ConfigTests(unittest.TestCase):
    def test_adaptive_defaults_are_registered(self):
        from my_rag.config import DEFAULT_EXPERT_FEEDBACK

        self.assertEqual(DEFAULT_EXPERT_FEEDBACK, Path("data/processed/adaptive/expert_feedback.txt"))


class CliTests(unittest.TestCase):
    def test_adaptive_flags_parse(self):
        args = build_parser().parse_args(
            ["--rag", "adaptive", "--ranking", "3", "--rating", "5", "--num-words", "40"]
        )
        self.assertEqual((args.rag, args.ranking, args.rating, args.num_words), ("adaptive", 3, 5, 40))
        self.assertEqual(args.query, None)

    def test_unmatched_keyword_fails_cleanly(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            status = main(["--rag", "adaptive", "--ranking", "5", "nothing relevant"])
        self.assertEqual(status, 1)
        self.assertIn("no relevant keywords found", stderr.getvalue())

    def test_ranking_without_fetch_runs_offline_with_a_fake_model(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch("my_rag.adaptive.create_llm", return_value=FakeLLM()):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                status = main(["--rag", "adaptive", "--ranking", "3", "--rating", "3"])
        self.assertEqual(status, 0)
        out = stdout.getvalue()
        self.assertIn("A large language model is a neural network.", out)
        self.assertIn("Evaluator Score: 3", out)
        self.assertIn("Score history :  3.0", out)

    def test_default_adaptive_query_is_used(self):
        self.assertEqual(DEFAULT_ADAPTIVE_QUERY, "What is an LLM?")


if __name__ == "__main__":
    unittest.main()
