"""Offline tests for chapter 8 (dynamic RAG with Chroma and a Hugging Face Llama).

Chroma runs in memory, so the index round-trip is tested against the **real**
client with a fake embedding; the SciQ network fetch is replayed by a fake
session, and the LLM is a fake. No server, key, model download or network needed.
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

warnings.filterwarnings("ignore", category=ResourceWarning)

from my_rag.cli import build_parser, main
from my_rag.config import (
    DEFAULT_CHROMA_COLLECTION,
    DEFAULT_SCIENCE_LIMIT,
    DEFAULT_SCIENCE_QA,
    PROVIDER_DEFAULTS,
    Settings,
    load_hf_token,
)
from my_rag.dynamic import (
    DEFAULT_QUESTION,
    DEFAULT_TOP_K,
    DROP_COLUMNS,
    GENERATION_TEMPERATURE,
    SIMILARITY_THRESHOLD,
    SUMMARY_INSTRUCTION,
    ChromaIndex,
    DynamicRAG,
    QAMatch,
    build_summary_prompt,
    collect_science_qa,
    fetch_science_qa,
    open_science_collection,
    prepare_row,
    prepare_rows,
    read_science_qa,
    text_similarities,
    text_similarity,
)
from my_rag.llm import HuggingFaceLLM, LLM, _hf_load_hint, available_providers, create_llm

RAW_ROWS = [
    {
        "question": "What is water made of?",
        "distractor1": "sand",
        "distractor2": "rock",
        "distractor3": "air",
        "correct_answer": "H2O",
        "support": "Water is H2O, two hydrogen atoms and one oxygen.",
    },
    {
        "question": "What gas do plants absorb?",
        "distractor1": "oxygen",
        "distractor2": "nitrogen",
        "distractor3": "helium",
        "correct_answer": "carbon dioxide",
        "support": "Plants absorb carbon dioxide during photosynthesis.",
    },
    {"question": "empty support?", "correct_answer": "x", "support": ""},
    {"question": "empty answer?", "correct_answer": "", "support": "something"},
]

PREPARED = [
    {
        "question": "What is water made of?",
        "correct_answer": "H2O",
        "support": "Water is H2O, two hydrogen atoms and one oxygen.",
        "completion": "H2O because Water is H2O, two hydrogen atoms and one oxygen.",
    },
    {
        "question": "What gas do plants absorb?",
        "correct_answer": "carbon dioxide",
        "support": "Plants absorb carbon dioxide during photosynthesis.",
        "completion": "carbon dioxide because Plants absorb carbon dioxide during photosynthesis.",
    },
]


class FakeEmbedding:
    """Word-count vectors keyed to a small vocabulary (deterministic, offline)."""

    vocabulary = ("water", "plants", "h2o", "carbon")

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)

    def _vector(self, text):
        words = text.lower().split()
        return [float(words.count(term)) + 0.1 for term in self.vocabulary]


class FakeLLM(LLM):
    def __init__(self, reply: str = "A beginner summary.") -> None:
        self.reply = reply
        self.calls: list[tuple] = []

    def complete(self, prompt, *, system=None, temperature=None) -> str:
        self.calls.append((prompt, system, temperature))
        return self.reply


class FakeIndex:
    """Minimal ChromaIndex stand-in: replays canned matches and records calls."""

    def __init__(self, matches=None, *, by_question=None, name: str = "sciq_supports6") -> None:
        self.matches = list(matches or [])
        self.by_question = by_question
        self.name = name
        self.rows = []
        self.queries = []
        self.deleted = False

    def add_rows(self, rows):
        self.rows.extend(rows)
        return len(rows)

    def query(self, question, *, top_k=DEFAULT_TOP_K):
        self.queries.append(question)
        if self.by_question is not None:
            return [QAMatch(question, 0.0, self.by_question[question])]
        return self.matches[:top_k]

    def count(self):
        return len(self.rows)

    def delete(self):
        self.deleted = True


class FakeSession:
    """Replays datasets-server pages of rows, honouring offset/length."""

    def __init__(self, rows) -> None:
        self.rows = list(rows)
        self.calls: list[dict] = []

    def get(self, url, params=None, headers=None, timeout=None):
        params = dict(params or {})
        self.calls.append(params)
        offset = int(params.get("offset", 0))
        length = int(params.get("length", 100))
        page = self.rows[offset : offset + length]
        payload = {"rows": [{"row": row} for row in page], "num_rows_total": len(self.rows)}
        return SimpleNamespace(json=lambda: payload, raise_for_status=lambda: None)


def _settings(**overrides) -> Settings:
    values = dict(
        provider="ollama",
        model="qwen2.5:3b",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        embedding_model="all-minilm",
        embedding_dimension=4,
        chroma_collection="test_chroma",
    )
    values.update(overrides)
    return Settings(**values)


class PrepareTests(unittest.TestCase):
    def test_prepare_row_drops_distractors_and_builds_completion(self):
        prepared = prepare_row(RAW_ROWS[0])
        self.assertNotIn(DROP_COLUMNS[0], prepared)
        self.assertEqual(prepared["correct_answer"], "H2O")
        self.assertEqual(prepared["completion"], "H2O because Water is H2O, two hydrogen atoms and one oxygen.")

    def test_prepare_rows_filters_empty_support_or_answer(self):
        prepared = prepare_rows(RAW_ROWS)
        self.assertEqual(len(prepared), 2)
        self.assertTrue(all(row["support"] and row["correct_answer"] for row in prepared))


class FetchTests(unittest.TestCase):
    def test_fetch_pages_until_limit_and_skips_invalid_rows(self):
        session = FakeSession(RAW_ROWS)
        # Four raw rows hold two valid ones; asking for 3 exhausts the two pages.
        rows = fetch_science_qa(3, session=session, page_size=2)
        self.assertEqual(rows, PREPARED)
        self.assertGreaterEqual(len(session.calls), 2)
        self.assertEqual(session.calls[0]["dataset"], "allenai/sciq")

    def test_bad_limits_are_rejected(self):
        with self.assertRaises(ValueError):
            fetch_science_qa(0, session=FakeSession(RAW_ROWS))
        with self.assertRaises(ValueError):
            fetch_science_qa(1, session=FakeSession(RAW_ROWS), page_size=0)

    def test_collect_writes_jsonl_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sciq" / "sciq.jsonl"
            written = collect_science_qa(path, limit=2, session=FakeSession(RAW_ROWS), page_size=2)
            self.assertEqual(written, path)
            self.assertEqual(read_science_qa(path), PREPARED)

    def test_read_missing_dataset_fails_clearly(self):
        with self.assertRaises(FileNotFoundError):
            read_science_qa("/nonexistent/sciq.jsonl")


class SimilarityTests(unittest.TestCase):
    def test_identical_texts_score_one(self):
        self.assertAlmostEqual(text_similarity("water plants", "water plants", FakeEmbedding()), 1.0)

    def test_pairwise_returns_one_score_per_pair(self):
        scores = text_similarities(["water", "plants"], ["water", "carbon"], FakeEmbedding())
        self.assertEqual(len(scores), 2)
        self.assertGreater(scores[0], scores[1])

    def test_mismatched_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            text_similarities(["a"], ["a", "b"], FakeEmbedding())

    def test_empty_input_returns_empty(self):
        self.assertEqual(text_similarities([], [], FakeEmbedding()), [])


class PromptTests(unittest.TestCase):
    def test_summary_prompt_prefixes_the_instruction(self):
        prompt = build_summary_prompt("H2O because water is H2O.")
        self.assertTrue(prompt.startswith(SUMMARY_INSTRUCTION))
        self.assertTrue(prompt.endswith("H2O because water is H2O."))

    def test_empty_context_is_just_the_instruction(self):
        self.assertEqual(build_summary_prompt(""), SUMMARY_INSTRUCTION)


class ChromaIndexTests(unittest.TestCase):
    """The real in-memory Chroma client, no server — a genuine round-trip."""

    def test_add_query_count_and_delete(self):
        index = open_science_collection(_settings(), embedding_function=FakeEmbedding())
        self.assertIsInstance(index, ChromaIndex)
        self.assertEqual(index.add_rows(PREPARED), 2)
        self.assertEqual(index.count(), 2)
        matches = index.query("What is water made of?", top_k=1)
        self.assertEqual(len(matches), 1)
        self.assertIsInstance(matches[0], QAMatch)
        self.assertIn("H2O", matches[0].text)
        index.delete()

    def test_missing_collection_fails_clearly(self):
        with self.assertRaises(FileNotFoundError):
            open_science_collection(_settings(), collection="absent_collection", create=False)


class DynamicRAGTests(unittest.TestCase):
    def _rag(self, matches=None, llm=None, **overrides):
        index = FakeIndex(matches=matches or [QAMatch("0", 0.1, "H2O because water is H2O.")])
        rag = DynamicRAG(
            _settings(),
            index,
            embedding_function=FakeEmbedding(),
            llm=llm or FakeLLM(),
            top_k=1,
            **overrides,
        )
        return rag, index

    def test_answer_augments_the_retrieved_support(self):
        llm = FakeLLM("Water is two hydrogens and an oxygen.")
        rag, _ = self._rag(llm=llm)
        answer = rag.answer("What is water made of?")
        self.assertEqual(answer.retrieved, "H2O because water is H2O.")
        self.assertTrue(answer.prompt.startswith(SUMMARY_INSTRUCTION))
        self.assertEqual(answer.response, llm.reply)
        prompt, system, temperature = llm.calls[0]
        self.assertEqual(temperature, GENERATION_TEMPERATURE)
        self.assertIn("H2O because water is H2O.", prompt)

    def test_default_question_matches_the_book(self):
        rag, index = self._rag()
        rag.answer()
        self.assertEqual(index.queries, [DEFAULT_QUESTION])

    def test_evaluate_reports_accuracy_above_the_threshold(self):
        rows = PREPARED
        # Each question retrieves its own completion -> similarity 1.0 -> 100%.
        by_question = {row["question"]: row["completion"] for row in rows}
        index = FakeIndex(by_question=by_question)
        rag = DynamicRAG(_settings(), index, embedding_function=FakeEmbedding(), llm=FakeLLM())
        report = rag.evaluate(rows)
        self.assertEqual(report.documents, 2)
        self.assertEqual(report.threshold, SIMILARITY_THRESHOLD)
        self.assertAlmostEqual(report.accuracy, 1.0)
        self.assertEqual(report.correct, 2)

    def test_evaluate_rejects_an_empty_dataset(self):
        rag, _ = self._rag()
        with self.assertRaises(ValueError):
            rag.evaluate([])


class ProviderTests(unittest.TestCase):
    def test_hf_provider_is_registered(self):
        self.assertIn("hf", available_providers())

    def test_hf_provider_builds_without_loading_a_model(self):
        llm = create_llm(_settings(provider="hf", model="meta-llama/Llama-2-7b-chat-hf"))
        self.assertIsInstance(llm, HuggingFaceLLM)
        self.assertIsNone(llm._pipeline)  # the transformers pipeline is lazy
        self.assertIsNone(llm.device_map)  # "auto" segfaults on Apple-silicon macOS

    def test_hf_default_model_is_a_small_ungated_local_model(self):
        self.assertEqual(PROVIDER_DEFAULTS["hf"]["model"], "Qwen/Qwen2.5-0.5B-Instruct")

    def test_gated_load_error_gets_an_actionable_hint(self):
        hint = _hf_load_hint(
            "meta-llama/Llama-2-7b-chat-hf",
            RuntimeError("403 Client Error: you are not in the authorized list for a gated repo"),
        )
        self.assertIn("hf auth login", hint)
        self.assertIn("Qwen2.5-0.5B-Instruct", hint)
        self.assertIn("~/.config/hugging_face/token", hint)

    def test_other_load_error_passes_through(self):
        self.assertEqual(_hf_load_hint("m", ValueError("boom")), "boom")

    def test_gated_model_falls_back_to_the_ungated_default(self):
        llm = HuggingFaceLLM(
            "meta-llama/Llama-2-7b-chat-hf",
            fallback_model="Qwen/Qwen2.5-0.5B-Instruct",
        )
        calls = []

        def fake_build(pipeline, model, dtype):
            calls.append(model)
            if model == "meta-llama/Llama-2-7b-chat-hf":
                raise RuntimeError("403 Client Error: not in the authorized list for a gated repo")
            return "pipeline"

        llm._build = fake_build  # stub the network/weights
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline = llm._load()
        self.assertEqual(pipeline, "pipeline")
        self.assertEqual(calls, ["meta-llama/Llama-2-7b-chat-hf", "Qwen/Qwen2.5-0.5B-Instruct"])
        self.assertEqual(llm.model, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertTrue(any("falling back" in str(w.message) for w in caught))

    def test_gated_model_without_fallback_raises_with_hint(self):
        llm = HuggingFaceLLM("meta-llama/Llama-2-7b-chat-hf")

        def fake_build(pipeline, model, dtype):
            raise RuntimeError("403 Client Error: gated repo")

        llm._build = fake_build
        with self.assertRaises(RuntimeError) as caught:
            llm._load()
        self.assertIn("hf auth login", str(caught.exception))


class ConfigTests(unittest.TestCase):
    def test_chapter_8_defaults_are_registered(self):
        self.assertEqual(DEFAULT_SCIENCE_QA, Path("data/raw/sciq/sciq.jsonl"))
        self.assertEqual(DEFAULT_CHROMA_COLLECTION, "sciq_supports6")
        self.assertEqual(DEFAULT_SCIENCE_LIMIT, 500)
        self.assertEqual(DEFAULT_QUESTION, "Millions of years ago, plants used energy from the sun to form what?")
        self.assertEqual(DEFAULT_TOP_K, 1)

    def _no_token_env(self, **extra):
        env = {"HF_TOKEN": "", "HUGGING_FACE_HUB_TOKEN": "", "MY_RAG_HF_TOKEN_FILE": ""}
        env.update(extra)
        return mock.patch.dict("os.environ", env, clear=False)

    def test_load_hf_token_reads_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "token"
            path.write_text("hf_secret_token\n", encoding="utf-8")
            with self._no_token_env():
                token = load_hf_token(path)
        self.assertEqual(token, "hf_secret_token")

    def test_load_hf_token_takes_the_first_line_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "token"
            path.write_text("first_line\nsecond_line\n", encoding="utf-8")
            with self._no_token_env():
                token = load_hf_token(path)
        self.assertEqual(token, "first_line")

    def test_existing_env_token_wins(self):
        with self._no_token_env(HF_TOKEN="env_token"):
            self.assertEqual(load_hf_token("/nonexistent/token"), "env_token")

    def test_missing_token_file_returns_none(self):
        with self._no_token_env():
            self.assertIsNone(load_hf_token("/nonexistent/token"))

    def test_token_file_env_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "token"
            path.write_text("override_token\n", encoding="utf-8")
            with self._no_token_env(MY_RAG_HF_TOKEN_FILE=str(path)):
                self.assertEqual(load_hf_token(), "override_token")

    def test_settings_read_science_and_chroma_env(self):
        with mock.patch.dict(
            "os.environ",
            {
                "MY_RAG_SCIENCE_QA": "/tmp/sciq.jsonl",
                "MY_RAG_SCIENCE_LIMIT": "42",
                "MY_RAG_CHROMA_COLLECTION": "my_collection",
            },
            clear=False,
        ):
            settings = Settings.from_env(provider="ollama")
        self.assertEqual(settings.science_qa, Path("/tmp/sciq.jsonl"))
        self.assertEqual(settings.science_limit, 42)
        self.assertEqual(settings.chroma_collection, "my_collection")


class CliTests(unittest.TestCase):
    def test_chapter_8_flags_parse(self):
        args = build_parser().parse_args(
            ["--rag", "dynamic", "--evaluate", "--keep-collection", "--top-k", "3", "--collection", "mine"]
        )
        self.assertEqual((args.rag, args.evaluate, args.keep_collection, args.top_k, args.collection),
                         ("dynamic", True, True, 3, "mine"))
        corpus = build_parser().parse_args(["--collect", "--corpus", "sciq", "--limit", "7"])
        self.assertEqual((corpus.corpus, corpus.limit), ("sciq", 7))

    def test_dynamic_query_runs_offline_and_deletes_the_collection(self):
        index = FakeIndex(matches=[QAMatch("0", 0.1, "H2O because water is H2O.")])
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch("my_rag.dynamic.read_science_qa", return_value=PREPARED), mock.patch(
            "my_rag.dynamic.open_science_collection", return_value=index
        ), mock.patch("my_rag.dynamic.create_embedding_function", return_value=FakeEmbedding()), mock.patch(
            "my_rag.dynamic.create_llm", return_value=FakeLLM("Water is H2O.")
        ):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                status = main(["--rag", "dynamic", "--show-context", "--evaluate"])
        self.assertEqual(status, 0)
        out = stdout.getvalue()
        self.assertIn("Water is H2O.", out)
        self.assertIn("H2O because water is H2O.", out)
        self.assertIn("Matches: 1", out)
        self.assertIn("Retrieval accuracy over 2 questions", out)
        self.assertIn("Session time:", out)
        self.assertTrue(index.deleted)

    def test_keep_collection_skips_deletion(self):
        index = FakeIndex(matches=[QAMatch("0", 0.1, "H2O because water is H2O.")])
        with mock.patch("my_rag.dynamic.read_science_qa", return_value=PREPARED), mock.patch(
            "my_rag.dynamic.open_science_collection", return_value=index
        ), mock.patch("my_rag.dynamic.create_embedding_function", return_value=FakeEmbedding()), mock.patch(
            "my_rag.dynamic.create_llm", return_value=FakeLLM()
        ):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                status = main(["--rag", "dynamic", "--keep-collection"])
        self.assertEqual(status, 0)
        self.assertFalse(index.deleted)

    def test_collect_sciq_runs_offline_with_fakes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sciq.jsonl"
            with mock.patch("my_rag.dynamic.collect_science_qa", return_value=path):
                stderr = io.StringIO()
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                    status = main(["--collect", "--corpus", "sciq", "--limit", "2", "--output", str(path)])
        self.assertEqual(status, 0)
        self.assertIn("wrote", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
