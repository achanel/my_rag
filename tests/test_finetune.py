"""Offline tests for chapter 9 (local Apple MLX fine-tuning over the SciQ data).

No OpenAI, no key, no network, no ``mlx-lm`` install and no model run: the MLX
trainer gets a fake ``subprocess.run``, the LLMs are fakes, and the SciQ session
is replayed.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from my_rag.cli import build_parser, main
from my_rag.config import (
    DEFAULT_FINETUNE_DATA,
    DEFAULT_FINETUNE_LIMIT,
    DEFAULT_MLX_ADAPTERS,
    DEFAULT_MLX_DATA,
    DEFAULT_MLX_ITERS,
    PROVIDER_DEFAULTS,
    Settings,
)
from my_rag.dynamic import QAMatch
from my_rag.finetune import (
    EXPLANATION_SEPARATOR,
    FINE_TUNE_SYSTEM,
    RUN_SYSTEM,
    RUN_TEMPERATURE,
    Comparison,
    FineTunedRAG,
    MLXFineTuner,
    MlxRun,
    build_dataset,
    build_detailed_answer,
    build_example,
    build_mlx_fuse_command,
    build_mlx_lora_command,
    collect_finetune_dataset,
    load_science_rows,
    open_mlx_fine_tuner,
    prepare_finetune_dataset,
    read_finetune_jsonl,
    split_examples,
    write_finetune_jsonl,
    write_mlx_dataset,
)
from my_rag.llm import LLM, MlxLLM, _mlx_install_hint, available_providers, create_llm

ROWS = [
    {
        "question": "What gas do plants absorb?",
        "correct_answer": "carbon dioxide",
        "support": "Plants absorb carbon dioxide during photosynthesis.",
        "completion": "carbon dioxide because Plants absorb carbon dioxide during photosynthesis.",
    },
    {
        "question": "What is water made of?",
        "correct_answer": "H2O",
        "support": "Water is H2O, two hydrogen atoms and one oxygen.",
        "completion": "H2O because Water is H2O, two hydrogen atoms and one oxygen.",
    },
]

EXAMPLE = {
    "messages": [
        {"role": "system", "content": FINE_TUNE_SYSTEM},
        {"role": "user", "content": "What gas do plants absorb?"},
        {
            "role": "assistant",
            "content": "carbon dioxide" + EXPLANATION_SEPARATOR
            + "Plants absorb carbon dioxide during photosynthesis.",
        },
    ]
}


class FakeLLM(LLM):
    def __init__(self, reply: str = "A complete explanation.") -> None:
        self.reply = reply
        self.calls: list[tuple] = []

    def complete(self, prompt, *, system=None, temperature=None) -> str:
        self.calls.append((prompt, system, temperature))
        return self.reply


class FakeRetriever:
    def __init__(self, text: str = "Plants absorb carbon dioxide.") -> None:
        self.text = text
        self.queries: list[str] = []

    def retrieve(self, question):
        self.queries.append(question)
        return [QAMatch("0", 0.0, self.text)]


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, command, *, check=True):
        self.calls.append(list(command))
        return SimpleNamespace(returncode=0)


def _settings(**overrides) -> Settings:
    values = dict(
        provider="mlx",
        model="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        embedding_model="all-minilm",
        embedding_dimension=384,
    )
    values.update(overrides)
    return Settings(**values)


class DatasetTests(unittest.TestCase):
    def test_detailed_answer_glues_the_book_separator(self):
        self.assertEqual(
            build_detailed_answer(ROWS[0]),
            "carbon dioxide" + EXPLANATION_SEPARATOR
            + "Plants absorb carbon dioxide during photosynthesis.",
        )

    def test_example_is_the_book_messages_item(self):
        self.assertEqual(build_example(ROWS[0]), EXAMPLE)

    def test_dataset_builds_one_example_per_row(self):
        dataset = build_dataset(ROWS)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]["messages"][0]["content"], FINE_TUNE_SYSTEM)

    def test_jsonl_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "finetune" / "qa.jsonl"
            written = write_finetune_jsonl(build_dataset(ROWS), path)
            self.assertEqual(written, path)
            self.assertEqual(read_finetune_jsonl(path), build_dataset(ROWS))
            self.assertEqual(len(path.read_text().splitlines()), 2)

    def test_write_empty_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_finetune_jsonl([], Path(tmp) / "qa.jsonl")

    def test_read_missing_file_fails_clearly(self):
        with self.assertRaises(FileNotFoundError):
            read_finetune_jsonl("/nonexistent/finetune.jsonl")

    def test_prepare_dataset_writes_and_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qa.jsonl"
            prepare_finetune_dataset(ROWS, path, limit=1)
            self.assertEqual(len(read_finetune_jsonl(path)), 1)

    def test_prepare_dataset_rejects_empty_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                prepare_finetune_dataset([], Path(tmp) / "qa.jsonl")

    def test_load_science_rows_reads_the_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            science = Path(tmp) / "sciq.jsonl"
            science.write_text("\n".join(json.dumps(row) for row in ROWS), encoding="utf-8")
            self.assertEqual(load_science_rows(science), ROWS)
            self.assertEqual(load_science_rows(science, limit=1), ROWS[:1])

    def test_load_science_rows_refetches_when_cache_is_smaller_than_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            science = Path(tmp) / "sciq.jsonl"
            science.write_text(json.dumps(ROWS[0]), encoding="utf-8")

            def fake_collect(path, *, limit, session=None):
                Path(path).write_text("\n".join(json.dumps(row) for row in ROWS), encoding="utf-8")
                return Path(path)

            with mock.patch("my_rag.finetune.collect_science_qa", side_effect=fake_collect) as fetch:
                self.assertEqual(load_science_rows(science, limit=len(ROWS)), ROWS)
            fetch.assert_called_once()

    def test_collect_fetches_when_cache_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            science = Path(tmp) / "sciq.jsonl"
            out = Path(tmp) / "qa.jsonl"

            def fake_collect(path, *, limit, session=None):
                Path(path).write_text("\n".join(json.dumps(row) for row in ROWS), encoding="utf-8")
                return Path(path)

            lines: list[str] = []
            with mock.patch("my_rag.finetune.collect_science_qa", side_effect=fake_collect):
                collect_finetune_dataset(out, science_qa=science, limit=2, progress=lines.append)
            self.assertEqual(len(read_finetune_jsonl(out)), 2)
            self.assertTrue(any("fine-tuning examples" in line for line in lines))


class MlxDatasetTests(unittest.TestCase):
    def test_split_keeps_a_train_split_and_a_validation_tail(self):
        groups = split_examples(list(range(10)), valid_ratio=0.2)
        self.assertEqual(groups["train"], list(range(8)))
        self.assertEqual(groups["valid"], [8, 9])
        self.assertEqual(groups["test"], [])

    def test_split_keeps_at_least_one_training_example(self):
        groups = split_examples([1, 2], valid_ratio=0.9)
        self.assertEqual(len(groups["train"]), 1)
        self.assertEqual(groups["train"] + groups["valid"] + groups["test"], [1, 2])

    def test_split_rejects_empty_and_bad_ratios(self):
        with self.assertRaises(ValueError):
            split_examples([])
        with self.assertRaises(ValueError):
            split_examples([1, 2, 3], valid_ratio=1.0)
        with self.assertRaises(ValueError):
            split_examples([1, 2, 3], valid_ratio=0.6, test_ratio=0.6)

    def test_write_mlx_dataset_writes_chat_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            written = write_mlx_dataset(build_dataset(ROWS), tmp, valid_ratio=0.5)
            self.assertIn("train", written)
            self.assertIn("valid", written)
            self.assertEqual(read_finetune_jsonl(written["train"])[0], EXAMPLE)


class MlxCommandTests(unittest.TestCase):
    def test_lora_command_targets_mlx_lm_lora_train(self):
        command = build_mlx_lora_command("mlx-community/m", "data", "adapters", iters=7)
        self.assertEqual(command[1:4], ["-m", "mlx_lm", "lora"])
        self.assertIn("--train", command)
        self.assertEqual(command[command.index("--iters") + 1], "7")
        self.assertEqual(command[command.index("--adapter-path") + 1], "adapters")
        self.assertIn("--grad-checkpoint", command)

    def test_fuse_command_targets_mlx_lm_fuse(self):
        command = build_mlx_fuse_command("m", "adapters", "fused")
        self.assertEqual(command[1:4], ["-m", "mlx_lm", "fuse"])
        self.assertEqual(command[command.index("--save-path") + 1], "fused")

    def test_run_serve_command_uses_the_module_form(self):
        run = MlxRun(model="m", data=Path("d"), adapters=Path("a"), iters=1, command=[])
        self.assertEqual(run.serve_command()[1:4], ["-m", "mlx_lm", "server"])
        self.assertIn("--adapter-path", run.serve_command())


class MlxFineTunerTests(unittest.TestCase):
    def _tuner(self, runner=None, **kw):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        return MLXFineTuner(
            "mlx-community/m",
            data_dir=base / "mlx",
            adapters=base / "adapters",
            runner=runner or FakeRunner(),
            **kw,
        )

    def test_prepare_writes_the_split(self):
        tuner = self._tuner()
        written = tuner.prepare(build_dataset(ROWS))
        self.assertTrue((tuner.data_dir / "train.jsonl").is_file())
        self.assertIn("train", written)

    def test_train_requires_a_prepared_split(self):
        with self.assertRaises(FileNotFoundError):
            self._tuner().train()

    def test_train_runs_the_lora_command_and_returns_the_run(self):
        tuner = self._tuner(iters=9)
        tuner.prepare(build_dataset(ROWS))
        run = tuner.train()
        self.assertIsInstance(run, MlxRun)
        self.assertEqual(run.iters, 9)
        self.assertEqual(run.adapters, tuner.adapters)
        self.assertEqual(run.command, tuner.command())

    def test_fuse_runs_the_fuse_command(self):
        runner = FakeRunner()
        tuner = self._tuner(runner=runner)
        tuner.fuse("fused")
        self.assertEqual(runner.calls[-1][1:4], ["-m", "mlx_lm", "fuse"])

    def test_open_mlx_fine_tuner_reads_settings(self):
        settings = _settings(
            model="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            mlx_data=Path("/tmp/d"),
            mlx_adapters=Path("/tmp/a"),
            mlx_iters=3,
        )
        tuner = open_mlx_fine_tuner(settings)
        self.assertEqual((tuner.model, tuner.data_dir, tuner.adapters, tuner.iters),
                         ("mlx-community/Qwen2.5-0.5B-Instruct-4bit", Path("/tmp/d"), Path("/tmp/a"), 3))


class MlxLLMTests(unittest.TestCase):
    def test_mlx_provider_is_registered_and_builds_lazily(self):
        self.assertIn("mlx", available_providers())
        llm = create_llm(_settings(mlx_adapters=Path("/nonexistent/adapters")))
        self.assertIsInstance(llm, MlxLLM)
        self.assertIsNone(llm._model)  # nothing is loaded just by selecting the provider
        self.assertIsNone(llm.adapter_path)  # a missing adapter dir is not passed

    def test_complete_uses_the_chat_template_and_generate(self):
        llm = MlxLLM("m", temperature=0.0)
        captured: dict = {}

        class Tokenizer:
            def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
                return "|".join(message["content"] for message in messages)

        llm._load = lambda: ("MODEL", Tokenizer())

        def fake_generate(model, tokenizer, *, prompt, **kwargs):
            captured["prompt"] = prompt
            captured["kwargs"] = kwargs
            return "  Coriolis effect.  "

        llm._generate = fake_generate
        out = llm.complete("Why do winds turn?", system=RUN_SYSTEM)
        self.assertEqual(out, "Coriolis effect.")
        self.assertIn(RUN_SYSTEM, captured["prompt"])
        self.assertIn("Why do winds turn?", captured["prompt"])
        self.assertEqual(captured["kwargs"]["temp"], 0.0)

    def test_complete_falls_back_without_a_chat_template(self):
        llm = MlxLLM("m")

        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                raise ValueError("no chat template")

        llm._load = lambda: ("MODEL", Tokenizer())
        llm._generate = lambda model, tokenizer, *, prompt, **kwargs: prompt
        out = llm.complete("Q", system="S")
        self.assertIn("S", out)
        self.assertIn("Q", out)

    def test_missing_mlx_install_gets_a_hint(self):
        hint = _mlx_install_hint(ImportError("No module named 'mlx_lm'"))
        self.assertIn("mlx-lm[train]", hint)
        self.assertIn("Apple", hint)

    @unittest.skipIf(importlib.util.find_spec("mlx_lm") is not None, "mlx-lm is installed")
    def test_load_without_mlx_raises_the_hint(self):
        with self.assertRaises(RuntimeError) as caught:
            MlxLLM("m")._load()
        self.assertIn("mlx-lm[train]", str(caught.exception))


class FineTunedRAGTests(unittest.TestCase):
    def test_answer_uses_the_book_prompt(self):
        llm = FakeLLM("The Coriolis effect.")
        rag = FineTunedRAG(_settings(), llm=llm)
        answer = rag.answer("Why do winds turn?")
        self.assertEqual(answer.response, "The Coriolis effect.")
        self.assertEqual(answer.model, "mlx-community/Qwen2.5-0.5B-Instruct-4bit")
        prompt, system, temperature = llm.calls[0]
        self.assertEqual((prompt, system, temperature), ("Why do winds turn?", RUN_SYSTEM, RUN_TEMPERATURE))

    def test_compare_returns_fine_tuned_base_and_retrieval(self):
        rag = FineTunedRAG(
            _settings(),
            llm=FakeLLM("fine-tuned answer"),
            base_llm=FakeLLM("base answer"),
            retriever=FakeRetriever("retrieved support"),
        )
        comparison = rag.compare("Why?")
        self.assertIsInstance(comparison, Comparison)
        self.assertEqual(comparison.fine_tuned, "fine-tuned answer")
        self.assertEqual(comparison.base, "base answer")
        self.assertEqual(comparison.retrieved, "retrieved support")

    def test_compare_without_a_retriever_has_no_retrieval(self):
        rag = FineTunedRAG(_settings(), llm=FakeLLM("f"), base_llm=FakeLLM("b"))
        self.assertEqual(rag.compare("Why?").retrieved, "")


class ConfigTests(unittest.TestCase):
    def test_chapter_9_defaults_are_registered(self):
        self.assertEqual(DEFAULT_FINETUNE_LIMIT, 500)
        self.assertEqual(
            DEFAULT_FINETUNE_DATA, Path("data/processed/finetune/qa_prompts_and_completions.jsonl")
        )

    def test_mlx_defaults_are_registered(self):
        self.assertEqual(PROVIDER_DEFAULTS["mlx"]["model"], "mlx-community/Qwen2.5-0.5B-Instruct-4bit")
        self.assertEqual(DEFAULT_MLX_DATA, Path("data/processed/finetune/mlx"))
        self.assertEqual(DEFAULT_MLX_ADAPTERS, Path("data/processed/finetune/adapters"))
        self.assertEqual(DEFAULT_MLX_ITERS, 600)

    def test_settings_read_the_finetune_env(self):
        with mock.patch.dict(
            "os.environ",
            {"MY_RAG_FINETUNE_DATA": "/tmp/qa.jsonl", "MY_RAG_FINETUNE_LIMIT": "7"},
            clear=False,
        ):
            settings = Settings.from_env(provider="mlx")
        self.assertEqual(settings.finetune_data, Path("/tmp/qa.jsonl"))
        self.assertEqual(settings.finetune_limit, 7)

    def test_settings_read_the_mlx_env(self):
        with mock.patch.dict(
            "os.environ",
            {"MY_RAG_MLX_DATA": "/tmp/mlx", "MY_RAG_MLX_ADAPTERS": "/tmp/adapters", "MY_RAG_MLX_ITERS": "12"},
            clear=False,
        ):
            settings = Settings.from_env(provider="mlx")
        self.assertEqual(settings.mlx_data, Path("/tmp/mlx"))
        self.assertEqual(settings.mlx_adapters, Path("/tmp/adapters"))
        self.assertEqual(settings.mlx_iters, 12)


class CliTests(unittest.TestCase):
    def test_chapter_9_flags_parse(self):
        args = build_parser().parse_args(
            ["--provider", "mlx", "--fine-tune", "--iters", "10", "--adapter-path", "a", "--fuse"]
        )
        self.assertEqual((args.provider, args.fine_tune, args.iters, args.adapter_path, args.fuse),
                         ("mlx", True, 10, "a", True))
        compare = build_parser().parse_args(["--provider", "mlx", "--rag", "finetune", "--compare"])
        self.assertEqual((compare.rag, compare.compare), ("finetune", True))

    def test_fine_tune_requires_the_mlx_provider(self):
        stderr = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            status = main(["--provider", "ollama", "--fine-tune"])
        self.assertEqual(status, 1)
        self.assertIn("--provider mlx", stderr.getvalue())

    def test_mlx_fine_tune_prepares_and_trains_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "qa.jsonl"
            write_finetune_jsonl(build_dataset(ROWS), dataset)
            adapters = Path(tmp) / "adapters"
            data = Path(tmp) / "mlx"

            class FakeTuner:
                def __init__(self):
                    self.data_dir = data
                    self.adapters = adapters
                    self.iters = 600

                def prepare(self, examples):
                    self.data_dir.mkdir(parents=True, exist_ok=True)
                    return {"train": self.data_dir / "train.jsonl"}

                def train(self):
                    return MlxRun(model="mlx-community/m", data=self.data_dir,
                                  adapters=self.adapters, iters=self.iters, command=["x"])

            stdout = io.StringIO()
            with mock.patch("my_rag.finetune.open_mlx_fine_tuner", return_value=FakeTuner()):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                    status = main(
                        ["--provider", "mlx", "--fine-tune", "--output", str(dataset), "--limit", "2"]
                    )
        self.assertEqual(status, 0)
        out = stdout.getvalue()
        self.assertIn("MLX LoRA training done", out)
        self.assertIn("python -m mlx_lm server", out)

    def test_mlx_fine_tune_status_reports_adapters(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "mlx"
            data.mkdir()
            (data / "train.jsonl").write_text("{}\n{}\n", encoding="utf-8")
            adapters = Path(tmp) / "adapters"
            adapters.mkdir()
            (adapters / "adapters.safetensors").write_text("x", encoding="utf-8")
            stdout = io.StringIO()
            with mock.patch.dict(
                "os.environ",
                {"MY_RAG_MLX_DATA": str(data), "MY_RAG_MLX_ADAPTERS": str(adapters)},
                clear=False,
            ):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                    status = main(["--provider", "mlx", "--fine-tune-status"])
        self.assertEqual(status, 0)
        out = stdout.getvalue()
        self.assertIn("train.jsonl: 2 examples", out)
        self.assertIn("adapters.safetensors", out)

    def test_finetune_query_requires_an_adapter(self):
        stderr = io.StringIO()
        with mock.patch.dict("os.environ", {"MY_RAG_MLX_ADAPTERS": "/nonexistent/adapters"}, clear=False):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                status = main(["--provider", "mlx", "--rag", "finetune", "Why?"])
        self.assertEqual(status, 1)
        self.assertIn("no MLX adapter", stderr.getvalue())

    def test_finetune_query_runs_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapters = Path(tmp) / "adapters"
            adapters.mkdir()
            stdout = io.StringIO()
            with mock.patch.dict("os.environ", {"MY_RAG_MLX_ADAPTERS": str(adapters)}, clear=False), mock.patch(
                "my_rag.finetune.create_llm", return_value=FakeLLM("Coriolis effect.")
            ):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                    status = main(["--provider", "mlx", "--rag", "finetune", "Why?"])
        self.assertEqual(status, 0)
        self.assertIn("Coriolis effect.", stdout.getvalue())

    def test_finetune_compare_runs_offline(self):
        index = mock.Mock()
        index.query.return_value = [QAMatch("0", 0.0, "Plants absorb carbon dioxide.")]
        stdout = io.StringIO()

        class FakeBase(MlxLLM):
            def complete(self, prompt, *, system=None, temperature=None):
                return "base answer"

        with tempfile.TemporaryDirectory() as tmp:
            adapters = Path(tmp) / "adapters"
            adapters.mkdir()
            with mock.patch.dict(
                "os.environ", {"MY_RAG_MLX_ADAPTERS": str(adapters)}, clear=False
            ), mock.patch("my_rag.dynamic.read_science_qa", return_value=ROWS), mock.patch(
                "my_rag.dynamic.open_science_collection", return_value=index
            ), mock.patch(
                "my_rag.finetune.create_llm", return_value=FakeLLM("fine-tuned answer")
            ), mock.patch("my_rag.finetune.MlxLLM", FakeBase):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                    status = main(["--provider", "mlx", "--rag", "finetune", "--compare", "Why?"])
        self.assertEqual(status, 0)
        out = stdout.getvalue()
        self.assertIn("Fine-tuned adapter (no context):", out)
        self.assertIn("Base model (no adapter, no context):", out)
        self.assertIn("Pure retrieval (nearest SciQ support, no model):", out)
        self.assertIn("Plants absorb carbon dioxide.", out)
        index.delete.assert_called_once()


if __name__ == "__main__":
    unittest.main()
