"""Local MLX fine-tuning of RAG-derived data (chapter 9).

Open-source reimplementation of the book's
``Chapter09/Fine_tuning_OpenAI_GPT-4o-mini.ipynb`` — but **without** OpenAI: the
SciQ science-QA dataset (whose human-written supports are the assistant targets)
becomes a chat fine-tuning file, a small model is fine-tuned *locally* with
Apple MLX / ``mlx-lm`` LoRA, and the resulting adapter is run — and compared
against the base model and pure retrieval. No API key, no network, no paid
service.

The notebook's three steps map onto three commands:

- **Prepare** — :func:`build_example` reproduces the notebook's fine-tuning item
  (a ``messages`` array: a ``system`` instruction, the ``user`` question and the
  ``assistant`` target ``correct_answer + " Explanation: " + support``);
  :func:`collect_finetune_dataset` reads the SciQ cache chapter 8 already built
  (fetching it first if absent) and writes the JSONL.
- **Fine-tune** — :func:`write_mlx_dataset` splits the JSONL into MLX's
  ``train.jsonl``/``valid.jsonl`` and :class:`MLXFineTuner` runs
  ``python -m mlx_lm lora --train`` (plus ``mlx_lm.fuse`` on request).
- **Run** — :class:`FineTunedRAG` asks the adapter through the ``mlx`` provider
  (the book's step-3 ``system`` prompt, ``temperature=0.0``);
  :meth:`FineTunedRAG.compare` runs the same question through the base model
  (no adapter) and, when a retriever is passed, through pure retrieval.

Substitutions, consistent with the rest of ``my_rag``:

- **Apple MLX LoRA instead of OpenAI's hosted fine-tuning.** There is no
  open-source drop-in for the hosted fine-tuning API; ``mlx-lm`` is a local
  trainer. It runs on Apple silicon (``pip install "mlx-lm[train]"``) and needs
  no key, so chapter 9 is usable out of the box.
- **The SciQ cache from chapter 8, not a second ``load_dataset``.** The notebook
  reloads SciQ with ``datasets``; here :func:`~my_rag.dynamic.read_science_qa`
  reuses ``data/raw/sciq/sciq.jsonl``, so the chapter needs neither the
  ``datasets``/``pyarrow`` dependency nor a second download.
- **Plain ``json`` instead of ``jsonlines``.** The Q&A file is one JSON object
  per line; :func:`write_finetune_jsonl` / :func:`read_finetune_jsonl` write and
  read it directly, so no extra dependency.

The book's Colab scaffolding (the Drive mount, ``api_key.txt``, the ``!pip
install`` cells) and its inline ``display()`` cells are not ported.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from .config import (
    DEFAULT_FINETUNE_DATA,
    DEFAULT_FINETUNE_LIMIT,
    DEFAULT_MLX_ADAPTERS,
    DEFAULT_MLX_DATA,
    DEFAULT_MLX_ITERS,
    DEFAULT_SCIENCE_QA,
    Settings,
)
from .dynamic import collect_science_qa, read_science_qa
from .llm import LLM, MlxLLM, create_llm

#: The book's fine-tuning ``system`` message (the dataset's instruction).
FINE_TUNE_SYSTEM = (
    "Given a science question, provide the correct answer with a detailed explanation."
)

#: The book's step-3 ``system`` message (when running the fine-tuned model).
RUN_SYSTEM = "Given a question, reply with a complete explanation for students."

#: The label the book glues between the answer and the support in the target.
EXPLANATION_SEPARATOR = " Explanation: "

#: The book's generation temperature for the fine-tuned model (``0.0``).
RUN_TEMPERATURE = 0.0


# --------------------------------------------------------------------------- #
# Prepare — the fine-tuning dataset
# --------------------------------------------------------------------------- #
def build_detailed_answer(row: dict) -> str:
    """The book's assistant target: ``correct_answer + " Explanation: " + support``."""
    return f"{row['correct_answer']}{EXPLANATION_SEPARATOR}{row['support']}"


def build_example(row: dict) -> dict:
    """One fine-tuning example in the chat ``messages`` format (the book's item)."""
    return {
        "messages": [
            {"role": "system", "content": FINE_TUNE_SYSTEM},
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": build_detailed_answer(row)},
        ]
    }


def build_dataset(rows: Sequence[dict]) -> list[dict]:
    """Turn SciQ rows into the notebook's list of ``{"messages": [...]}`` items."""
    return [build_example(row) for row in rows]


def write_finetune_jsonl(examples: Sequence[dict], path: Path | str = DEFAULT_FINETUNE_DATA) -> Path:
    """Write fine-tuning examples to ``path`` as JSONL (one object per line)."""
    path = Path(path)
    examples = list(examples)
    if not examples:
        raise ValueError("cannot write an empty fine-tuning file")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")
    return path


def read_finetune_jsonl(path: Path | str = DEFAULT_FINETUNE_DATA) -> list[dict]:
    """Read a fine-tuning JSONL file into a list of example dicts."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"fine-tuning file {path} not found; build it with: "
            "python -m my_rag --collect --corpus finetune"
        )
    examples = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    if not examples:
        raise ValueError(f"fine-tuning file {path} is empty")
    return examples


def load_science_rows(
    path: Path | str = DEFAULT_SCIENCE_QA,
    *,
    limit: Optional[int] = None,
    session=None,
) -> list[dict]:
    """Read the cached SciQ rows, fetching and caching them first when absent."""
    path = Path(path)
    try:
        rows = read_science_qa(path)
    except FileNotFoundError:
        rows = []
    if not rows or (limit and len(rows) < limit):
        # Absent, or cached by chapter 8 with fewer rows than asked for: (re)fetch.
        collect_science_qa(path, limit=limit or DEFAULT_FINETUNE_LIMIT, session=session)
        rows = read_science_qa(path)
    return rows[:limit] if limit else rows


def prepare_finetune_dataset(
    rows: Sequence[dict],
    path: Path | str = DEFAULT_FINETUNE_DATA,
    *,
    limit: Optional[int] = None,
) -> Path:
    """Build the chat fine-tuning JSONL from SciQ rows and write it."""
    sample = list(rows)[:limit] if limit else list(rows)
    if not sample:
        raise ValueError("cannot build a fine-tuning file from an empty dataset")
    return write_finetune_jsonl(build_dataset(sample), path)


def collect_finetune_dataset(
    path: Path | str = DEFAULT_FINETUNE_DATA,
    *,
    science_qa: Path | str = DEFAULT_SCIENCE_QA,
    limit: Optional[int] = None,
    session=None,
    progress: Optional[Callable[[str], None]] = None,
) -> Path:
    """Fetch/cache SciQ, then write the fine-tuning JSONL (``--collect --corpus finetune``)."""
    rows = load_science_rows(science_qa, limit=limit, session=session)
    if progress is not None:
        progress(f"prepared {len(rows)} fine-tuning examples")
    return prepare_finetune_dataset(rows, path, limit=limit)


# --------------------------------------------------------------------------- #
# Local fine-tuning — Apple MLX
# --------------------------------------------------------------------------- #
#: The ``mlx-lm`` CLI modules used for training and (optional) fusing.
#: (``python -m mlx_lm <subcommand>``; the ``python -m mlx_lm.<cmd>`` form is deprecated).
MLX_LORA_MODULE = "lora"
MLX_FUSE_MODULE = "fuse"
MLX_SERVE_MODULE = "server"

#: MLX LoRA defaults (all overridable via flags / env).
DEFAULT_MLX_BATCH_SIZE = 4
DEFAULT_MLX_NUM_LAYERS = 16
DEFAULT_MLX_VALID_RATIO = 0.1

#: MLX expects one file per split in the data directory.
MLX_SPLITS: tuple[str, ...] = ("train.jsonl", "valid.jsonl", "test.jsonl")


def split_examples(
    examples: Sequence[dict],
    *,
    valid_ratio: float = DEFAULT_MLX_VALID_RATIO,
    test_ratio: float = 0.0,
) -> dict[str, list[dict]]:
    """Split examples into MLX's ``train``/``valid``/``test`` groups.

    The tail of the list is used for validation (and then test), so a small
    dataset still yields a non-empty train split and the split is deterministic.
    """
    items = list(examples)
    if not items:
        raise ValueError("cannot split an empty dataset")
    if not 0.0 <= valid_ratio < 1.0 or not 0.0 <= test_ratio < 1.0:
        raise ValueError("valid_ratio/test_ratio must be in [0, 1)")
    if valid_ratio + test_ratio >= 1.0:
        raise ValueError("valid_ratio + test_ratio must be < 1")
    count = len(items)
    test_count = int(count * test_ratio)
    valid_count = int(count * valid_ratio)
    train_end = count - valid_count - test_count
    if train_end < 1:
        # Keep at least one training example on tiny datasets.
        train_end = 1
        remaining = count - 1
        test_count = min(test_count, remaining)
        valid_count = min(valid_count, remaining - test_count)
    groups = {
        "train": items[:train_end],
        "valid": items[train_end : train_end + valid_count],
        "test": items[train_end + valid_count : train_end + valid_count + test_count],
    }
    return groups


def write_mlx_dataset(
    examples: Sequence[dict],
    data_dir: Path | str = DEFAULT_MLX_DATA,
    *,
    valid_ratio: float = DEFAULT_MLX_VALID_RATIO,
    test_ratio: float = 0.0,
) -> dict[str, Path]:
    """Write the MLX ``train.jsonl`` (and non-empty ``valid``/``test``) for ``--data``."""
    data_dir = Path(data_dir)
    groups = split_examples(examples, valid_ratio=valid_ratio, test_ratio=test_ratio)
    data_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, rows in groups.items():
        if not rows:
            continue
        path = data_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        written[name] = path
    return written


def build_mlx_lora_command(
    model: str,
    data_dir: Path | str = DEFAULT_MLX_DATA,
    adapters: Path | str = DEFAULT_MLX_ADAPTERS,
    *,
    iters: int = DEFAULT_MLX_ITERS,
    batch_size: int = DEFAULT_MLX_BATCH_SIZE,
    num_layers: int = DEFAULT_MLX_NUM_LAYERS,
    grad_checkpoint: bool = True,
) -> list[str]:
    """The ``mlx_lm.lora --train`` argv (``python -m mlx_lm lora ...``)."""
    command = [
        sys.executable,
        "-m",
        "mlx_lm",
        MLX_LORA_MODULE,
        "--model",
        str(model),
        "--train",
        "--data",
        str(data_dir),
        "--adapter-path",
        str(adapters),
        "--iters",
        str(iters),
        "--batch-size",
        str(batch_size),
        "--num-layers",
        str(num_layers),
    ]
    if grad_checkpoint:
        command.append("--grad-checkpoint")
    return command


def build_mlx_fuse_command(
    model: str,
    adapters: Path | str = DEFAULT_MLX_ADAPTERS,
    fused: Path | str = "fused_model",
) -> list[str]:
    """The ``mlx_lm.fuse`` argv that merges the adapter into a standalone model."""
    return [
        sys.executable,
        "-m",
        "mlx_lm",
        MLX_FUSE_MODULE,
        "--model",
        str(model),
        "--adapter-path",
        str(adapters),
        "--save-path",
        str(fused),
    ]


@dataclass(frozen=True)
class MlxRun:
    """Outcome of :meth:`MLXFineTuner.train`: the model, data, adapters and argv."""

    model: str
    data: Path
    adapters: Path
    iters: int
    command: list[str]

    def serve_command(self) -> list[str]:
        """The ``python -m mlx_lm server`` argv that serves the adapter."""
        return [
            sys.executable,
            "-m",
            "mlx_lm",
            MLX_SERVE_MODULE,
            "--model",
            self.model,
            "--adapter-path",
            str(self.adapters),
        ]


class MLXFineTuner:
    """Apple-MLX LoRA fine-tuning (chapter 9's open-source, key-free path).

    ``runner`` is the ``subprocess.run`` used to launch ``mlx_lm.lora``; tests
    inject a fake so nothing needs MLX or the weights installed.
    """

    def __init__(
        self,
        model: str,
        *,
        data_dir: Path | str = DEFAULT_MLX_DATA,
        adapters: Path | str = DEFAULT_MLX_ADAPTERS,
        iters: int = DEFAULT_MLX_ITERS,
        batch_size: int = DEFAULT_MLX_BATCH_SIZE,
        num_layers: int = DEFAULT_MLX_NUM_LAYERS,
        valid_ratio: float = DEFAULT_MLX_VALID_RATIO,
        runner: Optional[Callable[..., object]] = None,
    ) -> None:
        self.model = model
        self.data_dir = Path(data_dir)
        self.adapters = Path(adapters)
        self.iters = iters
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.valid_ratio = valid_ratio
        self.runner = runner or subprocess.run

    def prepare(self, examples: Sequence[dict]) -> dict[str, Path]:
        """Write the MLX train/valid JSONL split under :attr:`data_dir`."""
        return write_mlx_dataset(examples, self.data_dir, valid_ratio=self.valid_ratio)

    def command(self) -> list[str]:
        """The ``mlx_lm.lora --train`` argv for the current settings."""
        return build_mlx_lora_command(
            self.model,
            self.data_dir,
            self.adapters,
            iters=self.iters,
            batch_size=self.batch_size,
            num_layers=self.num_layers,
        )

    def train(self) -> MlxRun:
        """Run the LoRA training (requires ``mlx_lm`` and the base model weights)."""
        if not (self.data_dir / "train.jsonl").is_file():
            raise FileNotFoundError(
                f"MLX train split {self.data_dir / 'train.jsonl'} not found; "
                "run prepare() first (python -m my_rag --provider mlx --fine-tune)"
            )
        command = self.command()
        self.adapters.parent.mkdir(parents=True, exist_ok=True)
        self.runner(command, check=True)
        return MlxRun(
            model=self.model,
            data=self.data_dir,
            adapters=self.adapters,
            iters=self.iters,
            command=command,
        )

    def fuse(self, fused: Path | str = "fused_model") -> list[str]:
        """Merge the adapter into a standalone model at ``fused`` (``mlx_lm.fuse``)."""
        command = build_mlx_fuse_command(self.model, self.adapters, fused)
        self.runner(command, check=True)
        return command


def open_mlx_fine_tuner(
    settings: Settings,
    *,
    model: Optional[str] = None,
    runner: Optional[Callable[..., object]] = None,
) -> MLXFineTuner:
    """Build an :class:`MLXFineTuner` from ``settings`` (no MLX import needed)."""
    return MLXFineTuner(
        model or settings.model,
        data_dir=settings.mlx_data,
        adapters=settings.mlx_adapters,
        iters=settings.mlx_iters,
        runner=runner,
    )


# --------------------------------------------------------------------------- #
# Run + compare
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FineTuneAnswer:
    """Outcome of :meth:`FineTunedRAG.answer`: the model, prompt and response."""

    question: str
    model: str
    system: str
    response: str
    elapsed: float


@dataclass(frozen=True)
class Comparison:
    """The three answers side by side: fine-tuned, base generation, pure retrieval."""

    question: str
    fine_tuned: str
    base: str
    retrieved: str = ""


class FineTunedRAG:
    """Run the locally fine-tuned MLX model, and compare it against the base model.

    The fine-tuned model is the base model plus the LoRA adapter
    (``settings.mlx_adapters``); the baseline is the same base model without the
    adapter. Without a ``retriever`` :meth:`compare` contrasts only those two;
    pass a chapter-8 :class:`~my_rag.dynamic.DynamicRAG` (or anything with a
    ``retrieve(query)`` method) to add the pure-retrieval answer.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        llm: Optional[LLM] = None,
        base_llm: Optional[LLM] = None,
        retriever=None,
    ) -> None:
        self.settings = settings
        self.model = settings.model
        self._llm = llm
        self._base_llm = base_llm
        self.retriever = retriever

    def _fine_llm(self) -> LLM:
        if self._llm is None:
            self._llm = create_llm(self.settings)
        return self._llm

    def _base(self) -> LLM:
        if self._base_llm is None:
            # The baseline is the same base model without the LoRA adapter.
            self._base_llm = MlxLLM(self.settings.model, temperature=self.settings.temperature)
        return self._base_llm

    def answer(self, question: str) -> FineTuneAnswer:
        """Ask the fine-tuned (adapter) model, with no retrieval context."""
        start = time.perf_counter()
        response = self._fine_llm().complete(
            question, system=RUN_SYSTEM, temperature=RUN_TEMPERATURE
        )
        elapsed = time.perf_counter() - start
        return FineTuneAnswer(
            question=question,
            model=self.model,
            system=RUN_SYSTEM,
            response=response,
            elapsed=elapsed,
        )

    def retrieve(self, question: str) -> str:
        """The nearest retrieved support ("" when no retriever is configured)."""
        if self.retriever is None:
            return ""
        matches = self.retriever.retrieve(question)
        if not matches:
            return ""
        first = matches[0]
        return getattr(first, "text", None) or str(first)

    def compare(self, question: str, *, retrieve: bool = True) -> Comparison:
        """Fine-tuned answer next to the base model and (optionally) pure retrieval."""
        fine = self.answer(question)
        base = self._base().complete(question, system=RUN_SYSTEM, temperature=RUN_TEMPERATURE)
        retrieved = self.retrieve(question) if retrieve else ""
        return Comparison(
            question=question, fine_tuned=fine.response, base=base, retrieved=retrieved
        )


__all__ = [
    "FINE_TUNE_SYSTEM",
    "RUN_SYSTEM",
    "EXPLANATION_SEPARATOR",
    "RUN_TEMPERATURE",
    "build_detailed_answer",
    "build_example",
    "build_dataset",
    "write_finetune_jsonl",
    "read_finetune_jsonl",
    "load_science_rows",
    "prepare_finetune_dataset",
    "collect_finetune_dataset",
    "MLX_LORA_MODULE",
    "MLX_FUSE_MODULE",
    "MLX_SERVE_MODULE",
    "MLX_SPLITS",
    "DEFAULT_MLX_BATCH_SIZE",
    "DEFAULT_MLX_NUM_LAYERS",
    "DEFAULT_MLX_VALID_RATIO",
    "split_examples",
    "write_mlx_dataset",
    "build_mlx_lora_command",
    "build_mlx_fuse_command",
    "MlxRun",
    "MLXFineTuner",
    "open_mlx_fine_tuner",
    "FineTuneAnswer",
    "Comparison",
    "FineTunedRAG",
]
