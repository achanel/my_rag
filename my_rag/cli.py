"""Command-line entry point.

Examples::

    python -m my_rag "define a rag store"
    python -m my_rag --provider openai --model gpt-4o "..."
    python -m my_rag --no-stream --width 100 "..."
    python -m my_rag --think "why does RAG reduce hallucinations?"
    python -m my_rag --list-providers

Chapter 2, part 1 (data collection) fetches the Wikipedia corpus::

    python -m my_rag --collect
    python -m my_rag --collect --output data/raw/corpus.md

Chapter 2, part 2 (embeddings + vector store) embeds that corpus into a local
Deep Lake store and retrieves from it::

    python -m my_rag --embed
    python -m my_rag --rag embeddings "Tell me about space exploration on the Moon and Mars."

Chapter 1, part 2 adds the retrieval-augmented variants::

    python -m my_rag --rag naive "define a rag store"
    python -m my_rag --rag advanced "define a rag store"
    python -m my_rag --rag modular --method vector "define a rag store"
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import Optional

from .collection import DEFAULT_OUTPUT as DEFAULT_CORPUS
from .collection import WIKI_URLS, collect
from .config import CHUNK_SIZE, DEFAULT_VECTOR_STORE, Settings, load_env
from .corpus import DB_RECORDS
from .embeddings import create_embedding_function
from .llm import LLM, available_providers, create_llm
from .pipeline import RAGPipeline
from .retrieval import create_retriever

DEFAULT_QUERY = "define a rag store"


def format_response(response: str, width: int = 80) -> str:
    """Wrap the answer in the book's header/footer framing."""
    body = textwrap.fill(response, width=width)
    return f"Response:\n---------------\n{body}\n---------------"


def print_context(pipeline: RAGPipeline, query: str) -> None:
    """Print the record(s) the retriever fed into the prompt."""
    if pipeline.retriever is None:
        return
    records = pipeline.retriever.retrieve(query)
    print("Retrieved context:")
    print("---------------")
    for record in records:
        print(textwrap.fill(record, width=80))
    print("---------------")


def stream_response(pipeline: RAGPipeline, query: str) -> None:
    """Print the framed answer as it is generated, token by token."""
    chunks = pipeline.stream(query)
    first = next(chunks, "")  # opening the stream may raise (e.g. unknown model)
    print("Response:")
    print("---------------")
    if first:
        sys.stdout.write(first)
    for chunk in chunks:
        sys.stdout.write(chunk)
        sys.stdout.flush()
    print("\n---------------")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="my_rag",
        description="RAG playground: retrieval -> augmentation -> generation (ch. 1) and data collection (ch. 2).",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=DEFAULT_QUERY,
        help=f"question to send (default: {DEFAULT_QUERY!r})",
    )
    parser.add_argument("--provider", help="LLM provider (env MY_RAG_PROVIDER; default: ollama)")
    parser.add_argument("--model", help="model name (env MY_RAG_MODEL; default: provider default)")
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint (env OPENAI_BASE_URL)")
    parser.add_argument("--temperature", type=float, help="sampling temperature (env MY_RAG_TEMPERATURE)")
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stream tokens as they are generated (default: on)",
    )
    parser.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable model reasoning for thinking models such as qwen3 (env MY_RAG_THINK; default: off)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=80,
        help="wrap width for the printed answer (only with --no-stream)",
    )
    parser.add_argument(
        "--rag",
        choices=["none", "naive", "advanced", "modular", "embeddings"],
        default="none",
        help="RAG variant: none (plain generation), naive (keyword), advanced (TF-IDF vector/index), "
        "modular (selectable method), embeddings (vector store search)",
    )
    parser.add_argument(
        "--method",
        choices=["keyword", "vector", "indexed"],
        default="indexed",
        help="retrieval method for --rag advanced|modular (default: indexed)",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="print the retrieved record(s) before the answer",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="fetch the chapter 2 Wikipedia corpus and exit",
    )
    parser.add_argument(
        "--output",
        help=f"corpus path for --collect, or corpus to embed for --embed (default: {DEFAULT_CORPUS})",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="embed the collected corpus into the vector store and exit (chapter 2, part 2)",
    )
    parser.add_argument(
        "--vector-store",
        help=f"vector store path for --embed / --rag embeddings (default: {DEFAULT_VECTOR_STORE})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"character chunk size for --embed (default: {CHUNK_SIZE})",
    )
    parser.add_argument("--list-providers", action="store_true", help="list registered providers and exit")
    return parser


def _print_model_hint(llm: LLM, settings: Settings) -> None:
    """On a missing model, list what the server does have and how to pull it."""
    if not hasattr(llm, "list_models"):
        return
    try:
        installed = llm.list_models()
    except Exception:
        return
    if not installed:
        return
    print(f"hint: model {settings.model!r} is not installed.", file=sys.stderr)
    print(f"      installed models: {', '.join(installed)}", file=sys.stderr)
    if settings.provider == "ollama":
        print(f"      pull it with: ollama pull {settings.model}", file=sys.stderr)
    else:
        print("      pick one with: --model <name>", file=sys.stderr)


def collect_corpus(output: Optional[str]) -> int:
    """Fetch the chapter 2 Wikipedia corpus and write it to ``output``."""
    path = Path(output) if output else DEFAULT_CORPUS
    print(f"collecting {len(WIKI_URLS)} articles ...", file=sys.stderr)
    try:
        result = collect(output=path, progress=lambda line: print(line, file=sys.stderr))
    except OSError as exc:  # e.g. unwritable output path
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {result.path} ({len(result.collected)}/{len(WIKI_URLS)} articles)", file=sys.stderr)
    return 1 if result.failed else 0


def embed_corpus(corpus: Path, store_path: Path, chunk_size: int, settings: Settings) -> int:
    """Embed ``corpus`` into the Deep Lake vector store at ``store_path`` (ch. 2, part 2)."""
    from .vectorstore import DeepLakeVectorStore  # lazy: Deep Lake is a heavy import

    if not corpus.is_file():
        print(f"error: corpus {corpus} not found; run: python -m my_rag --collect", file=sys.stderr)
        return 1
    embedding_function = create_embedding_function(settings)
    print(
        f"embedding {corpus} -> {store_path} (model: {settings.embedding_model}) ...",
        file=sys.stderr,
    )
    try:
        result = DeepLakeVectorStore(store_path, embedding_function, chunk_size=chunk_size).build(corpus)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # embedding endpoint unreachable, model missing, ...
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {result.chunks} vectors to {result.path}", file=sys.stderr)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_providers:
        print("\n".join(available_providers()))
        return 0

    if args.collect:
        return collect_corpus(args.output)

    load_env()
    settings = Settings.from_env(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        think=args.think,
    )

    if args.embed:
        store_path = Path(args.vector_store) if args.vector_store else DEFAULT_VECTOR_STORE
        corpus = Path(args.output) if args.output else DEFAULT_CORPUS
        return embed_corpus(corpus, store_path, args.chunk_size, settings)

    info = f"[{settings.provider}] {settings.model} @ {settings.base_url}"
    if settings.provider == "ollama":
        info += f" (think={'on' if settings.think else 'off'})"
    print(info, file=sys.stderr)
    if settings.think:
        print("note: reasoning enabled — the first answer tokens may take a while.", file=sys.stderr)

    try:
        llm = create_llm(settings)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.rag == "embeddings":
        from .vectorstore import open_vector_store  # lazy: Deep Lake is a heavy import

        store_path = Path(args.vector_store) if args.vector_store else DEFAULT_VECTOR_STORE
        try:
            retriever = open_vector_store(store_path, create_embedding_function(settings))
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"rag: embeddings (retrieval: {settings.embedding_model})", file=sys.stderr)
    else:
        try:
            retriever = create_retriever(args.rag, DB_RECORDS, method=args.method)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if retriever is not None:
            mode = args.method if args.rag in {"advanced", "modular"} else "keyword"
            print(f"rag: {args.rag} (retrieval: {mode})", file=sys.stderr)

    pipeline = RAGPipeline(llm, retriever)
    try:
        if args.show_context:
            print_context(pipeline, args.query)
        if args.stream:
            stream_response(pipeline, args.query)
        else:
            print(format_response(pipeline.answer(args.query), width=args.width))
    except Exception as exc:  # surface provider/connection errors without a traceback
        print(f"error: {exc}", file=sys.stderr)
        if "not found" in str(exc).lower():
            _print_model_hint(llm, settings)
        return 1

    return 0
