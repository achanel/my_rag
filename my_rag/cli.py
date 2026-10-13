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

Chapter 3 adds LlamaIndex index-based semantic search (Deep Lake + Ollama)::

    python -m my_rag --collect --corpus drone
    python -m my_rag --rag index --index-type vector "How do drones identify vehicles?"
    python -m my_rag --rag index --index-type tree --show-context "..."

Chapter 4 answers from the drone text and drone images together (needs a vision
model, default ``gemma3:12b``)::

    python -m my_rag --collect --corpus visdrone --limit 20
    python -m my_rag --rag multimodal "How do drones identify a truck?"
    python -m my_rag --rag multimodal --show-context "How do drones identify a truck?"

Chapter 5 adapts the generator input to a panel ranking and runs the human
feedback loop (rankings 1-4 need no network; 5 fetches a Wikipedia page)::

    python -m my_rag --rag adaptive --ranking 5 "What is an LLM?"
    python -m my_rag --rag adaptive --ranking 3 "What is an LLM?"
    python -m my_rag --rag adaptive --ranking 1 --rating 3 "What is an LLM?"

Chapter 6 scales the index onto Qdrant over the bank-customer-churn dataset
(local and key-free by default)::

    python -m my_rag --collect --corpus churn
    python -m my_rag --build-index
    python -m my_rag --rag qdrant "Customer Henderson CreditScore 599 ..."

Chapter 7 builds a knowledge graph over the Marketing citation list and indexes
it with LlamaIndex (Ollama only; the graph build is one LLM call per chunk, so
``--limit`` picks how many pages it walks)::

    python -m my_rag --collect --corpus marketing --limit 15
    python -m my_rag --graph --visualize
    python -m my_rag --rag graph --show-context "What are the 4Ps?"
    python -m my_rag --rag graph --metrics "What are the 4Ps?"
    python -m my_rag --visualize --pairs        # the book's Tree_2_Graph example

Chapter 8 builds a temporary (ephemeral) Chroma collection over the SciQ dataset
and summarizes the retrieved support (Ollama by default; ``--provider hf`` runs
the book's local Hugging Face Llama)::

    python -m my_rag --collect --corpus sciq
    python -m my_rag --rag dynamic "Millions of years ago, plants used energy from the sun to form what?"
    python -m my_rag --rag dynamic --evaluate --limit 50

Chapter 9 fine-tunes a small model locally on the SciQ data with Apple MLX (LoRA;
the open-source, key-free path — the book's OpenAI backend is not used)::

    python -m my_rag --collect --corpus finetune       # SciQ -> fine-tuning JSONL (offline)
    python -m my_rag --provider mlx --fine-tune         # split + local MLX LoRA train
    python -m my_rag --provider mlx --fine-tune-status  # the data split + adapters
    python -m my_rag --provider mlx --rag finetune "..."            # run the adapter
    python -m my_rag --provider mlx --rag finetune --compare "..."  # + base model + retrieval
    # serve the adapter for any OpenAI-compatible client:
    python -m mlx_lm server --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
        --adapter-path data/processed/finetune/adapters
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

from .collection import DEFAULT_DRONE_OUTPUT
from .collection import DEFAULT_OUTPUT as DEFAULT_CORPUS
from .collection import DRONE_URLS, WIKI_URLS, collect
from .config import (
    CHUNK_SIZE,
    DEFAULT_CHURN_CSV,
    DEFAULT_FINETUNE_DATA,
    DEFAULT_FINETUNE_LIMIT,
    DEFAULT_GRAPH_HTML_DIR,
    DEFAULT_GRAPH_LIMIT,
    DEFAULT_GRAPH_STORE,
    DEFAULT_IMAGE_DIR,
    DEFAULT_INDEX_STORE,
    DEFAULT_MLX_ADAPTERS,
    DEFAULT_MLX_ITERS,
    DEFAULT_QDRANT_COLLECTION,
    DEFAULT_SCIENCE_LIMIT,
    DEFAULT_SCIENCE_QA,
    DEFAULT_VECTOR_STORE,
    Settings,
    load_env,
)
from .corpus import DB_RECORDS
from .embeddings import create_embedding_function
from .graph import (
    DEFAULT_MARKETING_OUTPUT,
    DEFAULT_TOP_K as DEFAULT_GRAPH_TOP_K,
    METRIC_RECORDS,
    GraphIndex,
    GraphRAG,
    TOY_FRIENDS,
    TOY_LAYOUTS,
    TOY_PAIRS,
    build_networkx_graph,
    collect_marketing,
    draw_networkx_graph,
    load_documents as load_graph_documents,
    write_report,
)
from .llm import LLM, available_providers, create_llm
from .multimodal import DEFAULT_IMAGE_LIMIT, IMAGE_TOP_K, TEXT_TOP_K
from .pipeline import RAGPipeline
from .retrieval import create_retriever
from .scaling import DEFAULT_DSIZE, DEFAULT_QUERY_RECORD

DEFAULT_QUERY = "define a rag store"

#: The book's chapter 3 question over the drone/UAV corpus.
DEFAULT_INDEX_QUERY = "How do drones identify vehicles?"

#: The book's chapter 4 question: answered from the drone text and the VisDrone images.
DEFAULT_MULTIMODAL_QUERY = "How do drones identify a truck?"

#: The book's chapter 5 question, adapted by the panel's ranking.
DEFAULT_ADAPTIVE_QUERY = "What is an LLM?"

#: The book's chapter 7 question over the Marketing knowledge graph.
DEFAULT_GRAPH_QUERY = "What is the primary goal of marketing for the consumer market?"

#: Where chapter 7's metrics run writes its numbers (git-ignored).
DEFAULT_GRAPH_REPORT = Path("data/processed/graph/metrics.json")

#: Where the chapter 7 pyvis graph is written by default.
DEFAULT_GRAPH_HTML = "Knowledge_graph"

#: The book's chapter 8 question over the SciQ supports.
DEFAULT_DYNAMIC_QUERY = "Millions of years ago, plants used energy from the sun to form what?"

#: The book's chapter 9 question to the fine-tuned model.
DEFAULT_FINETUNE_QUERY = (
    "What phenomenon makes global winds blow northeast to southwest or the reverse in the "
    "northern hemisphere and northwest to southeast or the reverse in the southern hemisphere?"
)


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
        default=None,
        help=f"question to send (default: {DEFAULT_QUERY!r}; "
        f"for --rag index: {DEFAULT_INDEX_QUERY!r}; for --rag multimodal: {DEFAULT_MULTIMODAL_QUERY!r}; "
        f"for --rag adaptive: {DEFAULT_ADAPTIVE_QUERY!r}; for --rag qdrant: a customer record; "
        f"for --rag graph: {DEFAULT_GRAPH_QUERY!r})",
    )
    parser.add_argument("--provider", help="LLM provider (env MY_RAG_PROVIDER; default: ollama; "
        "openai, hf — local Hugging Face — or mlx — local Apple MLX fine-tuning/generation)")
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
        choices=[
            "none", "naive", "advanced", "modular", "embeddings", "index", "multimodal",
            "adaptive", "qdrant", "graph", "dynamic", "finetune",
        ],
        default="none",
        help="RAG variant: none (plain generation), naive (keyword), advanced (TF-IDF vector/index), "
        "modular (selectable method), embeddings (vector store search), index (LlamaIndex index, ch. 3), "
        "multimodal (text + drone images + vision model, ch. 4), adaptive (ranking-driven RAG with "
        "human feedback, ch. 5), qdrant (Qdrant collection over bank-churn records, ch. 6), "
        "graph (LlamaIndex knowledge graph, ch. 7), dynamic (temporary Chroma collection over SciQ, ch. 8), "
        "finetune (run the locally fine-tuned MLX adapter, ch. 9)",
    )
    parser.add_argument(
        "--method",
        choices=["keyword", "vector", "indexed"],
        default="indexed",
        help="retrieval method for --rag advanced|modular (default: indexed)",
    )
    parser.add_argument(
        "--index-type",
        choices=["vector", "tree", "list", "keyword"],
        default="vector",
        help="LlamaIndex index for --rag index (default: vector)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="similarity_top_k for --rag index|graph (default: 3, the book's value)",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="print the retrieved record(s) before the answer",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="fetch a corpus and exit (chapter 2 Wikipedia, chapter 3 drone, chapter 4 VisDrone images, "
        "chapter 7 marketing)",
    )
    parser.add_argument(
        "--corpus",
        choices=["wiki", "drone", "visdrone", "churn", "marketing", "sciq", "finetune"],
        default="wiki",
        help="corpus for --collect: wiki (ch. 2), drone (ch. 3 text), visdrone (ch. 4 images), "
        "churn (ch. 6 bank-customer dataset), marketing (ch. 7 knowledge-graph corpus), "
        "sciq (ch. 8 SciQ science-QA dataset), or finetune (ch. 9 chat fine-tuning JSONL "
        "built from SciQ; default: wiki)",
    )
    parser.add_argument(
        "--output",
        help=f"path for --collect / --embed / --rag index / --rag multimodal / --churn-report / "
        f"--build-index / --graph / --rag dynamic / --fine-tune (default: {DEFAULT_CORPUS}; drone: "
        f"{DEFAULT_DRONE_OUTPUT}; churn: {DEFAULT_CHURN_CSV}; marketing: {DEFAULT_MARKETING_OUTPUT}; "
        f"sciq: {DEFAULT_SCIENCE_QA}; finetune: {DEFAULT_FINETUNE_DATA})",
    )
    parser.add_argument(
        "--image-dir",
        help=f"folder of VisDrone images and annotations for --collect --corpus visdrone or --rag multimodal "
        f"(default: {DEFAULT_IMAGE_DIR})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=f"number of samples/pages: VisDrone images for --collect --corpus visdrone "
        f"(default: {DEFAULT_IMAGE_LIMIT}), Marketing pages for --collect --corpus marketing and "
        f"--graph (default: {DEFAULT_GRAPH_LIMIT}), SciQ rows for --collect --corpus sciq and "
        f"--rag dynamic (default: {DEFAULT_SCIENCE_LIMIT}), fine-tuning examples for "
        f"--collect --corpus finetune (default: {DEFAULT_FINETUNE_LIMIT})",
    )
    parser.add_argument(
        "--vision-model",
        help="vision model for --rag multimodal (env MY_RAG_VISION_MODEL; default: provider default, "
        "gemma3:12b for ollama)",
    )
    parser.add_argument(
        "--ranking",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=5,
        help="panel ranking for --rag adaptive: 1-2 no RAG, 3-4 human feedback, 5 retrieved document "
        "(default: 5)",
    )
    parser.add_argument(
        "--num-words",
        type=int,
        default=100,
        help="words pulled from the retrieved page for --rag adaptive (default: 100, the book's value)",
    )
    parser.add_argument(
        "--rating",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help="human rating (1-5) to record in the --rag adaptive feedback loop (default: none)",
    )
    parser.add_argument(
        "--churn-report",
        action="store_true",
        help="print the churn EDA and KMeans segmentation, then exit (chapter 6, part 1)",
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="embed the churn dataset and upsert it into Qdrant, then exit (chapter 6, part 2)",
    )
    parser.add_argument(
        "--dsize",
        type=int,
        default=DEFAULT_DSIZE,
        help=f"duplication factor for --build-index (default: {DEFAULT_DSIZE}, the book's value)",
    )
    parser.add_argument(
        "--collection",
        help=f"collection name for --build-index / --rag qdrant (Qdrant) or --rag dynamic (Chroma) "
        f"(env MY_RAG_QDRANT_COLLECTION / MY_RAG_CHROMA_COLLECTION; default: {DEFAULT_QDRANT_COLLECTION} / "
        f"sciq_supports6)",
    )
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        help="vector dimension for the Qdrant collection (env MY_RAG_EMBEDDING_DIMENSION; default: "
        "provider embedding dimension, 384 for Ollama all-minilm)",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="embed the collected corpus into the vector store and exit (chapter 2, part 2)",
    )
    parser.add_argument(
        "--vector-store",
        help=f"vector store path for --embed / --rag embeddings, or LlamaIndex Deep Lake store "
        f"for --rag index (default: {DEFAULT_VECTOR_STORE}; index: {DEFAULT_INDEX_STORE})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"character chunk size for --embed / --graph (default: {CHUNK_SIZE})",
    )

    graph = parser.add_argument_group("chapter 7 — knowledge graph (Marketing + LlamaIndex)")
    graph.add_argument(
        "--graph",
        action="store_true",
        help="build the knowledge graph from the corpus, persist it, and exit (ch. 7, part 2)",
    )
    graph.add_argument(
        "--graph-store",
        help=f"folder the knowledge graph is persisted to / loaded from "
        f"(env MY_RAG_GRAPH_STORE; default: {DEFAULT_GRAPH_STORE})",
    )
    graph.add_argument(
        "--visualize",
        action="store_true",
        help="write the graph as a pyvis HTML file (with --pairs: the book's Tree_2_Graph example)",
    )
    graph.add_argument(
        "--pairs",
        action="store_true",
        help="draw the book's Tree_2_Graph toy graph (pairs + friend/dashed edges) instead of the "
        "knowledge graph",
    )
    graph.add_argument(
        "--layout",
        choices=list(TOY_LAYOUTS),
        default="spring",
        help="networkx layout for --visualize --pairs (default: spring)",
    )
    graph.add_argument(
        "--graph-html",
        help=f"output HTML/SVG path for --visualize (default: "
        f"{DEFAULT_GRAPH_HTML_DIR}/{DEFAULT_GRAPH_HTML}.html; --pairs: tree_<layout>.svg)",
    )
    graph.add_argument(
        "--max-nodes",
        type=int,
        default=150,
        help="nodes drawn in the pyvis graph, largest component first (default: 150; 0 = all)",
    )
    graph.add_argument(
        "--max-triplets",
        type=int,
        default=2,
        help="max_triplets_per_chunk for the graph index (default: 2, the book's value)",
    )
    graph.add_argument(
        "--metrics",
        action="store_true",
        help="run the book's metrics sample over the graph index and write the report (ch. 7, part 3)",
    )
    graph.add_argument(
        "--report",
        help=f"where --metrics writes its JSON report (default: {DEFAULT_GRAPH_REPORT})",
    )
    graph.add_argument(
        "--crawl",
        type=int,
        metavar="N",
        help="collect the corpus by crawling N of the Marketing page's links instead of the "
        "book's URL list (needs the network, no index)",
    )

    dynamic = parser.add_argument_group("chapter 8 — dynamic RAG (SciQ + Chroma)")
    dynamic.add_argument(
        "--evaluate",
        action="store_true",
        help="with --rag dynamic, run the book's retrieval-accuracy loop (cosine > 0.7) as well",
    )
    dynamic.add_argument(
        "--keep-collection",
        action="store_true",
        help="with --rag dynamic, keep the Chroma collection instead of deleting it at the end "
        "(the book's ephemeral session deletes it)",
    )

    finetune = parser.add_argument_group("chapter 9 — local fine-tuning (SciQ + Apple MLX)")
    finetune.add_argument(
        "--fine-tune",
        action="store_true",
        help="split the fine-tuning JSONL and run a local Apple MLX LoRA training, then "
        "exit (ch. 9; needs --provider mlx and mlx-lm on Apple silicon — no key)",
    )
    finetune.add_argument(
        "--fine-tune-status",
        action="store_true",
        help="show the local MLX data split and LoRA adapters (ch. 9)",
    )
    finetune.add_argument(
        "--compare",
        action="store_true",
        help="with --rag finetune, also run the base model (no adapter) and the chapter-8 "
        "pure-retrieval answer (needs the SciQ cache) and print all three",
    )
    finetune.add_argument(
        "--adapter-path",
        help=f"LoRA adapter directory for --provider mlx (env MY_RAG_MLX_ADAPTERS; "
        f"training output and load path; default: {DEFAULT_MLX_ADAPTERS})",
    )
    finetune.add_argument(
        "--iters",
        type=int,
        help=f"MLX LoRA training iterations for --provider mlx --fine-tune "
        f"(env MY_RAG_MLX_ITERS; default: {DEFAULT_MLX_ITERS})",
    )
    finetune.add_argument(
        "--fuse",
        action="store_true",
        help="with --provider mlx --fine-tune, also merge the LoRA adapter into a "
        "standalone model (mlx_lm.fuse) so it can be served without an adapter",
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


def collect_corpus(path: Path, urls: list[str]) -> int:
    """Fetch ``urls`` and write the corpus to ``path``."""
    print(f"collecting {len(urls)} articles ...", file=sys.stderr)
    try:
        result = collect(urls=urls, output=path, progress=lambda line: print(line, file=sys.stderr))
    except OSError as exc:  # e.g. unwritable output path
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {result.path} ({len(result.collected)}/{len(urls)} articles)", file=sys.stderr)
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


def run_index_query(args, settings: Settings) -> int:
    """Build a LlamaIndex index over the corpus and answer ``args.query`` (ch. 3)."""
    from .indexing import DEFAULT_TOP_K, IndexQueryEngine, load_documents  # lazy: LlamaIndex

    args.top_k = args.top_k or DEFAULT_TOP_K
    corpus = Path(args.output) if args.output else DEFAULT_DRONE_OUTPUT
    store_path = Path(args.vector_store) if args.vector_store else DEFAULT_INDEX_STORE
    print(f"rag: index ({args.index_type}, top_k={args.top_k})", file=sys.stderr)
    print(f"loading {corpus} ...", file=sys.stderr)
    try:
        documents = load_documents(corpus)
        engine = IndexQueryEngine(
            args.index_type, documents, settings, vector_store_path=store_path, top_k=args.top_k
        )
        if args.show_context:
            print("Retrieved context:")
            print("---------------")
            for text in engine.retrieve(args.query):
                print(textwrap.fill(text, width=args.width))
                print()
            print("---------------")
        answer = engine.query(args.query)
    except Exception as exc:  # missing corpus, Ollama unreachable, bad model, ...
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"indexed {len(documents)} documents in {answer.elapsed:.2f}s", file=sys.stderr)
    print(format_response(answer.response, width=args.width))
    return 0


def collect_churn_dataset(path: Path) -> int:
    """Download the chapter 6 bank-customer-churn dataset to ``path`` (ch. 6, part 1)."""
    from .scaling import collect_churn

    print(f"downloading the bank-customer churn dataset -> {path} ...", file=sys.stderr)
    try:
        result = collect_churn(path, progress=lambda line: print(line, file=sys.stderr))
    except Exception as exc:  # no network, unwritable path, ...
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {result}", file=sys.stderr)
    return 0


def collect_images(image_dir: Path, limit: int) -> int:
    """Save the first ``limit`` VisDrone samples to ``image_dir`` (ch. 4)."""
    from .multimodal import collect_visdrone  # lazy: Deep Lake is a heavy import

    print(f"collecting {limit} VisDrone samples from Deep Lake Hub ...", file=sys.stderr)
    try:
        samples = collect_visdrone(image_dir, limit, progress=lambda line: print(line, file=sys.stderr))
    except Exception as exc:  # no network to Activeloop, unwritable folder, ...
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {len(samples)} images and annotations to {image_dir}", file=sys.stderr)
    return 0


def print_multimodal_answer(answer, settings: Settings, width: int) -> None:
    """Print the text module's response, the vision module's response, and the scores (ch. 4)."""
    print(format_response(answer.text_response, width=width))
    print()
    print(f"Vision ({settings.vision_model}) on {answer.sample.path.name}, object: {answer.label}")
    print("---------------")
    print(textwrap.fill(answer.vision_response, width=width))
    print("---------------")
    print(f"Boxed image: {answer.boxed_image}")
    print(
        f"Scores: text {answer.text_score:.3f} | vision {answer.vision_score:.3f} "
        f"| modular {answer.modular_score:.3f}"
    )


def run_multimodal_query(args, settings: Settings) -> int:
    """Answer ``args.query`` from the drone text and the drone images together (ch. 4)."""
    from openai import NotFoundError

    from .multimodal import MultimodalRAG, load_samples, load_text_chunks

    corpus = Path(args.output) if args.output else DEFAULT_DRONE_OUTPUT
    image_dir = Path(args.image_dir) if args.image_dir else DEFAULT_IMAGE_DIR
    print(
        f"rag: multimodal (text top_k={TEXT_TOP_K}, image top_k={IMAGE_TOP_K}, "
        f"vision: {settings.vision_model})",
        file=sys.stderr,
    )
    try:
        rag = MultimodalRAG(settings, load_samples(image_dir), load_text_chunks(corpus))
        answer = rag.answer(args.query)
    except NotFoundError as exc:  # a configured model is not installed on the server
        print(f"error: {exc}", file=sys.stderr)
        if settings.provider == "ollama":
            print(
                f"hint: pull the models with: ollama pull {settings.vision_model} "
                f"&& ollama pull {settings.embedding_model}",
                file=sys.stderr,
            )
        return 1
    except Exception as exc:  # missing data, Ollama unreachable, ...
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.show_context:
        print("Retrieved text context:")
        print("---------------")
        for text in answer.text_context:
            print(textwrap.fill(text, width=args.width))
            print()
        print(f"Retrieved image: {answer.sample.path} (labels: {answer.sample.label_text()})")
        print("---------------")
    print_multimodal_answer(answer, settings, width=args.width)
    return 0


def churn_report(csv_path: Path) -> int:
    """Print the chapter 6 EDA and KMeans segmentation (ch. 6, part 1)."""
    from .scaling import (
        CLUSTER_FEATURES,
        age_overview,
        cluster_churn,
        exited_overview,
        load_churn,
        salary_overview,
        score_clusters,
    )

    try:
        rows = load_churn(csv_path)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Loaded {len(rows)} customers from {csv_path}")
    exited = exited_overview(rows)
    age = age_overview(rows)
    salary = salary_overview(rows)
    print(
        f"Exited=1: {exited.sum_exited} | Complain=1: {exited.sum_complain} | "
        f"complain over exited: {exited.complain_over_exited:.2f}%"
    )
    print(f"Age >= {age.age:g} among exited: {age.aged_and_over} ({age.aged_over_among_exited:.2f}%)")
    print(
        f"EstimatedSalary >= {salary.threshold:g} among exited: {salary.salary_over} "
        f"({salary.salary_over_among_exited:.2f}%)"
    )
    print(f"Cluster selection on {', '.join(CLUSTER_FEATURES)}:")
    for score in score_clusters(rows):
        print(
            f"  k={score.n_clusters}: silhouette {score.silhouette:.4f}, "
            f"Davies-Bouldin {score.davies_bouldin:.4f}"
        )
    _, summary = cluster_churn(rows)
    print("KMeans k=2 classes:")
    for cluster in sorted(summary.counts):
        print(
            f"  class {cluster}: {summary.counts[cluster]} customers, "
            f"{summary.complaints[cluster]} complain, {summary.exited[cluster]} exited"
        )
    return 0


def build_vector_index(args, settings: Settings) -> int:
    """Embed the churn dataset and upsert it into Qdrant (ch. 6, part 2)."""
    from .scaling import load_churn, open_qdrant_index, to_records, upsert_records

    csv_path = Path(args.output) if args.output else DEFAULT_CHURN_CSV
    collection = args.collection or settings.qdrant_collection
    print(f"building Qdrant collection {collection} (dsize={args.dsize}) from {csv_path} ...", file=sys.stderr)
    index = None
    try:
        records = to_records(load_churn(csv_path))
        index = open_qdrant_index(
            settings, collection=args.collection, dimension=args.embedding_dimension
        )
        result = upsert_records(
            records,
            create_embedding_function(settings),
            index,
            collection=index.collection,
            dsize=args.dsize,
            progress=lambda line: print(line, file=sys.stderr),
        )
    except Exception as exc:  # missing dataset, embedding endpoint down, ...
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if index is not None:
            index.close()
    print(
        f"upserted {result.vectors} vectors ({result.records} records × {result.dsize}, "
        f"{result.dimension}-dim) into {result.collection}",
        file=sys.stderr,
    )
    return 0


def run_qdrant_query(args, settings: Settings) -> int:
    """Draft a retention email from the record's nearest Qdrant neighbours (ch. 6)."""
    from .scaling import DEFAULT_QUERY_RECORD, DEFAULT_TOP_K, CustomerRAG, open_qdrant_index

    args.top_k = args.top_k or DEFAULT_TOP_K
    record = args.query or DEFAULT_QUERY_RECORD
    index = None
    try:
        index = open_qdrant_index(
            settings, collection=args.collection, dimension=args.embedding_dimension, create=False
        )
        print(f"rag: qdrant (collection: {index.collection}, top_k={args.top_k})", file=sys.stderr)
        answer = CustomerRAG(settings, index, top_k=args.top_k).answer(record)
    except Exception as exc:  # missing collection, embedding endpoint down, ...
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if index is not None:
            index.close()
    if not answer.matches:
        print("note: the collection is empty; run: python -m my_rag --build-index", file=sys.stderr)
    if args.show_context:
        print("Retrieved context:")
        print("---------------")
        for match in answer.matches:
            print(f"[{match.id}] score {match.score:.4f}")
            print(textwrap.fill(match.text, width=args.width))
            print()
        print("---------------")
    print(format_response(answer.response, width=args.width))
    print()
    print(f"Matches: {len(answer.matches)} | Response Time: {answer.elapsed:.2f} seconds")
    return 0


def run_adaptive_query(args, settings: Settings) -> int:
    """Answer ``args.query`` through the ranking-adaptive RAG with human feedback (ch. 5)."""
    from .adaptive import AdaptiveRAG, NoMatchError, evaluate, strategy_for_ranking

    print(
        f"rag: adaptive (ranking {args.ranking} → {strategy_for_ranking(args.ranking)}, "
        f"doc words: {args.num_words})",
        file=sys.stderr,
    )
    try:
        rag = AdaptiveRAG(settings)
        result = rag.answer(args.query, ranking=args.ranking, num_words=args.num_words)
    except NoMatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "no relevant keywords found. Please enter a query related to "
            "'LLM', 'LLMs', or 'Prompt Engineering'.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # network down, Ollama unreachable, ...
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(format_response(result.response, width=args.width))
    print()
    print(f"Strategy: {result.strategy}" + (f" ({result.source})" if result.source else ""))
    print(f"Response Time: {result.elapsed:.2f} seconds")

    evaluation = evaluate(result, rating=args.rating)
    print(f"Cosine Similarity Score: {evaluation.similarity:.3f}")
    if evaluation.rating is not None:
        print("Evaluator Score:", evaluation.rating)
        print("Rankings      :", evaluation.state.counter)
        print("Score history : ", evaluation.mean_score)
        if evaluation.expert_review_required:
            print("Human expert evaluation is required for the feedback loop.")
    return 0


def collect_marketing_corpus(args, limit: int) -> int:
    """Fetch the chapter 7 Marketing corpus, from the book's URL list or by crawling (ch. 7)."""
    from .graph import DEFAULT_URLS_OUTPUT, crawl_summaries

    output = Path(args.output) if args.output else DEFAULT_MARKETING_OUTPUT
    if args.crawl:
        print(f"crawling {args.crawl} links from the Marketing page ...", file=sys.stderr)
        try:
            articles = crawl_summaries("Marketing", args.crawl)
        except Exception as exc:  # no network, unknown page, ...
            print(f"error: {exc}", file=sys.stderr)
            return 1
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n\n".join(article.to_markdown() for article in articles) + "\n", encoding="utf-8")
        print(f"wrote {output} ({len(articles)} pages via the Wikipedia API)", file=sys.stderr)
        return 0

    print(f"collecting the first {limit} Marketing pages -> {output} ...", file=sys.stderr)
    try:
        path, collected, failed = collect_marketing(
            output, limit=limit, progress=lambda line: print(line, file=sys.stderr)
        )
    except Exception as exc:  # no network to GitHub, unwritable path, ...
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {path} ({len(collected)}/{limit} pages, {len(failed)} skipped)", file=sys.stderr)
    return 0


def _graph_index(args, settings: Settings) -> GraphIndex:
    """Chapter 7's graph index, pointed at ``--graph-store``/``--output``."""
    return GraphIndex(
        settings,
        store_path=Path(args.graph_store) if args.graph_store else settings.graph_store,
        max_triplets_per_chunk=args.max_triplets,
        chunk_size=args.chunk_size,
    )


def build_knowledge_graph(args, settings: Settings) -> int:
    """Build the chapter 7 knowledge graph from the corpus and persist it (ch. 7, part 2)."""
    corpus = Path(args.output) if args.output else DEFAULT_MARKETING_OUTPUT
    limit = args.limit or settings.graph_limit
    index = _graph_index(args, settings)
    print(
        f"graph: {corpus} (first {limit} pages, max_triplets_per_chunk={args.max_triplets}) "
        f"-> {index.store_path}",
        file=sys.stderr,
    )
    try:
        documents = load_graph_documents(corpus, limit=limit)
        print(f"extracting triplets from {len(documents)} documents ...", file=sys.stderr)
        build = index.build(
            documents,
            progress=lambda done, total: print(f"  chunk {done}/{total}", file=sys.stderr),
        )
    except Exception as exc:  # missing corpus, Ollama unreachable, ...
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"built {build.stats} from {build.chunks} chunks in {build.elapsed:.1f}s; "
        f"wrote {index.store_path}",
        file=sys.stderr,
    )
    return 0


def run_graph_query(args, settings: Settings) -> int:
    """Answer ``args.query`` from the knowledge graph, re-ranked (ch. 7, part 3)."""
    index = _graph_index(args, settings)
    top_k = args.top_k or DEFAULT_GRAPH_TOP_K
    print(f"rag: graph (top_k={top_k}, store: {index.store_path})", file=sys.stderr)
    try:
        if not index.exists:
            _build_graph_inline(index, args, settings)
        if args.metrics:
            return _graph_metrics(args, settings, index)
        answer = GraphRAG(index, top_k=top_k).answer(args.query)
    except Exception as exc:  # missing graph/corpus, Ollama unreachable, ...
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.show_context:
        print("Retrieved context:")
        print("---------------")
        for position, source in enumerate(answer.sources, start=1):
            print(f"[{position}] graph score {source.score}\n{textwrap.fill(source.text, width=args.width)}\n")
        print("---------------")
        if answer.ranking.best_rank is not None:
            print(
                f"Re-ranking (cosine vs the question): best rank {answer.ranking.best_rank}, "
                f"score {answer.ranking.best_score:.3f}"
            )
            print("---------------")
    print(format_response(answer.response, width=args.width))
    print()
    print(
        f"Triplets: {answer.triplets} | Sources: {len(answer.sources)} | "
        f"Cosine Similarity Score: {answer.similarity:.3f} | Response Time: {answer.elapsed:.2f} seconds"
    )
    return 0


def _build_graph_inline(index: GraphIndex, args, settings: Settings) -> None:
    """Build and persist the graph when a graph query finds no store yet."""
    corpus = Path(args.output) if args.output else DEFAULT_MARKETING_OUTPUT
    print(f"no graph at {index.store_path}; building from {corpus} ...", file=sys.stderr)
    build = index.build(
        load_graph_documents(corpus, limit=args.limit or settings.graph_limit),
        progress=lambda done, total: print(f"  chunk {done}/{total}", file=sys.stderr),
    )
    print(f"built {build.stats} from {build.chunks} chunks in {build.elapsed:.1f}s", file=sys.stderr)


def _graph_metrics(args, settings: Settings, index: GraphIndex) -> int:
    """Run the book's metrics sample and write the JSON report (ch. 7, part 3)."""
    report = Path(args.report) if args.report else DEFAULT_GRAPH_REPORT
    try:
        answers, metrics, comparison = GraphRAG(index, top_k=args.top_k or DEFAULT_GRAPH_TOP_K).evaluate(
            limit=METRIC_RECORDS,
            progress=lambda done, total, question: print(
                f"  [{done}/{total}] {question}", file=sys.stderr
            ),
        )
        path = write_report(report, answers=answers, metrics=metrics, comparison=comparison)
    except Exception as exc:  # empty graph, Ollama unreachable, ...
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"\nMetrics over {len(answers)} questions ({comparison.source} human side):")
    for line in metrics.lines():
        print(f"  {line}")
    print(
        f"  Mean cosine {comparison.mean_cosine:.3f} vs mean human {comparison.mean_human:.3f} "
        f"(within 0.10: {comparison.agreement:.0%})"
    )
    print(f"wrote {path}", file=sys.stderr)
    return 0


def visualize_graph(args) -> int:
    """Draw the book's graphs: the Tree_2_Graph example, or the knowledge graph (ch. 7)."""
    if args.pairs:
        output = Path(args.graph_html) if args.graph_html else Path(f"data/processed/graph/tree_{args.layout}.svg")
        graph, root = build_networkx_graph(TOY_PAIRS)
        try:
            path = draw_networkx_graph(
                graph,
                output=output,
                layout=args.layout,
                friends=TOY_FRIENDS,
                title=f"Visual Representation of the Tree ({args.layout} layout)",
            )
        except Exception as exc:  # bad layout, unwritable path, ...
            print(f"error: {exc}", file=sys.stderr)
            return 1
        for first, second in TOY_PAIRS:
            print(f"Pair ({first}, {second}): {'friend' if (first, second) in TOY_FRIENDS else 'not friend'}")
        print(f"root: {root} | wrote {path}")
        return 0

    index = _graph_index(args, _settings_for_visualize(args))
    output = Path(args.graph_html) if args.graph_html else Path(DEFAULT_GRAPH_HTML_DIR) / f"{DEFAULT_GRAPH_HTML}.html"
    print(f"visualizing the knowledge graph at {index.store_path} -> {output}", file=sys.stderr)
    try:
        path = index.visualize(
            output, max_nodes=args.max_nodes, progress=lambda line: print(line, file=sys.stderr)
        )
    except Exception as exc:  # missing graph, unwritable path, ...
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {path}")
    return 0


def _settings_for_visualize(args) -> Settings:
    """Settings for a drawing: no model needed, so nothing is contacted."""
    return Settings.from_env(provider=args.provider, model=args.model, base_url=args.base_url)


def collect_science_qa_corpus(args, limit: int) -> int:
    """Download the chapter 8 SciQ dataset and cache it as JSONL (ch. 8)."""
    from .dynamic import collect_science_qa

    output = Path(args.output) if args.output else DEFAULT_SCIENCE_QA
    print(f"downloading {limit} SciQ rows -> {output} ...", file=sys.stderr)
    try:
        path = collect_science_qa(
            output, limit=limit, progress=lambda line: print(line, file=sys.stderr)
        )
    except Exception as exc:  # no network, unwritable path, bad response, ...
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {path}", file=sys.stderr)
    return 0


def run_dynamic_query(args, settings: Settings) -> int:
    """Build a temporary Chroma collection over SciQ and summarize a retrieval (ch. 8)."""
    from .dynamic import DEFAULT_TOP_K, DynamicRAG, open_science_collection, read_science_qa

    top_k = args.top_k or DEFAULT_TOP_K
    limit = args.limit or settings.science_limit
    dataset = Path(args.output) if args.output else settings.science_qa
    collection = args.collection or settings.chroma_collection
    session_start = time.perf_counter()
    index = None
    answer = None
    report = None
    try:
        rows = read_science_qa(dataset)[:limit]
        index = open_science_collection(settings, collection=collection)
        print(
            f"rag: dynamic (collection: {index.name}, {len(rows)} completions, top_k={top_k})",
            file=sys.stderr,
        )
        index.add_rows(rows)
        rag = DynamicRAG(settings, index, top_k=top_k)
        answer = rag.answer(args.query)
        if args.evaluate:
            report = rag.evaluate(
                rows,
                progress=lambda done, total, question: print(
                    f"  [{done}/{total}] {question}", file=sys.stderr
                ),
            )
    except Exception as exc:  # missing dataset, embedding endpoint down, ...
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if index is not None and not args.keep_collection:
            try:
                index.delete()
            except Exception:  # cleanup must not mask the run's real result
                pass

    if args.show_context:
        print("Retrieved context:")
        print("---------------")
        for match in answer.matches:
            print(f"[{match.id}] distance {match.distance:.4f}")
            print(textwrap.fill(match.text, width=args.width))
            print()
        print("---------------")
    print(format_response(answer.response, width=args.width))
    print()
    print(f"Matches: {len(answer.matches)} | Response Time: {answer.elapsed:.2f} seconds")
    if report is not None:
        print(
            f"\nRetrieval accuracy over {report.documents} questions "
            f"(cosine > {report.threshold:.1f}): {report.accuracy:.2%} "
            f"({report.correct}/{report.documents})"
        )
    print(f"Session time: {time.perf_counter() - session_start:.2f} seconds")
    return 0


def collect_finetune_corpus(args, limit: int) -> int:
    """Build the chapter 9 fine-tuning JSONL from the SciQ dataset (ch. 9, part 1)."""
    from .finetune import collect_finetune_dataset

    output = Path(args.output) if args.output else DEFAULT_FINETUNE_DATA
    science = Path(os.environ.get("MY_RAG_SCIENCE_QA") or DEFAULT_SCIENCE_QA)
    print(f"preparing {limit} fine-tuning examples from {science} -> {output} ...", file=sys.stderr)
    try:
        path = collect_finetune_dataset(
            output,
            science_qa=science,
            limit=limit,
            progress=lambda line: print(line, file=sys.stderr),
        )
    except Exception as exc:  # no SciQ cache + no network, unwritable path, ...
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {path}", file=sys.stderr)
    return 0


def _require_mlx(settings: Settings) -> Optional[int]:
    """Chapter 9 is a local Apple MLX path; reject other providers with a hint."""
    if settings.provider == "mlx":
        return None
    print(
        "error: chapter 9 fine-tuning is local Apple MLX; run with --provider mlx "
        "(needs mlx-lm on Apple silicon: pip install \"mlx-lm[train]\")",
        file=sys.stderr,
    )
    return 1


def run_fine_tune(args, settings: Settings) -> int:
    """Build the MLX dataset split and run a local LoRA training (ch. 9)."""
    from .finetune import collect_finetune_dataset, open_mlx_fine_tuner, read_finetune_jsonl

    error = _require_mlx(settings)
    if error is not None:
        return error

    dataset = Path(args.output) if args.output else settings.finetune_data
    limit = args.limit or settings.finetune_limit
    tuner = open_mlx_fine_tuner(settings)
    if args.iters:
        tuner.iters = args.iters
    if args.adapter_path:
        tuner.adapters = Path(args.adapter_path)
    try:
        if not dataset.is_file():
            collect_finetune_dataset(
                dataset,
                science_qa=settings.science_qa,
                limit=limit,
                progress=lambda line: print(line, file=sys.stderr),
            )
        examples = read_finetune_jsonl(dataset)
        written = tuner.prepare(examples)
        print(f"mlx data: {tuner.data_dir} ({len(examples)} examples: {', '.join(sorted(written))})")
        run = tuner.train()
    except Exception as exc:  # mlx-lm missing, no weights, no SciQ cache, ...
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"MLX LoRA training done -> {run.adapters} ({run.iters} iterations)")
    print(f"  run:   python -m my_rag --provider mlx --rag finetune \"...\"")
    print(f"  serve: {' '.join(run.serve_command())}")
    if args.fuse:
        fused = run.adapters.parent / "fused_model"
        try:
            tuner.fuse(fused)
        except Exception as exc:
            print(f"error: mlx_lm.fuse failed: {exc}", file=sys.stderr)
            return 1
        print(f"  fused: python -m mlx_lm generate --model {fused} --prompt ...")
    return 0


def run_fine_tune_status(args, settings: Settings) -> int:
    """Show the local MLX dataset split and LoRA adapters (ch. 9)."""
    error = _require_mlx(settings)
    if error is not None:
        return error

    data_dir = settings.mlx_data
    adapters = Path(args.adapter_path) if args.adapter_path else settings.mlx_adapters
    model = settings.model
    print(f"mlx data dir: {data_dir}")
    found = False
    for name in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        path = data_dir / name
        if path.is_file():
            lines = sum(1 for line in path.open(encoding="utf-8") if line.strip())
            print(f"  {name}: {lines} examples")
            found = True
    if not found:
        print("  (no split yet; build one with: python -m my_rag --provider mlx --fine-tune)")
    if adapters.is_dir():
        names = sorted(item.name for item in adapters.iterdir())
        print(f"adapters: {adapters} ({', '.join(names) or 'empty'})")
        print(f"run it with: python -m my_rag --provider mlx --rag finetune \"...\"")
    else:
        print(f"no adapters at {adapters}; train with: python -m my_rag --provider mlx --fine-tune")
    return 0


def run_finetune_query(args, settings: Settings) -> int:
    """Run the locally fine-tuned MLX adapter, optionally vs. base + retrieval (ch. 9)."""
    from .dynamic import DEFAULT_TOP_K, DynamicRAG, open_science_collection, read_science_qa
    from .finetune import FineTunedRAG

    error = _require_mlx(settings)
    if error is not None:
        return error

    adapters = Path(args.adapter_path) if args.adapter_path else settings.mlx_adapters
    if not adapters.exists():
        print(
            f"error: no MLX adapter at {adapters}; train one with: "
            "python -m my_rag --provider mlx --fine-tune",
            file=sys.stderr,
        )
        return 1
    settings = replace(settings, mlx_adapters=adapters)

    index = None
    retriever = None
    if args.compare:
        try:
            rows = read_science_qa(settings.science_qa)[: (args.limit or settings.science_limit)]
            index = open_science_collection(
                settings, collection=args.collection or settings.chroma_collection
            )
            index.add_rows(rows)
            retriever = DynamicRAG(settings, index, top_k=DEFAULT_TOP_K)
        except Exception as exc:  # no SciQ cache, embedding endpoint down, ...
            print(
                f"note: no SciQ retrieval for the comparison ({exc}); "
                "comparing against the base model only",
                file=sys.stderr,
            )
            index = None
            retriever = None

    try:
        rag = FineTunedRAG(settings, retriever=retriever)
        if args.compare:
            comparison = rag.compare(args.query)
        else:
            answer = rag.answer(args.query)
    except Exception as exc:  # mlx-lm missing, no weights, ...
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if index is not None:
            try:
                index.delete()
            except Exception:  # cleanup must not mask the run's real result
                pass

    if args.compare:
        print(f"Base model: {settings.model} (+ LoRA adapter)")
        print(f"Question: {comparison.question}\n")
        print("Fine-tuned adapter (no context):")
        print(textwrap.fill(comparison.fine_tuned, width=args.width))
        print("\nBase model (no adapter, no context):")
        print(textwrap.fill(comparison.base, width=args.width))
        if comparison.retrieved:
            print("\nPure retrieval (nearest SciQ support, no model):")
            print(textwrap.fill(comparison.retrieved, width=args.width))
    else:
        print(format_response(answer.response, width=args.width))
        print(f"\nModel: {answer.model} | Response Time: {answer.elapsed:.2f} seconds")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.query is None:
        if args.rag == "index":
            args.query = DEFAULT_INDEX_QUERY
        elif args.rag == "multimodal":
            args.query = DEFAULT_MULTIMODAL_QUERY
        elif args.rag == "adaptive":
            args.query = DEFAULT_ADAPTIVE_QUERY
        elif args.rag == "qdrant":
            args.query = DEFAULT_QUERY_RECORD
        elif args.rag == "graph":
            args.query = DEFAULT_GRAPH_QUERY
        elif args.rag == "dynamic":
            args.query = DEFAULT_DYNAMIC_QUERY
        elif args.rag == "finetune":
            args.query = DEFAULT_FINETUNE_QUERY
        else:
            args.query = DEFAULT_QUERY

    if args.pairs and not args.visualize:
        args.visualize = True

    if args.visualize:
        load_env()
        return visualize_graph(args)

    if args.list_providers:
        print("\n".join(available_providers()))
        return 0

    if args.collect:
        if args.corpus == "visdrone":
            limit = args.limit or DEFAULT_IMAGE_LIMIT
            return collect_images(Path(args.image_dir) if args.image_dir else DEFAULT_IMAGE_DIR, limit)
        if args.corpus == "drone":
            return collect_corpus(Path(args.output) if args.output else DEFAULT_DRONE_OUTPUT, DRONE_URLS)
        if args.corpus == "churn":
            return collect_churn_dataset(Path(args.output) if args.output else DEFAULT_CHURN_CSV)
        if args.corpus == "marketing":
            limit = args.limit or int(os.environ.get("MY_RAG_GRAPH_LIMIT") or DEFAULT_GRAPH_LIMIT)
            return collect_marketing_corpus(args, limit)
        if args.corpus == "sciq":
            limit = args.limit or int(os.environ.get("MY_RAG_SCIENCE_LIMIT") or DEFAULT_SCIENCE_LIMIT)
            return collect_science_qa_corpus(args, limit)
        if args.corpus == "finetune":
            limit = args.limit or int(os.environ.get("MY_RAG_FINETUNE_LIMIT") or DEFAULT_FINETUNE_LIMIT)
            return collect_finetune_corpus(args, limit)
        return collect_corpus(Path(args.output) if args.output else DEFAULT_CORPUS, WIKI_URLS)

    if args.churn_report:
        return churn_report(Path(args.output) if args.output else DEFAULT_CHURN_CSV)

    load_env()
    settings = Settings.from_env(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        think=args.think,
        vision_model=args.vision_model,
    )

    if args.build_index:
        return build_vector_index(args, settings)

    if args.fine_tune:
        return run_fine_tune(args, settings)

    if args.fine_tune_status:
        return run_fine_tune_status(args, settings)

    if args.embed:
        store_path = Path(args.vector_store) if args.vector_store else DEFAULT_VECTOR_STORE
        corpus = Path(args.output) if args.output else DEFAULT_CORPUS
        return embed_corpus(corpus, store_path, args.chunk_size, settings)

    if args.graph:
        return build_knowledge_graph(args, settings)

    if args.rag == "graph":
        return run_graph_query(args, settings)

    if args.rag == "dynamic":
        return run_dynamic_query(args, settings)

    if args.rag == "finetune":
        return run_finetune_query(args, settings)

    if args.rag == "index":
        return run_index_query(args, settings)

    if args.rag == "multimodal":
        return run_multimodal_query(args, settings)

    if args.rag == "adaptive":
        return run_adaptive_query(args, settings)

    if args.rag == "qdrant":
        return run_qdrant_query(args, settings)

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
