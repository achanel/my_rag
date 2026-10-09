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
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import Optional

from .collection import DEFAULT_DRONE_OUTPUT
from .collection import DEFAULT_OUTPUT as DEFAULT_CORPUS
from .collection import DRONE_URLS, WIKI_URLS, collect
from .config import (
    CHUNK_SIZE,
    DEFAULT_CHURN_CSV,
    DEFAULT_IMAGE_DIR,
    DEFAULT_INDEX_STORE,
    DEFAULT_QDRANT_COLLECTION,
    DEFAULT_VECTOR_STORE,
    Settings,
    load_env,
)
from .corpus import DB_RECORDS
from .embeddings import create_embedding_function
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
        f"for --rag adaptive: {DEFAULT_ADAPTIVE_QUERY!r}; for --rag qdrant: a customer record)",
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
        choices=[
            "none", "naive", "advanced", "modular", "embeddings", "index", "multimodal",
            "adaptive", "qdrant",
        ],
        default="none",
        help="RAG variant: none (plain generation), naive (keyword), advanced (TF-IDF vector/index), "
        "modular (selectable method), embeddings (vector store search), index (LlamaIndex index, ch. 3), "
        "multimodal (text + drone images + vision model, ch. 4), adaptive (ranking-driven RAG with "
        "human feedback, ch. 5), qdrant (Qdrant collection over bank-churn records, ch. 6)",
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
        default=3,
        help="similarity_top_k for --rag index (default: 3, the book's value)",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="print the retrieved record(s) before the answer",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="fetch a corpus and exit (chapter 2 Wikipedia, chapter 3 drone, or chapter 4 VisDrone images)",
    )
    parser.add_argument(
        "--corpus",
        choices=["wiki", "drone", "visdrone", "churn"],
        default="wiki",
        help="corpus for --collect: wiki (ch. 2), drone (ch. 3 text), visdrone (ch. 4 images), "
        "or churn (ch. 6 bank-customer dataset; default: wiki)",
    )
    parser.add_argument(
        "--output",
        help=f"path for --collect / --embed / --rag index / --rag multimodal / --churn-report / --build-index "
        f"(default: {DEFAULT_CORPUS}; drone: {DEFAULT_DRONE_OUTPUT}; churn: {DEFAULT_CHURN_CSV})",
    )
    parser.add_argument(
        "--image-dir",
        help=f"folder of VisDrone images and annotations for --collect --corpus visdrone or --rag multimodal "
        f"(default: {DEFAULT_IMAGE_DIR})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_IMAGE_LIMIT,
        help=f"number of VisDrone samples for --collect --corpus visdrone (default: {DEFAULT_IMAGE_LIMIT})",
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
        help=f"Qdrant collection for --build-index / --rag qdrant "
        f"(env MY_RAG_QDRANT_COLLECTION; default: {DEFAULT_QDRANT_COLLECTION})",
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
    from .indexing import IndexQueryEngine, load_documents  # lazy: LlamaIndex is heavy

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
    from .scaling import DEFAULT_QUERY_RECORD, CustomerRAG, open_qdrant_index

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
        else:
            args.query = DEFAULT_QUERY

    if args.list_providers:
        print("\n".join(available_providers()))
        return 0

    if args.collect:
        if args.corpus == "visdrone":
            return collect_images(Path(args.image_dir) if args.image_dir else DEFAULT_IMAGE_DIR, args.limit)
        if args.corpus == "drone":
            return collect_corpus(Path(args.output) if args.output else DEFAULT_DRONE_OUTPUT, DRONE_URLS)
        if args.corpus == "churn":
            return collect_churn_dataset(Path(args.output) if args.output else DEFAULT_CHURN_CSV)
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

    if args.embed:
        store_path = Path(args.vector_store) if args.vector_store else DEFAULT_VECTOR_STORE
        corpus = Path(args.output) if args.output else DEFAULT_CORPUS
        return embed_corpus(corpus, store_path, args.chunk_size, settings)

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
