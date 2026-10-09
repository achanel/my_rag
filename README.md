# my_rag

A study RAG playground built while working through
[*RAG-Driven Generative AI*](https://github.com/Denis2054/RAG-Driven-Generative-AI)
(Denis Rothman, Packt). Every chapter's notebook is reimplemented as one growing,
extensible Python package — the notebooks stay as references, not as the code.

- **`my_rag/`** — the application: one pipeline the chapters extend chapter by
  chapter, with a CLI.
- **`notebooks/`** — one folder per chapter; each `README.md` maps the reference
  notebook to the code and lists every command.
- **`docs/atlas.html`** — a self-contained infographic (pipelines, retrievers,
  the chapter 5 feedback loop); open it in a browser.
- **`tests/`** — offline unit tests with fakes for Ollama, the vision model and
  the web fetch.

## Quick start

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # OPENAI_API_KEY / ACTIVELOOP_TOKEN (ch. 6 Qdrant is local, no key)
```

### 2. Models (free, local, via Ollama)

[Ollama](https://ollama.com) serves an OpenAI-compatible API, so the book's
`openai` client code runs unchanged — only the endpoint and model names differ.

```bash
ollama serve                  # or launch Ollama.app
ollama pull qwen2.5:3b        # text generation (light; qwen3:8b if you have RAM)
ollama pull all-minilm        # embeddings (384-dim, ~45 MB; bge-m3 for multilingual/Russian)
ollama pull gemma3:12b        # vision — chapter 4 only
```

In `.env`, set `OPENAI_API_KEY=ollama` and uncomment `OPENAI_BASE_URL`; the
`openai` SDK picks it up automatically. Then swap the book's model names:

| OpenAI (book) | Ollama |
|---|---|
| `gpt-4o`, `gpt-4o-mini`, … | `qwen2.5:3b` |
| `text-embedding-3-small`, `text-embedding-ada-002` | `all-minilm` (or `nomic-embed-text`) |
| vision requests (ch. 4) | `gemma3:12b` |

Caveats:

- Embedding size differs (OpenAI 1536, `all-minilm` 384, `nomic-embed-text` 768,
  `bge-m3` 1024). Re-embed data with the same model you query with, and match the
  vector collection's dimension.
- LlamaIndex (ch. 3, 7): use `llama-index-llms-ollama` /
  `llama-index-embeddings-ollama`.
- Fine-tuning (ch. 9) has no Ollama equivalent; on Apple Silicon use LoRA via
  `mlx-lm` on a small model instead.
- Sized for a 16 GB Apple Silicon machine: `qwen2.5:3b` (~2 GB) is the default;
  an 8B model needs ~6 GB and, next to PyCharm/browser, pushes a 16 GB machine
  into swap. See [ollama.com/library](https://ollama.com/library) for newer
  versions of these model families.

### 3. Run

```bash
python -m my_rag "define a rag store"        # plain generation (chapter 1)
python -m my_rag --list-providers            # registered LLM backends
python -m my_rag --model qwen2.5:0.5b "..."  # even lighter / faster
```

## Chapters

The chapters are reimplemented in order. Each one's README is the source of truth
for its commands, substitutions and cleanup — the table below is the map.

| Chapter | What the package gains |
|---|---|
| [1 · RAG overview](notebooks/01_rag_overview/README.md) | `RAGPipeline` (retrieve → augment → generate) with Naive / Advanced / Modular retrievers |
| [2 · Data, embeddings, vector store](notebooks/02_data_embeddings_generation/README.md) | Web corpus collection, embeddings, and a local Deep Lake vector store |
| [3 · Deep Lake + LlamaIndex](notebooks/03_deep_lake_llamaindex/README.md) | Four LlamaIndex index types over a drone/UAV corpus |
| [4 · Multimodal, modular RAG](notebooks/04_multimodal_modular_rag/README.md) | Text + VisDrone images + a vision model, with similarity scores |
| [5 · Adaptive RAG with human feedback](notebooks/05_adaptive_rag_human_feedback/README.md) | Panel ranking picks the generator input; a human-rating loop |
| [6 · Scaling RAG with Qdrant](notebooks/06_scaling_qdrant/README.md) | Bank-churn dataset, NumPy KMeans segmentation, and a Qdrant-backed index (embed → duplicate → upsert) |

`notebooks/07…10` are placeholders for the remaining chapters.

## How it works

Every chapter plugs into the same skeleton. Retrieval is optional, so the same
pipeline also does plain generation:

```text
query ──▶ retrieve ──▶ augment ──▶ generate ──▶ answer
          top-k records   "{query}: {context}"   llm.complete()
```

`Retriever` is a one-method protocol — `retrieve(query, top_k) -> list[str]`.
Each chapter adds a new implementation of it, or a new backend:

| Chapter | Retriever |
|---|---|
| 1 | `KeywordRetriever`, `VectorRetriever`, `IndexRetriever`, `ModularRetriever` |
| 2 | `DeepLakeVectorStore` |
| 3 | `IndexQueryEngine` (answers itself via LlamaIndex) |
| 4 | `EmbeddingIndex` (in memory) |
| 5 | none — `RAGPipeline` over the ranking-selected input |
| 6 | a Qdrant collection (`my_rag.scaling.open_qdrant_index`) |

## Command line

```bash
python -m my_rag "define a rag store"          # default: local Ollama, qwen2.5:3b
python -m my_rag --list-providers              # registered LLM backends
python -m my_rag --model qwen2.5:0.5b "..."    # even lighter / faster
python -m my_rag --think "..."                 # reasoning, for thinking models (qwen3)
python -m my_rag --no-stream --width 100 "..." # buffer + wrap instead of streaming
```

`--rag` selects the retrieval variant. Each chapter documents its own:

| `--rag` | Chapter | README |
|---|---|---|
| `naive`, `advanced`, `modular` | 1 | [link](notebooks/01_rag_overview/README.md) |
| `embeddings` | 2 | [link](notebooks/02_data_embeddings_generation/README.md) |
| `index` | 3 | [link](notebooks/03_deep_lake_llamaindex/README.md) |
| `multimodal` | 4 | [link](notebooks/04_multimodal_modular_rag/README.md) |
| `adaptive` | 5 | [link](notebooks/05_adaptive_rag_human_feedback/README.md) |
| `qdrant` | 6 | [link](notebooks/06_scaling_qdrant/README.md) |

`--provider openai` runs everything against OpenAI instead of Ollama.

## Configuration

Resolved as CLI flags → environment → per-provider defaults (`my_rag/config.py`).
Read from `.env`: `MY_RAG_PROVIDER`, `MY_RAG_MODEL`, `MY_RAG_EMBEDDING_MODEL`,
`MY_RAG_EMBEDDING_DIMENSION`, `MY_RAG_VISION_MODEL`, `MY_RAG_TEMPERATURE`,
`MY_RAG_THINK`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`. Chapter 6 adds
`MY_RAG_QDRANT_COLLECTION`, `MY_RAG_QDRANT_PATH`, and `QDRANT_URL` /
`QDRANT_API_KEY` for a remote Qdrant server.

## Extending

- **LLM providers** — subclass `my_rag.llm.LLM` and register a factory with
  `@register_provider("name")`; `ollama` and `openai` already share the
  OpenAI-compatible client.
- **Embeddings** — `create_embedding_function(settings)` builds the
  `embedding_function` the vector store calls.
- **Retrieval** — `create_retriever(variant, records, method=…)`, or pass any
  object implementing the `my_rag.pipeline.Retriever` protocol to `RAGPipeline`;
  without one the pipeline does plain generation.

## Layout

- `my_rag/` — the application package (`config`, `llm`, `embeddings`,
  `vectorstore`, `indexing`, `multimodal`, `adaptive`, `scaling`, `collection`,
  `corpus`, `pipeline`, `retrieval`, `cli`).
- `notebooks/01_rag_overview` … `notebooks/10_video_stock_production` — one per
  chapter of the reference book; see each folder's `README.md`.
- `tests/` — offline unit tests for chapters 4–6. Run with
  `python -m unittest discover -s tests`.
- `docs/atlas.html` — infographic for chapters 1–6.
- `commons/` — shared helpers (API keys, HTTP sessions) extracted as you go.
- `data/raw`, `data/processed` — input and derived datasets (git-ignored).
- `models/` — local model checkpoints / cache (git-ignored).

## Key dependencies

`deeplake`, `openai`, `transformers`, `numpy` — core RAG stack from the
reference book; chapter 3 adds `llama-index-core` with the Ollama LLM/embedding
and Deep Lake vector-store integrations. `accelerate`, `deepspeed`,
`bitsandbytes`, `neural_compressor`, `onnx` are for local Hugging Face model
training/fine-tuning and inference optimization if you extend beyond the book's
notebooks (note: `deepspeed`/`bitsandbytes` target Linux+CUDA primarily — expect
limited functionality on macOS). `pandas`, `scipy`, `beautifulsoup4`, `requests`
cover data prep and scraping. Chapter 6 adds `qdrant-client` for the vector index;
its embedded engine runs locally, so no server or key is needed (the data, EDA and
clustering commands work without it).
