# my_rag

RAG playground, structured after
[RAG-Driven-Generative-AI](https://github.com/Denis2054/RAG-Driven-Generative-AI)
(Denis Rothman, Packt) for study purposes — one folder per chapter/topic,
each with a `README.md` pointing at the reference notebook and the tools it
uses. The book's code is reimplemented locally in the `my_rag` package
(a single, extensible application) as you work through the chapters — not
copied from the source repo; the notebooks stay as references.

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY / ACTIVELOOP_TOKEN / PINECONE_API_KEY
```

## Free local LLM (Ollama instead of OpenAI)

[Ollama](https://ollama.com) serves an OpenAI-compatible API, so the book's
`openai` client code runs unchanged — only the endpoint and model names differ.

```bash
ollama serve                  # or launch Ollama.app
ollama pull qwen2.5:3b        # text generation (light; qwen3:8b if you have RAM)
ollama pull all-minilm        # embeddings (384-dim, ~45 MB; bge-m3 for multilingual/Russian)
ollama pull gemma3:12b        # vision, chapter 4 only
```

In `.env`, set `OPENAI_API_KEY=ollama` and uncomment `OPENAI_BASE_URL` — the
`openai` SDK picks up `OPENAI_BASE_URL` automatically. Then replace model
names in the notebooks:

| OpenAI (book) | Ollama |
|---|---|
| `gpt-4o`, `gpt-4o-mini`, … | `qwen2.5:3b` (see the size note below) |
| `text-embedding-3-small`, `text-embedding-ada-002` | `all-minilm` (or `nomic-embed-text`) |
| vision requests (ch. 4) | `gemma3:12b` |

Caveats:
- Embedding size differs (OpenAI 1536, `all-minilm` 384, `nomic-embed-text` 768,
  `bge-m3` 1024). The book's prebuilt Deep Lake datasets can't be queried with
  local embeddings — re-embed your data with the same model you query with, and
  create Pinecone indexes with the matching dimension.
- LlamaIndex (ch. 3, 7): use `llama-index-llms-ollama` /
  `llama-index-embeddings-ollama`.
- Fine-tuning (ch. 9) has no Ollama equivalent; on Apple Silicon use LoRA via
  `mlx-lm` on a small model instead.
- Sized for a 16 GB Apple Silicon machine: `qwen2.5:3b` (~2 GB) is the default
  here; an 8B model needs ~6 GB and, alongside PyCharm/browser, pushes a 16 GB
  machine into swap. Check
  [ollama.com/library](https://ollama.com/library) for newer versions of these
  model families.

## Application

Chapter work lives in the `my_rag` package (a growing, extensible application)
rather than in notebooks.

```bash
python -m my_rag "define a rag store"       # default: local Ollama, qwen2.5:3b
python -m my_rag --list-providers           # registered LLM backends
python -m my_rag --model qwen2.5:0.5b "..."  # even lighter / faster
python -m my_rag --think "..."              # reasoning, for thinking models (qwen3)
python -m my_rag --no-stream --width 100 "…" # buffer + wrap instead of streaming
```

### RAG variants (chapter 1, part 2)

The reference notebook introduces Naive, Advanced, and Modular RAG. Select a
variant with `--rag` and (for the modular/advanced retrievers) a `--method`;
`--show-context` prints the retrieved record(s) that augment the prompt. All
retrieval runs locally with numpy — no embedding model or extra service:

```bash
python -m my_rag --rag naive    "define a rag store"                    # Naive RAG: keyword search
python -m my_rag --rag advanced "define a rag store"                    # Advanced RAG: TF-IDF index search
python -m my_rag --rag advanced --method vector "define a rag store"    # Advanced RAG: brute-force vector search
python -m my_rag --rag modular  --method indexed "define a rag store"   # Modular RAG: pick keyword/vector/indexed
python -m my_rag --rag naive --show-context "define a rag store"        # print the retrieved context too
```

Without `--rag` (the default `none`) the pipeline does plain generation, with no
retrieved context, as in part 1.

The answer streams token by token by default. Reasoning models such as `qwen3`
"think" before answering, which makes the first visible token take a while, so
thinking is **off by default** for Ollama (`think=off` in the header line);
pass `--think` to turn it on. `--no-stream` restores the wrapped book-style
framing.

### Chapter 2 — data collection, embeddings + vector store

Part 1 collects the corpus; part 2 embeds it into a local **Deep Lake**
(Activeloop) vector store; part 3 reuses the pipeline with that store as the
retriever. Embeddings come from the same OpenAI-compatible endpoint as the LLM —
by default the free local `all-minilm` (384-dim) via Ollama instead of OpenAI's
`text-embedding-3-small`:

```bash
python -m my_rag --collect                 # part 1: -> data/raw/llm.md
python -m my_rag --collect --output my_corpus.md          # custom corpus path
python -m my_rag --embed                    # part 2: -> data/processed/vector_store
python -m my_rag --embed --output my_corpus.md --vector-store /tmp/vs --chunk-size 500
python -m my_rag --rag embeddings "Tell me about space exploration on the Moon and Mars."
python -m my_rag --rag embeddings --show-context "…"   # print the retrieved chunk(s)
```

`--embed` reads `--output` (default `data/raw/llm.md`) and writes to
`--vector-store` (default `data/processed/vector_store`), chunking by
`--chunk-size` characters (default 1000) as the book does. The store is local, so
no `ACTIVELOOP_TOKEN` is needed; re-running `--embed` rebuilds it. A store built
with one embedding model can only be queried with the same model/dimension. See
`notebooks/02_data_embeddings_generation/README.md` for the notebook mapping.

### Chapter 3 — LlamaIndex index-based semantic search

Builds on LlamaIndex: collect the book's drone/UAV corpus, index it, and query it
through four index types (the vector one persists embeddings in a local Deep Lake
store, as chapter 2 does):

```bash
python -m my_rag --collect --corpus drone                      # -> data/raw/drone.md
python -m my_rag --rag index --index-type vector "How do drones identify vehicles?"
python -m my_rag --rag index --index-type tree   "How do drones identify vehicles?"
python -m my_rag --rag index --index-type list   "How do drones identify vehicles?"
python -m my_rag --rag index --index-type keyword "How do drones identify vehicles?"
python -m my_rag --rag index --index-type vector --show-context "How do drones identify vehicles?"
```

`--rag index` reads `--output` (default `data/raw/drone.md`) and, for the vector
type, writes to `--vector-store` (default `data/processed/index_store`);
`--top-k` is the book's `similarity_top_k` (default 3). Chapter 3 uses the Ollama
LlamaIndex integrations, so it is scoped to `--provider ollama` (the default) —
the OpenAI integrations pin `openai<3`, which would downgrade the version used
elsewhere. See `notebooks/03_deep_lake_llamaindex/README.md` for the mapping.

Configuration resolves in this order: CLI flags → environment → per-provider
defaults (`my_rag/config.py`). `MY_RAG_PROVIDER`, `MY_RAG_MODEL`,
`MY_RAG_EMBEDDING_MODEL`, `MY_RAG_TEMPERATURE`, `MY_RAG_THINK`, `OPENAI_BASE_URL`,
`OPENAI_API_KEY` are read from `.env`.

Extension points:

- **LLM providers** — subclass `my_rag.llm.LLM` and register a factory with
  `@register_provider("name")`; `ollama` and `openai` already share the
  OpenAI-compatible client.
- **Embeddings** — `my_rag.embeddings.OpenAICompatibleEmbedding` is the
  `embedding_function` the vector store calls; `create_embedding_function(settings)`
  builds it from configuration.
- **Retrieval** — `my_rag.retrieval` ships the chapter 1, part 2 retrievers
  (`KeywordRetriever`, `VectorRetriever`, `IndexRetriever`, `ModularRetriever`);
  `create_retriever(variant, records, method=…)` builds one by name. The chapter 2
  `DeepLakeVectorStore` also implements `Retriever`. Pass any object implementing
  the `my_rag.pipeline.Retriever` protocol to `RAGPipeline` to add your own;
  without one the pipeline does plain generation (part 1).

## Layout

- `my_rag/` — the application package (`config`, `llm`, `embeddings`,
  `vectorstore`, `indexing`, `collection`, `corpus`, `pipeline`, `retrieval`,
  `cli`).
- `notebooks/01_rag_overview` … `notebooks/10_video_stock_production` — one
  per chapter of the reference book, see each folder's `README.md`.
- `commons/` — shared helpers (API keys, HTTP sessions) extracted as you go.
- `data/raw`, `data/processed` — input and derived datasets (git-ignored).
- `models/` — local model checkpoints / cache (git-ignored).

## Key dependencies

`deeplake`, `openai`, `transformers`, `numpy` — core RAG stack from the
reference book; chapter 3 adds `llama-index-core` with the Ollama LLM/embedding
and Deep Lake vector-store integrations. `accelerate`, `deepspeed`,
`bitsandbytes`, `neural_compressor`,
`onnx` are for local Hugging Face model training/fine-tuning and inference
optimization if you extend beyond the book's notebooks (note: `deepspeed`/
`bitsandbytes` target Linux+CUDA primarily — expect limited functionality on
macOS). `pandas`, `scipy`, `beautifulsoup4`, `requests` cover data prep and
scraping.
