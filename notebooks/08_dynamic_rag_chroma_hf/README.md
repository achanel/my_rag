# Chapter 8 — Dynamic RAG (Chroma + Hugging Face Llama)

*Part of [**my_rag**](../../README.md) — the root README has setup and the chapter map.*

A *dynamic* (temporary, per-session) RAG over the SciQ science-QA dataset:
build a throw-away **Chroma** collection, retrieve the support for a question,
score the retrieval with a semantic cosine, and summarize it with a local
**Hugging Face Llama** — then delete the collection. The book's scenario is one
daily-meeting session: nothing persists between runs.

Source notebook:
- [Dynamic_RAG_with_Chroma_and_Hugging_Face.ipynb](https://github.com/Denis2054/RAG-Driven-Generative-AI/blob/main/Chapter08/Dynamic_RAG_with_Chroma_and_Hugging_Face.ipynb)

Supporting data: `sciq` (11 679 train rows of science questions with a support and
a correct answer).
Tools: Chroma, Hugging Face `transformers`, Ollama / OpenAI (the book: Chroma's
default embedding + `meta-llama/Llama-2-7b-chat-hf`).

## Implementation

Implemented in `my_rag/dynamic.py` instead of the notebook. It mirrors the book's
one-session pipeline:

- **Prepare.** `fetch_science_qa()` pages the Hugging Face datasets server for
  `allenai/sciq`, keeps rows with a support *and* a correct answer, drops the
  three `distractor*` columns, and builds the completion
  (`correct_answer + " because " + support`). `collect_science_qa()` caches the
  result as JSONL (`data/raw/sciq/sciq.jsonl`), so later runs are offline;
  `read_science_qa()` reads it back.
- **Index.** `open_science_collection()` opens the book's **ephemeral** Chroma
  collection (`chromadb.Client()`, in memory) and `ChromaIndex.add_rows()` embeds
  the completions through the configured embedding model and stores them with the
  book's `{"type": "completion"}` metadata.
- **Query / evaluate.** `ChromaIndex.query()` returns the nearest completion;
  `DynamicRAG.evaluate()` reproduces the notebook's retrieval loop — query by
  question, score the retrieved completion against the known one
  (`text_similarities`, a semantic cosine), and report the share above
  `0.7` (`SIMILARITY_THRESHOLD`).
- **Generate.** `DynamicRAG.answer()` concatenates the book's instruction
  (`SUMMARY_INSTRUCTION`) with the retrieved support and asks the model for a
  beginner summary. `ChromaIndex.delete()` then drops the collection.

```bash
# Prepare — cache SciQ (needs the network once)
python -m my_rag --collect --corpus sciq            # -> data/raw/sciq/sciq.jsonl (500 rows)
python -m my_rag --collect --corpus sciq --limit 2000

# One dynamic session — temporary Chroma collection, query, summarize, delete
python -m my_rag --rag dynamic "Millions of years ago, plants used energy from the sun to form what?"
python -m my_rag --rag dynamic --show-context "What gas do plants absorb?"
python -m my_rag --rag dynamic --evaluate --limit 50  # + the book's accuracy loop (cosine > 0.7)
python -m my_rag --rag dynamic --top-k 3 --temperature 0.5 "..."

# Local Hugging Face model (small ungated default Qwen2.5-0.5B; needs the weights)
python -m my_rag --provider hf --rag dynamic "..."
# The book's exact (gated) model, only once you have access to it:
python -m my_rag --provider hf --model meta-llama/Llama-2-7b-chat-hf --rag dynamic "..."
```

`--rag dynamic` prints the answer, then `Matches | Response Time`, the session
time, and — with `--evaluate` — the retrieval accuracy:

```text
rag: dynamic (collection: sciq_supports6, 500 completions, top_k=1)
...
Matches: 1 | Response Time: 0.42 seconds

Retrieval accuracy over 50 questions (cosine > 0.7): 86.00% (43/50)
Session time: 12.80 seconds
```

The collection is created, used and deleted inside the run, so there is no Chroma
directory to clean up — only the cached dataset:

```bash
rm -rf data/raw/sciq
```

Substitutions, consistent with the rest of `my_rag`:

- **Local Ollama by default instead of the Hugging Face Llama pipeline.** The book
  loads `meta-llama/Llama-2-7b-chat-hf` in `transformers` (a 7B model, a GPU, a HF
  token); the default here is the same local `qwen2.5:3b` as the other chapters.
  `--provider hf` selects `my_rag.llm.HuggingFaceLLM`, which runs the book's
  `transformers.pipeline("text-generation", …)` (the pipeline is built lazily, so
  merely selecting the provider loads nothing; the book's `device_map="auto"` is
  dropped — it segfaults transformers 5.x on Apple-silicon macOS, so the pipeline
  picks the default device, MPS/CPU). Its default
  model is the **small ungated local** `Qwen/Qwen2.5-0.5B-Instruct` (~1 GB, runs
  on CPU, no HF token and no license gate): the book's Llama-2 repo is *gated*, so
  `--model meta-llama/Llama-2-7b-chat-hf` is opt-in after accepting its terms. If
  that gated model is inaccessible, generation **falls back** to the ungated
  default (with a warning) instead of failing the whole run; a direct
  load failure without a fallback is reported with an actionable hint
  (`hf auth login` / the token file).
- **Ollama `all-minilm` embeddings instead of Chroma's default embedding.** The
  book lets Chroma pick `all-MiniLM-L6-v2` (downloaded as ONNX on first use); here
  the embedding function is chapter 2's `all-minilm` (the same all-MiniLM-L6-v2
  family), so one embedding stack covers the whole repo. It is injected into the
  collection, which also lets the tests pass a fake.
- **The datasets-server REST API instead of the `datasets` library.** The book
  calls `load_dataset("sciq")` (pulls `pyarrow`); `my_rag` fetches the same rows
  from the public datasets server over `requests` and caches them — no `datasets`
  or `pyarrow` dependency.
- **A semantic embedding cosine instead of spaCy `en_core_web_md`.** The book's
  `simple_text_similarity` loads a spaCy model to score the retrieval;
  `text_similarities` reuses the configured embedding model and chapter 2's
  `cosine_similarity`. Like chapter 3, the book's separate evaluation dependency
  is not ported.
- **A smaller default.** The book embeds all 11 679 supports in every session;
  each embedding here is one HTTP round-trip to Ollama, so `MY_RAG_SCIENCE_LIMIT`
  (default 500) keeps a session runnable, and `--limit` scales it up.

Not ported: the Colab scaffolding (the Drive mount, `hf_token.txt` / the HF secret,
the `!pip install` pins and `spacy download`), the notebook's inline
`display()`/`print` cells, and the `collection.get(include=['embeddings'])` dump
(a display step, not part of the pipeline).

## Configuration

`MY_RAG_SCIENCE_QA` (the cached JSONL path), `MY_RAG_SCIENCE_LIMIT` (rows embedded
by default) and `MY_RAG_CHROMA_COLLECTION` (default `sciq_supports6`) override the
chapter's defaults. `--provider hf` also reads `MY_RAG_MODEL` for the Hugging Face
model id.

The default `hf` model needs **no token**. A Hugging Face access token is only
read for gated models from `~/.config/hugging_face/token`, mirroring the book's
"read the token file, set `HF_TOKEN`" cell: `load_hf_token()` exports it into
`HF_TOKEN` just before the `transformers` pipeline is built; an existing
`HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` wins, and `MY_RAG_HF_TOKEN_FILE` overrides
the path. The token is never logged.

## Tests

`tests/test_dynamic.py` covers the row preparation, the paged fetch (with a fake
session), the JSONL round-trip, the similarity cosine, the prompt, the
`DynamicRAG` answer/evaluate flow, the config, and the CLI. Chroma runs in memory,
so the index round-trip (add → query → delete) is exercised against the **real
client** with a fake embedding; the LLM and the network are fakes — no server,
model download or key needed:

```bash
python -m unittest tests.test_dynamic -v
```
