# Chapter 9 — Fine-Tuning RAG Data on Human Feedback (Local MLX)

*Part of [**my_rag**](../../README.md) — the root README has setup and the chapter map.*

Fine-tune a small model on RAG-derived data — the SciQ science-QA dataset, whose
**human-written** supports become the assistant targets — then run the fine-tuned
model and compare its answer against the base model and pure retrieval. This is
the one chapter that touches training.

The book fine-tunes OpenAI's `gpt-4o-mini` through the hosted API. That API is
paid, closed, and needs a real `OPENAI_API_KEY`, so `my_rag` implements the
**open-source, key-free** analogue instead: a local LoRA run on **Apple MLX**
(`mlx-lm`) over the exact same dataset and target format. It runs on Apple
silicon and needs no account, no network and no cost:

```bash
pip install "mlx-lm[train]"
```

Source notebook:
- [Fine_tuning_OpenAI_GPT-4o-mini.ipynb](https://github.com/Denis2054/RAG-Driven-Generative-AI/blob/main/Chapter09/Fine_tuning_OpenAI_GPT-4o-mini.ipynb)

Supporting data: `sciq` (the same 11 679-row dataset as chapter 8).
Tools: Apple MLX / `mlx-lm` (local LoRA training, fusing and an OpenAI-compatible
server).

## Implementation

Implemented in `my_rag/finetune.py` instead of the notebook, with a 1:1 step
mapping onto a local MLX run:

- **Prepare.** `build_example()` reproduces the notebook's fine-tuning item — a
  `messages` array with the system instruction, the `user` question and the
  `assistant` target `correct_answer + " Explanation: " + support` (the
  human-authored answer). `collect_finetune_dataset()` reads the SciQ cache
  chapter 8 already built (`data/raw/sciq/sciq.jsonl`, fetching it first if
  absent) and writes the JSONL.
- **Fine-tune.** `write_mlx_dataset()` splits the JSONL into `train.jsonl` /
  `valid.jsonl` (the layout `mlx_lm.lora --data` expects) and `MLXFineTuner`
  runs `python -m mlx_lm lora --train …` (LoRA, gradient checkpointing). `--fuse`
  additionally runs `python -m mlx_lm fuse` to merge the adapter into a
  standalone model.
- **Run.** `FineTunedRAG.answer()` asks the adapter through the `mlx` provider
  (the book's step-3 system prompt, `temperature=0.0`); `FineTunedRAG.compare()`
  runs the same question through the base model (no adapter) and — when a
  retriever is passed — through pure retrieval, so the three answers sit side by
  side.

```bash
# Prepare — SciQ -> fine-tuning JSONL (offline once SciQ is cached)
python -m my_rag --collect --corpus finetune                 # 500 examples
python -m my_rag --collect --corpus finetune --limit 2000
# (builds on the chapter 8 cache; fetch it first if needed:)
python -m my_rag --collect --corpus sciq

# Fine-tune — split + local LoRA training (Open source, no key; Apple silicon)
python -m my_rag --provider mlx --fine-tune                    # 600 iterations
python -m my_rag --provider mlx --fine-tune --iters 300 --fuse # fewer iters + merge

# Status — show the data split and the adapters
python -m my_rag --provider mlx --fine-tune-status

# Run the adapter (compared against the base model and pure retrieval)
python -m my_rag --provider mlx --rag finetune "…"
python -m my_rag --provider mlx --rag finetune --compare "…"
```

`--provider mlx --fine-tune` splits the JSONL and runs `mlx_lm.lora`:

```text
mlx data: data/processed/finetune/mlx (500 examples: train, valid)
MLX LoRA training done -> data/processed/finetune/adapters (600 iterations)
  run:   python -m my_rag --provider mlx --rag finetune "..."
  serve: …/python -m mlx_lm server --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --adapter-path data/processed/finetune/adapters
```

`--rag finetune --compare` prints the three answers:

```text
Base model: mlx-community/Qwen2.5-0.5B-Instruct-4bit (+ LoRA adapter)
Question: What phenomenon makes global winds blow northeast to southwest ...?

Fine-tuned adapter (no context):
Coriolis effect Explanation: ...

Base model (no adapter, no context):
...

Pure retrieval (nearest SciQ support, no model):
...
```

The adapter can also be served for any OpenAI-compatible client:

```bash
python -m mlx_lm server --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --adapter-path data/processed/finetune/adapters
```

Two artifacts are written — the fine-tuning JSONL and the MLX split/adapters:

```bash
rm -rf data/processed/finetune
```

Substitutions, consistent with the rest of `my_rag`:

- **Apple MLX LoRA instead of OpenAI's hosted fine-tuning.** There is no mature
  open-source drop-in for the hosted fine-tuning API; `mlx-lm`, HF TRL+PEFT,
  LLaMA-Factory and Unsloth are local trainers. MLX is the one that targets this
  machine (Apple silicon), needs no key, and accepts the exact ``messages``
  dataset the notebook builds.
- **The chapter 8 SciQ cache, not a second `load_dataset`.** The notebook reloads
  SciQ with the `datasets` library (pulling `pyarrow`); chapter 9 reuses
  `data/raw/sciq/sciq.jsonl`, so it needs neither the dependency nor a second
  download.
- **Plain `json` instead of `jsonlines`.** The Q&A file is one JSON object per
  line; `write_finetune_jsonl()` / `read_finetune_jsonl()` write and read it
  directly — no extra dependency.
- **A smaller default.** The book fine-tunes on all 10 481 filtered rows;
  `MY_RAG_FINETUNE_LIMIT` (default 500) keeps a local run quick and `--limit`
  scales it up.
- **A comparison, not just a run.** The notebook's step 3 only runs the
  fine-tuned model; `--compare` adds the base model (same checkpoint, no adapter)
  and the chapter-8 pure-retrieval answer so the effect of fine-tuning is visible.

Not ported: the OpenAI-specific parts of the notebook (`client.files.create`,
`fine_tuning.jobs.create`, the job-monitoring dataframe, the `ft:` model id), the
Colab scaffolding (the Drive mount, `api_key.txt`, the `!pip install` pins) and
the inline `display()` cells.

## Configuration

`MY_RAG_FINETUNE_DATA` (the JSONL path), `MY_RAG_FINETUNE_LIMIT` (SciQ rows turned
into examples, default 500), `MY_RAG_MLX_DATA` (the `train.jsonl` directory),
`MY_RAG_MLX_ADAPTERS` (the LoRA adapter directory — training output and load
path) and `MY_RAG_MLX_ITERS` (default 600) override the chapter's defaults.
`--adapter-path`, `--iters` and `--limit` override them per run. The `--compare`
retrieval side reuses `MY_RAG_SCIENCE_QA` / `MY_RAG_SCIENCE_LIMIT` and the
chapter 8 Chroma collection (embeddings still come from Ollama).

The `mlx` provider needs **Apple silicon macOS** and `pip install "mlx-lm[train]"`
(it is not in `requirements.txt`, so non-Apple installs keep working); the same
install provides the `mlx_lm.server` used for serving.

## Tests

`tests/test_finetune.py` covers the dataset build (messages format, JSONL
round-trip, SciQ cache reuse and the fetch fallback), the MLX split, the command
build, `MLXFineTuner` train/fuse (with a fake `subprocess.run`), the `MlxLLM`
provider and its chat-template/generate path, the `FineTunedRAG` answer/compare
flow, the config, and the CLI (`--collect --corpus finetune`, `--fine-tune`,
`--fine-tune-status`, `--rag finetune [--compare]`). No OpenAI call, key,
network, `mlx-lm` install or model run is needed:

```bash
python -m unittest tests.test_finetune -v
```
