# Chapter 1 — RAG Overview

*Part of [**my_rag**](../../README.md) — the root README has setup and the chapter map.*

Introductory concepts: what RAG is, why it improves LLM accuracy, and the
retrieval → augmentation → generation pipeline.

Source notebooks (Packt, *RAG-Driven Generative AI*, Denis Rothman):
- [RAG_Overview.ipynb](https://github.com/Denis2054/RAG-Driven-Generative-AI/blob/main/Chapter01/RAG_Overview.ipynb)
- [RAG_Overview_Grok.ipynb](https://github.com/Denis2054/RAG-Driven-Generative-AI/blob/main/Chapter01/RAG_Overview_Grok.ipynb)

Tools: OpenAI API.

## Implementation

Both parts are implemented in the `my_rag` package instead of a notebook.
Everything runs with Ollama (no GPT).

**Part 1 — Foundations and Basic Implementation** (generation without
augmentation): the query goes straight to the model.

```bash
python -m my_rag "define a rag store"
python -m my_rag --no-stream --width 100 "define a rag store"    # buffer + wrap instead of streaming
python -m my_rag --think "why does RAG reduce hallucinations?"   # reasoning, for thinking models (qwen3)
```

The answer streams token by token by default. Reasoning models such as `qwen3`
"think" before answering, which makes the first visible token take a while, so
thinking is **off by default** for Ollama (`think=off` in the header line); pass
`--think` to turn it on. `--no-stream` restores the wrapped book-style framing.

**Part 2 — Advanced Techniques and Evaluation** adds the three RAG variants
(`my_rag/retrieval.py`), selected with `--rag`. Retrieval is local TF-IDF /
keyword scoring (numpy only), so no embedding model is needed. Without `--rag`
(the default `none`) the pipeline does plain generation, as in part 1:

```bash
python -m my_rag --rag naive    "define a rag store"                  # Naive RAG — keyword search
python -m my_rag --rag advanced "define a rag store"                  # Advanced RAG — TF-IDF index search
python -m my_rag --rag advanced --method vector "define a rag store"  # Advanced RAG — brute-force per-pair TF-IDF
python -m my_rag --rag modular  --method indexed "define a rag store" # Modular RAG — keyword | vector | indexed
python -m my_rag --rag naive --show-context "define a rag store"      # print the retrieved record(s)
```

`--method` picks the advanced/modular retriever: `indexed` (default; one TF-IDF
matrix over the corpus), `vector` (a vectorizer fit per query/record pair), or
`keyword`. `--show-context` prints the retrieved record(s) that augment the
prompt.

The notebook's cosine-similarity metric is what the advanced/modular retrievers
use internally. Its *enhanced similarity* metric (spaCy lemmas + WordNet
synonyms) is not ported — it would add an NLP dependency and model download for
an evaluation-only score, not one of the three RAG variants.
