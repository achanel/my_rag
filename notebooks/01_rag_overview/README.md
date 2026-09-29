# Chapter 1 — RAG Overview

Introductory concepts: what RAG is, why it improves LLM accuracy, and the
retrieval → augmentation → generation pipeline.

Source notebooks (Packt, *RAG-Driven Generative AI*, Denis Rothman):
- [RAG_Overview.ipynb](https://github.com/Denis2054/RAG-Driven-Generative-AI/blob/main/Chapter01/RAG_Overview.ipynb)
- [RAG_Overview_Grok.ipynb](https://github.com/Denis2054/RAG-Driven-Generative-AI/blob/main/Chapter01/RAG_Overview_Grok.ipynb)

Tools: OpenAI API.

## Implementation

Both parts are implemented in the `my_rag` package instead of a notebook — see
the root `README.md` → *Application*. Everything runs with Ollama (no GPT).

**Part 1 — Foundations and Basic Implementation** (generation without
augmentation):

```bash
python -m my_rag "define a rag store"
```

**Part 2 — Advanced Techniques and Evaluation** adds the three RAG variants
(`my_rag/retrieval.py`), selected with `--rag`. Retrieval is local TF-IDF /
keyword scoring (numpy only), so no embedding model is needed:

```bash
python -m my_rag --rag naive    "define a rag store"                  # Naive RAG — keyword search
python -m my_rag --rag advanced "define a rag store"                  # Advanced RAG — TF-IDF index search
python -m my_rag --rag modular  --method vector "define a rag store"  # Modular RAG — keyword | vector | indexed
```

Add `--show-context` to print the retrieved record(s), and `--method indexed`
(or `vector`) to switch the advanced/modular retriever.

The notebook's cosine-similarity metric is what the advanced/modular retrievers
use internally. Its *enhanced similarity* metric (spaCy lemmas + WordNet
synonyms) is not ported — it would add an NLP dependency and model download for
an evaluation-only score, not one of the three RAG variants.
