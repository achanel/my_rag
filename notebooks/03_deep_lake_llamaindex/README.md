# Chapter 3 — Deep Lake + LlamaIndex + OpenAI RAG

An end-to-end RAG pipeline built on LlamaIndex with Deep Lake as the vector
store and OpenAI as the generator.

Source notebook:
- [Deep_Lake_LlamaIndex_OpenAI_RAG.ipynb](https://github.com/Denis2054/RAG-Driven-Generative-AI/blob/main/Chapter03/Deep_Lake_LlamaIndex_OpenAI_RAG.ipynb)

Tools: LlamaIndex, Deep Lake, OpenAI API.

## Implementation

Implemented in `my_rag/indexing.py` instead of a notebook, plus a chapter-3
corpus (the Drone/UAV URLs) in `my_rag/collection.py`. It mirrors the book's
three pipelines: collect the drone/UAV documents, build a LlamaIndex index (the
vector one persists its embeddings in Deep Lake), then run index-based semantic
search over it. The book's four index types are all ported and selectable with
`--index-type`:

```bash
python -m my_rag --collect --corpus drone          # -> data/raw/drone.md
python -m my_rag --rag index --index-type vector   "How do drones identify vehicles?"
python -m my_rag --rag index --index-type tree     "How do drones identify vehicles?"
python -m my_rag --rag index --index-type list     "How do drones identify vehicles?"
python -m my_rag --rag index --index-type keyword  "How do drones identify vehicles?"
python -m my_rag --rag index --index-type vector --show-context "How do drones identify vehicles?"
rm -rf data/raw/drone.md data/processed/index_store   # cleanup (tree/list/keyword are in-memory)
```

`--show-context` prints the retrieved source nodes before the answer;
`--top-k` sets `similarity_top_k` (default `3`, the book's value). The vector
index writes to `--vector-store` (default `data/processed/index_store`), kept
apart from chapter 2's store because the LlamaIndex integration uses a different
dataset schema.

Substitutions, consistent with the rest of `my_rag`:

- **Local Ollama instead of OpenAI.** The book calls `gpt-4o` and OpenAI
  embeddings. Chapter 3 uses `llama-index-llms-ollama` /
  `llama-index-embeddings-ollama` with the configured model, so it is scoped to
  `--provider ollama` (the default). The OpenAI LlamaIndex integrations pin
  `openai<3` and would downgrade the `openai` package used by chapters 1–2.
- **Local Deep Lake path instead of `hub://denis76/drone_v2`.** No
  `ACTIVELOOP_TOKEN`/network is required, as in chapter 2.
- **One Markdown corpus instead of one `.txt` per URL.** `load_documents`
  splits the collected file into one document per `#` heading; pointing it at a
  directory uses LlamaIndex's `SimpleDirectoryReader` (one file per document) as
  the book does.

Not ported: the book's cosine-similarity / performance-metric cell uses
`sentence-transformers` (`all-MiniLM-L6-v2`) — a heavy evaluation-only
dependency, so the index answer reports source-node scores and timings instead.
