# Chapter 7 — Knowledge-Graph RAG (Wikipedia + LlamaIndex)

*Part of [**my_rag**](../../README.md) — the root README has setup and the chapter map.*

Building a knowledge-graph-based RAG system over Wikipedia content, using
**LlamaIndex**'s `KnowledgeGraphIndex`: every chunk is turned into
`(subject, predicate, object)` triplets, stored as a graph, and queried through a
graph retriever whose source nodes are re-ranked by a semantic cosine.

Source notebooks:
- [Wikipedia_API.ipynb](https://github.com/Denis2054/RAG-Driven-Generative-AI/blob/main/Chapter07/Wikipedia_API.ipynb)
- [Tree_2_Graph.ipynb](https://github.com/Denis2054/RAG-Driven-Generative-AI/blob/main/Chapter07/Tree_2_Graph.ipynb)
- [Knowledge_Graph__Deep_Lake_LlamaIndex_OpenAI_RAG.ipynb](https://github.com/Denis2054/RAG-Driven-Generative-AI/blob/main/Chapter07/Knowledge_Graph__Deep_Lake_LlamaIndex_OpenAI_RAG.ipynb)

Supporting data: the book's `Marketing_urls.txt` (100 Wikipedia URLs).
Tools: LlamaIndex, Wikipedia API, Ollama (the book: Deep Lake + OpenAI).

## Implementation

Implemented in `my_rag/graph.py` instead of the notebooks. It mirrors the book's
three pipelines, plus the two notebooks it leans on:

- **Pipeline 1 — collect and prepare.** `collect_urls()` downloads the book's
  ready-made `Marketing_urls.txt`; `collect_marketing()` fetches the first
  `--limit` pages into one Markdown corpus (one `#` heading per page, chapter 2's
  layout). `crawl_summaries()` reproduces `Wikipedia_API.ipynb` from the API alone
  — walk the Marketing page's links and keep each page's intro extract — with
  `--crawl N`.
- **Pipeline 2 — the graph index.** `GraphIndex.build()` splits the corpus, runs
  `KnowledgeGraphIndex` (one LLM call per chunk to extract triplets), embeds them
  and persists to `--graph-store`; `GraphIndex.stats` reports the graph's shape
  (triplets / nodes / edges / density). `load()` reopens the store, so a later
  `--rag graph` skips triplet extraction. `Tree_2_Graph.ipynb` is ported as
  `build_networkx_graph()` / `draw_networkx_graph()` for the book's toy `pairs`
  example.
- **Pipeline 3 — query, re-rank, metrics.** `GraphRAG.answer()` queries the graph
  engine, then `rerank()` scores each source node against the question by cosine
  (the book's re-ranking) and reports `best_rank`/`best_score`. `evaluate()` runs
  the book's ten questions and aggregates the cosine sample into the notebook's
  descriptive statistics via `cosine_metrics()`; `comparison_metrics()` compares
  that mean against the book's human scores, and `write_report()` persists the run
  as JSON. `visualize()` writes the book's pyvis graph.

```bash
# Pipeline 1 — the Marketing corpus (needs network)
python -m my_rag --collect --corpus marketing --limit 15   # -> data/raw/marketing.md + marketing_urls.txt
python -m my_rag --collect --corpus marketing --crawl 15   # re-derive the list from the Wikipedia API

# Pipeline 2 — build and persist the knowledge graph (needs Ollama)
python -m my_rag --graph                                   # -> data/processed/graph_store
python -m my_rag --graph --limit 30 --max-triplets 3       # a larger graph
python -m my_rag --graph --visualize                       # also write the pyvis HTML

# Pipeline 3 — query, re-rank, metrics
python -m my_rag --rag graph --show-context "What are the 4Ps?"
python -m my_rag --rag graph --metrics                     # 10 questions -> data/processed/graph/metrics.json

# Tree_2_Graph — the book's toy example (offline, no model)
python -m my_rag --visualize --pairs --layout spiral       # -> data/processed/graph/tree_spiral.svg
```

`--rag graph` prints the answer, then `Triplets | Sources | Cosine Similarity
Score | Response Time`. With `--show-context` it prints each graph source node and
its re-ranking; with `--metrics` it runs the book's ten questions and prints the
statistics:

```text
Metrics over 10 questions (rubric human side):
  Mean: 0.52
  Median: 0.53
  ...
  Mean cosine 0.52 vs mean human 0.58 (within 0.10: 70%)
```

`--visualize --pairs` draws the book's friendship tree (solid edges for the friend
pairs, dashed otherwise) and prints each pair's label plus the root node.

Cleanup:

```bash
rm -rf data/raw/marketing.md data/raw/marketing_urls.txt data/processed/graph_store data/processed/graph
```

Substitutions, consistent with the rest of `my_rag`:

- **Local Ollama instead of GPT-4o**, through the same LLM/embedding builders as
  chapter 3 (the OpenAI LlamaIndex integrations pin `openai<3`). Chapter 7 is
  therefore `ollama`-only.
- **A local index folder instead of `hub://denis76/marketing01`.** The book writes
  both the vector store and the dataset to Activeloop; the graph index persists to
  `data/processed/graph_store` (no token/network), which also lets `--rag graph`
  reload it instead of re-extracting triplets.
- **The ready-made `Marketing_urls.txt` from the book's GitHub**, not a Colab copy
  of the 100 URLs; `--crawl` re-derives the list from the Wikipedia API.
- **Chapter 2's `clean_text` / parser** (`my_rag.collection`) instead of the
  notebook's second copy of the fetcher.
- **pyvis instead of the notebook's inline HTML cell** for the graph display, so
  the graph opens as a file in a browser.
- **The TF-IDF/numpy cosine instead of `sentence-transformers`.** Both the book's
  re-ranking and its metrics cell call `all-MiniLM-L6-v2`: the re-ranking asks the
  configured embedding model, the metrics reuse chapter 5's
  `my_rag.adaptive.tfidf_similarity`. Like chapter 3, the book's separate
  evaluation dependency is not ported.

Not ported: the Colab scaffolding (Drive mount, `api_key.txt`/`activeloop.txt` key
files, the `/etc/resolv.conf` DNS cell, the Pillow-version restart, `!pip install`
pins), the notebook's `display()`/`print` cells, and `deeplake.load(...).summary()`
(a DataFrame dump chapter 2/3 already cover) — `GraphIndex.stats` reports the
graph's own shape instead.

## Tests

`tests/test_graph.py` covers URL parsing, the Wikipedia walk, the corpus
collection (including a failed page), the `Tree_2_Graph` build and drawing, the
re-ranking cosine, the metrics and the CLI. LlamaIndex runs for real with a typed
`MockLLM` (fixed triplets) and an in-memory `MockEmbedding`, so the whole
triplet → persist → query → re-rank pipeline runs without a server, network or
Ollama:

```bash
python -m unittest tests.test_graph -v
```
