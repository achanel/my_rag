# Chapter 6 — Scaling RAG with Qdrant

*Part of [**my_rag**](../../README.md) — the root README has setup and the chapter map.*

Three-stage pipeline for scaling a RAG system on bank customer data, using the
open-source **Qdrant** vector database instead of the book's Pinecone. Qdrant's
Python client embeds a local on-disk engine, so the index needs **no API key and
no server** — point `QDRANT_URL` at a running server to get the book's managed
behaviour back.

Source notebooks (the book uses Pinecone):
- [Pipeline_1_Collecting_and_preparing_the_dataset.ipynb](https://github.com/Denis2054/RAG-Driven-Generative-AI/blob/main/Chapter06/Pipeline_1_Collecting_and_preparing_the_dataset.ipynb)
- [Pipeline_2_Scaling_a_Pinecone_Index.ipynb](https://github.com/Denis2054/RAG-Driven-Generative-AI/blob/main/Chapter06/Pipeline_2_Scaling_a_Pinecone_Index.ipynb)
- [Pipeline_3_RAG_Generative_AI.ipynb](https://github.com/Denis2054/RAG-Driven-Generative-AI/blob/main/Chapter06/Pipeline_3_RAG_Generative_AI.ipynb)

Supporting data: `data1.csv` (Bank Customer Churn, 10 000 rows).
Tools: Qdrant, Ollama / OpenAI (the book: Pinecone + OpenAI).

## Implementation

Implemented in `my_rag/scaling.py` instead of the notebooks. It mirrors the book's
three pipelines:

- **Pipeline 1 — collect and prepare.** `collect_churn()` downloads the book's
  prepared `data1.csv` (the four dropped columns — `RowNumber`, `Surname`,
  `Gender`, `Geography` — are already gone). `to_records()` turns each row into one
  `"column: value"` line, which is the unit the index stores. The EDA and the
  segmentation are `exited_overview()`, `age_overview()`, `salary_overview()` and
  a NumPy KMeans (`churn_features` → `kmeans` → `cluster_churn`) that adds the
  `class` column.
- **Pipeline 2 — scale the index.** `upsert_records()` embeds the records
  (`embed_in_batches`, batches of 1 000), duplicates each vector `dsize` times
  (10 000 × 5 = the book's 50 000), and upserts them into a Qdrant collection in
  size-bounded batches (the notebook's `batch_upsert`).
- **Pipeline 3 — RAG generation.** `CustomerRAG.answer()` embeds a target record,
  fetches the nearest `top_k` points from Qdrant, concatenates
  `EMAIL_QUERY_PROMPT + record + context`, and asks the model — with the book's
  community-manager system prompt — for a retention email.

The pipeline 1 commands run offline; pipelines 2 and 3 also run offline by default
(local Qdrant + local Ollama), and only need network when `QDRANT_URL` / a remote
embedding model is configured.

```bash
# Pipeline 1 — dataset, EDA and KMeans (offline, ~3 s on 10 000 rows)
python -m my_rag --collect --corpus churn          # -> data/raw/churn/data1.csv
python -m my_rag --churn-report                    # EDA + cluster selection + class sums

# Pipeline 2 — build the Qdrant collection (local engine, no key)
python -m my_rag --build-index                     # embed + upsert, dsize 5
python -m my_rag --build-index --dsize 1           # a smaller collection for testing

# Pipeline 3 — retrieve + generate a retention email
python -m my_rag --rag qdrant --show-context --top-k 1
python -m my_rag --rag qdrant "Customer Robertson CreditScore 632 Age 21 ..."
```

`--churn-report` reproduces the book's numbers on the real dataset:

```text
Exited=1: 2038 | Complain=1: 2044 | complain over exited: 100.29%
Age >= 50 among exited: 634 (31.11%)
EstimatedSalary >= 100000 among exited: 1045 (51.28%)
KMeans k=2 classes:
  class 0: 2039 customers, 2036 complain, 2037 exited
  class 1: 7961 customers, 8 complain, 1 exited
```

The `class` column is the book's punchline: one cluster is almost exactly the
customers who complained *and* left (2 036 of 2 039), the other holds everyone
else. Re-running `--build-index` upserts the same deterministic ids, so it
overwrites rather than duplicates.

`--rag qdrant` prints the retrieved records, then the drafted email. The default
record is the book's `Customer Henderson …` (Pipeline 3); pass your own as the
positional argument. `--show-context` prints each Qdrant hit with its id and score.

Cleanup:

```bash
rm -rf data/raw/churn data/processed/qdrant
```

Substitutions, consistent with the rest of `my_rag`:

- **Qdrant instead of Pinecone.** Pinecone is a paid managed service with an API
  key; Qdrant is open-source and its client embeds a local on-disk engine, so the
  index lives at `data/processed/qdrant` with **no key and no server**. The book's
  index name (`bank-index-50000`) becomes the collection name; the `ServerlessSpec`
  cloud/region becomes Qdrant's local path or a `QDRANT_URL`. Set `QDRANT_URL`
  (and `QDRANT_API_KEY`) to talk to a Qdrant server exactly as the book talks to
  Pinecone.
- **Local Ollama instead of GPT-4o + `text-embedding-3-small`.** The default is
  `qwen2.5:3b` + `all-minilm`; `--provider openai` uses `gpt-4o` +
  `text-embedding-3-small`. Because the embedding **dimension** depends on the
  model (384 vs 1 536), the collection size is configured
  (`MY_RAG_EMBEDDING_DIMENSION`, default per provider) instead of hard-coded to 1 536.
- **The ready-to-use `data1.csv` from the book's GitHub**, not a Kaggle download
  with credentials. It is already the dropped-columns dataset Pipeline 1 produces.
- **NumPy KMeans instead of scikit-learn.** `minmax_scale`, `kmeans` (k-means++
  over several restarts), `silhouette_score` and `davies_bouldin_score` reproduce
  the notebook's cluster selection with no scikit-learn — the same
  dependency-ladder choice as chapter 1's TF-IDF. Cluster *numbering* can differ
  from scikit-learn's, but the segmentation is the same. Silhouette is O(n²), so it
  is computed on a 1 000-point sample (the book computes all 10 000).
- **CSV via the standard library.** `csv.DictReader` replaces pandas; the EDA
  coerces the cells it needs instead of relying on pandas dtypes.
- **Qdrant and the OpenAI client are imported where used**, so the data, EDA and
  clustering commands need neither the Qdrant SDK nor a running server.

Not ported: the Colab scaffolding (Drive mount, `api_key.txt` / `pinecone.txt` key
files, Kaggle credentials and `!pip install`), the notebook's inline
`display()`/`print` cells, and the Seaborn correlation heatmap (its information is
already in the EDA).

## Tests

`tests/test_scaling.py` covers the record builder, the EDA counts, MinMax scaling,
KMeans on separable blobs, the cluster summary, dsize duplication and ids, the
query, the prompt assembly and the CLI. Qdrant runs locally, so the index
round-trip (build → query → reopen) is exercised against the **real client** with a
fake embedding, while the embeddings and the model are fakes elsewhere — no server,
key or network needed:

```bash
python -m unittest tests.test_scaling -v
```
