"""Scaling RAG with a Qdrant vector index over bank-churn data (chapter 6).

Reimplementation of the book's three chapter 6 notebooks, which together scale a
RAG system onto a managed vector index:

- **Pipeline 1 — collect and prepare the dataset.** Download the *Bank Customer
  Churn* dataset (``radheshyamkollipara/bank-customer-churn``), drop the columns
  the book drops, and run its EDA: complaints vs. exits, age vs. exits, salary vs.
  exits, and a KMeans segmentation that adds a ``class`` column (the book's
  "market deep-learning segment analysis").
- **Pipeline 2 — scale the vector index.** Each customer row becomes one text
  record (``"column: value"`` pairs), the records are embedded, each embedding is
  duplicated ``dsize`` times, and the vectors are upserted into a Qdrant
  collection in size-bounded batches.
- **Pipeline 3 — RAG generation.** A target customer record is embedded, the
  nearest records are fetched from Qdrant, concatenated into an augmented prompt,
  and the model drafts a customer-retention email.

Substitutions, consistent with the rest of ``my_rag``:

- **Qdrant instead of Pinecone.** Pinecone is a paid managed service; Qdrant is
  open-source and its Python client embeds a local on-disk engine, so the index
  needs **no API key and no server** by default (state under
  ``data/processed/qdrant``). Point ``QDRANT_URL`` at a running server to keep the
  book's managed-index behaviour without changing any other code.
- **Free local Ollama by default** (``qwen2.5:3b`` + ``all-minilm``) instead of
  GPT-4o + ``text-embedding-3-small``. The embeddings are **1536-dim** for OpenAI's
  model but only **384-dim** for ``all-minilm``, so the collection size is
  configured (``MY_RAG_EMBEDDING_DIMENSION``) rather than hard-coded to 1536.
- **The ready-to-use ``data1.csv`` from the book's GitHub** instead of a Kaggle
  download with credentials; it already has ``RowNumber``, ``Surname``, ``Gender``
  and ``Geography`` dropped, which is exactly the book's prepared dataset.
- **NumPy KMeans instead of scikit-learn.** ``kmeans()`` (k-means++ with several
  restarts), ``minmax_scale()``, ``silhouette_score()`` and
  ``davies_bouldin_score()`` reproduce the notebook's cluster selection without
  pulling scikit-learn — the same dependency-ladder choice as chapter 1's TF-IDF.
- **Qdrant and the OpenAI client are imported where they are used**, so light
  commands (data collection, EDA, clustering) need neither the Qdrant SDK nor a
  running server.

Not ported: the Colab scaffolding (Drive mount, ``api_key.txt`` / ``pinecone.txt``
key files, Kaggle credentials), the notebook's inline ``display()``/``print``
cells, and its Seaborn correlation heatmap (the numbers it would show are already
in the EDA).
"""
from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import requests

from .config import DEFAULT_CHURN_CSV, Settings
from .embeddings import EmbeddingFunction, create_embedding_function
from .llm import LLM, create_llm

#: The book's ready-to-use dataset on GitHub (its Kaggle source needs credentials).
CHURN_URL = (
    "https://raw.githubusercontent.com/Denis2054/"
    "RAG-Driven-Generative-AI/main/Chapter06/data1.csv"
)

#: Columns the book drops right after loading the Kaggle CSV. Kept here for
#: documentation; ``data1.csv`` on GitHub is already the dropped dataset.
DROP_COLUMNS: tuple[str, ...] = ("RowNumber", "Surname", "Gender", "Geography")

#: The six standardized features the book clusters on (`data2[features]`).
CLUSTER_FEATURES: tuple[str, ...] = (
    "CreditScore",
    "Age",
    "EstimatedSalary",
    "Exited",
    "Complain",
    "Point Earned",
)

#: Qdrant distance metric and the book's duplication factor (each record is repeated 5×).
QDRANT_DISTANCE = "Cosine"
DEFAULT_DSIZE = 5

#: The book's ``similarity_top_k`` for the target-record query.
DEFAULT_TOP_K = 1

#: Upsert batches are bounded by an approximate 4 MB payload, as in the notebook.
BATCH_BYTES = 4_000_000

#: Embedding requests are batched so a local server is not asked for 10 000 texts at once.
EMBED_BATCH_SIZE = 1000

#: The book's retention prompt and community-manager system message.
EMAIL_QUERY_PROMPT = (
    "I have this customer bank record with interesting information on age, credit "
    "score and more and similar customers. What could I suggest to keep them in my "
    "bank in an email with an url to get new advantages based on the fields for each "
    "Customer ID:"
)
EMAIL_SYSTEM_PROMPT = (
    "You are the community manager can write engaging email based on the text you "
    "have. Do not use a surname but simply Dear Valued Customer instead."
)

#: The book's target record (Pipeline 3). The pipeline 2 notebook queries Robertson.
DEFAULT_QUERY_RECORD = (
    "Customer Henderson CreditScore 599 Age 37Tenure 2Balance 0.0NumOfProducts "
    "1HasCrCard 1IsActiveMember 1EstimatedSalary 107000.88Exited 1Complain 1"
    "Satisfaction Score 2Card Type DIAMONDPoint Earned 501"
)

_HEADERS = {"User-Agent": "my_rag/0.1 (study project; RAG-Driven-Generative-AI chapter 6)"}


# --------------------------------------------------------------------------- #
# Pipeline 1 — collecting and preparing the dataset
# --------------------------------------------------------------------------- #
def collect_churn(
    path: Path | str = DEFAULT_CHURN_CSV,
    *,
    session: Optional[requests.Session] = None,
    timeout: float = 60.0,
    progress: Optional[Callable[[str], None]] = None,
) -> Path:
    """Download the book's prepared ``data1.csv`` to ``path`` and return it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        progress(f"downloading {CHURN_URL}")
    client = session or requests
    response = client.get(CHURN_URL, headers=_HEADERS, timeout=timeout)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def load_churn(path: Path | str = DEFAULT_CHURN_CSV) -> list[dict[str, str]]:
    """Read the churn CSV into a list of column-name → value dicts."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"dataset {path} not found; collect it with: python -m my_rag --collect --corpus churn"
        )
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_records(rows: Sequence[dict[str, str]]) -> list[str]:
    """Turn each row into the book's ``"column: value"`` text record (one line per chunk)."""
    if not rows:
        raise ValueError("cannot build records from an empty dataset")
    columns = list(rows[0])
    return [" ".join(f"{column}: {row[column]}" for column in columns) for row in rows]


def _number(value: str) -> float:
    """Parse a CSV cell as a number (the book's pandas dtypes, made explicit)."""
    return float(value)


@dataclass(frozen=True)
class ExitedOverview:
    """The book's "Complain and exited" counts."""

    sum_exited: int
    sum_complain: int

    @property
    def complain_over_exited(self) -> float:
        """Complaints as a percentage of exits (0 when nobody exited)."""
        return self.sum_complain / self.sum_exited * 100 if self.sum_exited else 0.0


def exited_overview(rows: Sequence[dict[str, str]]) -> ExitedOverview:
    """Count ex-customers and complainers (the book's first EDA cell)."""
    return ExitedOverview(
        sum_exited=sum(1 for row in rows if _number(row["Exited"]) == 1),
        sum_complain=sum(1 for row in rows if _number(row["Complain"]) == 1),
    )


@dataclass(frozen=True)
class AgeOverview:
    """The book's "Age and exited" counts for a threshold age."""

    age: float
    aged_and_over: int
    sum_exited: int

    @property
    def aged_over_among_exited(self) -> float:
        """Customers at or above ``age`` as a percentage of the ex-customers."""
        return self.aged_and_over / self.sum_exited * 100 if self.sum_exited else 0.0


def age_overview(rows: Sequence[dict[str, str]], age: float = 50) -> AgeOverview:
    """Count ex-customers aged ``age`` and over (the book's second EDA cell)."""
    returned = 0
    exited = 0
    for row in rows:
        if _number(row["Exited"]) == 1:
            exited += 1
            if _number(row["Age"]) >= age:
                returned += 1
    return AgeOverview(age=age, aged_and_over=returned, sum_exited=exited)


@dataclass(frozen=True)
class SalaryOverview:
    """The book's "Salary and exited" counts for a salary threshold."""

    threshold: float
    salary_over: int
    sum_exited: int

    @property
    def salary_over_among_exited(self) -> float:
        """Customers above ``threshold`` as a percentage of the ex-customers."""
        return self.salary_over / self.sum_exited * 100 if self.sum_exited else 0.0


def salary_overview(rows: Sequence[dict[str, str]], threshold: float = 100000) -> SalaryOverview:
    """Count ex-customers with an estimated salary over ``threshold`` (the book's third cell)."""
    returned = 0
    exited = 0
    for row in rows:
        if _number(row["Exited"]) == 1:
            exited += 1
            if _number(row["EstimatedSalary"]) >= threshold:
                returned += 1
    return SalaryOverview(threshold=threshold, salary_over=returned, sum_exited=exited)


def minmax_scale(features: np.ndarray) -> np.ndarray:
    """Scale each column to ``[0, 1]`` (the book's ``MinMaxScaler``)."""
    array = np.asarray(features, dtype=float)
    low = array.min(axis=0)
    high = array.max(axis=0)
    span = np.where(high == low, 1.0, high - low)
    return (array - low) / span


def churn_features(
    rows: Sequence[dict[str, str]], features: Sequence[str] = CLUSTER_FEATURES
) -> np.ndarray:
    """The MinMax-scaled feature matrix the book clusters (``data2[features]``)."""
    if not rows:
        raise ValueError("cannot cluster an empty dataset")
    returned = np.array([[float(row[name]) for name in features] for row in rows], dtype=float)
    return minmax_scale(returned)


def _kmeans_plusplus(array: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding: spread the initial centroids by distance, as sklearn does."""
    centers = [array[rng.integers(len(array))]]
    for _ in range(1, k):
        distances = np.linalg.norm(
            array[:, None, :] - np.asarray(centers)[None, :, :], axis=2
        ).min(axis=1) ** 2
        total = distances.sum()
        if total <= 0:  # duplicate points: fall back to a random one
            centers.append(array[rng.integers(len(array))])
            continue
        centers.append(array[rng.choice(len(array), p=distances / total)])
    return np.asarray(centers)


def _assign(array: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Index of the nearest centroid per point."""
    return np.linalg.norm(array[:, None, :] - centers[None, :, :], axis=2).argmin(axis=1)


def _centroids(
    array: np.ndarray, labels: np.ndarray, k: int, fallback: np.ndarray
) -> np.ndarray:
    """Mean of each cluster; an empty cluster keeps its previous centroid."""
    returned = np.empty((k, array.shape[1]))
    for cluster in range(k):
        members = array[labels == cluster]
        returned[cluster] = members.mean(axis=0) if len(members) else fallback[cluster]
    return returned


def kmeans(
    features: np.ndarray,
    n_clusters: int,
    *,
    n_init: int = 10,
    seed: int = 0,
    max_iter: int = 300,
    tol: float = 1e-4,
) -> tuple[np.ndarray, float]:
    """Cluster ``features`` with k-means++ over ``n_init`` restarts.

    Returns ``(labels, inertia)``. A small numpy stand-in for
    ``sklearn.cluster.KMeans``: the best (lowest-inertia) of the restarts wins, so
    the result is stable for a fixed ``seed``. Cluster numbering may differ from
    scikit-learn's, but the segmentation is the same.
    """
    array = np.asarray(features, dtype=float)
    if array.ndim != 2:
        raise ValueError("features must be a 2-D array")
    if not 1 <= n_clusters <= len(array):
        raise ValueError(f"n_clusters must be between 1 and {len(array)}, got {n_clusters}")

    rng = np.random.default_rng(seed)
    best_inertia = float("inf")
    best_labels = np.zeros(len(array), dtype=int)
    for _ in range(n_init):
        centers = _kmeans_plusplus(array, n_clusters, rng)
        for _ in range(max_iter):
            labels = _assign(array, centers)
            moved = _centroids(array, labels, n_clusters, centers)
            if np.allclose(moved, centers, atol=tol):
                centers = moved
                break
            centers = moved
        labels = _assign(array, centers)
        inertia = float(((array - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels
    return best_labels, best_inertia


def silhouette_score(
    features: np.ndarray, labels: np.ndarray, *, sample: Optional[int] = None, seed: int = 0
) -> float:
    """Mean silhouette coefficient, as the book prints for each candidate ``k``.

    The coefficient is O(n²); ``sample`` bounds the number of points (the book
    computes it on all 10 000, which is ~800 MB of distances). ``None`` uses all.
    """
    array = np.asarray(features, dtype=float)
    labels = np.asarray(labels)
    if sample is not None and sample < len(array):
        picked = np.random.default_rng(seed).choice(len(array), sample, replace=False)
        array, labels = array[picked], labels[picked]
    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0
    distances = np.linalg.norm(array[:, None, :] - array[None, :, :], axis=2)
    scores = []
    for index in range(len(array)):
        own = distances[index][labels == labels[index]]
        a = own.sum() / (len(own) - 1) if len(own) > 1 else 0.0
        b = min(
            distances[index][labels == cluster].mean()
            for cluster in unique
            if cluster != labels[index]
        )
        scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(scores))


def davies_bouldin_score(features: np.ndarray, labels: np.ndarray) -> float:
    """Davies-Bouldin index (lower is better), the book's second selection metric."""
    array = np.asarray(features, dtype=float)
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0
    centroids = np.array([array[labels == cluster].mean(axis=0) for cluster in unique])
    scatters = np.array(
        [
            np.linalg.norm(array[labels == cluster] - centroids[index], axis=1).mean()
            for index, cluster in enumerate(unique)
        ]
    )
    total = 0.0
    for first in range(len(unique)):
        worst = 0.0
        for second in range(len(unique)):
            if first == second:
                continue
            distance = np.linalg.norm(centroids[first] - centroids[second])
            ratio = (scatters[first] + scatters[second]) / distance if distance > 0 else 0.0
            worst = max(worst, ratio)
        total += worst
    return float(total / len(unique))


@dataclass(frozen=True)
class ClusterScore:
    """One row of the book's "experiment with different numbers of clusters" loop."""

    n_clusters: int
    silhouette: float
    davies_bouldin: float


def score_clusters(
    rows: Sequence[dict[str, str]],
    *,
    range_: Sequence[int] = (2, 3, 4),
    n_init: int = 20,
    seed: int = 0,
    sample: int = 1000,
) -> list[ClusterScore]:
    """Silhouette and Davies-Bouldin for each ``k`` in ``range_`` (the book's loop)."""
    features = churn_features(rows)
    scores = []
    for n_clusters in range_:
        labels, _ = kmeans(features, n_clusters, n_init=n_init, seed=seed)
        scores.append(
            ClusterScore(
                n_clusters=n_clusters,
                silhouette=silhouette_score(features, labels, sample=sample, seed=seed),
                davies_bouldin=davies_bouldin_score(features, labels),
            )
        )
    return scores


@dataclass(frozen=True)
class ClusterSummary:
    """Counts per cluster, split by complaints and exits (the book's final cells)."""

    counts: dict[int, int]
    complaints: dict[int, int]
    exited: dict[int, int]


def cluster_churn(
    rows: Sequence[dict[str, str]], *, n_clusters: int = 2, n_init: int = 10, seed: int = 0
) -> tuple[np.ndarray, ClusterSummary]:
    """Run the book's final KMeans and summarize each ``class``."""
    labels, _ = kmeans(churn_features(rows), n_clusters, n_init=n_init, seed=seed)
    summary = ClusterSummary(
        counts={cluster: int((labels == cluster).sum()) for cluster in range(n_clusters)},
        complaints={
            cluster: sum(
                1
                for index, row in enumerate(rows)
                if labels[index] == cluster and _number(row["Complain"]) == 1
            )
            for cluster in range(n_clusters)
        },
        exited={
            cluster: sum(
                1
                for index, row in enumerate(rows)
                if labels[index] == cluster and _number(row["Exited"]) == 1
            )
            for cluster in range(n_clusters)
        },
    )
    return labels, summary


# --------------------------------------------------------------------------- #
# Pipeline 2 — the Qdrant index
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VectorMatch:
    """One Qdrant query hit: its id, similarity score, and stored text."""

    id: str
    score: float
    text: str

    def __str__(self) -> str:  # so --show-context prints the record text
        return self.text


@dataclass(frozen=True)
class BuildResult:
    """Outcome of :func:`upsert_records`."""

    collection: str
    records: int
    vectors: int
    dimension: int
    dsize: int


class QdrantIndex:
    """A thin wrapper over one Qdrant collection (the ``index`` of the pipeline)."""

    def __init__(self, client, collection: str) -> None:
        self.client = client
        self.collection = collection

    def upsert(self, points: Sequence[dict], **kwargs) -> None:
        """Upsert ``{"id", "vector", "payload"}`` dicts as Qdrant points."""
        from qdrant_client import models

        structs = [
            point
            if isinstance(point, models.PointStruct)
            else models.PointStruct(id=point["id"], vector=point["vector"], payload=point["payload"])
            for point in points
        ]
        self.client.upsert(collection_name=self.collection, points=structs, **kwargs)

    def query(self, vector: Sequence[float], *, limit: int, with_payload: bool = True):
        """The ``limit`` nearest scored points for ``vector``."""
        return self.client.query_points(
            collection_name=self.collection, query=list(vector), limit=limit, with_payload=with_payload
        ).points

    def count(self) -> int:
        """Number of points currently stored in the collection."""
        return self.client.count(self.collection).count

    def close(self) -> None:
        """Release the local engine (or the connection) held by the client."""
        self.client.close()


def open_qdrant_index(
    settings: Settings,
    *,
    collection: Optional[str] = None,
    dimension: Optional[int] = None,
    create: bool = True,
) -> QdrantIndex:
    """Open the Qdrant collection, creating it when absent.

    With ``settings.qdrant_url`` set the client talks to that server; otherwise it
    uses Qdrant's embedded on-disk engine at ``settings.qdrant_path`` (no server, no
    key). The collection size must match the embedding model (384 for the local
    ``all-minilm``, 1536 for OpenAI's ``text-embedding-3-small``). The client is
    imported lazily so the data/EDA commands do not need it.
    """
    from qdrant_client import QdrantClient, models

    name = collection or settings.qdrant_collection
    if settings.qdrant_url:
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    else:
        path = Path(settings.qdrant_path)
        path.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(path))
    if not client.collection_exists(name):
        if not create:
            client.close()
            raise FileNotFoundError(
                f"Qdrant collection {name!r} not found; build it with: python -m my_rag --build-index"
            )
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=dimension or settings.embedding_dimension,
                distance=models.Distance[QDRANT_DISTANCE.upper()],
            ),
        )
    return QdrantIndex(client, name)


def embed_in_batches(
    chunks: Sequence[str],
    embedding_function: EmbeddingFunction,
    *,
    batch_size: int = EMBED_BATCH_SIZE,
) -> list[list[float]]:
    """Embed ``chunks`` in batches, as the book's ``embed_chunks`` does."""
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), batch_size):
        vectors.extend(embedding_function.embed_documents(list(chunks[start : start + batch_size])))
    return vectors


def _batch_size(data: Sequence[dict], *, limit: int = BATCH_BYTES) -> int:
    """How many leading items fit within ``limit`` bytes (the book's ``get_batch_size``)."""
    total = 0
    for count, item in enumerate(data, start=1):
        size = sum(sys.getsizeof(value) for value in item.values())
        if total + size > limit:
            return max(1, count - 1)
        total += size
    return len(data)


def upsert_records(
    records: Sequence[str],
    embedding_function: EmbeddingFunction,
    index: QdrantIndex | object,
    *,
    collection: str = "",
    dsize: int = DEFAULT_DSIZE,
    batch_bytes: int = BATCH_BYTES,
    progress: Optional[Callable[[str], None]] = None,
) -> BuildResult:
    """Embed ``records``, duplicate each ``dsize`` times, and upsert into ``index``.

    The duplication mirrors the book's scaling step: it pads the index to its
    advertised size (``bank-index-50000`` = 10 000 records × 5). Upserts are
    size-bounded batches, like the notebook's ``batch_upsert``.
    """
    if dsize < 1:
        raise ValueError(f"dsize must be at least 1, got {dsize}")
    if not records:
        raise ValueError("cannot upsert an empty dataset")
    embeddings = embed_in_batches(records, embedding_function)
    if len(embeddings) != len(records):
        raise ValueError(
            f"got {len(embeddings)} embeddings for {len(records)} records"
        )
    dimension = len(embeddings[0])
    data: list[dict] = []
    for text, vector in zip(records, embeddings):
        for _ in range(dsize):
            data.append(
                {"id": len(data) + 1, "vector": vector, "payload": {"text": text}}
            )
    total = len(data)
    done = 0
    while done < total:
        size = _batch_size(data[done:], limit=batch_bytes)
        index.upsert(data[done : done + size], wait=True)
        done += size
        if progress is not None:
            progress(f"upserted {done}/{total} vectors")
    return BuildResult(
        collection=collection,
        records=len(records),
        vectors=total,
        dimension=dimension,
        dsize=dsize,
    )


def query_index(
    index: QdrantIndex | object,
    embedding_function: EmbeddingFunction,
    record: str,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> list[VectorMatch]:
    """Embed ``record`` and return the ``top_k`` nearest Qdrant matches."""
    points = index.query(embedding_function.embed_query(record), limit=top_k, with_payload=True)
    return [
        VectorMatch(
            id=str(point.id),
            score=float(point.score),
            text=str((point.payload or {}).get("text", "")),
        )
        for point in points
    ]


# --------------------------------------------------------------------------- #
# Pipeline 3 — augmented generation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RetentionEmail:
    """Outcome of :meth:`CustomerRAG.answer`: the matched records and the draft email."""

    record: str
    matches: list[VectorMatch]
    prompt: str
    response: str
    elapsed: float


def build_email_prompt(record: str, context: str = "") -> str:
    """The book's augmented prompt: query prompt + target record + retrieved context."""
    return f"{EMAIL_QUERY_PROMPT}{record}{context}"


class CustomerRAG:
    """Draft a customer-retention email from the record's nearest Qdrant neighbours."""

    def __init__(
        self,
        settings: Settings,
        index,
        *,
        embedding_function: Optional[EmbeddingFunction] = None,
        llm: Optional[LLM] = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.settings = settings
        self.index = index
        self.embedding_function = embedding_function or create_embedding_function(settings)
        self.llm = llm or create_llm(settings)
        self.top_k = top_k

    def answer(self, record: str = DEFAULT_QUERY_RECORD) -> RetentionEmail:
        """Retrieve similar customers, augment the prompt, and generate the email."""
        matches = query_index(
            self.index, self.embedding_function, record, top_k=self.top_k
        )
        context = "\n".join(match.text for match in matches)
        prompt = build_email_prompt(record, context)
        start = time.perf_counter()
        response = self.llm.complete(prompt, system=EMAIL_SYSTEM_PROMPT, temperature=0.0)
        elapsed = time.perf_counter() - start
        return RetentionEmail(
            record=record,
            matches=matches,
            prompt=prompt,
            response=response,
            elapsed=elapsed,
        )
