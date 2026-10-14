"""RAG for video stock production (chapter 10).

Reimplementation of the book's four Chapter 10 notebooks — a three-stage
pipeline that turns raw stock videos into searchable, described frames:

- ``Video_dataset_visualization.ipynb`` — download the videos and show their
  thumbnails and durations (:func:`video_info` / :func:`thumbnail`).
- ``Pipeline_1_Generator_and_Commentator.ipynb`` — split a video into frames
  (OpenCV), sample every *n*-th frame, ask a vision model "What is happening in
  this image?" for each, and save the comments to a CSV
  (:func:`comment_video`).
- ``Pipeline_2_The_Vector_Store_Administrator.ipynb`` — load the comment CSVs,
  treat each comment as one chunk, embed it, upsert the vectors into an index
  (the book's ``videos-sports-us``), then query it (:func:`upsert_comments`).
- ``Pipeline_3_The_Video_Expert.ipynb`` — RAG: the query retrieves a comment, the
  model rewrites it (cognitive dissonances / engaging style / label), and the
  evaluator scores the rewrite against human-feedback flashcards
  (:class:`VideoExpert`, :class:`EvaluationReport`).

Substitutions, consistent with the rest of ``my_rag``:

- **Qdrant instead of Pinecone.** Pinecone is a paid managed service; chapter 6
  already swaps it for Qdrant, whose Python client embeds a local on-disk engine
  (no key, no server). The same collection name, cosine metric and batch upsert
  are kept, but the vector size follows the embedding model (384 for the local
  ``all-minilm`` vs. Pinecone's hard-coded 1536).
- **A configurable vision model instead of GPT-4o Vision.** ``gemma3:12b`` by
  default through Ollama's OpenAI-compatible endpoint; ``--provider openai``
  uses ``gpt-4o``. The frame travels as the same ``data:image/jpeg;base64`` URL.
- **The book's ready-made comment CSVs from GitHub** (``Chapter10/comments``) so
  the vector-store pipelines run with no video, no OpenCV and no vision model;
  OpenCV is needed only to make new comments (``--comment-video``).
- **The configured embedding model instead of sentence-transformers / spaCy**
  for the evaluator, and numpy instead of scikit-learn for the metrics
  (``describe`` / ``classify``), the same dependency-ladder choice as the other
  chapters.

Not ported: the Colab scaffolding (the Drive mount, ``api_key.txt`` /
``pinecone.txt`` key files, the ``!pip install`` pins), the notebooks' inline
``display()`` / ``print`` cells, and the notebook's video-saving cells (kept as
the ``save`` / ``save_frames`` switches) — the frames under
``data/raw/video_frames`` already are the "save" step.
"""
from __future__ import annotations

import csv
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import requests

from .config import (
    DEFAULT_VIDEO_COLLECTION,
    DEFAULT_VIDEO_COMMENTS,
    DEFAULT_VIDEO_DIR,
    DEFAULT_VIDEO_FRAMES,
    DEFAULT_VIDEO_SAMPLE,
    Settings,
)
from .dynamic import text_similarity
from .embeddings import EmbeddingFunction, create_embedding_function
from .llm import LLM, create_llm

#: The book's GitHub repository, and the folders its videos/comments live in.
VIDEO_BASE_URL = "https://raw.githubusercontent.com/Denis2054/RAG-Driven-Generative-AI/main/"
VIDEO_DIRECTORY = "Chapter10/videos"
COMMENTS_DIRECTORY = "Chapter10/comments"

#: The book's 21 stock videos (the visualization notebook's ``lfiles``).
VIDEO_FILES: tuple[str, ...] = (
    "jogging1.mp4",
    "jogging2.mp4",
    "skiing1.mp4",
    "soccer_pass.mp4",
    "soccer_player_head.mp4",
    "soccer_player_running.mp4",
    "surfer1.mp4",
    "surfer2.mp4",
    "swimming1.mp4",
    "walking1.mp4",
    "alpinist1.mp4",
    "ball_passing_goal.mp4",
    "basketball1.mp4",
    "basketball2.mp4",
    "basketball3.mp4",
    "basketball4.mp4",
    "basketball5.mp4",
    "female_player_after_scoring.mp4",
    "football1.mp4",
    "football2.mp4",
    "hockey1.mp4",
)

#: The book's ready-made comment CSVs (``Pipeline 2``'s ``lfiles``, one per
#: video). ``save_comment()`` names a video's file ``<video>.csv``, so the videos
#: and the comment files line up one-to-one.
COMMENT_FILES: tuple[str, ...] = tuple(f"{name}.csv" for name in VIDEO_FILES)

#: The video Pipeline 1 comments by default (the notebook picks ``alpinist1``).
DEFAULT_VIDEO = "alpinist1.mp4"

#: The book's frame-sampling frequency (``nb`` in ``generate_openai_comments``).
DEFAULT_SAMPLE_RATE = 3

#: The book's vision question and its generation cap.
VISION_QUESTION = "What is happening in this image?"
VISION_MAX_TOKENS = 150

#: The comment CSV header written by ``save_comment()``.
COMMENT_COLUMNS: tuple[str, ...] = ("ID", "FrameNumber", "Comment", "FileName")

#: The book's Pipeline 3 system prompt (rewrite + label).
VIDEO_EXPERT_SYSTEM = (
    "You will be provided with comments of an image frame taken from a video. "
    "Analyze the text and 1. Point out the cognitive dissonances 2. Rewrite the "
    "comment in a logical engaging style. 3. Provide a label for this image such "
    "as Label: basketball, football, soccer or other label."
)

#: The book's generation temperature for the rewrite (``temperature=0``).
REWRITE_TEMPERATURE = 0.0

#: The book's accuracy threshold, used to bin the retrieval and similarity scores.
DEFAULT_THRESHOLD = 0.6

#: The book's Pipeline 3 query and its ``k``.
DEFAULT_QUERY = "Find a basketball player that is scoring with a dunk."
DEFAULT_TOP_K = 1

#: The book's Pipeline 3 evaluation sample: the query and the human-feedback
#: flashcard comment it is scored against.
EVALUATION_SAMPLE: tuple[tuple[str, str], ...] = (
    ("Find a female soccer player that is playing.",
     "This image shows soccer players on a field dribbling and passing the ball."),
    ("Find a basketball player scoring with a slam dunk.",
     "This image shows an incredible dunk by a basketball player."),
    ("Find a skiier skiing down a slope.",
     "This image shows people skiing down a slope on white snow."),
    ("Show a hockey player on the ice rink.",
     "This image shows a hokey player pushing the puck with his hockey stick."),
    ("Show somebody walking in a forest.",
     "This image shows a person walking through a forest with boots."),
    ("Show a surfer.", "This is a cool image of a surfer on a big wave."),
    ("Show a person who is swimming.", "This image shows a good swimmer swimming fast."),
    ("Show a person jogging.", "This image shows more than one person running and jogging."),
    ("Show a person with a blue helmet climbing the rock of a mountain.",
     "This image shows a person with a white cap climbing a mountain."),
    ("Show a ball passing a goal keepeer.",
     "This image shows a team of female soccer players scoring a goal."),
)

_HEADERS = {"User-Agent": "my_rag/0.1 (study project; RAG-Driven-Generative-AI chapter 10)"}


# --------------------------------------------------------------------------- #
# Videos — download, inspect, split (the visualization notebook + Pipeline 1)
# --------------------------------------------------------------------------- #
def download(
    directory: str,
    filename: str,
    dest: Path | str,
    *,
    session: Optional[requests.Session] = None,
    timeout: float = 120.0,
) -> Path:
    """Download ``<base>/<directory>/<filename>`` into ``dest`` and return the path."""
    url = f"{VIDEO_BASE_URL}{directory}/{filename}"
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    client = session or requests
    response = client.get(url, headers=_HEADERS, timeout=timeout)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def download_video(
    name: str = DEFAULT_VIDEO,
    dest_dir: Path | str = DEFAULT_VIDEO_DIR,
    *,
    session: Optional[requests.Session] = None,
    timeout: float = 300.0,
    progress: Optional[Callable[[str], None]] = None,
) -> Path:
    """Download one of the book's videos into ``dest_dir`` (skipping a cached file)."""
    dest_dir = Path(dest_dir)
    dest = dest_dir / Path(name).name
    if dest.is_file() and dest.stat().st_size > 0:
        if progress is not None:
            progress(f"using cached {dest}")
        return dest
    if progress is not None:
        progress(f"downloading {name} -> {dest}")
    return download(VIDEO_DIRECTORY, Path(name).name, dest, session=session, timeout=timeout)


def download_comments(
    dest_dir: Path | str = DEFAULT_VIDEO_COMMENTS,
    *,
    names: Sequence[str] = COMMENT_FILES,
    session: Optional[requests.Session] = None,
    timeout: float = 60.0,
    progress: Optional[Callable[[str], None]] = None,
) -> list[Path]:
    """Download the book's ready-made comment CSVs into ``dest_dir``.

    A missing file is reported and skipped rather than aborting the batch (the
    book's ``download()`` swallows a failed ``curl`` the same way).
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in names:
        dest = dest_dir / Path(name).name
        if dest.is_file() and dest.stat().st_size > 0:
            written.append(dest)
            if progress is not None:
                progress(f"using cached {dest}")
            continue
        try:
            written.append(
                download(COMMENTS_DIRECTORY, Path(name).name, dest, session=session, timeout=timeout)
            )
        except Exception as exc:  # a 404 for one file must not stop the rest
            if progress is not None:
                progress(f"skipped {name}: {exc}")
            continue
        if progress is not None:
            progress(f"downloaded {dest}")
    return written


def extract_frames(
    video: Path | str,
    out_dir: Path | str,
    *,
    limit: Optional[int] = None,
    progress: Optional[Callable[[int], None]] = None,
) -> list[Path]:
    """Split ``video`` into ``frame_<n>.jpg`` files under ``out_dir`` (OpenCV).

    Names match the book's ``split_file``: the first written file is
    ``frame_0.jpg``. Imported lazily, so the comment/index pipelines do not need
    OpenCV; a missing install gets an actionable hint.
    """
    try:
        import cv2
    except ImportError as exc:  # OpenCV is only needed to make new comments
        raise RuntimeError(
            "splitting a video into frames needs OpenCV: pip install opencv-python "
            "(the vector-store pipelines can use the book's comment CSVs instead)"
        ) from exc

    video = Path(video)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    frames: list[Path] = []
    try:
        if not capture.isOpened():
            raise FileNotFoundError(f"cannot open video {video}")
        number = 0
        while True:
            if limit is not None and number >= limit:
                break
            ok, frame = capture.read()
            if not ok:
                break
            path = out_dir / f"frame_{number}.jpg"
            cv2.imwrite(str(path), frame)
            frames.append(path)
            number += 1
            if progress is not None:
                progress(number)
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"video {video} produced no frames")
    return frames


def sample_positions(total: int, *, rate: int = DEFAULT_SAMPLE_RATE) -> list[int]:
    """The frames the book comments: every ``rate``-th of ``total`` (its ``nb``).

    ``generate_openai_comments`` increments a counter per frame and comments when
    it hits ``nb`` while ``counter < total_frames`` — i.e. frames ``nb-1``,
    ``2*nb-1``, … up to (but excluding) the final frame. For ``nb=3`` that is
    2, 5, 8, … (the notebook prints "Analyzing frame 2…" first).
    """
    if rate < 1:
        raise ValueError(f"sample rate must be at least 1, got {rate}")
    return [number for number in range(total) if (number + 1) % rate == 0 and number + 1 < total]


def video_info(video: Path | str) -> "VideoInfo":
    """Frame count, frame rate and duration of ``video`` (OpenCV).

    Mirrors the visualization notebook's ``cv2.CAP_PROP_*`` cell. Imported
    lazily, with the same OpenCV hint as :func:`extract_frames`.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "reading a video's properties needs OpenCV: pip install opencv-python"
        ) from exc

    video = Path(video)
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise FileNotFoundError(f"cannot open video {video}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    duration = frame_count / frame_rate if frame_rate > 0 else 0.0
    return VideoInfo(
        file_name=video.name,
        frame_count=frame_count,
        frame_rate=frame_rate,
        duration=duration,
    )


def thumbnail(
    video: Path | str,
    output: Path | str,
    *,
    frame_number: int = 5,
    size: tuple[int, int] = (135, 90),
) -> Path:
    """Save frame ``frame_number`` of ``video`` as a resized JPEG (the notebook's frame cell)."""
    try:
        import cv2
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "saving a video thumbnail needs OpenCV: pip install opencv-python"
        ) from exc

    video = Path(video)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise FileNotFoundError(f"cannot open video {video}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"frame {frame_number} of {video} could not be read")
    finally:
        capture.release()
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if size:
        image = image.resize(size)
    image.save(output, format="JPEG")
    return output


@dataclass(frozen=True)
class VideoInfo:
    """A video's shape, as the visualization notebook reports it."""

    file_name: str
    frame_count: int
    frame_rate: float
    duration: float


# --------------------------------------------------------------------------- #
# Comments — the Pipeline 1 CSV
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FrameComment:
    """One commented frame: the book's ``ID``/``FrameNumber``/``Comment``/``FileName`` row."""

    id: str
    frame_number: int
    comment: str
    file_name: str

    def to_row(self) -> list[str]:
        """The CSV row for this comment."""
        return [self.id, str(self.frame_number), self.comment, self.file_name]

    def __str__(self) -> str:  # so --show-context prints the comment
        return self.comment


def comment_frame(
    image_url: str,
    settings: Settings,
    *,
    client,
    question: str = VISION_QUESTION,
    max_tokens: int = VISION_MAX_TOKENS,
) -> str:
    """Ask the vision model what happens in a frame (the book's vision call).

    ``image_url`` is a ``data:image/jpeg;base64,…`` URL; the book sends exactly
    this shape with ``max_tokens=150``.
    """
    response = client.chat.completions.create(
        model=settings.vision_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def save_comments(
    comments: Iterable[FrameComment],
    path: Path | str,
    *,
    file_name: str = "",
) -> Path:
    """Write comments to a CSV with the book's header (one row per comment).

    A fresh UUID is minted per comment when it has none, matching ``save_comment``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(COMMENT_COLUMNS)
        for comment in comments:
            row = comment.to_row()
            if not row[0]:
                row[0] = str(uuid.uuid4())
            if file_name and not row[3]:
                row[3] = file_name
            writer.writerow(row)
    return path


def _is_header(row: Sequence[str]) -> bool:
    """Whether a CSV row is the book's header rather than a comment."""
    return bool(row) and row[0].strip().strip('"').upper() == "ID"


def read_comments(path: Path | str) -> list[FrameComment]:
    """Read one comment CSV, skipping its header and any blank/N/A rows.

    The book's CSVs carry a header (``ID,FrameNumber,Comment,FileName``); the
    notebook re-reads them with ``header=None``, which keeps the header as data —
    here it is dropped, so only real comments are indexed.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"comment file {path} not found; collect them with: "
            "python -m my_rag --collect --corpus video-comments"
        )
    comments: list[FrameComment] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for position, row in enumerate(csv.reader(handle)):
            if not row or (position == 0 and _is_header(row)):
                continue
            row = row + [""] * (len(COMMENT_COLUMNS) - len(row))
            comment = row[2].strip()
            if not comment or comment.lower() in {"nan", "n/a"}:
                continue
            try:
                frame_number = int(float(row[1]))
            except (TypeError, ValueError):
                frame_number = 0
            comments.append(
                FrameComment(
                    id=row[0].strip(),
                    frame_number=frame_number,
                    comment=comment,
                    file_name=row[3].strip(),
                )
            )
    return comments


def load_comments(
    paths: Iterable[Path | str],
    *,
    limit: Optional[int] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> list[FrameComment]:
    """Read and concatenate several comment CSVs into one list (the book's ``dfl``)."""
    comments: list[FrameComment] = []
    for path in paths:
        try:
            loaded = read_comments(path)
        except (OSError, ValueError) as exc:
            if progress is not None:
                progress(f"skipped {path}: {exc}")
            continue
        comments.extend(loaded)
        if progress is not None:
            progress(f"{Path(path).name}: {len(loaded)} comments")
        if limit is not None and len(comments) >= limit:
            return comments[:limit]
    if not comments:
        raise ValueError("no comments loaded; collect them with: python -m my_rag --collect --corpus video-comments")
    return comments


def comment_video(
    video: Path | str,
    output: Path | str,
    settings: Settings,
    *,
    frames_dir: Optional[Path | str] = None,
    sample: int = DEFAULT_SAMPLE_RATE,
    limit: Optional[int] = None,
    client=None,
    progress: Optional[Callable[[str], None]] = None,
) -> list[FrameComment]:
    """Comment a video end-to-end (Pipeline 1): split, sample, comment, save CSV.

    Frames are written to ``frames_dir`` (default: ``settings.video_frames/<root>``)
    and the sampled frames are described by ``settings.vision_model``. ``limit``
    caps how many sampled frames are commented (a local vision model is slow), so
    a run stays practical; pass ``limit=0`` to comment every sampled frame. A fresh
    ``client`` is built when none is injected (tests pass a fake).
    """
    video = Path(video)
    root = video.name.split(".")[0]
    frames_dir = Path(frames_dir) if frames_dir else Path(settings.video_frames) / root
    if progress is not None:
        progress(f"splitting {video} -> {frames_dir}")
    frames = extract_frames(video, frames_dir, progress=None)
    positions = sample_positions(len(frames), rate=sample)
    if limit:
        positions = positions[:limit]
    if not positions:
        raise ValueError(f"video {video} is too short to sample every {sample} frame(s)")

    if client is None:
        from openai import OpenAI

        client = OpenAI(base_url=settings.base_url, api_key=settings.api_key)
    from .multimodal import image_data_url  # the same data: URL the vision call uses

    comments: list[FrameComment] = []
    for number in positions:
        if progress is not None:
            progress(f"analyzing frame {number}...")
        caption = comment_frame(image_data_url(frames[number]), settings, client=client)
        comments.append(
            FrameComment(
                id=str(uuid.uuid4()),
                frame_number=number,
                comment=caption,
                file_name=video.name,
            )
        )
    return_comments = save_comments(comments, output, file_name=video.name)
    if progress is not None:
        progress(f"wrote {len(comments)} comments -> {return_comments}")
    return comments


# --------------------------------------------------------------------------- #
# Index — the vector store administrator (Pipeline 2)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VideoMatch:
    """One index hit: its id, similarity score, stored comment and frame metadata."""

    id: str
    score: float
    text: str
    frame_number: int = 0
    file_name: str = ""

    def __str__(self) -> str:  # so --show-context prints the comment
        return self.text


@dataclass(frozen=True)
class VideoIndexResult:
    """Outcome of :func:`upsert_comments`."""

    collection: str
    comments: int
    vectors: int
    dimension: int


def build_chunks(comments: Sequence[FrameComment]) -> list[str]:
    """One chunk per comment (the book's step 2 — each comment is its own chunk)."""
    if not comments:
        raise ValueError("cannot build chunks from an empty comment set")
    return [comment.comment for comment in comments]


def duplicate_chunks(chunks: Sequence[str]) -> int:
    """How many chunks repeat an earlier one (the book's duplicate check)."""
    seen: set[str] = set()
    duplicates = 0
    for chunk in chunks:
        if chunk in seen:
            duplicates += 1
        else:
            seen.add(chunk)
    return duplicates


def open_video_index(
    settings: Settings,
    *,
    collection: Optional[str] = None,
    dimension: Optional[int] = None,
    create: bool = True,
):
    """Open the video Qdrant collection (reusing chapter 6's local engine).

    Pinecone is replaced by Qdrant; the collection keeps the book's name
    (``videos-sports-us``) and cosine metric. Imported lazily via chapter 6.
    """
    from .scaling import open_qdrant_index

    return open_qdrant_index(
        settings,
        collection=collection or settings.video_collection,
        dimension=dimension or settings.embedding_dimension,
        create=create,
    )


def _point_id(raw: str) -> str:
    """A Qdrant-legal point id: the comment's UUID, else a deterministic UUID5."""
    try:
        return str(uuid.UUID(str(raw)))
    except (ValueError, AttributeError, TypeError):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, str(raw)))


def upsert_comments(
    comments: Sequence[FrameComment],
    embedding_function: EmbeddingFunction,
    index,
    *,
    collection: str = "",
    progress: Optional[Callable[[str], None]] = None,
) -> VideoIndexResult:
    """Embed each comment and upsert it with its frame metadata (Pipeline 2, step 3).

    The vector payload mirrors the book's ``data_for_upsert``: ``text`` (the
    comment), ``frame_number`` and ``file_name``. Upserts are batched like the
    notebook's ``upsert_in_batches``.
    """
    comments = list(comments)
    if not comments:
        raise ValueError("cannot index an empty comment set")
    chunks = build_chunks(comments)
    embeddings = embedding_function.embed_documents(chunks)
    if len(embeddings) != len(comments):
        raise ValueError(f"got {len(embeddings)} embeddings for {len(comments)} comments")
    dimension = len(embeddings[0]) if embeddings else 0
    points = [
        {
            "id": _point_id(comment.id),
            "vector": vector,
            "payload": {
                "text": comment.comment,
                "frame_number": comment.frame_number,
                "file_name": comment.file_name,
            },
        }
        for comment, vector in zip(comments, embeddings)
    ]
    for start in range(0, len(points), VIDEO_UPSERT_BATCH):
        batch = points[start : start + VIDEO_UPSERT_BATCH]
        index.upsert(batch, wait=True)
        if progress is not None:
            progress(f"upserted {min(start + VIDEO_UPSERT_BATCH, len(points))}/{len(points)} comments")
    return VideoIndexResult(
        collection=collection,
        comments=len(comments),
        vectors=len(points),
        dimension=dimension,
    )


#: The book's ``batch_size`` for the Pinecone upsert.
VIDEO_UPSERT_BATCH = 100


def query_video_index(
    index,
    embedding_function: EmbeddingFunction,
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> list[VideoMatch]:
    """Embed ``query`` and return the ``top_k`` nearest comment matches."""
    points = index.query(embedding_function.embed_query(query), limit=top_k, with_payload=True)
    return [
        VideoMatch(
            id=str(point.id),
            score=float(point.score),
            text=str((point.payload or {}).get("text", "")),
            frame_number=int((point.payload or {}).get("frame_number", 0) or 0),
            file_name=str((point.payload or {}).get("file_name", "")),
        )
        for point in points
    ]


# --------------------------------------------------------------------------- #
# Video Expert — RAG + evaluation (Pipeline 3)
# --------------------------------------------------------------------------- #
def _strip_markdown(value: str) -> str:
    """Strip surrounding markdown/whitespace/quotes from a parsed value."""
    return value.strip().strip("*_`# ").strip().strip("\"'").strip()


def extract_rewritten_comment(response: str) -> str:
    """Pull the rewrite out of the model's reply (the book's parser, plus alternates).

    The book looks for the literal ``Rewritten Comment:`` marker; the prompt asks
    for "a logical engaging style", so a local model often labels it *"Logical
    Engaging Style:"* (or just *"Rewritten:"*) instead — all three are accepted.
    """
    markers = ("Rewritten Comment:", "Logical Engaging Style:", "Rewritten:")
    lines = response.split("\n")
    rewritten: list[str] = []
    started = False
    for line in lines:
        if not started and any(marker in line for marker in markers):
            started = True
            continue
        if started:
            if not line.strip():
                break
            rewritten.append(line.strip())
    return _strip_markdown(" ".join(rewritten))


def extract_label(response: str) -> str:
    """Pull the text after ``Label:`` out of the model's reply ("" when absent)."""
    for line in response.split("\n"):
        if "Label:" in line:
            return _strip_markdown(line.split("Label:", 1)[1]).rstrip(".").strip()
    return ""


@dataclass(frozen=True)
class VideoAnswer:
    """Outcome of :meth:`VideoExpert.answer`: the retrieval, prompt and rewrite."""

    query: str
    matches: list[VideoMatch]
    retrieved: str
    prompt: str
    response: str
    rewritten: str
    label: str
    elapsed: float


@dataclass(frozen=True)
class DescriptiveStats:
    """The book's mean/median/spread block over one score list."""

    mean: float
    median: float
    std: float
    variance: float
    minimum: float
    maximum: float
    range: float
    q1: float
    q3: float
    iqr: float


@dataclass(frozen=True)
class ClassificationStats:
    """The book's F1 / precision / recall / accuracy block (binarized at a threshold)."""

    threshold: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    true_labels: list[int]
    predicted_labels: list[int]
    confusion: list[list[int]]


def describe(scores: Sequence[float]) -> DescriptiveStats:
    """Mean, median, spread and percentiles of ``scores`` (the book's metrics cell)."""
    values = np.asarray(list(scores), dtype=float)
    if values.size == 0:
        raise ValueError("cannot describe an empty score list")
    q1, q3 = np.percentile(values, [25, 75])
    return DescriptiveStats(
        mean=float(values.mean()),
        median=float(np.median(values)),
        std=float(values.std()),
        variance=float(values.var()),
        minimum=float(values.min()),
        maximum=float(values.max()),
        range=float(values.max() - values.min()),
        q1=float(q1),
        q3=float(q3),
        iqr=float(q3 - q1),
    )


def classify(
    scores: Sequence[float],
    retrieval_scores: Sequence[float],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> ClassificationStats:
    """Bin similarity and retrieval scores at ``threshold`` and report the book's metrics.

    ``scores`` are the predicted side (rewrite vs. human flashcard) and
    ``retrieval_scores`` the "ground truth" side (the index score), exactly the
    roles the notebook gives them.
    """
    scores = list(scores)
    retrieval_scores = list(retrieval_scores)
    if len(scores) != len(retrieval_scores):
        raise ValueError("scores and retrieval_scores must have the same length")
    if not scores:
        raise ValueError("cannot classify an empty score list")
    predicted = [1 if value > threshold else 0 for value in scores]
    true = [1 if value > threshold else 0 for value in retrieval_scores]
    tp = sum(1 for t, p in zip(true, predicted) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(true, predicted) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true, predicted) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(true, predicted) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(scores)
    return ClassificationStats(
        threshold=threshold,
        f1=f1,
        precision=precision,
        recall=recall,
        accuracy=accuracy,
        true_labels=true,
        predicted_labels=predicted,
        confusion=[[tn, fp], [fn, tp]],
    )


@dataclass(frozen=True)
class EvaluationItem:
    """One row of :meth:`VideoExpert.evaluate`: query, human flashcard, rewrite, scores."""

    query: str
    human: str
    retrieved: str
    rewritten: str
    retrieval_score: float
    similarity: float


@dataclass(frozen=True)
class EvaluationReport:
    """The book's evaluator output: per-item scores plus the descriptive/classification blocks."""

    items: list[EvaluationItem] = field(default_factory=list)
    threshold: float = DEFAULT_THRESHOLD

    @property
    def scores(self) -> list[float]:
        """The similarity scores (rewrite vs. human flashcard)."""
        return [item.similarity for item in self.items]

    @property
    def retrieval_scores(self) -> list[float]:
        """The index scores of the retrieved comments."""
        return [item.retrieval_score for item in self.items]

    @property
    def descriptive(self) -> DescriptiveStats:
        """The mean/median/spread block over :attr:`scores`."""
        return describe(self.scores)

    @property
    def classification(self) -> ClassificationStats:
        """The F1/precision/recall/accuracy block over :attr:`scores` and :attr:`retrieval_scores`."""
        return classify(self.scores, self.retrieval_scores, threshold=self.threshold)


class VideoExpert:
    """The chapter 10 RAG pipeline over the video-comment index (Pipeline 3)."""

    def __init__(
        self,
        settings: Settings,
        index,
        *,
        embedding_function: Optional[EmbeddingFunction] = None,
        llm: Optional[LLM] = None,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.settings = settings
        self.index = index
        self.embedding_function = embedding_function or create_embedding_function(settings)
        self.llm = llm or create_llm(settings)
        self.top_k = top_k
        self.threshold = threshold

    def retrieve(self, query: str) -> list[VideoMatch]:
        """Query the index for the nearest comments."""
        return query_video_index(
            self.index, self.embedding_function, query, top_k=self.top_k
        )

    def answer(self, query: str = DEFAULT_QUERY) -> VideoAnswer:
        """Retrieve the nearest comment, rewrite it, and label it (Pipeline 3, step 4)."""
        matches = self.retrieve(query)
        retrieved = matches[0].text if matches else ""
        # The book feeds the retrieved text straight in as the user prompt.
        prompt = retrieved
        start = time.perf_counter()
        response = self.llm.complete(prompt, system=VIDEO_EXPERT_SYSTEM, temperature=REWRITE_TEMPERATURE)
        elapsed = time.perf_counter() - start
        return VideoAnswer(
            query=query,
            matches=matches,
            retrieved=retrieved,
            prompt=prompt,
            response=response,
            rewritten=extract_rewritten_comment(response),
            label=extract_label(response),
            elapsed=elapsed,
        )

    def evaluate(
        self,
        sample: Sequence[tuple[str, str]] = EVALUATION_SAMPLE,
        *,
        limit: Optional[int] = None,
        progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> EvaluationReport:
        """Run the book's Pipeline 3 evaluation loop over ``sample``.

        For each ``(query, human flashcard)``: retrieve the nearest comment (its
        index score is the ground-truth side), rewrite it, and measure the
        embedding cosine between the rewrite and the flashcard.
        """
        items_pairs = list(sample)[:limit] if limit else list(sample)
        if not items_pairs:
            raise ValueError("cannot evaluate an empty sample")
        items: list[EvaluationItem] = []
        for position, (query, human) in enumerate(items_pairs, start=1):
            address = self.answer(query)
            similarity = text_similarity(human, address.rewritten or address.response, self.embedding_function)
            items.append(
                EvaluationItem(
                    query=query,
                    human=human,
                    retrieved=address.retrieved,
                    rewritten=address.rewritten or address.response,
                    retrieval_score=address.matches[0].score if address.matches else 0.0,
                    similarity=similarity,
                )
            )
            if progress is not None:
                progress(position, len(items_pairs), query)
        return EvaluationReport(items=items, threshold=self.threshold)


__all__ = [
    "VIDEO_BASE_URL",
    "VIDEO_DIRECTORY",
    "COMMENTS_DIRECTORY",
    "VIDEO_FILES",
    "COMMENT_FILES",
    "DEFAULT_VIDEO",
    "DEFAULT_SAMPLE_RATE",
    "VISION_QUESTION",
    "VISION_MAX_TOKENS",
    "COMMENT_COLUMNS",
    "VIDEO_EXPERT_SYSTEM",
    "REWRITE_TEMPERATURE",
    "DEFAULT_THRESHOLD",
    "DEFAULT_QUERY",
    "DEFAULT_TOP_K",
    "EVALUATION_SAMPLE",
    "VIDEO_UPSERT_BATCH",
    "VideoInfo",
    "FrameComment",
    "VideoMatch",
    "VideoIndexResult",
    "VideoAnswer",
    "DescriptiveStats",
    "ClassificationStats",
    "EvaluationItem",
    "EvaluationReport",
    "download",
    "download_video",
    "download_comments",
    "extract_frames",
    "sample_positions",
    "video_info",
    "thumbnail",
    "comment_frame",
    "save_comments",
    "read_comments",
    "load_comments",
    "comment_video",
    "build_chunks",
    "duplicate_chunks",
    "open_video_index",
    "upsert_comments",
    "query_video_index",
    "extract_rewritten_comment",
    "extract_label",
    "describe",
    "classify",
    "VideoExpert",
]
