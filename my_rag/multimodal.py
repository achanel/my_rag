"""Multimodal, modular RAG over text and drone images (chapter 4).

Reimplementation of ``Chapter04/Multimodal_Modular_RAG_Drones.ipynb`` ("Multimodal
Multimodular RAG for Drone Technology"). The book answers one question from three
modules and averages two similarity scores:

- **text module** — a semantic index over the drone text corpus; the LLM answers
  from the top chunks (the chapter 1 pipeline, :class:`my_rag.pipeline.RAGPipeline`);
- **image module** — the VisDrone detection dataset. Each image is indexed by the
  object classes in its bounding boxes, and the best match for the query is picked;
- **vision module** — the matched image is redrawn with red boxes around the object
  class the question names, and a vision model is asked about those boxes.

The book's scores are cosine similarities between the question and each answer, and
the modular score is their mean. They are rough heuristics, kept for the comparison.

Substitutions, consistent with the rest of ``my_rag``:

- **Local vision model instead of GPT-4o.** The default is ``gemma3:12b`` through
  Ollama's OpenAI-compatible endpoint; ``--provider openai`` uses ``gpt-4o``.
- **Embeddings instead of OpenAI's, and no sentence-transformers.** The same
  ``all-minilm`` embedding (all-MiniLM-L6-v2 family) serves retrieval and the
  scores, so the metric needs no extra model download.
- **Images saved locally.** The book streams ``hub://activeloop/visdrone-det-train``
  on every run. :func:`collect_visdrone` reads the first samples once and writes JPEG
  files plus ``annotations.json``; Deep Lake is only needed for that download.
- **Query-driven target object.** The book asks the vision model about
  ``unique_words[0]``, the alphabetically first word of the retrieved text, which is
  not tied to the question. :func:`target_label` uses the class the query names.

Deep Lake, PIL and the OpenAI client are imported where they are used, so light
commands do not pay for them.
"""
from __future__ import annotations

import base64
import itertools
import json
import os
import re
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from openai import OpenAI

from .config import CHUNK_SIZE, DEFAULT_BOXED_DIR, DEFAULT_IMAGE_DIR, Settings
from .embeddings import EmbeddingFunction, EmbeddingIndex, cosine_similarity, create_embedding_function
from .llm import LLM, create_llm
from .pipeline import RAGPipeline
from .vectorstore import chunk_text

#: Deep Lake's analytics reporter can hang a short-lived CLI (see chapter 2).
os.environ.setdefault("BUGGER_OFF", "1")

#: Deep Lake prints a PyPI update banner on import; irrelevant CLI noise.
warnings.filterwarnings("ignore", message="A newer version of deeplake")

#: The book's image dataset: VisDrone-DET train split, public on Activeloop (no token).
VISDRONE_DATASET = "hub://activeloop/visdrone-det-train"

#: Samples saved by ``--collect --corpus visdrone``; the dataset has 6,471.
DEFAULT_IMAGE_LIMIT = 20

#: ``similarity_top_k`` of the book's text and image modules.
TEXT_TOP_K = 2
IMAGE_TOP_K = 1

#: Annotation file written next to the images.
ANNOTATIONS_FILE = "annotations.json"

#: VisDrone classes that are not objects of interest; never the object a question is about.
_UNANNOTATED = frozenset({"ignored regions", "others"})


@dataclass(frozen=True)
class Sample:
    """One annotated drone image: its file, boxes ``[x, y, w, h]`` in pixels, and class per box."""

    id: str
    path: Path
    boxes: list[list[float]]
    labels: list[str]

    def label_text(self) -> str:
        """The image's retrieval document: its distinct classes, most frequent first.

        The book joins one label per box. The local ``all-minilm`` has a 512-token
        context, and a dense frame (211 boxes in VisDrone sample 10) would spend most
        of it on repeats; for retrieval only the presence of a class matters. An image
        without boxes gets a placeholder, since an empty string cannot be embedded.
        """
        return " ".join(label for label, _ in Counter(self.labels).most_common()) or "no objects"


def collect_visdrone(
    image_dir: Path | str = DEFAULT_IMAGE_DIR,
    limit: int = DEFAULT_IMAGE_LIMIT,
    *,
    progress: Optional[Callable[[str], None]] = None,
) -> list[Sample]:
    """Save the first ``limit`` VisDrone samples as JPEG files plus ``annotations.json``.

    Mirrors the book's cells that read ``sample.images``, ``sample.boxes`` and
    ``sample.labels``. The arrays Deep Lake returns are already RGB, so no BGR→RGB
    conversion is needed (the book's OpenCV step).
    """
    import deeplake  # lazy: Deep Lake is a heavy import
    from PIL import Image

    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    dataset = deeplake.load(VISDRONE_DATASET, read_only=True, verbose=False)
    count = min(limit, len(dataset))
    records = []
    # Iterating the dataset is much faster than ds[i] per index (Deep Lake's own advice).
    for index, sample in enumerate(itertools.islice(dataset, count)):
        file_name = f"{index:05d}.jpg"
        Image.fromarray(sample.images.numpy()).save(image_dir / file_name, quality=90)
        boxes = np.asarray(sample.boxes.numpy(), dtype=float).reshape(-1, 4)
        labels = list(sample.labels.data()["text"])
        records.append(
            {
                "id": f"{index:05d}",
                "file": file_name,
                "boxes": boxes.tolist(),
                "labels": labels,
            }
        )
        if progress is not None:
            progress(f"[{index + 1}/{count}] {file_name}: {len(labels)} objects")

    (image_dir / ANNOTATIONS_FILE).write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return load_samples(image_dir)


def load_samples(image_dir: Path | str = DEFAULT_IMAGE_DIR) -> list[Sample]:
    """Read the annotated images written by :func:`collect_visdrone`."""
    image_dir = Path(image_dir)
    annotations = image_dir / ANNOTATIONS_FILE
    if not annotations.is_file():
        raise FileNotFoundError(
            f"annotations {annotations} not found; collect them with: "
            "python -m my_rag --collect --corpus visdrone"
        )
    samples = []
    for record in json.loads(annotations.read_text(encoding="utf-8")):
        if len(record["boxes"]) != len(record["labels"]):
            raise ValueError(
                f"{record['file']}: {len(record['boxes'])} boxes but {len(record['labels'])} labels"
            )
        samples.append(
            Sample(
                id=record["id"],
                path=image_dir / record["file"],
                boxes=record["boxes"],
                labels=record["labels"],
            )
        )
    if not samples:
        raise ValueError(f"{annotations} lists no samples")
    return samples


def load_text_chunks(path: Path | str) -> list[str]:
    """Read the text corpus (chapter 3's ``drone.md``) and split it into 1000-character chunks."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"corpus {path} not found; collect it with: python -m my_rag --collect --corpus drone"
        )
    chunks = [chunk for chunk in chunk_text(path.read_text(encoding="utf-8"), CHUNK_SIZE) if chunk.strip()]
    if not chunks:
        raise ValueError(f"corpus {path} produced no chunks")
    return chunks


def target_label(query: str, sample: Sample) -> str:
    """The object class the vision model is asked about.

    A class the query names ("truck", also "trucks") wins, the most frequent one
    among them. Otherwise the sample's most frequent class is used, so the answer
    can differ from the question — the caller reports which class was analysed.
    """
    counts = Counter(label for label in sample.labels if label not in _UNANNOTATED)
    if not counts:
        raise ValueError(f"image {sample.id} has no annotated objects")
    named = [
        label
        for label in counts
        if re.search(rf"\b{re.escape(label)}s?\b", query, flags=re.IGNORECASE)
    ]
    return max(named or list(counts), key=counts.__getitem__)


def draw_boxes(sample: Sample, label: str, output: Path | str) -> Path:
    """Save the sample's image with a red box around every object of class ``label``.

    Boxes are VisDrone's ``[x, y, w, h]`` in pixels, as in the book's display function.
    """
    from PIL import Image, ImageDraw

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(sample.path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for (x, y, w, h), name in zip(sample.boxes, sample.labels):
        if name == label:
            draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
            draw.text((x, y), label, fill="red")
    image.save(output, quality=90)
    return output


def image_data_url(path: Path | str) -> str:
    """The JPEG file as a ``data:`` URL, the vision input format of OpenAI-compatible servers."""
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def describe_objects(
    image_url: str,
    label: str,
    settings: Settings,
    *,
    client: OpenAI,
) -> str:
    """Ask the vision model about the boxed objects of class ``label`` (the book's vision call)."""
    response = client.chat.completions.create(
        model=settings.vision_model,
        messages=[
            {
                "role": "system",
                "content": f"You are a helpful assistant that analyzes images that contain {label}.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Analyze the following image, tell me if there is one {label} or more "
                        "in the bounding boxes and analyse them:",
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def similarity(embedding_function: EmbeddingFunction, first: str, second: str) -> float:
    """Cosine similarity of two texts under the embedding model (the book's metric)."""
    vectors = embedding_function.embed_documents([first, second])
    return cosine_similarity(vectors[0], vectors[1])


@dataclass(frozen=True)
class MultimodalAnswer:
    """Outcome of :meth:`MultimodalRAG.answer`: both modules' output and their scores."""

    text_response: str
    text_context: list[str]
    sample: Sample
    label: str
    boxed_image: Path
    vision_response: str
    text_score: float
    vision_score: float

    @property
    def modular_score(self) -> float:
        """The book's combined score: the mean of the text and vision scores."""
        return (self.text_score + self.vision_score) / 2


class MultimodalRAG:
    """Answer a question from the text corpus and the drone images together."""

    def __init__(
        self,
        settings: Settings,
        samples: Sequence[Sample],
        text_chunks: Sequence[str],
        *,
        embedding_function: Optional[EmbeddingFunction] = None,
        llm: Optional[LLM] = None,
        vision_client: Optional[OpenAI] = None,
        boxed_dir: Path | str = DEFAULT_BOXED_DIR,
    ) -> None:
        self.settings = settings
        self.samples = list(samples)
        self.embedding_function = embedding_function or create_embedding_function(settings)
        self.llm = llm or create_llm(settings)
        self.vision_client = vision_client or OpenAI(base_url=settings.base_url, api_key=settings.api_key)
        self.boxed_dir = Path(boxed_dir)
        self.text_index = EmbeddingIndex(text_chunks, self.embedding_function, top_k=TEXT_TOP_K)
        self.image_index = EmbeddingIndex(
            [sample.label_text() for sample in self.samples],
            self.embedding_function,
            top_k=IMAGE_TOP_K,
        )

    def answer(self, query: str) -> MultimodalAnswer:
        """Run the text module, the image module, and the vision module on ``query``."""
        text_response = RAGPipeline(self.llm, self.text_index).answer(query)
        text_context = self.text_index.retrieve(query)

        position, _ = self.image_index.search(query)[0]
        sample = self.samples[position]
        label = target_label(query, sample)
        boxed = draw_boxes(sample, label, self.boxed_dir / f"boxed_{sample.id}.jpg")
        vision_response = describe_objects(
            image_data_url(boxed), label, self.settings, client=self.vision_client
        )

        return MultimodalAnswer(
            text_response=text_response,
            text_context=text_context,
            sample=sample,
            label=label,
            boxed_image=boxed,
            vision_response=vision_response,
            text_score=similarity(self.embedding_function, query, text_response),
            # The book compares the vision answer with the question plus the object, pluralised.
            vision_score=similarity(self.embedding_function, f"{query} {label}s", vision_response),
        )
