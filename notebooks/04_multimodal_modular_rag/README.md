# Chapter 4 — Multimodal, Modular RAG

Combining text and image data in a modular RAG pipeline (drone-imagery use
case).

Source notebook:
- [Multimodal_Modular_RAG_Drones.ipynb](https://github.com/Denis2054/RAG-Driven-Generative-AI/blob/main/Chapter04/Multimodal_Modular_RAG_Drones.ipynb)

Tools: OpenAI (vision + text), Deep Lake.

## Implementation

Implemented in `my_rag/multimodal.py` instead of a notebook. It mirrors the book's
three modules and its scoring:

- **Text module** — a semantic index over the drone text corpus; the LLM answers from
  the two best chunks (the chapter 1 `RAGPipeline`).
- **Image module** — VisDrone images, each indexed by the object classes in its
  bounding boxes; the image that best matches the question is retrieved.
- **Vision module** — the retrieved image is redrawn with red boxes around the object
  the question names, and a vision model is asked about those boxes.
- **Scores** — cosine similarity between the question and the text answer, and
  between the question (plus the object, pluralised) and the vision answer; the
  modular score is their mean, as in the book.

It needs a vision model and two collections, the second one from chapter 3:

```bash
ollama pull gemma3:12b                                  # vision model (~8 GB)
python -m my_rag --collect --corpus drone               # text corpus -> data/raw/drone.md (ch. 3)
python -m my_rag --collect --corpus visdrone --limit 20 # images + annotations -> data/raw/drone_images
```

Then answer the book's question:

```bash
python -m my_rag --rag multimodal "How do drones identify a truck?"
python -m my_rag --rag multimodal --show-context "How do drones identify a truck?"
python -m my_rag --rag multimodal --vision-model gemma3:4b "How do drones identify a truck?"
python -m my_rag --rag multimodal "How do drones identify a car?"
```

`--show-context` prints the text chunks and the matched image with its classes. The
boxed image is saved to `data/processed/multimodal/boxed_<id>.jpg`; open it to see
which objects the vision model was asked about. `--limit` sets how many VisDrone
samples are saved (the book's dataset has 6,471; 20 covers trucks, cars and
pedestrians). Re-running `--collect` overwrites the images.

Cleanup (chapter 3's `data/raw/drone.md` is shared with chapter 3):

```bash
rm -rf data/raw/drone_images data/processed/multimodal
```

Substitutions, consistent with the rest of `my_rag`:

- **Local vision model instead of GPT-4o.** The default is `gemma3:12b` through
  Ollama's OpenAI-compatible endpoint, the same one the text model uses. Set
  `MY_RAG_VISION_MODEL` or `--vision-model` to change it. `--provider openai` uses
  `gpt-4o` for both text and vision.
- **Local embeddings for retrieval and scores.** Retrieval and the cosine scores use
  Ollama's `all-minilm`, the all-MiniLM-L6-v2 family the book uses for its scores.
  The book's `sentence-transformers` dependency (it pulls in PyTorch) is not needed.
- **Images saved locally.** The book streams `hub://activeloop/visdrone-det-train`
  each run. `--collect --corpus visdrone` reads the first samples of that public
  dataset once (no `ACTIVELOOP_TOKEN`) and writes JPEG files plus
  `annotations.json`; Deep Lake is used only for that download.
- **Chapter 3's text corpus instead of `hub://denis76/drone_v2`.** The text module
  reads the web-collected drone corpus, split into 1000-character chunks as in chapter 2.
- **Image documents are the distinct classes.** The book repeats one label per box.
  The local `all-minilm` has a 512-token context, and a dense frame (211 boxes in
  VisDrone sample 10) would spend most of it on repeats; a class's presence is what
  retrieval needs.
- **The object comes from the question.** The book asks the vision model about
  `unique_words[0]`, the alphabetically first word of the retrieved text, which has
  no link to the question. The class the question names is used instead; if the
  matched image has none, the most frequent class is used and printed.
- **No colour conversion.** The book converts BGR→RGB with OpenCV. The arrays Deep
  Lake returns are already RGB (checked on the sample images), so Pillow saves them as is.

Not ported: the notebook's inline displays (`ds.visualize()`, `display(img)`), the
Colab setup (Drive key files, `resolv.conf` DNS workaround), and the book's
`get_unique_words` printout, which only served the `unique_words[0]` choice above.
