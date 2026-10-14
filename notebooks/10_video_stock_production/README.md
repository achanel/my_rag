# Chapter 10 — RAG for Video Stock Production (Pinecone + OpenAI)

*Part of [**my_rag**](../../README.md) — the root README has setup and the chapter map.*

Turn raw stock videos into a searchable library of *described frames*: split a
video into frames, ask a vision model what happens in each sampled frame, embed
the comments into a vector index, then answer a query by retrieving a comment and
rewriting it into an engaging, labelled shot description.

The book splits this across four notebooks — one visualization tour and three
pipelines. The book's index is **Pinecone** and its vision/generation model is
**GPT-4o Vision** (both paid/closed), so `my_rag` keeps the exact pipeline shape
but swaps:

- **Pinecone → Qdrant.** Chapter 6 already replaces Pinecone; the same
  key-free, embedded Qdrant engine backs the ``videos-sports-us`` collection
  (cosine metric, batch upsert). Only the vector size follows the embedding model
  (384 for the local `all-minilm`, vs. Pinecone's hard-coded 1536).
- **GPT-4o Vision → a configurable vision model.** `gemma3:12b` through Ollama by
  default; `--provider openai` uses `gpt-4o`. The frame travels as the same
  `data:image/jpeg;base64` URL.
- **The evaluator's sentence-transformers / spaCy → the configured embeddings**,
  and **scikit-learn metrics → numpy**, the same dependency-ladder choice as
  chapters 5 and 7.

Source notebooks:
- [Video_dataset_visualization.ipynb](https://github.com/Denis2054/RAG-Driven-Generative-AI/blob/main/Chapter10/Video_dataset_visualization.ipynb)
- [Pipeline_1_Generator_and_Commentator.ipynb](https://github.com/Denis2054/RAG-Driven-Generative-AI/blob/main/Chapter10/Pipeline_1_Generator_and_Commentator.ipynb)
- [Pipeline_2_The_Vector_Store_Administrator.ipynb](https://github.com/Denis2054/RAG-Driven-Generative-AI/blob/main/Chapter10/Pipeline_2_The_Vector_Store_Administrator.ipynb)
- [Pipeline_3_The_Video_Expert.ipynb](https://github.com/Denis2054/RAG-Driven-Generative-AI/blob/main/Chapter10/Pipeline_3_The_Video_Expert.ipynb)

Supporting data: the book's 21 stock videos and their ready-made comment CSVs
(`Chapter10/videos`, `Chapter10/comments`, public GitHub — no auth).
Tools: OpenCV (`opencv-python-headless`), a local vision model (Ollama
`gemma3:12b`), local Ollama generation/embeddings, the local Qdrant engine. No
token or paid service is required; `--provider openai` is optional.

## Implementation

Implemented in `my_rag/video.py`, with a 1:1 step mapping onto the notebooks:

- **Visualization.** `video_info()` reports frame count, frame rate and duration
  (`cv2.CAP_PROP_*`); `thumbnail()` saves a resized frame, the notebook's
  "display a frame of video" cell.
- **Pipeline 1 — the generator and commentator.** `extract_frames()` splits a
  video into `frame_<n>.jpg` (`cv2.VideoCapture`, imported lazily);
  `sample_positions()` reproduces the notebook's `nb` counter (every 3rd frame:
  2, 5, 8, …); `comment_frame()` asks the vision model *"What is happening in
  this image?"*; `comment_video()` ties it together and `save_comments()` writes
  the `ID,FrameNumber,Comment,FileName` CSV the notebook builds.
- **Pipeline 2 — the vector store administrator.** `load_comments()` reads the
  comment CSVs (skipping the header, unlike the notebook's `header=None`),
  `build_chunks()` treats each comment as one chunk, `upsert_comments()` embeds
  them and upserts each with its `text`/`frame_number`/`file_name` payload into
  the Qdrant collection, and `query_video_index()` runs the book's `k=1` query.
- **Pipeline 3 — the video expert.** `VideoExpert.answer()` retrieves the nearest
  comment, feeds it as the prompt under the book's rewrite-and-label system
  message (`temperature=0`), and parses the reply with
  `extract_rewritten_comment()` / `extract_label()`. `VideoExpert.evaluate()`
  runs the book's sample: for each query it rewrites the hit and scores the
  rewrite against the human-feedback flashcard (`describe()` for the
  mean/median/spread block, `classify()` for F1/precision/recall/accuracy and the
  confusion matrix, binarized at the book's `0.6`).

```bash
# Visualization — inspect a video and save a thumbnail (needs OpenCV)
python -m my_rag --video-info --video alpinist1.mp4
python -m my_rag --collect --corpus videos --limit 3   # download the videos (heavy)

# Pipeline 1 — comment a video (needs OpenCV + a vision model)
python -m my_rag --comment-video --video alpinist1.mp4          # 6 frames, every 3rd
python -m my_rag --comment-video --video alpinist1.mp4 --sample 5 --limit 10
python -m my_rag --comment-video --video alpinist1.mp4 --limit 0  # the whole video
# the book's ready-made comments, so the index below needs no video/vision:
python -m my_rag --collect --corpus video-comments

# Pipeline 2 — embed the comments into Qdrant (local, no key)
python -m my_rag --build-video-index
python -m my_rag --build-video-index --limit 200

# Pipeline 3 — the video expert
python -m my_rag --rag video "Find a basketball player that is scoring with a dunk."
python -m my_rag --rag video --show-context "Show a surfer."
python -m my_rag --rag video --evaluate                 # the book's 10-query evaluator
```

`--build-video-index` embeds the comment CSVs and upserts them:

```text
building Qdrant collection videos-sports-us from 21 comment files in data/raw/video_comments ...
upserted 500 vectors (500 comments, 384-dim) into videos-sports-us
```

`--rag video` retrieves a comment, rewrites it and prints the label:

```text
Response:
---------------
1. No cognitive dissonance. ...
Rewritten Comment:
A basketball player in mid-air slams the ball through the hoop ...
Label: basketball
---------------
Label: basketball
Matches: 1 | Response Time: 0.42 seconds
```

`--rag video --evaluate` prints the book's Pipeline-3 metrics:

```text
Evaluation over 10 queries:
  Mean: 0.63
  Median: 0.66
  ...
Classification (threshold 0.6):
  F1 Score: 0.80
  Precision: 0.80
  Recall: 1.00
  Accuracy: 0.80
  Confusion matrix: TN=1 FP=1 FN=0 TP=8
```

Artifacts written (all git-ignored):

```bash
rm -rf data/raw/videos data/raw/video_comments data/raw/video_frames
rm -rf data/processed/qdrant          # also holds chapter 6's collection
```

Substitutions, consistent with the rest of `my_rag`:

- **Qdrant instead of Pinecone.** The book creates a Pinecone serverless index
  (`videos-sports-us`, `dimension=1536`, `metric='cosine'`, an API key and a
  cloud/region). `open_video_index()` reuses chapter 6's
  `open_qdrant_index`, so the same collection name and cosine metric ride on the
  embedded, key-free Qdrant engine; `MY_RAG_EMBEDDING_DIMENSION` sets the size.
- **A configurable vision model instead of GPT-4o.** `comment_frame()` sends the
  frame as a `data:image/jpeg;base64` URL exactly like the notebook, but the
  model is `settings.vision_model` (`gemma3:12b` for Ollama, `gpt-4o` for OpenAI).
- **The book's comment CSVs, not a video download per run.** The vector-store and
  expert pipelines run from the ready-made `Chapter10/comments` files; a video is
  downloaded (and OpenCV is required) only for `--comment-video`.
- **A header-aware reader.** The notebook reads the CSVs with `header=None`, which
  would keep the header row as data; `read_comments()` drops it, so only real
  comments are indexed.
- **Embeddings instead of sentence-transformers / spaCy, numpy instead of
  scikit-learn.** The evaluator's `calculate_cosine_similarity_with_embeddings`
  becomes `text_similarity` over the configured model, and the F1 / precision /
  recall / accuracy / confusion-matrix block is plain numpy.
- **A practical frame cap.** The book comments every third frame of its ~208-frame
  videos (≈69 vision calls). A local vision model takes tens of seconds per frame,
  so `MY_RAG_VIDEO_LIMIT` (default 6) caps how many sampled frames
  `--comment-video` describes and `--limit` scales it (``--limit 0`` = every
  sampled frame); `--sample` changes the frequency and `--limit` also caps how
  many comments the index ingests.
- **`--comment-video` writes the book's file name.** It saves
  ``<video>.csv`` (e.g. ``alpinist1.mp4.csv``) in the comments dir, exactly the
  name the book's ``save_comment()`` uses — so it *replaces* a downloaded CSV for
  the same video. Pass ``--output`` to keep both.

Not ported: the Colab scaffolding (the Drive mount, `api_key.txt` /
`pinecone.txt` key files, the `!pip install` pins), the notebook's inline
`display()`/`HTML()` cells (the videos and frames are files here), the book's
`save`/`save_frames` copy-to-Drive cells (frames already land in
`data/raw/video_frames`) and the fixed static-similarity demo cell (the
evaluator covers it).

## Configuration

`MY_RAG_VIDEO_DIR` (video downloads), `MY_RAG_VIDEO_COMMENTS` (comment CSVs),
`MY_RAG_VIDEO_FRAMES` (split frames and thumbnails),
`MY_RAG_VIDEO_COLLECTION` (default `videos-sports-us`), `MY_RAG_VIDEO_SAMPLE`
(default 3) and `MY_RAG_VIDEO_LIMIT` (default 6 sampled frames per
`--comment-video` run; 0 = all) override the chapter's defaults. `--video`,
`--sample`, `--limit`, `--video-dir`, `--comments-dir`, `--frames-dir` and
`--collection` override them per run. The index reuses chapter 6's Qdrant settings
(`MY_RAG_QDRANT_PATH` / `QDRANT_URL`) and the embedding model
(`MY_RAG_EMBEDDING_MODEL` / `MY_RAG_EMBEDDING_DIMENSION`); the vision model is
`MY_RAG_VISION_MODEL`.

Frames need **OpenCV** — `opencv-python-headless` in `requirements.txt` (free,
local, no key). Everything else the chapter needs is already there: the vision
and generation models run through **Ollama** (`gemma3:12b`, `qwen2.5:3b`) and the
embeddings through `all-minilm`; the index is the **local Qdrant** engine. No
token, account or paid service is used, and the book's videos/comments come from
its public GitHub repo (no auth).

## Tests

`tests/test_video.py` covers the comment CSV round-trip and header skip, the
book's frame sampler, the downloads (with a fake session, including a skipped
404), `comment_video` (frames and vision client faked), the upsert payload and a
real local Qdrant round-trip, the rewrite/label parsers, `VideoExpert`
answer/evaluate, the descriptive and classification metrics, the config, and the
CLI (`--collect --corpus video-comments`, `--build-video-index`, `--rag video
[--evaluate]`, `--comment-video`). No video, OpenCV, vision call, key, network or
Qdrant server is needed:

```bash
python -m unittest tests.test_video -v
```
