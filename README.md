# my_rag

RAG playground, structured after
[RAG-Driven-Generative-AI](https://github.com/Denis2054/RAG-Driven-Generative-AI)
(Denis Rothman, Packt) for study purposes — one folder per chapter/topic,
each with a `README.md` pointing at the reference notebook and the tools it
uses. Notebook content is written locally as you work through the book, not
copied from the source repo.

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY / ACTIVELOOP_TOKEN / PINECONE_API_KEY
```

## Layout

- `notebooks/01_rag_overview` … `notebooks/10_video_stock_production` — one
  per chapter of the reference book, see each folder's `README.md`.
- `commons/` — shared helpers (API keys, HTTP sessions) extracted as you go.
- `data/raw`, `data/processed` — input and derived datasets (git-ignored).
- `models/` — local model checkpoints / cache (git-ignored).

## Key dependencies

`deeplake`, `openai`, `transformers`, `numpy` — core RAG stack from the
reference book. `accelerate`, `deepspeed`, `bitsandbytes`, `neural_compressor`,
`onnx` are for local Hugging Face model training/fine-tuning and inference
optimization if you extend beyond the book's notebooks (note: `deepspeed`/
`bitsandbytes` target Linux+CUDA primarily — expect limited functionality on
macOS). `pandas`, `scipy`, `beautifulsoup4`, `requests` cover data prep and
scraping.
