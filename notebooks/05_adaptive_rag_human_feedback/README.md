# Chapter 5 — Adaptive RAG with Human Feedback

*Part of [**my_rag**](../../README.md) — the root README has setup and the chapter map.*

Using human feedback signals to refine retrieval accuracy dynamically.

Source notebook:
- [Adaptive_RAG.ipynb](https://github.com/Denis2054/RAG-Driven-Generative-AI/blob/main/Chapter05/Adaptive_RAG.ipynb)

Supporting data: `human_feedback.txt`, `raw_markdown_rag.txt`.
Tools: OpenAI API.

## Implementation

Implemented in `my_rag/adaptive.py` instead of a notebook. It mirrors the book's
three phases and its adaptivity:

- **Retriever** (`retrieve`) — a keyword→Wikipedia map (`ADAPTIVE_SOURCES`):
  `prompt engineering`, `artificial intelligence`, `llm`/`llms`. The matched page
  is fetched and cleaned with `my_rag.collection` (requests + BeautifulSoup), and
  its first `num_words` words become the augmented document wrapped in the book's
  prompt.
- **Generator** (`AdaptiveRAG`) — the panel's 1–5 *ranking* of earlier answers
  picks what the model sees, as the notebook's section 2.3 does:

  | Ranking | `--ranking` | Input to the LLM |
  |---|---|---|
  | 1–2 | `--ranking 1`/`2` | the bare query — **no RAG** |
  | 3–4 | `--ranking 3`/`4` | the **human-expert feedback** flashcard (`HUMAN_FEEDBACK`) |
  | 5 | `--ranking 5` | the **retrieved document** — RAG only, no feedback |

- **Evaluator** (`evaluate`) — response time, a TF-IDF cosine between the input and
  the answer (the book's `calculate_cosine_similarity`), and the human-rating loop
  (`RatingState`: `counter` / `score_history` / mean) that flags when a human
  expert should review the responses.

Only ranking 5 needs the network (the Wikipedia fetch); rankings 1–4 run offline.

```bash
python -m my_rag --rag adaptive --ranking 5 "What is an LLM?"        # retrieved document
python -m my_rag --rag adaptive --ranking 3 "What is an LLM?"        # human feedback only
python -m my_rag --rag adaptive --ranking 1 "What is an LLM?"        # no RAG
python -m my_rag --rag adaptive --ranking 5 --num-words 50 "What is an LLM?"   # shorter excerpt
python -m my_rag --rag adaptive --ranking 1 --rating 3 "What is an LLM?"       # record a rating
```

The answer is printed, then the strategy, the response time and the cosine score.
With `--rating 1..5` the book's rating loop is recorded too — seeded from the
notebook's values (20 earlier rankings, 60 points) so a single rating reproduces
its output: `Evaluator Score: 3`, `Rankings: 21`, `Score history: 3.0`, and the
`Human expert evaluation is required for the feedback loop.` hint when it fires.
`save_expert_feedback()` / `load_expert_feedback()` write and read the book's
`expert_feedback.txt` at `data/processed/adaptive/expert_feedback.txt`.

Substitutions, consistent with the rest of `my_rag`:

- **Local LLM instead of GPT-4o.** The default is `qwen2.5:3b` through Ollama; the
  book's system prompt (*You are an expert Natural Language Processing exercise
  expert.*) and instruction (*Please summarize or elaborate on the following
  content:*) are kept verbatim. `--provider openai` uses `gpt-4o`.
- **Fetching reused from chapter 2.** `my_rag.collection.fetch_article` /
  `parse_article` replace a second copy of the downloader; the book extracts only
  `<p>` tags, while chapter 2 keeps the whole article body (a slightly longer
  excerpt, same source).
- **TF-IDF cosine on numpy.** `my_rag.retrieval.TfidfIndex` replaces
  scikit-learn's `TfidfVectorizer` + `cosine_similarity` for the evaluator score.
- **The feedback flashcard is a constant.** The book downloads
  `human_feedback.txt`; here its text is the module constant `HUMAN_FEEDBACK`.
  The book's `raw_markdown_rag.txt` is the same RAG explainer already held as
  `corpus.DB_RECORDS` in chapter 1, so it is not duplicated.
- **The expert trigger is kept verbatim.** The book compares the *cumulative*
  `score_history` (not its mean) to the threshold; `RatingState.expert_review_required`
  preserves that quirk.

Not ported: the Colab scaffolding — the Drive mount and `api_key.txt`, the
`grequests.download` helper, the `display(Image(...))` of `rag_strategy.png`, the
in-browser thumbs-up/down widgets with their base64 images, and the interactive
`input()` rating (replaced by `--rating`).

## Tests

`tests/test_adaptive.py` covers the keyword match, the ranking→strategy map, the
retrieval excerpt, each strategy's input, the TF-IDF score, the rating loop and
the CLI — with the fetch and the model replaced by fakes, so no network or Ollama
is needed:

```bash
python -m unittest tests.test_adaptive -v
```
