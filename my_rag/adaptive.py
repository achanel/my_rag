"""Adaptive RAG with human feedback (chapter 5).

Reimplementation of ``Chapter05/Adaptive_RAG.ipynb`` ("Hybrid Adaptive
RAG-driven Generative AI"). The notebook ties three phases into one loop:

- **Retriever** — a keyword→Wikipedia map ("prompt engineering", "artificial
  intelligence", "llm"/"llms"). The matched page is fetched, cleaned, and its
  first ``num_words`` words become the augmented document, wrapped in the
  book's prompt.
- **Generator** — the *ranking* a panel gave earlier answers adapts what the
  model sees: a low rank (1–2) sends the bare query (no RAG), a middle rank
  (3–4) sends a human-expert feedback flashcard, the top rank (5) sends the
  retrieved document. The LLM then summarizes/elaborates that input.
- **Evaluator** — response time, a TF-IDF cosine between input and answer, and
  the human rating loop (``counter`` / ``score_history`` / mean) that decides
  when human-expert feedback is needed.

Substitutions, consistent with the rest of ``my_rag``:

- **Free local Ollama by default** (``qwen2.5:3b``) instead of GPT-4o; the
  system/instruction prompts are the book's.
- **Fetching reuses :mod:`my_rag.collection`** (``requests`` + BeautifulSoup)
  instead of a second copy of the downloader, and **the TF-IDF cosine reuses
  :class:`my_rag.retrieval.TfidfIndex`** (numpy) instead of scikit-learn.
- **The human feedback flashcard is a module constant** (the book ships it as
  ``human_feedback.txt``), and the expert-feedback loop can write a local file.

Not ported: the Colab scaffolding (Drive mount, ``google.colab`` thumbs-up/down
widgets and their base64 images, the interactive ``input()`` rating), and the
book's ``raw_markdown_rag.txt`` sample, which duplicates the chapter 1
``corpus.DB_RECORDS`` text.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .collection import Article, fetch_article
from .config import DEFAULT_EXPERT_FEEDBACK
from .llm import LLM, create_llm
from .pipeline import RAGPipeline
from .retrieval import TfidfIndex

#: Keyword → Wikipedia page, in the book's order. ``llm`` deliberately precedes
#: ``llms``: the match is a substring test, so "what is an llm" hits ``llm``.
ADAPTIVE_SOURCES: dict[str, str] = {
    "prompt engineering": "https://en.wikipedia.org/wiki/Prompt_engineering",
    "artificial intelligence": "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "llm": "https://en.wikipedia.org/wiki/Large_language_model",
    "llms": "https://en.wikipedia.org/wiki/Large_language_model",
}

#: The book's ``human_feedback.txt``: domain feedback for the C-phone product,
#: injected when the ranking is middling (3–4) instead of the retrieved page.
HUMAN_FEEDBACK = (
    "A Large Language Model (LLM) is an advanced AI system trained on vast amounts of text "
    "data to generate human-like text responses. It understands and generates language based on "
    "the patterns and information it has learned during training. LLMs are highly effective in "
    "various language-based tasks, including answering questions, making recommendations, and "
    "facilitating conversations. They can be continually updated with new information and trained "
    "to understand specific domains or industries. For the C-phone series customer support, "
    "incorporating an LLM could significantly enhance service quality and efficiency. The "
    "conversational agent powered by an LLM can provide instant responses to customer inquiries, "
    "reducing wait times and freeing up human agents for more complex issues. It can be programmed "
    "to handle common technical questions about the C-phone series, troubleshoot problems, guide "
    "users through setup processes, and offer tips for optimizing device performance. Additionally, "
    "it can be used to gather customer feedback, providing valuable insights into user experiences "
    "and product performance. This feedback can then be used to improve products and services. "
    "Furthermore, the LLM can be designed to escalate issues to human agents when necessary, "
    "ensuring that customers receive the best possible support at all levels. The agent can also "
    "provide personalized recommendations for customers based on their usage patterns and "
    "preferences, enhancing user satisfaction and loyalty."
)

#: The book's instruction (chapter 1's ``RAGPipeline`` default adds only "elaborate").
ADAPTIVE_INSTRUCTION = "Please summarize or elaborate on the following content:"

#: How the ranking picks the generator input.
STRATEGY_NONE = "none"          # ranking 1–2: bare query, no RAG
STRATEGY_FEEDBACK = "feedback"  # ranking 3–4: human-expert feedback only
STRATEGY_DOCUMENT = "rag"       # ranking 5: retrieved document only

#: Word limit on the retrieved page (the book's ``max_words``).
DEFAULT_NUM_WORDS = 100

#: Default ranking: 5, the case the book's notebook runs (RAG without feedback).
DEFAULT_RANKING = 5

#: The book's query and its feedback-loop seed values.
DEFAULT_QUERY = "What is an LLM?"
COUNTER_SEED = 20
SCORE_HISTORY_SEED = 60
DEFAULT_COUNTER_THRESHOLD = 10
DEFAULT_SCORE_THRESHOLD = 4

#: A fetcher takes a URL and returns a cleaned :class:`~my_rag.collection.Article`.
Fetcher = Callable[[str], Article]


class NoMatchError(ValueError):
    """The query named none of the retriever's keywords."""


def match_keyword(query: str) -> Optional[str]:
    """Return the first source keyword contained in ``query`` (case-insensitive)."""
    lowered = query.lower()
    return next((keyword for keyword in ADAPTIVE_SOURCES if keyword in lowered), None)


def strategy_for_ranking(ranking: int) -> str:
    """Map a 1–5 human ranking to its adaptive strategy, as the book does."""
    if not 1 <= ranking <= 5:
        raise ValueError(f"ranking must be between 1 and 5, got {ranking}")
    if ranking <= 2:
        return STRATEGY_NONE
    if ranking <= 4:
        return STRATEGY_FEEDBACK
    return STRATEGY_DOCUMENT


def first_words(text: str, num_words: int = DEFAULT_NUM_WORDS) -> str:
    """Return the first ``num_words`` whitespace-separated words of ``text``."""
    if num_words < 1:
        raise ValueError(f"num_words must be at least 1, got {num_words}")
    return " ".join(text.split()[:num_words])


@dataclass(frozen=True)
class Retrieved:
    """One fetched source document and the excerpt/prompt built from it."""

    keyword: str
    url: str
    text: str
    excerpt: str
    prompt: str


def retrieve(
    query: str,
    num_words: int = DEFAULT_NUM_WORDS,
    *,
    fetcher: Fetcher = fetch_article,
) -> Optional[Retrieved]:
    """Fetch and clean the page for ``query``'s keyword, or ``None`` if none matches.

    Mirrors the book's ``process_query``: the first ``num_words`` words of the
    cleaned text become both the display excerpt and the prompt context.
    """
    keyword = match_keyword(query)
    if keyword is None:
        return None
    article = fetcher(ADAPTIVE_SOURCES[keyword])
    excerpt = first_words(article.text, num_words)
    return Retrieved(
        keyword=keyword,
        url=article.url,
        text=article.text,
        excerpt=excerpt,
        prompt=f"{ADAPTIVE_INSTRUCTION}\n{excerpt}",
    )


@dataclass(frozen=True)
class AdaptiveResult:
    """Outcome of :meth:`AdaptiveRAG.answer`: the chosen input and the answer."""

    query: str
    ranking: int
    strategy: str
    source: Optional[str]
    text_input: str
    response: str
    elapsed: float


@dataclass
class RatingState:
    """The book's human-rating loop: ``counter`` ratings totalling ``score_history``.

    Seeded with the book's values (20 earlier rankings, 60 accumulated points) so
    a single recorded score reproduces its printed mean.
    """

    counter: int = COUNTER_SEED
    score_history: float = SCORE_HISTORY_SEED

    def record(self, score: int) -> "RatingState":
        """Add one human score (1–5) and return ``self`` for chaining."""
        if not 1 <= score <= 5:
            raise ValueError(f"score must be between 1 and 5, got {score}")
        self.counter += 1
        self.score_history += score
        return self

    @property
    def mean(self) -> float:
        """Mean score so far (the book's ``mean_score``)."""
        return round(self.score_history / self.counter, 2) if self.counter else 0.0

    def expert_review_required(
        self,
        counter_threshold: int = DEFAULT_COUNTER_THRESHOLD,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> bool:
        """Whether the loop should call in a human expert.

        Kept verbatim from the book, which compares the *cumulative*
        ``score_history`` (not its mean) to the threshold — a quirk of the
        notebook, so the trigger only fires for a large, very low-scoring panel.
        """
        return self.counter > counter_threshold and self.score_history <= score_threshold


@dataclass(frozen=True)
class Evaluation:
    """Outcome of :func:`evaluate`: input/answer similarity and the rating loop."""

    similarity: float
    rating: Optional[int] = None
    state: Optional[RatingState] = None

    @property
    def mean_score(self) -> Optional[float]:
        """The loop's mean score, or ``None`` when no rating was recorded."""
        return None if self.state is None else self.state.mean

    @property
    def expert_review_required(self) -> bool:
        """Whether the rating loop asks for a human expert (``False`` without a rating)."""
        return False if self.state is None else self.state.expert_review_required()


def tfidf_similarity(first: str, second: str) -> float:
    """Cosine similarity of two texts under a per-pair TF-IDF (the book's metric)."""
    index = TfidfIndex([first, second])
    return float(index.matrix[0] @ index.matrix[1])


def evaluate(result: AdaptiveResult, *, rating: Optional[int] = None) -> Evaluation:
    """Score ``result``: TF-IDF cosine, plus the human-rating loop when given a rating."""
    state = RatingState().record(rating) if rating is not None else None
    return Evaluation(
        similarity=tfidf_similarity(result.text_input, result.response),
        rating=rating,
        state=state,
    )


class AdaptiveRAG:
    """Answer a query through the book's ranking-adaptive RAG with human feedback."""

    def __init__(
        self,
        settings,
        *,
        llm: Optional[LLM] = None,
        fetcher: Fetcher = fetch_article,
    ) -> None:
        self.settings = settings
        self.llm = llm or create_llm(settings)
        self.fetcher = fetcher

    def input_for(self, query: str, ranking: int, num_words: int) -> tuple[str, Optional[str], str]:
        """Select the generator's input for ``ranking`` (the book's 2.3 branch).

        Returns ``(strategy, source, text_input)``: ``source`` is the fetched URL
        for the document strategy and ``None`` otherwise.
        """
        strategy = strategy_for_ranking(ranking)
        if strategy == STRATEGY_NONE:
            return strategy, None, query
        if strategy == STRATEGY_FEEDBACK:
            return strategy, None, HUMAN_FEEDBACK

        retrieved = retrieve(query, num_words, fetcher=self.fetcher)
        if retrieved is None:
            raise NoMatchError(
                f"none of {', '.join(ADAPTIVE_SOURCES)} appear in the query; "
                "the retriever has no document to fetch"
            )
        return strategy, retrieved.url, retrieved.excerpt

    def answer(
        self,
        query: str,
        *,
        ranking: int = DEFAULT_RANKING,
        num_words: int = DEFAULT_NUM_WORDS,
    ) -> AdaptiveResult:
        """Run the adaptive selection and generation, timing the model call."""
        strategy, source, text_input = self.input_for(query, ranking, num_words)
        pipeline = RAGPipeline(self.llm, instruction=ADAPTIVE_INSTRUCTION)
        start = time.perf_counter()
        response = pipeline.answer(text_input)
        elapsed = time.perf_counter() - start
        return AdaptiveResult(
            query=query,
            ranking=ranking,
            strategy=strategy,
            source=source,
            text_input=text_input,
            response=response,
            elapsed=elapsed,
        )


def save_expert_feedback(
    feedback: str,
    path: Path | str = DEFAULT_EXPERT_FEEDBACK,
) -> Path:
    """Write expert feedback to ``path``, the book's ``expert_feedback.txt``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(feedback, encoding="utf-8")
    return path


def load_expert_feedback(path: Path | str = DEFAULT_EXPERT_FEEDBACK) -> Optional[str]:
    """Read expert feedback written by :func:`save_expert_feedback`, if any."""
    path = Path(path)
    return path.read_text(encoding="utf-8") if path.is_file() else None
