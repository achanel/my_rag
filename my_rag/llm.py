"""LLM providers.

Extension point: subclass :class:`LLM` and register a factory with
``@register_provider("name")`` to add a new backend. The built-in
``ollama``/``openai`` providers share :class:`OpenAICompatibleLLM`, because both
speak the OpenAI chat-completions API; adding e.g. Anthropic means writing one
new :class:`LLM` class and nothing else.
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Callable, Iterator, Optional

from openai import OpenAI

from .config import PROVIDER_DEFAULTS, Settings, load_hf_token


class LLM(ABC):
    """Minimal text-completion interface consumed by the pipeline."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Return the model's completion for ``prompt``."""

    def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        """Yield response text as it is generated.

        Default implementation falls back to a single chunk so that providers
        without streaming support still work with the CLI.
        """
        yield self.complete(prompt, system=system, temperature=temperature)


class OpenAICompatibleLLM(LLM):
    """Client for any server exposing the OpenAI chat-completions API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.1,
        extra_body: Optional[dict] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._extra_body = extra_body
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def _request(self, prompt: str, system: Optional[str], temperature: Optional[float]) -> dict:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        request: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if self._extra_body is not None:
            request["extra_body"] = self._extra_body
        return request

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        response = self._client.chat.completions.create(**self._request(prompt, system, temperature))
        return response.choices[0].message.content.strip()

    def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        chunks = self._client.chat.completions.create(
            **self._request(prompt, system, temperature), stream=True
        )
        for chunk in chunks:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    def list_models(self) -> list[str]:
        """Return the model ids the server reports (used for error hints)."""
        return [model.id for model in self._client.models.list().data]


#: Markers that identify a gated/auth Hugging Face load failure (llama-2 etc.).
_HF_ACCESS_MARKERS = ("gated", "unauthorized", "401", "403", "authorized list", "access to model")


def _hf_is_gated_error(error: Exception) -> bool:
    """Whether ``error`` looks like a gated repo / missing HF authorization."""
    lowered = str(error).lower()
    return any(marker in lowered for marker in _HF_ACCESS_MARKERS)


def _hf_load_hint(model: str, error: Exception) -> str:
    """Add an actionable hint to a gated/auth Hugging Face load failure."""
    message = str(error)
    if _hf_is_gated_error(error):
        return (
            f"{message}\n\n"
            f"hint: {model!r} needs Hugging Face access. Accept its terms at "
            f"https://huggingface.co/{model}, then authenticate with 'hf auth login' "
            "(the modern CLI; 'huggingface-cli' is removed) or a token at "
            "~/.config/hugging_face/token (or HF_TOKEN / MY_RAG_HF_TOKEN_FILE). Or "
            "pick a small ungated local model with --model, e.g. Qwen/Qwen2.5-0.5B-Instruct."
        )
    return message


class HuggingFaceLLM(LLM):
    """Local generation through a ``transformers`` text-generation pipeline.

    Chapter 8's book loads ``meta-llama/Llama-2-7b-chat-hf`` with
    ``transformers.pipeline(...)``. Reproduced here so ``--provider hf`` runs a
    local Hugging Face model; by default that is the small ungated
    ``Qwen/Qwen2.5-0.5B-Instruct`` (no token, runs on CPU), and the gated Llama-2
    can be requested with ``--model``. The pipeline is imported and built lazily on
    the first :meth:`complete`, so merely selecting the provider does not load a
    checkpoint. ``torch_dtype="auto"`` uses float16 where the device supports it
    (CUDA/MPS) and float32 on CPU. If the requested model is gated and
    inaccessible, ``fallback_model`` (the provider's ungated default) is used
    instead of failing, with a warning.
    """

    def __init__(
        self,
        model: str,
        *,
        fallback_model: Optional[str] = None,
        temperature: float = 0.5,
        max_new_tokens: int = 100,
        top_k: int = 10,
        repetition_penalty: float = 2.0,
        torch_dtype: str = "auto",
        device_map: Optional[str] = None,
    ) -> None:
        self.model = model
        self.fallback_model = fallback_model if fallback_model != model else None
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self._pipeline = None

    def _resolve_dtype(self, torch):
        """The torch dtype: an explicit name, or float16 on GPU/MPS and float32 on CPU."""
        if self.torch_dtype != "auto":
            return getattr(torch, self.torch_dtype, torch.float32)
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if torch.cuda.is_available() or (mps is not None and mps.is_available()):
            return torch.float16
        return torch.float32

    def _load(self):
        """Build the text-generation pipeline once (lazy heavy import)."""
        if self._pipeline is None:
            import torch
            from transformers import pipeline

            # The book exports its HF token before loading; do the same from
            # ~/.config/hugging_face/token (or HF_TOKEN / MY_RAG_HF_TOKEN_FILE) so
            # gated checkpoints such as Llama-2 authenticate.
            load_hf_token()
            dtype = self._resolve_dtype(torch)
            try:
                self._pipeline = self._build(pipeline, self.model, dtype)
            except Exception as exc:
                if not self._is_gated(exc):
                    raise RuntimeError(_hf_load_hint(self.model, exc)) from exc
                if self.fallback_model is None:
                    raise RuntimeError(_hf_load_hint(self.model, exc)) from exc
                # A gated model without access: use the ungated fallback instead of
                # failing the whole run, and say so loudly.
                warnings.warn(
                    f"Hugging Face model {self.model!r} is gated and not accessible "
                    f"({exc.__class__.__name__}); falling back to {self.fallback_model!r}. "
                    "Accept the terms / authenticate to use it, or pass an ungated --model.",
                    UserWarning,
                    stacklevel=2,
                )
                self.model = self.fallback_model
                self._pipeline = self._build(pipeline, self.model, dtype)
        return self._pipeline

    def _build(self, pipeline, model: str, dtype):
        """Construct the text-generation pipeline (split out so tests can stub it).

        ``device_map`` stays ``None`` by default: transformers' ``"auto"`` path
        segfaults while materialising the weights on Apple-silicon macOS, whereas
        ``None`` lets the pipeline pick the default device (MPS/CPU) and works.
        Pass ``device_map="auto"`` explicitly only on a CUDA box.
        """
        return pipeline(
            "text-generation",
            model=model,
            dtype=dtype,  # transformers ≥5 name (was torch_dtype)
            device_map=self.device_map,
        )

    @staticmethod
    def _is_gated(error: Exception) -> bool:
        return _hf_is_gated_error(error)

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        text = f"{system}\n{prompt}" if system else prompt
        sequences = self._load()(
            text,
            do_sample=True,
            top_k=self.top_k,
            num_return_sequences=1,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature if temperature is None else temperature,
            repetition_penalty=self.repetition_penalty,
            truncation=True,
        )
        generated = str(sequences[0]["generated_text"])
        # The book strips the prompt from the generated text (``replace(iprompt, '')``).
        return generated.replace(text, "", 1).strip()


ProviderFactory = Callable[[Settings], LLM]
_PROVIDERS: dict[str, ProviderFactory] = {}


def register_provider(name: str) -> Callable[[ProviderFactory], ProviderFactory]:
    """Decorator registering a provider factory under ``name``."""

    def decorator(factory: ProviderFactory) -> ProviderFactory:
        _PROVIDERS[name] = factory
        return factory

    return decorator


@register_provider("ollama")
def _ollama(settings: Settings) -> LLM:
    # Ollama understands a `think` flag for reasoning models (e.g. qwen3).
    return OpenAICompatibleLLM(
        base_url=settings.base_url,
        api_key=settings.api_key,
        model=settings.model,
        temperature=settings.temperature,
        extra_body={"think": settings.think},
    )


@register_provider("openai")
def _openai(settings: Settings) -> LLM:
    return OpenAICompatibleLLM(
        base_url=settings.base_url,
        api_key=settings.api_key,
        model=settings.model,
        temperature=settings.temperature,
    )


@register_provider("hf")
def _huggingface(settings: Settings) -> LLM:
    # Local transformers pipeline (chapter 8's Hugging Face Llama). Embeddings are
    # not affected: they keep coming from ``settings.base_url`` (local Ollama). If
    # the requested model is the gated Llama-2 and it is not accessible, fall back
    # to the provider's ungated default rather than failing the run.
    return HuggingFaceLLM(
        settings.model,
        fallback_model=PROVIDER_DEFAULTS["hf"].get("model"),
        temperature=settings.temperature,
    )


def available_providers() -> list[str]:
    """Return the sorted names of all registered providers."""
    return sorted(_PROVIDERS)


def create_llm(settings: Settings) -> LLM:
    """Instantiate the provider requested by ``settings``."""
    try:
        factory = _PROVIDERS[settings.provider]
    except KeyError:
        raise ValueError(
            f"Unknown provider {settings.provider!r}; available: {', '.join(available_providers())}"
        ) from None
    return factory(settings)
