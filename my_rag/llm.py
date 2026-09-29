"""LLM providers.

Extension point: subclass :class:`LLM` and register a factory with
``@register_provider("name")`` to add a new backend. The built-in
``ollama``/``openai`` providers share :class:`OpenAICompatibleLLM`, because both
speak the OpenAI chat-completions API; adding e.g. Anthropic means writing one
new :class:`LLM` class and nothing else.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Iterator, Optional

from openai import OpenAI

from .config import Settings


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
