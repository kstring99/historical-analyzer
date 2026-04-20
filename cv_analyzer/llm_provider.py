"""LLM provider abstraction for vision + text calls.

Supports Anthropic (Claude), OpenAI (GPT-4o), or local ollama text-only.
The provider is chosen at request time based on user settings or env vars.
"""

import base64
import os
from abc import ABC, abstractmethod
from typing import Optional


class LLMProvider(ABC):
    name: str = "base"
    supports_vision: bool = False

    @abstractmethod
    def analyze_image(self, image_bytes: bytes, prompt: str, media_type: str = "image/jpeg") -> str:
        ...

    @abstractmethod
    def generate_text(self, prompt: str, system: str = "") -> str:
        ...


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    supports_vision = True

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def analyze_image(self, image_bytes: bytes, prompt: str, media_type: str = "image/jpeg") -> str:
        import time
        b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        for attempt in range(3):
            try:
                msg = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                            {"type": "text", "text": prompt},
                        ],
                    }],
                )
                return msg.content[0].text
            except Exception as e:
                if "overloaded" in str(e).lower() or "500" in str(e):
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                raise

    def generate_text(self, prompt: str, system: str = "") -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        msg = self.client.messages.create(**kwargs)
        return msg.content[0].text


class OpenAIProvider(LLMProvider):
    name = "openai"
    supports_vision = True

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def analyze_image(self, image_bytes: bytes, prompt: str, media_type: str = "image/jpeg") -> str:
        b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.choices[0].message.content

    def generate_text(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            messages=messages,
        )
        return response.choices[0].message.content


class OllamaProvider(LLMProvider):
    """Local text-only fallback. Cannot analyze images — analyze_image raises."""
    name = "ollama"
    supports_vision = False

    def __init__(self, model: str = "llama3.2:3b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def analyze_image(self, image_bytes: bytes, prompt: str, media_type: str = "image/jpeg") -> str:
        raise NotImplementedError(
            "Local ollama provider does not support vision analysis. "
            "Add an Anthropic or OpenAI API key in Settings to enable topo map and "
            "city directory analysis."
        )

    def generate_text(self, prompt: str, system: str = "") -> str:
        from cv_analyzer.ollama_client import generate
        return generate(prompt, model=self.model, system=system, temperature=0.3)


def resolve_provider(
    provider_name: str = "auto",
    api_key: Optional[str] = None,
    ollama_model: str = "llama3.2:3b",
) -> LLMProvider:
    """Pick a provider based on user input + environment.

    Priority order for "auto":
    1. User-supplied api_key → Anthropic
    2. ANTHROPIC_API_KEY env → Anthropic
    3. OPENAI_API_KEY env → OpenAI
    4. Ollama (text-only, vision paths will error)
    """
    name = (provider_name or "auto").lower()

    if name == "auto":
        if api_key:
            name = "anthropic"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            name = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            name = "openai"
        else:
            name = "ollama"

    if name == "anthropic":
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError(
                "Anthropic selected but no API key provided. "
                "Paste a key into Settings or set ANTHROPIC_API_KEY."
            )
        return AnthropicProvider(api_key=key)

    if name == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError(
                "OpenAI selected but no API key provided. "
                "Paste a key into Settings or set OPENAI_API_KEY."
            )
        return OpenAIProvider(api_key=key)

    if name == "ollama":
        return OllamaProvider(model=ollama_model)

    raise ValueError(f"Unknown provider: {name!r}")
