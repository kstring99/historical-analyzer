import base64
import json
import os
from abc import ABC, abstractmethod

import anthropic
import openai


class LLMProvider(ABC):
    """Abstract interface — swap models via config."""

    @abstractmethod
    def analyze_image(self, image_bytes: bytes, prompt: str, media_type: str = "image/png") -> str:
        ...

    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        ...


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def analyze_image(self, image_bytes: bytes, prompt: str, media_type: str = "image/png") -> str:
        import time
        b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        # Retry up to 3 times — OAuth tokens intermittently 500 on vision
        for attempt in range(3):
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": b64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }],
                )
                return message.content[0].text
            except Exception as e:
                if "500" in str(e) and attempt < 2:
                    print(f"[LLM] Vision 500 error, retry {attempt + 1}/3...")
                    time.sleep(1)
                    continue
                raise

    def generate_text(self, prompt: str) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def analyze_image(self, image_bytes: bytes, prompt: str, media_type: str = "image/png") -> str:
        b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.choices[0].message.content

    def generate_text(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


class OpenClawProvider(LLMProvider):
    """Route through local OpenClaw gateway (uses Claude Max subscription)."""
    
    def __init__(self, base_url: str = None, token: str = None, model: str = None):
        self.base_url = base_url or os.environ.get("OPENCLAW_URL", "http://127.0.0.1:18789/v1/chat/completions")
        self.token = token or os.environ.get("OPENCLAW_TOKEN", "")
        self.model = model or os.environ.get("OPENCLAW_MODEL", "anthropic/claude-sonnet-4-6")
        import httpx
        self._client = httpx.Client(timeout=300.0)
    
    def _call(self, messages: list) -> str:
        resp = self._client.post(
            self.base_url,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
            json={"model": self.model, "messages": messages, "max_tokens": 2048},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    
    def analyze_image(self, image_bytes: bytes, prompt: str, media_type: str = "image/png") -> str:
        b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }]
        return self._call(messages)
    
    def generate_text(self, prompt: str) -> str:
        return self._call([{"role": "user", "content": prompt}])


def get_provider(provider_name: str = "auto", api_key: str | None = None) -> LLMProvider:
    """Factory to get the configured LLM provider.
    
    Priority for API keys:
    1. Explicitly passed api_key
    2. Environment variable (OPENAI_API_KEY or ANTHROPIC_API_KEY)
    
    Default provider is 'auto' — tries OpenAI first (enterprise), falls back to Anthropic.
    """
    if provider_name == "auto":
        # Anthropic first (supports vision), then OpenAI, then OpenClaw (text-only)
        if api_key or os.environ.get("ANTHROPIC_API_KEY"):
            provider_name = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            provider_name = "openai"
        elif os.environ.get("OPENCLAW_TOKEN"):
            provider_name = "openclaw"
        else:
            provider_name = "anthropic"
    
    if provider_name == "openclaw":
        return OpenClawProvider()
    elif provider_name == "anthropic":
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError(
                "No Anthropic API key found. Set ANTHROPIC_API_KEY environment variable "
                "or pass an API key in Settings."
            )
        return AnthropicProvider(api_key=key)
    elif provider_name == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError(
                "No OpenAI API key found. Set OPENAI_API_KEY environment variable "
                "or pass an API key in Settings."
            )
        return OpenAIProvider(api_key=key)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


