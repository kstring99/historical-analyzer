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
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def analyze_image(self, image_bytes: bytes, prompt: str, media_type: str = "image/png") -> str:
        b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
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


def get_provider(provider_name: str = "auto", api_key: str | None = None) -> LLMProvider:
    """Factory to get the configured LLM provider.
    
    Priority for API keys:
    1. Explicitly passed api_key
    2. Environment variable (OPENAI_API_KEY or ANTHROPIC_API_KEY)
    
    Default provider is 'auto' — tries OpenAI first (enterprise), falls back to Anthropic.
    """
    if provider_name == "auto":
        # Try OpenAI first (enterprise subscription), fall back to Anthropic
        if api_key or os.environ.get("OPENAI_API_KEY"):
            provider_name = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            provider_name = "anthropic"
        else:
            # Default to OpenAI — enterprise deployments set OPENAI_API_KEY
            provider_name = "openai"
    
    if provider_name == "anthropic":
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


