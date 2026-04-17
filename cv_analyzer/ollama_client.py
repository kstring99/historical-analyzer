"""Ollama REST API client for local LLM inference."""

import json
import urllib.request
import urllib.error


DEFAULT_URL = "http://localhost:11434"


def check_available(base_url: str = DEFAULT_URL) -> bool:
    """Check if ollama is running and reachable."""
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def list_models(base_url: str = DEFAULT_URL) -> list[str]:
    """List available model names."""
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return []


def generate(
    prompt: str,
    model: str = "llama3.2:3b",
    system: str = "",
    base_url: str = DEFAULT_URL,
    temperature: float = 0.3,
    timeout: int = 120,
) -> str:
    """Generate a completion from a local ollama model.

    Uses the non-streaming /api/generate endpoint for simplicity.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if system:
        payload["system"] = system

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip()
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama error {e.code}: {error_body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach ollama at {base_url}: {e}") from e
