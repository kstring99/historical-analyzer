"""Optional VLM fallback via ollama for low-confidence detections."""

import base64
import json
import subprocess

import cv2
import numpy as np


PROMPT_TEMPLATE = """You are analyzing a cropped section of an aerial photograph for a Phase I Environmental Site Assessment.

This image patch was flagged by automated detection as a potential environmental concern.
The automated system classified it as: {cv_class} (confidence: {confidence}).

Look at this image and determine what the feature actually is. Choose ONE of:
- UST farm (underground storage tanks — circular features, fill ports, dispensers)
- AST (above-ground storage tank — cylindrical or rectangular tank)
- Lagoon/pit (waste lagoon, retention pond, settling pit)
- Industrial structure (manufacturing facility, processing equipment)
- Vegetation stress (dying/stressed vegetation from contamination)
- Surface staining (soil/pavement discoloration from spills)
- Benign (normal structure, shadow, artifact — not an environmental concern)

Respond with ONLY a JSON object:
{{"class": "your_choice", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""


def check_ollama() -> bool:
    """Check if ollama is installed and running."""
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def classify_with_vlm(
    crop: np.ndarray,
    cv_class: str,
    cv_confidence: float,
    model: str = "llava:13b",
) -> dict | None:
    """Send a crop to a local VLM via ollama for classification.

    Returns {"class", "confidence", "reasoning"} or None on failure.
    """
    _, buf = cv2.imencode(".jpg", crop)
    b64 = base64.b64encode(buf).decode("utf-8")

    prompt = PROMPT_TEMPLATE.format(cv_class=cv_class, confidence=cv_confidence)

    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            input=b64,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            return None

        text = result.stdout.strip()
        # Try to extract JSON from the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        return None

    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return None
