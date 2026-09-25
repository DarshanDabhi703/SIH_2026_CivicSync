"""
llm_client.py
=============
CivicSync Phase 3 — Local Ollama LLM Client

Interacts with the local Ollama HTTP API (http://localhost:11434)
to generate text using the local Llama model (e.g. llama3.1:8b).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Optional

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


class OllamaClientError(Exception):
    """Base exception for Ollama client errors."""
    pass


class OllamaConnectionError(OllamaClientError):
    """Raised when Ollama server cannot be reached."""
    pass


class OllamaModelNotFoundError(OllamaClientError):
    """Raised when the specified model is not installed/found in Ollama."""
    pass


def generate(
    prompt: str,
    temperature: float = 0.1,
    model: Optional[str] = None,
    host: Optional[str] = None,
    timeout: float = 120.0,
) -> str:
    """
    Send a generation request to the local Ollama HTTP API.

    Parameters
    ----------
    prompt : str
        The prompt to send to the LLM.
    temperature : float, optional
        Sampling temperature (default 0.1 for deterministic factual output).
    model : str, optional
        Ollama model tag to use. Defaults to OLLAMA_MODEL env var or 'llama3.1:8b'.
    host : str, optional
        Ollama host endpoint. Defaults to OLLAMA_HOST env var or 'http://localhost:11434'.
    timeout : float, optional
        Request timeout in seconds (default 120.0).

    Returns
    -------
    str
        Generated text response from the model.
    """
    selected_model = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL_NAME)
    base_host = (host or os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)).rstrip("/")
    url = f"{base_host}/api/generate"

    payload = {
        "model": selected_model,
        "prompt": prompt,
        "temperature": temperature,
        "stream": False,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            result = json.loads(body)
            return result.get("response", "").strip()

    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise OllamaModelNotFoundError(
                f"Model '{selected_model}' not found in Ollama at {base_host}. "
                f"Please run 'ollama pull {selected_model}'."
            ) from exc
        body_text = exc.read().decode("utf-8", errors="ignore")
        raise OllamaClientError(
            f"Ollama HTTP {exc.code} Error: {body_text}"
        ) from exc

    except urllib.error.URLError as exc:
        raise OllamaConnectionError(
            f"Failed to connect to Ollama server at {base_host}. "
            "Ensure Ollama is running locally ('ollama serve')."
        ) from exc

    except Exception as exc:
        raise OllamaClientError(f"Unexpected error communicating with Ollama: {exc}") from exc
