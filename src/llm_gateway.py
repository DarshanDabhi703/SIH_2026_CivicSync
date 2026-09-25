"""
src/llm_gateway.py
==================
CivicSync Phase 6E — Unified LLM Provider Gateway

Abstracts LLM calls to Groq, Gemini, or local Ollama based on LLM_PROVIDER
environment variable. Automatically falls back to local Ollama if the primary
provider is not configured or fails.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from src.llm_client import generate as ollama_generate, OllamaClientError

logger = logging.getLogger("civicsync.llm_gateway")


class LLMGatewayError(Exception):
    """Raised when all configured LLM providers fail."""
    pass


def generate(
    prompt: str,
    temperature: float = 0.1,
    model: Optional[str] = None,
) -> str:
    """
    Generate response using the configured LLM provider (groq, gemini, ollama).
    Automatically falls back to local Ollama if any API errors occur or if API keys are missing.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not configured. Falling back to local Ollama.")
            return _ollama_generate_fallback(prompt, temperature, model)
        try:
            return _generate_groq(prompt, api_key, temperature, model)
        except Exception as exc:
            logger.warning("Groq generation failed (%s). Falling back to local Ollama.", repr(exc))
            return _ollama_generate_fallback(prompt, temperature, model)

    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not configured. Falling back to local Ollama.")
            return _ollama_generate_fallback(prompt, temperature, model)
        try:
            return _generate_gemini(prompt, api_key, temperature, model)
        except Exception as exc:
            logger.warning("Gemini generation failed (%s). Falling back to local Ollama.", repr(exc))
            return _ollama_generate_fallback(prompt, temperature, model)

    elif provider == "ollama":
        return _ollama_generate_fallback(prompt, temperature, model)

    else:
        logger.warning("Unknown LLM_PROVIDER '%s'. Falling back to local Ollama.", provider)
        return _ollama_generate_fallback(prompt, temperature, model)


def _ollama_generate_fallback(prompt: str, temperature: float, model: Optional[str]) -> str:
    """Invokes local Ollama client."""
    try:
        return ollama_generate(prompt=prompt, temperature=temperature, model=model)
    except Exception as exc:
        logger.error("Local Ollama fallback failed: %s", repr(exc))
        raise LLMGatewayError(f"All LLM generation paths failed: {repr(exc)}") from exc


def _generate_groq(prompt: str, api_key: str, temperature: float, model: Optional[str]) -> str:
    """Invokes Groq Cloud Chat Completion API."""
    # Llama-3.3-70b-specdec is the current state-of-the-art model on Groq
    groq_model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    url = "https://api.groq.com/openai/v1/chat/completions"

    payload = {
        "model": groq_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=30.0) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"].strip()


def _generate_gemini(prompt: str, api_key: str, temperature: float, model: Optional[str]) -> str:
    """Invokes Google Gemini Content Generation API."""
    gemini_model = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={api_key}"

    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": temperature
        }
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=30.0) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        return body["candidates"][0]["content"]["parts"][0]["text"].strip()
