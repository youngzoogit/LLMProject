"""LangChain <-> Ollama (Gemma) wrapper with graceful degradation.

All entry points are safe to call even when ``langchain-ollama`` is not installed
or the Ollama server is down: availability checks return False and ``generate``
returns None, so the caller can fall back to a rule-based answer.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request

logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e2b")
_PING_TIMEOUT = 2
# Hard cap on generation so a slow/CPU-only model fails fast to the rule-based
# fallback instead of freezing the UI for minutes. Override via OLLAMA_TIMEOUT.
_GEN_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "120"))


def ollama_up(timeout: int = _PING_TIMEOUT) -> bool:
    """True if the Ollama server responds on /api/tags."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=timeout) as resp:
            json.loads(resp.read())
        return True
    except Exception:
        return False


def langchain_installed() -> bool:
    try:
        import langchain_ollama  # noqa: F401

        return True
    except Exception:
        return False


def gemma_available() -> bool:
    """True only if both the package is importable and Ollama is reachable."""
    return langchain_installed() and ollama_up()


def generate_verbose(
    system_prompt: str, user_prompt: str, temperature: float = 0.0
) -> tuple[str | None, str | None]:
    """Run Gemma via ChatOllama. Returns (text, error_reason).

    On success: (text, None). On failure: (None, short error description) so the
    caller can surface *why* it fell back instead of failing silently.

    Note: on this project's Windows/Ollama setup, prompts beyond roughly 500
    characters reproducibly crash the Ollama server (``wsarecv`` / connection
    forcibly closed) regardless of whether the content is split across a
    SystemMessage or merged into one HumanMessage -- confirmed by direct testing.
    This is an Ollama-server-side issue, not a prompt-formatting bug, so no
    client-side merge/rewrite can work around it; only a smaller prompt, a
    different model, or a fixed Ollama install can.
    """
    prompt_length = len(system_prompt) + len(user_prompt)
    start = time.time()
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=MODEL,
            temperature=temperature,
            base_url=OLLAMA_URL,
            num_predict=256,
            client_kwargs={"timeout": _GEN_TIMEOUT},
        )
        response = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        text = getattr(response, "content", None)
        if isinstance(text, list):  # some providers return content parts
            text = " ".join(str(t) for t in text)
        text = (text or "").strip()
        elapsed = time.time() - start
        if not text:
            logger.warning(
                "Gemma empty response | model=%s prompt_len=%d elapsed=%.1fs",
                MODEL, prompt_length, elapsed,
            )
            return None, f"빈 응답 (model={MODEL}, prompt_len={prompt_length})"
        logger.info(
            "Gemma ok | model=%s prompt_len=%d elapsed=%.1fs resp_len=%d",
            MODEL, prompt_length, elapsed, len(text),
        )
        return text, None
    except Exception as exc:
        # Do NOT swallow silently: log full context for the runtime diagnosis.
        elapsed = time.time() - start
        error_type = type(exc).__name__
        error_message = str(exc)[:200]
        logger.error(
            "Gemma generation failed | model=%s error_type=%s prompt_len=%d "
            "elapsed=%.1fs error=%s",
            MODEL, error_type, prompt_length, elapsed, error_message,
        )
        return None, (
            f"{error_type}: {error_message} "
            f"(model={MODEL}, prompt_len={prompt_length}, elapsed={elapsed:.1f}s)"
        )


def generate(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str | None:
    """Run Gemma via ChatOllama. Returns text, or None on any failure."""
    text, _ = generate_verbose(system_prompt, user_prompt, temperature)
    return text
