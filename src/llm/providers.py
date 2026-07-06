"""LLM provider chain: Gemini API -> Ollama Gemma -> rule-based fallback.

Selected by LLM_PROVIDER:
- gemini/api (default): try Google Gemini, then fallback
- ollama: try only local Ollama, then fallback
- fallback: skip LLM providers

The app never raises provider errors to the UI. It records attempts and lets the
caller use the rule-based/document answer when all LLM providers fail.
"""

from __future__ import annotations

import logging
import os

from src.llm import langchain_gemma

logger = logging.getLogger(__name__)

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", os.environ.get("API_MODEL", "gemini-2.5-flash"))
API_TIMEOUT = int(os.environ.get("API_TIMEOUT", "30"))
_BLOCKING_PROXY_VALUES = {"http://127.0.0.1:9", "https://127.0.0.1:9"}

PROVIDER_LABEL = {
    "gemini": "Gemini API",
    "api": "Gemini API",
    "ollama": "Ollama Gemma",
    "fallback": "문서 기반 fallback",
}


def provider_label(provider: str | None) -> str:
    return PROVIDER_LABEL.get(provider or "fallback", "문서 기반 fallback")


def selected_provider() -> str:
    """LLM_PROVIDER env: gemini/api(default) | ollama | fallback."""
    return os.environ.get("LLM_PROVIDER", "gemini").strip().lower()


def _ollama_fallback_enabled() -> bool:
    return os.environ.get("LLM_ENABLE_OLLAMA_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"}


def _provider_chain() -> list[str]:
    pref = selected_provider()
    if pref == "ollama":
        return ["ollama"]
    if pref == "fallback":
        return []
    if pref in ("gemini_ollama", "api_ollama", "google_ollama"):
        return ["gemini", "ollama"]
    if pref in ("gemini", "api", "google"):
        return ["gemini", "ollama"] if _ollama_fallback_enabled() else ["gemini"]
    return ["gemini", "ollama"] if _ollama_fallback_enabled() else ["gemini"]



def _clear_blocking_proxy_env() -> None:
    """Remove local deny-proxy values that make Google API calls fail with 10061."""
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        value = os.environ.get(key, "")
        if value.rstrip("/").lower() in _BLOCKING_PROXY_VALUES:
            os.environ.pop(key, None)

# --- Gemini API ------------------------------------------------------------- #
def api_available() -> bool:
    """True if a Google Gemini key is set and LangChain Gemini is importable."""
    if not os.environ.get("GOOGLE_API_KEY"):
        return False
    try:
        import langchain_google_genai  # noqa: F401

        return True
    except Exception:
        return False


def gemini_generate(system_prompt: str, user_prompt: str) -> tuple[str | None, str | None]:
    """Call Google Gemini through LangChain. Returns (text, error). Never raises."""
    _clear_blocking_proxy_env()
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0,
            timeout=API_TIMEOUT,
            max_retries=1,
        )
        resp = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        text = getattr(resp, "content", "")
        if isinstance(text, list):
            text = " ".join(str(part) for part in text)
        text = (text or "").strip()
        if not text:
            return None, f"빈 응답 (model={GEMINI_MODEL})"
        return text, None
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {str(exc)[:180]}"
        logger.error("Gemini API failed | model=%s error=%s", GEMINI_MODEL, error)
        return None, error


# Backward-compatible name used by older code/reports.
def api_generate(system_prompt: str, user_prompt: str) -> tuple[str | None, str | None]:
    return gemini_generate(system_prompt, user_prompt)


# --- Unified entry ---------------------------------------------------------- #
def generate_answer(
    system_prompt: str,
    user_prompt: str,
    ollama_prompt: str | None = None,
) -> dict:
    """Try providers in order and return the first successful generation."""
    attempts: list[tuple[str, str]] = []

    for provider in _provider_chain():
        if provider == "gemini":
            if not api_available():
                attempts.append(("gemini", "GOOGLE_API_KEY 없음 또는 langchain_google_genai 미설치"))
                continue
            text, error = gemini_generate(system_prompt, user_prompt)
            if text:
                return {"text": text, "provider": "gemini", "error": None, "attempts": attempts}
            attempts.append(("gemini", error or "실패"))

        elif provider == "ollama":
            if not langchain_gemma.gemma_available():
                attempts.append(("ollama", "Ollama/langchain-ollama 미연결"))
                continue
            text, error = langchain_gemma.generate_verbose(system_prompt, ollama_prompt or user_prompt)
            if text:
                return {"text": text, "provider": "ollama", "error": None, "attempts": attempts}
            attempts.append(("ollama", error or "실패"))

    last_error = attempts[-1][1] if attempts else "LLM provider 미사용(fallback)"
    return {"text": None, "provider": None, "error": last_error, "attempts": attempts}