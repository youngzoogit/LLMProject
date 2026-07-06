"""Layer-isolating diagnostic for the Ollama/Gemma RAG crash.

Reproduces the "short prompt works, long RAG prompt kills the server" symptom by
hitting the Ollama REST API directly (no app, no LangChain), then repeating the
same prompts through LangChain (ChatOllama). Comparing the two paths tells us
which layer is at fault:

  - REST API also fails  -> Ollama / model / driver problem (not our code).
  - REST ok but LangChain fails -> LangChain configuration problem.

Usage:
    python scripts/test_ollama_prompt_lengths.py
    python scripts/test_ollama_prompt_lengths.py --model llama3.2:1b
    python scripts/test_ollama_prompt_lengths.py --models gemma2:2b llama3.2:1b
    python scripts/test_ollama_prompt_lengths.py --lengths 100 300 500 800 1200
    python scripts/test_ollama_prompt_lengths.py --json reports/_ollama_probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
GENERATE_ENDPOINT = f"{OLLAMA_URL}/api/generate"
DEFAULT_MODEL = "gemma2:2b"
DEFAULT_LENGTHS = [100, 300, 500, 800, 1200]
NUM_PREDICT = 64
TIMEOUT = 120

# A realistic RAG-like Korean prompt head; padded to the target length so the
# test mirrors the actual failing content, not random bytes.
_PROMPT_HEAD = (
    "당신은 TCGA 유전자 발현 기반 암종 예측 결과를 설명하는 연구용 도우미입니다. "
    "다음 컨텍스트에 근거해 한국어로 3문장 이내로 답하세요.\n"
    "[질문] 이 환자는 왜 갑상선암으로 예측됐어?\n"
    "[예측 요약] 예측 암종: 갑상선암(THCA) / 모델 동의: 3/3 / 평균 확률: 99.6%\n"
    "[유전자 근거] "
)
_PADDING = (
    "TG는 티로글로불린으로 갑상선 여포세포가 만드는 대형 당단백이며 갑상선호르몬 "
    "합성의 전구체입니다. TPO는 갑상선 과산화효소로 호르몬 생합성에 필수적입니다. "
    "TSHR은 갑상선자극호르몬 수용체입니다. "
)


def build_prompt(length: int) -> str:
    """Build a prompt of approximately ``length`` characters."""
    text = _PROMPT_HEAD
    while len(text) < length:
        text += _PADDING
    return text[:length]


def probe_rest(model: str, prompt: str, timeout: int = TIMEOUT) -> dict:
    """Call Ollama /api/generate directly. Never raises."""
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "5m",
            "options": {"num_predict": NUM_PREDICT, "temperature": 0},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        GENERATE_ENDPOINT, data=payload, headers={"Content-Type": "application/json"}
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
        elapsed = time.time() - start
        text = (body.get("response") or "").strip()
        return {
            "ok": True,
            "elapsed": elapsed,
            "resp_len": len(text),
            "error_type": None,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - diagnostic must capture everything
        return {
            "ok": False,
            "elapsed": time.time() - start,
            "resp_len": 0,
            "error_type": type(exc).__name__,
            "error": str(exc)[:200],
        }


def probe_langchain(model: str, prompt: str, timeout: int = TIMEOUT) -> dict:
    """Call the same prompt through LangChain ChatOllama. Never raises."""
    start = time.time()
    try:
        from langchain_core.messages import HumanMessage
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=model,
            temperature=0,
            base_url=OLLAMA_URL,
            num_predict=NUM_PREDICT,
            client_kwargs={"timeout": timeout},
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        elapsed = time.time() - start
        text = getattr(resp, "content", "")
        if isinstance(text, list):
            text = " ".join(str(t) for t in text)
        return {
            "ok": True,
            "elapsed": elapsed,
            "resp_len": len(str(text).strip()),
            "error_type": None,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "elapsed": time.time() - start,
            "resp_len": 0,
            "error_type": type(exc).__name__,
            "error": str(exc)[:200],
        }


def ollama_reachable() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as resp:
            json.loads(resp.read())
        return True
    except Exception:
        return False


def run(models: list[str], lengths: list[int], timeout: int) -> dict:
    results: dict = {"reachable": ollama_reachable(), "models": {}}
    if not results["reachable"]:
        print(f"[!] Ollama unreachable at {OLLAMA_URL} — is `ollama serve` running?")
        return results

    for model in models:
        print(f"\n=== model: {model} ===")
        header = f"{'chars':>6} | {'REST':>6} {'time':>7} {'len':>5} | {'LC':>6} {'time':>7} {'len':>5} | notes"
        print(header)
        print("-" * len(header))
        rows = []
        for length in lengths:
            prompt = build_prompt(length)
            rest = probe_rest(model, prompt, timeout)
            lc = probe_langchain(model, prompt, timeout)
            note = ""
            if not rest["ok"]:
                note = f"REST {rest['error_type']}: {rest['error']}"
            elif not lc["ok"]:
                note = f"LC {lc['error_type']}: {lc['error']}"
            print(
                f"{length:>6} | {('OK' if rest['ok'] else 'FAIL'):>6} "
                f"{rest['elapsed']:>6.1f}s {rest['resp_len']:>5} | "
                f"{('OK' if lc['ok'] else 'FAIL'):>6} "
                f"{lc['elapsed']:>6.1f}s {lc['resp_len']:>5} | {note[:80]}"
            )
            rows.append({"chars": length, "rest": rest, "langchain": lc})
        results["models"][model] = rows
    return results


def summarize(results: dict) -> None:
    print("\n=== 결론 요약 ===")
    if not results.get("reachable"):
        print("Ollama에 연결할 수 없어 판정 불가.")
        return
    for model, rows in results["models"].items():
        rest_fail = [r["chars"] for r in rows if not r["rest"]["ok"]]
        lc_fail = [r["chars"] for r in rows if not r["langchain"]["ok"]]
        rest_only = sorted(set(rest_fail))
        lc_only = sorted(set(lc_fail) - set(rest_fail))
        print(f"- {model}: REST 실패 길이={rest_only or '없음'}, "
              f"LangChain 단독 실패 길이={lc_only or '없음'}")
        if rest_only:
            print(f"    -> REST API도 실패 → Ollama/모델/드라이버 문제 (앱/LangChain 아님).")
        elif lc_only:
            print(f"    -> REST는 되고 LangChain만 실패 → LangChain 설정 문제.")
        else:
            print(f"    -> 모든 길이 성공 → 이 환경에서는 재현 안 됨.")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Ollama prompt-length crash probe")
    parser.add_argument("--model", default=None, help="single model (default gemma2:2b)")
    parser.add_argument("--models", nargs="+", default=None, help="multiple models")
    parser.add_argument("--lengths", nargs="+", type=int, default=DEFAULT_LENGTHS)
    parser.add_argument("--timeout", type=int, default=TIMEOUT)
    parser.add_argument("--json", default=None, help="write raw results to this JSON path")
    args = parser.parse_args(argv)

    if args.models:
        models = args.models
    elif args.model:
        models = [args.model]
    else:
        models = [DEFAULT_MODEL]

    results = run(models, args.lengths, args.timeout)
    summarize(results)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[+] raw results -> {args.json}")


if __name__ == "__main__":
    # Allow running from the project root without installation.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main(sys.argv[1:])
