#!/usr/bin/env python3
"""
Compare SSR outputs across embedding models (local vs OpenAI).

Usage:
  python backend/scripts/compare_ssr_embeddings.py \
    --input /path/to/project.json \
    --language pl \
    --mode intent \
    --limit 80
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, List

import numpy as np

from app.i18n import Language
from app.services.embedding_client import LocalEmbeddingClient, OpenAIEmbeddingClient
from app.services.ssr_engine import SSREngine


def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().split(" #", 1)[0].strip()
    return env


def _extract_purchase_intent_text(text: str, language: Language) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return text

    section = cleaned
    heading_match = re.search(r"(?mi)^\s*3[.)]\s*(.*)$", cleaned)
    if heading_match:
        section = cleaned[heading_match.start():]
        next_heading = re.search(r"(?mi)^\s*[1245][.)]\s+", section[1:])
        if next_heading:
            section = section[: next_heading.start() + 1]
        section = re.sub(r"(?mi)^\s*3[.)]\s*", "", section).strip()
    else:
        if language == Language.PL:
            phrase_match = re.search(r"(?i)\bczy\s+kup", cleaned)
        else:
            phrase_match = re.search(r"(?i)\bwould\s+you\s+buy\b", cleaned)
        if phrase_match:
            section = cleaned[phrase_match.start():].strip()

    section = re.sub(r"[ \t]+", " ", section)
    section = re.sub(r"\n{2,}", "\n", section).strip()

    sentences = re.split(r"(?<=[.!?])\s+", section)
    for sentence in sentences:
        candidate = sentence.strip()
        if not candidate:
            continue
        if re.search(r"(?i)^(czy\s+kup|would\s+you\s+buy)", candidate):
            continue
        return candidate

    for line in section.splitlines():
        candidate = line.strip()
        if candidate:
            return candidate

    return cleaned


def _load_texts(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        # Project JSON schema
        research = data.get("research", {})
        sim = research.get("simulation", {}) if isinstance(research, dict) else {}
        result = sim.get("result", {}) if isinstance(sim, dict) else {}
        agents = result.get("agent_responses", []) if isinstance(result, dict) else []
        texts = [
            a.get("text_response")
            for a in agents
            if isinstance(a, dict) and a.get("text_response")
        ]
        if texts:
            return texts
    if isinstance(data, list):
        return [str(x) for x in data if isinstance(x, str) and x.strip()]
    raise ValueError("Unsupported input format: expected project JSON or list of strings.")


def _likert_array(dist) -> np.ndarray:
    return np.array(
        [dist.scale_1, dist.scale_2, dist.scale_3, dist.scale_4, dist.scale_5],
        dtype=float,
    )


def _ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.cumsum(a) - np.cumsum(b))))


def _summarize(label: str, scores: Iterable[float]) -> None:
    arr = np.array(list(scores), dtype=float)
    print(f"{label}: mean={arr.mean():.4f} std={arr.std(ddof=0):.4f} min={arr.min():.4f} max={arr.max():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SSR outputs across embedding models.")
    parser.add_argument("--input", required=True, help="Path to project JSON or list-of-strings JSON.")
    parser.add_argument("--language", default="pl", choices=["pl", "en"], help="Language for anchors.")
    parser.add_argument("--mode", default="intent", choices=["intent", "full"], help="SSR input mode.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of responses.")
    parser.add_argument("--openai-model", default="text-embedding-3-small")
    parser.add_argument("--local-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--env-file", default="backend/.env")
    parser.add_argument("--top", type=int, default=10, help="Show top N per-response diffs.")
    parser.add_argument("--plot-out", default="", help="Optional PNG path for comparison chart.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    env_path = Path(args.env_file).expanduser().resolve()

    env = _parse_env_file(env_path)
    openai_key = os.getenv("OPENAI_API_KEY") or env.get("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment or .env file.")

    language = Language.PL if args.language == "pl" else Language.EN

    texts = _load_texts(input_path)
    if args.limit and args.limit > 0:
        texts = texts[: args.limit]

    if args.mode == "intent":
        texts = [_extract_purchase_intent_text(t, language) for t in texts]

    local_client = LocalEmbeddingClient(model_name=args.local_model)
    openai_client = OpenAIEmbeddingClient(api_key=openai_key, model=args.openai_model)

    ssr_local = SSREngine(embedding_client=local_client, language=language)
    ssr_openai = SSREngine(embedding_client=openai_client, language=language)

    local_results = ssr_local.rate_responses(texts)
    openai_results = ssr_openai.rate_responses(texts)

    local_scores = [r.expected_score for r in local_results]
    openai_scores = [r.expected_score for r in openai_results]
    diffs = [o - l for o, l in zip(openai_scores, local_scores)]

    local_dist = ssr_local.aggregate_to_survey_pmf(local_results)
    openai_dist = ssr_openai.aggregate_to_survey_pmf(openai_results)

    local_arr = _likert_array(local_dist)
    openai_arr = _likert_array(openai_dist)

    print("=== SSR Embedding Comparison ===")
    print(f"Responses: {len(texts)} | mode={args.mode} | language={args.language}")
    print(f"Local model: {args.local_model}")
    print(f"OpenAI model: {args.openai_model}")
    print("")
    _summarize("Local mean scores", local_scores)
    _summarize("OpenAI mean scores", openai_scores)
    _summarize("OpenAI - Local (per-response)", diffs)
    print("")
    print("Aggregate distribution (scale_1..scale_5)")
    print(f"Local:  {local_arr.round(6).tolist()}")
    print(f"OpenAI: {openai_arr.round(6).tolist()}")
    print("")
    l1 = float(np.sum(np.abs(local_arr - openai_arr)))
    ks = _ks_distance(local_arr, openai_arr)
    print(f"Aggregate L1 distance: {l1:.6f}")
    print(f"Aggregate KS distance: {ks:.6f}")

    if args.top > 0:
        print("")
        print(f"Top {args.top} per-response diffs (by |OpenAI-Local|):")
        ranked = sorted(
            enumerate(diffs),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[: args.top]
        for idx, diff in ranked:
            loc = local_scores[idx]
            opn = openai_scores[idx]
            snippet = texts[idx].replace("\n", " ").strip()
            if len(snippet) > 140:
                snippet = snippet[:137] + "..."
            print(f"- #{idx+1:02d}  local={loc:.3f} openai={opn:.3f} diff={diff:+.3f} | {snippet}")

    if args.plot_out:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise RuntimeError(f"matplotlib not available: {exc}") from exc

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        x = np.arange(1, 6)
        width = 0.35
        axes[0].bar(x - width / 2, local_arr, width=width, label="Local")
        axes[0].bar(x + width / 2, openai_arr, width=width, label="OpenAI")
        axes[0].set_xticks(x)
        axes[0].set_xlabel("Likert scale")
        axes[0].set_ylabel("Probability")
        axes[0].set_title("Aggregate Distribution")
        axes[0].legend()

        axes[1].scatter(local_scores, openai_scores, alpha=0.7, s=20)
        axes[1].plot([1, 5], [1, 5], linestyle="--", color="gray")
        axes[1].set_xlim(1, 5)
        axes[1].set_ylim(1, 5)
        axes[1].set_xlabel("Local expected score")
        axes[1].set_ylabel("OpenAI expected score")
        axes[1].set_title("Per-response Scores")

        fig.tight_layout()
        out_path = Path(args.plot_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
        print(f"\nSaved chart: {out_path}")


if __name__ == "__main__":
    main()
