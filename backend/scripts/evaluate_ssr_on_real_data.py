#!/usr/bin/env python3
"""Evaluate SSR quality on real labeled data and compare embedding models.

Default dataset:
  Hugging Face `amazon_reviews_multi` (English), using `review_body` and 1-5 stars.

This script answers two questions:
1) How well SSR scores align with real ratings for a real-world dataset?
2) How much do results change between embedding models?
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import spearmanr

from app.i18n import Language, get_anchor_sets
from app.services.embedding_client import LocalEmbeddingClient
from app.services.ssr_engine import SSREngine


@dataclass
class EvalMetrics:
    model: str
    n: int
    mae: float
    spearman: float
    exact_acc: float
    off_by_one_acc: float
    mean_pred: float
    mean_true: float


def _load_hf_amazon_reviews(limit: int, seed: int) -> tuple[list[str], list[int]]:
    from datasets import load_dataset

    ds = load_dataset("amazon_reviews_multi", "en", split="test")
    # Keep examples with non-empty text and 1..5 star labels.
    rows = [
        (str(x["review_body"]).strip(), int(x["stars"]))
        for x in ds
        if str(x["review_body"]).strip() and int(x["stars"]) in {1, 2, 3, 4, 5}
    ]
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    if limit > 0:
        rows = rows[:limit]
    texts, labels = zip(*rows)
    return list(texts), list(labels)


def _score_texts_ssr(
    texts: Iterable[str],
    *,
    model_name: str,
    language: Language,
    temperature: float,
    epsilon: float,
) -> np.ndarray:
    client = LocalEmbeddingClient(model_name=model_name)
    engine = SSREngine(
        embedding_client=client,
        language=language,
        temperature=temperature,
        epsilon=epsilon,
    )
    results = engine.rate_responses(list(texts))
    return np.array([float(r.expected_score) for r in results], dtype=float)


def _evaluate(pred_scores: np.ndarray, labels: list[int], model: str) -> EvalMetrics:
    true = np.array(labels, dtype=float)
    rounded = np.clip(np.rint(pred_scores), 1, 5).astype(int)
    true_int = true.astype(int)
    abs_err = np.abs(pred_scores - true)
    off_by_one = np.abs(rounded - true_int) <= 1
    corr = spearmanr(pred_scores, true).correlation
    corr = float(corr) if corr is not None and not np.isnan(corr) else 0.0
    return EvalMetrics(
        model=model,
        n=len(labels),
        mae=float(abs_err.mean()),
        spearman=corr,
        exact_acc=float((rounded == true_int).mean()),
        off_by_one_acc=float(off_by_one.mean()),
        mean_pred=float(pred_scores.mean()),
        mean_true=float(true.mean()),
    )


def _print_metrics(m: EvalMetrics) -> None:
    print(f"\n=== {m.model} ===")
    print(f"n={m.n}")
    print(f"MAE: {m.mae:.4f}")
    print(f"Spearman: {m.spearman:.4f}")
    print(f"Exact accuracy (rounded): {m.exact_acc:.4f}")
    print(f"Off-by-one accuracy: {m.off_by_one_acc:.4f}")
    print(f"Mean predicted: {m.mean_pred:.4f}")
    print(f"Mean true: {m.mean_true:.4f}")


def _compare_models(name_a: str, scores_a: np.ndarray, name_b: str, scores_b: np.ndarray) -> None:
    diffs = np.abs(scores_a - scores_b)
    print(f"\n--- Embedding impact: {name_a} vs {name_b} ---")
    print(f"Mean |Δ score|: {float(diffs.mean()):.4f}")
    print(f"Median |Δ score|: {float(statistics.median(diffs.tolist())):.4f}")
    print(f"Max |Δ score|: {float(diffs.max()):.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SSR on real labeled data.")
    p.add_argument("--limit", type=int, default=1200, help="Number of test examples (0 = all).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--language", choices=["pl", "en"], default="en")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--epsilon", type=float, default=0.0)
    p.add_argument(
        "--models",
        default="all-MiniLM-L6-v2,BAAI/bge-m3",
        help="Comma-separated embedding models to compare.",
    )
    p.add_argument(
        "--csv-out",
        default="",
        help="Optional output CSV path with per-example predictions.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise ValueError("No embedding models provided.")

    lang = Language.EN if args.language == "en" else Language.PL
    if lang != Language.EN:
        print("Note: default dataset labels are English review stars; EN anchors are recommended.")

    texts, labels = _load_hf_amazon_reviews(limit=args.limit, seed=args.seed)
    print(f"Loaded real dataset rows: {len(labels)}")
    print(f"Anchor sets: {len(get_anchor_sets(lang))} | temperature={args.temperature} | epsilon={args.epsilon}")

    model_to_scores: dict[str, np.ndarray] = {}
    model_to_metrics: dict[str, EvalMetrics] = {}
    for model in models:
        scores = _score_texts_ssr(
            texts,
            model_name=model,
            language=lang,
            temperature=args.temperature,
            epsilon=args.epsilon,
        )
        metrics = _evaluate(scores, labels, model=model)
        model_to_scores[model] = scores
        model_to_metrics[model] = metrics
        _print_metrics(metrics)

    if len(models) >= 2:
        for i in range(len(models) - 1):
            for j in range(i + 1, len(models)):
                a, b = models[i], models[j]
                _compare_models(a, model_to_scores[a], b, model_to_scores[b])

    if args.csv_out:
        import pandas as pd

        out = pd.DataFrame({"text": texts, "label": labels})
        for model in models:
            out[f"score_{model}"] = model_to_scores[model]
        out_path = Path(args.csv_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"\nSaved per-example predictions: {out_path}")


if __name__ == "__main__":
    main()
