#!/usr/bin/env python3
"""Benchmark SSR on 20 industry datasets (Amazon Reviews 2023 categories).

Compares embedding models on real star ratings (1-5) across 20 categories.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from huggingface_hub import HfFileSystem
from scipy.stats import spearmanr

from app.i18n import Language, get_anchor_sets
from app.services.embedding_client import LocalEmbeddingClient


CATEGORIES_20 = [
    "Electronics",
    "Books",
    "Clothing_Shoes_and_Jewelry",
    "Home_and_Kitchen",
    "Automotive",
    "Sports_and_Outdoors",
    "Health_and_Personal_Care",
    "Beauty_and_Personal_Care",
    "Toys_and_Games",
    "Video_Games",
    "Office_Products",
    "Pet_Supplies",
    "Grocery_and_Gourmet_Food",
    "Appliances",
    "Tools_and_Home_Improvement",
    "Cell_Phones_and_Accessories",
    "Musical_Instruments",
    "Software",
    "Patio_Lawn_and_Garden",
    "Industrial_and_Scientific",
]


@dataclass
class Metrics:
    n: int
    mae: float
    spearman: float
    exact_acc: float
    off_by_one_acc: float
    mean_pred: float
    mean_true: float


def _scale_pmf(pmf: np.ndarray, temperature: float) -> np.ndarray:
    if temperature == 1.0:
        return pmf
    if temperature == 0.0:
        out = np.zeros_like(pmf)
        idx = int(np.argmax(pmf))
        out[idx] = 1.0
        return out
    hist = pmf ** (1 / temperature)
    return hist / hist.sum()


def _response_embeddings_to_pmf(
    response_embeddings: np.ndarray,
    likert_embeddings: np.ndarray,
    epsilon: float = 0.0,
) -> np.ndarray:
    m_left = response_embeddings
    m_right = likert_embeddings
    if m_left.shape[0] == 0:
        return np.empty((0, m_right.shape[1]))
    norm_right = np.linalg.norm(m_right, axis=0)
    m_right = m_right / norm_right[None, :]
    norm_left = np.linalg.norm(m_left, axis=1)
    m_left = m_left / norm_left[:, None]
    cos = (1 + m_left.dot(m_right)) / 2
    cos_min = cos.min(axis=1)[:, None]
    numerator = cos - cos_min
    if epsilon > 0:
        mins = np.argmin(cos, axis=1)
        for i, j in enumerate(mins):
            numerator[i, j] += epsilon
    denominator = cos.sum(axis=1)[:, None] - cos.shape[1] * cos_min + epsilon
    return numerator / denominator


def _expected_scores_from_texts(
    texts: list[str],
    *,
    model_name: str,
    temperature: float,
    epsilon: float,
) -> np.ndarray:
    client = LocalEmbeddingClient(model_name=model_name)
    anchors = get_anchor_sets(Language.EN)

    # Precompute all anchor embeddings once.
    anchor_mats: list[np.ndarray] = []
    for a in anchors:
        anchor_texts = [a[i] for i in range(1, 6)]
        emb = client.embed(anchor_texts)  # shape (5, d)
        anchor_mats.append(emb.T)  # shape (d, 5)

    responses = client.embed(texts)  # shape (n, d)
    pmf_sets = [_response_embeddings_to_pmf(responses, m, epsilon=epsilon) for m in anchor_mats]
    pmf_avg = np.mean(np.stack(pmf_sets, axis=0), axis=0)

    # Normalize and apply temperature row-wise.
    out_scores = np.zeros(pmf_avg.shape[0], dtype=float)
    for i in range(pmf_avg.shape[0]):
        row = pmf_avg[i]
        row = row / row.sum()
        row = _scale_pmf(row, temperature=temperature)
        out_scores[i] = np.dot(np.arange(1, 6, dtype=float), row)
    return out_scores


def _category_rows(
    category: str,
    *,
    sample_size: int,
    max_scan: int,
    seed: int,
) -> tuple[list[str], list[int]]:
    fs = HfFileSystem()
    path = f"datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories/{category}.jsonl"
    rng = np.random.default_rng(seed)
    reservoir: list[tuple[str, int]] = []
    seen = 0

    with fs.open(path, "r") as f:
        for line in f:
            if max_scan > 0 and seen >= max_scan:
                break
            seen += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rating = obj.get("rating")
            text = (obj.get("text") or "").strip()
            title = (obj.get("title") or "").strip()
            if rating is None:
                continue
            try:
                y = int(round(float(rating)))
            except Exception:
                continue
            if y < 1 or y > 5:
                continue
            if not text and not title:
                continue
            x = text if text else title

            if len(reservoir) < sample_size:
                reservoir.append((x, y))
            else:
                j = rng.integers(0, seen)
                if j < sample_size:
                    reservoir[j] = (x, y)

    texts = [r[0] for r in reservoir]
    labels = [r[1] for r in reservoir]
    return texts, labels


def _metrics(scores: np.ndarray, labels: list[int]) -> Metrics:
    y = np.array(labels, dtype=float)
    rounded = np.clip(np.rint(scores), 1, 5).astype(int)
    yi = y.astype(int)
    abs_err = np.abs(scores - y)
    corr = spearmanr(scores, y).correlation
    corr = float(corr) if corr is not None and not np.isnan(corr) else 0.0
    return Metrics(
        n=len(labels),
        mae=float(abs_err.mean()),
        spearman=corr,
        exact_acc=float((rounded == yi).mean()),
        off_by_one_acc=float((np.abs(rounded - yi) <= 1).mean()),
        mean_pred=float(scores.mean()),
        mean_true=float(y.mean()),
    )


def _print_metrics(prefix: str, m: Metrics) -> None:
    print(
        f"{prefix} n={m.n} mae={m.mae:.4f} spearman={m.spearman:.4f} "
        f"exact={m.exact_acc:.4f} off_by_1={m.off_by_one_acc:.4f} "
        f"mean_pred={m.mean_pred:.4f} mean_true={m.mean_true:.4f}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-per-category", type=int, default=200)
    p.add_argument("--max-scan-per-category", type=int, default=12000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--epsilon", type=float, default=0.0)
    p.add_argument("--models", default="all-MiniLM-L6-v2,BAAI/bge-m3")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if len(models) < 2:
        raise ValueError("Provide at least two models in --models")

    print(f"Categories: {len(CATEGORIES_20)} | sample/category={args.sample_per_category}")
    all_texts: dict[str, list[str]] = {}
    all_labels: dict[str, list[int]] = {}

    for i, cat in enumerate(CATEGORIES_20):
        texts, labels = _category_rows(
            cat,
            sample_size=args.sample_per_category,
            max_scan=args.max_scan_per_category,
            seed=args.seed + i,
        )
        all_texts[cat] = texts
        all_labels[cat] = labels
        print(f"Loaded {cat}: n={len(labels)}")

    model_scores: dict[str, dict[str, np.ndarray]] = {m: {} for m in models}
    model_metrics: dict[str, dict[str, Metrics]] = {m: {} for m in models}

    for model in models:
        print(f"\n=== Scoring model: {model} ===")
        for cat in CATEGORIES_20:
            scores = _expected_scores_from_texts(
                all_texts[cat],
                model_name=model,
                temperature=args.temperature,
                epsilon=args.epsilon,
            )
            m = _metrics(scores, all_labels[cat])
            model_scores[model][cat] = scores
            model_metrics[model][cat] = m
            _print_metrics(f"[{cat}]", m)

    # Aggregate summary (unweighted and weighted by n)
    print("\n=== Aggregate Summary ===")
    for model in models:
        per = list(model_metrics[model].values())
        # Unweighted mean across categories.
        mae_u = statistics.mean(m.mae for m in per)
        sp_u = statistics.mean(m.spearman for m in per)
        exact_u = statistics.mean(m.exact_acc for m in per)
        off_u = statistics.mean(m.off_by_one_acc for m in per)

        # Weighted by number of examples.
        w = np.array([m.n for m in per], dtype=float)
        w = w / w.sum()
        mae_w = float(np.dot(w, np.array([m.mae for m in per])))
        sp_w = float(np.dot(w, np.array([m.spearman for m in per])))
        exact_w = float(np.dot(w, np.array([m.exact_acc for m in per])))
        off_w = float(np.dot(w, np.array([m.off_by_one_acc for m in per])))
        print(
            f"{model}: unweighted(mae={mae_u:.4f}, spearman={sp_u:.4f}, exact={exact_u:.4f}, off1={off_u:.4f}) | "
            f"weighted(mae={mae_w:.4f}, spearman={sp_w:.4f}, exact={exact_w:.4f}, off1={off_w:.4f})"
        )

    base, alt = models[0], models[1]
    diffs = []
    for cat in CATEGORIES_20:
        d = np.abs(model_scores[base][cat] - model_scores[alt][cat])
        diffs.append(d)
    d_all = np.concatenate(diffs) if diffs else np.array([], dtype=float)
    if d_all.size:
        print(
            f"\nEmbedding impact ({base} vs {alt}): "
            f"mean|Δ|={float(d_all.mean()):.4f}, median|Δ|={float(np.median(d_all)):.4f}, max|Δ|={float(d_all.max()):.4f}"
        )


if __name__ == "__main__":
    main()
