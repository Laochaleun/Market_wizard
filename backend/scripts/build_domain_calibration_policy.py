#!/usr/bin/env python3
"""Build domain-aware calibration policy artifact."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import HfFileSystem

from app.i18n import DEFAULT_ANCHOR_VARIANT, Language, get_anchor_sets, get_anchor_variants
from app.services.embedding_client import LocalEmbeddingClient
from app.services.score_calibration import (
    DomainCalibrationPolicy,
    IsotonicCalibrator,
    fit_isotonic_calibrator,
)


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


def _response_embeddings_to_pmf(response_embeddings: np.ndarray, likert_embeddings: np.ndarray, epsilon: float) -> np.ndarray:
    if response_embeddings.shape[0] == 0:
        return np.empty((0, 5))
    m_right = likert_embeddings / np.linalg.norm(likert_embeddings, axis=0)[None, :]
    m_left = response_embeddings / np.linalg.norm(response_embeddings, axis=1)[:, None]
    cos = (1.0 + m_left.dot(m_right)) / 2.0
    cos_min = cos.min(axis=1)[:, None]
    numerator = cos - cos_min
    if epsilon > 0:
        mins = np.argmin(cos, axis=1)
        for i, j in enumerate(mins):
            numerator[i, j] += epsilon
    denominator = cos.sum(axis=1)[:, None] - cos.shape[1] * cos_min + epsilon
    return numerator / denominator


def _apply_temperature(pmf: np.ndarray, temperature: float) -> np.ndarray:
    if temperature == 1.0:
        return pmf
    if temperature == 0.0:
        out = np.zeros_like(pmf)
        idx = np.argmax(pmf, axis=1)
        out[np.arange(len(idx)), idx] = 1.0
        return out
    scaled = pmf ** (1.0 / temperature)
    return scaled / scaled.sum(axis=1)[:, None]


def _score_texts(
    client: LocalEmbeddingClient,
    texts: list[str],
    *,
    language: Language,
    anchor_variant: str,
    temperature: float,
    epsilon: float,
) -> np.ndarray:
    anchors = get_anchor_sets(language, variant=anchor_variant)
    anchor_mats: list[np.ndarray] = []
    for a in anchors:
        anchor_texts = [a[i] for i in range(1, 6)]
        emb = client.embed(anchor_texts)
        anchor_mats.append(emb.T)
    responses = client.embed(texts)
    pmf_sets = [_response_embeddings_to_pmf(responses, m, epsilon=epsilon) for m in anchor_mats]
    pmf_avg = np.mean(np.stack(pmf_sets, axis=0), axis=0)
    pmf_avg = pmf_avg / pmf_avg.sum(axis=1)[:, None]
    pmf_avg = _apply_temperature(pmf_avg, temperature=temperature)
    return pmf_avg.dot(np.arange(1, 6, dtype=float))


def _load_yelp(limit: int, seed: int) -> tuple[list[str], np.ndarray]:
    ds = load_dataset("Yelp/yelp_review_full", split="test")
    rows = [
        (str(x["text"]).strip(), int(x["label"]) + 1)
        for x in ds
        if str(x["text"]).strip() and int(x["label"]) in {0, 1, 2, 3, 4}
    ]
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    rows = rows[:limit]
    return [x for x, _ in rows], np.array([y for _, y in rows], dtype=float)


def _load_app_reviews(limit: int, seed: int) -> tuple[list[str], np.ndarray]:
    ds = load_dataset("app_reviews", split="train")
    rows = [
        (str(x["review"]).strip(), int(x["star"]))
        for x in ds
        if str(x["review"]).strip() and 1 <= int(x["star"]) <= 5
    ]
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    rows = rows[:limit]
    return [x for x, _ in rows], np.array([y for _, y in rows], dtype=float)


def _load_allegro(limit: int, seed: int) -> tuple[list[str], np.ndarray]:
    ds = load_dataset("allegro/klej-allegro-reviews", split="train")
    rows = [
        (str(x["text"]).strip(), int(round(float(x["rating"]))))
        for x in ds
        if str(x["text"]).strip() and 1 <= int(round(float(x["rating"]))) <= 5
    ]
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    rows = rows[:limit]
    return [x for x, _ in rows], np.array([y for _, y in rows], dtype=float)


def _load_amazon_2023(limit_per_category: int, max_scan: int, seed: int) -> tuple[list[str], np.ndarray]:
    fs = HfFileSystem()
    all_rows: list[tuple[str, int]] = []
    for i, cat in enumerate(CATEGORIES_20):
        rng = np.random.default_rng(seed + i)
        reservoir: list[tuple[str, int]] = []
        seen = 0
        path = f"datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories/{cat}.jsonl"
        with fs.open(path, "r") as f:
            for line in f:
                if max_scan > 0 and seen >= max_scan:
                    break
                seen += 1
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                text = (obj.get("text") or "").strip() or (obj.get("title") or "").strip()
                rating = obj.get("rating")
                if not text or rating is None:
                    continue
                try:
                    y = int(round(float(rating)))
                except Exception:
                    continue
                if not (1 <= y <= 5):
                    continue
                row = (text, y)
                if len(reservoir) < limit_per_category:
                    reservoir.append(row)
                else:
                    j = rng.integers(0, seen)
                    if j < limit_per_category:
                        reservoir[j] = row
        all_rows.extend(reservoir)
    return [x for x, _ in all_rows], np.array([y for _, y in all_rows], dtype=float)


def _fit_domain_calibrator(scores: np.ndarray, labels: np.ndarray, holdout_ratio: float, seed: int) -> tuple[IsotonicCalibrator, dict[str, float]]:
    n = labels.size
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_hold = int(round(n * holdout_ratio))
    n_hold = max(1, min(n - 1, n_hold))
    hold_idx = idx[:n_hold]
    train_idx = idx[n_hold:]
    cal = fit_isotonic_calibrator(scores[train_idx], labels[train_idx])

    def mae(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.abs(a - b).mean())

    raw_train = scores[train_idx]
    cal_train = cal.transform(raw_train)
    raw_hold = scores[hold_idx]
    cal_hold = cal.transform(raw_hold)
    return cal, {
        "train_n": float(train_idx.size),
        "holdout_n": float(hold_idx.size),
        "train_mae_raw": mae(raw_train, labels[train_idx]),
        "train_mae_cal": mae(cal_train, labels[train_idx]),
        "holdout_mae_raw": mae(raw_hold, labels[hold_idx]),
        "holdout_mae_cal": mae(cal_hold, labels[hold_idx]),
    }


def _off1(pred: np.ndarray, labels: np.ndarray) -> float:
    rounded = np.clip(np.rint(pred), 1, 5).astype(int)
    yi = labels.astype(int)
    return float((np.abs(rounded - yi) <= 1).mean())


def _exact(pred: np.ndarray, labels: np.ndarray) -> float:
    rounded = np.clip(np.rint(pred), 1, 5).astype(int)
    yi = labels.astype(int)
    return float((rounded == yi).mean())


def _blend_calibrator(cal: IsotonicCalibrator, alpha: float) -> IsotonicCalibrator:
    x = cal.x_sorted.astype(float)
    y = x + alpha * (cal.y_fitted.astype(float) - x)
    return IsotonicCalibrator(
        x_sorted=x,
        y_fitted=y,
        clip_min=cal.clip_min,
        clip_max=cal.clip_max,
    )


def _fit_domain_calibrator_off1_tuned(
    scores: np.ndarray,
    labels: np.ndarray,
    holdout_ratio: float,
    seed: int,
) -> tuple[IsotonicCalibrator, dict[str, float]]:
    n = labels.size
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_hold = int(round(n * holdout_ratio))
    n_hold = max(1, min(n - 1, n_hold))
    hold_idx = idx[:n_hold]
    train_idx = idx[n_hold:]

    train_scores = scores[train_idx]
    train_labels = labels[train_idx]
    hold_scores = scores[hold_idx]
    hold_labels = labels[hold_idx]

    iso_train = fit_isotonic_calibrator(train_scores, train_labels)
    iso_hold = iso_train.transform(hold_scores)

    best_alpha = 1.0
    best_tuple = (-1.0, float("inf"), -1.0)  # off1 max, mae min, exact max
    for alpha in np.linspace(0.0, 1.0, 21):
        pred = np.clip(hold_scores + alpha * (iso_hold - hold_scores), 1.0, 5.0)
        off1 = _off1(pred, hold_labels)
        mae = float(np.abs(pred - hold_labels).mean())
        exact = _exact(pred, hold_labels)
        key = (off1, -mae, exact)
        if key > best_tuple:
            best_tuple = key
            best_alpha = float(alpha)

    iso_full = fit_isotonic_calibrator(scores, labels)
    blended = _blend_calibrator(iso_full, best_alpha)

    raw_hold = hold_scores
    cal_hold = blended.transform(hold_scores)
    raw_mae = float(np.abs(raw_hold - hold_labels).mean())
    cal_mae = float(np.abs(cal_hold - hold_labels).mean())
    return blended, {
        "train_n": float(train_idx.size),
        "holdout_n": float(hold_idx.size),
        "best_alpha": float(best_alpha),
        "holdout_mae_raw": raw_mae,
        "holdout_mae_cal": cal_mae,
        "holdout_off1_raw": _off1(raw_hold, hold_labels),
        "holdout_off1_cal": _off1(cal_hold, hold_labels),
        "holdout_exact_raw": _exact(raw_hold, hold_labels),
        "holdout_exact_cal": _exact(cal_hold, hold_labels),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build domain calibration policy.")
    p.add_argument("--model", default="BAAI/bge-m3")
    p.add_argument("--anchor-variant", default=DEFAULT_ANCHOR_VARIANT, choices=get_anchor_variants())
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--epsilon", type=float, default=0.0)
    p.add_argument("--limit-yelp", type=int, default=4000)
    p.add_argument("--limit-app", type=int, default=4000)
    p.add_argument("--limit-allegro", type=int, default=4000)
    p.add_argument("--amazon-per-category", type=int, default=250)
    p.add_argument("--amazon-max-scan", type=int, default=25000)
    p.add_argument("--holdout-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--optimize",
        choices=["mae", "off1"],
        default="off1",
        help="Objective used to tune domain calibrator on holdout.",
    )
    p.add_argument(
        "--out",
        default="/Users/pawel/Market_wizard/backend/app/data/ssr_calibration_policy_default.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Model={args.model} anchor_variant={args.anchor_variant} T={args.temperature} eps={args.epsilon}")
    client = LocalEmbeddingClient(model_name=args.model)

    # General long-review domain (EN): Yelp reviews
    yelp_x, yelp_y = _load_yelp(args.limit_yelp, args.seed)
    gen_x = yelp_x
    gen_y = yelp_y
    gen_scores = _score_texts(
        client,
        gen_x,
        language=Language.EN,
        anchor_variant=args.anchor_variant,
        temperature=args.temperature,
        epsilon=args.epsilon,
    )
    if args.optimize == "off1":
        general_cal, general_diag = _fit_domain_calibrator_off1_tuned(
            gen_scores,
            gen_y,
            holdout_ratio=args.holdout_ratio,
            seed=args.seed,
        )
    else:
        general_cal, general_diag = _fit_domain_calibrator(
            gen_scores,
            gen_y,
            holdout_ratio=args.holdout_ratio,
            seed=args.seed,
        )

    # Purchase-intent short domains:
    # - EN: Amazon 2023 + App Reviews
    # - PL: Allegro
    app_x, app_y = _load_app_reviews(args.limit_app, args.seed + 1)
    amz_x, amz_y = _load_amazon_2023(args.amazon_per_category, args.amazon_max_scan, args.seed + 2)
    all_x, all_y = _load_allegro(args.limit_allegro, args.seed + 3)
    amz_scores = _score_texts(
        client,
        amz_x,
        language=Language.EN,
        anchor_variant=args.anchor_variant,
        temperature=args.temperature,
        epsilon=args.epsilon,
    )
    app_scores = _score_texts(
        client,
        app_x,
        language=Language.EN,
        anchor_variant=args.anchor_variant,
        temperature=args.temperature,
        epsilon=args.epsilon,
    )
    allegro_scores = _score_texts(
        client,
        all_x,
        language=Language.PL,
        anchor_variant=args.anchor_variant,
        temperature=args.temperature,
        epsilon=args.epsilon,
    )
    ecom_en_y = np.concatenate([amz_y, app_y])
    ecom_en_scores = np.concatenate([amz_scores, app_scores])
    ecom_pl_y = all_y
    ecom_pl_scores = allegro_scores
    ecom_y = np.concatenate([ecom_en_y, ecom_pl_y])
    ecom_scores = np.concatenate([ecom_en_scores, ecom_pl_scores])

    def fit_domain(scores: np.ndarray, labels: np.ndarray, seed: int):
        if args.optimize == "off1":
            return _fit_domain_calibrator_off1_tuned(
                scores,
                labels,
                holdout_ratio=args.holdout_ratio,
                seed=seed,
            )
        return _fit_domain_calibrator(
            scores,
            labels,
            holdout_ratio=args.holdout_ratio,
            seed=seed,
        )

    ecommerce_en_cal, ecommerce_en_diag = fit_domain(
        ecom_en_scores,
        ecom_en_y,
        args.seed + 7,
    )
    ecommerce_pl_cal, ecommerce_pl_diag = fit_domain(
        ecom_pl_scores,
        ecom_pl_y,
        args.seed + 8,
    )
    ecommerce_cal, ecommerce_diag = fit_domain(
        ecom_scores,
        ecom_y,
        args.seed + 9,
    )

    policy = DomainCalibrationPolicy(
        default_domain="general",
        calibrators={
            "general": general_cal,
            "review_long_en": general_cal,
            "purchase_intent_short_en": ecommerce_en_cal,
            "purchase_intent_short_pl": ecommerce_pl_cal,
            "purchase_intent": ecommerce_cal,
            "ecommerce": ecommerce_cal,
        },
    )
    out = Path(args.out).expanduser().resolve()
    policy.save_json(
        out,
        metadata={
            "created_at": datetime.now().isoformat(),
            "model": args.model,
            "anchor_variant": args.anchor_variant,
            "temperature": args.temperature,
            "epsilon": args.epsilon,
            "optimize": args.optimize,
            "general_train_rows": int(gen_y.size),
            "ecommerce_train_rows": int(ecom_y.size),
            "purchase_intent_short_en_train_rows": int(ecom_en_y.size),
            "purchase_intent_short_pl_train_rows": int(ecom_pl_y.size),
            "purchase_intent_train_rows": int(ecom_y.size),
            "general_diagnostics": general_diag,
            "review_long_en_diagnostics": general_diag,
            "purchase_intent_short_en_diagnostics": ecommerce_en_diag,
            "purchase_intent_short_pl_diagnostics": ecommerce_pl_diag,
            "ecommerce_diagnostics": ecommerce_diag,
            "purchase_intent_diagnostics": ecommerce_diag,
        },
    )
    print(f"Saved policy: {out}")
    print(f"General holdout MAE raw->cal: {general_diag['holdout_mae_raw']:.4f} -> {general_diag['holdout_mae_cal']:.4f}")
    print(f"Ecommerce holdout MAE raw->cal: {ecommerce_diag['holdout_mae_raw']:.4f} -> {ecommerce_diag['holdout_mae_cal']:.4f}")


if __name__ == "__main__":
    main()
