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
    AnyCalibrator,
    EntropyAwareCalibrator,
    IsotonicCalibrator,
    PiecewiseIsotonicCalibrator,
    Piecewise3IsotonicCalibrator,
    TrustRegionCalibrator,
    fit_isotonic_calibrator,
    fit_piecewise_isotonic_calibrator,
    fit_piecewise3_isotonic_calibrator,
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
) -> tuple[np.ndarray, np.ndarray]:
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
    scores = pmf_avg.dot(np.arange(1, 6, dtype=float))
    entropy = -np.sum(pmf_avg * np.log(np.clip(pmf_avg, 1e-12, 1.0)), axis=1) / np.log(5.0)
    return scores, entropy


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


def _fit_base_calibrator(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    calibration_mode: str,
    piecewise_split_quantile: float,
    piecewise3_split_quantiles: tuple[float, float],
    uncertainty: np.ndarray | None = None,
    entropy_split_quantile: float = 0.6,
) -> AnyCalibrator:
    if calibration_mode == "entropy_piecewise":
        if uncertainty is None or uncertainty.shape != scores.shape:
            return fit_piecewise_isotonic_calibrator(scores, labels, split_quantile=piecewise_split_quantile)
        thr = float(np.quantile(uncertainty, float(np.clip(entropy_split_quantile, 0.2, 0.8))))
        low_mask = uncertainty <= thr
        high_mask = ~low_mask
        if low_mask.sum() < 10 or high_mask.sum() < 10:
            return fit_piecewise_isotonic_calibrator(scores, labels, split_quantile=piecewise_split_quantile)
        low_cal = fit_piecewise_isotonic_calibrator(scores[low_mask], labels[low_mask], split_quantile=piecewise_split_quantile)
        high_cal = fit_piecewise_isotonic_calibrator(scores[high_mask], labels[high_mask], split_quantile=piecewise_split_quantile)
        return EntropyAwareCalibrator(entropy_threshold=thr, low_entropy=low_cal, high_entropy=high_cal)
    if calibration_mode == "piecewise3":
        return fit_piecewise3_isotonic_calibrator(
            scores,
            labels,
            split_quantiles=piecewise3_split_quantiles,
        )
    if calibration_mode == "piecewise":
        return fit_piecewise_isotonic_calibrator(
            scores,
            labels,
            split_quantile=piecewise_split_quantile,
        )
    return fit_isotonic_calibrator(scores, labels)


def _fit_domain_calibrator(
    scores: np.ndarray,
    labels: np.ndarray,
    holdout_ratio: float,
    seed: int,
    *,
    calibration_mode: str,
    piecewise_split_quantile: float,
    piecewise3_split_quantiles: tuple[float, float],
    mae_off1_floor: float,
    mae_off1_max_drop: float,
    uncertainty: np.ndarray | None = None,
    entropy_split_quantile: float = 0.6,
) -> tuple[AnyCalibrator, dict[str, float]]:
    n = labels.size
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_hold = int(round(n * holdout_ratio))
    n_hold = max(1, min(n - 1, n_hold))
    hold_idx = idx[:n_hold]
    train_idx = idx[n_hold:]
    base_cal = _fit_base_calibrator(
        scores[train_idx],
        labels[train_idx],
        calibration_mode=calibration_mode,
        piecewise_split_quantile=piecewise_split_quantile,
        piecewise3_split_quantiles=piecewise3_split_quantiles,
        uncertainty=uncertainty[train_idx] if uncertainty is not None else None,
        entropy_split_quantile=entropy_split_quantile,
    )
    hold_unc = uncertainty[hold_idx] if uncertainty is not None else None
    hold_base = base_cal.transform(scores[hold_idx], uncertainty=hold_unc)
    raw_hold = scores[hold_idx]
    hold_labels = labels[hold_idx]

    # For MAE optimization, keep Off1 from drifting too low on holdout.
    raw_off1 = _off1(raw_hold, hold_labels)
    off1_floor = float(mae_off1_floor) if mae_off1_floor >= 0 else max(0.0, raw_off1 - float(mae_off1_max_drop))
    best_alpha = 1.0
    best_mae = float("inf")
    constrained_hit = False
    fallback_alpha = 1.0
    fallback_mae = float("inf")
    for alpha in np.linspace(0.0, 1.0, 21):
        pred = np.clip(raw_hold + alpha * (hold_base - raw_hold), 1.0, 5.0)
        mae = float(np.abs(pred - hold_labels).mean())
        if mae < fallback_mae:
            fallback_mae = mae
            fallback_alpha = float(alpha)
        if _off1(pred, hold_labels) >= off1_floor and mae < best_mae:
            best_mae = mae
            best_alpha = float(alpha)
            constrained_hit = True
    if not constrained_hit:
        best_alpha = fallback_alpha

    cal = _blend_calibrator(base_cal, best_alpha)

    def mae(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.abs(a - b).mean())

    raw_train = scores[train_idx]
    train_unc = uncertainty[train_idx] if uncertainty is not None else None
    cal_train = cal.transform(raw_train, uncertainty=train_unc)
    cal_hold = cal.transform(raw_hold, uncertainty=hold_unc)
    return cal, {
        "train_n": float(train_idx.size),
        "holdout_n": float(hold_idx.size),
        "best_alpha": float(best_alpha),
        "holdout_off1_floor": float(off1_floor),
        "holdout_off1_constraint_hit": float(1.0 if constrained_hit else 0.0),
        "train_mae_raw": mae(raw_train, labels[train_idx]),
        "train_mae_cal": mae(cal_train, labels[train_idx]),
        "holdout_mae_raw": mae(raw_hold, labels[hold_idx]),
        "holdout_mae_cal": mae(cal_hold, labels[hold_idx]),
        "holdout_off1_raw": _off1(raw_hold, labels[hold_idx]),
        "holdout_off1_cal": _off1(cal_hold, labels[hold_idx]),
        "holdout_exact_raw": _exact(raw_hold, labels[hold_idx]),
        "holdout_exact_cal": _exact(cal_hold, labels[hold_idx]),
    }


def _off1(pred: np.ndarray, labels: np.ndarray) -> float:
    rounded = np.clip(np.rint(pred), 1, 5).astype(int)
    yi = labels.astype(int)
    return float((np.abs(rounded - yi) <= 1).mean())


def _exact(pred: np.ndarray, labels: np.ndarray) -> float:
    rounded = np.clip(np.rint(pred), 1, 5).astype(int)
    yi = labels.astype(int)
    return float((rounded == yi).mean())


def _blend_isotonic(cal: IsotonicCalibrator, alpha: float) -> IsotonicCalibrator:
    x = cal.x_sorted.astype(float)
    y = x + alpha * (cal.y_fitted.astype(float) - x)
    return IsotonicCalibrator(
        x_sorted=x,
        y_fitted=y,
        clip_min=cal.clip_min,
        clip_max=cal.clip_max,
    )


def _blend_calibrator(cal: AnyCalibrator, alpha: float) -> AnyCalibrator:
    if isinstance(cal, EntropyAwareCalibrator):
        return EntropyAwareCalibrator(
            entropy_threshold=cal.entropy_threshold,
            low_entropy=_blend_calibrator(cal.low_entropy, alpha),
            high_entropy=_blend_calibrator(cal.high_entropy, alpha),
        )
    if isinstance(cal, TrustRegionCalibrator):
        return TrustRegionCalibrator(
            base=_blend_calibrator(cal.base, alpha),
            max_delta=cal.max_delta,
        )
    if isinstance(cal, Piecewise3IsotonicCalibrator):
        return Piecewise3IsotonicCalibrator(
            split_x1=cal.split_x1,
            split_x2=cal.split_x2,
            lower=_blend_isotonic(cal.lower, alpha),
            middle=_blend_isotonic(cal.middle, alpha),
            upper=_blend_isotonic(cal.upper, alpha),
        )
    if isinstance(cal, PiecewiseIsotonicCalibrator):
        return PiecewiseIsotonicCalibrator(
            split_x=cal.split_x,
            lower=_blend_isotonic(cal.lower, alpha),
            upper=_blend_isotonic(cal.upper, alpha),
        )
    return _blend_isotonic(cal, alpha)


def _fit_domain_calibrator_off1_tuned(
    scores: np.ndarray,
    labels: np.ndarray,
    holdout_ratio: float,
    seed: int,
    *,
    calibration_mode: str,
    piecewise_split_quantile: float,
    piecewise3_split_quantiles: tuple[float, float],
    uncertainty: np.ndarray | None = None,
    entropy_split_quantile: float = 0.6,
) -> tuple[AnyCalibrator, dict[str, float]]:
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

    iso_train = _fit_base_calibrator(
        train_scores,
        train_labels,
        calibration_mode=calibration_mode,
        piecewise_split_quantile=piecewise_split_quantile,
        piecewise3_split_quantiles=piecewise3_split_quantiles,
        uncertainty=uncertainty[train_idx] if uncertainty is not None else None,
        entropy_split_quantile=entropy_split_quantile,
    )
    hold_unc = uncertainty[hold_idx] if uncertainty is not None else None
    iso_hold = iso_train.transform(hold_scores, uncertainty=hold_unc)

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

    iso_full = _fit_base_calibrator(
        scores,
        labels,
        calibration_mode=calibration_mode,
        piecewise_split_quantile=piecewise_split_quantile,
        piecewise3_split_quantiles=piecewise3_split_quantiles,
        uncertainty=uncertainty,
        entropy_split_quantile=entropy_split_quantile,
    )
    blended = _blend_calibrator(iso_full, best_alpha)

    raw_hold = hold_scores
    cal_hold = blended.transform(hold_scores, uncertainty=hold_unc)
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
        "--calibration-mode",
        choices=["isotonic", "piecewise", "piecewise3", "entropy_piecewise"],
        default="piecewise",
        help="Calibration mapping family for each domain.",
    )
    p.add_argument(
        "--purchase-calibration-mode",
        choices=["same_as_global", "piecewise", "piecewise3"],
        default="piecewise3",
        help="Calibration mode override for purchase domains.",
    )
    p.add_argument(
        "--piecewise-split-quantile",
        type=float,
        default=0.6,
        help="Split quantile for piecewise mode (ignored for isotonic).",
    )
    p.add_argument("--piecewise3-split-q1", type=float, default=0.4)
    p.add_argument("--piecewise3-split-q2", type=float, default=0.75)
    p.add_argument("--entropy-split-quantile", type=float, default=0.6)
    p.add_argument(
        "--mae-off1-floor",
        type=float,
        default=-1.0,
        help="Absolute Off1 floor for MAE objective on holdout. Negative means auto floor.",
    )
    p.add_argument(
        "--mae-off1-max-drop",
        type=float,
        default=0.002,
        help="Allowed Off1 drop vs raw holdout when MAE objective is used and floor is auto.",
    )
    p.add_argument(
        "--optimize",
        choices=["mae", "off1", "off1_constrained_mae"],
        default="off1_constrained_mae",
        help="Objective used to tune domain calibrator on holdout.",
    )
    p.add_argument(
        "--objective-eval-per-domain",
        type=int,
        default=1200,
        help="Rows per domain used for external objective mix when optimize=off1_constrained_mae.",
    )
    p.add_argument(
        "--objective-off1-min",
        type=float,
        default=0.92,
        help="Minimum Off1 required on objective mix for optimize=off1_constrained_mae.",
    )
    p.add_argument(
        "--objective-alpha-grid",
        default="0.7,0.8,0.9,1.0",
        help="Comma-separated alpha candidates for domain-specific objective search.",
    )
    p.add_argument(
        "--objective-piecewise3-grid",
        default="0.35:0.70,0.40:0.75,0.45:0.80",
        help="Comma-separated q1:q2 candidates for purchase piecewise3 objective search.",
    )
    p.add_argument("--trust-region-delta-general", type=float, default=-1.0)
    p.add_argument("--trust-region-delta-purchase-en", type=float, default=-1.0)
    p.add_argument("--trust-region-delta-purchase-pl", type=float, default=-1.0)
    p.add_argument("--trust-region-delta-purchase-agg", type=float, default=-1.0)
    p.add_argument(
        "--out",
        default="/Users/pawel/Market_wizard/backend/app/data/ssr_calibration_policy_default.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"Model={args.model} anchor_variant={args.anchor_variant} "
        f"T={args.temperature} eps={args.epsilon} mode={args.calibration_mode}"
    )
    piecewise3_split_quantiles = (float(args.piecewise3_split_q1), float(args.piecewise3_split_q2))
    alpha_grid = [
        float(x.strip()) for x in str(args.objective_alpha_grid).split(",") if x.strip()
    ]
    alpha_grid = sorted(set(a for a in alpha_grid if 0.0 <= a <= 1.0))
    if not alpha_grid:
        alpha_grid = [1.0]

    def parse_qpair_grid(raw: str) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for tok in str(raw).split(","):
            tok = tok.strip()
            if not tok or ":" not in tok:
                continue
            left, right = tok.split(":", 1)
            try:
                q1 = float(left.strip())
                q2 = float(right.strip())
            except Exception:
                continue
            if 0.0 < q1 < q2 < 1.0:
                out.append((q1, q2))
        if not out:
            out = [piecewise3_split_quantiles]
        return out

    piecewise3_grid = parse_qpair_grid(args.objective_piecewise3_grid)

    def mode_for_domain(domain_name: str) -> str:
        if not domain_name.startswith("purchase"):
            return args.calibration_mode
        if args.purchase_calibration_mode == "same_as_global":
            return args.calibration_mode
        return args.purchase_calibration_mode

    def maybe_trust(cal: AnyCalibrator, delta: float) -> AnyCalibrator:
        if delta is None or float(delta) < 0:
            return cal
        return TrustRegionCalibrator(base=cal, max_delta=float(delta))

    def apply_trust_to_policy(curr_policy: DomainCalibrationPolicy) -> DomainCalibrationPolicy:
        cals = dict(curr_policy.calibrators)
        cals["general"] = maybe_trust(cals["general"], args.trust_region_delta_general)
        cals["review_long_en"] = maybe_trust(cals["review_long_en"], args.trust_region_delta_general)
        cals["purchase_intent_short_en"] = maybe_trust(cals["purchase_intent_short_en"], args.trust_region_delta_purchase_en)
        cals["purchase_intent_short_pl"] = maybe_trust(cals["purchase_intent_short_pl"], args.trust_region_delta_purchase_pl)
        cals["purchase_intent"] = maybe_trust(cals["purchase_intent"], args.trust_region_delta_purchase_agg)
        cals["ecommerce"] = maybe_trust(cals["ecommerce"], args.trust_region_delta_purchase_agg)
        return DomainCalibrationPolicy(default_domain=curr_policy.default_domain, calibrators=cals)

    client = LocalEmbeddingClient(model_name=args.model)

    # General long-review domain (EN): Yelp reviews
    yelp_x, yelp_y = _load_yelp(args.limit_yelp, args.seed)
    gen_x = yelp_x
    gen_y = yelp_y
    gen_scores, gen_unc = _score_texts(
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
            calibration_mode=mode_for_domain("general"),
            piecewise_split_quantile=args.piecewise_split_quantile,
            piecewise3_split_quantiles=piecewise3_split_quantiles,
            uncertainty=gen_unc,
            entropy_split_quantile=args.entropy_split_quantile,
        )
    else:
        general_cal, general_diag = _fit_domain_calibrator(
            gen_scores,
            gen_y,
            holdout_ratio=args.holdout_ratio,
            seed=args.seed,
            calibration_mode=mode_for_domain("general"),
            piecewise_split_quantile=args.piecewise_split_quantile,
            piecewise3_split_quantiles=piecewise3_split_quantiles,
            mae_off1_floor=args.mae_off1_floor,
            mae_off1_max_drop=args.mae_off1_max_drop,
            uncertainty=gen_unc,
            entropy_split_quantile=args.entropy_split_quantile,
        )

    # Purchase-intent short domains:
    # - EN: Amazon 2023 + App Reviews
    # - PL: Allegro
    app_x, app_y = _load_app_reviews(args.limit_app, args.seed + 1)
    amz_x, amz_y = _load_amazon_2023(args.amazon_per_category, args.amazon_max_scan, args.seed + 2)
    all_x, all_y = _load_allegro(args.limit_allegro, args.seed + 3)
    amz_scores, amz_unc = _score_texts(
        client,
        amz_x,
        language=Language.EN,
        anchor_variant=args.anchor_variant,
        temperature=args.temperature,
        epsilon=args.epsilon,
    )
    app_scores, app_unc = _score_texts(
        client,
        app_x,
        language=Language.EN,
        anchor_variant=args.anchor_variant,
        temperature=args.temperature,
        epsilon=args.epsilon,
    )
    allegro_scores, allegro_unc = _score_texts(
        client,
        all_x,
        language=Language.PL,
        anchor_variant=args.anchor_variant,
        temperature=args.temperature,
        epsilon=args.epsilon,
    )
    ecom_en_y = np.concatenate([amz_y, app_y])
    ecom_en_scores = np.concatenate([amz_scores, app_scores])
    ecom_en_unc = np.concatenate([amz_unc, app_unc])
    ecom_pl_y = all_y
    ecom_pl_scores = allegro_scores
    ecom_pl_unc = allegro_unc
    ecom_y = np.concatenate([ecom_en_y, ecom_pl_y])
    ecom_scores = np.concatenate([ecom_en_scores, ecom_pl_scores])
    ecom_unc = np.concatenate([ecom_en_unc, ecom_pl_unc])

    def fit_domain(
        scores: np.ndarray,
        labels: np.ndarray,
        seed: int,
        *,
        domain_name: str,
        override_piecewise3: tuple[float, float] | None = None,
        uncertainty: np.ndarray | None = None,
    ):
        if args.optimize == "off1":
            return _fit_domain_calibrator_off1_tuned(
                scores,
                labels,
                holdout_ratio=args.holdout_ratio,
                seed=seed,
                calibration_mode=mode_for_domain(domain_name),
                piecewise_split_quantile=args.piecewise_split_quantile,
                piecewise3_split_quantiles=override_piecewise3 or piecewise3_split_quantiles,
                uncertainty=uncertainty,
                entropy_split_quantile=args.entropy_split_quantile,
            )
        return _fit_domain_calibrator(
            scores,
            labels,
            holdout_ratio=args.holdout_ratio,
            seed=seed,
            calibration_mode=mode_for_domain(domain_name),
            piecewise_split_quantile=args.piecewise_split_quantile,
            piecewise3_split_quantiles=override_piecewise3 or piecewise3_split_quantiles,
            mae_off1_floor=args.mae_off1_floor,
            mae_off1_max_drop=args.mae_off1_max_drop,
            uncertainty=uncertainty,
            entropy_split_quantile=args.entropy_split_quantile,
        )

    ecommerce_en_cal, ecommerce_en_diag = fit_domain(
        ecom_en_scores,
        ecom_en_y,
        args.seed + 7,
        domain_name="purchase_intent_short_en",
        uncertainty=ecom_en_unc,
    )
    ecommerce_pl_cal, ecommerce_pl_diag = fit_domain(
        ecom_pl_scores,
        ecom_pl_y,
        args.seed + 8,
        domain_name="purchase_intent_short_pl",
        uncertainty=ecom_pl_unc,
    )
    ecommerce_cal, ecommerce_diag = fit_domain(
        ecom_scores,
        ecom_y,
        args.seed + 9,
        domain_name="purchase_intent",
        uncertainty=ecom_unc,
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

    def sample_eval(scores: np.ndarray, labels: np.ndarray, limit: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        if limit <= 0 or limit >= labels.size:
            return scores, labels
        rng = np.random.default_rng(seed)
        idx = rng.permutation(labels.size)[:limit]
        return scores[idx], labels[idx]

    objective_diag: dict[str, float | str] | None = None
    if args.optimize == "off1_constrained_mae":
        eval_sets = [
            ("general",) + sample_eval(gen_scores, gen_y, args.objective_eval_per_domain, args.seed + 101),
            ("purchase_intent_short_en",) + sample_eval(ecom_en_scores, ecom_en_y, args.objective_eval_per_domain, args.seed + 102),
            ("purchase_intent_short_pl",) + sample_eval(ecom_pl_scores, ecom_pl_y, args.objective_eval_per_domain, args.seed + 103),
        ]
        best_cfg = (1.0, 1.0, 1.0, piecewise3_split_quantiles[0], piecewise3_split_quantiles[1])
        best_mae = float("inf")
        best_off1 = -1.0
        constrained_hit = False
        fallback_cfg = best_cfg
        fallback_tuple = (-1.0, float("inf"))  # off1 max, mae min

        def eval_policy(curr_policy: DomainCalibrationPolicy) -> tuple[float, float]:
            total_n = 0
            total_abs = 0.0
            off1_hits = 0
            for domain_hint, s_eval, y_eval in eval_sets:
                cal = curr_policy.calibrators[domain_hint]
                pred = cal.transform(s_eval)
                total_n += int(y_eval.size)
                total_abs += float(np.abs(pred - y_eval).sum())
                rounded = np.clip(np.rint(pred), 1, 5).astype(int)
                off1_hits += int((np.abs(rounded - y_eval.astype(int)) <= 1).sum())
            mae = total_abs / max(1, total_n)
            off1 = off1_hits / max(1, total_n)
            return mae, off1

        for q1, q2 in piecewise3_grid:
            c_en, _ = fit_domain(
                ecom_en_scores,
                ecom_en_y,
                args.seed + 107,
                domain_name="purchase_intent_short_en",
                override_piecewise3=(q1, q2),
            )
            c_pl, _ = fit_domain(
                ecom_pl_scores,
                ecom_pl_y,
                args.seed + 108,
                domain_name="purchase_intent_short_pl",
                override_piecewise3=(q1, q2),
            )
            c_all, _ = fit_domain(
                ecom_scores,
                ecom_y,
                args.seed + 109,
                domain_name="purchase_intent",
                override_piecewise3=(q1, q2),
            )
            base_policy = DomainCalibrationPolicy(
                default_domain="general",
                calibrators={
                    "general": general_cal,
                    "review_long_en": general_cal,
                    "purchase_intent_short_en": c_en,
                    "purchase_intent_short_pl": c_pl,
                    "purchase_intent": c_all,
                    "ecommerce": c_all,
                },
            )
            for a_gen in alpha_grid:
                for a_en in alpha_grid:
                    for a_pl in alpha_grid:
                        a_all = float((a_en + a_pl) / 2.0)
                        cal_map = {
                            "general": _blend_calibrator(base_policy.calibrators["general"], a_gen),
                            "review_long_en": _blend_calibrator(base_policy.calibrators["review_long_en"], a_gen),
                            "purchase_intent_short_en": _blend_calibrator(base_policy.calibrators["purchase_intent_short_en"], a_en),
                            "purchase_intent_short_pl": _blend_calibrator(base_policy.calibrators["purchase_intent_short_pl"], a_pl),
                            "purchase_intent": _blend_calibrator(base_policy.calibrators["purchase_intent"], a_all),
                            "ecommerce": _blend_calibrator(base_policy.calibrators["ecommerce"], a_all),
                        }
                        curr_policy = DomainCalibrationPolicy(default_domain="general", calibrators=cal_map)
                        curr_policy = apply_trust_to_policy(curr_policy)
                        mae, off1 = eval_policy(curr_policy)
                        if (off1, -mae) > fallback_tuple:
                            fallback_tuple = (off1, -mae)
                            fallback_cfg = (a_gen, a_en, a_pl, q1, q2)
                        if off1 >= float(args.objective_off1_min) and mae < best_mae:
                            best_mae = mae
                            best_off1 = off1
                            best_cfg = (a_gen, a_en, a_pl, q1, q2)
                            constrained_hit = True
        if not constrained_hit:
            best_cfg = fallback_cfg
            best_off1 = fallback_tuple[0]
            best_mae = -fallback_tuple[1] if np.isfinite(fallback_tuple[1]) else float("inf")

        a_gen, a_en, a_pl, q1_best, q2_best = best_cfg
        c_en_best, _ = fit_domain(
            ecom_en_scores,
            ecom_en_y,
            args.seed + 207,
            domain_name="purchase_intent_short_en",
            override_piecewise3=(q1_best, q2_best),
        )
        c_pl_best, _ = fit_domain(
            ecom_pl_scores,
            ecom_pl_y,
            args.seed + 208,
            domain_name="purchase_intent_short_pl",
            override_piecewise3=(q1_best, q2_best),
        )
        c_all_best, _ = fit_domain(
            ecom_scores,
            ecom_y,
            args.seed + 209,
            domain_name="purchase_intent",
            override_piecewise3=(q1_best, q2_best),
        )
        a_all = float((a_en + a_pl) / 2.0)
        policy = DomainCalibrationPolicy(
            default_domain=policy.default_domain,
            calibrators={
                "general": _blend_calibrator(general_cal, a_gen),
                "review_long_en": _blend_calibrator(general_cal, a_gen),
                "purchase_intent_short_en": _blend_calibrator(c_en_best, a_en),
                "purchase_intent_short_pl": _blend_calibrator(c_pl_best, a_pl),
                "purchase_intent": _blend_calibrator(c_all_best, a_all),
                "ecommerce": _blend_calibrator(c_all_best, a_all),
            },
        )
        policy = apply_trust_to_policy(policy)
        objective_diag = {
            "best_alpha_general": float(a_gen),
            "best_alpha_purchase_en": float(a_en),
            "best_alpha_purchase_pl": float(a_pl),
            "best_q1": float(q1_best),
            "best_q2": float(q2_best),
            "best_objective_mae": float(best_mae),
            "best_objective_off1": float(best_off1),
            "off1_min": float(args.objective_off1_min),
            "objective_eval_per_domain": float(args.objective_eval_per_domain),
            "constraint_hit": float(1.0 if constrained_hit else 0.0),
        }
    else:
        policy = apply_trust_to_policy(policy)
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
            "calibration_mode": args.calibration_mode,
            "purchase_calibration_mode": args.purchase_calibration_mode,
            "piecewise_split_quantile": args.piecewise_split_quantile,
            "piecewise3_split_q1": args.piecewise3_split_q1,
            "piecewise3_split_q2": args.piecewise3_split_q2,
            "mae_off1_floor": args.mae_off1_floor,
            "mae_off1_max_drop": args.mae_off1_max_drop,
            "objective_off1_min": args.objective_off1_min,
            "objective_eval_per_domain": args.objective_eval_per_domain,
            "objective_alpha_grid": args.objective_alpha_grid,
            "objective_piecewise3_grid": args.objective_piecewise3_grid,
            "trust_region_delta_general": args.trust_region_delta_general,
            "trust_region_delta_purchase_en": args.trust_region_delta_purchase_en,
            "trust_region_delta_purchase_pl": args.trust_region_delta_purchase_pl,
            "trust_region_delta_purchase_agg": args.trust_region_delta_purchase_agg,
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
            "objective_optimization": objective_diag,
        },
    )
    print(f"Saved policy: {out}")
    print(f"General holdout MAE raw->cal: {general_diag['holdout_mae_raw']:.4f} -> {general_diag['holdout_mae_cal']:.4f}")
    print(f"Ecommerce holdout MAE raw->cal: {ecommerce_diag['holdout_mae_raw']:.4f} -> {ecommerce_diag['holdout_mae_cal']:.4f}")


if __name__ == "__main__":
    main()
