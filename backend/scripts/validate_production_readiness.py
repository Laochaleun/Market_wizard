#!/usr/bin/env python3
"""External validation suite for Market Wizard production readiness."""

from __future__ import annotations

import argparse
import json
import os
from glob import glob
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset
from huggingface_hub import HfFileSystem
from scipy.stats import spearmanr

from app.config import get_settings
from app.i18n import (
    DEFAULT_ANCHOR_VARIANT,
    Language,
    get_anchor_sets,
    get_anchor_variants,
)
from app.services.embedding_client import LocalEmbeddingClient
from app.services.score_calibration import DomainCalibrationPolicy, IsotonicCalibrator


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

HF_DATASETS_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "datasets"


@dataclass
class DatasetSlice:
    name: str
    language: Language
    texts: list[str]
    labels: np.ndarray  # float 1..5
    timestamps: np.ndarray | None = None


@dataclass
class Metrics:
    n: int
    mae: float
    spearman: float
    exact: float
    off1: float


@dataclass
class GateResult:
    name: str
    metrics: Metrics
    ks_distance: float
    ks_similarity: float
    split_half_ceiling_proxy: float
    correlation_attainment: float
    mae_ci: tuple[float, float]
    off1_ci: tuple[float, float]
    spearman_ci: tuple[float, float]
    mae_pass: bool
    off1_pass: bool
    sp_pass: bool
    reg_pass: bool
    overall_pass: bool
    spearman_drop: float
    max_dataset_mae_regression: float


def _metrics(scores: np.ndarray, labels: np.ndarray) -> Metrics:
    y = labels.astype(float)
    rounded = np.clip(np.rint(scores), 1, 5).astype(int)
    yi = y.astype(int)
    corr = spearmanr(scores, y).correlation
    corr = float(corr) if corr is not None and not np.isnan(corr) else 0.0
    return Metrics(
        n=int(y.size),
        mae=float(np.abs(scores - y).mean()),
        spearman=corr,
        exact=float((rounded == yi).mean()),
        off1=float((np.abs(rounded - yi) <= 1).mean()),
    )


def _bootstrap_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = labels.size
    if n == 0:
        return {}
    stats = {"mae": [], "off1": [], "exact": [], "spearman": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = _metrics(scores[idx], labels[idx])
        stats["mae"].append(m.mae)
        stats["off1"].append(m.off1)
        stats["exact"].append(m.exact)
        stats["spearman"].append(m.spearman)
    out: dict[str, tuple[float, float]] = {}
    for k, arr in stats.items():
        a = np.array(arr, dtype=float)
        out[k] = (float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975)))
    return out


def _discrete_ks_distance(scores: np.ndarray, labels: np.ndarray) -> float:
    """KS distance over discrete 1..5 stars using rounded predictions."""
    pred = np.clip(np.rint(scores), 1, 5).astype(int)
    true = np.clip(np.rint(labels), 1, 5).astype(int)
    n_pred = max(1, pred.size)
    n_true = max(1, true.size)
    pred_hist = np.array([(pred == k).sum() for k in range(1, 6)], dtype=float) / n_pred
    true_hist = np.array([(true == k).sum() for k in range(1, 6)], dtype=float) / n_true
    pred_cdf = np.cumsum(pred_hist)
    true_cdf = np.cumsum(true_hist)
    return float(np.max(np.abs(pred_cdf - true_cdf)))


def _split_half_ceiling_proxy(labels: np.ndarray, *, n_splits: int, seed: int) -> float:
    """
    Proxy ceiling from split-half agreement of label distributions.

    Note: this is a pragmatic proxy for single-label datasets, not a full
    human split-half ceiling from repeated ratings per item.
    """
    if labels.size < 20 or n_splits < 1:
        return 0.0
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    idx = np.arange(labels.size, dtype=int)
    for _ in range(n_splits):
        perm = rng.permutation(idx)
        half = perm.size // 2
        if half < 10:
            continue
        a = np.clip(np.rint(labels[perm[:half]]), 1, 5).astype(int)
        b = np.clip(np.rint(labels[perm[half : half * 2]]), 1, 5).astype(int)
        a_hist = np.array([(a == k).sum() for k in range(1, 6)], dtype=float)
        b_hist = np.array([(b == k).sum() for k in range(1, 6)], dtype=float)
        corr = spearmanr(a_hist, b_hist).correlation
        if corr is None or np.isnan(corr):
            continue
        vals.append(float(corr))
    if not vals:
        return 0.0
    return float(np.median(np.array(vals, dtype=float)))


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


def _sample_rows(rows: list[tuple[str, int, int | None]], limit: int, seed: int) -> DatasetSlice:
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    if limit > 0:
        rows = rows[:limit]
    texts = [r[0] for r in rows]
    labels = np.array([r[1] for r in rows], dtype=float)
    if any(r[2] is not None for r in rows):
        ts = np.array([int(r[2]) if r[2] is not None else -1 for r in rows], dtype=int)
    else:
        ts = None
    return DatasetSlice(name="", language=Language.EN, texts=texts, labels=labels, timestamps=ts)


def _latest_cached_file(pattern: str) -> Path | None:
    matches = [Path(p) for p in glob(pattern, recursive=True)]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _load_yelp(limit: int, seed: int) -> DatasetSlice:
    ds = None
    try:
        ds = load_dataset("Yelp/yelp_review_full", split="test")
    except Exception:
        yelp_arrow = _latest_cached_file(str(HF_DATASETS_CACHE_ROOT / "**" / "yelp_review_full-test.arrow"))
        if yelp_arrow is None:
            raise
        ds = Dataset.from_file(str(yelp_arrow))
    rows: list[tuple[str, int, int | None]] = []
    for x in ds:
        text = str(x["text"]).strip()
        label = int(x["label"]) + 1
        if text and 1 <= label <= 5:
            rows.append((text, label, None))
    out = _sample_rows(rows, limit, seed)
    out.name = "Yelp Review Full (EN)"
    out.language = Language.EN
    return out


def _load_app_reviews(limit: int, seed: int) -> DatasetSlice:
    ds = None
    try:
        ds = load_dataset("app_reviews", split="train")
    except Exception:
        app_arrow = _latest_cached_file(str(HF_DATASETS_CACHE_ROOT / "**" / "app_reviews-train.arrow"))
        if app_arrow is None:
            raise
        ds = Dataset.from_file(str(app_arrow))
    rows: list[tuple[str, int, int | None]] = []
    for x in ds:
        text = str(x["review"]).strip()
        star = int(x["star"])
        date_raw = str(x.get("date", "")).strip()
        ts = None
        if date_raw:
            try:
                ts = int(datetime.fromisoformat(date_raw).timestamp())
            except Exception:
                ts = None
        if text and 1 <= star <= 5:
            rows.append((text, star, ts))
    out = _sample_rows(rows, limit, seed)
    out.name = "App Reviews (EN)"
    out.language = Language.EN
    return out


def _load_allegro(limit: int, seed: int) -> DatasetSlice:
    ds = None
    try:
        ds = load_dataset("allegro/klej-allegro-reviews", split="train")
    except Exception:
        allegro_arrow = _latest_cached_file(str(HF_DATASETS_CACHE_ROOT / "**" / "klej-allegro-reviews-train.arrow"))
        if allegro_arrow is None:
            raise
        ds = Dataset.from_file(str(allegro_arrow))
    rows: list[tuple[str, int, int | None]] = []
    for x in ds:
        text = str(x["text"]).strip()
        rating = int(round(float(x["rating"])))
        if text and 1 <= rating <= 5:
            rows.append((text, rating, None))
    out = _sample_rows(rows, limit, seed)
    out.name = "KLEJ Allegro Reviews (PL)"
    out.language = Language.PL
    return out


def _load_amazon_2023(limit_per_category: int, max_scan: int, seed: int) -> DatasetSlice:
    fs = HfFileSystem()
    all_rows: list[tuple[str, int, int | None]] = []
    for i, cat in enumerate(CATEGORIES_20):
        rng = np.random.default_rng(seed + i)
        reservoir: list[tuple[str, int, int | None]] = []
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
                ts = None
                raw_ts = obj.get("timestamp")
                if raw_ts is not None:
                    try:
                        ts = int(raw_ts)
                    except Exception:
                        ts = None
                row = (text, y, ts)
                if len(reservoir) < limit_per_category:
                    reservoir.append(row)
                else:
                    j = rng.integers(0, seen)
                    if j < limit_per_category:
                        reservoir[j] = row
        all_rows.extend(reservoir)
    out = _sample_rows(all_rows, 0, seed)
    out.name = "Amazon Reviews 2023 - 20 categories (EN)"
    out.language = Language.EN
    if len(out.texts) == 0:
        raise RuntimeError("Amazon Reviews 2023 loader produced zero rows.")
    return out


def _temporal_tail_metrics(scores: np.ndarray, labels: np.ndarray, timestamps: np.ndarray | None, tail_ratio: float) -> Metrics | None:
    if timestamps is None:
        return None
    valid = timestamps >= 0
    if not np.any(valid):
        return None
    idx = np.where(valid)[0]
    idx = idx[np.argsort(timestamps[idx])]
    n_tail = int(round(idx.size * tail_ratio))
    if n_tail < 100:
        return None
    tail_idx = idx[-n_tail:]
    return _metrics(scores[tail_idx], labels[tail_idx])


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    p = argparse.ArgumentParser(description="External production-readiness validation.")
    p.add_argument("--model", default=settings.embedding_model)
    p.add_argument("--anchor-variant", default=DEFAULT_ANCHOR_VARIANT, choices=get_anchor_variants())
    p.add_argument("--temperature", type=float, default=settings.ssr_temperature)
    p.add_argument("--epsilon", type=float, default=settings.ssr_epsilon)
    p.add_argument("--calibrator-path", default=settings.ssr_calibration_artifact_path)
    p.add_argument("--policy-path", default=settings.ssr_calibration_policy_path)
    p.add_argument("--disable-calibration", action="store_true")
    p.add_argument("--limit-yelp", type=int, default=3000)
    p.add_argument("--limit-app", type=int, default=3000)
    p.add_argument("--limit-allegro", type=int, default=3000)
    p.add_argument("--amazon-per-category", type=int, default=200)
    p.add_argument("--amazon-max-scan", type=int, default=20000)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--split-half-splits", type=int, default=300)
    p.add_argument("--temporal-tail-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gate-mae-max", type=float, default=0.60)
    p.add_argument("--gate-off1-min", type=float, default=0.92)
    p.add_argument("--gate-spearman-drop-max", type=float, default=0.01)
    p.add_argument("--gate-max-dataset-mae-regression", type=float, default=0.10)
    p.add_argument("--report-out", default="")
    return p.parse_args()


def _evaluate_policy(
    *,
    name: str,
    labels: np.ndarray,
    raw_scores: np.ndarray,
    policy_scores: np.ndarray,
    per_dataset_rows: list[dict],
    args: argparse.Namespace,
) -> GateResult:
    m_raw = _metrics(raw_scores, labels)
    m_pol = _metrics(policy_scores, labels)
    ks_distance = _discrete_ks_distance(policy_scores, labels)
    ks_similarity = 1.0 - ks_distance
    ceiling_proxy = _split_half_ceiling_proxy(
        labels, n_splits=args.split_half_splits, seed=args.seed + 123
    )
    attainment = m_pol.spearman / max(ceiling_proxy, 1e-6)
    ci_pol = _bootstrap_ci(policy_scores, labels, n_boot=args.bootstrap, seed=args.seed + 99)
    mae_pass = ci_pol["mae"][1] <= args.gate_mae_max
    off1_pass = ci_pol["off1"][0] >= args.gate_off1_min
    spearman_drop = m_raw.spearman - m_pol.spearman
    sp_pass = spearman_drop <= args.gate_spearman_drop_max
    max_reg = max(r["mae_regression_for_policy"][name] for r in per_dataset_rows)
    reg_pass = max_reg <= args.gate_max_dataset_mae_regression
    return GateResult(
        name=name,
        metrics=m_pol,
        ks_distance=ks_distance,
        ks_similarity=ks_similarity,
        split_half_ceiling_proxy=ceiling_proxy,
        correlation_attainment=attainment,
        mae_ci=ci_pol["mae"],
        off1_ci=ci_pol["off1"],
        spearman_ci=ci_pol["spearman"],
        mae_pass=mae_pass,
        off1_pass=off1_pass,
        sp_pass=sp_pass,
        reg_pass=reg_pass,
        overall_pass=mae_pass and off1_pass and sp_pass and reg_pass,
        spearman_drop=spearman_drop,
        max_dataset_mae_regression=max_reg,
    )


def main() -> None:
    args = parse_args()
    print(f"Model={args.model} T={args.temperature} eps={args.epsilon}")
    print(f"Anchor variant={args.anchor_variant}")
    client = LocalEmbeddingClient(model_name=args.model)
    calibrator: IsotonicCalibrator | None = None
    domain_policy: DomainCalibrationPolicy | None = None
    if not args.disable_calibration and args.calibrator_path:
        cpath = Path(args.calibrator_path).expanduser().resolve()
        if cpath.exists():
            calibrator = IsotonicCalibrator.load_json(cpath)
            print(f"Loaded calibrator: {cpath}")
        else:
            print(f"Calibrator path not found: {cpath}")
    if args.policy_path:
        ppath = Path(args.policy_path).expanduser().resolve()
        if ppath.exists():
            try:
                domain_policy = DomainCalibrationPolicy.load_json(ppath)
                print(f"Loaded domain policy: {ppath}")
            except Exception as exc:
                print(f"Failed to load domain policy: {exc}")

    datasets: list[DatasetSlice] = []
    data_warnings: list[str] = []
    loaders = [
        ("amazon_2023", lambda: _load_amazon_2023(args.amazon_per_category, args.amazon_max_scan, args.seed)),
        ("yelp", lambda: _load_yelp(args.limit_yelp, args.seed + 1)),
        ("app_reviews", lambda: _load_app_reviews(args.limit_app, args.seed + 2)),
        ("allegro", lambda: _load_allegro(args.limit_allegro, args.seed + 3)),
    ]
    for key, loader in loaders:
        try:
            datasets.append(loader())
        except Exception as exc:
            msg = f"Skipped dataset `{key}`: {type(exc).__name__}: {exc}"
            print(msg)
            data_warnings.append(msg)
    if not datasets:
        raise RuntimeError("No datasets available for validation (all loaders failed).")

    pooled_labels: list[np.ndarray] = []
    pooled_raw: list[np.ndarray] = []
    pooled_cal: list[np.ndarray] = []
    pooled_policy_scores: dict[str, list[np.ndarray]] = {
        "raw_only": [],
        "global_calibrated": [],
        "pl_only_calibrated": [],
        "purchase_intent_only_calibrated": [],
    }
    has_purchase_domain_calibrator = False
    purchase_hybrid_alphas = [i / 10.0 for i in range(1, 10)]
    if domain_policy is not None:
        pooled_policy_scores["domain_policy_artifact"] = []
        if (
            domain_policy.select(domain_hint="purchase_intent_short_en") is not None
            or domain_policy.select(domain_hint="purchase_intent_short_pl") is not None
            or domain_policy.select(domain_hint="purchase_intent") is not None
        ):
            has_purchase_domain_calibrator = True
            pooled_policy_scores["purchase_domain_only_calibrated"] = []
            for alpha in purchase_hybrid_alphas:
                pooled_policy_scores[f"purchase_hybrid_{alpha:.1f}"] = []
    for alpha_i in range(1, 10):
        pooled_policy_scores[f"blend_{alpha_i / 10:.1f}"] = []
    per_rows: list[dict] = []
    temporal_rows: list[dict] = []

    for ds in datasets:
        print(f"Scoring: {ds.name} | n={len(ds.labels)}")
        raw_scores = _score_texts(
            client,
            ds.texts,
            language=ds.language,
            anchor_variant=args.anchor_variant,
            temperature=args.temperature,
            epsilon=args.epsilon,
        )
        if calibrator is None:
            cal_scores = raw_scores
        else:
            cal_scores = calibrator.transform(raw_scores)
        blend_scores: dict[str, np.ndarray] = {}
        for alpha_i in range(1, 10):
            alpha = alpha_i / 10.0
            blend_scores[f"blend_{alpha:.1f}"] = np.clip(raw_scores + alpha * (cal_scores - raw_scores), 1.0, 5.0)
        is_pl = ds.language == Language.PL
        is_purchase_intent = "Amazon" in ds.name or "Allegro" in ds.name or "App Reviews" in ds.name
        if "Allegro" in ds.name:
            domain_hint = "purchase_intent_short_pl"
            purchase_hint = "purchase_intent_short_pl"
        elif is_purchase_intent:
            domain_hint = "purchase_intent_short_en"
            purchase_hint = "purchase_intent_short_en"
        else:
            domain_hint = "review_long_en"
            purchase_hint = "purchase_intent_short_en"
        pl_only = cal_scores if is_pl else raw_scores
        purchase_intent_only = cal_scores if is_purchase_intent else raw_scores
        domain_policy_scores = raw_scores
        purchase_domain_only = raw_scores
        purchase_hybrid_scores: dict[str, np.ndarray] = {}
        if domain_policy is not None:
            chosen = domain_policy.select(domain_hint=domain_hint)
            if chosen is not None:
                domain_policy_scores = chosen.transform(raw_scores)
            purchase_cal = domain_policy.select(domain_hint=purchase_hint)
            if has_purchase_domain_calibrator and purchase_cal is not None:
                if is_purchase_intent:
                    purchase_domain_only = purchase_cal.transform(raw_scores)
                    for alpha in purchase_hybrid_alphas:
                        purchase_hybrid_scores[f"purchase_hybrid_{alpha:.1f}"] = np.clip(
                            cal_scores + alpha * (purchase_domain_only - cal_scores), 1.0, 5.0
                        )
                else:
                    purchase_domain_only = raw_scores
                    for alpha in purchase_hybrid_alphas:
                        purchase_hybrid_scores[f"purchase_hybrid_{alpha:.1f}"] = raw_scores

        m_raw = _metrics(raw_scores, ds.labels)
        m_cal = _metrics(cal_scores, ds.labels)
        mae_reg_by_policy: dict[str, float] = {
            "raw_only": 0.0,
            "global_calibrated": (m_cal.mae - m_raw.mae) / m_raw.mae if m_raw.mae > 0 else 0.0,
            "pl_only_calibrated": (_metrics(pl_only, ds.labels).mae - m_raw.mae) / m_raw.mae if m_raw.mae > 0 else 0.0,
            "purchase_intent_only_calibrated": (_metrics(purchase_intent_only, ds.labels).mae - m_raw.mae) / m_raw.mae if m_raw.mae > 0 else 0.0,
        }
        if domain_policy is not None:
            mae_reg_by_policy["domain_policy_artifact"] = (
                (_metrics(domain_policy_scores, ds.labels).mae - m_raw.mae) / m_raw.mae if m_raw.mae > 0 else 0.0
            )
        if has_purchase_domain_calibrator:
            mae_reg_by_policy["purchase_domain_only_calibrated"] = (
                (_metrics(purchase_domain_only, ds.labels).mae - m_raw.mae) / m_raw.mae if m_raw.mae > 0 else 0.0
            )
            for k, v in purchase_hybrid_scores.items():
                mae_reg_by_policy[k] = (_metrics(v, ds.labels).mae - m_raw.mae) / m_raw.mae if m_raw.mae > 0 else 0.0
        for k, v in blend_scores.items():
            mae_reg_by_policy[k] = (_metrics(v, ds.labels).mae - m_raw.mae) / m_raw.mae if m_raw.mae > 0 else 0.0
        per_rows.append(
            {
                "name": ds.name,
                "n": m_raw.n,
                "raw": m_raw,
                "cal": m_cal,
                "mae_regression": mae_reg_by_policy["global_calibrated"],
                "mae_regression_for_policy": mae_reg_by_policy,
            }
        )
        t_raw = _temporal_tail_metrics(raw_scores, ds.labels, ds.timestamps, args.temporal_tail_ratio)
        t_cal = _temporal_tail_metrics(cal_scores, ds.labels, ds.timestamps, args.temporal_tail_ratio)
        if t_raw is not None and t_cal is not None:
            temporal_rows.append({"name": ds.name, "raw": t_raw, "cal": t_cal})

        pooled_labels.append(ds.labels)
        pooled_raw.append(raw_scores)
        pooled_cal.append(cal_scores)
        pooled_policy_scores["raw_only"].append(raw_scores)
        pooled_policy_scores["global_calibrated"].append(cal_scores)
        pooled_policy_scores["pl_only_calibrated"].append(pl_only)
        pooled_policy_scores["purchase_intent_only_calibrated"].append(purchase_intent_only)
        if domain_policy is not None:
            pooled_policy_scores["domain_policy_artifact"].append(domain_policy_scores)
        if has_purchase_domain_calibrator:
            pooled_policy_scores["purchase_domain_only_calibrated"].append(purchase_domain_only)
            for k, v in purchase_hybrid_scores.items():
                pooled_policy_scores[k].append(v)
        for k, v in blend_scores.items():
            pooled_policy_scores[k].append(v)

    y = np.concatenate(pooled_labels)
    raw_all = np.concatenate(pooled_raw)
    cal_all = np.concatenate(pooled_cal)
    m_raw_all = _metrics(raw_all, y)
    m_cal_all = _metrics(cal_all, y)

    ci_cal = _bootstrap_ci(cal_all, y, n_boot=args.bootstrap, seed=args.seed + 99)

    policy_results: list[GateResult] = []
    for policy_name, parts in pooled_policy_scores.items():
        policy_scores = np.concatenate(parts)
        policy_results.append(
            _evaluate_policy(
                name=policy_name,
                labels=y,
                raw_scores=raw_all,
                policy_scores=policy_scores,
                per_dataset_rows=per_rows,
                args=args,
            )
        )
    policy_results = sorted(
        policy_results,
        key=lambda r: (
            r.overall_pass,
            int(r.mae_pass) + int(r.off1_pass) + int(r.sp_pass) + int(r.reg_pass),
            -max(0.0, r.mae_ci[1] - args.gate_mae_max),
            -max(0.0, args.gate_off1_min - r.off1_ci[0]),
            r.metrics.off1,
            -r.metrics.mae,
        ),
        reverse=True,
    )
    best_policy = policy_results[0]

    now = datetime.now()
    if args.report_out:
        out_path = Path(args.report_out).expanduser().resolve()
    else:
        out_path = (
            Path(__file__).resolve().parents[2]
            / "reports"
            / f"production_readiness_validation_{now.strftime('%Y-%m-%d')}.md"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# Production Readiness Validation ({now.strftime('%Y-%m-%d')})")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- Model: `{args.model}`")
    lines.append(f"- Anchor variant: `{args.anchor_variant}`")
    lines.append(f"- Temperature: `{args.temperature}`")
    lines.append(f"- Epsilon: `{args.epsilon}`")
    lines.append(f"- Calibration enabled: `{calibrator is not None}`")
    lines.append(f"- Calibrator path: `{args.calibrator_path}`")
    lines.append(f"- Total evaluated rows: `{len(y)}`")
    lines.append(f"- Split-half ceiling proxy splits: `{args.split_half_splits}`")
    lines.append(f"- Offline mode: `{bool(os.environ.get('HF_HUB_OFFLINE'))}`")
    if data_warnings:
        lines.append("- Dataset warnings:")
        for warning in data_warnings:
            lines.append(f"  - {warning}")
    lines.append("")

    lines.append("## Pooled Metrics")
    lines.append(
        f"- Raw: mae={m_raw_all.mae:.4f}, spearman={m_raw_all.spearman:.4f}, "
        f"off1={m_raw_all.off1:.4f}, exact={m_raw_all.exact:.4f}"
    )
    lines.append(
        f"- Calibrated: mae={m_cal_all.mae:.4f}, spearman={m_cal_all.spearman:.4f}, "
        f"off1={m_cal_all.off1:.4f}, exact={m_cal_all.exact:.4f}"
    )
    lines.append(
        f"- Calibrated 95% CI: mae=[{ci_cal['mae'][0]:.4f}, {ci_cal['mae'][1]:.4f}], "
        f"off1=[{ci_cal['off1'][0]:.4f}, {ci_cal['off1'][1]:.4f}], "
        f"spearman=[{ci_cal['spearman'][0]:.4f}, {ci_cal['spearman'][1]:.4f}]"
    )
    lines.append("")

    lines.append("## Dataset Breakdown")
    lines.append("| Dataset | n | Raw MAE | Cal MAE | Raw Spearman | Cal Spearman | Raw Off1 | Cal Off1 | MAE Regression |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in per_rows:
        raw = r["raw"]
        cal = r["cal"]
        lines.append(
            f"| {r['name']} | {raw.n} | {raw.mae:.4f} | {cal.mae:.4f} | {raw.spearman:.4f} | "
            f"{cal.spearman:.4f} | {raw.off1:.4f} | {cal.off1:.4f} | {r['mae_regression']:+.4f} |"
        )
    lines.append("")

    if temporal_rows:
        lines.append("## Temporal Tail Check")
        lines.append(
            f"- Tail ratio: `{args.temporal_tail_ratio}` (latest portion by available timestamp/date)"
        )
        lines.append("| Dataset | Raw MAE | Cal MAE | Raw Spearman | Cal Spearman | Raw Off1 | Cal Off1 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for t in temporal_rows:
            raw = t["raw"]
            cal = t["cal"]
            lines.append(
                f"| {t['name']} | {raw.mae:.4f} | {cal.mae:.4f} | {raw.spearman:.4f} | "
                f"{cal.spearman:.4f} | {raw.off1:.4f} | {cal.off1:.4f} |"
            )
        lines.append("")

    lines.append("## Policy Search")
    lines.append(
        "| Policy | MAE | Off1 | Exact | Spearman | KS sim | Ceiling proxy | Corr attainment | MAE 95% CI | Off1 95% CI | Spearman drop | Max dataset MAE regression | Pass |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|")
    for pr in policy_results:
        lines.append(
            f"| {pr.name} | {pr.metrics.mae:.4f} | {pr.metrics.off1:.4f} | {pr.metrics.exact:.4f} | {pr.metrics.spearman:.4f} | "
            f"{pr.ks_similarity:.4f} | {pr.split_half_ceiling_proxy:.4f} | {pr.correlation_attainment:.4f} | "
            f"[{pr.mae_ci[0]:.4f}, {pr.mae_ci[1]:.4f}] | [{pr.off1_ci[0]:.4f}, {pr.off1_ci[1]:.4f}] | "
            f"{pr.spearman_drop:+.4f} | {pr.max_dataset_mae_regression:+.4f} | {'PASS' if pr.overall_pass else 'FAIL'} |"
        )
    lines.append("")

    lines.append("## Paper-native Diagnostics (Proxy)")
    lines.append(
        "- `KS sim` is `1 - KS distance` between rounded prediction-star CDF and true-star CDF (higher is better)."
    )
    lines.append(
        "- `Ceiling proxy` uses split-half Spearman agreement of label distributions over many random half-splits."
    )
    lines.append(
        "- `Corr attainment` is `policy Spearman / ceiling proxy` and should be treated as a proxy on single-label datasets."
    )
    lines.append(f"- Best policy KS sim: `{best_policy.ks_similarity:.4f}`")
    lines.append(f"- Best policy ceiling proxy: `{best_policy.split_half_ceiling_proxy:.4f}`")
    lines.append(f"- Best policy corr attainment: `{best_policy.correlation_attainment:.4f}`")
    lines.append("")

    lines.append("## Production Gates (Best Policy)")
    lines.append(f"- Selected policy: `{best_policy.name}`")
    lines.append(f"- Gate MAE <= {args.gate_mae_max:.2f}: `{best_policy.mae_pass}`")
    lines.append(f"- Gate Off-by-one >= {args.gate_off1_min:.2f}: `{best_policy.off1_pass}`")
    lines.append(
        f"- Gate Spearman drop <= {args.gate_spearman_drop_max:.3f}: `{best_policy.sp_pass}` "
        f"(drop={best_policy.spearman_drop:.4f})"
    )
    lines.append(
        f"- Gate max dataset MAE regression <= {args.gate_max_dataset_mae_regression:.2f}: "
        f"`{best_policy.reg_pass}` (max={best_policy.max_dataset_mae_regression:.4f})"
    )
    lines.append(f"- Final decision: `{'PASS' if best_policy.overall_pass else 'FAIL'}`")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report saved: {out_path}")
    print(f"Best policy: {best_policy.name}")
    print(f"Decision: {'PASS' if best_policy.overall_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
