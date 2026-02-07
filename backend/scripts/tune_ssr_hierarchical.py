#!/usr/bin/env python3
"""Hierarchical SSR tuning on large real datasets.

What it does:
1) Runs grid-search over SSR mapping parameters (temperature, epsilon, anchors).
2) Evaluates each config on many industry groups (default: 20 Amazon categories).
3) Finds a global best config and per-group best configs.
4) Recommends when per-group tuning is worth enabling.
5) Writes a Markdown report to /reports.

Optional:
- Load additional grouped CSV data (e.g., per-country, including PL datasets)
  via --extra-csv with columns for group/text/label.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from huggingface_hub import HfFileSystem
from scipy.stats import spearmanr

from app.i18n import Language, get_anchor_sets
from app.services.embedding_client import LocalEmbeddingClient
from app.services.score_calibration import IsotonicCalibrator, fit_isotonic_calibrator


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


@dataclass(frozen=True)
class Config:
    """Single SSR mapping configuration."""

    temperature: float
    epsilon: float
    anchor_mode: str  # "avg6" or "single:<idx>"

    @property
    def key(self) -> str:
        return f"T={self.temperature:g}|eps={self.epsilon:g}|anchor={self.anchor_mode}"


@dataclass
class Metrics:
    """Evaluation metrics for one (group, config)."""

    n: int
    mae: float
    spearman: float
    exact_acc: float
    off_by_one_acc: float
    mean_pred: float
    mean_true: float


@dataclass
class GroupData:
    """Cached inputs and embeddings for one group."""

    name: str
    texts: list[str]
    labels: list[int]
    response_embeddings: np.ndarray


def _parse_float_list(raw: str) -> list[float]:
    vals: list[float] = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    if not vals:
        raise ValueError("Expected at least one numeric value.")
    return vals


def _clip_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _embed_texts(
    client: LocalEmbeddingClient,
    texts: list[str],
    *,
    batch_size: int,
    max_text_chars: int,
) -> np.ndarray:
    clipped = [_clip_text(t, max_text_chars) for t in texts]
    # Use direct SentenceTransformer encode with controlled batch size to avoid OOM.
    if hasattr(client, "model"):
        emb = client.model.encode(
            clipped,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        return np.array(emb)
    return client.embed(clipped)


def _anchor_indices(mode: str, n_sets: int) -> list[int]:
    if mode == "avg6":
        return list(range(n_sets))
    if mode.startswith("single:"):
        idx = int(mode.split(":", 1)[1])
        if idx < 0 or idx >= n_sets:
            raise ValueError(f"anchor index out of range: {idx}")
        return [idx]
    raise ValueError(f"Unknown anchor mode: {mode}")


def _response_embeddings_to_pmf(
    response_embeddings: np.ndarray,
    likert_embeddings: np.ndarray,  # shape (d, 5)
    epsilon: float,
) -> np.ndarray:
    """Map response embeddings to PMFs for one anchor set."""
    m_left = response_embeddings
    m_right = likert_embeddings
    if m_left.shape[0] == 0:
        return np.empty((0, 5))

    m_right = m_right / np.linalg.norm(m_right, axis=0)[None, :]
    m_left = m_left / np.linalg.norm(m_left, axis=1)[:, None]

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
    """Apply SSR temperature scaling row-wise."""
    if temperature == 1.0:
        return pmf
    if temperature == 0.0:
        out = np.zeros_like(pmf)
        idx = np.argmax(pmf, axis=1)
        out[np.arange(len(idx)), idx] = 1.0
        return out
    scaled = pmf ** (1.0 / temperature)
    denom = scaled.sum(axis=1)[:, None]
    return scaled / denom


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


def _build_configs(
    temperatures: list[float],
    epsilons: list[float],
    include_single_anchors: bool,
    n_anchor_sets: int,
) -> list[Config]:
    modes = ["avg6"]
    if include_single_anchors:
        modes.extend([f"single:{i}" for i in range(n_anchor_sets)])
    configs = [
        Config(temperature=t, epsilon=e, anchor_mode=m)
        for t in temperatures
        for e in epsilons
        for m in modes
    ]
    return configs


def _weighted_summary(metrics: Iterable[Metrics]) -> dict[str, float]:
    rows = list(metrics)
    w = np.array([r.n for r in rows], dtype=float)
    w = w / w.sum()
    return {
        "mae": float(np.dot(w, np.array([r.mae for r in rows]))),
        "spearman": float(np.dot(w, np.array([r.spearman for r in rows]))),
        "exact": float(np.dot(w, np.array([r.exact_acc for r in rows]))),
        "off1": float(np.dot(w, np.array([r.off_by_one_acc for r in rows]))),
    }


def _global_rank_key(summary: dict[str, float]) -> tuple[float, float, float]:
    """Higher is better."""
    return (summary["spearman"], summary["off1"], -summary["mae"])


def _stratified_holdout_indices(
    labels: np.ndarray,
    *,
    holdout_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels_i = labels.astype(int)
    train_parts: list[np.ndarray] = []
    holdout_parts: list[np.ndarray] = []
    all_idx = np.arange(labels.shape[0], dtype=int)
    for cls in np.unique(labels_i):
        idx = all_idx[labels_i == cls]
        if idx.size == 0:
            continue
        idx = idx.copy()
        rng.shuffle(idx)
        n_hold = int(round(idx.size * holdout_ratio))
        if holdout_ratio > 0 and n_hold == 0 and idx.size >= 2:
            n_hold = 1
        if n_hold >= idx.size and idx.size >= 2:
            n_hold = idx.size - 1
        holdout_parts.append(idx[:n_hold])
        train_parts.append(idx[n_hold:])
    train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=int)
    holdout_idx = np.concatenate(holdout_parts) if holdout_parts else np.array([], dtype=int)
    return train_idx, holdout_idx


def _stratified_kfold_indices(
    labels: np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    labels_i = labels.astype(int)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    all_idx = np.arange(labels.shape[0], dtype=int)
    for cls in np.unique(labels_i):
        idx = all_idx[labels_i == cls].copy()
        rng.shuffle(idx)
        for i, j in enumerate(idx):
            folds[i % n_splits].append(int(j))
    return [np.array(sorted(f), dtype=int) for f in folds]


def _evaluate_calibration_oof(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    folds: int,
    min_samples: int,
    seed: int,
) -> tuple[Metrics, Metrics]:
    if scores.size < min_samples or np.unique(scores).size < 2 or folds < 2:
        m = _metrics(scores, labels.astype(int).tolist())
        return m, m
    folds_idx = _stratified_kfold_indices(labels, n_splits=folds, seed=seed)
    raw_pred = scores.copy()
    cal_pred = scores.copy()
    for i in range(folds):
        val_idx = folds_idx[i]
        if val_idx.size == 0:
            continue
        train_parts = [folds_idx[j] for j in range(folds) if j != i and folds_idx[j].size > 0]
        if not train_parts:
            continue
        train_idx = np.concatenate(train_parts)
        if train_idx.size < min_samples or np.unique(scores[train_idx]).size < 2:
            continue
        cal = fit_isotonic_calibrator(scores[train_idx], labels[train_idx])
        cal_pred[val_idx] = cal.transform(scores[val_idx])
    raw_m = _metrics(raw_pred, labels.astype(int).tolist())
    cal_m = _metrics(cal_pred, labels.astype(int).tolist())
    return raw_m, cal_m


def _load_industry_group_rows(
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


def _load_grouped_csv(
    path: Path,
    *,
    group_col: str,
    text_col: str,
    label_col: str,
    min_group_n: int,
    sample_per_group: int,
    seed: int,
) -> dict[str, tuple[list[str], list[int]]]:
    groups: dict[str, list[tuple[str, int]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            group = str(row.get(group_col, "")).strip()
            text = str(row.get(text_col, "")).strip()
            label_raw = str(row.get(label_col, "")).strip()
            if not group or not text or not label_raw:
                continue
            try:
                y = int(round(float(label_raw)))
            except Exception:
                continue
            if y < 1 or y > 5:
                continue
            groups.setdefault(group, []).append((text, y))

    rng = np.random.default_rng(seed)
    out: dict[str, tuple[list[str], list[int]]] = {}
    for group, rows in groups.items():
        if len(rows) < min_group_n:
            continue
        rng.shuffle(rows)
        if sample_per_group > 0:
            rows = rows[:sample_per_group]
        texts = [r[0] for r in rows]
        labels = [r[1] for r in rows]
        out[group] = (texts, labels)
    return out


def _score_group(
    group: GroupData,
    cfg: Config,
    anchor_mats: list[np.ndarray],  # each shape (d, 5)
) -> Metrics:
    idxs = _anchor_indices(cfg.anchor_mode, len(anchor_mats))
    pmf_sets = [
        _response_embeddings_to_pmf(group.response_embeddings, anchor_mats[i], epsilon=cfg.epsilon)
        for i in idxs
    ]
    pmf_avg = np.mean(np.stack(pmf_sets, axis=0), axis=0)
    pmf_avg = pmf_avg / pmf_avg.sum(axis=1)[:, None]
    pmf_avg = _apply_temperature(pmf_avg, temperature=cfg.temperature)
    scores = pmf_avg.dot(np.arange(1, 6, dtype=float))
    return _metrics(scores, group.labels)


def _score_group_with_scores(
    group: GroupData,
    cfg: Config,
    anchor_mats: list[np.ndarray],  # each shape (d, 5)
) -> np.ndarray:
    idxs = _anchor_indices(cfg.anchor_mode, len(anchor_mats))
    pmf_sets = [
        _response_embeddings_to_pmf(group.response_embeddings, anchor_mats[i], epsilon=cfg.epsilon)
        for i in idxs
    ]
    pmf_avg = np.mean(np.stack(pmf_sets, axis=0), axis=0)
    pmf_avg = pmf_avg / pmf_avg.sum(axis=1)[:, None]
    pmf_avg = _apply_temperature(pmf_avg, temperature=cfg.temperature)
    return pmf_avg.dot(np.arange(1, 6, dtype=float))


def _infer_group_anchor_language(group_name: str, default_language: Language) -> Language:
    """Infer anchor language for a given group.

    Current rule:
    - Groups named `extra::PL_*` use Polish anchors.
    - All others use default language.
    """
    name = group_name.strip().lower()
    if name.startswith("extra::pl_"):
        return Language.PL
    return default_language


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hierarchical SSR tuning on large grouped datasets.")
    p.add_argument("--model", default="BAAI/bge-m3", help="Local embedding model.")
    p.add_argument("--language", choices=["pl", "en"], default="en", help="Anchor language.")
    p.add_argument(
        "--anchor-language-mode",
        choices=["fixed", "auto"],
        default="fixed",
        help="Anchor language strategy: fixed=single language, auto=PL anchors for extra::PL_* groups.",
    )
    p.add_argument(
        "--temperatures",
        default="0.7,0.85,1.0,1.15,1.3",
        help="Comma-separated SSR temperatures.",
    )
    p.add_argument(
        "--epsilons",
        default="0,1e-4,5e-4,1e-3,5e-3",
        help="Comma-separated epsilon values.",
    )
    p.add_argument(
        "--include-single-anchors",
        action="store_true",
        help="Also test single-anchor-set configs (in addition to avg6).",
    )
    p.add_argument(
        "--sample-per-industry",
        type=int,
        default=500,
        help="Sample size per industry (20 categories).",
    )
    p.add_argument(
        "--max-scan-per-industry",
        type=int,
        default=30000,
        help="Max streamed rows scanned per industry category.",
    )
    p.add_argument(
        "--skip-industries",
        action="store_true",
        help="Skip Amazon 20-industry loading (useful for offline/local-only runs).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--embed-batch-size",
        type=int,
        default=8,
        help="Embedding batch size for local model (reduce if OOM).",
    )
    p.add_argument(
        "--max-text-chars",
        type=int,
        default=1200,
        help="Max chars per text before embedding (0 = no clipping).",
    )
    p.add_argument(
        "--extra-csv",
        default="",
        help="Optional grouped CSV path (e.g., country datasets incl. PL).",
    )
    p.add_argument("--extra-group-col", default="group")
    p.add_argument("--extra-text-col", default="text")
    p.add_argument("--extra-label-col", default="label")
    p.add_argument(
        "--sample-per-extra-group",
        type=int,
        default=1200,
        help="Sample size per extra CSV group (0 = all).",
    )
    p.add_argument(
        "--min-extra-group-n",
        type=int,
        default=300,
        help="Minimum rows required for an extra CSV group.",
    )
    p.add_argument(
        "--group-specific-min-n",
        type=int,
        default=500,
        help="Minimum group size to recommend group-specific config.",
    )
    p.add_argument(
        "--group-specific-min-uplift",
        type=float,
        default=0.02,
        help="Minimum Spearman uplift vs global baseline for group-specific recommendation.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top configs to print in report.",
    )
    p.add_argument(
        "--global-calibration",
        choices=["none", "isotonic"],
        default="none",
        help="Optional global post-SSR calibration over pooled predictions.",
    )
    p.add_argument(
        "--calibration-min-samples",
        type=int,
        default=1000,
        help="Minimum pooled sample size required to fit global calibrator.",
    )
    p.add_argument(
        "--calibration-cv-folds",
        type=int,
        default=5,
        help="Number of folds for out-of-fold calibration validation.",
    )
    p.add_argument(
        "--calibration-holdout-ratio",
        type=float,
        default=0.2,
        help="Holdout share for final calibration validation (0 disables holdout).",
    )
    p.add_argument(
        "--calibration-artifact-out",
        default="",
        help="Optional path to save fitted global calibration artifact (.json).",
    )
    p.add_argument(
        "--report-out",
        default="",
        help="Optional explicit report path (.md). Defaults to reports/ssr_hierarchical_tuning_YYYY-MM-DD.md",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    temperatures = _parse_float_list(args.temperatures)
    epsilons = _parse_float_list(args.epsilons)
    language = Language.PL if args.language == "pl" else Language.EN

    anchor_sets = get_anchor_sets(language)
    configs = _build_configs(
        temperatures=temperatures,
        epsilons=epsilons,
        include_single_anchors=args.include_single_anchors,
        n_anchor_sets=len(anchor_sets),
    )

    print(f"Embedding model: {args.model}")
    print(f"Anchor language: {language.value}")
    print(f"Anchor language mode: {args.anchor_language_mode}")
    print(f"Configs to evaluate: {len(configs)}")
    if args.skip_industries:
        print("Industry sample: skipped")
    else:
        print(
            f"Industry sample: {args.sample_per_industry} x {len(CATEGORIES_20)} "
            f"(max scan/category={args.max_scan_per_industry})"
        )

    client = LocalEmbeddingClient(model_name=args.model)

    anchor_mats_by_language: dict[Language, list[np.ndarray]] = {}
    languages_to_prepare: list[Language] = [language]
    if args.anchor_language_mode == "auto":
        languages_to_prepare = sorted(set([language, Language.PL]), key=lambda x: x.value)

    for lang in languages_to_prepare:
        mats: list[np.ndarray] = []
        for anchor_set in get_anchor_sets(lang):
            anchor_texts = [anchor_set[i] for i in range(1, 6)]
            emb = _embed_texts(
                client,
                anchor_texts,
                batch_size=args.embed_batch_size,
                max_text_chars=args.max_text_chars,
            )
            mats.append(emb.T)
        anchor_mats_by_language[lang] = mats

    groups: list[GroupData] = []
    if not args.skip_industries:
        for i, cat in enumerate(CATEGORIES_20):
            texts, labels = _load_industry_group_rows(
                cat,
                sample_size=args.sample_per_industry,
                max_scan=args.max_scan_per_industry,
                seed=args.seed + i,
            )
            if len(labels) == 0:
                continue
            embeddings = _embed_texts(
                client,
                texts,
                batch_size=args.embed_batch_size,
                max_text_chars=args.max_text_chars,
            )
            groups.append(
                GroupData(
                    name=f"industry::{cat}",
                    texts=texts,
                    labels=labels,
                    response_embeddings=embeddings,
                )
            )
            print(f"Loaded industry group {cat}: n={len(labels)}")

    if args.extra_csv:
        extra_path = Path(args.extra_csv).expanduser().resolve()
        extra = _load_grouped_csv(
            extra_path,
            group_col=args.extra_group_col,
            text_col=args.extra_text_col,
            label_col=args.extra_label_col,
            min_group_n=args.min_extra_group_n,
            sample_per_group=args.sample_per_extra_group,
            seed=args.seed,
        )
        for idx, (group_name, (texts, labels)) in enumerate(sorted(extra.items())):
            if len(labels) == 0:
                continue
            embeddings = _embed_texts(
                client,
                texts,
                batch_size=args.embed_batch_size,
                max_text_chars=args.max_text_chars,
            )
            groups.append(
                GroupData(
                    name=f"extra::{group_name}",
                    texts=texts,
                    labels=labels,
                    response_embeddings=embeddings,
                )
            )
            print(f"Loaded extra group {group_name}: n={len(labels)}")

    if not groups:
        raise RuntimeError("No groups loaded. Nothing to evaluate.")

    group_names = [g.name for g in groups]
    by_cfg_group_raw: dict[str, dict[str, Metrics]] = {c.key: {} for c in configs}
    cfg_summaries_raw: dict[str, dict[str, float]] = {}

    for c_idx, cfg in enumerate(configs, start=1):
        print(f"Evaluating config {c_idx}/{len(configs)}: {cfg.key}")
        for g in groups:
            lang_for_group = (
                _infer_group_anchor_language(g.name, language)
                if args.anchor_language_mode == "auto"
                else language
            )
            mats = anchor_mats_by_language[lang_for_group]
            scores = _score_group_with_scores(g, cfg, mats)
            by_cfg_group_raw[cfg.key][g.name] = _metrics(scores, g.labels)
        cfg_summaries_raw[cfg.key] = _weighted_summary(by_cfg_group_raw[cfg.key].values())
    rank_summaries = cfg_summaries_raw
    by_cfg_group_ranked = by_cfg_group_raw

    ranked_cfgs = sorted(
        configs,
        key=lambda c: _global_rank_key(rank_summaries[c.key]),
        reverse=True,
    )
    global_best = ranked_cfgs[0]
    global_best_key = global_best.key

    group_best: dict[str, Config] = {}
    group_uplift: dict[str, float] = {}
    for g in group_names:
        best_cfg = max(
            configs,
            key=lambda c: (
                by_cfg_group_ranked[c.key][g].spearman,
                by_cfg_group_ranked[c.key][g].off_by_one_acc,
                -by_cfg_group_ranked[c.key][g].mae,
            ),
        )
        group_best[g] = best_cfg
        group_uplift[g] = (
            by_cfg_group_ranked[best_cfg.key][g].spearman
            - by_cfg_group_ranked[global_best_key][g].spearman
        )

    recommend_group_specific: list[str] = []
    for g in group_names:
        n = by_cfg_group_ranked[global_best_key][g].n
        uplift = group_uplift[g]
        if n >= args.group_specific_min_n and uplift >= args.group_specific_min_uplift:
            recommend_group_specific.append(g)

    # Calibration validation for the selected global-best config.
    pooled_raw_scores: list[np.ndarray] = []
    pooled_labels: list[np.ndarray] = []
    for g in groups:
        lang_for_group = (
            _infer_group_anchor_language(g.name, language)
            if args.anchor_language_mode == "auto"
            else language
        )
        mats = anchor_mats_by_language[lang_for_group]
        scores = _score_group_with_scores(g, global_best, mats)
        pooled_raw_scores.append(scores)
        pooled_labels.append(np.array(g.labels, dtype=float))
    all_scores = np.concatenate(pooled_raw_scores) if pooled_raw_scores else np.array([], dtype=float)
    all_labels = np.concatenate(pooled_labels) if pooled_labels else np.array([], dtype=float)

    apparent_raw = _metrics(all_scores, all_labels.astype(int).tolist())
    apparent_cal = apparent_raw
    oof_raw = apparent_raw
    oof_cal = apparent_raw
    holdout_raw = apparent_raw
    holdout_cal = apparent_raw
    final_calibrator: IsotonicCalibrator | None = None
    holdout_n = 0
    train_n = int(all_scores.size)

    if (
        args.global_calibration == "isotonic"
        and all_scores.size >= args.calibration_min_samples
        and np.unique(all_scores).size >= 2
    ):
        final_calibrator = fit_isotonic_calibrator(all_scores, all_labels)
        apparent_cal = _metrics(final_calibrator.transform(all_scores), all_labels.astype(int).tolist())
        oof_raw, oof_cal = _evaluate_calibration_oof(
            all_scores,
            all_labels,
            folds=args.calibration_cv_folds,
            min_samples=args.calibration_min_samples,
            seed=args.seed,
        )
        if args.calibration_holdout_ratio > 0:
            train_idx, holdout_idx = _stratified_holdout_indices(
                all_labels,
                holdout_ratio=args.calibration_holdout_ratio,
                seed=args.seed,
            )
            if (
                holdout_idx.size > 0
                and train_idx.size >= args.calibration_min_samples
                and np.unique(all_scores[train_idx]).size >= 2
            ):
                train_n = int(train_idx.size)
                holdout_n = int(holdout_idx.size)
                holdout_fit = fit_isotonic_calibrator(all_scores[train_idx], all_labels[train_idx])
                holdout_raw = _metrics(
                    all_scores[holdout_idx],
                    all_labels[holdout_idx].astype(int).tolist(),
                )
                holdout_cal = _metrics(
                    holdout_fit.transform(all_scores[holdout_idx]),
                    all_labels[holdout_idx].astype(int).tolist(),
                )

    now = datetime.now()
    if args.report_out:
        report_path = Path(args.report_out).expanduser().resolve()
    else:
        report_path = (
            Path(__file__).resolve().parents[2]
            / "reports"
            / f"ssr_hierarchical_tuning_{now.strftime('%Y-%m-%d')}.md"
        )
    artifact_path: Path | None = None
    if args.calibration_artifact_out:
        artifact_path = Path(args.calibration_artifact_out).expanduser().resolve()
    elif args.global_calibration != "none":
        artifact_path = (
            Path(__file__).resolve().parents[2]
            / "reports"
            / f"ssr_calibrator_{now.strftime('%Y-%m-%d')}.json"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# SSR Hierarchical Tuning Report ({now.strftime('%Y-%m-%d')})")
    lines.append("")
    lines.append("## Run Setup")
    lines.append(f"- Embedding model: `{args.model}`")
    lines.append(f"- Anchor language: `{language.value}`")
    lines.append(f"- Anchor language mode: `{args.anchor_language_mode}`")
    lines.append(f"- Configs tested: `{len(configs)}`")
    lines.append(f"- Skip industries: `{args.skip_industries}`")
    lines.append(f"- Global calibration: `{args.global_calibration}`")
    lines.append(f"- Calibration min samples: `{args.calibration_min_samples}`")
    lines.append(f"- Calibration CV folds: `{args.calibration_cv_folds}`")
    lines.append(f"- Calibration holdout ratio: `{args.calibration_holdout_ratio}`")
    lines.append(
        f"- Grid: temperatures={temperatures}, epsilons={epsilons}, "
        f"single_anchor_sets={bool(args.include_single_anchors)}"
    )
    if args.skip_industries:
        lines.append("- Industry sample: `skipped`")
    else:
        lines.append(
            f"- Industry sample: `{args.sample_per_industry}` x `{len(CATEGORIES_20)}` "
            f"(max_scan/category={args.max_scan_per_industry})"
        )
    lines.append(f"- Total groups evaluated: `{len(groups)}`")
    lines.append(f"- Total examples evaluated: `{sum(len(g.labels) for g in groups)}`")
    lines.append("")

    lines.append("## Calibration Effect (Global Best Config)")
    lines.append(f"- Config: `{global_best_key}`")
    lines.append(f"- Apparent sample size: `{apparent_raw.n}`")
    lines.append(
        f"- Apparent Spearman: `{apparent_raw.spearman:.4f} -> {apparent_cal.spearman:.4f}` "
        f"(`{apparent_cal.spearman - apparent_raw.spearman:+.4f}`)"
    )
    lines.append(
        f"- Apparent MAE: `{apparent_raw.mae:.4f} -> {apparent_cal.mae:.4f}` "
        f"(`{apparent_cal.mae - apparent_raw.mae:+.4f}`)"
    )
    lines.append(
        f"- Apparent Off-by-one: `{apparent_raw.off_by_one_acc:.4f} -> {apparent_cal.off_by_one_acc:.4f}` "
        f"(`{apparent_cal.off_by_one_acc - apparent_raw.off_by_one_acc:+.4f}`)"
    )
    lines.append(
        f"- Apparent Exact: `{apparent_raw.exact_acc:.4f} -> {apparent_cal.exact_acc:.4f}` "
        f"(`{apparent_cal.exact_acc - apparent_raw.exact_acc:+.4f}`)"
    )
    lines.append("")

    lines.append("## Calibration Validation (OOF)")
    lines.append(f"- Folds: `{args.calibration_cv_folds}`")
    lines.append(f"- Spearman: `{oof_raw.spearman:.4f} -> {oof_cal.spearman:.4f}` (`{oof_cal.spearman - oof_raw.spearman:+.4f}`)")
    lines.append(f"- MAE: `{oof_raw.mae:.4f} -> {oof_cal.mae:.4f}` (`{oof_cal.mae - oof_raw.mae:+.4f}`)")
    lines.append(
        f"- Off-by-one: `{oof_raw.off_by_one_acc:.4f} -> {oof_cal.off_by_one_acc:.4f}` "
        f"(`{oof_cal.off_by_one_acc - oof_raw.off_by_one_acc:+.4f}`)"
    )
    lines.append(
        f"- Exact: `{oof_raw.exact_acc:.4f} -> {oof_cal.exact_acc:.4f}` "
        f"(`{oof_cal.exact_acc - oof_raw.exact_acc:+.4f}`)"
    )
    lines.append("")

    lines.append("## Calibration Validation (Holdout)")
    lines.append(f"- Holdout ratio: `{args.calibration_holdout_ratio}`")
    lines.append(f"- Train n: `{train_n}` | Holdout n: `{holdout_n}`")
    lines.append(
        f"- Spearman: `{holdout_raw.spearman:.4f} -> {holdout_cal.spearman:.4f}` "
        f"(`{holdout_cal.spearman - holdout_raw.spearman:+.4f}`)"
    )
    lines.append(f"- MAE: `{holdout_raw.mae:.4f} -> {holdout_cal.mae:.4f}` (`{holdout_cal.mae - holdout_raw.mae:+.4f}`)")
    lines.append(
        f"- Off-by-one: `{holdout_raw.off_by_one_acc:.4f} -> {holdout_cal.off_by_one_acc:.4f}` "
        f"(`{holdout_cal.off_by_one_acc - holdout_raw.off_by_one_acc:+.4f}`)"
    )
    lines.append(
        f"- Exact: `{holdout_raw.exact_acc:.4f} -> {holdout_cal.exact_acc:.4f}` "
        f"(`{holdout_cal.exact_acc - holdout_raw.exact_acc:+.4f}`)"
    )
    lines.append("")

    lines.append("## Global Best Config")
    gb = rank_summaries[global_best_key]
    lines.append(f"- Config: `{global_best_key}`")
    lines.append(f"- Weighted Spearman: `{gb['spearman']:.4f}`")
    lines.append(f"- Weighted MAE: `{gb['mae']:.4f}`")
    lines.append(f"- Weighted Off-by-one: `{gb['off1']:.4f}`")
    lines.append(f"- Weighted Exact: `{gb['exact']:.4f}`")
    lines.append("")

    lines.append(f"## Top {min(args.top_k, len(ranked_cfgs))} Global Configs")
    for idx, cfg in enumerate(ranked_cfgs[: args.top_k], start=1):
        s = rank_summaries[cfg.key]
        lines.append(
            f"{idx}. `{cfg.key}` | spearman={s['spearman']:.4f} | "
            f"mae={s['mae']:.4f} | off1={s['off1']:.4f} | exact={s['exact']:.4f}"
        )
    lines.append("")

    lines.append("## Per-Group Best vs Global")
    lines.append("| Group | n | Global Spearman | Best Config | Best Spearman | Uplift | Recommendation |")
    lines.append("|---|---:|---:|---|---:|---:|---|")
    for g in sorted(group_names):
        n = by_cfg_group_ranked[global_best_key][g].n
        g_sp = by_cfg_group_ranked[global_best_key][g].spearman
        b_cfg = group_best[g]
        b_sp = by_cfg_group_ranked[b_cfg.key][g].spearman
        uplift = b_sp - g_sp
        rec = "group-specific" if g in recommend_group_specific else "global"
        lines.append(
            f"| `{g}` | {n} | {g_sp:.4f} | `{b_cfg.key}` | {b_sp:.4f} | {uplift:+.4f} | {rec} |"
        )
    lines.append("")

    lines.append("## Recommendation Summary")
    lines.append(f"- Default (global): `{global_best_key}`")
    lines.append(
        "- Rule for group-specific tuning: "
        f"`n >= {args.group_specific_min_n}` and "
        f"`Spearman uplift >= {args.group_specific_min_uplift:.4f}`"
    )
    if recommend_group_specific:
        lines.append(f"- Groups recommended for group-specific config: `{len(recommend_group_specific)}`")
        for g in sorted(recommend_group_specific):
            lines.append(f"  - `{g}` -> `{group_best[g].key}`")
    else:
        lines.append("- No groups passed the group-specific recommendation threshold.")
    lines.append("")

    if args.extra_csv:
        lines.append("## Extra CSV Data")
        lines.append(f"- Source: `{Path(args.extra_csv).expanduser()}`")
        lines.append(
            f"- Columns: group=`{args.extra_group_col}`, text=`{args.extra_text_col}`, "
            f"label=`{args.extra_label_col}`"
        )
        lines.append(
            f"- Filters: min_group_n={args.min_extra_group_n}, "
            f"sample_per_group={args.sample_per_extra_group}"
        )
        lines.append("")

    if final_calibrator is not None and artifact_path is not None:
        final_calibrator.save_json(
            artifact_path,
            metadata={
                "created_at": now.isoformat(),
                "config_key": global_best_key,
                "embedding_model": args.model,
                "anchor_language": language.value,
                "anchor_language_mode": args.anchor_language_mode,
                "sample_count": int(all_scores.size),
                "calibration_cv_folds": int(args.calibration_cv_folds),
                "calibration_holdout_ratio": float(args.calibration_holdout_ratio),
            },
        )
        lines.append("## Calibration Artifact")
        lines.append(f"- Path: `{artifact_path}`")
        lines.append("- Format: `isotonic_v1`")
        lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== Done ===")
    print(f"Global best: {global_best_key}")
    print(f"Groups: {len(groups)} | Examples: {sum(len(g.labels) for g in groups)}")
    print(f"Report saved: {report_path}")
    if final_calibrator is not None and artifact_path is not None:
        print(f"Calibration artifact saved: {artifact_path}")

    if recommend_group_specific:
        print("\nGroups with recommended group-specific configs:")
        for g in sorted(recommend_group_specific):
            print(f"- {g}: {group_best[g].key} (uplift={group_uplift[g]:+.4f})")


if __name__ == "__main__":
    main()
