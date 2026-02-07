"""Parity test: Market Wizard SSR math vs semantic-similarity-rating compute.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_reference_compute_module():
    path = Path("/Users/pawel/semantic-similarity-rating/semantic_similarity_rating/compute.py")
    if not path.exists():
        pytest.skip("Reference semantic-similarity-rating repo not available locally.")
    spec = importlib.util.spec_from_file_location("ssr_compute", str(path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mw_response_embeddings_to_pmf(matrix_responses, matrix_likert_sentences, epsilon=0.0):
    m_left = matrix_responses
    m_right = matrix_likert_sentences
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


def _mw_scale_pmf(pmf, temperature):
    pmf = np.asarray(pmf, dtype=float)
    if temperature == 1.0:
        return pmf
    if temperature == 0.0:
        if np.all(pmf == pmf[0]):
            return pmf
        out = np.zeros_like(pmf)
        out[np.argmax(pmf)] = 1.0
        return out
    hist = pmf ** (1 / temperature)
    return hist / hist.sum()


def test_ssr_core_math_parity_with_reference_repo():
    ref = _load_reference_compute_module()
    ref_pmf = ref.response_embeddings_to_pmf
    ref_scale = ref.scale_pmf

    rng = np.random.default_rng(123)

    for eps in [0.0, 1e-6, 0.01, 0.2]:
        for _ in range(40):
            responses = rng.normal(size=(6, 384))
            likert = rng.normal(size=(384, 5))
            a = ref_pmf(responses, likert, epsilon=eps)
            b = _mw_response_embeddings_to_pmf(responses, likert, epsilon=eps)
            assert np.allclose(a, b, atol=1e-12, rtol=1e-12)

    for t in [0.0, 0.1, 1.0, 2.0, 10.0]:
        for _ in range(40):
            pmf = rng.random(5)
            pmf = pmf / pmf.sum()
            a = ref_scale(pmf, t)
            b = _mw_scale_pmf(pmf, t)
            assert np.allclose(a, b, atol=1e-12, rtol=1e-12)
