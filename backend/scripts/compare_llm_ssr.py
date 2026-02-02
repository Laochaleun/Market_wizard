#!/usr/bin/env python3
"""
Compare SSR outputs across two LLMs (current Gemini vs GPT-4o) and two embeddings.

Example:
  TMPDIR=/path/to/tmp /path/to/python backend/scripts/compare_llm_ssr.py \
    --input /path/to/project.json \
    --language pl \
    --limit 20 \
    --temperature 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, List, Dict

import numpy as np

from app.i18n import Language, get_persona_prompt
from app.models import Persona
from app.services.embedding_client import LocalEmbeddingClient, OpenAIEmbeddingClient
from app.services.llm_client import get_llm_client
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

def _load_project(path: Path) -> tuple[List[Persona], str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    research = data.get("research", {}) if isinstance(data, dict) else {}
    sim = research.get("simulation", {}) if isinstance(research, dict) else {}
    result = sim.get("result", {}) if isinstance(sim, dict) else {}
    agents = result.get("agent_responses", []) if isinstance(result, dict) else []
    personas = []
    for item in agents:
        if not isinstance(item, dict):
            continue
        persona_dict = item.get("persona")
        if isinstance(persona_dict, dict):
            personas.append(Persona(**persona_dict))
    if not personas:
        raise ValueError("No personas found in project JSON.")

    inputs = sim.get("inputs", {}) if isinstance(sim, dict) else {}
    product_description = inputs.get("product_description") or data.get("product_description")
    if not product_description:
        raise ValueError("No product_description found in project JSON.")
    return personas, product_description


def _likert_array(dist) -> np.ndarray:
    return np.array(
        [dist.scale_1, dist.scale_2, dist.scale_3, dist.scale_4, dist.scale_5],
        dtype=float,
    )


def _ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.cumsum(a) - np.cumsum(b))))


def _summarize(label: str, scores: Iterable[float]) -> None:
    arr = np.array(list(scores), dtype=float)
    print(
        f"{label}: mean={arr.mean():.4f} std={arr.std(ddof=0):.4f} min={arr.min():.4f} max={arr.max():.4f}"
    )


def _generate_gpt4o_responses(
    *,
    personas: List[Persona],
    product_description: str,
    language: Language,
    temperature: float,
    openai_key: str,
    model: str,
) -> List[str]:
    from openai import OpenAI

    client = OpenAI(api_key=openai_key)
    outputs: List[str] = []
    for persona in personas:
        prompt = get_persona_prompt(
            language=language,
            name=persona.name,
            age=persona.age,
            gender=persona.gender,
            location=persona.location,
            income=persona.income,
            occupation=persona.occupation,
            product_description=product_description,
        )
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content or ""
        outputs.append(text.strip())
    return outputs


def _build_full_prompt(persona: Persona, product_description: str, language: Language) -> str:
    if language == Language.PL:
        gender_word = "kobieta" if persona.gender == "F" else "mężczyzna"
        occupation_line = f"\nPracujesz jako {persona.occupation}." if persona.occupation else ""
        return (
            f"Jesteś {persona.name}, {persona.age}-letni {gender_word} mieszkający w {persona.location}."
            f"\nTwój miesięczny dochód to około {persona.income} PLN.{occupation_line}\n\n"
            f"PRODUKT DO OCENY:\n{product_description}\n\n"
            "Odpowiedz naturalnie, jak w rozmowie:\n"
            "1. Jak ten produkt wypada vs konkurencja?\n"
            "2. Czy cena jest atrakcyjna biorąc pod uwagę Twoje zarobki?\n"
            "3. Czy kupiłbyś/kupiłabyś? Dlaczego?"
        )
    gender_word = "woman" if persona.gender == "F" else "man"
    occupation_line = f"\nYou work as a {persona.occupation}." if persona.occupation else ""
    return (
        f"You are {persona.name}, a {persona.age}-year-old {gender_word} living in {persona.location}."
        f"\nYour monthly income is about ${persona.income}.{occupation_line}\n\n"
        f"PRODUCT TO EVALUATE:\n{product_description}\n\n"
        "Answer naturally, as in conversation:\n"
        "1. How does this product compare to competition?\n"
        "2. Is the price attractive given your income?\n"
        "3. Would you buy it? Why?"
    )


async def _generate_gemini_responses(
    *,
    personas: List[Persona],
    product_description: str,
    language: Language,
    temperature: float,
    model_override: str | None,
) -> List[str]:
    client = get_llm_client(model_override)
    outputs: List[str] = []
    for persona in personas:
        text = await client.generate_opinion(
            persona,
            product_description,
            language=language,
            temperature=temperature,
        )
        outputs.append((text or "").strip())
    return outputs


def _load_cache(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {"gemini": [], "gpt4o": [], "prompt": ""}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "gemini": list(data.get("gemini", [])),
        "gpt4o": list(data.get("gpt4o", [])),
        "prompt": data.get("prompt", ""),
    }


def _save_cache(path: Path, cache: Dict[str, List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SSR across LLMs and embeddings.")
    parser.add_argument("--input", required=True, help="Path to project JSON.")
    parser.add_argument("--language", default="pl", choices=["pl", "en"])
    parser.add_argument("--limit", type=int, default=20, help="Limit personas for cost control.")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--env-file", default="backend/.env")
    parser.add_argument("--gemini-model", default=None, help="Optional override for Gemini model.")
    parser.add_argument("--gpt4o-model", default="gpt-4o")
    parser.add_argument("--openai-embed-model", default="text-embedding-3-small")
    parser.add_argument("--local-embed-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--out", default="", help="Optional JSON output path.")
    parser.add_argument("--cache", default="", help="Optional cache JSON path to resume long runs.")
    parser.add_argument("--chunk-size", type=int, default=0, help="Generate only N new responses per run.")
    parser.add_argument("--compute-partial", action="store_true", help="Compute SSR even if cache is incomplete.")
    parser.add_argument("--prompt-style", default="short", choices=["short", "full"])
    parser.add_argument("--plot-out", default="", help="Optional PNG path for comparison chart.")
    parser.add_argument("--compare-intent-full", action="store_true", help="Compare SSR on full vs intent text.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    env_path = Path(args.env_file).expanduser().resolve()
    env = _parse_env_file(env_path)

    openai_key = os.getenv("OPENAI_API_KEY") or env.get("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment or .env file.")

    personas, product_description = _load_project(input_path)
    if args.limit and args.limit > 0:
        personas = personas[: args.limit]

    language = Language.PL if args.language == "pl" else Language.EN

    cache_path = Path(args.cache).expanduser().resolve() if args.cache else None
    cache = {"gemini": [], "gpt4o": [], "prompt": ""}
    if cache_path:
        cache = _load_cache(cache_path)
        if cache.get("prompt") and cache.get("prompt") != args.prompt_style:
            raise RuntimeError("Cache prompt style mismatch. Use a different cache file.")
    cache["prompt"] = args.prompt_style

    target_n = len(personas)
    start_idx = 0
    existing = min(len(cache["gemini"]), len(cache["gpt4o"]))
    start_idx = existing

    if start_idx < target_n:
        if args.chunk_size and args.chunk_size > 0:
            end_idx = min(target_n, start_idx + args.chunk_size)
        else:
            end_idx = target_n
        batch = personas[start_idx:end_idx]

        import asyncio

        if args.prompt_style == "short":
            gemini_batch = asyncio.run(
                _generate_gemini_responses(
                    personas=batch,
                    product_description=product_description,
                    language=language,
                    temperature=args.temperature,
                    model_override=args.gemini_model,
                )
            )
            gpt4o_batch = _generate_gpt4o_responses(
                personas=batch,
                product_description=product_description,
                language=language,
                temperature=args.temperature,
                openai_key=openai_key,
                model=args.gpt4o_model,
            )
        else:
            gemini_client = get_llm_client(args.gemini_model)
            gemini_batch = []
            for persona in batch:
                prompt = _build_full_prompt(persona, product_description, language)
                from google.genai import types
                response = gemini_client.client.models.generate_content(
                    model=gemini_client.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=args.temperature,
                        max_output_tokens=2048,
                    ),
                )
                text = response.text or ""
                gemini_batch.append((text or "").strip())

            gpt4o_batch = []
            from openai import OpenAI

            client = OpenAI(api_key=openai_key)
            for persona in batch:
                prompt = _build_full_prompt(persona, product_description, language)
                response = client.chat.completions.create(
                    model=args.gpt4o_model,
                    temperature=args.temperature,
                    max_tokens=700,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.choices[0].message.content or ""
                gpt4o_batch.append(text.strip())

        cache["gemini"].extend(gemini_batch)
        cache["gpt4o"].extend(gpt4o_batch)
        if cache_path:
            _save_cache(cache_path, cache)

    if len(cache["gemini"]) < target_n or len(cache["gpt4o"]) < target_n:
        print(f"Generated {len(cache['gemini'])}/{target_n} responses. Re-run to continue.")
        if not args.compute_partial:
            return

    gemini_responses = cache["gemini"][:target_n]
    gpt4o_responses = cache["gpt4o"][:target_n]

    local_client = LocalEmbeddingClient(model_name=args.local_embed_model)
    openai_client = OpenAIEmbeddingClient(api_key=openai_key, model=args.openai_embed_model)

    ssr_local = SSREngine(embedding_client=local_client, language=language)
    ssr_openai = SSREngine(embedding_client=openai_client, language=language)

    results = {}
    for llm_label, responses in (("gemini", gemini_responses), ("gpt4o", gpt4o_responses)):
        local_scores = ssr_local.rate_responses(responses)
        openai_scores = ssr_openai.rate_responses(responses)
        results[llm_label] = {
            "local": {
                "scores": [r.expected_score for r in local_scores],
                "distribution": _likert_array(ssr_local.aggregate_to_survey_pmf(local_scores)).tolist(),
            },
            "openai": {
                "scores": [r.expected_score for r in openai_scores],
                "distribution": _likert_array(ssr_openai.aggregate_to_survey_pmf(openai_scores)).tolist(),
            },
        }
        if args.compare_intent_full:
            intent_texts = [_extract_purchase_intent_text(t, language) for t in responses]
            local_intent_scores = ssr_local.rate_responses(intent_texts)
            openai_intent_scores = ssr_openai.rate_responses(intent_texts)
            results[llm_label]["intent_local"] = {
                "scores": [r.expected_score for r in local_intent_scores],
                "distribution": _likert_array(
                    ssr_local.aggregate_to_survey_pmf(local_intent_scores)
                ).tolist(),
            }
            results[llm_label]["intent_openai"] = {
                "scores": [r.expected_score for r in openai_intent_scores],
                "distribution": _likert_array(
                    ssr_openai.aggregate_to_survey_pmf(openai_intent_scores)
                ).tolist(),
            }

    print("=== LLM + Embedding SSR Comparison ===")
    print(f"Responses: {len(personas)} | language={args.language} | temp={args.temperature}")
    print(f"Gemini model override: {args.gemini_model or 'default'}")
    print(f"GPT-4o model: {args.gpt4o_model}")
    print(f"Embeddings: local={args.local_embed_model}, openai={args.openai_embed_model}")
    print(f"Prompt style: {args.prompt_style}")
    print("")

    for llm_label in ("gemini", "gpt4o"):
        for emb_label in ("local", "openai"):
            scores = results[llm_label][emb_label]["scores"]
            _summarize(f"{llm_label}/{emb_label} mean scores", scores)
        print("")

    for emb_label in ("local", "openai"):
        g = np.array(results["gemini"][emb_label]["distribution"], dtype=float)
        o = np.array(results["gpt4o"][emb_label]["distribution"], dtype=float)
        l1 = float(np.sum(np.abs(g - o)))
        ks = _ks_distance(g, o)
        print(f"LLM diff on {emb_label} embeddings:")
        print(f"  Gemini dist: {np.round(g, 6).tolist()}")
        print(f"  GPT-4o dist: {np.round(o, 6).tolist()}")
        print(f"  L1={l1:.6f} KS={ks:.6f}")
        print("")

    if args.compare_intent_full:
        for llm_label in ("gemini", "gpt4o"):
            for emb_label in ("local", "openai"):
                full_dist = np.array(results[llm_label][emb_label]["distribution"], dtype=float)
                intent_key = f"intent_{emb_label}"
                intent_dist = np.array(results[llm_label][intent_key]["distribution"], dtype=float)
                l1 = float(np.sum(np.abs(full_dist - intent_dist)))
                ks = _ks_distance(full_dist, intent_dist)
                print(f"Intent vs Full ({llm_label}/{emb_label}): L1={l1:.6f} KS={ks:.6f}")
            print("")

    if args.plot_out:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise RuntimeError(f"matplotlib not available: {exc}") from exc

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        x = np.arange(1, 6)
        width = 0.35

        gem_local = np.array(results["gemini"]["local"]["distribution"])
        gem_open = np.array(results["gemini"]["openai"]["distribution"])
        axes[0].bar(x - width / 2, gem_local, width=width, label="Local")
        axes[0].bar(x + width / 2, gem_open, width=width, label="OpenAI")
        axes[0].set_title("Gemini")
        axes[0].set_xticks(x)
        axes[0].set_xlabel("Likert scale")
        axes[0].set_ylabel("Probability")
        axes[0].legend()

        gpt_local = np.array(results["gpt4o"]["local"]["distribution"])
        gpt_open = np.array(results["gpt4o"]["openai"]["distribution"])
        axes[1].bar(x - width / 2, gpt_local, width=width, label="Local")
        axes[1].bar(x + width / 2, gpt_open, width=width, label="OpenAI")
        axes[1].set_title("GPT-4o")
        axes[1].set_xticks(x)
        axes[1].set_xlabel("Likert scale")
        axes[1].set_ylabel("Probability")
        axes[1].legend()

        fig.tight_layout()
        out_path = Path(args.plot_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
        print(f"Saved chart: {out_path}")

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "meta": {
                        "input": str(input_path),
                        "language": args.language,
                        "temperature": args.temperature,
                        "gemini_model": args.gemini_model,
                        "gpt4o_model": args.gpt4o_model,
                        "local_embedding": args.local_embed_model,
                        "openai_embedding": args.openai_embed_model,
                        "prompt_style": args.prompt_style,
                        "n": len(personas),
                    },
                    "results": results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved JSON: {out_path}")


if __name__ == "__main__":
    main()
