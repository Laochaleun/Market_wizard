#!/usr/bin/env python3
"""External grounding probe for Gemini Search.

Runs a few prompt strategies and prints source counts and domains.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse

from app.config import get_settings
from app.i18n import Language
from app.models import Persona
from app.services.llm_client import GeminiClient


@dataclass
class ProbeResult:
    name: str
    sources: list[object]


def _extract_urls(sources: Iterable[object]) -> list[str]:
    urls: list[str] = []
    for item in sources:
        if isinstance(item, str):
            url = item
        elif isinstance(item, dict):
            url = str(item.get("url") or "")
        else:
            url = ""
        if url:
            urls.append(url)
    return urls


def _domains(urls: Iterable[str]) -> list[str]:
    domains: list[str] = []
    for url in urls:
        try:
            netloc = urlparse(url).netloc.lower()
        except Exception:
            continue
        if netloc:
            domains.append(netloc)
    return domains


async def _run_prompt(
    client: GeminiClient,
    prompt: str,
    resolve_urls: bool,
) -> tuple[list[str], dict[str, int]]:
    from google.genai import types

    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(
        tools=[grounding_tool],
        temperature=0.2,
        max_output_tokens=256,
    )
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.client.models.generate_content(
            model=client.model_name,
            contents=prompt,
            config=config,
        ),
    )
    sources: list[str] = []
    stats = {"has_metadata": 0, "chunks": 0, "web_uris": 0}
    metadata = None
    if response.candidates:
        candidate = response.candidates[0]
        metadata = getattr(candidate, "grounding_metadata", None) or getattr(
            candidate, "groundingMetadata", None
        )
    if not metadata:
        metadata = getattr(response, "grounding_metadata", None) or getattr(
            response, "groundingMetadata", None
        )
    if metadata:
        stats["has_metadata"] = 1
        chunks = getattr(metadata, "grounding_chunks", None) or getattr(
            metadata, "groundingChunks", None
        )
        if chunks:
            stats["chunks"] = len(chunks)
        sources = client._extract_grounding_urls(metadata)
        stats["web_uris"] = len(sources)
    if resolve_urls:
        return await client._resolve_grounding_urls(sources), stats
    return sources, stats


def _build_global_prompts(product: str, language: Language) -> list[str]:
    if language == Language.PL:
        return [
            f"Znajdź aktualne produkty konkurencyjne i ceny dla: {product}. "
            "Preferuj różne domeny (sklepy, porównywarki, recenzje).",
            f"Wyszukaj alternatywy w tej samej kategorii oraz ceny rynkowe: {product}. "
            "Zwróć źródła z różnych domen.",
            f"Znajdź recenzje i porównania cenowe dla podobnych produktów: {product}. "
            "Zwróć możliwie różnorodne domeny.",
            f"Znajdź strony producentów i oficjalne karty produktów dla: {product}. "
            "Zwróć różne domeny.",
            f"Wyszukaj oferty sklepów zagranicznych i międzynarodowe porównania cen dla: {product}. "
            "Zwróć różne domeny.",
        ]
    return [
        f"Find competing products and current prices for: {product}. "
        "Prefer diverse domains (shops, comparisons, reviews).",
        f"Search for alternatives in the same category and market prices: {product}. "
        "Return sources from varied domains.",
        f"Find reviews and price comparisons for similar products: {product}. "
        "Return a diverse set of domains.",
        f"Find manufacturer pages and official product sheets for: {product}. "
        "Return diverse domains.",
        f"Search for international retailers and global price comparisons for: {product}. "
        "Return diverse domains.",
    ]


async def _probe_product(
    client: GeminiClient,
    product: str,
    language: Language,
    timeout_s: float,
    run_per_agent: bool,
    resolve_urls: bool,
    run_global_prompts: bool,
    run_global_raw: bool,
) -> list[ProbeResult]:
    results: list[ProbeResult] = []

    if run_global_prompts:
        print("Running: global_prompts")
        try:
            sources_global = await asyncio.wait_for(
                client.generate_market_sources(product, language=language),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            sources_global = []
            results.append(ProbeResult("global_prompts_timeout", sources_global))
        else:
            results.append(ProbeResult("global_prompts", sources_global))
    elif run_global_raw:
        print("Running: global_prompts_raw")
        combined: list[str] = []
        for prompt in _build_global_prompts(product, language):
            try:
                sources, stats = await asyncio.wait_for(
                    _run_prompt(client, prompt, resolve_urls=False),
                    timeout=timeout_s,
                )
                print(f"  stats: {stats}")
                combined.extend(sources)
            except asyncio.TimeoutError:
                continue
        results.append(ProbeResult("global_prompts_raw", combined))

    if language == Language.PL:
        prompt_a = (
            f"Znajdz 5-8 zrodel (tylko linki) o produktach konkurencyjnych i cenach dla: {product}. "
            "Preferuj rozne domeny."
        )
        prompt_b = (
            f"Wyszukaj w sieci informacje i ceny dla: {product}. "
            "Zwracaj uwage na porownania i rozne domeny."
        )
        prompt_c = (
            f"Znajdz zrodla zagraniczne (EN) dla: {product}. "
            "Zwracaj rozne domeny."
        )
    else:
        prompt_a = (
            f"Find 5-8 sources (links only) about competing products and prices for: {product}. "
            "Prefer diverse domains."
        )
        prompt_b = (
            f"Search the web for information and prices for: {product}. "
            "Focus on comparisons and diverse domains."
        )
        prompt_c = (
            f"Find international (EN) sources for: {product}. "
            "Return diverse domains."
        )

    print("Running: single_prompt_links")
    try:
        sources, stats = await asyncio.wait_for(
            _run_prompt(client, prompt_a, resolve_urls=resolve_urls),
            timeout=timeout_s,
        )
        print(f"  stats: {stats}")
        results.append(ProbeResult("single_prompt_links", sources))
    except asyncio.TimeoutError:
        results.append(ProbeResult("single_prompt_links_timeout", []))

    print("Running: single_prompt_general")
    try:
        sources, stats = await asyncio.wait_for(
            _run_prompt(client, prompt_b, resolve_urls=resolve_urls),
            timeout=timeout_s,
        )
        print(f"  stats: {stats}")
        results.append(ProbeResult("single_prompt_general", sources))
    except asyncio.TimeoutError:
        results.append(ProbeResult("single_prompt_general_timeout", []))

    print("Running: single_prompt_international")
    try:
        sources, stats = await asyncio.wait_for(
            _run_prompt(client, prompt_c, resolve_urls=resolve_urls),
            timeout=timeout_s,
        )
        print(f"  stats: {stats}")
        results.append(ProbeResult("single_prompt_international", sources))
    except asyncio.TimeoutError:
        results.append(ProbeResult("single_prompt_international_timeout", []))

    # Per-agent style prompt (no grounding tool, just to compare behavior)
    persona = Persona(
        name="Test",
        age=35,
        gender="M",
        location="Warsaw" if language == Language.EN else "Warszawa",
        income=6000 if language == Language.PL else 4000,
        location_type="urban",
        education="higher",
        occupation="engineer",
    )
    if run_per_agent:
        print("Running: per_agent_search")
        try:
            opinion, sources = await asyncio.wait_for(
                client.generate_opinion_with_search(
                    persona=persona,
                    product_description=product,
                    language=language,
                ),
                timeout=timeout_s,
            )
            _ = opinion  # keep for future diagnostics
            results.append(ProbeResult("per_agent_search", sources))
        except asyncio.TimeoutError:
            results.append(ProbeResult("per_agent_search_timeout", []))

    return results


def _print_results(product: str, results: list[ProbeResult]) -> None:
    print(f"\n=== Product: {product}")
    for result in results:
        urls = _extract_urls(result.sources)
        domains = _domains(urls)
        domain_counts = Counter(domains)
        top_domains = ", ".join([f"{d}({c})" for d, c in domain_counts.most_common(5)])
        print(f"- {result.name}: sources={len(result.sources)} domains={len(domain_counts)}")
        if top_domains:
            print(f"  top_domains: {top_domains}")
        if urls:
            sample = urls[:5]
            print(f"  sample: {sample}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["pl", "en"], default="pl")
    parser.add_argument("--product", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--skip-per-agent", action="store_true")
    parser.add_argument("--model", default="")
    parser.add_argument("--skip-resolve", action="store_true")
    parser.add_argument("--skip-global", action="store_true")
    parser.add_argument("--global-raw", action="store_true")
    args = parser.parse_args()

    language = Language.PL if args.language == "pl" else Language.EN
    settings = get_settings()
    if not settings.google_api_key:
        raise SystemExit("GOOGLE_API_KEY is required for grounding probe.")

    products = args.product or [
        "Unikatowa szklana karafka w ksztalcie srodkowego palca, 1L, 100 PLN",
        "Apple iPhone 15 Pro clear case with MagSafe, price 59 USD",
        "LEGO Star Wars Millennium Falcon 75192, price 849.99 USD",
    ]

    model_name = args.model or settings.llm_model
    client = GeminiClient(
        api_key=settings.google_api_key,
        model=model_name,
        temperature=settings.llm_temperature,
    )

    for product in products:
        results = await _probe_product(
            client,
            product,
            language,
            args.timeout,
            run_per_agent=not args.skip_per_agent,
            resolve_urls=not args.skip_resolve,
            run_global_prompts=not args.skip_global and not args.global_raw,
            run_global_raw=bool(args.global_raw),
        )
        _print_results(product, results)


if __name__ == "__main__":
    asyncio.run(main())
