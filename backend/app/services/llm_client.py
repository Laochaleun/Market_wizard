"""LLM client for generating synthetic consumer responses."""

from typing import Protocol

import logging

from app.config import get_settings
from app.models import Persona
from app.i18n import Language, get_persona_prompt


class LLMClient(Protocol):
    """Protocol for LLM clients."""

    async def generate_opinion(
        self, 
        persona: Persona, 
        product_description: str,
        language: Language = Language.PL,
        temperature: float | None = None,
    ) -> str:
        """Generate a purchase intent opinion from a synthetic consumer."""
        ...


class GeminiClient:
    """Google Gemini LLM client using new google-genai SDK."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-001", temperature: float | None = None):
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model_name = model
        self.temperature = temperature if temperature is not None else get_settings().llm_temperature
        self._url_cache: dict[str, str] = {}
        self._research_cache: dict[str, dict[str, object]] = {}
        self._logger = logging.getLogger(__name__)
        settings = get_settings()
        self.research_model_name = getattr(settings, "research_llm_model", self.model_name)
        from datetime import timedelta
        from pathlib import Path
        self._research_cache_ttl = timedelta(hours=getattr(settings, "research_cache_ttl_hours", 24))
        self._research_cache_path = Path(__file__).resolve().parents[2] / "data" / "search_cache.json"
        self._load_research_cache()

    async def generate_opinion(
        self, 
        persona: Persona, 
        product_description: str,
        language: Language = Language.PL,
        temperature: float | None = None,
    ) -> str:
        """Generate opinion using Gemini."""
        from google.genai import types
        
        # Get language-specific prompt from i18n module
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

        # Use synchronous call wrapped for async (genai doesn't have native async yet)
        import asyncio
        loop = asyncio.get_event_loop()
        
        temp = self.temperature if temperature is None else temperature
        config = types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=2048,  # Enough for full responses
        )
        
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
        )

        return response.text or ""

    def _load_research_cache(self) -> None:
        import json
        from datetime import datetime

        if not self._research_cache_path.exists():
            return
        try:
            raw = json.loads(self._research_cache_path.read_text(encoding="utf-8"))
            entries = raw.get("entries", {})
            now = datetime.now()
            cleaned: dict[str, dict[str, object]] = {}
            for key, entry in entries.items():
                cached_at = entry.get("cached_at")
                sources = entry.get("sources")
                if not cached_at or not isinstance(sources, list):
                    continue
                try:
                    cached_dt = datetime.fromisoformat(cached_at)
                except ValueError:
                    continue
                if now - cached_dt <= self._research_cache_ttl and sources:
                    cleaned[key] = {"cached_at": cached_at, "sources": sources}
            self._research_cache = cleaned
        except Exception:
            self._research_cache = {}

    def _save_research_cache(self) -> None:
        import json
        from datetime import datetime

        data = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "entries": self._research_cache,
        }
        self._research_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._research_cache_path.write_text(
            json.dumps(data, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def _make_research_cache_key(self, product_description: str, language: Language) -> str:
        import hashlib

        cache_version = "v2"
        key = f"{cache_version}|{self.research_model_name}|{language.value}|{product_description.strip()}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()

    def _get_cached_sources(self, key: str) -> list[dict[str, str]] | None:
        entry = self._research_cache.get(key)
        if not entry:
            return None
        sources = entry.get("sources")
        if isinstance(sources, list) and sources:
            # Accept records with url (summary optional).
            if all(isinstance(s, dict) and s.get("url") for s in sources):
                return sources
        return None

    def _set_cached_sources(self, key: str, sources: list[dict[str, str]]) -> None:
        from datetime import datetime

        if not sources:
            return
        self._research_cache[key] = {
            "cached_at": datetime.now().isoformat(timespec="seconds"),
            "sources": sources,
        }
        self._save_research_cache()

    def _select_sources_for_persona(
        self,
        sources: list[dict[str, str]],
        persona: Persona,
        language: Language,
        max_sources: int = 6,
    ) -> list[dict[str, str]]:
        if not sources:
            return []

        def is_preferred(url: str) -> bool:
            u = url.lower()
            if language == Language.PL:
                return u.endswith(".pl") or "/pl/" in u or "pln" in u
            return any(u.endswith(suffix) for suffix in (".com", ".co", ".uk", ".us")) or "/en/" in u

        eligible = [s for s in sources if s.get("summary")]
        preferred = [s for s in eligible if is_preferred(s.get("url", ""))]
        other = [s for s in eligible if s not in preferred]

        import random
        seed = hash(f"{persona.name}|{persona.age}|{persona.location}")
        rng = random.Random(seed)
        rng.shuffle(preferred)
        rng.shuffle(other)

        picked: list[str] = []
        ordered = preferred + other if preferred else other
        for source in ordered:
            if source not in picked:
                picked.append(source)
            if len(picked) >= max_sources:
                break
        return picked

    @staticmethod
    def _response_has_required_citations(text: str, sources_count: int) -> bool:
        import re

        if sources_count <= 0:
            return True

        # Require at least one citation anywhere.
        if not re.search(r"\[[1-9]\d*\]", text):
            return False

        # If prices are mentioned, require nearby citation.
        price_patterns = [
            r"\b\d{1,3}(?:[ .]\d{3})*(?:[.,]\d{1,2})?\s?(?:PLN|zł|zl|USD|EUR|GBP)\b",
            r"\b(?:PLN|zł|zl|USD|EUR|GBP)\s?\d{1,3}(?:[ .]\d{3})*(?:[.,]\d{1,2})?\b",
        ]
        price_regex = re.compile("|".join(price_patterns), re.IGNORECASE)
        for match in price_regex.finditer(text):
            start, end = match.span()
            window = text[max(0, start - 60) : min(len(text), end + 60)]
            if not re.search(r"\[[1-9]\d*\]", window):
                return False

        # Require citations near product/competitor names (format-based heuristics).
        name_patterns = [
            r"\"[^\"]{3,}\"",          # quoted names
            r"„[^”]{3,}”",             # Polish quotes
            r"\*\*[^*]{3,}\*\*",       # markdown bold
            r"\b[A-Z]{2,}[-_ ]?\d+[A-Z0-9-]*\b",  # model-like codes
        ]
        name_regex = re.compile("|".join(name_patterns))
        for match in name_regex.finditer(text):
            start, end = match.span()
            window = text[max(0, start - 80) : min(len(text), end + 80)]
            if not re.search(r"\[[1-9]\d*\]", window):
                return False
        # Also enforce citations on list lines that look like product labels.
        for line in text.splitlines():
            if line.strip().startswith(("-", "*")) or ":" in line:
                if not re.search(r"\[[1-9]\d*\]", line):
                    return False

        return True

    def _extract_grounding_urls(self, metadata) -> list[str]:
        urls: list[str] = []

        def add_url(value):
            if not value:
                return
            if isinstance(value, str) and value.startswith("http"):
                urls.append(value)

        def try_get_attr(obj, *names):
            if obj is None:
                return None
            for name in names:
                if isinstance(obj, dict) and name in obj:
                    return obj[name]
                if hasattr(obj, name):
                    return getattr(obj, name)
            return None

        chunks = try_get_attr(metadata, "grounding_chunks", "groundingChunks")
        if chunks:
            for chunk in chunks:
                web = try_get_attr(chunk, "web")
                if web:
                    add_url(try_get_attr(web, "uri", "url", "link"))

        search_entry = try_get_attr(metadata, "search_entry_point", "searchEntryPoint")
        if search_entry:
            add_url(try_get_attr(search_entry, "uri", "url", "link"))

        return urls

    def _log_grounding_status(self, context: str, response, sources: list[str]) -> None:
        try:
            candidates = getattr(response, "candidates", None)
            candidate_count = len(candidates) if candidates else 0
            metadata = None
            if candidates:
                candidate = candidates[0]
                metadata = getattr(candidate, "grounding_metadata", None) or getattr(candidate, "groundingMetadata", None)
            if not metadata:
                metadata = getattr(response, "grounding_metadata", None) or getattr(response, "groundingMetadata", None)
            has_metadata = bool(metadata)
            self._logger.info(
                "Grounding %s | model=%s | candidates=%s | metadata=%s | sources=%s",
                context,
                self.model_name,
                candidate_count,
                has_metadata,
                len(sources),
            )
        except Exception:
            self._logger.debug("Grounding %s | logging failed", context)

    @staticmethod
    def _is_grounding_redirect(url: str) -> bool:
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)
        except Exception:
            return False
        return (
            parsed.netloc.endswith("vertexaisearch.cloud.google.com")
            and parsed.path.startswith("/grounding-api-redirect/")
        )

    async def _resolve_grounding_urls(self, urls: list[str]) -> list[str]:
        if not urls:
            return []

        import asyncio
        import httpx

        timeout = httpx.Timeout(10.0, connect=5.0)
        headers = {"User-Agent": "MarketWizard/1.0"}
        semaphore = asyncio.Semaphore(5)

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            headers=headers,
        ) as client:

            async def resolve(url: str) -> str:
                cached = self._url_cache.get(url)
                if cached:
                    return cached
                if not self._is_grounding_redirect(url):
                    self._url_cache[url] = url
                    return url
                async with semaphore:
                    try:
                        resp = await client.head(url)
                        if resp.status_code >= 400:
                            resp = await client.get(url)
                        final_url = str(resp.url)
                        resolved = final_url or url
                        self._url_cache[url] = resolved
                        return resolved
                    except Exception:
                        self._url_cache[url] = url
                        return url

            resolved = await asyncio.gather(*(resolve(u) for u in urls))

        seen: set[str] = set()
        deduped: list[str] = []
        for url in resolved:
            if url and url not in seen:
                seen.add(url)
                deduped.append(url)
        return deduped

    async def generate_market_sources(
        self,
        product_description: str,
        language: Language = Language.PL,
        max_sources: int = 25,
    ) -> list[dict[str, str]]:
        cache_key = self._make_research_cache_key(product_description, language)
        cached = self._get_cached_sources(cache_key)
        if cached:
            return cached

        from google.genai import types
        import asyncio

        if language == Language.PL:
            prompts = [
                f"Znajdź 5-8 źródeł (linki) o produktach konkurencyjnych i cenach dla: {product_description}. "
                "Preferuj różne domeny.",
                f"Wyszukaj w sieci informacje i ceny dla: {product_description}. "
                "Zwróć uwagę na porównania i różne domeny.",
                f"Znajdź źródła zagraniczne (EN) dla: {product_description}. "
                "Zwróć różne domeny.",
            ]
        else:
            prompts = [
                f"Find 5-8 sources (links) about competing products and prices for: {product_description}. "
                "Prefer diverse domains.",
                f"Search the web for information and prices for: {product_description}. "
                "Focus on comparisons and diverse domains.",
                f"Find international (EN) sources for: {product_description}. "
                "Return diverse domains.",
            ]

        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(
            tools=[grounding_tool],
            temperature=0.2,
            max_output_tokens=256,
        )

        loop = asyncio.get_event_loop()

        async def run_prompt(prompt: str) -> list[str]:
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.research_model_name,
                    contents=prompt,
                    config=config,
                ),
            )
            sources: list[str] = []
            metadata = None
            if response.candidates:
                candidate = response.candidates[0]
                metadata = getattr(candidate, "grounding_metadata", None) or getattr(candidate, "groundingMetadata", None)
            if not metadata:
                metadata = getattr(response, "grounding_metadata", None) or getattr(response, "groundingMetadata", None)
            if metadata:
                sources = self._extract_grounding_urls(metadata)
                self._logger.info("Market sources grounding: extracted %s urls", len(sources))
            else:
                self._logger.info("Market sources grounding: no metadata")
            return sources

        prompt_timeout_s = 18.0

        async def run_prompt_safe(prompt: str) -> list[str]:
            try:
                return await asyncio.wait_for(run_prompt(prompt), timeout=prompt_timeout_s)
            except Exception as exc:
                self._logger.warning("Market sources prompt failed: %s", exc)
                return []

        results = await asyncio.gather(*(run_prompt_safe(p) for p in prompts))
        combined = [u for batch in results for u in batch]
        combined = await self._resolve_grounding_urls(combined)

        def _is_product_like(u: str) -> bool:
            import re
            u_lc = (u or "").lower()
            if any(x in u_lc for x in ["/product/", "/p/", "/sku/", "/item/", "/dp/"]):
                return True
            return bool(re.search(r"[a-zA-Z]+[-_]*\d{2,}", u_lc))

        def _is_listing_or_guide(u: str) -> bool:
            u_lc = (u or "").lower()
            tokens = [
                "/blog", "/poradnik", "/poradniki", "/ranking", "/review", "/reviews",
                "/news", "/article", "/artykul", "/artykuly", "/listing", "/category",
                "/categories", "/c/", "/g-", "/search", "listing?", "category?", "?q=",
            ]
            if any(tok in u_lc for tok in tokens):
                return True
            if "youtube.com" in u_lc or "youtu.be" in u_lc:
                return True
            return False

        seen: set[str] = set()
        product_urls: list[str] = []
        product_like_urls: list[str] = []
        other_urls: list[str] = []
        for url in combined:
            if url and self._is_grounding_redirect(url):
                continue
            if url and url not in seen:
                seen.add(url)
                if _is_listing_or_guide(url):
                    other_urls.append(url)
                elif _is_product_like(url):
                    product_like_urls.append(url)
                else:
                    product_urls.append(url)
            if len(product_urls) + len(other_urls) >= max_sources:
                break
        deduped = (product_like_urls + product_urls + other_urls)[:max_sources]
        if deduped:
            self._logger.info("Market sources top-5 URLs: %s", " | ".join(deduped[:5]))

        # Build knowledge records by fetching page text (HTTP-only) and letting LLM extract facts.
        from app.services.product_extractor import (
            extract_product_summary_fast,
            extract_product_summary_with_playwright,
            extract_product_text_and_structured_fast,
            extract_product_text_and_structured_with_playwright,
        )

        semaphore = asyncio.Semaphore(5)
        fallback_lock = asyncio.Lock()
        from app.config import get_settings
        cfg = get_settings()
        fallback_used = 0
        fallback_limit = getattr(cfg, "research_playwright_fallback_limit", 2)
        fallback_timeout_ms = getattr(cfg, "research_playwright_timeout_ms", 15000)

        async def build_record(url: str) -> dict[str, str] | None:
            nonlocal fallback_used
            async with semaphore:
                text, structured = await extract_product_text_and_structured_fast(url)
                if not text and not any(structured.values()):
                    async with fallback_lock:
                        if fallback_used >= fallback_limit:
                            return None
                        fallback_used += 1
                    text, structured = await extract_product_text_and_structured_with_playwright(
                        url,
                        timeout_ms=fallback_timeout_ms,
                    )
                if not text and not any(structured.values()):
                    return None
            return {"url": url, "text": text, "structured": structured}

        tasks = [build_record(u) for u in deduped[:max_sources]]
        records_raw = await asyncio.gather(*tasks)
        def _summary_confidence(summary: str) -> str:
            if not summary:
                return "none"
            import re
            tokens = summary.lower().split()
            has_digit = any(ch.isdigit() for ch in summary)
            has_currency = any(c in summary for c in ["PLN", "USD", "EUR", "$", "€", "zł", "zl"])
            base = "low"
            if len(tokens) >= 8 and (has_digit or has_currency):
                base = "high"
            elif len(tokens) >= 4 and (has_digit or has_currency):
                base = "medium"
            elif len(tokens) >= 6:
                base = "medium"

            s_lc = summary.lower()
            has_model_like = bool(re.search(r"[a-zA-Z]+\d+|\d+[a-zA-Z]+", summary))
            is_ranking = any(t in s_lc for t in ["ranking", "rank", "best", "top", "najleps", "polecam"])
            is_comparison = any(t in s_lc for t in ["porównanie", "porownanie", "compare", "comparison"])
            is_price_list = any(t in s_lc for t in ["porównanie cen", "porownanie cen", "comparison of prices"])
            if is_ranking and not has_model_like:
                if base == "high":
                    base = "medium"
                elif base == "medium":
                    base = "low"
            if (is_comparison or is_price_list) and not has_model_like and not (has_digit or has_currency):
                base = "low"
            return base

        texts_for_llm = [r for r in records_raw if r and (r.get("text") or r.get("structured"))]
        summaries_by_url: dict[str, str] = {}
        if texts_for_llm:
            summaries_by_url = await self._summarize_product_texts_llm(
                items=texts_for_llm,
                language=language,
            )

        # Keep all URLs; summary may be empty for some.
        records: list[dict[str, str]] = []
        low_conf_urls: list[str] = []
        for url in deduped[:max_sources]:
            summary = summaries_by_url.get(url, "")
            if not summary:
                summary = await extract_product_summary_fast(url, language.value)
                if not summary:
                    summary = await extract_product_summary_with_playwright(
                        url,
                        language.value,
                        timeout_ms=fallback_timeout_ms,
                    )
            conf = _summary_confidence(summary)
            if conf in {"low", "none"}:
                low_conf_urls.append(url)
            summary = self._append_conf_tag(summary or "", conf)
            records.append({"url": url, "summary": summary or ""})

        if records:
            self._set_cached_sources(cache_key, records)
        if low_conf_urls:
            self._logger.info("Market sources: low-confidence summaries for %s URLs", len(low_conf_urls))
        self._logger.info(
            "Market sources | model=%s | language=%s | prompts=%s | sources=%s | records=%s",
            self.research_model_name,
            language.value,
            len(prompts),
            len(deduped),
            len(records),
        )
        return records

    async def _summarize_product_text_llm(
        self,
        *,
        text: str,
        url: str,
        language: Language,
    ) -> str:
        from google.genai import types
        import asyncio
        import json as _json
        import re as _re

        if not text:
            return ""

        max_chars = 6000
        clipped = text[:max_chars]

        if language == Language.PL:
            prompt = (
                "Masz tekst ze strony produktu. Wyodrębnij TYLKO fakty zawarte w tekście. "
                "Nie zgaduj i nie dopisuj. Zwróć WYŁĄCZNIE JSON o schemacie:\n"
                '{"name":"","brand":"","price":"","currency":"","facts":[]}\n'
                "Zasady:\n"
                "- jeśli brak danych, wstaw pusty string lub pustą listę\n"
                "- NIE traktuj kosztu dostawy/wysyłki jako ceny produktu\n"
                "- facts: max 4 krótkie fakty (pojemność, materiał, wymiary itp.)\n"
                "- żadnych komentarzy, żadnego markdown\n\n"
                f"URL: {url}\n"
                f"TEKST:\n{clipped}"
            )
        else:
            prompt = (
                "You are given product page text. Extract ONLY facts present in the text. "
                "Do not guess or add info. Return ONLY JSON with schema:\n"
                '{"name":"","brand":"","price":"","currency":"","facts":[]}\n'
                "Rules:\n"
                "- if missing, use empty string or empty list\n"
                "- do NOT treat shipping/delivery fees as product price\n"
                "- facts: up to 4 short facts (capacity, material, dimensions, etc.)\n"
                "- no commentary, no markdown\n\n"
                f"URL: {url}\n"
                f"TEXT:\n{clipped}"
            )

        config = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=400,
        )

        loop = asyncio.get_event_loop()
        from app.config import get_settings
        cfg = get_settings()
        model_name = getattr(cfg, "research_interpretation_model", self.research_model_name)

        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            ),
        )
        raw = (response.text or "").strip()
        if not raw:
            return ""

        if "```" in raw:
            raw = _re.sub(r"```(?:json)?", "", raw, flags=_re.IGNORECASE).strip()

        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return ""

        try:
            data = _json.loads(raw[start:end + 1])
        except Exception:
            return ""

        name = str(data.get("name") or "").strip()
        brand = str(data.get("brand") or "").strip()
        price = str(data.get("price") or "").strip()
        currency = str(data.get("currency") or "").strip()
        facts = data.get("facts") or []
        if not isinstance(facts, list):
            facts = []
        facts = [str(f).strip() for f in facts if str(f).strip()]
        facts = self._filter_shipping_facts(facts)
        if self._is_shipping_text(price):
            price = ""

        if not any([name, brand, price, facts]):
            return ""

        if language == Language.PL:
            parts = []
            if name and brand:
                parts.append(f"{name} marki {brand}")
            elif name:
                parts.append(name)
            if price:
                parts.append(f"cena {price} {currency}".strip())
            desc = ", ".join(parts) if parts else ""
            if facts:
                facts_text = ". ".join(facts[:2])
                desc = f"{desc}. {facts_text}" if desc else facts_text
            return desc.strip()

    def _format_summary_from_fields(
        self,
        *,
        name: str,
        brand: str,
        price: str,
        currency: str,
        facts: list[str],
        language: Language,
    ) -> str:
        if language == Language.PL:
            parts = []
            if name and brand:
                parts.append(f"{name} marki {brand}")
            elif name:
                parts.append(name)
            elif brand:
                parts.append(brand)
            if price:
                if currency:
                    parts.append(f"cena {price} {currency}".strip())
                else:
                    parts.append(f"cena {price}".strip())
            desc = ", ".join(parts) if parts else ""
            if facts:
                facts_text = ". ".join(facts[:2])
                desc = f"{desc}. {facts_text}" if desc else facts_text
            return desc.strip()
        else:
            parts = []
            if name and brand:
                parts.append(f"{name} by {brand}")
            elif name:
                parts.append(name)
            elif brand:
                parts.append(brand)
            if price:
                if currency:
                    parts.append(f"price {price} {currency}".strip())
                else:
                    parts.append(f"price {price}".strip())
            desc = ", ".join(parts) if parts else ""
            if facts:
                facts_text = ". ".join(facts[:2])
                desc = f"{desc}. {facts_text}" if desc else facts_text
            return desc.strip()

    def _append_conf_tag(self, summary: str, confidence: str) -> str:
        if not summary:
            return summary
        conf = (confidence or "").strip().lower()
        if not conf:
            return summary
        return f"{summary} [conf:{conf}]"

    def _is_shipping_text(self, text: str) -> bool:
        if not text:
            return False
        t = text.lower()
        keywords = [
            "shipping", "delivery", "postage", "courier", "ship", "freight", "handling",
            "dispatch", "versand", "liefer", "porto", "spedizione", "consegna", "envio",
            "entrega", "livraison", "expedition", "dostaw", "wysyl", "przesyl", "kurier",
            "transport",
        ]
        return any(k in t for k in keywords)

    def _filter_shipping_facts(self, facts: list[str]) -> list[str]:
        return [f for f in facts if not self._is_shipping_text(f)]

    def _strip_conf_tag(self, summary: str) -> str:
        if not summary:
            return summary
        import re
        return re.sub(r"\s*\[conf:(low|medium|high|none)\]\s*$", "", summary, flags=re.IGNORECASE).strip()

    async def _summarize_product_texts_llm(
        self,
        *,
        items: list[dict[str, str]],
        language: Language,
    ) -> dict[str, str]:
        from google.genai import types
        import asyncio
        import json as _json
        import re as _re

        if not items:
            return {}

        def chunk_items() -> list[list[dict[str, str]]]:
            max_chars = 12000
            max_items = 6
            chunks: list[list[dict[str, str]]] = []
            current: list[dict[str, str]] = []
            current_chars = 0
            for item in items:
                text = item.get("text") or ""
                clipped = text[:6000]
                structured = item.get("structured") or {}
                entry_size = len(clipped)
                if current and (current_chars + entry_size > max_chars or len(current) >= max_items):
                    chunks.append(current)
                    current = []
                    current_chars = 0
                current.append({"url": item.get("url", ""), "text": clipped, "structured": structured})
                current_chars += entry_size
            if current:
                chunks.append(current)
            return chunks

        async def summarize_chunk(chunk: list[dict[str, str]]) -> dict[str, str]:
            if language == Language.PL:
                prompt = (
                    "Masz teksty ze stron produktów. Wyodrębnij TYLKO fakty zawarte w tekście. "
                    "Nie zgaduj i nie dopisuj. Zwróć WYŁĄCZNIE JSON array o schemacie:\n"
                    '[{"url":"","name":"","brand":"","price":"","currency":"","facts":[]}]\\n'
                    "Zasady:\n"
                    "- jeśli brak danych, wstaw pusty string lub pustą listę\n"
                    "- NIE traktuj kosztu dostawy/wysyłki jako ceny produktu\n"
                    "- price/currency: wpisz dokładnie jak w tekście; jeśli wiele cen, wybierz najniższą\n"
                    "- facts: max 4 krótkie fakty (pojemność, materiał, wymiary itp.)\n"
                    "- jeśli w polu structured jest cena produktu, możesz ją użyć jako preferowane źródło\n"
                    "- żadnych komentarzy, żadnego markdown\n\n"
                    f"DANE:\n{_json.dumps(chunk, ensure_ascii=True)}"
                )
            else:
                prompt = (
                    "You are given product page texts. Extract ONLY facts present in the text. "
                    "Do not guess or add info. Return ONLY a JSON array with schema:\n"
                    '[{"url":"","name":"","brand":"","price":"","currency":"","facts":[]}]\\n'
                    "Rules:\n"
                    "- if missing, use empty string or empty list\n"
                    "- do NOT treat shipping/delivery fees as product price\n"
                    "- price/currency: copy exactly from text; if multiple prices, choose the lowest\n"
                    "- facts: up to 4 short facts (capacity, material, dimensions, etc.)\n"
                    "- if structured contains product price, you may use it as preferred source\n"
                    "- no commentary, no markdown\n\n"
                    f"DATA:\n{_json.dumps(chunk, ensure_ascii=True)}"
                )

            config = types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=800,
            )

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.research_model_name,
                    contents=prompt,
                    config=config,
                ),
            )
            raw = (response.text or "").strip()
            if not raw:
                return {}
            if "```" in raw:
                raw = _re.sub(r"```(?:json)?", "", raw, flags=_re.IGNORECASE).strip()
            start = raw.find("[")
            end = raw.rfind("]")
            if start == -1 or end == -1 or end <= start:
                return {}
            try:
                data = _json.loads(raw[start:end + 1])
            except Exception:
                return {}
            if not isinstance(data, list):
                return {}

            out: dict[str, str] = {}
            for row in data:
                if not isinstance(row, dict):
                    continue
                url = str(row.get("url") or "").strip()
                if not url:
                    continue
                name = str(row.get("name") or "").strip()
                brand = str(row.get("brand") or "").strip()
                price = str(row.get("price") or "").strip()
                currency = str(row.get("currency") or "").strip()
                facts = row.get("facts") or []
                if not isinstance(facts, list):
                    facts = []
                facts = [str(f).strip() for f in facts if str(f).strip()]
                facts = self._filter_shipping_facts(facts)
                if self._is_shipping_text(price):
                    price = ""

                # Prefer structured price if LLM output has no valid price.
                structured = {}
                for item in chunk:
                    if item.get("url") == url:
                        structured = item.get("structured") or {}
                        break
                if structured:
                    s_price = str(structured.get("price") or "").strip()
                    s_currency = str(structured.get("currency") or "").strip()
                    if not price and s_price and not self._is_shipping_text(s_price):
                        price = s_price
                        currency = s_currency or currency
                    if not name:
                        name = str(structured.get("name") or "").strip()
                    if not brand:
                        brand = str(structured.get("brand") or "").strip()
                summary = self._format_summary_from_fields(
                    name=name,
                    brand=brand,
                    price=price,
                    currency=currency,
                    facts=facts,
                    language=language,
                )
                out[url] = summary
            return out

        results = await asyncio.gather(*(summarize_chunk(c) for c in chunk_items()))
        merged: dict[str, str] = {}
        for result in results:
            merged.update(result)
        return merged

    async def generate_opinion_with_sources(
        self,
        persona: Persona,
        product_description: str,
        sources: list[dict[str, str]],
        language: Language = Language.PL,
        temperature: float | None = None,
    ) -> tuple[str, list[str]]:
        from google.genai import types
        import asyncio

        selected_sources = self._select_sources_for_persona(sources, persona, language)
        sources_block = "\n".join(
            f"[{i}] {self._strip_conf_tag(s.get('summary') or '')} (Źródło: {s.get('url')})"
            for i, s in enumerate(selected_sources[:6], 1)
            if s.get("summary") and s.get("url")
        )
        no_sources_note_pl = (
            "UWAGA: Nie znaleziono wiarygodnych źródeł. "
            "Wyraźnie zaznacz niepewność, nie podawaj nazw i cen konkurencji. "
            "Oceń produkt głównie przez pryzmat dopasowania do persony (dochód, potrzeby, styl życia)."
        )
        no_sources_note_en = (
            "NOTE: No reliable sources were found. "
            "Clearly state uncertainty, do not provide specific competitor names or prices. "
            "Base the evaluation mainly on persona fit (income, needs, lifestyle)."
        )
        if language == Language.PL:
            prompt = f"""Jesteś {persona.name}, masz {persona.age} lat, płeć: {persona.gender}.
Mieszkasz w {persona.location}, zarabiasz {persona.income:,} PLN miesięcznie.
{f'Pracujesz jako {persona.occupation}.' if persona.occupation else ''}

PRODUKT DO OCENY: {product_description}

ŹRÓDŁA DO WYKORZYSTANIA (wybierz 1-3 różne domeny):
{sources_block}
{no_sources_note_pl if not selected_sources else ""}

PORÓWNANIE (uwzględnij w odpowiedzi):
- Podaj 1-2 konkretne produkty konkurencji z cenami
- Jeśli produkty różnią się cechami, zaznacz to
- Oceń czy cena analizowanego produktu jest uczciwa

TWOJA ODPOWIEDŹ (jako konsument):
1. Jak ten produkt wypada vs konkurencja (lub zaznacz że jest unikalny)?
2. Czy cena jest atrakcyjna biorąc pod uwagę Twoje zarobki ({persona.income:,} PLN)?
3. Czy kupiłbyś/kupiłabyś? Dlaczego?

Odpowiedz naturalnie, jak w rozmowie. Nie wymyślaj faktów poza podanymi źródłami.
Jeśli podajesz ceny lub nazwy konkretnych produktów, dodaj cytowanie źródła w formacie [1], [2] itd."""
        else:
            prompt = f"""You are {persona.name}, {persona.age} years old, {persona.gender}.
Living in {persona.location}, earning ${persona.income:,} monthly.
{f'Working as {persona.occupation}.' if persona.occupation else ''}

PRODUCT TO EVALUATE: {product_description}

SOURCES TO USE (pick 1-3 different domains):
{sources_block}
{no_sources_note_en if not selected_sources else ""}

COMPARISON (include in your response):
- Name 1-2 competing products with prices
- Note differences in features if relevant
- Assess if the product's price is fair

YOUR RESPONSE (as a consumer):
1. How does this product compare to competition (or note if it's unique)?
2. Is the price attractive given your income (${persona.income:,})?
3. Would you buy it? Why?

Answer naturally. Do not invent facts beyond the provided sources.
If you mention prices or specific competitor products, cite sources like [1], [2]."""

        temp = self.temperature if temperature is None else temperature
        config = types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=2048,
        )
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            ),
        )
        text = response.text or ""
        if not self._response_has_required_citations(text, len(selected_sources)):
            if language == Language.PL:
                retry_note = (
                    "UWAGA: Jeśli nie potrafisz wskazać źródeł, "
                    "nie podawaj nazw i cen konkurencji. "
                    "Wtedy napisz, że brak potwierdzonych danych w źródłach."
                )
            else:
                retry_note = (
                    "NOTE: If you cannot cite sources, do not provide specific "
                    "competitor names or prices. State that sources don't confirm them."
                )
            retry_prompt = f"{prompt}\n\n{retry_note}"
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=retry_prompt,
                    config=config,
                ),
            )
            text = response.text or ""
        selected_urls = [s.get("url") for s in selected_sources if s.get("url")]
        return text, selected_urls

    async def generate_opinion_with_search(
        self,
        persona: Persona,
        product_description: str,
        language: Language = Language.PL,
        temperature: float | None = None,
    ) -> tuple[str, list[str]]:
        """Generate opinion with Google Search grounding.
        
        The agent will search the web for market information before forming opinion.
        
        Returns:
            tuple of (opinion_text, list_of_source_urls)
        """
        from google.genai import types
        import asyncio
        
        # Enhanced prompt with proper comparison instructions
        if language == Language.PL:
            prompt = f"""Jesteś {persona.name}, masz {persona.age} lat, płeć: {persona.gender}.
Mieszkasz w {persona.location}, zarabiasz {persona.income:,} PLN miesięcznie.
{f'Pracujesz jako {persona.occupation}.' if persona.occupation else ''}

PRODUKT DO OCENY: {product_description}

INSTRUKCJE WYSZUKIWANIA:
1. Wyszukaj PODOBNE produkty (ta sama kategoria, zbliżone parametry)
2. Zwróć uwagę na AKTUALNE ceny (2024-2025)
3. Jeśli nie ma identycznego produktu, znajdź najbliższy odpowiednik

PORÓWNANIE (uwzględnij w odpowiedzi):
- Podaj 1-2 konkretne produkty konkurencji z cenami (np. "Model X kosztuje 120 zł")
- Jeśli produkty różnią się cechami (pojemność, marka, funkcje), zaznacz to
- Oceń czy cena analizowanego produktu jest uczciwa biorąc pod uwagę różnice

JEŚLI NIE MA ANALOGICZNYCH PRODUKTÓW:
- Zaznacz że produkt jest UNIKALNY na rynku
- Oceń DLACZEGO nie ma konkurencji:
  * Czy to INNOWACJA (nowa wartość, rozwiązuje problem)?
  * Czy to NISZA (mały rynek, ale realny)?
  * Czy to produkt BEZSENSOWNY (nikt tego nie potrzebuje)?
- Rozważ czy cena jest uzasadniona

TWOJA ODPOWIEDŹ (jako konsument):
1. Jak ten produkt wypada vs konkurencja (lub zaznacz że jest unikalny)?
2. Czy cena jest atrakcyjna biorąc pod uwagę Twoje zarobki ({persona.income:,} PLN)?
3. Czy kupiłbyś/kupiłabyś? Dlaczego?

Odpowiedz naturalnie, jak w rozmowie. Wspomnij konkretne znalezione produkty i ceny."""
        else:
            prompt = f"""You are {persona.name}, {persona.age} years old, {persona.gender}.
Living in {persona.location}, earning ${persona.income:,} monthly.
{f'Working as {persona.occupation}.' if persona.occupation else ''}

PRODUCT TO EVALUATE: {product_description}

SEARCH INSTRUCTIONS:
1. Search for SIMILAR products (same category, similar specs)
2. Focus on CURRENT prices (2024-2025)
3. If no identical product exists, find closest equivalent

COMPARISON (include in your response):
- Name 1-2 specific competing products with prices (e.g., "Model X costs $120")
- If products differ in features (capacity, brand, functions), note it
- Assess if product's price is fair considering the differences

IF NO ANALOGOUS PRODUCTS EXIST:
- Note that product is UNIQUE on the market
- Assess WHY there's no competition:
  * Is it INNOVATIVE (new value, solves a problem)?
  * Is it a NICHE (small market, but real)?
  * Is it POINTLESS (nobody needs this)?
- Consider if the price is justified

YOUR RESPONSE (as a consumer):
1. How does this product compare to competition (or note if it's unique)?
2. Is the price attractive given your income (${persona.income:,})?
3. Would you buy it? Why?

Answer naturally, as in conversation. Mention specific products and prices you found."""

        # Enable Google Search grounding
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        
        temp = self.temperature if temperature is None else temperature
        config = types.GenerateContentConfig(
            tools=[grounding_tool],
            temperature=temp,
            max_output_tokens=2048,
        )
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
        )
        
        # Extract source URLs from grounding metadata
        sources = []
        metadata = None
        if response.candidates:
            candidate = response.candidates[0]
            metadata = getattr(candidate, "grounding_metadata", None) or getattr(candidate, "groundingMetadata", None)
        if not metadata:
            metadata = getattr(response, "grounding_metadata", None) or getattr(response, "groundingMetadata", None)
        if metadata:
            sources = self._extract_grounding_urls(metadata)

        sources = await self._resolve_grounding_urls(sources)
        self._log_grounding_status("per-agent-search", response, sources)
        return response.text or "", sources

    async def extract_product_from_url(
        self,
        url: str,
        language: Language = Language.PL,
    ) -> str:
        """Extract product description from a shop URL using hybrid fetch + Gemini fallback.
        
        Args:
            url: URL of the product page
            language: Output language
            
        Returns:
            Extracted product description
        """
        import asyncio
        from google.genai import types
        from app.services.product_extractor import extract_product_description
        
        # Try direct extraction first (HTTP + optional Playwright)
        extracted = await extract_product_description(url, language.value)
        if extracted:
            return extracted

        if language == Language.PL:
            prompt = f"""Przeanalizuj stronę produktu: {url}

Zwróć TYLKO opis produktu w formacie:
"[Nazwa produktu] marki [marka], cena [cena]. [Kluczowe cechy w 1-2 zdaniach]."

WAŻNE:
- NIE dodawaj żadnych komentarzy, pytań ani wyjaśnień
- NIE pisz "Name included?" ani podobnych uwag
- Odpowiedź ma zawierać WYŁĄCZNIE opis produktu
- Maksymalnie 2-3 zdania"""
        else:
            prompt = f"""Analyze product page: {url}

Return ONLY the product description in format:
"[Product name] by [brand], price [price]. [Key features in 1-2 sentences]."

IMPORTANT:
- Do NOT add any comments, questions or explanations
- Do NOT write "Name included?" or similar notes
- Response must contain ONLY the product description
- Maximum 2-3 sentences"""

        # Enable URL Context tool
        url_context_tool = types.Tool(url_context=types.UrlContext())
        
        config = types.GenerateContentConfig(
            tools=[url_context_tool],
            temperature=0.1,  # Very low temp for factual extraction
            max_output_tokens=300,
        )
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
        )
        
        text = response.text or ""
        
        # Post-process to remove thinking artifacts
        text = self._clean_extraction_artifacts(text)
        
        return text

    def _clean_extraction_artifacts(self, text: str) -> str:
        """Remove internal thinking artifacts from Gemini response."""
        import re
        
        # Remove lines with common thinking patterns
        patterns_to_remove = [
            r'\*.*included\?.*\*',  # *Name included? Yes*
            r'\*.*Format.*\*',
            r'\*.*Self-Correction.*\*',
            r'\*.*Let me.*\*',
            r'\*.*Wait.*\*',
            r'\*.*double-check.*\*',
            r'\*.*browsing.*\*',
            r'\*Only description\?.*\*',
            r'\*No comments\?.*\*',
            r'\*Max.*sentences\?.*\*',
            r'\*.*Polish Version.*\*',
            r'\*.*Final.*Version.*\*',
            r'^\s*\*\s*\*.*$',  # Lines with just asterisks
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove multiple spaces and newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Get first coherent paragraph (the actual description)
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        clean_lines = []
        for line in lines:
            # Skip lines that look like internal notes
            if any(x in line.lower() for x in ['included?', 'format?', 'let me', 'self-correction', 'double-check']):
                continue
            clean_lines.append(line)
        
        return ' '.join(clean_lines).strip()


class OpenAIClient:
    """OpenAI GPT client (alternative)."""

    def __init__(self, api_key: str, model: str = "gpt-4o", temperature: float | None = None):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature if temperature is not None else get_settings().llm_temperature

    async def generate_opinion(
        self, 
        persona: Persona, 
        product_description: str,
        language: Language = Language.PL,
        temperature: float | None = None,
    ) -> str:
        """Generate opinion using OpenAI."""
        # Get language-specific prompt from i18n module
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

        temp = self.temperature if temperature is None else temperature
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=2048,
        )

        return response.choices[0].message.content or ""


def get_llm_client(model_override: str | None = None) -> LLMClient:
    """Factory function to get the configured LLM client."""
    settings = get_settings()
    model = model_override or settings.llm_model

    if model.startswith("gemini"):
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY required for Gemini models")
        return GeminiClient(
            api_key=settings.google_api_key,
            model=model,
            temperature=settings.llm_temperature,
        )
    elif model.startswith("gpt"):
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI models")
        return OpenAIClient(
            api_key=settings.openai_api_key,
            model=model,
            temperature=settings.llm_temperature,
        )
    else:
        raise ValueError(f"Unknown model: {model}")
