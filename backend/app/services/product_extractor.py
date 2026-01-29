"""Product page extractor with HTTP + optional Playwright fallback."""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from app.config import get_settings

logger = logging.getLogger(__name__)


def _is_grounding_redirect(url: str) -> bool:
    return "vertexaisearch.cloud.google.com/grounding-api-redirect" in (url or "")


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def _tokenize_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9ąćęłńóśżź]+", (text or "").lower())
    stop = {
        "the","and","for","with","this","that","from","your","you","all","one","two",
        "of","to","in","on","a","an","or","is","are","by","as","at","it","be","do",
        "oraz","aby","dla","jako","jest","czy","tak","nie","się","oraz","oraz","bez",
        "z","za","do","na","w","od","po","or","and","the","with","dla","oraz",
        "produkt","produkty","item","model","new","nowa","nowy",
    }
    keywords = [t for t in tokens if len(t) >= 3 and t not in stop]
    # Keep top unique tokens
    seen = set()
    out = []
    for t in keywords:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out[:12]


def _extract_url_keywords(url: str) -> list[str]:
    try:
        slug = urlparse(url).path
    except Exception:
        slug = ""
    slug = slug.replace("-", " ")
    return _tokenize_keywords(slug)


def _name_variants(name: str, brand: str | None) -> list[str]:
    variants = [name]
    for sep in [" - ", " | ", " / "]:
        if sep in name:
            left = name.rsplit(sep, 1)[0]
            variants.append(left)
        if brand:
            variants.append(name.replace(f"{sep}{brand}", ""))
    if brand:
        variants.append(name.replace(brand, ""))
    cleaned: list[str] = []
    seen = set()
    for v in variants:
        v = _clean_text(v).strip(" -|/")
        if not v:
            continue
        if v.lower() in seen:
            continue
        seen.add(v.lower())
        cleaned.append(v)
    return cleaned


def _score_sentence(sentence: str, keywords: set[str]) -> int:
    score = 0
    s_lc = sentence.lower()
    if re.search(r"\d", sentence):
        score += 1
    if re.search(r"\b(cm|mm|m|l|ml|g|kg|oz|in|inch|lit|w|watt|mah|db)\b", s_lc):
        score += 1
    if keywords and any(k in s_lc for k in keywords):
        score += 1
    if len(sentence.split()) >= 8:
        score += 1
    return score


def _strip_marketing_noise(text: str, keywords: set[str] | None = None) -> str:
    keywords = keywords or set()
    text = text.replace("•", ". ").replace(" | ", ". ")
    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned: list[str] = []
    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        s_lc = s.lower()
        if ":" in s:
            prefix, rest = s.split(":", 1)
            if len(prefix.split()) <= 2 and not re.search(r"\d", prefix):
                s = rest.strip()
                s_lc = s.lower()
        promo = False
        if "!" in s and not re.search(r"\d", s):
            promo = True
        if re.search(r"[\U0001F300-\U0001FAFF]", s):
            promo = True
        score = _score_sentence(s, keywords)
        if promo and score < 2:
            continue
        if len(s.split()) < 6 and score < 2:
            continue
        if s:
            cleaned.append(s)
    return " ".join(cleaned).strip()


def _clean_features_text(text: str, name: str | None, url: str) -> str:
    text = _clean_text(text)
    if not text:
        return ""
    keywords = set(_tokenize_keywords(name or "") + _extract_url_keywords(url))

    boilerplate = [
        "czytaj więcej",
        "pokaż mniej",
        "pokaż więcej",
        "zobacz więcej",
        "dodaj do koszyka",
        "add to cart",
    ]
    text_lc = text.lower()
    for phrase in boilerplate:
        idx = text_lc.find(phrase)
        if idx != -1:
            text = text[:idx].strip()
            text_lc = text.lower()

    if name:
        for variant in _name_variants(name, None):
            if not variant:
                continue
            words = [re.escape(w) for w in variant.split()]
            if not words:
                continue
            # Match words separated by optional whitespace and dashes.
            pattern = r"\s*[-–—]?\s*".join(words)
            text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    # Hard cap for short blurbs.
    if len(text.split()) <= 30 and len(text) <= 200:
        return _strip_marketing_noise(text, keywords) or text

    parts = re.split(r"(?<=[.!?])\s+", text)
    if len(parts) >= 2:
        trimmed = " ".join(parts[:2]).strip()
    else:
        words = text.split()
        trimmed = " ".join(words[:80]).strip()
        if len(words) > 60:
            trimmed = " ".join(words[:50]).strip()
    if len(trimmed) > 420:
        trimmed = trimmed[:420].rsplit(" ", 1)[0].strip()
    # If there is no sentence punctuation, keep it short.
    if not re.search(r"[.!?]", trimmed) and len(trimmed.split()) > 45:
        trimmed = " ".join(trimmed.split()[:45]).strip()
    # Final word cap to avoid long lists.
    if len(trimmed.split()) > 45:
        trimmed = " ".join(trimmed.split()[:45]).strip()
    trimmed = _strip_marketing_noise(trimmed, keywords) or trimmed
    return trimmed


def _truncate_sentences(text: str, max_sentences: int = 2) -> str:
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(parts[:max_sentences]).strip()


def _dedupe_lines(lines: list[str]) -> list[str]:
    seen = set()
    out = []
    for line in lines:
        norm = line.lower()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(line)
    return out


def _collect_description_blocks(soup: BeautifulSoup) -> str:
    selectors = [
        "[itemprop='description']",
        ".product-description",
        ".product__description",
        ".woocommerce-Tabs-panel--description",
        "#description",
        ".description",
        ".product-desc",
        ".product-details__description",
    ]
    blocks: list[str] = []
    for sel in selectors:
        for el in soup.select(sel):
            text = _clean_text(el.get_text(" ", strip=True))
            if text and len(text.split()) >= 5:
                blocks.append(text)

    blocks = _dedupe_lines(blocks)
    return " ".join(blocks).strip()


def _strip_noise_sections(soup: BeautifulSoup) -> None:
    noise_tokens = [
        "related", "recommend", "upsell", "cross-sell", "crosssell", "similar",
        "you-may-like", "youmaylike", "also-bought", "suggest", "carousel", "slider", "swiper",
        "footer", "header", "nav", "menu", "breadcrumb", "social", "share", "newsletter",
        "subscribe", "popup", "modal", "cookie", "consent", "banner", "reviews", "review",
        "rating", "comment", "comments", "opinia", "opinie",
    ]
    for tag in soup.find_all(True):
        if tag.name in {"html", "body"}:
            continue
        try:
            class_attr = " ".join(tag.get("class", []))
            id_attr = tag.get("id") or ""
        except Exception:
            continue
        hay = f"{class_attr} {id_attr}".lower()
        if any(tok in hay for tok in noise_tokens):
            tag.decompose()


def _link_density(tag) -> float:
    try:
        text = _clean_text(tag.get_text(" ", strip=True))
        if not text:
            return 0.0
        link_text = ""
        for a in tag.find_all("a"):
            link_text += " " + _clean_text(a.get_text(" ", strip=True))
        return min(1.0, len(link_text) / max(len(text), 1))
    except Exception:
        return 0.0


def _score_block(tag) -> float:
    text = _clean_text(tag.get_text(" ", strip=True))
    if not text:
        return 0.0
    length = len(text)
    if length < 80:
        return 0.0
    ld = _link_density(tag)
    score = length * (1.0 - ld)
    if tag.name in {"article", "main"}:
        score *= 1.2
    if tag.find(["h1", "h2", "h3"]):
        score *= 1.05
    return score


def _select_main_content(soup: BeautifulSoup) -> str:
    candidates = soup.find_all(["article", "main", "section", "div"])
    best = None
    best_score = 0.0
    for tag in candidates:
        score = _score_block(tag)
        if score > best_score:
            best_score = score
            best = tag
    if not best:
        return ""
    parts: list[str] = []
    for el in best.find_all(["h1", "h2", "h3", "p", "li"]):
        text = _clean_text(el.get_text(" ", strip=True))
        if text and len(text.split()) >= 4:
            parts.append(text)
    parts = _dedupe_lines(parts)
    return " ".join(parts).strip()


def _collect_specs(soup: BeautifulSoup) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []

    # Table-based specs
    for table in soup.find_all("table"):
        class_attr = " ".join(table.get("class", []))
        if class_attr and not any(x in class_attr.lower() for x in ["spec", "param", "product", "data", "attributes"]):
            continue
        for row in table.find_all("tr"):
            cells = row.find_all(["th", "td"])
            if len(cells) < 2:
                continue
            key = _clean_text(cells[0].get_text(" ", strip=True))
            value = _clean_text(cells[-1].get_text(" ", strip=True))
            if key and value and key != value:
                specs.append((key, value))

    # Definition list specs
    for dl in soup.find_all("dl"):
        class_attr = " ".join(dl.get("class", []))
        if class_attr and not any(x in class_attr.lower() for x in ["spec", "param", "product", "data", "attributes"]):
            continue
        dts = dl.find_all("dt")
        dds = dl.find_all("dd")
        for dt, dd in zip(dts, dds):
            key = _clean_text(dt.get_text(" ", strip=True))
            value = _clean_text(dd.get_text(" ", strip=True))
            if key and value and key != value:
                specs.append((key, value))

    # List-based specs (e.g., features lists)
    for ul in soup.find_all(["ul", "ol"]):
        class_attr = " ".join(ul.get("class", []))
        if class_attr and not any(x in class_attr.lower() for x in ["spec", "param", "feature", "attributes"]):
            continue
        items = [ _clean_text(li.get_text(" ", strip=True)) for li in ul.find_all("li") ]
        for item in items:
            if not item:
                continue
            if ":" in item:
                parts = item.split(":", 1)
                key = _clean_text(parts[0])
                value = _clean_text(parts[1])
                if key and value:
                    specs.append((key, value))

    # Dedupe specs
    seen = set()
    deduped = []
    for key, value in specs:
        norm = f"{key.lower()}::{value.lower()}"
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append((key, value))
    return deduped


def _extract_json_ld(html: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    soup = BeautifulSoup(html, "lxml")
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or ""
        raw = raw.strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            blocks.extend([item for item in data if isinstance(item, dict)])
        elif isinstance(data, dict):
            blocks.append(data)
    return blocks


def _find_product_ld(blocks: list[dict[str, Any]]) -> dict[str, Any] | None:
    for block in blocks:
        if str(block.get("@type", "")).lower() == "product":
            return block
        # Sometimes @type is a list or nested
        block_type = block.get("@type")
        if isinstance(block_type, list) and any(str(t).lower() == "product" for t in block_type):
            return block
    return None


def _extract_price_from_ld(product: dict[str, Any]) -> tuple[str | None, str | None]:
    offers = product.get("offers")
    if isinstance(offers, list) and offers:
        offers = offers[0]
    if isinstance(offers, dict):
        price = offers.get("price")
        currency = offers.get("priceCurrency")
        if price is not None:
            return str(price), str(currency or "")
    return None, None


def _extract_price_from_text(text: str) -> tuple[str | None, str | None]:
    # Simple regex for price + currency
    pattern = re.compile(r"(\d{1,3}(?:[\s.,]\d{3})*(?:[.,]\d{2})?)\s*(PLN|z[lł]|EUR|USD|\$|€)", re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return None, None
    price = match.group(1).replace(" ", "")
    currency = match.group(2).upper().replace("ZŁ", "PLN").replace("ZL", "PLN")
    return price, currency


def _build_description(
    *,
    name: str | None,
    brand: str | None,
    price: str | None,
    currency: str | None,
    features: str | None,
    language: str,
) -> str:
    name = _clean_text(name or "")
    brand = _clean_text(brand or "")
    price = _clean_text(price or "")
    currency = _clean_text(currency or "")
    features = _truncate_sentences(_clean_text(features or ""), max_sentences=6)

    if language.lower().startswith("pl"):
        parts = []
        if name and brand:
            parts.append(f"{name} marki {brand}")
        elif name:
            parts.append(name)
        if price:
            parts.append(f"cena {price} {currency}".strip())
        desc = ", ".join(parts) if parts else ""
        if features:
            if desc:
                desc = f"{desc}. {features}"
            else:
                desc = features
        return desc.strip()

    # EN
    parts = []
    if name and brand:
        parts.append(f"{name} by {brand}")
    elif name:
        parts.append(name)
    if price:
        parts.append(f"price {price} {currency}".strip())
    desc = ", ".join(parts) if parts else ""
    if features:
        if desc:
            desc = f"{desc}. {features}"
        else:
            desc = features
    return desc.strip()


def _extract_from_html(html: str, url: str, language: str) -> str:
    html = html or ""
    soup = BeautifulSoup(html, "lxml")
    _strip_noise_sections(soup)
    json_ld_only = bool(get_settings().research_json_ld_only)
    # JSON-LD first
    blocks = _extract_json_ld(html)
    product = _find_product_ld(blocks)
    name = brand = description = price = currency = None

    if product:
        name = product.get("name")
        brand_val = product.get("brand")
        if isinstance(brand_val, dict):
            brand = brand_val.get("name")
        elif isinstance(brand_val, str):
            brand = brand_val
        description = product.get("description")
        price, currency = _extract_price_from_ld(product)

    specs = [] if json_ld_only else _collect_specs(soup)
    has_structured = bool(product and (name or description or price))
    desc_blocks = _collect_description_blocks(soup) if (not json_ld_only and not has_structured) else ""

    # Fallbacks
    if not description:
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            description = meta_desc.get("content")
        if not description:
            og_desc = soup.find("meta", attrs={"property": "og:description"})
            if og_desc and og_desc.get("content"):
                description = og_desc.get("content")
    if not name:
        if soup.title and soup.title.string:
            name = soup.title.string

    if not price and not json_ld_only:
        text_blob = soup.get_text(" ", strip=True)
        price, currency = _extract_price_from_text(text_blob)

    # Prefer page-specific description blocks only when structured data is missing.
    if desc_blocks and not has_structured and len(desc_blocks) > len(description or ""):
        description = desc_blocks

    keywords = set(_tokenize_keywords(name or "") + _extract_url_keywords(url))
    description = _clean_features_text(description or "", name, url)

    base_desc = _build_description(
        name=name,
        brand=brand,
        price=price,
        currency=currency,
        features=description,
        language=language,
    )
    base_desc = base_desc.strip()

    # Add specs if available
    if specs:
        allowed_spec_terms = {
            "pojemność", "objętość", "litr", "ml", "wymiary", "materiał", "szkło",
            "borosilikat", "szerokość", "wysokość", "głębokość", "waga",
            "capacity", "volume", "dimensions", "material", "glass", "height", "width", "depth", "weight",
        }
        desc_lc = (description or "").lower()
        if any(term in desc_lc for term in allowed_spec_terms):
            specs = []
        spec_items = []
        for key, value in specs[:12]:
            kv = f"{key} {value}".lower()
            if keywords and not any(k in kv for k in keywords) and not any(t in kv for t in allowed_spec_terms):
                continue
            spec_items.append(f"{key}: {value}")
        specs_text = "; ".join(spec_items)
        if specs_text:
            if language.lower().startswith("pl"):
                base_desc = f"{base_desc} Parametry: {specs_text}."
            else:
                base_desc = f"{base_desc} Specifications: {specs_text}."

    return base_desc.strip()


def _extract_main_text(html: str) -> str:
    html = html or ""
    soup = BeautifulSoup(html, "lxml")
    _strip_noise_sections(soup)
    for tag in soup.find_all(["script", "style", "noscript", "svg", "canvas"]):
        tag.decompose()

    blocks: list[str] = []
    main_text = _select_main_content(soup)
    if main_text:
        blocks.append(main_text)

    desc_text = _collect_description_blocks(soup)
    if desc_text:
        blocks.append(desc_text)

    containers = soup.select("main, article, [role='main']")
    if not containers:
        containers = [soup.body] if soup.body else [soup]

    for container in containers:
        for el in container.find_all(["h1", "h2", "h3", "p", "li"]):
            text = _clean_text(el.get_text(" ", strip=True))
            if not text or len(text.split()) < 4:
                continue
            blocks.append(text)

    blocks = _dedupe_lines(blocks)
    text = " ".join(blocks).strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text


async def _fetch_html_httpx(url: str) -> str:
    logger.info("Product extract: HTTP fetch start: %s", url)
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    timeout = httpx.Timeout(15.0, connect=10.0)
    async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=timeout) as client:
        resp = await client.get(url)
        if resp.status_code >= 400:
            logger.info("Product extract: HTTP fetch failed (%s): %s", resp.status_code, url)
            return ""
        logger.info("Product extract: HTTP fetch OK (%s): %s", resp.status_code, url)
        return resp.text


async def _try_accept_cookies(page) -> None:
    selectors = [
        "button#onetrust-accept-btn-handler",
        "button[aria-label='Accept cookies']",
        "button[aria-label='Akceptuj']",
        "button[aria-label='Akceptuj wszystkie']",
        "button:has-text('Akceptuj')",
        "button:has-text('Akceptuję')",
        "button:has-text('Akceptuj wszystkie')",
        "button:has-text('Zgadzam się')",
        "button:has-text('Zgoda')",
        "button:has-text('Accept all')",
        "button:has-text('Accept')",
        "button:has-text('I agree')",
        "button:has-text('Agree')",
        "button:has-text('OK')",
    ]

    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if await locator.is_visible(timeout=1000):
                await locator.click(timeout=1000)
                logger.info("Product extract: Cookie consent clicked via selector: %s", selector)
                return
        except Exception:
            continue


async def _fetch_html_playwright(url: str, timeout_ms: int = 30000) -> str:
    try:
        from playwright.async_api import async_playwright
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Playwright not available") from exc

    logger.info("Product extract: Playwright fetch start: %s", url)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
                locale="pl-PL",
            )
            page = await context.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            await _try_accept_cookies(page)
            await page.wait_for_load_state("networkidle", timeout=timeout_ms)
            html = await page.content()
            await context.close()
            logger.info("Product extract: Playwright fetch OK: %s", url)
            return html
        finally:
            await browser.close()


async def extract_product_description(url: str, language: str) -> str:
    """Hybrid extractor: HTTP -> Playwright (if needed)."""
    html = ""
    try:
        html = await _fetch_html_httpx(url)
    except Exception:
        logger.info("Product extract: HTTP fetch failed, falling back to Playwright: %s", url)
        html = ""

    if html:
        desc = _extract_from_html(html, url, language)
        if desc:
            logger.info("Product extract: HTTP extract OK: %s", url)
            return desc

    try:
        html = await _fetch_html_playwright(url)
    except Exception:
        logger.info("Product extract: Playwright fetch failed: %s", url)
        html = ""

    if html:
        desc = _extract_from_html(html, url, language)
        if desc:
            logger.info("Product extract: Playwright extract OK: %s", url)
            return desc

    logger.info("Product extract: No content extracted: %s", url)
    return ""


async def extract_product_summary_fast(url: str, language: str) -> str:
    """Fast extractor: HTTP only, no Playwright."""
    if _is_grounding_redirect(url):
        logger.info("Product extract fast: Skip grounding redirect: %s", url)
        return ""
    try:
        html = await _fetch_html_httpx(url)
    except Exception:
        logger.info("Product extract fast: HTTP fetch failed: %s", url)
        return ""

    if not html:
        return ""

    desc = _extract_from_html(html, url, language)
    if desc:
        logger.info("Product extract fast: HTTP extract OK: %s", url)
        return desc

    logger.info("Product extract fast: No content extracted: %s", url)
    return ""


async def extract_product_text_fast(url: str) -> str:
    """Fast extractor: HTTP only, no Playwright. Returns main page text."""
    if _is_grounding_redirect(url):
        logger.info("Product text fast: Skip grounding redirect: %s", url)
        return ""
    try:
        html = await _fetch_html_httpx(url)
    except Exception:
        logger.info("Product text fast: HTTP fetch failed: %s", url)
        return ""
    if not html:
        logger.info("Product text fast: HTTP fetch failed: %s", url)
        return ""
    text = _extract_main_text(html)
    if text:
        logger.info("Product text fast: HTTP extract OK: %s", url)
        return text
    logger.info("Product text fast: No content extracted: %s", url)
    return ""


async def extract_product_summary_with_playwright(
    url: str,
    language: str,
    timeout_ms: int = 15000,
) -> str:
    """Fallback extractor: Playwright only with shorter timeout."""
    if _is_grounding_redirect(url):
        logger.info("Product extract fallback: Skip grounding redirect: %s", url)
        return ""
    try:
        html = await _fetch_html_playwright(url, timeout_ms=timeout_ms)
    except Exception:
        logger.info("Product extract fallback: Playwright fetch failed: %s", url)
        return ""

    if not html:
        return ""

    desc = _extract_from_html(html, url, language)
    if desc:
        logger.info("Product extract fallback: Playwright extract OK: %s", url)
        return desc

    logger.info("Product extract fallback: No content extracted: %s", url)
    return ""


async def extract_product_text_with_playwright(
    url: str,
    timeout_ms: int = 15000,
) -> str:
    """Fallback extractor: Playwright only with shorter timeout. Returns main page text."""
    if _is_grounding_redirect(url):
        logger.info("Product text fallback: Skip grounding redirect: %s", url)
        return ""
    try:
        html = await _fetch_html_playwright(url, timeout_ms=timeout_ms)
    except Exception:
        logger.info("Product text fallback: Playwright fetch failed: %s", url)
        return ""
    if not html:
        logger.info("Product text fallback: Playwright fetch failed: %s", url)
        return ""
    text = _extract_main_text(html)
    if text:
        logger.info("Product text fallback: Playwright extract OK: %s", url)
        return text
    logger.info("Product text fallback: No content extracted: %s", url)
    return ""
