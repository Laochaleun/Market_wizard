"""Product page extractor with HTTP + optional Playwright fallback."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


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

    specs = _collect_specs(soup)
    desc_blocks = _collect_description_blocks(soup)

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

    if not price:
        text_blob = soup.get_text(" ", strip=True)
        price, currency = _extract_price_from_text(text_blob)

    # Prefer longer, page-specific description blocks when available.
    if desc_blocks and len(desc_blocks) > len(description or ""):
        description = desc_blocks

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
        spec_items = []
        for key, value in specs[:12]:
            spec_items.append(f"{key}: {value}")
        specs_text = "; ".join(spec_items)
        if language.lower().startswith("pl"):
            base_desc = f"{base_desc} Parametry: {specs_text}."
        else:
            base_desc = f"{base_desc} Specifications: {specs_text}."

    return base_desc.strip()


async def _fetch_html_httpx(url: str) -> str:
    logger.info("Product extract: HTTP fetch start: %s", url)
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    timeout = httpx.Timeout(15.0, connect=10.0)
    async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=timeout) as client:
        resp = await client.get(url)
        resp.raise_for_status()
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


async def _fetch_html_playwright(url: str) -> str:
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
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await _try_accept_cookies(page)
            await page.wait_for_load_state("networkidle", timeout=30000)
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
