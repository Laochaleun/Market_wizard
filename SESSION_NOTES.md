# Session Notes (handoff)

Date: 2026-01-29

## Goal
Stabilize web grounding sources, make links clickable and consistent, reduce hallucinations, and improve source summaries so agents cite real products/prices with [n] citations. Also improve report rendering and diagnostics. The app is Market_wizard.

## Summary of Work Done (Working Changes)

### 1) Grounding + Research Pipeline
- Research uses **Gemini 2.5 flash lite** and response generation uses **Gemini 3 flash preview**.
- Global sources are collected once (global search + cache), then **per‑agent selection** is done from global sources.
- Per‑agent grounding/search is **disabled** (determinism, lower cost).
- Grounding redirect URLs (vertexaisearch) are **resolved** to real URLs; redirects are filtered/avoided in fetching.
- Added **cache versioning** so new extraction logic invalidates old cache: `cache_version = "v2"`.

Files:
- `backend/app/services/llm_client.py`
  - `generate_market_sources` uses `self.research_model_name`.
  - Shorter prompts, per‑prompt timeout, error handling.
  - Skip grounding redirect URLs in dedup.
  - Research cache key changed to include version + `research_model_name`.
- `backend/app/services/simulation_engine.py`
  - Per‑agent search removed; always use global sources when web search enabled.

### 2) Product extraction improvements (summary quality)
- Added HTML noise stripping to reduce unrelated content in summaries.
- Prefer **JSON‑LD** if present; only use page description blocks if no structured data.
- Aggressive length limits (2 sentences, max ~45 words when no punctuation).
- Remove duplicate product name fragments (handles separators like “ - ”).
- More robust cleaning and marketing noise trimming; **no product‑specific keywords**.

Files:
- `backend/app/services/product_extractor.py`
  - `_strip_noise_sections`: removes DOM nodes with related/upsell/carousel/reviews/footer etc.
  - `_clean_features_text`: trims boilerplate, shortens to 2 sentences, caps words, strips repeated name.
  - `_strip_marketing_noise`: generic removal of short labels + hype with exclamations (no product-specific list now).
  - Safer DOM traversal (guard for bad tags).
  - Stop adding specs when description already contains key spec terms.

### 3) Report output improvements
- HTML report now renders markdown-like text into proper HTML (no ### / ** artifacts).
- Source list shows **domain count**.
- Option to show **only cited sources** is available in UI.
- Citations [n] required near prices and product-like terms; if missing, regeneration prompt is used.

Files:
- `backend/app/services/report_generator.py` (HTML formatting and source filtering)
- `backend/app/models/schemas.py` (`AgentResponse` has `sources`)
- `frontend/main.py` (checkbox: only cited sources)

### 4) Grounding probe improvements
- `backend/scripts/grounding_probe.py` updated to handle sources as dicts or URLs (domains displayed correctly).

### 5) Playwright fallback
- Playwright only used when HTTP extraction fails.
- Playwright is limited by `research_playwright_fallback_limit` and `research_playwright_timeout_ms`.
- Playwright **skips vertexaisearch redirect** URLs.

Config:
- `backend/app/config.py`: new settings for research model and Playwright fallback.
- `backend/.env`: comments added for RESEARCH_* settings.

## Verified Results (from user tests)
- Global sources return real URLs and multiple domains.
- Agents include citations [n] near prices and product names.
- Report now shows sources with clickable links.
- Summary still sometimes contains marketing fluff like “Boskie moce” from some shops, but after the latest changes the noise is reduced and summaries are shorter and mostly relevant.

## Current Status / What’s Still Open

1) **Summary quality still occasionally noisy**
   - Example: GodsToys summary had “Boskie moce: Pojemność 1,75 l! ...”
   - It is now shorter but can still include some marketing phrasing. Should we tighten the filter further?

2) **Confirm final summary quality**
   - Need one more local test after latest changes (cache cleared) to ensure summary is concise and relevant.

3) **Optional**: JSON‑LD only mode
   - If you want zero HTML noise, implement a strict mode that only uses JSON‑LD + meta description fallback (no page content extraction).

## Plan for Next Session
1) Run the local test once (cache cleared) and capture `sample_source` output.
2) Decide whether to keep current generic trimming or add a strict “JSON‑LD only” extraction mode.
3) If summaries are still noisy, implement JSON‑LD only fallback and re‑test.
4) Verify report output (HTML) on a recent simulation and confirm sources list + citations [n].

## Session Kickoff Prompt (for Codex)
You are continuing Market_wizard. Read `SESSION_NOTES.md`. Goal: verify summary quality after latest extractor changes. First step: run the local test (cache cleared) to get `sample_source` and decide if we need JSON‑LD only mode. If summaries are still noisy, implement JSON‑LD only fallback and re‑test. Avoid product-specific keyword lists.

## How to Re‑run the local test
```
cd /Users/pawel/Market_wizard/backend
set -a; source .env; set +a
PYTHONPATH=/Users/pawel/Market_wizard/backend python - <<'PY'
import asyncio
from app.services.llm_client import get_llm_client
from app.services.persona_manager import PersonaManager
from app.i18n import Language

async def main():
    llm = get_llm_client()
    persona = PersonaManager(language=Language.PL).generate_population(n_agents=1)[0]
    product = (
        "Karafka w kształcie środkowego palca, szkło borokrzemowe, 1L, cena ok. 100 PLN. "
        "Do domu/bar, prezent humorystyczny."
    )
    sources = await llm.generate_market_sources(product, language=Language.PL)
    print("global_sources:", len(sources))
    if sources:
        print("sample_source:", sources[0])
    opinion, used = await llm.generate_opinion_with_sources(
        persona, product, sources, language=Language.PL
    )
    print("used_sources:", len(used))
    print("opinion:\n", opinion)

asyncio.run(main())
PY
```

## Cleanups performed
- Deleted `backend/data/search_cache.json` multiple times to force fresh summaries. If you need a fresh run now, delete the cache file again before testing.

## Commit message suggestions (per logical change)
1) `feat(research): use global sources only; improve grounding prompts and cache versioning`
2) `fix(extraction): strip noisy sections; prefer JSON-LD; shorten summaries`
3) `feat(report): render HTML, add domain counts, allow only-cited sources`

## Notes / Constraints
- The assistant runtime sometimes lacks DNS; when it fails, tests must be run locally.
- Avoid adding product‑specific keyword lists; keep filters generic.

## 2026-01-30 Updates
- Added separate model for interpretation: `research_interpretation_model` (default `gemini-3-flash-preview`) while grounding remains `gemini-2.5-flash-lite`.
- Grounding now always uses the research model (fix for empty sources when interpretation model was used).
- Added JSON-LD structured field extraction to prefer product price when available.
- Added multilingual shipping fee filtering (prompt + post-process) to avoid treating shipping cost as product price.
- Added product-like URL prioritization and top-5 URL logging; listing/guide sources remain as backup.
