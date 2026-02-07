# Session Notes: HF Spaces File Download Issue

## Problem Description
Reports (HTML and PDF) are generated successfully on Hugging Face Spaces but cannot be downloaded by users. The error message shown is "File wasn't available on site".

**Evidence:**
- Logs confirm files are created: `Report exported to: /tmp/ssr_report_20260130_193617.html | exists=True`
- Both HTML and PDF exports show `exists=True`
- But clicking download shows "File wasn't available on site"

## Environment
- **Platform**: Hugging Face Spaces (Docker-based)
- **Framework**: Gradio (version likely 5.x or 6.x)
- **Files saved to**: `/tmp/` directory
- **App file**: `frontend/main.py`

## Root Cause Analysis
Gradio has a known bug in version 6.x (GitHub Issue #12452) where dynamically generated files cannot be downloaded. The problem appears to be in how Gradio's file serving/caching mechanism handles files that are created after the app starts.

According to Gradio documentation, files from `tempfile.gettempdir()` should automatically be cached and served, but this doesn't work on HF Spaces.

## Attempts (Historical)

### Attempt 1: Save to project-relative `reports/` directory
- Changed output path from `/tmp/` to `PROJECT_BASE_DIR / "reports"`
- Added `allowed_paths=[str(reports_dir)]` to `demo.launch()`
- Modified Dockerfile to create `/app/reports` with `chmod 777`
- **Result**: Same error - files not available

### Attempt 2: Save directly to `tempfile.gettempdir()` without subdirectory
- Changed from `/tmp/market_wizard_reports/` to just `/tmp/`
- **Result**: Same error - files not available

### Attempt 3: Return `gr.File()` wrapper instead of string path
- Changed return from `str(output_path)` to `gr.File(value=str(output_path), visible=True)`
- **Result**: Caused app restart loop on HF Spaces

### Attempt 4: Replace `gr.File` with `gr.HTML` for download link
- Changed output component from `gr.File` to `gr.HTML`
- Export functions return styled HTML `<a>` tag with download link
- Link points to `/download-report/{filename}` endpoint
- **Result**: Still shows "File wasn't available on site" - the HTML link appears but clicking it fails

## Current Implementation

### Export functions
We cache every generated HTML/PDF via `processing_utils.save_file_to_cache` and return the cached path to `gr.File`, letting Gradio expose it on `/file=...`. The helper still honors `MARKET_WIZARD_EXTERNAL_BASE_URL` plus `__sign` tokens so HF proxies receive a valid absolute URL.

### Frontend components
`export_file` and `fg_export_file` now use `gr.File`, so the download button appears as a native Gradio file download even on HF Spaces.

## Resolution (Local)
- Root cause confirmed: HF Spaces would never hit `/download-report/...` because requests expire without the proper Gradio cache signature.
- Final solution: copy the generated file to Gradio's cache and return that path from `export_file`/`fg_export_file` (with optional override and `__sign` propagation). Gradio now serves the download using its built-in `/file=...` route, so HF proxies stay happy.
- Logs show the cached path and any `__sign` additions, making future debugging straightforward.

## Status
- ✅ Reports export still happens in `/tmp/`, but the download link is now always a Gradio-managed cached file.
- ✅ Download works on HF Spaces and locally without app restarts.

## Follow-up (2026-02-01)
- SSR implementation aligned with the reference SSR tool (cosine scaled to [0,1], epsilon on min only, temperature scaling after PMF).
- Defaults set to temperature=1.0 and epsilon=0.0.
- Local embedding default changed to `all-MiniLM-L6-v2`.
- Added embedding warmup on startup (API + UI) to force local model download in both local and HF environments.
- Added `EMBEDDING_WARMUP` flag (default on) to disable warmup in CI/offline runs if needed.
- SSR scoring now uses only the purchase-intent answer (Q3) when opinions include longer multi-question responses.
- When web-search opinions are long, SSR now generates a separate short purchase-intent statement to match paper-style elicitation.
- Updated UI default to show SSR temperature = 1.0 in Advanced Settings.

## Open Items (HF Spaces)
1. **Confirm proxy routing**: verify whether `/download-report/...` is reachable in HF after deploy.
2. **Check root_path/base_url**: verify request URL, root_path, and forwarded headers in logs.
3. **Try absolute URL in UI**: if root_path mismatch persists, force full URL using HF Space domain.

## Files Changed During This Session
- `/Users/pawel/Market_wizard/frontend/main.py` - Multiple changes to export logic and UI components
- `/Users/pawel/Market_wizard/Dockerfile` - Added `/app/reports` directory creation
- `/Users/pawel/Market_wizard/frontend/main.py` - Mounted Gradio into FastAPI and added request-aware download URLs/logs

## Relevant Documentation Links
- Gradio File Access: https://www.gradio.app/guides/file-access
- Gradio Issue #12452: https://github.com/gradio-app/gradio/issues/12452
- HF Spaces Docker: https://huggingface.co/docs/hub/spaces-sdks-docker
- Gradio mount_gradio_app: https://www.gradio.app/docs/gradio/mount_gradio_app

## Screenshot of Error
![Download error](/Users/pawel/.gemini/antigravity/brain/55155640-d6a9-476c-8f95-31038b0f77a2/uploaded_media_1769801944368.png)

## Follow-up (2026-02-07)

### Demography/GUS and UI
- Fixed gender mapping in UI/backend normalization so Polish `K` maps to backend `F`; English mode supports `F/M` directly.
- Added tests for gender mapping and localization in reports:
  - `backend/tests/test_frontend_gender_mapping.py`
  - `backend/tests/test_report_gender_localization.py`
- Added region/voivodeship targeting in persona generation flow and region-aware test:
  - `backend/tests/test_persona_region_filter.py`
- Unified demographic settings usage across analyses (simulation, A/B, price analysis, focus group), with focus-group size still controlled separately.

### SSR parity and benchmarking vs semantic-similarity-rating
- Added SSR parity test path and aligned Market Wizard SSR behavior with the reference `semantic-similarity-rating` engine.
- Added/updated SSR test:
  - `backend/tests/test_ssr_engine.py`
- Added benchmark scripts for real datasets:
  - `backend/scripts/evaluate_ssr_on_real_data.py`
  - `backend/scripts/evaluate_ssr_20_industries.py`
- Stored benchmark summary report:
  - `reports/ssr_embedding_benchmark_2026-02-07.md`

### Embedding model decision
- Compared `all-MiniLM-L6-v2` vs `BAAI/bge-m3` on real-data benchmarks.
- Set default embedding fallback in code to `BAAI/bge-m3`:
  - `backend/app/config.py`
- Note: HF Spaces may not use local `.env`; code fallback/default in config is the effective safety net for deployments.

### Documentation and release tracking
- Updated `README.md` with testing workflow, including compatibility checks against `semantic-similarity-rating`.
- Updated `changelog.txt` with the latest entries (v0.5.2).

### Git status / remotes
- Latest released commit in this thread: `0f86363` (`Update changelog for SSR benchmarks and bge-m3 default`).
- Pushed to both remotes:
  - `origin/main` (GitHub)
  - `hf/main` (Hugging Face Spaces)

## Follow-up (2026-02-07) - Pre-calibration baseline checkpoint

### Objective
- Created a rollback-safe baseline commit before starting SSR calibration work.
- Baseline chosen from large hybrid benchmarks: fixed-anchor setup (EN anchors), `T=0.7`, `epsilon=0.0`.

### What was prepared
- Added tuning and data-prep scripts:
  - `backend/scripts/tune_ssr_hierarchical.py`
  - `backend/scripts/validate_grouped_ratings_csv.py`
  - `backend/scripts/build_poland_grouped_csv.py`
  - `backend/scripts/ssr_grouped_ratings_template.csv`
- Generated benchmark and hybrid reports:
  - `reports/ssr_hierarchical_tuning_2026-02-07.md`
  - `reports/ssr_hierarchical_tuning_2026-02-07_large.md`
  - `reports/ssr_hybrid_tuning_pl_2026-02-07.md`
  - `reports/ssr_hybrid_tuning_pl_precise_2026-02-07.md`
  - `reports/ssr_hybrid_tuning_pl_precise_auto_anchor_2026-02-07.md`
- Generated PL grouped datasets:
  - `reports/pl_grouped_ratings_2026-02-07.csv`
  - `reports/pl_grouped_ratings_precise_2026-02-07.csv`

### Runtime defaults set from benchmark winner
- `SSR_TEMPERATURE=0.7`
- `SSR_EPSILON=0.0`
- Applied in:
  - `backend/app/config.py`
  - `backend/app/services/simulation_engine.py`
  - `frontend/main.py` (Advanced Settings default slider value)

### Environment template cleanup
- Canonical template: `backend/.env.example`
- Removed duplicate `backend/.env.sample`
- Updated setup docs in `README.md` to use `.env.example`

### Calibration readiness note
- This checkpoint is explicitly intended as the "before calibration" recovery point.
- If post-calibration changes degrade quality, revert/reset to this commit and rerun from baseline.
