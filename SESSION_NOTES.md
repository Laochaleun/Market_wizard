# Session Notes: HF Spaces File Download Issue

## Reference paper location (local context)
- Main methodological reference for this app:
  - arXiv: `2510.08338v3`
- Optional local copy (kept outside git, via `.gitignore`):
  - `/Users/pawel/Market_wizard/.local_context/papers/2510.08338v3.pdf`

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

## Follow-up (2026-02-07) - Test matrix, data links, and next-session plan

### Test matrix completed in this session
- **Embedding benchmark (real labeled datasets)**:
  - Script: `backend/scripts/evaluate_ssr_on_real_data.py`
  - Report: `reports/ssr_embedding_benchmark_2026-02-07.md`
  - Input datasets:
    - HF `yelp_review_full` (EN)
    - HF `amazon_reviews_multi` (EN; when available in runtime)
- **20-industry benchmark (large, EN)**:
  - Script: `backend/scripts/evaluate_ssr_20_industries.py`
  - Report: `reports/ssr_hierarchical_tuning_2026-02-07.md`
  - Input dataset:
    - HF `McAuley-Lab/Amazon-Reviews-2023` (`review_categories/*.jsonl`, 20 categories)
- **Hierarchical SSR tuning (large variants)**:
  - Script: `backend/scripts/tune_ssr_hierarchical.py`
  - Reports:
    - `reports/ssr_hierarchical_tuning_2026-02-07.md` (20k examples, 25 configs)
    - `reports/ssr_hierarchical_tuning_2026-02-07_large.md` (stability run)
- **Hybrid tuning (20 industries + PL groups)**:
  - Script: `backend/scripts/tune_ssr_hierarchical.py`
  - Reports:
    - `reports/ssr_hybrid_tuning_pl_2026-02-07.md` (coarse PL groups)
    - `reports/ssr_hybrid_tuning_pl_precise_2026-02-07.md` (precise PL groups)
    - `reports/ssr_hybrid_tuning_pl_precise_auto_anchor_2026-02-07.md` (auto anchor language)
  - Input data:
    - `reports/pl_grouped_ratings_2026-02-07.csv` (coarse PL grouping)
    - `reports/pl_grouped_ratings_precise_2026-02-07.csv` (precise PL_industry grouping)
    - Generated by `backend/scripts/build_poland_grouped_csv.py` from HF `allegro/klej-allegro-reviews`
  - Input CSV validation:
    - `backend/scripts/validate_grouped_ratings_csv.py`

### Baseline outcome to carry into calibration phase
- Best fixed setup in this checkpoint:
  - `temperature=0.7`
  - `epsilon=0.0`
  - fixed anchors (EN in tested hybrid setup)
- This setup is now the runtime default baseline (pre-calibration).

### PI-only status (important clarification)
- The "PI-only sentence" idea is **already implemented** and should **not** be duplicated in next steps:
  - Extraction helper: `backend/app/services/simulation_engine.py` (`_extract_purchase_intent_text`)
  - SSR input path:
    - uses `ssr_text` when generated,
    - otherwise falls back to extracted concise purchase-intent sentence.
- For web-search mode, a dedicated extra opinion is generated for SSR scoring and passed via `ssr_text`.

### Next session plan (fresh start) - calibration-first
- Do **not** run another broad raw grid as the first step.
- Prioritize:
  1. Add **global post-SSR calibration** (`pred_score -> calibrated_score`, monotonic/isotonic).
  2. Add **per-group calibration** only for sufficiently large groups (e.g., `n >= 1000`) and only if validated uplift is real.
  3. Improve anchor quality for domain fit (e-commerce wording: price/value/quality) and evaluate weighted anchor sets.
  4. Use multi-metric objective for selection (not Spearman-only), e.g. `0.5*Spearman - 0.3*MAE + 0.2*OffByOne`.
  5. Optionally increase response stability in simulation by averaging 2-3 textual samples per persona.
- Deliverable target for next session:
  - calibration script(s) + `before vs after` report on current benchmark suites.

## Follow-up (2026-02-07) - Stage 1 calibration execution log (detailed)

### Scope completed in this session
- Implemented first-stage calibration infrastructure in code and runtime.
- Added external production-readiness benchmark suite with hard pass/fail gates.
- Added domain-aware calibration routing and policy artifacts for HF-compatible fallback.
- Repeated full validation runs on external datasets after each material change.

### 1) Core calibration infrastructure added

#### New module
- `backend/app/services/score_calibration.py`
  - `IsotonicCalibrator`:
    - `transform()`
    - JSON serialization/deserialization (`isotonic_v1`)
    - file save/load helpers.
  - `DomainCalibrationPolicy`:
    - domain-to-calibrator routing
    - fallback to default domain
    - JSON format `domain_calibration_v1`.
  - `fit_isotonic_calibrator()` (PAV fit).

#### SSR runtime integration
- `backend/app/services/ssr_engine.py`
  - Added calibration-aware scoring:
    - supports single calibrator and domain policy artifact.
    - added `domain_hint` in `rate_response()` and `rate_responses()`.
  - Added `raw_expected_score` in `SSRResult` for diagnostics/comparison.
  - Loading behavior:
    - if policy artifact available and enabled -> policy routing,
    - else fallback to single calibrator artifact,
    - robust fail-safe (scoring does not crash on artifact load errors).

#### Simulation routing
- `backend/app/services/simulation_engine.py`
  - SSR scoring call now passes `domain_hint="ecommerce"` for product PI path.

#### Config + HF fallback defaults
- `backend/app/config.py`
  - `ssr_temperature` switched to `1.0` to match calibrator training setup.
  - Added:
    - `ssr_calibration_enabled`
    - `ssr_calibration_artifact_path`
    - `ssr_calibration_policy_path`
  - Default fallback paths point to repo-stored artifacts under `backend/app/data/`.

#### Env templates
- `backend/.env.example`
  - Added calibration keys and updated defaults for stage-1 runtime.

### 2) Tuning script extended with calibration validation

- `backend/scripts/tune_ssr_hierarchical.py`
  - Added global isotonic calibration options:
    - `--global-calibration`
    - `--calibration-min-samples`
    - `--calibration-cv-folds`
    - `--calibration-holdout-ratio`
    - `--calibration-artifact-out`
  - Added OOF + holdout sections in report.
  - Added artifact saving for winning config.
  - Added `--skip-industries` to support offline fallback runs.

### 3) Domain policy builder script added

- `backend/scripts/build_domain_calibration_policy.py`
  - Builds `domain_calibration_v1` artifact with two domains:
    - `general`
    - `ecommerce`
  - Supports optimization modes:
    - `--optimize mae`
    - `--optimize off1`
  - Uses holdout split per domain and stores diagnostics in artifact metadata.

### 4) External production-readiness suite added

- `backend/scripts/validate_production_readiness.py`
  - External datasets used:
    - Amazon Reviews 2023 (20 categories; HF JSONL stream)
    - Yelp Review Full
    - App Reviews
    - Allegro KLEJ Reviews (PL)
  - Metrics:
    - MAE, Spearman, Exact, Off-by-one
    - Bootstrap 95% CI (MAE/Off1/Spearman)
    - Temporal tail check where timestamp/date is available
  - Policy search included:
    - `raw_only`
    - `global_calibrated`
    - `pl_only_calibrated`
    - `ecommerce_only_calibrated`
    - `blend_0.1 ... blend_0.9`
    - `domain_policy_artifact` (if policy file present)
  - Hard production gates:
    - MAE <= 0.60
    - Off-by-one >= 0.92
    - Spearman drop <= 0.01
    - max per-dataset MAE regression <= 0.10

### 5) Artifacts produced / updated

- `backend/app/data/ssr_calibrator_default.json`
- `backend/app/data/ssr_calibration_policy_default.json`
- `reports/ssr_hybrid_tuning_pl_precise_calibrated_2026-02-07.md`
- `reports/ssr_calibrator_hybrid_pl_precise_2026-02-07.json`
- `reports/production_readiness_validation_2026-02-07.md`

### 6) Validation history and key numeric outcomes

#### A) Hybrid tuning with calibration (HF + PL grouped)
- Report: `reports/ssr_hybrid_tuning_pl_precise_calibrated_2026-02-07.md`
- Scale:
  - 27 groups
  - 15,406 examples
- Best config:
  - `T=1|eps=0|anchor=avg6`
- Holdout (for calibration effect):
  - MAE improved strongly (`~0.99 -> ~0.53`)
  - Exact improved strongly
  - Spearman near-stable (expected for monotonic mapping).

#### B) External readiness baseline (global calibrator)
- Report: `reports/production_readiness_validation_2026-02-07.md`
- Total rows:
  - 13,000
- Result:
  - `FAIL`
- Main issue:
  - global calibrator hurt some general-review domains (notably Yelp off-by-one).

#### C) Policy search without trained general domain
- Best policy candidate:
  - `ecommerce_only_calibrated`
- Still `FAIL`:
  - off-by-one below 0.92 gate
  - MAE above 0.60 gate.

#### D) Domain policy artifact (general + ecommerce) with separate fitting
- Artifact chosen by policy search at one stage:
  - `domain_policy_artifact` (best by aggregate objective)
- Still `FAIL` vs hard gates:
  - close-to-threshold MAE in one run, but off-by-one CI still below 0.92.

#### E) Off1-optimized domain policy retrain
- Re-trained `domain_calibration_v1` with `--optimize off1`.
- Holdout diagnostics from builder script:
  - general MAE: `0.9311 -> 0.8065`
  - ecommerce MAE: `0.9925 -> 0.6575`
- Re-ran full production validation:
  - best policy again `ecommerce_only_calibrated`
  - final decision remained `FAIL`.

### 7) Runtime decisions applied

- Switched runtime defaults to `SSR_TEMPERATURE=1.0` (critical consistency with calibration training).
- Enabled calibration by default in code fallback for HF deployments (where local `.env` may be absent).
- Left stage-1 policy as default artifact, but readiness verdict remains non-production.

### 8) Test coverage updates

- Added/updated tests:
  - `backend/tests/test_score_calibration.py`
  - `backend/tests/test_ssr_engine.py`
- Status in-session:
  - calibration + SSR tests passing repeatedly (`12 passed` on latest run set).

### 9) Why Stage 1 is intentionally marked as incomplete for production

- Infrastructure is complete for:
  - training calibrators,
  - loading artifacts in runtime,
  - routing by domain,
  - running external objective benchmark with CIs and gates.
- However, objective production bars are still not met on external benchmark suite.

### 10) Stage 2 recommendation (next session)

1. Revisit base SSR signal quality before calibration:
   - anchor refinement (domain language),
   - embedding sensitivity tests (`BAAI/bge-m3` vs `text-embedding-3-small` in paper mode).
2. Add paper-native evaluation outputs:
   - KS similarity for distribution match,
   - correlation attainment with split-half ceiling.
3. Refit domain policies on expanded and cleaner splits:
   - isolate Yelp-like “general reviews” from e-commerce intent framing.
4. Keep same hard gates and rerun identical external validation:
   - do not relax gates unless explicitly decided by product owner.

### 11) Git checkpoints from this session

- Calibration stage-1 implementation commit:
  - `1918c6b`
  - message: `Stage 1 calibration: add domain policy runtime and external readiness validation`
- Local paper-context documentation commit:
  - `03ad485`
  - message: `Document local reference paper context and ignore .local_context`

### 12) Remote push status (end of session)

- Pushed: `origin/main` (GitHub) up to `03ad485`.
- Not pushed intentionally: `hf/main` (Hugging Face) to keep deployment pending while calibration work continues.
