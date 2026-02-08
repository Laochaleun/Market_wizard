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

## Follow-up (2026-02-08) - Stage 2 (anchor + calibration) detailed execution log

### Session objective
- Continue Stage 2 work to close remaining production gaps after Stage 1.
- Constraints held constant:
  - hard gates unchanged:
    - MAE <= 0.60
    - Off-by-one >= 0.92
    - Spearman drop <= 0.01
    - max per-dataset MAE regression <= 0.10
  - no relaxation of acceptance criteria.

### High-level strategy used
1. Improve base SSR signal first via anchor refinement (PL + EN), aligned to paper-style purchase-intent elicitation and **not** e-commerce-only semantics.
2. Keep runtime temperature/epsilon fixed at calibration-consistent values (`T=1.0`, `eps=0.0`).
3. Re-evaluate repeatedly on exactly the same external validation suite (Amazon 2023, Yelp, App Reviews, Allegro).
4. Expand policy search space:
   - global, domain, per-language policies,
   - blend families,
   - purchase-intent-specific hybrids.
5. Refit artifacts after anchor updates.
6. Compare “artifact A + artifact B” cross-combinations (old/new calibrator vs old/new policy) to identify best compromise.

### Code-level changes completed in this stage

#### 1) Anchor variants and selection framework
- Added anchor-variant registry in `backend/app/i18n.py`:
  - `paper_general_v1`
  - `paper_general_v2`
  - `paper_general_v3`
- Added:
  - `DEFAULT_ANCHOR_VARIANT`
  - `get_anchor_variants()`
  - `get_anchor_sets(language, variant=...)`
- Default switched to: `paper_general_v3`.

#### 2) Runtime domain naming alignment
- `backend/app/services/simulation_engine.py`:
  - changed SSR routing hint from `domain_hint="ecommerce"` to `domain_hint="purchase_intent"`.
- `backend/app/services/score_calibration.py`:
  - added backward-compatible domain aliasing:
    - `purchase_intent -> ecommerce` when needed,
    - `ecommerce -> purchase_intent` when needed.

#### 3) Validation script expansion and diagnostics
- `backend/scripts/validate_production_readiness.py`:
  - added `--anchor-variant`,
  - added paper-native diagnostics:
    - KS similarity (`1 - KS distance`) on rounded 1..5 distributions,
    - split-half ceiling proxy,
    - correlation attainment proxy.
  - added robust local-cache fallbacks for HF datasets in constrained environments.
  - renamed/extended policy families:
    - `purchase_intent_only_calibrated`,
    - `purchase_domain_only_calibrated`,
    - `purchase_hybrid_0.1 ... 0.9`,
    - retained raw/global/pl/domain/blend baselines.

#### 4) Training scripts upgraded for anchor variants
- `backend/scripts/tune_ssr_hierarchical.py`:
  - added `--anchor-variant`,
  - propagated variant to report metadata and anchor embedding generation.
- `backend/scripts/build_domain_calibration_policy.py`:
  - added `--anchor-variant`,
  - policy keys include `purchase_intent` (with `ecommerce` kept for compatibility),
  - metadata expanded.
- Added new script:
  - `backend/scripts/build_global_calibrator.py`
  - purpose: build single global calibrator with objective modes:
    - `mae`
    - `off1`
    - `balanced`

### Detailed experiment matrix and outcomes

#### A) Anchor screening (small/fast)
- Reports:
  - `reports/production_readiness_validation_2026-02-07_anchor_screen_v1.md`
  - `reports/production_readiness_validation_2026-02-07_anchor_screen_v2.md`
  - `reports/production_readiness_validation_2026-02-07_anchor_screen_v3.md`
- Sample sizing used for screening:
  - Amazon: `80/category` (20 categories),
  - Yelp/App/Allegro: `1200` each,
  - bootstrap: `300`.
- Summary:
  - `v2` and `v3` improved objective versus `v1`,
  - selected for full run: `paper_general_v3`.

#### B) Full run after anchor refinement (`paper_general_v3`)
- Report:
  - `reports/production_readiness_validation_2026-02-07_stage2_anchor_refined_v3_full.md`
- Notable change vs previous anchor baseline:
  - stronger raw signal on multiple datasets,
  - best policy switched behaviorally toward purchase-intent family,
  - still overall `FAIL`.

#### C) Retraining artifacts on `paper_general_v3`
- Global calibrator retrain report:
  - `reports/ssr_hierarchical_tuning_2026-02-07_anchor_v3_recal.md`
- Full readiness with retrained artifacts:
  - `reports/production_readiness_validation_2026-02-07_stage2_anchor_v3_retrained_full.md`
- Best policy in this run:
  - `purchase_intent_only_calibrated`
- Key near-pass:
  - MAE `0.6092` (target `<=0.60`)
  - Off1 `0.9189` (target `>=0.92`)
  - Spearman-drop and max-regression gates passed.

#### D) Hybrid policy-space expansion
- Report:
  - `reports/production_readiness_validation_2026-02-07_stage2_policy_hybrid_search.md`
- Added candidate families yielded expected tradeoff curve:
  - lower MAE around `purchase_intent_only`,
  - higher Off1 around stronger purchase-domain hybrids/domain policy.
- No candidate met both MAE and Off1 gates simultaneously.

#### E) Artifact combination experiments
- Old calibrator + new policy:
  - `reports/production_readiness_validation_2026-02-07_stage2_oldcal_newpolicy_hybrid.md`
  - still `FAIL`.
- New calibrator + old policy:
  - `reports/production_readiness_validation_2026-02-07_stage2_newcal_oldpolicy_hybrid.md`
  - still `FAIL`.

#### F) Domain-policy data partition change
- Modified policy builder to train:
  - `general`: Yelp,
  - `purchase_intent`: Amazon + App + Allegro.
- Full run:
  - `reports/production_readiness_validation_2026-02-07_stage2_policy_retrained_purchase_app.md`
- Result:
  - Off1 improved with stronger domain policies,
  - MAE drifted upward too much,
  - overall `FAIL`.

#### G) Global calibrator objective sweeps (new script)
- Off1-optimized global calibrator:
  - report: `reports/production_readiness_validation_2026-02-07_stage2_global_off1_calibrator.md`
  - outcome: Off1 improved, MAE worsened significantly -> `FAIL`.
- Balanced-objective global calibrator:
  - report: `reports/production_readiness_validation_2026-02-07_stage2_global_balanced_calibrator.md`
  - outcome: still MAE too high -> `FAIL`.
- Conclusion:
  - global calibrator objective retuning alone cannot close both MAE + Off1 gates simultaneously on this benchmark mix.

### Best known point retained at end of session
- Chosen practical checkpoint for continuation:
  - anchors: `paper_general_v3` (default),
  - global calibrator: isotonic baseline retrained via hierarchical script,
  - policy: baseline-compatible artifact (restored for stability).
- Best near-pass metric snapshot to beat next:
  - policy `purchase_intent_only_calibrated`
  - MAE `0.6092`
  - Off1 `0.9189`
  - Spearman drop: pass
  - max dataset MAE regression: pass

### Why still not production-ready
- Remaining gaps are small but persistent and coupled:
  - reducing MAE tends to reduce Off1 below threshold,
  - increasing Off1 tends to raise MAE above threshold.
- Current gating failure is not due to missing infrastructure; it is due to unresolved objective tradeoff on external data.

### Strong recommendation for next session
1. Add `paper_general_v4` anchors (PL+EN) tuned explicitly for preserving high Off1 while avoiding MAE inflation on Yelp/App.
2. Refit domain policy using stricter per-domain holdout diagnostics and possibly separate purchase-intent subdomains (`amazon_like`, `app_like`, `pl_retail`) if data supports it.
3. Keep same hard gates and run only full-comparison experiments (no gate relaxation).
4. Track a fixed “best-known” benchmark row and require explicit improvement before replacing runtime artifacts.

## Follow-up (2026-02-08) - Stage 2B (v4 typed-domain calibration), full execution log

### Critical methodology requirement for all next sessions (MUST)
- The calibration and SSR workflow **must always** be guided by the reference paper:
  - `arXiv:2510.08338v3`
  - local copy available at: `/Users/pawel/Documents/!Praca/Subverse/Human purchase intent/2510.08338v3.pdf`
- This is not optional context. It is a hard requirement for future iterations.
- Practical implications enforced in this session:
  1. Anchor statements must remain short, generic, and domain-independent.
  2. Anchors must represent purchase-intent likelihood gradient (unlikely -> neutral/uncertain -> likely).
  3. SSR remains text-first; no direct Likert elicitation logic introduced.
  4. Temperature/epsilon consistency maintained (`T=1.0`, `eps=0.0`) unless explicitly re-validated.

### Session objective
- Continue Stage 2 after v4 anchors and close the MAE/Off1 gate gap without violating paper-style PI elicitation.
- Replace platform-like domain split assumption with app-aligned split by **response type**:
  - `purchase_intent_short`
  - `review_long`
  - with language-specific variants (`_en`, `_pl`) where relevant.

### Starting point at session entry
- Prior best v4 full validation (before typed-domain runtime changes):
  - report: `reports/production_readiness_validation_2026-02-08_stage2_v4_full_best.md`
  - best policy: `domain_policy_artifact`
  - MAE: `0.6229`
  - Off1: `0.9200`
  - Decision: `FAIL`
- Legacy near-pass from earlier stage remained:
  - MAE `0.6092`
  - Off1 `0.9189`

### Code changes completed (Stage 2B)

#### 1) Domain policy selection upgraded to typed/language-aware fallback graph
- File: `backend/app/services/score_calibration.py`
- Added hierarchical candidate resolution in `DomainCalibrationPolicy`:
  - accepts typed hints such as:
    - `purchase_intent_short_en`
    - `purchase_intent_short_pl`
    - `review_long_en`
  - preserves backward compatibility via fallback chain to:
    - `purchase_intent`
    - `ecommerce`
    - `general`
- Behavior:
  - first tries exact key,
  - then removes language suffix if needed,
  - then applies type-level fallback,
  - finally uses `default_domain`.

#### 2) SSR batch scoring now supports per-row domain hints
- File: `backend/app/services/ssr_engine.py`
- `rate_responses()` signature extended:
  - supports `domain_hints: List[str | None]`
  - validates length consistency with response list
  - keeps existing `domain_hint` behavior for backward compatibility.

#### 3) Simulation runtime now classifies SSR input by type + language
- File: `backend/app/services/simulation_engine.py`
- Added `_build_ssr_input_and_hint(opinion, ssr_text)`:
  - if dedicated SSR text exists -> `purchase_intent_short_{lang}`
  - else extracted PI sentence is used,
  - if extracted text equals long full opinion and is long enough, classified as `review_long_{lang}`.
- `run_simulation()` now builds:
  - `text_responses`
  - matching per-response `domain_hints`
- SSR scoring call updated to pass `domain_hints` list instead of one global hint.

#### 4) Domain policy training script upgraded to typed domains
- File: `backend/scripts/build_domain_calibration_policy.py`
- Training partition now explicit:
  - `review_long_en` <- Yelp (general long-review proxy)
  - `purchase_intent_short_en` <- Amazon 2023 + App Reviews
  - `purchase_intent_short_pl` <- Allegro KLEJ
  - aggregated compatibility calibrator still emitted as `purchase_intent`/`ecommerce`
- Metadata extended with typed diagnostics and train row counts.

#### 5) Validation routing aligned with typed domains
- File: `backend/scripts/validate_production_readiness.py`
- Domain hints for policy routing changed to:
  - Allegro -> `purchase_intent_short_pl`
  - Amazon/App -> `purchase_intent_short_en`
  - Yelp -> `review_long_en`
- Purchase-domain hybrid logic now checks typed purchase calibrators first, then compatibility key.

#### 6) Test coverage extended
- Updated tests:
  - `backend/tests/test_score_calibration.py`
  - `backend/tests/test_ssr_engine.py`
- Added/updated assertions for:
  - typed-domain fallback behavior,
  - per-row domain hint scoring in SSR batch path.
- Additional anchor integrity test introduced in this broader v4 stream:
  - `backend/tests/test_i18n_anchors.py`

### Validation and training runs executed in this session

#### A) Reference v4 full run (baseline for this session branch)
- Report:
  - `reports/production_readiness_validation_2026-02-08_stage2_v4_full_best.md`
- Best policy:
  - `domain_policy_artifact`
- Metrics:
  - MAE `0.6229`
  - Off1 `0.9200`
- Verdict:
  - `FAIL`

#### B) Typed-domain smoke training + smoke validation
- Policy artifact (small sample smoke):
  - `reports/ssr_policy_v4_typed_smoke.json`
- Validation report (small sample smoke):
  - `reports/production_readiness_validation_2026-02-08_stage2_v4_typed_smoke.md`
- Purpose:
  - verify end-to-end typed routing/training works before expensive full runs.
- Result:
  - technically successful execution,
  - still `FAIL` (expected at smoke scale).

#### C) Full typed policy training artifacts
- MAE objective:
  - `reports/ssr_policy_v4_typed_full_mae.json`
  - holdout diagnostics printed by trainer:
    - general MAE `0.7248 -> 0.6295`
    - ecommerce/purchase aggregate MAE `0.9389 -> 0.6132`
- Off1 objective:
  - `reports/ssr_policy_v4_typed_full_off1.json`
  - holdout diagnostics printed by trainer:
    - general MAE `0.7248 -> 0.6257`
    - ecommerce/purchase aggregate MAE `0.9389 -> 0.7680`

#### D) Full production-readiness runs for typed policies
- MAE typed policy full report:
  - `reports/production_readiness_validation_2026-02-08_stage2_v4_typed_full_mae.md`
  - best policy: `domain_policy_artifact`
  - MAE: `0.6143`
  - Off1: `0.9187`
  - Spearman: `0.7674`
  - Decision: `FAIL`
- Off1 typed policy full report:
  - `reports/production_readiness_validation_2026-02-08_stage2_v4_typed_full_off1.md`
  - best policy: `domain_policy_artifact`
  - MAE: `0.7183`
  - Off1: `0.9278`
  - Spearman: `0.7546`
  - Decision: `FAIL`

### Comparative conclusions from this session
- Typed MAE policy produced the best overall compromise for current app goals:
  - MAE improved versus prior v4 full best:
    - `0.6229 -> 0.6143`
  - Off1 slightly decreased:
    - `0.9200 -> 0.9187`
  - Ranking/correlation-related behavior improved (Spearman up in best policy row).
- Typed Off1 policy improved Off1 but with unacceptable MAE cost:
  - MAE inflated to `0.7183`.
- Gate status still unresolved at full scale:
  - MAE gate: fail
  - Off1 gate: fail (for selected MAE typed policy)
  - Spearman drop gate: pass
  - max per-dataset MAE regression gate: pass

### Runtime artifacts applied at end of session
- Global calibrator default kept as:
  - `backend/app/data/ssr_calibrator_default.json`
  - source artifact lineage: v4 balanced global calibrator
- Domain policy default updated to typed MAE full policy:
  - `backend/app/data/ssr_calibration_policy_default.json`
  - source: `reports/ssr_policy_v4_typed_full_mae.json`
- Confirmed policy contains typed domains + compatibility keys:
  - `general`
  - `review_long_en`
  - `purchase_intent_short_en`
  - `purchase_intent_short_pl`
  - `purchase_intent`
  - `ecommerce`

### Reporting/documentation updates made
- Added calibration summary report for stakeholders:
  - `reports/calibration_report_08022026.md`
  - includes gate goals/results and practical expected deviation interpretation.
- Changelog updated with:
  - v4 default anchors,
  - typed-domain calibration routing,
  - full report references,
  - selected runtime artifact decision.

### Current known best checkpoints (important)
1. Historical near-pass (legacy stage):
   - MAE `0.6092`, Off1 `0.9189`
2. Best v4 full before typed split:
   - MAE `0.6229`, Off1 `0.9200`
3. Best typed full (selected now for runtime):
   - MAE `0.6143`, Off1 `0.9187`

### Why this still matters despite FAIL
- Infrastructure and routing now align much better with real app semantics:
  - calibration by response type + language, not by platform branding.
- This removes a major conceptual mismatch noted in-session.
- Remaining work is now mostly numerical optimization under fixed gates, not architecture gap.

### Explicit next-session plan (carry-forward)
1. Keep paper-first constraints active (required).
2. Add `paper_general_v4.1` micro-edits (minimal anchor text adjustments at 3/4 intensity only).
3. Explore piecewise calibration (two-segment monotonic mapping with continuity) to improve MAE without sacrificing Off1 CI.
4. Run full protocol only (13k rows, bootstrap 1000, split-half 300).
5. Replace runtime artifacts **only** when new run strictly improves best-known row per agreed objective.

### Operational notes
- Full external runs require out-of-sandbox network access to HF.
- In sandbox/offline mode, scripts can skip datasets or stall on model/data fetch; prefer escalated runs for canonical results.
- Do not interpret smoke-run metrics as decision-grade.

### Final state at session end
- Stage: **Stage 2B** (typed-domain calibration alignment with app purpose)
- Decision status: still **not production-ready** under hard gates.
- Key achievement: architecture and calibration routing now reflect app use-case and paper methodology more faithfully.

## Follow-up (2026-02-08) - Stage 2C (v4.1 anchors + piecewise calibration)

### Session objective
- Execute the carry-forward plan from Stage 2B without changing hard gates:
  - add `paper_general_v4.1` minimal anchor edits (only intensity 3/4 phrasing),
  - compare isotonic vs piecewise domain calibration on full protocol.

### Method constraints kept fixed
- Paper-first constraints preserved.
- Runtime-consistent SSR params preserved:
  - `T=1.0`, `eps=0.0`
- Typed-domain routing unchanged.
- Full protocol used for decision-grade validation:
  - 13k rows (4k Amazon, 3k Yelp, 3k App, 3k Allegro),
  - bootstrap `1000`,
  - split-half `300`.

### Code changes completed
- Added new anchor variant `paper_general_v4.1`:
  - file: `backend/app/i18n.py`
  - scope: minimal micro-edits focused on scale `3` / `4` wording in PL+EN.
- Added piecewise calibration support (continuous two-segment isotonic):
  - file: `backend/app/services/score_calibration.py`
  - new calibrator type: `piecewise_isotonic_v1`
  - runtime policy loader now supports both `isotonic_v1` and `piecewise_isotonic_v1`.
- Extended domain policy builder with calibration-mode controls:
  - file: `backend/scripts/build_domain_calibration_policy.py`
  - new CLI flags:
    - `--calibration-mode {isotonic,piecewise}`
    - `--piecewise-split-quantile`
- Tests updated:
  - `backend/tests/test_i18n_anchors.py`
  - `backend/tests/test_score_calibration.py`

### Tests run (local)
- `pytest -q backend/tests/test_i18n_anchors.py backend/tests/test_score_calibration.py`
  - result: `9 passed`
- `pytest -q backend/tests/test_ssr_engine.py backend/tests/test_score_calibration.py`
  - result: `15 passed`

### Full runs executed
1. `v4.1 + typed MAE policy + isotonic`
   - policy artifact:
     - `reports/ssr_policy_v41_typed_full_mae_isotonic.json`
   - full validation report:
     - `reports/production_readiness_validation_2026-02-08_stage2_v41_typed_full_mae_isotonic.md`
   - best policy row (`domain_policy_artifact`):
     - MAE `0.6126`
     - Off1 `0.9182`
     - Spearman `0.7677`
   - gates:
     - MAE gate: fail
     - Off1 gate: fail
     - Spearman-drop gate: pass
     - max dataset MAE regression gate: pass
   - decision: `FAIL`

2. `v4.1 + typed MAE policy + piecewise`
   - policy artifact:
     - `reports/ssr_policy_v41_typed_full_mae_piecewise.json`
   - full validation report:
     - `reports/production_readiness_validation_2026-02-08_stage2_v41_typed_full_mae_piecewise.md`
   - best policy row (`domain_policy_artifact`):
     - MAE `0.5797`
     - Off1 `0.9156`
     - Spearman `0.7687`
   - gates:
     - MAE gate: pass
     - Off1 gate: fail
     - Spearman-drop gate: pass
     - max dataset MAE regression gate: pass
   - decision: `FAIL`

### Comparison vs current runtime best (typed v4 MAE)
- Current runtime best baseline:
  - report: `reports/production_readiness_validation_2026-02-08_stage2_v4_typed_full_mae.md`
  - MAE `0.6143`, Off1 `0.9187`, Spearman `0.7674`
- Isotonic v4.1:
  - MAE improved (`0.6143 -> 0.6126`),
  - Off1 decreased (`0.9187 -> 0.9182`).
- Piecewise v4.1:
  - MAE improved strongly (`0.6143 -> 0.5797`),
  - Off1 decreased more (`0.9187 -> 0.9156`).

### Runtime replacement decision
- **No runtime artifact replacement performed** in this session.
- Reason: neither candidate satisfies the hard Off1 gate (`>= 0.92`) and neither is a strict overall gate-closing improvement.

### Additional experiment: MAE objective with explicit Off1 guard (2026-02-08)
- Implemented Off1-guard controls in policy builder:
  - `--mae-off1-floor`
  - `--mae-off1-max-drop`
- Run executed:
  - policy: `reports/ssr_policy_v41_typed_full_mae_piecewise_off1guard.json`
  - validation: `reports/production_readiness_validation_2026-02-08_stage2_v41_typed_full_mae_piecewise_off1guard.md`
- Result:
  - metrics were unchanged vs plain piecewise run:
    - MAE `0.5797`
    - Off1 `0.9156`
    - Spearman `0.7687`
  - decision: `FAIL`
- Interpretation:
  - Off1 guard was non-binding on the internal holdouts for trained domains (`best_alpha=1.0`),
  - so it did not alter the fitted calibrators in this setup.

### Sequential micro-edit run: `paper_general_v4.1a` (2026-02-08)
- Change scope:
  - one micro-edit at intensity level 4 in anchor set #1 (PL+EN),
  - `v4.1a` reverted this phrase to stronger "probably buy" wording vs `v4.1` "leaning toward buying".
- Artifacts:
  - policy: `reports/ssr_policy_v41a_typed_full_mae_piecewise.json`
  - report: `reports/production_readiness_validation_2026-02-08_stage2_v41a_typed_full_mae_piecewise.md`
- Best policy metrics:
  - MAE `0.5841`
  - Off1 `0.9138`
  - Spearman `0.7659`
  - Decision: `FAIL`
- Comparison:
  - vs `v4.1 piecewise`: MAE worse (`0.5797 -> 0.5841`), Off1 worse (`0.9156 -> 0.9138`)
  - vs runtime typed v4 MAE: MAE better (`0.6143 -> 0.5841`), Off1 worse (`0.9187 -> 0.9138`)
- Runtime decision:
  - no replacement.

### Sequential micro-edit run: `paper_general_v4.1b` (2026-02-08)
- Change scope:
  - one micro-edit at intensity level 4 in anchor set #2 (PL+EN),
  - `v4.1b` tightened level-4 likelihood wording (`dość/fairly likely` -> `raczej/somewhat likely`) while keeping paper-style generic phrasing.
- Artifacts:
  - policy: `reports/ssr_policy_v41b_typed_full_mae_piecewise.json`
  - report: `reports/production_readiness_validation_2026-02-08_stage2_v41b_typed_full_mae_piecewise.md`
- Best policy metrics:
  - MAE `0.5804`
  - Off1 `0.9178`
  - Spearman `0.7697`
  - Decision: `FAIL`
- Comparison:
  - vs `v4.1 piecewise`: MAE slightly worse (`0.5797 -> 0.5804`), Off1 better (`0.9156 -> 0.9178`)
  - vs `v4.1a piecewise`: better on both MAE and Off1
  - vs runtime typed v4 MAE: MAE much better (`0.6143 -> 0.5804`), Off1 still lower (`0.9187 -> 0.9178`)
- Runtime decision:
  - no replacement (Off1 gate still below `0.92`).

### Sequential micro-edit run: `paper_general_v4.1c` (2026-02-08)
- Change scope:
  - one micro-edit at intensity level 4 in anchor set #3 (PL+EN),
  - shifted level-4 wording toward likelihood phrasing while staying short and domain-neutral.
- Artifacts:
  - policy: `reports/ssr_policy_v41c_typed_full_mae_piecewise.json`
  - report: `reports/production_readiness_validation_2026-02-08_stage2_v41c_typed_full_mae_piecewise.md`
- Best policy metrics:
  - MAE `0.5833`
  - Off1 `0.9121`
  - Spearman `0.7665`
  - Decision: `FAIL`
- Comparison:
  - vs `v4.1b`: worse MAE and worse Off1
  - vs `v4.1`: worse MAE and worse Off1
  - vs runtime typed v4 MAE: better MAE, worse Off1
- Runtime decision:
  - no replacement.

## Runtime decision update (2026-02-08, post v4.1x series)
- Applied `paper_general_v4.1b` as active default anchor variant:
  - `backend/app/i18n.py` -> `DEFAULT_ANCHOR_VARIANT = "paper_general_v4.1b"`
- Applied `v4.1b` typed piecewise policy as active runtime default:
  - `backend/app/data/ssr_calibration_policy_default.json`
  - source artifact: `reports/ssr_policy_v41b_typed_full_mae_piecewise.json`
- Rationale:
  - among tested `v4.1/v4.1a/v4.1b/v4.1c`, `v4.1b` gave best Off1 while keeping MAE near the best MAE regime.

## Follow-up (2026-02-08) - Steps 1+2+3 execution (off1-constrained objective, piecewise3 purchase, micro-sweep)

### Step 1 implemented: constrained objective
- `backend/scripts/build_domain_calibration_policy.py` extended with:
  - `--optimize off1_constrained_mae`
  - objective mix sampled from external benchmark domains
  - objective constraint: Off1 floor (`--objective-off1-min`, default `0.92`)
  - objective selector over blend alpha with MAE minimization under Off1 constraint

### Step 2 implemented: 3-segment calibration for purchase domains
- `backend/app/services/score_calibration.py` extended with:
  - `piecewise3_isotonic_v1` calibrator type
  - fit function with continuity across both split boundaries
  - runtime policy loader compatibility for new type
- `backend/scripts/build_domain_calibration_policy.py` extended with:
  - `--purchase-calibration-mode` (default `piecewise3`)
  - `--piecewise3-split-q1`, `--piecewise3-split-q2`

### Step 1+2 full run result (`v4.1b`)
- Policy artifact:
  - `reports/ssr_policy_v41b_typed_full_off1constr_piecewise3.json`
- Validation report:
  - `reports/production_readiness_validation_2026-02-08_stage2_v41b_off1constr_piecewise3.md`
- Best policy metrics:
  - MAE `0.5914`
  - Off1 `0.9164`
  - Spearman `0.7582`
  - Decision: `FAIL`
- Comparison vs baseline `v4.1b piecewise+mae`:
  - baseline: MAE `0.5804`, Off1 `0.9178`
  - constrained+piecewise3 run is worse on both MAE and Off1

### Step 3 implemented: local micro-sweep around `v4.1b` (single level-4 edits)
1. `v4.1d` (single level-4 edit in anchor set #4)
   - policy: `reports/ssr_policy_v41d_typed_full_off1constr_piecewise3.json`
   - report: `reports/production_readiness_validation_2026-02-08_stage2_v41d_off1constr_piecewise3.md`
   - result: MAE `0.5985`, Off1 `0.9121`, Spearman `0.7594` (`FAIL`)
2. `v4.1e` (single level-4 edit in anchor set #6)
   - policy: `reports/ssr_policy_v41e_typed_full_off1constr_piecewise3.json`
   - report: `reports/production_readiness_validation_2026-02-08_stage2_v41e_off1constr_piecewise3.md`
   - result: MAE `0.6298`, Off1 `0.9129`, Spearman `0.7581` (`FAIL`)

### Outcome and runtime decision
- None of steps 1+2+3 candidates improved the selected runtime baseline (`v4.1b piecewise+mae`).
- Runtime defaults should remain unchanged:
  - anchors default: `paper_general_v4.1b`
  - policy default artifact from `reports/ssr_policy_v41b_typed_full_mae_piecewise.json`

## Follow-up (2026-02-08) - Domain-specific alpha + piecewise3 quantile-grid rerun

### What was changed
- Step 1 refined:
  - `off1_constrained_mae` now searches objective on domain-mix sample with **domain-specific alphas**:
    - separate alpha for `general`, `purchase_intent_short_en`, `purchase_intent_short_pl`
- Step 2 refined:
  - purchase `piecewise3` now supports **quantile-grid search** over `(q1, q2)` candidates.
- Step 3 rerun:
  - micro-sweep around `v4.1b` using this new method (`v4.1b`, `v4.1d`, `v4.1e`).

### Full-run results (new method)
1. `v4.1b`:
   - policy: `reports/ssr_policy_v41b_typed_full_off1constr_piecewise3_domainalpha.json`
   - report: `reports/production_readiness_validation_2026-02-08_stage2_v41b_off1constr_piecewise3_domainalpha.md`
   - MAE `0.6188`, Off1 `0.9199`, Spearman `0.7587` (`FAIL`)
2. `v4.1d`:
   - policy: `reports/ssr_policy_v41d_typed_full_off1constr_piecewise3_domainalpha.json`
   - report: `reports/production_readiness_validation_2026-02-08_stage2_v41d_off1constr_piecewise3_domainalpha.md`
   - MAE `0.6143`, Off1 `0.9147`, Spearman `0.7548` (`FAIL`)
3. `v4.1e`:
   - policy: `reports/ssr_policy_v41e_typed_full_off1constr_piecewise3_domainalpha.json`
   - report: `reports/production_readiness_validation_2026-02-08_stage2_v41e_off1constr_piecewise3_domainalpha.md`
   - MAE `0.6131`, Off1 `0.9130`, Spearman `0.7603` (`FAIL`)

### Diagnostic observation
- Objective optimizer selected same alpha pattern in all runs:
  - `alpha_general = alpha_purchase_en = alpha_purchase_pl = 1.0`
- Best quantile pair selected consistently:
  - `(q1, q2) = (0.45, 0.80)`
- This indicates current objective search remained effectively non-binding on alpha and mostly shifted quantiles.

### Decision
- New domain-alpha + quantile-grid method did **not** beat baseline `v4.1b piecewise+mae` (`MAE 0.5804`, `Off1 0.9178`).
- Runtime defaults remain unchanged.

## Follow-up (2026-02-08) - Trust region trial

### Implementation
- Added `trust_region_v1` calibrator wrapper:
  - file: `backend/app/services/score_calibration.py`
  - behavior: limits calibrated shift relative to raw score by per-domain `max_delta`.
- Added trust-region controls to policy builder:
  - file: `backend/scripts/build_domain_calibration_policy.py`
  - flags:
    - `--trust-region-delta-general`
    - `--trust-region-delta-purchase-en`
    - `--trust-region-delta-purchase-pl`
    - `--trust-region-delta-purchase-agg`
- Added tests:
  - `backend/tests/test_score_calibration.py` (trust region roundtrip + bounds assertions)

### Run executed
- Policy:
  - `reports/ssr_policy_v41b_typed_full_mae_piecewise_trustregion_v1.json`
- Validation:
  - `reports/production_readiness_validation_2026-02-08_stage2_v41b_typed_full_mae_piecewise_trustregion_v1.md`
- Deltas used:
  - general `0.35`
  - purchase EN `0.20`
  - purchase PL `0.20`
  - purchase aggregate `0.20`

### Result
- Best policy row:
  - MAE `0.7965`
  - Off1 `0.9266`
  - Spearman `0.7244`
  - Decision: `FAIL`
- Interpretation:
  - trust region can push Off1 above gate,
  - but with these deltas it over-degraded MAE.
- Runtime decision:
  - no replacement.

### Trust-region quick grid (near no-trust)
- Base policy used: `v4.1b piecewise+mae`
  - `reports/ssr_policy_v41b_typed_full_mae_piecewise.json`
- Grid executed (general_delta, purchase_delta):
  1. `g12_p10` -> (1.2, 1.0)
     - report: `reports/production_readiness_validation_2026-02-08_stage2_v41b_trustgrid_g12_p10.md`
     - MAE `0.5824`, Off1 `0.9178`
  2. `g10_p08` -> (1.0, 0.8)
     - report: `reports/production_readiness_validation_2026-02-08_stage2_v41b_trustgrid_g10_p08.md`
     - MAE `0.6094`, Off1 `0.9179`
  3. `g09_p07` -> (0.9, 0.7)
     - report: `reports/production_readiness_validation_2026-02-08_stage2_v41b_trustgrid_g09_p07.md`
     - MAE `0.6330`, Off1 `0.9183`
  4. `g08_p06` -> (0.8, 0.6)
     - report: `reports/production_readiness_validation_2026-02-08_stage2_v41b_trustgrid_g08_p06.md`
     - MAE `0.6606`, Off1 `0.9203`
- Baseline reference:
  - `reports/production_readiness_validation_2026-02-08_stage2_v41b_typed_full_mae_piecewise.md`
  - MAE `0.5804`, Off1 `0.9178`

### Grid conclusion
- Lower deltas increase Off1 but strongly inflate MAE.
- No grid point satisfies both hard gates (`MAE <= 0.60` and `Off1 >= 0.92`).
- Closest trust-grid compromise is `g12_p10`, but it is not better than baseline.

## Follow-up (2026-02-08) - Option 1 trial (entropy-aware / temperature-aware calibration)

### Implementation
- Added `entropy_aware_v1` calibrator:
  - file: `backend/app/services/score_calibration.py`
  - behavior: routes to low/high branch by normalized PMF entropy threshold.
- Extended SSR runtime to pass uncertainty signal (normalized entropy) into calibrator transform:
  - file: `backend/app/services/ssr_engine.py`
- Extended training/validation scoring paths to produce uncertainty from PMF:
  - files:
    - `backend/scripts/build_domain_calibration_policy.py`
    - `backend/scripts/validate_production_readiness.py`
- Added tests:
  - `backend/tests/test_score_calibration.py` (entropy-aware roundtrip/switch behavior)

### Run executed
- Policy artifact:
  - `reports/ssr_policy_v41b_typed_full_mae_entropy_piecewise_q60.json`
- Validation report:
  - `reports/production_readiness_validation_2026-02-08_stage2_v41b_typed_full_mae_entropy_piecewise_q60.md`
- Metrics (best policy row):
  - MAE `0.6554`
  - Off1 `0.9053`
  - Spearman `0.7318`
  - Decision: `FAIL`

### Conclusion
- In this first configuration, entropy-aware calibration underperformed strongly versus baseline `v4.1b piecewise+mae` (`0.5804 / 0.9178`).
- No runtime replacement.

## Detailed Experiment Registry (2026-02-08)

### Canonical baseline to beat
- Anchor variant: `paper_general_v4.1b`
- Policy artifact: `reports/ssr_policy_v41b_typed_full_mae_piecewise.json`
- Validation report: `reports/production_readiness_validation_2026-02-08_stage2_v41b_typed_full_mae_piecewise.md`
- Metrics:
  - MAE `0.5804`
  - Off1 `0.9178`
  - Spearman `0.7697`
- Status:
  - best practical compromise in current branch
  - still `FAIL` vs hard gates (`MAE <= 0.60`, `Off1 >= 0.92`)

### Runtime defaults currently active
- `backend/app/i18n.py`
  - `DEFAULT_ANCHOR_VARIANT = "paper_general_v4.1b"`
- `backend/app/data/ssr_calibration_policy_default.json`
  - sourced from: `reports/ssr_policy_v41b_typed_full_mae_piecewise.json`
- Global calibrator path unchanged from prior stage:
  - `backend/app/data/ssr_calibrator_default.json`

### Full validation reports generated in this continuation
1. `reports/production_readiness_validation_2026-02-08_stage2_v41b_off1constr_piecewise3.md`
   - MAE `0.5914`, Off1 `0.9164`, Spearman `0.7582`
2. `reports/production_readiness_validation_2026-02-08_stage2_v41d_off1constr_piecewise3.md`
   - MAE `0.5985`, Off1 `0.9121`, Spearman `0.7594`
3. `reports/production_readiness_validation_2026-02-08_stage2_v41e_off1constr_piecewise3.md`
   - MAE `0.6298`, Off1 `0.9129`, Spearman `0.7581`
4. `reports/production_readiness_validation_2026-02-08_stage2_v41b_off1constr_piecewise3_domainalpha.md`
   - MAE `0.6188`, Off1 `0.9199`, Spearman `0.7587`
5. `reports/production_readiness_validation_2026-02-08_stage2_v41d_off1constr_piecewise3_domainalpha.md`
   - MAE `0.6143`, Off1 `0.9147`, Spearman `0.7548`
6. `reports/production_readiness_validation_2026-02-08_stage2_v41e_off1constr_piecewise3_domainalpha.md`
   - MAE `0.6131`, Off1 `0.9130`, Spearman `0.7603`
7. `reports/production_readiness_validation_2026-02-08_stage2_v41b_typed_full_mae_piecewise_trustregion_v1.md`
   - MAE `0.7965`, Off1 `0.9266`, Spearman `0.7244`
8. `reports/production_readiness_validation_2026-02-08_stage2_v41b_trustgrid_g12_p10.md`
   - MAE `0.5824`, Off1 `0.9178`, Spearman `0.7696`
9. `reports/production_readiness_validation_2026-02-08_stage2_v41b_trustgrid_g10_p08.md`
   - MAE `0.6094`, Off1 `0.9179`, Spearman `0.7666`
10. `reports/production_readiness_validation_2026-02-08_stage2_v41b_trustgrid_g09_p07.md`
    - MAE `0.6330`, Off1 `0.9183`, Spearman `0.7664`
11. `reports/production_readiness_validation_2026-02-08_stage2_v41b_trustgrid_g08_p06.md`
    - MAE `0.6606`, Off1 `0.9203`, Spearman `0.7655`
12. `reports/production_readiness_validation_2026-02-08_stage2_v41b_typed_full_mae_entropy_piecewise_q60.md`
    - MAE `0.6554`, Off1 `0.9053`, Spearman `0.7318`

### Policy artifacts generated in this continuation
- `reports/ssr_policy_v41b_typed_full_off1constr_piecewise3.json`
- `reports/ssr_policy_v41d_typed_full_off1constr_piecewise3.json`
- `reports/ssr_policy_v41e_typed_full_off1constr_piecewise3.json`
- `reports/ssr_policy_v41b_typed_full_off1constr_piecewise3_domainalpha.json`
- `reports/ssr_policy_v41d_typed_full_off1constr_piecewise3_domainalpha.json`
- `reports/ssr_policy_v41e_typed_full_off1constr_piecewise3_domainalpha.json`
- `reports/ssr_policy_v41b_typed_full_mae_piecewise_trustregion_v1.json`
- `reports/ssr_policy_v41b_typed_full_mae_piecewise_trustgrid_g12_p10.json`
- `reports/ssr_policy_v41b_typed_full_mae_piecewise_trustgrid_g10_p08.json`
- `reports/ssr_policy_v41b_typed_full_mae_piecewise_trustgrid_g09_p07.json`
- `reports/ssr_policy_v41b_typed_full_mae_piecewise_trustgrid_g08_p06.json`
- `reports/ssr_policy_v41b_typed_full_mae_entropy_piecewise_q60.json`

### Code modules changed during this continuation
- `backend/app/services/score_calibration.py`
  - added: `piecewise3_isotonic_v1`, `trust_region_v1`, `entropy_aware_v1`
  - loader now supports these calibrator types.
- `backend/scripts/build_domain_calibration_policy.py`
  - added: constrained objective variants, purchase mode override, quantile grids, domain-specific alpha search, trust-region flags, entropy-aware training path.
- `backend/scripts/validate_production_readiness.py`
  - scoring path now computes uncertainty (PMF entropy) and passes it into calibrators.
- `backend/app/services/ssr_engine.py`
  - runtime path now computes per-response PMF entropy and passes uncertainty into calibrator `transform()`.
- `backend/app/i18n.py`
  - added anchor variants: `paper_general_v4.1d`, `paper_general_v4.1e`
  - default remains `paper_general_v4.1b`.
- tests:
  - `backend/tests/test_score_calibration.py`
  - `backend/tests/test_i18n_anchors.py`

### Validation protocol consistency (preserved)
- anchor methodology: paper-style, short generic statements, no direct Likert elicitation
- SSR core constants: `T=1.0`, `eps=0.0`
- external protocol: 13k rows, bootstrap `1000`, split-half `300`
- hard gates unchanged:
  - MAE `<= 0.60`
  - Off1 `>= 0.92`
  - Spearman drop `<= 0.01`
  - max dataset MAE regression `<= 0.10`

### Practical takeaway
- Current frontier remains a tight MAE/Off1 trade-off:
  - trust-region can push Off1 up but quickly raises MAE,
  - entropy-aware variant (tested at q60) degraded both MAE and Off1,
  - constrained objective / piecewise3 / domain-alpha searches did not beat baseline.
