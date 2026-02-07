# SSR Embedding Benchmark Report (2026-02-07)

## Scope
- Goal: compare SSR quality for two local embedding models in Market Wizard.
- Models:
  - `all-MiniLM-L6-v2`
  - `BAAI/bge-m3`
- Metrics:
  - MAE (lower is better)
  - Spearman correlation (higher is better)
  - Exact accuracy after rounding to 1..5
  - Off-by-one accuracy

## Dataset A (single-domain sanity check)
- Source: `yelp_review_full` (Hugging Face)
- Size: 1200 real reviews (EN), labels 1..5
- Settings: `temperature=1.0`, `epsilon=0.0`, EN anchors

### Results
- `all-MiniLM-L6-v2`
  - MAE: `1.1371`
  - Spearman: `0.5915`
  - Exact accuracy: `0.2292`
  - Off-by-one accuracy: `0.7033`
- `BAAI/bge-m3`
  - MAE: `0.7750`
  - Spearman: `0.8408`
  - Exact accuracy: `0.3442`
  - Off-by-one accuracy: `0.9192`
- Embedding impact (`all-MiniLM-L6-v2` vs `BAAI/bge-m3`):
  - Mean `|Δ score|`: `0.5434`
  - Median `|Δ score|`: `0.4674`
  - Max `|Δ score|`: `1.6601`

## Dataset B (20 industries)
- Source: `McAuley-Lab/Amazon-Reviews-2023`, 20 category files:
  - Electronics, Books, Clothing_Shoes_and_Jewelry, Home_and_Kitchen, Automotive,
    Sports_and_Outdoors, Health_and_Personal_Care, Beauty_and_Personal_Care,
    Toys_and_Games, Video_Games, Office_Products, Pet_Supplies,
    Grocery_and_Gourmet_Food, Appliances, Tools_and_Home_Improvement,
    Cell_Phones_and_Accessories, Musical_Instruments, Software,
    Patio_Lawn_and_Garden, Industrial_and_Scientific
- Sampling: 150 reviews/category (streaming reservoir), total 3000
- Settings: `temperature=1.0`, `epsilon=0.0`, EN anchors

### Aggregate results
- `all-MiniLM-L6-v2`
  - MAE: `1.3775`
  - Spearman: `0.3575`
  - Exact accuracy: `0.1043`
  - Off-by-one accuracy: `0.5327`
- `BAAI/bge-m3`
  - MAE: `0.9602`
  - Spearman: `0.6387`
  - Exact accuracy: `0.1857`
  - Off-by-one accuracy: `0.9460`
- Embedding impact (`all-MiniLM-L6-v2` vs `BAAI/bge-m3`):
  - Mean `|Δ score|`: `0.5106`
  - Median `|Δ score|`: `0.4638`
  - Max `|Δ score|`: `1.7315`

## Conclusion
- On both the single-domain and 20-industry benchmark, `BAAI/bge-m3` is consistently and substantially closer to real labels than `all-MiniLM-L6-v2`.
- Recommendation: use `BAAI/bge-m3` as default local embedding model in Market Wizard.

## Operational note (HF Spaces)
- HF deployment may not use local `.env`.
- If no environment variable is provided, runtime falls back to defaults from `backend/app/config.py`.
- Therefore the fallback default in code should match the recommended model (`BAAI/bge-m3`).
