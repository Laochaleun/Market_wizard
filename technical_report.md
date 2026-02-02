# Technical Report

## SSR Comparison Findings (BYD 3, 2026-02-01, N=80)

### Key conclusions
- **Embedding choice dominates SSR output.** Switching from local MiniLM to OpenAI `text-embedding-3-small` shifts mean purchase intent by ~+0.5 to +0.7 and moves mass from 2–3 toward 4–5. This is the primary driver of variance.
- **LLM choice (Gemini vs GPT‑4o) is secondary.** Differences between LLMs are modest relative to embedding choice (low L1/KS deltas).
- **Intent‑only vs full text matters.** SSR computed on full multi‑part responses diverges meaningfully from SSR computed on the short purchase‑intent statement, especially under OpenAI embeddings.

### Quantitative summary
All results below are for the same 80 personas (“BYD 3”) at temperature 0.5.

#### Short prompt (paper‑style) — LLM vs LLM, 2 embeddings
- **Local embeddings (all‑MiniLM‑L6‑v2):**
  - Gemini vs GPT‑4o: L1=0.0775, KS=0.0388
- **OpenAI embeddings (text‑embedding‑3‑small):**
  - Gemini vs GPT‑4o: L1=0.1376, KS=0.0432

#### Full prompt (3‑part responses) — LLM vs LLM, 2 embeddings
- **Local embeddings (all‑MiniLM‑L6‑v2):**
  - Gemini vs GPT‑4o: L1=0.0502, KS=0.0193
- **OpenAI embeddings (text‑embedding‑3‑small):**
  - Gemini vs GPT‑4o: L1=0.0934, KS=0.0344

#### Intent‑only vs full text (same LLM)
- **Gemini / Local:** L1=0.1423, KS=0.0698  
- **Gemini / OpenAI:** L1=0.2382, KS=0.0800  
- **GPT‑4o / Local:** L1=0.0572, KS=0.0282  
- **GPT‑4o / OpenAI:** L1=0.2463, KS=0.1232  

### Interpretation
- **Embedding model choice is the biggest lever** in SSR outcomes. It changes similarity structure and thus distributional mass on Likert anchors.
- **LLM choice matters less** than embedding choice for SSR distributions under this setup.
- **To align with the paper**, SSR should use the short purchase‑intent statement (single‑question elicitation), while longer multi‑part answers should be reserved for qualitative insights.
> **Clarification:** In the earlier “0% ‘no’” case (BYD 2), the zero tail occurred under **local embeddings** with a long, multi‑part response and sharp temperature — not because of OpenAI embeddings. The issue was prompt/elicitation + scoring mode, not embedding alone.

### Operational recommendations
- Keep **SSR scoring on intent‑only** text; use full responses only for qualitative “market‑insight” sections.
- If results must be comparable to the paper, prefer **OpenAI text‑embedding‑3‑small**; if local‑only is required, treat MiniLM results as a separate regime and avoid mixing benchmarks.
- When comparing runs, **hold embedding model constant**; otherwise distribution shifts can be misattributed to product changes.
- **Implemented:** SSR now uses intent‑only text for scoring (short elicitation), while full responses remain in qualitative sections.

### Artifacts
- Short prompt comparison:
  - `reports/byd3_llm_ssr_compare_short.json`
  - `reports/byd3_llm_ssr_compare_short.png`
- Full prompt comparison (includes intent vs full metrics):
  - `reports/byd3_llm_ssr_compare_full.json`
  - `reports/byd3_llm_ssr_compare_full.png`
