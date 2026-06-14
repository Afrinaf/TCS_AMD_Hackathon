# README — ScreeningCopilot: Adverse Media & Negative News Screening Copilot

---

## Project Overview

ScreeningCopilot is an end-to-end agentic pipeline that automates adverse media screening for compliance analysts. Given any company or person's name, it retrieves real-time news, detects adverse signals using a hybrid AI approach, scores entity-level risk on a 0–100 scale, and provides a conversational copilot to explain every decision in plain English.

Built for the TCS & AMD AI Hackathon — Challenge AGENTS_001.

---

## The Problem

Compliance analysts at banks, fintechs, and enterprises must screen entities for adverse media — fraud, legal violations, sanctions, ESG failures — before onboarding and continuously afterward. Current process: manual Google searches, inconsistent judgments, 2–4 hours per entity, impossible to scale.

ScreeningCopilot reduces that to under 2 minutes with auditable, explainable, consistent results.

---

## Pipeline Architecture

```
[Entity Name]
      ↓
Phase 1 — News Retrieval Agent
  Google News RSS + NewsData API
  Output: 30 articles → DataFrame (title, source, date, URL, summary)
      ↓
Phase 2 — Adverse Detection Agent
  Keyword layer (fast) + Groq LLaMA 3.3 70B (context-aware)
  Output: is_adverse, adverse_category, confidence_score, reason
      ↓
Phase 3 — Risk Scoring Engine
  4-dimension weighted scoring + diminishing returns aggregation
  Output: entity_risk_score (0–100), risk_label, JSON scorecard
      ↓
Phase 4 — Screening Copilot (Streamlit)
  Groq LLaMA 3.3 grounded on Phase 3 scorecard
  Output: natural language Q&A for analysts
```

---

## Phases

### Phase 1 — News Retrieval (`phase1_news_retrieval.ipynb`)

Retrieves recent news articles for a given entity from public sources.

**Sources:** Google News RSS, NewsData.io API (free tier)

**Output columns:**

| Column | Description |
|---|---|
| title | Article headline |
| source | Publisher name |
| publication_date | Date published |
| article_url | Direct link |
| article_summary | Short summary or RSS description |

**To run:**
```python
entity_name = "Tesla"   # set in Cell 3
# Run all cells — saves news_Tesla_phase1.csv
```

---

### Phase 2 — Adverse Detection (`phase2_adverse_detection.ipynb` + `phase2_patch_groq.ipynb`)

Classifies each article as adverse or non-adverse using a hybrid approach.

**Hybrid detection design:**
- Keyword layer: fast regex patterns catch obvious signals (fraud, recall, lawsuit, crash). Positive override patterns suppress false positives on regulatory approvals and editorial opinions.
- LLM layer: Groq LLaMA 3.3 reads each article in full context and classifies category, confidence, and reason. Catches nuanced adverse signals the keyword layer misses.

**Output columns added:**

| Column | Description |
|---|---|
| is_adverse | True / False |
| adverse_category | Operational Risk, Fraud, Legal, Sanctions, ESG, Reputational |
| confidence_score | 0.0 – 1.0 |
| reason | One-sentence compliance-grade explanation |
| detection_layer | keyword or llm |
| article_risk_score | Per-article score (0–100) |

**LLM backend:** Groq API (`llama-3.3-70b-versatile`). HuggingFace Inference API patch is included but Groq is recommended — confirmed reachable in restricted environments.

---

### Phase 3 — Risk Scoring (`phase3_risk_scoring.ipynb`)

Aggregates article-level signals into a single entity risk score.

**Scoring model — per article:**

```
Article Score = (category_severity × 40)
              + (llm_confidence    × 25)
              + (source_credibility× 20)
              + (recency           × 15)
```

**Category severity weights:**

| Category | Severity |
|---|---|
| Sanctions | 1.00 |
| Fraud | 0.95 |
| Legal | 0.80 |
| ESG | 0.70 |
| Reputational | 0.65 |
| Operational Risk | 0.60 |

**Entity-level aggregation — diminishing returns:**

```
Entity Score = normalize(A1×1.00 + A2×0.75 + A3×0.56 + A4×0.42 ...)
```

Prevents score inflation when multiple outlets cover the same single event.

**Risk labels:**

| Score | Label |
|---|---|
| 85–100 | Critical |
| 65–84 | High |
| 35–64 | Medium |
| 0–34 | Low |

**Output files:**
- `risk_scored_<entity>_phase3.csv` — article-level data with all score dimensions
- `risk_scorecard_<entity>_phase3.json` — full entity scorecard (consumed by Phase 4)

---

### Phase 4 — Screening Copilot (`phase4_copilot_app.py`)

Streamlit chat application. Analysts upload the Phase 3 scorecard and ask questions in plain English.

**Capabilities:**

| Question type | Example |
|---|---|
| Risk reason | "Why is Tesla rated Medium risk?" |
| Driving articles | "Which articles drove the score?" |
| Risk categories | "What categories of risk were detected?" |
| Source credibility | "How credible are the sources?" |
| Next steps | "What due diligence actions should I take?" |
| Score explanation | "How was the 52.17 score calculated?" |
| Executive summary | "Give me a summary." |

**Key design decisions:**

**Grounded generation** — the LLM is explicitly instructed to answer only from the Phase 3 scorecard context. It cannot use training memory. This eliminates hallucination of fake articles or fabricated scores in a compliance context.

**Intent routing** — user questions are pre-classified by regex before hitting the LLM. A focused prefix is injected to steer the model to the correct section of the scorecard. Improves answer precision and reduces hallucination further.

**Multi-turn conversation** — full conversation history is sent with every Groq call, enabling follow-up questions like "which of those had the highest score?"

**Scorecard normaliser** — automatically maps whatever key names Phase 3 produced (flat or nested JSON) to the canonical schema the app expects. No manual editing of JSON required.

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- Jupyter Notebook or JupyterLab (Phases 1–3)
- A free Groq API key: https://console.groq.com

### Install dependencies

**Phases 1–3 (notebooks):**
```bash
pip install feedparser requests pandas python-dateutil newspaper3k groq
```

**Phase 4 (Streamlit app):**
```bash
pip install streamlit groq pandas
```

### Configure API keys

**Groq (required for Phases 2 and 4):**
```bash
export GROQ_API_KEY="gsk_your_key_here"
```

**NewsData.io (optional, Phase 1):**
Register at https://newsdata.io — free tier gives 200 requests/day. Add the key in Phase 1 Cell 3.

---

## Running the Pipeline

**Phase 1 — retrieve news:**
```bash
jupyter notebook phase1_news_retrieval.ipynb
# Set entity_name in Cell 3, then Run All
```

**Phase 2 — detect adverse news:**
```bash
jupyter notebook phase2_adverse_detection.ipynb
# Loads Phase 1 output automatically, then Run All
```

**Phase 3 — score risk:**
```bash
jupyter notebook phase3_risk_scoring.ipynb
# Loads Phase 2 output automatically, then Run All
```

**Phase 4 — launch copilot:**
```bash
export GROQ_API_KEY="gsk_your_key_here"
streamlit run phase4_copilot_app.py
# Opens at http://localhost:8501
# Upload Phase 3 JSON and CSV via sidebar
```

---

## File Structure

```
screeningcopilot/
├── phase1_news_retrieval.ipynb         # News retrieval agent
├── phase2_adverse_detection.ipynb      # Adverse detection (HuggingFace)
├── phase2_patch_groq.ipynb             # Adverse detection (Groq — recommended)
├── phase3_risk_scoring.ipynb           # Risk scoring engine
├── phase4_copilot_app.py               # Streamlit copilot app
│
├── outputs/                            # Generated by running the pipeline
│   ├── news_<entity>_phase1.csv
│   ├── adverse_detected_<entity>_phase2.csv
│   ├── risk_scored_<entity>_phase3.csv
│   └── risk_scorecard_<entity>_phase3.json
│
└── README.md
```

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | LLaMA 3.3 70B via Groq API |
| News retrieval | Google News RSS, NewsData.io |
| Data processing | Python, pandas |
| Notebooks | Jupyter |
| Copilot UI | Streamlit |
| Key libraries | feedparser, groq SDK, newspaper3k, python-dateutil |

---

## End-to-End Latency

| Phase | Time |
|---|---|
| Phase 1 — 30 articles retrieved | ~10 seconds |
| Phase 2 — 30 articles classified | ~26 seconds |
| Phase 3 — scoring | < 1 second |
| Phase 4 — copilot response | 1–2 seconds |
| **Total new entity** | **~40 seconds** |

No local GPU required. All LLM inference runs on Groq's cloud hardware.

---

## Key Design Decisions

**Hybrid adverse detection** — keyword rules give speed and zero API cost for obvious signals. The LLM gives context-awareness for ambiguous cases. Neither alone achieves production accuracy.

**Diminishing returns aggregation** — a single event covered by 15 outlets should not score 15× higher than one covered by 1 outlet. The diminishing returns formula treats correlated coverage correctly.

**Grounded copilot** — in compliance, a hallucinated citation is a liability. The Phase 4 LLM is forbidden from answering outside the scorecard data. If it doesn't know, it says so.

**Source credibility weighting** — a Bloomberg article carries a credibility score of 1.0; an unknown blog defaults to 0.30. Same event, different weight.

**Explainability at every layer** — every adverse article has a reason field. Every entity score traces back to four numeric dimensions. Every copilot answer cites the article it drew from.

---

## Output Screenshot

<img width="1919" height="960" alt="Screenshot 2026-06-14 223114" src="https://github.com/user-attachments/assets/c538f988-3df6-49a4-9990-ae725d85d3d6" />


---

## Limitations

- English-language news sources only (current implementation)
- No entity resolution — "Elon Musk" and "Tesla CEO" are treated as different entities
- NewsData.io free tier: 200 requests/day
- Groq free tier: 1,000 requests/day on `llama-3.3-70b-versatile`

---

## Future Work

- Entity resolution across name variants and aliases
- Multilingual news sources (Arabic, Mandarin, Spanish)
- Regulatory database integration — OFAC, FCA, SEC EDGAR
- Continuous monitoring with scheduled re-screening and score-change alerts
- Analyst feedback loop to improve classification over time
- Batch screening for multiple entities in parallel

---

## Hackathon

**Challenge:** AGENTS_001 — Adverse Media / Negative News Screening Copilot
**Hackathon:** TCS & AMD AI Hackathon 2026
