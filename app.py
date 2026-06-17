import os, re, json, time, math, warnings, textwrap
from datetime import datetime
from urllib.parse import quote_plus

warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import requests
import feedparser
from dateutil import parser as date_parser
from groq import Groq, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type




NEWSDATA_API_KEY     = os.environ.get("NEWSDATA_API_KEY", "") # your API Key
USE_NEWSDATA_API     = True   # set False to disable
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")  # your API Key
GROQ_MODEL     = "llama-3.3-70b-versatile"
MAX_TOKENS     = 1024
TEMPERATURE_QA = 0.3    # for copilot Q&A
TEMPERATURE_CL = 0.0    # for classification (deterministic)

# Phase 1 settings
MAX_ARTICLES_PER_SOURCE = 20
REQUEST_DELAY           = 0.5
REQUEST_TIMEOUT         = 10

# Phase 2 settings
MAX_INPUT_CHARS      = 800
CONFIDENCE_THRESHOLD = 0.5
API_CALL_DELAY       = 0.3

# Phase 3 settings
WEIGHTS = {
    "category_severity" : 40,
    "llm_confidence"    : 25,
    "source_credibility": 20,
    "recency"           : 15,
}
RECENCY_HALF_LIFE_DAYS = 30
RECENCY_FLOOR          = 0.05
DIMINISHING_BASE       = 0.75

ADVERSE_CATEGORIES = [
    "Fraud & Financial Crime",
    "Legal & Regulatory",
    "Corruption & Bribery",
    "ESG & Environmental",
    "Cybersecurity & Data",
    "Labor & Human Rights",
    "Reputational",
    "Operational Risk",
    "Sanctions & Watchlist",
    "Not Adverse",
]

CATEGORY_SEVERITY = {
    "Sanctions & Watchlist"  : 1.00,
    "Fraud & Financial Crime": 0.95,
    "Corruption & Bribery"   : 0.90,
    "Legal & Regulatory"     : 0.80,
    "Cybersecurity & Data"   : 0.75,
    "Labor & Human Rights"   : 0.70,
    "ESG & Environmental"    : 0.65,
    "Operational Risk"       : 0.60,
    "Reputational"           : 0.40,
    "Not Adverse"            : 0.00,
}

SOURCE_CREDIBILITY = {
    "bloomberg": 1.00, "bloomberg.com": 1.00,
    "reuters": 1.00,
    "the wall street journal": 0.98, "wsj": 0.98,
    "financial times": 0.98, "ft": 0.98,
    "the new york times": 0.95, "nyt": 0.95,
    "associated press": 0.95, "ap": 0.95,
    "bbc": 0.93, "bbc news": 0.93,
    "the guardian": 0.88, "guardian": 0.88,
    "washington post": 0.88,
    "los angeles times": 0.85,
    "fortune": 0.85, "forbes": 0.82, "cnbc": 0.82,
    "yahoo finance": 0.75, "barron's": 0.80,
    "marketwatch": 0.78, "techcrunch": 0.75,
    "the verge": 0.73, "wired": 0.73,
    "king5.com": 0.65, "ktla": 0.65,
    "edmunds": 0.68, "motor1.com": 0.60,
    "autoblog": 0.60, "electrek": 0.60,
    "the drive": 0.58, "sherwood news": 0.55,
    "cleantechnica": 0.55,
    "yahoo autos": 0.50, "fool": 0.50,
    "google news": 0.40, "bing news": 0.40,
}
UNKNOWN_SOURCE_SCORE = 0.30

RISK_THRESHOLDS = {"Critical": 75, "High": 50, "Medium": 25, "Low": 0}
RISK_EMOJI      = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}

RECOMMENDED_ACTIONS = {
    "Critical": "🚨 Block onboarding. Escalate immediately to senior compliance.",
    "High"    : "🔴 Senior analyst review required before proceeding.",
    "Medium"  : "🟡 Enhanced due diligence recommended. Monitor for 30 days.",
    "Low"     : "🟢 Standard onboarding. Schedule periodic monitoring.",
}

KEYWORD_TAXONOMY = {
    "Fraud & Financial Crime": [
        r"\bfraud\b", r"\bembezzl", r"\bponzi\b", r"\bmoney launder",
        r"\baccounting fraud\b", r"\binsider trading\b", r"\bsecurities fraud\b",
        r"\bwire fraud\b", r"\bmisappropriat", r"\bdeceptive practice",
    ],
    "Legal & Regulatory": [
        r"\blawsuit\b", r"\blitigation\b", r"\bsued\b", r"\bindictment\b",
        r"\bclass action\b", r"\bsec investigation\b", r"\bregulatory fine\b",
        r"\bpenalt", r"\bviolation\b", r"\bcourt order\b", r"\bprosecuted\b",
        r"\bcriminal charge\b", r"\bdoj\b", r"\bfbi probe\b", r"\bsanctioned\b",
    ],
    "Corruption & Bribery": [
        r"\bbribery\b", r"\bcorruption\b", r"\bkickback\b", r"\bextortion\b",
        r"\bfcpa\b", r"\bgraft\b", r"\billicit payment", r"\bpayoff\b",
    ],
    "ESG & Environmental": [
        r"\bpollution\b", r"\benvironmental violation\b", r"\bgreenwash",
        r"\bhazardous waste\b", r"\boil spill\b", r"\bepa fine\b",
        r"\bsustainability fraud\b",
    ],
    "Cybersecurity & Data": [
        r"\bdata breach\b", r"\bhacked\b", r"\bcyberattack\b", r"\bransom",
        r"\bprivacy violation\b", r"\bdata leak\b", r"\bgdpr fine\b",
        r"\bmalware\b", r"\bphishing\b",
    ],
    "Labor & Human Rights": [
        r"\bchild labor\b", r"\bslave labor\b", r"\bhuman trafficking\b",
        r"\bworker abuse\b", r"\bdiscrimination\b", r"\bharassment\b",
        r"\bunion busting\b", r"\bforced labor\b", r"\bwhistleblower\b",
    ],
    "Reputational": [
        r"\bscandal\b", r"\bcontroversy\b", r"\bmisconduct\b",
        r"\bresigned amid\b", r"\bboycott\b", r"\bbacklash\b",
        r"\bpublic outrage\b", r"\baccused of\b", r"\ballegation\b",
    ],
    "Operational Risk": [
        # ── TIGHTENED from original Phase 2 ──
        # Each pattern now requires a harm-context word near the trigger.
        # This eliminates false positives on positive/neutral mentions.

        # Self-driving: only flag when paired with harm words
        r"\bself-driving.{0,40}(crash|accident|death|injur|fail|recall|ban|probe|halt)\b",
        r"(crash|accident|death|injur|fail|recall|ban|probe).{0,40}\bself-driving\b",

        # Autopilot: only flag when paired with failure/accident words
        r"\bautopilot.{0,40}(crash|malfunction|disengag|death|injur|fail|recall)\b",
        r"(crash|malfunction|death|injur|fail).{0,40}\bautopilot\b",

        # Robotaxi: only flag when paired with service failure or incident words
        r"\brobotaxi.{0,40}(stall|crash|fail|injur|halt|ban|probe|incident|sue|block)\b",
        r"(stall|crash|fail|injur|halt|ban|incident).{0,40}\brobotaxi\b",

        # Unambiguous operational failure terms (no proximity required)
        r"\bproduct recall\b", r"\bsafety failure\b",
        r"\bfatal (crash|accident|collision)\b",
        r"\b(fire|explosion|battery fire)\b.{0,30}\btesla\b",
        r"\btesla.{0,30}\b(fire|explosion|battery fire)\b",
        r"\bproduction halt\b", r"\bmanufacturing defect\b",
    ],
    "Sanctions & Watchlist": [
        r"\bofac\b", r"\bsanctions list\b", r"\bblacklist\b", r"\bwatchlist\b",
        r"\bun sanctions\b", r"\beu sanctions\b", r"\bterrorist financ",
        r"\bproliferation\b", r"\bdebarred\b",
    ],
}

POSITIVE_OVERRIDE_PATTERNS = [
    r"\bgets? (approval|go-ahead|clearance|approved|greenlit)\b",
    r"\b(approved|cleared) (to sell|for sale|by regulator|in [a-z]+)\b",
    r"\b(solves?|solved|breakthrough|bull|bullish|record (profit|revenue|delivery|sales))\b",
    r"\b(launches?|unveils?|releases?|expands?|wins? (contract|deal|bid))\b",
    r"\befficiency (champ|leader|record)\b",
]

INTENT_PATTERNS = {
    "risk_reason"  : r"\b(why|reason|explain|how|what made|flagged|rated|scored)\b",
    "articles"     : r"\b(article|news|source|headline|story|stories|drove|contributing)\b",
    "categories"   : r"\b(categor|type|kind|domain|class|fraud|legal|esg|sanction|operat|reputat)\b",
    "source_cred"  : r"\b(source|credib|reliable|outlet|publisher|bloomberg|reuters|trust)\b",
    "next_steps"   : r"\b(next|action|recommend|due diligence|investigate|escalat|step|do|should)\b",
    "score_explain": r"\b(score|number|point|100|scale|metric|how high|how low|risk level)\b",
    "summary"      : r"\b(summar|overview|brief|tldr|overall|total|snapshot)\b",
}


# ═════════════════════════════════════════════════════════════════════
# PHASE 1 — NEWS RETRIEVAL
# ═════════════════════════════════════════════════════════════════════

def normalize_date(raw: str) -> str:
    if not raw:
        return "Unknown"
    try:
        return date_parser.parse(raw, fuzzy=True).strftime("%Y-%m-%d")
    except Exception:
        return "Unknown"


def fetch_google_news_rss(entity: str, log) -> list:
    articles = []
    url = f"https://news.google.com/rss/search?q={quote_plus(entity)}&hl=en-US&gl=US&ceid=US:en"
    log(f"  📡 Google News RSS: querying for **{entity}**")
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries[:MAX_ARTICLES_PER_SOURCE]:
            source = "Google News"
            if hasattr(entry, "source") and hasattr(entry.source, "title"):
                source = entry.source.title
            raw_summary = getattr(entry, "summary", "") or ""
            summary = re.sub(r"<[^>]+>", "", raw_summary).strip()
            articles.append({
                "title"           : getattr(entry, "title", "No Title"),
                "source"          : source,
                "publication_date": normalize_date(getattr(entry, "published", "")),
                "article_url"     : getattr(entry, "link", ""),
                "article_summary" : summary or "No summary available",
            })
        log(f"  ✅ Google News RSS: {len(articles)} articles retrieved")
    except Exception as e:
        log(f"  ⚠️ Google News RSS error: {e}")
    return articles


def fetch_bing_news_rss(entity: str, log) -> list:
    articles = []
    url = f"https://www.bing.com/news/search?q={quote_plus(entity)}&format=RSS"
    log(f"  📡 Bing News RSS: querying for **{entity}**")
    try:
        resp = requests.get(
            url, timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (compatible; NewsScreener/1.0)"}
        )
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
        for entry in feed.entries[:MAX_ARTICLES_PER_SOURCE]:
            raw_summary = getattr(entry, "summary", "") or ""
            summary = re.sub(r"<[^>]+>", "", raw_summary).strip()
            articles.append({
                "title"           : getattr(entry, "title", "No Title"),
                "source"          : getattr(entry, "source", {}).get("title", "Bing News")
                                    if isinstance(getattr(entry, "source", None), dict)
                                    else "Bing News",
                "publication_date": normalize_date(getattr(entry, "published", "")),
                "article_url"     : getattr(entry, "link", ""),
                "article_summary" : summary or "No summary available",
            })
        log(f"  ✅ Bing News RSS: {len(articles)} articles retrieved")
    except Exception as e:
        log(f"  ⚠️ Bing News RSS error: {e}")
    return articles

def fetch_newsdata_api(entity: str, log) -> list:
    articles = []

    if not NEWSDATA_API_KEY or not USE_NEWSDATA_API:
        log("  ⏭ NewsData API: skipped (no API key set)")
        return articles

    log(f"  📡 NewsData API: querying for **{entity}**")

    params = {
        "apikey"  : NEWSDATA_API_KEY,
        "q"       : entity,
        "language": "en",
        "size"    : min(MAX_ARTICLES_PER_SOURCE, 10),  # free tier max = 10
    }

    try:
        response = requests.get(
            "https://newsdata.io/api/1/news",
            params=params,
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()

        data = response.json()

        if data.get("status") != "success":
            log(f"  ⚠️ NewsData API: non-success status — {data.get('status')}")
            return articles

        for item in data.get("results", [])[:MAX_ARTICLES_PER_SOURCE]:
            summary = (
                item.get("description")
                or item.get("content")
                or "No summary available"
            )
            articles.append({
                "title"           : item.get("title", "No Title"),
                "source"          : item.get("source_id", "NewsData"),
                "publication_date": normalize_date(item.get("pubDate", "")),
                "article_url"     : item.get("link", ""),
                "article_summary" : summary,
            })

        log(f"  ✅ NewsData API: {len(articles)} articles retrieved")

    except requests.exceptions.Timeout:
        log(f"  ⚠️ NewsData API: request timed out after {REQUEST_TIMEOUT}s.")
    except requests.exceptions.HTTPError as e:
        log(f"  ⚠️ NewsData API: HTTP error — {e}")
    except ValueError as e:
        log(f"  ⚠️ NewsData API: failed to parse JSON — {e}")
    except Exception as e:
        log(f"  ⚠️ NewsData API: ERROR — {e}")

    return articles


    


def run_phase1(entity: str, log) -> pd.DataFrame:
    log("### 📰 Phase 1 — News Retrieval")
    all_articles = []
    all_articles.extend(fetch_google_news_rss(entity, log))
    time.sleep(REQUEST_DELAY)
    all_articles.extend(fetch_bing_news_rss(entity, log))
    all_articles.extend(fetch_newsdata_api(entity, log))  

    if not all_articles:
        log("  ⚠️ No articles retrieved.")
        return pd.DataFrame(columns=["title", "source", "publication_date", "article_url", "article_summary"])

    df = pd.DataFrame(all_articles)
    before = len(df)
    df = df.drop_duplicates(subset=["article_url"], keep="first").reset_index(drop=True)
    df["_sort"] = pd.to_datetime(df["publication_date"], errors="coerce")
    df = df.sort_values("_sort", ascending=False, na_position="last").drop(columns=["_sort"]).reset_index(drop=True)
    log(f"  ✅ Phase 1 complete — **{len(df)} articles** ({before - len(df)} duplicates removed)")
    return df


# ═════════════════════════════════════════════════════════════════════
# PHASE 2 — ADVERSE DETECTION
# ═════════════════════════════════════════════════════════════════════

def keyword_screen(title: str, summary: str):
    title_lower = title.lower()
    text = f"{title} {summary}".lower()
    for pat in POSITIVE_OVERRIDE_PATTERNS:
        if re.search(pat, title_lower):
            return None
    matched_cats, matched_kws = [], []
    for cat, patterns in KEYWORD_TAXONOMY.items():
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                if cat not in matched_cats:
                    matched_cats.append(cat)
                kw = m.group().strip()
                if kw not in matched_kws:
                    matched_kws.append(kw)
    if matched_cats:
        return {
            "is_adverse"       : True,
            "adverse_category" : matched_cats[0],
            "confidence_score" : 0.85,
            "reason"           : f"Keyword match(es): [{', '.join(matched_kws[:5])}]. Categories: {', '.join(matched_cats)}.",
            "detection_layer"  : "keyword",
        }
    return None


def build_classification_prompt(entity: str, title: str, summary: str):
    combined = f"{title}. {summary}"
    if len(combined) > MAX_INPUT_CHARS:
        combined = combined[:MAX_INPUT_CHARS] + "..."
    cats = "\n".join(f"  - {c}" for c in ADVERSE_CATEGORIES)
    system = (
        "You are a senior compliance analyst specializing in adverse media screening. "
        "Respond ONLY with a valid JSON object. No preamble, no markdown."
    )
    user = f"""Classify this news article about "{entity}".

Article: {combined}

Categories (choose exactly one):
{cats}

Rules:
- Regulatory approvals and positive news → "Not Adverse"
- Only flag if article describes actual harm, failure, legal action, or material risk
- Competitive comparisons, analyst opinions, stock volatility → "Not Adverse"
- "Operational Risk" = crashes, malfunctions, recalls, safety incidents
- Confidence < 0.75 → default "Not Adverse" unless concrete harmful event described

Return JSON only:
{{"is_adverse": bool, "adverse_category": "...", "confidence_score": 0.0-1.0, "reason": "1-2 sentences"}}"""
    return system, user


class HFRateLimitError(Exception):
    pass


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=2, min=5, max=30),
    stop=stop_after_attempt(3)
)
def call_groq_classify(system: str, user: str) -> str:
    if not GROQ_API_KEY:
        return '{"is_adverse": false, "adverse_category": "Not Adverse", "confidence_score": 0, "reason": "No API key set."}'
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=TEMPERATURE_CL,
        max_tokens=250,
    )
    return resp.choices[0].message.content.strip()


def parse_llm_response(raw: str) -> dict:
    fallback = {
        "is_adverse": False, "adverse_category": "Not Adverse",
        "confidence_score": 0.0, "reason": "Parse error.", "detection_layer": "llm_error"
    }
    if not raw:
        return fallback
    raw = re.sub(r"```json|```", "", raw).strip()
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if not m:
        return fallback
    try:
        p = json.loads(m.group())
        p["is_adverse"]       = bool(p.get("is_adverse", False))
        p["confidence_score"] = float(max(0.0, min(1.0, p.get("confidence_score", 0.0))))
        p["adverse_category"] = str(p.get("adverse_category", "Not Adverse"))
        p["reason"]           = str(p.get("reason", ""))
        p["detection_layer"]  = "llm"
        if p["adverse_category"] not in ADVERSE_CATEGORIES:
            p["adverse_category"] = "Reputational"
        return p
    except Exception:
        return fallback


def classify_article(entity: str, title: str, summary: str) -> dict:
    try:
        sys_p, usr_p = build_classification_prompt(entity, title, summary)
        raw = call_groq_classify(sys_p, usr_p)
        return parse_llm_response(raw)
    except Exception as e:
        return {
            "is_adverse": False, "adverse_category": "Not Adverse",
            "confidence_score": 0.0,
            "reason": f"LLM error: {str(e)[:80]}",
            "detection_layer": "llm_error"
        }


def run_phase2(df_news: pd.DataFrame, entity: str, log, progress_bar) -> pd.DataFrame:
    log("### 🔍 Phase 2 — Adverse Detection")
    log(f"  Model: Groq / {GROQ_MODEL} | Mode: Hybrid (keyword + LLM)")
    results = []
    total = len(df_news)
    llm_calls = 0
    kw_hits = 0

    for i, row in df_news.iterrows():
        title   = str(row.get("title", ""))
        summary = str(row.get("article_summary", ""))

        kw_result = keyword_screen(title, summary)

        if kw_result:
            kw_hits += 1
            llm_result = classify_article(entity, title, summary)
            llm_calls += 1
            if llm_result["confidence_score"] >= CONFIDENCE_THRESHOLD:
                llm_result["confidence_score"] = max(
                    llm_result["confidence_score"], kw_result["confidence_score"]
                )
                results.append(llm_result)
            else:
                results.append(kw_result)
        else:
            llm_result = classify_article(entity, title, summary)
            results.append(llm_result)
            llm_calls += 1

        time.sleep(API_CALL_DELAY)
        progress_bar.progress((i + 1) / total)

    df_out = df_news.copy().reset_index(drop=True)
    results_df = pd.DataFrame(results)
    for col in ["is_adverse", "adverse_category", "confidence_score", "reason", "detection_layer"]:
        df_out[col] = results_df[col].values

    df_out = df_out.sort_values(
        by=["is_adverse", "confidence_score"], ascending=[False, False]
    ).reset_index(drop=True)

    adverse_count = int(df_out["is_adverse"].sum())
    log(f"  ✅ Phase 2 complete — **{adverse_count} adverse** / {total} articles | LLM calls: {llm_calls} | Keyword hits: {kw_hits}")
    return df_out


# ═════════════════════════════════════════════════════════════════════
# PHASE 3 — RISK SCORING
# ═════════════════════════════════════════════════════════════════════

def score_category(cat: str) -> float:
    return CATEGORY_SEVERITY.get(cat, 0.40)

def score_confidence(conf: float) -> float:
    return float(max(0.0, min(1.0, conf)))

def score_source(source: str) -> float:
    if not source or pd.isna(source):
        return UNKNOWN_SOURCE_SCORE
    sl = str(source).lower().strip()
    if sl in SOURCE_CREDIBILITY:
        return SOURCE_CREDIBILITY[sl]
    for k, v in SOURCE_CREDIBILITY.items():
        if k in sl or sl in k:
            return v
    return UNKNOWN_SOURCE_SCORE

def score_recency(pub_date: str) -> float:
    if not pub_date or pub_date == "Unknown" or pd.isna(pub_date):
        return 0.50
    try:
        age = max(0, (pd.Timestamp.now().normalize() - pd.to_datetime(pub_date)).days)
        return max(RECENCY_FLOOR, math.pow(2, -age / RECENCY_HALF_LIFE_DAYS))
    except Exception:
        return 0.50

def compute_article_score(row: pd.Series) -> dict:
    zero = {"article_risk_score": 0.0, "score_category": 0.0, "score_confidence": 0.0,
            "score_source": 0.0, "score_recency": 0.0, "raw_category": 0.0,
            "raw_confidence": 0.0, "raw_source": 0.0, "raw_recency": 0.0}
    if not row.get("is_adverse", False):
        return zero
    rc = score_category(row.get("adverse_category", "Not Adverse"))
    rf = score_confidence(row.get("confidence_score", 0.0))
    rs = score_source(row.get("source", ""))
    rr = score_recency(row.get("publication_date", ""))
    sc = rc * WEIGHTS["category_severity"]
    sf = rf * WEIGHTS["llm_confidence"]
    ss = rs * WEIGHTS["source_credibility"]
    sr = rr * WEIGHTS["recency"]
    total = round(min(100.0, sc + sf + ss + sr), 2)
    return {"article_risk_score": total, "score_category": round(sc, 2),
            "score_confidence": round(sf, 2), "score_source": round(ss, 2),
            "score_recency": round(sr, 2), "raw_category": round(rc, 3),
            "raw_confidence": round(rf, 3), "raw_source": round(rs, 3),
            "raw_recency": round(rr, 3)}

def aggregate_entity_score(df_scored: pd.DataFrame) -> float:
    scores = df_scored[df_scored["is_adverse"] == True]["article_risk_score"]\
             .sort_values(ascending=False).tolist()
    if not scores:
        return 0.0
    diminished = sum(s * (DIMINISHING_BASE ** i) for i, s in enumerate(scores))
    theoretical_max = 100.0 * (1.0 / (1.0 - DIMINISHING_BASE))
    return round(min(100.0, (diminished / theoretical_max) * 100.0), 2)

def get_risk_label(score: float) -> str:
    for label, threshold in sorted(RISK_THRESHOLDS.items(), key=lambda x: -x[1]):
        if score >= threshold:
            return label
    return "Low"

def build_scorecard(entity: str, df_scored: pd.DataFrame) -> dict:
    adverse_df  = df_scored[df_scored["is_adverse"] == True].copy()
    entity_score = aggregate_entity_score(df_scored)
    risk_label   = get_risk_label(entity_score)

    cat_breakdown = {}
    if not adverse_df.empty:
        for cat, grp in adverse_df.groupby("adverse_category"):
            cat_breakdown[cat] = {
                "count"    : int(len(grp)),
                "avg_score": round(grp["article_risk_score"].mean(), 2),
                "max_score": round(grp["article_risk_score"].max(), 2),
                "severity" : CATEGORY_SEVERITY.get(cat, 0.40),
            }

    src_breakdown = {}
    if not adverse_df.empty:
        for src, grp in adverse_df.groupby("source"):
            src_breakdown[str(src)] = {
                "count"            : int(len(grp)),
                "avg_credibility"  : round(grp["raw_source"].mean(), 3),
                "avg_article_score": round(grp["article_risk_score"].mean(), 2),
            }

    dim_scores = {}
    if not adverse_df.empty:
        dim_scores = {
            "category_severity" : round(adverse_df["score_category"].mean(), 2),
            "llm_confidence"    : round(adverse_df["score_confidence"].mean(), 2),
            "source_credibility": round(adverse_df["score_source"].mean(), 2),
            "recency"           : round(adverse_df["score_recency"].mean(), 2),
        }

    contributing = []
    if not adverse_df.empty:
        for _, row in adverse_df.sort_values("article_risk_score", ascending=False).iterrows():
            contributing.append({
                "title"             : str(row.get("title", "")),
                "source"            : str(row.get("source", "")),
                "publication_date"  : str(row.get("publication_date", "")),
                "article_url"       : str(row.get("article_url", "")),
                "adverse_category"  : str(row.get("adverse_category", "")),
                "confidence_score"  : float(row.get("confidence_score", 0.0)),
                "article_risk_score": float(row.get("article_risk_score", 0.0)),
                "score_breakdown"   : {
                    "category_severity" : float(row.get("score_category", 0.0)),
                    "llm_confidence"    : float(row.get("score_confidence", 0.0)),
                    "source_credibility": float(row.get("score_source", 0.0)),
                    "recency"           : float(row.get("score_recency", 0.0)),
                },
                "reason": str(row.get("reason", "")),
            })

    return {
        "entity_name"         : entity,
        "screening_date"      : datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entity_risk_score"   : entity_score,
        "risk_label"          : risk_label,
        "risk_emoji"          : RISK_EMOJI.get(risk_label, "⚫"),
        "total_articles"      : int(len(df_scored)),
        "adverse_articles"    : int(len(adverse_df)),
        "non_adverse_articles": int(len(df_scored) - len(adverse_df)),
        "adverse_rate_pct"    : round(100 * len(adverse_df) / max(len(df_scored), 1), 1),
        "category_breakdown"  : cat_breakdown,
        "source_breakdown"    : src_breakdown,
        "dimension_scores"    : dim_scores,
        "contributing_articles": contributing,
    }


def run_phase3(df_analyzed: pd.DataFrame, entity: str, log) -> tuple:
    log("### 📊 Phase 3 — Risk Scoring")
    score_records = [compute_article_score(row) for _, row in df_analyzed.iterrows()]
    scores_df     = pd.DataFrame(score_records)
    df_scored     = df_analyzed.copy().reset_index(drop=True)
    for col in scores_df.columns:
        df_scored[col] = scores_df[col].values

    scorecard    = build_scorecard(entity, df_scored)
    entity_score = scorecard["entity_risk_score"]
    risk_label   = scorecard["risk_label"]
    emoji        = scorecard["risk_emoji"]
    log(f"  ✅ Phase 3 complete — **{emoji} {risk_label}** | Score: **{entity_score:.1f} / 100**")
    return df_scored, scorecard


# ═════════════════════════════════════════════════════════════════════
# PHASE 4 — COPILOT Q&A HELPERS
# ═════════════════════════════════════════════════════════════════════

def scorecard_to_context(sc: dict) -> str:
    lines = ["=== ENTITY RISK SCORECARD ===",
             f"Entity           : {sc.get('entity_name', '?')}",
             f"Screened         : {sc.get('screening_date', '?')}",
             f"Total articles   : {sc.get('total_articles', '?')}",
             f"Adverse articles : {sc.get('adverse_articles', '?')}",
             "",
             "=== RISK SCORE ===",
             f"Entity Score : {sc.get('entity_risk_score', '?')} / 100",
             f"Risk Label   : {sc.get('risk_label', '?')}",
             f"Interpretation: {sc.get('risk_emoji','')} {sc.get('entity_name','')} scored "
             f"{sc.get('entity_risk_score',0)}/100 — {sc.get('risk_label','')} risk. "
             f"{sc.get('adverse_articles',0)} of {sc.get('total_articles',0)} articles flagged.",
             "",
             "=== RISK CATEGORIES ==="]
    for cat, info in sc.get("category_breakdown", {}).items():
        lines.append(f"  {cat}: {info.get('count',0)} article(s), avg score {info.get('avg_score',0):.1f}, max {info.get('max_score',0):.1f}")
    lines += ["", "=== SCORE DIMENSIONS ==="]
    for dim, val in sc.get("dimension_scores", {}).items():
        lines.append(f"  {dim}: {val:.1f} pts")
    lines += ["", "=== SOURCE CREDIBILITY ==="]
    for src, info in sc.get("source_breakdown", {}).items():
        cred = info.get("avg_credibility", 0)
        lines.append(f"  {src}: credibility={cred:.0%}, articles={info.get('count',0)}")
    lines += ["", "=== CONTRIBUTING ADVERSE ARTICLES ==="]
    for i, art in enumerate(sc.get("contributing_articles", []), 1):
        lines += [
            f"[{i}] {art.get('title','?')}",
            f"    Source    : {art.get('source','?')}",
            f"    Date      : {art.get('publication_date','?')}",
            f"    Category  : {art.get('adverse_category','?')}",
            f"    Confidence: {art.get('confidence_score','?')} (0–1 scale)",
            f"    Risk Score: {art.get('article_risk_score','?')} / 100",
            f"    Reason    : {art.get('reason','?')}",
            f"    URL       : {art.get('article_url','?')}",
            "",
        ]
    return "\n".join(lines)


def build_system_prompt(context: str, entity: str) -> str:
    return textwrap.dedent(f"""
    You are ScreeningCopilot, an AI assistant inside an Adverse Media Screening platform.
    Help compliance analysts understand the risk profile of "{entity}" using ONLY the scorecard below.

    CAPABILITIES:
    1. Explain WHY the entity received its risk label and score.
    2. Identify WHICH articles drove the score (cite title + source + date).
    3. Break down risk categories detected and their compliance significance.
    4. Explain source credibility and its effect on scoring.
    5. Suggest specific due diligence next steps.

    RULES:
    - Answer ONLY from the scorecard context. Never invent data.
    - Confidence scores are 0–1 scale (0.9 = 90% confident). Never multiply by 100.
    - Article risk scores are 0–100. Dimension contributions are pts within that.
    - Keep answers professional and concise. This is a compliance tool.
    - If info not in scorecard: say "That information is not in the current scorecard."

    ── SCORECARD CONTEXT ──
    {context}
    ───────────────────────
    """).strip()


def detect_intent(msg: str) -> str:
    ml = msg.lower()
    for intent, pat in INTENT_PATTERNS.items():
        if re.search(pat, ml):
            return intent
    return "general"


def get_intent_prefix(intent: str, entity: str) -> str:
    return {
        "risk_reason"  : f"Explain in detail why {entity} received its risk label and score. Reference specific articles.",
        "articles"     : f"List all articles contributing to {entity}'s risk score with source, date, category, and score.",
        "categories"   : f"Describe the risk categories detected for {entity} and their compliance significance.",
        "source_cred"  : f"Explain source credibility scores and how they affected {entity}'s overall risk score.",
        "next_steps"   : f"Recommend specific due diligence actions for a compliance analyst screening {entity}.",
        "score_explain": f"Explain how {entity}'s risk score was calculated and what it means.",
        "summary"      : f"Give a concise executive summary of {entity}'s adverse media risk profile.",
        "general"      : "",
    }.get(intent, "")


def call_groq_qa(system_prompt: str, history: list, user_msg: str, intent: str, entity: str) -> str:
    if not GROQ_API_KEY:
        return "⚠️ Groq API key not set. Run: export GROQ_API_KEY='gsk_...' then restart."
    try:
        client  = Groq(api_key=GROQ_API_KEY)
        prefix  = get_intent_prefix(intent, entity)
        augmented = f"{prefix}\n\nAnalyst question: {user_msg}" if prefix else user_msg
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": augmented})
        resp = client.chat.completions.create(
            model=GROQ_MODEL, messages=messages,
            max_tokens=MAX_TOKENS, temperature=TEMPERATURE_QA
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        err = str(e)
        if "api_key" in err.lower() or "auth" in err.lower():
            return "⚠️ Groq API key error. Check your GROQ_API_KEY environment variable."
        elif "rate" in err.lower():
            return "⚠️ Groq rate limit reached. Wait a moment and try again."
        return f"⚠️ LLM error: {err}"


# ═════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═════════════════════════════════════════════════════════════════════

def render_sidebar_history():
    """Shows all previously screened entities in the sidebar."""
    st.sidebar.markdown("## 📋 Screening History")
    history = st.session_state.get("entity_history", [])
    if not history:
        st.sidebar.caption("No entities screened yet.")
        return
    for i, record in enumerate(reversed(history)):
        sc = record["scorecard"]
        emoji  = sc.get("risk_emoji", "⚫")
        label  = sc.get("risk_label", "?")
        score  = sc.get("entity_risk_score", 0)
        entity = sc.get("entity_name", "?")
        adv    = sc.get("adverse_articles", 0)
        total  = sc.get("total_articles", 0)
        btn_label = f"{emoji} {entity}  ·  {score:.0f}/100  ({label})"
        if st.sidebar.button(btn_label, key=f"hist_{i}", use_container_width=True):
            # Switch active session to this historical record
            st.session_state.active_entity   = record["entity"]
            st.session_state.scorecard        = record["scorecard"]
            st.session_state.df_scored        = record["df_scored"]
            st.session_state.context          = record["context"]
            st.session_state.chat_messages    = []
            st.session_state.pipeline_done    = True
            st.rerun()
        st.sidebar.caption(f"  {adv}/{total} adverse  ·  {sc.get('screening_date','')[:10]}")

    if len(history) > 0:
        st.sidebar.markdown("---")
        if st.sidebar.button("🗑 Clear history", use_container_width=True):
            st.session_state.entity_history = []
            st.rerun()


def render_scorecard_panel(sc: dict):
    """Compact risk snapshot shown above the chat after pipeline completes."""
    emoji = sc.get("risk_emoji", "⚫")
    label = sc.get("risk_label", "?")
    score = sc.get("entity_risk_score", 0)
    entity = sc.get("entity_name", "?")
    adv    = sc.get("adverse_articles", 0)
    total  = sc.get("total_articles", 0)

    # Score bar
    filled = int(score / 5)
    bar = "█" * filled + "░" * (20 - filled)

    col1, col2, col3, col4 = st.columns([2, 1.5, 1.5, 3])
    with col1:
        st.metric("Entity Risk Score", f"{score:.1f} / 100")
        st.caption(f"`[{bar}]`")
    with col2:
        st.metric("Risk Label", f"{emoji} {label}")
    with col3:
        st.metric("Adverse Articles", f"{adv} / {total}")
    with col4:
        action = RECOMMENDED_ACTIONS.get(label, "⚠️ Manual review required.")
        st.info(action)

    # Category breakdown
    cats = sc.get("category_breakdown", {})
    if cats:
        with st.expander("📌 Risk Category Breakdown", expanded=False):
            cat_data = [
                {"Category": c, "Articles": v["count"],
                 "Avg Score": f"{v['avg_score']:.1f}", "Max Score": f"{v['max_score']:.1f}",
                 "Severity": f"{v['severity']:.2f}"}
                for c, v in cats.items()
            ]
            st.dataframe(pd.DataFrame(cat_data), use_container_width=True, hide_index=True)

    # Adverse articles table
    df_sc = st.session_state.get("df_scored")
    if df_sc is not None:
        adv_df = df_sc[df_sc["is_adverse"] == True].copy()
        if not adv_df.empty:
            with st.expander("📰 Adverse Articles Detail", expanded=False):
                disp_cols = [c for c in [
                    "title", "source", "publication_date", "adverse_category",
                    "confidence_score", "article_risk_score", "reason"
                ] if c in adv_df.columns]
                st.dataframe(adv_df[disp_cols].reset_index(drop=True),
                             use_container_width=True, height=280, hide_index=True)


def render_quick_questions(entity: str, risk_label: str):
    st.markdown("**Suggested questions — click to ask:**")
    qs = [
        f"Why is {entity} rated {risk_label.lower()} risk?",
        f"Which articles drove the risk score?",
        f"What risk categories were detected?",
        f"How credible are the news sources?",
        f"What due diligence steps should I take?",
        f"Give me a compliance executive summary.",
    ]
    cols = st.columns(3)
    for i, q in enumerate(qs):
        if cols[i % 3].button(q, key=f"qq_{i}_{entity}", use_container_width=True):
            st.session_state.pending_question = q


def main():
    st.set_page_config(
        page_title="Adverse Media Screening Copilot",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Sidebar ─────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🔍 Adverse Media\nScreening Copilot")
        st.caption("TCS & AMD AI Hackathon · AGENTS_001")
        st.divider()
        render_sidebar_history()
        st.divider()
        st.caption(f"LLM: {GROQ_MODEL}")
        if not GROQ_API_KEY:
            st.warning("⚠️ GROQ_API_KEY not set.\nRun:\n`export GROQ_API_KEY='gsk_...'`")

    # ── Session state init ───────────────────────────────────────
    for key, default in [
        ("pipeline_done", False),
        ("active_entity", ""),
        ("scorecard", {}),
        ("df_scored", None),
        ("context", ""),
        ("chat_messages", []),
        ("pending_question", None),
        ("entity_history", []),
        ("log_lines", []),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Header ───────────────────────────────────────────────────
    st.markdown(
        "<h1 style='margin-bottom:2px'>🔍 Adverse Media Screening Copilot</h1>"
        "<p style='color:gray;margin-top:0'>Type an entity name → pipeline runs automatically → ask questions</p>",
        unsafe_allow_html=True
    )
    st.divider()

    # ── Entity input ─────────────────────────────────────────────
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        entity_input = st.text_input(
            "Entity to screen",
            placeholder="e.g.  Tesla,  Binance,  Adani Group,  Sam Bankman-Fried",
            label_visibility="collapsed",
        )
    with col_btn:
        run_btn = st.button("🔎 Screen Now", type="primary", use_container_width=True)

    # ── Pipeline execution ───────────────────────────────────────
    if run_btn and entity_input.strip():
        entity = entity_input.strip()
        st.session_state.pipeline_done  = False
        st.session_state.active_entity  = entity
        st.session_state.chat_messages  = []
        st.session_state.pending_question = None
        st.session_state.log_lines      = []

        log_container = st.empty()
        progress_container = st.empty()

        log_lines = []

        def log(msg: str):
            log_lines.append(msg)
            log_container.markdown("\n\n".join(log_lines))

        log(f"## 🚀 Screening: **{entity}**")
        log("---")

        # Phase 1
        with st.spinner("Phase 1 — Retrieving news articles..."):
            df_news = run_phase1(entity, log)

        if df_news.empty:
            st.warning("No articles found. Try a different entity name.")
            st.stop()

        # Phase 2
        log("---")
        with st.spinner("Phase 2 — Classifying adverse media with Groq LLM..."):
            p2_bar = progress_container.progress(0)
            df_analyzed = run_phase2(df_news, entity, log, p2_bar)
            progress_container.empty()

        # Phase 3
        log("---")
        with st.spinner("Phase 3 — Computing risk scores..."):
            df_scored, scorecard = run_phase3(df_analyzed, entity, log)

        log("---")
        log(f"### ✅ Pipeline complete — opening Copilot Q&A for **{entity}**")

        # Persist results
        context = scorecard_to_context(scorecard)
        st.session_state.scorecard     = scorecard
        st.session_state.df_scored     = df_scored
        st.session_state.context       = context
        st.session_state.pipeline_done = True
        st.session_state.log_lines     = log_lines

        # Add to history (most recent first, avoid duplicates)
        history = st.session_state.entity_history
        history = [r for r in history if r["entity"].lower() != entity.lower()]
        history.insert(0, {
            "entity"   : entity,
            "scorecard": scorecard,
            "df_scored": df_scored,
            "context"  : context,
        })
        st.session_state.entity_history = history[:10]  # keep last 10

        time.sleep(0.5)
        st.rerun()

    # ── Post-pipeline: show scorecard + chat ─────────────────────
    if st.session_state.pipeline_done and st.session_state.scorecard:
        sc     = st.session_state.scorecard
        entity = st.session_state.active_entity

        # Show pipeline log in collapsed expander
        if st.session_state.log_lines:
            with st.expander("📋 Pipeline execution log", expanded=False):
                st.markdown("\n\n".join(st.session_state.log_lines))

        st.divider()

        # Scorecard summary
        st.markdown(f"### {sc.get('risk_emoji','')} Risk Scorecard — {entity}")
        render_scorecard_panel(sc)

        st.divider()

        # Chat interface
        st.markdown("### 💬 Ask the Copilot")

        # Welcome message
        if not st.session_state.chat_messages:
            risk_label   = sc.get("risk_label", "?")
            entity_score = sc.get("entity_risk_score", "?")
            adv          = sc.get("adverse_articles", 0)
            total        = sc.get("total_articles", 0)
            welcome = (
                f"I've completed the adverse media screening for **{entity}**.\n\n"
                f"**Result:** {sc.get('risk_emoji','')} **{risk_label}** risk — score **{entity_score}/100**\n\n"
                f"**{adv}** of **{total}** articles flagged as adverse. "
                f"Ask me anything about this entity's risk profile."
            )
            with st.chat_message("assistant", avatar="🔍"):
                st.markdown(welcome)

        # Render history
        for msg in st.session_state.chat_messages:
            avatar = "👤" if msg["role"] == "user" else "🔍"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

        # Quick questions
        st.divider()
        render_quick_questions(entity, sc.get("risk_label", "unknown"))

        # Chat input
        user_input = st.chat_input(f"Ask about {entity}'s risk profile...")

        # Handle quick question button clicks
        if st.session_state.pending_question and not user_input:
            user_input = st.session_state.pending_question
            st.session_state.pending_question = None

        # Process message
        if user_input:
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_input)

            intent = detect_intent(user_input)
            system_prompt = build_system_prompt(st.session_state.context, entity)

            with st.chat_message("assistant", avatar="🔍"):
                with st.spinner("Analyzing..."):
                    response = call_groq_qa(
                        system_prompt=system_prompt,
                        history=[{"role": m["role"], "content": m["content"]}
                                 for m in st.session_state.chat_messages[:-1]],
                        user_msg=user_input,
                        intent=intent,
                        entity=entity,
                    )
                st.markdown(response)
                st.caption(f"Intent: `{intent}` · Model: `{GROQ_MODEL}`")

            st.session_state.chat_messages.append({"role": "assistant", "content": response})

    elif not st.session_state.pipeline_done:
        # Empty state
        st.markdown(
            """
            <div style='text-align:center;padding:60px 0;color:#888'>
            <div style='font-size:48px'>🔍</div>
            <h3 style='color:#555'>Enter an entity name above and click Screen Now</h3>
            <p>The pipeline will retrieve live news, detect adverse media,<br>
            compute a risk score, and open a Q&amp;A copilot — automatically.</p>
            <p style='font-size:13px;margin-top:24px'>
            Works with companies, individuals, funds, or any named entity.<br>
            Examples: <code>Binance</code> · <code>FTX</code> · <code>Wirecard</code> · <code>Gautam Adani</code>
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
