

import streamlit as st
import json
import os
import re
import textwrap
from datetime import datetime
from pathlib import Path

import pandas as pd
from groq import Groq


GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "") # your API Key
GROQ_MODEL   = "llama-3.3-70b-versatile"  # best free-tier model
MAX_TOKENS   = 1024
TEMPERATURE  = 0.3   # low = more deterministic, compliance-appropriate


DEFAULT_JSON_PATH = ""
DEFAULT_CSV_PATH  = ""

# ── Copilot persona & capabilities ─────────────────────────────
COPILOT_NAME = "ScreeningCopilot"


RECOMMENDED_ACTIONS = {
    "Critical": "🚨 Block onboarding. Escalate immediately to senior compliance.",
    "High"    : "🔴 Senior analyst review required before proceeding.",
    "Medium"  : "🟡 Enhanced due diligence recommended. Monitor for 30 days.",
    "Low"     : "🟢 Standard onboarding process. Schedule periodic monitoring.",
}


INTENT_PATTERNS = {
    "risk_reason"    : r"\b(why|reason|explain|how|what made|flagged|rated|scored)\b",
    "articles"       : r"\b(article|news|source|headline|story|stories|drove|contributing)\b",
    "categories"     : r"\b(categor|type|kind|domain|class|fraud|legal|esg|sanction|operat|reputat)\b",
    "source_cred"    : r"\b(source|credib|reliable|outlet|publisher|bloomberg|reuters|trust)\b",
    "next_steps"     : r"\b(next|action|recommend|due diligence|investigate|escalat|step|do|should)\b",
    "score_explain"  : r"\b(score|number|point|100|scale|metric|how high|how low|risk level)\b",
    "summary"        : r"\b(summar|overview|brief|tldr|overall|total|snapshot)\b",
}



def load_scorecard(json_path: str) -> dict:
    """
    Load Phase 3 JSON scorecard into a dict.
    Returns an empty dict with a clear error key if load fails.
    The scorecard is the primary grounding document for the LLM.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return {"_error": f"Scorecard file not found: {json_path}"}
    except json.JSONDecodeError as e:
        return {"_error": f"Could not parse JSON: {e}"}
    except Exception as e:
        return {"_error": f"Unexpected error loading scorecard: {e}"}


def normalize_scorecard(raw: dict) -> dict:
    """
    Normalises the Phase 3 JSON into the canonical shape this app expects.

    Your Phase 3 JSON is FLAT — all keys sit at the top level with no
    nested metadata or risk_score blocks. Key names used by Phase 3:

      entity_name          → entity
      screening_date       → screened_at
      entity_risk_score    → entity_risk_score  (same, direct)
      risk_label           → risk_label         (same, direct)
      total_articles       → total_articles     (same, direct)
      adverse_articles     → adverse_articles   (same, direct)
      category_breakdown   → category_breakdown (same, but inner keys differ)
      source_breakdown     → source_breakdown   (same, but inner keys differ)
      dimension_scores     → dimension_contributions  (renamed)
      contributing_articles→ contributing_articles    (same, direct)

    Inner key differences:
      category_breakdown[cat]["avg_score"]       (Phase 3) vs "max_article_score" (app)
      source_breakdown[src]["avg_credibility"]   (Phase 3) vs "credibility_score" (app)
      source_breakdown[src]["count"]             (Phase 3) vs "article_count"     (app)
    """
    if "_error" in raw:
        return raw

    # ── Helper: try multiple key names, return first match ────────
    def pick(d: dict, *keys, default=None):
        for k in keys:
            if k in d:
                return d[k]
        return default

    # ── 1. Entity name ────────────────────────────────────────────
    entity = pick(
        raw,
        "entity_name", "entity", "name", "company",
        default=pick(raw.get("metadata", {}), "entity", "entity_name", default="Unknown Entity")
    ) or "Unknown Entity"

    # ── 2. Score & label — flat in your JSON ─────────────────────
    entity_score = pick(
        raw,
        "entity_risk_score", "risk_score", "score", "final_score",
        default=pick(raw.get("risk_score", {}), "entity_risk_score", "score", default=0)
    )
    risk_label = pick(
        raw,
        "risk_label", "label", "risk_level", "level", "rating",
        default=pick(raw.get("risk_score", {}), "risk_label", "label", default="Unknown")
    )

    # ── 3. Metadata fields — flat in your JSON ────────────────────
    screened_at = pick(
        raw,
        "screening_date", "screened_at", "timestamp", "generated_at",
        default=pick(raw.get("metadata", {}), "screened_at", "screening_date", default="")
    )
    total_articles = pick(
        raw,
        "total_articles", "total", "num_articles",
        default=pick(raw.get("metadata", {}), "total_articles", default="?")
    )
    adverse_articles = pick(
        raw,
        "adverse_articles", "adverse_count", "num_adverse",
        default=pick(raw.get("metadata", {}), "adverse_articles", default="?")
    )
    adverse_rate = pick(raw, "adverse_rate_pct", "adverse_rate", default="")
    risk_emoji   = pick(raw, "risk_emoji", default="")

    # ── 4. Category breakdown — normalise inner keys ──────────────
    raw_cats = pick(raw, "category_breakdown", "categories", "risk_categories", default={})
    if isinstance(raw_cats, list):
        raw_cats = {
            item.get("category", item.get("name", f"Cat{i}")): item
            for i, item in enumerate(raw_cats)
        }
    cat_block = {}
    for cat_name, cat_data in raw_cats.items():
        if isinstance(cat_data, dict):
            cat_block[cat_name] = {
                "count"            : cat_data.get("count", 0),
                "avg_confidence"   : cat_data.get("avg_score", cat_data.get("avg_confidence", 0)),
                "max_article_score": cat_data.get("max_score", cat_data.get("max_article_score", 0)),
                "severity"         : cat_data.get("severity", 0),
            }
        else:
            cat_block[cat_name] = cat_data

    # ── 5. Source breakdown — normalise inner keys ────────────────
    raw_srcs = pick(raw, "source_breakdown", "sources", "source_credibility", default={})
    src_block = {}
    for src_name, src_data in raw_srcs.items():
        if isinstance(src_data, dict):
            src_block[src_name] = {
                "article_count"    : src_data.get("count", src_data.get("article_count", 0)),
                "credibility_score": src_data.get("avg_credibility", src_data.get("credibility_score", 0)),
                "avg_article_score": src_data.get("avg_article_score", 0),
            }
        else:
            src_block[src_name] = src_data

    # ── 6. Dimension scores ───────────────────────────────────────
    raw_dims = pick(
        raw,
        "dimension_scores",
        "dimension_contributions",
        "dimensions",
        "score_dimensions",
        default={}
    )
    dim_block = {
        k: float(v) if isinstance(v, (int, float)) else 0.0
        for k, v in raw_dims.items()
    }

    # ── 7. Contributing articles ──────────────────────────────────
    articles = pick(
        raw,
        "contributing_articles", "adverse_articles", "articles", "flagged_articles",
        default=[]
    )
    if isinstance(articles, dict):
        articles = list(articles.values())

    # ── 8. Build canonical scorecard ──────────────────────────────
    return {
        "metadata": {
            "entity"          : entity,
            "screened_at"     : screened_at,
            "total_articles"  : total_articles,
            "adverse_articles": adverse_articles,
            "adverse_rate_pct": adverse_rate,
            "risk_emoji"      : risk_emoji,
            "phase"           : "Phase 3 — Risk Scoring",
        },
        "risk_score": {
            "entity_risk_score": entity_score,
            "risk_label"       : risk_label,
            "interpretation"   : (
                f"{risk_emoji} {entity} scored {entity_score}/100 — {risk_label} risk. "
                f"{adverse_articles} of {total_articles} articles flagged as adverse "
                f"({adverse_rate}%)."
            ),
        },
        "category_breakdown"     : cat_block,
        "source_breakdown"       : src_block,
        "dimension_contributions": dim_block,
        "contributing_articles"  : articles,
        "_raw"                   : raw,
    }


def load_articles_df(csv_path: str) -> pd.DataFrame:
    """
    Load Phase 3 CSV (article-level scored data) into a DataFrame.
    Used for structured queries (top articles, category filters, etc.)
    Returns an empty DataFrame on failure.

    FIX 2 (applied here): is_adverse is normalised from string
    "True"/"False" to boolean before filtering, matching the fix
    already present in the upload path.
    """
    try:
        df = pd.read_csv(csv_path, parse_dates=["publication_date"])
        # FIX 2: handle is_adverse stored as string "True"/"False" in CSV
        # (Phase 2 CSV stores it as a string, not a Python boolean)
        if "is_adverse" in df.columns:
            df["is_adverse"] = df["is_adverse"].apply(
                lambda x: str(x).strip().lower() in ("true", "1", "yes")
            )
        df = df[df["is_adverse"] == True].copy()
        df = df.sort_values("article_risk_score", ascending=False)
        return df
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not load articles CSV: {e}")
        return pd.DataFrame()


def scorecard_to_context_string(scorecard: dict) -> str:
    """
    Converts the Phase 3 JSON scorecard into a compact plain-text
    context block that fits inside the LLM system prompt.

    Why plain text, not raw JSON?
    Raw JSON wastes tokens on punctuation and nested braces.
    A structured plain-text summary gives the LLM the same
    information more efficiently, leaving more room for the
    conversation history.
    """
    if "_error" in scorecard:
        return f"[SCORECARD LOAD ERROR: {scorecard['_error']}]"

    lines = []
    meta        = scorecard.get("metadata", {})
    score_block = scorecard.get("risk_score", {})
    cat_block   = scorecard.get("category_breakdown", {})
    src_block   = scorecard.get("source_breakdown", {})
    dim_block   = scorecard.get("dimension_contributions", {})
    articles    = scorecard.get("contributing_articles", [])

    # ── Entity summary ──────────────────────────────────────────
    lines.append("=== ENTITY RISK SCORECARD ===")
    lines.append(f"Entity           : {meta.get('entity', 'Unknown')}")
    lines.append(f"Screened         : {meta.get('screened_at', 'Unknown')}")
    lines.append(f"Total articles   : {meta.get('total_articles', '?')}")
    lines.append(f"Adverse articles : {meta.get('adverse_articles', '?')}")
    lines.append("")

    # ── Risk score ──────────────────────────────────────────────
    lines.append("=== RISK SCORE ===")
    lines.append(f"Entity Score : {score_block.get('entity_risk_score', '?')} / 100")
    lines.append(f"Risk Label   : {score_block.get('risk_label', '?')}")
    lines.append(f"Interpretation: {score_block.get('interpretation', '')}")
    lines.append("")

    # ── Category breakdown ──────────────────────────────────────
    lines.append("=== RISK CATEGORIES ===")
    for cat, info in cat_block.items():
        if isinstance(info, dict):
            lines.append(
                f"  {cat}: {info.get('count', 0)} article(s), "
                f"avg article score {info.get('avg_confidence', 0):.1f}, "
                f"max article score {info.get('max_article_score', 0):.1f}"
            )
    lines.append("")

    # ── Dimension contributions ─────────────────────────────────
    lines.append("=== SCORE DIMENSIONS (avg pts contributed per adverse article) ===")
    for dim, val in dim_block.items():
        lines.append(f"  {dim}: {val:.1f} pts")
    lines.append("")

    # ── Source credibility ──────────────────────────────────────
    lines.append("=== SOURCE CREDIBILITY ===")
    for src, info in src_block.items():
        if isinstance(info, dict):
            cred = info.get('credibility_score', '?')
            # Display credibility as a percentage for clarity
            cred_display = f"{float(cred):.0%}" if isinstance(cred, (int, float)) else cred
            lines.append(
                f"  {src}: credibility={cred_display}, "
                f"articles={info.get('article_count', '?')}"
            )
    lines.append("")

    # ── Contributing articles ───────────────────────────────────
    lines.append("=== CONTRIBUTING ADVERSE ARTICLES ===")
    for i, art in enumerate(articles, 1):
        lines.append(f"[{i}] {art.get('title', 'No title')}")
        lines.append(f"    Source      : {art.get('source', '?')}")
        lines.append(f"    Date        : {art.get('publication_date', '?')}")
        lines.append(f"    Category    : {art.get('adverse_category', '?')}")
        lines.append(f"    Confidence  : {art.get('confidence_score', '?')} (0–1 scale)")
        lines.append(f"    Risk Score  : {art.get('article_risk_score', '?')} / 100")
        lines.append(f"    Reason      : {art.get('reason', '?')}")
        lines.append(f"    URL         : {art.get('article_url', '?')}")
        lines.append("")

    return "\n".join(lines)




def build_system_prompt(context: str, entity: str) -> str:
    """
    Builds the system prompt that anchors the Copilot to the
    Phase 3 scorecard data.

    Design principle — grounded generation:
    The LLM is explicitly told to answer only from the provided
    context, never from its training knowledge. This prevents
    hallucination of fake news articles or fictional risk scores.
    """
    return textwrap.dedent(f"""
    You are {COPILOT_NAME}, an AI assistant embedded inside an
    Adverse Media Screening platform for compliance analysts.

    Your role is to help analysts understand the risk profile of
    the entity "{entity}" based exclusively on the structured
    risk scorecard provided below.

    CAPABILITIES YOU HAVE:
    1. Explain WHY the entity received its risk label and score.
    2. Identify WHICH articles drove the score (cite title + source).
    3. Break down WHICH risk categories were detected and their severity.
    4. Explain SOURCE CREDIBILITY and how it affected the score.
    5. Suggest NEXT STEPS and due diligence actions for analysts.

    STRICT RULES:
    - Answer ONLY from the scorecard context below. Never invent articles,
      scores, or facts not present in the data.
    - If the analyst asks about something not in the scorecard,
      say: "That information is not in the current scorecard."
    - Keep answers concise, professional, and compliance-appropriate.
    - When citing articles, always include the source name and date.
    - For next steps, tailor recommendations to the detected risk category.
    - Do NOT use casual language. This is a compliance tool.
    - Confidence scores are on a 0–1 scale (e.g. 0.9 = 90% confidence).
      Never multiply them by 100 or report them as raw point values.
    - Article risk scores are on a 0–100 scale. Dimension contributions
      are sub-components of that score in points (e.g. 22.5 pts out of 25).

    ── SCORECARD CONTEXT ──────────────────────────────────────────
    {context}
    ───────────────────────────────────────────────────────────────
    """).strip()


def detect_intent(user_message: str) -> str:
    """
    Lightweight intent classifier using regex.
    Routes the question to the right prompt prefix before the LLM call.
    This reduces hallucination by pre-framing the LLM's focus area.
    Returns the best-matching intent label, or 'general' as fallback.
    """
    msg_lower = user_message.lower()
    for intent, pattern in INTENT_PATTERNS.items():
        if re.search(pattern, msg_lower):
            return intent
    return "general"


def get_intent_prefix(intent: str, entity: str) -> str:
    """
    Returns a prefix appended to the user message before sending to
    the LLM. This steers the model toward the right answer type
    without rewriting the user's question.
    """
    prefixes = {
        "risk_reason"  : f"Explain in detail why {entity} received its risk label and score. Reference specific articles and their dimensions.",
        "articles"     : f"List the specific articles that contributed to {entity}'s adverse risk score, with source, date, category, and article score.",
        "categories"   : f"Describe the risk categories detected for {entity}, their article counts, and what each category means in a compliance context.",
        "source_cred"  : f"Explain the source credibility scores for the outlets that reported on {entity} and how they affected the overall risk score.",
        "next_steps"   : f"Based on the risk profile of {entity}, recommend specific due diligence actions a compliance analyst should take next.",
        "score_explain": f"Explain how {entity}'s numeric risk score was calculated, what scale it uses, and what the score means.",
        "summary"      : f"Give a concise executive summary of {entity}'s adverse media risk profile.",
        "general"      : "",
    }
    return prefixes.get(intent, "")


def call_groq(
    system_prompt: str,
    conversation_history: list,
    user_message: str,
    intent: str,
    entity: str,
) -> str:
    """
    Calls Groq LLM with full conversation history for multi-turn Q&A.

    Multi-turn design:
    The full conversation history (roles: user/assistant) is sent
    with every call. This lets the analyst ask follow-up questions
    like "what about the second article?" and the LLM understands
    the reference from prior context.

    Error handling:
    Returns a user-friendly error string rather than crashing
    so the Streamlit UI stays functional.
    """
    # Guard: catch missing API key early with a clear message
    if not GROQ_API_KEY:
        return (
            "⚠️ Groq API key not set. "
            "Run: export GROQ_API_KEY='gsk_your_key_here' "
            "then restart the app."
        )

    try:
        client = Groq(api_key=GROQ_API_KEY)

        # Prepend the intent prefix to focus the LLM
        prefix = get_intent_prefix(intent, entity)
        augmented_message = f"{prefix}\n\nAnalyst question: {user_message}" if prefix else user_message

        # Build the messages payload: system + full history + current turn
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": augmented_message})

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            return "⚠️ Groq API key error. Set GROQ_API_KEY environment variable and restart."
        elif "rate" in error_msg.lower():
            return "⚠️ Groq rate limit reached. Wait a moment and try again."
        elif "connect" in error_msg.lower() or "network" in error_msg.lower():
            return "⚠️ Network error reaching Groq API. Check connectivity."
        else:
            return f"⚠️ LLM error: {error_msg}"



def render_scorecard_sidebar(scorecard: dict):
    """
    Renders a compact risk summary in the sidebar so analysts
    always have the key numbers in view while chatting.

    FIX 3 (bonus): Recommended action banner added below the metrics.
    This makes the tool actionable — not just a score, but a decision.
    """
    if "_error" in scorecard:
        st.sidebar.error(scorecard["_error"])
        return

    entity      = scorecard.get("metadata", {}).get("entity", "Unknown")
    score_block = scorecard.get("risk_score", {})
    cat_block   = scorecard.get("category_breakdown", {})
    meta        = scorecard.get("metadata", {})

    entity_score = score_block.get("entity_risk_score", 0)
    risk_label   = score_block.get("risk_label", "Unknown")

    # Risk label colour — handles both "HIGH" and "High" (mixed case)
    label_colours = {
        "CRITICAL" : "🔴",
        "HIGH"     : "🟠",
        "MEDIUM"   : "🟡",
        "LOW"      : "🟢",
        "MINIMAL"  : "⚪",
    }
    icon = (
        meta.get("risk_emoji")
        or label_colours.get(risk_label.upper(), "⚫")
    )

    st.sidebar.markdown(f"## {icon} {entity}")
    st.sidebar.metric("Risk Score", f"{entity_score} / 100")
    st.sidebar.metric("Risk Label", risk_label)
    st.sidebar.metric(
        "Adverse Articles",
        f"{meta.get('adverse_articles', '?')} / {meta.get('total_articles', '?')}"
    )

    # FIX 3: Recommended action — shown immediately below the score
    action = RECOMMENDED_ACTIONS.get(
        risk_label,
        RECOMMENDED_ACTIONS.get(risk_label.capitalize(), "⚠️ Manual review required.")
    )
    st.sidebar.info(action)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Risk Categories**")
    for cat, info in cat_block.items():
        if isinstance(info, dict) and info.get("count", 0) > 0:
            count = info["count"]
            st.sidebar.markdown(f"- `{cat}` — {count} article{'s' if count > 1 else ''}")

    st.sidebar.markdown("---")
    screened_at = meta.get("screened_at", "")
    if screened_at:
        st.sidebar.caption(f"Screened: {screened_at[:10]}")


def render_quick_questions(entity: str, risk_label: str):
    """
    Renders clickable pre-built question buttons below the chat input.
    Clicking a button sends that question as if the analyst typed it.

    FIX 1 (quick question label): Uses the actual risk_label from the
    scorecard instead of hardcoded "high risk", so the button always
    matches the entity's true rating.
    """
    st.markdown("**Quick questions:**")
    cols = st.columns(2)

    # FIX 1: risk_label passed in dynamically — no hardcoded "high risk"
    quick_qs = [
        f"Why is {entity} flagged as {risk_label.lower()} risk?",
        f"Which articles drove the risk score?",
        f"What risk categories were detected?",
        f"How credible are the sources?",
        f"What due diligence steps should I take?",
        f"Give me an executive summary.",
    ]

    for i, q in enumerate(quick_qs):
        col = cols[i % 2]
        if col.button(q, key=f"quick_{i}", use_container_width=True):
            st.session_state["pending_question"] = q


def main():
    """
    Main Streamlit app entry point.

    App architecture:
    - Session state holds: chat history, loaded scorecard, entity name
    - Sidebar: scorecard summary + file upload
    - Main area: chat interface + quick question buttons
    - Each user message → intent detection → LLM call → response appended to history
    """
    # ── Page config ─────────────────────────────────────────────
    st.set_page_config(
        page_title="Adverse Media Screening Copilot",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Header ──────────────────────────────────────────────────
    st.markdown(
        """
        <h1 style='margin-bottom:0'>🔍 Adverse Media Screening Copilot</h1>
        <p style='color:gray;margin-top:4px'>
        TCS &amp; AMD AI Hackathon · AGENTS_001 
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Session state initialisation ────────────────────────────
    if "messages"    not in st.session_state: st.session_state.messages    = []
    if "scorecard"   not in st.session_state: st.session_state.scorecard   = {}
    if "df_articles" not in st.session_state: st.session_state.df_articles = pd.DataFrame()
    if "entity"      not in st.session_state: st.session_state.entity      = ""
    if "context"     not in st.session_state: st.session_state.context     = ""
    if "pending_question" not in st.session_state: st.session_state.pending_question = None

    # ── Sidebar: file loading ────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📂 Load Phase 3 Data")

        # Option A: file uploader (drag & drop)
        st.markdown("**Upload scorecard files:**")
        uploaded_json = st.file_uploader(
            "Risk Scorecard JSON", type=["json"], key="json_upload"
        )
        uploaded_csv = st.file_uploader(
            "Scored Articles CSV", type=["csv"], key="csv_upload"
        )

        # Option B: path input (if files are already on disk)
        st.markdown("**Or enter file paths:**")
        json_path = st.text_input("JSON path", value=DEFAULT_JSON_PATH)
        csv_path  = st.text_input("CSV path",  value=DEFAULT_CSV_PATH)

        load_btn = st.button("▶ Load Data", type="primary", use_container_width=True)

        if load_btn or uploaded_json:
            with st.spinner("Loading scorecard..."):

                # ── Load JSON ───────────────────────────────────
                if uploaded_json:
                    try:
                        scorecard = json.load(uploaded_json)
                    except Exception as e:
                        scorecard = {"_error": str(e)}
                else:
                    scorecard = load_scorecard(json_path)

                # ── Load CSV ────────────────────────────────────
                if uploaded_csv:
                    try:
                        df = pd.read_csv(uploaded_csv, parse_dates=["publication_date"])
                        # FIX 2: handle is_adverse stored as string in CSV
                        if "is_adverse" in df.columns:
                            df["is_adverse"] = df["is_adverse"].apply(
                                lambda x: str(x).strip().lower() in ("true", "1", "yes")
                            )
                            df = df[df["is_adverse"] == True]
                        if "article_risk_score" in df.columns:
                            df = df.sort_values("article_risk_score", ascending=False)
                    except Exception as e:
                        st.warning(f"CSV load issue: {e}")
                        df = pd.DataFrame()
                else:
                    df = load_articles_df(csv_path)

                # ── Normalise scorecard to canonical schema ─────
                scorecard = normalize_scorecard(scorecard)

                # ── Persist to session ──────────────────────────
                st.session_state.scorecard   = scorecard
                st.session_state.df_articles = df
                st.session_state.entity = (
                    scorecard.get("metadata", {}).get("entity", "Unknown Entity")
                    if "_error" not in scorecard else "Unknown Entity"
                )
                st.session_state.context = scorecard_to_context_string(scorecard)

                # ── Reset conversation on new data load ─────────
                st.session_state.messages = []

                if "_error" not in scorecard:
                    st.success(f"✅ Loaded: {st.session_state.entity}")
                else:
                    st.error(scorecard["_error"])

                # ── Debug expander ──────────────────────────────
                with st.expander("🔎 JSON structure debug (click to inspect)", expanded=False):
                    raw = scorecard.get("_raw", scorecard)
                    st.caption("Top-level keys in your JSON:")
                    st.code(", ".join(raw.keys()) if isinstance(raw, dict) else str(type(raw)))
                    st.caption("Normalised metadata:")
                    st.json(scorecard.get("metadata", {}))
                    st.caption("Normalised risk_score:")
                    st.json(scorecard.get("risk_score", {}))

        st.divider()

        # ── Sidebar: scorecard snapshot ─────────────────────────
        if st.session_state.scorecard and "_error" not in st.session_state.scorecard:
            render_scorecard_sidebar(st.session_state.scorecard)

        st.divider()
        st.caption(f"LLM: {GROQ_MODEL}")
        st.caption("Phase 4 · Adverse Media Copilot")

    # ── Guard: require data before showing chat ──────────────────
    if not st.session_state.scorecard:
        st.info(
            "👈 Upload your Phase 3 scorecard JSON and CSV in the sidebar, "
            "then click **Load Data** to start chatting."
        )
        st.stop()

    if "_error" in st.session_state.scorecard:
        st.error(
            f"Could not load scorecard: {st.session_state.scorecard['_error']}\n\n"
            "Please check the file path or upload the file directly."
        )
        st.stop()

    entity = st.session_state.entity

    # Resolve risk label for use in quick questions and welcome message
    risk_label = st.session_state.scorecard.get(
        "risk_score", {}
    ).get("risk_label", "Unknown")

    # ── Chat area: render history ────────────────────────────────
    chat_container = st.container()
    with chat_container:
        # Welcome message on first load
        if not st.session_state.messages:
            with st.chat_message("assistant", avatar="🔍"):
                entity_score = st.session_state.scorecard.get(
                    "risk_score", {}
                ).get("entity_risk_score", "?")
                st.markdown(
                    f"Hello. I've loaded the risk scorecard for **{entity}**.\n\n"
                    f"The entity has been rated **{risk_label}** with a score of "
                    f"**{entity_score}/100**.\n\n"
                    f"Ask me anything about this entity's adverse media profile — "
                    f"what drove the score, which articles flagged, what to do next, and more."
                )

        # Render existing conversation history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🔍"):
                st.markdown(msg["content"])

    # ── Quick question buttons ───────────────────────────────────
    st.divider()
    # FIX 1: pass risk_label so button label matches actual entity rating
    render_quick_questions(entity, risk_label)

    # ── Chat input ───────────────────────────────────────────────
    user_input = st.chat_input(
        f"Ask about {entity}'s risk profile...",
        key="chat_input",
    )

    # Handle quick-question button clicks
    if st.session_state.pending_question and not user_input:
        user_input = st.session_state.pending_question
        st.session_state.pending_question = None

    # ── Process user message ─────────────────────────────────────
    if user_input:
        # Append user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # ── Detect intent ────────────────────────────────────────
        intent = detect_intent(user_input)

        # ── Build system prompt with scorecard context ───────────
        system_prompt = build_system_prompt(
            context=st.session_state.context,
            entity=entity,
        )

        # ── Call LLM ─────────────────────────────────────────────
        with st.chat_message("assistant", avatar="🔍"):
            with st.spinner("Analyzing..."):
                response = call_groq(
                    system_prompt=system_prompt,
                    conversation_history=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ],
                    user_message=user_input,
                    intent=intent,
                    entity=entity,
                )
            st.markdown(response)
            st.caption(f"Intent detected: `{intent}` · Model: `{GROQ_MODEL}`")

        # Append assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # ── Articles expander ────────────────────────────────────────
    if not st.session_state.df_articles.empty:
        with st.expander("📰 View Adverse Articles Table", expanded=False):
            df = st.session_state.df_articles.copy()
            display_cols = [
                c for c in [
                    "title", "source", "publication_date",
                    "adverse_category", "confidence_score",
                    "article_risk_score", "reason"
                ] if c in df.columns
            ]
            st.dataframe(
                df[display_cols].reset_index(drop=True),
                use_container_width=True,
                height=300,
            )


if __name__ == "__main__":
    main()
