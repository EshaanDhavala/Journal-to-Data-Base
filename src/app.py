# src/app.py
from __future__ import annotations

import datetime
import json
import re

import altair as alt
import pandas as pd
import streamlit as st

from extract import extract, client as openai_client
from schema import Extraction
from sheets import get_client, upsert_daily_row, append_signals
from validate import get_missing_questions, FIELD_TYPES

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Journal → Data",
    layout="wide",
    page_icon="📓",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1080px;
}

/* Metric cards */
div[data-testid="metric-container"] {
    border: 1px solid rgba(128,128,128,0.18);
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    transition: border-color 0.18s;
}
div[data-testid="metric-container"]:hover {
    border-color: rgba(128,128,128,0.40);
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    font-weight: 500;
    letter-spacing: 0.02em;
    transition: transform 0.12s, opacity 0.12s;
}
.stButton > button:hover {
    transform: translateY(-1px);
    opacity: 0.88;
}

/* Monospace journal textarea */
.stTextArea textarea {
    font-family: 'SF Mono', 'Monaco', 'Menlo', 'Courier New', monospace;
    font-size: 0.84rem;
    line-height: 1.55;
    border-radius: 10px;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-weight: 500;
    font-size: 0.97rem;
    padding: 0.55rem 1.4rem;
}

/* Dividers */
hr { opacity: 0.25; }

/* Dataframe */
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* Hide Vega-Embed toolbar (the "..." menu with source/download options) */
.vega-embed details { display: none !important; }
.vega-embed summary { display: none !important; }
.vega-embed .vega-actions { display: none !important; }

/* Streak badges */
.streak-card {
    border: 1px solid rgba(128,128,128,0.18);
    border-radius: 12px;
    padding: 0.85rem 1rem;
    text-align: center;
    transition: border-color 0.18s;
}
.streak-card:hover { border-color: rgba(128,128,128,0.40); }
.streak-count { font-size: 1.75rem; font-weight: 700; line-height: 1.2; }
.streak-sub { font-size: 0.78rem; opacity: 0.60; margin-top: 0.15rem; }
.streak-zero .streak-count { opacity: 0.35; }

/* NL query chat history */
.qa-block {
    border-left: 3px solid rgba(128,128,128,0.25);
    padding: 0.5rem 0.9rem;
    margin-bottom: 0.75rem;
    border-radius: 0 8px 8px 0;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Secrets
# ─────────────────────────────────────────────────────────────────────────────
_service_json = json.dumps(dict(st.secrets["gcp_service_account"]))
_sheet_name = st.secrets["GOOGLE_SHEET_NAME"]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_bool(s: str):
    s = s.strip().lower()
    if s in ("y", "yes", "true", "1"):
        return True
    if s in ("n", "no", "false", "0"):
        return False
    return None


def parse_value(field: str, raw: str):
    raw = str(raw).strip()
    ftype = FIELD_TYPES.get(field, "str")

    if ftype == "bool":
        b = parse_bool(raw)
        if b is None:
            raise ValueError("Enter yes/no.")
        return b

    if ftype == "time_hhmm":
        if not re.match(r"^\d{1,2}:\d{2}$", raw):
            raise ValueError("Must be HH:MM (e.g., 07:45 or 12:30).")
        hh, mm = raw.split(":")
        hh, mm = int(hh), int(mm)
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            raise ValueError("Invalid time.")
        return f"{hh:02d}:{mm:02d}"

    if ftype == "int_1_10":
        v = int(float(raw))
        if not (1 <= v <= 10):
            raise ValueError("Must be between 1 and 10.")
        return v

    if ftype == "int_nonneg":
        v = int(float(raw))
        if v < 0:
            raise ValueError("Must be >= 0.")
        return v

    if ftype == "float_1_10":
        v = float(raw)
        if not (1 <= v <= 10):
            raise ValueError("Must be between 1 and 10.")
        return v

    if ftype == "float_nonneg":
        v = float(raw)
        if v < 0:
            raise ValueError("Must be >= 0.")
        return v

    if ftype == "float":
        return float(raw)

    return raw


def _fmt(row, col, suffix="", as_int=False, decimals=1):
    """Safely format a value from a pandas Series for st.metric."""
    try:
        v = row[col]
        if pd.isna(v):
            return None
        if as_int:
            return f"{int(v)}{suffix}"
        return f"{round(float(v), decimals)}{suffix}"
    except Exception:
        return None


def _delta(latest, prev, col):
    """Return numeric delta between two Series rows, or None."""
    if prev is None:
        return None
    try:
        v, p = latest[col], prev[col]
        if pd.notna(v) and pd.notna(p):
            return round(float(v) - float(p), 2)
    except Exception:
        pass
    return None


@st.cache_data(ttl=120)
def load_sheet_data(_gc, sheet_name: str) -> pd.DataFrame:
    sh = _gc.open(sheet_name)
    ws = sh.worksheet("Daily")
    records = ws.get_all_records()
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    # Blank cells from gspread come back as "" — replace with NaN so dropna()
    # correctly skips columns that weren't tracked yet.
    df = df.replace("", None)
    if "date" not in df.columns:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    numeric_cols = [
        "sleep_hours", "sleep_quality_1_10", "workout_minutes", "cardio_minutes",
        "water_bottles", "screen_time_hours", "study_hours",
        "calories_est", "protein_est", "mood_1_10", "weight",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _trend(df: pd.DataFrame, y: str, title: str, color: str = "#6ca0dc", height: int = 195):
    sub = df[["date", y]].dropna()
    if sub.empty:
        return None
    base = alt.Chart(sub).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %d", labelAngle=-30, tickCount=6)),
        y=alt.Y(f"{y}:Q", title=None, scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip("date:T", title="Date", format="%b %d, %Y"),
            alt.Tooltip(f"{y}:Q", title=title, format=".1f"),
        ],
    )
    return (
        base.mark_line(color=color, strokeWidth=2.2)
        + base.mark_circle(color=color, size=48, opacity=0.9)
    ).properties(height=height, title=title).interactive()


def _bars(df: pd.DataFrame, y: str, title: str, color: str = "#6ca0dc", height: int = 195):
    sub = df[["date", y]].dropna()
    if sub.empty:
        return None
    return (
        alt.Chart(sub)
        .mark_bar(color=color, opacity=0.80)
        .encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %d", labelAngle=-30, tickCount=6)),
            y=alt.Y(f"{y}:Q", title=None),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%b %d, %Y"),
                alt.Tooltip(f"{y}:Q", title=title),
            ],
        )
        .properties(height=height, title=title)
        .interactive()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Streak helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_truthy(val) -> bool:
    """Normalize gym/creatine column values to bool."""
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes")


def _gym_streak(df: pd.DataFrame) -> int:
    """
    Gym streak where ONE consecutive rest day is allowed (Thursday is the
    planned rest day). TWO or more consecutive rest days = streak broken.
    Returns the number of days in the current unbroken window.
    """
    if "gym" not in df.columns or df.empty:
        return 0
    rows = df.sort_values("date", ascending=False)[["date", "gym"]]
    streak = 0
    consec_rest = 0
    for _, row in rows.iterrows():
        val = row["gym"]
        if pd.isna(val):
            continue
        if _is_truthy(val):
            streak += 1
            consec_rest = 0
        else:
            consec_rest += 1
            if consec_rest >= 2:
                break
            streak += 1  # single rest day stays inside the streak window
    return streak


def _simple_streak(df: pd.DataFrame, col: str, condition) -> int:
    """
    Consecutive days (most recent first) where condition(value) is True.
    NaN rows are skipped rather than treated as failures.
    """
    if col not in df.columns or df.empty:
        return 0
    rows = df.sort_values("date", ascending=False)[["date", col]]
    streak = 0
    for _, row in rows.iterrows():
        val = row[col]
        if pd.isna(val):
            continue
        try:
            if condition(val):
                streak += 1
            else:
                break
        except Exception:
            break
    return streak


def _streak_card(emoji: str, label: str, count: int, sub: str = "") -> str:
    """Return HTML for a streak badge card."""
    zero_cls = " streak-zero" if count == 0 else ""
    count_display = str(count) if count > 0 else "—"
    unit = f" day{'s' if count != 1 else ''}" if count > 0 else ""
    sub_html = f'<div class="streak-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="streak-card{zero_cls}">'
        f'<div style="font-size:1.5rem">{emoji}</div>'
        f'<div style="font-size:0.85rem;font-weight:600;margin:0.2rem 0">{label}</div>'
        f'<div class="streak-count">{count_display}{unit}</div>'
        f'{sub_html}'
        f'</div>'
    )


def _parse_foods_for_context(df: pd.DataFrame) -> str:
    """
    Parse the 'foods' column (JSON or Python repr string) into a human-readable
    text log for GPT context: one line per day listing food names.
    """
    if "foods" not in df.columns:
        return ""
    lines = []
    for _, row in df.sort_values("date", ascending=False).iterrows():
        raw = row.get("foods")
        if not raw or pd.isna(raw):
            continue
        date_str = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])
        items = []
        try:
            parsed = json.loads(str(raw))
            if isinstance(parsed, list):
                items = [f.get("name", "") for f in parsed if isinstance(f, dict)]
        except (json.JSONDecodeError, TypeError):
            # Fallback: extract quoted strings or name values via regex
            items = re.findall(r"'name':\s*'([^']+)'|\"name\":\s*\"([^\"]+)\"", str(raw))
            items = [a or b for a, b in items]
        if items:
            lines.append(f"{date_str}: {', '.join(i for i in items if i)}")
    return "\n".join(lines)


def _build_metrics_context(df: pd.DataFrame, max_rows: int = 90) -> str:
    """Build a compact CSV-like text of key metrics for GPT context."""
    cols = [c for c in [
        "date", "calories_est", "protein_est", "sleep_hours", "sleep_quality_1_10",
        "mood_1_10", "weight", "gym", "workout_type", "workout_minutes",
        "water_bottles", "study_hours", "screen_time_hours", "creatine",
    ] if c in df.columns]
    sub = df.sort_values("date", ascending=False).head(max_rows)[cols].copy()
    sub["date"] = sub["date"].dt.strftime("%Y-%m-%d")
    return sub.to_csv(index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
for _k in ("pending_data", "pending_entry", "pending_date", "weekly_review_text", "nl_answers"):
    if _k not in st.session_state:
        st.session_state[_k] = None
if st.session_state.nl_answers is None:
    st.session_state.nl_answers = []

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_log, tab_dash, tab_review, tab_ask = st.tabs(
    ["Log Entry", "Dashboard", "Weekly Review", "Ask Your Data"]
)

# ═════════════════════════════════════════════════════════════════════════════
# LOG ENTRY TAB
# ═════════════════════════════════════════════════════════════════════════════
with tab_log:
    st.markdown("## Journal → Data")
    st.caption("Extract structured health metrics from your daily journal entry.")

    c_date, _ = st.columns([1, 3])
    with c_date:
        d = st.date_input("Date", key="date_input")

    entry = st.text_area(
        "Journal entry",
        height=260,
        placeholder="Paste your journal entry here…",
        key="journal_entry",
    )

    col_extract, col_reset, col_pad = st.columns([2, 1, 2])
    with col_extract:
        extract_clicked = st.button("Extract", type="primary", use_container_width=True)
    with col_reset:
        reset_clicked = st.button("↺  Reset", use_container_width=True)

    if extract_clicked:
        if not entry.strip():
            st.error("Paste a journal entry first.")
        else:
            with st.spinner("Extracting with AI…"):
                try:
                    result = extract(entry=entry, date=str(d))
                    st.session_state.pending_data = result.model_dump()
                    st.session_state.pending_entry = entry
                    st.session_state.pending_date = str(d)
                except Exception as e:
                    st.error(f"Extraction failed: {e}")

    if reset_clicked:
        for _k in ("pending_data", "pending_entry", "pending_date"):
            st.session_state[_k] = None
        st.rerun()

    # ── Extracted data display ────────────────────────────────────────────
    data = st.session_state.pending_data
    if data:
        st.markdown("---")
        st.markdown("### Extracted Data")

        # Sleep
        st.markdown("#### 💤 Sleep")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Wake Time", data.get("wake_time") or "—")
        sleep_h = data.get("sleep_hours")
        sc2.metric("Sleep Hours", sleep_h if sleep_h is not None else "—")
        sq = data.get("sleep_quality_1_10")
        sc3.metric("Sleep Quality", f"{sq}/10" if sq is not None else "—")

        # Training
        st.markdown("#### 🏋️ Training")
        tc1, tc2, tc3, tc4 = st.columns(4)
        gym_v = data.get("gym")
        tc1.metric("Gym", "Yes" if gym_v is True else ("No" if gym_v is False else "—"))
        tc2.metric("Type", data.get("workout_type") or "—")
        wm = data.get("workout_minutes")
        tc3.metric("Workout Mins", wm if wm is not None else "—")
        cm = data.get("cardio_minutes")
        tc4.metric("Cardio Mins", cm if cm is not None else "—")

        # Nutrition
        st.markdown("#### 🥗 Nutrition")
        nc1, nc2, nc3 = st.columns(3)
        cal = data.get("calories_est")
        nc1.metric("Calories", cal if cal is not None else "—")
        pro = data.get("protein_est")
        nc2.metric("Protein (g)", pro if pro is not None else "—")
        wb = data.get("water_bottles")
        nc3.metric("Water Bottles", wb if wb is not None else "—")

        # Food breakdown
        foods = data.get("foods") or []
        if foods:
            rows = []
            for f in foods:
                if not isinstance(f, dict):
                    continue
                src = f.get("source", "model_estimate")
                name = f.get("name", "")
                if src == "known_food":
                    name = name + "  ✓"
                rows.append({
                    "Food": name,
                    "Quantity": f.get("quantity_text", ""),
                    "Calories": f.get("calories", ""),
                    "Protein (g)": f.get("protein_g", ""),
                    "Conf.": f"{float(f.get('confidence', 0.7)):.0%}",
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.caption("✓ = verified nutrition data from known-food database.")

        # Habits & Mood
        st.markdown("#### 🧠 Habits & Mood")
        hc1, hc2, hc3, hc4, hc5 = st.columns(5)
        mood = data.get("mood_1_10")
        hc1.metric("Mood", f"{mood}/10" if mood is not None else "—")
        st_h = data.get("screen_time_hours")
        hc2.metric("Screen Time", f"{st_h}h" if st_h is not None else "—")
        sh = data.get("study_hours")
        hc3.metric("Study Hours", sh if sh is not None else "—")
        wt = data.get("weight")
        hc4.metric("Weight", f"{wt} lbs" if wt is not None else "—")
        cre = data.get("creatine")
        hc5.metric("Creatine", "Yes" if cre is True else ("No" if cre is False else "—"))

        if data.get("summary"):
            st.info(f"**Summary:** {data['summary']}")

        # ── Missing fields ────────────────────────────────────────────────
        missing = get_missing_questions(data)
        answers_raw = {}

        if missing:
            st.markdown("---")
            st.markdown("### ⚠️ Missing Fields")
            st.caption("Fill in the blanks before saving.")
            mid = (len(missing) + 1) // 2
            m1, m2 = st.columns(2)
            for i, (field, q) in enumerate(missing):
                ftype = FIELD_TYPES.get(field, "str")
                col = m1 if i < mid else m2
                with col:
                    if ftype == "bool":
                        answers_raw[field] = st.selectbox(q, ["", "yes", "no"], key=f"ask_{field}")
                    else:
                        answers_raw[field] = st.text_input(q, value="", key=f"ask_{field}")

        # ── Save ─────────────────────────────────────────────────────────
        st.markdown("---")
        if st.button("💾  Confirm & Save", type="primary", use_container_width=True):
            answers = {}
            errors = []
            for field, val in answers_raw.items():
                if val is None:
                    continue
                val_str = str(val).strip()
                if val_str == "":
                    continue
                try:
                    answers[field] = parse_value(field, val_str)
                except Exception as e:
                    errors.append(f"{field}: {e}")

            if errors:
                st.error("Fix inputs:\n- " + "\n- ".join(errors))
                st.stop()

            data.update(answers)
            data["date"] = st.session_state.pending_date

            try:
                validated = Extraction.model_validate(data).model_dump()
            except Exception as e:
                st.error(f"Validation error: {e}")
                st.stop()

            gc = get_client(_service_json)
            sh = gc.open(_sheet_name)
            daily_ws = sh.worksheet("Daily")
            signals_ws = sh.worksheet("Signals")

            daily_row = dict(validated)
            daily_row["full_entry"] = st.session_state.pending_entry or ""
            upsert_daily_row(daily_ws, daily_row)
            append_signals(signals_ws, validated["date"], validated.get("signals", []))

            load_sheet_data.clear()
            st.success("✅ Saved to Google Sheets!")
            for _k in ("pending_data", "pending_entry", "pending_date"):
                st.session_state[_k] = None
            st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# DASHBOARD TAB
# ═════════════════════════════════════════════════════════════════════════════
with tab_dash:
    st.markdown("## Dashboard")

    try:
        gc_dash = get_client(_service_json)
        df_all = load_sheet_data(gc_dash, _sheet_name)
    except Exception as e:
        st.error(f"Could not load sheet data: {e}")
        df_all = pd.DataFrame()

    if df_all.empty:
        st.info("No data yet. Log some entries and they'll appear here.")
    else:
        # ── Time range selector ───────────────────────────────────────────
        range_map = {"Last 7 days": 7, "Last 14 days": 14, "Last 30 days": 30, "All time": 0}
        r_col, _ = st.columns([1, 3])
        with r_col:
            selected = st.selectbox("Time range", list(range_map.keys()), index=2)
        n = range_map[selected]
        if n > 0:
            cutoff = df_all["date"].max() - pd.Timedelta(days=n - 1)
            df = df_all[df_all["date"] >= cutoff].copy()
        else:
            df = df_all.copy()

        # ── Latest metrics with deltas ────────────────────────────────────
        latest = df_all.iloc[-1]
        prev = df_all.iloc[-2] if len(df_all) >= 2 else None

        st.markdown("### Latest Entry")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(
            "Weight",
            _fmt(latest, "weight", " lbs"),
            delta=_delta(latest, prev, "weight"),
        )
        m2.metric(
            "Calories",
            _fmt(latest, "calories_est", as_int=True),
            delta=_delta(latest, prev, "calories_est"),
        )
        m3.metric(
            "Protein",
            _fmt(latest, "protein_est", " g", as_int=True),
            delta=_delta(latest, prev, "protein_est"),
        )
        m4.metric(
            "Sleep",
            _fmt(latest, "sleep_hours", " h"),
            delta=_delta(latest, prev, "sleep_hours"),
        )
        m5.metric(
            "Mood",
            _fmt(latest, "mood_1_10", "/10"),
            delta=_delta(latest, prev, "mood_1_10"),
        )

        # ── Streaks ───────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔥 Streaks")
        s1, s2, s3, s4 = st.columns(4)
        gym_s = _gym_streak(df_all)
        cre_s = _simple_streak(df_all, "creatine", _is_truthy)
        sleep_s = _simple_streak(df_all, "sleep_hours", lambda v: float(v) >= 8)
        prot_s = _simple_streak(df_all, "protein_est", lambda v: float(v) >= 100)
        with s1:
            st.markdown(_streak_card("🏋️", "Gym", gym_s, "≤1 rest/row allowed"), unsafe_allow_html=True)
        with s2:
            st.markdown(_streak_card("💊", "Creatine", cre_s), unsafe_allow_html=True)
        with s3:
            st.markdown(_streak_card("💤", "Sleep ≥7h", sleep_s), unsafe_allow_html=True)
        with s4:
            st.markdown(_streak_card("🥩", "Protein ≥140g", prot_s), unsafe_allow_html=True)

        st.markdown("---")

        # ── Nutrition ─────────────────────────────────────────────────────
        st.markdown("### 🥗 Nutrition")
        n1, n2 = st.columns(2)
        with n1:
            c = _trend(df, "calories_est", "Calories", "#fab387")
            if c:
                st.altair_chart(c, use_container_width=True)
        with n2:
            c = _trend(df, "protein_est", "Protein (g)", "#a6e3a1")
            if c:
                st.altair_chart(c, use_container_width=True)

        # ── Body & Sleep ──────────────────────────────────────────────────
        st.markdown("### 💤 Body & Sleep")
        b1, b2, b3 = st.columns(3)
        with b1:
            c = _trend(df, "weight", "Weight (lbs)", "#89b4fa")
            if c:
                st.altair_chart(c, use_container_width=True)
        with b2:
            c = _bars(df, "sleep_hours", "Sleep Hours", "#b4befe")
            if c:
                st.altair_chart(c, use_container_width=True)
        with b3:
            c = _trend(df, "sleep_quality_1_10", "Sleep Quality (1–10)", "#cba6f7")
            if c:
                st.altair_chart(c, use_container_width=True)

        # ── Performance ───────────────────────────────────────────────────
        st.markdown("### 🧠 Performance")
        p1, p2, p3 = st.columns(3)
        with p1:
            c = _trend(df, "mood_1_10", "Mood (1–10)", "#f9e2af")
            if c:
                st.altair_chart(c, use_container_width=True)
        with p2:
            c = _bars(df, "study_hours", "Study Hours", "#89dceb")
            if c:
                st.altair_chart(c, use_container_width=True)
        with p3:
            c = _bars(df, "screen_time_hours", "Screen Time (h)", "#f38ba8")
            if c:
                st.altair_chart(c, use_container_width=True)

        # ── Training ──────────────────────────────────────────────────────
        st.markdown("### 🏋️ Training")
        t1, t2 = st.columns(2)
        with t1:
            c = _bars(df, "workout_minutes", "Workout Minutes", "#a6e3a1")
            if c:
                st.altair_chart(c, use_container_width=True)
        with t2:
            if "workout_type" in df.columns:
                wt_counts = df["workout_type"].dropna().value_counts().reset_index()
                wt_counts.columns = ["type", "count"]
                if not wt_counts.empty:
                    donut = (
                        alt.Chart(wt_counts)
                        .mark_arc(innerRadius=48)
                        .encode(
                            theta=alt.Theta("count:Q"),
                            color=alt.Color(
                                "type:N",
                                scale=alt.Scale(scheme="tableau10"),
                                legend=alt.Legend(title="Type"),
                            ),
                            tooltip=["type:N", "count:Q"],
                        )
                        .properties(height=195, title="Workout Type Distribution")
                    )
                    st.altair_chart(donut, use_container_width=True)

        # ── Habits ────────────────────────────────────────────────────────
        st.markdown("### 💧 Habits")
        h1, h2 = st.columns(2)
        with h1:
            c = _bars(df, "water_bottles", "Water Bottles", "#74c7ec")
            if c:
                st.altair_chart(c, use_container_width=True)
        with h2:
            if "creatine" in df.columns:
                cre_df = df[["date", "creatine"]].dropna().copy()
                cre_df["creatine"] = cre_df["creatine"].map(
                    lambda x: 1 if str(x).strip().lower() in ("true", "1", "yes") else 0
                )
                c = _bars(cre_df, "creatine", "Creatine (1 = taken)", "#cba6f7")
                if c:
                    st.altair_chart(c, use_container_width=True)

        # ── Raw data ──────────────────────────────────────────────────────
        st.markdown("---")
        with st.expander("📋 Raw Data Table"):
            show_cols = [c for c in [
                "date", "weight", "calories_est", "protein_est",
                "sleep_hours", "sleep_quality_1_10", "mood_1_10",
                "workout_type", "workout_minutes", "cardio_minutes",
                "water_bottles", "study_hours", "screen_time_hours",
            ] if c in df.columns]
            disp = df[show_cols].copy()
            disp["date"] = disp["date"].dt.strftime("%Y-%m-%d")
            st.dataframe(
                disp.sort_values("date", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

        if st.button("↺  Refresh Data"):
            load_sheet_data.clear()
            st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# WEEKLY REVIEW TAB
# ═════════════════════════════════════════════════════════════════════════════
with tab_review:
    st.markdown("## Weekly Review")
    st.caption("GPT-generated narrative summary of your last 7 days.")

    try:
        gc_review = get_client(_service_json)
        df_rev_all = load_sheet_data(gc_review, _sheet_name)
    except Exception as e:
        st.error(f"Could not load sheet data: {e}")
        df_rev_all = pd.DataFrame()

    if df_rev_all.empty:
        st.info("No data yet. Log some entries first.")
    else:
        today_dt = datetime.date.today()
        default_start = today_dt - datetime.timedelta(days=6)

        wc1, wc2, _ = st.columns([1, 1, 2])
        with wc1:
            week_start = st.date_input("Week start", value=default_start, key="review_start")
        with wc2:
            week_end = st.date_input("Week end", value=today_dt, key="review_end")

        gen_btn = st.button("Generate Review", type="primary")

        if gen_btn:
            # Clear cached review when user explicitly regenerates
            st.session_state.weekly_review_text = None

        if gen_btn or (st.session_state.weekly_review_text is None and not gen_btn):
            # Only auto-generate if there's no cached review (first load)
            pass

        if gen_btn:
            ws_dt = pd.Timestamp(week_start)
            we_dt = pd.Timestamp(week_end)
            week_df = df_rev_all[
                (df_rev_all["date"] >= ws_dt) & (df_rev_all["date"] <= we_dt)
            ].copy()

            if week_df.empty:
                st.warning("No entries found for that date range.")
            else:
                def _safe_avg(col):
                    if col in week_df.columns:
                        s = pd.to_numeric(week_df[col], errors="coerce").dropna()
                        return round(s.mean(), 1) if not s.empty else "n/a"
                    return "n/a"

                gym_days = 0
                if "gym" in week_df.columns:
                    gym_days = int(week_df["gym"].apply(
                        lambda v: 1 if _is_truthy(v) else 0
                    ).sum())

                workout_types = ""
                if "workout_type" in week_df.columns:
                    wt = week_df["workout_type"].dropna().tolist()
                    workout_types = ", ".join(wt) if wt else "none"

                creatine_days = 0
                if "creatine" in week_df.columns:
                    creatine_days = int(week_df["creatine"].apply(
                        lambda v: 1 if _is_truthy(v) else 0
                    ).sum())

                best_mood_row = None
                worst_mood_row = None
                if "mood_1_10" in week_df.columns:
                    mood_col = pd.to_numeric(week_df["mood_1_10"], errors="coerce")
                    if not mood_col.dropna().empty:
                        best_mood_row = week_df.loc[mood_col.idxmax()]
                        worst_mood_row = week_df.loc[mood_col.idxmin()]

                summaries = ""
                if "summary" in week_df.columns:
                    rows_s = week_df[["date", "summary"]].dropna(subset=["summary"])
                    if not rows_s.empty:
                        summaries = "\n".join(
                            f"  {r['date'].strftime('%a %b %d')}: {r['summary']}"
                            for _, r in rows_s.iterrows()
                        )

                foods_log = _parse_foods_for_context(week_df)

                stats = f"""
Week: {week_start} → {week_end} ({len(week_df)} entries)

TRAINING
  Gym sessions: {gym_days} / {len(week_df)} days
  Workout types: {workout_types}

NUTRITION
  Avg calories: {_safe_avg('calories_est')} kcal
  Avg protein:  {_safe_avg('protein_est')} g
  Avg water:    {_safe_avg('water_bottles')} bottles

SLEEP & RECOVERY
  Avg sleep:         {_safe_avg('sleep_hours')} h
  Avg sleep quality: {_safe_avg('sleep_quality_1_10')} / 10

MIND & PERFORMANCE
  Avg mood:     {_safe_avg('mood_1_10')} / 10{
  f" (best: {best_mood_row['mood_1_10']} on {best_mood_row['date'].strftime('%a')}, worst: {worst_mood_row['mood_1_10']} on {worst_mood_row['date'].strftime('%a')})"
  if best_mood_row is not None else ""}
  Avg study:    {_safe_avg('study_hours')} h
  Creatine:     {creatine_days} / {len(week_df)} days

WEIGHT
  Latest: {_safe_avg('weight')} lbs (avg over week)

DAILY SUMMARIES
{summaries if summaries else "  (none logged)"}

FOODS EATEN THIS WEEK
{foods_log if foods_log else "  (not available)"}
""".strip()

                with st.spinner("Generating review..."):
                    try:
                        resp = openai_client.chat.completions.create(
                            model="gpt-4.1-mini",
                            temperature=0.7,
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "You are a personal health coach reviewing someone's week of journal data. "
                                        "Write a concise, encouraging, and specific weekly review. "
                                        "Cover: training consistency, nutrition highlights, sleep quality, "
                                        "mood/energy trends, and 2-3 actionable suggestions for next week. "
                                        "Use markdown with headers. Keep it personal and motivating, not generic."
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": f"Here is my week's data:\n\n{stats}\n\nWrite my weekly review.",
                                },
                            ],
                        )
                        st.session_state.weekly_review_text = resp.choices[0].message.content
                    except Exception as e:
                        st.error(f"GPT error: {e}")

        if st.session_state.weekly_review_text:
            st.markdown("---")
            st.markdown(st.session_state.weekly_review_text)


# ═════════════════════════════════════════════════════════════════════════════
# ASK YOUR DATA TAB
# ═════════════════════════════════════════════════════════════════════════════
with tab_ask:
    st.markdown("## Ask Your Data")
    st.caption("Ask anything about your health history — foods, habits, trends, counts.")

    try:
        gc_ask = get_client(_service_json)
        df_ask_all = load_sheet_data(gc_ask, _sheet_name)
    except Exception as e:
        st.error(f"Could not load sheet data: {e}")
        df_ask_all = pd.DataFrame()

    if df_ask_all.empty:
        st.info("No data yet. Log some entries first.")
    else:
        question = st.text_input(
            "Your question",
            placeholder="How many times did I get Taco Bell this month?",
            key="nl_question",
        )
        ask_btn = st.button("Ask", type="primary", key="nl_ask_btn")

        if ask_btn and question.strip():
            metrics_ctx = _build_metrics_context(df_ask_all)
            foods_ctx = _parse_foods_for_context(df_ask_all)
            today_str = datetime.date.today().strftime("%B %d, %Y")

            system_msg = (
                f"You are a data analyst answering questions about someone's personal health journal. "
                f"Today is {today_str}. "
                "Answer precisely and concisely. Give counts or numbers whenever possible. "
                "If something isn't in the data, say so clearly. Do not make up information."
            )
            user_msg = (
                f"Here is my health data (CSV, most recent first):\n\n{metrics_ctx}\n\n"
                f"Here is a log of every food I ate, by date:\n\n{foods_ctx}\n\n"
                f"My question: {question.strip()}"
            )

            with st.spinner("Thinking..."):
                try:
                    resp = openai_client.chat.completions.create(
                        model="gpt-4.1-mini",
                        temperature=0,
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                        ],
                    )
                    answer = resp.choices[0].message.content
                    # Prepend to history (newest first)
                    st.session_state.nl_answers = [
                        {"q": question.strip(), "a": answer}
                    ] + st.session_state.nl_answers[:4]
                except Exception as e:
                    st.error(f"GPT error: {e}")

        # Display Q&A history
        if st.session_state.nl_answers:
            st.markdown("---")
            for qa in st.session_state.nl_answers:
                st.markdown(
                    f'<div class="qa-block">'
                    f'<strong>Q:</strong> {qa["q"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(qa["a"])
                st.markdown("")

        # Transparency expander
        if not df_ask_all.empty:
            with st.expander("🔍 Data context sent to GPT"):
                st.caption("Metrics (last 90 days):")
                st.code(_build_metrics_context(df_ask_all), language="")
                st.caption("Foods log:")
                st.code(_parse_foods_for_context(df_ask_all) or "(none)", language="")
