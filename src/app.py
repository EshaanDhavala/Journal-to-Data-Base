# src/app.py
from __future__ import annotations

import json
import re

import altair as alt
import pandas as pd
import streamlit as st

from extract import extract
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
        .mark_bar(color=color, opacity=0.80, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
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
# Session state
# ─────────────────────────────────────────────────────────────────────────────
for _k in ("pending_data", "pending_entry", "pending_date"):
    if _k not in st.session_state:
        st.session_state[_k] = None

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_log, tab_dash = st.tabs(["Log Entry", "Dashboard"])

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
                result = extract(entry=entry, date=str(d))
                st.session_state.pending_data = result.model_dump()
                st.session_state.pending_entry = entry
                st.session_state.pending_date = str(d)

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
                cre_df = df[["date", "creatine"]].copy()
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
