# src/app.py
from __future__ import annotations

import os
import re
import streamlit as st
from dotenv import load_dotenv

from extract import extract
from schema import Extraction
from sheets import get_client, upsert_daily_row, append_signals
from validate import get_missing_questions, FIELD_TYPES

import json

service_json = json.dumps(dict(st.secrets["gcp_service_account"]))
sheet_name = st.secrets["GOOGLE_SHEET_NAME"]


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
        hh = int(hh)
        mm = int(mm)
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


def build_followups_block(answers: dict) -> str:
    lines = ["", "FOLLOW-UPS (added after prompting):"]
    for k, v in answers.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


st.set_page_config(page_title="Journal → Data", layout="centered")
st.title("Journal → Data")

d = st.date_input("Date")
entry = st.text_area("Journal entry", height=300, placeholder="Paste your journal entry here...")

if "pending_data" not in st.session_state:
    st.session_state.pending_data = None
if "pending_entry" not in st.session_state:
    st.session_state.pending_entry = None
if "pending_date" not in st.session_state:
    st.session_state.pending_date = None

col1, col2 = st.columns(2)

with col1:
    if st.button("Extract"):
        if not entry.strip():
            st.error("Paste an entry first.")
        else:
            extraction = extract(entry=entry, date=str(d))
            st.session_state.pending_data = extraction.model_dump()
            st.session_state.pending_entry = entry
            st.session_state.pending_date = str(d)

with col2:
    if st.button("Reset"):
        st.session_state.pending_data = None
        st.session_state.pending_entry = None
        st.session_state.pending_date = None

data = st.session_state.pending_data
if data:
    st.subheader("Extracted JSON")
    st.json(data)

    missing = get_missing_questions(data)
    answers_raw = {}

    if missing:
        st.subheader("Missing required fields")
        st.caption("Fill these in and click **Confirm & Save**. Inputs are validated.")

        for field, q in missing:
            ftype = FIELD_TYPES.get(field, "str")

            if ftype == "bool":
                answers_raw[field] = st.selectbox(q, ["", "yes", "no"], key=f"ask_{field}")
            else:
                answers_raw[field] = st.text_input(q, value="", key=f"ask_{field}")

    if st.button("Confirm & Save"):
        # Apply followups with strict parsing
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
            st.error("Fix these inputs:\n- " + "\n- ".join(errors))
            st.stop()

        # Update data with followups
        data.update(answers)
        data["date"] = st.session_state.pending_date  # force date

        # Append followups into full journal entry text
        full_entry = st.session_state.pending_entry or ""
        if answers:
            full_entry = full_entry

        # Validate with Pydantic
        try:
            validated = Extraction.model_validate(data).model_dump()
        except Exception as e:
            st.error(f"Pydantic validation failed: {e}")
            st.stop()

        # Sheets config
        service_json = json.dumps(dict(st.secrets["gcp_service_account"]))
        sheet_name = st.secrets["GOOGLE_SHEET_NAME"]

        # Write to Sheets
        gc = get_client(service_json)
        sh = gc.open(sheet_name)
        daily_ws = sh.worksheet("Daily")
        signals_ws = sh.worksheet("Signals")

        daily_row = dict(validated)
        daily_row["full_entry"] = full_entry

        upsert_daily_row(daily_ws, daily_row)
        append_signals(signals_ws, validated["date"], validated.get("signals", []))

        st.success("✅ Logged to Google Sheets.")
        st.session_state.pending_data = None
        st.session_state.pending_entry = None
        st.session_state.pending_date = None
