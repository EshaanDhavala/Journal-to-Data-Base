# main.py
from __future__ import annotations
import re
import os
from dotenv import load_dotenv

from extract import extract
from schema import Extraction
from sheets import get_client, upsert_daily_row, append_signals
from validate import get_missing_questions, FIELD_TYPES


def parse_bool(s: str):
    s = s.strip().lower()
    if s in ("y", "yes", "true", "1"):
        return True
    if s in ("n", "no", "false", "0"):
        return False
    return None


def parse_value(field: str, raw: str):
    raw = raw.strip()
    ftype = FIELD_TYPES.get(field, "str")

    if ftype == "bool":
        b = parse_bool(raw)
        if b is None:
            raise ValueError("Enter yes/no.")
        return b

    if ftype == "int_1_10":
        v = int(float(raw))
        if not (1 <= v <= 10):
            raise ValueError("Must be between 1 and 10.")
        return v

    if ftype == "float_1_10":
        v = float(raw)
        if not (1 <= v <= 10):
            raise ValueError("Must be between 1 and 10.")
        return v

    if ftype == "int_nonneg":
        v = int(float(raw))
        if v < 0:
            raise ValueError("Must be >= 0.")
        return v

    if ftype == "float_nonneg":
        v = float(raw)
        if v < 0:
            raise ValueError("Must be >= 0.")
        return v

    if ftype == "float":
        return float(raw)
    
    if ftype == "time_hhmm":
        if not re.match(r"^\d{1,2}:\d{2}$", raw):
            raise ValueError("Must be HH:MM (e.g., 07:45 or 12:30).")
        hh, mm = raw.split(":")
        hh = int(hh)
        mm = int(mm)
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            raise ValueError("Invalid time.")
        return f"{hh:02d}:{mm:02d}"


    # str
    return raw


def append_followups(entry: str, answers: dict) -> str:
    lines = ["", "FOLLOW-UPS (added after prompting):"]
    for k, v in answers.items():
        lines.append(f"- {k}: {v}")
    return entry + "\n" + "\n".join(lines)


def prompt_for_missing_fields(data: dict, entry: str) -> tuple[dict, str]:
    questions = get_missing_questions(data)
    if not questions:
        return data, entry

    answers = {}

    print("\nMissing required fields. Please answer:")
    for field, q in questions:
        while True:
            raw = input(f"{q} ").strip()
            if raw == "":
                print("  (required) please enter a value.")
                continue
            try:
                val = parse_value(field, raw)
                data[field] = val
                answers[field] = val
                break
            except Exception as e:
                print(f"  Invalid: {e}")

    if answers:
        entry = append_followups(entry, answers)

    return data, entry


def main():
    load_dotenv()

    date = input("Date (YYYY-MM-DD): ").strip()

    print("Paste journal entry. End with a blank line:")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    entry = "\n".join(lines).strip()

    # Extract
    extraction = extract(entry=entry, date=date)
    data = extraction.model_dump()

    # Ensure date is set
    data["date"] = date

    # Prompt for any missing required fields and append to entry
    data, entry = prompt_for_missing_fields(data, entry)

    # Validate everything with Pydantic (hard stop if invalid)
    validated = Extraction.model_validate(data).model_dump()

    # Prepare row for Sheets (header-driven upsert uses keys that match headers)
    daily_row = dict(validated)
    daily_row["full_entry"] = entry

    # Write to Sheets
    service_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    sheet_name = os.getenv("GOOGLE_SHEET_NAME")
    if not service_json or not sheet_name:
        raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_JSON or GOOGLE_SHEET_NAME in .env")

    gc = get_client(service_json)
    sh = gc.open(sheet_name)
    daily_ws = sh.worksheet("Daily")
    signals_ws = sh.worksheet("Signals")

    upsert_daily_row(daily_ws, daily_row)
    append_signals(signals_ws, validated["date"], validated.get("signals", []))

    print("\n✅ Logged to Google Sheets.")


if __name__ == "__main__":
    main()
