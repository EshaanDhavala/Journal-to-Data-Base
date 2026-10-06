# export_snapshot.py
"""
Export a public, aggregate-only snapshot of the Daily sheet for the portfolio site.

Only weekly averages and headline totals are written -- no raw daily rows,
journal text, food logs, weight, or favorite moments.

Usage:
    python scripts/export_snapshot.py path/to/journal_snapshot.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sheets import get_client  # noqa: E402

METRICS = ["sleep_hours", "sleep_quality_1_10", "mood_1_10", "workout_minutes", "screen_time_hours"]


def load_daily() -> pd.DataFrame:
    load_dotenv()
    service_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    sheet_name = os.getenv("GOOGLE_SHEET_NAME")
    if not service_json or not sheet_name:
        raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_JSON or GOOGLE_SHEET_NAME in .env")
    if Path(service_json).is_file():  # .env may hold a path instead of inline JSON
        service_json = Path(service_json).read_text()
    ws = get_client(service_json).open(sheet_name).worksheet("Daily")
    df = pd.DataFrame(ws.get_all_records()).replace("", None)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    for c in METRICS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["gym"] = df.get("gym").map(lambda v: str(v).strip().lower() in ("true", "yes", "1"))
    return df


def half_change(df: pd.DataFrame, col: str, agg: str = "mean"):
    mid = len(df) // 2
    a, b = df[col].iloc[:mid], df[col].iloc[mid:]
    first, second = getattr(a, agg)(), getattr(b, agg)()
    if pd.isna(first) or pd.isna(second):
        return None
    return {"first_half": round(float(first), 2), "second_half": round(float(second), 2)}


def build_snapshot(df: pd.DataFrame) -> dict:
    weekly = (
        df.set_index("date")
        .resample("W-MON", label="left", closed="left")
        .agg({**{c: "mean" for c in METRICS if c in df.columns}, "gym": "sum"})
        .rename(columns={"gym": "gym_days"})
    )
    weekly["days_logged"] = df.set_index("date").resample("W-MON", label="left", closed="left").size()
    weekly = weekly[weekly["days_logged"] > 0].round(2)
    weeks = [
        {"week": d.strftime("%Y-%m-%d"), **{k: (None if pd.isna(v) else v) for k, v in row.items()}}
        for d, row in weekly.iterrows()
    ]
    return {
        "range": {"start": df["date"].min().strftime("%Y-%m-%d"), "end": df["date"].max().strftime("%Y-%m-%d")},
        "totals": {
            "days_logged": int(len(df)),
            "gym_days": int(df["gym"].sum()),
            "avg_sleep_hours": round(float(df["sleep_hours"].mean()), 2) if "sleep_hours" in df else None,
            "avg_mood": round(float(df["mood_1_10"].mean()), 2) if "mood_1_10" in df else None,
        },
        "halves": {
            "gym_rate": half_change(df, "gym"),
            "screen_time_hours": half_change(df, "screen_time_hours") if "screen_time_hours" in df else None,
            "sleep_hours": half_change(df, "sleep_hours") if "sleep_hours" in df else None,
            "mood_1_10": half_change(df, "mood_1_10") if "mood_1_10" in df else None,
        },
        "weeks": weeks,
    }


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "journal_snapshot.json")
    snap = build_snapshot(load_daily())
    out.write_text(json.dumps(snap, indent=2))
    print(f"Wrote {len(snap['weeks'])} weeks, {snap['totals']['days_logged']} days -> {out}")
