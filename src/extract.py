# src/extract.py
from __future__ import annotations

import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

import streamlit as st

from schema import Extraction

load_dotenv()
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

SYSTEM = """You extract structured health/life tracking data from journal entries.
Return ONLY valid JSON (no markdown, no extra text).

Nutrition MUST be represented as an itemized list of foods/drinks with calories and protein for each item.
Do not reuse the same calories/protein numbers across many items unless it is truly the same food/portion.
For ambiguous portions (e.g., "some", "few pieces", "40% of a family sized bag"), estimate conservatively but include macros.

signals must be a JSON array of objects with keys: key, value, unit, source, confidence (0..1).
"""

INSTRUCTIONS = """Return a single JSON object with these keys:

Top-level fields:
- date (YYYY-MM-DD) MUST equal the provided date
- wake_time (string HH:MM in 24h like "07:45" or "12:30", or null)
- sleep_hours (float or null)
- sleep_quality_1_10 (int 1-10 or null)
- gym (true/false or null)
- workout_type (string like: push/pull/legs/tennis/run/rest/other or null)
- workout_minutes (int or null)
- cardio_minutes (int or null)
- water_bottles (int or null)
- creatine (true/false or null)
- screen_time_hours (float or null) IMPORTANT: set null unless explicitly mentioned in the journal text
- study_hours (float or null)
- mood_1_10 (float 1.0-10.0 or null)
- weight (float in lbs, e.g., 165.4, or null)
- summary (short string or null)

Nutrition fields:
- foods: an array of food/drink items. Each item:
  - name (string)
  - quantity_text (string)
  - calories (int) for THIS portion
  - protein_g (int) for THIS portion
  - confidence (0..1)

- calories_est (int or null)  (optional; will be recomputed in code)
- protein_est (int or null)   (optional; will be recomputed in code)

- signals: list of objects like {"key":"...", "value":"...", "unit":"", "source":"journal", "confidence":0.0-1.0}

Rules:
- If food is mentioned, foods must include it and calories/protein must be integers (not null).
- Use conservative typical estimates when exact nutrition is unknown.
- Do not reuse the same calories/protein pair across many different foods unless they are identical.
"""

# Small set of known items (OPTIONAL but helpful). Only applied when the FOOD NAME matches.
KNOWN_FOODS = [
    {
        "known_name": "cafe_1919_personal_goat_cheese_sundried_tomato_pizza",
        "patterns": [
            r"\bgoat\s+cheese\b.*\bsun[- ]dried\s+tomato\b.*\bpizza\b",
            r"\bsun[- ]dried\s+tomato\b.*\bgoat\s+cheese\b.*\bpizza\b",
            r"\bcafe\s*1919\b.*\bpizza\b",
        ],
        "cal": 600,
        "protein": 20,
    },
    {
        "known_name": "built_protein_bar",
        "patterns": [r"\bbuilt\s+protein\s+bar\b", r"\bbuilt\s+bar\b", r"\bbuilt\s+puff\b"],
        "cal": 140,
        "protein": 17,
    },
    {
        "known_name": "fairlife_core_power_shake",
        "patterns": [r"\bcore\s*power\b", r"\bfair\s*life\b", r"\bfairlife\b"],
        "cal": 230,
        "protein": 42,
    },
]


def _safe_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(round(float(x)))
    except Exception:
        return None


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def mentioned_screen_time(entry_text: str) -> bool:
    t = (entry_text or "").lower()
    return any(
        phrase in t
        for phrase in [
            "screen time",
            "screentime",
            "phone time",
            "time on my phone",
            "hours on my phone",
        ]
    )


def normalize_foods(raw_foods: Any) -> List[Dict[str, Any]]:
    if raw_foods is None or not isinstance(raw_foods, list):
        return []
    out: List[Dict[str, Any]] = []
    for f in raw_foods:
        if not isinstance(f, dict):
            continue
        name = str(f.get("name", "")).strip()
        if not name:
            continue
        qty = str(f.get("quantity_text", "")).strip()
        cal = _safe_int(f.get("calories"))
        pro = _safe_int(f.get("protein_g"))
        conf = _safe_float(f.get("confidence"))
        if conf is None:
            conf = 0.7
        out.append({"name": name, "quantity_text": qty, "calories": cal, "protein_g": pro, "confidence": conf})
    return out


def match_known_food_by_name(food_name: str) -> Optional[Dict[str, Any]]:
    """
    IMPORTANT: Match ONLY on the food item's name (NOT the full entry text),
    to avoid overriding everything just because one known item is present.
    """
    fname = (food_name or "").lower()
    for item in KNOWN_FOODS:
        if any(re.search(p, fname, re.IGNORECASE) for p in item["patterns"]):
            return {"calories": item["cal"], "protein_g": item["protein"], "known_name": item["known_name"]}
    return None


def apply_known_overrides(foods: List[Dict[str, Any]], raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    overridden = []
    final = []

    for f in foods:
        known = match_known_food_by_name(f.get("name", ""))
        if known:
            model_cal = f.get("calories")
            model_pro = f.get("protein_g")

            # Override only if missing or far off
            should_override = (
                model_cal is None
                or model_pro is None
                or abs(int(model_cal) - known["calories"]) >= 80
                or abs(int(model_pro) - known["protein_g"]) >= 10
            )

            if should_override:
                f2 = dict(f)
                f2["calories"] = int(known["calories"])
                f2["protein_g"] = int(known["protein_g"])
                f2["confidence"] = max(float(f.get("confidence", 0.7)), 0.85)
                f2["source"] = "known_food"
                f2["matched_known"] = known["known_name"]
                overridden.append(known["known_name"])
                final.append(f2)
                continue

        f2 = dict(f)
        f2.setdefault("source", "model_estimate")
        final.append(f2)

    if overridden:
        raw.setdefault("signals", [])
        raw["signals"].append(
            {
                "key": "known_food_overrides",
                "value": "Overrode macros for: " + ", ".join(sorted(set(overridden))),
                "unit": "",
                "source": "heuristic",
                "confidence": 0.9,
            }
        )
    return final


def sum_food_macros(foods: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    total_cal = 0
    total_pro = 0
    counted = 0
    for f in foods:
        cal = f.get("calories")
        pro = f.get("protein_g")
        if isinstance(cal, int) and isinstance(pro, int):
            total_cal += cal
            total_pro += pro
            counted += 1
    return total_cal, total_pro, counted


def sanity_bad_uniform_macros(foods: List[Dict[str, Any]]) -> bool:
    """
    Catch the failure: lots of different foods all get identical macros (e.g., 600/20).
    """
    pairs = [
        (f.get("calories"), f.get("protein_g"))
        for f in foods
        if isinstance(f.get("calories"), int) and isinstance(f.get("protein_g"), int)
    ]
    if len(pairs) < 4:
        return False
    from collections import Counter

    c = Counter(pairs)
    most_common = c.most_common(1)[0]
    return most_common[1] >= 4


def call_gpt(entry: str, date: str, extra_rules: str = "") -> Dict[str, Any]:
    msg = f"""<RAW_JOURNAL>
{entry}
</RAW_JOURNAL>

Journal entry date (local): {date}

{INSTRUCTIONS}

{extra_rules}
"""
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": msg},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def extract(entry: str, date: str) -> Extraction:
    raw = call_gpt(entry, date)
    raw["date"] = date

    # signals must be list
    sigs = raw.get("signals", [])
    if sigs is None:
        raw["signals"] = []
    elif isinstance(sigs, dict):
        raw["signals"] = [
            {"key": k, "value": str(v), "unit": "", "source": "journal", "confidence": 0.6}
            for k, v in sigs.items()
        ]
    elif not isinstance(sigs, list):
        raw["signals"] = []

    foods = normalize_foods(raw.get("foods"))
    raw["foods"] = foods

    # Apply known overrides (safe — name-only match)
    foods = apply_known_overrides(foods, raw)
    raw["foods"] = foods

    # If sanity check trips, re-ask once with a strict rule against copy-paste macros
    if foods and sanity_bad_uniform_macros(foods):
        raw2 = call_gpt(
            entry,
            date,
            extra_rules="STRICT: Do NOT reuse the same calories/protein across different foods unless identical. Use the portion description to vary estimates conservatively.",
        )
        raw2["date"] = date

        # keep earlier non-nutrition fields from first pass unless second pass has them
        for k in [
            "wake_time",
            "sleep_hours",
            "sleep_quality_1_10",
            "gym",
            "workout_type",
            "workout_minutes",
            "cardio_minutes",
            "water_bottles",
            "creatine",
            "screen_time_hours",
            "study_hours",
            "mood_1_10",
            "weight",
            "summary",
        ]:
            raw2.setdefault(k, raw.get(k))

        # merge signals
        sigs2 = raw2.get("signals", [])
        if sigs2 is None:
            sigs2 = []
        if isinstance(sigs2, dict):
            sigs2 = [
                {"key": k, "value": str(v), "unit": "", "source": "journal", "confidence": 0.6}
                for k, v in sigs2.items()
            ]
        if not isinstance(sigs2, list):
            sigs2 = []
        raw2["signals"] = (raw.get("signals", []) or []) + sigs2
        raw2["signals"].append(
            {
                "key": "sanity_rerun",
                "value": "Re-asked GPT due to uniform macros across foods.",
                "unit": "",
                "source": "code",
                "confidence": 0.9,
            }
        )

        foods = normalize_foods(raw2.get("foods"))
        foods = apply_known_overrides(foods, raw2)
        raw = raw2
        raw["foods"] = foods

    # Compute totals from foods
    if raw.get("foods"):
        total_cal, total_pro, counted = sum_food_macros(raw["foods"])
        raw["calories_est"] = int(total_cal)
        raw["protein_est"] = int(total_pro)
        raw.setdefault("signals", [])
        raw["signals"].append(
            {
                "key": "macro_totals_computed",
                "value": f"Summed {counted} food items to compute totals.",
                "unit": "",
                "source": "code",
                "confidence": 0.95,
            }
        )

    # Normalize signals
    cleaned = []
    for s in raw.get("signals", []):
        if not isinstance(s, dict):
            continue
        s.setdefault("key", "unknown")
        s.setdefault("unit", "")
        s.setdefault("source", "journal")
        s.setdefault("confidence", 0.7)
        v = s.get("value", "")
        if v is None:
            continue
        s["value"] = str(v)
        cleaned.append(s)
    raw["signals"] = cleaned

    # Force user-prompt for screen_time_hours unless explicitly mentioned
    if not mentioned_screen_time(entry):
        raw["screen_time_hours"] = None

    return Extraction.model_validate(raw)
