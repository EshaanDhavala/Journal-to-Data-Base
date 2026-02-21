# src/extract.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from openai import APIStatusError, APIConnectionError, RateLimitError, AuthenticationError

import streamlit as st

from schema import Extraction

load_dotenv()
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

SYSTEM = """You are a precise nutritionist and health data extractor.
Extract structured health/life tracking data from journal entries.
Return ONLY valid JSON (no markdown, no extra text).

NUTRITION ACCURACY: Use USDA/standard nutritional data as your reference.
- Never assign identical calories AND protein to different, distinct food items.
- Each food item's macros must reflect its specific type and portion size.
- When a brand is mentioned, use that brand's actual nutrition label values.
- Be conservative but realistic — do not undercount by more than 10%.

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
  - quantity_text (string, e.g., "2 slices", "6 oz", "1 cup")
  - calories (int) for THIS specific portion
  - protein_g (int) for THIS specific portion
  - confidence (0..1)

- calories_est (int or null)  (optional; will be recomputed in code)
- protein_est (int or null)   (optional; will be recomputed in code)

- signals: list of objects like {"key":"...", "value":"...", "unit":"", "source":"journal", "confidence":0.0-1.0}

NUTRITION REFERENCE VALUES — use these as anchors for estimation:

Proteins:
  chicken breast 6oz ≈ 280cal/52g | chicken breast 4oz ≈ 190cal/35g
  egg large ≈ 70cal/6g | egg whites 3 ≈ 50cal/11g
  Greek yogurt 6oz ≈ 100cal/17g | cottage cheese 1/2cup ≈ 110cal/13g
  canned tuna 3oz ≈ 100cal/22g | salmon 6oz ≈ 350cal/34g
  ground beef 4oz 80/20 ≈ 290cal/20g | ground beef 4oz 93/7 ≈ 190cal/23g
  steak 6oz ≈ 350cal/44g | turkey breast 4oz ≈ 150cal/28g
  shrimp 4oz ≈ 120cal/23g | tofu firm 4oz ≈ 90cal/10g

Grains/Starches:
  cooked white rice 1cup ≈ 200cal/4g | cooked brown rice 1cup ≈ 215cal/5g
  cooked pasta 1cup ≈ 220cal/8g | bread slice ≈ 80cal/3g
  oats 1/2cup dry ≈ 150cal/5g | bagel plain ≈ 270cal/10g
  tortilla 10" flour ≈ 200cal/5g | sweet potato medium ≈ 115cal/2g
  white potato medium ≈ 160cal/4g

Dairy:
  whole milk 1cup ≈ 150cal/8g | 2% milk 1cup ≈ 125cal/8g
  cheddar cheese 1oz ≈ 110cal/7g | mozzarella 1oz ≈ 85cal/6g

Fats/Nuts:
  avocado 1/2 ≈ 120cal/2g | almonds 1oz ≈ 165cal/6g
  peanut butter 2tbsp ≈ 190cal/8g | olive oil 1tbsp ≈ 120cal/0g
  butter 1tbsp ≈ 100cal/0g

Vegetables:
  leafy greens 2cups ≈ 20cal/2g | broccoli 1cup ≈ 55cal/4g
  mixed veggies 1cup ≈ 50cal/3g

Fruits:
  banana medium ≈ 105cal/1g | apple medium ≈ 95cal/0g
  berries 1cup ≈ 65cal/1g | orange medium ≈ 65cal/1g

Beverages:
  whole milk 1cup ≈ 150cal/8g | orange juice 1cup ≈ 110cal/2g
  sports drink 20oz ≈ 130cal/0g | coffee black ≈ 5cal/0g

Restaurants (typical portions):
  burger + fries ≈ 950-1100cal/40-50g | pizza slice cheese ≈ 280cal/12g
  pizza slice with toppings ≈ 320-380cal/16-20g
  burrito (Chipotle-style) ≈ 750-900cal/40-55g
  pasta entrée ≈ 650-950cal/25-35g | sushi roll ≈ 300-380cal/12-18g
  sandwich/sub 6" ≈ 400-550cal/25-35g | salad with protein ≈ 400-600cal/30-45g
  stir fry with rice ≈ 500-700cal/25-35g

PORTION ESTIMATION RULES:
- "a serving" = standard serving size per package or label
- "a handful" = ~1oz nuts / ~1cup leafy greens / ~0.5cup grains
- "some" or "a bit" = 1/4 to 1/3 of a typical serving
- "a lot" or "a ton" = 1.5–2x typical serving
- "half" = exactly 0.5x the reference value
- Restaurant portions ≈ 1.5x home cooking portions
- "X% of a bag/box" = multiply total package nutrition by X%

Rules:
- If food is mentioned, foods must include it with integer calories and protein (not null).
- Do NOT reuse the same (calories, protein) pair across multiple different food items.
- If unsure of exact portion, state it in quantity_text and estimate conservatively.
"""

# Known foods with verified nutrition data
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
    {
        "known_name": "quest_protein_bar",
        "patterns": [r"\bquest\s+bar\b", r"\bquest\s+protein\s+bar\b"],
        "cal": 190,
        "protein": 21,
    },
    {
        "known_name": "premier_protein_shake",
        "patterns": [r"\bpremier\s+protein\b", r"\bpremier\s+shake\b"],
        "cal": 160,
        "protein": 30,
    },
    {
        "known_name": "rxbar",
        "patterns": [r"\brx\s*bar\b", r"\brxbar\b"],
        "cal": 210,
        "protein": 12,
    },
    {
        "known_name": "chobani_plain_greek_yogurt_nonfat",
        "patterns": [r"\bchobani\b.*\bgreek\b", r"\bchobani\b.*\byogurt\b"],
        "cal": 90,
        "protein": 17,
    },
    {
        "known_name": "oikos_pro_greek_yogurt",
        "patterns": [r"\boikos\s+pro\b", r"\boikos\b.*\byogurt\b"],
        "cal": 130,
        "protein": 20,
    },
    {
        "known_name": "isopure_zero_carb_protein_shake",
        "patterns": [r"\bisopure\b"],
        "cal": 160,
        "protein": 40,
    },
    {
        "known_name": "muscle_milk_pro_series",
        "patterns": [r"\bmuscle\s+milk\b"],
        "cal": 280,
        "protein": 40,
    },
    {
        "known_name": "kodiak_cakes_protein_waffle",
        "patterns": [r"\bkodiak\b.*\bwaffle\b", r"\bkodiak\b.*\bpancake\b"],
        "cal": 250,
        "protein": 14,
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
    Match ONLY on the food item's name (NOT the full entry text),
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

            # Override if missing or significantly off
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
                f2["confidence"] = max(float(f.get("confidence", 0.7)), 0.95)
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
                "confidence": 0.95,
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


def _parse_json_object(text: str) -> Dict[str, Any]:
    payload = (text or "").strip()
    if not payload:
        raise ValueError("Model returned empty content.")

    # Strict path
    try:
        obj = json.loads(payload)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback for markdown-wrapped or extra-text responses
    start = payload.find("{")
    end = payload.rfind("}")
    if start != -1 and end != -1 and end > start:
        obj = json.loads(payload[start : end + 1])
        if isinstance(obj, dict):
            return obj

    raise ValueError("Model did not return a valid JSON object.")


def call_gpt(entry: str, date: str, extra_rules: str = "") -> Dict[str, Any]:
    msg = f"""<RAW_JOURNAL>
{entry}
</RAW_JOURNAL>

Journal entry date (local): {date}

{INSTRUCTIONS}

{extra_rules}
"""
    models = ["gpt-4.1-mini", "gpt-4o-mini"]
    last_err: Optional[Exception] = None

    for model in models:
        req = dict(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": msg},
            ],
            temperature=0.1,
        )

        # Try strict JSON mode first, then plain completion mode.
        for use_json_mode in (True, False):
            try:
                if use_json_mode:
                    resp = client.chat.completions.create(
                        **req,
                        response_format={"type": "json_object"},
                    )
                else:
                    resp = client.chat.completions.create(**req)
                return _parse_json_object(resp.choices[0].message.content or "")
            except Exception as e:
                last_err = e
                continue

    raise RuntimeError(f"OpenAI extraction request failed after retries. Last error: {last_err}")


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
            extra_rules="CRITICAL: Every food item must have UNIQUE (calories, protein) values. "
            "Different foods cannot share identical macros. "
            "Re-read each food item and assign accurate, distinct nutrition values based on type and portion.",
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
                "value": f"Summed {counted} food items: {total_cal} cal, {total_pro}g protein.",
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
