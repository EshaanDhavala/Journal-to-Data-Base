# validate.py
from __future__ import annotations

from typing import Dict, List, Tuple, Any


# All fields (besides date and signals) are required.
# If any of these are missing after extraction, we will prompt the user.
REQUIRED_FIELDS: List[str] = [
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
    "calories_est",
    "protein_est",
    "mood_1_10",
    "weight",
    "summary",
]

# Human prompts for each required field
QUESTIONS: Dict[str, str] = {
    "wake_time": "Wake-up time? (HH:MM, e.g., 12:30)",
    "sleep_hours": "Sleep hours? (e.g., 7.5)",
    "sleep_quality_1_10": "Sleep quality (1-10)?",
    "gym": "Gym today? (yes/no)",
    "workout_type": "Workout type? (push/pull/legs/run/tennis/rest/other)",
    "workout_minutes": "Workout minutes? (0+)",
    "cardio_minutes": "Cardio minutes? (0+)",
    "water_bottles": "Total bottles of water? (0+)",
    "creatine": "Creatine today? (yes/no)",
    "screen_time_hours": "Screen time hours? (e.g., 6.0)",
    "study_hours": "Study hours? (e.g., 3.5)",
    "calories_est": "Total calories estimate? (integer, e.g., 2400)",
    "protein_est": "Total protein estimate (grams)? (integer, e.g., 130)",
    "mood_1_10": "Mood (1-10)?",
    "weight": "Body weight? (lbs, e.g., 165.4)",
    "summary": "One-line summary of the day?",
}

# Simple type declarations for parsing (used by main.py/app.py parse_value)
FIELD_TYPES: Dict[str, str] = {
    "wake_time": "time_hhmm",
    "sleep_hours": "float_nonneg",
    "sleep_quality_1_10": "int_1_10",
    "gym": "bool",
    "workout_type": "str",
    "workout_minutes": "int_nonneg",
    "cardio_minutes": "int_nonneg",
    "water_bottles": "int_nonneg",
    "creatine": "bool",
    "screen_time_hours": "float_nonneg",
    "study_hours": "float_nonneg",
    "calories_est": "int_nonneg",
    "protein_est": "int_nonneg",
    "mood_1_10": "float_1_10",
    "weight": "float_nonneg",
    "summary": "str",
}


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def get_missing_fields(extraction_dict: dict) -> List[str]:
    missing: List[str] = []
    for f in REQUIRED_FIELDS:
        if is_missing(extraction_dict.get(f)):
            missing.append(f)
    return missing


def get_missing_questions(extraction_dict: dict) -> List[Tuple[str, str]]:
    return [(f, QUESTIONS.get(f, f"What is {f}?")) for f in get_missing_fields(extraction_dict)]
