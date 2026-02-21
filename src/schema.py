# schema.py
from __future__ import annotations

import re
from typing import Optional, List

from pydantic import BaseModel, Field, field_validator


class Signal(BaseModel):
    key: str
    value: str
    unit: Optional[str] = ""
    source: str = "journal"
    confidence: float = 0.7


class FoodItem(BaseModel):
    name: str
    quantity_text: str = ""
    calories: Optional[int] = None
    protein_g: Optional[int] = None
    confidence: float = 0.7
    source: str = "model_estimate"


class Extraction(BaseModel):
    # Core
    date: Optional[str] = None
    summary: Optional[str] = None

    # Sleep
    wake_time: Optional[str] = None          # HH:MM
    sleep_hours: Optional[float] = None
    sleep_quality_1_10: Optional[int] = None

    # Training
    gym: Optional[bool] = None
    workout_type: Optional[str] = None
    workout_minutes: Optional[int] = None
    cardio_minutes: Optional[int] = None

    # Health / habits
    water_bottles: Optional[int] = None
    creatine: Optional[bool] = None
    screen_time_hours: Optional[float] = None
    study_hours: Optional[float] = None
    weight: Optional[float] = None

    # Nutrition
    foods: List[FoodItem] = Field(default_factory=list)
    calories_est: Optional[int] = None
    protein_est: Optional[int] = None

    # Mood
    mood_1_10: Optional[float] = None

    # Meta
    signals: List[Signal] = Field(default_factory=list)

    # ---------- Validators ----------

    @field_validator("wake_time")
    @classmethod
    def validate_wake_time(cls, v):
        if v is None or v == "":
            return None
        if not isinstance(v, str):
            raise ValueError("wake_time must be a string HH:MM")
        if not re.match(r"^\d{1,2}:\d{2}$", v):
            raise ValueError("wake_time must be HH:MM")
        hh, mm = v.split(":")
        hh = int(hh)
        mm = int(mm)
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            raise ValueError("wake_time must be a valid 24h time")
        return f"{hh:02d}:{mm:02d}"

    @field_validator(
        "sleep_quality_1_10",
        "mood_1_10",
    )
    @classmethod
    def validate_1_10(cls, v):
        if v is None:
            return None
        v = float(v)
        if not (1 <= v <= 10):
            raise ValueError("Must be between 1 and 10")
        return v

    @field_validator(
        "workout_minutes",
        "cardio_minutes",
        "water_bottles",
        "calories_est",
        "protein_est",
    )
    @classmethod
    def validate_nonneg_int(cls, v):
        if v is None:
            return None
        v = int(v)
        if v < 0:
            raise ValueError("Must be >= 0")
        return v

    @field_validator(
        "sleep_hours",
        "screen_time_hours",
        "study_hours",
        "weight",
    )
    @classmethod
    def validate_nonneg_float(cls, v):
        if v is None:
            return None
        v = float(v)
        if v < 0:
            raise ValueError("Must be >= 0")
        return v
