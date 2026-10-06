# JournalToData

Write a normal journal entry at night; get a row of clean, validated daily metrics in Google Sheets.

An LLM reads each entry and extracts 15+ structured fields (sleep, training, food, screen time, mood, and
more). Pydantic validates them, the app asks follow-up questions for anything missing, and a Streamlit
dashboard sits on top. I used it for 131 of 132 nights from January to June 2026.

**Write-up with charts:** https://eshaandhavala.github.io/entries/journal-to-data/

## What it does

| Tab | |
|---|---|
| **Log Entry** | Paste or type the day's entry. The model extracts metrics, validation flags gaps, and you answer follow-ups before it saves. Optional daily photo. |
| **Dashboard** | Trends, streaks (gym, sleep, and so on), and day-over-day deltas. |
| **Weekly Review** | Averages and highlights for the week. |
| **Ask Your Data** | Chat with your own metrics and journal history. |

## How it works

```
journal text ──▶ extract.py (OpenAI, JSON mode) ──▶ schema.py (Pydantic) ──▶ validate.py (follow-up questions)
                                                                                     │
                       Streamlit dashboard ◀── sheets.py (Google Sheets, upsert by date) ◀─┘
```

- `src/extract.py`: prompt and JSON extraction, food normalization, known-food macro overrides, sanity checks.
- `src/schema.py`: the `Extraction` model with typed, range-checked fields.
- `src/validate.py`: decides which missing fields to ask about.
- `src/sheets.py`: Google Sheets client; one row per date in the `Daily` worksheet, signals in `Signals`.
- `src/app.py`: Streamlit UI (four tabs above).
- `src/main.py`: the same pipeline as a CLI.
- `scripts/export_snapshot.py`: exports weekly aggregates only (no raw entries) to JSON for the portfolio charts.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/app.py
```

Secrets go in `.streamlit/secrets.toml` (never commit it):

| Key | |
|---|---|
| `OPENAI_API_KEY` | OpenAI key |
| `OPENAI_MODEL` | optional; defaults to `gpt-4.1-mini` |
| `GOOGLE_SHEET_NAME` | spreadsheet with `Daily` and `Signals` worksheets |
| `[gcp_service_account]` | service-account JSON as a TOML table; share the sheet with its email |
| `CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY`, `CLOUDINARY_API_SECRET` | optional, for daily photos |

The CLI (`python src/main.py`) and `scripts/export_snapshot.py` read `OPENAI_API_KEY`,
`GOOGLE_SERVICE_ACCOUNT_JSON` (inline JSON or a path), and `GOOGLE_SHEET_NAME` from `.env`.

## Status

Finished. I stopped logging in June 2026. The live Streamlit app is retired; the portfolio page renders a
static snapshot so it never goes to sleep.
