import gspread
from google.oauth2.service_account import Credentials  # noqa: F401

def get_client(service_account_json_str: str):
    import json
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    info = json.loads(service_account_json_str)
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(creds)

# sheets.py
def upsert_daily_row(sheet, row_dict: dict):
    date = row_dict["date"]

    # Read headers from row 1
    headers = sheet.row_values(1)
    if not headers:
        raise ValueError("Daily sheet has no header row (row 1).")

    # Build values aligned to headers
    values = [row_dict.get(h, "") for h in headers]

    # Find date row by matching the "date" column
    # Assumes one of the headers is literally "date"
    if "date" not in headers:
        raise ValueError('Daily sheet headers must include a "date" column.')

    date_col_idx = headers.index("date") + 1
    col_vals = sheet.col_values(date_col_idx)

    if date in col_vals:
        idx = col_vals.index(date) + 1
        sheet.update(f"A{idx}:{chr(64+len(headers))}{idx}", [values])
    else:
        sheet.append_row(values, value_input_option="USER_ENTERED")


def ensure_column(sheet, col_name: str):
    """Append col_name as a new header column if it doesn't already exist."""
    headers = sheet.row_values(1)
    if col_name not in headers:
        sheet.update_cell(1, len(headers) + 1, col_name)


def append_signals(sheet, date: str, signals: list):
    rows = []
    for s in signals:
        rows.append([
            date,
            s.get("key", ""),
            s.get("value", ""),
            s.get("unit", ""),
            s.get("source", "journal"),
            s.get("confidence", 0.7),
        ])
    if rows:
        sheet.append_rows(rows, value_input_option="USER_ENTERED")
