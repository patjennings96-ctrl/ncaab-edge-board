"""
sheets.py
=========
Push model outputs to Google Sheets so the daily card lives in a spreadsheet you
can open on any device.

Three ways to get data into Sheets — pick what fits your setup:

A) gspread + service account  (recommended, fully automated)
   - Create a Google Cloud service account, enable the Sheets + Drive APIs,
     download credentials.json, and SHARE the target spreadsheet with the
     service-account email (it acts like another user).
   - `write_dataframe()` below clears a tab and writes a DataFrame.

B) Publish a CSV + IMPORTDATA  (no API, near-zero setup)
   - Have `run_daily.py` write `daily_card.csv` somewhere with a public URL
     (e.g. a GitHub repo, GCS bucket). In a cell:  =IMPORTDATA("<csv-url>")
   - Sheet refreshes on its own roughly hourly.

C) Google Apps Script  (if you can't run Python on a schedule)
   - A time-driven Apps Script trigger fetches your published CSV/JSON and
     writes it. Good when the only always-on environment you have is Sheets.

For a hands-off "updates daily" loop, run `run_daily.py` from GitHub Actions
(cron schedule) or Google Cloud Scheduler -> Cloud Run, using option A.
"""

from __future__ import annotations

import pandas as pd


def write_dataframe(df: pd.DataFrame, spreadsheet_id: str, worksheet: str,
                    credentials_path: str = "credentials.json") -> None:
    """
    Clear `worksheet` in the given spreadsheet and write `df` (header + rows).
    Requires: pip install gspread google-auth, and the sheet shared with the
    service-account email.
    """
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file(credentials_path, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(spreadsheet_id)

    try:
        ws = sh.worksheet(worksheet)
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet, rows=max(100, len(df) + 10),
                              cols=max(26, len(df.columns) + 2))

    # gspread wants plain Python types; cast everything to str-safe values.
    safe = df.where(pd.notna(df), "")
    values = [list(map(str, safe.columns))] + safe.astype(object).values.tolist()
    ws.update(values, value_input_option="USER_ENTERED")


def write_csv(df: pd.DataFrame, path: str) -> None:
    """Option B helper: dump a CSV you can serve and pull with IMPORTDATA."""
    df.to_csv(path, index=False)
