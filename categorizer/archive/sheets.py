import os
import re

import pandas as pd

from categorizer.archive.google_auth import build_sheets_service


MONTH_NAMES = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)
# Match tab names like "May", "May 2024", "May-24", "May_2024" — month name
# followed by end-of-string or a non-letter separator. Case-insensitive.
_MONTH_TAB_RE = re.compile(
    r"^(" + "|".join(MONTH_NAMES) + r")($|[\s\-_])",
    re.IGNORECASE,
)

ARCHIVE_RANGE = "A33:E"
ARCHIVE_COLUMNS = ("Date", "Name", "Amount", "Category", "Account")

# Google Sheets serial-date epoch: serial 0 == 1899-12-30. Date cells fetched
# with valueRenderOption="UNFORMATTED_VALUE" come back as days-since-epoch
# numbers rather than strings, so we convert against this origin below.
_SHEETS_DATE_ORIGIN = "1899-12-30"


def _list_month_tabs(service, spreadsheet_id):
    meta = service.spreadsheets().get(
        spreadsheetId=spreadsheet_id,
        fields="sheets(properties(title))",
    ).execute()
    titles = [
        s["properties"]["title"]
        for s in meta.get("sheets", [])
        if "properties" in s and "title" in s["properties"]
    ]
    return [t for t in titles if _MONTH_TAB_RE.match(t.strip())]


def load_archive_from_sheets(spreadsheet_id: str = None) -> pd.DataFrame:
    """Fetch the archive by concatenating month-named tabs in the spreadsheet.

    Reads A33:E (Date, Name, Amount, Category, Account) from every tab whose
    name starts with a month name. Returns a DataFrame with columns
    Date, Name, Amount, Category, Account.
    """
    if spreadsheet_id is None:
        spreadsheet_id = os.environ.get("CURRENT_YEAR_ARCHIVE_SHEET_ID", "")
    if not spreadsheet_id:
        raise RuntimeError(
            "CURRENT_YEAR_ARCHIVE_SHEET_ID env var is not set. Add the Google "
            "Sheets spreadsheet ID for your transaction archive to .env."
        )

    service = build_sheets_service()
    month_tabs = _list_month_tabs(service, spreadsheet_id)
    if not month_tabs:
        raise RuntimeError(
            f"No month-named tabs (e.g. 'May', 'June') found in spreadsheet "
            f"{spreadsheet_id}."
        )

    ranges = [f"'{tab}'!{ARCHIVE_RANGE}" for tab in month_tabs]
    resp = service.spreadsheets().values().batchGet(
        spreadsheetId=spreadsheet_id,
        ranges=ranges,
        valueRenderOption="UNFORMATTED_VALUE",
    ).execute()

    frames = []
    for value_range in resp.get("valueRanges", []):
        values = value_range.get("values", [])
        # The Sheets API truncates trailing empty cells, so a 5-col row may
        # come back with <5 entries. Pad with "" so DataFrame construction
        # doesn't error; empty Name/Category rows are filtered out below.
        ncols = len(ARCHIVE_COLUMNS)
        padded = [(row + [""] * (ncols - len(row)))[:ncols] for row in values]
        df = pd.DataFrame(padded, columns=list(ARCHIVE_COLUMNS))
        # Drop rows with no Name or no Category — these are incomplete
        # archive entries (blank separator rows, partial drafts) and are
        # silently skipped rather than errored on.
        df = df[df["Name"].astype(str).str.strip() != ""]
        df = df[df["Category"].astype(str).str.strip() != ""]
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=list(ARCHIVE_COLUMNS))

    combined = pd.concat(frames, ignore_index=True)
    # Amount may come back as float, int, or str depending on cell format.
    # Non-numeric entries become NaN — rows are kept so the caller can decide
    # whether to surface them; they are not silently dropped here.
    combined["Amount"] = pd.to_numeric(combined["Amount"], errors="coerce")
    # Date arrives as a Google Sheets serial number (days since 1899-12-30)
    # because of UNFORMATTED_VALUE. ASSUMPTION: every Date cell uses that serial
    # epoch; a cell that is blank or already a string becomes NaT rather than
    # erroring, and the row is kept so the caller can surface/drop it (the eval
    # drops NaT-dated rows since it can't assign them to a month).
    serials = pd.to_numeric(combined["Date"], errors="coerce")
    combined["Date"] = pd.to_datetime(
        serials, unit="D", origin=_SHEETS_DATE_ORIGIN, errors="coerce"
    )
    return combined


if __name__ == "__main__":
    # .env is loaded by categorizer/__init__ on package import.
    df = load_archive_from_sheets()
    print(df.head(10))
    print(f"\nTotal rows: {len(df)}")
    print(f"Categories: {sorted(df['Category'].unique().tolist())}")
