import argparse
import os
import re

import pandas as pd

from categorizer.archive.google_auth import build_sheets_service
from categorizer.paths import OUTPUT_DIR


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

# First data row of a month tab; rows 1-32 hold the per-month summary block.
ARCHIVE_START_ROW = 33
# Columns written to a new month tab. A:E are the archive columns the reader
# consumes; F (Reasoning) is written for review only and is ignored by
# load_archive_from_sheets, which reads A33:E.
WRITE_COLUMNS = ARCHIVE_COLUMNS + ("Reasoning",)
WRITE_LAST_COLUMN = "F"

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


def _tab_month(title: str) -> int | None:
    """Return the 1-12 month number a tab name starts with, or None."""
    match = _MONTH_TAB_RE.match(title.strip())
    if match is None:
        return None
    return MONTH_NAMES.index(match.group(1).lower()) + 1


def _month_tab_properties(service, spreadsheet_id):
    """Return [(month, sheet_id, index, title)] for every month-named tab."""
    meta = service.spreadsheets().get(
        spreadsheetId=spreadsheet_id,
        fields="sheets(properties(title,sheetId,index))",
    ).execute()

    tabs = []
    for sheet in meta.get("sheets", []):
        props = sheet.get("properties", {})
        title = props.get("title")
        if title is None:
            continue
        month = _tab_month(title)
        if month is None:
            continue
        tabs.append((month, props["sheetId"], props["index"], title))
    return tabs


def _pick_template_tab(tabs, target_month: int):
    """Pick the month tab to duplicate as the template for a new month.

    Prefers the latest tab for a month *before* the target month (so August
    copies July's layout). If the target is the earliest month in the sheet
    (e.g. January in a fresh year sheet), falls back to the latest tab present
    — its summary block is still the right shape, just from a later month.
    """
    earlier = [t for t in tabs if t[0] < target_month]
    pool = earlier or tabs
    return max(pool, key=lambda t: (t[0], t[2]))


def _tabs_are_ascending(tabs) -> bool:
    """True if tab position increases with month (Jan leftmost).

    ASSUMPTION: a sheet whose month tabs are neither ascending nor descending by
    position is treated as ascending, so the new tab lands to the right of its
    template rather than erroring. Tab position has no effect on correctness —
    load_archive_from_sheets() reads tabs by name, not order.
    """
    months = [month for month, _, _, _ in sorted(tabs, key=lambda t: t[2])]
    if months == sorted(months, reverse=True) and months != sorted(months):
        return False
    return True


def _new_tab_title(template_title: str, target_month: int) -> str:
    """Derive the new tab's name by swapping the month word in the template.

    "July 2026" -> "August 2026", "July" -> "August", preserving whatever
    suffix convention the existing tabs use.
    """
    month_name = MONTH_NAMES[target_month - 1].capitalize()
    return _MONTH_TAB_RE.sub(
        lambda m: month_name + m.group(2), template_title.strip(), count=1
    )


def _cell_value(value):
    """Coerce a pandas cell into something the Sheets JSON API accepts.

    Missing values (NaT/NaN/None) become "" rather than being dropped, so a row
    with an unparseable date still lands in the tab with its other fields intact
    and is visibly blank instead of silently absent.
    """
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (int, float)):
        # numpy scalars are not JSON-serializable; float() normalizes them.
        return float(value)
    return str(value)


def write_month_tab(
    frame: pd.DataFrame,
    month: int,
    spreadsheet_id: str = None,
) -> str:
    """Create a new month tab from the latest existing one and fill it in.

    Duplicates the most recent prior month tab (preserving the rows 1-32 summary
    block, its formulas and formatting), renames it for `month`, clears the old
    data below row 32, and writes `frame`'s Date/Name/Amount/Category/Account/
    Reasoning columns starting at A33.

    Returns the new tab's title. Raises if a tab for that month already exists —
    existing archive data is never overwritten.
    """
    if spreadsheet_id is None:
        spreadsheet_id = os.environ.get("CURRENT_YEAR_ARCHIVE_SHEET_ID", "")
    if not spreadsheet_id:
        raise RuntimeError(
            "CURRENT_YEAR_ARCHIVE_SHEET_ID env var is not set. Add the Google "
            "Sheets spreadsheet ID for your transaction archive to .env."
        )

    service = build_sheets_service()
    tabs = _month_tab_properties(service, spreadsheet_id)
    if not tabs:
        raise RuntimeError(
            f"No month-named tabs (e.g. 'May', 'June') found in spreadsheet "
            f"{spreadsheet_id}; there is no template tab to duplicate."
        )

    existing = [t for t in tabs if t[0] == month]
    if existing:
        raise RuntimeError(
            f"Spreadsheet {spreadsheet_id} already has a tab for month {month} "
            f"({', '.join(t[3] for t in existing)}). Rename or delete it before "
            "re-running, so existing archive data isn't overwritten."
        )

    template = _pick_template_tab(tabs, month)
    _, template_sheet_id, template_index, template_title = template
    new_title = _new_tab_title(template_title, month)
    insert_index = (
        template_index + 1 if _tabs_are_ascending(tabs) else template_index
    )

    service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={
            "requests": [
                {
                    "duplicateSheet": {
                        "sourceSheetId": template_sheet_id,
                        "insertSheetIndex": insert_index,
                        "newSheetName": new_title,
                    }
                }
            ]
        },
    ).execute()

    # Wipe the template's transactions, keeping rows 1-32 and all formatting.
    service.spreadsheets().values().clear(
        spreadsheetId=spreadsheet_id,
        range=f"'{new_title}'!A{ARCHIVE_START_ROW}:{WRITE_LAST_COLUMN}",
        body={},
    ).execute()

    values = [
        [_cell_value(row[col]) for col in WRITE_COLUMNS]
        for _, row in frame.iterrows()
    ]
    if values:
        service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=f"'{new_title}'!A{ARCHIVE_START_ROW}",
            # USER_ENTERED (not RAW) so the YYYY-MM-DD strings are parsed into
            # real date cells; load_archive_from_sheets reads dates back as
            # serial numbers and would produce NaT for text dates.
            valueInputOption="USER_ENTERED",
            body={"values": values},
        ).execute()

    return new_title


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


def _push_csv(csv_path: str, month: int = None) -> None:
    """Write an already-categorized CSV to a new month tab.

    Lets a run whose Sheets write failed be retried without re-paying for the
    LLM calls. The month is inferred from the CSV's dates unless given; a CSV
    spanning more than one month is an error rather than a guess.
    """
    frame = pd.read_csv(csv_path, parse_dates=["Date"])
    if month is None:
        months = sorted(frame["Date"].dt.month.dropna().unique().tolist())
        if len(months) != 1:
            raise RuntimeError(
                f"{csv_path} spans months {months}; pass --month explicitly."
            )
        month = int(months[0])
    new_tab = write_month_tab(frame, month)
    print(f"Wrote {len(frame)} rows to tab '{new_tab}'.")


if __name__ == "__main__":
    # .env is loaded by categorizer/__init__ on package import.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--push",
        nargs="?",
        const=os.path.join(OUTPUT_DIR, "categorized_transactions.csv"),
        metavar="CSV",
        help="Write an existing categorized CSV to a new month tab instead of "
             "running the archive-fetch smoke test.",
    )
    parser.add_argument(
        "--month",
        type=int,
        help="Target month 1-12 (inferred from the CSV's dates if omitted).",
    )
    args = parser.parse_args()

    if args.push:
        _push_csv(args.push, args.month)
    else:
        df = load_archive_from_sheets()
        print(df.head(10))
        print(f"\nTotal rows: {len(df)}")
        print(f"Categories: {sorted(df['Category'].unique().tolist())}")
