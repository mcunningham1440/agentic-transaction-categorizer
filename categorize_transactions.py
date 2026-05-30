import asyncio
import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from Levenshtein import distance as levenshtein_distance
from openai import AsyncOpenAI

load_dotenv()

from agent import categorize_dataframe
from sheets_archive import load_archive_from_sheets
from tools import CATEGORIES


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PERSONAL_PROFILE_PATH = os.path.join(SCRIPT_DIR, "personal_profile.txt")


def load_personal_profile() -> str:
    if not os.path.exists(PERSONAL_PROFILE_PATH):
        raise FileNotFoundError(
            f"personal_profile.txt not found at {PERSONAL_PROFILE_PATH}. "
            "Copy personal_profile.sample.txt to personal_profile.txt and fill "
            "it in with details that help the agent disambiguate categories."
        )
    with open(PERSONAL_PROFILE_PATH) as f:
        return f.read().strip()


def find_close_strings(query_string, df, cutoff=0.8):
    result_df = df.copy()

    result_df['Distance'] = result_df['Name'].apply(
        lambda x: levenshtein_distance(x.lower(), query_string.lower())
        )

    result_df['MaxLength'] = result_df['Name'].apply(
        lambda x: max(len(str(x)), len(query_string)))

    result_df['Similarity'] = 1 - (result_df['Distance'] / result_df['MaxLength'])

    close_matches = result_df.sort_values('Similarity', ascending=False)
    close_matches = close_matches[close_matches['Similarity'] >= cutoff]

    close_matches = close_matches.drop(['Distance', 'MaxLength'], axis=1)

    return close_matches


# Rows in any category not present in CATEGORIES are silently dropped from the
# similarity pool — this is intentional, since "Income" / "Internal transfer"
# etc. are not spending categories the agent should learn from. The same filter
# also drops rows from prior years whose category has since been retired: the
# category list evolves year to year, so the past-year sheet may contain labels
# that no longer exist in CATEGORIES, and those rows are excluded here.
def load_archive() -> pd.DataFrame:
    # Combine the current- and prior-year archive sheets so the agent always has
    # a full trailing year of history available (the per-transaction 12-month
    # window is applied later, inside agent.categorize_one). PAST_YEAR is
    # required, not optional: without it, January/early-year transactions would
    # have an empty example pool. A missing ID raises rather than degrading.
    past_year_id = os.environ.get("PAST_YEAR_ARCHIVE_SHEET_ID", "")
    if not past_year_id:
        raise RuntimeError(
            "PAST_YEAR_ARCHIVE_SHEET_ID env var is not set. It supplies the "
            "trailing-year history the agent needs for early-year transactions. "
            "Add the prior year's archive spreadsheet ID to .env."
        )

    current = load_archive_from_sheets()  # reads CURRENT_YEAR_ARCHIVE_SHEET_ID
    past = load_archive_from_sheets(past_year_id)
    archive = pd.concat([current, past], ignore_index=True)
    archive = archive[archive["Category"].isin(CATEGORIES)].reset_index(drop=True)
    return archive


async def main():
    # Calculate previous month as default
    today = datetime.now()
    if today.month == 1:
        default_month = 12
        default_year = (today.year - 1) % 100
    else:
        default_month = today.month - 1
        default_year = today.year % 100

    use_default = input(f"Use {default_month}/{default_year}? (Y/n): ").strip().lower()
    if use_default == "" or use_default == "y":
        month = default_month
        year = default_year
    else:
        month = int(input("Enter the month number: "))
        year = int(input("Enter the last two digits of the year: "))

    month_folder = os.path.join(SCRIPT_DIR, "data", f"{month}-{year}")

    sams_frames = []
    chase_frame = apple_frame = venmo_frame = ally_frame = None

    missing_accounts = {"Chase", "Apple", "Venmo", "Sam's Club", "Ally"}

    # See transaction_csv_sources.md (gitignored) for where to download each
    # account's CSV and the filename-prefix convention used below.
    for item in os.listdir(month_folder):
        item_path = os.path.join(month_folder, item)

        if item[:5] == "Chase":
            chase_frame = pd.read_csv(item_path)
            missing_accounts.remove("Chase")

        elif item[:5] == "Apple":
            apple_frame = pd.read_csv(item_path)
            missing_accounts.remove("Apple")

        elif item[:5] == "Venmo":
            venmo_frame = pd.read_csv(item_path, usecols=[2,5,6,7,8], skiprows=[0,1])
            missing_accounts.remove("Venmo")

        elif item[:11] == "Transaction":
            sams_frames.append(pd.read_csv(item_path))

            if "Sam's Club" in missing_accounts:
                missing_accounts.add("Sam's Club (partial)")
                missing_accounts.remove("Sam's Club")
            elif "Sam's Club (partial)" in missing_accounts:
                missing_accounts.remove("Sam's Club (partial)")

        elif item[:12] == "transactions":
            ally_frame = pd.read_csv(item_path)
            missing_accounts.remove("Ally")

    if missing_accounts:
        print(f"Not all account files found. Missing: {missing_accounts}")
    else:
        print("All account files found.")

    print("Capital One frame not implemented yet")

    frames_to_concat = []

    if chase_frame is not None:
        chase_frame = chase_frame[["Transaction Date", "Description", "Amount"]]
        chase_frame.columns = ["Date", "Name", "Amount"]
        chase_frame["Date"] = pd.to_datetime(chase_frame["Date"])
        chase_frame = chase_frame[chase_frame["Date"].dt.month == month]
        chase_frame["Account"] = "Chase"
        frames_to_concat.append(chase_frame)

    if apple_frame is not None:
        apple_frame = apple_frame[["Transaction Date", "Description", "Amount (USD)"]]
        apple_frame["Amount (USD)"] = -apple_frame["Amount (USD)"]
        apple_frame.columns = ["Date", "Name", "Amount"]
        apple_frame["Date"] = pd.to_datetime(apple_frame["Date"])
        apple_frame = apple_frame[apple_frame["Date"].dt.month == month]
        apple_frame["Account"] = "Apple"
        frames_to_concat.append(apple_frame)

    if venmo_frame is not None:
        venmo_holder = os.environ.get("VENMO_ACCOUNT_HOLDER_NAME", "")
        if not venmo_holder:
            raise RuntimeError(
                "VENMO_ACCOUNT_HOLDER_NAME env var is not set. It must match the "
                "name in the 'To' column of your Venmo CSV exactly, otherwise "
                "Transaction partner labels will be wrong. See .env.sample."
            )
        venmo_frame = venmo_frame.drop([0,len(venmo_frame)-1])
        venmo_frame["Amount (total)"] = venmo_frame["Amount (total)"].str.replace(r"\+ \$", "", regex=True).str.replace(r"- \$", "-", regex=True).str.replace(",", "").astype(float)
        venmo_frame["Transaction partner"] = venmo_frame.apply(lambda x: x["From"] if x["To"] == venmo_holder else x["To"], axis=1)
        venmo_frame["Transaction partner"] = venmo_frame["Transaction partner"].fillna("Unknown").astype(str)
        venmo_frame["Note"] = venmo_frame["Note"].fillna("").astype(str)
        venmo_frame["Description"] = venmo_frame.apply(lambda x: "Venmo from " + x["Transaction partner"] + " for " + x['Note'] if x["Amount (total)"] > 0 else "Venmo to " + x["Transaction partner"] + " for " + x['Note'], axis=1)
        venmo_frame = venmo_frame[["Datetime", "Description", "Amount (total)"]]
        venmo_frame = venmo_frame.rename(columns={"Datetime": "Date", "Description": "Name", "Amount (total)": "Amount"})
        venmo_frame["Date"] = pd.to_datetime(venmo_frame["Date"])
        venmo_frame["Account"] = "Venmo"
        frames_to_concat.append(venmo_frame)

    if sams_frames:
        sam_frame = pd.concat(sams_frames)
        sam_frame = sam_frame.rename(columns={"Transaction Date": "Date", "Description": "Name", "Amount": "Amount"})
        sam_frame["Date"] = pd.to_datetime(sam_frame["Date"])
        sam_frame = sam_frame[sam_frame["Date"].dt.month == month]
        sam_frame = sam_frame[["Date", "Name", "Amount"]]
        sam_frame["Account"] = "Sam's Club"
        frames_to_concat.append(sam_frame)

    if ally_frame is not None:
        ally_frame = ally_frame[["Date", " Amount", " Description"]]
        ally_frame["Date"] = pd.to_datetime(ally_frame["Date"])
        ally_frame = ally_frame[ally_frame["Date"].dt.month == month]
        ally_frame = ally_frame.rename(columns={" Amount": "Amount", " Description": "Name"})
        ally_frame["Account"] = "Ally"
        frames_to_concat.append(ally_frame)

    all_frames = pd.concat(frames_to_concat)

    archive = load_archive()
    personal_profile = load_personal_profile()

    print(f"\nCategorizing {len(all_frames)} transactions via LLM agent...")
    client = AsyncOpenAI()
    results = await categorize_dataframe(
        all_frames, archive, client, personal_profile
    )
    # Preserve prior behavior: failed categorizations land in the CSV as
    # "ERROR: ..." strings rather than aborting the whole run.
    all_frames["Category"] = [
        r.category if r.error is None else f"ERROR: {r.error}" for r in results
    ]

    all_frames = all_frames.sort_values(by=["Account", "Date"], ascending=[True, False])

    all_frames = all_frames[["Date", "Name", "Amount", "Category", "Account"]]

    all_frames.to_csv(os.path.join(SCRIPT_DIR, "categorized_transactions.csv"), index=False)

    print("\nTransactions per account:")
    account_counts = all_frames["Account"].value_counts(dropna=False)
    for account, count in account_counts.items():
        print(f"  {account}: {count}")


if __name__ == "__main__":
    asyncio.run(main())
