import json
import os

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from categorizer.paths import CREDENTIALS_PATH, TOKEN_PATH

SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    # Read/write, not .readonly: the pipeline writes each month's categorized
    # transactions into a new tab at the end of a run (see sheets.write_month_tab).
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/gmail.readonly",
]


def _stored_scopes():
    """Scopes recorded in token.json, or None if unreadable.

    Read straight from the file rather than from a Credentials object: passing
    `scopes=SCOPES` to from_authorized_user_file makes `creds.scopes` echo back
    what we *asked* for (google/oauth2/credentials.py only falls back to the
    stored value when `scopes is None`), so comparing against it is a no-op.
    """
    try:
        with open(TOKEN_PATH) as f:
            scopes = json.load(f).get("scopes")
    except (OSError, ValueError):
        # Unreadable/corrupt token file: report unknown so the caller discards
        # it and re-runs the OAuth flow rather than trusting it.
        return None
    if isinstance(scopes, str):
        scopes = scopes.split(" ")
    return scopes


def load_credentials():
    creds = None
    if os.path.exists(TOKEN_PATH):
        # If the stored token was issued for a narrower scope set than we now
        # require (e.g. read-only Sheets tokens predating the archive writer),
        # force a fresh OAuth flow instead of silently using a token that
        # would 403 on the new API.
        stored = _stored_scopes()
        if stored is None or not set(SCOPES).issubset(set(stored)):
            os.remove(TOKEN_PATH)
        else:
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds or not creds.valid:
        refreshed = False
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                refreshed = True
            except RefreshError:
                # The stored refresh token was revoked or expired (e.g. OAuth
                # clients in "Testing" status expire refresh tokens after 7
                # days). Discard the dead token and fall through to a fresh
                # browser flow instead of crashing on invalid_grant.
                if os.path.exists(TOKEN_PATH):
                    os.remove(TOKEN_PATH)
                creds = None
        if not refreshed:
            if not os.path.exists(CREDENTIALS_PATH):
                raise FileNotFoundError(
                    f"Google OAuth client secrets not found at {CREDENTIALS_PATH}. "
                    "Create an OAuth client at console.cloud.google.com (Calendar + Sheets APIs enabled), "
                    "download the JSON, and save it to that path."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())
    return creds


def build_calendar_service():
    return build("calendar", "v3", credentials=load_credentials(), cache_discovery=False)


def build_sheets_service():
    return build("sheets", "v4", credentials=load_credentials(), cache_discovery=False)


def build_gmail_service():
    return build("gmail", "v1", credentials=load_credentials(), cache_discovery=False)
