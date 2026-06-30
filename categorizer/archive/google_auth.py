import os

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from categorizer.paths import CREDENTIALS_PATH, TOKEN_PATH

SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/gmail.readonly",
]


def load_credentials():
    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
        # If the stored token was issued for a narrower scope set than we now
        # require (e.g. calendar-only tokens predating the Sheets scope),
        # force a fresh OAuth flow instead of silently using a token that
        # would 403 on the new API.
        if creds and creds.scopes and not set(SCOPES).issubset(set(creds.scopes)):
            creds = None
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
