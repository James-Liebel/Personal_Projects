"""Auth utilities for Google Docs API (Installed App OAuth 2.0)."""
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import json

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

CREDENTIALS_DIR = Path(".credentials")
TOKEN_PATH = CREDENTIALS_DIR / "token.json"
CLIENT_SECRET_PATH = Path("credentials") / "client_secret.json"
DEFAULT_SCOPES: Sequence[str] = ("https://www.googleapis.com/auth/documents",)

def get_docs_client(scopes: Sequence[str] = DEFAULT_SCOPES):
    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, scopes)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CLIENT_SECRET_PATH.exists():
                raise FileNotFoundError(
                    f"Missing {CLIENT_SECRET_PATH}. Download OAuth client JSON and place it there."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_PATH), scopes)
            creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json())
    return build("docs", "v1", credentials=creds)
