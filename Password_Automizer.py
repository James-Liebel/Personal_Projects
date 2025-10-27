import os, sys, pickle, argparse
from typing import List
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# ---- Optional dependency: cryptography ----
try:
    from cryptography.fernet import Fernet
except ImportError:
    Fernet = None  # We'll guard usage and show a friendly error.

# ---------------- CONFIG ----------------
SCOPES = [
    'https://www.googleapis.com/auth/documents',
    'https://www.googleapis.com/auth/drive.file'
]
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.pickle'
KEY_FILE = 'master.key'  # Local Fernet key (keep OUT of git)

# ---------------- AUTHENTICATION ----------------
def authenticate():
    """Robust auth: attempts token refresh; on failure, forces a clean login.
       Also supports console flow if localhost callback is blocked."""
    creds = None

    # 0) ensure credentials.json exists
    if not os.path.exists(CREDENTIALS_FILE):
        print(f"[Auth] Missing {CREDENTIALS_FILE} in {os.getcwd()}")
        sys.exit(1)

    # 1) load cached creds if present
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, 'rb') as f:
                creds = pickle.load(f)
        except Exception:
            creds = None  # corrupt cache -> force re-auth

    # 2) try refresh if possible
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception as e:
            # stale/invalid refresh token -> drop and force fresh login
            print(f"[Auth] Refresh failed ({e}); wiping cache and re-authenticating…")
            try:
                os.remove(TOKEN_FILE)
            except Exception:
                pass
            creds = None

    # 3) fresh login if no valid creds
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
        try:
            # preferred: opens a localhost callback
            creds = flow.run_local_server(port=0)
        except Exception as e:
            print(f"[Auth] Local server auth failed ({e}); falling back to console copy/paste flow.")
            # fallback: copy/paste code from browser (works behind strict firewalls)
            creds = flow.run_console()

        with open(TOKEN_FILE, 'wb') as f:
            pickle.dump(creds, f)

    return build('docs', 'v1', credentials=creds)

# ---------------- ENCRYPTION HELPERS ----------------
def ensure_crypto_available():
    if Fernet is None:
        print("This feature requires 'cryptography'. Install it: pip install cryptography")
        sys.exit(1)

def load_or_create_key(path: str = KEY_FILE) -> bytes:
    """
    Loads a Fernet key from path, or creates a new one if missing.
    Store this file securely; anyone with it can decrypt the passwords.
    """
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return f.read()
    key = Fernet.generate_key()
    # Write with restricted permissions where possible
    with open(path, 'wb') as f:
        f.write(key)
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass  # best-effort on non-POSIX systems
    print(f"[Info] Created new encryption key at {path}. Keep it safe and OUT of git.")
    return key

def encrypt_password(fernet: Fernet, plaintext: str) -> str:
    return fernet.encrypt(plaintext.encode('utf-8')).decode('utf-8')

def decrypt_password(fernet: Fernet, token: str) -> str:
    return fernet.decrypt(token.encode('utf-8')).decode('utf-8')

# ---------------- DOC HELPERS ----------------
def get_document_text(service, doc_id) -> str:
    doc = service.documents().get(documentId=doc_id).execute()
    text = []
    for content in doc.get('body', {}).get('content', []):
        paragraph = content.get('paragraph')
        if not paragraph:
            continue
        for el in paragraph.get('elements', []):
            segment = el.get('textRun', {}).get('content')
            if segment:
                text.append(segment)
    return ''.join(text).strip()

def update_document(service, doc_id, text: str):
    """Replace the entire document body with `text` (avoid deleting the final newline)."""
    doc = service.documents().get(documentId=doc_id).execute()
    content = doc.get('body', {}).get('content', [])
    # endIndex is the *end of the document*, and includes a trailing newline.
    end_index = content[-1].get('endIndex', 1) if content else 1

    # Do not include the terminal newline in delete range.
    delete_end = max(1, end_index - 1)

    requests = []
    if delete_end > 1:
        requests.append({
            'deleteContentRange': {
                'range': {'startIndex': 1, 'endIndex': delete_end}
            }
        })
    # Always insert at index 1 (start of body), we keep one trailing newline in text.
    requests.append({
        'insertText': {
            'location': {'index': 1},
            'text': text if text.endswith('\n') else text + '\n'
        }
    })

    service.documents().batchUpdate(documentId=doc_id, body={'requests': requests}).execute()


def parse_lines(existing_text: str) -> List[str]:
    return [ln.strip() for ln in existing_text.splitlines() if ln.strip()]

# ---------------- CORE LOGIC ----------------
def add_password_entry_encrypted(service, doc_id, company, username, password):
    """
    Stores: Company | username | ENC:<token>
    Only the password is encrypted; company and username stay plaintext for sorting.
    """
    ensure_crypto_available()
    key = load_or_create_key()
    fernet = Fernet(key)

    existing_text = get_document_text(service, doc_id)
    lines = parse_lines(existing_text)

    enc_pwd = encrypt_password(fernet, password)
    new_line = f"{company} | {username} | ENC:{enc_pwd}"
    lines.append(new_line)

    # Sort by company name (before the first '|')
    lines.sort(key=lambda line: line.split('|')[0].strip().lower())

    update_document(service, doc_id, '\n'.join(lines) + '\n')

def list_entries_decrypted(service, doc_id):
    """
    Reads the doc and prints entries with decrypted passwords (if ENC: is found).
    """
    ensure_crypto_available()
    key = load_or_create_key()
    fernet = Fernet(key)

    text = get_document_text(service, doc_id)
    lines = parse_lines(text)
    if not lines:
        print("(No entries found.)")
        return

    print("\nCompany | Username | Password")
    print("-" * 60)
    for ln in lines:
        parts = [p.strip() for p in ln.split('|')]
        if len(parts) != 3:
            print(ln)  # unexpected format; print raw
            continue
        company, user, pwd_field = parts
        if pwd_field.startswith("ENC:"):
            token = pwd_field[4:].strip()
            try:
                dec = decrypt_password(fernet, token)
                pwd_display = dec
            except Exception as e:
                pwd_display = f"[decrypt-error: {e}]"
        else:
            pwd_display = pwd_field  # legacy/plaintext
        print(f"{company} | {user} | {pwd_display}")
    print()

# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser(description="Password list manager (Google Docs + Fernet encryption).")
    parser.add_argument("--doc-id", required=True, help="Google Doc ID")
    parser.add_argument("--list", action="store_true", help="List entries with decrypted passwords (local key required)")
    args = parser.parse_args()

    service = authenticate()

    if args.list:
        list_entries_decrypted(service, args.doc_id)
        return

    print("Add credentials (leave Company blank to quit).")
    while True:
        company = input("Company name: ").strip()
        if not company:
            print("Done.")
            break
        username = input("Username: ").strip()
        password = input("Password: ").strip()
        if not username or not password:
            print("Username and password are required.\n")
            continue

        add_password_entry_encrypted(service, args.doc_id, company, username, password)
        print(f"✅ Added and sorted entry for {company}\n")

if __name__ == "__main__":
    main()
