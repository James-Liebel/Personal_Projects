# --- Click Typer main script (final simplified version) ---
import sys, pathlib, random, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.auth.google_auth import get_docs_client
from app.docloc.insertion_point import resolve_insertion_point
from app.humantyping.human_typer import _replace_all_text, _batch_insert

# 🧠 CONFIG: set everything here once
DOC_ID = "1ShQP5NtUoK14GznH14Hta3g3MX0Ckfs5y58BgoSGOtY"
MARKER = "*"

TEXT_TO_TYPE = """This is my automated text typed in human style.
You can change me anytime right here in main.py!
"""

# typing feel
CHUNK = 10             # characters per "keystroke"
MIN_DELAY_MS = 40      # fastest possible typing delay
MAX_DELAY_MS = 120     # slowest typing delay

def type_human_style(docs_client, doc_id, start_index, text, marker):
    """Remove marker and type slowly."""
    # Remove the marker safely
    _replace_all_text(docs_client, doc_id, marker, "")
    idx = int(start_index)
    pos = 0
    while pos < len(text):
        end = min(pos + CHUNK, len(text))
        slice_ = text[pos:end]
        _batch_insert(docs_client, doc_id, idx, slice_)
        idx += len(slice_)
        pos = end
        time.sleep(random.randint(MIN_DELAY_MS, MAX_DELAY_MS) / 1000.0)
    print("✅ Typing complete!")

def main():
    docs = get_docs_client()
    loc = resolve_insertion_point(docs, DOC_ID, marker=MARKER)
    print(f"Typing into doc using marker strategy at index {loc['index']}...")
    type_human_style(docs, DOC_ID, loc["index"], TEXT_TO_TYPE, MARKER)

if __name__ == "__main__":
    main()
