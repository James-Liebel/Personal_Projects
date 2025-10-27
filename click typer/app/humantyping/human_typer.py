# app/typing/human_typer.py
from __future__ import annotations
import time
import random
from typing import Optional
from tenacity import retry, wait_exponential_jitter, stop_after_attempt, retry_if_exception_type
from googleapiclient.errors import HttpError

def _sleep_ms(ms: int) -> None:
    if ms > 0:
        time.sleep(ms / 1000.0)

@retry(
    retry=retry_if_exception_type(HttpError),
    wait=wait_exponential_jitter(initial=0.25, max=4.0),
    stop=stop_after_attempt(5),
)
def _batch_update(docs_client, doc_id: str, requests: list[dict], dry_run: bool = False):
    body = {"requests": requests}
    if dry_run:
        print(f"[dry-run] batchUpdate: {body}")
        return {"dryRun": True}
    return docs_client.documents().batchUpdate(documentId=doc_id, body=body).execute()

def _batch_insert(docs_client, doc_id: str, index: int, text_slice: str, dry_run: bool = False):
    return _batch_update(
        docs_client,
        doc_id,
        [{"insertText": {"location": {"index": index}, "text": text_slice}}],
        dry_run=dry_run,
    )

def _delete_range(docs_client, doc_id: str, start_index: int, end_index: int, dry_run: bool = False):
    return _batch_update(
        docs_client,
        doc_id,
        [{"deleteContentRange": {"range": {"startIndex": start_index, "endIndex": end_index}}}],
        dry_run=dry_run,
    )

def _replace_all_text(docs_client, doc_id: str, needle: str, replacement: str, dry_run: bool = False):
    return _batch_update(
        docs_client,
        doc_id,
        [{
            "replaceAllText": {
                "containsText": {"text": needle, "matchCase": True},
                "replaceText": replacement
            }
        }],
        dry_run=dry_run,
    )

def type_string_human_style(
    docs_client,
    doc_id: str,
    start_index: int,
    text: str,
    *,
    chunk: int = 12,
    min_delay_ms: int = 40,
    max_delay_ms: int = 120,
    fast: bool = False,
    dry_run: bool = False,
    strategy: Optional[str] = None,
    marker_text: Optional[str] = None,
    # marker_start / marker_end no longer needed with replaceAllText
) -> None:
    if not isinstance(text, str) or not text:
        raise ValueError("text must be a non-empty string")
    if chunk <= 0:
        chunk = len(text)

    # FAST PATH: replace marker in one shot (safest & quickest)
    if fast and marker_text:
        _replace_all_text(docs_client, doc_id, marker_text, text, dry_run=dry_run)
        return
    if fast:
        _batch_insert(docs_client, doc_id, int(start_index), text, dry_run=dry_run)
        return

    # HUMAN TYPING PATH
    idx = int(start_index)

    # If we targeted a marker, remove it first to avoid grapheme-split errors
    if strategy == "marker" and marker_text:
        _replace_all_text(docs_client, doc_id, marker_text, "", dry_run=dry_run)
        # After removal, we still insert at the original start_index
        idx = int(start_index)

    pos = 0
    n = len(text)
    while pos < n:
        end = min(pos + chunk, n)
        slice_ = text[pos:end]         # never mutate original 'text'
        _batch_insert(docs_client, doc_id, idx, slice_, dry_run=dry_run)
        idx += len(slice_)
        pos = end
        _sleep_ms(random.randint(min_delay_ms, max_delay_ms))
