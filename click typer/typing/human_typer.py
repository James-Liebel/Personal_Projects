"""Human-style typing for Google Docs insertText."""
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
def _batch_insert(docs_client, doc_id: str, index: int, text_slice: str, dry_run: bool = False):
    req = {"requests": [{"insertText": {"location": {"index": index}, "text": text_slice}}]}
    if dry_run:
        print(f"[dry-run] batchUpdate: index={index}, text={repr(text_slice)}")
        return {"dryRun": True}
    return docs_client.documents().batchUpdate(documentId=doc_id, body=req).execute()

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
) -> None:
    if not isinstance(text, str) or text == "":
        raise ValueError("text must be a non-empty string")
    if chunk <= 0:
        chunk = len(text)

    idx = int(start_index)
    if fast:
        _batch_insert(docs_client, doc_id, idx, text, dry_run=dry_run)
        return

    # IMPORTANT: never mutate original 'text'
    pos = 0
    n = len(text)
    while pos < n:
        end = min(pos + chunk, n)
        slice_ = text[pos:end]          # new substring; original text untouched
        _batch_insert(docs_client, doc_id, idx, slice_, dry_run=dry_run)
        idx += len(slice_)              # subsequent inserts move forward
        pos = end
        _sleep_ms(random.randint(min_delay_ms, max_delay_ms))
