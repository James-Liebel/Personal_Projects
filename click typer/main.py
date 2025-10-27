"""CLI entrypoint: auth -> locate insertion point -> human typing."""
from __future__ import annotations
import argparse
import os
from dotenv import load_dotenv

from app.auth.google_auth import get_docs_client
from app.docloc.insertion_point import resolve_insertion_point
from app.typing.human_typer import type_string_human_style

def parse_args():
    p = argparse.ArgumentParser(description="Type a string into a Google Doc.")
    p.add_argument("--doc-id", default=os.getenv("DOC_ID"), required=False)
    p.add_argument("--text", default=os.getenv("TEXT"), required=False)
    p.add_argument("--named-range", default=os.getenv("NAMED_RANGE"))
    p.add_argument("--bookmark-id", default=os.getenv("BOOKMARK_ID"))
    p.add_argument("--marker", default=os.getenv("MARKER"))
    p.add_argument("--fallback", choices=["start", "end"], default=os.getenv("FALLBACK", "end"))
    p.add_argument("--fast", action="store_true", default=os.getenv("FAST") == "1")
    p.add_argument("--dry-run", action="store_true", default=os.getenv("DRY_RUN") == "1")
    p.add_argument("--chunk", type=int, default=int(os.getenv("CHUNK", "12")))
    p.add_argument("--min-delay-ms", type=int, default=int(os.getenv("MIN_DELAY_MS", "40")))
    p.add_argument("--max-delay-ms", type=int, default=int(os.getenv("MAX_DELAY_MS", "120")))
    return p.parse_args()

def main():
    load_dotenv()
    args = parse_args()
    if not args.doc_id or not args.text:
        raise SystemExit("Missing --doc-id and/or --text (or set DOC_ID/TEXT in .env).")

    docs = get_docs_client()
    loc = resolve_insertion_point(
        docs,
        args.doc_id,
        named_range=args.named_range,
        bookmark_id=args.bookmark_id,
        marker=args.marker,
        fallback=args.fallback,
    )
    type_string_human_style(
        docs,
        args.doc_id,
        loc["index"],
        args.text,
        chunk=args.chunk,
        min_delay_ms=args.min_delay_ms,
        max_delay_ms=args.max_delay_ms,
        fast=args.fast,
        dry_run=args.dry_run,
    )
    print(f"Typed using strategy={loc['strategy']} at index={loc['index']}.")

if __name__ == "__main__":
    main()
