"""Resolve where to insert text in a Google Doc."""
from __future__ import annotations
from typing import Optional, Dict, Any

def _get_end_index(doc_json: Dict[str, Any]) -> int:
    body = doc_json.get("body", {})
    content = body.get("content", []) or []
    if not content:
        return 1  # start of body
    # Find last element's endIndex
    for elt in reversed(content):
        if "endIndex" in elt:
            return int(elt["endIndex"]) - 1  # Docs endIndex is 1 past end
    return 1

def _find_marker_index(doc_json: Dict[str, Any], marker: str) -> Optional[int]:
    for elt in doc_json.get("body", {}).get("content", []) or []:
        para = elt.get("paragraph")
        if not para:
            continue
        for el in para.get("elements", []) or []:
            text_run = el.get("textRun")
            if not text_run:
                continue
            txt = text_run.get("content", "")
            if marker in txt:
                # startIndex of this element
                start = el.get("startIndex")
                if start is not None:
                    # Position at the beginning of the marker within the run
                    return int(start) + txt.index(marker)
    return None

def resolve_insertion_point(
    docs_client,
    doc_id: str,
    *,
    named_range: Optional[str] = None,
    bookmark_id: Optional[str] = None,
    marker: Optional[str] = None,
    fallback: str = "end",
) -> Dict[str, Any]:
    doc = docs_client.documents().get(documentId=doc_id).execute()

    # 1) NamedRange
    if named_range and "namedRanges" in doc:
        ranges = doc["namedRanges"].get(named_range)
        if ranges and ranges.get("ranges"):
            start = ranges["ranges"][0]["startIndex"]
            return {"index": int(start), "strategy": "namedRange"}

    # 2) Bookmark
    if bookmark_id and "bookmarks" in doc:
        bm = doc["bookmarks"].get(bookmark_id)
        if bm and "position" in bm and "index" in bm["position"]:
            return {"index": int(bm["position"]["index"]), "strategy": "bookmark"}

    # 3) Marker token
    if marker:
        idx = _find_marker_index(doc, marker)
        if idx is not None:
            return {"index": idx, "strategy": "marker"}

    # 4) Fallback
    if fallback == "start":
        return {"index": 1, "strategy": "fallback-start"}
    # default end
    return {"index": _get_end_index(doc), "strategy": "fallback-end"}
