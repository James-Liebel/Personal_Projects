# app/docloc/insertion_point.py
from __future__ import annotations
from typing import Optional, Dict, Any, Tuple

def _get_end_index(doc_json: Dict[str, Any]) -> int:
    body = doc_json.get("body", {})
    content = body.get("content", []) or []
    if not content:
        return 1
    for elt in reversed(content):
        if "endIndex" in elt:
            return int(elt["endIndex"]) - 1
    return 1

def _find_marker_range(doc_json: Dict[str, Any], marker: str) -> Optional[Tuple[int, int]]:
    for elt in doc_json.get("body", {}).get("content", []) or []:
        para = elt.get("paragraph")
        if not para:
            continue
        for el in para.get("elements", []) or []:
            text_run = el.get("textRun")
            if not text_run:
                continue
            txt = text_run.get("content", "")
            if not txt:
                continue
            pos = txt.find(marker)
            if pos != -1:
                start = int(el.get("startIndex"))
                mstart = start + pos
                mend = mstart + len(marker)
                return (mstart, mend)
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

    if named_range and "namedRanges" in doc:
        ranges = doc["namedRanges"].get(named_range)
        if ranges and ranges.get("ranges"):
            start = int(ranges["ranges"][0]["startIndex"])
            return {"index": start, "strategy": "namedRange"}

    if bookmark_id and "bookmarks" in doc:
        bm = doc["bookmarks"].get(bookmark_id)
        if bm and "position" in bm and "index" in bm["position"]:
            return {"index": int(bm["position"]["index"]), "strategy": "bookmark"}

    if marker:
        rng = _find_marker_range(doc, marker)
        if rng:
            mstart, mend = rng
            return {
                "index": mstart,
                "strategy": "marker",
                "marker_start": mstart,
                "marker_end": mend,
            }

    if fallback == "start":
        return {"index": 1, "strategy": "fallback-start"}
    return {"index": _get_end_index(doc), "strategy": "fallback-end"}
