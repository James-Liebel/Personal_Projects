from pathlib import Path
import re
import sys


INDEX_PATH = Path(__file__).resolve().parents[1] / "index.html"


def assert_contains(text: str, snippet: str, message: str) -> None:
    if snippet not in text:
        raise AssertionError(message)


def assert_regex(text: str, pattern: str, message: str) -> None:
    if not re.search(pattern, text, flags=re.DOTALL):
        raise AssertionError(message)


def run() -> None:
    html = INDEX_PATH.read_text(encoding="utf-8")
    print_block_match = re.search(r"@media print\s*\{(.*)\}\s*@media \(prefers-reduced-motion: reduce\)", html, flags=re.DOTALL)
    if not print_block_match:
        raise AssertionError("Missing @media print CSS block.")
    print_block = print_block_match.group(1)

    assert_contains(html, 'id="pdfBtn"', "Missing PDF button id='pdfBtn'.")
    assert_contains(html, 'id="resume"', "Missing resume section id='resume'.")
    assert_contains(html, "@media print", "Missing @media print CSS block.")
    assert_regex(
        print_block,
        r"@page\s*\{\s*size:\s*letter;\s*margin:\s*1\.5cm;\s*\}",
        "Print page size/margins are not configured to letter + 1.5cm.",
    )
    assert_regex(
        print_block,
        r"font-size:\s*11pt;",
        "Print body font-size should be print-safe (~10-12pt).",
    )
    assert_regex(
        print_block,
        r"\.resume-name\s*\{[^}]*font-size:\s*16pt;",
        "Print heading size should be in a print-safe heading range.",
    )
    assert_regex(
        print_block,
        r"page-break-inside:\s*avoid",
        "Print styles should protect major blocks from breaking across pages.",
    )
    assert_regex(
        print_block,
        r"a\[href\^=\"http\"\]::after",
        "Print styles should include inline href rendering for external links.",
    )
    assert_regex(
        print_block,
        r"body\.print-mode\s+#resume\s*\{\s*display:\s*block\s*!important;",
        "Print styles should show resume only in print-mode.",
    )
    assert_regex(
        print_block,
        r"body\.print-mode\s+main,\s*body\.print-mode\s+footer\s*\{\s*display:\s*none\s*!important;",
        "Print styles should hide portfolio sections in print-mode.",
    )
    assert_regex(
        print_block,
        r"overflow:\s*visible\s*!important",
        "Print styles should force visible overflow on key content containers.",
    )
    for decl in re.findall(r"max-height:\s*([^;]+);", print_block):
        if not decl.strip().startswith("none"):
            raise AssertionError("Print styles should not use max-height clamps.")
    if "overflow: hidden" in print_block:
        raise AssertionError("Print styles should not use overflow:hidden clamps.")

    assert_regex(
        html,
        r"function\s+printResume\s*\(\)\s*\{.*?classList\.add\('print-mode'\).*?window\.print\(\);",
        "printResume() must enable print-mode and call window.print().",
    )
    if "body > *:not(#resume)" in html:
        raise AssertionError("printResume() should not hide live screen DOM via body > *:not(#resume).")
    assert_regex(
        html,
        r"window\.addEventListener\('afterprint',\s*restoreResumeView,\s*\{\s*once:\s*true\s*\}\)",
        "printResume() is missing afterprint restore handler.",
    )
    assert_regex(
        html,
        r"window\.addEventListener\('focus',\s*restoreResumeView,\s*\{\s*once:\s*true\s*\}\)",
        "printResume() is missing focus-based restore fallback.",
    )
    assert_regex(
        html,
        r"printRestoreTimer\s*=\s*setTimeout\(restoreResumeView,\s*4000\)",
        "printResume() is missing timeout-based restore fallback.",
    )
    assert_contains(
        html,
        "JAMES LIEBEL",
        "Resume heading text is missing.",
    )
    assert_contains(
        html,
        "jliebel@uchicago.edu",
        "Updated resume contact info is missing.",
    )
    assert_contains(
        html,
        "Incoming University of Chicago student (Class of 2030)",
        "Professional summary text is missing.",
    )

    print("PASS: PDF resume setup checks passed.")


if __name__ == "__main__":
    try:
        run()
    except AssertionError as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
