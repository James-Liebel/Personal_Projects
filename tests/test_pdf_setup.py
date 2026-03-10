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

    assert_contains(html, 'id="pdfBtn"', "Missing PDF button id='pdfBtn'.")
    assert_contains(html, 'id="resume"', "Missing resume section id='resume'.")
    assert_contains(html, "@media print", "Missing @media print CSS block.")
    assert_contains(
        html,
        "@page { size: letter portrait; margin: 0.45in; }",
        "Print page size/margins are not configured for one-page letter output.",
    )
    assert_contains(
        html,
        "max-height: calc(11in - 0.9in);",
        "Missing one-page max-height constraint for printed resume.",
    )
    assert_contains(
        html,
        "overflow: hidden;",
        "Printed resume does not enforce overflow clipping.",
    )

    assert_regex(
        html,
        r"function\s+printResume\s*\(\)\s*\{.*?document\.querySelectorAll\('body > \*:not\(#resume\)'\).*?window\.print\(\);",
        "printResume() must hide non-resume content and call window.print().",
    )
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
