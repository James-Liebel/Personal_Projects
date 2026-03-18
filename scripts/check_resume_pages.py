from __future__ import annotations

import contextlib
import http.server
import os
import shutil
import socketserver
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(r"c:\Users\james\Personal_Projects")
PAGES = [
    ROOT / "index.html",
    ROOT / "projects" / "web" / "resume.html",
    ROOT / "projects" / "web" / "data_science_resume.html",
]


@dataclass
class PageReport:
    path: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)


class RefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.refs: list[tuple[str, str]] = []
        self.meta_viewport = False
        self.iframes = 0
        self.media_queries = 0
        self._in_style = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        if tag == "meta" and attr_map.get("name", "").lower() == "viewport":
            self.meta_viewport = True
        if tag in {"a", "link"} and attr_map.get("href"):
            self.refs.append((tag, attr_map["href"]))
        if tag in {"script", "img", "iframe"} and attr_map.get("src"):
            self.refs.append((tag, attr_map["src"]))
        if tag == "iframe":
            self.iframes += 1
        if tag == "style":
            self._in_style = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "style":
            self._in_style = False

    def handle_data(self, data: str) -> None:
        if self._in_style and "@media" in data:
            self.media_queries += data.count("@media")


def is_external(ref: str) -> bool:
    lowered = ref.lower()
    return lowered.startswith(("http://", "https://", "mailto:", "tel:", "javascript:"))


def resolve_ref(page: Path, ref: str) -> Path | None:
    if not ref or ref.startswith("#"):
        return None
    clean = ref.split("#", 1)[0].split("?", 1)[0]
    if not clean or is_external(clean):
        return None
    return (page.parent / clean).resolve()


def scan_page(path: Path) -> PageReport:
    report = PageReport(path=path)
    parser = RefParser()
    try:
        text = path.read_text(encoding="utf-8")
        parser.feed(text)
    except Exception as exc:  # pragma: no cover - defensive
        report.errors.append(f"Failed to parse HTML: {exc}")
        return report

    if not parser.meta_viewport:
        report.errors.append("Missing viewport meta tag.")
    else:
        report.checks.append("Viewport meta tag present.")

    if parser.media_queries == 0:
        report.warnings.append("No CSS media queries detected.")
    else:
        report.checks.append(f"Detected {parser.media_queries} CSS media query blocks.")

    if parser.iframes:
        report.checks.append(f"Found {parser.iframes} iframe embeds.")

    for tag, ref in parser.refs:
        resolved = resolve_ref(path, ref)
        if resolved is None:
            continue
        if not resolved.exists():
            report.errors.append(f"Missing local asset for <{tag}>: {ref}")

    if "data_science_resume.html" in path.name:
        if "grid-template-columns: repeat(2, 1fr);" not in text:
            report.warnings.append("Expected skills grid pattern not found.")
        if "@media (max-width:" not in text:
            report.errors.append("No small-screen breakpoint found.")
    return report


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


def start_server() -> tuple[socketserver.TCPServer, int]:
    os.chdir(ROOT)
    server = socketserver.TCPServer(("127.0.0.1", 0), QuietHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port


def fetch(url: str) -> tuple[int | None, str]:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            body = response.read(4096).decode("utf-8", errors="ignore")
            return response.status, body
    except urllib.error.URLError as exc:
        return None, str(exc)


def find_browser() -> Path | None:
    candidates = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def browser_smoke(browser: Path, url: str, width: int, height: int, marker: str) -> str | None:
    user_data_dir = ROOT / ".tmp-browser-profile"
    with contextlib.suppress(Exception):
        shutil.rmtree(user_data_dir, ignore_errors=True)
    user_data_dir.mkdir(exist_ok=True)
    try:
        command = [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-crash-reporter",
            "--disable-features=Crashpad,RendererCodeIntegrity",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=3000",
            "--user-data-dir=" + str(user_data_dir),
            "--window-size=%d,%d" % (width, height),
            "--dump-dom",
            url,
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return f"Browser smoke failed to start: {exc}"
    finally:
        with contextlib.suppress(Exception):
            shutil.rmtree(user_data_dir, ignore_errors=True)

    if result.returncode != 0:
        return f"Browser smoke failed with exit code {result.returncode}: {result.stderr.strip()[:300]}"
    if marker not in result.stdout:
        return f"Browser smoke missing expected marker '{marker}'."
    return None


def main() -> int:
    reports = [scan_page(page) for page in PAGES]

    server, port = start_server()
    time.sleep(0.5)
    try:
        for report in reports:
            rel = report.path.relative_to(ROOT).as_posix()
            status, body = fetch(f"http://127.0.0.1:{port}/{rel}")
            if status != 200:
                report.errors.append(f"HTTP fetch failed for {rel}: {body}")
            else:
                report.checks.append(f"HTTP fetch passed for {rel}.")
    finally:
        with contextlib.suppress(Exception):
            server.shutdown()
            server.server_close()

    browser = find_browser()
    if browser is None:
        for report in reports:
            report.warnings.append("No Chrome/Edge executable found for browser smoke tests.")
    else:
        marker_map = {
            "index.html": "Resume + Portfolio",
            "resume.html": "Interactive Resume",
            "data_science_resume.html": "Data Science Resume",
        }
        server, port = start_server()
        time.sleep(0.5)
        try:
            for report in reports:
                rel = report.path.relative_to(ROOT).as_posix()
                url = f"http://127.0.0.1:{port}/{rel}"
                marker = marker_map[report.path.name]
                for width, height, label in [(1440, 1024, "desktop"), (390, 844, "mobile")]:
                    error = browser_smoke(browser, url, width, height, marker)
                    if error:
                        report.errors.append(f"{label} browser smoke: {error}")
                    else:
                        report.checks.append(f"{label} browser smoke passed.")
        finally:
            with contextlib.suppress(Exception):
                server.shutdown()
                server.server_close()

    has_errors = False
    for report in reports:
        print(f"\n== {report.path.relative_to(ROOT)} ==")
        for check in report.checks:
            print(f"PASS: {check}")
        for warning in report.warnings:
            print(f"WARN: {warning}")
        for error in report.errors:
            has_errors = True
            print(f"FAIL: {error}")

    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
