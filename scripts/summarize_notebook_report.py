from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "notebook_execution_report.txt"


def first_error_line(section: str) -> str:
    ansi = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")
    clean = ansi.sub("", section)
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    for line in reversed(lines):
        if re.match(
            r"^(ModuleNotFoundError|StdinNotImplementedError|FileNotFoundError|KeyError|ValueError|IndexError|NameError|RuntimeError|URLError|ConnectionError|HTTPError)\b",
            line,
        ):
            return line

    patterns = [
        r"(?:ModuleNotFoundError|StdinNotImplementedError|FileNotFoundError|KeyError|ValueError|IndexError|NameError|RuntimeError|URLError|ConnectionError|HTTPError):.*",
        r"(?:ModuleNotFoundError|StdinNotImplementedError|FileNotFoundError|KeyError|ValueError|IndexError|NameError|RuntimeError|URLError|ConnectionError|HTTPError)\b.*",
    ]
    for pattern in patterns:
        match = re.search(pattern, clean)
        if match:
            return match.group(0).strip()
    return "Unknown error signature"


def main() -> int:
    if not REPORT.exists():
        print(f"Report not found: {REPORT}")
        return 1

    text = REPORT.read_text(encoding="utf-8", errors="ignore")
    blocks = [b for b in text.split("\n## ") if b.strip()]
    parsed: list[tuple[str, str]] = []
    for raw in blocks:
        if raw.startswith("## "):
            raw = raw[3:]
        lines = raw.splitlines()
        if not lines:
            continue
        path = lines[0].strip()
        section = "\n".join(lines[1:])
        parsed.append((path, first_error_line(section)))

    print(f"Failures parsed: {len(parsed)}")
    for path, err in parsed:
        print(f"- {path}: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
