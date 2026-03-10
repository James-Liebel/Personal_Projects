from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


def main() -> int:
    parser = argparse.ArgumentParser(description="Print code cells from a notebook.")
    parser.add_argument("notebook", help="Path to .ipynb file")
    args = parser.parse_args()

    nb_path = Path(args.notebook)
    if not nb_path.exists():
        print(f"Notebook not found: {nb_path}")
        return 1

    with nb_path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    idx = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        idx += 1
        print(f"\n--- code cell {idx} ---")
        print(cell.get("source", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
