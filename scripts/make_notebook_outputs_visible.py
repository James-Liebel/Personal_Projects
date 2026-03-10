from __future__ import annotations

from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]


def normalize_notebook(path: Path) -> tuple[bool, int, int]:
    changed = False
    with path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    output_cells = 0
    hidden_cells = 0

    for cell in cells:
        if cell.get("cell_type") != "code":
            continue

        outputs = cell.get("outputs", [])
        if outputs:
            output_cells += 1

        metadata = cell.setdefault("metadata", {})

        # Remove common flags that hide outputs or collapse code/output views.
        if metadata.pop("collapsed", None) is not None:
            changed = True
        if metadata.pop("scrolled", None) is not None:
            changed = True

        jupyter_meta = metadata.get("jupyter")
        if isinstance(jupyter_meta, dict):
            if "source_hidden" in jupyter_meta and jupyter_meta["source_hidden"]:
                jupyter_meta["source_hidden"] = False
                changed = True
                hidden_cells += 1
            if "outputs_hidden" in jupyter_meta and jupyter_meta["outputs_hidden"]:
                jupyter_meta["outputs_hidden"] = False
                changed = True
                hidden_cells += 1
            if not jupyter_meta:
                metadata.pop("jupyter", None)
                changed = True

        # Some environments store hide_input under top-level metadata.
        if metadata.get("hide_input") is True:
            metadata["hide_input"] = False
            changed = True
            hidden_cells += 1

    if changed:
        with path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
            f.write("\n")

    return changed, output_cells, hidden_cells


def main() -> int:
    notebooks = sorted(ROOT.rglob("*.ipynb"))
    changed_count = 0
    zero_output = []
    invalid_notebooks = []

    print(f"Found {len(notebooks)} notebooks.")
    for nb in notebooks:
        rel = nb.relative_to(ROOT)
        try:
            changed, output_cells, hidden_cells = normalize_notebook(nb)
        except Exception as exc:  # noqa: BLE001
            invalid_notebooks.append((rel, str(exc)))
            print(f"{rel} | INVALID_NOTEBOOK | {exc}")
            continue
        if changed:
            changed_count += 1
        if output_cells == 0:
            zero_output.append(rel)
        print(
            f"{rel} | outputs:{output_cells} | hidden_flags_fixed:{hidden_cells} | changed:{changed}"
        )

    print("\n=== Summary ===")
    print(f"Notebooks updated: {changed_count}")
    print(f"Invalid notebooks: {len(invalid_notebooks)}")
    print(f"Notebooks with zero saved outputs: {len(zero_output)}")
    if invalid_notebooks:
        print("Invalid notebook files:")
        for rel, exc in invalid_notebooks:
            print(f"- {rel} ({exc})")
    if zero_output:
        print("Zero-output notebooks:")
        for rel in zero_output:
            print(f"- {rel}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
