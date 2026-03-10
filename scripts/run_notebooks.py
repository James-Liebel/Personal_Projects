from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import traceback

import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import NoSuchKernel


ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_SECONDS = 600


@dataclass
class Result:
    path: Path
    ok: bool
    error: str = ""


def has_saved_outputs(nb: nbformat.NotebookNode) -> bool:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code" and cell.get("outputs"):
            return True
    return False


def execute_notebook(path: Path, timeout_seconds: int) -> Result:
    errors: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        declared = nb.metadata.get("kernelspec", {}).get("name")
        candidates = [k for k in [declared, "python3", "python"] if k]

        seen = set()
        kernels = []
        for kernel in candidates:
            if kernel not in seen:
                kernels.append(kernel)
                seen.add(kernel)

        for kernel in kernels:
            try:
                client = NotebookClient(
                    nb,
                    timeout=timeout_seconds,
                    kernel_name=kernel,
                    resources={"metadata": {"path": str(path.parent)}},
                    allow_errors=False,
                )
                client.execute()
                with path.open("w", encoding="utf-8") as f:
                    nbformat.write(nb, f)
                return Result(path=path, ok=True)
            except NoSuchKernel:
                errors.append(f"NoSuchKernel: {kernel}")
            except Exception:  # noqa: BLE001
                errors.append(f"[kernel={kernel}]\n{traceback.format_exc(limit=8)}")

        return Result(path=path, ok=False, error="\n".join(errors))
    except Exception:  # noqa: BLE001
        return Result(path=path, ok=False, error=traceback.format_exc(limit=8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute notebooks in-place and persist outputs."
    )
    parser.add_argument(
        "--only-missing-outputs",
        action="store_true",
        help="Execute only notebooks that currently have zero saved code-cell outputs.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=TIMEOUT_SECONDS,
        help=f"Per-cell timeout in seconds (default: {TIMEOUT_SECONDS}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_notebooks = sorted(ROOT.rglob("*.ipynb"))
    print(f"Found {len(all_notebooks)} notebooks.")

    notebooks: list[Path] = []
    scan_failures: list[Result] = []
    skipped_with_outputs = 0

    for nb_path in all_notebooks:
        if not args.only_missing_outputs:
            notebooks.append(nb_path)
            continue

        try:
            with nb_path.open("r", encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)
            if has_saved_outputs(nb):
                skipped_with_outputs += 1
                continue
            notebooks.append(nb_path)
        except Exception:  # noqa: BLE001
            scan_failures.append(
                Result(path=nb_path, ok=False, error=traceback.format_exc(limit=8))
            )

    if args.only_missing_outputs:
        print(f"Selected notebooks with missing outputs: {len(notebooks)}")
        print(f"Skipped notebooks that already had outputs: {skipped_with_outputs}")
        print(f"Unreadable notebooks during scan: {len(scan_failures)}")

    if not notebooks and not scan_failures:
        print("Nothing to execute.")
        return 0

    results: list[Result] = []
    for idx, nb_path in enumerate(notebooks, start=1):
        rel = nb_path.relative_to(ROOT)
        print(f"[{idx}/{len(notebooks)}] Executing {rel} ...")
        result = execute_notebook(nb_path, args.timeout_seconds)
        results.append(result)
        if result.ok:
            print("  PASS")
        else:
            print("  FAIL")
            print(result.error.splitlines()[-1] if result.error else "Unknown error")

    failures = [r for r in results if not r.ok] + scan_failures
    print("\n=== Summary ===")
    print(f"Passed: {len(results) - len([r for r in results if not r.ok])}")
    print(f"Failed: {len(failures)}")

    if failures:
        print("\nFailed notebooks:")
        for failure in failures:
            print(f"- {failure.path.relative_to(ROOT)}")
        report = ROOT / "notebook_execution_report.txt"
        with report.open("w", encoding="utf-8") as f:
            for failure in failures:
                f.write(f"## {failure.path.relative_to(ROOT)}\n")
                f.write(failure.error)
                if not failure.error.endswith("\n"):
                    f.write("\n")
                f.write("\n")
        print(f"\nDetailed traceback written to {report.relative_to(ROOT)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
