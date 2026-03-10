from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "projects" / "machine_learning" / "cnn_digit_recognition.ipynb"


def main() -> int:
    with NOTEBOOK.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    code_cells = [cell for cell in nb.cells if cell.get("cell_type") == "code"]
    if len(code_cells) < 8:
        raise RuntimeError("Unexpected notebook structure: fewer than 8 code cells")

    code_cells[5]["source"] = """from google.colab import files
import zipfile
import io
import shutil
import tempfile
from pathlib import Path

uploaded = files.upload()

extract_dir = tempfile.mkdtemp(prefix="mnist_upload_")

for filename, payload in uploaded.items():
    print(f"Uploaded file: {filename}")
    name = str(filename).lower()
    if name.endswith(".zip"):
        archive = io.BytesIO(payload) if isinstance(payload, (bytes, bytearray)) else filename
        with zipfile.ZipFile(archive, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
    else:
        target = Path(extract_dir) / Path(str(filename)).name
        if isinstance(payload, (bytes, bytearray)):
            target.write_bytes(payload)
        else:
            shutil.copy(str(filename), target)

print(f"Contents extracted to temp dir: {extract_dir}")"""

    code_cells[6]["source"] = """from pathlib import Path

image_paths = [
    p for p in Path(extract_dir).rglob("*")
    if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
]
print(f"Found {len(image_paths)} image files")
for path in image_paths[:20]:
    print(path.name)"""

    code_cells[7]["source"] = """import shutil

for img_path in image_paths:
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        if img is None:
            continue

        img = cv2.resize(img, (28, 28))

        # invert colors
        img = cv2.bitwise_not(img)

        # Normalize and flatten the image
        img = img.astype("float32") / 255.0
        img = img.flatten()
        img = img.reshape(1, 784)

        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")

        plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
        plt.title(f"Predicted: {np.argmax(prediction)}")
        plt.show()
    except Exception as e:
        print(f"Error processing file {img_path}: {e}")

# Cleanup temp extraction directory so image trees are not persisted.
shutil.rmtree(extract_dir, ignore_errors=True)"""

    with NOTEBOOK.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Patched notebook: {NOTEBOOK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
