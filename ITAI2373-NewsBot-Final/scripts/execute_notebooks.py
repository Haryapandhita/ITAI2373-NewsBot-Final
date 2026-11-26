"""Execute every notebook in place so the outputs are stored (``python scripts/execute_notebooks.py [names...]``)."""

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def run(path: Path) -> None:
    t0 = time.time()
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(NB_DIR)}}).execute()
    nbformat.write(nb, path)
    print(f"✓ {path.name} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(p.name for p in NB_DIR.glob("0*.ipynb"))
    failed = []
    for name in names:
        try:
            run(NB_DIR / name)
        except Exception as exc:  # keep going, report at the end
            failed.append(name)
            print(f"✗ {name}: {str(exc)[:1500]}", flush=True)
    print("FAILED:", failed if failed else "none")
