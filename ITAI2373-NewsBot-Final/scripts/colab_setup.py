"""
One-call environment setup used by every notebook.

* In Google Colab: clones the repository (if needed), installs the
  requirements and spaCy models, and moves into the project folder.
* Locally: just puts the project root on ``sys.path``.

Usage inside a notebook::

    import os, sys, subprocess
    ... (see the first cell of any notebook)
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/Haryapandhita/ITAI2373-NewsBot-Final.git"
REPO_DIR = Path("/content/ITAI2373-NewsBot-Final")
PROJECT_SUBDIR = "ITAI2373-NewsBot-Final"


def _run(cmd):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def setup(install: bool = True) -> Path:
    in_colab = "google.colab" in sys.modules
    if in_colab:
        if not REPO_DIR.exists():
            _run(["git", "clone", "-q", REPO_URL, str(REPO_DIR)])
        root = REPO_DIR / PROJECT_SUBDIR
        os.chdir(root)
        if install and not Path("/content/.newsbot_installed").exists():
            _run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
            for model in ("en_core_web_sm", "xx_ent_wiki_sm"):
                _run([sys.executable, "-m", "spacy", "download", model, "-q"])
            Path("/content/.newsbot_installed").touch()
    else:
        root = Path(__file__).resolve().parents[1]
        os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    print(f"Project root: {root}  |  Colab: {in_colab}")
    return root


if __name__ == "__main__":
    setup()
