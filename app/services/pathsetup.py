"""Make the existing research code in scripts/ importable.

Must be imported before anything that touches the scripts package:
- scripts/explainer/explainer_utils.py uses flat imports (`from dataloader import ...`),
  so scripts/ itself has to be on sys.path.
- scripts/explainer/ig_explainer.py imports `from scripts.explainer...`,
  so the repo root has to be on sys.path as well.
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")

for path in (REPO_ROOT, SCRIPTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)
