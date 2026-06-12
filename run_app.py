"""GNN4NMR web app entry point.

Usage:
    .venv\\Scripts\\python.exe run_app.py

Then open http://127.0.0.1:5000 in a browser.

The app loads the default dataset (data/all_graphs_with_length_filtered.pkl)
on startup. Train a model in the "Train & Models" tab (no checkpoint ships
with the repo), load it, and explore predictions/explanations in "Explore".

Notes:
- The Flask reloader is disabled on purpose: the app holds in-process state
  (dataset registry, active model, training thread) that must live in a
  single process.
- Device is auto-selected (CUDA if available, else CPU). Override with the
  environment variable GNN4NMR_DEVICE=cpu|cuda.
"""
from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True)
