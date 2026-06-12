# gnn-for-nmr
Anwendung von Graph Neural Networks zur Korrektur chemischer Verschiebungen von DFT zu CCSD(T)-Qualität in der Kernspinresonanzspektroskopie

## Web App

A Flask web app for building/training/loading the GNN models, predicting NMR
chemical shifts per atom, and explaining node-level predictions (GNNExplainer
and Integrated Gradients) on an interactive 3D molecule viewer.

### Setup (once)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

(With an NVIDIA GPU, install the matching CUDA torch wheel instead of the CPU one.)

### Run

```powershell
.\.venv\Scripts\python.exe run_app.py
```

Then open http://127.0.0.1:5000. The default dataset
`data/all_graphs_with_length_filtered.pkl` is loaded on startup; other `.pkl`
files in `data/` (same networkx graph schema) can be selected in the UI.

- **Train & Models**: configure and train a model (all PyG operators from
  `scripts/operators.py`), watch live loss curves, stop early, and load saved
  checkpoints (stored in `models/` as `.pt` + `.json` + `.stats.pkl`).
- **Explore**: pick a molecule, inspect it in 3D (3Dmol.js, the engine behind
  py3Dmol), click an atom to see its predicted shift, ground truth, raw
  features, and run per-atom explanations — feature/neighbor/bond importances
  are shown as bars and highlighted in the 3D view.
