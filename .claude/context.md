# gnn4nmr — Context for Claude Code

Project: Graph Neural Networks that correct DFT NMR chemical shifts toward
CCSD(T) quality. Node-level regression on H and C atoms of molecular graphs.

## Environment (important!)

- **Always use the repo venv:** `.\.venv\Scripts\python.exe` (Python 3.13,
  torch CPU, torch_geometric, captum, flask). The active conda base
  (miniforge) has **no torch/PyG/numpy** — running `python` directly will fail.
- No NVIDIA GPU on this machine (Intel Iris Xe) → CPU-only torch.
- Recreate the env with `requirements.txt` + CPU torch index
  (`pip install torch --index-url https://download.pytorch.org/whl/cpu`).

## Layout

- `scripts/` — original research code (CLI/notebook driven). Treated as
  **read-only library** by the web app: `dataloader.py` (ShiftDataset),
  `model.py` (HeteroGNNModel), `operators.py`, `explainer/` (GNNExplainer +
  Captum IG utilities). Imports are flat (`from dataloader import ...`) AND
  package-style (`from scripts.explainer...`), so **both** the repo root and
  `scripts/` must be on sys.path (`app/services/pathsetup.py` does this).
- `app/` + `run_app.py` — Flask web app (train/load models, predict shifts,
  explain atoms on a 3Dmol.js viewer). English code + UI.
  Brand colors: `#004e9f` (primary), `#fcba00` (accent), `#909085` (gray).
- `data/` — `all_graphs_with_length_filtered.pkl` (712 networkx graphs,
  73 compounds; default) and `all_graphs_with_length.pkl` (940 graphs).
  3D coords are **not** in the graphs: SDF V2000 files live at
  `data/orca_xyz_formate/{compound}/{compound}_{structure:02d}.sdf`
  (fallback: `data/converted_sdf_files/`).
- `models/` — checkpoints written by the app: `<id>.pt` (state_dict),
  `<id>.json` (config/metrics/history sidecar), `<id>.stats.pkl`
  (normalization stats). Git-ignored.
- `notebooks/` — exploration only; not part of the app.

## Running / verifying

```powershell
.\.venv\Scripts\python.exe run_app.py     # http://127.0.0.1:5000
```
Reloader/debug are intentionally off (in-process state + threads).
API smoke: `/api/health`, `/api/graphs`, POST `/api/train`
(`{"num_epochs":3,"hidden_dim":32,"out_dim":32,"batch_size":8}`),
`/api/train/status`, POST `/api/models/<id>/load`,
`/api/graphs/0/predictions`, POST `/api/explain`, `/api/explain/jobs/<id>`.

## Data model facts

- Node types H (33 features), C (39), Others (16); 13 element one-hots first,
  continuous features z-scored from index 13 (stats over the whole dataset).
  Edge features: 10. Target attr: `shift_high-low` (ppm), NaN = unlabeled.
- Node ordering: iterate `nx_g.nodes()`, bucket per element class →
  type-local index. **nx node IDs == SDF atom indices (0-based)** — this is
  the contract behind 3D click ↔ tensor-row mapping
  (`dataset_service.get_node_map` is the single source of truth).
- `nx_g.graph["compound"]` is an **int** (e.g. 1); SDF paths need
  zero-padded `001` (`dataset_service.compound_str`).

## Known traps (all worked around in app/, don't re-introduce)

1. `scripts/main.py` hardcodes `in_dim_dict["H"] = 34` — **wrong**, the
   dataloader produces 33. Always infer dims from a sample HeteroData.
2. `scripts/train.py` has a module-level `import wandb` → never import it
   (wandb is not installed). The app has its own equivalent loop in
   `app/services/training_service.py`.
3. `HeteroGNNModel.forward` **mutates** `x_dict` (and `edge_attr_dict` for
   GINEConv) in place → every inference goes through
   `model_service.SafeModelAdapter` (dict copy + edge_attr clone). Without it,
   repeated forwards (IG steps) crash for GINEConv.
4. PyG's GNNExplainer installs masks on the model's MessagePassing modules →
   all forwards/explanations are serialized via `STATE.inference_lock`,
   masks cleared in `finally`.
5. JSON: NaN/torch/numpy never go to the frontend directly — everything is
   passed through `jsonutil.to_jsonable` (NaN → null).
6. Repo `.gitignore` ignores **all `__init__.py`** files; `app/__init__.py`
   and `app/services/__init__.py` are explicitly un-ignored (`!` rules).
   Keep that in mind when adding new packages.
7. 3Dmol.js notes: `addStyle` merges styles and a base `colorscheme` beats an
   explicit `color` → use `setStyle` per atom to recolor.
   `viewer.modelToScreen()` returns **absolute page coordinates**.
   Atom click callbacks deliver `atom.index` = SDF file order.

## App architecture (brief)

Singletons + locks in `app/state.py` (dataset registry, active model,
single training job, explanation jobs + LRU cache). Training runs in a
daemon thread, polled via `/api/train/status`; explanations run as jobs
polled via `/api/explain/jobs/<id>`. A model checkpoint carries its own
normalization stats; predictions/explanations use a dataset "view" built
with those stats (`dataset_service.make_normalized_view`, no pickle re-read).
Training can target any pkl in `data/` via the `dataset_file` parameter
(loaded into the registry without switching the Explore view's active
dataset; split into train/val/test by compound with the configured ratio).