/* Explore view: molecule list, atom detail panel, explanations. */
"use strict";

const Explore = {
  init() {
    document.getElementById("molecule-search").addEventListener("input", (e) => {
      AppState.searchTerm = e.target.value.trim().toLowerCase();
      Explore.renderMoleculeList();
    });
    document.getElementById("btn-reset-view").addEventListener("click", () => Viewer.resetZoom());
    document.getElementById("toggle-labels").addEventListener("change", (e) => {
      AppState.labelsOn = e.target.checked;
      Explore.syncLabels();
    });
    document.getElementById("dataset-select").addEventListener("change", async (e) => {
      try {
        const result = await Api.postJSON("/api/datasets/select", { file_name: e.target.value });
        showToast(`Dataset loaded: ${result.file_name} (${result.num_graphs} graphs)`);
        Explore.resetMoleculeState();
        await Explore.loadGraphs();
        await Explore.loadDatasets();
      } catch (err) {
        showToast(err.message, "error");
        await Explore.loadDatasets(); // restore selection
      }
    });
  },

  resetMoleculeState() {
    AppState.currentGraphIdx = null;
    AppState.molecule = null;
    AppState.atomsByIndex = {};
    AppState.predictions = null;
    AppState.predByAtom = {};
    AppState.selectedAtomIndex = null;
    AppState.explanation = null;
    Explore.stopExplainPolling();
    Viewer.clear();
    document.getElementById("molecule-title").textContent = "No molecule selected";
    document.getElementById("viewer-overlay").style.display = "flex";
    document.getElementById("viewer-overlay").innerHTML = "<span>Select a molecule from the list</span>";
    document.getElementById("viewer-hint").style.display = "none";
    document.getElementById("detail-panel").innerHTML =
      '<div class="empty">Select a molecule to see details</div>';
  },

  async loadDatasets() {
    const data = await Api.getJSON("/api/datasets");
    const select = document.getElementById("dataset-select");
    select.innerHTML = "";
    for (const ds of data.datasets) {
      const option = document.createElement("option");
      option.value = ds.file_name;
      option.textContent = ds.num_graphs
        ? `${ds.file_name} (${ds.num_graphs})`
        : ds.file_name;
      if (ds.active) option.selected = true;
      select.appendChild(option);
    }
  },

  async loadGraphs() {
    try {
      const data = await Api.getJSON("/api/graphs");
      AppState.graphs = data.graphs;
    } catch (err) {
      AppState.graphs = [];
      showToast(err.message, "error");
    }
    Explore.renderMoleculeList();
  },

  renderMoleculeList() {
    const container = document.getElementById("molecule-list");
    const term = AppState.searchTerm;
    const filtered = term
      ? AppState.graphs.filter(
          (g) =>
            g.label.toLowerCase().includes(term) ||
            (g.formula || "").toLowerCase().includes(term) ||
            g.compound.toLowerCase().includes(term)
        )
      : AppState.graphs;

    container.innerHTML = "";
    const fragment = document.createDocumentFragment();
    for (const g of filtered) {
      const item = document.createElement("button");
      item.type = "button";
      item.className = "molecule-item" + (g.graph_idx === AppState.currentGraphIdx ? " active" : "");
      item.innerHTML = `<span class="molecule-label">${escapeHtml(g.label)}</span>
        <span class="molecule-meta">${escapeHtml(g.formula)} · ${g.num_atoms}</span>`;
      item.addEventListener("click", () => Explore.selectMolecule(g.graph_idx));
      fragment.appendChild(item);
    }
    container.appendChild(fragment);
    document.getElementById("molecule-count").textContent =
      `${filtered.length} of ${AppState.graphs.length} structures`;
  },

  async selectMolecule(graphIdx) {
    if (AppState.currentGraphIdx === graphIdx) return;
    AppState.currentGraphIdx = graphIdx;
    AppState.selectedAtomIndex = null;
    AppState.explanation = null;
    Explore.stopExplainPolling();
    Explore.renderMoleculeList();

    const overlay = document.getElementById("viewer-overlay");
    overlay.style.display = "flex";
    overlay.innerHTML = '<div class="spinner big"></div><span>Loading molecule…</span>';

    try {
      const [molecule, predictions] = await Promise.all([
        Api.getJSON(`/api/graphs/${graphIdx}`),
        Explore.fetchPredictions(graphIdx),
      ]);
      if (AppState.currentGraphIdx !== graphIdx) return; // user clicked elsewhere
      AppState.molecule = molecule;
      AppState.atomsByIndex = {};
      for (const atom of molecule.atoms) AppState.atomsByIndex[atom.atom_index] = atom;
      Explore.applyPredictions(predictions);

      document.getElementById("molecule-title").innerHTML =
        `${escapeHtml(molecule.label)}<span class="sub">${escapeHtml(molecule.formula)} · compound ${escapeHtml(molecule.compound)}</span>`;

      if (molecule.sdf) {
        overlay.style.display = "none";
        document.getElementById("viewer-hint").style.display = "block";
        Viewer.load(molecule.sdf, (atomIndex) => Explore.onAtomClick(atomIndex));
        Explore.syncLabels();
      } else {
        Viewer.clear();
        document.getElementById("viewer-hint").style.display = "none";
        overlay.innerHTML = "<span>No 3D structure (SDF) found for this graph</span>";
      }
      Explore.renderMoleculePanel();
    } catch (err) {
      overlay.innerHTML = `<span>Failed to load molecule</span>`;
      showToast(err.message, "error");
    }
  },

  async fetchPredictions(graphIdx) {
    try {
      return await Api.getJSON(`/api/graphs/${graphIdx}/predictions`);
    } catch (err) {
      if (err.code === "no_active_model") return null;
      showToast(err.message, "error");
      return null;
    }
  },

  applyPredictions(predictions) {
    AppState.predictions = predictions;
    AppState.predByAtom = {};
    if (predictions) {
      for (const ntype of ["H", "C"]) {
        for (const row of predictions.predictions[ntype] || []) {
          AppState.predByAtom[row.atom_index] = row;
        }
      }
    }
  },

  /* Re-fetch predictions after a model was loaded/changed. */
  async refreshPredictions() {
    if (AppState.currentGraphIdx === null) return;
    const predictions = await Explore.fetchPredictions(AppState.currentGraphIdx);
    Explore.applyPredictions(predictions);
    AppState.explanation = null;
    Viewer.clearOverlay();
    Explore.syncLabels();
    if (AppState.selectedAtomIndex !== null) Explore.renderAtomPanel();
    else Explore.renderMoleculePanel();
  },

  syncLabels() {
    if (!AppState.labelsOn || !AppState.predictions || !AppState.molecule || !AppState.molecule.sdf) {
      Viewer.clearLabels();
      return;
    }
    const rows = [];
    for (const ntype of ["H", "C"]) {
      for (const row of AppState.predictions.predictions[ntype] || []) {
        rows.push({ atom_index: row.atom_index, text: fmt(row.pred, 2) });
      }
    }
    Viewer.setLabels(rows);
  },

  onAtomClick(atomIndex) {
    const atom = AppState.atomsByIndex[atomIndex];
    if (!atom) return;
    AppState.selectedAtomIndex = atomIndex;
    AppState.explanation = null;
    Explore.stopExplainPolling();
    Viewer.clearOverlay();
    Viewer.select(atomIndex);
    Explore.renderAtomPanel();
  },

  /* ------------------------------------------------------------ panels */
  renderMoleculePanel() {
    const panel = document.getElementById("detail-panel");
    const molecule = AppState.molecule;
    if (!molecule) {
      panel.innerHTML = '<div class="empty">Select a molecule to see details</div>';
      return;
    }
    const predictions = AppState.predictions;
    let modelBlock;
    if (!predictions) {
      modelBlock = `<div class="empty">No model loaded — train or load one in <b>Train &amp; Models</b> to see predictions.</div>`;
    } else {
      const s = predictions.summary;
      const rows = [];
      for (const ntype of ["H", "C"]) {
        for (const row of predictions.predictions[ntype] || []) {
          rows.push({ ...row, ntype });
        }
      }
      rows.sort((a, b) => a.atom_index - b.atom_index);
      modelBlock = `
        <div class="panel-section">
          <h3>Predictions <span class="note">(${escapeHtml(predictions.model_id)})</span></h3>
          <div class="metrics-row" style="margin:0 0 8px">
            <div class="m"><span>MAE H </span><b>${fmt(s.mae_H)}</b></div>
            <div class="m"><span>MAE C </span><b>${fmt(s.mae_C)}</b></div>
          </div>
          <table class="pred-table">
            <thead><tr><th>Atom</th><th>Pred</th><th>Truth</th><th>|Δ|</th></tr></thead>
            <tbody>
              ${rows
                .map(
                  (r) => `<tr class="clickable" data-atom="${r.atom_index}">
                    <td>${escapeHtml(r.ntype)}${r.atom_index}</td>
                    <td>${fmt(r.pred, 2)}</td>
                    <td>${fmt(r.ground_truth, 2)}</td>
                    <td>${fmt(r.abs_error, 2)}</td>
                  </tr>`
                )
                .join("")}
            </tbody>
          </table>
        </div>`;
    }

    panel.innerHTML = `
      <div class="panel-section">
        <h3>Molecule</h3>
        <dl class="kv">
          <dt>Structure</dt><dd>${escapeHtml(molecule.label)}</dd>
          <dt>Formula</dt><dd>${escapeHtml(molecule.formula)}</dd>
          <dt>Atoms</dt><dd>${molecule.num_atoms} (${molecule.n_H} H, ${molecule.n_C} C)</dd>
        </dl>
      </div>
      ${modelBlock}
      <div class="note">Click an atom in the 3D view (or a table row) to inspect features and explanations.</div>
    `;

    panel.querySelectorAll("tr.clickable").forEach((tr) => {
      tr.addEventListener("click", () => Explore.onAtomClick(parseInt(tr.dataset.atom, 10)));
    });
  },

  renderAtomPanel() {
    const panel = document.getElementById("detail-panel");
    const atom = AppState.atomsByIndex[AppState.selectedAtomIndex];
    if (!atom) {
      Explore.renderMoleculePanel();
      return;
    }
    const predRow = AppState.predByAtom[atom.atom_index];
    const isPredicted = atom.node_type === "H" || atom.node_type === "C";

    const shiftBlock = isPredicted
      ? `<div class="shift-numbers">
          <div class="shift-block">
            <div class="label">Predicted shift</div>
            <div class="value">${predRow ? fmt(predRow.pred, 2) : '<span class="muted">no model</span>'}<span class="unit">${predRow ? "ppm" : ""}</span></div>
          </div>
          <div class="shift-block">
            <div class="label">Ground truth</div>
            <div class="value gt">${atom.ground_truth !== null ? fmt(atom.ground_truth, 2) : '<span class="muted">n/a</span>'}<span class="unit">${atom.ground_truth !== null ? "ppm" : ""}</span></div>
          </div>
          <div class="shift-block">
            <div class="label">|Error|</div>
            <div class="value gt">${predRow && predRow.abs_error !== null ? fmt(predRow.abs_error, 2) : "–"}</div>
          </div>
        </div>`
      : `<div class="note" style="margin-top:8px">No shift prediction for this atom type — the model predicts H and C only.</div>`;

    const explanationBlock = isPredicted
      ? `<div class="panel-section">
          <h3>Explanation</h3>
          ${Explore.explainControlsHtml(atom)}
          <div id="explain-status"></div>
        </div>
        <div id="explanation-results"></div>`
      : "";

    panel.innerHTML = `
      <div class="panel-section">
        <button class="show-more" id="btn-back-molecule">← Molecule overview</button>
        <div class="atom-card" style="margin-top:6px">
          <div class="element-badge type-${atom.node_type}">${escapeHtml(atom.element)}</div>
          <div>
            <div class="atom-title">${escapeHtml(atom.element)} · atom ${atom.atom_index}</div>
            <div class="atom-sub">node type ${atom.node_type} · local index ${atom.type_local_idx}</div>
          </div>
        </div>
        ${shiftBlock}
      </div>
      ${explanationBlock}
      <div class="panel-section" id="feature-values-section">
        <h3>Feature values</h3>
        <div id="feature-values"></div>
      </div>
    `;

    document.getElementById("btn-back-molecule").addEventListener("click", () => {
      AppState.selectedAtomIndex = null;
      AppState.explanation = null;
      Explore.stopExplainPolling();
      Viewer.select(null);
      Viewer.clearOverlay();
      Explore.renderMoleculePanel();
    });

    if (isPredicted) {
      Explore.wireExplainControls(atom);
      if (AppState.explanation && AppState.explanation.atom_index === atom.atom_index) {
        Explore.renderExplanationResults(AppState.explanation);
      }
    }
    Explore.renderFeatureValues(atom);
  },

  renderFeatureValues(atom) {
    const target = document.getElementById("feature-values");
    if (!target) return;
    const names = AppState.molecule.feature_names[atom.node_type] || [];
    const rows = names
      .map(
        (name, i) => `<div class="bar-row" style="grid-template-columns: 1fr 80px">
          <span class="bar-name" title="${escapeHtml(name)}">${escapeHtml(name)}</span>
          <span class="bar-value">${fmt(atom.feature_values[i])}</span>
        </div>`
      )
      .join("");
    target.innerHTML = `<div style="max-height:200px;overflow-y:auto">${rows}</div>
      <div class="note" style="margin-top:4px">Raw (unnormalized) input features of this atom.</div>`;
  },

  /* ------------------------------------------------------- explanations */
  explainControlsHtml(atom) {
    const hasGT = atom.ground_truth !== null;
    return `
      <div class="explain-controls">
        <div class="explain-row">
          <label for="explain-method">Method</label>
          <select id="explain-method">
            <option value="gnn_explainer">GNNExplainer</option>
            <option value="integrated_gradients">Integrated Gradients</option>
          </select>
        </div>
        <div class="params-grid" id="explain-params"></div>
        <button class="btn" id="btn-explain" ${AppState.predictions ? "" : "disabled"}>Explain this atom</button>
        ${AppState.predictions ? "" : '<div class="note">Load a model first to compute explanations.</div>'}
        ${hasGT ? "" : '<div class="note">No ground truth for this atom — only "model" explanations are possible.</div>'}
      </div>`;
  },

  paramsHtml(method, hasGT) {
    const expl = `
      <label>Explanation target
        <select data-param="explanation_type">
          <option value="model">model (prediction)</option>
          <option value="phenomenon" ${hasGT ? "" : "disabled"}>phenomenon (ground truth)</option>
        </select>
      </label>`;
    if (method === "gnn_explainer") {
      return `
        <label>Epochs <input type="number" data-param="epochs" value="100" min="10" max="500"></label>
        <label>Neighborhood (hops) <input type="number" data-param="k_hops" value="2" min="1" max="4"></label>
        ${expl}
        <div class="params-subhead">Regularization (λ)</div>
        <label title="Edge-mask sparsity — higher keeps fewer edges (more compact subgraph). Effect saturates by ~5.">Edge sparsity <input type="number" data-param="edge_size" value="0.005" min="0" max="100" step="0.001"></label>
        <label title="Node-feature-mask sparsity — averaged over all features, so its effect stays mild even at high values">Node Feature sparsity <input type="number" data-param="node_feat_size" value="1" min="0" max="100" step="0.1"></label>
        <label title="Edge-mask entropy — higher pushes edge-mask values toward a crisp 0/1 selection">Edge entropy <input type="number" data-param="edge_ent" value="1" min="0" max="100" step="0.1"></label>
        <label title="Node-feature-mask entropy — higher pushes feature-mask values toward a crisp 0/1 selection">Node feature entropy <input type="number" data-param="node_feat_ent" value="0.1" min="0" max="100" step="0.05"></label>`;
    }
    return `
      <label>IG steps <input type="number" data-param="n_steps" value="50" min="8" max="128"></label>
      <label>Baseline
        <select data-param="baseline_type">
          <option value="zero">zero</option>
          <option value="mean">mean</option>
          <option value="min">min</option>
          <option value="max">max</option>
          <option value="random">random</option>
        </select>
      </label>
      <label>Neighborhood (hops) <input type="number" data-param="k_hops" value="1" min="1" max="4"></label>
      <label class="param-check"><input type="checkbox" data-param="include_neighbors" checked> Neighbor attributions</label>
      ${expl}`;
  },

  wireExplainControls(atom) {
    const methodSelect = document.getElementById("explain-method");
    const paramsBox = document.getElementById("explain-params");
    const hasGT = atom.ground_truth !== null;
    const renderParams = () => {
      paramsBox.innerHTML = Explore.paramsHtml(methodSelect.value, hasGT);
    };
    renderParams();
    methodSelect.addEventListener("change", renderParams);
    document.getElementById("btn-explain").addEventListener("click", () => Explore.runExplanation(atom));
  },

  collectParams() {
    const params = {};
    document.querySelectorAll("#explain-params [data-param]").forEach((el) => {
      const key = el.dataset.param;
      if (el.type === "checkbox") params[key] = el.checked;
      else if (el.type === "number") params[key] = parseFloat(el.value);
      else params[key] = el.value;
    });
    return params;
  },

  async runExplanation(atom) {
    const method = document.getElementById("explain-method").value;
    const payload = {
      graph_idx: AppState.currentGraphIdx,
      node_type: atom.node_type,
      type_local_idx: atom.type_local_idx,
      method,
      params: Explore.collectParams(),
    };
    const statusBox = document.getElementById("explain-status");
    const button = document.getElementById("btn-explain");
    button.disabled = true;
    statusBox.innerHTML =
      '<div class="explain-status"><div class="spinner"></div><span>Computing explanation… <b id="explain-elapsed">0 s</b></span></div>';
    Explore.stopExplainPolling();

    try {
      const response = await Api.postJSON("/api/explain", payload);
      if (response.status === "done") {
        Explore.onExplanationDone(response.result, atom, !!response.cached);
        return;
      }
      const jobId = response.job_id;
      const started = Date.now();
      AppState.explainPolling = setInterval(async () => {
        const elapsedEl = document.getElementById("explain-elapsed");
        if (elapsedEl) elapsedEl.textContent = `${Math.round((Date.now() - started) / 1000)} s`;
        try {
          const job = await Api.getJSON(`/api/explain/jobs/${jobId}`);
          if (job.status === "done") {
            Explore.stopExplainPolling();
            Explore.onExplanationDone(job.result, atom, false);
          } else if (job.status === "error") {
            Explore.stopExplainPolling();
            Explore.onExplanationFailed(job.error);
          }
        } catch (err) {
          Explore.stopExplainPolling();
          Explore.onExplanationFailed(err.message);
        }
      }, 700);
    } catch (err) {
      Explore.onExplanationFailed(err.message);
    }
  },

  stopExplainPolling() {
    if (AppState.explainPolling) {
      clearInterval(AppState.explainPolling);
      AppState.explainPolling = null;
    }
  },

  onExplanationDone(result, atom, cached) {
    // The user may have clicked elsewhere while the job ran.
    if (AppState.currentGraphIdx !== result.graph_idx) return;
    if (AppState.selectedAtomIndex !== result.atom_index) return;
    AppState.explanation = result;
    const statusBox = document.getElementById("explain-status");
    if (statusBox) {
      statusBox.innerHTML = `<div class="note">${cached ? "Cached result · " : ""}computed in ${result.elapsed_s}s${
        result.convergence_delta !== null && result.convergence_delta !== undefined
          ? ` · convergence Δ ${fmt(result.convergence_delta)}`
          : ""
      }</div>`;
    }
    const button = document.getElementById("btn-explain");
    if (button) button.disabled = false;
    Explore.renderExplanationResults(result);
    Viewer.setOverlay({ neighbors: result.neighbors || [], bonds: result.bonds || [] });
  },

  onExplanationFailed(message) {
    const statusBox = document.getElementById("explain-status");
    if (statusBox) statusBox.innerHTML = `<div class="note" style="color:var(--danger)">${escapeHtml(message)}</div>`;
    const button = document.getElementById("btn-explain");
    if (button) button.disabled = false;
    showToast(`Explanation failed: ${message}`, "error");
  },

  renderExplanationResults(result) {
    const container = document.getElementById("explanation-results");
    if (!container) return;
    const methodLabel =
      result.method === "gnn_explainer" ? "GNNExplainer" : "Integrated Gradients";

    const featureBars = (showAll) => {
      const items = showAll ? result.own_features : result.own_features.slice(0, 10);
      return items
        .map((f) => {
          const neg = f.signed < 0;
          return `<div class="bar-row">
            <span class="bar-name" title="${escapeHtml(f.name)}">${escapeHtml(f.name)}</span>
            <div class="bar-track"><div class="bar-fill${neg ? " neg" : ""}" style="width:${Math.round(f.importance * 100)}%"></div></div>
            <span class="bar-value" title="importance ${(f.importance * 100).toFixed(0)}%">${neg ? "−" : ""}${fmt(f.raw_value)}</span>
          </div>`;
        })
        .join("");
    };

    const neighborBars = (result.neighbors || [])
      .map(
        (n) => `<div class="bar-row">
          <span class="bar-name">${escapeHtml(n.element)}${n.atom_index}<span class="hops">${n.hops ? `${n.hops} hop${n.hops > 1 ? "s" : ""}` : ""}</span></span>
          <div class="bar-track"><div class="bar-fill accent" style="width:${Math.round(n.importance * 100)}%"></div></div>
          <span class="bar-value">${(n.importance * 100).toFixed(0)}%</span>
        </div>`
      )
      .join("");

    const bondBars =
      result.bonds === null || result.bonds === undefined
        ? ""
        : result.bonds
            .slice(0, 10)
            .map((b) => {
              const e1 = AppState.atomsByIndex[b.a1];
              const e2 = AppState.atomsByIndex[b.a2];
              return `<div class="bar-row">
                <span class="bar-name">${escapeHtml(e1 ? e1.element : "?")}${b.a1}–${escapeHtml(e2 ? e2.element : "?")}${b.a2}</span>
                <div class="bar-track"><div class="bar-fill accent" style="width:${Math.round(b.importance * 100)}%"></div></div>
                <span class="bar-value">${(b.importance * 100).toFixed(0)}%</span>
              </div>`;
            })
            .join("");

    container.innerHTML = `
      <div class="panel-section">
        <h3>Own features <span class="method-tag">${methodLabel}</span></h3>
        <div id="feature-bars">${featureBars(false)}</div>
        ${
          result.own_features.length > 10
            ? `<button class="show-more" id="btn-show-all-features">Show all ${result.own_features.length} features</button>`
            : ""
        }
        <div class="note">Bar = relative importance · number = raw feature value${
          result.method === "integrated_gradients" ? " · gray bar = negative attribution" : ""
        }</div>
      </div>
      <div class="panel-section">
        <h3>Neighbor atoms</h3>
        ${neighborBars || '<div class="note">No neighbor attributions available.</div>'}
      </div>
      ${
        result.bonds !== null && result.bonds !== undefined
          ? `<div class="panel-section"><h3>Bonds</h3>${bondBars || '<div class="note">No bond importances.</div>'}</div>`
          : `<div class="panel-section"><h3>Bonds</h3><div class="note">Integrated Gradients attributes node features only — switch to GNNExplainer for bond importances.</div></div>`
      }
    `;

    const showAllBtn = document.getElementById("btn-show-all-features");
    if (showAllBtn) {
      showAllBtn.addEventListener("click", () => {
        document.getElementById("feature-bars").innerHTML = featureBars(true);
        showAllBtn.remove();
      });
    }
  },
};
