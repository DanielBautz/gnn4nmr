/* Client-side store shared by the explore/train modules. */
"use strict";

const AppState = {
  device: null,

  // explore
  graphs: [],
  searchTerm: "",
  currentGraphIdx: null,
  molecule: null,          // /api/graphs/<i> payload
  atomsByIndex: {},        // atom_index -> atom object
  predictions: null,       // /api/graphs/<i>/predictions payload or null
  predByAtom: {},          // atom_index -> prediction row
  selectedAtomIndex: null,
  explanation: null,       // last explanation result for the selected atom
  explainPolling: null,    // interval handle
  labelsOn: false,

  // models / training
  activeModel: null,
  trainPolling: null,
  lastTrainStatus: null,
};

function setModelChip(model, device) {
  const chip = document.getElementById("model-chip");
  const text = document.getElementById("model-chip-text");
  if (model) {
    chip.classList.add("has-model");
    text.textContent = `${model.display_name || model.model_id} · ${device || ""}`;
  } else {
    chip.classList.remove("has-model");
    text.textContent = "No model loaded";
  }
}
