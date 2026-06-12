/* Boot + tab switching. */
"use strict";

document.addEventListener("DOMContentLoaded", async () => {
  // Tabs
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
      const target = tab.dataset.view;
      document.getElementById("view-explore").classList.toggle("hidden", target !== "explore");
      document.getElementById("view-train").classList.toggle("hidden", target !== "train");
      if (target === "explore") Viewer.resize();
      if (target === "train") {
        Train.loadModels();
        Train.pollOnce();
        Train.loadDatasetOptions();
      }
    });
  });

  window.addEventListener("resize", () => Viewer.resize());

  Explore.init();
  Train.init();

  try {
    const health = await Api.getJSON("/api/health");
    AppState.device = health.device;
    const deviceEl = document.getElementById("train-device");
    if (deviceEl) deviceEl.textContent = health.device;
  } catch (err) {
    showToast("Backend not reachable", "error");
  }

  try {
    const active = await Api.getJSON("/api/models/active");
    AppState.activeModel = active.model;
    setModelChip(active.model, active.device);
  } catch (err) {
    /* non-fatal */
  }

  await Explore.loadDatasets();
  await Explore.loadGraphs();
});
