/* Train & Models view: hyperparameter form, live status + chart, checkpoints. */
"use strict";

const Train = {
  chart: null,

  init() {
    document.getElementById("train-form").addEventListener("submit", (e) => {
      e.preventDefault();
      Train.start();
    });
    document.getElementById("btn-train-stop").addEventListener("click", () => Train.stop());
    Train.buildChart();
    Train.pollOnce();
    Train.loadModels();
    Train.loadDatasetOptions();
  },

  /* Populate the dataset dropdown; keeps the user's selection on refresh. */
  async loadDatasetOptions() {
    let data;
    try {
      data = await Api.getJSON("/api/datasets");
    } catch (err) {
      return;
    }
    const select = document.getElementById("train-dataset-select");
    const previous = select.value;
    select.innerHTML = "";
    for (const ds of data.datasets) {
      const option = document.createElement("option");
      option.value = ds.file_name;
      option.textContent = ds.num_graphs
        ? `${ds.file_name} (${ds.num_graphs} graphs)`
        : ds.file_name;
      select.appendChild(option);
    }
    select.value =
      previous && [...select.options].some((o) => o.value === previous)
        ? previous
        : data.active || (select.options[0] && select.options[0].value) || "";
  },

  collectConfig() {
    const form = document.getElementById("train-form");
    const data = new FormData(form);
    const cfg = {};
    for (const [key, value] of data.entries()) {
      if (key === "model_name" || key === "operator_type" || key === "dataset_file") cfg[key] = value;
      else if (key === "split_ratio") {
        cfg[key] = value
          .split(",")
          .map((part) => parseFloat(part.trim()))
          .filter((x) => !Number.isNaN(x));
      } else cfg[key] = parseFloat(value);
    }
    return cfg;
  },

  async start() {
    try {
      await Api.postJSON("/api/train", Train.collectConfig());
      showToast("Training started");
      Train.startPolling();
    } catch (err) {
      showToast(err.message, "error");
    }
  },

  async stop() {
    try {
      await Api.postJSON("/api/train/stop", {});
      showToast("Stop requested — finishing current batch…");
    } catch (err) {
      showToast(err.message, "error");
    }
  },

  startPolling() {
    if (AppState.trainPolling) return;
    AppState.trainPolling = setInterval(() => Train.pollOnce(), 1000);
    Train.pollOnce();
  },

  stopPolling() {
    if (AppState.trainPolling) {
      clearInterval(AppState.trainPolling);
      AppState.trainPolling = null;
    }
  },

  async pollOnce() {
    let job;
    try {
      job = await Api.getJSON("/api/train/status");
    } catch (err) {
      return;
    }
    const previous = AppState.lastTrainStatus;
    AppState.lastTrainStatus = job;
    Train.renderStatus(job);
    Train.updateChart(job);

    if (job.status === "running" || job.status === "stopping") {
      Train.startPolling();
    } else {
      Train.stopPolling();
      if (previous && (previous.status === "running" || previous.status === "stopping")) {
        if (job.status === "done") {
          showToast(`Training finished — checkpoint ${job.checkpoint_id} saved`);
          Train.loadModels();
        } else if (job.status === "error") {
          showToast(`Training failed: ${job.error}`, "error");
        }
      }
    }
  },

  renderStatus(job) {
    const pill = document.getElementById("train-pill");
    const summary = document.getElementById("train-summary");
    const progress = document.getElementById("train-progress");
    const metrics = document.getElementById("train-metrics");
    const startBtn = document.getElementById("btn-train-start");
    const stopBtn = document.getElementById("btn-train-stop");

    const status = job.status || "idle";
    pill.className = `pill ${status}`;
    pill.textContent = status;

    const running = status === "running" || status === "stopping";
    startBtn.disabled = running;
    stopBtn.disabled = !running;

    if (status === "idle") {
      summary.textContent = "No training has run in this session.";
      progress.style.width = "0%";
      metrics.innerHTML = "";
      return;
    }

    const parts = [`${job.model_name || ""}`, `epoch ${job.epoch}/${job.num_epochs}`];
    if (job.config && job.config.dataset_file) parts.push(job.config.dataset_file);
    if (job.elapsed_s !== undefined) parts.push(`${Math.round(job.elapsed_s)}s`);
    if (job.val_fallback) parts.push(`val = ${job.val_fallback} split`);
    if (job.status === "error") parts.push(job.error || "");
    if (job.checkpoint_id) parts.push(`checkpoint: ${job.checkpoint_id}`);
    summary.textContent = parts.filter(Boolean).join(" · ");

    progress.style.width = job.num_epochs
      ? `${Math.round((100 * job.epoch) / job.num_epochs)}%`
      : "0%";

    const last = job.history && job.history.length ? job.history[job.history.length - 1] : null;
    if (last) {
      metrics.innerHTML = `
        <div class="m"><span>train MAE H </span><b>${fmt(last.train_mae_H)}</b></div>
        <div class="m"><span>val MAE H </span><b>${fmt(last.val_mae_H)}</b></div>
        <div class="m"><span>train MAE C </span><b>${fmt(last.train_mae_C)}</b></div>
        <div class="m"><span>val MAE C </span><b>${fmt(last.val_mae_C)}</b></div>
        <div class="m"><span>best val score </span><b>${fmt(job.best_val_score)}${
          job.best_epoch ? ` <span class="note">(epoch ${job.best_epoch})</span>` : ""
        }</b></div>`;
    } else {
      metrics.innerHTML = '<div class="note">Waiting for the first epoch…</div>';
    }
  },

  buildChart() {
    const ctx = document.getElementById("loss-chart").getContext("2d");
    const mkSet = (label, color, dash, axis) => ({
      label,
      data: [],
      borderColor: color,
      backgroundColor: color,
      borderDash: dash,
      borderWidth: 2,
      pointRadius: 2,
      pointHoverRadius: 4,
      tension: 0.25,
      yAxisID: axis,
    });
    this.chart = new Chart(ctx, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          mkSet("train MAE H", "#004e9f", [], "yH"),
          mkSet("val MAE H", "#004e9f", [6, 4], "yH"),
          mkSet("train MAE C", "#fcba00", [], "yC"),
          mkSet("val MAE C", "#fcba00", [6, 4], "yC"),
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: "index", intersect: false },
        scales: {
          x: { title: { display: true, text: "epoch" }, grid: { color: "#f0efec" } },
          yH: {
            position: "left",
            title: { display: true, text: "MAE H (ppm)", color: "#004e9f" },
            grid: { color: "#f0efec" },
          },
          yC: {
            position: "right",
            title: { display: true, text: "MAE C (ppm)", color: "#bb8a00" },
            grid: { drawOnChartArea: false },
          },
        },
        plugins: { legend: { labels: { boxWidth: 16, font: { size: 11 } } } },
      },
    });
  },

  updateChart(job) {
    if (!this.chart) return;
    const history = job.history || [];
    this.chart.data.labels = history.map((h) => h.epoch);
    this.chart.data.datasets[0].data = history.map((h) => h.train_mae_H);
    this.chart.data.datasets[1].data = history.map((h) => h.val_mae_H);
    this.chart.data.datasets[2].data = history.map((h) => h.train_mae_C);
    this.chart.data.datasets[3].data = history.map((h) => h.val_mae_C);
    this.chart.update("none");
  },

  async loadModels() {
    const container = document.getElementById("models-table");
    let data;
    try {
      data = await Api.getJSON("/api/models");
    } catch (err) {
      container.innerHTML = `<div class="note">Could not load models: ${escapeHtml(err.message)}</div>`;
      return;
    }
    if (!data.models.length) {
      container.innerHTML =
        '<div class="empty">No saved models yet — train one with the form on the left.</div>';
      return;
    }
    container.innerHTML = `
      <table class="pred-table">
        <thead><tr>
          <th>Model</th><th>Operator</th><th>val MAE H</th><th>val MAE C</th>
          <th>test MAE H</th><th>test MAE C</th><th></th>
        </tr></thead>
        <tbody>
          ${data.models
            .map(
              (m) => `<tr>
                <td>${m.active ? '<span class="active-dot"></span>' : ""}${escapeHtml(m.display_name)}</td>
                <td>${escapeHtml(m.operator_type || "")}</td>
                <td>${fmt(m.best_val_mae_H)}</td>
                <td>${fmt(m.best_val_mae_C)}</td>
                <td>${fmt(m.test_mae_H)}</td>
                <td>${fmt(m.test_mae_C)}</td>
                <td>${
                  m.active
                    ? '<span class="note">active</span>'
                    : `<button class="btn btn-ghost" data-model="${escapeHtml(m.model_id)}">Load</button>`
                }</td>
              </tr>`
            )
            .join("")}
        </tbody>
      </table>`;

    container.querySelectorAll("button[data-model]").forEach((button) => {
      button.addEventListener("click", () => Train.activateModel(button.dataset.model));
    });
  },

  async activateModel(modelId) {
    try {
      const result = await Api.postJSON(`/api/models/${encodeURIComponent(modelId)}/load`);
      AppState.activeModel = result.model;
      setModelChip(result.model, AppState.device);
      showToast(`Model loaded: ${result.model.display_name}`);
      Train.loadModels();
      Explore.refreshPredictions();
    } catch (err) {
      showToast(err.message, "error");
    }
  },
};
