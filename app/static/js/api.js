/* Fetch helpers + toast notifications. */
"use strict";

const Api = {
  async getJSON(url) {
    const response = await fetch(url);
    return Api._handle(response);
  },

  async postJSON(url, body) {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    return Api._handle(response);
  },

  async _handle(response) {
    let payload = null;
    try {
      payload = await response.json();
    } catch (err) {
      /* non-JSON response */
    }
    if (!response.ok) {
      const error = new Error((payload && payload.message) || `HTTP ${response.status}`);
      error.code = payload && payload.error;
      error.status = response.status;
      throw error;
    }
    return payload;
  },
};

function showToast(message, type = "info", timeout = 5000) {
  const container = document.getElementById("toasts");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), timeout);
}

/* Compact number formatting for shifts/feature values. */
function fmt(value, digits) {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  if (typeof value !== "number") return String(value);
  if (digits !== undefined) return value.toFixed(digits);
  const abs = Math.abs(value);
  if (abs === 0) return "0";
  if (abs >= 100) return value.toFixed(1);
  if (abs >= 1) return value.toFixed(2);
  if (abs >= 0.001) return value.toPrecision(3);
  return value.toExponential(1);
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text == null ? "" : String(text);
  return div.innerHTML;
}
