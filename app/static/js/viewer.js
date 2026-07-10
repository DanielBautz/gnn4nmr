/* Thin wrapper around 3Dmol.js — the only module touching the 3D canvas.
   Atom indices everywhere are SDF file order (= backend atom_index). */
"use strict";

const Viewer = {
  viewer: null,
  coords: [],
  selectedIdx: null,
  overlay: null, // {neighbors: [{atom_index, importance}], bonds: [{a1,a2,importance}]|null}
  labelHandles: [],
  onClick: null,

  baseStyle: {
    stick: { radius: 0.15, colorscheme: "Jmol" },
    sphere: { scale: 0.25, colorscheme: "Jmol" },
  },

  ensure() {
    if (!this.viewer) {
      this.viewer = $3Dmol.createViewer(document.getElementById("viewer3d"), {
        backgroundColor: "#f7f7f5",
        antialias: true,
      });
    }
    return this.viewer;
  },

  load(sdfText, onClick) {
    const v = this.ensure();
    this.onClick = onClick;
    this.selectedIdx = null;
    this.overlay = null;
    this.labelHandles = [];
    v.removeAllLabels();
    v.removeAllShapes();
    v.removeAllModels();

    const model = v.addModel(sdfText, "sdf");
    this.coords = model.selectedAtoms({}).map((a) => ({ x: a.x, y: a.y, z: a.z }));

    v.setClickable({}, true, (atom) => {
      if (this.onClick && atom && atom.index !== undefined) this.onClick(atom.index);
    });

    this.restyle();
    v.zoomTo();
    v.render();
  },

  clear() {
    if (!this.viewer) return;
    this.viewer.removeAllLabels();
    this.viewer.removeAllShapes();
    this.viewer.removeAllModels();
    this.viewer.render();
    this.coords = [];
    this.selectedIdx = null;
    this.overlay = null;
    this.labelHandles = [];
  },

  select(atomIndex) {
    this.selectedIdx = atomIndex;
    this.restyle();
  },

  setOverlay(overlay) {
    this.overlay = overlay;
    this.restyle();
  },

  clearOverlay() {
    if (this.overlay) {
      this.overlay = null;
      this.restyle();
    }
  },

  restyle() {
    const v = this.viewer;
    if (!v || !this.coords.length) return;
    v.setStyle({}, this.baseStyle);
    v.removeAllShapes();

    // setStyle (not addStyle) for recolored atoms: the base "colorscheme"
    // would otherwise win over an explicit "color" after a style merge.
    if (this.overlay) {
      const neighbors = this.overlay.neighbors || [];
      const bonds = this.overlay.bonds || [];
      // Importance per atom index — drives both the atom color and the
      // two-tone bond split. Atoms absent here are outside the computational
      // graph (out of range).
      const impByAtom = new Map(neighbors.map((n) => [n.atom_index, n.importance]));
      // Color for a bond endpoint: the explained atom is yellow, in-range
      // atoms follow the blue -> red heatmap, anything else is grey.
      const endColor = (idx) =>
        idx === this.selectedIdx
          ? SELECTED_YELLOW
          : impByAtom.has(idx)
          ? heatColor(impByAtom.get(idx))
          : OUTSIDE_GREY;
      // Split weight: the explained atom always predominates over its bonds.
      const endWeight = (idx) =>
        idx === this.selectedIdx ? 1 : impByAtom.get(idx) ?? 0;

      // Atoms outside the computational graph (and their stick halves) stay
      // muted grey; in-range atoms are recolored by importance below.
      v.setStyle(
        {},
        {
          stick: { radius: this.baseStyle.stick.radius, color: OUTSIDE_GREY },
          sphere: { scale: this.baseStyle.sphere.scale, color: OUTSIDE_GREY },
        }
      );
      for (const n of neighbors) {
        const c = heatColor(n.importance);
        // Same size as every other atom — importance is shown by color only.
        v.setStyle(
          { index: n.atom_index },
          {
            stick: { radius: this.baseStyle.stick.radius, color: c },
            sphere: { scale: this.baseStyle.sphere.scale, color: c },
          }
        );
      }
      // Each bond half takes its adjacent atom's color. The graph edges are
      // undirected, so the split point is biased toward the more important
      // endpoint — its color predominates (50/50 only when equally important).
      for (const b of bonds) {
        const start = this.coords[b.a1];
        const end = this.coords[b.a2];
        if (!start || !end) continue;
        const wA = endWeight(b.a1);
        const wB = endWeight(b.a2);
        const fA = wA + wB > 0 ? wA / (wA + wB) : 0.5;
        const mid = {
          x: start.x + (end.x - start.x) * fA,
          y: start.y + (end.y - start.y) * fA,
          z: start.z + (end.z - start.z) * fA,
        };
        v.addCylinder({ start, end: mid, radius: 0.16, color: endColor(b.a1), fromCap: 1, toCap: 0 });
        v.addCylinder({ start: mid, end, radius: 0.16, color: endColor(b.a2), fromCap: 0, toCap: 1 });
      }
    }

    if (this.selectedIdx !== null && this.selectedIdx !== undefined) {
      // Same size as every other atom — selection is shown by color only.
      v.setStyle(
        { index: this.selectedIdx },
        {
          stick: { radius: this.baseStyle.stick.radius, color: SELECTED_YELLOW },
          sphere: { scale: this.baseStyle.sphere.scale, color: SELECTED_YELLOW },
        }
      );
    }
    v.addStyle({}, { clicksphere: { radius: 0.45 } });
    v.render();
  },

  setLabels(rows) {
    this.clearLabels();
    const v = this.viewer;
    if (!v) return;
    for (const row of rows) {
      const pos = this.coords[row.atom_index];
      if (!pos) continue;
      this.labelHandles.push(
        v.addLabel(row.text, {
          position: pos,
          fontSize: 11,
          fontColor: "#004e9f",
          backgroundColor: "white",
          backgroundOpacity: 0.8,
          borderColor: "#909085",
          borderThickness: 0.4,
          inFront: true,
        })
      );
    }
    v.render();
  },

  clearLabels() {
    if (!this.viewer || !this.labelHandles.length) return;
    for (const handle of this.labelHandles) this.viewer.removeLabel(handle);
    this.labelHandles = [];
    this.viewer.render();
  },

  resetZoom() {
    if (!this.viewer) return;
    this.viewer.zoomTo();
    this.viewer.render();
  },

  resize() {
    if (this.viewer) this.viewer.resize();
  },
};

// Continuous structure-importance heatmap: a direct blue -> red RGB blend that
// passes through purple at the midpoint. Low importance = blue, mid = purple,
// high = red.
const HEAT_LOW = "#1414ff";
const HEAT_HIGH = "#ff1414";
// Explained/selected atom (yellow) and atoms outside the computational graph.
const SELECTED_YELLOW = "#f0e400";
const OUTSIDE_GREY = "#b8b6b0";

function heatColor(t) {
  return lerpHex(HEAT_LOW, HEAT_HIGH, t);
}

function lerpHex(fromHex, toHex, t) {
  const f = parseInt(fromHex.slice(1), 16);
  const to = parseInt(toHex.slice(1), 16);
  const channel = (shift) => {
    const a = (f >> shift) & 0xff;
    const b = (to >> shift) & 0xff;
    return Math.round(a + (b - a) * Math.max(0, Math.min(1, t)));
  };
  const rgb = (channel(16) << 16) | (channel(8) << 8) | channel(0);
  return `#${rgb.toString(16).padStart(6, "0")}`;
}
