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
      for (const n of this.overlay.neighbors || []) {
        v.setStyle(
          { index: n.atom_index },
          {
            stick: this.baseStyle.stick,
            sphere: {
              // Same size as every other atom — importance is shown by color only.
              scale: this.baseStyle.sphere.scale,
              color: lerpHex("#e8e6e1", "#fcba00", n.importance),
            },
          }
        );
      }
      for (const b of this.overlay.bonds || []) {
        const start = this.coords[b.a1];
        const end = this.coords[b.a2];
        if (!start || !end) continue;
        v.addCylinder({
          start,
          end,
          radius: 0.06 + 0.1 * b.importance,
          color: "#fcba00",
          opacity: 0.35 + 0.6 * b.importance,
          fromCap: 1,
          toCap: 1,
        });
      }
    }

    if (this.selectedIdx !== null && this.selectedIdx !== undefined) {
      // Same size as every other atom — selection is shown by color only.
      v.setStyle(
        { index: this.selectedIdx },
        {
          stick: this.baseStyle.stick,
          sphere: { scale: this.baseStyle.sphere.scale, color: "#004e9f" },
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
