/**
 * Shared renderer for figure 4e (2x2 intensity correlations) and 4f
 * (aerenchyma-ratio correlation). One scatter panel per call to drawPanel.
 *
 * Species → marker shape + colour mapping is fixed across panels.
 */
(function () {

// Palette mirrors fig1b sunburst - keep species colours aligned across the paper.
//   Rice    #9ab2d4 (light blue)   |  Millet  #9393c9 (periwinkle)
//   Sorghum #514a8d (dark navy)    |  Solanum #ab8b58 (dicot tan, used for ALL
//                                     Solanum samples per user request - the
//                                     fig1b sub-species shade #5c3f4b is not
//                                     used here.)
const SPECIES_INFO = {
  "Rice":    { color: "#9ab2d4", symbol: d3.symbolCircle,    label: "Rice"    },
  "Millet":  { color: "#9393c9", symbol: d3.symbolSquare,    label: "Millet"  },
  "Sorghum": { color: "#514a8d", symbol: d3.symbolTriangle,  label: "Sorghum" },
  "Solanum": { color: "#ab8b58", symbol: d3.symbolDiamond,   label: "Solanum" },
};
const SPECIES_ORDER = ["Rice", "Millet", "Sorghum", "Solanum"];

// Mulberry32 - small, deterministic PRNG. Used to shuffle scatter points so
// species are interleaved (no single species sits entirely on top) while the
// figure remains reproducible across reloads.
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function linreg(x, y) {
  const n = x.length;
  const mx = d3.mean(x), my = d3.mean(y);
  let sxy = 0, sxx = 0;
  for (let i = 0; i < n; i++) {
    sxy += (x[i] - mx) * (y[i] - my);
    sxx += (x[i] - mx) ** 2;
  }
  const slope = sxy / sxx;
  const intercept = my - slope * mx;
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < n; i++) {
    const yp = slope * x[i] + intercept;
    ssRes += (y[i] - yp) ** 2;
    ssTot += (y[i] - my) ** 2;
  }
  return { slope, intercept, r2: 1 - ssRes / ssTot, n };
}

/**
 * Draw one scatter subpanel into `parent` (a d3 selection on a <g> or <svg>).
 *
 * @param {object} opts
 *   x0, y0, w, h      Subpanel box on the parent canvas (mm).
 *   title             Subpanel title (mm-bold, above plot).
 *   points            [{species, x, y}, ...] in data units.
 *   xLabel, yLabel    Axis titles.
 *   tickFmt           d3.format string ("d", ".2f", etc.) for tick labels.
 *   tickCount         Approximate tick count per axis (default 4).
 *   margins           Optional {top,right,bottom,left} override (mm).
 */
function drawPanel(parent, opts) {
  const {
    x0, y0, w, h, title, points,
    xLabel = "Measured", yLabel = "Predicted",
    tickFmt = "d", tickCount = 4,
    margins,
  } = opts;
  const M = Object.assign({ top: 5.5, right: 2.5, bottom: 10, left: 12 }, margins || {});
  const plotW = w - M.left - M.right;
  const plotH = h - M.top - M.bottom;

  const xVals = points.map(p => p.x);
  const yVals = points.map(p => p.y);
  const allVals = xVals.concat(yVals);
  const dataMin = d3.min(allVals);
  const dataMax = d3.max(allVals);
  const pad = (dataMax - dataMin) * 0.06;
  const lo = Math.max(0, dataMin - pad);
  const hi = dataMax + pad;

  const x = d3.scaleLinear().domain([lo, hi]).nice(tickCount).range([0, plotW]);
  const y = d3.scaleLinear().domain([lo, hi]).nice(tickCount).range([plotH, 0]);
  const [xLo, xHi] = x.domain();

  const g = parent.append("g").attr("transform", `translate(${x0 + M.left},${y0 + M.top})`);

  // Regression line (solid black) - drawn FIRST so the y = x dashed reference
  // sits on top and stays visible where the two overlap.
  const reg = linreg(xVals, yVals);
  const ry1 = reg.slope * xLo + reg.intercept;
  const ry2 = reg.slope * xHi + reg.intercept;
  g.append("line")
    .attr("x1", x(xLo)).attr("y1", y(ry1))
    .attr("x2", x(xHi)).attr("y2", y(ry2))
    .attr("stroke", "#222").attr("stroke-width", 0.3);

  // y = x reference (grey dashed) - painted after the regression line.
  g.append("line")
    .attr("x1", x(xLo)).attr("y1", y(xLo))
    .attr("x2", x(xHi)).attr("y2", y(xHi))
    .attr("stroke", "#bbb").attr("stroke-width", 0.18)
    .attr("stroke-dasharray", "0.7 0.5");

  // Axes
  g.append("line").attr("x1", 0).attr("x2", plotW)
    .attr("y1", plotH).attr("y2", plotH)
    .attr("stroke", "black").attr("stroke-width", 0.2);
  g.append("line").attr("x1", 0).attr("x2", 0)
    .attr("y1", 0).attr("y2", plotH)
    .attr("stroke", "black").attr("stroke-width", 0.2);

  const fmt = d3.format(tickFmt);
  const xTicks = x.ticks(tickCount);
  const yTicks = y.ticks(tickCount);
  xTicks.forEach(d => {
    const xx = x(d);
    g.append("line").attr("x1", xx).attr("x2", xx)
      .attr("y1", plotH).attr("y2", plotH + 0.8)
      .attr("stroke", "black").attr("stroke-width", 0.2);
    g.append("text").attr("x", xx).attr("y", plotH + 2.7)
      .attr("text-anchor", "middle").attr("font-size", 1.8)
      .text(fmt(d));
  });
  yTicks.forEach(d => {
    const yy = y(d);
    g.append("line").attr("x1", -0.8).attr("x2", 0)
      .attr("y1", yy).attr("y2", yy)
      .attr("stroke", "black").attr("stroke-width", 0.2);
    g.append("text").attr("x", -1.0).attr("y", yy + 0.6)
      .attr("text-anchor", "end").attr("font-size", 1.8)
      .text(fmt(d));
  });

  // Data markers - interleave points across species so no single species
  // sits entirely on top. Uses a seeded Fisher-Yates shuffle so the order
  // is deterministic across reloads (important for reproducible figures).
  const rng = mulberry32(0xC0FFEE);
  const symCache = {};
  for (const sp of SPECIES_ORDER) {
    const info = SPECIES_INFO[sp]; if (!info) continue;
    symCache[sp] = d3.symbol().type(info.symbol).size(0.9)();
  }
  const shuffled = points.slice();
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  shuffled.forEach(p => {
    const info = SPECIES_INFO[p.species]; if (!info) return;
    g.append("path")
      .attr("d", symCache[p.species])
      .attr("transform", `translate(${x(p.x)},${y(p.y)})`)
      .attr("fill", info.color)
      .attr("fill-opacity", 0.5);
  });

  // R² annotation - top-left corner of the plot area (6.5 pt bold).
  g.append("text").attr("x", 0.6).attr("y", 2.2)
    .attr("text-anchor", "start")
    .attr("font-size", 2.29).attr("font-weight", "bold")
    .text(`R² = ${reg.r2.toFixed(3)}`);

  // Subpanel title - centred over the plot region. Accepts either a string
  // (single line) or an array of strings (multi-line rendered as <tspan>s).
  const titleLines = Array.isArray(title) ? title : [title];
  const titleFont = titleLines.length > 1 ? 2.2 : 2.47;
  const titleLineH = 2.6;
  const titleFirstY = titleLines.length > 1 ? 2.1 : 3.4;
  const titleCx = x0 + M.left + plotW / 2;
  const titleEl = parent.append("text")
    .attr("text-anchor", "middle").attr("font-size", titleFont)
    .attr("font-weight", "bold");
  titleLines.forEach((line, i) => {
    titleEl.append("tspan")
      .attr("x", titleCx)
      .attr("y", y0 + titleFirstY + i * titleLineH)
      .text(line);
  });

  // Axis titles (6 pt, bold). Tight gap to the tick labels (~0.5 mm).
  parent.append("text")
    .attr("x", x0 + M.left + plotW / 2)
    .attr("y", y0 + M.top + plotH + 5.5)
    .attr("text-anchor", "middle").attr("font-size", 2.12)
    .attr("font-weight", "bold")
    .text(xLabel);
  parent.append("text")
    .attr("transform",
          `translate(${x0 + M.left - 5},${y0 + M.top + plotH / 2}) rotate(-90)`)
    .attr("text-anchor", "middle").attr("font-size", 2.12)
    .attr("font-weight", "bold")
    .text(yLabel);
}

/**
 * Render a horizontal species legend (4 entries, marker + name).
 */
function drawLegend(parent, x0, y0, opts) {
  const { itemGap = 14, markerSize = 2.6, fontSize = 2.0 } = opts || {};
  let cx = x0;
  SPECIES_ORDER.forEach(sp => {
    const info = SPECIES_INFO[sp];
    const sym = d3.symbol().type(info.symbol).size(markerSize)();
    parent.append("path")
      .attr("d", sym)
      .attr("transform", `translate(${cx},${y0})`)
      .attr("fill", info.color)
      .attr("fill-opacity", 0.85);
    parent.append("text")
      .attr("x", cx + 1.6).attr("y", y0 + fontSize * 0.35)
      .attr("font-size", fontSize)
      .text(info.label);
    cx += itemGap;
  });
}

window.Fig4Corr = { drawPanel, drawLegend, SPECIES_INFO, SPECIES_ORDER, linreg };

})();
