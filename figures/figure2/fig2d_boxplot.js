/**
 * fig2d_boxplot.js
 *
 * Shared renderer for Figure 2c: per-class Bio-7 IoU across the Strategy A
 * test split (all species pooled). Six boxes (Epidermis, Exodermis, Cortex,
 * Aerenchyma, Endodermis, Vascular) - one per anatomical class. Jittered
 * sample points overlay each box. Colours use the canonical Fig-1a anatomy
 * palette (same colours as the Bio-7 semantic legend in the figure).
 *
 * No internal legend - this panel shares the class legend that sits in the
 * 2b/legend row of the figure 2 assembly.
 *
 * Loaded by both fig2d_boxplot.html (interactive builder) and
 * assemble_figure2.html (live inline rendering).
 */
(function () {

// Six bio classes + Fig-1a anatomy palette. Order goes innermost-anatomy
// → outermost (Vascular at the centre of the root, Epidermis at the edge),
// with Aerenchyma at the end since it's absent in half the dataset.
const CLASSES = [
    { name: "Vascular",   color: "#e76f61" },
    { name: "Exodermis",  color: "#f4a261" },
    { name: "Endodermis", color: "#f6e48e" },
    { name: "Cortex",     color: "#94d2bd" },
    { name: "Epidermis",  color: "#0a9396" },
    { name: "Aerenchyma", color: "#264653" },
];

// Lighten a hex colour toward white by `amount` (0 = no change, 1 = white).
// Used to render the annotator-baseline boxes in a lighter shade of the
// class colour - model boxes use the original colour, baseline boxes use
// this lighter tint, so the visual hierarchy is "darker = main result".
function lighten(hex, amount) {
    const h = hex.replace("#", "");
    const r = parseInt(h.slice(0, 2), 16);
    const g = parseInt(h.slice(2, 4), 16);
    const b = parseInt(h.slice(4, 6), 16);
    const lr = Math.round(r + (255 - r) * amount);
    const lg = Math.round(g + (255 - g) * amount);
    const lb = Math.round(b + (255 - b) * amount);
    return `#${[lr,lg,lb].map(v => v.toString(16).padStart(2,"0")).join("")}`;
}

// Panel geometry (mm) - defaults match the original 113×57 layout. When the
// panel is narrower, the renderer scales everything proportionally because
// all drawing is in mm units inside the SVG viewBox.
const DEFAULT_PANEL_W = 113, DEFAULT_PANEL_H = 57;
// X-title "Class" sits below the −25° rotated class-name tick labels.
// Sample-count row ("n=…") sits ABOVE the plot in the M.top whitespace
// band - M.top widened from 3 to 3.5 mm so the descender clears the
// topmost y-tick label by the canonical 0.5 mm.
const M = { top: 6.5, right: 3, bottom: 11, left: 8 };
const TITLE_FONT_MM = 2.47;                 // 7 pt
const LABEL_TILT = -25;
const Y_TITLE_CENTRE_OFFSET = 5.94;
// Title baseline placed so:
//   • the gap between the title cap-top and the rotated tick label's
//     bottom-most baseline (≈ "A" of "Aerenchyma" at panel y 34.46) is
//     the canonical 0.5 mm - matching the y-tick-label-to-y-title gap.
//   • paired with the assembler shifting fig2e's panel top up by 1.5 mm,
//     the title's canvas y baseline lands at 77.66 = bottom of the last
//     image row in fig2b.
// Set 1 mm larger than fig2d_species_boxplot.js (panel d, 9.47 mm) so the
// visual gap from the rotated class-name tick labels' BOTTOM to the
// "Anatomical Class" x-title matches panel d. Panels b/c have longer labels
// ("Aerenchyma", "Endodermis" - 10 chars) than panel d ("Solanums" - 8 chars),
// and the −25° rotation makes the longer labels extend ~1 mm further down;
// the larger offset here absorbs that so the title-to-label gap is uniform.
const X_TITLE_BASELINE_OFFSET = 10.47;


const _csvCache = new Map();
async function _loadCsv(path) {
    if (!_csvCache.has(path)) {
        _csvCache.set(path, d3.csv(path + "?t=" + Date.now()));
    }
    return _csvCache.get(path);
}


function boxStats(vals) {
    const s = vals.filter(v => v != null && !isNaN(v)).sort(d3.ascending);
    if (s.length === 0) return null;
    const q1 = d3.quantile(s, 0.25);
    const med = d3.quantile(s, 0.50);
    const q3 = d3.quantile(s, 0.75);
    const iqr = q3 - q1;
    const whiskerLo = Math.max(d3.min(s), q1 - 1.5 * iqr);
    const whiskerHi = Math.min(d3.max(s), q3 + 1.5 * iqr);
    return { q1, med, q3, whiskerLo, whiskerHi, n: s.length, vals: s };
}


/** Gaussian kernel density estimate on a fixed grid. */
function kde(values, grid, bandwidth) {
    const n = values.length;
    if (n === 0) return grid.map(() => 0);
    const norm = 1 / (n * bandwidth * Math.sqrt(2 * Math.PI));
    return grid.map(x => {
        let s = 0;
        for (const v of values) {
            const u = (x - v) / bandwidth;
            s += Math.exp(-0.5 * u * u);
        }
        return norm * s;
    });
}


async function render(svgEl, opts) {
    const {
        yMin = 0.0, yMax = 1.0,
        csvPath = "per_sample_iou.csv",
        humanCsvPath = "human_baseline_iou.csv",
        // Optional: restrict to rows where row.split === splitFilter.
        // Used by figure 2 panels b (splitFilter="test") and
        // c (splitFilter="zero-shot") so each subset plots separately.
        splitFilter = null,
        // Optional override for the final box width (subBoxW). If set, the
        // renderer ignores the bandwidth-based default and computes
        // fullBoxW from this value, so a single boxWidth value yields
        // equal-size boxes whether the panel renders paired (model + human)
        // or single boxes. Used by assemble_figure2 to keep panel b and
        // panel c's boxes the same physical width while their panels share
        // the same bandwidth (and therefore the same inter-class gap).
        boxWidth = null,
        panelW = DEFAULT_PANEL_W,
        panelH = DEFAULT_PANEL_H,
        title = "",
    } = opts || {};
    const PANEL_W = panelW;
    const PANEL_H = panelH;
    const PLOT_W = PANEL_W - M.left - M.right;
    const PLOT_H = PANEL_H - M.top - M.bottom;

    const raw = await _loadCsv(csvPath);
    const filtered = splitFilter ? raw.filter(r => r.split === splitFilter) : raw;

    // Pool IoUs across all species per class - model predictions.
    const byClass = {};
    for (const c of CLASSES) byClass[c.name] = [];
    for (const row of filtered) {
        for (const c of CLASSES) {
            const v = row[`${c.name}_IoU`];
            if (v === "" || v == null) continue;
            const f = parseFloat(v);
            if (!isNaN(f)) byClass[c.name].push(f);
        }
    }

    // Inter-annotator (human–human) baseline. Optional - if the CSV is missing,
    // the plot falls back to the model-only style.
    const byClassHuman = {};
    for (const c of CLASSES) byClassHuman[c.name] = [];
    if (humanCsvPath) {
        try {
            const hraw = await _loadCsv(humanCsvPath);
            for (const row of hraw) {
                for (const c of CLASSES) {
                    const v = row[`${c.name}_IoU`];
                    if (v === "" || v == null) continue;
                    const f = parseFloat(v);
                    if (!isNaN(f)) byClassHuman[c.name].push(f);
                }
            }
        } catch (e) {
            console.warn(`fig2d: no human baseline CSV at ${humanCsvPath} (${e.message})`);
        }
    }
    const hasHuman = Object.values(byClassHuman).some(v => v.length > 0);

    const svg = d3.select(svgEl);
    svg.selectAll("*").remove();
    svg.attr("font-family", "Helvetica, Arial, sans-serif");

    // Optional panel title - 6 pt bold Helvetica, centred horizontally inside
    // the existing M.top so the plot area is unchanged. Panel-letter sits in
    // the top-left corner (HTML span over the SVG); a centred title clears it.
    if (title) {
        svg.append("text")
            .attr("x", PANEL_W / 2).attr("y", 3.0)
            .attr("text-anchor", "middle")
            .attr("font-size", 2.12).attr("font-weight", "bold")
            .text(title);
    }

    const x = d3.scaleBand()
        .domain(CLASSES.map(c => c.name))
        .range([0, PLOT_W])
        .paddingInner(0.30);
    const y = d3.scaleLinear().domain([yMin, yMax]).range([PLOT_H, 0]);

    const g = svg.append("g").attr("transform", `translate(${M.left},${M.top})`);

    // Grid
    const yTicks = y.ticks(5);
    g.selectAll("line.hg").data(yTicks).enter().append("line")
        .attr("x1", 0).attr("x2", PLOT_W)
        .attr("y1", d => y(d)).attr("y2", d => y(d))
        .attr("stroke", "#e4e4e4").attr("stroke-width", 0.1);

    // Axes (slightly thicker than tick marks)
    g.append("line").attr("x1", 0).attr("x2", PLOT_W)
        .attr("y1", PLOT_H).attr("y2", PLOT_H)
        .attr("stroke", "black").attr("stroke-width", 0.2);
    g.append("line").attr("x1", 0).attr("x2", 0)
        .attr("y1", 0).attr("y2", PLOT_H)
        .attr("stroke", "black").attr("stroke-width", 0.2);

    // Y ticks + labels
    g.selectAll(".yt").data(yTicks).enter().append("g").each(function (d) {
        const yy = y(d);
        d3.select(this).append("line")
            .attr("x1", -0.8).attr("x2", 0)
            .attr("y1", yy).attr("y2", yy)
            .attr("stroke", "black").attr("stroke-width", 0.2);
        d3.select(this).append("text")
            .attr("x", -1.2).attr("y", yy + 0.7)
            .attr("text-anchor", "end").attr("font-size", 2.12)
            .text(d3.format(".1f")(d));
    });

    // Y-title (7 pt bold): 0.5 mm left of the y-tick labels
    svg.append("text")
        .attr("transform",
              `translate(${M.left - Y_TITLE_CENTRE_OFFSET},${M.top + PLOT_H / 2}) rotate(-90)`)
        .attr("text-anchor", "middle").attr("font-size", TITLE_FONT_MM)
        .attr("font-weight", "bold")
        .text("IoU");
    // X-title "Class" (7 pt bold): sits below the rotated class-name tick labels
    svg.append("text")
        .attr("x", M.left + PLOT_W / 2)
        .attr("y", M.top + PLOT_H + X_TITLE_BASELINE_OFFSET)
        .attr("text-anchor", "middle").attr("font-size", TITLE_FONT_MM)
        .attr("font-weight", "bold")
        .text("Anatomical Class");

    // X ticks + rotated class-name labels (n=N is rendered separately below).
    for (const c of CLASSES) {
        const cx = x(c.name) + x.bandwidth() / 2;
        g.append("line")
            .attr("x1", cx).attr("x2", cx)
            .attr("y1", PLOT_H).attr("y2", PLOT_H + 0.8)
            .attr("stroke", "black").attr("stroke-width", 0.2);
        // Rotated label translate y = PLOT_H + 2.5 so the topmost point of
        // the rotated body sits ~0.7 mm below the tick mark end (matches
        // fig2f's tick-to-label visual gap). Smaller offsets crowd the tick.
        g.append("text")
            .attr("transform", `translate(${cx},${PLOT_H + 2.5}) rotate(${LABEL_TILT})`)
            .attr("text-anchor", "end").attr("font-size", 2.12)
            .attr("font-weight", "bold")
            .attr("fill", c.color)
            .text(c.name);
    }

    // Each class column hosts two side-by-side boxes when human baseline is
    // available: model (left, solid filled) and human-baseline (right, outline
    // only). Both share the class color; the fill style is the only cue.
    const subGap   = 0.35;                    // mm between paired boxes
    // If the caller supplied an explicit box width, derive fullBoxW from
    // it so paired panels and single panels render at the same physical
    // box size. Otherwise fall back to the bandwidth-based 0.75 default.
    const fullBoxW = (boxWidth != null)
        ? (hasHuman ? 2 * boxWidth + subGap : boxWidth)
        : Math.max(1.2, x.bandwidth() * 0.75);
    const subBoxW  = hasHuman ? (fullBoxW - subGap) / 2 : fullBoxW;

    // drawBox: render whiskers + caps + box + jittered points + median.
    // `style` ∈ "model" | "human". Both styles are SOLID FILLED with the
    // class colour; "human" uses a lighter tint to set the model apart.
    const HUMAN_LIGHTEN = 0.85;        // 0=identical, 1=white. Used for the
                                       // (currently unused) box-fill ramp and
                                       // the box-stroke (× 0.45 below).
    const HUMAN_POINT_LIGHTEN = 0.25;  // scatter points: less lightening than
                                       // the box-fill ramp so the dots stay
                                       // visible against a white panel.
    function drawBox(cx, stats, vals, color, classLen, style) {
        const fill = style === "model" ? color : lighten(color, HUMAN_LIGHTEN);
        const stroke = style === "model" ? color : lighten(color, HUMAN_LIGHTEN * 0.45);
        const pointFill = style === "model" ? color : lighten(color, HUMAN_POINT_LIGHTEN);
        // Whiskers
        g.append("line")
            .attr("x1", cx).attr("x2", cx)
            .attr("y1", y(stats.whiskerLo)).attr("y2", y(stats.whiskerHi))
            .attr("stroke", "black").attr("stroke-width", 0.15);
        [stats.whiskerLo, stats.whiskerHi].forEach(v => {
            g.append("line")
                .attr("x1", cx - subBoxW * 0.35).attr("x2", cx + subBoxW * 0.35)
                .attr("y1", y(v)).attr("y2", y(v))
                .attr("stroke", "black").attr("stroke-width", 0.15);
        });
        // Box: model = solid fill, baseline = outline only (no fill).
        const rect = g.append("rect")
            .attr("x", cx - subBoxW / 2).attr("y", y(stats.q3))
            .attr("width", subBoxW).attr("height", y(stats.q1) - y(stats.q3))
            .attr("stroke", stroke).attr("stroke-width", 0.25);
        if (style === "model") {
            rect.attr("fill", fill).attr("fill-opacity", 0.55);
        } else {
            rect.attr("fill", "none");
        }
        // Jittered points
        const jitterW = subBoxW * 0.60;
        vals.forEach((v, i) => {
            const t = (Math.sin(i * 9301 + 49297 + classLen + (style === "human" ? 17 : 0)) + 1) / 2;
            const jx = cx + (t - 0.5) * jitterW;
            g.append("circle")
                .attr("cx", jx).attr("cy", y(v))
                .attr("r", 0.22)
                .attr("fill", pointFill).attr("fill-opacity", 0.55)
                .attr("stroke", "none");
        });
        // Median line - black across all boxes for consistency
        g.append("line")
            .attr("x1", cx - subBoxW / 2).attr("x2", cx + subBoxW / 2)
            .attr("y1", y(stats.med)).attr("y2", y(stats.med))
            .attr("stroke", "black").attr("stroke-width", 0.25);
    }

    for (const c of CLASSES) {
        const vals = byClass[c.name];
        if (!vals.length) continue;
        const stats = boxStats(vals);
        if (!stats) continue;
        const cx = x(c.name) + x.bandwidth() / 2;

        if (hasHuman) {
            // Two sub-boxes: model (left, full colour), human-baseline (right, lighter)
            const cxModel = cx - (subBoxW + subGap) / 2;
            const cxHuman = cx + (subBoxW + subGap) / 2;
            drawBox(cxModel, stats, vals, c.color, c.name.length, "model");
            const hVals = byClassHuman[c.name] || [];
            const hStats = boxStats(hVals);
            if (hStats) drawBox(cxHuman, hStats, hVals, c.color, c.name.length, "human");
        } else {
            drawBox(cx, stats, vals, c.color, c.name.length, "model");
        }
    }
    // Expose subBoxW for the n=N renderer below
    const boxW = subBoxW;

    // Inline legend distinguishing the two box styles, positioned at the
    // BOTTOM-CENTER of the plot (horizontally centred across the full plot
    // width). Wrapped in a bordered rectangle to set it apart from the
    // data marks. Always rendered: panel b shows both rows (Model + human
    // baseline), panel c shows only the Model row since the annotator
    // dataset does not cover the out-of-distribution split.
    {
        const LG_FS = 2.12;                           // 6 pt
        const LG_BOX_W = 1.75, LG_BOX_H = 1.75;       // square swatch sized to text cap-height (≈ 0.71 × em)
        const LG_GAP = 0.6;                           // gap between swatch and label
        const LG_ROW_PITCH = 3.2;                     // vertical step between rows (extra breathing room between the two legend entries)
        const LG_PAD = 1.4;                           // padding inside the legend frame (text breathing room from frame edge)
        const LG_GREY = "#888";
        const labels = hasHuman
            ? ["Model Prediction vs Ground Truth", "Internal Annotator vs Ground Truth"]
            : ["Model Prediction vs Ground Truth"];
        // Measure widest label
        const measureG = svg.append("g").attr("visibility", "hidden");
        const widths = labels.map(t => measureG.append("text")
            .attr("font-size", LG_FS).text(t).node().getComputedTextLength());
        measureG.remove();
        const widestLabel = Math.max(...widths);
        const innerW = LG_BOX_W + LG_GAP + widestLabel;
        const innerH = LG_ROW_PITCH * (labels.length - 1) + LG_BOX_H;
        const frameW = innerW + 2 * LG_PAD;
        const frameH = innerH + 2 * LG_PAD;
        // Centre horizontally across the full plot width.
        const frameX = (PLOT_W - frameW) / 2;
        const frameY = PLOT_H - frameH - 0.5;          // 0.5 mm above x-axis
        // Frame: white fill + light grey border
        g.append("rect")
            .attr("x", frameX).attr("y", frameY)
            .attr("width", frameW).attr("height", frameH)
            .attr("fill", "white").attr("fill-opacity", 0.95)
            .attr("stroke", "#888").attr("stroke-width", 0.2)
            .attr("rx", 0.4);
        const lgX = frameX + LG_PAD;
        const lgY = frameY + LG_PAD;
        const drawRow = (rowIdx, label, fillOpts, strokeColor) => {
            const yRow = lgY + rowIdx * LG_ROW_PITCH;
            const rect = g.append("rect")
                .attr("x", lgX).attr("y", yRow)
                .attr("width", LG_BOX_W).attr("height", LG_BOX_H)
                .attr("stroke", strokeColor).attr("stroke-width", 0.25);
            if (fillOpts) {
                rect.attr("fill", fillOpts.fill).attr("fill-opacity", fillOpts.opacity);
            } else {
                rect.attr("fill", "none");
            }
            g.append("text")
                .attr("x", lgX + LG_BOX_W + LG_GAP)
                .attr("y", yRow + LG_BOX_H)
                .attr("font-size", LG_FS).attr("fill", "#333")
                .text(label);
        };
        // Row 1: filled grey box (model prediction style).
        drawRow(0, labels[0], { fill: LG_GREY, opacity: 0.55 }, LG_GREY);
        // Row 2 (only when human baseline exists): outline-only box.
        if (hasHuman) drawRow(1, labels[1], null, LG_GREY);
    }

    // Sample counts - "n=" key sits once in the M.top whitespace at the
    // y-axis end; each box's count number floats just ABOVE its own upper
    // whisker cap (per the user's preference for a tight per-box label).
    const N_FS = 1.76;                       // 5 pt
    const N_KEY_BASELINE_Y = -1.7;           // 0.5 mm gap above topmost y-tick
    const N_GAP = 0.85;                      // baseline 0.85 mm above whisker cap
    g.append("text")
        .attr("x", -0.5).attr("y", N_KEY_BASELINE_Y)
        .attr("text-anchor", "end").attr("font-size", N_FS)
        .attr("fill", "#555")
        .text("n=");
    for (const c of CLASSES) {
        const cx = x(c.name) + x.bandwidth() / 2;
        const vals = byClass[c.name] || [];
        if (vals.length > 0) {
            const stats = boxStats(vals);
            const cxM = hasHuman ? cx - (subBoxW + subGap) / 2 : cx;
            g.append("text")
                .attr("x", cxM).attr("y", y(stats.whiskerHi) - N_GAP)
                .attr("text-anchor", "middle").attr("font-size", N_FS)
                .attr("fill", "#555")
                .text(`${vals.length}`);
        }
        if (hasHuman) {
            const hVals = byClassHuman[c.name] || [];
            if (hVals.length > 0) {
                const hStats = boxStats(hVals);
                const cxH = cx + (subBoxW + subGap) / 2;
                g.append("text")
                    .attr("x", cxH).attr("y", y(hStats.whiskerHi) - N_GAP)
                    .attr("text-anchor", "middle").attr("font-size", N_FS)
                    .attr("fill", "#555")
                    .text(`${hVals.length}`);
            }
        }
    }

    return byClass;
}


window.Fig2dBoxplot = { render, CLASSES,
                        PANEL_W: DEFAULT_PANEL_W, PANEL_H: DEFAULT_PANEL_H };

})();
