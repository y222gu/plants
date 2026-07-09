/**
 * fig2e_microscope_boxplot.js
 *
 * Shared renderer for Figure 2b: per-sample mIoU (Bio-7) grouped by
 * microscope. Three boxes side-by-side - Olympus (test) and C10 (test)
 * are part of the Strategy A test split; Zeiss (oneshot) is the zero-shot
 * held-out split. Each box shows Q1/median/Q3 with whiskers at 1.5 × IQR
 * clamped to observed min/max; jittered sample points overlay the box.
 *
 * Loaded by both fig2e_microscope_boxplot.html (interactive builder) and
 * assemble_figure2.html (live inline rendering).
 *
 * Data source: per_sample_miou_by_microscope.csv
 *   columns = microscope, split, sample_id, species, mean_IoU
 */
(function () {

// Display order + colour. Zeiss is the zero-shot microscope - this is
// clarified via the Figure 2 caption rather than crammed into the x-label.
const GROUPS = [
    { microscope: "Olympus", split: "test",    color: "#4e79a7", suffix: "" },
    { microscope: "C10",     split: "test",    color: "#59a14f", suffix: "" },
    { microscope: "Zeiss",   split: "zero-shot", color: "#e15759", suffix: "" },
];

// Panel geometry (mm) - defaults match the original 113 × 57 slot.
const DEFAULT_PANEL_W = 113, DEFAULT_PANEL_H = 57;
// M.bottom = 11 matches 2a's so the panel bottoms line up; combined with
// M.top = 3.5 and panelH = 42.16 the x-axis lands at canvas y = 31.5 mm
// - same as 2a's. Single-line rotated tick labels drop ~3.5 mm below the
// axis so the title baseline (axis + 7.47) sits ~4 mm below the labels.
// M.right is widened to 11 mm so the "Zero-shot" annotation + arrow fit
// to the right of the rotated "Zeiss" label without affecting PLOT_W.
// M.top = 3 - the absolute minimum so the "n=" key (1.76 mm font,
// cap-top at M.top − 2.95) just clears the panel's top edge. Matches
// panels d (species) and f (scatter) so all three row-2-bottom panels
// have their plot regions flush with the panel top.
const M = { top: 3, right: 8, bottom: 13, left: 8 };
const TITLE_FONT_MM = 2.47;                 // 7 pt
const LABEL_TILT = -25;
// X-title baseline matched across panels d/e/f at axis + 9.47 mm so the
// "Species" / "Microscope" / "Test mIoU" titles share the same canvas y.
const X_TITLE_BASELINE_OFFSET = 9.47;
const Y_TITLE_CENTRE_OFFSET   = 5.94;


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
        yMin = 0.5, yMax = 1.0,
        csvPath = "per_sample_miou_by_microscope.csv",
        panelW = DEFAULT_PANEL_W,
        panelH = DEFAULT_PANEL_H,
        title = "",
    } = opts || {};
    const PANEL_W = panelW;
    const PANEL_H = panelH;
    const PLOT_W = PANEL_W - M.left - M.right;
    const PLOT_H = PANEL_H - M.top - M.bottom;

    const raw = await _loadCsv(csvPath);

    // Bucket per (microscope, split) pair - each group in GROUPS gets its own
    // array of mean_IoU values.
    const byGroup = GROUPS.map(g => {
        const vals = raw
            .filter(r => r.microscope === g.microscope && r.split === g.split)
            .map(r => parseFloat(r.mean_IoU))
            .filter(v => !isNaN(v));
        return { ...g, vals };
    });

    const svg = d3.select(svgEl);
    svg.selectAll("*").remove();
    svg.attr("font-family", "Helvetica, Arial, sans-serif");

    // Optional panel title - 6 pt bold, centred above the plot in the existing
    // M.top region. Panel-letter sits top-left; centred title clears it.
    // Supports multi-line titles via "\n" in the string - each segment becomes
    // a separate <tspan> stacked at the title font's line height.
    if (title) {
        const TITLE_FS = 2.12;
        const TITLE_LH = TITLE_FS * 1.2;       // 2.54 mm - standard line height
        const lines = title.split("\n");
        const textEl = svg.append("text")
            .attr("y", 3.0)
            .attr("text-anchor", "middle")
            .attr("font-size", TITLE_FS).attr("font-weight", "bold");
        lines.forEach((line, i) => {
            textEl.append("tspan")
                .attr("x", PANEL_W / 2)
                .attr("dy", i === 0 ? 0 : TITLE_LH)
                .text(line);
        });
    }

    const g = svg.append("g").attr("transform", `translate(${M.left},${M.top})`);

    // ── Scales ─────────────────────────────────────────────────────
    const x = d3.scaleBand()
        .domain(GROUPS.map(d => d.microscope))
        .range([0, PLOT_W])
        .paddingInner(0.35);
    const y = d3.scaleLinear().domain([yMin, yMax]).range([PLOT_H, 0]);

    // ── Grid ──────────────────────────────────────────────────────
    const yTicks = y.ticks(5);
    g.selectAll("line.hg").data(yTicks).enter().append("line")
        .attr("x1", 0).attr("x2", PLOT_W)
        .attr("y1", d => y(d)).attr("y2", d => y(d))
        .attr("stroke", "#e4e4e4").attr("stroke-width", 0.1);

    // ── Axes (slightly thicker than tick marks) ───────────────────
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
        .text("Sample-level mIoU");
    // X-title (7 pt bold) - centred under the plot, baseline at M.top +
    // PLOT_H + 7.47 so it shares 2a's title canvas y.
    svg.append("text")
        .attr("x", M.left + PLOT_W / 2)
        .attr("y", M.top + PLOT_H + X_TITLE_BASELINE_OFFSET)
        .attr("text-anchor", "middle").attr("font-size", TITLE_FONT_MM)
        .attr("font-weight", "bold")
        .text("Microscope");

    // ── Boxes per microscope group ────────────────────────────────
    const boxW = Math.max(2.0, x.bandwidth() * 0.65);
    for (const gp of byGroup) {
        if (gp.vals.length === 0) continue;
        const stats = boxStats(gp.vals);
        if (!stats) continue;
        const cx = x(gp.microscope) + x.bandwidth() / 2;
        const color = gp.color;

        // Whiskers - stem and caps both 0.15 mm to match fig2e
        g.append("line")
            .attr("x1", cx).attr("x2", cx)
            .attr("y1", y(stats.whiskerLo)).attr("y2", y(stats.whiskerHi))
            .attr("stroke", "black").attr("stroke-width", 0.15);
        [stats.whiskerLo, stats.whiskerHi].forEach(v => {
            g.append("line")
                .attr("x1", cx - boxW * 0.3).attr("x2", cx + boxW * 0.3)
                .attr("y1", y(v)).attr("y2", y(v))
                .attr("stroke", "black").attr("stroke-width", 0.15);
        });
        // Box
        g.append("rect")
            .attr("x", cx - boxW / 2).attr("y", y(stats.q3))
            .attr("width", boxW).attr("height", y(stats.q1) - y(stats.q3))
            .attr("fill", color).attr("fill-opacity", 0.35)
            .attr("stroke", color).attr("stroke-width", 0.25);

        // Jittered points first, then median bar on top
        const jitterW = boxW * 0.55;
        gp.vals.forEach((v, i) => {
            const t = (Math.sin(i * 9301 + 49297 + gp.microscope.length) + 1) / 2;
            const jx = cx + (t - 0.5) * jitterW;
            g.append("circle")
                .attr("cx", jx).attr("cy", y(v))
                .attr("r", 0.22)
                .attr("fill", color).attr("fill-opacity", 0.55)
                .attr("stroke", "none");
        });
        // Median line - black 0.25 mm to match fig2e
        g.append("line")
            .attr("x1", cx - boxW / 2).attr("x2", cx + boxW / 2)
            .attr("y1", y(stats.med)).attr("y2", y(stats.med))
            .attr("stroke", "black").attr("stroke-width", 0.25);
    }

    // ── X tick marks + rotated microscope labels (n=N rendered separately below).
    for (const gp of byGroup) {
        const cx = x(gp.microscope) + x.bandwidth() / 2;
        g.append("line")
            .attr("x1", cx).attr("x2", cx)
            .attr("y1", PLOT_H).attr("y2", PLOT_H + 0.8)
            .attr("stroke", "black").attr("stroke-width", 0.2);
        // Rotated label translate y = PLOT_H + 2.5 so the topmost point of
        // the rotated body sits ~0.7 mm below the tick mark end (matches
        // fig2f's tick-to-label visual gap). Multi-line labels (containing
        // a literal "\n") are split into one <tspan> per line, sharing the
        // parent rotation. Each line is CENTER-aligned with respect to the
        // longest line: the longest line's right end sits at the tick (x=0
        // under anchor=end) and shorter lines are shifted right by
        // (width − longestWidth) / 2 so their centres line up. dy="1em"
        // drops each subsequent line by one font-size in pre-rotation y.
        const labelEl = g.append("text")
            .attr("transform", `translate(${cx},${PLOT_H + 2.5}) rotate(${LABEL_TILT})`)
            .attr("text-anchor", "end").attr("font-size", 2.12)
            .attr("font-weight", "bold")
            .attr("fill", gp.color);
        const lines = `${gp.microscope}${gp.suffix}`.split("\n");
        // Measure each line's true rendered width so we can centre-align them.
        const measureG = svg.append("g").attr("visibility", "hidden");
        const widths = lines.map(line => {
            const t = measureG.append("text")
                .attr("font-size", 2.12).attr("font-weight", "bold")
                .text(line);
            return t.node().getComputedTextLength();
        });
        measureG.remove();
        const longestW = Math.max(...widths);
        lines.forEach((line, i) => {
            const xOffset = (widths[i] - longestW) / 2;   // 0 for longest, negative for shorter
            labelEl.append("tspan")
                .attr("x", xOffset)
                .attr("dy", i === 0 ? 0 : "1em")
                .text(line);
        });
    }

    // ── "Zero-shot" annotation pointing at the Zeiss tick label ─────
    // Reminds the reader that Zeiss is the held-out (zero-shot) microscope.
    // Coloured to match the Zeiss group; small arrow points up-left from the
    // text to just below the right end of the rotated "Zeiss" label.
    const zeissGp = GROUPS.find(gp => gp.microscope === "Zeiss");
    if (zeissGp && x("Zeiss") != null) {
        const zCxPanel = M.left + x("Zeiss") + x.bandwidth() / 2;
        const xAxisY   = M.top  + PLOT_H;
        const annoFS   = 1.76;                                    // 5 pt
        const annoColor = zeissGp.color;
        const arrowId  = "fig2e-zero-shot-arrow";

        // Horizontal arrow pointing left at the rotated "Zeiss" label.
        // Y = visual centre of "Zeiss" on screen:
        //   cap top of right-end letter (highest pt) ≈ xAxisY + 1.37
        //   baseline of left-end letter (lowest pt)  ≈ xAxisY + 4.61
        //   midpoint                                  ≈ xAxisY + 2.99
        const zeissCenterY = xAxisY + 2.99;
        const headX = zCxPanel + 0.4;
        const headY = zeissCenterY;
        const tailX = zCxPanel + 3.0;
        const tailY = zeissCenterY;

        const defs = svg.append("defs");
        defs.append("marker")
            .attr("id", arrowId)
            .attr("markerUnits", "userSpaceOnUse")
            .attr("viewBox", "0 0 1 1")
            .attr("refX", 0.9).attr("refY", 0.5)
            .attr("markerWidth", 1.0).attr("markerHeight", 1.0)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M 0 0 L 1 0.5 L 0 1 z")
            .attr("fill", annoColor);
        svg.append("line")
            .attr("x1", tailX).attr("y1", tailY)
            .attr("x2", headX).attr("y2", headY)
            .attr("stroke", annoColor).attr("stroke-width", 0.25)
            .attr("marker-end", `url(#${arrowId})`);
        // Text - visual centre aligned with arrow line (= "Zeiss" label's
        // visual centre y). Baseline = centre_y + ½ cap height for 5 pt
        // Helvetica (cap height ≈ 0.71 em → half = 0.355 em).
        svg.append("text")
            .attr("x", tailX + 0.4).attr("y", zeissCenterY + annoFS * 0.355)
            .attr("text-anchor", "start").attr("font-size", annoFS)
            .attr("font-style", "italic")
            .attr("fill", annoColor)
            .text("Zero-shot");
    }

    // Sample counts - "n=" key sits once in the M.top whitespace at the
    // y-axis end; each microscope's count number floats just ABOVE its own
    // upper whisker cap.
    const N_FS = 1.76;                       // 5 pt
    const N_KEY_BASELINE_Y = -1.7;           // 0.5 mm gap above topmost y-tick
    const N_GAP = 0.85;                      // baseline 0.85 mm above whisker cap
    g.append("text")
        .attr("x", -0.5).attr("y", N_KEY_BASELINE_Y)
        .attr("text-anchor", "end").attr("font-size", N_FS)
        .attr("fill", "#555")
        .text("n=");
    for (const gp of byGroup) {
        if (gp.vals.length === 0) continue;
        const stats = boxStats(gp.vals);
        if (!stats) continue;
        const cx = x(gp.microscope) + x.bandwidth() / 2;
        g.append("text")
            .attr("x", cx).attr("y", y(stats.whiskerHi) - N_GAP)
            .attr("text-anchor", "middle").attr("font-size", N_FS)
            .attr("fill", "#555")
            .text(`${gp.vals.length}`);
    }

    return byGroup;
}


window.Fig2eMicroscopeBoxplot = { render, GROUPS,
                                  PANEL_W: DEFAULT_PANEL_W, PANEL_H: DEFAULT_PANEL_H };

})();
