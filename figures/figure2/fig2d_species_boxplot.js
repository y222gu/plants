/**
 * fig2d_species_boxplot.js
 *
 * Renderer for Figure 2 panel d: per-sample mIoU (Bio-7) grouped by
 * species, restricted to the in-distribution test split. Four boxes
 * (Tomato, Rice, Sorghum, Millet). Each box: Q1 / median / Q3 with
 * 1.5×IQR whiskers; jittered sample points overlay the box.
 *
 * Data source: per_sample_iou.csv
 *   relevant columns = species, mean_IoU, split
 *
 * Loaded by assemble_figure2.html for live inline rendering.
 */
(function () {

// Display order: Rice → Millet → Sorghum → Solanums. `dataKey` is the
// value in per_sample_iou.csv's `species` column (the CSV uses "Solanum"
// singular for the tomato samples since the set spans multiple Solanum
// species - S. lycopersicum, S. cheesmaniae). `label` is the plural form
// shown on the x-axis tick. Colours match figure 4e–h (SPECIES_INFO in
// fig4_corr_common.js) so the species palette is consistent across the
// paper's downstream-correlation and per-species-mIoU panels.
const GROUPS = [
    { dataKey: "Rice",    label: "Rice",     color: "#9ab2d4" },
    { dataKey: "Millet",  label: "Millet",   color: "#9393c9" },
    { dataKey: "Sorghum", label: "Sorghum",  color: "#514a8d" },
    { dataKey: "Solanum", label: "Solanums", color: "#ab8b58" },
];

const DEFAULT_PANEL_W = 80, DEFAULT_PANEL_H = 42.16;
// Margins mirror fig2e_microscope_boxplot and fig2f_scatter so all three
// row-2-bottom panels share the same x-axis canvas y AND the same x-title
// canvas y. M.top = 3 - the absolute minimum so the "n=" key (1.76 mm
// font, cap-top at M.top − 2.95) just clears the panel's top edge. The
// chart title is empty so no extra room is needed at the top.
const M = { top: 3, right: 3, bottom: 13, left: 8 };
const TITLE_FONT_MM = 2.47;                 // 7 pt
const LABEL_TILT = -25;
// X-title offset matched across panels d/e/f (9.47 mm below the axis) so
// the three "Species" / "Microscope" / "Test mIoU" titles share the same
// canvas y baseline. The offset is 2 mm larger than before to absorb the
// 2 mm plot-region lift while keeping the title at the same canvas y.
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


async function render(svgEl, opts) {
    const {
        yMin = 0.0, yMax = 1.0,
        csvPath = "per_sample_iou.csv",
        splitFilter = "test",
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

    const byGroup = GROUPS.map(g => {
        const vals = filtered
            .filter(r => r.species === g.dataKey)
            .map(r => parseFloat(r.mean_IoU))
            .filter(v => !isNaN(v));
        return { ...g, vals };
    });

    const svg = d3.select(svgEl);
    svg.selectAll("*").remove();
    svg.attr("font-family", "Helvetica, Arial, sans-serif");

    if (title) {
        const TITLE_FS = 2.12;
        const TITLE_LH = TITLE_FS * 1.2;
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

    const x = d3.scaleBand()
        .domain(GROUPS.map(d => d.label))
        .range([0, PLOT_W])
        .paddingInner(0.35);
    const y = d3.scaleLinear().domain([yMin, yMax]).range([PLOT_H, 0]);

    const yTicks = y.ticks(5);
    g.selectAll("line.hg").data(yTicks).enter().append("line")
        .attr("x1", 0).attr("x2", PLOT_W)
        .attr("y1", d => y(d)).attr("y2", d => y(d))
        .attr("stroke", "#e4e4e4").attr("stroke-width", 0.1);

    g.append("line").attr("x1", 0).attr("x2", PLOT_W)
        .attr("y1", PLOT_H).attr("y2", PLOT_H)
        .attr("stroke", "black").attr("stroke-width", 0.2);
    g.append("line").attr("x1", 0).attr("x2", 0)
        .attr("y1", 0).attr("y2", PLOT_H)
        .attr("stroke", "black").attr("stroke-width", 0.2);

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

    svg.append("text")
        .attr("transform",
              `translate(${M.left - Y_TITLE_CENTRE_OFFSET},${M.top + PLOT_H / 2}) rotate(-90)`)
        .attr("text-anchor", "middle").attr("font-size", TITLE_FONT_MM)
        .attr("font-weight", "bold")
        .text("Sample-level mIoU");
    svg.append("text")
        .attr("x", M.left + PLOT_W / 2)
        .attr("y", M.top + PLOT_H + X_TITLE_BASELINE_OFFSET)
        .attr("text-anchor", "middle").attr("font-size", TITLE_FONT_MM)
        .attr("font-weight", "bold")
        .text("Species");

    const boxW = Math.max(2.0, x.bandwidth() * 0.65);
    for (const gp of byGroup) {
        if (gp.vals.length === 0) continue;
        const stats = boxStats(gp.vals);
        if (!stats) continue;
        const cx = x(gp.label) + x.bandwidth() / 2;
        const color = gp.color;

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
        g.append("rect")
            .attr("x", cx - boxW / 2).attr("y", y(stats.q3))
            .attr("width", boxW).attr("height", y(stats.q1) - y(stats.q3))
            .attr("fill", color).attr("fill-opacity", 0.35)
            .attr("stroke", color).attr("stroke-width", 0.25);

        const jitterW = boxW * 0.55;
        gp.vals.forEach((v, i) => {
            const t = (Math.sin(i * 9301 + 49297 + gp.label.length) + 1) / 2;
            const jx = cx + (t - 0.5) * jitterW;
            g.append("circle")
                .attr("cx", jx).attr("cy", y(v))
                .attr("r", 0.22)
                .attr("fill", color).attr("fill-opacity", 0.55)
                .attr("stroke", "none");
        });
        g.append("line")
            .attr("x1", cx - boxW / 2).attr("x2", cx + boxW / 2)
            .attr("y1", y(stats.med)).attr("y2", y(stats.med))
            .attr("stroke", "black").attr("stroke-width", 0.25);
    }

    for (const gp of byGroup) {
        const cx = x(gp.label) + x.bandwidth() / 2;
        g.append("line")
            .attr("x1", cx).attr("x2", cx)
            .attr("y1", PLOT_H).attr("y2", PLOT_H + 0.8)
            .attr("stroke", "black").attr("stroke-width", 0.2);
        g.append("text")
            .attr("transform", `translate(${cx},${PLOT_H + 2.5}) rotate(${LABEL_TILT})`)
            .attr("text-anchor", "end").attr("font-size", 2.12)
            .attr("font-weight", "bold")
            .attr("fill", gp.color)
            .text(gp.label);
    }

    // Sample counts - "n=" key in the M.top whitespace; per-box counts
    // float just above each box's upper whisker cap.
    const N_FS = 1.76;
    const N_KEY_BASELINE_Y = -1.7;
    const N_GAP = 0.85;
    g.append("text")
        .attr("x", -0.5).attr("y", N_KEY_BASELINE_Y)
        .attr("text-anchor", "end").attr("font-size", N_FS)
        .attr("fill", "#555")
        .text("n=");
    for (const gp of byGroup) {
        if (gp.vals.length === 0) continue;
        const stats = boxStats(gp.vals);
        const cx = x(gp.label) + x.bandwidth() / 2;
        g.append("text")
            .attr("x", cx).attr("y", y(stats.whiskerHi) - N_GAP)
            .attr("text-anchor", "middle").attr("font-size", N_FS)
            .attr("fill", "#555")
            .text(`${gp.vals.length}`);
    }
}


window.Fig2SpeciesBoxplot = { render, GROUPS,
                              PANEL_W: DEFAULT_PANEL_W, PANEL_H: DEFAULT_PANEL_H };

})();
