/**
 * fig2f_scatter.js
 *
 * Shared renderer for the Figure 2f test-vs-oneshot mIoU scatter. Loaded by
 * both fig2f_benchmark.html (interactive builder with controls) and
 * assemble_figure2.html (final assembly). Both pages therefore render from
 * the same model list + colour map with no save/reload step.
 *
 * Usage:
 *   <svg id="fig2f" viewBox="0 0 67 57"></svg>
 *   <script src="d3.v7.min.js"></script>
 *   <script src="fig2f_scatter.js"></script>
 *   <script>
 *     Fig2fScatter.render(document.getElementById("fig2f"), {
 *       xMin: 0.80, xMax: 0.90, yMin: 0.870, yMax: 0.895,
 *       labelMode: "off",            // or "numbers"
 *       csvPath: "../model_runs_summary.csv",
 *     });
 *   </script>
 */
(function () {

const WANTED = [
  "dpt_meta_facebook_dinov3-vits16-pretrain-lvd1689m_equalw_drop_shuf_dfcel_semantic7c_A",
  "dpt_meta_vit_small_patch14_dinov2_equalw_drop_shuf_dfcel_semantic7c_A",
  "unetpp_resnet34_imagenet_equalw_drop_shuf_dfcel_semantic7c_A",
  "ms_linear_vit_small_patch14_dinov2_equalw_drop_shuf_dfcel_semantic7c_A",
  "segdino_mlp_facebook_dinov3-vits16-pretrain-lvd1689m_equalw_drop_shuf_dfcel_semantic7c_A",
  "unetplusplus_resnet50_imagenet_equalw_drop_shuf_dfcel_semantic7c_A",
  "unetr_sam_vit_b_lm_lora_r4_equalw_drop_shuf_dfcel_semantic7c_A",
  "yolo26m_coco_noweight_drop_shuf_ultralytics_instance6c_A",
];

// `csv` maps each run to its label in per_sample_miou_all_models.csv (the
// scatter's data source - see render()). The `label` field is the in-plot
// display string and may differ (shorter / no parenthesis suffix).
const MODEL_INFO = {
  "dpt_meta_facebook_dinov3-vits16-pretrain-lvd1689m_equalw_drop_shuf_dfcel_semantic7c_A":
    { csv: "DINOv3 + DPT-meta",      label: "RADIX (DINOv3+DPT)", color: "#4e79a7" },
  "dpt_meta_vit_small_patch14_dinov2_equalw_drop_shuf_dfcel_semantic7c_A":
    { csv: "DINOv2 + DPT-meta",      label: "DINOv2 + DPT",       color: "#f28e2b" },
  "unetpp_resnet34_imagenet_equalw_drop_shuf_dfcel_semantic7c_A":
    { csv: "ResNet34 + UNet++ (IN)", label: "ResNet34 + UNet++",  color: "#76b7b2" },
  "ms_linear_vit_small_patch14_dinov2_equalw_drop_shuf_dfcel_semantic7c_A":
    { csv: "DINOv2 + MS-Linear",     label: "DINOv2 + MS",        color: "#59a14f" },
  "segdino_mlp_facebook_dinov3-vits16-pretrain-lvd1689m_equalw_drop_shuf_dfcel_semantic7c_A":
    { csv: "DINOv3 + SegDINO-MLP",   label: "SegDINO",            color: "#b07aa1" },
  "unetplusplus_resnet50_imagenet_equalw_drop_shuf_dfcel_semantic7c_A":
    { csv: "ResNet50 + UNet++ (IN)", label: "ResNet50 + UNet++",  color: "#9c755f" },
  "unetr_sam_vit_b_lm_lora_r4_equalw_drop_shuf_dfcel_semantic7c_A":
    { csv: "MicroSAM + UNETR",       label: "MicroSAM + UNETR",   color: "#edc948" },
  "yolo26m_coco_noweight_drop_shuf_ultralytics_instance6c_A":
    { csv: "YOLO26m-seg (COCO)",     label: "YOLO26m",            color: "#ff9da7" },
};

// Headline mIoU = mean over samples of (per-sample mean over 6 anatomy
// classes, Whole Root excluded). This is what the eval pipeline reports
// and what per_sample_iou.csv / the violin / the microscope box plot all
// use - so the scatter loads the same per-sample file and averages by
// (model, split). Earlier versions used mean-of-per-class-macros from
// model_runs_summary.csv, which gave a different (lower) test mIoU
// whenever a class was NaN for some samples (e.g. Aerenchyma absent in
// tomato dropped test mIoU by ~0.012 across all 8 models).
function meanBySplit(rows) {
    const acc = new Map();   // "model||split" → {sum, n}
    for (const r of rows) {
        const v = +r.mean_IoU;
        if (!Number.isFinite(v)) continue;
        const k = r.model + "||" + r.split;
        const a = acc.get(k) || { sum: 0, n: 0 };
        a.sum += v; a.n += 1;
        acc.set(k, a);
    }
    return (model, split) => {
        const a = acc.get(model + "||" + split);
        return a && a.n > 0 ? a.sum / a.n : NaN;
    };
}

// Panel geometry (mm). Default 67 × 57; overridable via render({ panelW, panelH }).
// Host SVG viewBox must match whatever panelW/panelH the caller passes.
// M.left/M.bottom chosen to match fig2e so the two panels' y-axes align in
// the figure grid and title-to-axis distances are equal on both panels.
const DEFAULT_PANEL_W = 67, DEFAULT_PANEL_H = 57;
// M.top / M.bottom match panels d (species) and e (microscope) so the
// three row-2-bottom panels share the same x-axis and x-title canvas y.
// M.top = 3 - plot region flush with the panel top (chart title was
// removed so no room needed for it). M.bottom 13 hosts the rotated
// tick labels plus the 9.47 mm x-title offset.
const M = { top: 3, right: 4, bottom: 13, left: 8 };
// LABEL_TITLE_GAP = 0.5 mm is the canonical gap between any axis tick-label
// and its axis-title (matches the 2d column-label → image gap). Axis-tick
// labels render at 6 pt (2.12 mm), axis-titles at 7 pt (2.47 mm):
//   x-title baseline (canonical, horizontal labels)
//                    = axis + 3 + 0.4 + 0.5 + 1.98 = axis + 5.88
//   y-title centre   = axis − (1.2 + W_tick + GAP + title_half_thickness(7pt))
//                    = axis − (1.2 + 4.3 + 0.5 + 1.235) = axis − 7.24  (4-char tick labels)
// However, 2f's x-title is forced to match 2d's offset (7.47) so that the
// "Out-of-distribution mIoU" and "Microscope" titles share the same canvas baseline.
// 2d's larger offset comes from its −25° rotated tick labels needing extra
// vertical room - 2f's horizontal tick labels just have ~1.6 mm of extra
// whitespace between them and the title.
const LABEL_TITLE_GAP = 0.5;
const TITLE_FONT_MM = 2.47;                 // 7 pt
const X_TITLE_BASELINE_OFFSET = 9.47;       // matches panels d and e - aligns x-titles at same canvas y
// 4-char tick labels ("0.78") at 6 pt ≈ 4.7 mm wide → y-title centre at
// axis − (1.2 + 4.7 + 0.5 + 1.235) = axis − 7.635 mm for single-line.
// Bumped to 9.0 mm because the y-title is broken into two lines
// ("Out-of-distribution" / "Test mIoU") - the extra line adds ~1.36 mm of
// thickness on the inner side that would otherwise crowd the tick labels.
const Y_TITLE_CENTRE_OFFSET   = 9.0;      // from y-axis line


/**
 * Fetch the CSV once and memoise so concurrent callers share data.
 */
const _csvCache = new Map();
async function _loadCsv(path) {
    if (!_csvCache.has(path)) {
        _csvCache.set(path, d3.csv(path + "?t=" + Date.now()));
    }
    return _csvCache.get(path);
}


async function render(svgEl, opts) {
    const {
        xMin = 0.82, xMax = 0.88,
        yMin = 0.78, yMax = 0.88,
        labelMode = "off",
        csvPath = "per_sample_miou_all_models.csv",
        panelW = DEFAULT_PANEL_W,
        panelH = DEFAULT_PANEL_H,
        title = "",
    } = opts || {};
    const PANEL_W = panelW;
    const PANEL_H = panelH;
    const PLOT_W = PANEL_W - M.left - M.right;
    const PLOT_H = PANEL_H - M.top - M.bottom;

    const raw = await _loadCsv(csvPath);
    const lookup = meanBySplit(raw);
    const rows = WANTED.map((n, i) => {
        const info = MODEL_INFO[n];
        if (!info) return null;
        return {
            idx: i + 1, run: n, label: info.label, color: info.color,
            test:    lookup(info.csv, "test"),
            oneshot: lookup(info.csv, "zero-shot"),
        };
    }).filter(x => Number.isFinite(x.test) && Number.isFinite(x.oneshot));

    const svg = d3.select(svgEl);
    svg.selectAll("*").remove();

    // Ensure host SVG has Helvetica (paper-wide font convention)
    svg.attr("font-family", "Helvetica, Arial, sans-serif");

    // Optional panel title - 6 pt bold, centred above the plot in the existing
    // M.top region. Panel-letter sits top-left; centred title clears it.
    if (title) {
        svg.append("text")
            .attr("x", PANEL_W / 2).attr("y", 3.0)
            .attr("text-anchor", "middle")
            .attr("font-size", 2.12).attr("font-weight", "bold")
            .text(title);
    }

    const x = d3.scaleLinear().domain([xMin, xMax]).range([0, PLOT_W]);
    const y = d3.scaleLinear().domain([yMin, yMax]).range([PLOT_H, 0]);

    const g = svg.append("g").attr("transform", `translate(${M.left},${M.top})`);

    // Grid
    const xTicks = x.ticks(5), yTicks = y.ticks(5);
    g.append("g").selectAll("line.vg").data(xTicks).enter().append("line")
        .attr("x1", d => x(d)).attr("x2", d => x(d))
        .attr("y1", 0).attr("y2", PLOT_H)
        .attr("stroke", "#e4e4e4").attr("stroke-width", 0.1);
    g.append("g").selectAll("line.hg").data(yTicks).enter().append("line")
        .attr("x1", 0).attr("x2", PLOT_W)
        .attr("y1", d => y(d)).attr("y2", d => y(d))
        .attr("stroke", "#e4e4e4").attr("stroke-width", 0.1);

    // y = x reference line (grey dashed) - clipped to the intersection of
    // the x-domain and y-domain so it never extends past the plot box.
    const lineLo = Math.max(xMin, yMin);
    const lineHi = Math.min(xMax, yMax);
    if (lineLo < lineHi) {
        g.append("line")
            .attr("x1", x(lineLo)).attr("y1", y(lineLo))
            .attr("x2", x(lineHi)).attr("y2", y(lineHi))
            .attr("stroke", "#aaa").attr("stroke-width", 0.2)
            .attr("stroke-dasharray", "0.8 0.6");
        // Label centred on the line midpoint, rotated to match the line
        // angle, offset ~1 mm perpendicular ABOVE the line so it sits in
        // the upper-triangle region (most data points have oneshot < test
        // and therefore live BELOW the line). Italic 5 pt grey.
        const midVal = (lineLo + lineHi) / 2;
        const mx = x(midVal);
        const my = y(midVal);
        const angleDeg = Math.atan2(y(lineHi) - y(lineLo),
                                    x(lineHi) - x(lineLo)) * 180 / Math.PI;
        g.append("text")
            .attr("transform", `translate(${mx},${my}) rotate(${angleDeg})`)
            .attr("x", 0).attr("y", -0.875)
            .attr("text-anchor", "middle")
            .attr("font-size", 1.76)
            .attr("font-style", "italic")
            .attr("fill", "#888")
            .text("Out-of-distribution mIoU = Test mIoU");
    }

    // Axes (slightly thicker than tick marks)
    g.append("line").attr("x1", 0).attr("x2", PLOT_W)
        .attr("y1", PLOT_H).attr("y2", PLOT_H)
        .attr("stroke", "black").attr("stroke-width", 0.2);
    g.append("line").attr("x1", 0).attr("x2", 0)
        .attr("y1", 0).attr("y2", PLOT_H)
        .attr("stroke", "black").attr("stroke-width", 0.2);

    g.selectAll(".xt").data(xTicks).enter().append("g").each(function (d) {
        const xx = x(d);
        d3.select(this).append("line")
            .attr("x1", xx).attr("x2", xx)
            .attr("y1", PLOT_H).attr("y2", PLOT_H + 0.8)
            .attr("stroke", "black").attr("stroke-width", 0.2);
        d3.select(this).append("text")
            .attr("x", xx).attr("y", PLOT_H + 3)
            .attr("text-anchor", "middle").attr("font-size", 2.12)
            .text(d3.format(".3f")(d));
    });
    g.selectAll(".yt").data(yTicks).enter().append("g").each(function (d) {
        const yy = y(d);
        d3.select(this).append("line")
            .attr("x1", -0.8).attr("x2", 0)
            .attr("y1", yy).attr("y2", yy)
            .attr("stroke", "black").attr("stroke-width", 0.2);
        d3.select(this).append("text")
            .attr("x", -1.2).attr("y", yy + 0.7)
            .attr("text-anchor", "end").attr("font-size", 2.12)
            .text(d3.format(".3f")(d));
    });

    // X-title (7 pt bold): 0.5 mm below the x-tick labels
    svg.append("text")
        .attr("x", M.left + PLOT_W / 2)
        .attr("y", M.top + PLOT_H + X_TITLE_BASELINE_OFFSET)
        .attr("text-anchor", "middle").attr("font-size", TITLE_FONT_MM)
        .attr("font-weight", "bold")
        .text("In-distribution Test mIoU");
    // Y-title (7 pt bold): 0.5 mm left of the y-tick labels. Two lines
    // because "Out-of-distribution Test mIoU" is too long for one row along
    // the y-axis. tspan#1 sits further from the axis (top when reading), #2
    // closer to the axis (bottom when reading).
    const yTitle = svg.append("text")
        .attr("transform",
              `translate(${M.left - Y_TITLE_CENTRE_OFFSET},${M.top + PLOT_H / 2}) rotate(-90)`)
        .attr("text-anchor", "middle").attr("font-size", TITLE_FONT_MM)
        .attr("font-weight", "bold");
    yTitle.append("tspan").attr("x", 0).attr("dy", "-0.55em").text("Out-of-distribution");
    yTitle.append("tspan").attr("x", 0).attr("dy", "1.1em").text("Test mIoU");

    const pts = rows.map(d => ({ ...d, px: x(d.test), py: y(d.oneshot) }));

    // Data markers - small filled circles (no stroke), labels render right
    // next to each point so we don't need a separate legend.
    const MARKER_R = 0.6;                    // smaller marker
    g.selectAll(".pt").data(pts).enter().append("circle")
        .attr("cx", d => d.px).attr("cy", d => d.py)
        .attr("r", MARKER_R)
        .attr("fill", d => d.color);

    // Per-point inline labels: model name placed to the right of each marker
    // by default; flipped to the LEFT if the right-side label would extend
    // past the right plot edge. "RADIX (DINOv3+DPT)" rendered bold as the headline.
    const LABEL_FS = 2.47;                   // 7 pt
    const LABEL_GAP = 0.7;                   // mm between marker edge and text
    const BOLD_LABEL = "RADIX (DINOv3+DPT)";

    // Measure label widths for placement-flip logic
    const measure = svg.append("g").attr("visibility", "hidden");
    const labelWidth = pts.map(p => {
        const t = measure.append("text")
            .attr("font-size", LABEL_FS).text(p.label);
        if (p.label === BOLD_LABEL) t.attr("font-weight", "bold");
        return t.node().getComputedTextLength();
    });
    measure.remove();

    pts.forEach((p, i) => {
        const w = labelWidth[i];
        const flipLeft = p.px + MARKER_R + LABEL_GAP + w > PLOT_W - 0.5;
        const xText = flipLeft
            ? p.px - MARKER_R - LABEL_GAP
            : p.px + MARKER_R + LABEL_GAP;
        const t = g.append("text")
            .attr("x", xText)
            .attr("y", p.py + LABEL_FS * 0.32)   // vertical center on the marker
            .attr("text-anchor", flipLeft ? "end" : "start")
            .attr("font-size", LABEL_FS)
            .attr("fill", p.color)               // match marker color
            .text(p.label);
        if (p.label === BOLD_LABEL) t.attr("font-weight", "bold");
    });

    return { rows, pts };
}


// Expose
window.Fig2fScatter = { render, WANTED, MODEL_INFO,
                        PANEL_W: DEFAULT_PANEL_W, PANEL_H: DEFAULT_PANEL_H };

})();
