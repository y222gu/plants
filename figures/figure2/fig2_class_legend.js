/**
 * fig2_class_legend.js
 *
 * Shared Bio-7 semantic-map legend used by Figure 2's bottom-right
 * sub-grid. Colours match the canonical Figure 1a palette.
 *
 * Layout: swatches LEFT-aligned to the panel; class-name labels sit just
 * to the right of each swatch and are LEFT-aligned. Title sits above the
 * swatches column, also left-aligned.
 */
(function () {

// Order matches fig2b box order (top-to-bottom = left-to-right in 2b):
// Vascular → Exodermis → Endodermis → Cortex → Epidermis → Aerenchyma.
const CLASSES = [
    { name: "Vascular",   color: "#e76f61" },
    { name: "Exodermis",  color: "#f4a261" },
    { name: "Endodermis", color: "#f6e48e" },
    { name: "Cortex",     color: "#94d2bd" },
    { name: "Epidermis",  color: "#0a9396" },
    { name: "Aerenchyma", color: "#264653" },
];

function render(svgEl, opts) {
    const {
        panelW, panelH,
        titleText = "Anatomical Class",
        // Optional: bottom-align the visible block (title + 6 rows) so the
        // last row's baseline lands at this panel y. Used by assemble_figure2
        // to align the legend's bottom with the x-axis of fig2c.
        contentBottom = null,
        // Optional: top-align the visible block so the title's top edge sits
        // at this panel y. Mutually exclusive with contentBottom; takes
        // priority if both are supplied.
        contentTop = null,
    } = opts || {};
    const PANEL_W = panelW, PANEL_H = panelH;

    const svg = d3.select(svgEl);
    svg.selectAll("*").remove();
    svg.attr("font-family", "Helvetica, Arial, sans-serif");

    const titleFS = 2.47;                  // 7 pt
    const rowFS   = 2.47;                  // 7 pt (all legend text)
    const swatchSide = 2.4;
    const rowPitch   = 3.3;
    const titleGap   = 1.5;
    const blockH = titleFS + titleGap + CLASSES.length * rowPitch;
    // Title baseline. contentTop wins if supplied (title top edge = contentTop),
    // otherwise contentBottom (last row baseline = contentBottom), otherwise
    // vertically centre the title + 6 rows.
    const y0 = (contentTop != null)
        ? contentTop + titleFS
        : (contentBottom != null)
            ? contentBottom - blockH + titleFS
            : (PANEL_H - blockH) / 2 + titleFS;
    // Swatches LEFT-aligned to panel. Text sits to the RIGHT of each
    // swatch with a small gap, also left-aligned.
    const leftPad = 0;
    const swatchTextGap = 1.0;
    const swatchLeftX  = leftPad;
    const textLeftX    = swatchLeftX + swatchSide + swatchTextGap;

    // Title - left-aligned, sits above the swatches column.
    svg.append("text")
        .attr("x", swatchLeftX).attr("y", y0)
        .attr("text-anchor", "start")
        .attr("font-size", titleFS).attr("font-weight", "bold")
        .text(titleText);

    // Rows - swatch on the LEFT, text left-aligned right of the swatch.
    let y = y0 + titleGap + rowFS;
    for (const cls of CLASSES) {
        svg.append("rect")
            .attr("x", swatchLeftX)
            .attr("y", y - swatchSide + 0.2)
            .attr("width", swatchSide).attr("height", swatchSide)
            .attr("fill", cls.color);
        svg.append("text")
            .attr("x", textLeftX).attr("y", y)
            .attr("text-anchor", "start")
            .attr("font-size", rowFS)
            .text(cls.name);
        y += rowPitch;
    }
}

window.Fig2ClassLegend = { render, CLASSES };

})();
