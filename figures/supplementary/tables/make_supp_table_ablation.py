"""Build supplementary ablation table.

Single table that pools two ablation studies on the headline model
(RADIX = DINOv3-S/16 + DPT-meta, Strategy A, Bio-7 semantic):
  1. Encoder training strategy  (frozen vs fine-tuned)
  2. Loss function              (Dice + Focal + wCE  vs  + Lovasz)

For each variant we report sample-level mIoU (mean ± s.d.) on the
in-distribution test split and the Zeiss out-of-distribution split, computed
directly from each run's `eval/{split}/metrics_bio7.csv`.

Outputs (in this directory):
  supp_table_ablation.csv
  supp_table_ablation.md
  supp_table_ablation.html

Open the HTML via `python3 -m http.server` and click Save PNG / Save SVG.
"""

import csv
from pathlib import Path

# Reuse the shared helpers + style from the figure-2 supp-table builder.
from make_supp_tables_fig2 import (
    summary, fmt_mean_std, write_csv, write_markdown,
    render_table_svg, wrap_html,
)

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent / "eval"

# Ablation rows: (section, label, run-short-name, is_baseline). Short names
# map to subdirectories of supplementary/eval/.
ABLATIONS = [
    ("Encoder training strategy", "Frozen encoder",                                "ablation/frozen_encoder",  False),
    ("Encoder training strategy", "Fine-tuned encoder (RADIX)",                    "radix",                    True),
    ("Loss function",             "Dice + Focal + weighted CE",                    "ablation/loss_no_lovasz",  False),
    ("Loss function",             "Dice + Focal + weighted CE + Lovasz (RADIX)",   "radix",                    True),
]


def load_miou(run_rel: str, split: str) -> list[float]:
    """Read mean_IoU (sample-level, Bio-7) from a single run's split."""
    p = EVAL / run_rel / split / "metrics_bio7.csv"
    if not p.exists():
        print(f"  MISSING: {p}")
        return []
    out = []
    with open(p) as f:
        for r in csv.DictReader(f):
            v = r.get("mean_IoU", "")
            if v in ("", None) or str(v).lower() == "nan":
                continue
            try:
                out.append(float(v))
            except ValueError:
                pass
    return out


def build_rows():
    rows = []
    for section, label, run_rel, _is_base in ABLATIONS:
        test = load_miou(run_rel, "test")
        zs   = load_miou(run_rel, "oneshot")
        s_t  = summary(test)
        s_z  = summary(zs)
        rows.append({
            "Section":   section,
            "Variant":   label,
            "Test n":    s_t["n"],
            "Test mIoU": fmt_mean_std(s_t),
            "Zero n":    s_z["n"],
            "Zero mIoU": fmt_mean_std(s_z),
        })
    return rows


def main() -> None:
    rows = build_rows()

    # ── CSV / Markdown ──────────────────────────────────────────────
    fields = ["Section", "Variant",
              "Test n", "Test mIoU",
              "Zero n", "Zero mIoU"]
    csv_path = HERE / "supp_table_ablation.csv"
    md_path  = HERE / "supp_table_ablation.md"
    write_csv(rows, fields, csv_path)

    n_test = rows[0]["Test n"]
    n_zero = rows[0]["Zero n"]
    write_markdown(
        rows, fields, md_path,
        "Supplementary Table. RADIX ablation studies",
        f"Sample-level Bio-7 mIoU (mean ± s.d. across samples) for the "
        f"headline RADIX model under two ablations: encoder training "
        f"strategy (frozen vs end-to-end fine-tuned) and loss function "
        f"(with vs without the Lovasz term added to the Dice + Focal + "
        f"weighted-CE base). Each variant retrains the baseline "
        f"DINOv3-S/16 + DPT-meta architecture on Strategy A with only the "
        f"indicated component changed. Test n = {n_test} is the "
        f"in-distribution test split; Out-of-distribution n = {n_zero} is the "
        f"held-out Zeiss split. Whole Root is excluded from the mIoU "
        f"computation per project convention. Rows labelled \"RADIX\" "
        f"mark the recipe reported as RADIX throughout the manuscript.",
    )

    # ── HTML / SVG ──────────────────────────────────────────────────
    columns = [
        {"key": "Variant", "label": "Variant",
         "width_mm": 70, "align": "start"},
        {"key": "Test mIoU",
         "label": ["In-distribution Test mIoU",
                   f"(mean ± s.d., n = {n_test})"],
         "width_mm": 40, "align": "middle"},
        {"key": "Zero mIoU",
         "label": ["Out-of-distribution Test mIoU",
                   f"(mean ± s.d., n = {n_zero})"],
         "width_mm": 36, "align": "middle"},
    ]
    # Drop section column from per-row body; expose section breaks via
    # the booktabs-style section_dividers feature instead.
    sec_seen = set()
    section_dividers = []
    body_rows = []
    for r in rows:
        if r["Section"] not in sec_seen:
            section_dividers.append({"row": len(body_rows),
                                     "label": r["Section"]})
            sec_seen.add(r["Section"])
        body_rows.append({k: v for k, v in r.items() if k != "Section"})

    svg_inner, w_mm, h_mm = render_table_svg(
        columns, body_rows,
        group_headers=None,
        section_dividers=section_dividers,
        title=None, footnote=[],
    )
    html_path = HERE / "supp_table_ablation.html"
    html_path.write_text(wrap_html(
        svg_inner, w_mm, h_mm,
        "supp_table_ablation",
        "Supplementary Table: RADIX ablation studies",
    ))

    print(f"→ {csv_path.relative_to(HERE)}")
    print(f"→ {md_path.relative_to(HERE)}")
    print(f"→ {html_path.relative_to(HERE)}   viewBox = {w_mm:.1f} × {h_mm:.1f} mm")
    print("\nOpen the HTML via `python3 -m http.server` and click Save PNG / Save SVG.")


if __name__ == "__main__":
    main()
