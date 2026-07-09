"""Build supplementary tables for Figure 2.

Source CSVs (in this directory):
  per_sample_iou.csv                : model per-class IoU, panels b/c/d
  human_baseline_iou.csv            : annotator-vs-annotator baseline, panel b
  per_sample_miou_by_microscope.csv : per-sample mIoU by microscope, panel e
  per_sample_miou_all_models.csv    : all-model benchmark, panel f
  diversity_counts.csv              : UID -> species/genotype/split

Headline model: RADIX (Root Anatomy Deep Identification across species),
DINOv3-S/16 + DPT-meta. Whole Root is excluded from all per-class and mIoU
summaries per project convention.

HTML tables follow paper conventions: Helvetica, mm-sized SVG, 600 dpi PNG export.
Open the HTML via `python3 -m http.server` from this directory (or any parent that
contains it), then click Save PNG / Save SVG to export the figure.
"""

import csv
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIVERSITY_CSV = HERE.parent / "diversity_counts.csv"


def _normalize_uid(s: str) -> str:
    """Collapse runs of underscores to a single underscore. Source sample IDs
    occasionally use '__' where the spreadsheet uses '_'."""
    import re
    return re.sub(r"_+", "_", s) if s else s


def _cell_to_str(v) -> str:
    """Convert an openpyxl cell value to a clean display string. Whole-number
    floats lose their trailing '.0' (e.g. accession ID 235464.0 → '235464').
    """
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def load_genotype_map() -> dict:
    if not DIVERSITY_CSV.exists():
        return {}
    m = {}
    with open(DIVERSITY_CSV) as f:
        for r in csv.DictReader(f):
            uid, gen = r.get("uid", ""), r.get("genotype", "")
            if uid and gen:
                m[_normalize_uid(uid)] = gen
    return m

# Display order matches the panels: vascular → exodermis → endodermis → cortex
# → epidermis → aerenchyma (Fig 2b/c). Panel d order is Rice → Millet → Sorghum
# → Solanums; panel e is Olympus → C10 → Zeiss.
CLASSES = ["Vascular", "Exodermis", "Endodermis", "Cortex", "Epidermis", "Aerenchyma"]
SPECIES_ORDER = ["Rice", "Millet", "Sorghum", "Solanum"]
SPECIES_LABEL = {"Rice": "Rice", "Millet": "Millet",
                 "Sorghum": "Sorghum", "Solanum": "Solanums"}
MICROSCOPE_ROWS = [
    ("Olympus", "test",      "Olympus"),
    ("C10",     "test",      "C10"),
    ("Zeiss",   "zero-shot", "Zeiss"),
]


def floats(rows: list[dict], col: str) -> list[float]:
    out = []
    for r in rows:
        v = r.get(col, "")
        if v == "" or v.lower() == "nan":
            continue
        out.append(float(v))
    return out


def summary(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "mean": None, "std": None,
                "median": None, "q1": None, "q3": None}
    s = sorted(vals)
    return {
        "n": len(s),
        "mean": statistics.fmean(s),
        "std": statistics.pstdev(s) if len(s) > 1 else 0.0,
        "median": statistics.median(s),
        "q1": statistics.quantiles(s, n=4)[0] if len(s) > 1 else s[0],
        "q3": statistics.quantiles(s, n=4)[2] if len(s) > 1 else s[0],
    }


def fmt_mean_std(s: dict) -> str:
    if s["n"] == 0:
        return "n/a"
    return f"{s['mean']:.3f} ± {s['std']:.3f}"


def load_csv(p: Path) -> list[dict]:
    with open(p) as f:
        return list(csv.DictReader(f))


def build_per_class_table() -> tuple[list[dict], list[str]]:
    """Panels b (in-distribution test) and c (Zeiss zero-shot).

    Mean ± std of IoU across samples, per class, for RADIX on the in-distribution
    test split and the Zeiss zero-shot split.
    """
    model_rows = load_csv(HERE.parent / "per_sample_iou.csv")

    test = [r for r in model_rows if r["split"] == "test"]
    zshot = [r for r in model_rows if r["split"] == "zero-shot"]

    out = []
    for c in CLASSES:
        test_iou  = summary(floats(test,  f"{c}_IoU"))
        zs_iou    = summary(floats(zshot, f"{c}_IoU"))
        out.append({
            "Anatomical Class": c,
            "In-distribution Test n":                test_iou["n"],
            "In-distribution Test IoU (mean ± std)": fmt_mean_std(test_iou),
            "Zero-shot Test n":                      zs_iou["n"],
            "Zero-shot Test IoU (mean ± std)":       fmt_mean_std(zs_iou),
        })
    fields = list(out[0].keys())
    return out, fields


# Per-model architectural metadata for the panel f supplementary table.
# Parameter counts (in millions) verified by instantiating each model and
# summing parameters under encoder vs. decoder submodules. MicroSAM uses
# LoRA r=4, so only LoRA adapter params are fine-tuned; all other models
# fine-tune their encoder end-to-end (ft_encoder_M == encoder_M).
# Encoder/pretrained-dataset labels follow notes/_make_results_docx.py.
MODEL_META = {
    #                          encoder         pretrained      enc_M ft_enc_M dec_M  total_M
    "DINOv3 + DPT-meta":       ("DINOv3-S/16", "LVD-1.69B",     21.6, 21.6,    19.3,  40.9),
    "DINOv2 + DPT-meta":       ("DINOv2-S/14", "LVD-142M",      23.6, 23.6,    19.3,  42.9),
    "ResNet34 + UNet++ (IN)":  ("ResNet34",    "ImageNet-1k",   21.3, 21.3,     4.8,  26.1),
    "DINOv2 + MS-Linear":      ("DINOv2-S/14", "LVD-142M",      23.6, 23.6,     0.01, 23.7),
    "DINOv3 + SegDINO-MLP":    ("DINOv3-S/16", "LVD-1.69B",     21.6, 21.6,     1.0,  22.6),
    "ResNet50 + UNet++ (IN)":  ("ResNet50",    "ImageNet-1k",   23.5, 23.5,    25.5,  49.0),
    "MicroSAM + UNETR":        ("SAM ViT-B",   "SA-1B + LM",    89.7,  0.2,    99.7, 189.4),
    "YOLO26m-seg (COCO)":      ("YOLO26m",     "COCO",          10.4, 10.4,    16.7,  27.1),
}


def build_panel_f_table() -> tuple[list[dict], list[str]]:
    """Panel f: per-model sample-level mIoU on in-distribution test and
    Zeiss zero-shot test, summarised as mean ± std across samples, plus
    total parameter count per model.

    Reads ../figure2/per_sample_miou_all_models.csv (the same file the
    fig2f scatter renders from) so the table and the panel agree.
    Rows are sorted by in-distribution test mIoU descending (best first).
    """
    csv_path = HERE.parent / "per_sample_miou_all_models.csv"
    rows = load_csv(csv_path)

    # Group by (model, split).
    per_ms: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        v = r.get("mean_IoU", "")
        if v in ("", None) or v.lower() == "nan":
            continue
        try:
            f = float(v)
        except ValueError:
            continue
        per_ms.setdefault((r["model"], r["split"]), []).append(f)

    models = sorted({m for (m, _s) in per_ms})

    def _ms(x):
        return "n/a" if x is None else (f"{x:.2f}" if x < 1 else f"{x:.1f}")

    import re
    def _clean_model(s: str) -> str:
        # Drop trailing " (XYZ)" since the pretrained dataset already
        # carries that info in its own column, strip the "-meta"
        # qualifier from DPT variants, and shorten the SegDINO row since
        # SegDINO is itself a DINOv3-based pipeline.
        s = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()
        s = s.replace("-meta", "")
        s = s.replace("DINOv3 + SegDINO-MLP", "SegDINO-MLP")
        # The DINOv3 + DPT model is branded as RADIX in paper-facing
        # figures and tables.
        if s == "DINOv3 + DPT":
            s = "RADIX"
        return s

    out = []
    for m in models:
        test = per_ms.get((m, "test"), [])
        zs   = per_ms.get((m, "zero-shot"), [])
        s_test = summary(test)
        s_zs   = summary(zs)
        meta = MODEL_META.get(m, (None, None, None, None, None, None))
        enc_name, pretrained, enc_M, ft_enc_M, dec_M, total_M = meta
        out.append({
            "Model":                                   _clean_model(m),
            "Encoder":                                 enc_name or "n/a",
            "Pretrained Dataset":                      pretrained or "n/a",
            "Fine-tuned Encoder (M)":                  _ms(ft_enc_M),
            "Decoder (M)":                             _ms(dec_M),
            "Total Params (M)":                        _ms(total_M),
            "In-distribution Test n":                  s_test["n"],
            "In-distribution Test mIoU (mean ± std)":  fmt_mean_std(s_test),
            "_t_sort":                                 s_test["mean"] or 0.0,
            "Zero-shot Test n":                        s_zs["n"],
            "Zero-shot Test mIoU (mean ± std)":        fmt_mean_std(s_zs),
        })
    out.sort(key=lambda r: r["_t_sort"], reverse=True)
    for r in out:
        del r["_t_sort"]
    fields = list(out[0].keys())
    return out, fields


def build_panel_f_html_svg(rows_f: list[dict]) -> tuple[str, float, float]:
    """Compact Nature-style table for panel f: per-model sample-level mIoU
    on in-distribution test and Zeiss zero-shot test."""
    columns = [
        {"key": "Model",      "label": "Model",
         "width_mm": 32, "align": "start"},
        {"key": "Encoder",    "label": "Encoder",
         "width_mm": 22, "align": "start"},
        {"key": "Pretrained", "label": ["Pretrained", "Dataset"],
         "width_mm": 24, "align": "start"},
        {"key": "t_iou",
         "label": ["In-distribution Test mIoU", "(mean ± s.d., n = 185)"],
         "width_mm": 38, "align": "middle"},
        {"key": "z_iou",
         "label": ["Zero-shot Test mIoU", "(mean ± s.d., n = 35)"],
         "width_mm": 32, "align": "middle"},
    ]
    f_rows = []
    for r in rows_f:
        f_rows.append({
            "Model":      r["Model"],
            "Encoder":    r["Encoder"],
            "Pretrained": r["Pretrained Dataset"],
            "t_iou":      r["In-distribution Test mIoU (mean ± std)"],
            "z_iou":      r["Zero-shot Test mIoU (mean ± std)"],
        })
    return render_table_svg(columns, f_rows,
                            group_headers=None,
                            title=None, footnote=[])


def build_per_class_by_species_table() -> tuple[list[dict], list[str]]:
    """Per-class IoU broken down by species for RADIX. Rice pools both test
    sets (in-distribution + Zeiss zero-shot), matching the species block of
    the panel d/e table; the other species use the in-distribution test
    split only. Aerenchyma is computed only over samples in which the class
    is present (Solanums lack the class entirely and show n/a).
    """
    model_rows = load_csv(HERE.parent / "per_sample_iou.csv")
    test = [r for r in model_rows if r["split"] == "test"]

    out = []
    for sp in SPECIES_ORDER:
        if sp == "Rice":
            sub = [r for r in model_rows if r["species"] == sp]
            suffix = " (from both test sets)"
        else:
            sub = [r for r in test if r["species"] == sp]
            suffix = " (from in-distribution test)"
        row = {
            "Species": SPECIES_LABEL[sp] + suffix,
            "n":       len(sub),
        }
        for c in CLASSES:
            s = summary(floats(sub, f"{c}_IoU"))
            row[f"{c} IoU (mean ± std)"] = fmt_mean_std(s)
        out.append(row)
    fields = list(out[0].keys())
    return out, fields


def build_per_class_by_species_html_svg(rows: list[dict]) -> tuple[str, float, float]:
    """Nature-style table: rows = species, columns = anatomy class IoUs."""
    columns = [
        {"key": "Species", "label": "Species",    "width_mm": 40, "align": "start"},
        {"key": "n",       "label": "n",          "width_mm": 10, "align": "middle"},
        {"key": "Vascular",   "label": "Vascular",   "width_mm": 18, "align": "middle"},
        {"key": "Exodermis",  "label": "Exodermis",  "width_mm": 18, "align": "middle"},
        {"key": "Endodermis", "label": "Endodermis", "width_mm": 18, "align": "middle"},
        {"key": "Cortex",     "label": "Cortex",     "width_mm": 18, "align": "middle"},
        {"key": "Epidermis",  "label": "Epidermis",  "width_mm": 18, "align": "middle"},
        {"key": "Aerenchyma", "label": "Aerenchyma", "width_mm": 18, "align": "middle"},
    ]
    body_rows = []
    for r in rows:
        d = {"Species": r["Species"], "n": r["n"]}
        for c in CLASSES:
            d[c] = r[f"{c} IoU (mean ± std)"]
        body_rows.append(d)
    return render_table_svg(columns, body_rows,
                            group_headers=None,
                            title=None, footnote=[])


def build_diversity_table() -> tuple[list[dict], list[str]]:
    """Per-genotype sample counts across the four data splits, read from
    figure1/Diversity counts.xlsx (the same workbook that supplies the
    annotator-baseline genotype labels). One row per (Species, genotype)
    pair, sorted by species in the canonical paper order (Millet, Rice,
    Sorghum, Solanum lycopersicum, then the wild Solanum species
    alphabetically), then by genotype.

    Species names are written with underscores replaced by spaces. Solanum
    species are wrapped in `*...*` for italic binomial display, as is
    standard for binomial nomenclature. Genotypes that are all-lowercase
    letters/digits/hyphens are likewise italicised (gene/allele convention);
    the redundant `mutant` qualifier is stripped from those labels.
    """
    import re
    if not DIVERSITY_CSV.exists():
        return [], []

    from collections import defaultdict
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"train": 0, "val": 0, "test": 0, "oneshot": 0})
    with open(DIVERSITY_CSV) as f:
        for r in csv.DictReader(f):
            sp = r.get("species", "")
            gen_s = r.get("genotype", "")
            spl_s = r.get("split", "").strip().lower()
            if not sp:
                continue
            if spl_s in counts[(sp, gen_s)]:
                counts[(sp, gen_s)][spl_s] += 1

    # Species ordering: monocots (Millet, Rice, Sorghum) then dicots; among
    # Solanum, the cultivated tomato S. lycopersicum first, then the wild
    # species alphabetically.
    cultivated = ["Millet", "Rice", "Sorghum", "Solanum_lycopersicum"]
    wild_solanum = sorted({
        sp for (sp, _g) in counts
        if sp.startswith("Solanum_") and sp != "Solanum_lycopersicum"
    })
    species_order = cultivated + wild_solanum
    sp_rank = {s: i for i, s in enumerate(species_order)}

    def _format_species(sp_raw: str) -> str:
        if sp_raw.startswith("Solanum_"):
            return "*" + sp_raw.replace("_", " ") + "*"
        return sp_raw

    def _format_genotype(g_raw: str) -> str:
        if not g_raw:
            return "-"
        g = re.sub(r"[_\s]*mutants?[_\s]*", " ", g_raw).strip()
        # Paper-facing spelling fix (see project_kitaake_spelling memory).
        g = g.replace("Kittake", "Kitaake")
        # Italicise gene/allele names: all-lowercase letters/digits/hyphens
        # with at least one letter (pure-digit accession IDs stay upright).
        if (g and g == g.lower()
                and re.fullmatch(r"[a-z0-9\-]+", g)
                and re.search(r"[a-z]", g)):
            return f"*{g}*"
        return g

    out = []
    for (sp, g) in sorted(counts.keys(),
                          key=lambda x: (sp_rank.get(x[0], 999), x[1].lower())):
        c = counts[(sp, g)]
        total = c["train"] + c["val"] + c["test"] + c["oneshot"]
        out.append({
            "Species":   _format_species(sp),
            "Genotype":  _format_genotype(g),
            "Train":     c["train"],
            "Val":       c["val"],
            "Test":      c["test"],
            "Zero-shot": c["oneshot"],
            "Total":     total,
        })
    fields = list(out[0].keys()) if out else []
    return out, fields


def _diversity_single_split_columns(header: str) -> list[dict]:
    return [
        {"key": "Species",  "label": "Species",  "width_mm": 44, "align": "start"},
        {"key": "Genotype", "label": "Genotype", "width_mm": 32, "align": "start"},
        {"key": "n",        "label": header,    "width_mm": 14, "align": "middle"},
    ]


def _build_diversity_single_split_html_svg(
        rows: list[dict], header: str,
) -> tuple[str, float, float]:
    return render_table_svg(_diversity_single_split_columns(header), rows,
                            group_headers=None, title=None, footnote=[])


def _collapse_repeated_groups(rows: list[dict], key: str = "Species") -> list[dict]:
    """Return a copy of `rows` where each consecutive run with the same
    `key` value is treated as a group: the value is shown only on the
    middle row of the group (so it sits vertically centred between the
    group's first and last row) and a thin rule is drawn above each new
    group (via the `_separator` marker that render_table_svg interprets).
    """
    if not rows:
        return rows
    out = [dict(r) for r in rows]
    # Identify group spans.
    groups: list[tuple[int, int, object]] = []  # (start, end_inclusive, value)
    start = 0
    cur = out[0].get(key)
    for i in range(1, len(out)):
        v = out[i].get(key)
        if v != cur:
            groups.append((start, i - 1, cur))
            start = i
            cur = v
    groups.append((start, len(out) - 1, cur))
    # Blank every row's group cell, then write the value back at the
    # middle row of each group. Mark a separator on each non-first group.
    for r in out:
        r[key] = ""
    for gi, (s, e, val) in enumerate(groups):
        mid = (s + e) // 2
        out[mid][key] = val
        if gi > 0:
            out[s]["_separator"] = True
    return out


def _half_split_with_species_snap(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split rows in two roughly equal halves. If a species transition sits
    within three rows of the geometric midpoint, snap to it so each half
    begins / ends on a clean species boundary; otherwise just split at N//2.
    """
    n = len(rows)
    if n <= 1:
        return rows, []
    mid = n // 2
    boundaries = [i for i in range(1, n)
                  if rows[i]["Species"] != rows[i - 1]["Species"]]
    if boundaries:
        nearest = min(boundaries, key=lambda i: abs(i - mid))
        if abs(nearest - mid) <= 3:
            mid = nearest
    return rows[:mid], rows[mid:]


# build_diversity_val_html_svg removed - main() now calls
# _build_diversity_single_split_html_svg directly with already-filtered rows.


def build_diversity_train_parallel_html_svg(
        rows_a: list[dict], rows_b: list[dict]
) -> tuple[str, float, float]:
    """Render the two halves of the training-set table (train + val) side
    by side in one SVG canvas. Each sub-table has Species, Genotype, Train,
    Val columns; the two combined widths plus an inter-table gutter sit
    just under the 180 mm canvas cap.
    """
    cols = [
        {"key": "Species",  "label": "Species",  "width_mm": 32, "align": "start"},
        {"key": "Genotype", "label": "Genotype", "width_mm": 24, "align": "start"},
        {"key": "Train",    "label": "Train",    "width_mm": 10, "align": "middle"},
        {"key": "Val",      "label": "Val",      "width_mm": 10, "align": "middle"},
    ]
    svg_a, wa, ha = render_table_svg(cols, rows_a,
                                     group_headers=None, title=None, footnote=[])
    svg_b, wb, hb = render_table_svg(cols, rows_b,
                                     group_headers=None, title=None, footnote=[])
    gap_mm = 6.0
    combined = (
        f'<g transform="translate(0, 0)">\n{svg_a}\n</g>\n'
        f'<g transform="translate({wa + gap_mm:.3f}, 0)">\n{svg_b}\n</g>'
    )
    return combined, wa + gap_mm + wb, max(ha, hb)


def build_diversity_test_html_svg(test_rows: list[dict]) -> tuple[str, float, float]:
    """Test half of the dataset-composition table. Rows must already be the
    pre-filtered (Species, Genotype, In-distribution Test, Zero-shot Test,
    Total) shape produced in main()."""
    columns = [
        {"key": "Species",   "label": "Species",                            "width_mm": 44, "align": "start"},
        {"key": "Genotype",  "label": "Genotype",                           "width_mm": 32, "align": "start"},
        {"key": "In-distribution Test", "label": ["In-distribution", "Test"], "width_mm": 22, "align": "middle"},
        {"key": "Zero-shot Test",       "label": ["Zero-shot", "Test"],     "width_mm": 16, "align": "middle"},
        {"key": "Total",     "label": "Total",                              "width_mm": 14, "align": "middle"},
    ]
    return render_table_svg(columns, test_rows,
                            group_headers=None,
                            title=None, footnote=[])


def build_annotator_table() -> tuple[list[dict], list[str]]:
    """Inter-annotator agreement table: per-class IoU for each of the 16
    re-annotated samples, paired side-by-side with the corresponding RADIX
    model prediction IoU on the same sample. One row per sample.

    Sources:
      ../figure2/human_baseline_iou.csv : second-annotator vs primary GT IoU
      ../figure2/per_sample_iou.csv     : RADIX prediction vs primary GT IoU
                                          (lookup by sample_id)

    Splits the original sample_id into Species, Microscope, and Genotype.
    Genotype labels come from figure1/Diversity counts.xlsx (column
    'genotype'); falls back to a heuristic if a sample is missing. Gene
    names that are all-lowercase letters/digits/hyphens are wrapped in
    `*...*` markers so the renderer typesets them in italic.

    Sample-level mIoU is the unweighted mean of the available per-class
    IoUs (Aerenchyma is skipped where the class is absent from the GT).
    """
    import re

    human_rows = load_csv(HERE.parent / "human_baseline_iou.csv")
    model_rows = load_csv(HERE.parent / "per_sample_iou.csv")
    model_by_sid = {r["sample_id"]: r for r in model_rows}
    genotype_map = load_genotype_map()

    def _parse(v):
        if v in ("", None) or (isinstance(v, str) and v.lower() == "nan"):
            return None
        try:
            return float(v)
        except ValueError:
            return None

    out = []
    for r in human_rows:
        sp = r.get("species", "")
        mic = r.get("microscope", "")
        sid = r.get("sample_id", "")
        # Strip the "{raw_species}_{microscope}_" prefix.
        raw_prefix = f"{sid.split('_', 1)[0]}_{mic}_"
        tail = sid[len(raw_prefix):] if sid.startswith(raw_prefix) else sid
        # Prefer the curated genotype from the diversity spreadsheet; fall
        # back to a heuristic stripped sample identifier if not found.
        genotype = genotype_map.get(_normalize_uid(sid))
        if not genotype:
            genotype = re.sub(r"^Exp\d+_", "", tail)
            genotype = re.sub(r"(_\d+){1,3}$", "", genotype)
        genotype = re.sub(r"\s*\bmutants?\b\s*", "", genotype).strip()
        if (genotype and genotype == genotype.lower()
                and re.fullmatch(r"[a-z0-9\-]+", genotype)
                and re.search(r"[a-z]", genotype)):
            genotype = f"*{genotype}*"

        model_row = model_by_sid.get(sid, {})
        row = {
            "Species":    SPECIES_LABEL.get(sp, sp),
            "Microscope": mic,
            "Genotype":   genotype,
        }
        m_vals, a_vals = [], []
        for c in CLASSES:
            mv = _parse(model_row.get(f"{c}_IoU", ""))
            av = _parse(r.get(f"{c}_IoU", ""))
            row[f"{c}_M"] = f"{mv:.3f}" if mv is not None else "n/a"
            row[f"{c}_A"] = f"{av:.3f}" if av is not None else "n/a"
            if mv is not None:
                m_vals.append(mv)
            if av is not None:
                a_vals.append(av)
        row["mIoU_M"] = f"{sum(m_vals) / len(m_vals):.3f}" if m_vals else "n/a"
        row["mIoU_A"] = f"{sum(a_vals) / len(a_vals):.3f}" if a_vals else "n/a"
        out.append(row)
    fields = list(out[0].keys())
    return out, fields


def build_per_class_by_group_table() -> tuple[list[dict], list[str]]:
    """Merged panel d + e + by-species per-class table. Two sections:

      Section 1 - By species  (Rice, Millet, Sorghum, Solanums)
      Section 2 - By microscope (Olympus, C10, Zeiss)

    For each group, report per-class IoU mean ± std plus sample-level mIoU
    mean ± std. Rice (the only species spanning both test sets) pools the
    in-distribution test and Zeiss zero-shot splits, matching the panel d
    convention; the other species use in-distribution test only. Microscope
    rows are non-overlapping by construction (each sample has one
    microscope); Zeiss is the zero-shot row.
    """
    miou_rows = load_csv(HERE.parent / "per_sample_iou.csv")
    test = [r for r in miou_rows if r["split"] == "test"]

    def _make_row(group_label: str, sub: list[dict],
                  grouping: str) -> dict:
        row = {
            "Grouping": grouping,
            "Group":    group_label,
            "n":        len(sub),
        }
        for c in CLASSES:
            s = summary(floats(sub, f"{c}_IoU"))
            row[f"{c} IoU (mean ± std)"] = fmt_mean_std(s)
        s_m = summary(floats(sub, "mean_IoU"))
        row["mIoU (mean ± std)"] = fmt_mean_std(s_m)
        return row

    out = []
    # ── By species ────────────────────────────────────────────────────
    # The split-source qualifier (Rice pools both test sets; the others use
    # in-distribution only) is described in the caption, not the row label.
    for sp in SPECIES_ORDER:
        if sp == "Rice":
            sub = [r for r in miou_rows if r["species"] == sp]
        else:
            sub = [r for r in test if r["species"] == sp]
        out.append(_make_row(SPECIES_LABEL[sp], sub, "Species"))

    # ── By microscope ─────────────────────────────────────────────────
    for mic, split, label in MICROSCOPE_ROWS:
        sub = [r for r in miou_rows
               if r["microscope"] == mic and r["split"] == split]
        out.append(_make_row(label, sub, "Microscope"))

    fields = list(out[0].keys())
    return out, fields


def write_csv(rows: list[dict], fields: list[str], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_markdown(rows: list[dict], fields: list[str], path: Path,
                   title: str, caption: str) -> None:
    lines = [f"# {title}", "", caption, ""]
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("|" + "|".join(["---"] * len(fields)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(r[f]) for f in fields) + " |")
    path.write_text("\n".join(lines) + "\n")


# ─── Nature-style SVG table renderer ────────────────────────────────────────
# Booktabs convention: thick top + bottom rules, thin midrule under the
# column-header row, thin cmidrule spans under group-header text, NO vertical
# rules anywhere. Helvetica everywhere (paper-wide convention).

# All body text and headers are 7 pt Helvetica (= 2.469 mm at 1 pt = 0.3528 mm).
# Row / header / line heights are sized so the 7 pt cap-height + descender
# leaves the canonical Nature-style 0.5 mm gap above the midrule.
T_RULE   = 0.40   # mm: top and bottom rules (booktabs \toprule / \bottomrule)
M_RULE   = 0.20   # mm: midrule and cmidrule
ROW_H    = 6.0    # mm: data row height
HDR_H    = 5.4    # mm: column-header row height (single line)
LINE_H   = 3.1    # mm: additional height per extra line in a multi-line header
GRP_H    = 5.4    # mm: group-header row height
SEC_H    = 5.4    # mm: section-divider row height (table d/e)
TITLE_H  = 6.0    # mm: title row above the table
FOOT_H   = 4.4    # mm: footnote line height
SIDE_PAD = 3.0    # mm: left + right canvas padding
CELL_PAD = 0.8    # mm: horizontal pad inside each cell
FONT_BODY  = 2.47 # mm ≈ 7 pt
FONT_HDR   = 2.47 # mm ≈ 7 pt bold
FONT_GRP   = 2.47 # mm ≈ 7 pt bold
FONT_TITLE = 2.65 # mm ≈ 7.5 pt bold (unused: titles are removed)
FONT_FOOT  = 2.10 # mm ≈ 6 pt (unused: footnotes are removed)
FONT_SEC   = 2.47 # mm ≈ 7 pt bold italic (section dividers)


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;")
                  .replace("<", "&lt;")
                  .replace(">", "&gt;"))


def _render_inline(s) -> str:
    """Render text that may contain `*italic*` substrings into SVG-safe
    content with <tspan font-style="italic"> for the marked segments.
    Markdown-style single-asterisk pairs are detected; everything else is
    escaped and emitted verbatim.
    """
    import re
    s = "" if s is None else str(s)
    if not s or "*" not in s:
        return _esc(s)
    parts = re.split(r"(\*[^*]+\*)", s)
    out = []
    for p in parts:
        if not p:
            continue
        if len(p) >= 2 and p[0] == "*" and p[-1] == "*":
            out.append(f'<tspan font-style="italic">{_esc(p[1:-1])}</tspan>')
        else:
            out.append(_esc(p))
    return "".join(out)


def _text(x: float, y: float, s: str, *, size=FONT_BODY,
          anchor="middle", weight="normal", style="normal",
          fill="black") -> str:
    style_attr = f' font-style="{style}"' if style != "normal" else ""
    weight_attr = f' font-weight="{weight}"' if weight != "normal" else ""
    return (f'<text x="{x:.3f}" y="{y:.3f}" font-size="{size}" '
            f'text-anchor="{anchor}"{weight_attr}{style_attr} '
            f'fill="{fill}">{_render_inline(s)}</text>')


def _line(x1, y1, x2, y2, w) -> str:
    return (f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" '
            f'stroke="black" stroke-width="{w}"/>')


def render_table_svg(
    columns: list[dict],
    rows: list[dict],
    *,
    group_headers: list[dict] = None,   # [{label, start, end}]  (col idx inclusive)
    section_dividers: list[dict] = None, # [{row, label}]  row = data-row idx where divider sits ABOVE it
    title: str = None,
    footnote: list[str] = None,
) -> tuple[str, float, float]:
    """Render a Nature/booktabs-style table as inline SVG.

    columns: [{key, label, width_mm, align: 'start'|'middle'|'end'}]
    rows:    list of dicts keyed by column key
    Returns: (svg_xml, viewbox_w, viewbox_h).
    """
    group_headers = group_headers or []
    section_dividers = section_dividers or []
    footnote = footnote or []

    table_w = sum(c["width_mm"] for c in columns)
    canvas_w = table_w + 2 * SIDE_PAD

    # Vertical layout
    y = 0.0
    if title:
        y += TITLE_H
    y_top = y
    has_group = bool(group_headers)
    if has_group:
        y += GRP_H + 0.2     # +cmidrule gap

    # Account for section dividers inside the data block.
    n_dividers = len(section_dividers)
    data_h = len(rows) * ROW_H + n_dividers * SEC_H
    # Multi-line column headers grow HDR_H by LINE_H per extra line.
    def _hdr_lines(label):
        return label if isinstance(label, (list, tuple)) else [label]
    n_lines_max = max(len(_hdr_lines(c["label"])) for c in columns)
    effective_hdr_h = HDR_H + (n_lines_max - 1) * LINE_H
    body_top = y + effective_hdr_h + 0.2
    y_bottom = body_top + data_h
    canvas_h = y_bottom + (FOOT_H * len(footnote) if footnote else 0) + 1.0

    parts = []

    # Title
    if title:
        parts.append(_text(SIDE_PAD, TITLE_H - 1.2, title,
                           size=FONT_TITLE, anchor="start", weight="bold"))

    # Column x ranges
    x_starts = [SIDE_PAD]
    for c in columns:
        x_starts.append(x_starts[-1] + c["width_mm"])

    def col_x_text(i):
        c = columns[i]
        x0, x1 = x_starts[i], x_starts[i + 1]
        if c["align"] == "start":
            return x0 + CELL_PAD, "start"
        if c["align"] == "end":
            return x1 - CELL_PAD, "end"
        return (x0 + x1) / 2, "middle"

    # Top rule
    parts.append(_line(SIDE_PAD, y_top, SIDE_PAD + table_w, y_top, T_RULE))

    # Group headers + cmidrule
    if has_group:
        y_gtext = y_top + GRP_H - 1.2
        y_cmid  = y_top + GRP_H + 0.2
        for g in group_headers:
            s, e = g["start"], g["end"]
            gx0 = x_starts[s]
            gx1 = x_starts[e + 1]
            cx  = (gx0 + gx1) / 2
            parts.append(_text(cx, y_gtext, g["label"],
                               size=FONT_GRP, anchor="middle", weight="bold"))
            parts.append(_line(gx0 + 0.5, y_cmid, gx1 - 0.5, y_cmid, M_RULE))

    # Column headers. Multi-line labels (label given as list/tuple) stack
    # from the bottom up, so the last line of every header sits on the
    # same baseline regardless of how many lines that header has.
    y_hdr_top = (y_top + GRP_H + 0.2) if has_group else y_top
    y_last_baseline = y_hdr_top + effective_hdr_h - 1.2
    for i, c in enumerate(columns):
        tx, ta = col_x_text(i)
        lines = _hdr_lines(c["label"])
        for k, line in enumerate(reversed(lines)):
            y_line = y_last_baseline - k * LINE_H
            parts.append(_text(tx, y_line, line,
                               size=FONT_HDR, anchor=ta, weight="bold"))

    # Midrule under column headers
    y_midrule = y_hdr_top + effective_hdr_h + 0.2
    parts.append(_line(SIDE_PAD, y_midrule, SIDE_PAD + table_w, y_midrule, M_RULE))

    # Body rows (interleave section dividers and group separators)
    div_at = {d["row"]: d["label"] for d in section_dividers}
    y_cursor = body_top
    for ridx, row in enumerate(rows):
        if ridx in div_at:
            # Italic section title centred over the table, with a thin
            # rule below it (booktabs-style internal break).
            cx = SIDE_PAD + table_w / 2
            parts.append(_text(SIDE_PAD + CELL_PAD,
                               y_cursor + SEC_H - 1.4,
                               div_at[ridx],
                               size=FONT_SEC, anchor="start",
                               weight="bold", style="italic"))
            y_cursor += SEC_H
            parts.append(_line(SIDE_PAD, y_cursor - 0.1,
                               SIDE_PAD + table_w, y_cursor - 0.1, M_RULE))
        # Thin rule above this row if it begins a new sub-group (e.g. a
        # new Species after the previous block). The special `_separator`
        # key on the row triggers it.
        if row.get("_separator"):
            parts.append(_line(SIDE_PAD, y_cursor,
                               SIDE_PAD + table_w, y_cursor, M_RULE))
        # Render the data row
        ybase = y_cursor + ROW_H - 1.4
        for i, c in enumerate(columns):
            if c["key"].startswith("_"):
                continue
            tx, ta = col_x_text(i)
            val = row.get(c["key"], "")
            parts.append(_text(tx, ybase, val, size=FONT_BODY, anchor=ta))
        y_cursor += ROW_H

    # Bottom rule
    parts.append(_line(SIDE_PAD, y_bottom, SIDE_PAD + table_w, y_bottom, T_RULE))

    # Footnote
    y_fn = y_bottom + FOOT_H - 0.5
    for line in footnote:
        parts.append(_text(SIDE_PAD, y_fn, line,
                           size=FONT_FOOT, anchor="start", style="italic"))
        y_fn += FOOT_H

    svg_inner = "\n".join(parts)
    return svg_inner, canvas_w, canvas_h


# Standard physical canvas width for every saved table figure (mm). Tables
# narrower than this are padded with whitespace and centred so that, when
# all four tables are dropped into the manuscript at the same column width,
# none of them need to be rescaled and 7 pt stays 7 pt across the whole set.
CANVAS_W_MM = 180.0


def wrap_html(svg_inner: str, w_mm: float, h_mm: float,
              filename_stem: str, title_tag: str) -> str:
    """Wrap rendered SVG in an HTML page with Save PNG / Save SVG buttons.

    The saved canvas is forced to CANVAS_W_MM wide (or wider if the table
    naturally exceeds that). The table content is horizontally centred in
    that canvas via a top-level <g transform="translate(...)">.
    """
    target_w = max(CANVAS_W_MM, w_mm)
    x_offset = (target_w - w_mm) / 2.0
    if x_offset > 0.01:
        svg_inner = (f'<g transform="translate({x_offset:.3f}, 0)">\n'
                     f'{svg_inner}\n</g>')
    w_mm = target_w
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_esc(title_tag)}</title>
<style>
  :root {{ --w-mm: {w_mm:.3f}; --h-mm: {h_mm:.3f}; }}
  body {{
    margin: 0; padding: 18px;
    font-family: Helvetica, Arial, sans-serif;
    background: #f4f4f4;
  }}
  #controls {{
    display: flex; gap: 10px; align-items: center;
    margin-bottom: 12px; flex-wrap: wrap;
  }}
  button {{
    padding: 8px 14px; font-size: 13px;
    background: #333; color: white; border: 0;
    border-radius: 4px; cursor: pointer;
  }}
  input[type="number"] {{ padding: 5px 8px; font-size: 13px; }}
  svg#fig {{
    display: block;
    width: calc(var(--w-mm) * 1mm);
    height: calc(var(--h-mm) * 1mm);
    background: white;
    box-shadow: 0 1px 6px rgba(0,0,0,0.15);
    outline: 1px solid #bbb;
  }}
  svg#fig text {{ font-family: Helvetica, Arial, sans-serif; }}
</style>
</head>
<body>

<div id="controls">
  <button id="save">Save PNG</button>
  <button id="save-svg">Save SVG</button>
  <label>DPI <input type="number" id="dpi" value="600" min="72" max="1200" step="50" style="width:70px"></label>
</div>

<svg id="fig" viewBox="0 0 {w_mm:.3f} {h_mm:.3f}" xmlns="http://www.w3.org/2000/svg"
     font-family="Helvetica, Arial, sans-serif">
{svg_inner}
</svg>

<script>
function saveSVG() {{
    const svgNode = document.getElementById("fig");
    const clone = svgNode.cloneNode(true);
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    const src = new XMLSerializer().serializeToString(clone);
    const blob = new Blob([src], {{ type: "image/svg+xml" }});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "{filename_stem}.svg"; a.click();
    URL.revokeObjectURL(url);
}}
// PNG CRC table for embedding a pHYs chunk so the saved file declares its
// DPI. Without this, programs like Word and PowerPoint assume 96 DPI and
// render the image ~6× larger than the live preview. With pHYs, DPI-aware
// programs scale it to the correct mm size.
const _CRC_TABLE = (() => {{
    const t = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {{
        let c = n;
        for (let k = 0; k < 8; k++) c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
        t[n] = c;
    }}
    return t;
}})();
function _crc32(buf, start, len) {{
    let c = 0xffffffff;
    for (let i = 0; i < len; i++) c = _CRC_TABLE[(c ^ buf[start + i]) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
}}
function _injectPHYs(pngBytes, ppm) {{
    // PNG = 8-byte signature + IHDR (4+4+13+4=25 bytes) + ... Insert pHYs
    // chunk right after IHDR (bytes 8..32 → insertion at byte 33).
    const ihdrEnd = 33;
    const phys = new Uint8Array(21);
    // length = 9
    phys[0]=0; phys[1]=0; phys[2]=0; phys[3]=9;
    // type "pHYs"
    phys[4]=0x70; phys[5]=0x48; phys[6]=0x59; phys[7]=0x73;
    const writeU32 = (off, v) => {{
        phys[off]=(v>>>24)&0xff; phys[off+1]=(v>>>16)&0xff;
        phys[off+2]=(v>>>8)&0xff; phys[off+3]=v&0xff;
    }};
    writeU32(8,  ppm);   // pixels per metre, X
    writeU32(12, ppm);   // pixels per metre, Y
    phys[16] = 1;        // unit = metre
    writeU32(17, _crc32(phys, 4, 13));  // CRC over type+data
    const out = new Uint8Array(pngBytes.length + 21);
    out.set(pngBytes.subarray(0, ihdrEnd), 0);
    out.set(phys, ihdrEnd);
    out.set(pngBytes.subarray(ihdrEnd), ihdrEnd + 21);
    return out;
}}
function savePNG() {{
    const dpi = +document.getElementById("dpi").value;
    const w_mm = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--w-mm'));
    const h_mm = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--h-mm'));
    const pxPerMm = dpi / 25.4;
    const pxW = Math.round(w_mm * pxPerMm), pxH = Math.round(h_mm * pxPerMm);
    const svgNode = document.getElementById("fig");
    const clone = svgNode.cloneNode(true);
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    clone.setAttribute("width",  pxW);
    clone.setAttribute("height", pxH);
    const src = new XMLSerializer().serializeToString(clone);
    const blob = new Blob([src], {{ type: "image/svg+xml;charset=utf-8" }});
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = function() {{
        const cv = document.createElement("canvas");
        cv.width = pxW; cv.height = pxH;
        const ctx = cv.getContext("2d");
        ctx.fillStyle = "white"; ctx.fillRect(0, 0, pxW, pxH);
        ctx.drawImage(img, 0, 0, pxW, pxH);
        URL.revokeObjectURL(url);
        cv.toBlob(async b2 => {{
            const raw = new Uint8Array(await b2.arrayBuffer());
            const ppm = Math.round(dpi / 0.0254);    // pixels per metre
            const withDpi = _injectPHYs(raw, ppm);
            const final = new Blob([withDpi], {{ type: "image/png" }});
            const u2 = URL.createObjectURL(final);
            const a = document.createElement("a");
            a.href = u2; a.download = "{filename_stem}.png"; a.click();
            URL.revokeObjectURL(u2);
        }}, "image/png");
    }};
    img.src = url;
}}
document.getElementById("save").addEventListener("click", savePNG);
document.getElementById("save-svg").addEventListener("click", saveSVG);
</script>
</body>
</html>
"""


def build_bc_html_svg(rows_bc: list[dict]) -> tuple[str, float, float]:
    """Compact Nature-style table for panels b + c. IoU mean ± s.d. across
    samples for the in-distribution test split, inter-annotator baseline,
    and Zeiss zero-shot split."""
    columns = [
        {"key": "Class", "label": "Anatomical Class",
         "width_mm": 28, "align": "start"},
        {"key": "t_iou",
         "label": ["In-distribution Test IoU", "(mean ± s.d., n = 185)"],
         "width_mm": 40, "align": "middle"},
        {"key": "z_iou",
         "label": ["Zero-shot Test IoU", "(mean ± s.d., n = 35)"],
         "width_mm": 34, "align": "middle"},
    ]
    bc_rows = []
    for r in rows_bc:
        bc_rows.append({
            "Class": r["Anatomical Class"],
            "t_iou": r["In-distribution Test IoU (mean ± std)"],
            "z_iou": r["Zero-shot Test IoU (mean ± std)"],
        })
    return render_table_svg(columns, bc_rows,
                            group_headers=None,
                            title=None, footnote=[])


def build_annotator_html_svg(rows_ann: list[dict]) -> tuple[str, float, float]:
    """Side-by-side table: for each of the 16 re-annotated samples show the
    model-vs-GT IoU and the second-annotator-vs-GT IoU per class, plus the
    sample-level mIoU pair. Class names span two narrow sub-columns
    (Model / Human) via group headers + cmidrules."""
    sub_w = 9    # mm per Model or Human sub-column at 7 pt
    columns = [
        {"key": "Species",    "label": "Species",    "width_mm": 13, "align": "start"},
        {"key": "Microscope", "label": "Microscope", "width_mm": 15, "align": "start"},
        {"key": "Genotype",   "label": "Genotype",   "width_mm": 18, "align": "start"},
    ]
    # Six anatomy classes, then mIoU. Each spans a Model / Human pair.
    groups = []
    for c in CLASSES:
        idx = len(columns)
        groups.append({"label": c, "start": idx, "end": idx + 1})
        columns.append({"key": f"{c}_M", "label": "Model", "width_mm": sub_w, "align": "middle"})
        columns.append({"key": f"{c}_A", "label": "Human", "width_mm": sub_w, "align": "middle"})
    idx = len(columns)
    groups.append({"label": "mIoU", "start": idx, "end": idx + 1})
    columns.append({"key": "mIoU_M", "label": "Model", "width_mm": sub_w, "align": "middle"})
    columns.append({"key": "mIoU_A", "label": "Human", "width_mm": sub_w, "align": "middle"})

    return render_table_svg(columns, rows_ann,
                            group_headers=groups,
                            title=None, footnote=[])


def build_de_html_svg(rows_de: list[dict]) -> tuple[str, float, float]:
    """Compact Nature-style table merging the previous panel d/e mIoU table
    with the per-class breakdown. Two sections (By species / By microscope);
    each row carries per-class IoU plus sample-level mIoU."""
    columns = [
        {"key": "Group",      "label": "Group",      "width_mm": 16, "align": "start"},
        {"key": "n",          "label": "n",          "width_mm": 10, "align": "middle"},
        {"key": "Vascular",   "label": "Vascular",   "width_mm": 20, "align": "middle"},
        {"key": "Exodermis",  "label": "Exodermis",  "width_mm": 20, "align": "middle"},
        {"key": "Endodermis", "label": "Endodermis", "width_mm": 20, "align": "middle"},
        {"key": "Cortex",     "label": "Cortex",     "width_mm": 20, "align": "middle"},
        {"key": "Epidermis",  "label": "Epidermis",  "width_mm": 20, "align": "middle"},
        {"key": "Aerenchyma", "label": "Aerenchyma", "width_mm": 20, "align": "middle"},
        {"key": "mIoU",       "label": "mIoU",       "width_mm": 20, "align": "middle"},
    ]
    body = []
    section_dividers = []
    section_dividers.append({"row": 0, "label": "By species"})
    for r in rows_de:
        if r["Grouping"] == "Species":
            d = {"Group": r["Group"], "n": r["n"]}
            for c in CLASSES:
                d[c] = r[f"{c} IoU (mean ± std)"]
            d["mIoU"] = r["mIoU (mean ± std)"]
            body.append(d)
    section_dividers.append({"row": len(body), "label": "By microscope"})
    for r in rows_de:
        if r["Grouping"] == "Microscope":
            d = {"Group": r["Group"], "n": r["n"]}
            for c in CLASSES:
                d[c] = r[f"{c} IoU (mean ± std)"]
            d["mIoU"] = r["mIoU (mean ± std)"]
            body.append(d)
    return render_table_svg(columns, body,
                            group_headers=None,
                            section_dividers=section_dividers,
                            title=None, footnote=[])


def main() -> None:
    rows_bc, fields_bc   = build_per_class_table()
    rows_de, fields_de   = build_per_class_by_group_table()
    rows_f,  fields_f    = build_panel_f_table()
    rows_div, fields_div = build_diversity_table()
    rows_ann, fields_ann = build_annotator_table()

    csv_bc  = HERE / "Per-class_IoU.csv"
    csv_de  = HERE / "sample-level_mIoU_by_species_and_microscope.csv"
    csv_f   = HERE / "Sample-level_mIoU_eight_benchmarked_architectures.csv"
    csv_div_train_a = HERE / "supp_table_dataset_diversity_train_part1.csv"
    csv_div_train_b = HERE / "supp_table_dataset_diversity_train_part2.csv"
    csv_div_test    = HERE / "supp_table_dataset_diversity_test.csv"
    csv_ann = HERE / "supp_table_annotator_baseline.csv"
    md_bc   = HERE / "Per-class_IoU.md"
    md_de   = HERE / "sample-level_mIoU_by_species_and_microscope.md"
    md_f    = HERE / "Sample-level_mIoU_eight_benchmarked_architectures.md"
    md_div_train_a = HERE / "supp_table_dataset_diversity_train_part1.md"
    md_div_train_b = HERE / "supp_table_dataset_diversity_train_part2.md"
    md_div_test    = HERE / "supp_table_dataset_diversity_test.md"
    md_ann  = HERE / "supp_table_annotator_baseline.md"

    # Combined train+val rows: include any genotype with at least one sample
    # in either training or validation, so the merged table captures
    # everything the model was exposed to during training. Sorted in the
    # same monocots-then-Solanums order; split in two halves for the
    # parallel side-by-side layout.
    div_trainval_all = []
    for r in rows_div:
        t, v = (r.get("Train") or 0), (r.get("Val") or 0)
        if t + v == 0:
            continue
        div_trainval_all.append({
            "Species":  r["Species"],
            "Genotype": r["Genotype"],
            "Train":    t,
            "Val":      v,
        })
    div_train_a, div_train_b = _half_split_with_species_snap(div_trainval_all)

    div_test_rows = []
    for r in rows_div:
        tt = (r["Test"] or 0) + (r["Zero-shot"] or 0)
        if tt > 0:
            div_test_rows.append({
                "Species":            r["Species"],
                "Genotype":           r["Genotype"],
                "In-distribution Test": r["Test"],
                "Zero-shot Test":     r["Zero-shot"],
                "Total":              tt,
            })
    div_test_fields = ["Species", "Genotype",
                       "In-distribution Test", "Zero-shot Test", "Total"]
    div_trainval_fields = ["Species", "Genotype", "Train", "Val"]

    write_csv(rows_bc,  fields_bc,  csv_bc)
    write_csv(rows_de,  fields_de,  csv_de)
    write_csv(rows_f,   fields_f,   csv_f)
    write_csv(div_train_a,  div_trainval_fields, csv_div_train_a)
    write_csv(div_train_b,  div_trainval_fields, csv_div_train_b)
    write_csv(div_test_rows, div_test_fields,    csv_div_test)
    write_csv(rows_ann, fields_ann, csv_ann)
    write_markdown(
        rows_bc, fields_bc, md_bc,
        "Supplementary Table. Figure 2b,c per-class IoU",
        "Per-class IoU for RADIX (Root Anatomy Deep Identification across "
        "species) on the in-distribution test split (n=185 samples; panel b) "
        "and on the Zeiss zero-shot split (n=35 samples; panel c). Aerenchyma "
        "is reported only over samples in which the class is present (tomato "
        "samples are excluded). Whole Root is omitted per project convention; "
        "it scores ~0.98 IoU across all splits and contributes no signal "
        "between models.",
    )
    write_markdown(
        rows_de, fields_de, md_de,
        "Supplementary Table. Figure 2d,e per-class IoU and mIoU by species and microscope",
        "Per-tissue-class IoU and per-sample mIoU (mean ± s.d. across samples) "
        "for RADIX, broken down by species (top block) and by microscope "
        "(bottom block). Sample-level mIoU is the unweighted mean over the six "
        "anatomy classes for each sample. Rice pools both test sets "
        "(in-distribution and Zeiss zero-shot); the other species use the "
        "in-distribution test split only. Microscope rows are non-overlapping "
        "by construction; Zeiss is the held-out zero-shot row. n is the "
        "number of samples per row. Aerenchyma is computed only over samples "
        "in which the class is present; Solanums lack the class entirely. "
        "Values correspond to Figure 2d and 2e.",
    )
    write_markdown(
        rows_f, fields_f, md_f,
        "Supplementary Table. Figure 2f per-model sample-level mIoU",
        "Sample-level mIoU (mean over the six anatomy classes per sample) for "
        "every model in the Figure 2f benchmark, summarised as mean ± standard "
        "deviation across samples. In-distribution Test (n = 185) is the "
        "Strategy A test split pooled across all species and microscopes; "
        "Zero-shot Test (n = 35) is the held-out Zeiss split. Rows are sorted "
        "by in-distribution mIoU, best first.",
    )
    train_total = sum((r.get("Train") or 0) for r in div_trainval_all)
    val_total   = sum((r.get("Val")   or 0) for r in div_trainval_all)
    tv_total    = train_total + val_total
    write_markdown(
        div_train_a, div_trainval_fields, md_div_train_a,
        f"Supplementary Table. Training dataset composition by species and "
        f"genotype, part 1 of 2 (n = {tv_total:,} samples in total: "
        f"{train_total:,} training, {val_total:,} validation)",
        "Per-genotype sample counts in the training and validation splits, "
        "first half of two parallel sub-tables. Filtered to genotypes with "
        "at least one sample in either split. Read this sub-table "
        "top-to-bottom, then continue into part 2. Rows are sorted by "
        "species in the monocots-then-Solanums order used throughout the "
        "paper. Solanum binomials are italicised per botanical convention; "
        "gene and allele names in the genotype column are also italicised.",
    )
    write_markdown(
        div_train_b, div_trainval_fields, md_div_train_b,
        f"Supplementary Table. Training dataset composition by species and "
        f"genotype, part 2 of 2 (n = {tv_total:,} samples in total: "
        f"{train_total:,} training, {val_total:,} validation)",
        "Continued from part 1. Per-genotype sample counts in the training "
        "and validation splits, second half. Same column structure, "
        "sorting, and italic conventions as part 1.",
    )
    test_in   = sum((r.get("In-distribution Test") or 0) for r in div_test_rows)
    test_zero = sum((r.get("Zero-shot Test") or 0) for r in div_test_rows)
    test_total = test_in + test_zero
    write_markdown(
        div_test_rows, div_test_fields, md_div_test,
        f"Supplementary Table. Test dataset composition by species and "
        f"genotype (n = {test_total:,} test samples in total: "
        f"{test_in:,} in-distribution, {test_zero:,} Zeiss zero-shot)",
        "Per-genotype sample counts in the test splits: the in-distribution "
        "test split and the Zeiss zero-shot test split. Filtered to "
        "genotypes with at least one sample in either test split. Rows are "
        "sorted by species in the monocots-then-Solanums order used "
        "throughout the paper. Solanum binomials are italicised per "
        "botanical convention; gene/allele names in the genotype column are "
        "also italicised. The Total column is in-distribution + zero-shot "
        "for that row. All zero-shot samples are Rice (Kitaake) on the "
        "held-out Zeiss microscope.",
    )
    write_markdown(
        rows_ann, fields_ann, md_ann,
        "Supplementary Table. Inter-annotator IoU baseline",
        "Per-class IoU between two expert annotators on each of the 16 "
        "re-annotated samples. Rows are individual samples. n/a indicates "
        "the class is absent from the sample's ground truth (tomato samples "
        "lack aerenchyma; one Sorghum sample also lacked it). Provides a "
        "human-level reference for the model IoU reported elsewhere in the "
        "supplementary tables.",
    )

    # ── HTML figure tables ────────────────────────────────────────────
    svg_bc, w_bc, h_bc    = build_bc_html_svg(rows_bc)
    svg_de, w_de, h_de    = build_de_html_svg(rows_de)
    svg_f,  w_f,  h_f     = build_panel_f_html_svg(rows_f)
    # Collapse repeated Species names + insert thin rules between groups,
    # purely for visual rendering. The CSV/MD outputs above keep every row
    # explicit (so downstream programs can sort/filter them freely).
    div_train_a_v = _collapse_repeated_groups(div_train_a)
    div_train_b_v = _collapse_repeated_groups(div_train_b)
    div_test_v    = _collapse_repeated_groups(div_test_rows)
    svg_div_train, w_dt, h_dt = build_diversity_train_parallel_html_svg(
        div_train_a_v, div_train_b_v)
    svg_div_test,  w_div_test,  h_div_test  = build_diversity_test_html_svg(div_test_v)
    rows_ann_v = _collapse_repeated_groups(rows_ann, key="Species")
    svg_ann, w_ann, h_ann = build_annotator_html_svg(rows_ann_v)
    html_bc  = HERE / "Per-class_IoU.html"
    html_de  = HERE / "sample-level_mIoU_by_species_and_microscope.html"
    html_f   = HERE / "Sample-level_mIoU_eight_benchmarked_architectures.html"
    html_div_train = HERE / "supp_table_dataset_diversity_train.html"
    html_div_test  = HERE / "supp_table_dataset_diversity_test.html"
    html_ann = HERE / "supp_table_annotator_baseline.html"
    html_bc.write_text(wrap_html(
        svg_bc, w_bc, h_bc, "Per-class_IoU",
        "Supplementary Table for Figure 2b,c"))
    html_de.write_text(wrap_html(
        svg_de, w_de, h_de, "sample-level_mIoU_by_species_and_microscope",
        "Supplementary Table for Figure 2d,e"))
    html_f.write_text(wrap_html(
        svg_f, w_f, h_f, "Sample-level_mIoU_eight_benchmarked_architectures",
        "Supplementary Table for Figure 2f"))
    html_div_train.write_text(wrap_html(
        svg_div_train, w_dt, h_dt,
        "supp_table_dataset_diversity_train",
        "Supplementary Table: training composition"))
    html_div_test.write_text(wrap_html(
        svg_div_test, w_div_test, h_div_test, "supp_table_dataset_diversity_test",
        "Supplementary Table: test composition"))
    html_ann.write_text(wrap_html(
        svg_ann, w_ann, h_ann, "supp_table_annotator_baseline",
        "Supplementary Table: inter-annotator baseline"))

    # Remove the superseded combined / monocots-vs-solanums train tables.
    for stale in [
        HERE / "supp_table_dataset_diversity.csv",
        HERE / "supp_table_dataset_diversity.md",
        HERE / "supp_table_dataset_diversity.html",
        HERE / "supp_table_dataset_diversity_train_val.csv",
        HERE / "supp_table_dataset_diversity_train_val.md",
        HERE / "supp_table_dataset_diversity_train_val.html",
        # Note: the bare "_train.csv/.md" files predate the merged figure
        # and were the single-split outputs; they have been re-introduced
        # for the parallel HTML so DO NOT delete them here.
        HERE / "supp_table_dataset_diversity_train_monocots.csv",
        HERE / "supp_table_dataset_diversity_train_monocots.md",
        HERE / "supp_table_dataset_diversity_train_monocots.html",
        HERE / "supp_table_dataset_diversity_train_solanums.csv",
        HERE / "supp_table_dataset_diversity_train_solanums.md",
        HERE / "supp_table_dataset_diversity_train_solanums.html",
        # Superseded by the merged side-by-side train table.
        HERE / "supp_table_dataset_diversity_train_part1.html",
        HERE / "supp_table_dataset_diversity_train_part2.html",
        # Val is now merged into the train table.
        HERE / "supp_table_dataset_diversity_val.csv",
        HERE / "supp_table_dataset_diversity_val.md",
        HERE / "supp_table_dataset_diversity_val.html",
    ]:
        if stale.exists():
            stale.unlink()

    # Remove the now-superseded standalone per-species-only table.
    for stale in [
        HERE / "supp_table_per_class_iou_by_species.csv",
        HERE / "supp_table_per_class_iou_by_species.md",
        HERE / "supp_table_per_class_iou_by_species.html",
        HERE / "supp_table_fig2de_sample_miou.csv",   # legacy clean-up
        HERE / "supp_table_fig2de_sample_miou.md",    # legacy clean-up
    ]:
        if stale.exists():
            stale.unlink()

    for p in [csv_bc, csv_de, csv_f,
              csv_div_train_a, csv_div_train_b, csv_div_test, csv_ann,
              md_bc,  md_de,  md_f,
              md_div_train_a, md_div_train_b, md_div_test, md_ann]:
        print(f"→ {p.relative_to(HERE.parent.parent)}")
    for p, w, h in [(html_bc,           w_bc,    h_bc),
                    (html_de,           w_de,    h_de),
                    (html_f,            w_f,     h_f),
                    (html_div_train,    w_dt,    h_dt),
                    (html_div_test,     w_div_test, h_div_test),
                    (html_ann,          w_ann,   h_ann)]:
        print(f"→ {p.relative_to(HERE.parent.parent)}   "
              f"viewBox = {w:.1f} × {h:.1f} mm")
    print()
    print("Open each HTML via `python3 -m http.server` "
          "(or directly in browser since no external CSV is loaded), "
          "then click Save PNG / Save SVG.")


if __name__ == "__main__":
    main()
