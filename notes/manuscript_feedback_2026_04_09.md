# Manuscript Feedback Conversation — 2026-04-09

Full conversation about framing the root anatomy segmentation paper for Nature Communications / Nature Plants.

---

## Q1: For publishing a Nature Communications / Nature Plants level paper, what model performance or metrics could I compare with to justify that my model is superior or very powerful, or made significant improvement for the field? There is no model can segment root anatomical structures yet. There is only general model like cellpose, micro-sam and plantseg for general cell segmentation. It does not learn the anatomical structure.

Great question — this is about framing your contribution for maximum impact. Since no existing model segments root anatomical structures, your comparison strategy needs multiple layers:

### 1. Zero-shot baselines (most important)

Run pretrained general models **without fine-tuning** on your test set and evaluate on Bio-7 classes:

- **Cellpose cyto3** — will segment individual cells, not anatomical regions. Bio-7 mIoU will be near 0.
- **micro-SAM vit_b_lm** — same: cell-level, no anatomy understanding.
- **PlantSeg (laid-back-lobster)** — Arabidopsis cell boundaries, not structure-level.
- **SAM2 / MedSAM** — general foundation models, no plant domain knowledge.

These will all fail catastrophically at your task (mIoU likely <0.1 for anatomical classes). That gap — from ~0 to your 0.87 — **is your contribution**. It demonstrates the task requires specialized training, not just bigger models.

### 2. Fine-tuned general architectures (you mostly have this)

You already trained micro-SAM fine-tuned (per-class) and PanSeg UNet2D. Add these to your benchmark table alongside YOLO and U-Net++. This shows that even when you give general architectures your data, purpose-built pipelines outperform adapted ones.

### 3. Generalization metrics (what reviewers will scrutinize)

- **Cross-species** (Strategy B): unified model vs monocot/dicot specialists
- **Cross-microscope** (Zeiss zero-shot): 35 held-out samples from unseen platform
- **Per-species breakdown**: show it works on all 4 species, not just the majority class

This is what elevates the paper from "we trained a model" to "this generalizes across the biological diversity that matters."

### 4. Downstream biological validity (the killer argument for Nature Plants)

Reviewers in biology care less about mIoU and more about: **does the segmentation enable correct biological conclusions?**

- **Predicted vs GT measurements**: aerenchyma ratio, barrier intensity — R² and scatter plots
- **Phenotypic discrimination**: can the model detect known differences between genotypes/treatments that biologists already know exist?
- **Throughput**: "1,600 samples in X minutes vs Y hours manual" — quantify the speedup

### 5. Human performance upper bound

If feasible, have a second annotator label ~20-30 samples. Then:
- Human-vs-human IoU = ceiling
- Model-vs-human IoU = your performance
- If these are close, you can claim **"approaching expert-level agreement"**

### Summary table for the paper

| Comparison | What it proves |
|---|---|
| Pretrained general models (zero-shot) | Task requires specialized training; no existing tool works |
| Fine-tuned general architectures | Your pipeline outperforms adapted baselines |
| Multi-architecture benchmark (5 models) | Thorough, not cherry-picked |
| Cross-species/microscope generalization | Practical utility across labs |
| Downstream measurement accuracy (R²) | Biologically valid, not just pixel-accurate |
| Human agreement upper bound | Performance contextualized against expert variability |
| Throughput comparison | Practical impact for the field |

**The strongest framing**: "No existing tool can perform this task at all (layer 1). When we adapt existing tools, they underperform (layer 2). Our best model generalizes across species and microscopes (layer 3) and produces biologically accurate measurements (layer 4)."

The zero-shot general model comparison is the one you're currently missing and would add the most value — it takes very little compute and makes the gap viscerally clear in a figure.

---

## Q2: I have a question. Aren't unet and yolo also general models? How is fine-tuning sam different fine-tuning yolo or unet?

You're raising an important point, and you're right — U-Net++ and YOLO are also general architectures. The distinction isn't architecture novelty. Let me clarify the framing:

### What's actually the same

All of these are general-purpose architectures pretrained on non-plant data:
- YOLO → COCO (everyday objects)
- U-Net++ → ImageNet (natural photos)
- micro-SAM → light microscopy (cells, not anatomy)
- Cellpose → cytoplasm/nuclei

You fine-tune all of them on your dataset. There's no fundamental difference in the approach.

### So what IS your contribution?

**It's not a novel architecture. It's the task, dataset, and validation.** Specifically:

1. **Task definition** — nobody has formalized "segment 6 root anatomical structures from fluorescence microscopy" as a learnable task before
2. **Dataset** — 1,671 expert-annotated samples across 4 species, 3 microscopes, with careful split design
3. **Comprehensive benchmarking** — you tested 5 architectures so the community knows what works, not just "our method"
4. **Cross-species/microscope generalization** — proving it's not overfit to one lab's setup
5. **Biological validation** — showing the segmentation enables real phenotyping

### How to frame the comparisons honestly

Don't frame it as "our model vs general models." Frame it as:

| Comparison | What it tests |
|---|---|
| Pretrained models zero-shot (no fine-tuning) | Can existing tools do this task at all? → **No** |
| 5 architectures fine-tuned on your data | Which architecture suits this task best? |
| Specialist vs unified training (Strategy B) | Does cross-species training help? |
| Zeiss zero-shot | Does the trained model generalize to new hardware? |

The zero-shot baselines don't prove "our architecture is better." They prove **the task requires dedicated training data** — which is the dataset contribution.

### Analogies in high-impact papers

This is exactly how landmark papers in the field are structured:

- **Cellpose** (Nature Methods) — used a U-Net variant, not a novel architecture. Contribution = flow representation + diverse training data + generalization
- **StarDist** (MICCAI) — standard U-Net backbone. Contribution = star-convex polygon representation + dataset
- **PlantSeg** (eLife) — standard U-Net. Contribution = boundary prediction pipeline + plant-specific evaluation

None of these claimed architectural novelty. They claimed **"we solved a task nobody could solve before, validated it thoroughly, and released tools for the community."**

### Bottom line

Your paper's story is: *"Root anatomical segmentation was previously manual. No existing tool — general or specialized — can do it. We created a large annotated dataset, benchmarked 5 architectures, and show the best one generalizes across species and microscopes while producing biologically accurate measurements."*

The 5-architecture benchmark actually strengthens the paper — it shows rigor, not weakness. You're not selling one model, you're establishing a field.

---

## Q3: For Cellpose, StarDist, and PlantSeg, they all have their own little pieces they modified or add on to the model, but we just fine-tuned yolo or unet without modify anything for the pipeline. so is techniqually still different from Cellpose, StarDist, and PlantSeg. How do you frame this paper, so it is strong, can be accepted by nature comminications?

You're being honest with yourself, which is exactly the right instinct. Let me give you the real answer.

### The hard truth

You're right. Cellpose/StarDist/PlantSeg each have a **methodological contribution** (flow fields, star-convex polygons, boundary+watershed). If you submit a "we fine-tuned YOLO" paper to Nature Methods, it will be desk-rejected — they want method novelty.

**But Nature Communications is different.** It publishes plenty of papers where the contribution is *resource + biological discovery* rather than algorithmic novelty. That's the door you should walk through.

### Reframe: this is a biology paper, not a methods paper

Stop pitching it as "we built a segmentation model." Pitch it as:

> *"Root anatomical barriers (endodermis, exodermis) regulate water/nutrient uptake and stress tolerance, but quantifying them at scale has been impossible. We built the first pipeline to do this across species and microscopes, and used it to discover [X biology]."*

The model becomes the **enabling tool**, not the headline result. Nature Comms will judge this paper on the biology it enables, not on whether you invented a new loss function.

### What you need to make it Nature Comms-strong

#### 1. A real biological discovery (the most important missing piece)

Look at your existing data and find at least one of:

- **Cross-species comparison**: how does barrier formation timing/extent differ between cereals (millet/rice/sorghum) and dicots (tomato)? Is there a pattern nobody has quantified before?
- **Genotype phenotyping**: you have tomato mutants and sorghum genotypes. Show your pipeline detects known mutant phenotypes (validation), then reveals something unexpected.
- **Treatment/condition response**: any stress, hormone, or developmental time series in your data?
- **Quantitative law**: e.g., "exodermis thickness scales with root diameter as X across species" — a finding only possible because you can measure 1,600 roots.

**Without a biological discovery, this is a methods paper with weak methods. With one, it's a biology paper with a powerful tool.**

#### 2. Resource contribution (frame it explicitly)

Nature Comms loves "community resources." Make this an explicit selling point:

- **Largest annotated dataset** for root cross-section anatomy (1,671 samples, 4 species, 3 microscopes) — release it
- **First public benchmark** for the task — release leaderboard + eval code
- **Pretrained models** for community use — release weights for all 5 architectures
- **Annotation tool** (your polygon editor) — release the GUI

A reusable resource has *concrete* impact reviewers can point to.

#### 3. Technical contributions you DO have (lean into these)

You're underselling yourself. You have several real technical findings:

- **The `overlap_mask=False` discovery** — this is a genuine, generalizable finding for anyone training instance segmentation on nested annotations. +13 mAP is huge. Write it up as a section, not a footnote.
- **Bio-7 evaluation framework** — converting raw overlapping annotations into 7 mutually exclusive biological classes via ring derivation is non-trivial and reusable
- **Cross-microscope generalization protocol** — Zeiss zero-shot is a real test design
- **Fluorescence-specific augmentation strategy** — channel dropout/shuffle ablation gives the field actionable guidance
- **Semantic-to-instance conversion** for downstream polygon use

These aren't novel architectures, but they're **publishable engineering contributions** that the community will actually use.

#### 4. Multi-architecture benchmark = rigor

Frame your 5-model benchmark as the contribution itself: *"We provide the first systematic comparison of segmentation paradigms (instance, semantic, foundation, multilabel) on root anatomy."* This positions the paper as the field's reference point.

### The story arc for Nature Comms

1. **Hook**: Root barriers are critical for plant function but quantification is the bottleneck
2. **Gap**: No existing tool can do this; even general models fail (your zero-shot baselines)
3. **Resource**: We built the dataset, annotation protocol, and benchmark
4. **Method**: We benchmark 5 architectures and identify what works (with technical findings: overlap_mask, channel aug, etc.)
5. **Generalization**: Across species and microscopes, including zero-shot transfer
6. **Biology** ← **the most important section**: Using the pipeline, we discovered [X about barriers across species/genotypes]
7. **Validation**: Predicted vs expert measurements (R²), comparison to known biology
8. **Release**: Dataset, models, tool, all open

### What to do this week

1. **Identify the biology angle.** Look at your existing data — which subset would tell the most compelling story? Talk to your PI/Siobhan: what's the biological question they care most about that your pipeline can now answer?
2. **Don't add more ML experiments.** You have enough. Adding a 6th architecture won't help. A biological discovery will.
3. **Polish the technical findings into proper sections** — overlap_mask, channel augmentation ablation, cross-microscope transfer. Each is a small but real contribution.
4. **Plan the resource release** — dataset DOI (Zenodo), GitHub repo, model weights on HuggingFace. Reviewers love this.

### Honest journal targeting

- **Nature Methods**: probably not (no algorithmic novelty)
- **Nature Plants**: yes if biology discovery is strong; this is your top target
- **Nature Communications**: yes, especially with biology + resource framing
- **Plant Cell**: very strong fit (biology-focused)
- **eLife**: also a strong fit (PlantSeg published here)
- **Bioinformatics / Plant Methods**: solid fallback if biology angle stays weak

**My honest recommendation**: aim for Nature Plants first. It values resources + biology over algorithmic novelty more than Nature Comms does, and this paper fits its scope perfectly. Your CLAUDE.md already lists it as the target — stick with it.

The single biggest thing that will determine acceptance is whether you can show **a biological insight that was previously impossible**. The model is the means, not the end. Find that insight, and the rest of the paper writes itself.

---

## Q4: '/Users/yifeigu/Downloads/Manuscript.docx' here is my outline so far. can you check if some of the sections should be emphased or removed to the supplyment? how do these story fit together? I am not a biologist, I am a imaging scientist with a little ML background. The collaborating lab is plant biologists, they will be the main user but i am the first author on this paper. So it is a little awakward for me write a biology paper

I read the full manuscript. The good news: **you have all the right pieces.** Result 5 (the biological applications) is genuinely strong and is exactly what Nature Comms wants. The bad news: the current structure buries your strongest material under ML methodology, which makes it read like a methods paper.

Let me give you concrete, section-by-section feedback.

### Big picture: your story is in the wrong order

Your paper currently reads as: *dataset → ML benchmark → ablations → explainability → biology*

Nature Comms reviewers (especially the biology editor) will lose interest before they reach Result 5. You need to flip the emphasis: **the model is plumbing, the biology is the result.**

The four discoveries in Result 5 are the strongest material in the whole manuscript:
1. Sorghum genotype generalization (Stefan)
2. Microbial perturbations reshape barriers (Stefan)
3. WOX10 aerenchyma is environment-dependent (Tamera/Adeel)
4. Tomato exodermis suberin mutant (Kevin)

These collectively say: *"barriers are multi-dimensional, context-dependent, and were previously inaccessible at this scale."* That's a Nature Comms-worthy claim. Lead with it everywhere — abstract, intro hook, figure prominence.

### Section-by-section verdict

#### Abstract — REWRITE
Currently leads with "dataset + architectures." Should lead with the biological question and the discovery. Try this structure:
1. Sentence 1: barriers are central to plant function but quantification is the bottleneck (you have this)
2. Sentence 2: we built a generalizable framework (one sentence — don't list 3 architectures)
3. Sentence 3-4: **using this framework, we resolved [genotype × environment effects on barriers across 4 species]** ← this is the headline
4. Sentence 5: the framework is openly released

Move "we domain-adapted YOLO11/U-Net++/micro-SAM" to one phrase. Don't list architectures in the abstract — that screams "methods paper."

#### Result 1: Dataset — KEEP, but trim
Keep prominent. This is a real contribution. Trim the bullet list of growth conditions/microscopes to the figure caption.

#### Result 2: Benchmark + generalization — COMPRESS HEAVILY
This is currently your second-longest section and your weakest selling point. Reviewers don't need 4 panels of bar charts.

**Main text keeps:**
- One figure: representative qualitative results (panel a) + best-model per-class IoU (panel b) + Zeiss + Solanum zero-shot (one combined panel)
- One sentence on the architecture comparison: "We benchmarked YOLO11-seg, U-Net++, and micro-SAM (Supp. Table 1) and selected [best] for all downstream analysis."

**Move to supplement:**
- Per-architecture comparison table
- Per-species and per-microscope breakdown bar charts (keep as supp figs)
- Inference speed / model size comparison
- Most of Table 1

#### Result 3 (current) — SPLIT AND DEMOTE

This section currently mixes three unrelated things: augmentation ablation, mono vs dico specialist comparison, and GradCAM/UMAP explainability. They don't belong together.

**Augmentation ablation (3.1)** → fully supplementary. This is engineering, not biology.

**Mono vs dico unified-vs-specialist (3.2)** → keep in main text as a brief subsection of Result 2. It's actually a generalization claim, not explainability. One short paragraph + one panel.

**Explainability (3.3, GradCAM + UMAP)** → cut to ONE panel in main text. Show only the UMAP colored by species and by microscope (proves the model learned biology, not platform artifacts). Move GradCAM and channel occlusion to supplement. Biologists honestly do not care about GradCAM as much as ML reviewers think they do.

#### Result 4: Downstream validation — KEEP, EMPHASIZE
This is critical. R² scatter plots showing automated = expert measurements is exactly what biology reviewers need to trust the rest. Keep all panels.

One addition: explicitly include **a throughput/speedup statement in the main text** ("X samples in Y minutes vs Z hours manually"). Reviewers love a clean efficiency number.

#### Result 5: Biological applications — EXPAND, MAKE IT THE CENTERPIECE
Currently four subsections that read almost like vignettes. To carry a Nature Comms paper, this needs to:

1. **Open with a stronger framing paragraph** that states the biological hypothesis being tested across all four experiments — not just "we applied our framework." Something like: *"Whether root barrier traits are coordinated, modular, or independently regulated across genetic and environmental contexts has been impossible to test at scale."* Then the four experiments answer that question.

2. **Make sure each experiment delivers a clear quantitative finding** (not just "the model captured differences"). Each subsection needs at least one number reviewers can quote. Example: *"WOX10 reduced aerenchyma area by X% in rhizotron but had no effect in pouches (p=Y)."*

3. **End with a synthesis paragraph** that pulls the four experiments into a single biological message — what did we learn about barrier biology that we didn't know before? This is the hardest paragraph to write and the one your collaborators must help you with.

4. **Consider splitting Result 5 into two figures** if there's room. Right now you're cramming 4 biological experiments into one figure with 4 panels. That undersells each one. Two figures (e.g., "natural + microbial variation" and "genetic regulation") would give each more space and signal "this is the centerpiece."

#### Methods — mostly fine, but fix factual errors
I noticed several inconsistencies between the manuscript and your CLAUDE.md (which I assume reflects what you actually trained):

- Manuscript says **RTX 4090 24GB**; CLAUDE.md says **H100 86GB** (and Lambda GH200 for some runs)
- Manuscript says **ResNet50** encoder; CLAUDE.md says **ResNet34**
- Manuscript says **SAM 2 (ViT-B)** with frozen encoder + only mask decoder trainable; you actually trained **micro-SAM (vit_b_lm)** which trains prompt encoder + mask decoder + UNETR decoder. These are different methods.
- Manuscript says **4-channel sigmoid output** for U-Net++; you have **6 classes** (root, aer, o.endo, i.endo, o.exo, i.exo)
- Manuscript abstract lists **2 microscopes** but Methods mentions Zeiss too — clarify "trained on 2, deployed on a 3rd"

Fix these before submission — reviewers will catch them.

### The "I'm not a biologist" problem — actually solvable

You're worried about being first author on a biology paper. Three things to know:

1. **This is normal and accepted.** Tool/resource papers in Nature Comms regularly have computational/imaging first authors with biology co-authors. Examples: Cellpose (Stringer is an applied mathematician). StarDist (Weigert is a physicist). PlantSeg (Wolny is a CS researcher). You're in good company.

2. **Frame your contribution honestly in the paper.** Your contribution is: the framework, the dataset, the benchmark, the validation that it produces biologically meaningful measurements, and **demonstrating how it can be used for discovery** in collaboration with experts. Your biology co-authors contribute the specific biological interpretation in Result 5.

3. **Make your collaborators write Result 5 with you, not for you.** Each subsection (microbes, WOX10, tomato mutant) should have a biologist co-author (Stefan, Tamera/Adeel, Kevin) co-drafting that paragraph. They will:
   - Provide the biological framing and interpretation
   - Ensure the claims are appropriately calibrated (not overclaiming)
   - Give you the "what does this mean biologically" sentences you can't write alone
   
   Schedule writing sessions where they tell you the story of their experiment and you take notes. Then write the draft and circulate.

4. **The intro and discussion need a senior biologist's voice.** You can write the technical sections solo. But the intro hook ("why does this matter for plant biology"), the discussion synthesis, and the framing of Result 5 need real biological judgment. Make sure Siobhan or Randy gives these sections a hard pass before submission.

### Suggested new structure

| Section | Status | Length |
|---|---|---|
| Result 1: Dataset | Keep | Same |
| Result 2: Unified model generalizes across species and platforms | Compress (merge benchmark + Zeiss zero-shot + mono/dico into one tighter section) | Halved |
| Result 3: Automated measurements agree with expert annotations | Promote (was Result 4) | Same |
| Result 4: Genotype × environment shapes barrier traits (was Result 5) | **Expand to centerpiece** | Doubled |
| Discussion | Write — biology-led | New |

| Moved to supplement |
|---|
| Per-architecture comparison details, full Table 1 |
| Augmentation ablation (was 3.1) |
| GradCAM and channel occlusion (was 3.3) |
| Per-species and per-microscope IoU breakdowns |
| Annotation SOP |

### What to do next

1. **Talk to Siobhan/Randy this week** about the reframe. Get their buy-in on demoting the ML benchmark and centering the biology. They will likely agree — this is what they want too.
2. **Schedule a working session with each biology co-author** (Stefan, Tamera, Kevin) to draft their Result 5 subsection together. Each session can be 1 hour. This is the highest-value writing you can do.
3. **Fix the methods inconsistencies** (RTX 4090 → H100, ResNet50 → 34, SAM 2 → micro-SAM, 4 → 6 channels). Quick win.
4. **Don't add more ML experiments.** You have enough. Adding architectures or ablations will not help acceptance.

You're closer than you think. The manuscript already contains a Nature Comms-level paper — it just needs to be reordered so the biology breathes and the ML recedes into supporting infrastructure.

---

## Q5: How about figures? how many figures should I prepare, and what should I include in each figure? I want to have visually appealing figures. My microscopy images of different root a very beautiful. The biologiest put in a lot of work to get those images

Great instinct — for a Nature Comms paper with stunning microscopy, **figures carry more weight than text**. Reviewers (and especially editors) flip through figures first, decide if it's interesting, then read. You should plan 5–6 main figures that each tell one clear story and showcase your imagery prominently.

Here's a concrete plan based on the restructured narrative.

### Overall principles before we get to specifics

1. **5 main figures is the sweet spot.** Nature Comms allows up to 8, but 5 punchy figures > 8 cluttered ones. You can put a 6th if your biology section needs to split.
2. **Every figure should have at least one large image panel.** Your microscopy is your asset. Don't bury it in tiny thumbnails.
3. **One consistent color scheme for the 7 anatomical classes**, used in EVERY figure. Reviewers should learn the colors in Fig 1 and recognize them by Fig 5. You already have `TARGET_CLASS_COLORS_RGB` — stick to it religiously.
4. **No bar charts in main figures unless absolutely necessary.** Use box plots, scatter plots, or strip plots. Bar charts read as "methods paper."
5. **Reserve one figure as the visual showpiece** — the one you'd put on the journal cover. For you, that's Figure 1.
6. **Avoid panel labels beyond "g."** If you need (h), (i), (j), the figure is too busy — split it.

---

### Figure 1 — Dataset & root anatomy (THE COVER FIGURE)

**Goal:** Hook the reader with biology and beautiful imagery. Establish what root barriers are, why they matter, and what your dataset looks like.

| Panel | Content | Size |
|---|---|---|
| **(a)** | **Biological schematic** — labeled cross-section illustration showing epidermis → exodermis → cortex (with aerenchyma) → endodermis (Casparian strip) → vasculature, with brief functional annotations ("regulates water/nutrient flux", "gas exchange", etc.). Get a scientific illustrator or use BioRender. | Large (top-left, ~⅓ width) |
| **(b)** | **Hero gallery** — 4 large representative fluorescence images, one per species (millet, rice, sorghum, tomato). Show the raw 3-channel composite. These should be beautiful, large, full-width. | LARGE — half the figure |
| **(c)** | Same 4 images but with annotation overlay using the consistent color scheme. Side-by-side or just below (b). | Same size as (b) |
| **(d)** | **Dataset composition** — sunburst or stacked donut chart: outer ring = species, middle = microscope, inner = growth condition. Avoid a stacked bar chart — sunburst is more visually appealing and Nature-style. | Small (bottom corner) |

**Why this works:** A reviewer who only looks at Fig 1 immediately gets: (1) the biology, (2) the visual diversity of your data, (3) what segmentation classes you predict, (4) the scale and breadth of your dataset.

**Key advice:** Make panel (b) huge. Big, beautiful microscopy is your single best persuasion tool. If the journal cover team picks up your paper, this is the panel they'll use.

---

### Figure 2 — Unified model performance and generalization

**Goal:** Convince the reviewer the model works across the full diversity of your data, including unseen species and microscopes. Compress the ML benchmark into one panel.

| Panel | Content |
|---|---|
| **(a)** | **Qualitative results gallery** — 4–6 representative samples showing original | GT | prediction columns. Pick: 1 easy + 1 hard per species. **This should dominate the figure.** |
| **(b)** | **Per-class performance** — strip/swarm plot of per-sample IoU for each of the 7 classes (best model only). Each dot = one test sample, colored by species. Median bar overlaid. Strip plots show distribution; bar charts hide it. |
| **(c)** | **Cross-microscope generalization** — paired comparison: best-model performance on Olympus vs C10 vs Zeiss (zero-shot). Could be a small connected dot plot or grouped strip plot. One sentence: "Performance held within X% on the unseen Zeiss platform." |
| **(d)** | **Cross-species generalization** — qualitative panel showing zero-shot prediction on unseen Solanum / monkeyflower species. 2-3 example images with overlays. Frame as "deployment to species not in training." |

**Move to supplement:**
- Per-architecture comparison table (YOLO vs U-Net++ vs micro-SAM)
- mAP@0.5 / mAP@0.5:0.95 numbers
- Per-species IoU breakdown bar charts
- Inference speed comparison
- Mono vs dico specialist comparison (Strategy B)

**Key advice:** Don't include 3 architectures' results in Fig 2. Pick the winner and show only its results in the main figure. The architecture comparison goes in Supp Table 1 with one sentence in the main text. This single decision will make Fig 2 dramatically less methods-paper-like.

---

### Figure 3 — Automated measurements agree with expert quantification

**Goal:** Convince biology reviewers the model produces measurements they can trust. This is the bridge from "ML works" to "biology discoveries are valid."

| Panel | Content |
|---|---|
| **(a)** | **Aerenchyma area ratio scatter** — automated vs expert, points colored by species, regression line, R² in corner. |
| **(b)** | **Suberin (TRITC in endodermis) scatter** — same format. |
| **(c)** | **Lignin (FITC in endodermis + vasculature) scatter** — same format. |
| **(d)** | **Visualization of measurement extraction** — one sample showing: original image → predicted endodermis mask → intensity heatmap within mask. Makes the workflow concrete and visually beautiful. |
| **(e)** | **Throughput** — small inset: "1,600 samples in X min (model) vs Y hours (manual)" as a clean bar pair or just a callout box with big numbers. Reviewers love a single dramatic number. |

**Key advice:** The R² values need to be high (>0.85 ideally) for this figure to land. If any are weak, drop that panel rather than show it. A figure with three strong scatter plots is better than four mediocre ones.

---

### Figure 4 — Genotype variation and environmental plasticity (Biology Part 1)

**Goal:** First half of the biological discovery — natural variation and environmental responses.

| Panel | Content |
|---|---|
| **(a)** | **Sorghum genotype comparison** — 2 representative samples (one per genotype) with prediction overlays, side-by-side. Beautiful imagery + clear visual difference. |
| **(b)** | **Quantitative comparison across sorghum genotypes** — box plot of barrier traits (e.g., aerenchyma ratio, endodermis suberin intensity) split by genotype. Include the held-out genotype prominently — annotate "not in training." |
| **(c)** | **Microbial perturbation gallery** — 6 representative samples (one per microbial isolate) with overlays. Visual diversity in barrier responses. |
| **(d)** | **Microbial trait heatmap** — rows = barrier traits (aerenchyma ratio, suberin, lignin), columns = microbial isolates, color = z-score. Shows that responses are trait-specific, not uniform. This is the headline visualization for the microbe story. |
| **(e)** | **Quantitative box plots** for the 1–2 most striking microbial effects with statistical annotations (asterisks for significance). |

**Key advice:** The heatmap in (d) is critical — it visually proves your "barriers are multi-dimensional, not uniform" thesis. Make it prominent.

---

### Figure 5 — Genetic regulation of barrier traits (Biology Part 2)

**Goal:** Second half of the biological discovery — targeted genetic perturbations validate the framework on cell-type-specific phenotypes.

| Panel | Content |
|---|---|
| **(a)** | **WOX10 mutant rice imagery** — WT vs wox10 across growth systems (paper pouch, sand, rhizotron). 6 images in a 2×3 grid with prediction overlays. |
| **(b)** | **WOX10 quantification** — box plot of aerenchyma ratio: WT vs wox10 × 3 growth systems. Shows the rhizotron-specific effect. The biological punchline: "the genetic effect only manifests in one environment." |
| **(c)** | **Tomato suberin mutant imagery** — WT vs mutant, 2-4 representative images with prediction overlays. Highlight the exodermal layer specifically. |
| **(d)** | **Tomato exodermis quantification** — box plot of exodermal suberin intensity: WT vs mutant. Show that the effect is restricted to the exodermis (compare with endodermis as a negative control panel). |
| **(e)** | **Synthesis schematic** — small conceptual diagram tying together: "barriers are regulated by genotype × environment × tissue, and our framework resolves all three axes." Optional but powerful as a closing visual. |

**Key advice:** The contrast between WOX10 (environment-dependent) and tomato (cell-type-specific) is the core synthesis. Make sure (b) and (d) tell that story side by side. If panel (e) feels too speculative, drop it — the schematic should only go in if Siobhan/Randy can defend it.

---

### Decision: 5 vs 6 figures

If your biological data is strong, **split Figs 4+5 into 2 figures** (as above) and aim for 5 total. If the biology data is thinner than expected, combine Result 5 into one figure and aim for 4 main figs + 1 generalization fig.

**My recommendation: 5 figures.** Two biology figures signal "this is a biology paper with a tool," not "a tool paper that mentions biology." That positioning is what gets you past the editor.

---

### Visual design checklist

- **Use a single, consistent color palette across all figures.** Pick one for the 7 classes and never deviate. Color blindness: avoid red/green pairs — use the colors you already have (`TARGET_CLASS_COLORS_RGB`) but verify with a simulator.
- **Image panels should fill ≥40% of total figure area.** Quantitative plots ≤60%. Tilt the ratio toward imagery.
- **Use fluorescence pseudocolor consistently.** TRITC=red, FITC=green, DAPI=blue throughout. Don't switch.
- **Scale bars on every microscopy image** in white, lower-right corner, with the value in the figure caption (not on the bar).
- **Font: sans-serif (Helvetica or Arial), 7pt minimum.** Nature journals require 5–7pt for axes and 8pt for labels. Use the same font everywhere.
- **No 3D effects, no gradients, no drop shadows.** Flat design only.
- **High DPI:** 300 dpi minimum, 600 dpi for line art. Save as PDF (vector) or TIFF (raster) — never PNG for submission.
- **Annotate inside the panel, not in the caption,** when possible. e.g., "n.s.", "P<0.01", "R²=0.94" inside the panel saves the reader from cross-referencing.
- **Use uniform panel margins.** Make all subpanel borders align — Nature reviewers notice misaligned grids.
- **Test at print size.** Print Fig 1 at the actual journal column width (typically 89mm single-column or 183mm double-column) before deciding panel sizes are large enough.

---

### Tools I'd use for this

- **Layout & assembly:** Affinity Designer, Inkscape, or Illustrator. Avoid PowerPoint for final figures (rasterization issues).
- **Plots:** matplotlib/seaborn with a single shared style file. Save as SVG, then assemble in your layout tool.
- **Schematic illustrations (Fig 1a, Fig 5e):** BioRender (paid but Nature-quality) or hire a scientific illustrator on Fiverr/Upwork for $100–300. Worth it.
- **Microscopy display:** Use Fiji/ImageJ for consistent contrast adjustments, save as PNG/TIFF, then place in the layout. Apply the **same** percentile clip across all images of the same modality so they look comparable.
- **Color palette:** ColorBrewer or Okabe-Ito palette (colorblind-safe).

---

### What I'd do this week

1. **Sketch all 5 figures by hand on paper first.** 30 minutes. Decide what story each figure tells before opening any software.
2. **Curate your "hero images"** — go through your dataset and pick the 8–12 most beautiful, representative samples. These will appear in Figs 1, 2, 4, 5. Spending an afternoon on image selection pays off enormously.
3. **Build a shared matplotlib style file** with the consistent color palette, font, and figure dimensions. Use it for every plot from now on.
4. **Make Fig 1 first.** It's the most impactful and will set the visual tone for the rest.

The biology team put in real effort on those microscopy images — honor that work by giving them the visual prominence they deserve. A figure where 60% of the space is gorgeous root cross-sections does more for acceptance than any ablation table ever will.

---

## Q6: For figure 2, I think I should at least have one figure to show my metrics between different models and human-to-human upper bound.

You're right, and actually that's a stronger instinct than what I suggested. **The human upper bound completely changes the framing of Figure 2** — it transforms "our model beats other models" (which reads as cherry-picked benchmark) into "our model approaches expert-level agreement" (which is a Nature-level claim). Keep it in the main figure and make it prominent.

Two important questions before the panel design:

### First — do you actually have the human-to-human data?

I didn't see this in your manuscript outline. To make this claim you need a second expert annotator (not Labellerr — a domain expert) to independently re-annotate at least ~20-30 test samples. Then you compute IoU between annotator A vs annotator B, and that becomes your upper bound.

If you don't have this yet, **this is the single highest-value experiment you can run before submission.** It costs you ~1 day of a biologist co-author's time and adds enormously to the paper. Ideal candidates: Stefan, Tamera, or Kevin re-annotating 20-30 samples.

If you can swing it, also do **intra-annotator agreement** (same person re-annotating after a few weeks) — this gives you a second reference line and is even cleaner methodologically.

### Revised Figure 2 — keep the model comparison, anchor to human bound

| Panel | Content |
|---|---|
| **(a)** | **Qualitative gallery** — 4-6 representative samples (one per species, easy + hard) showing original \| GT \| best-model prediction. Still the dominant visual element. |
| **(b)** | **The headline quantitative panel** — per-class IoU on the 7 bio classes, with: <br>• Each model (YOLO, U-Net++, micro-SAM) as a distinct color/marker <br>• **Horizontal dashed line per class showing human-vs-human IoU** <br>• Strip plot (each dot = one test sample) with model means as larger markers <br>• Annotation: "best model approaches inter-annotator agreement on 5/7 classes" |
| **(c)** | **Cross-platform generalization** — best model on Olympus / C10 / Zeiss (zero-shot). Compact strip plot or paired connected dots. |
| **(d)** | **Cross-species generalization** — qualitative panel: zero-shot prediction on Solanum / unseen species. 2-3 example images with overlays. |

This is one panel busier than my original suggestion, but the human bound is worth it. Drop my original (c) and (d) into one combined "generalization" panel to keep total panel count at 4.

### Why panel (b) is the most important plot in your paper

Visual format matters here. Bad version: bar chart with 3 models × 7 classes = 21 bars, looks like a methods paper, eye glazes over.

**Good version**: per-class strip plot, one column per class (whole root, epidermis, exodermis, cortex, aerenchyma, endodermis, vasculature). Within each column:
- 3 colored swarms of dots (one per model), each dot = one test sample
- A larger black marker showing the model's mean
- A **horizontal dashed gray line spanning the column showing inter-annotator IoU**
- If your best model's distribution overlaps the human line, that visually ends the argument

This single panel says, in one glance:
1. We benchmarked 3 architectures (rigor)
2. The best one is consistently strong across classes (robustness)
3. It approaches human-level agreement (the killer claim)
4. The hard classes (probably aerenchyma) are hard for humans too (proper context for any class where you don't hit the bound)

That's exactly the framing reviewers can't argue with. "Your model isn't perfect" → "neither are humans, and we're within their disagreement range."

### Two related design choices

**1. Show the gap honestly.** If your model is below human on aerenchyma or exodermis (likely), show it. Don't hide weak classes. Reviewers respect honesty, and the discussion can frame it: *"Aerenchyma remains challenging even for expert annotators (inter-rater IoU = 0.X), reflecting genuine biological ambiguity in cavity boundaries."* That turns a weakness into a domain-knowledge insight.

**2. Architecture comparison stays in main text but tight.** Now that you have the human bound as the y-axis anchor, you can keep all 3 architectures in panel (b) without it feeling like a methods paper — because the comparison is no longer "model vs model" but "models vs human ceiling." Different framing, same data. The architecture details (params, FLOPs, inference speed, mAP@0.5:0.95) still go to Supp Table 1.

### Updated supplementary figures from this section

| Supp Fig | Content |
|---|---|
| Supp Fig X | Full architecture comparison table (params, mAP, inference speed) |
| Supp Fig Y | Per-species IoU breakdown for the best model |
| Supp Fig Z | Per-microscope IoU breakdown for the best model |
| Supp Fig W | Mono vs dico specialist vs unified (Strategy B) |
| Supp Fig V | Augmentation ablation |
| Supp Fig U | Intra-annotator agreement (if you collect it) |

### Action item

**Before anything else: send a message to Stefan/Tamera/Kevin this week** asking if one of them can re-annotate ~20-30 test samples for inter-annotator agreement. Frame it: *"This will let us claim our model approaches expert-level performance in the paper, which significantly strengthens acceptance odds. It's about 1 day of work and will be acknowledged in the methods."*

This is the single most cost-effective addition you can make to the entire manuscript. Don't skip it.

---

## Q7: What about the visualization for embeddings of images of different species? Is it weird to have that in figure 2?

Not weird at all — actually it **belongs** in Figure 2 if you frame it right. Here's why.

### The UMAP isn't really about "explainability" — it's evidence for your generalization claim

In your current outline, UMAP sits in the explainability section (Result 3), which positions it as a "look how interpretable our model is" curiosity. That's the wrong framing and it's why it feels disconnected.

The real claim a UMAP supports is: **"the model learned biological structure, not imaging artifacts."** That's exactly what you're already asserting in the cross-microscope/cross-species generalization panels of Figure 2. The empirical evidence is "performance holds on Zeiss" — the UMAP is the mechanistic evidence for *why* it holds.

So don't think of it as "should I add explainability to Fig 2?" Think of it as "does the UMAP make my generalization claim more convincing?" Yes — strongly.

### How to include it without overcrowding Fig 2

Use **two small UMAP panels side by side** as a single composite, not two separate panels:

| Panel | Content |
|---|---|
| **(a)** | Qualitative gallery (4–6 representative samples) |
| **(b)** | Per-class IoU strip plot with human-vs-human upper bound |
| **(c)** | Cross-platform: Olympus / C10 / Zeiss zero-shot performance |
| **(d)** | Cross-species: zero-shot prediction overlays on unseen species |
| **(e)** | **UMAP composite — two small plots side by side**: same embedding, colored by species (left) and by microscope (right). One short caption: *"Embeddings cluster by species but not by microscope, indicating the model encodes biological structure rather than platform artifacts."* |

Five panels is a lot, but (c), (d), and (e) can be small — they're all serving the same generalization argument and can sit on one row at the bottom.

If that feels too crowded, the cleanest alternative is to **merge (c) and (e)**: cross-platform IoU on the left, side-by-side UMAPs on the right, as a single "generalization across platforms" panel. The IoU bar/strip plot says "performance holds"; the UMAPs say "and here's why."

### The pairing that makes it land

The two-color UMAP comparison only works if you show **both colorings of the same plot**. Don't show one. The comparison is the entire argument:

- UMAP colored by **species** → clusters cleanly = "model learned species-distinguishing biology"
- UMAP colored by **microscope** → clusters mix = "model is invariant to imaging platform"

If both are shown together, the reviewer's eye does the work in 2 seconds. If you show only one, the claim doesn't land.

**Critical:** check this actually works on your data before committing to the panel. If your embeddings *do* cluster by microscope (they might — pretrained encoders often pick up illumination differences), then this panel undermines your generalization claim instead of supporting it. Run the UMAP first, look at the result, decide.

### What about the rest of explainability (GradCAM, channel occlusion)?

Still supplement. They're nice-to-have but they don't make a load-bearing claim that justifies main-text real estate. The UMAP is the only explainability piece that does double duty as evidence for a primary claim.

A possible exception: if your **channel occlusion** result is dramatic (e.g., zeroing TRITC causes endodermis IoU to collapse), that's a strong figure for the discussion section because it ties the model directly to known biology (suberin/lignin staining). But it's not a Fig 2 thing — it's a Discussion or supplementary insight.

### Quick action

Generate the UMAPs this week before locking the figure layout. Two outcomes:
1. **Embeddings cluster by species, not microscope** → put it in Fig 2 as panel (e). Major win.
2. **Embeddings cluster by microscope too** → leave it in supplement, tone down the generalization framing slightly, and rely on the empirical Zeiss IoU as your main evidence.

You don't have to commit to including it until you see the plot. But the upside if it works is large — it converts a "trust the numbers" argument into a "see for yourself" argument, which is always more persuasive.

---

## Q8: Are you sure including embedding umap is a better than including GradCAM to show where the model is paying attention?

Honestly? You're pushing me to think harder, and I think you're right to question it. Let me revise.

### The honest comparison

I was anchoring on "UMAP supports the generalization claim" but I undersold GradCAM. For **a biology audience reading a biology journal**, GradCAM is probably the better choice. Here's why:

| | UMAP | GradCAM |
|---|---|---|
| **What it shows** | Embeddings cluster by biology, not platform | Where the model looks for each class |
| **What a biologist sees** | Abstract scatter — needs interpretation | Heatmap on microscopy — instant intuition |
| **Validates against domain knowledge?** | Indirectly | Directly — "yes, that's the endodermis" |
| **Ties to your stains?** | No | **Yes** — can show FITC drives endodermis, TRITC drives suberin, etc. |
| **Robust to method choice?** | UMAP hyperparameters can be fiddly | Has known limitations but is standard |
| **Redundancy with other panels?** | Overlaps with empirical Zeiss IoU | No redundancy — only way to show attention |
| **Risk if it doesn't work** | Embeddings might cluster by microscope → no panel | Heatmaps might look diffuse → less dramatic but still publishable |

Two things I underweighted before:

1. **GradCAM is the only piece that directly ties your model to your stains.** Your three fluorophores have specific biological meaning (TRITC = suberin, FITC = lignin, DAPI = cell walls). If GradCAM shows the model uses FITC signal to find endodermis (because of the lignified Casparian strip), that's a *biological* validation, not just a *statistical* one. UMAP can't do that.

2. **UMAP partially duplicates what your Zeiss zero-shot panel already proves.** The empirical claim "model generalizes across microscopes" is already supported by the IoU numbers. UMAP is mechanistic backup. GradCAM, in contrast, answers a question nothing else in your paper answers.

For a biology reviewer, "the model looks at the lignified ring to find the endodermis" is far more convincing than "embeddings don't cluster by microscope."

### Revised recommendation

**Move GradCAM into Figure 2 (or its own panel set), demote UMAP to supplement.**

A small GradCAM panel could be the closing panel of Fig 2 — or, if it's strong, become the centerpiece of a brief "model uses biologically meaningful features" subsection that bridges Fig 2 and Fig 3.

### Even stronger: pair GradCAM with channel occlusion

If you want the strongest possible "the model uses biology" argument, pair two complementary pieces of evidence:

1. **GradCAM** (gradient-based) — *spatially* where does it look? → cell wall ring vs cavity vs vascular bundle
2. **Channel occlusion** (intervention-based) — *which channel* does it depend on? → drop TRITC and watch endodermis IoU fall

These two together are much stronger than either alone, because GradCAM has known reliability criticisms in ML and channel occlusion is direct causal intervention. A reviewer who dismisses GradCAM still has to accept the occlusion result.

| Panel | Content |
|---|---|
| (left) | **GradCAM heatmaps** — 4 columns (one per key class: endodermis, exodermis, aerenchyma, vasculature), each on a representative sample. Overlay on raw image. |
| (right) | **Channel occlusion bar plot** — for each class, IoU drop when DAPI / FITC / TRITC is zeroed. Shows endodermis depends on FITC, suberin/exodermis depends on TRITC, etc. |

Caption pulls it together: *"The model attends to lignified rings for endodermis (a) and depends causally on the FITC channel (b), consistent with Casparian strip biology."*

That's a one-paragraph claim that biologists will love.

### So where does this go in the figure layout?

You have two options:

**Option A — Fold into Fig 2 as panel (e)**
Keep my earlier Fig 2 layout but replace the UMAP panel with a small GradCAM panel (just 2-3 example heatmaps).

**Option B — Make it a small dedicated figure (becomes Fig 3)**
A short figure with: GradCAM panel + channel occlusion panel + one sentence of biological interpretation. This bumps everything else down and gives you 6 main figures, but explainability gets its own moment instead of being squeezed into the corner of Fig 2.

I'd lean **Option B** if your biology reviewers are likely to scrutinize "is the model really learning biology or just memorizing?" — a dedicated mini-figure makes this argument cleanly. **Option A** if you're tight on figure count and want to keep momentum toward the biology figures.

### And UMAP?

Move it to supplement. It's still useful evidence, just not load-bearing. One supp figure with both colorings (species, microscope) and one paragraph in the methods/results saying "consistent with the GradCAM analysis, embeddings clustered by species rather than microscope (Supp. Fig. X)."

### TL;DR

You were right to push back. **GradCAM > UMAP for this paper.** It directly ties the model's decisions to your fluorescent stains, which is the most biologically interpretable evidence you can give. Pair it with channel occlusion for a one-two punch that even skeptical reviewers will accept. UMAP becomes supporting material in the supplement.

Sorry for the back-and-forth — your instinct on this was better than mine.

---

*End of conversation. Saved 2026-04-09.*
