# Retinal Disease Classification — Technical Project Summary

**Repository:** `github.com/pshashik0710/retinal-disease-classification`
**Active branch:** `rebuild` (original submission preserved on `archive/ieee-access-submission`)
**Evidence reference:** `reports/master_evidence_table.md` (commit `40715e9`). Every quantitative result below is taken from that table or the committed report it cites. Items not re-verified against a committed file are marked as such.

---

## Contents

1. Project overview
2. The original submission and why it was rebuilt
3. Rebuild design principles
4. Compute environment
5. Datasets and curation
6. Pipeline architecture
7. Phase 1 — Rebuild baseline
8. Phase 2 — OCT harmonisation with CycleGAN
9. Phase 3 — OCT↔CFP cross-modal evaluation
10. Claims supported by the evidence
11. How the rebuild answers the original weaknesses
12. Limitations
13. Engineering and reproducibility notes
14. Commit history
15. Open items

---

## 1. Project overview

The project studies automated classification of age-related macular degeneration (AMD) and related retinal disease from two imaging modalities:

- **Optical coherence tomography (OCT)** — depth-resolved, cross-sectional B-scans of the retina, acquired as single-wavelength backscatter intensity (hence grayscale).
- **Colour fundus photography (CFP)** — an en-face photograph of the retinal surface.

It was rebuilt from an earlier submission to IEEE Access, which was rejected. The rebuild replaced a single headline-accuracy result with an evidence-first study in three phases:

| Phase | Question | Outcome |
|---|---|---|
| 1. Rebuild baseline | How well can a leakage-controlled pipeline classify pooled OCT, and what confounds it? | 0.840 ± 0.004 macro-F1 (4-class, 5 seeds); cohorts near-perfectly separable in feature space |
| 2. OCT harmonisation | Can a CycleGAN remove cohort information between two OCT datasets? | No — a controlled negative result |
| 3. Cross-modal evaluation | Do disease representations transfer between OCT and CFP, and how do the modalities compare? | No transfer in either direction; CFP matches OCT for AMD vs healthy, OCT far ahead for AMD vs diabetic disease |

The project supervisor required a GAN component (Phase 2) and an OCT↔CFP cross-modal experiment with 5-fold evaluation (Phase 3).

---

## 2. The original submission and why it was rebuilt

### 2.1 What the original system did

Preserved on `archive/ieee-access-submission`. The core files were `train_cyclegan.py`, `generate_synthetic_images.py`, `train_two_stream.py` and `train_evaluate_two_stream.py`, with separate `train_oct.py` and `train_amd.py` streams.

The evaluated model (`train_evaluate_two_stream.py`) was a **two-stream fusion network**:

- **Stream inputs:** a real OCT image from `oct2017/test` and a synthetic OCT image from `generated_images/synthetic_OCT`.
- **Backbone:** one shared ConvNeXt-Tiny applied to both inputs.
- **Fusion:** the two feature vectors concatenated, then `Linear(2d → 512) → ReLU → Dropout(0.5) → Linear(512 → 4)`.
- **Data loading:** `torchvision.ImageFolder` over the dataset's own folder structure; no patient-level split control.

The reported headline accuracy of the submission was approximately 96.77%. That figure is not reproduced in the rebuild and is not used as evidence.

### 2.2 The structural flaw

The test dataset paired the real image at index `idx` with the file `synthetic_OCT_{idx+1}.png`. The pairing was by **array position only** — not by patient, eye, class or any deliberate correspondence. Consequences:

1. **Both streams were OCT.** The "multimodal" model was one modality presented twice, with no colour fundus input anywhere in the evaluated file.
2. **The second stream carried no aligned information.** The fusion layer concatenated features of two unrelated images.

### 2.3 Documented reviewer concerns

From the project record, the reviewers identified:

- **Incompatible label structures** across the combined datasets.
- **Single-modality results presented as multimodal** — explained exactly by §2.2.

The rebuild additionally hardened leakage control, dataset integrity, synthetic-image validation, ablations and reproducibility. The full review text was not re-examined in this summary; mapping each individual reviewer comment to a committed artifact remains an open item (§15).

---

## 3. Rebuild design principles

1. **Patient/group-disjoint evaluation everywhere.** No patient's images appear in more than one of train / validation / test, or in more than one cross-validation fold.
2. **Committed manifests as the single source of truth.** Every image is a manifest row; splits are precomputed, verified and committed; training code never re-splits.
3. **Controls before claims.** High numbers are tested with label-shuffle controls, learning curves, matched-preprocessing comparisons and sample-size controls.
4. **Negative results are kept.** A result that fails is reported with the evidence showing why.
5. **Provenance.** Configs, checkpoints metadata, loss logs and every report are committed; the evidence table names the file behind each number.
6. **Frozen-feature linear probes as the primary protocol** (rationale in §4 and §6.3).

---

## 4. Compute environment

| Resource | Use |
|---|---|
| Local Windows machine, CPU only, conda environment `retinal` | Data curation, manifests, feature caching, all linear-probe experiments, all cross-modal experiments |
| Kaggle notebook, 2× Tesla T4 (15 GB each) | CycleGAN training, image translation, GAN evaluation, translated-feature caching |

**Why CPU forced the frozen-feature design.** Measured on the local CPU (ConvNeXt-Tiny, batch 16, 224 px), recorded in `scripts/cache_features.py`:

| Mode | Time per batch | Time per epoch (42,838 images) |
|---|---:|---:|
| Full fine-tuning | ~32.6 s | ~8 h |
| Frozen backbone, images each epoch | ~14.5 s | ~3.5 h |
| Cached features (one pass, then head only) | one traversal | seconds per epoch |

A frozen backbone produces identical features every epoch, so caching them once converts a 100+ hour training schedule into one feature pass plus seconds of head training. Individual CPU tasks otherwise exceeded 12 hours.

**GPU speed-up for the GAN.** On one T4 at 128 px: ~18 minutes per epoch at batch 4 (1.85 it/s, 2,007 iterations), ~14.5 minutes per epoch at batch 8 (1.15 it/s, 1,003 iterations).

---

## 5. Datasets and curation

### 5.1 Sources

| Dataset | Modality | Classes used | Notes |
|---|---|---|---|
| Kermany OCT2017 | OCT | NORMAL, DRUSEN, CNV, DME | Public; Kaggle mirror `paultimothymooney/kermany2018`. Five export widths (512 / 768 / 1024 / 1536 × 496, and 512 × 512). DME occurs only in this dataset. |
| NEH (NEH_UT_2021) | OCT | NORMAL, DRUSEN, CNV | Uploaded as `neh-cleaned-12565` |
| HYAMD | CFP | CONTROL, AMD (binary); CONTROL, AMD_EARLY, AMD_LATE (staging) | Single site, single Topcon camera, two export widths (2576 and 1960, both 1934 high). Labels from full clinical evaluation **including OCT**. CONTROL = diabetic-retinopathy patients **without** AMD, not healthy eyes. |
| AMDNet23 | CFP | NORMAL, AMD, DIABETIC, CATARACT | Compiled from multiple public sources (`aria`, `bare`, `img`, `odir`, `other`); 751 distinct image sizes |
| ODIR (subset of AMDNet23) | CFP | NORMAL, AMD, DIABETIC, CATARACT | The only AMDNet23 source contributing all four classes |

### 5.2 Manifest schema

Each manifest row records: `Directory` (path relative to a cohort root), `cohort`, `patient_key`, `group_key`, `eye_key`, class labels (`Class`, `Label`, `y_label`) and integer label `y`, `Eye`, `B-scan`, `width`, `height`, `root_hint`, and `split`.

- **Relative paths plus per-cohort roots** make manifests portable: the same manifest runs on Windows (`D:\datasets\...`) and Kaggle (`/kaggle/input/...`) by changing only the root mapping.
- **`group_key`** is stricter than `patient_key`: it links patients who share duplicate images, so splitting by group is automatically patient-disjoint as well. In the OCT AMD-vs-NORMAL subset, 4,755 patients form 4,753 groups.
- **Class order is load-bearing.** `CLASSES = [NORMAL, DRUSEN, CNV, DME]` defines the integer labels written into the manifests; reordering it would silently mislabel every metric.

### 5.3 Pooled OCT manifest (`manifests/pooled_split.csv`)

| Cohort | Images | Patients |
|---|---:|---:|
| Kermany | 54,742 | 4,656 |
| NEH | 11,712 | 437 |
| **Total** | **66,454** | **5,093** |

Split: **train 42,838 / validation 10,244 / test 13,372**, patient/group-disjoint. At most 50 images per patient.

The split is a patient-level reassignment, not Kermany's original folders: images stored under Kermany's `train/` folder can carry `split = val`. The `split` column is authoritative; the folder name is only part of the file path.

### 5.4 Task subsets

| Subset | Images | Patients / groups | Composition |
|---|---:|---:|---|
| OCT AMD vs NORMAL | 58,224 | 4,755 / 4,753 | NORMAL 29,794; AMD = DRUSEN + CNV 28,430; DME excluded. 404 groups carry both labels (all Kermany). |
| OCT AMD vs DME, Kermany only | 26,649 | 1,767 groups | DME 7,035; AMD 19,614; 144 mixed AMD/DME groups dropped |
| HYAMD binary | 1,560 | 325 patients | AMD 532 (135 patients); CONTROL 1,028 (190 patients); no mixed patients |
| ODIR | 1,303 | 879 patients | Splits 558 / 150 / 171 patients, zero overlap; zero patient overlap with other AMDNet23 sources. AMD + NORMAL subset: 745 images, 512 groups. |

**Mixed-label groups.** Kermany assigns labels per scan while IDs are per patient, so one patient can appear under two classes — plausibly a diseased eye and a healthy or differently-diseased fellow eye. Grouping keeps such a patient inside one fold, so there is no leakage; the issue is only label ambiguity. Both were tested (§9.6, §9.4).

### 5.5 Dataset-audit figures not re-verified in this summary

Kermany ~76,677 images before capping; 466 cross-class-conflict images from 56 patients removed; NEH 550 eyes. These come from the dataset audit and should be checked against `reports/integrity_*.txt` before citation.

---

## 6. Pipeline architecture

### 6.1 Configuration tracks (`scripts/config.py`)

One config holds several studies. A **track** fixes the manifest, cohort roots, class list, resize strategy and an optional row filter. Feature caches and outputs are namespaced by track so that switching tracks cannot silently reuse the wrong vectors.

| Track | Manifest | Classes | Resize |
|---|---|---|---|
| `oct` | `pooled_split.csv` | NORMAL, DRUSEN, CNV, DME | normalize_768 |
| `cfp_hyamd` | `hyamd_binary.csv` | CONTROL, AMD | resize_crop |
| `cfp_hyamd3` | `hyamd_staging.csv` | CONTROL, AMD_EARLY, AMD_LATE | resize_crop |
| `cfp_amdnet23` | `amdnet23_clean.csv` | NORMAL, AMD, DIABETIC, CATARACT | resize_crop |
| `cfp_odir` | `amdnet23_clean.csv`, filtered to `source = odir` | NORMAL, AMD, DIABETIC, CATARACT | resize_crop |
| `oct_match_*` | patient-count-matched OCT subsets | NORMAL, AMD | normalize_768 |
| `oct_translated` | `pooled_translated_neh.csv`, root = translated images | NORMAL, DRUSEN, CNV, DME | resize_crop |
| `oct_resized_baseline` | same manifest, root = resized-only images | NORMAL, DRUSEN, CNV, DME | resize_crop |

The two translation tracks read their roots from environment variables (`OCT_TRANSLATED_ROOT`, `OCT_BASELINE_ROOT`), defaulting to the Kaggle paths, so the committed config is portable.

### 6.2 Resize strategies

OCT B-scans occur at six native aspect ratios (0.774–3.097), so the resize step can differ between cohorts in ways correlated with the cohort itself.

- **resize_crop** — resize the short side, then centre-crop.
- **normalize_768** — pass both cohorts through a common intermediate resolution first, so the centre crop costs them equally.
- **pad** — letterbox to square.
- **squash** — resize to square ignoring aspect ratio.

### 6.3 Feature extraction (`scripts/cache_features.py`)

- Backbone: ConvNeXt-Tiny, ImageNet-pretrained, frozen; input 224 px; ImageNet normalisation.
- Output per split: `.npz` holding `features (N, 768)`, `labels (N,)` and `index (N,)` — the row index into that split's manifest rows, so every prediction traces back to its patient, cohort and file.
- Features are computed without augmentation (an augmented image produces different features each epoch and cannot be cached). This is the standard linear-probe protocol.

### 6.4 Classification heads

- **Baseline experiments** (`scripts/train.py`, recorded in each `summary.json`): linear head on cached features; learning rate 0.001; batch 512; weight decay 1e-4; label smoothing 0.1; class-weighted loss; up to 30 epochs with early stopping, model selected on validation macro-F1.
- **Probe and cross-modal experiments:** scikit-learn `LogisticRegression(max_iter=2000, class_weight="balanced")`. Logistic regression on fixed features is convex and deterministic, so refitting with different seeds returns the same model; uncertainty is therefore estimated by resampling data (bootstrap or cross-validation), never by seed sweeps.

---

## 7. Phase 1 — Rebuild baseline

### 7.1 Pooled OCT 4-class baseline — final configuration

normalize_768, five seeds. Sources: `outputs/normalize_768`, `outputs/seed1`–`seed4`.

| Seed | Accuracy | Macro-F1 |
|---|---:|---:|
| 42 | 0.8619 | 0.8390 |
| 1 | 0.8578 | 0.8365 |
| 2 | 0.8716 | 0.8467 |
| 3 | 0.8621 | 0.8403 |
| 4 | 0.8573 | 0.8368 |
| **Mean ± SD** | **0.862 ± 0.006** | **0.840 ± 0.004** |

The seed SD of **0.004 macro-F1** is the noise floor for reading every single-seed ablation.

**Per-class F1** (mean of the same five runs): NORMAL ≈ 0.914, CNV ≈ 0.883, DME ≈ 0.840, **DRUSEN ≈ 0.723**. DRUSEN is consistently the weakest class, with precision (~0.64–0.71) well below recall (~0.75–0.83): the model over-predicts drusen. Whether residual drusen errors arise from resolution loss in preprocessing or from label ambiguity is an open question (§15).

### 7.2 Ablations (single seed 42)

| Factor | Run | Accuracy | Macro-F1 | Reading |
|---|---|---:|---:|---|
| Resize | normalize_768 (final) | 0.8619 | 0.8390 | Reference |
| Resize | resize_crop (`baseline_v2`) | 0.8579 | 0.8347 | Within seed noise |
| Resize | squash | 0.8440 | 0.8157 | −0.023 |
| Resize | pad | 0.8315 | 0.8102 | −0.029, worst |
| Dropout | `do0` | 0.8579 | 0.8347 | Identical to `baseline_v2` |
| Class weights | `no_weights` | 0.8572 | 0.8199 | Weighting adds +0.015 macro-F1 at unchanged accuracy |
| Learning rate | `lr3e-3` | 0.8480 | 0.8223 | −0.012 |

Preprocessing is not neutral: the choice between normalize_768 and pad moves macro-F1 by ~0.03, several times the seed noise.

The first run, `baseline_cached_probe` (accuracy 0.8494, macro-F1 0.8241), is superseded by the final configuration and kept only as history.

### 7.3 Cohort probe — the central confound (`reports/probe_controls.json`)

A linear probe trained to predict **which dataset** an OCT image came from (Kermany vs NEH), on the same frozen features:

| Measure | Value |
|---|---|
| Held-out test accuracy | **0.9964** (train 0.9997; majority baseline 0.8158) |
| Train–test gap | 0.0033 — not overfitting |
| Label-shuffle control (10 permutations) | 0.5201 ± 0.0034; real result 140.3 SD above it |
| Learning curve | 100 images → 0.967; 500 → 0.975; 2,000 → 0.985; 10,000 → 0.994; 42,838 → 0.9964 |
| Within-Kermany export-width probe | 0.8982 vs baseline 0.6589 |

**Interpretation.**

- The shuffle control collapses, so the pipeline is not inflating scores; the signal is real.
- The learning curve saturates at 100 images, so the cohort difference is not subtle — a few obvious directions in feature space.
- The width probe is the key: **inside Kermany alone**, with NEH not involved, the features predict the export width at 0.898. A large share of the "cohort" signal is therefore preprocessing/export properties, not pathology or scanner optics.
- Kermany and NEH differ at once in device, export resolution, file format, site, population and labelling practice. The supportable claim is **cohort information**, not "scanner bias".

### 7.4 Patient-matched OCT, binary NORMAL vs AMD

Binary task, five seeds each — not comparable to the 4-class baseline.

| Matched to CFP cohort size | Patients | Macro-F1 |
|---|---:|---:|
| ODIR | 879 | 0.934 ± 0.004 |
| AMDNet23 | 1,420 | 0.926 ± 0.003 |

Not monotonic in patient count, so not a learning curve. The 325-patient (HYAMD-matched) track has no committed output.

### 7.5 CFP baselines (single split, five seeds)

| Dataset / task | Macro-F1 | Caveat |
|---|---:|---|
| AMDNet23, 4-class | 0.856 ± 0.019 | **Source confounded with class**: a source probe scores 0.9176 vs baseline 0.6841 (shuffle 0.448 ± 0.029). `aria` is 100% AMD; `other` is 99.5% cataract — recognising the source predicts the label. |
| HYAMD, AMD vs CONTROL | 0.607 ± 0.046 | Hard task: CONTROL is diabetic retinopathy, not healthy |
| HYAMD, 3-class staging | 0.426 (one run) | AMD_EARLY has 20 patients; secondary only |

The AMDNet23 source confound is why ODIR (single source, all four classes) is used as the clean CFP comparison.

---

## 8. Phase 2 — OCT harmonisation with CycleGAN

### 8.1 Rationale

The cohort probe (§7.3) shows Kermany and NEH are nearly perfectly separable. If a model partly learns "which dataset" rather than "which disease", mapping both cohorts into one appearance might reduce that shortcut. This is **same-modality harmonisation**: both domains are OCT B-scans of the same anatomy, differing in acquisition.

### 8.2 Architecture and training (`harmonization/train_cyclegan.py`)

| Component | Setting |
|---|---|
| Generators G_AB (Kermany→NEH), G_BA (NEH→Kermany) | ResNet generator, 6 residual blocks, **7,837,699 parameters each**, tanh output on [−1, 1] |
| Discriminators D_A, D_B | 70×70 PatchGAN, **2,764,737 parameters each** |
| Losses | adversarial + cycle consistency (λ = 10) + identity (λ = 5) |
| Image pool | 50 generated images |
| Resolution / batch | 128 × 128, batch 8 |
| Schedule | 12 epochs; learning rate 2e-4, constant for 6 epochs then linear decay to zero |
| Sampling | Domains balanced each epoch by subsampling the larger domain **at the patient level**, redrawn every epoch: 8,028 images per domain per epoch, 1,003 iterations |
| Training data | Training split only: Kermany 34,810 images / 2,961 patients; NEH 8,028 images / 295 patients |
| Checkpointing | Atomic save every 500 iterations and at each epoch end; automatic resume from `latest.pth`; refuses to resume from a corrupt checkpoint |
| Seed | 42 |

A smoke test (64 px, 2 blocks, 2 epochs) and a one-epoch timing run preceded the full run.

**Run:** 12,036 iterations in **2 h 54 m 41 s** on one T4. Inference weights `generators_ep012.pth` (62.7 MB) verified to load with keys `G_AB, G_BA, epoch, domains, args`. Config in `training_logs/cyclegan_run1_meta.json`; losses in `training_logs/cyclegan_run1_losses.csv` (12 observations per epoch).

### 8.3 Training dynamics

| Loss | Epoch 0 | Epoch 11 | Change |
|---|---:|---:|---|
| Cycle consistency | 2.753 | 0.869 | −68%; still declining in epochs 9–11 (0.909 → 0.814 → 0.869) |
| Identity | 1.364 | 0.384 | −72% |
| D_A | 0.382 | 0.041 | Dropped to 0.012 at epoch 10 |
| D_B | 0.499 | 0.116 | — |

The discriminators behaved asymmetrically: D_A approached zero, meaning it separated real from generated Kermany almost perfectly and G_BA's adversarial gradient largely vanished at those points. Training was **not demonstrated to be converged**; the result is an evaluated checkpoint.

Sample grids show clear progression: at iteration 500, colour fringing and checkerboard artefacts; by iteration 12,000, grayscale output with retinal layer structure and visible pathology (e.g. subretinal fluid, irregular RPE) retained.

### 8.4 Translation (`harmonization/translate.py`)

- Direction `--to-domain neh`: Kermany images translated through G_AB; NEH images only resized.
- **All splits translated.** The generator was fitted on training images only; applying frozen weights to validation and test images is a fixed preprocessing function, like a resize — not leakage.
- **Resized baseline written alongside** (`--write-resized-baseline`): every image resized to 128 px but not translated. Without it, any difference could be the extra resize rather than the translation.
- Output: 66,454 images plus manifest `manifests/pooled_translated_neh.csv`, where `cohort = translated` and the original cohort is kept in `source_cohort`. Verification confirmed patient, group and file disjointness inherited across splits, and 66,454 / 66,454 files on disk.
- Sanity check: mean absolute pixel difference translated vs resized-only = **72.9** for Kermany and **0.0** for NEH — the generator changed Kermany images and passed NEH through untouched, as designed.

### 8.5 Evaluation (`harmonization/evaluate_gan.py` plus matched controls)

| Measure | Result | Reading |
|---|---|---|
| **Cohort probe, matched 128 px preprocessing** | Resized-only 0.9998 / **0.9984** (train / test); translated 1.0000 / **0.9996** | Cohort separability unchanged at ceiling |
| **Feature distance, matched resolution** | NEH vs itself (floor) 29.22; resized Kermany vs NEH 258.08; translated Kermany vs NEH **323.52** | Distance to target **increased** by 25% |
| Feature distance, unmatched (auxiliary) | Real gap 191.67; translated 466.25 | Inflated by resolution mismatch; the matched row is the fair test |
| Held-out cycle L1 | Kermany train / test 0.0448 / 0.0443; NEH 0.0341 / 0.0365 | Generalises to unseen images |
| Memorisation | Translated→nearest real NEH, median 0.2084; real NEH→real NEH, median 0.0392; ratio 5.31 | No copying of training images |
| Intensity (mean / SD) | Kermany 0.1893 / 0.1984; NEH 0.1286 / 0.1956; translated 0.1113 / 0.1763 | Real change, overshooting the target |
| Translated-probe controls | Test 0.9996; shuffle 0.5276 ± 0.0047 (z = 99.6); 100 training images → 0.9919 | Signal genuine; recoverable from 100 images |

"Feature distance" is a Fréchet distance in the frozen ConvNeXt feature space used by the classifiers — **not** Inception FID and not comparable to published FID values.

The matched comparison used two dedicated tracks (`oct_resized_baseline`, `oct_translated`) sharing one manifest, split and feature extractor, differing only in whether images passed through the generator.

### 8.6 Interpretation

The generator learned a **stable, generalising, non-memorised** mapping that substantially changed image intensity — and did not harmonise the cohorts. Training losses and plausible images were not evidence of harmonisation.

The width probe (§7.3) explains why: much of the cohort signal is export resolution present inside a single cohort, which an appearance-level transformation cannot remove regardless of training length.

---

## 9. Phase 3 — OCT↔CFP cross-modal evaluation

### 9.1 Rationale

The original submission claimed a multimodal result from two OCT streams (§2.2). The rebuild asks the real cross-modal questions: does a disease representation learned on one modality work on the other, and how do the modalities compare within themselves?

The cohorts share no patients, so there is no paired OCT + CFP data to fuse; this is a **transfer** study, not multimodal fusion.

### 9.2 Shared label space

| Modality / dataset | Negative class (0) | AMD (1) | Dropped |
|---|---|---|---|
| OCT (default) | NORMAL | DRUSEN + CNV | DME |
| OCT (task-matched variant) | DME | DRUSEN + CNV | NORMAL |
| HYAMD | CONTROL | AMD | — |
| ODIR | NORMAL | AMD | DIABETIC, CATARACT |

DME has no fundus counterpart in these cohorts; cataract is a lens opacity that does not appear in a retinal B-scan.

### 9.3 Protocol (`scripts/crossmodal_cv.py`)

- Each modality split independently into **5 group-disjoint, class-stratified folds** (`StratifiedGroupKFold`, seed 42). The script verifies every group sits in exactly one fold before computing anything.
- In fold *k*, **one model per modality** is fitted on folds ≠ *k* and scored on **both** modalities' fold *k*:

```
OCT model  →  within_OCT (OCT fold k)   and  OCT_to_CFP (CFP fold k)
CFP model  →  within_CFP (CFP fold k)   and  CFP_to_OCT (OCT fold k)
```

- **Paired deltas on identical test folds:**
  `Δ OCT→CFP = within_CFP − OCT_to_CFP`  and  `Δ CFP→OCT = within_OCT − CFP_to_OCT`.
  Positive = what is lost by training on the wrong modality (for log loss, cross − within).
- **Metrics:** accuracy, balanced accuracy, macro-F1, AUC, sensitivity (AMD recall), specificity (NORMAL recall), and **log loss on training data and each test fold** — the training and evaluation losses.
- **Modality probe:** 5-fold group-disjoint probe predicting OCT vs CFP.
- **Options** (all off by default, so committed defaults stay reproducible — verified identical after each change): `--oct-map amd_dme`, `--oct-cohort kermany`, `--drop-mixed`, `--match-train images|groups` with `--match-repeats`.

Fold scores share training data, so the SD describes spread across folds, not a standard error; naive paired t-tests over folds would overstate significance.

The earlier single-split version (`scripts/crossmodal_probe.py`, bootstrap intervals) gave consistent results — for HYAMD, OCT→CFP AUC 0.479 and CFP→OCT 0.397 — confirming the 5-fold result is not a lucky split.

### 9.4 Results

**(a) OCT AMD-vs-NORMAL ↔ HYAMD AMD-vs-CONTROL** — `reports/crossmodal_cv_cfp_hyamd.json`
Transfer results valid; the within-OCT vs within-CFP comparison here is **not** task-matched (healthy vs diabetic negative class).

| Condition | Balanced acc. | Macro-F1 | AUC | Log loss |
|---|---:|---:|---:|---:|
| Within OCT | 0.944 ± 0.005 | 0.944 ± 0.005 | 0.984 ± 0.003 | 0.157 |
| Within CFP (HYAMD) | 0.643 ± 0.031 | 0.640 ± 0.030 | 0.710 ± 0.035 | 0.992 |
| OCT → CFP | 0.512 ± 0.042 | 0.392 ± 0.116 | 0.490 ± 0.067 | 4.804 |
| CFP → OCT | 0.487 ± 0.026 | 0.426 ± 0.041 | 0.448 ± 0.047 | 3.487 |

**(b) OCT AMD-vs-NORMAL ↔ ODIR AMD-vs-NORMAL** (task-matched, healthy negatives) — `reports/crossmodal_cv_cfp_odir.json`

| Condition | Balanced acc. | Macro-F1 | AUC | Log loss |
|---|---:|---:|---:|---:|
| Within OCT | 0.944 ± 0.005 | 0.944 ± 0.005 | 0.984 ± 0.003 | 0.157 |
| Within CFP (ODIR) | 0.943 ± 0.030 | 0.946 ± 0.025 | **0.990 ± 0.006** | 0.126 |
| OCT → CFP | 0.471 ± 0.020 | 0.349 ± 0.066 | 0.482 ± 0.083 | 5.557 |
| CFP → OCT | 0.501 ± 0.007 | 0.346 ± 0.027 | 0.572 ± 0.053 | 2.675 |

**(c) Diabetic-disease-matched: Kermany AMD-vs-DME ↔ HYAMD AMD-vs-CONTROL** — `reports/crossmodal_cv_cfp_hyamd_amd_dme_kermany_nomixed.json`
Both negative classes are diabetic retinal disease; both sources single-site. Kermany-only because DME is Kermany-only and the cohorts are separable at 0.996 — a pooled version would partly be a cohort task. 144 mixed AMD/DME groups dropped.

| Condition | Balanced acc. | Macro-F1 | AUC | Log loss |
|---|---:|---:|---:|---:|
| Within OCT (Kermany) | 0.944 ± 0.005 | 0.936 ± 0.003 | **0.983 ± 0.003** | 0.187 |
| Within CFP (HYAMD) | 0.643 ± 0.031 | 0.640 ± 0.030 | **0.710 ± 0.035** | 0.992 |
| OCT → CFP | 0.540 ± 0.029 | 0.459 ± 0.117 | 0.567 ± 0.027 | 4.611 |
| CFP → OCT | 0.489 ± 0.031 | 0.351 ± 0.107 | 0.498 ± 0.043 | 4.631 |

Class prevalence differs (OCT 74% AMD, HYAMD 34%), so AUC and balanced accuracy are the comparable metrics.

**(d) Sample-size control for (c)** — `..._matchimages.json`, `..._matchgroups.json`
OCT's training set subsampled per fold to HYAMD's training size (whole groups, class ratio kept, 5 draws per fold averaged); OCT test folds untouched.

| OCT training data per fold | Within-OCT AUC | Within-OCT bal. acc. | HYAMD AUC | Gap |
|---|---:|---:|---:|---:|
| Full (~21,300 images / ~1,414 groups) | 0.983 ± 0.003 | 0.944 | 0.710 | 0.273 |
| Matched by patients (~4,000 images / 261 groups) | 0.973 ± 0.002 | 0.922 | 0.710 | 0.263 |
| Matched by images (~1,280 images / ~84 groups) | 0.956 ± 0.004 | 0.885 | 0.710 | 0.246 |
| HYAMD training data | 1,248 images / 260 groups | | | |

Images-matching is the harsher test: OCT receives the same number of scans as HYAMD from about a third of the patients, and still reaches 0.956. Reducing data explains only ~0.03 of the 0.27 gap. At matched size both modalities fit their training sets fully (OCT 1.000, HYAMD 0.9996), yet generalise very differently.

**(e) Mixed-label sensitivity** — `reports/crossmodal_cv_cfp_hyamd_nomixed.json`
Excluding the 404 mixed NORMAL/AMD groups (58,224 → 49,957 images): within-OCT AUC 0.9836 → 0.9870; OCT→CFP 0.4902 → 0.4356; CFP→OCT 0.4479 → 0.4488. No material effect.

**(f) Modality probe.** 1.000 in every configuration (baselines 0.945–0.987). In frozen ImageNet features, OCT and CFP occupy completely separate regions.

### 9.5 Reading the transfer failures

- Every cross-modal direction was worse than within-modal in **5/5 folds** on AUC, macro-F1 and log loss, across both CFP datasets and both directions.
- **Log loss rises from ~0.13–0.99 within modality to ~2.7–6.3 across modality**: the models are confidently wrong, not merely uncertain.
- Some cross-modal sensitivities look high only because specificity collapses (OCT→HYAMD sensitivity 0.80 with specificity 0.23; ODIR→OCT specificity 0.02). These are biases toward predicting AMD, not gains, and are never reported as improvements.
- CFP→OCT sensitivity and specificity swing widely between folds (e.g. 0.28 ± 0.26), the signature of chance-level decisions.

### 9.6 What the within-modality comparison shows

The result is task-dependent:

- **AMD vs healthy:** CFP (ODIR) 0.990 ≈ OCT 0.984. Fundus performs as well as OCT.
- **AMD vs diabetic retinal disease:** OCT 0.983 ≫ CFP (HYAMD) 0.710, and the gap survives equal training data.

This is consistent with the imaging physics: AMD's defining pathology — drusen height, sub-RPE and subretinal fluid, RPE disruption — lies in depth, which OCT resolves and an en-face photograph collapses. Distinguishing AMD from other disease that also alters the retinal surface is where depth information matters most. The clinical reasoning comes from the literature; the experiment measures only linear separability in frozen features.

---

## 10. Claims supported by the evidence

| Claim | Status |
|---|---|
| Leakage-controlled pooled OCT 4-class baseline ≈ 0.840 macro-F1 | Supported |
| Preprocessing materially affects performance | Supported |
| Kermany and NEH are near-perfectly separable, partly by export properties | Supported ("cohort information", not "scanner bias") |
| CycleGAN translation harmonised the OCT cohorts | **Not supported** |
| The generator was well-behaved despite not harmonising | Supported |
| Disease representations do not transfer between OCT and CFP | Supported (frozen features) |
| Cross-modal models are confidently wrong | Supported |
| For AMD vs healthy, CFP matches OCT | Supported in ODIR |
| For AMD vs diabetic retinal disease, OCT far outperforms CFP | Supported for this comparison |
| That gap is not a sample-size artifact | Supported |
| Mixed-label groups do not drive the results | Supported |

**Not claimed:** that OCT is generally superior to CFP for AMD; any clinical superiority; successful harmonisation.

---

## 11. How the rebuild answers the original weaknesses

| Weakness in the original submission | Rebuild |
|---|---|
| "Multimodal" model used two OCT streams paired by array index | Genuine OCT↔CFP transfer study, 5-fold, both directions, stated explicitly as transfer rather than fusion |
| Incompatible label structures across datasets | Explicit shared label space with documented mappings and dropped classes; task-matched variants (AMD-vs-healthy with ODIR, AMD-vs-diabetic with Kermany DME) |
| No patient-level split control | Patient/group-disjoint splits and folds, verified in code on every run |
| Headline accuracy without controls | Seed variance, label-shuffle controls, learning curves, matched-preprocessing and sample-size controls |
| Synthetic images assumed useful | Four-part GAN evaluation plus cohort probe under matched preprocessing; negative result reported |
| Data-source confounding unmeasured | Cohort probe, within-cohort width probe, AMDNet23 source probe |
| Limited reproducibility | Committed manifests, configs, reports, loss logs, checkpoint metadata; evidence table with provenance |

---

## 12. Limitations

1. **Frozen ImageNet features.** Results describe linear separability in a fixed ConvNeXt-Tiny representation, not fine-tuned or modality-specific models, nor clinicians.
2. **One dataset per modality in each matched comparison.** HYAMD and ODIR lead to different conclusions; neither generalises to all CFP.
3. **DME is not diabetic retinopathy.** The diabetic-matched comparison is task-aligned, not label-identical.
4. **HYAMD's reference standard is OCT-informed.** Some AMD labels may rest on findings visible only on OCT — supporting the clinical argument, but grading the CFP classifier against an OCT-informed standard.
5. **ODIR.** Trained to 1.000 on 745 images; its AMD cases may be predominantly advanced.
6. **Cohort confounding.** Kermany and NEH differ on several axes at once.
7. **CycleGAN not converged; one checkpoint evaluated.**
8. **Single-seed ablations**, read against a 0.004 noise floor.
9. **No clinical claim.** Findings characterise performance on these datasets and tasks only.

---

## 13. Engineering and reproducibility notes

- **Portable roots.** Manifests hold relative paths; roots are supplied per environment via `--data-root COHORT=PATH` or environment variables.
- **Kaggle dataset path.** The Kermany mirror nests images in a directory named `OCT2017 ` with a trailing space; a symlink (`/kaggle/working/kermany_root`) removes the fragility.
- **Kaggle persistence.** `/kaggle/working` does not survive a session, and Quick Save captures the notebook, not working-directory files. Outputs were lost twice; the fix was to export results immediately (small JSON reports and the 62.7 MB generator file) and commit them, and to upload the checkpoint as a private Kaggle dataset (`cyclegan-ep012`) for later sessions.
- **Duplicate-process incident.** A queued shell cell launched a second training process writing to the same output directory. Both were killed, the directory wiped, and training restarted once, so the checkpoint's provenance is clean.
- **Misfiled outputs.** Runs named `odir` were produced with the OCT track active and wrote OCT results into `outputs/cfp_odir/`. An audit of every `summary.json` against its manifest caught this; the runs are exact duplicates of `normalize_768` and `seed1–4` and are excluded.
- **Verification habit.** Every modification to `crossmodal_cv.py` was followed by re-running the default configuration and confirming its summary was identical to the committed report.

### Key commands

```
# cross-modal, 5-fold
python scripts\crossmodal_cv.py                                   # HYAMD
python scripts\crossmodal_cv.py --cfp-track cfp_odir              # ODIR
python scripts\crossmodal_cv.py --oct-map amd_dme --oct-cohort kermany --drop-mixed
python scripts\crossmodal_cv.py --oct-map amd_dme --oct-cohort kermany --drop-mixed --match-train images
python scripts\crossmodal_cv.py --drop-mixed                      # mixed-label sensitivity

# cohort probe with controls
python scripts\probe_controls.py --track oct

# GAN (Kaggle)
python harmonization/train_cyclegan.py --epochs 12 --decay-start 6 --image-size 128 --batch-size 8 --blocks 6 --workers 2 --out outputs/cyclegan_run1 --data-root kermany=... --data-root neh=...
python harmonization/translate.py --checkpoint .../generators_ep012.pth --to-domain neh --out-images /kaggle/working/translated_neh --out-manifest manifests/pooled_translated_neh.csv --write-resized-baseline /kaggle/working/baseline_resized --data-root kermany=... --data-root neh=...
python harmonization/evaluate_gan.py --checkpoint ... --translated-manifest manifests/pooled_translated_neh.csv --translated-root /kaggle/working/translated_neh ...
```

---

## 14. Commit history (rebuild phases)

| Commit | Content |
|---|---|
| `5ced7cb` | CycleGAN run 1: loss log and training metadata |
| `d122e16` | Harmonisation evaluation reports and translated manifest |
| `d822efc` | Portable `oct_translated` and `oct_resized_baseline` tracks |
| `018b15a` | 5-fold cross-modal script; HYAMD and ODIR reports |
| `cb661cf` | Task-matched AMD-vs-DME variant (Kermany-only, mixed groups dropped) |
| `b7dcae2` | Size-matched OCT training control |
| `5b1003c` | Mixed-label sensitivity run |
| `40715e9` | Master evidence table |

---

## 15. Open items

1. **Supervisor review** of the ODIR finding, which narrows the expected "OCT outperforms CFP" claim to the AMD-vs-diabetic-disease comparison.
2. **Target venue**, which decides whether the paper leads with the modality comparison, the cross-modal methodology, or the negative harmonisation result.
3. **Reviewer-comment mapping:** place each original review comment beside the committed artifact that answers it.
4. **Dataset-audit figures** in §5.5, to be verified against `reports/integrity_*.txt` if cited.
5. **DRUSEN errors:** whether they stem from resolution loss in preprocessing or label ambiguity.
6. **Optional:** a longer CycleGAN run, only if a reviewer insists on the convergence question — the width-probe argument (§8.6) addresses it without further training.
