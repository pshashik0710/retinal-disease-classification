# Master Evidence Table — Retinal OCT/CFP Rebuild

**Branch:** `rebuild`
**Scope:** the three completed phases — rebuild baseline, OCT harmonisation, cross-modal evaluation.

**Provenance rule.** Every number below is taken from a committed file in `reports/`, `outputs/*/metrics/summary.json`, `training_logs/`, or a committed manifest, and the source is named. Anything not re-verified against a committed file is listed separately in §0 and is not used as evidence.

**Representation.** Unless stated otherwise, every classifier is a logistic-regression probe on frozen, ImageNet-pretrained ConvNeXt-Tiny features (768-d, cached). All splits and folds are patient/group-disjoint.

---

## 0. Excluded or not re-verified

| Item | Reason |
|---|---|
| `outputs/cfp_odir/odir*` | Misfiled: manifest is `pooled_split.csv` (OCT), not ODIR. Results are exact duplicates of `normalize_768` and `seed1–4`. Excluded; counted once via the originals. |
| `outputs/oct/recheck` | Exact duplicate of `normalize_768` (acc 0.8619, macro-F1 0.8390). Counted once. |
| `outputs/baseline_cached_probe` (acc 0.8494, macro-F1 0.8241) | Initial run, superseded by the final configuration (§1.2). Historical only. |
| Kermany 76,677 images pre-cap; 466 cross-class-conflict images from 56 patients; NEH 550 eyes | From the dataset audit; not re-verified in this pass. Check against `reports/integrity_*.txt` before citing. |
| Patient-matched OCT at 325 patients (`oct_match_hyamd`) | Track exists in config; no committed output. Not reported. |

---

## 1. Rebuild baseline

### 1.1 Data (verified from committed manifests)

| Set | Images | Patients / groups | Notes |
|---|---:|---:|---|
| Pooled OCT (`pooled_split.csv`) | 66,454 | 5,093 patients | Kermany 54,742 / 4,656 patients; NEH 11,712 / 437 patients. Split train / val / test = 42,838 / 10,244 / 13,372. Max 50 images per patient. |
| OCT AMD-vs-NORMAL subset | 58,224 | 4,755 patients / 4,753 groups | NORMAL 29,794; AMD (DRUSEN+CNV) 28,430; DME excluded. 404 groups carry both labels (all Kermany). |
| OCT AMD-vs-DME subset, Kermany only | 26,649 | 1,767 groups | DME 7,035; AMD 19,614; 144 mixed AMD/DME groups dropped. DME is Kermany-only, hence the cohort restriction. |
| HYAMD CFP (`hyamd_binary.csv`) | 1,560 | 325 patients | AMD 532 / 135 patients; CONTROL 1,028 / 190 patients; no mixed patients. CONTROL = diabetic retinopathy without AMD. |
| ODIR CFP (source within `amdnet23_clean.csv`) | 1,303 | 879 patients | Patient-disjoint splits (558 / 150 / 171); 0 patient overlap with other AMDNet23 sources. AMD-vs-NORMAL subset: 745 images, 512 groups. |

### 1.2 Pooled OCT 4-class baseline — final configuration

`normalize_768` resize, five seeds. Sources: `outputs/normalize_768`, `outputs/seed1`–`seed4`.

| Seed | Accuracy | Macro-F1 |
|---|---:|---:|
| 42 | 0.8619 | 0.8390 |
| 1 | 0.8578 | 0.8365 |
| 2 | 0.8716 | 0.8467 |
| 3 | 0.8621 | 0.8403 |
| 4 | 0.8573 | 0.8368 |
| **Mean ± SD** | **0.862 ± 0.006** | **0.840 ± 0.004** |

The seed SD of **0.004 macro-F1** is the noise floor for every single-seed ablation below.

### 1.3 Ablations (single seed 42)

| Factor | Run | Accuracy | Macro-F1 | Δ macro-F1 | Reading |
|---|---|---:|---:|---:|---|
| Resize | `normalize_768` (final) | 0.8619 | 0.8390 | — | Reference |
| Resize | `baseline_v2` (resize_crop) | 0.8579 | 0.8347 | −0.004 | Within seed noise; matches the 0.840-vs-0.835 comparison recorded in `config.py` |
| Resize | `squash` | 0.8440 | 0.8157 | −0.023 | Clearly worse |
| Resize | `pad` | 0.8315 | 0.8102 | −0.029 | Worst |
| Dropout | `do0` | 0.8579 | 0.8347 | 0.000 vs `baseline_v2` | Identical to `baseline_v2` |
| Class weights | `no_weights` | 0.8572 | 0.8199 | −0.015 vs `baseline_v2` | Weighting adds macro-F1 at unchanged accuracy (0.8579 vs 0.8572) |
| Learning rate | `lr3e-3` | 0.8480 | 0.8223 | −0.012 vs `baseline_v2` | Worse |

### 1.4 Cohort probe (Kermany vs NEH) — `reports/probe_controls.json`

| Measure | Value |
|---|---|
| Test / train accuracy | 0.9964 / 0.9997 (majority baseline 0.8158) |
| Label-shuffle control | 0.5201 ± 0.0034, n = 10; real probe 140.3 SD above it |
| Learning curve | 100 → 0.967; 500 → 0.975; 2,000 → 0.985; 10,000 → 0.994; 42,838 → 0.9964 |
| Within-Kermany export-width probe | 0.8982 vs baseline 0.6589 (widths 512 / 768 / 1024 / 1536) |

A large share of the cohort signal is present inside a single cohort as export resolution — a preprocessing property, not pathology.

### 1.5 Patient-matched OCT, binary NORMAL-vs-AMD (five seeds)

A **binary** task: not comparable to the 4-class baseline in §1.2.

| Matched to | Patients | Macro-F1 (5 seeds) | Mean ± SD |
|---|---:|---|---:|
| ODIR | 879 | 0.9354, 0.9380, 0.9323, 0.9355, 0.9276 | 0.934 ± 0.004 |
| AMDNet23 | 1,420 | 0.9257, 0.9254, 0.9286, 0.9277, 0.9213 | 0.926 ± 0.003 |

Not monotonic in patient count, so it should not be described as a learning curve.

### 1.6 CFP baselines (single split, five seeds)

| Dataset / task | Macro-F1 per seed | Mean ± SD | Caveat |
|---|---|---:|---|
| AMDNet23, 4-class | 0.8387, 0.8467, 0.8480, 0.8580, 0.8863 | 0.856 ± 0.019 | Source confounded with class: source probe 0.9176 vs baseline 0.6841, shuffle 0.448 ± 0.029 (`reports/probe_controls_amdnet23.json`). ARIA is 100% AMD; "other" is 99.5% cataract. |
| HYAMD, AMD vs CONTROL | 0.5505, 0.6111, 0.6400, 0.6616, 0.5708 | 0.607 ± 0.046 | CONTROL = diabetic retinopathy without AMD. Consistent with the 5-fold estimate in §3.1 (0.640 ± 0.030). |
| HYAMD, 3-class staging | 0.4263 (one run) | — | AMD_EARLY has 20 patients. Secondary only. |

---

## 2. OCT harmonisation — CycleGAN Kermany → NEH

**Setup:** 128×128, 6 residual blocks, batch 8, 12 epochs (LR decay from 6), λ_cycle 10, λ_identity 5, pool 50, seed 42, patient-level balanced sampling. 12,036 iterations in 2:54:41 on a Tesla T4. Checkpoint `generators_ep012.pth`; config in `training_logs/cyclegan_run1_meta.json`.

### 2.1 Training dynamics — `training_logs/cyclegan_run1_losses.csv`

| Loss | Epoch 0 | Epoch 11 | Note |
|---|---:|---:|---|
| Cycle | 2.753 | 0.869 | −68%; still declining (0.909 → 0.814 → 0.869 over epochs 9–11) |
| Identity | 1.364 | 0.384 | −72% |
| D_A | 0.382 | 0.041 | Fell to 0.012 at epoch 10; asymmetric with D_B (0.116 at epoch 11) |

Not demonstrated to be converged. Report as an evaluated checkpoint.

### 2.2 Harmonisation outcome — `reports/gan_evaluation.json`, `reports/harmonisation_summary.json`, `reports/probe_translated_controls.json`

| Measure | Result | Reading |
|---|---|---|
| Cohort probe, matched 128px preprocessing | Resized-only 0.9998 / **0.9984** (train / test); translated 1.0000 / **0.9996** | Cohort separability unchanged at ceiling |
| ConvNeXt-feature distance, matched resolution | NEH self-floor 29.22; resized Kermany↔NEH 258.08; translated Kermany↔NEH **323.52** | Distance increased (+25%) |
| Same, unmatched resolution (auxiliary) | Real gap 191.67; translated 466.25 | Confounded by resolution; §2.2 row above is the fair comparison |
| Held-out cycle L1 | Kermany train / test 0.0448 / 0.0443; NEH 0.0341 / 0.0365 | Generalises; no train/test gap |
| Memorisation | Translated→real median 0.2084 vs real→real 0.0392 (ratio 5.31; min 0.0627) | No copying signature |
| Intensity (mean / SD) | Kermany 0.1893 / 0.1984; NEH 0.1286 / 0.1956; translated 0.1113 / 0.1763 | Real change, overshooting the target mean |
| Translated probe controls | Test 0.9996; shuffle 0.5276 ± 0.0047 (z 99.6); 100 samples → 0.9919 | Signal genuine and recoverable from 100 images |

Distance values are Fréchet distances in the frozen ConvNeXt feature space — **not** Inception FID, and not comparable to published FID values.

**Limitation.** Kermany and NEH differ at once in device, export resolution, file format, site, population and labelling practice. This supports a claim about **cohort information**, not scanner bias specifically.

---

## 3. Cross-modal evaluation — 5-fold, group-disjoint

Script: `scripts/crossmodal_cv.py`. In each fold one model per modality is fitted and scored on both modalities' held-out fold, so every delta is paired on an identical test fold. Values are mean ± SD over 5 folds.

### 3.1 OCT AMD-vs-NORMAL ↔ HYAMD AMD-vs-CONTROL — `reports/crossmodal_cv_cfp_hyamd.json`

**Task mismatch:** OCT's negative class is healthy eyes; HYAMD's is diabetic retinopathy without AMD. The **transfer** results are valid; the **within-OCT vs within-CFP** comparison in this table is not task-matched (see §3.3).

| Condition | Accuracy | Balanced acc. | Macro-F1 | AUC | Log loss |
|---|---:|---:|---:|---:|---:|
| Within OCT | 0.944 ± 0.005 | 0.944 ± 0.005 | 0.944 ± 0.005 | 0.984 ± 0.003 | 0.157 ± 0.017 |
| Within CFP (HYAMD) | 0.673 ± 0.030 | 0.643 ± 0.031 | 0.640 ± 0.030 | 0.710 ± 0.035 | 0.992 ± 0.083 |
| OCT → CFP | 0.422 ± 0.092 | 0.512 ± 0.042 | 0.392 ± 0.116 | 0.490 ± 0.067 | 4.804 ± 2.382 |
| CFP → OCT | 0.492 ± 0.031 | 0.487 ± 0.026 | 0.426 ± 0.041 | 0.448 ± 0.047 | 3.487 ± 0.692 |

Both transfer directions worse than within-modal in 5/5 folds on AUC, macro-F1 and log loss. OCT→CFP sensitivity (0.80) exceeds within-CFP only because specificity collapses to 0.23 — a bias toward predicting AMD, not a gain. Modality probe 1.000 vs baseline 0.974.

### 3.2 OCT AMD-vs-NORMAL ↔ ODIR AMD-vs-NORMAL — `reports/crossmodal_cv_cfp_odir.json`

Task-matched: both negative classes are healthy eyes.

| Condition | Accuracy | Balanced acc. | Macro-F1 | AUC | Log loss |
|---|---:|---:|---:|---:|---:|
| Within OCT | 0.944 ± 0.005 | 0.944 ± 0.005 | 0.944 ± 0.005 | 0.984 ± 0.003 | 0.157 ± 0.017 |
| Within CFP (ODIR) | 0.953 ± 0.021 | 0.943 ± 0.030 | 0.946 ± 0.025 | 0.990 ± 0.006 | 0.126 ± 0.050 |
| OCT → CFP | 0.373 ± 0.044 | 0.471 ± 0.020 | 0.349 ± 0.066 | 0.482 ± 0.083 | 5.557 ± 1.721 |
| CFP → OCT | 0.490 ± 0.008 | 0.501 ± 0.007 | 0.346 ± 0.027 | 0.572 ± 0.053 | 2.675 ± 0.902 |

On a matched AMD-vs-healthy task, CFP performs as well as OCT. CFP→OCT's 0.572 AUC is not usable transfer: specificity 0.02, balanced accuracy 0.501. ODIR train accuracy is 1.000 on 745 images; ODIR AMD may be predominantly advanced.

### 3.3 Diabetic-disease-matched: Kermany AMD-vs-DME ↔ HYAMD AMD-vs-CONTROL — `reports/crossmodal_cv_cfp_hyamd_amd_dme_kermany_nomixed.json`

Both negative classes are diabetic retinal disease; both sources are single-site. **Not label-identical:** DME is a complication of diabetic retinopathy, not the same diagnosis.

| Condition | Accuracy | Balanced acc. | Macro-F1 | AUC | Log loss |
|---|---:|---:|---:|---:|---:|
| Within OCT (Kermany) | 0.950 ± 0.002 | 0.944 ± 0.005 | 0.936 ± 0.003 | 0.983 ± 0.003 | 0.187 ± 0.015 |
| Within CFP (HYAMD) | 0.673 ± 0.030 | 0.643 ± 0.031 | 0.640 ± 0.030 | 0.710 ± 0.035 | 0.992 ± 0.083 |
| OCT → CFP | 0.480 ± 0.087 | 0.540 ± 0.029 | 0.459 ± 0.117 | 0.567 ± 0.027 | 4.611 ± 3.065 |
| CFP → OCT | 0.378 ± 0.135 | 0.489 ± 0.031 | 0.351 ± 0.107 | 0.498 ± 0.043 | 4.631 ± 1.697 |

Class prevalence differs (OCT 74% AMD, HYAMD 34%), so compare **AUC and balanced accuracy**, not accuracy.

### 3.4 Sample-size control for §3.3 — does the within-modality gap survive equal training data?

OCT's **training** set subsampled per fold to HYAMD's training size; whole groups drawn, class ratio kept, 5 random draws per fold averaged. OCT test folds untouched. Sources: `..._matchimages.json`, `..._matchgroups.json`.

| OCT training data | OCT train size per fold | Within-OCT AUC | Within-OCT bal. acc. | Within-CFP (HYAMD) AUC | Gap (AUC) |
|---|---|---:|---:|---:|---:|
| Full | ~21,300 img / ~1,414 groups | 0.983 ± 0.003 | 0.944 | 0.710 ± 0.035 | 0.273 |
| Matched by patients | ~4,000 img / 261 groups | 0.973 ± 0.002 | 0.922 | 0.710 ± 0.035 | 0.263 |
| Matched by images | ~1,280 img / ~84 groups | 0.956 ± 0.004 | 0.885 | 0.710 ± 0.035 | 0.246 |
| HYAMD reference | 1,248 img / 260 groups | — | — | — | — |

The **within-modality OCT advantage over HYAMD survives equal training data.** Images-matching is the harsher test: OCT gets the same number of scans but about a third of the patients (84 vs 260) and still reaches 0.956. Reducing OCT's data explains ~0.03 of the 0.27 AUC gap. At matched size both modalities fit their training sets fully (OCT 1.000, HYAMD 0.9996), yet generalise very differently.

Cross-modal transfer stays near chance under matching (OCT→CFP 0.523 and 0.513).

### 3.5 Mixed-label sensitivity — `reports/crossmodal_cv_cfp_hyamd_nomixed.json`

404 OCT groups carry both NORMAL and AMD images (all Kermany; consistent with a diseased eye and a healthy fellow eye under patient-level IDs). Excluding them: 58,224 → 49,957 images, 4,753 → 4,349 groups.

| Measure | Mixed kept | Mixed excluded |
|---|---:|---:|
| Within-OCT AUC | 0.9836 | 0.9870 |
| OCT → CFP AUC | 0.4902 | 0.4356 |
| CFP → OCT AUC | 0.4479 | 0.4488 |

No material effect. The committed default runs stand. The OCT side is identical in the HYAMD and ODIR runs, so this covers both.

---

## 4. Claim → evidence map

| Claim | Evidence | Status | Qualification |
|---|---|---|---|
| Rebuilt pooled OCT 4-class baseline is ~0.84 macro-F1 | 0.840 ± 0.004 over 5 seeds (§1.2) | Supported | Frozen-feature linear probe |
| Preprocessing matters; normalize_768 ≈ resize_crop > squash > pad | §1.3 | Supported | Single-seed ablations, read against SD 0.004 |
| Kermany vs NEH are near-perfectly separable, partly by export properties | Probe 0.9964, shuffle 0.520, width probe 0.898 (§1.4) | Supported | "Cohort information", not "scanner bias" |
| CycleGAN translation harmonised the cohorts | Probe 0.9984 → 0.9996; distance 258 → 324 (§2.2) | **Not supported** | 12-epoch checkpoint, not converged |
| The generator is well-behaved despite not harmonising | Cycle L1 generalises; memorisation ratio 5.31 (§2.2) | Supported | Convergence ≠ valid harmonisation |
| Disease representations do not transfer across OCT and CFP | Cross-modal AUC 0.44–0.57, worse in 5/5 folds, both CFP sets, both directions (§3) | Supported | Frozen ImageNet features |
| Cross-modal models are confidently wrong | Within log loss 0.13–0.99; cross-modal 2.7–6.3 (§3) | Supported | — |
| OCT and CFP are trivially separable in these features | Modality probe 1.000 (§3) | Supported | Cohort differences contribute |
| For AMD vs healthy, CFP matches OCT | ODIR 0.990 vs OCT 0.984 (§3.2) | Supported, in ODIR | ODIR may be an easy / advanced-AMD cohort |
| For AMD vs diabetic retinal disease, OCT far outperforms CFP | Kermany 0.983 vs HYAMD 0.710 (§3.3) | Supported, for this comparison | DME ≠ DR; HYAMD reference standard OCT-informed |
| That gap is not a sample-size artifact | 0.956 at equal images, 0.973 at equal patients (§3.4) | Supported | One dataset per modality |
| Mixed-label OCT groups do not drive results | 0.9836 → 0.9870 (§3.5) | Supported | — |

---

## 5. Standing limitations

1. **Frozen ImageNet features.** All results measure linear separability in a fixed ConvNeXt-Tiny representation, not what a fine-tuned or modality-specific model — or a clinician — could achieve.
2. **One dataset per modality in each matched comparison.** HYAMD and ODIR support different conclusions; neither generalises to all CFP.
3. **DME ≠ diabetic retinopathy.** §3.3 is diabetic-disease-matched, not label-identical.
4. **HYAMD's reference standard was OCT-informed.** Some AMD diagnoses may rest on findings visible only on OCT. This supports the clinical argument but also means the CFP classifier is graded against an OCT-informed reference.
5. **ODIR caveat.** Train accuracy 1.000 on 745 images; its AMD cases may be predominantly advanced.
6. **Cohort confounding.** Kermany and NEH differ in several dimensions at once.
7. **CycleGAN not converged.** Report the harmonisation result as an evaluated checkpoint, not an exhaustive optimisation.
8. **Single-seed ablations.** §1.3 differences smaller than ~0.004 macro-F1 are within seed noise.
9. **No clinical superiority claim.** These experiments characterise performance and transfer on these datasets and tasks only.

---

## 6. Paper-level synthesis

Under patient-disjoint evaluation, the rebuilt pooled OCT classifier reaches 0.840 ± 0.004 macro-F1, while a linear probe separates the two OCT cohorts almost perfectly — partly on export properties present within a single cohort.

A CycleGAN trained to map Kermany toward NEH produced stable, generalising, non-memorised translations, yet left cohort separability at ceiling and moved the translated images further from the target cohort. Well-behaved training and plausible images did not amount to harmonisation.

Disease representations did not transfer between OCT and colour fundus photography in either direction, with near-chance AUC and sharply increased log loss. Within each modality the picture depends on the task: fundus matched OCT when AMD had only to be told apart from healthy eyes, but fell well behind OCT when AMD had to be separated from diabetic retinal disease — a gap that persisted when OCT was given no more training data than fundus.