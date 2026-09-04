"""
config.py -- single source of truth for the rebuilt OCT classification
pipeline.

Scope of the current experiment:

    pooled NEH + Kermany OCT  ->  4-class B-scan classification
    (NORMAL / DRUSEN / CNV / DME) on a patient-disjoint split,
    ConvNeXt-Tiny, CPU-only hardware.

Two things in here are load-bearing and must not be edited casually:

  CLASSES
      The order defines the integer labels. It MUST match the order used
      when the manifests were written (pool_manifests.py, kermany_clean.py),
      because manifests/pooled_split.csv already carries a 'y' column
      built from it. Reordering this list silently mislabels every
      confusion matrix and per-class metric -- the numbers stay plausible
      and the class names become wrong.

  POOLED_MANIFEST
      The split is precomputed, verified patient/group/eye/file-disjoint,
      and committed. Training code reads the 'split' column; it must never
      re-split the data.
"""

from __future__ import annotations

import os


class Config:

    # =====================================================================
    # PATHS
    # =====================================================================

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    MANIFEST_DIR = os.path.join(BASE_DIR, "manifests")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

    # =====================================================================
    # TRACK
    # =====================================================================
    #
    # One config, several studies. A track fixes the manifest, the data
    # roots, the class list and the preprocessing that suits that
    # modality. Everything downstream -- dataset.py, cache_features.py,
    # train.py -- reads Config and needs no changes.
    #
    # Set TRACK here, or override at the command line if a script exposes
    # it. Feature caches and experiment outputs are namespaced by track,
    # so switching cannot silently reuse the wrong vectors.
    #
    #   oct           pooled NEH + Kermany, 4-class B-scan  (the main study)
    #   cfp_hyamd     HYAMD fundus, CONTROL vs AMD          (clean cohort)
    #   cfp_hyamd3    HYAMD fundus, 3-class AMD staging     (thin early class)
    #   cfp_amdnet23  AMDNet23 fundus, 4-class              (multi-source)
    #
    TRACK = "oct"

    TRACKS = {
        "oct": {
            "manifest": "pooled_split.csv",
            "classes": ["NORMAL", "DRUSEN", "CNV", "DME"],
            "roots": {
                "neh": r"D:\datasets\neh\NEH_UT_2021RetinalOCTDataset",
                "kermany": r"D:\datasets\kermany2018\OCT2017",
            },
            # Six native aspect ratios (0.774-3.097). normalize_768 puts
            # both cohorts through a common intermediate resolution so the
            # centre crop costs them equally; measured equivalent to
            # resize_crop (0.840 vs 0.835 macro-F1, inside 1.04 sd of seed
            # noise) and clearly better than squash or pad.
            "resize": "normalize_768",
            "note": "pooled OCT B-scans; DME comes only from Kermany",
        },
        "cfp_hyamd": {
            "manifest": "hyamd_binary.csv",
            "classes": ["CONTROL", "AMD"],
            "roots": {"hyamd": r"D:\datasets\hyamd"},
            # Two export widths (2576 and 1960, both 1934 high) in the same
            # proportion across all classes, so no class-correlated
            # geometry. resize_crop is enough.
            "resize": "resize_crop",
            "note": ("HYAMD: single site, single Topcon camera, labels from "
                     "full clinical evaluation with OCT. Controls are "
                     "diabetic-retinopathy patients WITHOUT AMD, so "
                     "'CONTROL' is not 'healthy'."),
        },
        "cfp_hyamd3": {
            "manifest": "hyamd_staging.csv",
            "classes": ["CONTROL", "AMD_EARLY", "AMD_LATE"],
            "roots": {"hyamd": r"D:\datasets\hyamd"},
            "resize": "resize_crop",
            "note": ("AMD_EARLY has 20 patients (12/4/4 across splits). "
                     "Secondary analysis only -- do not select a model on "
                     "its metrics."),
        },
        "cfp_amdnet23": {
            "manifest": "amdnet23_clean.csv",
            "classes": ["NORMAL", "AMD", "DIABETIC", "CATARACT"],
            "roots": {"amdnet23": r"D:\datasets\amdnet23"},
            # 751 distinct sizes from six source datasets; no common
            # intermediate resolution is meaningful.
            "resize": "resize_crop",
            "note": ("Compiled from six public datasets, and source is "
                     "confounded with class: ARIA is 100% AMD, 'other' is "
                     "99.5% cataract. Run the source probe and report per-"
                     "source metrics, or restrict to ODIR."),
        },
    }

    # ---- resolved from TRACK ------------------------------------------
    _T = TRACKS[TRACK]

    POOLED_MANIFEST = os.path.join(MANIFEST_DIR, _T["manifest"])
    DATA_ROOTS = _T["roots"]
    TRACK_NOTE = _T["note"]

    # Namespaced so an OCT cache can never be read as a CFP one.
    FEATURE_CACHE_DIR = os.path.join(BASE_DIR, "features", TRACK)

    # =====================================================================
    # CLASSES  -- order is load-bearing, see module docstring
    # =====================================================================

    CLASSES = _T["classes"]
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
    NUM_CLASSES = len(CLASSES)

    # Clinical reading, for the paper:
    #   NORMAL  no AMD
    #   DRUSEN  early / intermediate AMD (sub-RPE deposits)
    #   CNV     wet / neovascular AMD
    #   DME     diabetic macular edema -- NOT AMD, and present only in the
    #           Kermany cohort. See COHORT_CONFOUND_NOTE below.

    COHORT_CONFOUND_NOTE = (
        "DME images come exclusively from the Kermany cohort. A model can "
        "score well on DME by recognising the cohort (scanner, resolution, "
        "preprocessing) rather than the pathology. Run the cohort probe and "
        "report per-class metrics split by cohort."
    )

    # =====================================================================
    # REPRODUCIBILITY
    # =====================================================================

    SEED = 42

    # torch.use_deterministic_algorithms(True) raises for
    # adaptive_avg_pool2d_backward on CUDA (ConvNeXt's head pools that way).
    # There is a deterministic CPU path, so it is safe here, but warn_only
    # keeps the code portable to GPU.
    DETERMINISTIC = True
    DETERMINISTIC_WARN_ONLY = True

    # These must be exported in the shell BEFORE python starts; setting them
    # from inside the process has no effect:
    #     set PYTHONHASHSEED=42
    #     set CUBLAS_WORKSPACE_CONFIG=:4096:8

    # =====================================================================
    # HARDWARE  (12 logical cores, 16 GB RAM, no CUDA)
    # =====================================================================

    import torch as _torch
    DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
    del _torch

    TORCH_NUM_THREADS = 10      # leave headroom for the dataloader workers
    NUM_WORKERS = 2
    PIN_MEMORY = False          # meaningless without CUDA
    PERSISTENT_WORKERS = True   # only honoured when NUM_WORKERS > 0
    PREFETCH_FACTOR = 2
    USE_AMP = False             # torch.amp autocast here is CUDA-only

    # =====================================================================
    # IMAGE / TRANSFORMS
    # =====================================================================

    IMAGE_SIZE = 224            # ConvNeXt pretrain resolution
    IMAGE_CHANNELS = 3

    # The pooled data has six native aspect ratios (0.774 .. 3.097) because
    # Kermany B-scans come at 512/768/1024/1536 x 496 and 512x512, while NEH
    # is uniformly 768x496. Height is retinal DEPTH and width is lateral
    # EXTENT, so squashing to a square distorts the two axes by different
    # amounts -- and by different amounts for different images.
    #
    #   "resize_crop" : resize the short side to IMAGE_SIZE, centre-crop.
    #                   Preserves geometry, loses the lateral periphery.
    #   "pad"         : resize the long side, pad the short side.
    #                   Preserves geometry and full field, adds black bars.
    #   "squash"      : plain resize to (IMAGE_SIZE, IMAGE_SIZE).
    #                   What the rejected pipeline did. Kept only so the
    #                   distortion can be quantified as an ablation.
    RESIZE_STRATEGY = _T["resize"]

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # GAN normalisation, for the tanh-output generator ([-1, 1])
    GAN_MEAN = [0.5, 0.5, 0.5]
    GAN_STD = [0.5, 0.5, 0.5]

    # =====================================================================
    # AUGMENTATION
    # =====================================================================

    # Horizontal flip mirrors nasal <-> temporal. Anatomically valid on a
    # B-scan and the standard augmentation for retinal images.
    HFLIP_PROB = 0.5

    # VERTICAL FLIP IS NEVER VALID HERE. The vertical axis is retinal depth:
    # ILM is always superior to RPE, and DRUSEN / CNV are defined by lesion
    # position RELATIVE to the RPE. Flipping teaches an impossible anatomy
    # and destroys the label's meaning. There is deliberately no config
    # entry to enable it.

    # Rotation is kept small for the same reason. The rejected pipeline
    # stacked Rotate(15) with ShiftScaleRotate(10), giving up to 25 degrees.
    ROTATION_LIMIT = 5
    ROTATION_PROB = 0.25

    BRIGHTNESS_LIMIT = 0.10
    CONTRAST_LIMIT = 0.10
    BRIGHTNESS_CONTRAST_PROB = 0.30

    # Gaussian blur is omitted: OCT speckle texture carries information
    # about tissue reflectivity, so blurring removes signal, not noise.

    # =====================================================================
    # MODEL
    # =====================================================================

    MODEL_NAME = "convnext_tiny"
    PRETRAINED = True
    DROPOUT = 0.30              # ConvNeXt also regularises via stoch. depth

    # =====================================================================
    # TRAINING
    # =====================================================================

    # Measured on this machine, ConvNeXt-Tiny, batch 16, 224px:
    #   full fine-tune   ~32.6 s/batch  ->  ~8   h/epoch over 42,838 images
    #   frozen backbone  ~14.5 s/batch  ->  ~3.5 h/epoch
    #
    # Hence the default protocol is CACHED LINEAR PROBE: run the frozen
    # backbone over every image once (~6-8 h), store the 768-d features,
    # then train the head on cached vectors in seconds per epoch. Same
    # result as an on-the-fly linear probe, and it makes the ablations,
    # the cohort probe and any hyperparameter search actually affordable.
    #
    #   "cached_probe" : cache features once, train head on vectors
    #   "linear_probe" : frozen backbone, images each epoch (no cache)
    #   "finetune"     : whole network trainable (GPU strongly advised)
    TRAIN_MODE = "cached_probe"

    BATCH_SIZE = 16             # image batches (fits comfortably in 16 GB)
    FEATURE_BATCH_SIZE = 512    # cached-vector batches; tiny and fast

    EPOCHS = 30                 # cheap once features are cached
    FINETUNE_EPOCHS = 10

    # Separate rates: a fresh head tolerates a large LR, pretrained weights
    # do not. models.ConvNeXtTiny.param_groups() consumes both.
    HEAD_LR = 1e-3
    BACKBONE_LR = 1e-5

    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.10
    OPTIMIZER = "AdamW"
    BETAS = (0.9, 0.999)

    # Pooled class counts are roughly NORMAL 29.8k / CNV 15.5k / DRUSEN 9.7k
    # / DME 8.2k, about 3.6:1. Weight the loss rather than resampling, so
    # every image is still seen once per epoch.
    USE_CLASS_WEIGHTS = True

    # =====================================================================
    # SCHEDULER / EARLY STOPPING
    # =====================================================================

    SCHEDULER = "ReduceLROnPlateau"
    SCHEDULER_MODE = "max"      # tracking macro-F1, so higher is better
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 3
    MIN_LR = 1e-6

    PATIENCE = 7                # must exceed SCHEDULER_PATIENCE
    BEST_METRIC = "macro_f1"

    # =====================================================================
    # CYCLEGAN  (not used by the baseline; kept for the augmentation study)
    # =====================================================================

    GAN_IMAGE_SIZE = 256
    RESIDUAL_BLOCKS = 9         # 9 for 256px, 6 for 128px
    GENERATOR_DROPOUT = 0.0     # the reference uses none for this mode
    PATCHGAN_N_LAYERS = 3       # -> 70x70 receptive field, verified in models.py

    GAN_LOSS_TYPE = "LSGAN"
    LAMBDA_CYCLE = 10.0
    LAMBDA_IDENTITY = 5.0       # applied directly, i.e. 0.5 * LAMBDA_CYCLE

    CYCLEGAN_BATCH_SIZE = 2
    CYCLEGAN_EPOCHS = 200       # the reference schedule; 10 does not converge
    CYCLEGAN_LR = 2e-4
    CYCLEGAN_BETAS = (0.5, 0.999)   # NOT the classifier's (0.9, 0.999)
    CYCLEGAN_DECAY_START = 100      # linear decay to zero after this epoch
    IMAGE_POOL_SIZE = 50            # discriminator history buffer

    # =====================================================================
    # EXPERIMENTS
    # =====================================================================

    # Every run writes to OUTPUT_DIR/<EXPERIMENT_NAME>/ so arms cannot
    # overwrite each other -- the rejected pipeline shared one
    # best_model.pth across the baseline and augmented arms.
    EXPERIMENT_NAME = "baseline_cached_probe"

    USE_SYNTHETIC_IMAGES = False
    SYNTHETIC_RATIO = 0.0

    RUN_COHORT_PROBE = True     # NEH vs Kermany; measures the domain gap

    # =====================================================================
    # LOGGING / METRICS
    # =====================================================================

    PRINT_FREQ = 50

    METRICS = [
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "roc_auc_ovr_macro",
    ]

    # =================================================================
    # DERIVED PATHS
    # =================================================================

    @classmethod
    def experiment_dir(cls, name: str | None = None) -> str:
        # namespaced by track, so an OCT run and a CFP run of the same
        # experiment name cannot overwrite each other
        return os.path.join(cls.OUTPUT_DIR, cls.TRACK,
                            name or cls.EXPERIMENT_NAME)

    @classmethod
    def paths(cls, name: str | None = None) -> dict:
        """Per-experiment output paths; creates the directories."""
        root = cls.experiment_dir(name)
        p = {
            "root": root,
            "checkpoints": os.path.join(root, "checkpoints"),
            "metrics": os.path.join(root, "metrics"),
            "plots": os.path.join(root, "plots"),
            "logs": os.path.join(root, "logs"),
        }
        for v in p.values():
            os.makedirs(v, exist_ok=True)
        return p

    @classmethod
    def feature_cache_path(cls, split: str) -> str:
        os.makedirs(cls.FEATURE_CACHE_DIR, exist_ok=True)
        tag = f"{cls.MODEL_NAME}_{cls.IMAGE_SIZE}_{cls.RESIZE_STRATEGY}"
        return os.path.join(cls.FEATURE_CACHE_DIR, f"{tag}_{split}.npz")

    # =================================================================
    # VALIDATION
    # =================================================================

    @classmethod
    def validate(cls, check_data: bool = True) -> list[str]:
        """Fail fast on misconfiguration. Returns a list of problems."""
        problems = []

        if cls.TRACK not in cls.TRACKS:
            problems.append(f"unknown TRACK {cls.TRACK!r}; "
                            f"valid: {sorted(cls.TRACKS)}")
        if cls.NUM_CLASSES != len(cls.CLASSES):
            problems.append("NUM_CLASSES does not match len(CLASSES)")
        if len(set(cls.CLASSES)) != len(cls.CLASSES):
            problems.append("CLASSES contains duplicates")
        if cls.PATIENCE <= cls.SCHEDULER_PATIENCE:
            problems.append("PATIENCE must exceed SCHEDULER_PATIENCE, "
                            "or early stopping fires before the LR drops")
        if cls.RESIZE_STRATEGY not in ("resize_crop", "pad", "squash",
                                       "normalize_768"):
            problems.append(f"unknown RESIZE_STRATEGY {cls.RESIZE_STRATEGY!r}")
        if cls.TRAIN_MODE not in ("cached_probe", "linear_probe", "finetune"):
            problems.append(f"unknown TRAIN_MODE {cls.TRAIN_MODE!r}")
        if cls.USE_AMP and cls.DEVICE == "cpu":
            problems.append("USE_AMP is CUDA-only; set it False on CPU")
        if cls.PERSISTENT_WORKERS and cls.NUM_WORKERS == 0:
            problems.append("PERSISTENT_WORKERS requires NUM_WORKERS > 0")
        if cls.PIN_MEMORY and cls.DEVICE == "cpu":
            problems.append("PIN_MEMORY does nothing without CUDA")
        if cls.SCHEDULER_MODE == "max" and "loss" in cls.BEST_METRIC:
            problems.append("SCHEDULER_MODE 'max' with a loss metric")

        if check_data:
            if not os.path.isfile(cls.POOLED_MANIFEST):
                problems.append(f"manifest not found: {cls.POOLED_MANIFEST}")
            else:
                # the manifest's cohorts must all have a configured root,
                # or dataset.py cannot build absolute paths
                import csv as _csv
                with open(cls.POOLED_MANIFEST, newline="") as _f:
                    _r = _csv.DictReader(_f)
                    _cohorts, _labels = set(), set()
                    for _i, _row in enumerate(_r):
                        _cohorts.add(_row.get("cohort", ""))
                        _labels.add(_row.get("y_label", ""))
                        if _i > 20000:
                            break
                _miss = _cohorts - set(cls.DATA_ROOTS)
                if _miss:
                    problems.append(f"manifest has cohort(s) with no root: "
                                    f"{sorted(_miss)}")
                _bad = _labels - set(cls.CLASSES)
                if _bad:
                    problems.append(f"manifest has labels outside "
                                    f"CLASSES: {sorted(_bad)}")
            for k, v in cls.DATA_ROOTS.items():
                if not os.path.isdir(v):
                    problems.append(f"data root missing for '{k}': {v}")

        return problems

    @classmethod
    def summary(cls) -> str:
        lines = [
            "=" * 66,
            "CONFIGURATION",
            "=" * 66,
            f"  TRACK           : {cls.TRACK}",
            f"  device          : {cls.DEVICE}",
            f"  torch threads   : {cls.TORCH_NUM_THREADS}",
            f"  train mode      : {cls.TRAIN_MODE}",
            f"  model           : {cls.MODEL_NAME} "
            f"(pretrained={cls.PRETRAINED})",
            f"  classes         : {cls.CLASSES}",
            f"  image size      : {cls.IMAGE_SIZE} "
            f"({cls.RESIZE_STRATEGY})",
            f"  batch size      : {cls.BATCH_SIZE}",
            f"  epochs          : {cls.EPOCHS}",
            f"  head LR         : {cls.HEAD_LR}",
            f"  backbone LR     : {cls.BACKBONE_LR}",
            f"  class weights   : {cls.USE_CLASS_WEIGHTS}",
            f"  hflip / rot     : {cls.HFLIP_PROB} / "
            f"{cls.ROTATION_LIMIT} deg",
            f"  best metric     : {cls.BEST_METRIC}",
            f"  experiment      : {cls.EXPERIMENT_NAME}",
            f"  manifest        : {cls.POOLED_MANIFEST}",
            f"  cache           : {cls.FEATURE_CACHE_DIR}",
            "=" * 66,
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    print()
    print(Config.summary())

    probs = Config.validate(check_data=True)
    if probs:
        print("\n  PROBLEMS:")
        for p in probs:
            print(f"    - {p}")
    else:
        print("\n  validate(): no problems")

    print(f"\n  experiment dir : {Config.experiment_dir()}")
    print(f"  feature cache  : {Config.feature_cache_path('train')}")
    print(f"\n  track note: {Config.TRACK_NOTE}")
    if Config.TRACK == "oct":
        print(f"\n  {Config.COHORT_CONFOUND_NOTE}")

    print(f"\n  available tracks:")
    for _t, _c in Config.TRACKS.items():
        _mark = " <-- active" if _t == Config.TRACK else ""
        print(f"    {_t:<14} {_c['manifest']:<22} "
              f"{len(_c['classes'])} classes{_mark}")
    print()