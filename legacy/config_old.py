import os
import torch


# =========================================================
# PROJECT CONFIGURATION
# =========================================================

class Config:

    # =====================================================
    # REPRODUCIBILITY
    # =====================================================

    SEED = 42

    DETERMINISTIC = True

    # =====================================================
    # DATASET SPLITS
    # =====================================================

    # Main split
    TRAIN_RATIO = 0.85
    TEST_RATIO = 0.15

    # Validation split inside train
    VAL_RATIO = 0.20

    # =====================================================
    # CROSS VALIDATION
    # =====================================================

    NUM_FOLDS = 5

    # =====================================================
    # IMAGE SETTINGS
    # =====================================================

    IMAGE_SIZE = 256

    IMAGE_CHANNELS = 3

    # =====================================================
    # NORMALIZATION
    # =====================================================

    # CycleGAN normalization
    # maps [0,255] -> [-1,1]
    GAN_MEAN = [0.5, 0.5, 0.5]
    GAN_STD = [0.5, 0.5, 0.5]

    # ConvNeXt ImageNet normalization
    # approximately maps near [-2,2]
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # =====================================================
    # DEVICE
    # =====================================================

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =====================================================
    # GPU SPEED OPTIMIZATION
    # =====================================================

    NUM_WORKERS = 8

    PIN_MEMORY = True

    PERSISTENT_WORKERS = True

    PREFETCH_FACTOR = 4

    # Mixed Precision
    USE_AMP = True

    # =====================================================
    # CLASSIFICATION TRAINING
    # =====================================================

    BATCH_SIZE = 64

    EPOCHS = 25

    LEARNING_RATE = 1e-4

    WEIGHT_DECAY = 1e-4

    LABEL_SMOOTHING = 0.1

    # =====================================================
    # EARLY STOPPING
    # =====================================================

    PATIENCE = 6

    # =====================================================
    # OPTIMIZER
    # =====================================================

    OPTIMIZER = "AdamW"

    BETAS = (0.9, 0.999)

    # =====================================================
    # SCHEDULER
    # =====================================================

    SCHEDULER = "ReduceLROnPlateau"

    SCHEDULER_FACTOR = 0.5

    SCHEDULER_PATIENCE = 3

    MIN_LR = 1e-6

    # =====================================================
    # MODEL SETTINGS
    # =====================================================

    MODEL_NAME = "convnext_tiny"

    PRETRAINED = True

    DROPOUT = 0.5

    NUM_CLASSES = 4

    # =====================================================
    # CLASS NAMES
    # =====================================================

    OCT_CLASSES = [
        "CNV",
        "DME",
        "DRUSEN",
        "NORMAL"
    ]

    AMD_CLASSES = [
        "amd",
        "cataract",
        "diabetes",
        "normal"
    ]

    # =====================================================
    # BEST MODEL SELECTION
    # =====================================================

    BEST_METRIC = "macro_f1"

    # =====================================================
    # CYCLEGAN SETTINGS
    # =====================================================

    # Generator architecture
    GENERATOR_TYPE = "ResNet"

    RESIDUAL_BLOCKS = 9

    GENERATOR_DROPOUT = 0.5

    # Discriminator architecture
    DISCRIMINATOR_TYPE = "PatchGAN"

    PATCHGAN_SIZE = 70

    # GAN training
    GAN_LOSS_TYPE = "LSGAN"

    # Cycle consistency
    LAMBDA_CYCLE = 10.0

    # Identity loss
    LAMBDA_IDENTITY = 5.0

    # =====================================================
    # CYCLEGAN TRAINING
    # =====================================================

    CYCLEGAN_BATCH_SIZE = 2
    CYCLEGAN_EPOCHS = 10

    CYCLEGAN_LR = 2e-4

    # =====================================================
    # SYNTHETIC IMAGE GENERATION
    # =====================================================

    USE_SYNTHETIC_IMAGES = True

    SYNTHETIC_RATIO = 0.30

    # Example:
    # 70% real + 30% synthetic

    # =====================================================
    # ABLATION STUDY
    # =====================================================

    RUN_BASELINE = True

    RUN_CYCLEGAN_AUGMENTED = True

    # =====================================================
    # PATHS
    # =====================================================

    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    DATASET_DIR = os.path.join(
        BASE_DIR,
        "datasets"
    )

    PROCESSED_DIR = os.path.join(
        DATASET_DIR,
        "processed"
    )

    OCT_RAW_DIR = os.path.join(
        DATASET_DIR,
        "oct2017_raw",
        "OCT2017"
    )

    AMD_RAW_DIR = os.path.join(
        DATASET_DIR,
        "amd_raw",
        "AMDNet23 Dataset"
    )

    # =====================================================
    # PROCESSED DATASETS
    # =====================================================

    PROCESSED_OCT_DIR = os.path.join(
        PROCESSED_DIR,
        "oct2017"
    )

    PROCESSED_AMD_DIR = os.path.join(
        PROCESSED_DIR,
        "amd"
    )

    # =====================================================
    # PROCESSED OCT DATASET
    # =====================================================

    OCT_TRAIN_DIR = os.path.join(
        PROCESSED_OCT_DIR,
        "train"
    )

    OCT_VAL_DIR = os.path.join(
        PROCESSED_OCT_DIR,
        "val"
    )

    OCT_TEST_DIR = os.path.join(
        PROCESSED_OCT_DIR,
        "test"
    )

    # =====================================================
    # PROCESSED AMD DATASET
    # =====================================================

    AMD_TRAIN_DIR = os.path.join(
        PROCESSED_AMD_DIR,
        "train"
    )

    AMD_VAL_DIR = os.path.join(
        PROCESSED_AMD_DIR,
        "val"
    )

    AMD_TEST_DIR = os.path.join(
        PROCESSED_AMD_DIR,
        "test"
    )

    # =====================================================
    # CYCLEGAN DATA
    # =====================================================

    CYCLEGAN_DIR = os.path.join(
        DATASET_DIR,
        "cyclegan"
    )

    # Domain A -> OCT
    TRAIN_A_DIR = os.path.join(
        CYCLEGAN_DIR,
        "trainA"
    )

    TEST_A_DIR = os.path.join(
        CYCLEGAN_DIR,
        "testA"
    )

    # Domain B -> AMD
    TRAIN_B_DIR = os.path.join(
        CYCLEGAN_DIR,
        "trainB"
    )

    TEST_B_DIR = os.path.join(
        CYCLEGAN_DIR,
        "testB"
    )

    # =====================================================
    # OUTPUT DIRECTORIES
    # =====================================================

    CHECKPOINT_DIR = os.path.join(
        BASE_DIR,
        "checkpoints_label_smoothing"
    )

    LOG_DIR = os.path.join(
        BASE_DIR,
        "logs"
    )

    METRICS_DIR = os.path.join(
        BASE_DIR,
        "metrics_label_smoothing"
    )

    PLOTS_DIR = os.path.join(
        BASE_DIR,
        "plots_label_smoothing"
    )

    # =====================================================
    # CHECKPOINT FILES
    # =====================================================

    BEST_MODEL_PATH = os.path.join(
        CHECKPOINT_DIR,
        "best_model.pth"
    )

    RESUME_CHECKPOINT = os.path.join(
        CHECKPOINT_DIR,
        "resume_checkpoint.pth"
    )
    # =====================================================
    # LOGGING
    # =====================================================

    PRINT_FREQ = 50

    SAVE_FREQ = 1

    # =====================================================
    # EVALUATION METRICS
    # =====================================================

    METRICS = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ]