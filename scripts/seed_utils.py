import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set complete reproducibility for training and evaluation.
    """

    # Python seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Random seeds
    random.seed(seed)
    np.random.seed(seed)

    # Torch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CUDA reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Extra deterministic behavior
    torch.use_deterministic_algorithms(True)

    # CUBLAS reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    print(f"Seed set to: {seed}")