import os
from glob import glob

import numpy as np

from PIL import Image

import torch
from torch.utils.data import Dataset

from config import Config



# =========================================================
# GENERIC RETINAL DATASET
# =========================================================

class RetinalDataset(Dataset):

    def __init__(
        self,
        image_paths,
        class_to_idx,
        transform=None
    ):

        self.image_paths = image_paths

        self.class_to_idx = class_to_idx

        self.transform = transform

    # =====================================================
    # TOTAL SAMPLES
    # =====================================================

    def __len__(self):

        return len(self.image_paths)

    # =====================================================
    # GET ITEM
    # =====================================================

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        # ---------------------------------------------
        # CLASS NAME
        # ---------------------------------------------

        class_name = os.path.basename(
            os.path.dirname(image_path)
        )

        label = self.class_to_idx[class_name]

        # ---------------------------------------------
        # LOAD IMAGE
        # ---------------------------------------------

        image = Image.open(image_path).convert("RGB")

        # ---------------------------------------------
        # TRANSFORMS
        # ---------------------------------------------

        if self.transform is not None:

            if self.transform is not None:

                augmented = self.transform(
                image=np.array(image)
            )

            image = augmented["image"]
        return image, label


# =========================================================
# GET IMAGE PATHS
# =========================================================

def get_image_paths(data_dir):

    image_paths = []

    for extension in ["*.jpg", "*.jpeg", "*.png"]:

        image_paths.extend(
            glob(
                os.path.join(
                    data_dir,
                    "*",
                    extension
                )
            )
        )

    image_paths = sorted(image_paths)

    return image_paths


# =========================================================
# OCT DATASET PATHS
# =========================================================

def get_oct_datasets():

    train_dir = os.path.join(
        Config.PROCESSED_OCT_DIR,
        "train"
    )

    val_dir = os.path.join(
        Config.PROCESSED_OCT_DIR,
        "val"
    )

    test_dir = os.path.join(
        Config.PROCESSED_OCT_DIR,
        "test"
    )

    train_paths = get_image_paths(train_dir)

    val_paths = get_image_paths(val_dir)

    test_paths = get_image_paths(test_dir)

    class_to_idx = {
        class_name: index
        for index, class_name in enumerate(
            Config.OCT_CLASSES
        )
    }

    return (
        train_paths,
        val_paths,
        test_paths,
        class_to_idx
    )


# =========================================================
# AMD DATASET PATHS
# =========================================================

def get_amd_datasets():

    train_dir = os.path.join(
        Config.PROCESSED_AMD_DIR,
        "train"
    )

    val_dir = os.path.join(
        Config.PROCESSED_AMD_DIR,
        "val"
    )

    test_dir = os.path.join(
        Config.PROCESSED_AMD_DIR,
        "test"
    )

    train_paths = get_image_paths(train_dir)

    val_paths = get_image_paths(val_dir)

    test_paths = get_image_paths(test_dir)

    class_to_idx = {
        class_name: index
        for index, class_name in enumerate(
            Config.AMD_CLASSES
        )
    }

    return (
        train_paths,
        val_paths,
        test_paths,
        class_to_idx
    )


# =========================================================
# PRINT DATASET STATS
# =========================================================

def print_dataset_stats(name, train_paths, val_paths, test_paths):

    print("\n===================================")
    print(f"{name} DATASET")
    print("===================================")

    print(f"Train Images : {len(train_paths)}")
    print(f"Val Images   : {len(val_paths)}")
    print(f"Test Images  : {len(test_paths)}")

    print("===================================\n")


# =========================================================
# MAIN TEST
# =========================================================

if __name__ == "__main__":

    # ---------------------------------------------
    # OCT
    # ---------------------------------------------

    (
        oct_train,
        oct_val,
        oct_test,
        _
    ) = get_oct_datasets()

    print_dataset_stats(
        "OCT2017",
        oct_train,
        oct_val,
        oct_test
    )

    # ---------------------------------------------
    # AMD
    # ---------------------------------------------

    (
        amd_train,
        amd_val,
        amd_test,
        _
    ) = get_amd_datasets()

    print_dataset_stats(
        "AMD",
        amd_train,
        amd_val,
        amd_test
    )