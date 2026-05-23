import os
import shutil
from glob import glob

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import Config
from seed_utils import set_seed


# =========================================================
# SET SEED
# =========================================================

set_seed(Config.SEED)


# =========================================================
# CREATE DIRECTORY
# =========================================================

def create_dir(path):
    os.makedirs(path, exist_ok=True)


# =========================================================
# COPY FILES
# =========================================================

def copy_files(file_list, destination_dir):

    create_dir(destination_dir)

    for file_path in tqdm(file_list):

        filename = os.path.basename(file_path)

        destination_path = os.path.join(
            destination_dir,
            filename
        )

        shutil.copy2(file_path, destination_path)


# =========================================================
# OCT2017 PREPROCESSING
# =========================================================

def preprocess_oct2017():

    print("\n==============================")
    print("PREPROCESSING OCT2017")
    print("==============================\n")

    for class_name in Config.OCT_CLASSES:

        print(f"\nProcessing class: {class_name}")

        # ---------------------------------------------
        # RAW CLASS PATH
        # ---------------------------------------------

        class_path = os.path.join(
            Config.OCT_RAW_DIR,
            "train",
            class_name
        )

        image_paths = sorted(
            glob(os.path.join(class_path, "*"))
        )

        print(f"Total images: {len(image_paths)}")

        # ---------------------------------------------
        # MAIN TRAIN / TEST SPLIT
        # 85% TRAIN
        # 15% TEST
        # ---------------------------------------------

        train_images, test_images = train_test_split(
            image_paths,
            test_size=Config.TEST_RATIO,
            random_state=Config.SEED,
            shuffle=True
        )

        # ---------------------------------------------
        # TRAIN / VAL SPLIT
        # 80% TRAIN
        # 20% VAL
        # ---------------------------------------------

        train_images, val_images = train_test_split(
            train_images,
            test_size=Config.VAL_RATIO,
            random_state=Config.SEED,
            shuffle=True
        )

        print(f"Train: {len(train_images)}")
        print(f"Val  : {len(val_images)}")
        print(f"Test : {len(test_images)}")

        # ---------------------------------------------
        # DESTINATION PATHS
        # ---------------------------------------------

        train_dir = os.path.join(
            Config.PROCESSED_OCT_DIR,
            "train",
            class_name
        )

        val_dir = os.path.join(
            Config.PROCESSED_OCT_DIR,
            "val",
            class_name
        )

        test_dir = os.path.join(
            Config.PROCESSED_OCT_DIR,
            "test",
            class_name
        )

        # ---------------------------------------------
        # COPY FILES
        # ---------------------------------------------

        copy_files(train_images, train_dir)

        copy_files(val_images, val_dir)

        copy_files(test_images, test_dir)

    print("\nOCT2017 preprocessing completed.\n")


# =========================================================
# AMD PREPROCESSING
# =========================================================

def preprocess_amd():

    print("\n==============================")
    print("PREPROCESSING AMD")
    print("==============================\n")

    for class_name in Config.AMD_CLASSES:

        print(f"\nProcessing class: {class_name}")

        # ---------------------------------------------
        # ORIGINAL TRAIN PATH
        # ---------------------------------------------

        train_class_path = os.path.join(
            Config.AMD_RAW_DIR,
            "train",
            class_name
        )

        image_paths = sorted(
            glob(os.path.join(train_class_path, "*"))
        )

        print(f"Original train images: {len(image_paths)}")

        # ---------------------------------------------
        # CREATE TEST SPLIT
        # ---------------------------------------------

        train_images, test_images = train_test_split(
            image_paths,
            test_size=Config.TEST_RATIO,
            random_state=Config.SEED,
            shuffle=True
        )

        print(f"Train: {len(train_images)}")
        print(f"Test : {len(test_images)}")

        # ---------------------------------------------
        # VALIDATION PATH
        # ---------------------------------------------

        valid_class_path = os.path.join(
            Config.AMD_RAW_DIR,
            "valid",
            class_name
        )

        val_images = sorted(
            glob(os.path.join(valid_class_path, "*"))
        )

        print(f"Val  : {len(val_images)}")

        # ---------------------------------------------
        # DESTINATION PATHS
        # ---------------------------------------------

        train_dir = os.path.join(
            Config.PROCESSED_AMD_DIR,
            "train",
            class_name
        )

        val_dir = os.path.join(
            Config.PROCESSED_AMD_DIR,
            "val",
            class_name
        )

        test_dir = os.path.join(
            Config.PROCESSED_AMD_DIR,
            "test",
            class_name
        )

        # ---------------------------------------------
        # COPY FILES
        # ---------------------------------------------

        copy_files(train_images, train_dir)

        copy_files(val_images, val_dir)

        copy_files(test_images, test_dir)

    print("\nAMD preprocessing completed.\n")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    preprocess_oct2017()

    preprocess_amd()

    print("\n===================================")
    print("ALL DATASETS PREPROCESSED")
    print("===================================\n")