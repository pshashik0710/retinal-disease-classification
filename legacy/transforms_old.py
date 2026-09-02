import albumentations as A

from albumentations.pytorch import ToTensorV2

from config import Config


# =========================================================
# CONVNEXT TRAIN TRANSFORMS
# =========================================================

def get_train_transforms():

    return A.Compose([

        # ---------------------------------------------
        # Resize
        # ---------------------------------------------

        A.Resize(
            height=Config.IMAGE_SIZE,
            width=Config.IMAGE_SIZE
        ),

        # ---------------------------------------------
        # MEDICAL IMAGE AUGMENTATION
        # ---------------------------------------------

        A.HorizontalFlip(p=0.5),

        A.VerticalFlip(p=0.2),

        A.Rotate(
            limit=15,
            p=0.5
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.5
        ),

        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=10,
            border_mode=0,
            p=0.5
        ),

        A.GaussianBlur(
            blur_limit=(3, 3),
            p=0.2
        ),

        # ---------------------------------------------
        # NORMALIZATION
        # ConvNeXt / ImageNet
        # approximately [-2, 2]
        # ---------------------------------------------

        A.Normalize(
            mean=Config.IMAGENET_MEAN,
            std=Config.IMAGENET_STD
        ),

        # ---------------------------------------------
        # TO TENSOR
        # ---------------------------------------------

        ToTensorV2()

    ])


# =========================================================
# VALIDATION TRANSFORMS
# =========================================================

def get_valid_transforms():

    return A.Compose([

        A.Resize(
            height=Config.IMAGE_SIZE,
            width=Config.IMAGE_SIZE
        ),

        A.Normalize(
            mean=Config.IMAGENET_MEAN,
            std=Config.IMAGENET_STD
        ),

        ToTensorV2()

    ])


# =========================================================
# TEST TRANSFORMS
# =========================================================

def get_test_transforms():

    return A.Compose([

        A.Resize(
            height=Config.IMAGE_SIZE,
            width=Config.IMAGE_SIZE
        ),

        A.Normalize(
            mean=Config.IMAGENET_MEAN,
            std=Config.IMAGENET_STD
        ),

        ToTensorV2()

    ])


# =========================================================
# CYCLEGAN TRANSFORMS
# =========================================================

def get_gan_transforms():

    return A.Compose([

        # ---------------------------------------------
        # Resize
        # ---------------------------------------------

        A.Resize(
            height=Config.IMAGE_SIZE,
            width=Config.IMAGE_SIZE
        ),

        # ---------------------------------------------
        # GAN NORMALIZATION
        # maps image to [-1, 1]
        # ---------------------------------------------

        A.Normalize(
            mean=Config.GAN_MEAN,
            std=Config.GAN_STD
        ),

        # ---------------------------------------------
        # TO TENSOR
        # ---------------------------------------------

        ToTensorV2()

    ])


# =========================================================
# MAIN TEST
# =========================================================

if __name__ == "__main__":

    train_transform = get_train_transforms()

    val_transform = get_valid_transforms()

    test_transform = get_test_transforms()

    gan_transform = get_gan_transforms()

    print("\nTransforms initialized successfully.\n")