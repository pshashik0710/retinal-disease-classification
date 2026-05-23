# cyclegan_dataset.py

import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CycleGANDataset(Dataset):
    """
    CycleGAN Dataset Loader

    Handles:
    - Unpaired image loading
    - Domain X ↔ Domain Y loading
    - OCT ↔ AMD image translation datasets

    Expected Folder Structure:

    dataset/
    ├── trainA/
    ├── trainB/
    ├── testA/
    └── testB/

    Example:
    trainA -> Fundus / AMD images
    trainB -> OCT images
    """

    def __init__(
        self,
        root_dir,
        mode="train",
        transform=None,
        image_size=256
    ):
        """
        Args:
            root_dir (str): Dataset root path
            mode (str): train or test
            transform (callable): Optional transforms
            image_size (int): Resize dimension
        """

        self.root_dir = root_dir
        self.mode = mode

        # Domain X and Domain Y
        self.dir_X = os.path.join(root_dir, f"{mode}A")
        self.dir_Y = os.path.join(root_dir, f"{mode}B")

        # Image lists
        self.X_images = sorted(os.listdir(self.dir_X))
        self.Y_images = sorted(os.listdir(self.dir_Y))

        # Default transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ])
        else:
            self.transform = transform

    def __len__(self):
        """
        Returns maximum domain length
        """

        return max(len(self.X_images), len(self.Y_images))

    def __getitem__(self, index):
        """
        Returns:
            real_X : image from domain X
            real_Y : image from domain Y
        """

        # Domain X image
        X_path = os.path.join(
            self.dir_X,
            self.X_images[index % len(self.X_images)]
        )

        # Random unpaired image from domain Y
        Y_path = os.path.join(
            self.dir_Y,
            random.choice(self.Y_images)
        )

        # Open images
        image_X = Image.open(X_path).convert("RGB")
        image_Y = Image.open(Y_path).convert("RGB")

        # Apply transforms
        image_X = self.transform(image_X)
        image_Y = self.transform(image_Y)

        return {
            "X": image_X,
            "Y": image_Y,
            "X_path": X_path,
            "Y_path": Y_path
        }


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":

    dataset = CycleGANDataset(
        root_dir="/home/user24/retinal_project/dataset",
        mode="train",
        image_size=256
    )

    print("Dataset Size:", len(dataset))

    sample = dataset[0]

    print("Domain X Shape:", sample["X"].shape)
    print("Domain Y Shape:", sample["Y"].shape)
    print("X Path:", sample["X_path"])
    print("Y Path:", sample["Y_path"])