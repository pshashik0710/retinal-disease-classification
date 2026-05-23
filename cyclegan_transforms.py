# cyclegan_transforms.py

from torchvision import transforms


def get_transforms(image_size=256):
    """
    CycleGAN Image Transformations

    Handles:
    - Resizing
    - Random cropping
    - Random horizontal flipping
    - Tensor conversion
    - Normalization to [-1, +1]

    Args:
        image_size (int): Final image size

    Returns:
        torchvision.transforms.Compose
    """

    transform = transforms.Compose([

        # Resize slightly larger before crop
        transforms.Resize((image_size + 30, image_size + 30)),

        # Random crop for robustness
        transforms.RandomCrop(image_size),

        # Random horizontal flip for augmentation
        transforms.RandomHorizontalFlip(p=0.5),

        # Convert PIL image → Tensor
        transforms.ToTensor(),

        # Normalize to [-1, +1]
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    return transform


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":

    transform = get_transforms(image_size=256)

    print("CycleGAN Transforms Loaded Successfully")
    print(transform)