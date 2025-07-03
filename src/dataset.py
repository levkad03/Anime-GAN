import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class AnimeFacesDataset(Dataset):
    def __init__(
        self,
        root_dir,
        transform=None,
        image_size=64,
        extensions=(".jpg", ".jpeg", ".png", ".bmp"),
    ):
        self.root_dir = root_dir
        self.image_size = image_size
        self.extensions = extensions

        # Get all image files
        self.image_files = []

        for file in os.listdir(root_dir):
            if file.lower().endswith(self.extensions):
                self.image_files.append(os.path.join(root_dir, file))

        if len(self.image_files) == 0:
            raise ValueError(
                f"No images found in {root_dir} with extensions {extensions}"
            )

        # Set default transform if none provided
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                    ),  # Normalize to [-1, 1]
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image
