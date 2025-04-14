import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class StainDataset(Dataset):
    """Dataset for stain detection from images"""

    def __init__(self, data_dir, image_size=224, transform=None, mode="train"):
        """
        Args:
            data_dir (str): Directory with class subdirectories of images
            image_size (int): Size to resize images to
            transform (callable, optional): Optional transform to apply to images
            mode (str): 'train', 'val', or 'test' mode
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.mode = mode

        # Find class directories
        self.class_dirs = [
            d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
        ]
        self.class_dirs.sort()  # Ensure consistent ordering
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_dirs)}

        # Collect image paths and labels
        self.image_paths = []
        self.labels = []

        for class_name in self.class_dirs:
            class_dir = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".tif", ".tiff")
                ):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])

        # Use provided transform or create default
        if transform:
            self.transform = transform
        else:
            if mode == "train":
                self.transform = transforms.Compose(
                    [
                        transforms.Resize((image_size, image_size)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomRotation(20),
                        transforms.ColorJitter(
                            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
                        ),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
            else:
                self.transform = transforms.Compose(
                    [
                        transforms.Resize((image_size, image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    @property
    def classes(self):
        return self.class_dirs

    @property
    def num_classes(self):
        return len(self.class_dirs)


def get_data_loaders(config):
    """Create data loaders based on configuration

    Args:
        config: Configuration object with data parameters

    Returns:
        dict: Dictionary with train, val, and test data loaders
    """
    image_size = config.data.image_size
    batch_size = config.data.batch_size

    # Create datasets
    train_dataset = StainDataset(
        data_dir=config.data.train_dir, image_size=image_size, mode="train"
    )

    val_dataset = StainDataset(
        data_dir=config.data.val_dir, image_size=image_size, mode="val"
    )

    test_dataset = StainDataset(
        data_dir=config.data.test_dir, image_size=image_size, mode="test"
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "num_classes": train_dataset.num_classes,
        "classes": train_dataset.classes,
    }
