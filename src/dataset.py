import logging
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


class StainDataset(Dataset):
    """Dataset for stain detection from images"""

    def __init__(
        self,
        data_dir,
        expected_num_classes,
        image_size=224,
        transform=None,
        mode="train",
    ):
        """
        Args:
            data_dir (str): Directory with class subdirectories of images
            image_size (int): Size to resize images to
            transform (callable, optional): Optional transform to apply to images
            mode (str): 'train', 'val', or 'test' mode
            expected_num_classes (int, optional): Expected number of classes from config
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.mode = mode

        # Find class directories
        self.class_dirs = [
            d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
        ]
        self.class_dirs.sort()  # Ensure consistent ordering

        # Validate number of classes if expected value is provided
        actual_num_classes = len(self.class_dirs)
        if actual_num_classes != expected_num_classes:
            error_msg = (
                f"Expected {expected_num_classes} classes in {data_dir}, "
                f"but found {actual_num_classes}. Class directories: {self.class_dirs}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

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
                        transforms.RandomChoice(
                            [
                                transforms.RandomRotation([0, 0]),  # No rotation
                                transforms.RandomRotation([90, 90]),  # 90 degrees
                                transforms.RandomRotation([180, 180]),  # 180 degrees
                                transforms.RandomRotation([270, 270]),  # 270 degrees
                            ]
                        ),
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
