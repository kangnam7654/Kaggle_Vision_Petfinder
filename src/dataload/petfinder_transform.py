import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD = [0.229, 0.224, 0.225]  # RGB


def load_transforms(config):
    transform = {
        "train": A.Compose(
            [
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomRotate90(),
                A.ColorJitter(p=0.8),
                A.CoarseDropout(),
                A.LongestMaxSize(max_size=config["transform"]["image_size"]),
                A.PadIfNeeded(
                    min_height=config["transform"]["image_size"],
                    min_width=config["transform"]["image_size"],
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5,
                ),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                A.Resize(224, 224),
                ToTensorV2(),
            ]
        ),
        "valid": A.Compose(
            [
                A.LongestMaxSize(max_size=config["transform"]["image_size"]),
                A.PadIfNeeded(
                    min_height=config["transform"]["image_size"],
                    min_width=config["transform"]["image_size"],
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5,
                ),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                A.Resize(224, 224),
                ToTensorV2(),
            ]
        ),
        "test": A.Compose(
            [
                A.LongestMaxSize(max_size=config["transform"]["image_size"]),
                A.PadIfNeeded(
                    min_height=config["transform"]["image_size"],
                    min_width=config["transform"]["image_size"],
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5,
                ),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                A.Resize(224, 224),
                ToTensorV2(),
            ]
        ),
    }
    return transform

