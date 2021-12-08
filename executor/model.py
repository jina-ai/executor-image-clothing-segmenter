import os
import cv2
import gdown
import torch
import numpy as np
import torch.nn.functional as f
import torchvision.transforms as transforms

from PIL import Image
from typing import List, TypeVar, Optional

from .utils import load_checkpoint_mgpu, NormalizeImage
from .u2net import U2NET


PillowImage = TypeVar("PillowImage")


class ClothingSegmentationModel:
    """Perform segmentation on images with fashion items using a pre-trained U2Net"""

    URL = 'https://drive.google.com/u/0/uc?id=1mhF3yqd7R-Uje092eypktNl-RoZNuiCJ&export=download'
    CACHE = './'
    NAME = 'cloth_segm_u2net_latest.pth'

    def __init__(self, model_path: Optional[str] = None, size: int = 768, masking_threshold: float = 0.05):
        """Initialization"""
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._net = U2NET(in_ch=3, out_ch=4)
        self._model_path = model_path or os.path.join(self.CACHE, self.NAME)
        self._net = load_checkpoint_mgpu(self._net, self._model_path)
        self._net = self._net.to(self._device)
        self._net = self._net.eval()
        self._transformations = transforms.Compose([transforms.ToTensor(), NormalizeImage(0.5, 0.5)])
        self._size = size
        self._masking_threshold = masking_threshold

    def __call__(self, images: List[PillowImage]) -> List[PillowImage]:
        """Callable object"""
        return self.segment(images=images)

    @property
    def device(self) -> str:
        """Device getter"""
        return self._device

    @classmethod
    def download(cls) -> None:
        """Download model"""
        gdown.download(cls.URL, cls.NAME, quiet=False)

    def preprocess(self, images: List[PillowImage]) -> List[PillowImage]:
        """Preprocess images"""
        preprocessed = []
        for image in images:
            img = image.convert("RGB")
            img.thumbnail((self._size, self._size), Image.ANTIALIAS)
            preprocessed.append(img)
        return preprocessed

    def transform(self, images: List[PillowImage]) -> torch.Tensor:
        """Transform images"""
        return torch.stack([self._transformations(image) for image in images], dim=0)

    def forward(self, x: torch.Tensor) -> np.ndarray:
        """Model forward pass"""
        y = self._net(x.to(self._device))
        y = f.log_softmax(y[0], dim=1)
        y = torch.max(y, dim=1, keepdim=True)[1]
        y = y.cpu().numpy()
        return y

    def reconstruct(self, originals: List[PillowImage], mask: np.ndarray) -> List[PillowImage]:
        """Merge original images with the segmentation masks"""
        merged = []
        for i, original in enumerate(originals):
            cv2img = np.array(original)
            _mask = mask[i, 0, :]
            _mask[_mask > 0] = 1
            if np.sum(_mask) < self._masking_threshold * sum(_mask.shape):
                merged.append(original)
                continue
            _mask = np.repeat(_mask[:, :, np.newaxis], 3, axis=2).astype('uint8')
            _mask[_mask == 1] = 255
            merged.append(Image.fromarray(cv2.bitwise_and(cv2img, _mask)))
        return merged

    def segment(self, images: List[PillowImage]) -> List[PillowImage]:
        """Segment images"""
        preprocessed = self.preprocess(images)
        x = self.transform(preprocessed)
        y = self.forward(x=x)
        return self.reconstruct(preprocessed, y)
