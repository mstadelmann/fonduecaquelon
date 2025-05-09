from PIL import Image
import torch
import torchvision.transforms.functional as TF


class ResizeMax(object):
    def __init__(self, max_size=256, interpolation=Image.NEAREST):
        self.max_size = max_size
        self.interpolation = interpolation

    def __call__(self, img):
        # Handle PIL Image
        if isinstance(img, Image.Image):
            w, h = img.size
            scale = min(self.max_size / w, self.max_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            return TF.resize(img, (new_h, new_w), interpolation=self.interpolation)

        # Handle Tensor (C, H, W) or (H, W)
        elif isinstance(img, torch.Tensor):
            if img.ndim == 3:
                _, h, w = img.shape
            elif img.ndim == 2:
                h, w = img.shape
            else:
                raise ValueError("Unsupported tensor shape: expected 2D or 3D tensor")

            scale = min(self.max_size / w, self.max_size / h)
            new_w, new_h = int(w * scale), int(h * scale)

            return TF.resize(img, [new_h, new_w], interpolation=self.interpolation)

        else:
            raise TypeError(f"Unsupported image type: {type(img)}")
