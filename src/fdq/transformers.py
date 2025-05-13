from PIL import Image
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision.transforms.v2 import Transform


class ResizeMaxDimPad(Transform):
    def __init__(self, max_dim: int, interpol_mode="bilinear"):
        super().__init__()
        self.max_dim = max_dim
        self.interpol_mode = interpol_mode

        if interpol_mode not in ["nearest", "linear", "bilinear", "bicubic"]:
            raise ValueError("Mode must be 'bilinear' or 'nearest'")

    def transform(self, inpt: torch.Tensor, params=None):
        if not isinstance(inpt, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")

        # Input shape: (C, H, W)
        c, h, w = inpt.shape

        # Scale to max_dim
        scale = self.max_dim / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize
        inpt = F.interpolate(
            inpt.unsqueeze(0),
            size=(new_h, new_w),
            mode=self.interpol_mode,
            # align_corners=False,
        ).squeeze(0)

        # Padding
        pad_h = self.max_dim - new_h
        pad_w = self.max_dim - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        inpt = F.pad(
            inpt, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0
        )

        return inpt


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
