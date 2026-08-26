from typing import Any
from collections.abc import Callable
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms
from omegaconf import DictConfig


class AddValueTransform:
    """A transform that adds a specified value to the input tensor."""

    def __init__(self, value):
        """Initialize the AddValueTransform.

        Args:
            value: The value to add to the input tensor.
        """
        self.value = value

    def __call__(self, t):
        return t + self.value


class MultValueTransform:
    """A transform that multiplies the input tensor by a specified value."""

    def __init__(self, value):
        """Initialize the MultValueTransform.

        Args:
            value: The value to multiply the input tensor by.
        """
        self.value = value

    def __call__(self, t):
        return t * self.value


class DivValueTransform:
    """A transform that divides the input tensor by a specified value."""

    def __init__(self, value):
        """Initialize the DivValueTransform.

        Args:
            value: The value to divide the input tensor by.
        """
        self.value = value

    def __call__(self, t):
        return t / self.value


class ClampAbsTransform:
    """A transform that clamps the input tensor to the specified lower and upper bounds."""

    def __init__(self, lower, upper):
        """Initialize the ClampAbsTransform.

        Args:
            lower: The lower bound for clamping.
            upper: The upper bound for clamping.
        """
        self.lower = lower
        self.upper = upper

    def __call__(self, t):
        return t.clamp(self.lower, self.upper)


class ClampPercTransform:
    """A transform that clamps the input tensor based on lower and upper percentiles.

    Args:
        lower_perc (float): The lower percentile for clamping.
        upper_perc (float): The upper percentile for clamping.
    """

    def __init__(self, lower_perc, upper_perc):
        """Initialize the ClampPercTransform.

        Args:
            lower_perc (float): The lower percentile for clamping.
            upper_perc (float): The upper percentile for clamping.
        """
        self.lower_perc = lower_perc
        self.upper_perc = upper_perc

    def __call__(self, t):
        lower = torch.quantile(self.rdm_reduce(t), self.lower_perc)
        upper = torch.quantile(self.rdm_reduce(t), self.upper_perc)
        return t.clamp(lower, upper)

    def rdm_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        # random reduce tensor size for quantile computation -> estimation
        n = 12582912  # random defined as 8*3*32*128*128
        if tensor.numel() > n:
            n = min(n, tensor.numel())
            random_indices = torch.randperm(tensor.numel())[:n]
            # random_indices = torch.multinomial(
            #     torch.ones(tensor.numel()), n, replacement=False
            # )
            return tensor.view(-1)[random_indices]
        return tensor


class ReRangeTransform:
    """Re-ranges the input tensor from [in_min, in_max] to [out_min, out_max]."""

    def __init__(self, in_min, in_max, out_min, out_max):
        """Initialize the ReRangeTransform.

        Args:
            in_min: Minimum value of the input range.
            in_max: Maximum value of the input range.
            out_min: Minimum value of the output range.
            out_max: Maximum value of the output range.
        """
        self.in_min = in_min
        self.in_max = in_max
        self.out_min = out_min
        self.out_max = out_max

    def __call__(self, t):
        return (t - self.in_min) / (self.in_max - self.in_min) * (self.out_max - self.out_min) + self.out_min


class ReRangeMinMaxTransform:
    """Re-ranges the input tensor so that its minimum and maximum values are mapped to out_min and out_max, respectively."""

    def __init__(self, out_min, out_max):
        """Initialize the ReRangeMinMaxTransform.

        Args:
            out_min: Minimum value of the output range.
            out_max: Maximum value of the output range.
        """
        self.out_min = out_min
        self.out_max = out_max

    def __call__(self, t):
        t_min = t.min()
        t_max = t.max()
        return (t - t_min) / (t_max - t_min) * (self.out_max - self.out_min) + self.out_min


class Stack3DTransform:
    """A transform that stacks a 2D image tensor along a new dimension to create a 3D tensor.

    Args:
        stack_n (int): The number of times to stack the input tensor along the new dimension.
    """

    def __init__(self, stack_n):
        """Initialize the Stack3DTransform.

        Args:
            stack_n (int): The number of times to stack the input tensor along the new dimension.
        """
        self.stack_n = stack_n

    def __call__(self, t):
        # # Assumes t is [B, C, H, W] or [C, H, W]
        # if t.dim() == 3:
        #     t = t.unsqueeze(1)  # [C, H, W] -> [C, 1, H, W]
        # elif t.dim() == 4:
        #     t = t.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]
        # return t.repeat_interleave(self.stack_n, dim=-3)  # Repeat along D

        # [B,C,H,W] -> [B,C,D,H,W] -> dim = 2
        return torch.stack([t] * self.stack_n, dim=2)


class ResizeMaxDimPadTransform:
    """Resizes an image tensor so its largest dimension matches 'max_dim' and pads the rest to make it square.

    Args:
        img (torch.Tensor): Input image tensor of shape (C, H, W).
        max_dim (int): The maximum dimension for resizing.
        interpol_mode (str): Interpolation mode for resizing (e.g., 'bilinear').
        mode (str): Padding mode ('constant', 'edge', 'replicate', or 'circular').
        value (int or float): Fill value for 'constant' padding.

    Returns:
        torch.Tensor: The resized and padded image tensor.
    """

    def __init__(self, max_dim, interpol_mode="bilinear", mode="constant", value=0):
        """Initialize the ResizeMaxDimPadTransform.

        Args:
            max_dim (int): The maximum dimension for resizing.
            interpol_mode (str): Interpolation mode for resizing (e.g., 'bilinear').
            mode (str): Padding mode ('constant', 'edge', 'replicate', or 'circular').
            value (int or float): Fill value for 'constant' padding.
        """
        self.max_dim = max_dim
        self.interpol_mode = interpol_mode
        self.mode = mode
        self.value = value

    def __call__(self, t):
        _, h, w = t.shape

        # Scale to max_dim
        scale = self.max_dim / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize
        t = F.interpolate(
            t.unsqueeze(0),
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

        return F.pad(t, (pad_left, pad_right, pad_top, pad_bottom), self.mode, self.value)


class PaddingTransform:
    """Add padding to an N-dimensional input tensor.

    Img = N-di m input tensor
    pad = padding values (tuple)
    mode = padding mode (constant, edge, replicate, circular)
        Default = 'constant'
    value =  fill value for 'constant' padding. Default: 0
    pad = (1,1) :        pad last dim by 1 on both sides
    pad = (1,1,2,2):     pad last dim by 1 on both sides,
                         pad second last dim by 2 on both sides
    pad = (1,1,2,2,3,3): pad last dim by 1 on both sides,
                         pad second last dim by 2 on both sides,
                         pad third last dim by 3 on both sides
    """

    def __init__(self, padding_size, padding_mode="constant", padding_value=0):
        """Initialize the PaddingTransform.

        Args:
            padding_size: Padding values (tuple).
            padding_mode (str): Padding mode ('constant', 'edge', 'replicate', or 'circular').
            padding_value: Fill value for 'constant' padding.
        """
        if padding_mode not in ["constant", "edge", "replicate", "circular"]:
            raise ValueError(f"Padding mode {padding_mode} not supported!")

        self.padding_size = padding_size
        self.padding_mode = padding_mode
        self.padding_value = padding_value

    def __call__(self, t):
        return F.pad(
            input=t,
            pad=self.padding_size,
            mode=self.padding_mode,
            value=self.padding_value,
        )


class UnPaddingTransform:
    """Removes padding from tensor.

    img = N-di m input tensor
    pad = padding values (tuple) - same format as used in add_padding
    """

    def __init__(self, padding_size):
        """Initialize the UnPaddingTransform.

        Args:
            padding_size: Padding values (tuple) - same format as used in PaddingTransform.
        """
        self.padding_size = padding_size

    def __call__(self, t):
        pad_spec = list(self.padding_size)
        if len(pad_spec) % 2 != 0:
            raise ValueError("Padding specification must have an even number of elements.")

        num_padded_dims = len(pad_spec) // 2
        if num_padded_dims > t.dim():
            raise ValueError(f"Padding specification for {num_padded_dims} dims exceeds tensor dims {t.dim()}")

        slices = [slice(None)] * t.dim()
        # PyTorch F.pad order: (pad_left, pad_right, pad_top, pad_bottom, ...), i.e. last dim first.
        for i in range(num_padded_dims):
            left = pad_spec[2 * i]
            right = pad_spec[2 * i + 1]
            dim = t.dim() - 1 - i  # map i-th pair to trailing dimension
            start = left if left > 0 else 0
            # If right == 0 we leave end as None to take full extent
            end = -right if right > 0 else None
            slices[dim] = slice(start, end)

        return t[tuple(slices)]


class Float32Transform:
    """A transform that converts the input tensor to torch.float32 dtype."""

    def __call__(self, t):
        # return t.to(dtype=t.new_empty(0).float().dtype)
        return t.type(torch.float32)


class Uint8Transform:
    """A transform that converts the input tensor to torch.uint8 dtype."""

    def __call__(self, t):
        # return t.to(dtype=t.new_empty(0).byte().dtype)
        return t.type(torch.uint8)


class Get2DFrom3DTransform:
    """Selects a 2D slice from a 3D tensor along the specified axis and index.

    Args:
        img (torch.Tensor): The input 3D tensor.
        axis (int): The axis along which to select the slice.
        index (int, optional): The index of the slice to select. Defaults to the middle slice.

    Returns:
        torch.Tensor: The selected 2D slice.
    """

    def __init__(self, axis, index):
        """Initialize the Get2DFrom3DTransform.

        Args:
            axis (int): The axis along which to select the slice.
            index (int): The index of the slice to select.
        """
        self.axis = axis
        self.index = index

    def __call__(self, t):
        if self.axis < 0 or self.axis >= t.dim():
            raise ValueError(f"Axis {self.axis} is out of bounds for the tensor with {t.dim()} dimensions.")
        # Recompute per call rather than caching on self.index: this transform instance
        # is built once and reused for every tensor passed through it, and different
        # calls can have different sizes along self.axis (e.g. a full-resolution volume
        # vs. a downsampled latent) - caching the first call's middle index would select
        # an out-of-range index on any later, smaller tensor.
        index = self.index if self.index is not None else t.shape[self.axis] // 2
        return t.select(dim=self.axis, index=index)


class SynchronizedRandomVerticalFlip:
    """Apply synchronized random vertical flip (upside down along the vertical axis) to multiple tensors."""

    def __init__(self, p=0.5, generator=None):
        """Initialize the SynchronizedRandomVerticalFlip transform.

        Args:
            p (float): Probability of applying the flip. Default is 0.5.
            generator (torch.Generator, optional): Random number generator for deterministic behavior.
                       Allows consistent random decision across distributed processes.
        """
        self.p = p
        self.generator = generator

    def __call__(self, *tensors):
        if torch.rand((), generator=self.generator).item() < self.p:
            return tuple(torch.flip(tensor, dims=[-2]) for tensor in tensors)
        return tensors

    def apply_transform(self, tensors):
        """Apply the vertical flip transform to tensors.

        Args:
            tensors: Input tensors to transform.

        Returns:
            Transformed tensors.
        """
        return self(*tensors)


class SynchronizedRandomHorizontalFlip:
    """Apply synchronized random horizontal flip (left-right along the horizontal axis) to multiple tensors."""

    def __init__(self, p=0.5, generator=None):
        """Initialize the SynchronizedRandomHorizontalFlip transform.

        Args:
            p (float): Probability of applying the flip. Default is 0.5.
            generator (torch.Generator, optional): Random number generator for deterministic behavior.
                       Allows consistent random decision across distributed processes.
        """
        self.p = p
        self.generator = generator

    def __call__(self, *tensors):
        if torch.rand((), generator=self.generator).item() < self.p:
            return tuple(torch.flip(tensor, dims=[-1]) for tensor in tensors)
        return tensors


class RandomGaussianNoiseTransform:
    """Add zero-mean Gaussian noise with a randomly sampled standard deviation.

    Useful to synthesize noisy inputs for training/evaluating denoising models. Operates
    elementwise, so the same transform applies to 2D images (e.g. (C, H, W)) and 3D
    volumes (e.g. (C, D, H, W)).
    """

    def __init__(self, sigma_min=0.0, sigma_max=0.1, p=1.0, generator=None):
        """Initialize the RandomGaussianNoiseTransform.

        Args:
            sigma_min (float): Minimum standard deviation of the noise to sample from.
            sigma_max (float): Maximum standard deviation of the noise to sample from.
            p (float): Probability of applying the transform. Default is 1.0.
            generator (torch.Generator, optional): Random number generator for deterministic behavior.
        """
        if sigma_min < 0 or sigma_max < sigma_min:
            raise ValueError(f"Invalid sigma range [{sigma_min}, {sigma_max}].")
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.p = p
        self.generator = generator

    def __call__(self, t):
        if torch.rand((), generator=self.generator).item() >= self.p:
            return t

        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.rand((), generator=self.generator).item()
        noise = torch.randn(t.shape, generator=self.generator).to(dtype=t.dtype, device=t.device)
        return t + noise * sigma


class RandomPoissonNoiseTransform:
    """Add signal-dependent Poisson (shot) noise, as seen in low-dose CT / photon-limited imaging.

    The input is scaled up to a randomly sampled level, Poisson noise is drawn at that
    scale, and the result is scaled back down. Operates elementwise, so the same
    transform applies to 2D images and 3D volumes. Assumes non-negative input intensities
    (negative values are clamped to 0 before noise is drawn).
    """

    def __init__(self, scale_min=0.5, scale_max=1.0, p=1.0, generator=None):
        """Initialize the RandomPoissonNoiseTransform.

        Args:
            scale_min (float): Minimum scale factor applied before drawing Poisson noise.
                Lower scale means relatively more noise. Must be > 0.
            scale_max (float): Maximum scale factor applied before drawing Poisson noise.
            p (float): Probability of applying the transform. Default is 1.0.
            generator (torch.Generator, optional): Random number generator for deterministic behavior.
        """
        if scale_min <= 0 or scale_max < scale_min:
            raise ValueError(f"Invalid scale range [{scale_min}, {scale_max}].")
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.p = p
        self.generator = generator

    def __call__(self, t):
        if torch.rand((), generator=self.generator).item() >= self.p:
            return t

        scale = self.scale_min + (self.scale_max - self.scale_min) * torch.rand((), generator=self.generator).item()
        rate = t.clamp(min=0).cpu() * scale
        noisy = torch.poisson(rate, generator=self.generator) / scale
        return noisy.to(dtype=t.dtype, device=t.device)


class RandomSaltPepperNoiseTransform:
    """Randomly overwrite a fraction of elements with extreme 'salt' and 'pepper' values.

    Simulates impulse noise / dead-pixel artifacts. Operates elementwise, so the same
    transform applies to 2D images and 3D volumes.
    """

    def __init__(self, amount=0.02, salt_vs_pepper=0.5, salt_value=1.0, pepper_value=0.0, p=1.0, generator=None):
        """Initialize the RandomSaltPepperNoiseTransform.

        Args:
            amount (float): Fraction of elements to corrupt, in [0, 1].
            salt_vs_pepper (float): Fraction of corrupted elements set to `salt_value`
                (the remainder are set to `pepper_value`), in [0, 1].
            salt_value (float): Value written for 'salt' elements.
            pepper_value (float): Value written for 'pepper' elements.
            p (float): Probability of applying the transform. Default is 1.0.
            generator (torch.Generator, optional): Random number generator for deterministic behavior.
        """
        if not 0.0 <= amount <= 1.0:
            raise ValueError(f"amount must be in [0, 1], got {amount}.")
        if not 0.0 <= salt_vs_pepper <= 1.0:
            raise ValueError(f"salt_vs_pepper must be in [0, 1], got {salt_vs_pepper}.")
        self.amount = amount
        self.salt_vs_pepper = salt_vs_pepper
        self.salt_value = salt_value
        self.pepper_value = pepper_value
        self.p = p
        self.generator = generator

    def __call__(self, t):
        if torch.rand((), generator=self.generator).item() >= self.p:
            return t

        corrupted = torch.rand(t.shape, generator=self.generator) < self.amount
        salt = corrupted & (torch.rand(t.shape, generator=self.generator) < self.salt_vs_pepper)
        pepper = corrupted & ~salt

        out = t.clone()
        out[salt.to(out.device)] = self.salt_value
        out[pepper.to(out.device)] = self.pepper_value
        return out


class RandomCutoutTransform:
    """Overwrite one or more randomly placed, randomly sized hyper-rectangular regions.

    Covers parts of an image (or volume) with a constant fill value, which is useful for
    forcing a denoising model to reconstruct missing/occluded content. Operates on the
    trailing `n_spatial_dims` dimensions of the tensor, so the same transform works on 2D
    images shaped (..., H, W) with `n_spatial_dims=2` and on 3D volumes shaped
    (..., D, H, W) with `n_spatial_dims=3`.
    """

    def __init__(
        self, n_spatial_dims=2, num_patches=1, min_frac=0.1, max_frac=0.5, fill_value=0.0, p=0.5, generator=None
    ):
        """Initialize the RandomCutoutTransform.

        Args:
            n_spatial_dims (int): Number of trailing tensor dimensions considered spatial
                (2 for images (..., H, W), 3 for volumes (..., D, H, W)).
            num_patches (int): Number of independent patches to cut out.
            min_frac (float): Minimum patch size, as a fraction of each spatial dimension.
            max_frac (float): Maximum patch size, as a fraction of each spatial dimension.
            fill_value (float): Value written into the cutout region(s).
            p (float): Probability of applying the transform. Default is 0.5.
            generator (torch.Generator, optional): Random number generator for deterministic behavior.
        """
        if n_spatial_dims < 1:
            raise ValueError(f"n_spatial_dims must be >= 1, got {n_spatial_dims}.")
        if not 0.0 < min_frac <= max_frac <= 1.0:
            raise ValueError(f"Invalid fraction range [{min_frac}, {max_frac}].")
        self.n_spatial_dims = n_spatial_dims
        self.num_patches = num_patches
        self.min_frac = min_frac
        self.max_frac = max_frac
        self.fill_value = fill_value
        self.p = p
        self.generator = generator

    def __call__(self, t):
        if t.dim() < self.n_spatial_dims:
            raise ValueError(f"Tensor with {t.dim()} dims is smaller than n_spatial_dims={self.n_spatial_dims}.")
        if torch.rand((), generator=self.generator).item() >= self.p:
            return t

        out = t.clone()
        spatial_shape = t.shape[-self.n_spatial_dims :]
        leading_dims = t.dim() - self.n_spatial_dims

        for _ in range(self.num_patches):
            slices = [slice(None)] * leading_dims
            for dim_size in spatial_shape:
                frac = self.min_frac + (self.max_frac - self.min_frac) * torch.rand((), generator=self.generator).item()
                size = max(1, min(dim_size, int(round(dim_size * frac))))
                start = torch.randint(0, dim_size - size + 1, (), generator=self.generator).item()
                slices.append(slice(start, start + size))
            out[tuple(slices)] = self.fill_value

        return out


class RandomBrightnessContrastTransform:
    """Randomly jitter brightness (additive) and contrast (multiplicative around the mean).

    Standard photometric augmentation. Operates elementwise, so it applies identically to
    2D images and 3D volumes.
    """

    def __init__(self, brightness=0.2, contrast=0.2, p=0.5, generator=None):
        """Initialize the RandomBrightnessContrastTransform.

        Args:
            brightness (float): Maximum absolute brightness shift, sampled uniformly from
                [-brightness, brightness].
            contrast (float): Maximum relative contrast change, sampled uniformly from
                [1 - contrast, 1 + contrast].
            p (float): Probability of applying the transform. Default is 0.5.
            generator (torch.Generator, optional): Random number generator for deterministic behavior.
        """
        if brightness < 0 or contrast < 0:
            raise ValueError("brightness and contrast must be non-negative.")
        self.brightness = brightness
        self.contrast = contrast
        self.p = p
        self.generator = generator

    def __call__(self, t):
        if torch.rand((), generator=self.generator).item() >= self.p:
            return t

        brightness_delta = (torch.rand((), generator=self.generator).item() * 2 - 1) * self.brightness
        contrast_factor = 1.0 + (torch.rand((), generator=self.generator).item() * 2 - 1) * self.contrast

        mean = t.mean()
        return (t - mean) * contrast_factor + mean + brightness_delta


class RandomGammaTransform:
    """Apply a randomly sampled gamma correction: sign(t) * |t| ** gamma.

    Simulates non-linear intensity variations (e.g. scanner/detector response). Operates
    elementwise, so it applies identically to 2D images and 3D volumes.
    """

    def __init__(self, gamma_min=0.7, gamma_max=1.5, p=0.5, generator=None):
        """Initialize the RandomGammaTransform.

        Args:
            gamma_min (float): Minimum gamma value to sample from. Must be > 0.
            gamma_max (float): Maximum gamma value to sample from. Must be >= gamma_min.
            p (float): Probability of applying the transform. Default is 0.5.
            generator (torch.Generator, optional): Random number generator for deterministic behavior.
        """
        if gamma_min <= 0 or gamma_max < gamma_min:
            raise ValueError(f"Invalid gamma range [{gamma_min}, {gamma_max}].")
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.p = p
        self.generator = generator

    def __call__(self, t):
        if torch.rand((), generator=self.generator).item() >= self.p:
            return t

        gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * torch.rand((), generator=self.generator).item()
        return t.sign() * t.abs().pow(gamma)


def get_transformers(t_defs: Any) -> transforms.Compose:
    """Compose a sequence of transformers specified by their names or configuration dictionaries.

    Args:
        t_defs (list): List of transformer names or configuration dictionaries.

    Returns:
        torchvision.transforms.Compose: Composed transformer.
    """
    return transforms.Compose([get_transformer(t) for t in t_defs])


def get_transformer_by_names(transformer_name: str, parameters: dict[str, Any] | None, t_defs: Any) -> Callable:
    """Returns a torchvision transformer instance based on the given transformer name and parameters.

    Args:
        transformer_name (str): The name of the transformer to create.
        parameters (dict): Dictionary of parameters required for the transformer.
        t_defs: The original transformer definition (used for error messages).

    Returns:
        torchvision.transforms.Transform: The corresponding transformer instance.
    """
    if transformer_name == "Stack3D":
        transformer = Stack3DTransform(parameters["stack_n"])

    elif transformer_name == "Resize_HW":
        transformer = transforms.Resize((parameters.get("h"), parameters.get("w")), antialias=True)

    elif transformer_name == "ResizeMaxDimPad":
        transformer = ResizeMaxDimPadTransform(
            parameters["max_dim"],
            parameters.get("interpol_mode", "bilinear"),
            parameters.get("mode", "constant"),
            parameters.get("value", 0),
        )

    elif transformer_name == "CLAMP_abs":
        transformer = ClampAbsTransform(parameters["lower"], parameters["upper"])

    elif transformer_name == "CLAMP_perc":
        transformer = ClampPercTransform(parameters["lower_perc"], parameters["upper_perc"])

    elif transformer_name == "ReRange":
        transformer = ReRangeTransform(
            parameters["in_min"],
            parameters["in_max"],
            parameters["out_min"],
            parameters["out_max"],
        )

    elif transformer_name == "ReRange_minmax":
        transformer = ReRangeMinMaxTransform(parameters["out_min"], parameters["out_max"])

    elif transformer_name == "Gaussian_Blur":
        transformer = transforms.GaussianBlur(
            kernel_size=parameters.get("blur_kernel_size"),
            sigma=parameters.get("blur_sigma"),
        )

    elif transformer_name == "Padding":
        transformer = PaddingTransform(
            parameters["padding_size"],
            parameters.get("padding_mode", "constant"),
            parameters.get("padding_value", 0),
        )

    elif transformer_name == "UnPadding":
        transformer = UnPaddingTransform(parameters["padding_size"])

    elif transformer_name == "Get2DFrom3D":
        transformer = Get2DFrom3DTransform(parameters.get("axis", 0), parameters.get("index"))

    elif transformer_name == "ADD":
        transformer = AddValueTransform(parameters["value"])

    elif transformer_name == "DIV":
        transformer = DivValueTransform(parameters["value"])

    elif transformer_name == "NORM":
        transformer = transforms.Normalize(mean=(parameters.get("mean"),), std=(parameters.get("stdev"),))

    elif transformer_name == "MULT":
        transformer = MultValueTransform(parameters["value"])

    elif transformer_name == "RandomAffine":
        degrees = parameters.get("degrees", 0)
        translate = parameters.get("translate", (0, 0))
        scale = parameters.get("scale")
        shear = parameters.get("shear")
        fill = (parameters.get("fill", 0),)
        center = parameters.get("center")
        transformer = transforms.RandomAffine(
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            center=center,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=fill,
        )

    elif transformer_name == "RandomHorizontalFlip":
        transformer = transforms.RandomHorizontalFlip(p=0.5 if parameters is None else parameters.get("p", 0.5))

    elif transformer_name == "SynchronizedRandomHorizontalFlip":
        transformer = SynchronizedRandomHorizontalFlip(p=0.5 if parameters is None else parameters.get("p", 0.5))

    elif transformer_name == "RandomVerticalFlip":
        transformer = transforms.RandomVerticalFlip(p=0.5 if parameters is None else parameters.get("p", 0.5))

    elif transformer_name == "SynchronizedRandomVerticalFlip":
        transformer = SynchronizedRandomVerticalFlip(p=0.5 if parameters is None else parameters.get("p", 0.5))

    elif transformer_name == "RandomGaussianNoise":
        params = parameters or {}
        transformer = RandomGaussianNoiseTransform(
            sigma_min=params.get("sigma_min", 0.0),
            sigma_max=params.get("sigma_max", 0.1),
            p=params.get("p", 1.0),
        )

    elif transformer_name == "RandomPoissonNoise":
        params = parameters or {}
        transformer = RandomPoissonNoiseTransform(
            scale_min=params.get("scale_min", 0.5),
            scale_max=params.get("scale_max", 1.0),
            p=params.get("p", 1.0),
        )

    elif transformer_name == "RandomSaltPepperNoise":
        params = parameters or {}
        transformer = RandomSaltPepperNoiseTransform(
            amount=params.get("amount", 0.02),
            salt_vs_pepper=params.get("salt_vs_pepper", 0.5),
            salt_value=params.get("salt_value", 1.0),
            pepper_value=params.get("pepper_value", 0.0),
            p=params.get("p", 1.0),
        )

    elif transformer_name == "RandomCutout":
        params = parameters or {}
        transformer = RandomCutoutTransform(
            n_spatial_dims=params.get("n_spatial_dims", 2),
            num_patches=params.get("num_patches", 1),
            min_frac=params.get("min_frac", 0.1),
            max_frac=params.get("max_frac", 0.5),
            fill_value=params.get("fill_value", 0.0),
            p=params.get("p", 0.5),
        )

    elif transformer_name == "RandomBrightnessContrast":
        params = parameters or {}
        transformer = RandomBrightnessContrastTransform(
            brightness=params.get("brightness", 0.2),
            contrast=params.get("contrast", 0.2),
            p=params.get("p", 0.5),
        )

    elif transformer_name == "RandomGamma":
        params = parameters or {}
        transformer = RandomGammaTransform(
            gamma_min=params.get("gamma_min", 0.7),
            gamma_max=params.get("gamma_max", 1.5),
            p=params.get("p", 0.5),
        )

    elif transformer_name == "ToTensor":
        # transformer = transforms.ToTensor() # deprecated
        transformer = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])

    elif transformer_name == "Float32":
        transformer = Float32Transform()

    elif transformer_name == "Uint8":
        transformer = Uint8Transform()

    elif transformer_name == "RGB_Normalize":
        transformer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    elif transformer_name == "RGB2GRAY":
        transformer = transforms.Grayscale(num_output_channels=1)

    elif transformer_name == "ToPil":
        transformer = transforms.ToPILImage()

    elif transformer_name == "Squeeze":
        transformer = torch.squeeze

    elif transformer_name == "LOG":
        transformer = torch.log

    elif transformer_name == "EXP":
        transformer = torch.exp

    elif transformer_name == "FLOOR":
        transformer = torch.floor

    elif transformer_name == "NOP":

        def nop_transform(t):
            return t

        transformer = nop_transform

    else:
        raise ValueError(f"Transformation {t_defs} is not supported!")

    return transformer


def get_transformer(t_defs: Any) -> Callable:
    """Tensor Transformers for image processing.

    Stack3D
    Morphs a 2D image to 3D by stacking the image along a new D dimension
    [B,C,H,W] -> [B,C,D,H,W]
    where d is set by the parameter 'stack_n'

    Resize_HW:
    Resizes images to Resize_IMG_SIZE_H x Resize_IMG_SIZE_W (must be defined in settings file)

    ResizeMaxDimPad:
    Resizes an image so its largest dimension matches 'max_dim' and pads the rest to make it square.
    - max_dim (int): Maximum dimension for the resized image.
    - interpol_mode (str): Interpolation mode for resizing. Default is 'bilinear'.
    - mode (str): Padding mode, can be 'constant', 'edge', 'replicate', or 'circular'. Default is 'constant'.
    - value (int): Fill value for 'constant' padding. Default is 0.

    CLAMP_abs:
    Clamps input tensor to [lower,upper]
    (clamp = clip)

    CLAMP_perc:
    Clamps input tensor by percentiles [lower_perc,upper_perc]

    ReRange:
    Re-ranges input tensor from [in_min, in_max] to [out_min, out_max]

    ReRange_minmax:
    Re-ranges input tensor to [out_min, out_max]

    Gaussian_Blur:
    Blurs the image using the parameters 'blur_kernel_size' and 'blur_sigma'.

    Padding:
    Pads the input tensor using the parameters
    - padding_size (tuple)
    - padding_mode (Default = 'constant')
    - padding_value (Default = 0)

    UnPadding:
    Removes padding from the input tensor (e.g. to compute metrics on original image size)

    Get2DFrom3D:
    Selects a 2D slice from a 3D tensor along the specified axis and index.

    ADD:
    Adds the input tensor by the value specified in the parameter 'value'.

    DIV:
    Divides the input tensor by the value specified in the parameter 'value'.

    MULT:
    Multiplies the input tensor by the value specified in the parameter 'value'.

    RandomAffine:
    Applies a random affine transformation to the input tensor with specified parameters
    https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.RandomAffine.html
    - degrees (float or int): Range of degrees to select from for rotation.
    - translate (tuple): Tuple of maximum absolute fraction for horizontal and vertical translations.
    - scale (tuple): Tuple of minimum and maximum scaling factors.
    - shear (float): Shear angle in degrees.
    - fill (float): Fill color for the area outside the transformed image.
    - center (tuple): Center of rotation. If None, the center of the image is used.


    NORM:
    Normalize a tensor image with 'mean' and 'stdev'.


    RandomHorizontalFlip
    Horizontally flip the input with probability p (default=0.5).

    RandomVerticalFlip
    Vertically flip the input with probability p (default=0.5).

    RandomGaussianNoise:
    Adds zero-mean Gaussian noise with a standard deviation sampled uniformly from
    ['sigma_min', 'sigma_max'], applied with probability 'p'. Works on 2D images and
    3D volumes alike (elementwise).
    - sigma_min (float): Default 0.0.
    - sigma_max (float): Default 0.1.
    - p (float): Default 1.0.

    RandomPoissonNoise:
    Adds signal-dependent Poisson (shot) noise, as seen in low-dose CT / photon-limited
    imaging, applied with probability 'p'. Assumes non-negative input intensities.
    - scale_min (float): Default 0.5.
    - scale_max (float): Default 1.0.
    - p (float): Default 1.0.

    RandomSaltPepperNoise:
    Randomly overwrites a fraction 'amount' of elements with 'salt_value' or
    'pepper_value' (impulse noise / dead-pixel artifacts), applied with probability 'p'.
    - amount (float): Fraction of elements corrupted. Default 0.02.
    - salt_vs_pepper (float): Fraction of corrupted elements set to 'salt_value'. Default 0.5.
    - salt_value (float): Default 1.0.
    - pepper_value (float): Default 0.0.
    - p (float): Default 1.0.

    RandomCutout:
    Overwrites one or more randomly placed, randomly sized hyper-rectangular regions
    with 'fill_value', applied with probability 'p'. Operates on the trailing
    'n_spatial_dims' dimensions, so the same transform works on 2D images
    (n_spatial_dims=2) and 3D volumes (n_spatial_dims=3).
    - n_spatial_dims (int): Default 2.
    - num_patches (int): Default 1.
    - min_frac (float): Minimum patch size as a fraction of each spatial dim. Default 0.1.
    - max_frac (float): Maximum patch size as a fraction of each spatial dim. Default 0.5.
    - fill_value (float): Default 0.0.
    - p (float): Default 0.5.

    RandomBrightnessContrast:
    Randomly jitters brightness (additive) and contrast (multiplicative around the
    mean), applied with probability 'p'.
    - brightness (float): Max absolute shift, sampled from [-brightness, brightness]. Default 0.2.
    - contrast (float): Max relative change, sampled from [1-contrast, 1+contrast]. Default 0.2.
    - p (float): Default 0.5.

    RandomGamma:
    Applies a randomly sampled gamma correction sign(t) * |t| ** gamma, applied with
    probability 'p'.
    - gamma_min (float): Default 0.7.
    - gamma_max (float): Default 1.5.
    - p (float): Default 0.5.

    ToTensor::
    https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a
    torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] if the PIL Image
    belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the
    numpy.ndarray has dtype = np.uint8

    Float32:
    Converts input tenor to Float32

    Uint8:
    Converts input tenor to Uint8

    RGB_Normalize:
    Channel-wise normalize RGB images in the range[0,1] according to mean/stdev of imageNET.

    RGB2GRAY:
    Converts RGB image to grayscale image.

    ToPil:
    Converts a tensor to a PIL image.

    Squeeze:
    Squeezes the input tensor.

    LOG:
    Applies the natural logarithm to the input tensor.

    EXP:
    Applies the exponential function to the input tensor.

    FLOOR:
    Applies the floor function to the input tensor.

    NOP:
    Does nothing, just returns the input tensor.
    """
    all_required_params = {
        "Stack3D": {"stack_n": [int]},
        "Resize_HW": {"h": [int], "w": [int]},
        "ResizeMaxDimPad": {"max_dim": [int]},
        "CLAMP_abs": {"lower": [float, int], "upper": [float, int]},
        "CLAMP_perc": {"lower_perc": [float], "upper_perc": [float]},
        "ReRange": {
            "in_min": [float, int],
            "in_max": [float, int],
            "out_min": [float, int],
            "out_max": [float, int],
        },
        "ReRange_minmax": {"out_min": [float, int], "out_max": [float, int]},
        "Gaussian_Blur": {"blur_kernel_size": [float, int], "blur_sigma": [float, int]},
        "Padding": {
            "padding_size": [list, int],
            "padding_mode": [str],
            "padding_value": [int],
        },
        "UnPadding": {"padding_size": [list, int]},
        "Get2DFrom3D": {"axis": [int]},
        "ADD": {"value": [float, int]},
        "MULT": {"value": [float, int]},
        "RandomAffine": {"degrees": [float, int]},
        "DIV": {"value": [float, int]},
        "NORM": {"mean": [float], "stdev": [float]},
    }

    if isinstance(t_defs, DictConfig):
        keys = list(t_defs.keys())
        if len(keys) != 1:
            raise ValueError(f"Transformation {t_defs} does not correspond to the expected format!")
        transformer_name = keys[0]
        parameters = t_defs[transformer_name]
    elif isinstance(t_defs, str):
        transformer_name = t_defs
        parameters = None
    else:
        raise ValueError(f"Transformation {t_defs} does not correspond to the expected format!")

    # check if all required parameters are present and datatypes are correct
    for req_param, req_type in all_required_params.get(transformer_name, {}).items():
        if req_param not in parameters:
            raise ValueError(f"Parameter {req_param} is missing for transformation {transformer_name}.")
        if type(parameters[req_param]) not in req_type:
            raise ValueError(f"Parameter {req_param} for transformation {transformer_name} is not correct.")

    transformer = get_transformer_by_names(transformer_name, parameters, t_defs)

    return transformer
