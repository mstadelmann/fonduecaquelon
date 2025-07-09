from typing import Any
from collections.abc import Callable
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms


class AddValueTransform:
    def __init__(self, value):
        self.value = value

    def __call__(self, t):
        return t + self.value


class MultValueTransform:
    def __init__(self, value):
        self.value = value

    def __call__(self, t):
        return t * self.value


class DivValueTransform:
    def __init__(self, value):
        self.value = value

    def __call__(self, t):
        return t / self.value


class ClampAbsTransform:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def __call__(self, t):
        return t.clamp(self.lower, self.upper)


class ClampPercTransform:
    def __init__(self, lower_perc, upper_perc):
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
            return tensor.view(-1)[random_indices]
        return tensor


class ReRangeTransform:
    def __init__(self, in_min, in_max, out_min, out_max):
        self.in_min = in_min
        self.in_max = in_max
        self.out_min = out_min
        self.out_max = out_max

    def __call__(self, t):
        return (t - self.in_min) / (self.in_max - self.in_min) * (
            self.out_max - self.out_min
        ) + self.out_min


class ReRangeMinMaxTransform:
    def __init__(self, out_min, out_max):
        self.out_min = out_min
        self.out_max = out_max

    def __call__(self, t):
        t_min = t.min()
        t_max = t.max()
        return (t - t_min) / (t_max - t_min) * (
            self.out_max - self.out_min
        ) + self.out_min


class Stack3DTransform:
    def __init__(self, stack_n):
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

        return F.pad(
            t, (pad_left, pad_right, pad_top, pad_bottom), self.mode, self.value
        )


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
        self.padding_size = padding_size

    def __call__(self, t):
        req_pad_dim = t.dim() * 2
        pad = self.padding_size + [0] * (req_pad_dim - len(self.padding_size))
        ps_start = pad[::2]
        ps_stop = pad[1::2]

        for dim in range(t.dim()):
            if ps_stop[dim] == 0:
                ps_stop[dim] = t.shape[dim]
            else:
                ps_stop[dim] = t.shape[dim] - ps_stop[dim]

        if t.dim() == 4:
            return t[
                ps_start[0] : ps_stop[0],
                ps_start[1] : ps_stop[1],
                ps_start[2] : ps_stop[2],
                ps_start[3] : ps_stop[3],
            ]
        if t.dim() == 5:
            return t[
                ps_start[0] : ps_stop[0],
                ps_start[1] : ps_stop[1],
                ps_start[2] : ps_stop[2],
                ps_start[3] : ps_stop[3],
                ps_start[4] : ps_stop[4],
            ]
        raise ValueError("Only 4D and 5D tensors are supported!")


class Float32Transform:
    def __call__(self, t):
        # return t.to(dtype=t.new_empty(0).float().dtype)
        return t.type(torch.float32)


class Uint8Transform:
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
        self.axis = axis
        self.index = index

    def __call__(self, t):
        if self.index is None:
            self.index = t.shape[self.axis] // 2  # Default to the middle slice
        if self.axis < 0 or self.axis >= t.dim():
            raise ValueError(
                f"Axis {self.axis} is out of bounds for the tensor with {t.dim()} dimensions."
            )
        return t.select(dim=self.axis, index=self.index)


def get_transformers(t_defs: Any) -> transforms.Compose:
    """Compose a sequence of transformers specified by their names or configuration dictionaries.

    Args:
        t_defs (list): List of transformer names or configuration dictionaries.

    Returns:
        torchvision.transforms.Compose: Composed transformer.
    """
    return transforms.Compose([get_transformer(t) for t in t_defs])


def get_transformer_by_names(
    transformer_name: str, parameters: dict[str, Any] | None, t_defs: Any
) -> Callable:
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
        transformer = transforms.Resize(
            (parameters.get("h"), parameters.get("w")), antialias=True
        )

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
        transformer = ClampPercTransform(
            parameters["lower_perc"], parameters["upper_perc"]
        )

    elif transformer_name == "ReRange":
        transformer = ReRangeTransform(
            parameters["in_min"],
            parameters["in_max"],
            parameters["out_min"],
            parameters["out_max"],
        )

    elif transformer_name == "ReRange_minmax":
        transformer = ReRangeMinMaxTransform(
            parameters["out_min"], parameters["out_max"]
        )

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
        transformer = Get2DFrom3DTransform(
            parameters.get("axis", 0), parameters.get("index")
        )

    elif transformer_name == "ADD":
        transformer = AddValueTransform(parameters["value"])

    elif transformer_name == "DIV":
        transformer = DivValueTransform(parameters["value"])

    elif transformer_name == "NORM":
        transformer = transforms.Normalize(
            mean=(parameters.get("mean"),), std=(parameters.get("stdev"),)
        )

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
        transformer = transforms.RandomHorizontalFlip(
            p=0.5 if parameters is None else parameters.get("p", 0.5)
        )

    elif transformer_name == "RandomVerticalFlip":
        transformer = transforms.RandomVerticalFlip(
            p=0.5 if parameters is None else parameters.get("p", 0.5)
        )

    elif transformer_name == "ToTensor":
        transformer = transforms.ToTensor()

    elif transformer_name == "Float32":
        transformer = Float32Transform()

    elif transformer_name == "Uint8":
        transformer = Uint8Transform()

    elif transformer_name == "RGB_Normalize":
        transformer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

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
        transformer = lambda t: t

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

    if isinstance(t_defs, dict):
        keys = list(t_defs.keys())
        if len(keys) != 1:
            raise ValueError(
                f"Transformation {t_defs} does not correspond to the expected format!"
            )
        transformer_name = keys[0]
        parameters = t_defs[transformer_name]
    elif isinstance(t_defs, str):
        transformer_name = t_defs
        parameters = None
    else:
        raise ValueError(
            f"Transformation {t_defs} does not correspond to the expected format!"
        )

    # check if all required parameters are present and datatypes are correct
    for req_param, req_type in all_required_params.get(transformer_name, {}).items():
        if req_param not in parameters:
            raise ValueError(
                f"Parameter {req_param} is missing for transformation {transformer_name}."
            )
        if type(parameters[req_param]) not in req_type:
            raise ValueError(
                f"Parameter {req_param} for transformation {transformer_name} is not correct."
            )

    transformer = get_transformer_by_names(transformer_name, parameters, t_defs)

    return transformer
