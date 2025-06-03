from typing import Any
from collections.abc import Callable
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms


def resize_max_dim_pad(
    img: torch.Tensor,
    max_dim: int,
    interpol_mode: str,
    mode: str,
    value: int | float,
) -> torch.Tensor:
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
    _, h, w = img.shape

    # Scale to max_dim
    scale = max_dim / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize
    img = F.interpolate(
        img.unsqueeze(0),
        size=(new_h, new_w),
        mode=interpol_mode,
        # align_corners=False,
    ).squeeze(0)

    # Padding
    pad_h = max_dim - new_h
    pad_w = max_dim - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode, value)

    return img


def select_2d_from_3d(
    img: torch.Tensor, axis: int, index: int | None = None
) -> torch.Tensor:
    """Selects a 2D slice from a 3D tensor along the specified axis and index.

    Args:
        img (torch.Tensor): The input 3D tensor.
        axis (int): The axis along which to select the slice.
        index (int, optional): The index of the slice to select. Defaults to the middle slice.

    Returns:
        torch.Tensor: The selected 2D slice.
    """
    if index is None:
        index = img.shape[axis] // 2  # Default to the middle slice
    if axis < 0 or axis >= img.dim():
        raise ValueError(
            f"Axis {axis} is out of bounds for the tensor with {img.dim()} dimensions."
        )
    return img.select(dim=axis, index=index)


def add_padding(
    img: torch.Tensor,
    pad: list | tuple,
    mode: str = "constant",
    value: int | float = 0,
) -> torch.Tensor:
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
    if mode not in ["constant", "edge", "replicate", "circular"]:
        raise ValueError(f"Padding mode {mode} not supported!")

    return torch.nn.functional.pad(input=img, pad=pad, mode=mode, value=value)


def remove_padding(img: torch.Tensor, pad: list) -> torch.Tensor:
    """Removes padding from tensor.

    img = N-di m input tensor
    pad = padding values (tuple) - same format as used in add_padding
    """
    req_pad_dim = img.dim() * 2
    pad = pad + [0] * (req_pad_dim - len(pad))
    ps_start = pad[::2]
    ps_stop = pad[1::2]

    for dim in range(img.dim()):
        if ps_stop[dim] == 0:
            ps_stop[dim] = img.shape[dim]
        else:
            ps_stop[dim] = img.shape[dim] - ps_stop[dim]

    if img.dim() == 4:
        return img[
            ps_start[0] : ps_stop[0],
            ps_start[1] : ps_stop[1],
            ps_start[2] : ps_stop[2],
            ps_start[3] : ps_stop[3],
        ]
    if img.dim() == 5:
        return img[
            ps_start[0] : ps_stop[0],
            ps_start[1] : ps_stop[1],
            ps_start[2] : ps_stop[2],
            ps_start[3] : ps_stop[3],
            ps_start[4] : ps_stop[4],
        ]
    raise ValueError("Only 4D and 5D tensors are supported!")


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
        stack_n = parameters.get("stack_n")
        # [B,C,H,W] -> [B,C,D,H,W] -> dim = 2
        transformer = transforms.Lambda(lambda t: torch.stack([t] * stack_n, dim=2))

    elif transformer_name == "Resize_HW":
        transformer = transforms.Resize(
            (parameters.get("h"), parameters.get("w")), antialias=True
        )

    elif transformer_name == "ResizeMaxDimPad":
        transformer = transforms.Lambda(
            lambda t: resize_max_dim_pad(
                t,
                max_dim=parameters.get("max_dim"),
                interpol_mode=parameters.get("interpol_mode", "bilinear"),
                mode=parameters.get("mode", "constant"),
                value=parameters.get("value", 0),
            )
        )

    elif transformer_name == "CLAMP_abs":
        transformer = transforms.Lambda(
            lambda t: torch.clamp(t, parameters.get("lower"), parameters.get("upper"))
        )

    elif transformer_name == "CLAMP_perc":

        def _rdm_reduce(tensor: torch.Tensor) -> torch.Tensor:
            # random reduce tensor size for quantile computation -> estimation
            n = 12582912  # random defined as 8*3*32*128*128
            if tensor.numel() > n:
                n = min(n, tensor.numel())
                random_indices = torch.randperm(tensor.numel())[:n]
                return tensor.view(-1)[random_indices]
            return tensor

        transformer = transforms.Lambda(
            lambda t: torch.clamp(
                t,
                torch.quantile(_rdm_reduce(t), parameters.get("lower_perc")),
                torch.quantile(_rdm_reduce(t), parameters.get("upper_perc")),
            )
        )

    elif transformer_name == "ReRange":
        in_min = parameters.get("in_min")
        in_max = parameters.get("in_max")
        out_min = parameters.get("out_min")
        out_max = parameters.get("out_max")
        transformer = transforms.Lambda(
            lambda t: ((t - in_min) * (out_max - out_min)) / (in_max - in_min) + out_min
        )

    elif transformer_name == "ReRange_minmax":
        out_min = parameters.get("out_min")
        out_max = parameters.get("out_max")
        transformer = transforms.Lambda(
            lambda t: ((t - t.min()) * (out_max - out_min)) / (t.max() - t.min())
            + out_min
        )

    elif transformer_name == "Gaussian_Blur":
        transformer = transforms.GaussianBlur(
            kernel_size=parameters.get("blur_kernel_size"),
            sigma=parameters.get("blur_sigma"),
        )

    elif transformer_name == "Padding":
        pad = parameters.get("padding_size")
        if pad is None:
            raise ValueError("padding_size must be provided for Padding transformer.")
        transformer = transforms.Lambda(
            lambda t: add_padding(
                t,
                pad=pad,
                mode=parameters.get("padding_mode", "constant"),
                value=parameters.get("padding_value", 0),
            )
        )

    elif transformer_name == "UnPadding":
        pad = parameters.get("padding_size")
        if not isinstance(pad, list):
            raise ValueError(
                "padding_size must be provided as a list for UnPadding transformer."
            )
        transformer = transforms.Lambda(lambda t: remove_padding(t, pad=pad))

    elif transformer_name == "Get2DFrom3D":
        axis = parameters.get("axis", 0)
        index = parameters.get("index")
        transformer = transforms.Lambda(
            lambda t: select_2d_from_3d(t, axis=axis, index=index)
        )

    elif transformer_name == "ADD":
        transformer = transforms.Lambda(lambda t: t + parameters.get("value"))

    elif transformer_name == "DIV":
        transformer = transforms.Lambda(lambda t: t / parameters.get("value"))

    elif transformer_name == "NORM":
        transformer = transforms.Normalize(
            mean=(parameters.get("mean"),), std=(parameters.get("stdev"),)
        )

    elif transformer_name == "MULT":
        transformer = transforms.Lambda(lambda t: t * parameters.get("value"))

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
        transformer = transforms.RandomHorizontalFlip(p=parameters.get("p", 0.5))

    elif transformer_name == "RandomVerticalFlip":
        transformer = transforms.RandomVerticalFlip(p=parameters.get("p", 0.5))

    elif transformer_name == "ToTensor":
        transformer = transforms.ToTensor()

    elif transformer_name == "Float32":
        transformer = transforms.Lambda(lambda t: t.type(torch.float32))

    elif transformer_name == "Uint8":
        transformer = transforms.Lambda(lambda t: t.type(torch.uint8))

    elif transformer_name == "RGB_Normalize":
        transformer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    elif transformer_name == "RGB2GRAY":
        transformer = transforms.Grayscale(num_output_channels=1)

    elif transformer_name == "ToPil":
        transformer = transforms.Lambda(lambda t: transforms.ToPILImage()(t))

    elif transformer_name == "Squeeze":
        transformer = transforms.Lambda(lambda t: t.squeeze())

    elif transformer_name == "LOG":
        transformer = transforms.Lambda(torch.log)

    elif transformer_name == "EXP":
        transformer = transforms.Lambda(torch.exp)

    elif transformer_name == "NOP":
        transformer = transforms.Lambda(lambda t: t)

    else:
        raise ValueError(f"Transformation {t_defs} is not supported!")

    return transformer


def get_transformer(t_defs: Any) -> Callable:
    """Tensor Transformers for image processing.

    Stack3D
    Morphes a 2D image to 3D by stacking the image along a new D dimension
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
        "Gaussian_Blur": {"blur_kernel_size": int, "blur_sigma": float},
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
