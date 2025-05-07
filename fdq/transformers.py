import torch
from torchvision import transforms

# from torchvision.transforms import v2 as transforms

# https://github.com/Varian-Imaging-Utils/VImagingUtils/blob/43e634c78162e349427da8d7bf04acb7dab5650f/script/motion/sim_motion.py#L115
# HU_OFFSET = -1102.20
# HU_SLOPE = 35582.62
# HU_SLOPE_OTHER_SOURCE = 48200.2344


def hounsfield(vol, intercept, slope):
    # intercept=HU_OFFSET, slope=HU_SLOPE
    vol = slope * vol + intercept
    return vol


def de_hounsfield(vol, intercept, slope):
    # intercept=HU_OFFSET, slope=HU_SLOPE
    vol = (vol - intercept) / slope
    return vol


def rerange_log_squashed(t, in_min, in_max, out_min, out_max):
    t = t.clone()  # clone the tensor to avoid modifying the original one
    mask1 = t < in_min
    mask2 = t > in_max
    t[mask1] = out_min - torch.log(torch.abs(t[mask1] / in_min))
    t[mask2] = out_max + torch.log(torch.abs(t[mask2] / in_max))
    mask3 = ~mask1 & ~mask2  # values within the range
    t[mask3] = ((t[mask3] - in_min) * (out_max - out_min)) / (in_max - in_min) + out_min
    return t


def inverse_rerange_log_squashed(t, in_min, in_max, out_min, out_max):
    t = t.clone()  # clone the tensor to avoid modifying the original one
    mask1 = t < out_min
    mask2 = t > out_max
    t[mask1] = in_min * torch.exp(out_min - t[mask1])
    t[mask2] = in_max * torch.exp(t[mask2] - out_max)
    mask3 = ~mask1 & ~mask2  # values within the range
    t[mask3] = ((t[mask3] - out_min) * (in_max - in_min)) / (out_max - out_min) + in_min
    return t


def rerange_log10_squashed(t, in_min, in_max, out_min, out_max):
    t = t.clone()  # clone the tensor to avoid modifying the original one
    mask1 = t < in_min
    mask2 = t > in_max
    t[mask1] = out_min - torch.log10(torch.abs(t[mask1] / in_min))
    t[mask2] = out_max + torch.log10(torch.abs(t[mask2] / in_max))
    mask3 = ~mask1 & ~mask2  # values within the range
    t[mask3] = ((t[mask3] - in_min) * (out_max - out_min)) / (in_max - in_min) + out_min
    return t


def inverse_rerange_log10_squashed(t, in_min, in_max, out_min, out_max):
    t = t.clone()  # clone the tensor to avoid modifying the original one
    mask1 = t < out_min
    mask2 = t > out_max
    t[mask1] = in_min * torch.pow(10, out_min - t[mask1])
    t[mask2] = in_max * torch.pow(10, t[mask2] - out_max)
    mask3 = ~mask1 & ~mask2  # values within the range
    t[mask3] = ((t[mask3] - out_min) * (in_max - in_min)) / (out_max - out_min) + in_min
    return t


def get_transformers(names):
    return transforms.Compose([get_transformer(t) for t in names])


def add_padding(img, pad, mode="constant", value=0):
    """
    img = N-di m input tensor
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


def pad_dim_to_multiple_of(img, dim, multiple, mode="constant", value=0):
    """
    Pad dimension 'dim' to a multiple of 'multiple'.
    E.g. Z dimension to a multiple of 32 to match the model architecture.
    """

    if mode not in ["constant", "edge", "replicate", "circular"]:
        raise ValueError(f"Padding mode {mode} not supported!")

    size = img.shape[dim]
    tot_pad_size = (multiple - size % multiple) % multiple
    pad_a = tot_pad_size // 2
    pad_b = tot_pad_size - pad_a
    pad = [0] * (img.dim() * 2)
    pad[-2 * dim - 1] = pad_a
    pad[-2 * dim - 2] = pad_b

    return torch.nn.functional.pad(input=img, pad=pad, mode=mode, value=value)


def remove_padding(img, pad):
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
    elif img.dim() == 5:
        return img[
            ps_start[0] : ps_stop[0],
            ps_start[1] : ps_stop[1],
            ps_start[2] : ps_stop[2],
            ps_start[3] : ps_stop[3],
            ps_start[4] : ps_stop[4],
        ]
    else:
        raise ValueError("Only 4D and 5D tensors are supported!")


def get_transformer(name):
    """
    Stack3D
    Morphes a 2D image to 3D by stacking the image along a new D dimension
    [B,C,H,W] -> [B,C,D,H,W]
    where d is set by the parameter 'stack_n'

    Resize_HW:
    Resizes images to Resize_IMG_SIZE_H x Resize_IMG_SIZE_W (must be defined in settings file)

    RandomHorizontalFlip:

    ToTensor:
    https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a
    torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] if the PIL Image
    belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the
    numpy.ndarray has dtype = np.uint8

    CLAMP_abs:
    Clamps input tensor to [lower,upper]
    (clamp = clip)

    CLAMP_perc:
    Clamps input tensor by percentiles [lower_perc,upper_perc]

    ReRange:
    Re-ranges input tensor from [in_min, in_max] to [out_min, out_max]

    ReRange_minmax:
    Re-ranges input tensor to [out_min, out_max]

    Rerange_log_squashed:
    Re-ranges input tensor from [in_min, in_max] to [out_min, out_max]
    with log squashing for values outside the range

    Rerange_log10_squashed:
    Re-ranges input tensor from [in_min, in_max] to [out_min, out_max]
    with log10 squashing for values outside the range

    Float32:
    Converts input tenor to Float32

    Uint8:
    Converts input tenor to Uint8

    RGB_Normalize:
    Channel-wise normalize RGB images in the range[0,1] according to mean/stdev of imageNET.

    RGB2GRAY:
    Converts RGB image to grayscale image.

    Gaussian_Blur:
    Blurs the image using the parameters 'blur_kernel_size' and 'blur_sigma'.

    Padding:
    Pads the input tensor using the parameters
    - padding_size (tuple)
    - padding_mode (Default = 'constant')
    - padding_value (Default = 0)

    UnPadding:
    Removes padding from the input tensor (e.g. to compute metrics on original image size)

    NOP
    No operation. Returns the input tensor as is.
    This is the default transformation if no transformation is defined in the settings file.

    FFT:
    Computes the 3D fast fourier transform (FFT) of the input tensor and
    splits the real and imaginary parts into channels. Only works with positive values.

    IFFT:
    Computes the 3D inverse fast fourier transform (IFFT) of the input tensor.
    The input tensor has to be a 2-channel tensor with the real and imaginary parts.

    FFT_log:
    Computes the 3D fast fourier transform (FFT) of the input tensor, log transform and
    splits the real and imaginary parts into channels. Only works with positive values.

    IFFT_exp:
    Computes the 3D inverse fast fourier transform (IFFT) of the input tensor.exp().
    The input tensor has to be a 2-channel tensor with the real and imaginary parts.

    DIV:
    Disivides the input tensor by the value specified in the parameter 'value'.

    MULT:
    Multiplies the input tensor by the value specified in the parameter 'value'.

    ToPil:
    Converts a tensor to a PIL image.

    Squeeze:
    Squeezes the input tensor.
    """

    all_required_params = {
        "Stack3D": {"stack_n": [int]},
        "Resize_HW": {"h": [int], "w": [int]},
        "CLAMP_abs": {"lower": [float, int], "upper": [float, int]},
        "CLAMP_perc": {"lower_perc": [float], "upper_perc": [float]},
        "ReRange": {"in_min": [float, int], "in_max": [float, int], "out_min": [float, int], "out_max": [float, int]},
        "ReRange_minmax": {"out_min": [float, int], "out_max": [float, int]},
        "Rerange_log_squashed": {
            "in_min": [float, int],
            "in_max": [float, int],
            "out_min": [float, int],
            "out_max": [float, int],
        },
        "Rerange_log10_squashed": {
            "in_min": [float, int],
            "in_max": [float, int],
            "out_min": [float, int],
            "out_max": [float, int],
        },
        "Gaussian_Blur": {"blur_kernel_size": int, "blur_sigma": float},
        "Padding": {"padding_size": [list, int], "padding_mode": [str], "padding_value": [int]},
        "UnPadding": {"padding_size": [list, int]},
        "ADD": {"value": [float, int]},
        "MULT": {"value": [float, int]},
        "DIV": {"value": [float, int]},
    }

    if isinstance(name, dict):
        keys = list(name.keys())
        if len(keys) != 1:
            raise ValueError(f"Transformation {name} does not correspond to the expected format!")
        transformer_name = keys[0]
        parameters = name[transformer_name]
    elif isinstance(name, str):
        transformer_name = name
        parameters = None
    else:
        raise ValueError(f"Transformation {name} does not correspond to the expected format!")

    required_params = all_required_params.get(transformer_name, {})

    # check if all required parameters are present and datatypes are correct
    for req_param, req_type in required_params.items():
        if req_param not in parameters:
            raise ValueError(f"Parameter {req_param} is missing for transformation {transformer_name}.")
        if type(parameters[req_param]) not in req_type:
            raise ValueError(f"Parameter {req_param} for transformation {transformer_name} is not correct.")

    if transformer_name == "Stack3D":
        stack_n = parameters.get("stack_n")
        # [B,C,H,W] -> [B,C,D,H,W] -> dim = 2
        transformer = transforms.Lambda(lambda t: torch.stack([t] * stack_n, dim=2))

    elif transformer_name == "Resize_HW":
        transformer = transforms.Resize((parameters.get("h"), parameters.get("w")), antialias=True)

    elif transformer_name == "RandomHorizontalFlip":
        transformer = transforms.RandomHorizontalFlip()

    elif transformer_name == "ToTensor":
        transformer = transforms.ToTensor()

    elif transformer_name == "Float32":
        transformer = transforms.Lambda(lambda t: t.type(torch.float32))

    elif transformer_name == "Uint8":
        transformer = transforms.Lambda(lambda t: t.type(torch.uint8))

    elif transformer_name == "ADD":
        transformer = transforms.Lambda(lambda t: t + parameters.get("value"))

    elif transformer_name == "MULT":
        transformer = transforms.Lambda(lambda t: t * parameters.get("value"))

    elif transformer_name == "DIV":
        transformer = transforms.Lambda(lambda t: t / parameters.get("value"))

    elif transformer_name == "ToPil":
        # pylint: disable=W0108
        transformer = transforms.Lambda(lambda t: transforms.ToPILImage()(t))

    elif transformer_name == "Squeeze":
        transformer = transforms.Lambda(lambda t: t.squeeze())

    elif transformer_name == "LOG":
        # pylint: disable=W0108
        transformer = transforms.Lambda(lambda t: torch.log(t))

    elif transformer_name == "EXP":
        # pylint: disable=W0108
        transformer = transforms.Lambda(lambda t: torch.exp(t))

    elif transformer_name == "RGB_Normalize":
        transformer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    elif transformer_name == "CLAMP_abs":
        transformer = transforms.Lambda(lambda t: torch.clamp(t, parameters.get("lower"), parameters.get("upper")))

    elif transformer_name == "CLAMP_perc":

        def _rdm_reduce(tensor):
            # random reduce tensor size for quantile computation -> estimation
            n = 12582912  # random defined as 8*3*32*128*128
            if tensor.numel() > n:
                n = min(n, tensor.numel())
                random_indices = torch.randperm(tensor.numel())[:n]
                return tensor.view(-1)[random_indices]
            else:
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
        transformer = transforms.Lambda(lambda t: ((t - in_min) * (out_max - out_min)) / (in_max - in_min) + out_min)

    elif transformer_name == "Rerange_log_squashed":
        in_min = parameters.get("in_min")
        in_max = parameters.get("in_max")
        out_min = parameters.get("out_min")
        out_max = parameters.get("out_max")
        transformer = transforms.Lambda(lambda t: rerange_log_squashed(t, in_min, in_max, out_min, out_max))

    elif transformer_name == "Inverse_Rerange_log_squashed":
        in_min = parameters.get("in_min")
        in_max = parameters.get("in_max")
        out_min = parameters.get("out_min")
        out_max = parameters.get("out_max")
        transformer = transforms.Lambda(lambda t: inverse_rerange_log_squashed(t, in_min, in_max, out_min, out_max))

    elif transformer_name == "Rerange_log10_squashed":
        in_min = parameters.get("in_min")
        in_max = parameters.get("in_max")
        out_min = parameters.get("out_min")
        out_max = parameters.get("out_max")
        transformer = transforms.Lambda(lambda t: rerange_log10_squashed(t, in_min, in_max, out_min, out_max))

    elif transformer_name == "Inverse_Rerange_log10_squashed":
        in_min = parameters.get("in_min")
        in_max = parameters.get("in_max")
        out_min = parameters.get("out_min")
        out_max = parameters.get("out_max")
        transformer = transforms.Lambda(lambda t: inverse_rerange_log10_squashed(t, in_min, in_max, out_min, out_max))

    elif transformer_name == "ReRange_minmax":
        out_min = parameters.get("out_min")
        out_max = parameters.get("out_max")
        transformer = transforms.Lambda(lambda t: ((t - t.min()) * (out_max - out_min)) / (t.max() - t.min()) + out_min)

    elif transformer_name == "RGB2GRAY":
        transformer = transforms.Grayscale(num_output_channels=1)

    elif transformer_name == "Gaussian_Blur":
        transformer = transforms.GaussianBlur(
            kernel_size=parameters.get("blur_kernel_size"), sigma=parameters.get("blur_sigma")
        )

    elif transformer_name == "Padding":
        transformer = transforms.Lambda(
            lambda t: add_padding(
                t,
                pad=parameters.get("padding_size"),
                mode=parameters.get("padding_mode"),
                value=parameters.get("padding_value"),
            )
        )

    elif transformer_name == "Padding_to_multiple":
        transformer = transforms.Lambda(
            lambda t: pad_dim_to_multiple_of(
                t,
                dim=parameters.get("dim"),
                multiple=parameters.get("multiple"),
                mode=parameters.get("padding_mode"),
                value=parameters.get("padding_value"),
            )
        )

    elif transformer_name == "UnPadding":
        transformer = transforms.Lambda(lambda t: remove_padding(t, pad=parameters.get("padding_size")))

    elif transformer_name == "NOP":
        transformer = transforms.Lambda(lambda t: t)

    elif transformer_name == "FFT":
        # pylint: disable=W0108
        # fft and split real and imaginary parts into channel
        transformer = transforms.Lambda(lambda t: torch.stack([torch.fft.fftn(t).real, torch.fft.fftn(t).imag], dim=0))

    elif transformer_name == "IFFT":
        # pylint: disable=W0108
        # ifft transform from fourier space to real space
        transformer = transforms.Lambda(lambda t: torch.abs(torch.fft.ifftn((t[0] + 1j * t[1]))))

    elif transformer_name == "FFT_log":
        # pylint: disable=W0108
        # fft and split real and imaginary parts into channel
        transformer = transforms.Lambda(
            lambda t: torch.stack([torch.fft.fftn(t).log().real, torch.fft.fftn(t).log().imag], dim=0)
        )

    elif transformer_name == "IFFT_exp":
        # pylint: disable=W0108
        # ifft transform from fourier space to real space
        transformer = transforms.Lambda(lambda t: torch.abs(torch.fft.ifftn((t[0] + 1j * t[1]).exp())))

    else:
        raise ValueError(f"Transformation {name} is not supported!")

    return transformer


def convert_string_to_type(s):
    if "." in s:
        return float(s)
    else:
        return int(s)


def create_transformers(experiment):
    """
    hdf_ds_key_request:
    if hdf_ds_key_request is used (hdf_preparator), then the transformations have to be defined
    as dict per hdf_ds_key.TODO: add a test for this case

    The following transformations are supported:
    train a/b are two slots to define transformations applied to the train data loader.
    -> these have to be defined in the respective preparators!

    val a/b and test a/b are exactly the same for validation and test data.

    dataset_transforms_eval a/b are two slots to define additional transformations for the evaluators,
    e.g. a reverse transformation to convert the output back to the original space such as Houndsfield Units.

    dataset_transforms_export is a slot to store a transformation for image exports
    (e.g normalizations, clamping, etc..)

    dataset_transforms_sampling is yet another slot that allows to define transformations for the sampling.
    (e.g. clamping during DDPM sampling)

    IMPORTANT: these are only slots to define generic transforms in the config file. The actual transformations
    has to be manually implemented in the preparator, training loop or evaluator codes!

    """

    # HDF loader specific:
    # the hdf loader allows to specify instead of a list also a dict of transformations for each group
    hdf_ds_key_request_all = experiment.data_params.get("hdf_ds_key_request", [])
    hdf_ds_key_request_test = experiment.data_params.get("hdf_ds_key_request_test", None)

    if hdf_ds_key_request_test is None:
        hdf_ds_key_request_test = hdf_ds_key_request_all

    DEFAULT_TRANSFORM = ["NOP"]

    if len(hdf_ds_key_request_all) > 0:
        DEFAULT_TRANSFORM_TRAIN = {}
        for key in hdf_ds_key_request_all:
            DEFAULT_TRANSFORM_TRAIN[key] = ["NOP"]
    else:
        DEFAULT_TRANSFORM_TRAIN = DEFAULT_TRANSFORM

    if len(hdf_ds_key_request_test) > 0:
        DEFAULT_TRANSFORM_TEST = {}
        for key in hdf_ds_key_request_test:
            DEFAULT_TRANSFORM_TEST[key] = ["NOP"]
    else:
        DEFAULT_TRANSFORM_TEST = DEFAULT_TRANSFORM

    transform_config = {
        "train_a": experiment.data_params.get("dataset_transforms_train_a", DEFAULT_TRANSFORM_TRAIN),
        "train_b": experiment.data_params.get("dataset_transforms_train_b", DEFAULT_TRANSFORM_TRAIN),
        "val_a": experiment.data_params.get("dataset_transforms_val_a", DEFAULT_TRANSFORM_TRAIN),
        "val_b": experiment.data_params.get("dataset_transforms_val_b", DEFAULT_TRANSFORM_TRAIN),
        "test_a": experiment.data_params.get("dataset_transforms_test_a", DEFAULT_TRANSFORM_TEST),
        "test_b": experiment.data_params.get("dataset_transforms_test_b", DEFAULT_TRANSFORM_TEST),
        "eval_a": experiment.data_params.get("dataset_transforms_eval_a", DEFAULT_TRANSFORM),
        "eval_b": experiment.data_params.get("dataset_transforms_eval_b", DEFAULT_TRANSFORM),
        "export": experiment.data_params.get("dataset_transforms_export", DEFAULT_TRANSFORM),
        "sampling": experiment.data_params.get("dataset_transforms_sampling", DEFAULT_TRANSFORM),
    }

    experiment.transform_config = transform_config

    hdf_ds_key_request_defs = {
        "train_a": hdf_ds_key_request_all,
        "train_b": hdf_ds_key_request_all,
        "val_a": hdf_ds_key_request_all,
        "val_b": hdf_ds_key_request_all,
        "test_a": hdf_ds_key_request_test,
        "test_b": hdf_ds_key_request_test,
        "eval_a": hdf_ds_key_request_all,
        "eval_b": hdf_ds_key_request_all,
        "export": hdf_ds_key_request_all,
        "sampling": hdf_ds_key_request_all,
    }

    transformers = {}

    for mode, transform_def in transform_config.items():
        if len(transform_def) == 0:
            continue

        hdf_ds_key_request = hdf_ds_key_request_defs[mode]

        if mode in ["sampling", "export", "eval_a", "eval_b"]:
            if isinstance(transform_def, dict):
                raise ValueError(f"dataset_transforms_{mode} has to be defined as a list (not per dataset key)!")
            transformers[mode] = transforms.Compose([get_transformer(t) for t in transform_def])

        elif len(hdf_ds_key_request) == 0:
            if isinstance(transform_def, dict):
                raise ValueError(
                    "Current settings file has no hdf_ds_key_request, "
                    "but the transformations are defined per hdf_ds_key!"
                    f"current mode: {mode}"
                )
            transformers[mode] = transforms.Compose([get_transformer(t) for t in transform_def])

        else:
            if not isinstance(transform_def, dict):
                raise ValueError(f"Transformations have to be written as dict per hdf_ds_key_request. (Mode: {mode})")

            transforms_dict = {}
            for ds_key in hdf_ds_key_request:
                ds_transform = transform_def.get(ds_key, [])
                if len(ds_transform) > 0:
                    transforms_dict[ds_key] = transforms.Compose([get_transformer(t) for t in ds_transform])
            transformers[mode] = transforms_dict

    experiment.trainTransformer_a = transformers.get("train_a", None)
    experiment.trainTransformer_b = transformers.get("train_b", None)
    experiment.valTransformer_a = transformers.get("val_a", None)
    experiment.valTransformer_b = transformers.get("val_b", None)
    experiment.testTransformer_a = transformers.get("test_a", None)
    experiment.testTransformer_b = transformers.get("test_b", None)
    experiment.exportTransformer = transformers.get("export", None)
    experiment.evaluatorTransformer_a = transformers.get("eval_a", None)
    experiment.evaluatorTransformer_b = transformers.get("eval_b", None)
    experiment.samplingTransformer = transformers.get("sampling", None)
