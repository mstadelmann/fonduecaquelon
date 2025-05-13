import logging
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import wandb
from functools import wraps
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

import cv2

# from evaluator.evaluator_common import compute_histo
from fdq.ui_functions import iprint, wprint, eprint

import fdq.transformers

from typing import Sequence
from PIL import Image
from scipy.ndimage import zoom
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def compute_histo(image, nb_bins=100):
    bins = torch.linspace(image.min(), image.max(), nb_bins).cpu()
    return torch.histogram(image.cpu().float(), bins=bins)


def no_grad_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    return wrapper


def showImg_cv(tensor_image, window_name="Image"):
    """
    Displays a PyTorch tensor image using OpenCV.

    Supports:
    - [H, W]  (2D grayscale)
    - [1, H, W] (grayscale)
    - [3, H, W] (RGB)
    """
    # Detach and move to CPU just in case
    tensor_image = tensor_image.detach().cpu()

    if tensor_image.ndim == 2:
        # [H, W] grayscale
        np_img = tensor_image.numpy()
    elif tensor_image.ndim == 3:
        if tensor_image.shape[0] == 1:
            # [1, H, W] grayscale
            np_img = tensor_image[0].numpy()
        elif tensor_image.shape[0] == 3:
            # [3, H, W] RGB → HWC and convert to BGR for OpenCV
            np_img = tensor_image.mul(255).byte().numpy()
            np_img = np.transpose(np_img, (1, 2, 0))  # [H, W, C]
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError("Expected 1 or 3 channels in [C, H, W] tensor.")
    else:
        raise ValueError("Tensor must be 2D or 3D (C x H x W or H x W).")

    # Normalize grayscale if float
    if np_img.dtype in [np.float32, np.float64]:
        np_img = np.clip(np_img, 0, 1)
        np_img = (np_img * 255).astype(np.uint8)

    cv2.imshow(window_name, np_img)
    # cv2.waitKey(0)

    # Wait for a key press or window closure
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key != 255:  # A key was pressed
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


def showImg(img, batch_nb=0, quietMode=True, grayscale=False):
    """
    pyplot.imshow expects [H W C]
    """

    if isinstance(img, torch.Tensor):
        if len(img.shape) > 3:
            img = img[batch_nb, ...]
            img = torch.squeeze(img, 0)

        if len(img.shape) == 3:
            img = img.permute(1, 2, 0)

        if quietMode:  # do not show warning messages
            logger = logging.getLogger()
            old_level = logger.level
            logger.setLevel(100)

        if grayscale:
            plt.imshow(img.cpu(), cmap="gray")
        else:
            plt.imshow(img.cpu())

        plt.show()

        if quietMode:
            logger.setLevel(old_level)

    else:
        img.show()


def fix_histo_ticks(axs, hist):
    # if bulk of data is in +- 1 range, set ticks manually
    data = np.array(hist.bin_edges)
    close_to_zero = np.sum((data >= -0.2) & (data <= 0.2)) / data.size
    norm_range = (
        np.sum((data >= -1.2) & (data <= 1.2)) / data.size
    )  # most data between +- 1.2
    if close_to_zero < 0.2 and norm_range > 0.8:
        axs.set_xticks(np.linspace(-1, 1, num=5))


def createSubplots(
    image_list,
    batch_nb=0,
    labels=None,
    grayscale=False,
    experiment=None,
    histogram=False,
    histogram3d=False,
    save_path=None,
    show_plot=True,
    figure_title=None,
    max_cols=None,
    hide_ticks=False,
    apply_global_range=True,
    lower_percentile_cutoff=0.01,
    upper_percentile_cutoff=0.99,
    fig_size=3,
):
    if labels is None:
        labels = len(image_list) * [""]
    elif len(labels) != len(image_list):
        raise ValueError("ERROR - nb labels does not correspond to nb images!")

    histos_3d = None

    # it might be a 3d experiment, but data was presliced to reduce storage footprint
    reslicing_required = len(image_list[0].shape) > 4

    if experiment is not None and experiment.is_3d and reslicing_required:
        if histogram3d:
            histos_3d = [
                compute_histo(torch.squeeze(img3d).cpu()) for img3d in image_list
            ]

        nb_images = len(image_list) * len(experiment.img_export_dims)
        nb_slices_per_img = len(experiment.img_export_dims)
        image_list = sum(
            [extract_2d_slices(img, experiment, strict=False) for img in image_list], []
        )
        labels = sum(
            [[lbl + "_" + dir for dir in experiment.img_export_dims] for lbl in labels],
            [],
        )

    else:
        nb_images = len(image_list)
        nb_slices_per_img = 1

    clamper_perc = transformers.get_transformer(
        {
            "CLAMP_perc": {
                "lower_perc": lower_percentile_cutoff,
                "upper_perc": upper_percentile_cutoff,
            }
        }
    )

    img_clamp = [clamper_perc(img) for img in image_list]
    global_min = min(float(img.min()) for img in img_clamp)
    global_max = max(float(img.max()) for img in img_clamp)

    if histogram3d:
        nb_rows = 3
        nb_cols = nb_images
    elif histogram:
        nb_rows = 2
        nb_cols = nb_images
    elif nb_images < 3:
        nb_rows = 1
        nb_cols = nb_images
    else:
        nb_rows = 2
        nb_cols = int(np.ceil(nb_images / nb_rows))

    if not histogram and max_cols is not None and max_cols < nb_cols:
        nb_cols = max_cols
        nb_rows = int(np.ceil(nb_images / nb_cols))

    fig, axs = plt.subplots(
        nb_rows, nb_cols, figsize=(fig_size * nb_cols, fig_size * nb_rows)
    )

    if figure_title is not None:
        if histogram3d:
            figure_title += " - green = 3D histogram"
        plt.suptitle(figure_title)

    if not hide_ticks:
        plt.subplots_adjust(wspace=0.4)

    for i, (img, lbl) in enumerate(zip(image_list, labels)):
        is_last = i == len(image_list) - 1

        # to make this work for numpy arrays
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)

        # subplot(nrows, ncols..)
        c_row = int(np.floor(i / nb_cols))
        c_col = i - (c_row * nb_cols)

        if len(img.shape) > 3:
            img = img[batch_nb, ...]
            img = torch.squeeze(img, 0)

        if len(img.shape) == 3:
            img = img.permute(1, 2, 0)

        if not grayscale and img.dtype == torch.float and img.max() > 50:
            # assume RGB image in float values
            # TODO where is this used? this might not be a good idea!
            img = img.int()

        if not apply_global_range:
            global_min = float(img.min())
            global_max = float(img.max())

        if grayscale:
            if nb_rows > 1:
                im = axs[c_row, c_col].imshow(
                    img.cpu(), cmap="gray", vmin=global_min, vmax=global_max
                )
                if not apply_global_range:
                    fig.colorbar(im, ax=axs[c_row, c_col])

                elif is_last:
                    fig.colorbar(im, ax=axs[c_row, c_col])
            else:
                axs[c_col].imshow(
                    img.cpu(), cmap="gray", vmin=global_min, vmax=global_max
                )
        else:
            if nb_rows > 1:
                axs[c_row, c_col].imshow(img.cpu(), vmin=global_min, vmax=global_max)
            else:
                axs[c_col].imshow(img.cpu(), vmin=global_min, vmax=global_max)

        if lbl is not None:
            if nb_rows > 1:
                axs[c_row, c_col].set_title(lbl)
            else:
                axs[c_col].set_title(lbl)

        if histogram:
            if grayscale:
                hist = compute_histo(torch.squeeze(img).cpu())
                axs[1, c_col].plot(hist.bin_edges[:-1], hist.hist, color="r")
                fix_histo_ticks(axs[1, c_col], hist)

            else:
                hist = compute_histo(torch.squeeze(img).cpu()[0])
                axs[1, c_col].plot(hist.bin_edges[:-1], hist.hist, color="r")
                hist = compute_histo(torch.squeeze(img).cpu()[1])
                axs[1, c_col].plot(hist.bin_edges[:-1], hist.hist, color="g")
                hist = compute_histo(torch.squeeze(img).cpu()[2])
                axs[1, c_col].plot(hist.bin_edges[:-1], hist.hist, color="b")

        if i % len(experiment.img_export_dims) == 0 and histos_3d is not None:
            hist = histos_3d[int(i / len(experiment.img_export_dims))]
            axs[2, c_col].plot(hist.bin_edges[:-1], hist.hist, color="g")
            fix_histo_ticks(axs[2, c_col], hist)

            # Make the plot wider?
            if nb_slices_per_img > 1:
                for sl in range(1, nb_slices_per_img):
                    axs[2, c_col + sl].remove()

                axs[2, c_col].set_position(
                    [
                        axs[2, c_col].get_position().x0,
                        axs[2, c_col].get_position().y0,
                        axs[2, c_col].get_position().width * nb_slices_per_img,
                        axs[2, c_col].get_position().height,
                    ]
                )

    if hide_ticks:
        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    if save_path is not None:
        plt.savefig(save_path)
    elif experiment is not None:
        plt.savefig(experiment.get_next_export_fn())
    else:
        eprint("No save path provided!")

    if show_plot:
        plt.show()


def saveImg(img, path, experiment=None):
    """Saves image or list of images to path.

    Args:
        img (torch.tensor or list): image or list of images
        path (str): destination path
    """

    def _saveImgC(img, path):
        try:
            if len(img.shape) == 2 or img.shape[0] in (1, 3):
                save_image(img, path)
            else:
                for C, img_c in enumerate(img):
                    save_image(img_c, path.replace(".", f"_C{C:02}."))
        except Exception:
            eprint(f"Error saving image with shape {img.shape} to {path}")

    if experiment is not None:
        img_to_store = experiment.exportTransformer(img)
    else:
        img_to_store = img

    if experiment is not None:
        img_to_store = extract_2d_slices(img_to_store, experiment)

    if isinstance(img_to_store, torch.Tensor):
        img_to_store = [img_to_store]

    if isinstance(img_to_store, list):
        for i, img_i in enumerate(img_to_store):
            if len(img_i.shape) == 4:
                for B, img_b in enumerate(img_i):
                    temp_path = path.replace(".", f"_B{B:02}_{i:02}.")
                    _saveImgC(img_b, temp_path)
            else:
                _saveImgC(img_i, path)


def check_if_image(data):
    """
    Check if tensor is (possibly) an image.
    """
    if data is None:
        return False
    elif len(data.shape) < 2:
        # at least two dimensions
        return False
    # at least 5 pixels in 2 dims to be considered as image
    elif not data.shape[-1] >= 5 or not data.shape[-2] >= 5:
        return False

    return True


def extract_2d_slices(
    images: torch.tensor,
    experiment,
    nb_slices=1,
    select_channel=None,
    squeeze_last=False,
    strict=True,
):
    """
    Extracts 2D slice(s) from tensor.
    Returns list of requested tensors.

    Expected image data format 2D: [H,W], [C,H,W] or [B,C,H,W]
    (Fewer dims in 2D are expected to be singleton dimensions.)
    Expected image data format 3D: [B,C,D,H,W]
    (Fewer dimensions in 3D fill fail.)

    For 3D data, experiment.img_export_dims defines the slicing direction.
    Supported values are  D, W, H or a list of these.

    nb_slices: How many slices to extract. Defaults to 1. Not yet implemented!

    squeeze_last: if true, returns only the last image in the listed
                  as squeezed tensor (for direct plotting or saving).

    strict: if True, raises an error if the image shape is not as expected.

    supported input tensor shapes:
    a)       [H,W] --> [1,1,H,W]
    b)     [C,H,W] --> [1,C,H,W] or with select_channel set:    [1,1,H,W]
    c)   [B,C,H,W] --> [1,C,H,W] or with select_channel set:    [1,1,H,W]
                                 or with nb_slices set: [nb_slices,C,H,W]
    d) [B,C,D,H,W] --> [1,C,H,W] or with select_channel set:    [1,1,H,W]
                                 or with nb_slices set: [nb_slices,C,H,W]
                                 or with multiple dimensions set: [[1,C,H,W]]
    """

    if nb_slices != 1:
        raise NotImplementedError("Only one slice is supported at the moment.")

    if select_channel is not None:
        raise NotImplementedError("Selecting a channel is not yet implemented.")

    if not check_if_image(images):
        return []

    image_list = []

    # select center batch img
    batch_size = images.shape[0]
    batch_center = int(batch_size / 2)

    nb_input_dim = len(images.shape)

    # A) [H,W] or B) [C,H,W]
    if nb_input_dim in [2, 3]:
        image_list.append(images)

    # C) [B,C,H,W]
    elif nb_input_dim == 4 and not experiment.is_3d:
        image_list.append(images[batch_center, ...])

    # D) [B,C,D,H,W]
    elif nb_input_dim == 5:
        if not experiment.is_3d:
            raise ValueError("Image data is 3D, but experiment is not 3D.")

        for slicing_dir in experiment.img_export_dims:
            dim = ["B", "C", "D", "H", "W"].index(slicing_dir)
            slice_idx = int(images.shape[dim] / 2)
            if dim == 2:
                image_list.append(images[batch_center, :, slice_idx, :, :])
            elif dim == 3:
                image_list.append(images[batch_center, :, :, slice_idx, :])
            elif dim == 4:
                image_list.append(images[batch_center, :, :, :, slice_idx])

    # 3d experiment, but pre-sliced data
    elif (
        nb_input_dim == 4
        and images.shape[0] == 1
        and images.shape[1] == 1
        and not strict
    ):
        image_list.append(images[0, 0, :, :])

    else:
        raise ValueError(
            f"ERROR, not supported image shape {images.shape}, 3D experiment: {experiment.is_3d}."
        )

    nb_expected_dims = 4

    for i, img in enumerate(image_list):
        while img.dim() < nb_expected_dims:
            img = img.unsqueeze(0)
            image_list[i] = img

    if not squeeze_last:
        return image_list
    else:
        return torch.squeeze(image_list[-1]).cpu()


def init_tensorboard(experiment):
    if not experiment.useTensorboard:
        return
    experiment.tb_writer = SummaryWriter(f"{experiment.results_dir}/tb/")
    experiment.tb_graph_stored = False
    iprint("-------------------------------------------------------")
    iprint("Start tensorboard typing:")
    iprint(f"tensorboard --logdir={experiment.results_dir}/tb/ --bind_all")
    iprint("-------------------------------------------------------")


def add_graph(experiment, inputs):
    if experiment.tb_graph_stored:
        return
    try:
        experiment.tb_writer.add_graph(experiment.model, inputs)
        experiment.tb_graph_stored = True
    except Exception:
        wprint("Unable to add graph to Tensorboard.")


def save_tensorboard_loss(experiment, inputs=None):
    if not experiment.useTensorboard:
        return

    if experiment.tb_writer is None:
        init_tensorboard(experiment)

    experiment.tb_writer.add_scalar(
        "train_loss", experiment.trainLoss, experiment.current_epoch
    )
    experiment.tb_writer.add_scalar(
        "val_loss", experiment.valLoss, experiment.current_epoch
    )


@no_grad_decorator
def save_tensorboard(experiment, images=None, scalars=None, max_batch_size=None):
    """Log images and scalars to tensorboard.

    Scalars: {name: value}
    Train and Val loss are logged automatically.
    Images are expected to be in shape [B,C,D,H,W]
    """
    if not experiment.useTensorboard:
        return

    if experiment.tb_writer is None:
        init_tensorboard(experiment)

    if scalars is None:
        scalars = {}
    scalars["train_loss"] = experiment.trainLoss
    scalars["val_loss"] = experiment.valLoss

    for scalar_name, scalar_value in scalars.items():
        experiment.tb_writer.add_scalar(
            scalar_name, scalar_value, experiment.current_epoch
        )

    for image in images:
        img = image["data"]
        dataformat = image.get("dataformats", "NCHW")
        if max_batch_size is not None:
            img = img[:max_batch_size, ...]

        experiment.tb_writer.add_images(
            tag=image["name"],
            img_tensor=img,
            global_step=experiment.current_epoch,
            dataformats=dataformat,
        )


def init_wandb(experiment):
    """
    Initialize weights and biases
    """
    if (
        not experiment.useWandb
        or experiment.wandb_project is None
        or experiment.wandb_entity is None
    ):
        return

    if not experiment.wandb_initialized:
        exp_name = os.path.basename(experiment.results_dir)
        if experiment.previous_slurm_job_id is not None:
            try:
                exp_name_all = exp_name.split("__")
                exp_name = (
                    exp_name_all[0]
                    + "__"
                    + exp_name_all[1]
                    + "__"
                    + experiment.previous_slurm_job_id
                    + "->"
                    + exp_name_all[2]
                )
            except Exception:
                exp_name = os.path.basename(experiment.results_dir)

        # trunk-ignore(pylint/I1101)
        wandb.login(key=experiment.wandb_key)
        # trunk-ignore(pylint/I1101)
        wandb.init(
            # set the wandb project where this run will be logged
            project=experiment.wandb_project,
            entity=experiment.wandb_entity,
            name=exp_name,
            # track hyperparameters and run metadata
            config=experiment.exp_file,
        )

        experiment.wandb_initialized = True

        iprint(f"Init Wandb -  log path: {wandb.run.dir}")


@no_grad_decorator
def save_wandb(experiment, images):
    """
    Track experiment data with weights and biases.

    Args:
        experiment (class): experiment object.
        images (list): stacked list of [[image_name, image_data]].

    Returns:
        type: Description of returned object.
    """
    # bool flag that defines when images are uploaded (defined by the parameter img_exp_nb in the config file)
    store_more_images = (
        experiment.current_val_batch * experiment.val_batch_size < experiment.img_exp_nb
    )
    use_wandb = (
        experiment.useWandb
        and experiment.wandb_project is not None
        and experiment.wandb_entity is not None
    )

    if not (use_wandb and store_more_images):
        return

    init_wandb(experiment)

    image_grid = []
    img_list = []

    # remove outliers for logging...
    t_clamp = transformers.get_transformer(
        {"CLAMP_perc": {"lower_perc": 0.01, "upper_perc": 0.99}}
    )

    for img_name, img_data in images:
        for slice_dir, img_2d in enumerate(
            extract_2d_slices(images=img_data, experiment=experiment)
        ):
            img_list.append([f"{img_name}_{slice_dir}", t_clamp(img_2d)])

    global_min = min(float(img.min()) for _, img in img_list)
    global_max = max(float(img.max()) for _, img in img_list)

    t_pil = transformers.get_transformer("ToPil")
    t_range = transformers.get_transformer(
        {
            "ReRange": {
                "in_min": global_min,
                "in_max": global_max,
                "out_min": 0,
                "out_max": 1,
            }
        }
    )

    for img_name, img in img_list:
        if img is None:
            continue

        nb_idx = min(img.shape[0], experiment.img_exp_nb)
        nb_channels = img.shape[1]

        if nb_channels in (1, 3):
            if experiment.wandb_generate_image_grid:
                image_grid.append((t_range(img[:nb_idx, ...].clone()), img_name))
            else:
                # trunk-ignore(pylint/I1101)
                wandb_images = wandb.Image(
                    t_pil(t_range(img[:nb_idx, ...]).squeeze()), caption=img_name
                )
                # trunk-ignore(pylint/I1101)
                wandb.log({img_name: wandb_images})

        else:
            raise NotImplementedError(
                "WANDB currently not implemented for more than 3 channels."
            )
            # for C in list(range(img.shape[1])):
            #     if experiment.wandb_generate_image_grid:
            #         image_grid.append((torch.unsqueeze(img[:nb_idx, C, ...], 1).clone(), f"{img_name}_C{C}"))
            #     else:
            #         # trunk-ignore(pylint/I1101)
            #         wandb_images = wandb.Image(
            #             torch.unsqueeze(img[:nb_idx, C, ...], 1), caption=f"{img_name}_C{C}"
            #         )
            #         # trunk-ignore(pylint/I1101)
            #         wandb.log({f"{img_name}_C{C}": wandb_images})

    if experiment.wandb_generate_image_grid:
        transform = T.ToPILImage()
        grid = transform(make_grid(torch.cat([tens for tens, _ in image_grid])))
        caption = ", ".join([name for _, name in image_grid])
        # trunk-ignore(pylint/I1101)
        wandb.log(
            {
                f"val_batch_{experiment.current_val_batch:02}": wandb.Image(
                    grid, caption=caption
                )
            }
        )


def save_wandb_loss(experiment, additional_metrics=None):
    """
    Track experiment loss with weights and biases
    """
    if (
        not experiment.useWandb
        or not experiment.wandb_project
        or not experiment.wandb_entity
    ):
        return

    init_wandb(experiment)

    log_metrics = {
        "train_loss": experiment.trainLoss,
        "val_loss": experiment.valLoss,
        "epoch": experiment.current_epoch,
    }

    if additional_metrics is not None and isinstance(additional_metrics, dict):
        log_metrics.update(additional_metrics)

    wandb.log(log_metrics)


def draw_diffusion_history(
    experiment,
    img_history: list,
    nb_denoise_steps: int,
    ground_truth_img: torch.tensor = None,
    condition_img: torch.tensor = None,
    add_eval_transforms: bool = False,
):
    def _extract_from_3d(img):
        return extract_2d_slices(
            img, experiment=experiment, squeeze_last=True, strict=False
        ).cpu()

    is_grayscale = experiment.nb_out_channels == 1
    img_list = []
    lbl_list = []
    history_lbls = [
        f"step {int(idx)}"
        for idx in torch.linspace(
            0, nb_denoise_steps, len(img_history), dtype=torch.int32
        )
    ]

    # add result converted to HU
    res = _extract_from_3d(img_history[-1])
    if add_eval_transforms:
        img_list.append(experiment.evaluatorTransformer_a(res))
        lbl_list.append("Res [HU]")

    if condition_img is not None:
        condition_img = _extract_from_3d(condition_img)
        img_list.append(condition_img)
        lbl_list.append("Condition")
        if add_eval_transforms:
            img_list.append(experiment.evaluatorTransformer_a(condition_img))
            lbl_list.append("Condition HU")

    if ground_truth_img is not None:
        ground_truth_img = _extract_from_3d(ground_truth_img)
        img_list.append(ground_truth_img)
        lbl_list.append("GT")
        if add_eval_transforms:
            img_list.append(experiment.evaluatorTransformer_a(ground_truth_img))
            lbl_list.append("GT [HU]")

        # add diff
        diff = ground_truth_img - res
        img_list.append(diff)
        lbl_list.append("GT - RES")
        if add_eval_transforms:
            img_list.append(experiment.evaluatorTransformer_a(diff))
            lbl_list.append("GT - RES [HU]")

    createSubplots(
        image_list=img_history + img_list,
        labels=history_lbls + lbl_list,
        grayscale=is_grayscale,
        experiment=experiment,
        histogram=True,
        histogram3d=False,
        save_path=None,
        show_plot=False,
        figure_title="Diffusion History",
        hide_ticks=False,
        apply_global_range=not add_eval_transforms,
    )


def createSubplots_CT(
    volume_list: Sequence[torch.tensor],
    vol_res: Sequence[float],
    labels: Sequence[str] = None,
    histogram=True,
    img_save_path=None,
):
    nb_images = len(volume_list)

    # image_list = [plot_vol_ax_cor_sag(vol, vol_res) for vol in volume_list]
    # labels = sum([[lbl + "_" + dir for dir in experiment.img_export_dims] for lbl in labels], [])

    if histogram:
        nb_rows = 2
        nb_cols = nb_images
    elif nb_images < 3:
        nb_rows = 1
        nb_cols = nb_images
    else:
        nb_rows = 2
        nb_cols = int(np.ceil(nb_images / nb_rows))

    if labels is not None:
        if len(labels) != len(volume_list):
            eprint(
                f"ERROR - nb labels ({len(labels)}) does not correspond to nb images ({len(volume_list)})!"
            )
            sys.exit()
    else:
        labels = len(volume_list) * [None]

    fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(3 * nb_cols, 3 * nb_rows))

    for i, (vol, lbl) in enumerate(zip(volume_list, labels)):
        # subplot(nrows, ncols..)
        c_row = int(np.floor(i / nb_cols))
        c_col = i - (c_row * nb_cols)

        if vol.ndim != 3:
            vol = vol.squeeze()

        if vol.min() == 0 and vol.max() == 1:
            # its a mask
            _fig, _axs = plot_vol_ax_cor_sag(vol.cpu().numpy(), vol_res, window=2)
        else:
            _fig, _axs = plot_vol_ax_cor_sag(vol.cpu().numpy(), vol_res)

        # Draw the figure first
        _fig.canvas.draw()

        # Now we can save it to a numpy array.
        img_data = np.frombuffer(_fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_data = img_data.reshape(_fig.canvas.get_width_height()[::-1] + (4,))
        if nb_rows > 1:
            axs[c_row, c_col].imshow(img_data)
            axs[c_row, c_col].axis("off")
            axs[c_row, c_col].set_xticks([])
            axs[c_row, c_col].set_yticks([])
        else:
            axs[c_col].imshow(img_data)
            axs[c_col].axis("off")
            axs[c_col].set_xticks([])
            axs[c_col].set_yticks([])

        if lbl is not None:
            if nb_rows > 1:
                axs[c_row, c_col].set_title(lbl)
            else:
                continue

        if histogram:
            hist = compute_histo(vol)
            axs[1, c_col].plot(hist.bin_edges[:-1], hist.hist, color="r")

    plt.figure(fig.number)
    plt.show()

    if img_save_path is not None:
        plt.savefig(img_save_path, dpi=300)

    plt.close()
    return fig, axs


def plot_vol_ax_cor_sag(
    vol: np.ndarray,
    vol_res: Sequence[float],
    x=0.5,
    y=0.5,
    z=0.5,
    level=0,
    window=2000,
    cmap="gray",
    show_colorbar=True,
    show_slice_marker=True,
    marker_width=1.5,
    marker_colors: Sequence[Sequence[float]] = None,
    dashes=(5, 10),
    compact_cor_sag=False,
    figsize=(10, 10),
    ax=None,
):
    """plot three orthogonal views for the given volume
    https://github.com/fpschill/ac3t/blob/master/ac3t/visualization/plot.py

    :param vol: volume to plot (D,H,W)
    :param vol_res: voxel size (x,y,z)
    :param x-slice position: , defaults to 0.5
    :type x: float, optional
    :param y: y-slice position, defaults to 0.5
    :type y: float, optional
    :param z: z-slice position, defaults to 0.5
    :type z: float, optional
    :param level: intensity level for plotting, use with 'window'. defaults to 0
    :param window: window of the intensity values to plot, vmin/vmax=level+-window/2. defaults to 2000
    :param cmap: colormap for the plot, see matplotlib colormaps for more info. defaults to 'gray'
    :param show_colorbar: if True, a colorbar is added to the plot, defaults to False
    :param show_slice_marker: shows a marker to indicate the slice position of the slices, defaults to True
    :param marker_width: width of the slice marker line, defaults to 1.5
    :param marker_colors: color for the slice markers, defaults to [(1,0,0),(0,1,0),(0,0.4,1)]
    :param dashes: dash pattern for slice markers (see matplotlib for more info), defaults to (5,10)
    :param compact_cor_sag: if True, the coronal and saggital slices are both display under the axial
                            view instead of the saggital slice being to the right, defaults to False
    :param figsize: figure size, ignored if ax is passed, defaults to (10,10)
    :param ax: the axis object to use for the plot. If None: a new figure is create automaticaqlly. Defaults to None
    :return: fig, ax: the matplotlib figure and axes object of the plot
    """
    if vol.ndim != 3:
        raise ValueError("Volume must be 3D")

    slice_idx = np.multiply(vol.shape, [z, y, x]).astype(int)

    # make pixels square
    axial = vol[slice_idx[0], :, :]
    axial_aspect = vol_res[0] / vol_res[1]
    axial = Image.fromarray(axial).resize(
        (int(axial.shape[1]), int(axial.shape[0] * 1 / axial_aspect)), Image.BILINEAR
    )
    axial = np.array(axial)

    coronal = vol[:, slice_idx[1], :]
    coronal_aspect = vol_res[0] / vol_res[2]
    coronal = Image.fromarray(coronal).resize(
        (int(coronal.shape[1]), int(coronal.shape[0] * 1 / coronal_aspect)),
        Image.BILINEAR,
    )
    coronal = np.array(coronal)

    saggital = vol[:, :, slice_idx[2]]
    saggital_aspect = vol_res[1] / vol_res[2]
    saggital = Image.fromarray(saggital).resize(
        (int(saggital.shape[1]), int(saggital.shape[0] * 1 / saggital_aspect)),
        Image.BILINEAR,
    )
    saggital = np.array(saggital)

    padding = 1  # empty pixels between different views

    if (
        compact_cor_sag
    ):  # plot coronal and saggital slices side-by-side under axial view
        w = axial.shape[1]
        wc = coronal.shape[1]
        ws = saggital.shape[1]
        cor_w = int(wc / (wc + ws) * w - padding / 2)
        sag_w = int(w - cor_w - padding / 2)

        coronal = np.array(
            Image.fromarray(coronal).resize(
                (cor_w, int(cor_w * coronal.shape[0] / coronal.shape[1])),
                Image.BILINEAR,
            )
        )
        saggital = np.array(
            Image.fromarray(saggital).resize((sag_w, coronal.shape[0]), Image.BILINEAR)
        )

    # assemble all views in combined image
    combined = np.ma.zeros(
        (
            axial.shape[0] + coronal.shape[0] + padding,
            coronal.shape[1] + saggital.shape[1] + padding,
        )
    )
    combined.mask = np.ones(combined.shape)
    combined[0 : axial.shape[0], 0 : axial.shape[1]] = axial
    combined[
        axial.shape[0] + padding : axial.shape[0] + padding + coronal.shape[0],
        0 : coronal.shape[1],
    ] = coronal
    combined[
        axial.shape[0] + padding : axial.shape[0] + padding + saggital.shape[0],
        coronal.shape[1] + padding :,
    ] = saggital

    if axial.shape[0] == 256:
        # scale from -1000 - 1000 to 0-1
        combined_scale01 = (combined + 1000) / 2000
        # make the image double the size
        combined_scale01_big = zoom(combined_scale01, 2)
        # scale from 0-1 to -1000 - 1000
        combined = combined_scale01_big * 2000 - 1000
        axial = zoom(axial, 2)
        coronal = zoom(coronal, 2)
        saggital = zoom(saggital, 2)
        padding *= 2
        marker_width *= 2

    if show_colorbar:
        # add colorbar to the right of the combined image
        # Create a new figure
        fig_colorbar, ax_colorband = plt.subplots(
            figsize=(axial.shape[0] / 100, axial.shape[0] / 400)
        )
        fig_colorbar.subplots_adjust(bottom=0.5)

        # Create a colorbar in the figure
        cmap = plt.get_cmap(cmap)
        norm = plt.Normalize(vmin=level - window / 2, vmax=level + window / 2)
        cb1 = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax_colorband,
            orientation="horizontal",
        )
        cb1.ax.tick_params(labelsize=18)

        # Draw the figure
        canvas = FigureCanvas(fig_colorbar)
        canvas.draw()
        # Convert the figure to a numpy array
        width, height = fig_colorbar.get_size_inches() * fig_colorbar.get_dpi()
        colorbar_img_buffer = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        colorbar_img = colorbar_img_buffer.reshape((int(height), int(width), 4))
        img_data_gray = np.dot(colorbar_img[..., :3], [0.2989, 0.5870, 0.1140])
        # replace 254.9745 with 255 other values do not change
        img_data_gray = (
            np.where(img_data_gray >= img_data_gray.max(), 255, img_data_gray) / 255
        )

        # colorbar_img_bw = np.array(colorbar_img_bw)
        # scale from 0-1 to -1000 - 1000
        img_data_gray = img_data_gray * (2000) - 1000

        # fill top right corner with white
        combined[0 : axial.shape[0], axial.shape[1] + padding :] = (
            np.ones((axial.shape[0], axial.shape[1])) * 1000
        )
        # add colorbar to the right of the combined image
        combined[
            int(axial.shape[0] / 2) : int(axial.shape[0] / 2 + img_data_gray.shape[0]),
            int(axial.shape[1] + padding) : int(
                axial.shape[1] + padding + img_data_gray.shape[1]
            ),
        ] = img_data_gray

    if ax is None:
        fig, axs = plt.subplots(figsize=figsize, ncols=1, nrows=1, squeeze=False)
        ax = axs[0, 0]
    else:
        fig = ax.figure

    ax.imshow(
        combined, cmap=cmap, aspect=1, vmin=level - window / 2, vmax=level + window / 2
    )
    ax.axis(False)
    ax.grid(False)
    ax.set_xlim(0, combined.shape[1])
    ax.set_ylim(combined.shape[0], 0)

    if show_slice_marker:
        x_idx, y_idx, z_idx = (
            x * axial.shape[1],
            y * axial.shape[0],
            z * coronal.shape[0],
        )

        if marker_colors is None:
            marker_colors = [(1, 0, 0), (0, 1, 0), (0, 0.4, 1)]
        # x & y axial view
        ax.plot(
            [x_idx, x_idx],
            [0, axial.shape[0]],
            linewidth=marker_width,
            color=marker_colors[0],
            dashes=dashes,
        )
        ax.plot(
            [0, axial.shape[1]],
            [y_idx, y_idx],
            linewidth=marker_width,
            color=marker_colors[1],
            dashes=dashes,
        )

        # x coronal view
        x = x * coronal.shape[1]
        ax.plot(
            [x, x],
            [axial.shape[0] + padding, axial.shape[0] + padding + coronal.shape[0]],
            linewidth=marker_width,
            color=marker_colors[0],
            dashes=dashes,
        )

        # y saggital view
        y = coronal.shape[1] + padding + y * saggital.shape[1]
        ax.plot(
            [y, y],
            [axial.shape[0], axial.shape[0] + saggital.shape[0]],
            linewidth=marker_width,
            color=marker_colors[1],
            dashes=dashes,
        )
        # z-marker spans coronal & saggital view
        z = z_idx + axial.shape[0] + padding
        ax.plot(
            [0, coronal.shape[1] + padding + saggital.shape[1]],
            [z, z],
            linewidth=marker_width,
            color=marker_colors[2],
            dashes=dashes,
        )

    fig.tight_layout()
    plt.close()
    return fig, ax
