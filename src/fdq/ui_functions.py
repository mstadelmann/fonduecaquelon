from typing import Any
from collections.abc import Sequence
import numpy as np
import progressbar
import termplotlib as tpl  # this requires GNUplot!
from colorama import init
from termcolor import colored


GLOB_COLORAMA_INITIALIZED: bool | None = None
GLOBAL_RANK: int = 0
GNUPLOT_WARNING_SHOWN: bool = False


def set_global_rank(rank: int) -> None:
    """Sets the global rank for the process."""
    global GLOBAL_RANK
    GLOBAL_RANK = rank


def getIntInput(message: str, drange: Sequence[int]) -> int:
    """UI helper function to get an integer input from the user within a specified range."""
    tmode: str | int | None = None
    while not isinstance(tmode, int):
        tmode = input(message)
        try:
            tmode = int(tmode)

            if not drange[0] <= tmode <= drange[1]:
                print(f"Value must be between {drange[0]} and {drange[1]}.")
                tmode = None
        except ValueError:
            print("Enter integer number!")

    return tmode


def getYesNoInput(message: str) -> bool:
    """UI helper function to get yes/no input from user.

    Returns True if 'y' is entered, False otherwise.
    """
    tmode: str | int | None = None
    while not isinstance(tmode, str):
        tmode = input(message)
        if tmode.lower() not in ["y", "n"]:
            print("Enter 'y' or 'n'!")
            tmode = None

    return tmode.lower() == "y"


def getFloatInput(message: str, drange: Sequence[float]) -> float:
    """UI helper function to get a float input from the user within a specified range."""
    tmode: str | float | None = None
    while not isinstance(tmode, float):
        tmode = input(message)
        try:
            tmode = float(tmode)

            if not drange[0] <= tmode <= drange[1]:
                print(f"Value must be between {drange[0]} and {drange[1]}.")
                tmode = None
        except ValueError:
            print("Enter real number!")

    return tmode


class CustomProgressBar(progressbar.ProgressBar):
    """A customizable progress bar that can be activated or deactivated."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the CustomProgressBar, optionally setting its active state."""
        if "is_active" in kwargs:
            self.is_active: bool = kwargs["is_active"]
            del kwargs["is_active"]
        else:
            self.is_active = True
        super().__init__(*args, **kwargs)

    def start(self) -> None:
        if self.is_active:
            super().start()

    def update(self, value: int | None = None) -> None:
        if self.is_active:
            super().update(value)

    def finish(self) -> None:
        if self.is_active:
            super().finish()


def startProgBar(nbstepts: int, message: str | None = None, is_active: bool = True) -> CustomProgressBar:
    """Starts and returns a progress bar with the specified number of steps and optional message."""
    global GLOBAL_RANK
    if GLOBAL_RANK != 0:
        # show prog. bar on rank 0 process only
        is_active = False

    elif message is not None:
        print(message)

    pbar = CustomProgressBar(
        maxval=nbstepts,
        widgets=[progressbar.Bar("=", "[", "]"), " ", progressbar.Percentage()],
        is_active=is_active,
    )
    pbar.start()
    return pbar


def show_train_progress(experiment: Any) -> None:
    """Displays training and validation loss progress for the given experiment.

    This function requires GNUplot to be installed.
    """
    global GNUPLOT_WARNING_SHOWN
    iprint(f"\nProject: {experiment.project} | Experiment name: {experiment.experimentName}")

    try:
        trainLoss = experiment.trainLoss_per_ep
        valLoss = experiment.valLoss_per_ep

        trainLossA = np.array(trainLoss)
        x = np.linspace(1, len(trainLoss), len(trainLoss))

        fig = tpl.figure()
        fig.plot(x, trainLossA, label="trainL", width=50, height=15)
        if any(valLoss):
            valLossA = np.array(valLoss)
            fig.plot(x, valLossA, label="valL", width=50, height=15)
        fig.show()

    except Exception:
        if not GNUPLOT_WARNING_SHOWN:
            wprint("GNUplot is not available, loss is not plotted.")
            GNUPLOT_WARNING_SHOWN = True

    iprint(f"Training Loss: {experiment.trainLoss:.4f}, Validation Loss: {experiment.valLoss:.4f}")


def iprint(msg: Any, dist_print=False) -> None:
    """Info print: plots information string in green.

    In distributed training, only the main process (rank 0) prints.
    Set `dist_print` to True to print from all processes.
    """
    cprint(msg, text_color="green", dist_print=dist_print)


def wprint(msg: Any, dist_print=False) -> None:
    """Warning print: plots warning string in yellow.

    In distributed training, only the main process (rank 0) prints.
    Set `dist_print` to True to print from all processes.
    """
    cprint(msg, text_color="yellow", dist_print=dist_print)


def eprint(msg: Any, dist_print=False) -> None:
    """Error print: plots error string in red.

    In distributed training, only the main process (rank 0) prints.
    Set `dist_print` to True to print from all processes.
    """
    cprint(msg, text_color="red", dist_print=dist_print)


def cprint(
    msg: Any,
    text_color: str | None = None,
    bg_color: str | None = None,
    dist_print: bool = False,
) -> None:
    """Prints a message with optional text and background color in the terminal."""
    global GLOBAL_RANK
    if not dist_print:
        # DDP, only print on rank 0
        if GLOBAL_RANK != 0:
            return
    else:
        msg = f"[Rank {GLOBAL_RANK}] {msg}"

    global GLOB_COLORAMA_INITIALIZED
    if not GLOB_COLORAMA_INITIALIZED:
        GLOB_COLORAMA_INITIALIZED = True
        init()

    supported_colors = ["red", "green", "yellow", "blue", "magenta"]
    supported_bg_colors = ["on_" + c for c in supported_colors]

    if text_color is not None and text_color not in supported_colors:
        raise ValueError(f"Text color {text_color} is not supported. Supported colors are {supported_colors}")
    if bg_color is not None and bg_color not in supported_bg_colors:
        raise ValueError(f"Background color {bg_color} is not supported. Supported colors are {supported_bg_colors}")

    if text_color is None:
        print(msg)
    elif bg_color is None:
        print(
            colored(
                msg,
                text_color,
            )
        )
    else:
        print(colored(msg, text_color, bg_color))


def save_images(
    images: Any,
    save_path: str,
    titles: list[str] | None = None,
    figsize: tuple[int, int] = (12, 8),
    normalize: bool = True,
    cmap: str = "gray",
    save_all_batch: bool = False,
) -> None:
    """Save images as a subplot figure to the specified path.

    Args:
        images: torch.Tensor, numpy array, or list of these. Can be:
                - Single image: (H, W), (C, H, W), or (1, C, H, W)
                - Batch of images: (B, C, H, W) or (B, H, W)
                - List of images with any of the above formats
        save_path: Path where to save the image file
        titles: Optional list of titles for each subplot
        figsize: Figure size as (width, height)
        normalize: Whether to normalize images to [0, 1] range
        cmap: Colormap for grayscale images
        save_all_batch: If True, save each batch item as separate file.
                       If False, only save first image from batches > 1.
    """
    try:
        import matplotlib.pyplot as plt
        import torch
        import os
    except ImportError as e:
        eprint(f"Required packages not available for image saving: {e}")
        return

    # Convert input to list of numpy arrays
    def to_numpy(img):
        if torch.is_tensor(img):
            return img.detach().cpu().numpy()
        return np.array(img)

    def normalize_image(img):
        if normalize:
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
        return img

    def process_single_image(img):
        img = to_numpy(img)
        img = normalize_image(img)

        # Handle different input shapes
        if img.ndim == 4:
            if img.shape[0] == 1:  # (1, C, H, W)
                img = img.squeeze(0)
            elif img.shape[0] > 1:  # (B, C, H, W) with B > 1
                # This should not happen in process_single_image
                # Take first image from batch
                img = img[0]

        if img.ndim == 3:  # (C, H, W)
            if img.shape[0] == 1:  # Grayscale
                img = img.squeeze(0)
            elif img.shape[0] == 3:  # RGB
                img = np.transpose(img, (1, 2, 0))
            elif img.shape[0] > 3:  # Multi-channel, take first 3
                img = np.transpose(img[:3], (1, 2, 0))

        return img

    # Convert input to list of processed images
    image_list = []

    # First pass: check batch sizes and collect batch information
    batch_sizes = []
    batch_detected = False

    if isinstance(images, list | tuple):
        for img in images:
            img_np = to_numpy(img)
            if img_np.ndim == 4:
                batch_sizes.append(img_np.shape[0])
                if img_np.shape[0] > 1:
                    batch_detected = True
            else:
                batch_sizes.append(1)
    else:
        # Single tensor or array
        img = to_numpy(images)
        if img.ndim == 4:
            batch_sizes.append(img.shape[0])
            if img.shape[0] > 1:
                batch_detected = True
        else:
            batch_sizes.append(1)

    # Check consistency of batch sizes
    if batch_detected and len(set(batch_sizes)) > 1:
        eprint(f"Inconsistent batch sizes detected: {batch_sizes}. All images should have the same batch size.")
        return

    max_batch_size = max(batch_sizes) if batch_sizes else 1

    if isinstance(images, list | tuple):
        for img in images:
            img_np = to_numpy(img)
            if img_np.ndim == 4 and img_np.shape[0] > 1:
                # Batch of images
                if save_all_batch:
                    # Add all images from batch
                    for i in range(img_np.shape[0]):
                        image_list.append(process_single_image(img_np[i]))
                else:
                    # Only add first image from batch
                    if max_batch_size > 1:
                        iprint(
                            f"Batch size {max_batch_size} detected, saving only first image from each batch. Use save_all_batch=True to save all."
                        )
                    image_list.append(process_single_image(img_np[0]))
            else:
                image_list.append(process_single_image(img))
    else:
        # Single tensor or array
        img = to_numpy(images)
        if img.ndim == 4 and img.shape[0] > 1:
            # Batch of images
            if save_all_batch:
                # Save each image in batch as separate file
                base_path, ext = os.path.splitext(save_path)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                for i in range(img.shape[0]):
                    individual_path = f"{base_path}_batch_{i:03d}{ext}"
                    single_img = process_single_image(img[i])

                    # Create single image plot
                    fig, ax = plt.subplots(1, 1, figsize=figsize)
                    if single_img.ndim == 2 or (single_img.ndim == 3 and single_img.shape[2] == 1):
                        if single_img.ndim == 3:
                            single_img = single_img.squeeze(2)
                        ax.imshow(single_img, cmap=cmap)
                    else:
                        ax.imshow(single_img)

                    ax.axis("off")
                    if titles and i < len(titles):
                        ax.set_title(titles[i])

                    plt.tight_layout()
                    plt.savefig(individual_path, dpi=300, bbox_inches="tight")
                    plt.close()

                iprint(f"Saved {img.shape[0]} individual images from batch to {base_path}_batch_XXX{ext}")
                return
            else:
                # Only add first image from batch
                iprint(
                    f"Batch size {img.shape[0]} detected, saving only first image. Use save_all_batch=True to save all."
                )
                image_list.append(process_single_image(img[0]))
        else:
            image_list.append(process_single_image(img))

    if not image_list:
        eprint("No images to save")
        return

    # Create subplot layout
    n_images = len(image_list)
    if n_images == 1:
        rows, cols = 1, 1
    elif n_images <= 4:
        rows, cols = 2, 2
    elif n_images <= 9:
        rows, cols = 3, 3
    elif n_images <= 16:
        rows, cols = 4, 4
    else:
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))

    # Create the figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_images == 1:
        axes = [axes]
    elif rows > 1 or cols > 1:
        axes = axes.flatten()

    # Plot images
    for i, img in enumerate(image_list):
        ax = axes[i] if n_images > 1 else axes[0]

        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            # Grayscale
            if img.ndim == 3:
                img = img.squeeze(2)
            ax.imshow(img, cmap=cmap)
        else:
            # RGB or multi-channel
            ax.imshow(img)

        ax.axis("off")

        if titles and i < len(titles):
            ax.set_title(titles[i])

    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    iprint(f"Saved {n_images} image(s) to {save_path}")
