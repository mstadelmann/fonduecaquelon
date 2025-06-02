import sys
import os
import copy
import json
import random
from datetime import datetime
from typing import Any
from collections.abc import Callable, Iterator

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import wandb

from fdq.ui_functions import iprint, wprint, eprint


class FCQmode:
    """Class to manage operation and test modes for the FondueCaquelon project."""

    def __init__(self) -> None:
        """Initialize the FCQmode object with default operation and test modes, and create dynamic setters."""
        self._op_mode: str = "init"
        self.allowed_op_modes: list[str] = [
            "init",  # initial state
            "train",  # training mode
            "test",  # testing
            "unittest",  # running unit tests
        ]
        self._test_mode: str = "best"
        self.allowed_test_modes: list[str] = [
            "best",  # test best model from last experiment - DEFAULT!
            "last",  # test last trained model from last experiment
            "custom_last",  # test last model from selected experiment
            "custom_best",  # test best model from selected experiment
            "custom_path",  # test with manually defined model path
        ]
        self._locked: bool = False  # Flag to lock the mode when set to unittest

        # Dynamically create setter methods
        for mode in self.allowed_op_modes:
            setattr(self, mode, self._create_setter("_op_mode", mode))

        for mode in self.allowed_test_modes:
            setattr(self, mode, self._create_setter("_test_mode", mode))

    def __repr__(self) -> str:
        """Return the string representation of the FCQmode object."""
        if self._op_mode == "test":
            return f"<{self.__class__.__name__}: {self._op_mode} / {self._test_mode}>"
        return f"<{self.__class__.__name__}: {self._op_mode}>"

    def _create_setter(self, attribute: str, value: str) -> Callable[[], None]:
        def setter() -> None:
            if self._locked and attribute == "_op_mode":
                wprint("Unittest mode is locked. Cannot change mode.")
            else:
                if value == "unittest" and attribute == "_op_mode":
                    self._locked = True
                setattr(self, attribute, value)

        return setter

    @property
    def op_mode(self) -> Any:
        class OpMode:
            """Helper class to provide boolean properties for each allowed operation mode."""

            def __init__(self, parent: "FCQmode") -> None:
                self.parent = parent

            def __repr__(self) -> str:
                return f"<{self.__class__.__name__}: {self.parent._op_mode}>"

            def __getattr__(self, name: str) -> bool:
                if name in self.parent.allowed_op_modes:
                    return self.parent._op_mode == name
                raise AttributeError(f"'OpMode' object has no attribute '{name}'")

        return OpMode(self)

    @property
    def test_mode(self) -> Any:
        class TestMode:
            """Helper class to provide boolean properties for each allowed test mode."""

            def __init__(self, parent: "FCQmode") -> None:
                self.parent = parent

            def __repr__(self) -> str:
                return f"<{self.__class__.__name__}: {self.parent._test_mode}>"

            def __getattr__(self, name: str) -> bool:
                if name in self.parent.allowed_test_modes:
                    return self.parent._test_mode == name
                raise AttributeError(f"'TestMode' object has no attribute '{name}'")

        return TestMode(self)


def recursive_dict_update(d_parent: dict, d_child: dict) -> dict:
    """Recursively update the parent dictionary with values from the child dictionary, merging nested dictionaries."""
    for key, value in d_child.items():
        if (
            isinstance(value, dict)
            and key in d_parent
            and isinstance(d_parent[key], dict)
        ):
            recursive_dict_update(d_parent[key], value)
        else:
            d_parent[key] = value

    return copy.deepcopy(d_parent)


class DictToObj:
    """A class that converts a dictionary into an object, recursively handling nested dictionaries."""

    def __init__(self, dictionary: dict) -> None:
        """Initialize the object by converting a dictionary into attributes, recursively handling nested dictionaries."""
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = DictToObj(value)
            setattr(self, key, value)

    def __getattr__(self, name: str) -> Any:
        """Return None if the requested attribute is not found."""
        return None

    def __repr__(self) -> str:
        """Return the string representation of the object."""
        keys = ", ".join(self.__dict__.keys())
        return f"<{self.__class__.__name__}: {keys}>"

    def __str__(self) -> str:
        """Return the string representation of the object."""
        return self.__repr__()

    def __iter__(self) -> Iterator:
        """Return an iterator over the object's dictionary items."""
        return iter(self.__dict__.items())

    def keys(self) -> Any:
        return self.__dict__.keys()

    def items(self) -> Any:
        return self.__dict__.items()

    def values(self) -> Any:
        return self.__dict__.values()

    def to_dict(self) -> dict:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, DictToObj):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        res = getattr(self, key)
        if res is None:
            return default
        return res

    def set(self, key: str, value: Any) -> None:
        if isinstance(value, dict):
            value = DictToObj(value)
        setattr(self, key, value)


def replace_tilde_with_abs_path(d: dict) -> None:
    """Fix user paths.

    Recursively traverse a dictionary and replace string values starting with "~/"
    with their absolute paths.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            replace_tilde_with_abs_path(value)
        elif isinstance(value, str) and value.startswith("~/"):
            d[key] = os.path.expanduser(value)


def get_subset(dataset: Any, subset_ratio: float) -> Any:
    """Return a random subset of the dataset according to the given ratio.

    Args:
        dataset: The dataset to subset.
        subset_ratio (float): The ratio of the dataset to include in the subset (0 < subset_ratio <= 1).

    Returns:
        Subset of the dataset if subset_ratio < 1, otherwise the original dataset.
    """
    if subset_ratio >= 1:
        return dataset
    n_dataset = len(dataset)
    n_subset = int(n_dataset * subset_ratio)
    new_set, _ = random_split(dataset, [n_subset, n_dataset - n_subset])
    return new_set


def print_nb_weights(experiment: Any) -> None:
    """Print the number of parameters for each model in the experiment."""
    for model_name, model in experiment.models.items():
        iprint("----------------------------------")
        iprint(f"Model: {model_name}")
        nbp = sum(p.numel() for p in model.parameters())
        iprint(f"nb parameters: {nbp / 1e6:.2f}M")
        iprint(f"Using Float32, This will require {nbp * 4 / 1e9:.3f} GB of memory.")
        iprint("----------------------------------")


def remove_file(path: str | None) -> None:
    """Remove the file at the given path if it exists."""
    if path is not None:
        try:
            os.remove(path)
        except FileNotFoundError:
            eprint(f"{path} does not exist!")


def store_processing_infos(experiment: Any) -> None:
    """Store experiment information to results directory."""
    experiment.run_info = collect_processing_infos(experiment=experiment)
    info_path = os.path.join(experiment.results_dir, "info.json")

    with open(info_path, "w", encoding="utf8") as write_file:
        json.dump(experiment.run_info, write_file, indent=4, sort_keys=True)


def collect_processing_infos(experiment: Any | None = None) -> dict:
    """Collect and return processing information about the current experiment and environment."""
    try:
        sysname = os.uname()[1]
    except AttributeError:
        sysname = None

    try:
        username = os.getlogin()
    except OSError:
        username = None

    try:
        create_dt_string = experiment.creation_time.strftime("%Y%m%d_%H_%M_%S")
    except (AttributeError, KeyError):
        create_dt_string = None

    try:
        stop_dt_string = experiment.finish_time.strftime("%Y%m%d_%H_%M_%S")
    except (AttributeError, KeyError):
        stop_dt_string = None

    try:
        td = experiment.run_time
        run_t_string = f"days: {td.days}, hours: {td.seconds // 3600}, minutes: {td.seconds % 3600 / 60.0:.0f}"
    except (AttributeError, KeyError):
        run_t_string = None

    data: dict = {
        "User": username,
        "System": sysname,
        "Python V.": sys.version,
        "Torch V.": torch.__version__,
        "Cuda V.": torch.version.cuda,
        "start_datetime": create_dt_string,
        "end_datetime": stop_dt_string,
        "total_runtime": run_t_string,
        "epochs": f"{experiment.current_epoch + 1} / {experiment.nb_epochs}",
        "last_update": datetime.now().strftime("%Y%m%d_%H_%M_%S"),
        "best_train_loss_epoch": experiment.new_best_train_loss_ep_id,
        "best_val_loss_epoch": experiment.new_best_val_loss_ep_id,
    }

    if experiment.early_stop_detected:
        data["early_stop_reason"] = experiment.early_stop_detected

    if experiment.is_slurm:
        data["slurm_job_id"] = experiment.slurm_job_id

    if experiment.inargs.resume_path is not None:
        data["job_continuation"] = True
        data["job_continuation_chpt_path"] = experiment.inargs.resume_path
        data["start_epoch"] = experiment.start_epoch
    else:
        data["job_continuation"] = False

    try:
        # add nb model parameters to info file
        model_weights = sum(p.numel() for p in experiment.model.parameters())
        data["Number of model parameters"] = f"{model_weights / 1e6:.2f}M"
    except AttributeError:
        pass

    try:
        # add dataset key-numbers to info file
        data["dataset_key_numbers"] = {
            "Nb samples train": experiment.trainset_size,
            "Train subset": experiment.train_subset,
            "Nb samples val": experiment.valset_size,
            "Validation subset": experiment.val_subset,
            "Nb samples test": experiment.testset_size,
            "Test subset": experiment.test_subset,
            "Validation set is a subset of the training set.": experiment.valset_is_train_subset,
            "Validation subset ratio": experiment.val_from_train_ratio,
        }
    except AttributeError:
        pass

    return data


def avoid_nondeterministic(experiment: Any, seed_overwrite: int = 0) -> None:
    """Avoid nondeterministic behavior.

    https://pytorch.org/docs/stable/notes/randomness.html

    The cuDNN library, used by CUDA convolution operations, can be a source of
    nondeterminism across multiple executions of an application. When a cuDNN
    convolution is called with a new set of size parameters, an optional feature
    can run multiple convolution algorithms, benchmarking them to find the fastest one.
    Then, the fastest algorithm will be used consistently during the rest of the process
    for the corresponding set of size parameters. Due to benchmarking noise and different
    hardware, the benchmark may select different algorithms on subsequent runs, even on the same machine.
    """
    if experiment.random_seed is None:
        experiment.random_seed = seed_overwrite
        random.seed(experiment.random_seed)
        np.random.seed(experiment.random_seed)
        torch.manual_seed(experiment.random_seed)

    torch.use_deterministic_algorithms(mode=True)


def save_train_history(experiment: Any) -> None:
    """Save training history to json and pdf."""
    try:
        out_json = os.path.join(experiment.results_dir, "history.json")
        out_pdf = os.path.join(experiment.results_dir, "history.pdf")
        loss_hist = {
            "train": experiment.trainLoss_per_ep,
            "validation": experiment.valLoss_per_ep,
        }

        with open(out_json, "w", encoding="utf8") as outfile:
            json.dump(loss_hist, outfile, default=float, indent=4, sort_keys=True)

        plt.rcParams.update({"font.size": 8})
        fig1, ax1 = plt.subplots()
        ax1.set_title("Loss")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
        loss_a = np.array(experiment.trainLoss_per_ep)
        val_loss_a = np.array(experiment.valLoss_per_ep)
        epochs = range(experiment.start_epoch, experiment.start_epoch + len(loss_a))
        ax1.plot(epochs, loss_a, color="red", label="train")
        ax1.plot(epochs, val_loss_a, color="green", label="val")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True, prune="both", nbins=10))
        ax1.legend(loc="best")
        fig1.savefig(out_pdf)
        plt.close(fig1)

    except (OSError, AttributeError, ValueError):
        wprint("Error - unable to store training history!")


def showImg_cv(tensor_image: torch.Tensor, window_name: str = "Image") -> None:
    """Displays a PyTorch tensor image using OpenCV.

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


def init_tensorboard(experiment: Any) -> None:
    """Initialize TensorBoard for the experiment and provide usage instructions."""
    if not experiment.useTensorboard:
        return
    experiment.tb_writer = SummaryWriter(f"{experiment.results_dir}/tb/")
    experiment.tb_graph_stored = False
    iprint("-------------------------------------------------------")
    iprint("Start tensorboard typing:")
    iprint(f"tensorboard --logdir={experiment.results_dir}/tb/ --bind_all")
    iprint("-------------------------------------------------------")


def add_graph(experiment: Any) -> None:
    """Add the model graph to TensorBoard using a sample input from the training data loader."""
    try:
        dummy_input = None
        sample = next(iter(experiment.data[list(experiment.data)[0]].train_data_loader))
        if isinstance(sample, tuple | list):
            dummy_input = sample[0]
        elif isinstance(sample, dict):
            dummy_input = next(iter(sample.values()))

        for model_name, _ in experiment.exp_def.models:
            experiment.tb_writer.add_graph(experiment.models[model_name], dummy_input.to(experiment.device))
            experiment.tb_graph_stored = True
    except (StopIteration, AttributeError, KeyError, TypeError):
        wprint("Unable to add graph to Tensorboard.")


@torch.no_grad()
def _log_tb_images(experiment: Any, images: list | dict | None) -> None:
    if images is not None:
        if not isinstance(images, list):
            if isinstance(images, dict):
                images = [images]
            else:
                raise ValueError(
                    "Images must be a dictionary or a list of dictionaries."
                )

        for image in images:
            img = image["data"]
            dataformat = image.get("dataformats", "NCHW")

            experiment.tb_writer.add_images(
                tag=image["name"],
                img_tensor=img,
                global_step=experiment.current_epoch,
                dataformats=dataformat,
            )


@torch.no_grad()
def save_tensorboard(
    experiment: Any,
    images: list | dict | None = None,
    scalars: dict | None = None,
    text: dict | None = None,
) -> None:
    """Log images and scalars to tensorboard.

    Scalars: {name: value}
    Train and Val loss are logged automatically.
    Images are expected to be in shape [B,C,D,H,W]
    """
    if not experiment.useTensorboard:
        return

    if experiment.tb_writer is None:
        init_tensorboard(experiment)

    # add model to tensorboard
    if not experiment.tb_graph_stored:
        add_graph(experiment)

    if scalars is None:
        scalars = {}
    elif not isinstance(scalars, dict):
        raise ValueError("Scalars must be a dictionary.")
    scalars["train_loss"] = experiment.trainLoss
    scalars["val_loss"] = experiment.valLoss

    for scalar_name, scalar_value in scalars.items():
        experiment.tb_writer.add_scalar(
            scalar_name, scalar_value, experiment.current_epoch
        )

    if text is not None:
        if not isinstance(text, dict):
            raise ValueError("Text must be a dictionary.")
        for text_name, text_value in text.items():
            experiment.tb_writer.add_text(
                text_name, text_value, experiment.current_epoch
            )

    _log_tb_images(experiment, images)


def init_wandb(experiment: Any) -> bool:
    """Initialize weights and biases."""
    if experiment.exp_def.store.wandb_project is None:
        raise ValueError(
            "Wandb project name is not set. Please set it in the experiment definition."
        )
    if experiment.exp_def.store.wandb_entity is None:
        raise ValueError(
            "Wandb entity name is not set. Please set it in the experiment definition."
        )
    if experiment.exp_def.store.wandb_key is None:
        raise ValueError(
            "Wandb key is not set. Please set it in the experiment definition."
        )

    slurm_str = ""
    if experiment.is_slurm:
        if experiment.previous_slurm_job_id is not None:
            slurm_str = (
                f"__{experiment.previous_slurm_job_id}->{experiment.slurm_job_id}"
            )
        else:
            slurm_str = f"__{experiment.slurm_job_id}"

    dt_string = experiment.creation_time.strftime("%Y%m%d_%H%M%S")
    wandb_name = f"{dt_string}__{experiment.experimentName[:20]}__{experiment.funky_name}{slurm_str}"

    try:
        wandb.login(key=experiment.exp_def.store.wandb_key)
        wandb.init(
            project=experiment.exp_def.store.wandb_project,
            entity=experiment.exp_def.store.wandb_entity,
            name=wandb_name,
            config=experiment.exp_file,
        )
        experiment.wandb_initialized = True
        iprint(f"Init Wandb -  log path: {wandb.run.dir}")
        return True

    except (
        wandb.errors.UsageError,
        wandb.errors.CommError,
        AttributeError,
        ValueError,
    ) as e:
        eprint("Unable to initialize wandb!")
        eprint(f"Error: {e}")
        experiment.useWandb = False
        return False


@torch.no_grad()
def _log_wandb_images(images: list | dict | None) -> None:
    if images is not None:
        if not isinstance(images, list):
            if isinstance(images, dict):
                images = [images]
            else:
                raise ValueError(
                    "Images must be a dictionary or a list of dictionaries."
                )

        for image in images:
            captions = image.get("captions", None)

            if image.get("data") is not None:
                img = image["data"]
            elif image.get("path") is not None:
                img = image["path"]
            else:
                continue

            wandb.log({image["name"]: wandb.Image(img, caption=captions)})


@torch.no_grad()
def save_wandb(
    experiment: Any,
    images: list | dict | None = None,
    scalars: dict | None = None,
) -> None:
    """Track experiment data with weights and biases.

    Args:
        experiment (class): experiment object.
        images (list): stacked list of [[image_name, image_data]].
        scalars (dict, optional): Dictionary of scalar values to log.

    Returns:
        type: Description of returned object.
    """
    if not experiment.useWandb:
        return

    if experiment.wandb_initialized is False:
        if not init_wandb(experiment):
            return

    if scalars is None:
        scalars = {}
    elif not isinstance(scalars, dict):
        raise ValueError("Scalars must be a dictionary.")
    scalars["train_loss"] = experiment.trainLoss
    scalars["val_loss"] = experiment.valLoss
    scalars["epoch"] = experiment.current_epoch

    wandb.log(scalars)
    _log_wandb_images(images)
