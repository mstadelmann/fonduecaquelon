from typing import Any
from collections.abc import Sequence
import numpy as np
import progressbar
import termplotlib as tpl  # this requires GNUplot!
from colorama import init
from termcolor import colored


GLOB_COLORAMA_INITIALIZED: bool | None = None
GLOBAL_RANK: int = 0  # Global variable to track the rank of the process in distributed training

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


def startProgBar(
    nbstepts: int, message: str | None = None, is_active: bool = True
) -> CustomProgressBar:
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
    This function requires GNUplot to be installed."""

    print(
        f"\nProject: {experiment.project} | Experiment name: {experiment.experimentName}"
    )

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
        print("GNUplot is not available, loss is not plotted.")

    iprint(
        f"Training Loss: {experiment.trainLoss:.4f}, Validation Loss: {experiment.valLoss:.4f}"
    )


def iprint(msg: Any, distributed=False) -> None:
    """Info print: plots information string in green.
    In distributed training, only the main process (rank 0) prints.
    Set `distributed` to True to print from all processes."""
    cprint(msg, text_color="green",dist_print=distributed)


def wprint(msg: Any, distributed=False) -> None:
    """Warning print: plots warning string in yellow.
    In distributed training, only the main process (rank 0) prints.
    Set `distributed` to True to print from all processes."""
    cprint(msg, text_color="yellow",dist_print=distributed)


def eprint(msg: Any, distributed=False) -> None:
    """Error print: plots error string in red.
    In distributed training, only the main process (rank 0) prints.
    Set `distributed` to True to print from all processes."""
    cprint(msg, text_color="red",dist_print=distributed)


def cprint(
    msg: Any, text_color: str | None = None, bg_color: str | None = None, dist_print: bool =False
) -> None:
    """Prints a message with optional text and background color in the terminal."""

    # Only print if this is the main process in distributed training
    if not dist_print:
        global GLOBAL_RANK
        if GLOBAL_RANK != 0:
            return  
    
    global GLOB_COLORAMA_INITIALIZED
    if "GLOB_COLORAMA_INITIALIZED" not in globals():
        GLOB_COLORAMA_INITIALIZED = True
        init()

    supported_colors = ["red", "green", "yellow", "blue", "magenta"]
    supported_bg_colors = ["on_" + c for c in supported_colors]

    if text_color is not None and text_color not in supported_colors:
        raise ValueError(
            f"Text color {text_color} is not supported. Supported colors are {supported_colors}"
        )
    if bg_color is not None and bg_color not in supported_bg_colors:
        raise ValueError(
            f"Background color {bg_color} is not supported. Supported colors are {supported_bg_colors}"
        )

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
