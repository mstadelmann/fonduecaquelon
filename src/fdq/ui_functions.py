import numpy as np
import progressbar
import termplotlib as tpl  # this requires GNUplot!
from colorama import init
from termcolor import colored


colorama_initialized = None


def getIntInput(message, drange=[0, 5000]):  # pylint: disable=W0102
    tmode = None
    while not isinstance(tmode, int):
        tmode = input(message)
        try:
            tmode = int(tmode)

            if not drange[0] <= tmode <= drange[1]:
                print(f"Value must be between {drange[0]} and {drange[1]}.")
                tmode = None
        except Exception:
            print("Enter integer number!")

    return tmode


def getYesNoInput(message):
    """
    UI helper function to get yes/no input from user.
    Returns True if 'y' is entered, False otherwise.
    """
    tmode = None
    while not isinstance(tmode, str):
        tmode = input(message)
        if tmode.lower() not in ["y", "n"]:
            print("Enter 'y' or 'n'!")
            tmode = None

    return tmode.lower() == "y"


def getFloatInput(message, drange=[-10, 10]):  # pylint: disable=W0102
    tmode = None
    while not isinstance(tmode, float):
        tmode = input(message)
        try:
            tmode = float(tmode)

            if not drange[0] <= tmode <= drange[1]:
                print(f"Value must be between {drange[0]} and {drange[1]}.")
                tmode = None
        except Exception:
            print("Enter real number!")

    return tmode


class CustomProgressBar(progressbar.ProgressBar):
    def __init__(self, *args, **kwargs):
        if "is_active" in kwargs:
            self.is_active = kwargs["is_active"]
            del kwargs["is_active"]
        else:
            self.is_active = True
        super().__init__(*args, **kwargs)

    def start(self):
        if self.is_active:
            super().start()

    def update(self, value=None):
        if self.is_active:
            super().update(value)

    def finish(self):
        if self.is_active:
            super().finish()


def startProgBar(nbstepts, message=None, is_active=True):
    if message is not None:
        print(message)
    pbar = CustomProgressBar(
        maxval=nbstepts,
        widgets=[progressbar.Bar("=", "[", "]"), " ", progressbar.Percentage()],
        is_active=is_active,
    )
    pbar.start()
    return pbar


def show_train_progress(experiment):
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


def iprint(msg):
    """
    Info print: plots information string in green.
    """
    cprint(msg, text_color="green")


def wprint(msg):
    """
    Warning print: plots warning string in yellow.
    """
    cprint(msg, text_color="yellow")


def eprint(msg):
    """
    Error print: plots error string in red.
    """
    cprint(msg, text_color="red")


def cprint(msg, text_color=None, bg_color=None):
    # pylint: disable=W0603
    global colorama_initialized
    if "colorama_initialized" not in globals():
        colorama_initialized = True
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
