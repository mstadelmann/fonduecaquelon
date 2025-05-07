import json
import os
import torch
import random


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from ui_functions import wprint


def avoid_nondeterministic(experiment, seed_overwrite=0):
    """
    Avoid nondeterministic behavior.
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


def save_train_history(experiment):
    """save training history to json and pdf"""

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

    except Exception:
        wprint("Error - unable to store training history!")
