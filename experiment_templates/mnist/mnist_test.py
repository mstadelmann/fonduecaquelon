"""MNIST test evaluation routines for fonduecaquelon experiments."""

import matplotlib
import torch
import torch.multiprocessing
import torch.nn.functional as F
from fdq.misc import showImg_cv
from fdq.ui_functions import getIntInput, startProgBar


# this is to fix
# RuntimeError: Too many open files. Communication with the workers is no longer possible.
torch.multiprocessing.set_sharing_strategy("file_system")

# to fix
# UserWarning: Matplotlib is currently using agg, which is a non-GUI backend
# matplotlib.use("TkAgg")  # --> this fails in CI!

# Dynamically set the backend
try:
    matplotlib.use("TkAgg")  # Use TkAgg if available
except ImportError:
    matplotlib.use("Agg")  # Fallback to Agg for headless environments


def interactive_test(experiment, test_loader):
    """Interactively tests the experiment model on the MNIST test set and displays predictions and accuracy.

    Args:
        experiment: The experiment object containing the model and device information.
        test_loader: DataLoader for the MNIST test dataset.

    Returns:
        accuracy (float or None): The computed accuracy of the model, or None if no samples were tested.
    """
    max_samples_to_print = getIntInput(
        "How many test batches do you want to check?\n", drange=[1, 1000]
    )
    labels_gt = []
    labels_pred = []
    total_accuracy = []

    for i, batch in enumerate(test_loader):
        inputs, targets = batch
        inputs = inputs.to(experiment.device)
        targets = targets.to(experiment.device)

        if i + 1 > max_samples_to_print:
            print("done testing..")
            break

        print("--------------------------------------")
        print(f"img shape fed to net: {inputs.shape}")

        pred = experiment.models["simpleNet"](inputs)
        pred_sm = F.softmax(pred, dim=1)
        pred_am = pred_sm.argmax()
        labels_gt.append(targets.item())
        labels_pred.append(pred_am.item())

        cor_pred = 0
        for lp, lg in zip(labels_pred, labels_gt):
            if lp == lg:
                cor_pred += 1

        accuracy = cor_pred / len(labels_pred)
        total_accuracy.append(accuracy)

        print(f"GT: {targets.item()}")
        print(
            f"Raw Prediction: {[round(p, 3) for p in sum(pred.tolist(), [])]} \nSoftmax: {[round(sm, 3) for sm in sum(pred_sm.tolist(), [])]}"
            f" \nCurrent accuracy: {accuracy:.3f}"
        )

        showImg_cv(torch.squeeze(inputs))

    accuracy = sum(total_accuracy) / len(total_accuracy) if total_accuracy else None

    return accuracy


def auto_test(experiment, test_loader):
    """Automatically tests the experiment model on the MNIST test set and prints the overall accuracy.

    Args:
        experiment: The experiment object containing the model and device information.
        test_loader: DataLoader for the MNIST test dataset.

    Returns:
        accuracy (float): The computed accuracy of the model.
    """
    print(f"Testset sample size: {experiment.data['MNIST'].n_test_samples}")
    pbar = startProgBar(experiment.data["MNIST"].n_test_samples, "evaluation...")

    labels_gt = []
    labels_pred = []
    cor_pred = 0

    for i, batch in enumerate(test_loader):
        inputs, targets = batch
        inputs = inputs.to(experiment.device)
        targets = targets.to(experiment.device)

        pbar.update(i)
        pred = experiment.models["simpleNet"](inputs)
        pred_sm = F.softmax(pred, dim=1)
        pred_am = pred_sm.argmax()
        labels_gt.append(targets)
        labels_pred.append(pred_am.item())

        if labels_gt[-1] == labels_pred[-1]:
            cor_pred += 1

    pbar.finish()

    accuracy = cor_pred / len(labels_pred)
    print(f"\nTotal accuracy: {accuracy}, Nb samples: {len(labels_pred)}")
    return accuracy


def fdq_test(experiment):
    """Evaluates the given experiment on the MNIST dataset using different test modes.

    Args:
        experiment: An experiment object containing model, data, and configuration.

    Returns:
        accuracy (float or None): The computed accuracy of the model, or None if not applicable.
    """
    test_loader = experiment.data["MNIST"].test_data_loader

    accuracy = None

    if experiment.exp_def.data.MNIST.args.test_batch_size != 1:
        raise ValueError(
            "Error: Test batch size must be 1 for this experiment. Please change the experiment file."
        )

    if experiment.mode.op_mode.unittest or experiment.inargs.test_model_auto:
        # no interactive for test experiments
        tmode = 2

    else:
        tmode = getIntInput(
            "\nSelect Testmode:\n1: Interactive test with predefined data.\n2: Automatic with predefined data."
            " \n3: Homogenous scalar tensor.",
            [1, 3],
        )

    if tmode == 1:
        accuracy = interactive_test(experiment, test_loader)

    elif tmode == 2:
        accuracy = auto_test(experiment, test_loader)

    if tmode == 3:
        in_scalar = getIntInput("Int input value?", drange=[0, 5000])
        in_tensor = in_scalar * torch.ones(
            (1, 3, experiment.net_input_size[0], experiment.net_input_size[1])
        )

        print(in_tensor.shape)

        pred = experiment.model(in_tensor.to(experiment.device))
        pred_sm = F.softmax(pred, dim=1)
        pred_am = pred_sm.argmax()

        print(
            f"Prediction: {pred.tolist()} \nSoftmax: {pred_sm.tolist()} \nArgmax: {pred_am.tolist()}"
        )

    return accuracy
