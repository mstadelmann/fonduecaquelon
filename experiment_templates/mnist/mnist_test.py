import os
import sys

import torch
import torch.multiprocessing
import torch.nn.functional as F
from PIL import Image


from ui_functions import iprint, eprint, wprint, getIntInput
from img_func import showImg, showImg_cv
from ui_functions import startProgBar
from experiment import FCQmode
import matplotlib

# import matplotlib
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


def createEvaluator(experiment):
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
            " \n3: Manual image load. \n4: Homogenous scalar tensor.",
            [1, 4],
        )

    if tmode == 1:
        max_samples_to_print = getIntInput(
            "How many random samples do you want to show?\n"
        )

        labels_gt = []
        labels_pred = []

        for i, batch in enumerate(test_loader):
            if isinstance(batch, dict):
                inputs = batch["inputs"]
                targets = batch["targets"]
            else:
                inputs, targets = batch

            inputs = inputs.to(experiment.device)
            targets = targets.to(experiment.device)

            if i + 1 > max_samples_to_print:
                print("done testing..")
                sys.exit()

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

            print(f"GT: {targets.item()}")
            print(
                f"Raw Prediction: {[round(p, 3) for p in sum(pred.tolist(), [])]} \nSoftmax: {[round(sm, 3) for sm in sum(pred_sm.tolist(), [])]}"
                f" \nCurrent accuracy: {accuracy:.3f}"
            )

            showImg_cv(torch.squeeze(inputs))

    elif tmode == 2:
        print(f"Testset sample size: {experiment.data['MNIST'].n_test_samples}")
        pbar = startProgBar(experiment.data["MNIST"].n_test_samples, "evaluation...")

        labels_gt = []
        labels_pred = []
        cor_pred = 0

        for i, batch in enumerate(test_loader):
            if isinstance(batch, dict):
                inputs = batch["inputs"]
                targets = batch["targets"]
            else:
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

    elif tmode == 3:
        while True:
            user_input = input("Enter path to image (or 'q' to quit).")

            if user_input == "q":
                sys.exit()

            if not os.path.exists(user_input):
                print(f"Error: File {user_input} not found.")
            else:
                try:
                    img = Image.open(user_input)

                    # imgt = np.array(img)
                    # print("-----------------------------")
                    # print(f"shape: {imgt.shape}")
                    # print(f"np.min(imgt): {np.min(imgt)}")
                    # print(f"np.max(imgt) input: {np.max(imgt)}")
                    # print(f"np.mean(imgt): {np.mean(imgt)}")
                    # print("-----------------------------")
                    # print(f"np.min(imgt[:,:,0]): {np.min(imgt[:,:,0])}")
                    # print(f"np.max(imgt) input: {np.max(imgt[:,:,0])}")
                    # print(f"np.mean(imgt): {np.mean(imgt[:,:,0])}")
                    # print(f"np.std(imgt): {np.std(imgt[:,:,0])}")

                    img = experiment.trainTransformer_a(img).to(experiment.device)

                    if len(img.shape) == 3:
                        img = torch.unsqueeze(img, 0)

                    # print("-----------------------------")
                    # print("-----------------------------")
                    # print(f"\nimg.dtype: {img.dtype}")
                    # print(f"\nimg.shape: {img.shape}")
                    # print(f"torch.min(img) total: {torch.min(img)}")
                    # print(f"torch.max(img): {torch.max(img)}")
                    # print(f"torch.mean(img): {torch.mean(img)}")

                    # print("-----------------------------")
                    # print(f"torch.min(img[:,0,...]): {torch.min(img[:,0,...])}")
                    # print(f"torch.max(img): {torch.max(img[:,0,...])}")
                    # print(f"torch.mean(img): {torch.mean(img[:,0,...])}")

                    # print("-----------------------------")
                    # print(f"torch.min(img[:,1,...]): {torch.min(img[:,1,...])}")
                    # print(f"torch.max(img): {torch.max(img[:,1,...])}")
                    # print(f"torch.mean(img): {torch.mean(img[:,1,...])}")

                    # print("-----------------------------")
                    # print(f"torch.min(img[:,2,...]): {torch.min(img[:,2,...])}")
                    # print(f"torch.max(img): {torch.max(img[:,2,...])}")
                    # print(f"torch.mean(img): {torch.mean(img[:,2,...])}")

                    print("--------------------------------------")
                    print(f"img shape forwarded: {img.shape}")
                    pred = experiment.model(img)
                    print(f"pred: {pred}")
                    pred_sm = F.softmax(pred, dim=1)
                    pred_am = pred_sm.argmax()
                    print(
                        f"Prediction: {pred.tolist()} \nSoftmax: {pred_sm.tolist()}"
                        f" \nPredicted Label: {experiment.class_names[pred_am.item()]}"
                    )
                    showImg(img)

                except Exception:
                    print(f"Error loading image {user_input}")

    if tmode == 4:
        in_scalar = getIntInput("Int input value?")
        in_tensor = in_scalar * torch.ones(
            (1, 3, experiment.net_input_size[0], experiment.net_input_size[1])
        )

        print(in_tensor.shape)

        pred = experiment.model(in_tensor.to(experiment.device))
        pred_sm = F.softmax(pred, dim=1)
        pred_am = pred_sm.argmax()

        print(f"Prediction: {pred.tolist()} \nSoftmax: {pred_sm.tolist()}")

    return accuracy
