"""This module defines the training procedure for the MNIST test experiment."""

import torch
import torchvision
from fdq.experiment import fdqExperiment
from fdq.ui_functions import show_train_progress, startProgBar, iprint
from fdq.misc import print_nb_weights
from fdq.img_func import (
    save_wandb_loss,
)


def train(experiment: fdqExperiment) -> None:
    """Train the model using the provided experiment configuration.

    Args:
        experiment (fdqExperiment): The experiment object containing data loaders, models, and training configurations.
    """
    iprint("Default training")
    print_nb_weights(experiment)

    data = experiment.data["MNIST"]
    model = experiment.models["simpleNet"]

    for epoch in range(experiment.start_epoch, experiment.nb_epochs):
        experiment.current_epoch = epoch
        iprint(f"\nEpoch: {epoch + 1} / {experiment.nb_epochs}")

        train_loss_sum = 0.0
        val_loss_sum = 0.0
        model.train()
        pbar = startProgBar(data.n_train_samples, "training...")

        for nb_tbatch, batch in enumerate(data.train_data_loader):
            pbar.update(nb_tbatch * experiment.exp_def.data.MNIST.args.train_batch_size)

            inputs, targets = batch

            inputs = inputs.to(experiment.device).type(torch.float32)
            targets = targets.to(experiment.device)

            if experiment.useAMP:
                device_type = (
                    "cuda" if experiment.device == torch.device("cuda") else "cpu"
                )

                with torch.autocast(device_type=device_type, enabled=True):
                    output = experiment.model(inputs)
                    train_loss_tensor = (
                        experiment.losses["cp"](output, targets)
                        / experiment.gradacc_iter
                    )

                experiment.scaler.scale(train_loss_tensor).backward()

            else:
                output = model(inputs)
                train_loss_tensor = (
                    experiment.losses["cp"](output, targets) / experiment.gradacc_iter
                )

                train_loss_tensor.backward()

            experiment.update_gradients(
                b_idx=nb_tbatch, loader_name="MNIST", model_name="simpleNet"
            )

            train_loss_sum += train_loss_tensor.detach().item()

        experiment.trainLoss = train_loss_sum / len(data.train_data_loader.dataset)
        pbar.finish()

        model.eval()

        pbar = startProgBar(data.n_val_samples, "validation...")

        for nb_vbatch, batch in enumerate(data.val_data_loader):
            experiment.current_val_batch = nb_vbatch
            pbar.update(nb_vbatch * experiment.exp_def.data.MNIST.args.val_batch_size)

            inputs, targets = batch

            with torch.no_grad():
                inputs = inputs.to(experiment.device)
                output = model(inputs)
                targets = targets.to(experiment.device)
                val_loss_tensor = experiment.losses["cp"](output, targets)

            val_loss_sum += val_loss_tensor.detach().item()
        experiment.valLoss = val_loss_sum / len(data.val_data_loader.dataset)

        pbar.finish()

        # Log the image grid
        img_grid = {
            "name": "inputs",
            "data": torchvision.utils.make_grid(inputs),
            "dataformats": "CHW",
        }

        # Log text predictions
        _, preds = torch.max(output, 1)
        log_txt = {
            f"Predictions/image_{idx}": f"Predicted: {preds[idx].item()}, True: {targets[idx].item()}"
            for idx in range(len(inputs))
        }

        experiment.finalize_epoch(log_images=img_grid, log_text=log_txt)

        if experiment.check_early_stop():
            break
