"""This module defines the training procedure for the MNIST test experiment."""

import torch
import torchvision
from fdq.experiment import fdqExperiment
from fdq.ui_functions import startProgBar, iprint
from fdq.misc import print_nb_weights


def fdq_train(experiment: fdqExperiment) -> None:
    """Train the model using the provided experiment configuration.

    Args:
        experiment (fdqExperiment): The experiment object containing data loaders, models, and training configurations.
    """
    iprint("Default training")
    print_nb_weights(experiment)

    data = experiment.data["MNIST"]
    model = experiment.models["simpleNet"]
    device_type = "cuda" if experiment.device == torch.device("cuda") else "cpu"

    for epoch in range(experiment.start_epoch, experiment.nb_epochs):
        experiment.current_epoch = epoch
        iprint(f"\nEpoch: {epoch + 1} / {experiment.nb_epochs}")

        train_loss_sum = 0.0
        val_loss_sum = 0.0
        model.train()
        pbar = startProgBar(data.n_train_samples, "training...")

        for nb_batch, batch in enumerate(data.train_data_loader):
            pbar.update(nb_batch * experiment.exp_def.data.MNIST.args.train_batch_size)

            inputs, targets = batch
            inputs = inputs.to(experiment.device).type(torch.float32)
            targets = targets.to(experiment.device)

            with torch.autocast(device_type=device_type, enabled=experiment.useAMP):
                output = model(inputs)
                loss_tensor = (
                    experiment.losses["cp"](output, targets) / experiment.gradacc_iter
                )
                if experiment.useAMP and experiment.scaler is not None:
                    experiment.scaler.scale(loss_tensor).backward()
                else:
                    loss_tensor.backward()

            experiment.update_gradients(
                b_idx=nb_batch, loader_name="MNIST", model_name="simpleNet"
            )

            train_loss_sum += loss_tensor.detach().item()

        experiment.trainLoss = train_loss_sum / len(data.train_data_loader.dataset)
        pbar.finish()

        model.eval()
        pbar = startProgBar(data.n_val_samples, "validation...")

        for nb_batch, batch in enumerate(data.val_data_loader):
            pbar.update(nb_batch * experiment.exp_def.data.MNIST.args.val_batch_size)

            inputs, targets = batch

            with torch.no_grad():
                inputs = inputs.to(experiment.device)
                output = model(inputs)
                targets = targets.to(experiment.device)
                loss_tensor = experiment.losses["cp"](output, targets)

            val_loss_sum += loss_tensor.detach().item()
        experiment.valLoss = val_loss_sum / len(data.val_data_loader.dataset)

        pbar.finish()

        # Log text predictions
        # only tensorboard!
        max_log_size = 8
        _, preds = torch.max(output, 1)
        log_txt = {
            f"Predictions/image_{idx}": f"Predicted: {preds[idx].item()}, True: {targets[idx].item()}"
            for idx in range(min(len(inputs), max_log_size))
        }

        # Log the images
        # tensorboard and wandb behave slightly different!
        # here two examples:
        imgs_tb = {
            "name": "inputs",
            "data": torchvision.utils.make_grid(inputs[:max_log_size, ...]),
            "dataformats": "CHW",
        }

        captions = [
            f"Predicted: {preds[idx].item()}, True: {targets[idx].item()}"
            for idx in range(len(preds))
        ]
        imgs_wandb = {
            "name": "inputs",
            "data": inputs[:max_log_size],
            "captions": captions[:max_log_size],
        }

        experiment.finalize_epoch(
            log_images_wandb=imgs_wandb,
            log_images_tensorboard=imgs_tb,
            log_text_tensorboard=log_txt,
        )

        if experiment.check_early_stop():
            break
