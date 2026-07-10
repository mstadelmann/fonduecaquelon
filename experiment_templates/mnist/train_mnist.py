"""This module defines the training procedure for the MNIST test experiment."""

import torch
import torchvision
from fdq.experiment import fdqExperiment
from fdq.ui_functions import startProgBar, iprint


def fdq_train(experiment: fdqExperiment) -> None:
    """Train the model using the provided experiment configuration.

    Args:
        experiment (fdqExperiment): The experiment object containing data loaders, models, and training configurations.
    """
    iprint("Default training")

    data = experiment.data["MNIST"]
    model = experiment.models["simpleNet"]
    device_type = "cuda" if experiment.device == torch.device("cuda") else "cpu"

    for epoch in range(experiment.start_epoch, experiment.nb_epochs):
        experiment.on_epoch_start(epoch=epoch)

        train_loss_sum = 0.0
        val_loss_sum = 0.0
        model.train()
        pbar = startProgBar(data.n_train_batches, "training...")

        for nb_batch, batch in enumerate(data.train_data_loader):
            pbar.update(nb_batch + 1)

            inputs, targets = batch
            inputs = inputs.to(experiment.device).type(torch.float32)
            targets = targets.to(experiment.device)

            with torch.autocast(device_type=device_type, enabled=experiment.useAMP):
                output = model(inputs)
                loss_tensor = experiment.losses["cross_ent"](output, targets) / experiment.gradacc_iter
                if experiment.useAMP and experiment.scaler is not None:
                    experiment.scaler.scale(loss_tensor).backward()
                else:
                    loss_tensor.backward()

            experiment.update_gradients(b_idx=nb_batch, loader_name="MNIST", model_name="simpleNet")

            train_loss_sum += loss_tensor.detach().item()

        experiment.trainLoss = train_loss_sum / len(data.train_data_loader.dataset)
        pbar.finish()

        model.eval()
        pbar = startProgBar(data.n_val_batches, "validation...")
        val_correct = 0
        val_total = 0

        for nb_batch, batch in enumerate(data.val_data_loader):
            pbar.update(nb_batch + 1)

            inputs, targets = batch

            with torch.no_grad():
                inputs = inputs.to(experiment.device)
                output = model(inputs)
                targets = targets.to(experiment.device)
                loss_tensor = experiment.losses["cross_ent"](output, targets)

            val_loss_sum += loss_tensor.detach().item()
            _, preds_batch = torch.max(output, 1)
            val_correct += (preds_batch == targets).sum().item()
            val_total += targets.size(0)

        experiment.valLoss = val_loss_sum / len(data.val_data_loader.dataset)
        val_accuracy = val_correct / val_total if val_total > 0 else 0.0

        pbar.finish()
        iprint(f"Validation accuracy: {val_accuracy:.4f}")

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
            "name": "val_samples",
            "data": torchvision.utils.make_grid(inputs[:max_log_size, ...]),
            "dataformats": "CHW",
        }

        captions = [f"Predicted: {preds[idx].item()}, True: {targets[idx].item()}" for idx in range(len(preds))]
        imgs_wandb = {
            "name": "val_samples",
            "data": inputs[:max_log_size],
            "captions": captions[:max_log_size],
        }

        experiment.on_epoch_end(
            log_scalars={"val_accuracy": val_accuracy},
            log_images_wandb=imgs_wandb,
            log_images_tensorboard=imgs_tb,
            log_text_tensorboard=log_txt,
        )

        if experiment.check_early_stop():
            break
