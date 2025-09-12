"""This module defines the training procedure for the MNIST test experiment."""

import torch
from fdq.experiment import fdqExperiment
from fdq.ui_functions import startProgBar, iprint


def fdq_train(experiment: fdqExperiment) -> None:
    """Train the model using the provided experiment configuration.

    Args:
        experiment (fdqExperiment): The experiment object containing data loaders, models, and training configurations.
    """
    iprint("Default training")

    data = experiment.data["OXPET"]
    model = experiment.models["ccUNET"]

    for epoch in range(experiment.start_epoch, experiment.nb_epochs):
        experiment.on_epoch_start(epoch=epoch)

        train_loss_sum = 0.0
        val_loss_sum = 0.0
        model.train()
        pbar = startProgBar(data.n_train_samples, "training...")

        for nb_tbatch, batch in enumerate(data.train_data_loader):
            pbar.update(nb_tbatch * len(batch["image"]))

            inputs = batch["image"].to(experiment.device).type(torch.float32)
            targets = batch["mask"].to(experiment.device).type(torch.float32)

            with torch.autocast(device_type=experiment.device.type, enabled=experiment.useAMP):
                output = model(inputs)
                train_loss_tensor = experiment.losses["cross_ent"](output, targets) / experiment.gradacc_iter
                if experiment.useAMP and experiment.scaler is not None:
                    experiment.scaler.scale(train_loss_tensor).backward()
                else:
                    train_loss_tensor.backward()

            experiment.update_gradients(b_idx=nb_tbatch, loader_name="OXPET", model_name="ccUNET")

            train_loss_sum += train_loss_tensor.detach().item()

        experiment.trainLoss = train_loss_sum / len(data.train_data_loader.dataset)
        pbar.finish()

        model.eval()

        pbar = startProgBar(data.n_val_samples, "validation...")

        with torch.no_grad():
            for nb_vbatch, batch in enumerate(data.val_data_loader):
                pbar.update(nb_vbatch * len(batch["image"]))

                inputs = batch["image"].to(experiment.device).type(torch.float32)
                targets = batch["mask"].to(experiment.device).type(torch.float32)
                output = model(inputs)
                val_loss_tensor = experiment.losses["cross_ent"](output, targets)
                val_loss_sum += val_loss_tensor.detach().item()

        experiment.valLoss = val_loss_sum / len(data.val_data_loader.dataset)

        pbar.finish()

        img = [
            {"name": "input", "data": inputs, "dataformats": "NCHW"},
            {"name": "output", "data": output, "dataformats": "NCHW"},
            {"name": "target", "data": targets, "dataformats": "NCHW"},
        ]

        experiment.on_epoch_end(log_images_wandb=img, log_images_tensorboard=img)

        if experiment.check_early_stop():
            break
