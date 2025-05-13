"""This module defines the training procedure for the MNIST test experiment."""

import torch
from fdq.experiment import fdqExperiment
from fdq.ui_functions import show_train_progress, startProgBar, iprint
from fdq.misc import print_nb_weights
from fdq.img_func import (
    save_tensorboard_loss,
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

        training_loss_value = 0.0
        valid_loss_value = 0.0
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

            training_loss_value += train_loss_tensor.data.item() * inputs.size(0)

        experiment.trainLoss = training_loss_value / len(data.train_data_loader.dataset)
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

            valid_loss_value += val_loss_tensor.data.item() * inputs.size(0)

        pbar.finish()
        experiment.valLoss = valid_loss_value / len(data.val_data_loader.dataset)

        save_wandb_loss(experiment)

        save_tensorboard_loss(experiment=experiment)
        show_train_progress(experiment)

        iprint(
            f"Training Loss: {experiment.trainLoss:.4f}, Validation Loss: {experiment.valLoss:.4f}"
        )

        experiment.finalize_epoch()

        if experiment.check_early_stop():
            break
