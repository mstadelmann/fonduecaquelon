"""This module defines the training procedure for the Oxford Pets segmentation experiment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from fdq.ui_functions import startProgBar, iprint

if TYPE_CHECKING:
    from fdq.experiment import fdqExperiment


def _target_to_labels(targets: torch.Tensor, num_classes: int, expected_ndim: int) -> torch.Tensor:
    if targets.ndim == expected_ndim and targets.shape[1] == num_classes:
        return targets.argmax(dim=1)
    if targets.ndim == expected_ndim and targets.shape[1] == 1:
        return targets.squeeze(1).long()
    return targets.long()


def multiclass_dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
    include_background: bool = True,
) -> torch.Tensor:
    """Compute mean Dice score for multiclass segmentation logits."""
    if logits.ndim < 3:
        raise ValueError("Expected logits with shape (N, C, ...).")
    if logits.shape[1] < 1:
        raise ValueError("Expected logits to contain at least one class channel.")

    num_classes = logits.shape[1]
    pred_labels = logits.argmax(dim=1)
    target_labels = _target_to_labels(targets, num_classes, logits.ndim).to(device=logits.device)

    if pred_labels.shape != target_labels.shape:
        raise ValueError(
            f"Prediction shape {tuple(pred_labels.shape)} does not match target shape {tuple(target_labels.shape)}."
        )

    start_class = 0 if include_background else 1
    scores = []
    for class_idx in range(start_class, num_classes):
        pred_mask = pred_labels == class_idx
        target_mask = target_labels == class_idx
        denominator = pred_mask.sum() + target_mask.sum()
        if denominator == 0:
            continue
        intersection = (pred_mask & target_mask).sum()
        scores.append((2.0 * intersection.float() + eps) / (denominator.float() + eps))

    if not scores:
        return torch.ones((), device=logits.device)

    return torch.stack(scores).mean()


def fdq_train(experiment: fdqExperiment) -> None:
    """Train the model using the provided experiment configuration.

    Args:
        experiment (fdqExperiment): The experiment object containing data loaders, models, and training configurations.
    """
    iprint("Default training")

    data = experiment.data["OXPET"]
    model = experiment.models["myUNET"]

    for epoch in range(experiment.start_epoch, experiment.nb_epochs):
        experiment.on_epoch_start(epoch=epoch)

        train_loss_sum = 0.0
        train_dice_sum = 0.0
        train_dice_samples = 0
        val_loss_sum = 0.0
        val_dice_sum = 0.0
        val_dice_samples = 0
        model.train()
        pbar = startProgBar(data.n_train_batches, "training...")

        for nb_tbatch, batch in enumerate(data.train_data_loader):
            pbar.update(nb_tbatch + 1)

            inputs = batch["image"].to(experiment.device).type(torch.float32)
            targets = batch["mask"].to(experiment.device).type(torch.float32)

            with torch.autocast(device_type=experiment.device.type, enabled=experiment.useAMP):
                output = model(inputs)
                train_loss_tensor = experiment.losses["cross_ent"](output, targets) / experiment.gradacc_iter
                if experiment.useAMP and experiment.scaler is not None:
                    experiment.scaler.scale(train_loss_tensor).backward()
                else:
                    train_loss_tensor.backward()

            experiment.update_gradients(b_idx=nb_tbatch, loader_name="OXPET", model_name="myUNET")

            batch_size = inputs.shape[0]
            train_dice_sum += multiclass_dice_score(output.detach(), targets).item() * batch_size
            train_dice_samples += batch_size
            train_loss_sum += train_loss_tensor.detach().item()

        experiment.trainLoss = train_loss_sum / len(data.train_data_loader.dataset)
        train_dice = train_dice_sum / max(1, train_dice_samples)
        pbar.finish()

        model.eval()

        pbar = startProgBar(data.n_val_batches, "validation...")

        with torch.no_grad():
            for nb_vbatch, batch in enumerate(data.val_data_loader):
                pbar.update(nb_vbatch + 1)

                inputs = batch["image"].to(experiment.device).type(torch.float32)
                targets = batch["mask"].to(experiment.device).type(torch.float32)
                output = model(inputs)
                val_loss_tensor = experiment.losses["cross_ent"](output, targets)
                batch_size = inputs.shape[0]
                val_dice_sum += multiclass_dice_score(output, targets).item() * batch_size
                val_dice_samples += batch_size
                val_loss_sum += val_loss_tensor.detach().item()

        experiment.valLoss = val_loss_sum / len(data.val_data_loader.dataset)
        val_dice = val_dice_sum / max(1, val_dice_samples)

        pbar.finish()

        img = [
            {"name": "input", "data": inputs, "dataformats": "NCHW"},
            {"name": "output", "data": output, "dataformats": "NCHW"},
            {"name": "target", "data": targets, "dataformats": "NCHW"},
        ]

        experiment.on_epoch_end(
            log_scalars={"train_dice": train_dice, "val_dice": val_dice},
            log_images_wandb=img,
            log_images_tensorboard=img,
        )

        if experiment.check_early_stop():
            break
