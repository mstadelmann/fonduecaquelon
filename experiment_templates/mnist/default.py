import torch

from experiment import fdqExperiment

from img_func import (
    save_checkpoint_img,
    save_tensorboard,
    save_tensorboard_loss,
    save_wandb,
    save_wandb_loss,
)
from ui_functions import show_train_progress, startProgBar, iprint
from utils import print_nb_weights


def train(experiment: fdqExperiment) -> None:
    iprint("Default training")
    print_nb_weights(experiment)

    train_loader = experiment.data["MNIST"].train_data_loader
    val_loader = experiment.data["MNIST"].val_data_loader
    n_train_samples = experiment.data["MNIST"].n_train_samples
    n_val_samples = experiment.data["MNIST"].n_val_samples

    for epoch in range(experiment.start_epoch, experiment.nb_epochs):
        experiment.current_epoch = epoch
        iprint(f"\nEpoch: {epoch + 1} / {experiment.nb_epochs}")

        training_loss_value = 0.0
        valid_loss_value = 0.0
        experiment.models["simpleNet"].train()
        pbar = startProgBar(n_train_samples, "training...")

        for nb_tbatch, batch in enumerate(train_loader):
            pbar.update(nb_tbatch * experiment.exp_def.data.MNIST.args.train_batch_size)

            inputs, targets = batch

            inputs = inputs.to(experiment.device).type(torch.float32)
            targets = targets.to(experiment.device)

            # this can be written without code repetition, however, goal is to keep both options flexible...
            # following: https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
            if experiment.useAMP:
                device_type = (
                    "cuda" if experiment.device == torch.device("cuda") else "cpu"
                )
                # torch.autocast(dtype=torch.float16) does not work on CPU
                # torch.autocast(dtype=torch.bfloat16) does not work on V100
                # torch.autocast(NO DTYPE) on V100 uses the same amount of mem as torch.float16

                with torch.autocast(device_type=device_type, enabled=True):
                    output = experiment.model(inputs)
                    train_loss_tensor = (
                        experiment.losses["cp"](output, targets)
                        / experiment.gradacc_iter
                    )

                experiment.scaler.scale(train_loss_tensor).backward()

            else:
                output = experiment.models["simpleNet"](inputs)
                train_loss_tensor = (
                    experiment.losses["cp"](output, targets) / experiment.gradacc_iter
                )

                train_loss_tensor.backward()

            experiment.update_gradients(
                b_idx=nb_tbatch, loader_name="MNIST", model_name="simpleNet"
            )

            training_loss_value += train_loss_tensor.data.item() * inputs.size(0)

        experiment.trainLoss = training_loss_value / len(train_loader.dataset)
        pbar.finish()

        experiment.models["simpleNet"].eval()

        pbar = startProgBar(n_val_samples, "validation...")

        for nb_vbatch, batch in enumerate(val_loader):
            experiment.current_val_batch = nb_vbatch
            pbar.update(nb_vbatch * experiment.exp_def.data.MNIST.args.val_batch_size)

            inputs, targets = batch

            with torch.no_grad():
                inputs = inputs.to(experiment.device)
                output = experiment.models["simpleNet"](inputs)
                targets = targets.to(experiment.device)
                val_loss_tensor = experiment.losses["cp"](output, targets)

            valid_loss_value += val_loss_tensor.data.item() * inputs.size(0)

        pbar.finish()
        experiment.valLoss = valid_loss_value / len(val_loader.dataset)

        save_wandb_loss(experiment)

        save_tensorboard_loss(experiment=experiment)
        show_train_progress(experiment)

        iprint(
            f"Training Loss: {experiment.trainLoss:.4f}, Validation Loss: {experiment.valLoss:.4f}"
        )

        experiment.finalize_epoch()

        if experiment.check_early_stop():
            break
