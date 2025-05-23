import torch
from fdq.ui_functions import startProgBar

# THIS IS A VERY BASIC DEMO AND NEEDS TO BE UPDATED!


def fdq_test(experiment):
    """Evaluate the experiment on the OXPET test dataset and return the mean loss as accuracy."""
    test_loader = experiment.data["OXPET"].test_data_loader

    losses = []

    print(f"Testset sample size: {experiment.data['OXPET'].n_test_samples}")
    pbar = startProgBar(experiment.data["OXPET"].n_test_samples, "evaluation...")

    for i, batch in enumerate(test_loader):
        if isinstance(batch, dict):
            inputs = batch["image"]
            targets = batch["mask"]
        else:
            inputs, targets = batch

        inputs = inputs.to(experiment.device)
        targets = targets.to(experiment.device)

        pbar.update(i)
        output = experiment.models["ccUNET"](inputs)

        losses.append(float(experiment.losses["cp"](output, targets)))

    pbar.finish()

    accuracy = float(torch.tensor(losses).mean())
    print(f"\nTotal accuracy: {accuracy}")

    return accuracy
