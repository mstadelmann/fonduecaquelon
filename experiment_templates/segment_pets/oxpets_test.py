import torch
from fdq.misc import save_wandb
from fdq.ui_functions import startProgBar, iprint


@torch.no_grad()
def fdq_test(experiment):
    """This is a very simple example evaluating the experiment on the OXPET test dataset and returning the mean loss as accuracy."""
    test_loader = experiment.data["OXPET"].test_data_loader

    targs = experiment.exp_def.test.args
    nb_test_samples = targs.get("nb_test_samples", 50)

    losses = []

    iprint(f"Testset sample size: {experiment.data['OXPET'].n_test_samples}")
    pbar = startProgBar(experiment.data["OXPET"].n_test_samples, "evaluation...")

    for nb_tbatch, batch in enumerate(test_loader):
        if nb_tbatch >= nb_test_samples:
            break

        if isinstance(batch, dict):
            inputs = batch["image"]
            targets = batch["mask"]
        else:
            inputs, targets = batch

        inputs = inputs.to(experiment.device)
        targets = targets.to(experiment.device)

        pbar.update(nb_tbatch)
        output = experiment.models["ccUNET"](inputs)
        current_loss = float(experiment.losses["cross_ent"](output, targets))
        losses.append(current_loss)

        imgs_to_log = [
            {"name": "test_in", "data": inputs},
            {"name": "test_out", "data": output},
            {"name": "test_targ", "data": targets},
        ]
        save_wandb(
            experiment,
            scalars={"img_nb": nb_tbatch, "cross_ent_loss": current_loss},
            images=imgs_to_log,
        )

    pbar.finish()

    accuracy = float(torch.tensor(losses).mean())
    iprint(f"\nTotal accuracy: {accuracy}")

    return accuracy
