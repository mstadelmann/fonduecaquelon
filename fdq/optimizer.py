from torch import optim


def createOptimizer(experiment):
    for model_name, margs in experiment.exp_def.models:
        oargs = margs.optimizer.to_dict()
        if margs.optimizer.name == "adam":
            # https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
            optimizer = optim.Adam(
                experiment.models[model_name].parameters(),
                lr=margs.optimizer.lr,
                betas=oargs.get("optimizer_betas", (0.9, 0.999)),
                eps=oargs.get("optimizer_eps", 1e-8),
                weight_decay=oargs.get("optimizer_weight_decay", 0),
            )

        elif margs.optimizer.name == "adamW":
            # https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW
            optimizer = optim.AdamW(
                experiment.models[model_name].parameters(),
                lr=margs.optimizer.lr,
                betas=oargs.get("optimizer_betas", (0.9, 0.999)),
                eps=oargs.get("optimizer_eps", 1e-8),
                weight_decay=oargs.get("optimizer_weight_decay", 1e-2),
            )

        elif margs.optimizer.name == "SGD":
            optimizer = optim.SGD(
                experiment.models[model_name].parameters(), lr=margs.optimizer.lr
            )

        elif margs.optimizer.name == "RMSprop":
            optimizer = optim.RMSprop(
                experiment.models[model_name].parameters(),
                lr=margs.optimizer.lr,
                weight_decay=oargs.get("optimizer_weight_decay", 1e-8),
            )

        elif margs.optimizer.name is None and experiment.training_strategy == "vanilla":
            optimizer = None

        else:
            raise ValueError(
                f"Optimizer {margs['optimizer']} is not defined in optimizer.py!"
            )

        if optimizer is not None:
            optimizer.zero_grad()

        experiment.optimizers[model_name] = optimizer


def set_lr_schedule(experiment):
    # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    # https://towardsdatascience.com/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863

    for model_name, margs in experiment.exp_def.models:
        if experiment.optimizers[model_name] is None:
            raise ValueError(f"ERROR - optimizer for model {model_name} is not set!")
        if margs.lr_scheduler is None:
            experiment.lr_schedulers[model_name] = None
            continue

        lr_scheduler_name = margs.lr_scheduler.name

        if lr_scheduler_name is None:
            experiment.lr_schedulers[model_name] = None
            continue

        elif lr_scheduler_name == "step_lr":
            # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR
            experiment.lr_schedulers[model_name] = optim.lr_scheduler.StepLR(
                experiment.optimizers[model_name],
                step_size=margs.lr_scheduler.step_size,
                gamma=margs.lr_scheduler.gamma,
            )

        else:
            raise NotImplementedError(
                f"ERROR - LR scheduler {lr_scheduler_name} is currently not supported!"
            )
