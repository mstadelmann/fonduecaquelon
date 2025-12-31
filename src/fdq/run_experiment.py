import argparse
import random
import sys
import os
import hydra
from typing import Any
from omegaconf import DictConfig
from omegaconf import OmegaConf

import numpy as np
import torch
import torch.multiprocessing as mp
from fdq.experiment import fdqExperiment
from fdq.testing import run_test
from fdq.ui_functions import iprint
from fdq.misc import load_conf_file
from fdq.dump import dump_model
from fdq.inference import inference_model


def start(rank: int, cfg: DictConfig) -> None:
    """Main entry point for running an FDQ experiment based on command-line arguments."""
    experiment: fdqExperiment = fdqExperiment(cfg, rank=rank)

    random_seed: Any = experiment.cfg.globals.set_random_seed
    if random_seed is not None:
        if not isinstance(random_seed, int):
            raise ValueError("ERROR, random seed must be integer number!")
        iprint(f"SETTING RANDOM SEED TO {random_seed} !!!")
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    if experiment.cfg.mode.print_model_summary:
        experiment.print_model()

    if experiment.cfg.mode.run_train:
        experiment.prepareTraining()
        experiment.trainer.fdq_train(experiment)
        experiment.clean_up_train()

    if experiment.cfg.mode.run_test_auto or experiment.cfg.mode.run_test_interactive:
        run_test(experiment)

    if experiment.cfg.mode.dump_model:
        dump_model(experiment)

    if experiment.cfg.mode.run_inference:
        inference_model(experiment)

    experiment.clean_up_distributed()

    iprint("done")

    # Return non-zero exit code to prevent automated launch of test job
    # if NaN or very early stop detected
    if experiment.early_stop_reason == "NaN_train_Loss":
        sys.exit(1)
    elif experiment.early_stop_detected and experiment.current_epoch < int(0.1 * experiment.nb_epochs):
        sys.exit(1)


def expand_paths(cfg):
    # convert to container (dict/list), walk recursively
    def _expand(v):
        if isinstance(v, str) and v.startswith("~"):
            return os.path.expanduser(v)
        elif isinstance(v, list):
            return [_expand(x) for x in v]
        elif isinstance(v, dict):
            return {k: _expand(val) for k, val in v.items()}
        else:
            return v

    return OmegaConf.create(_expand(OmegaConf.to_container(cfg, resolve=True)))


@hydra.main(
    version_base=None,
    config_path="/home/marc/dev/fonduecaquelon/experiment_templates/mnist",
    config_name="mnist_class_dense",
)
def main(cfg: DictConfig) -> None:
    """Main function to parse arguments, load configuration, and run the FDQ experiment."""
    cfg = expand_paths(cfg)
    use_GPU = cfg.train.args.use_GPU

    world_size = 1

    if cfg.mode.run_train:
        # DDP only on cluster, and only if GPU enabled
        if os.getenv("SLURM_JOB_ID") is not None and use_GPU:
            world_size = cfg.slurm_cluster.get("world_size", 1)

            if world_size > torch.cuda.device_count():
                raise ValueError(
                    f"ERROR, world size {world_size} is larger than available GPUs: {torch.cuda.device_count()}"
                )

    if world_size == 1:
        # No need for multiprocessing
        start(0, cfg)
    else:
        mp.spawn(start, args=(cfg,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
