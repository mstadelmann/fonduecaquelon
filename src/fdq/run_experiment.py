import argparse
import random
import sys
import os
import hydra
from hydra.core.hydra_config import HydraConfig
from typing import Any
from omegaconf import DictConfig
from omegaconf import OmegaConf
from omegaconf import open_dict

import numpy as np
import torch
import torch.multiprocessing as mp
from fdq.experiment import fdqExperiment
from fdq.testing import run_test
from fdq.ui_functions import iprint
from fdq.dump import dump_model
from fdq.inference import inference_model


def start(rank: int, cfg: DictConfig = None, cfg_container=None) -> None:
    """Main entry point for running an FDQ experiment based on command-line arguments."""
    if cfg is None:
        cfg = OmegaConf.create(cfg_container)

    experiment: fdqExperiment = fdqExperiment(cfg, rank=rank)

    random_seed: Any = experiment.cfg.globals.get("set_random_seed")
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


def get_hydra_paths():
    def collect(cfg_path: str) -> None:
        seen: set[str] = set()
        parents: list[str] = []
        try:
            cfg = OmegaConf.load(cfg_path)
        except Exception:
            return

        defaults = cfg.get("defaults", []) or []
        for item in defaults:
            name = None
            if isinstance(item, str):
                name = item
            elif isinstance(item, dict) and len(item) == 1:
                k, v = next(iter(item.items()))
                name = v if isinstance(v, str) else k

            if not name or name == "_self_":
                continue

            parent_path = os.path.join(config_dir, f"{name}.yaml")
            if "keys" in name:
                iprint(f"Skipping copying key files to results dir: {name}")
            elif os.path.exists(parent_path) and parent_path not in seen:
                seen.add(parent_path)
                parents.append(parent_path)
                collect(parent_path)  # recurse

        return parents

    # Get Hydra config paths
    try:
        hc = HydraConfig.get()
        config_name = hc.job.config_name
        for src in hc.runtime.config_sources:
            if src.schema == "file":
                config_dir = src.path
                break
        else:
            raise RuntimeError("No file-based config source found")
        root_config_path = os.path.join(config_dir, f"{config_name}.yaml")

    except Exception:
        raise RuntimeError("Could not determine Hydra config path")

    parents = collect(root_config_path)

    res = {
        "config_name": config_name,
        "config_dir": config_dir,
        "root_config_path": root_config_path,
        "parents": parents,
    }

    return res


@hydra.main(
    version_base=None,
    # Uncomment for easy debugging
    # config_path="/home/marc/dev/fonduecaquelon/experiment_templates/mnist",
    # config_name="mnist_class_dense",
)
def main(cfg: DictConfig) -> None:
    """Main function to parse arguments, load configuration, and run the FDQ experiment."""
    cfg = expand_paths(cfg)
    use_GPU = cfg.train.args.use_GPU
    use_slurm_cluster = cfg.get("slurm_cluster") is not None and os.getenv("SLURM_JOB_ID") is not None

    world_size = 1

    if cfg.mode.run_train:
        # DDP only on cluster, and only if GPU enabled
        if use_slurm_cluster and use_GPU:
            world_size = cfg.slurm_cluster.get("world_size", 1)

            if world_size > torch.cuda.device_count():
                raise ValueError(
                    f"ERROR, world size {world_size} is larger than available GPUs: {torch.cuda.device_count()}"
                )

    hydra_paths = get_hydra_paths()
    with open_dict(cfg):
        cfg.hydra_paths = hydra_paths

    if world_size == 1:
        # No need for multiprocessing
        start(0, cfg=cfg)
    else:
        # convert hydra cfg to a picklable container before spawning.
        cfg_container = OmegaConf.to_container(cfg, resolve=True)
        mp.spawn(
            start,
            args=(
                None,
                cfg_container,
            ),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    main()
