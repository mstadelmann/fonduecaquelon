import argparse
import random
import sys
import os
import json
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp
from fdq.experiment import fdqExperiment
from fdq.testing import run_test
from fdq.ui_functions import iprint
from fdq.misc import recursive_dict_update, DictToObj, replace_tilde_with_abs_path


def load_json(path: str) -> dict:
    """Load a JSON file and return its content as a dictionary."""
    with open(path, encoding="utf8") as fp:
        try:
            data = json.load(fp)
        except Exception as exc:
            raise ValueError(
                f"Error loading JSON file {path} (check syntax?)."
            ) from exc

    if data.get("globals") is None:
        raise ValueError(
            f"Error: experiment {path} does not contain 'globals' section. Check template!"
        )
    return data


def get_parent_path(path: str, exp_file_path: str) -> str:
    """Resolve the absolute path of a parent configuration file.

    Parameters
    ----------
    path : str
        Relative or absolute path to the parent configuration file.
    exp_file_path : str
        Path to the current experiment configuration file.

    Returns:
    -------
    str
        Absolute path to the parent configuration file.
    """
    if path[0] == "/":
        return path
    else:
        return os.path.abspath(os.path.join(os.path.split(exp_file_path)[0], path))


def load_conf_file(path) -> dict:
    """Load an experiment configuration file, recursively merging parent configurations.

    Parameters
    ----------
    path : str
        Path to the experiment configuration JSON file.

    Returns:
    -------
    dict
        The merged configuration as a dictionary-like object.
    """
    reached_leaf = False
    conf = load_json(path)
    parent_conf = conf.copy()
    parent = conf.get("globals").get("parent", {})
    conf["globals"]["parent_hierarchy"] = []

    while not reached_leaf:
        parent = parent_conf.get("globals").get("parent", {})
        if parent == {}:
            reached_leaf = True
        else:
            parent_path = get_parent_path(parent, path)
            conf["globals"]["parent_hierarchy"].append(parent_path)
            parent_conf = load_json(parent_path)
            conf = recursive_dict_update(d_parent=parent_conf, d_child=conf)

    replace_tilde_with_abs_path(conf)

    return DictToObj(conf)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for configuring and running an FDQ experiment.

    Returns:
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="FCQ deep learning framework."
    )
    parser.add_argument(
        "experimentfile", type=str, help="Path to experiment definition file."
    )
    parser.add_argument(
        "-nt", "-notrain", dest="train_model", default=True, action="store_false"
    )
    parser.add_argument(
        "-ti",
        "-test_interactive",
        dest="test_model_ia",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-ta",
        "-test_auto",
        dest="test_model_auto",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "-d", "-dump", dest="dump_model", default=False, action="store_true"
    )
    parser.add_argument(
        "-p", "-printmodel", dest="print_model", default=False, action="store_true"
    )
    parser.add_argument(
        "-rp",
        "-resume_path",
        dest="resume_path",
        type=str,
        default=None,
        help="Path to checkpoint.",
    )

    return parser.parse_args()


def start(rank: int, args: argparse.Namespace, conf: dict) -> None:
    """Main entry point for running an FDQ experiment based on command-line arguments."""
    experiment: fdqExperiment = fdqExperiment(args, exp_conf=conf, rank=rank)

    random_seed: Any = experiment.exp_def.globals.set_random_seed
    if random_seed is not None:
        if not isinstance(random_seed, int):
            raise ValueError("ERROR, random seed must be integer number!")
        iprint(f"SETTING RANDOM SEED TO {random_seed} !!!")
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    if experiment.inargs.print_model:
        experiment.print_model()

    if experiment.inargs.train_model:
        experiment.prepareTraining()
        experiment.trainer.fdq_train(experiment)
        experiment.clean_up_train()

    if experiment.inargs.test_model_auto or experiment.inargs.test_model_ia:
        run_test(experiment)

    if experiment.inargs.dump_model:
        experiment.dump_model()

    experiment.clean_up_distributed()

    iprint("done")

    # Return non-zero exit code to prevent automated launch of test job
    # if NaN or very early stop detected
    if experiment.early_stop_reason == "NaN_train_Loss":
        sys.exit(1)
    elif experiment.early_stop_detected and experiment.current_epoch < int(
        0.1 * experiment.nb_epochs
    ):
        sys.exit(1)


def main():
    """Main function to parse arguments, load configuration, and run the FDQ experiment."""
    inargs = parse_args()
    exp_config = load_conf_file(inargs.experimentfile)
    world_size = exp_config.get("slurm_cluster", {}).get("world_size", 1)

    if not inargs.train_model:
        world_size = 1

    if world_size > torch.cuda.device_count():
        raise ValueError(
            f"ERROR, world size {world_size} is larger than available GPUs: {torch.cuda.device_count()}"
        )

    if world_size == 1:
        # No need for multiprocessing
        start(0, inargs, exp_config)
    else:
        mp.spawn(start, args=(inargs, exp_config), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
