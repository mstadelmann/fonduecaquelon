import argparse
import random
import sys
import json
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp
from fdq.experiment import fdqExperiment
from fdq.testing import run_test
from fdq.ui_functions import iprint


def load_conf_file(path) -> None:
    with open(path, encoding="utf8") as fp:
        try:
            conf = json.load(fp)
        except Exception as exc:
            raise ValueError(
                f"Error loading experiment file {path} (check syntax?)."
            ) from exc
    return conf


def parse_args() -> argparse.Namespace:
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
        experiment.clean_up()

    if experiment.inargs.test_model_auto or experiment.inargs.test_model_ia:
        run_test(experiment)

    if experiment.inargs.dump_model:
        experiment.dump_model()

    iprint("done")

    # non zero exit code to prevent launch of test job
    # if NaN or very early stop detected
    if experiment.early_stop_detected == "NaN detected":
        sys.exit(1)
    elif experiment.early_stop_detected is not False and experiment.current_epoch < int(
        0.1 * experiment.nb_epochs
    ):
        sys.exit(1)


def main():
    inargs = parse_args()
    exp_config = load_conf_file(inargs.experimentfile)
    world_size = exp_config.get("slurm_cluster", {}).get("world_size", 1)

    if not inargs.train_model:
        world_size = 1

    if world_size > torch.cuda.device_count():
        raise ValueError(
            f"ERROR, world size {inargs.world_size} is larger than available GPUs: {torch.cuda.device_count()}"
        )

    mp.spawn(start, args=(inargs, exp_config), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
