import argparse
import random
import sys
import os
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp
from fdq.experiment import fdqExperiment
from fdq.testing import run_test
from fdq.ui_functions import iprint
from fdq.misc import load_conf_file
from fdq.dump import dump_model
from fdq.inference import inference_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for configuring and running an FDQ experiment.

    Returns:
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="FCQ deep learning framework.")
    parser.add_argument("experimentfile", type=str, help="Path to experiment definition file.")
    parser.add_argument("-nt", "-notrain", dest="train_model", default=True, action="store_false")
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
    parser.add_argument("-d", "-dump", dest="dump_model", default=False, action="store_true")
    parser.add_argument("-i", "-inference", dest="inference_model", default=False, action="store_true")
    parser.add_argument("-p", "-printmodel", dest="print_model", default=False, action="store_true")
    parser.add_argument(
        "-rp",
        "-resume_path",
        dest="resume_path",
        type=str,
        default=None,
        help="Path to checkpoint.",
    )

    return parser.parse_args()


def start(rank: int, args: argparse.Namespace) -> None:
    """Main entry point for running an FDQ experiment based on command-line arguments."""
    experiment: fdqExperiment = fdqExperiment(args, rank=rank)

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
        dump_model(experiment)

    if experiment.inargs.inference_model:
        inference_model(experiment)

    experiment.clean_up_distributed()

    iprint("done")

    # Return non-zero exit code to prevent automated launch of test job
    # if NaN or very early stop detected
    if experiment.early_stop_reason == "NaN_train_Loss":
        sys.exit(1)
    elif experiment.early_stop_detected and experiment.current_epoch < int(0.1 * experiment.nb_epochs):
        sys.exit(1)


def main():
    """Main function to parse arguments, load configuration, and run the FDQ experiment."""
    inargs = parse_args()
    use_GPU = load_conf_file(inargs.experimentfile).train.args.use_GPU

    world_size = 1

    if inargs.train_model:
        # DDP only on cluster, and only if GPU enabled
        if os.getenv("SLURM_JOB_ID") is not None and use_GPU:
            world_size = load_conf_file(inargs.experimentfile).get("slurm_cluster", {}).get("world_size", 1)

            if world_size > torch.cuda.device_count():
                raise ValueError(
                    f"ERROR, world size {world_size} is larger than available GPUs: {torch.cuda.device_count()}"
                )

    if world_size == 1:
        # No need for multiprocessing
        start(0, inargs)
    else:
        mp.spawn(start, args=(inargs,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
