import argparse
import random
import sys

import numpy as np
import torch

from fdq.experiment import fdqExperiment
from fdq.testing import run_test
from fdq.ui_functions import iprint


def main() -> None:
    parser = argparse.ArgumentParser(description="FCQ deep learning framework.")
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

    args = parser.parse_args()
    experiment = fdqExperiment(args)

    random_seed = experiment.exp_def.globals.set_random_seed
    if random_seed is not None:
        if not isinstance(random_seed, int):
            raise ValueError("ERROR, random seed must be integer number!")
        iprint(f"SETTING RANDOM SEED TO {random_seed} !!!")
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    # if experiment.inargs.print_model:
    #     experiment.init_models()
    #     iprint("\n-----------------------------------------------------------")
    #     iprint(experiment.model)
    #     iprint("-----------------------------------------------------------\n")

    #     try:
    #         experiment.setupData()
    #         iprint(
    #             f"Saving model graph to: {experiment.results_dir}/{experiment.networkName}_graph.png"
    #         )

    #         draw_graph(
    #             experiment.model,
    #             input_size=experiment.model_input_shape,
    #             device=experiment.device,
    #             save_graph=True,
    #             filename=experiment.networkName + "_graph",
    #             directory=experiment.results_dir,
    #             expand_nested=False,
    #         )
    #     except Exception as e:
    #         wprint("Failed to draw graph!")
    #         print(e)

    if experiment.inargs.train_model:
        experiment.prepareTraining()
        experiment.trainer.train(experiment)
        experiment.clean_up()

    if experiment.inargs.test_model_auto or experiment.inargs.test_model_ia:
        run_test(experiment)

    if experiment.inargs.dump_model:
        iprint("Dumping the best model of the last experiment")
        raise NotImplementedError(
            "Model dumping is not implemented yet. Please implement it in the fdqExperiment class."
        )
        # res_folder, net_name = find_model_path(experiment)
        # experiment.load_models(os.path.join(res_folder, net_name))
        # experiment.dump_model(res_folder)

    iprint("done")

    if experiment.early_stop_detected is not False:
        # non zero exit code to prevent job resubmission
        sys.exit(1)


if __name__ == "__main__":
    main()
