import json
import os
import sys
from datetime import datetime


from fdq.ui_functions import iprint, eprint, wprint, getIntInput


def get_nb_exp_epochs(path):
    """Returns the number of epochs of the experiment stored at 'path'."""
    path = os.path.join(path, "history.json")

    try:
        with open(path, "r", encoding="utf8") as f:
            data = json.load(f)
        return len(data["train"])
    except Exception:
        return 0


def find_experiment_result_dirs(experiment):
    if experiment.is_slurm and experiment.inargs.train_model:
        wprint(
            "WARNING: This is a slurm TRAINING session - looking only for results in temp_results_path!"
        )
        outbasepath = experiment.exp_def.get("slurm_cluster", {}).get(
            "temp_results_path"
        )

    elif experiment.is_slurm and not experiment.inargs.train_model:
        wprint(
            "WARNING: This is a slurm INFERENCE session - looking for results in regular path!"
        )
        outbasepath = experiment.exp_file.get("store", {}).get("results_path")

        if outbasepath[0] == "~":
            outbasepath = os.path.expanduser(outbasepath)

        elif outbasepath[0] != "/":
            raise ValueError("Error: The results path needs to be an absolute path!")

    else:
        # regular local use
        outbasepath = experiment.exp_file.get("store", {}).get("results_path")
        outbasepath = os.path.expanduser(outbasepath)

    if outbasepath is None:
        raise ValueError("Error: No store path specified in experiment file.")

    experiment_res_path = os.path.join(
        outbasepath, experiment.project, experiment.experimentName
    )
    subfolders = [
        f.path.split("/")[-1] for f in os.scandir(experiment_res_path) if f.is_dir()
    ]

    return experiment_res_path, subfolders


def manual_experiment_selection(subfolders_dict, res_root_path):
    """UI to manually select experiment ."""
    subfolders_datetime = [
        datetime.strptime(s, "%Y%m%d_%H_%M_%S") for s in subfolders_dict.keys()
    ]
    subfolders_datetime.sort()
    sorted_keys = [s.strftime("%Y%m%d_%H_%M_%S") for s in subfolders_datetime]

    print("\nSelect experiment:")
    for i, d in enumerate(sorted_keys):
        if subfolders_dict[d] == "":
            # old naming scheme, without funkyBob
            nb_epochs = get_nb_exp_epochs(os.path.join(res_root_path, d))
        else:
            nb_epochs = get_nb_exp_epochs(
                os.path.join(res_root_path, d + "__" + subfolders_dict[d])
            )
        print(
            f"{i:<3}: {d:<20} {subfolders_dict[d]:<25} {nb_epochs} epochs {'(!)' if nb_epochs == 0 else ''}"
        )
    exp_idx = getIntInput("Enter index: ", [0, len(subfolders_datetime) - 1])
    return sorted_keys[exp_idx]


def find_model_path(experiment):
    """Returns the path to the model file of a previous experiment."""
    experiment_res_path, subfolders = find_experiment_result_dirs(experiment)

    subfolders_date_str = []
    subfolders_dict = {}
    for s in subfolders:
        datestr = s.split("__")[0]
        subfolders_date_str.append(datestr)
        if len(s.split("__")) > 1:
            subfolders_dict[datestr] = s.replace(datestr + "__", "")
        else:
            subfolders_dict[datestr] = ""

    if experiment.mode.test_mode.custom_last or experiment.mode.test_mode.custom_best:
        selected_exp_date_str = manual_experiment_selection(
            subfolders_dict, experiment_res_path
        )
    else:
        selected_exp_date_str = sorted(subfolders_date_str)[-1]

    res = [i for i in subfolders if selected_exp_date_str in i]
    if not len(res) == 1:
        raise ValueError(
            f"No corresponding result folder was found in '{experiment_res_path}'. Specify path manually!"
        )

    find_last = experiment.mode.test_mode.custom_last or experiment.mode.test_mode.last
    search_string = "last_" if find_last else "best_"

    possible_files = []
    for fn in os.listdir(os.path.join(experiment_res_path, res[0])):
        if search_string in fn and fn.endswith(".fdqm"):
            possible_files.append(fn)

    if len(possible_files) == 0:
        raise ValueError(
            f"No corresponding model file was found in '{experiment_res_path}'. Specify path manually!"
        )

    elif len(possible_files) > 1:
        wprint(
            f"Multiple corresponding models files were found in '{experiment_res_path}':"
        )
        wprint(possible_files)
        wprint(
            f"Selecting automatically the first one for testing: '{possible_files[0]}'"
        )

    return os.path.join(experiment_res_path, res[0]), possible_files[0]


def save_test_results(test_results, experiment):
    if test_results is not None:
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d_%H_%M")

        results_fp = os.path.join(
            experiment.test_dir, f"00_test_results_{dt_string}.json"
        )

        with open(results_fp, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "experiment_name": experiment.experimentName,
                    "test results": test_results,
                },
                f,
                ensure_ascii=False,
                indent=4,
            )


def save_test_info(experiment, model=None, weights=None):
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H_%M")
    results_fp = os.path.join(experiment.test_dir, f"test_config_{dt_string}.json")

    with open(results_fp, "w", encoding="utf-8") as f:
        json.dump(
            {"model": model, "weights": weights},
            f,
            ensure_ascii=False,
            indent=4,
        )


def ui_ask_test_mode(experiment):
    """UI to select test mode."""
    exp_mode = getIntInput(
        "\nExperiment Selection:\n1: Last, 2: From List, 3: Path to model\n", [1, 3]
    )
    if exp_mode in [1, 2]:
        model_mode = getIntInput(
            "\nModel Selection:\n1: Last Model, 2: Best Model\n", [1, 2]
        )
        if exp_mode == 1 and model_mode == 1:
            experiment.mode.last()
        elif exp_mode == 1 and model_mode == 2:
            experiment.mode.best()
        elif exp_mode == 2 and model_mode == 1:
            experiment.mode.custom_last()
        elif exp_mode == 2 and model_mode == 2:
            experiment.mode.custom_best()
    else:
        if model_mode == 1:
            experiment.mode.custom_path()


def _set_test_mode(experiment):
    experiment.mode.test()
    best_or_last = experiment.exp_def.test.get("test_model", "best")
    if experiment.mode.op_mode.unittest:
        experiment.mode.last()

    elif experiment.inargs.test_model_auto:
        if best_or_last == "best":
            experiment.mode.best()
        else:
            experiment.mode.last()
        iprint(f"Auto test: Loading {best_or_last} model.")

    else:
        ui_ask_test_mode(experiment)


def _load_test_models(experiment):
    for model_name, _ in experiment.exp_def.models:
        if experiment.mode.test_mode.custom_path:
            while True:
                model_path = input(
                    f"Enter path to model for '{model_name}' (or 'q' to quit)."
                )
                if model_path == "q":
                    sys.exit()
                elif os.path.exists(model_path):
                    experiment.inference_model_paths[model_name] = model_path
                    break
                else:
                    eprint(f"Error: File {model_path} not found.")

        else:
            experiment._results_dir, net_name = find_model_path(experiment)
            experiment.inference_model_paths[model_name] = os.path.join(
                experiment._results_dir, net_name
            )

        experiment.load_models()


def run_test(experiment):
    iprint("-------------------------------------------")
    iprint("Starting Test...")
    iprint("-------------------------------------------")

    _set_test_mode(experiment)
    _load_test_models(experiment)

    experiment.copy_files_to_test_dir(experiment.experiment_file_path)
    if experiment.parent_file_path is not None:
        experiment.copy_files_to_test_dir(experiment.parent_file_path)

    save_test_info(
        experiment,
        model=experiment.inference_model_paths,
    )
    experiment.setupData()
    experiment.createLosses()
    test_results = experiment.runEvaluator()
    save_test_results(test_results, experiment)
