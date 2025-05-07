import json
import os

# trunk-ignore(bandit/B404)
import subprocess as sp
import sys
import time
from prettytable import PrettyTable
from datetime import datetime

import git
import numpy as np
import torch
import inspect
from ui_functions import iprint, wprint, eprint
import copy


class FCQmode:
    def __init__(self) -> None:
        self._op_mode = "init"
        self.allowed_op_modes = [
            "init",  # initial state
            "train",  # training mode
            "unittest",  # running unit tests
            "test",  # testing
        ]
        self._test_mode = "best"
        self.allowed_test_modes = [
            "best",  # test best model from last experiment - DEFAULT!
            "last",  # test last trained model from last experiment
            "custom_last",  # test last model from selected experiment
            "custom_best",  # test best model from selected experiment
            "custom_path",  # test with manually defined model path
        ]

        # Dynamically create setter methods
        for mode in self.allowed_op_modes:
            setattr(self, mode, self._create_setter("_op_mode", mode))

        for mode in self.allowed_test_modes:
            setattr(self, mode, self._create_setter("_test_mode", mode))

    def __repr__(self):
        if self._op_mode == "test":
            return f"<{self.__class__.__name__}: {self._op_mode} / {self._test_mode}>"
        else:
            return f"<{self.__class__.__name__}: {self._op_mode}>"

    @property
    def op_mode(self):
        class OpMode:
            def __init__(self, parent):
                self.parent = parent

            def __repr__(self):
                return f"<{self.__class__.__name__}: {self.parent._op_mode}>"

            def __getattr__(self, name):
                if name in self.parent.allowed_op_modes:
                    return self.parent._op_mode == name
                raise AttributeError(f"'OpMode' object has no attribute '{name}'")

        return OpMode(self)

    @property
    def test_mode(self):
        class TestMode:
            def __init__(self, parent):
                self.parent = parent

            def __repr__(self):
                return f"<{self.__class__.__name__}: {self.parent._test_mode}>"

            def __getattr__(self, name):
                if name in self.parent.allowed_test_modes:
                    return self.parent._test_mode == name
                raise AttributeError(f"'TestMode' object has no attribute '{name}'")

        return TestMode(self)

    def _create_setter(self, attribute, value):
        def setter():
            setattr(self, attribute, value)

        return setter


def recursive_dict_update(d_parent, d_child):
    for key, value in d_child.items():
        if (
            isinstance(value, dict)
            and key in d_parent
            and isinstance(d_parent[key], dict)
        ):
            recursive_dict_update(d_parent[key], value)
        else:
            d_parent[key] = value

    return copy.deepcopy(d_parent)


class DictToObj:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = DictToObj(value)
            setattr(self, key, value)

    def __getattr__(self, name):
        # if attribute not found
        return None

    def __repr__(self):
        keys = ", ".join(self.__dict__.keys())
        return f"<{self.__class__.__name__}: {keys}>"

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        return iter(self.__dict__.items())

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, DictToObj):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def get(self, key, default=None):
        res = getattr(self, key)
        if res is None:
            return default
        return res


def print_nb_weights(experiment, show_details=False):
    for model_name, model in experiment.models.items():
        iprint("----------------------------------")
        iprint(f"Model: {model_name}")
        nbp = sum(p.numel() for p in model.parameters())
        iprint(f"nb parameters: {nbp / 1e6:.2f}M")
        iprint("----------------------------------")


def remove_file(path):
    if path is not None:
        try:
            os.remove(path)
        except Exception:
            eprint(f"{path} does not exists!")


def get_nvidia_smi_memory():
    def _string_to_list(x):
        return x.decode("ascii").split("\n")[:-1]

    try:
        COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
        memory_used_info = _string_to_list(sp.check_output(COMMAND.split()))[1:]
        memory_used_values = [int(x.split()[0]) for i, x in enumerate(memory_used_info)]

        COMMAND = "nvidia-smi --query-gpu=memory.total --format=csv"
        memory_total_info = _string_to_list(sp.check_output(COMMAND.split()))[1:]
        memory_total_values = [
            int(x.split()[0]) for i, x in enumerate(memory_total_info)
        ]

        res = (
            memory_used_values,
            memory_total_values,
            list(
                np.round(
                    np.array(memory_used_values) / np.array(memory_total_values), 2
                )
            ),
        )

    except Exception:
        res = None

    return res


def store_processing_infos(experiment):
    """
    Store experiment information to results directory.
    """
    experiment.run_info = collect_processing_infos(experiment=experiment)
    info_path = os.path.join(experiment.results_dir, "info.json")

    with open(info_path, "w", encoding="utf8") as write_file:
        json.dump(experiment.run_info, write_file, indent=4, sort_keys=True)


def collect_fdq_git_hash():
    """
    Returns the git hash of the currently running FDQ environment,
    and checks, if all files were committed.
    None committed files are printed to the console and stored in the
    experiment info file.
    """
    dirty_files = None
    fdq_hash = None
    fdq_dirty = True  # unless proved otherwise...

    try:
        fdq_git = git.Repo(os.path.abspath(__file__), search_parent_directories=True)
    except Exception:
        wprint("Warning: Could not find git repo for FDQ!")
        fdq_git = None
        fdq_hash = "UNABLE TO LOCALIZE GIT REPO!"

    if fdq_git is not None:
        try:
            fdq_hash = fdq_git.head.object.hexsha
            fdq_dirty = fdq_git.is_dirty()

            if fdq_dirty:
                dirty_files = [f.b_path for f in fdq_git.index.diff(None)]

                wprint("---------------------------------------------")
                wprint("WARNING: fdq git repo is dirty!")
                wprint(dirty_files)
                wprint("---------------------------------------------")
                time.sleep(5)

        except Exception:
            wprint("Warning: Could not extract git hash for FDQ!")
            fdq_hash = "UNABLE TO LOCALIZE GIT REPO!"

    return fdq_hash, fdq_dirty, dirty_files


def collect_processing_infos(experiment=None):
    fdq_hash, fdq_dirty, dirty_files = collect_fdq_git_hash()

    try:
        sysname = os.uname()[1]
    except Exception:
        sysname = None

    try:
        username = os.getlogin()
    except Exception:
        username = None

    try:
        create_dt_string = experiment.creation_time.strftime("%Y%m%d_%H_%M_%S")
    except Exception:
        create_dt_string = None

    try:
        stop_dt_string = experiment.finish_time.strftime("%Y%m%d_%H_%M_%S")
    except Exception:
        stop_dt_string = None

    try:
        td = experiment.run_time
        run_t_string = f"days: {td.days}, hours: {td.seconds // 3600}, minutes: {td.seconds % 3600 / 60.0:.0f}"
    except Exception:
        run_t_string = None

    data = {
        "User": username,
        "System": sysname,
        "Python V.": sys.version,
        "Torch V.": torch.__version__,
        "Cuda V.": torch.version.cuda,
        "fdq-git": fdq_hash,
        "git-is-dirty": fdq_dirty,
        "dirty-files": dirty_files,
        "start_datetime": create_dt_string,
        "end_datetime": stop_dt_string,
        "total_runtime": run_t_string,
        # "epochs": f"{experiment.current_epoch + 1} / {experiment.nb_epochs}",
        "last_update": datetime.now().strftime("%Y%m%d_%H_%M_%S"),
        # "is_early_stop_val_loss": experiment.early_stop_val_loss_detected,
        # "is_early_stop_train_loss": experiment.early_stop_train_loss_detected,
        # "is_early_stop_nan": experiment.early_stop_nan_detected,
        # "best_train_loss_epoch": experiment.new_best_train_loss_ep_id,
        # "best_val_loss_epoch": experiment.new_best_val_loss_ep_id,
    }

    if experiment.is_slurm:
        data["slurm_job_id"] = experiment.slurm_job_id

    if experiment.inargs.resume_path is not None:
        data["job_continuation"] = True
        data["job_continuation_chpt_path"] = experiment.inargs.resume_path
        data["start_epoch"] = experiment.start_epoch
    else:
        data["job_continuation"] = False

    try:
        # add GPU memory usage
        if experiment.device == torch.device("cuda"):
            cur_str = f"{experiment.malloc_nvi_smi_current_list[-1] / 1000:.0f}"
            tot_str = f"{experiment.malloc_nvidia_smi_total / 1000:.0f}"
            mem_str = f"{experiment.malloc_nvidia_smi_percentage}%  ({cur_str}/{tot_str} [GB])"
            data["GPU memory usage estimation"] = mem_str
    except Exception:
        pass

    try:
        # add nb model parameters to info file
        model_weights = sum(p.numel() for p in experiment.model.parameters())
        data["Number of model parameters"] = f"{model_weights / 1e6:.2f}M"
    except Exception:
        pass

    try:
        # add dataset key-numbers to info file
        data["dataset_key_numbers"] = {
            "Nb samples train": experiment.trainset_size,
            "Train subset": experiment.train_subset,
            "Nb samples val": experiment.valset_size,
            "Validation subset": experiment.val_subset,
            "Nb samples test": experiment.testset_size,
            "Test subset": experiment.test_subset,
            "Validation set is a subset of the training set.": experiment.valset_is_train_subset,
            "Validation subset ratio": experiment.val_from_train_ratio,
        }
    except Exception:
        pass

    return data


def get_model_git(model):
    try:
        model_path = inspect.getfile(model)
        model_git = git.Repo(model_path, search_parent_directories=True)

    except Exception:
        model_git = None

    return model_git


def check_model_git_hash(experiment, current_model):
    """
    This function allows to check and checkout the correct version of external models.
    """

    # TODO: this does currently not work in a docker environment!

    ignore_model_git_hash = experiment.model_hash == "ignore"

    if ignore_model_git_hash:
        return

    if experiment.model_hash is None:
        raise ValueError(
            f"Could not find git hash for {experiment.networkName}! Set model_git_hash to 'ignore' to ignore check!"
        )

    model_git = get_model_git(current_model)

    if model_git is None:
        error_str = (
            "Unable to detect model git repository. Is it installed in editable mode? "
            "Set model_git_hash to 'ignore' False to ignore check!"
        )
        raise ValueError(error_str)

    try:
        current_model_hash = model_git.head.object.hexsha
    except Exception as exc:
        raise ValueError("Could not extract git hash for model!") from exc

    if current_model_hash == experiment.model_hash:
        iprint(f"Requested model version {experiment.model_hash} is already installed.")

    elif experiment.model_hash not in (current_model_hash, "ignore"):
        iprint(f"Trying to checkout model version {experiment.model_hash}.")

        try:
            model_git.git.checkout(experiment.model_hash)
            iprint("SUCCESS!")
        except Exception as exc:
            error_str = (
                f"Could not checkout model version {experiment.model_hash}. "
                "Set model_git_hash to 'ignore' False to ignore check!"
            )
            raise ValueError(error_str) from exc
