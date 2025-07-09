"""SLURM job submission utility for fonduecaquelon experiments."""

import sys
import os
import re
import json
import copy
import getpass
import subprocess
from datetime import datetime
from typing import Any


def get_template() -> str:
    """Return the SLURM job submission script template as a string."""
    return """#!/bin/bash
#SBATCH --time=#job_time#
#SBATCH --job-name=fdq-#job_name#
#SBATCH --ntasks=#ntasks#
#SBATCH --cpus-per-task=#cpus_per_task#
#SBATCH --nodes=#nodes#
#SBATCH --gres=#gres#
#SBATCH --mem=#mem#
#SBATCH --partition=#partition#
#SBATCH --account=#account#
#SBATCH --mail-user=#user#@zhaw.ch
#SBATCH --output=#log_path#/%j_%N__#job_name##job_tag#.out
#SBATCH --error=#log_path#/%j_%N__#job_name##job_tag#.err
#SBATCH --signal=B:SIGUSR1@#stop_grace_time#

script_start=$(date +%s.%N)

RUN_TRAIN=#run_train#
RUN_TEST=#run_test# # test will be run automatically, but not necessarily in this job
IS_TEST=#is_test# # if True, start test in this job
AUTO_RESUBMIT=#auto_resubmit# # resubmit the job if stopped due to time constraints
RESUME_CHPT_PATH=#resume_chpt_path# # path to checkpoint file to resume training
EXP_FILE_PATH=#exp_file_path#
SCRATCH_RESULTS_PATH=#scratch_results_path#
SCRATCH_DATA_PATH=#scratch_data_path#
RESULTS_PATH=#results_path#
SUBMIT_FILE_PATH=#submit_file_path#
PY_MODULE=#python_env_module#
UV_MODULE=#uv_env_module#
FDQ_VERSION=#fdq_version#
RETVALUE=1 # will become zero if training is successful, which will launch an optional test job

# Copy submit script to scratch for resubmission
cp $SUBMIT_FILE_PATH /scratch/
SCRATCH_SUBMIT_FILE_PATH="/scratch/$(basename "$SUBMIT_FILE_PATH")"

echo ------------------------------------------------------------
echo "FONDUE-CAQUELON - EXPERIMENT CONFIGURATION"
echo ------------------------------------------------------------
echo "START TIME: $(date)"
echo "SLURM JOB ID: $SLURM_JOB_ID"
echo "SOURCE SUBMIT FILE PATH: $SUBMIT_FILE_PATH"
echo "SCRATCH SUBMIT FILE PATH: $SCRATCH_SUBMIT_FILE_PATH"
echo "RUN_TRAIN: $RUN_TRAIN"
echo "RUN_TEST: $RUN_TEST"
echo "IS_TEST: $IS_TEST"
echo "AUTO_RESUBMIT: $AUTO_RESUBMIT"
echo "RESUME_CHPT_PATH: $RESUME_CHPT_PATH"
echo "EXP_FILE_PATH: $EXP_FILE_PATH"
echo "SCRATCH_RESULTS_PATH: $SCRATCH_RESULTS_PATH"
echo "SCRATCH_DATA_PATH: $SCRATCH_DATA_PATH"
echo "RESULTS_PATH: $RESULTS_PATH"
echo "PYTHON MODULE: $PY_MODULE"
echo "UV MODULE: $UV_MODULE"
echo "FDQ VERSION: $FDQ_VERSION"

echo ------------------------------------------------------------
echo "PREPARING ENVIRONMENT"
echo ------------------------------------------------------------
cd /scratch/
module load $PY_MODULE
VENV="fdqenv" module load $UV_MODULE
uv venv fdqenv
source /scratch/fdqenv/bin/activate
uv pip install fdq==$FDQ_VERSION

#additional_pip_packages#

echo "UV environment ready...!"

mkdir -p $SCRATCH_RESULTS_PATH
mkdir -p $SCRATCH_DATA_PATH
mkdir -p $RESULTS_PATH

# ----------------------------------------------------------------------------------
# Stop signal handler
# ----------------------------------------------------------------------------------
sig_handler_USR1()
{
    echo "++++++++++++++++++++++++++++++++++++++"
    echo "SLURM STOP SIGNAL DETECTED -  `date`"
    echo experiment file: $EXP_FILE_PATH
    echo "++++++++++++++++++++++++++++++++++++++"

    echo Copy files from $SCRATCH_RESULTS_PATH to $RESULTS_PATH.
    rsync -a $SCRATCH_RESULTS_PATH* $RESULTS_PATH
    sleep 1
    echo "---------------"
    echo "File copy done."
    echo "---------------"

    if [ "$AUTO_RESUBMIT" == True ]; then
        # resubmit the job pointing to the last checkpoint file.
        most_recent_chp=$(find $SCRATCH_RESULTS_PATH -name checkpoint* | head -n 1 | awk -F '/fdq_results/' '{print $2}')
        most_recent_chp_path=${RESULTS_PATH}/${most_recent_chp}
        echo "Most recent checkpoint_path: $most_recent_chp_path"

        sed -e "s|^RESUME_CHPT_PATH=.*|RESUME_CHPT_PATH=$most_recent_chp_path|g" $SCRATCH_SUBMIT_FILE_PATH > $SCRATCH_SUBMIT_FILE_PATH.resub
        rm $SCRATCH_SUBMIT_FILE_PATH
        mv $SCRATCH_SUBMIT_FILE_PATH.resub $SCRATCH_SUBMIT_FILE_PATH

        sleep 1
        echo submitting new job with the following command:
        echo "sbatch $SCRATCH_SUBMIT_FILE_PATH"
        sbatch $SCRATCH_SUBMIT_FILE_PATH
        sleep 1
    fi
    exit 0
}
sig_handler_USR2()
{
    echo "++++++++++++++++++++++++++++++++++++++"
    echo "USR2 - MANUAL STOP DETECTED -  `date`"
    echo experiment file: $EXP_FILE_PATH
    echo "copy files back to cluster and stop!"
    echo "++++++++++++++++++++++++++++++++++++++"

    echo Copy files from $SCRATCH_RESULTS_PATH to $RESULTS_PATH.
    rsync -a $SCRATCH_RESULTS_PATH* $RESULTS_PATH
    echo "---------------"
    echo "File copy done."
    echo "---------------"
    exit 0
}

trap 'sig_handler_USR1' USR1
trap 'sig_handler_USR2' USR2

if [ "$RUN_TRAIN" == True ]; then
    echo ------------------------------------------------------------
    echo "RUNNING TRAINING"
    echo ------------------------------------------------------------

    train_start=$(date +%s.%N)

    if [ "$RESUME_CHPT_PATH" == None ]; then
        fdq $EXP_FILE_PATH &
    elif [ -f "$RESUME_CHPT_PATH" ]; then
        echo "Resuming training from checkpoint: $RESUME_CHPT_PATH"
        fdq $EXP_FILE_PATH -rp $RESUME_CHPT_PATH &
    else
        echo "ERROR: Checkpoint path does not exist: $RESUME_CHPT_PATH"
    fi

    fdq_pid=$!
    wait $fdq_pid
    RETVALUE=$?
    train_stop=$(date +%s.%N)

    echo ------------------------------------------------------------
    echo "TRAINING DONE - Copying results back to storage cluster"
    echo ------------------------------------------------------------
    sleep 1
    copy_start=$(date +%s.%N)
    rsync -a $SCRATCH_RESULTS_PATH* $RESULTS_PATH
    copy_end=$(date +%s.%N)
    echo ------------------------------------------------------------
    echo "Copying results back to storage cluster - DONE"
    echo ------------------------------------------------------------
    train_time=$(echo "$train_stop - $train_start" | bc)
    copy_time=$(echo "$copy_end - $copy_start" | bc)
    script_time=$(echo "$copy_end - $script_start" | bc)
    echo ------------------------------------------------------------
    echo "Script execution time: $script_time s"
    echo "Train time: $train_time s"
    echo "Data copy time: $copy_time s"
    echo ------------------------------------------------------------
fi

if [ "$IS_TEST" == True ]; then
    echo ------------------------------------------------------------
    echo "RUNNING TEST"
    echo ------------------------------------------------------------
    RUN_TEST=False # dont launch test in new job
    test_start=$(date +%s.%N)
    fdq $EXP_FILE_PATH -nt -ta
    fdq_pid=$!
    wait $fdq_pid
    RETVALUE=$?
    test_stop=$(date +%s.%N)
    test_time=$(echo "$test_stop - $test_start" | bc)
    echo ------------------------------------------------------------
    echo "Test time: $test_time s"
    echo ------------------------------------------------------------
fi


# --------------------------------------------------------------
# Submit new job for test
# --------------------------------------------------------------
if [ "$RUN_TEST" == True ]; then
    if [ $RETVALUE -eq 0 ]; then
        echo ------------------------------------------------------------
        echo "Launching test job.."
        echo ------------------------------------------------------------
        sed -e "s|IS_TEST=False|IS_TEST=True|g" \
            -e "s|RUN_TRAIN=True|RUN_TRAIN=False|g" \
            -e "s|RUN_TEST=True|RUN_TEST=False|g" \
            -e "s|_train.|_test.|g" \
            -e "s|^#SBATCH --gpus-per-task=.*|#SBATCH --gpus-per-task=1|" \
            "$SCRATCH_SUBMIT_FILE_PATH" > "$SCRATCH_SUBMIT_FILE_PATH.resub"
        rm $SCRATCH_SUBMIT_FILE_PATH
        mv $SCRATCH_SUBMIT_FILE_PATH.resub $SCRATCH_SUBMIT_FILE_PATH
        sleep 1
        echo submitting new job with the following command:
        echo "sbatch --job-name=fdq-test $SCRATCH_SUBMIT_FILE_PATH"
        sbatch --job-name=fdq-test $SCRATCH_SUBMIT_FILE_PATH
        sleep 1
        exit 0
    else
        echo ----------------------------------------------------------------------
        echo "Automatic test not started due to non-zero fdq ret value: $retvalue"
        echo ----------------------------------------------------------------------
    fi
fi
"""


def recursive_dict_update(d_parent: dict, d_child: dict) -> dict:
    """Recursively update the parent dictionary with values from the child dictionary.

    Args:
        d_parent (dict): The dictionary to be updated.
        d_child (dict): The dictionary whose values will update the parent.

    Returns:
        dict: A deep copy of the updated parent dictionary.
    """
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
    """A class that converts a dictionary into an object, allowing attribute-style access to keys, including nested dictionaries."""

    def __init__(self, dictionary: dict) -> None:
        """Initialize the object by setting attributes from the given dictionary, converting nested dictionaries to DictToObj."""
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = DictToObj(value)
            setattr(self, key, value)

    def __getattr__(self, name: str) -> Any:
        """Return None if the requested attribute is not found."""
        return None

    def __repr__(self) -> str:
        """Return the string representation for debugging."""
        keys = ", ".join(self.__dict__.keys())
        return f"<{self.__class__.__name__}: {keys}>"

    def __str__(self) -> str:
        """Return the string representation of the object."""
        return self.__repr__()

    def __iter__(self):
        """Return an iterator over the object's dictionary items."""
        return iter(self.__dict__.items())

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def to_dict(self) -> dict:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, DictToObj):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        res = getattr(self, key)
        if res is None:
            return default
        return res


def parse_input_file(exp_file_path: str) -> DictToObj:
    """Parse the experiment JSON file, handle parent inheritance, and return a DictToObj representation.

    Args:
        exp_file_path (str): Path to the experiment JSON file.

    Returns:
        DictToObj: Parsed experiment configuration as an object.
    """
    if not os.path.isfile(exp_file_path):
        print(f"Error: The file '{exp_file_path}' does not exist.")
        sys.exit(1)

    try:
        with open(exp_file_path, encoding="utf8") as file:
            exp_file = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Error: The file '{exp_file_path}' is not a valid JSON file."
        ) from exc

    globals_def = exp_file.get("globals")
    parent = globals_def.get("parent", {})
    if parent != {}:
        if parent[0] == "/":
            parent_file_path = parent
        else:
            parent_file_path = os.path.abspath(
                os.path.join(os.path.split(exp_file_path)[0], parent)
            )

        if not os.path.exists(parent_file_path):
            raise FileNotFoundError(f"Error: File {parent_file_path} not found.")

        with open(parent_file_path, encoding="utf8") as fp:
            try:
                parent_expfile = json.load(fp)
            except Exception as exc:
                raise ValueError(
                    f"Error loading experiment file {parent_file_path} (check syntax?)."
                ) from exc

        exp_file = recursive_dict_update(d_parent=parent_expfile, d_child=exp_file)

    return DictToObj(exp_file)


def get_default_config(slurm_conf: Any) -> dict[str, Any]:
    """Return a job configuration dictionary with defaults, updated from the given SLURM config.

    Args:
        slurm_conf (dict): SLURM configuration dictionary.

    Returns:
        dict: Job configuration dictionary with updated values.
    """
    job_config: dict[str, Any] = {
        "job_name": None,
        "user": None,
        "job_time": None,
        "ntasks": 1,
        "cpus_per_task": 8,
        "nodes": 1,
        "gres": "gpu:1",
        "mem": "32G",
        "partition": None,
        "account": None,
        "run_train": True,
        "run_test": False,
        "is_test": False,
        "job_tag": "",
        "auto_resubmit": True,
        "resume_chpt_path": "",
        "log_path": None,
        "stop_grace_time": 15,
        "python_env_module": None,
        "uv_env_module": None,
        "fdq_version": None,
        "exp_file_path": None,
        "scratch_results_path": "/scratch/fdq_results/",
        "scratch_data_path": "/scratch/fdq_data/",
        "results_path": None,
        "submit_file_path": None,
    }

    for key in job_config:
        val = slurm_conf.get(key)
        if val is not None:
            job_config[key] = val

    return job_config


def check_config(job_config: dict[str, Any]) -> dict[str, Any]:
    """Check if all mandatory job configuration values are set and expand user paths.

    Args:
        job_config (dict): The job configuration dictionary to validate and update.

    Returns:
        dict: The validated and updated job configuration dictionary.

    Raises:
        ValueError: If any mandatory configuration value is None.
    """
    for key, value in job_config.items():
        if value is None:
            raise ValueError(
                f"Value for mandatory key'{key}' is None. Please update your config file!"
            )
        if value == "":
            job_config[key] = "None"
        elif isinstance(value, str) and value.startswith("~/"):
            job_config[key] = os.path.expanduser(value)

    return job_config


def create_submit_file(
    job_config: dict[str, Any], slurm_conf: Any, submit_path: str
) -> None:
    """Create a SLURM submit file from the job configuration and SLURM config.

    Args:
        job_config (dict): The job configuration dictionary.
        slurm_conf (DictToObj): The SLURM configuration object.
        submit_path (str): The path where the submit file will be written.
    """
    if os.path.exists(job_config["log_path"]):
        os.makedirs(job_config["log_path"], exist_ok=True)
    template_content: str = get_template()
    for key, value in job_config.items():
        template_content = template_content.replace(f"#{key}#", str(value))
    template_content = template_content.replace("//", "/")

    if slurm_conf.additional_pip_packages is None:
        template_content = template_content.replace("#additional_pip_packages#", "")
    elif isinstance(slurm_conf.additional_pip_packages, list):
        additional_pip_packages = "\n".join(
            [f"uv pip install {pkg}" for pkg in slurm_conf.additional_pip_packages]
        )
        template_content = template_content.replace(
            "#additional_pip_packages#", additional_pip_packages
        )
    else:
        raise ValueError("Error: additional_pip_packages must be a list of strings.")

    with open(submit_path, "w", encoding="utf8") as f:
        f.write(template_content)


def main() -> None:
    """Main entry point for submitting a job to SLURM using the provided experiment JSON file."""
    if len(sys.argv) != 2:
        raise ValueError(
            "Error: Exactly one argument is required which is the path to the JSON file."
        )

    in_args: DictToObj = parse_input_file(sys.argv[1])
    slurm_conf: Any = in_args.slurm_cluster
    if slurm_conf is None:
        raise ValueError(
            "Error: The 'slurm_cluster' section in the JSON config file is required to submit a job to the queue!"
        )

    job_config: dict[str, Any] = get_default_config(slurm_conf)

    # set exp file path
    exp_file_path: str = os.path.expanduser(sys.argv[1])
    if not os.path.isabs(exp_file_path):
        exp_file_path = os.path.abspath(exp_file_path)
    job_config["exp_file_path"] = exp_file_path
    exp_name: str = os.path.basename(exp_file_path).split(".")[0]

    job_config["job_name"] = exp_name[:30].replace(" ", "_")
    job_config["user"] = getpass.getuser()
    job_config["results_path"] = in_args.store.results_path
    job_config["log_path"] = job_config["log_path"]

    # define path of submit script
    base_path: str = os.path.join(
        os.path.expanduser(job_config["log_path"]),
        "submitted_jobs",
    )
    os.makedirs(base_path, exist_ok=True)
    submit_path: str = os.path.join(
        base_path,
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}__{exp_name.replace(' ', '_')}.submit",
    )
    job_config["submit_file_path"] = submit_path

    # if this is a pure test job, set the is_test flag
    if not job_config["run_train"] and job_config["run_test"]:
        job_config["is_test"] = True
        job_config["job_tag"] = "_test"
    else:
        job_config["job_tag"] = "_train"

    # check if all mandatory configs are set
    job_config = check_config(job_config)

    # create new submit file
    create_submit_file(job_config, slurm_conf, submit_path)

    # start slurm job
    result = subprocess.run(
        f"sbatch {submit_path}", shell=True, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Error submitting job. Exit code: {result.returncode}")
        print(f"stderr: {result.stderr}")
        sys.exit(result.returncode)

    match = re.search(r"(\d+)\s*$", result.stdout)
    if match:
        print(
            f"Submitted batch job.\nSlurm Job ID {match.group(1)}\nSubmit file: {submit_path}."
        )
    else:
        print(result.stdout)

    print("done.")


if __name__ == "__main__":
    print("----------")
    print("FDQ submit")
    main()
