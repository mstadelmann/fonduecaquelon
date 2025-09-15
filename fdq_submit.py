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


class FDQSubmitError(Exception):
    """Custom exception for FDQ submission errors."""

    pass


def log_info(message: str) -> None:
    """Log an info message."""
    print(f"[INFO] {message}")


def log_error(message: str) -> None:
    """Log an error message."""
    print(f"[ERROR] {message}", file=sys.stderr)


def log_warning(message: str) -> None:
    """Log a warning message."""
    print(f"[WARNING] {message}")


def validate_file_path(path: str, description: str) -> str:
    """Validate that a file exists and return its absolute path.

    Args:
        path: File path to validate
        description: Description of the file for error messages

    Returns:
        Absolute path to the file

    Raises:
        FDQSubmitError: If file doesn't exist
    """
    expanded_path = os.path.expanduser(path)
    abs_path = os.path.abspath(expanded_path)

    if not os.path.exists(abs_path):
        raise FDQSubmitError(f"{description} not found: {abs_path}")

    if not os.path.isfile(abs_path):
        raise FDQSubmitError(f"{description} is not a file: {abs_path}")

    return abs_path


def get_template() -> str:
    """Return the SLURM job submission script template as a string."""
    return """#!/bin/bash
#SBATCH --time=#job_time#
#SBATCH --job-name=fdq-#job_name#
#SBATCH --ntasks=#ntasks#
#SBATCH --cpus-per-task=#cpus_per_task#
#SBATCH --nodes=#nodes#
#NODELIST#
#SBATCH --gres=#gres#
#SBATCH --mem=#mem#
#SBATCH --partition=#partition#
#SBATCH --account=#account#
#SBATCH --mail-user=#user#@zhaw.ch
#SBATCH --output=#log_path#/%j_%N__#job_name##job_tag#.out
#SBATCH --error=#log_path#/%j_%N__#job_name##job_tag#.err
#SBATCH --signal=B:SIGUSR1@#stop_grace_time#

script_start=$(date +%s.%N)

# Job configuration variables
RUN_TRAIN=#run_train#
RUN_TEST=#run_test# # test will be run automatically, but not necessarily in this job
IS_TEST=#is_test# # if True, start test in this job
GRES_TEST=#gres_test#
MEM_TEST=#mem_test#
CPUS_TEST=#cpus_per_task_test#
AUTO_RESUBMIT=#auto_resubmit# # resubmit the job if stopped due to time constraints
RESUME_CHPT_PATH=#resume_chpt_path# # path to checkpoint file to resume training
EXP_FILE_PATH=#exp_file_path#
SCRATCH_RESULTS_PATH=#scratch_results_path#
SCRATCH_DATA_PATH=#scratch_data_path#
RESULTS_PATH=#results_path#
SUBMIT_FILE_PATH=#submit_file_path#
PY_MODULE=#python_env_module#
UV_MODULE=#uv_env_module#
CUDA_MODULE=#cuda_env_module#
FDQ_VERSION=#fdq_version#
FDQ_TEST_REPO=#fdq_test_repo# # if True, install fdq from https://test.pypi.org
RETVALUE=1 # will become zero if training is successful, which will launch an optional test job

# Function for safe file operations
safe_copy() {
    local src="$1"
    local dst="$2"
    echo "Copying $src to $dst..."
    if ! rsync -a "$src" "$dst"; then
        echo "WARNING: Failed to copy $src to $dst"
        return 1
    fi
    return 0
}

# Copy submit script to scratch for resubmission
if ! cp "$SUBMIT_FILE_PATH" /scratch/; then
    echo "ERROR: Failed to copy submit script to scratch"
    exit 1
fi
SCRATCH_SUBMIT_FILE_PATH="/scratch/$(basename "$SUBMIT_FILE_PATH")"

echo -----------------------------------------------------------
echo "FONDUE-CAQUELON - EXPERIMENT CONFIGURATION"
echo -----------------------------------------------------------
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
echo "CUDA MODULE: $CUDA_MODULE"
echo "FDQ VERSION: $FDQ_VERSION"

echo -----------------------------------------------------------
echo "PREPARING ENVIRONMENT"
echo -----------------------------------------------------------
cd /scratch/

# Load modules
echo "Loading Python module: $PY_MODULE"
if ! module load "$PY_MODULE"; then
    echo "ERROR: Failed to load Python module $PY_MODULE"
    exit 1
fi

echo "Loading UV module: $UV_MODULE"
if ! VENV="fdqenv" module load "$UV_MODULE"; then
    echo "ERROR: Failed to load UV module $UV_MODULE"
    exit 1
fi

if [ -n "$CUDA_MODULE" ] && [ "$CUDA_MODULE" != "None" ]; then
    echo "Loading CUDA module: $CUDA_MODULE"
    if ! VENV="fdqenv" module load "$CUDA_MODULE"; then
        echo "ERROR: Failed to load CUDA module $CUDA_MODULE"
        exit 1
    fi
fi

# Setup virtual environment
echo "Creating UV virtual environment..."
if ! uv venv fdqenv; then
    echo "ERROR: Failed to create UV virtual environment"
    exit 1
fi

echo "Activating virtual environment..."
if ! source /scratch/fdqenv/bin/activate; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi

echo "Installing FDQ version $FDQ_VERSION..."
if [ "$FDQ_TEST_REPO" == True ]; then
    echo "Installing from TestPyPI with PyPI fallback..."
    if ! uv pip install --index-url https://test.pypi.org/simple/ \
        --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-best-match "fdq[gpu]==$FDQ_VERSION"; then
        echo "ERROR: Failed to install fdq (test + fallback)"
        exit 1
    fi
else
    if ! uv pip install "fdq[gpu]==$FDQ_VERSION"; then
        echo "ERROR: Failed to install FDQ"
        exit 1
    fi
fi

# Install additional packages
#additional_pip_packages#

echo "Environment setup complete!"

# Create directories
mkdir -p "$SCRATCH_RESULTS_PATH" "$SCRATCH_DATA_PATH" "$RESULTS_PATH"

# -----------------------------------------------------------
# Stop signal handler
# -----------------------------------------------------------
sig_handler_USR1()
{
    echo "++++++++++++++++++++++++++++++++++++++"
    echo "SLURM STOP SIGNAL DETECTED - $(date)"
    echo "Experiment file: $EXP_FILE_PATH"
    echo "++++++++++++++++++++++++++++++++++++++"

    echo "Copying files from $SCRATCH_RESULTS_PATH to $RESULTS_PATH..."
    safe_copy "$SCRATCH_RESULTS_PATH"* "$RESULTS_PATH"
    
    if [ "$AUTO_RESUBMIT" == True ]; then
        echo "Preparing automatic resubmission..."
        # Find most recent checkpoint
        most_recent_chp=$(find "$SCRATCH_RESULTS_PATH" -name "checkpoint*" | head -n 1 | awk -F '/fdq_results/' '{print $2}')
        if [ -n "$most_recent_chp" ]; then
            most_recent_chp_path="${RESULTS_PATH}/${most_recent_chp}"
            echo "Most recent checkpoint: $most_recent_chp_path"

            # Update submit script for resubmission
            sed -e "s|^RESUME_CHPT_PATH=.*|RESUME_CHPT_PATH=$most_recent_chp_path|g" \
                "$SCRATCH_SUBMIT_FILE_PATH" > "$SCRATCH_SUBMIT_FILE_PATH.resub"
            mv "$SCRATCH_SUBMIT_FILE_PATH.resub" "$SCRATCH_SUBMIT_FILE_PATH"

            echo "Resubmitting job: sbatch $SCRATCH_SUBMIT_FILE_PATH"
            if sbatch "$SCRATCH_SUBMIT_FILE_PATH"; then
                echo "Job resubmitted successfully"
            else
                echo "ERROR: Failed to resubmit job"
            fi
        else
            echo "WARNING: No checkpoint found for resubmission"
        fi
    fi
    exit 0
}

sig_handler_USR2()
{
    echo "++++++++++++++++++++++++++++++++++++++"
    echo "USR2 - MANUAL STOP DETECTED - $(date)"
    echo "Experiment file: $EXP_FILE_PATH"
    echo "Copying files and stopping..."
    echo "++++++++++++++++++++++++++++++++++++++"

    safe_copy "$SCRATCH_RESULTS_PATH"* "$RESULTS_PATH"
    echo "Manual stop completed"
    exit 0
}

# Set signal handlers
trap 'sig_handler_USR1' USR1
trap 'sig_handler_USR2' USR2

if [ "$RUN_TRAIN" == True ]; then
    echo -----------------------------------------------------------
    echo "RUNNING TRAINING"
    echo -----------------------------------------------------------

    train_start=$(date +%s.%N)

    # Start training process
    if [ "$RESUME_CHPT_PATH" == None ]; then
        echo "Starting training from beginning..."
        fdq "$EXP_FILE_PATH" &
    elif [ -f "$RESUME_CHPT_PATH" ]; then
        echo "Resuming training from checkpoint: $RESUME_CHPT_PATH"
        fdq "$EXP_FILE_PATH" -rp "$RESUME_CHPT_PATH" &
    else
        echo "ERROR: Checkpoint path does not exist: $RESUME_CHPT_PATH"
        exit 1
    fi

    fdq_pid=$!
    echo "Training process started with PID: $fdq_pid"
    wait $fdq_pid
    RETVALUE=$?
    train_stop=$(date +%s.%N)

    echo -----------------------------------------------------------
    echo "TRAINING COMPLETED (exit code: $RETVALUE)"
    echo "Copying results back to $RESULTS_PATH"
    echo -----------------------------------------------------------
    
    copy_start=$(date +%s.%N)
    safe_copy "$SCRATCH_RESULTS_PATH"* "$RESULTS_PATH"
    copy_end=$(date +%s.%N)
    
    # Calculate timing
    train_time=$(echo "$train_stop - $train_start" | bc)
    copy_time=$(echo "$copy_end - $copy_start" | bc)
    script_time=$(echo "$copy_end - $script_start" | bc)
    
    echo -----------------------------------------------------------
    echo "TIMING SUMMARY"
    echo "Script execution time: $script_time s"
    echo "Training time: $train_time s"
    echo "Data copy time: $copy_time s"
    echo -----------------------------------------------------------
fi

if [ "$IS_TEST" == True ]; then
    echo -----------------------------------------------------------
    echo "RUNNING TEST"
    echo -----------------------------------------------------------
    
    test_start=$(date +%s.%N)
    fdq "$EXP_FILE_PATH" -nt -ta
    test_retval=$?
    test_stop=$(date +%s.%N)
    test_time=$(echo "$test_stop - $test_start" | bc)
    
    echo -----------------------------------------------------------
    echo "TEST COMPLETED (exit code: $test_retval)"
    echo "Test time: $test_time s"
    echo -----------------------------------------------------------
    
    # Set RETVALUE based on test result
    RETVALUE=$test_retval
fi

# -----------------------------------------------------------
# Submit new job for test
# -----------------------------------------------------------
if [ "$RUN_TEST" == True ] && [ $RETVALUE -eq 0 ] && [ "$IS_TEST" == False ]; then
    echo -----------------------------------------------------------
    echo "Launching test job..."
    echo -----------------------------------------------------------
    
    # Extract test-specific resource requirements
    GRES_TEST=$(awk -F= '/^GRES_TEST=/{print $2}' "$SUBMIT_FILE_PATH")
    MEM_TEST=$(awk -F= '/^MEM_TEST=/{print $2}' "$SUBMIT_FILE_PATH")
    CPUS_TEST=$(awk -F= '/^CPUS_TEST=/{print $2}' "$SUBMIT_FILE_PATH")
    
    # Create test job submit script
    sed -e "s|IS_TEST=False|IS_TEST=True|g" \
        -e "s|RUN_TRAIN=True|RUN_TRAIN=False|g" \
        -e "s|RUN_TEST=True|RUN_TEST=False|g" \
        -e "s|_train.|_test.|g" \
        -e "s|^#SBATCH --gres=.*|#SBATCH --gres=$GRES_TEST|g" \
        -e "s|^#SBATCH --mem=.*|#SBATCH --mem=$MEM_TEST|g" \
        -e "s|^#SBATCH --cpus-per-task=.*|#SBATCH --cpus-per-task=$CPUS_TEST|g" \
        "$SCRATCH_SUBMIT_FILE_PATH" > "$SCRATCH_SUBMIT_FILE_PATH.test"
    
    echo "Submitting test job: sbatch --job-name=fdq-test $SCRATCH_SUBMIT_FILE_PATH.test"
    if sbatch --job-name=fdq-test "$SCRATCH_SUBMIT_FILE_PATH.test"; then
        echo "Test job submitted successfully"
    else
        echo "ERROR: Failed to submit test job"
        exit 1
    fi
elif [ "$RUN_TEST" == True ]; then
    echo -----------------------------------------------------------
    echo "Test job not started due to training failure (exit code: $RETVALUE)"
    echo -----------------------------------------------------------
fi

echo -----------------------------------------------------------
echo "Job COMPLETED with exit code: $RETVALUE"
echo -----------------------------------------------------------
exit $RETVALUE
"""


def recursive_dict_update(d_parent: dict, d_child: dict) -> dict:
    """Merges two dictionaries recursively. The values of d_child will overwrite those in d_parent."""
    result = copy.deepcopy(d_parent)

    for key, value in d_child.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = recursive_dict_update(result[key], value)
        else:
            result[key] = value

    return result


def load_json(path: str) -> dict:
    """Load a JSON file and return its content as a dictionary.

    Args:
        path: Path to the JSON file

    Returns:
        Dictionary containing the JSON data

    Raises:
        FDQSubmitError: If file cannot be loaded or parsed
    """
    try:
        validated_path = validate_file_path(path, "JSON configuration file")
        with open(validated_path, encoding="utf8") as fp:
            data = json.load(fp)
    except json.JSONDecodeError as exc:
        raise FDQSubmitError(
            f"Invalid JSON syntax in {path}: {exc.msg} at line {exc.lineno}, column {exc.colno}"
        ) from exc
    except OSError as exc:
        raise FDQSubmitError(f"Cannot read file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise FDQSubmitError(f"JSON file {path} must contain a dictionary at root level")

    if data.get("globals") is None:
        raise FDQSubmitError(f"Experiment config {path} missing required 'globals' section")

    return data


def get_parent_path(path: str, exp_file_path: str) -> str:
    """Return the absolute path to the parent configuration file.

    Args:
        path: The path to the parent configuration file (absolute or relative)
        exp_file_path: The path to the current experiment file

    Returns:
        The absolute path to the parent configuration file

    Raises:
        FDQSubmitError: If parent path cannot be resolved
    """
    try:
        if os.path.isabs(path):
            return validate_file_path(path, "Parent configuration file")

        # Resolve relative path
        base_dir = os.path.dirname(exp_file_path)
        full_path = os.path.join(base_dir, path)
        return validate_file_path(full_path, "Parent configuration file")
    except Exception as exc:
        raise FDQSubmitError(f"Cannot resolve parent path '{path}': {exc}") from exc


def load_conf_file(path: str) -> "DictToObj":
    """Load an experiment configuration file with recursive parent merging.

    Args:
        path: Path to the experiment configuration JSON file

    Returns:
        The merged configuration as a DictToObj instance

    Raises:
        FDQSubmitError: If configuration cannot be loaded or is invalid
    """
    try:
        validated_path = validate_file_path(path, "Experiment configuration file")
        conf = load_json(validated_path)
        parent_conf = conf.copy()
        parent = conf.get("globals", {}).get("parent", {})
        conf["globals"]["parent_hierarchy"] = []

        # Track processed files to prevent infinite loops
        processed_files = {os.path.abspath(validated_path)}
        max_depth = 10  # Prevent infinite recursion
        depth = 0

        while parent and depth < max_depth:
            depth += 1
            parent_path = get_parent_path(parent, validated_path)

            # Check for circular references
            abs_parent_path = os.path.abspath(parent_path)
            if abs_parent_path in processed_files:
                raise FDQSubmitError(f"Circular reference detected in parent hierarchy: {parent_path}")
            processed_files.add(abs_parent_path)

            log_info(f"Loading parent configuration: {parent_path}")
            conf["globals"]["parent_hierarchy"].append(parent_path)
            parent_conf = load_json(parent_path)
            conf = recursive_dict_update(d_parent=parent_conf, d_child=conf)
            parent = parent_conf.get("globals", {}).get("parent", {})

        if depth >= max_depth:
            raise FDQSubmitError(f"Maximum parent hierarchy depth ({max_depth}) exceeded")

        return DictToObj(conf)

    except FDQSubmitError:
        raise
    except Exception as exc:
        raise FDQSubmitError(f"Failed to load configuration from {path}: {exc}") from exc


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
        "cpus_per_task_test": None,
        "nodes": 1,
        "nodelist": None,
        "gres": "gpu:1",
        "gres_test": None,
        "mem": "32G",
        "mem_test": None,
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
        "cuda_env_module": None,
        "fdq_version": None,
        "fdq_test_repo": False,
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

    # manually set test parameters if not set
    for param in ["gres_test", "mem_test", "cpus_per_task_test"]:
        if job_config[param] is None:
            job_config[param] = job_config[param.replace("_test", "")]

    return job_config


def check_config(job_config: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize job configuration values.

    Args:
        job_config: The job configuration dictionary to validate and update

    Returns:
        The validated and updated job configuration dictionary

    Raises:
        FDQSubmitError: If any mandatory configuration value is missing or invalid
    """
    # Check for mandatory fields
    mandatory_fields = [
        "job_time",
        "partition",
        "account",
        "python_env_module",
        "uv_env_module",
        "fdq_version",
        "results_path",
        "log_path",
    ]

    missing_fields = []
    for field in mandatory_fields:
        if job_config.get(field) is None:
            missing_fields.append(field)

    if missing_fields:
        raise FDQSubmitError(
            f"Missing mandatory configuration fields: {', '.join(missing_fields)}. Please update your config file!"
        )

    # Validate and normalize values
    for key, value in job_config.items():
        if value is None and key not in mandatory_fields:
            # Only set to "None" for optional fields
            job_config[key] = "None"
        elif value == "":
            job_config[key] = "None"
        elif isinstance(value, str) and value.startswith("~/"):
            expanded_path = os.path.expanduser(value)
            job_config[key] = expanded_path
            log_info(f"Expanded path for {key}: {expanded_path}")

    # Validate critical paths exist
    if not os.path.exists(job_config["results_path"]):
        try:
            os.makedirs(job_config["results_path"], exist_ok=True)
            log_info(f"Created results directory: {job_config['results_path']}")
        except OSError as exc:
            raise FDQSubmitError(f"Cannot create results directory {job_config['results_path']}: {exc}") from exc

    # Validate resource specifications
    if job_config.get("mem") and not re.match(r"^\d+[GMK]?$", str(job_config["mem"])):
        log_warning(f"Memory specification '{job_config['mem']}' may be invalid. Expected format: number + G/M/K")

    return job_config


def create_submit_file(job_config: dict[str, Any], slurm_conf: Any, submit_path: str) -> None:
    """Create a SLURM submit file from the job configuration.

    Args:
        job_config: The job configuration dictionary
        slurm_conf: The SLURM configuration object
        submit_path: The path where the submit file will be written

    Raises:
        FDQSubmitError: If submit file cannot be created
    """
    try:
        # Ensure log directory exists
        log_dir = job_config["log_path"]
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            log_info(f"Created log directory: {log_dir}")

        # Get template and substitute values
        template_content = get_template()

        # Replace all job config placeholders
        for key, value in job_config.items():
            placeholder = f"#{key}#"
            if placeholder in template_content:
                template_content = template_content.replace(placeholder, str(value))

        # set nodelist if specified
        nodelist_placeholder = "#NODELIST#"
        if job_config["nodelist"] == "None":
            nodelist_string = ""
        else:
            nodelist_string = f"#SBATCH --nodelist={job_config['nodelist']}"
        template_content = template_content.replace(nodelist_placeholder, nodelist_string)

        # Clean up double slashes in paths
        template_content = template_content.replace("//", "/")

        # Handle additional pip packages
        if slurm_conf.additional_pip_packages is None:
            template_content = template_content.replace("#additional_pip_packages#", "")
        elif isinstance(slurm_conf.additional_pip_packages, list):
            if slurm_conf.additional_pip_packages:  # Only if list is not empty
                packages_cmd = "\n".join(f"uv pip install '{pkg}'" for pkg in slurm_conf.additional_pip_packages)
                log_info(f"Adding {len(slurm_conf.additional_pip_packages)} additional pip packages")
                template_content = template_content.replace("#additional_pip_packages#", packages_cmd)
            else:
                template_content = template_content.replace("#additional_pip_packages#", "")
        else:
            raise FDQSubmitError(
                f"additional_pip_packages must be a list of strings, got {type(slurm_conf.additional_pip_packages)}"
            )

        # Ensure submit directory exists
        submit_dir = os.path.dirname(submit_path)
        if not os.path.exists(submit_dir):
            os.makedirs(submit_dir, exist_ok=True)
            log_info(f"Created submit directory: {submit_dir}")

        # Write the submit file
        with open(submit_path, "w", encoding="utf8") as f:
            f.write(template_content)

        # Make the file executable
        os.chmod(submit_path, 0o755)
        log_info(f"Created SLURM submit file: {submit_path}")

    except OSError as exc:
        raise FDQSubmitError(f"Cannot create submit file {submit_path}: {exc}") from exc
    except Exception as exc:
        raise FDQSubmitError(f"Failed to create submit file: {exc}") from exc


def parse_arguments() -> str:
    """Parse and validate command line arguments.

    Returns:
        Path to the experiment configuration file

    Raises:
        FDQSubmitError: If arguments are invalid
    """
    if len(sys.argv) != 2:
        raise FDQSubmitError(
            "Usage: python fdq_submit.py <path_to_experiment_config.json>\n"
            "Exactly one argument is required: the path to the experiment JSON file."
        )

    config_path = sys.argv[1]

    # Basic validation
    if not config_path.endswith(".json"):
        log_warning("Configuration file doesn't have .json extension")

    return validate_file_path(config_path, "Experiment configuration file")


def submit_slurm_job(submit_path: str) -> str:
    """Submit job to SLURM and return job ID.

    Args:
        submit_path: Path to the SLURM submit script

    Returns:
        SLURM job ID

    Raises:
        FDQSubmitError: If job submission fails
    """
    try:
        log_info(f"Submitting job to SLURM: sbatch {submit_path}")
        result = subprocess.run(
            f"sbatch {submit_path}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,  # Add timeout
        )

        if result.returncode != 0:
            raise FDQSubmitError(f"SLURM job submission failed (exit code {result.returncode}): {result.stderr}")

        # Extract job ID from output
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if match:
            job_id = match.group(1)
            log_info(f"Successfully submitted batch job {job_id}")
            return job_id
        else:
            # Fallback pattern
            match = re.search(r"(\d+)\s*$", result.stdout)
            if match:
                job_id = match.group(1)
                log_info(f"Successfully submitted batch job {job_id}")
                return job_id
            else:
                raise FDQSubmitError(f"Could not extract job ID from SLURM output: {result.stdout}")

    except subprocess.TimeoutExpired as exc:
        raise FDQSubmitError("SLURM submission timed out after 30 seconds") from exc
    except Exception as exc:
        raise FDQSubmitError(f"Failed to submit job to SLURM: {exc}") from exc


def main() -> None:
    """Main entry point for submitting a job to SLURM."""
    try:
        # Parse arguments
        exp_file_path = parse_arguments()
        log_info(f"Loading experiment configuration: {exp_file_path}")

        # Load and validate configuration
        exp_config = load_conf_file(exp_file_path)
        slurm_conf = exp_config.slurm_cluster

        if slurm_conf is None:
            raise FDQSubmitError(
                "Missing 'slurm_cluster' section in configuration file. "
                "This section is required for SLURM job submission."
            )

        # Setup job configuration
        job_config = get_default_config(slurm_conf)

        # Set paths and basic info
        job_config["exp_file_path"] = exp_file_path
        exp_name = os.path.splitext(os.path.basename(exp_file_path))[0]
        job_config["job_name"] = exp_name[:30].replace(" ", "_")
        job_config["user"] = getpass.getuser()

        # Validate required paths
        if not hasattr(exp_config, "store") or not hasattr(exp_config.store, "results_path"):
            raise FDQSubmitError("Configuration missing 'store.results_path' setting")

        job_config["results_path"] = exp_config.store.results_path

        # Setup submit file path
        base_path = os.path.join(
            os.path.expanduser(job_config["log_path"]),
            "submitted_jobs",
        )
        os.makedirs(base_path, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submit_filename = f"{timestamp}__{exp_name.replace(' ', '_')}.submit"
        submit_path = os.path.join(base_path, submit_filename)
        job_config["submit_file_path"] = submit_path

        # Configure job type
        if not job_config["run_train"] and job_config["run_test"]:
            job_config["is_test"] = True
            job_config["job_tag"] = "_test"
            log_info("Configured as test-only job")
        else:
            job_config["job_tag"] = "_train"
            log_info("Configured as training job")

        # Validate configuration
        job_config = check_config(job_config)

        # Create submit file
        create_submit_file(job_config, slurm_conf, submit_path)

        # Submit job
        job_id = submit_slurm_job(submit_path)

        # Success message
        print(f"\n{'=' * 60}")
        print("FDQ JOB SUBMISSION SUCCESSFUL")
        print(f"{'=' * 60}")
        print(f"SLURM Job ID:    {job_id}")
        print(f"Submit File:     {submit_path}")
        print(f"Experiment:      {exp_name}")
        print(f"Results Path:    {job_config['results_path']}")
        print(f"Log Path:        {job_config['log_path']}")
        print(f"{'=' * 60}")

    except FDQSubmitError as exc:
        log_error(str(exc))
        sys.exit(1)
    except KeyboardInterrupt:
        log_error("Operation cancelled by user")
        sys.exit(1)
    except Exception as exc:
        log_error(f"Unexpected error: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("FDQ SLURM Job Submission Utility")
    print("-" * 40)
    main()
