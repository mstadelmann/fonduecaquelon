"""SLURM job submission utility for fonduecaquelon experiments."""

import sys
import os
import re
import copy
import getpass
import subprocess
from datetime import datetime
from decimal import Decimal
from itertools import product
from typing import Any
from pathlib import Path

try:
    import yaml
    from yaml.constructor import SafeConstructor
except ModuleNotFoundError:
    yaml = None
    SafeConstructor = None


class FDQSubmitError(Exception):
    """Custom exception for FDQ submission errors."""

    pass


PARAMETER_RANGE_RE = re.compile(
    r"^\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"\s*:\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"\s*:\s*"
    r"(\d+)"
    r"\s*$"
)
PARAMETER_STUDY_SUFFIX = "@p"


if yaml is not None:

    class FDQYamlLoader(yaml.SafeLoader):
        """YAML loader that keeps colon-separated parameter ranges as strings."""

        pass

    def _construct_yaml_int(loader: yaml.Loader, node: yaml.Node) -> int | str:
        value = loader.construct_scalar(node)
        if ":" in value:
            return value
        return SafeConstructor.construct_yaml_int(loader, node)

    FDQYamlLoader.add_constructor("tag:yaml.org,2002:int", _construct_yaml_int)
else:
    FDQYamlLoader = None


def log_info(message: str) -> None:
    """Log an info message."""
    print(f"[INFO] {message}")


def log_error(message: str) -> None:
    """Log an error message."""
    print(f"[ERROR] {message}", file=sys.stderr)


def log_warning(message: str) -> None:
    """Log a warning message."""
    print(f"[WARNING] {message}")


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
CONFIG_PATH=#config_path#
CONFIG_NAME=#config_name#
SCRATCH_RESULTS_PATH=#scratch_results_path#
SCRATCH_DATA_PATH=#scratch_data_path#
RESULTS_PATH=#results_path#
SUBMIT_FILE_PATH=#submit_file_path#
PY_MODULE=#python_env_module#
UV_MODULE=#uv_env_module#
CUDA_MODULE=#cuda_env_module#
FDQ_VERSION=#fdq_version#
FDQ_TEST_REPO=#fdq_test_repo# # if True, install fdq from https://test.pypi.org
PARAMETER_OVERRIDES="#parameter_overrides#"
export FDQ_PARAMETER_RUN_TAG="#parameter_run_tag#"
export FDQ_PARAMETER_STUDY_PATHS="#parameter_study_paths#"
export FDQ_TEST_RESULTS_DIR="#test_results_dir#"
export FDQ_EXPERIMENT_NAME="#experiment_name#"
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
echo "CONFIG_PATH: $CONFIG_PATH"
echo "CONFIG_NAME: $CONFIG_NAME"
echo "SCRATCH_RESULTS_PATH: $SCRATCH_RESULTS_PATH"
echo "SCRATCH_DATA_PATH: $SCRATCH_DATA_PATH"
echo "RESULTS_PATH: $RESULTS_PATH"
echo "PYTHON MODULE: $PY_MODULE"
echo "UV MODULE: $UV_MODULE"
echo "CUDA MODULE: $CUDA_MODULE"
echo "FDQ VERSION: $FDQ_VERSION"
echo "PARAMETER OVERRIDES: $PARAMETER_OVERRIDES"
echo "PARAMETER RUN TAG: $FDQ_PARAMETER_RUN_TAG"
echo "PARAMETER STUDY PATHS: $FDQ_PARAMETER_STUDY_PATHS"
echo "TEST RESULTS DIR: $FDQ_TEST_RESULTS_DIR"

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
    echo "Experiment file: $CONFIG_PATH"/"$CONFIG_NAME"
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
    echo "Experiment file: $CONFIG_PATH"/"$CONFIG_NAME"
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
        echo "Starting training from beginning with command:"
        echo "fdq --config-path \"$CONFIG_PATH\" --config-name \"$CONFIG_NAME\" $PARAMETER_OVERRIDES mode.run_test_auto=false &"
        fdq --config-path "$CONFIG_PATH" --config-name "$CONFIG_NAME" $PARAMETER_OVERRIDES mode.run_test_auto=false &
    elif [ -f "$RESUME_CHPT_PATH" ]; then
        echo "Resuming training from checkpoint: $RESUME_CHPT_PATH"
        fdq --config-path "$CONFIG_PATH" --config-name "$CONFIG_NAME" $PARAMETER_OVERRIDES mode.resume_chpt_path="$RESUME_CHPT_PATH" mode.run_test_auto=false &
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
    echo "Starting test with command:"
    echo "fdq --config-path \"$CONFIG_PATH\" --config-name \"$CONFIG_NAME\" $PARAMETER_OVERRIDES mode.run_train=false mode.run_test_auto=true &"
    fdq --config-path "$CONFIG_PATH" --config-name "$CONFIG_NAME" $PARAMETER_OVERRIDES mode.run_train=false mode.run_test_auto=true &
    fdq_pid=$!
    echo "Testing process started with PID: $fdq_pid"
    wait $fdq_pid
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
    echo "Submit test in new job..."
    echo -----------------------------------------------------------
    
    # Extract test-specific resource requirements
    GRES_TEST=$(awk -F= '/^GRES_TEST=/{print $2}' "$SUBMIT_FILE_PATH")
    MEM_TEST=$(awk -F= '/^MEM_TEST=/{print $2}' "$SUBMIT_FILE_PATH")
    CPUS_TEST=$(awk -F= '/^CPUS_TEST=/{print $2}' "$SUBMIT_FILE_PATH")

    # Find the exact results folder created by this training job.
    if [ -n "$FDQ_PARAMETER_RUN_TAG" ]; then
        TRAINED_RESULTS_DIR=$(find "$SCRATCH_RESULTS_PATH" -type d -name "*__${SLURM_JOB_ID}${FDQ_PARAMETER_RUN_TAG}" | head -n 1)
    else
        TRAINED_RESULTS_DIR=$(find "$SCRATCH_RESULTS_PATH" -type d -name "*__${SLURM_JOB_ID}" | head -n 1)
    fi

    if [ -n "$TRAINED_RESULTS_DIR" ]; then
        TRAINED_RESULTS_DIR="${TRAINED_RESULTS_DIR%/}"
        SCRATCH_RESULTS_ROOT="${SCRATCH_RESULTS_PATH%/}"
        RESULTS_ROOT="${RESULTS_PATH%/}"
        case "$TRAINED_RESULTS_DIR" in
            "$SCRATCH_RESULTS_ROOT"/*)
                TRAINED_RESULTS_DIR="$RESULTS_ROOT/${TRAINED_RESULTS_DIR#"$SCRATCH_RESULTS_ROOT"/}"
                ;;
        esac
        echo "Auto-test will load trained results from: $TRAINED_RESULTS_DIR"
    else
        echo "WARNING: Could not determine exact trained results directory. Auto-test will fall back to the newest matching experiment."
    fi
    
    # Create test job submit script
    sed -e "s|IS_TEST=False|IS_TEST=True|g" \
        -e "s|RUN_TRAIN=True|RUN_TRAIN=False|g" \
        -e "s|RUN_TEST=True|RUN_TEST=False|g" \
        -e "s|^export FDQ_TEST_RESULTS_DIR=.*|export FDQ_TEST_RESULTS_DIR=\"$TRAINED_RESULTS_DIR\"|g" \
        -e "s|job_config[\"job_tag\"] = \"_train\"|job_config[\"job_tag\"] = \"_test\"|g" \
        -e "s|^\\(#SBATCH --output=.*\\)_train\\([^/[:space:]]*\\.out\\)|\\1_test\\2|g" \
        -e "s|^\\(#SBATCH --error=.*\\)_train\\([^/[:space:]]*\\.err\\)|\\1_test\\2|g" \
        -e "s|^#SBATCH --gres=.*|#SBATCH --gres=$GRES_TEST|g" \
        -e "s|^#SBATCH --mem=.*|#SBATCH --mem=$MEM_TEST|g" \
        -e "s|^#SBATCH --cpus-per-task=.*|#SBATCH --cpus-per-task=$CPUS_TEST|g" \
        "$SCRATCH_SUBMIT_FILE_PATH" > "$SCRATCH_SUBMIT_FILE_PATH.test"
        
    # Copy test submit script to source submit directory
    SUBMIT_SOURCE_PATH="${SUBMIT_FILE_PATH%/*}"
    cp "$SCRATCH_SUBMIT_FILE_PATH.test" "$SUBMIT_SOURCE_PATH"
    
    echo "Submitting test job: sbatch --job-name=fdq-test $SCRATCH_SUBMIT_FILE_PATH.test"
    if sbatch --job-name=fdq-test "$SCRATCH_SUBMIT_FILE_PATH.test"; then
        echo "Test job submitted successfully"
    else
        echo "ERROR: Failed to submit test job"
        exit 1
    fi
elif [ "$RUN_TEST" == True ] && [ $RETVALUE -ne 0 ] && [ "$IS_TEST" == False ]; then
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


def _get_default_config_name(default_item: Any) -> str | None:
    """Return the config name from a Hydra defaults entry."""
    if isinstance(default_item, str):
        return default_item
    if isinstance(default_item, dict) and len(default_item) == 1:
        key, value = next(iter(default_item.items()))
        return value if isinstance(value, str) else key
    return None


def _resolve_default_path(config_dir: Path, default_name: str) -> Path:
    """Resolve a defaults entry to a YAML path relative to the current config."""
    default_path = Path(default_name).expanduser()
    if not default_path.is_absolute():
        default_path = config_dir / default_path
    if default_path.suffix not in {".yaml", ".yml"}:
        default_path = default_path.with_suffix(".yaml")
    return default_path


def load_conf_file(path: str, _seen: set[Path] | None = None) -> dict:
    """Load an experiment YAML file with recursive Hydra defaults merging.

    Args:
        path: Path to the experiment configuration YAML file
        _seen: Internal recursion guard for inherited defaults

    Returns:
        The merged configuration as a dictionary

    Raises:
        FDQSubmitError: If configuration cannot be loaded or is invalid
    """
    if _seen is None:
        _seen = set()

    if yaml is None:
        raise FDQSubmitError(
            "PyYAML is required to load experiment YAML files. Install it with: python -m pip install pyyaml"
        )

    try:
        p = Path(path).expanduser().resolve()
        if p in _seen:
            raise ValueError(f"Recursive defaults reference detected for {p}")
        _seen.add(p)

        with p.open("r", encoding="utf-8") as f:
            conf = yaml.load(f, Loader=FDQYamlLoader)
        if not isinstance(conf, dict):
            raise ValueError("YAML root must be a mapping/dict")

        defaults = conf.get("defaults", []) or []
        merged_conf: dict[str, Any] = {}
        for default_item in defaults:
            default_name = _get_default_config_name(default_item)
            if not default_name or default_name == "_self_":
                continue

            default_path = _resolve_default_path(p.parent, default_name)
            if not default_path.exists():
                raise ValueError(f"Defaults entry '{default_name}' not found at {default_path}")

            parent_conf = load_conf_file(str(default_path), _seen)
            merged_conf = recursive_dict_update(merged_conf, parent_conf)

        conf_without_defaults = {key: value for key, value in conf.items() if key != "defaults"}
        return recursive_dict_update(merged_conf, conf_without_defaults)

    except Exception as exc:
        raise FDQSubmitError(f"Failed to load configuration from {path}: {exc}") from exc
    finally:
        _seen.discard(Path(path).expanduser().resolve())


def _decimal_to_parameter_value(value: Decimal) -> int | float:
    """Return a YAML-safe numeric scalar for a parsed parameter range value."""
    if value == value.to_integral_value():
        return int(value)
    return float(value)


def _parse_scalar_parameter_value(value: Any) -> Any:
    """Parse a parameter-study scalar into the type Hydra/YAML would normally use."""
    if not isinstance(value, str) or yaml is None:
        return value
    if value == "":
        return value

    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError:
        return value
    return value if isinstance(parsed, dict | list) else parsed


def _parse_parameter_values(value: Any) -> list[Any] | None:
    """Parse a parameter-study value list into concrete config values."""
    if not isinstance(value, list):
        return None

    if len(value) == 1 and isinstance(value[0], str):
        match = PARAMETER_RANGE_RE.match(value[0])
        if match:
            start = Decimal(match.group(1))
            stop = Decimal(match.group(2))
            count = int(match.group(3))
            if count < 1:
                raise FDQSubmitError(f"Parameter-study range '{value[0]}' must use a count of at least 1")
            if count == 1:
                return [_decimal_to_parameter_value(start)]

            step = (stop - start) / Decimal(count - 1)
            return [_decimal_to_parameter_value(start + step * Decimal(index)) for index in range(count)]

    categorical_values = _parse_categorical_parameter_values(value)
    if categorical_values is None:
        return None
    return categorical_values


def _parse_categorical_parameter_values(value: list[Any]) -> list[Any] | None:
    """Parse non-numeric `@p` values."""
    if _is_single_scalar_mapping(value):
        first, second = next(iter(value[0].items()))
        values = [first, second]
    elif len(value) == 1 and isinstance(value[0], str):
        values = [_parse_scalar_parameter_value(part.strip()) for part in value[0].split(":")]
    else:
        values = [_parse_scalar_parameter_value(part) for part in value]

    if len(values) < 2:
        return None
    if any(item == "" for item in values):
        raise FDQSubmitError(f"Parameter-study list '{value}' must not contain empty values")
    return values


def _is_single_scalar_mapping(value: Any) -> bool:
    """Return whether a loaded marker contains one scalar-to-scalar mapping."""
    if not (isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict) and len(value[0]) == 1):
        return False
    first, second = next(iter(value[0].items()))
    return not isinstance(first, dict | list) and not isinstance(second, dict | list)


def _format_parameter_value(value: Any) -> str:
    """Format a categorical parameter-study value for a Hydra CLI override."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def _unmarked_parameter_name(key: str) -> str:
    """Return a config key without the parameter-study suffix."""
    if not key.endswith(PARAMETER_STUDY_SUFFIX):
        return key
    unmarked = key[: -len(PARAMETER_STUDY_SUFFIX)]
    if not unmarked:
        raise FDQSubmitError(
            f"Parameter-study key '{key}' must include a parameter name before '{PARAMETER_STUDY_SUFFIX}'"
        )
    return unmarked


def _parameter_path(path: tuple[str, ...], key: str) -> str:
    """Return the dotted output path for a parameter-study key."""
    return ".".join((*path, _unmarked_parameter_name(key)))


def _is_parameter_key(key: Any) -> bool:
    """Return whether a config key marks a parameter study."""
    return isinstance(key, str) and key.endswith(PARAMETER_STUDY_SUFFIX)


def find_parameter_ranges(conf: dict[str, Any]) -> list[tuple[str, list[Any]]]:
    """Find all parameter-study markers in a merged experiment configuration."""
    ranges: list[tuple[str, list[Any]]] = []

    def visit(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if _is_parameter_key(key):
                    parameter_path = _parameter_path(path, key)
                    parsed_range = _parse_parameter_values(child)
                    if parsed_range is None:
                        raise FDQSubmitError(f"Parameter-study key '{parameter_path}' must use a list of values")
                    ranges.append((parameter_path, parsed_range))
                else:
                    visit(child, (*path, str(key)))

    visit(conf, ())
    return ranges


def _materialize_parameter_run(node: Any, path: tuple[str, ...], run_values: dict[str, Any]) -> Any:
    """Return a config copy with `@p` keys replaced by concrete unmarked keys."""
    if isinstance(node, dict):
        materialized: dict[str, Any] = {}
        parameter_keys = {_unmarked_parameter_name(key) for key in node if _is_parameter_key(key)}
        for key, value in node.items():
            if _is_parameter_key(key):
                unmarked_key = _unmarked_parameter_name(key)
                parameter_path = ".".join((*path, unmarked_key))
                if unmarked_key in materialized:
                    raise FDQSubmitError(
                        f"Parameter-study key '{parameter_path}' conflicts with an existing '{unmarked_key}' key"
                    )
                materialized[unmarked_key] = run_values[parameter_path]
            else:
                if key in parameter_keys:
                    continue
                if key in materialized:
                    raise FDQSubmitError(f"Duplicate materialized key '{'.'.join((*path, str(key)))}'")
                materialized[key] = _materialize_parameter_run(value, (*path, str(key)), run_values)
        return materialized
    if isinstance(node, list):
        return [_materialize_parameter_run(value, (*path, str(index)), run_values) for index, value in enumerate(node)]
    return copy.deepcopy(node)


def _write_concrete_parameter_config(
    config: dict[str, Any],
    submit_dir: str,
    timestamp: str,
    config_name: str,
    run_index: int,
) -> str:
    """Write a materialized parameter-study config and return its basename without extension."""
    if yaml is None:
        raise FDQSubmitError("PyYAML is required to write parameter-study configs")

    concrete_name = f"{timestamp}__{config_name.replace(' ', '_')}__p{run_index:03d}.yaml"
    concrete_path = os.path.join(submit_dir, concrete_name)
    try:
        with open(concrete_path, "w", encoding="utf8") as config_file:
            yaml.safe_dump(config, config_file, sort_keys=False)
    except OSError as exc:
        raise FDQSubmitError(f"Cannot create concrete parameter-study config {concrete_path}: {exc}") from exc

    log_info(f"Created concrete parameter-study config: {concrete_path}")
    return os.path.splitext(concrete_name)[0]


def build_parameter_study_runs(exp_config: dict[str, Any]) -> list[tuple[dict[str, Any], str, dict[str, Any]]]:
    """Build concrete configs and Hydra override strings for a parameter study.

    Study markers use config keys ending in `@p`, for example
    `lr@p: [0.001:0.005:5]` or `class_name@p: ["torch.optim.Adam":"torch.optim.SGD"]`.
    """
    ranges = find_parameter_ranges(exp_config)
    if not ranges:
        return [(exp_config, "", {})]

    parameter_names = [name for name, _values in ranges]
    value_lists = [values for _name, values in ranges]

    runs: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
    for combination in product(*value_lists):
        run_values = dict(zip(parameter_names, combination, strict=True))
        concrete_config = _materialize_parameter_run(exp_config, (), run_values)
        overrides = " ".join(f"{name}={_format_parameter_value(value)}" for name, value in run_values.items())
        runs.append((concrete_config, overrides, run_values))

    return runs


def get_default_config(slurm_conf: Any, mode_config: Any) -> dict[str, Any]:
    """Return a job configuration dictionary with defaults, updated from the given SLURM config.

    Args:
        slurm_conf (dict): SLURM configuration dictionary.
        mode_config (dict): Mode configuration dictionary controlling run and resume behavior.

    Returns:
        dict: Job configuration dictionary with updated values.
    """
    job_config: dict[str, Any] = {
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
        "config_path": None,
        "config_name": None,
        "job_name": None,
        "experiment_name": None,
        "scratch_results_path": "/scratch/fdq_results/",
        "scratch_data_path": "/scratch/fdq_data/",
        "results_path": None,
        "submit_file_path": None,
        "parameter_overrides": "",
        "parameter_run_tag": "",
        "parameter_study_paths": "",
        "test_results_dir": "",
    }

    for key in job_config:
        val = slurm_conf.get(key)
        if val is not None:
            job_config[key] = val

    job_config["run_train"] = mode_config.get("run_train", False)
    job_config["run_test"] = mode_config.get("run_test_auto", False)
    job_config["resume_chpt_path"] = mode_config.get("resume_chpt_path", "")
    if mode_config.get("run_test_interactive"):
        raise FDQSubmitError("Interactive test mode is not supported for SLURM job submission")
    if mode_config.get("dump_model"):
        raise FDQSubmitError("Model dumping is currently not supported for SLURM job submission")
    if mode_config.get("run_inference"):
        raise FDQSubmitError("Inference mode is currently not supported for SLURM job submission")
    if mode_config.get("print_model_summary"):
        raise FDQSubmitError("Printing model summary is not supported for SLURM job submission")

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
        elif (
            key in {"parameter_overrides", "parameter_run_tag", "parameter_study_paths", "test_results_dir"}
            and value == ""
        ):
            job_config[key] = ""
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
        job_config.setdefault("parameter_overrides", "")
        job_config.setdefault("parameter_run_tag", "")
        job_config.setdefault("parameter_study_paths", "")
        job_config.setdefault("test_results_dir", "")
        job_config.setdefault("job_name", job_config.get("config_name", ""))
        job_config.setdefault("experiment_name", job_config.get("job_name", job_config.get("config_name", "")))

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

        # Handle additional pip packages
        add_packages = slurm_conf.get("additional_pip_packages")
        if add_packages is None:
            template_content = template_content.replace("#additional_pip_packages#", "")
        elif isinstance(add_packages, list) and len(add_packages) > 0:
            packages_cmd = "\n".join(f"uv pip install '{pkg}'" for pkg in add_packages)
            log_info(f"Adding {len(add_packages)} additional pip packages")
            template_content = template_content.replace("#additional_pip_packages#", packages_cmd)

        else:
            raise FDQSubmitError(f"additional_pip_packages must be a list of strings, got {type(add_packages)}")

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


def get_config_path() -> str:
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
    expanded_path = os.path.expanduser(config_path)
    abs_path = os.path.abspath(expanded_path)

    if not os.path.exists(abs_path):
        raise FDQSubmitError(f"Experiment configuration file not found: {abs_path}")

    if not os.path.isfile(abs_path):
        raise FDQSubmitError(f"Experiment configuration file is not a file: {abs_path}")

    if not abs_path.endswith(".yaml"):
        raise FDQSubmitError(f"Experiment configuration file must have a .yaml extension: {abs_path}")

    return abs_path


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


def print_submission_summary(
    submitted_jobs: list[tuple[str, str, str]],
    config_name: str,
    config_path: str,
    last_job_config: dict[str, Any] | None,
    parameter_ranges: list[tuple[str, list[Any]]],
) -> None:
    """Print the final job submission summary."""
    print(f"\n{'=' * 60}")
    print("FDQ JOB SUBMISSION SUCCESSFUL")
    print(f"{'=' * 60}")
    if parameter_ranges:
        parameter_names = ", ".join(name for name, _values in parameter_ranges)
        print("Parameter Study: enabled")
        print(f"Parameter Runs:  {len(submitted_jobs)}")
        print(f"Sweep Params:    {parameter_names}")

    if len(submitted_jobs) == 1:
        job_id, submit_path, parameter_overrides = submitted_jobs[0]
        print(f"SLURM Job ID:    {job_id}")
        print(f"Submit File:     {submit_path}")
        if parameter_overrides:
            print(f"Overrides:       {parameter_overrides}")
    else:
        print(f"Submitted Jobs:  {len(submitted_jobs)}")
        for job_id, submit_path, parameter_overrides in submitted_jobs:
            print(f"  {job_id}: {parameter_overrides} ({submit_path})")
    print(f"Experiment Name: {config_name}")
    print(f"Experiment Path: {config_path}")
    if last_job_config is not None:
        print(f"Results Path:    {last_job_config['results_path']}")
        print(f"Log Path:        {last_job_config['log_path']}")
    print(f"{'=' * 60}")


def main() -> None:
    """Main entry point for submitting a job to SLURM."""
    try:
        full_config_path = get_config_path()

        log_info(f"Loading experiment configuration: {full_config_path}")

        exp_config = load_conf_file(full_config_path)
        config_path = os.path.dirname(full_config_path)
        config_name = os.path.basename(full_config_path).replace(".yaml", "")

        parameter_ranges = find_parameter_ranges(exp_config)
        parameter_runs = build_parameter_study_runs(exp_config)
        if parameter_ranges:
            parameter_names = ", ".join(name for name, _values in parameter_ranges)
            log_info(f"Parameter study enabled for {parameter_names}: submitting {len(parameter_runs)} jobs")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submitted_jobs: list[tuple[str, str, str]] = []
        last_job_config: dict[str, Any] | None = None

        for run_index, (run_exp_config, parameter_overrides, run_values) in enumerate(parameter_runs, start=1):
            slurm_conf = run_exp_config.get("slurm_cluster")
            mode_config = run_exp_config.get("mode")

            if slurm_conf is None:
                raise FDQSubmitError(
                    "Missing 'slurm_cluster' section in configuration file. "
                    "This section is required for SLURM job submission."
                )

            if mode_config is None:
                raise FDQSubmitError(
                    "Missing 'mode' section in configuration file. This section is required for SLURM job submission."
                )

            # Setup job configuration
            job_config = get_default_config(slurm_conf, mode_config)

            # Set paths and basic info
            job_config["config_path"] = config_path
            job_config["config_name"] = config_name
            job_config["job_name"] = config_name
            job_config["experiment_name"] = config_name
            job_config["user"] = getpass.getuser()
            job_config["parameter_overrides"] = parameter_overrides
            job_config["parameter_study_paths"] = " ".join(run_values.keys())

            parameter_run_tag = ""
            if parameter_ranges:
                parameter_run_tag = f"_p{run_index:03d}"
            job_config["parameter_run_tag"] = parameter_run_tag

            job_config["results_path"] = run_exp_config.get("store", {}).get("results_path")
            # validate results path
            if job_config["results_path"] is None:
                raise FDQSubmitError("Configuration missing 'store.results_path' setting")

            # Setup submit file path
            base_path = os.path.join(
                os.path.expanduser(job_config["log_path"]),
                "submitted_jobs",
            )
            os.makedirs(base_path, exist_ok=True)

            run_config_name = config_name
            if parameter_ranges:
                run_config_name = _write_concrete_parameter_config(
                    run_exp_config,
                    base_path,
                    timestamp,
                    config_name,
                    run_index,
                )
                job_config["parameter_overrides"] = ""

            submit_suffix = parameter_run_tag.replace("_", "__", 1)
            submit_filename = f"{timestamp}__{config_name.replace(' ', '_')}{submit_suffix}.submit"
            submit_path = os.path.join(base_path, submit_filename)
            job_config["submit_file_path"] = submit_path
            job_config["config_path"] = base_path if parameter_ranges else config_path
            job_config["config_name"] = run_config_name

            # Configure job type
            if not job_config["run_train"] and job_config["run_test"]:
                job_config["is_test"] = True
                job_config["job_tag"] = "_test"
                log_info("Configured as test-only job")
            else:
                job_config["job_tag"] = "_train"
                log_info("Configured as training job")

            if parameter_ranges:
                job_config["job_tag"] = f"{job_config['job_tag']}_p{run_index:03d}"
                run_values_msg = ", ".join(f"{key}={value}" for key, value in run_values.items())
                log_info(f"Preparing parameter run {run_index}/{len(parameter_runs)}: {run_values_msg}")

            # Validate configuration
            job_config = check_config(job_config)

            # Create submit file
            create_submit_file(job_config, slurm_conf, submit_path)

            # Submit job
            job_id = submit_slurm_job(submit_path)
            submitted_jobs.append((job_id, submit_path, parameter_overrides))
            last_job_config = job_config

        print_submission_summary(
            submitted_jobs,
            config_name,
            config_path,
            last_job_config,
            parameter_ranges,
        )

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
