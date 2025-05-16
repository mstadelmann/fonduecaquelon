import sys
import os
import json
import copy
import getpass


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


def parse_input_file(exp_file_path):
    if not os.path.isfile(exp_file_path):
        print(f"Error: The file '{exp_file_path}' does not exist.")
        sys.exit(1)

    try:
        with open(exp_file_path, "r") as file:
            exp_file = json.load(file)
    except json.JSONDecodeError:
        raise ValueError(f"Error: The file '{exp_file_path}' is not a valid JSON file.")

    globals = exp_file.get("globals")
    parent = globals.get("parent", {})
    if parent != {}:
        if parent[0] == "/":
            parent_file_path = parent
        else:
            parent_file_path = os.path.abspath(
                os.path.join(os.path.split(exp_file_path)[0], parent)
            )

        if not os.path.exists(parent_file_path):
            raise FileNotFoundError(f"Error: File {parent_file_path} not found.")

        with open(parent_file_path, "r", encoding="utf8") as fp:
            try:
                parent_expfile = json.load(fp)
            except Exception as exc:
                raise ValueError(
                    f"Error loading experiment file {parent_file_path} (check syntax?)."
                ) from exc

        exp_file = recursive_dict_update(d_parent=parent_expfile, d_child=exp_file)

    return DictToObj(exp_file)


def main():
    if len(sys.argv) != 2:
        print(
            "Error: Exactly one argument is required which is the path to the JSON file."
        )
        print("Usage: python fdqsubmit.py <path_to_json_file>")

    template_path = "/cluster/home/stmd/dev/fonduecaquelon/src/fdq/fdq.submit.template"
    submit_path = template_path.replace(".submit.template", ".run")

    in_args = parse_input_file(sys.argv[1])
    slurm_conf = in_args.slurm_cluster

    job_config = {
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
        "auto_resubmit": True,
        "resume_chpt_path": "",
        "log_path": None,
        "stop_grace_time": 15,
        "python_env_module": None,
        "uv_env_module": None,
        "fdq_version": None,
        "exp_file_path": None,
        "temp_results_path": "/scratch/fdq_results/",
        "cluster_results_path": None,
        "submit_file_path": None,
    }

    for key in job_config:
        val = slurm_conf.get(key)
        if val is not None:
            job_config[key] = val

    job_config["job_name"] = in_args.globals.project[:20]
    job_config["user"] = getpass.getuser()
    job_config["exp_file_path"] = sys.argv[1]
    job_config["cluster_results_path"] = os.path.expanduser(in_args.store.results_path)
    job_config["submit_file_path"] = submit_path

    if not job_config["run_train"] and not job_config["run_test"]:
        job_config["is_test"] = True

    for key, value in job_config.items():
        if value is None:
            raise ValueError(
                f"Value for mandatory key'{key}' is None. Please update your config file!"
            )

    with open(template_path, "r") as f:
        template_content = f.read()

    for key, value in job_config.items():
        template_content = template_content.replace(f"#{key}#", str(value))

    with open(submit_path, "w") as f:
        f.write(template_content)

    print("done")


# submit_file_path
# "/cluster/home/stmd/dev/fonduecaquelon/src/fdq/fdq.submit"

# exp_file_path
# "/cluster/home/stmd/dev/fonduecaquelon/experiment_templates/segment_pets/segment_pets.json"

# cluster_results_path
# "/cluster/home/stmd/data/ML_data/results/"

# test only submission

if __name__ == "__main__":
    main()
