import sys
import os
import json
import copy

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
        with open(exp_file_path, 'r') as file:
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
            raise FileNotFoundError(
                f"Error: File {parent_file_path} not found."
            )

        with open(parent_file_path, "r", encoding="utf8") as fp:
            try:
                parent_expfile = json.load(fp)
            except Exception as exc:
                raise ValueError(
                    f"Error loading experiment file {parent_file_path} (check syntax?)."
                ) from exc

        exp_file = recursive_dict_update(
            d_parent=parent_expfile, d_child=exp_file
        )

    return DictToObj(exp_file)




def main():

    if len(sys.argv) != 2:
        print("Error: Exactly one argument is required which is the path to the JSON file.")
        print("Usage: python fdqsubmit.py <path_to_json_file>")

    exp_def = parse_input_file(sys.argv[1]).slurm_cluster.to_dict()


    job_time = exp_def.get("time", 1000)
    ntasks = exp_def.get("ntasks", 1)
    cpus_per_task = exp_def.get("cpus-per-task", 8)
    nodes = exp_def.get("nodes", 1)
    gres = exp_def.get("gres", "")
    mem = exp_def.get("mem", "32G")
    partition = exp_def.get("partition", "gpu")
    account = exp_def.get("account", "admin")
    job_time = exp_def.get("time", 1000)
    log_root = exp_def.get("log_path", 1000)
                               
    print("done")






if __name__ == "__main__":
    main()