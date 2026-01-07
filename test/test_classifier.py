"""This module contains unit tests for the MNIST classifier experiment.

It verifies the training process, checks the existence of result files,
and ensures that the test results meet the expected criteria.
"""

import json
import os
import unittest
import glob
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra import compose
from hydra import initialize_config_dir
from fdq.experiment import fdqExperiment
from fdq.testing import run_test, find_model_path
from fdq.run_experiment import expand_paths


class TestMNISTClassifier(unittest.TestCase):
    """Unit tests for the MNIST classifier experiment."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        os.environ["FDQ_UNITTEST"] = "1"
        os.environ["FDQ_UNITTEST_DIR"] = "1"
        os.environ["FDQ_UNITTEST_CONF"] = "1"

        self.config_dir = os.path.join(os.path.dirname(__file__), "test_experiment")
        self.conf_name = "mnist_testexp_dense_ci" if os.getenv("GITHUB_ACTIONS") else "mnist_testexp_dense"

        os.environ["FDQ_UNITTEST_DIR"] = self.config_dir
        os.environ["FDQ_UNITTEST_CONF"] = self.conf_name

        workspace_root = os.getenv("GITHUB_WORKSPACE", os.getcwd())
        print("--------------------------------------------")
        print(f"Using workspace root: {workspace_root}")
        print(f"current file path: {os.path.abspath(__file__)}")
        print("--------------------------------------------")

    def _compose_cfg(self) -> DictConfig:
        """Compose Hydra config without changing CWD (like run_experiment)."""
        try:
            with initialize_config_dir(version_base=None, config_dir=self.config_dir):
                cfg: DictConfig = compose(
                    config_name=self.conf_name,
                    overrides=["hydra.run.dir=.", "hydra.job.chdir=False"],
                )
        except Exception:
            # Fallback: relative path init (older Hydra)
            from hydra import initialize

            conf_rel = os.path.relpath(self.config_dir, os.getcwd())
            with initialize(version_base=None, config_path=conf_rel):
                cfg = compose(
                    config_name=self.conf_name,
                    overrides=["hydra.run.dir=.", "hydra.job.chdir=False"],
                )
        # Inject dummy hydra_paths similar to run_experiment.get_hydra_paths
        config_dir = self.config_dir
        config_name = self.conf_name
        root_config_path = os.path.join(config_dir, f"{config_name}.yaml")

        def _collect_parents(cfg_path: str, seen: set[str]) -> list[str]:
            parents: list[str] = []
            try:
                y = OmegaConf.load(cfg_path)
            except Exception:
                return parents

            defaults = y.get("defaults", []) or []
            for item in defaults:
                name = None
                if isinstance(item, str):
                    name = item
                elif isinstance(item, dict) and len(item) == 1:
                    k, v = next(iter(item.items()))
                    name = v if isinstance(v, str) else k

                if not name or name == "_self_":
                    continue

                if "keys" in name:
                    continue

                parent_path = os.path.join(config_dir, f"{name}.yaml")
                if os.path.exists(parent_path) and parent_path not in seen:
                    seen.add(parent_path)
                    parents.append(parent_path)
                    parents.extend(_collect_parents(parent_path, seen))
            return parents

        parents = _collect_parents(root_config_path, set()) if os.path.exists(root_config_path) else []

        hydra_paths = {
            "config_name": config_name,
            "config_dir": config_dir,
            "root_config_path": root_config_path,
            "parents": parents,
        }

        with open_dict(cfg):
            cfg.hydra_paths = hydra_paths
        return cfg

    def test_run_train(self):
        """Test the training process and result validation for the MNIST classifier experiment."""
        cfg = self._compose_cfg()
        cfg = expand_paths(cfg)
        experiment = fdqExperiment(cfg, rank=0)
        # Set to unittest mode (suppress lint error for dynamic method)
        getattr(experiment.mode, "unittest")()  # Set to unittest mode

        # Store temp config path for cleanup if created
        # temp_config_path = exp_path if os.getenv("GITHUB_ACTIONS") else None
        experiment.prepareTraining()
        experiment.trainer.fdq_train(experiment)

        res_dir = experiment.results_dir

        # check if infofile exists (did experiment start at all?)
        info_file = os.path.join(res_dir, "info.json")
        self.assertTrue(os.path.exists(info_file))

        # check that loss is going down
        history_file = os.path.join(res_dir, "history.json")
        with open(history_file, encoding="utf8") as json_file:
            history = json.load(json_file)
            self.assertTrue(history["train"][0] > history["train"][-1])

        run_test(experiment)

        res_d, _ = find_model_path(experiment)

        # check if test results file exists
        res_paths = glob.glob(res_d + "/test/*/00_test_results_*")
        print("----------------------------------------")
        print("Found test results files:\n" + "\n".join(res_paths))
        print("----------------------------------------")
        self.assertTrue(len(res_paths) > 0)

        with open(res_paths[0], encoding="utf8") as json_file:
            testres = json.load(json_file)
            self.assertTrue(testres["test results"] > 0.2)

        # Cleanup temporary config file if created for CI
        # if temp_config_path and os.path.exists(temp_config_path):
        #     try:
        #         os.unlink(temp_config_path)
        #     except OSError:
        #         pass  # Ignore cleanup errors


if __name__ == "__main__":
    unittest.main()
