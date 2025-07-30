"""This module contains unit tests for the MNIST classifier experiment.

It verifies the training process, checks the existence of result files,
and ensures that the test results meet the expected criteria.
"""

import argparse
import json
import os
import unittest
import glob

from fdq.experiment import fdqExperiment
from fdq.testing import run_test, find_model_path


class TestMNISTClassifier(unittest.TestCase):
    """Unit tests for the MNIST classifier experiment."""

    def _create_ci_config(self, workspace_root):
        """Create a CI-specific config file with absolute paths."""
        import tempfile

        # Load the base CI config
        base_config_path = os.path.join(
            os.path.split(os.path.abspath(__file__))[0], "mnist_testexp_dense_ci.json"
        )

        return base_config_path

        with open(base_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Update paths to be absolute
        config["data"]["MNIST"]["processor"] = os.path.join(
            workspace_root, "experiment_templates/mnist/mnist_preparator.py"
        )
        config["models"]["simpleNet"]["path"] = os.path.join(
            workspace_root, "src/networks/simpleNet.py"
        )
        config["train"]["path"] = os.path.join(
            workspace_root, "experiment_templates/mnist/train_mnist.py"
        )
        config["test"]["processor"] = os.path.join(
            workspace_root, "experiment_templates/mnist/mnist_test.py"
        )

        # Create temporary config file
        fd, temp_config_path = tempfile.mkstemp(suffix=".json", prefix="mnist_test_ci_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            return temp_config_path
        except Exception:
            if fd:
                os.close(fd)
            raise

    def test_run_train(self):
        """Test the training process and result validation for the MNIST classifier experiment."""
        # Use different config file for CI vs local testing
        if os.getenv("GITHUB_ACTIONS"):
            # For CI, create config with absolute paths
            workspace_root = os.getenv("GITHUB_WORKSPACE", os.getcwd())
            print("--------------------------------------------")
            print(f"Using workspace root: {workspace_root}")
            print(f"current file path: {os.path.abspath(__file__)}")
            print("--------------------------------------------")
            config_file = self._create_ci_config(workspace_root)
        else:
            config_file = os.path.join(
                os.path.split(os.path.abspath(__file__))[0],
                "mnist_testexp_dense.json",
            )

        exp_path = config_file

        args = argparse.Namespace(
            experimentfile=exp_path,
            train_model=True,
            test_model_ia=False,
            test_model_auto=False,
            dump_model=False,
            print_model=False,
            resume_path=None,
        )

        experiment = fdqExperiment(args, rank=0)
        # Set to unittest mode (suppress lint error for dynamic method)
        getattr(experiment.mode, "unittest")()  # Set to unittest mode

        # Store temp config path for cleanup if created
        temp_config_path = exp_path if os.getenv("GITHUB_ACTIONS") else None
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
        self.assertTrue(len(res_paths) > 0)

        with open(res_paths[0], encoding="utf8") as json_file:
            testres = json.load(json_file)
            self.assertTrue(testres["test results"] > 0.2)

        # Cleanup temporary config file if created for CI
        if temp_config_path and os.path.exists(temp_config_path):
            try:
                os.unlink(temp_config_path)
            except OSError:
                pass  # Ignore cleanup errors


if __name__ == "__main__":
    unittest.main()
