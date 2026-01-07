"""Unit tests for fdqExperiment instantiation.

This module contains simple tests to verify that fdqExperiment objects
can be instantiated without errors.
"""

import os
import unittest
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra import compose
from hydra import initialize_config_dir
from fdq.experiment import fdqExperiment
from fdq.misc import build_dummy_hydra_paths


class TestFdqExperimentInstantiation(unittest.TestCase):
    """Test cases for fdqExperiment instantiation."""

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
        # Inject dummy hydra_paths via shared helper
        hydra_paths = build_dummy_hydra_paths(self.config_dir, self.conf_name)

        with open_dict(cfg):
            cfg.hydra_paths = hydra_paths
        return cfg

    def test_experiment_instantiation(self) -> None:
        """FdqExperiment can be instantiated from Hydra DictConfig."""
        cfg = self._compose_cfg()
        experiment = fdqExperiment(cfg, rank=0)

        self.assertIsNotNone(experiment)
        self.assertEqual(experiment.rank, 0)
        self.assertIsNotNone(experiment.project)
        self.assertIsNotNone(experiment.experimentName)

    def test_experiment_with_different_ranks(self):
        """Instantiate with different ranks (non-distributed)."""
        cfg = self._compose_cfg()
        for rank in [0, 1]:
            with self.subTest(rank=rank):
                experiment = fdqExperiment(cfg, rank=rank)
                self.assertEqual(experiment.rank, rank)
                self.assertIsNotNone(experiment)

    def test_experiment_attributes_after_instantiation(self):
        """Essential attributes exist after instantiation."""
        cfg = self._compose_cfg()
        experiment = fdqExperiment(cfg, rank=0)

        essential_attributes = [
            "rank",
            "project",
            "experimentName",
            "device",
            "mode",
            "creation_time",
            "current_epoch",
            "models",
            "data",
            "transformers",
            "optimizers",
            "losses",
        ]
        for attr in essential_attributes:
            with self.subTest(attribute=attr):
                self.assertTrue(
                    hasattr(experiment, attr),
                    f"Experiment missing essential attribute: {attr}",
                )


if __name__ == "__main__":
    unittest.main()
