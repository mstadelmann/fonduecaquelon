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

                # skip secret/key overlays
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
