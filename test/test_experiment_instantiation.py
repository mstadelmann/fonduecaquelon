"""Unit tests for fdqExperiment instantiation.

This module contains simple tests to verify that fdqExperiment objects
can be instantiated without errors.
"""

import os
import unittest
from unittest.mock import patch
from omegaconf import DictConfig, open_dict
from hydra import compose
from hydra import initialize_config_dir
from fdq.experiment import fdqExperiment
from fdq.misc import DictToObj, build_dummy_hydra_paths


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

    def test_experiment_name_can_be_overridden_for_generated_configs(self) -> None:
        """Generated parameter-study config filenames should not leak into run names."""
        cfg = self._compose_cfg()
        cfg.hydra_paths.config_name = "20260508_080633__mnist_class_dense_param_study__p001"

        with patch.dict(os.environ, {"FDQ_EXPERIMENT_NAME": "mnist_class_dense_param_study"}, clear=False):
            experiment = fdqExperiment(cfg, rank=0)

        self.assertEqual(experiment.experimentName, "mnist_class_dense_param_study")

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

    def test_loss_best_flags_reset_each_epoch(self):
        """Best-loss flags represent improvements in the current epoch only."""
        experiment = fdqExperiment.__new__(fdqExperiment)
        experiment._trainLoss = float("inf")
        experiment._valLoss = float("inf")
        experiment.bestTrainLoss = float("inf")
        experiment.bestValLoss = float("inf")
        experiment.trainLoss_per_ep = []
        experiment.valLoss_per_ep = []
        experiment.new_best_train_loss = False
        experiment.new_best_val_loss = False
        experiment.new_best_train_loss_ep_id = None
        experiment.new_best_val_loss_ep_id = None
        experiment.current_epoch = 0
        experiment.nb_epochs = 2
        experiment.world_size = 1

        experiment.trainLoss = 1.0
        experiment.valLoss = 2.0
        self.assertTrue(experiment.new_best_train_loss)
        self.assertTrue(experiment.new_best_val_loss)

        experiment.on_epoch_start(epoch=1)
        self.assertFalse(experiment.new_best_train_loss)
        self.assertFalse(experiment.new_best_val_loss)

        experiment.trainLoss = 1.5
        experiment.valLoss = 1.0
        self.assertFalse(experiment.new_best_train_loss)
        self.assertTrue(experiment.new_best_val_loss)

    def test_train_loss_early_stop_is_checked_with_val_loss_enabled(self):
        """Train-loss early stopping is independent from validation-loss checks."""
        experiment = fdqExperiment.__new__(fdqExperiment)
        experiment.cfg = DictToObj(
            {
                "train": {
                    "args": {
                        "early_stop_nan": None,
                        "early_stop_val_loss": 2,
                        "early_stop_train_loss": 2,
                    }
                }
            }
        )
        experiment.trainLoss_per_ep = [1.0, 2.0]
        experiment.valLoss_per_ep = [1.0, 0.5]
        experiment.bestTrainLoss = 0.5
        experiment.bestValLoss = 0.5
        experiment.current_epoch = 1
        experiment.early_stop_detected = False
        experiment.early_stop_reason = ""

        self.assertTrue(experiment.check_early_stop())
        self.assertEqual(experiment.early_stop_reason, "TrainLoss_stagnated")


class TestPrepareDdpDataArgs(unittest.TestCase):
    """Tests for _prepare_ddp_data_args DDP DataLoader argument handling."""

    def _make_experiment(self, world_size=2):
        """Create a minimal fdqExperiment stub with DDP enabled."""
        exp = fdqExperiment.__new__(fdqExperiment)
        exp.world_size = world_size
        return exp

    def _make_data_source(self, num_workers=4, prefetch_factor=2, ddp_num_workers=None):
        """Return an OmegaConf data-source stub."""
        from omegaconf import OmegaConf

        d = {"args": {"num_workers": num_workers, "prefetch_factor": prefetch_factor}}
        if ddp_num_workers is not None:
            d["args"]["ddp_num_workers"] = ddp_num_workers
        return OmegaConf.create(d)

    # ------------------------------------------------------------------
    # Non-distributed: no modifications expected
    # ------------------------------------------------------------------

    def test_no_op_when_not_distributed(self):
        """Non-distributed experiments leave args unchanged."""
        exp = self._make_experiment(world_size=1)
        ds = self._make_data_source(num_workers=4, prefetch_factor=2)
        exp._prepare_ddp_data_args("ds", ds)
        self.assertEqual(ds.args.num_workers, 4)
        self.assertEqual(ds.args.prefetch_factor, 2)

    # ------------------------------------------------------------------
    # DDP, no ddp_num_workers override → num_workers forced to 0
    # ------------------------------------------------------------------

    def test_forces_num_workers_to_zero_in_ddp(self):
        """DDP without ddp_num_workers forces num_workers to 0."""
        exp = self._make_experiment()
        ds = self._make_data_source(num_workers=4, prefetch_factor=2)
        exp._prepare_ddp_data_args("ds", ds)
        self.assertEqual(ds.args.num_workers, 0)

    def test_resets_prefetch_factor_when_num_workers_forced_to_zero(self):
        """prefetch_factor is reset to None when num_workers is forced to 0.

        DataLoader raises ValueError if prefetch_factor is set but num_workers=0,
        so _prepare_ddp_data_args must clear it together with num_workers.
        """
        exp = self._make_experiment()
        ds = self._make_data_source(num_workers=4, prefetch_factor=2)
        exp._prepare_ddp_data_args("ds", ds)
        self.assertIsNone(ds.args.prefetch_factor)

    def test_no_prefetch_factor_key_is_safe(self):
        """Absence of prefetch_factor in args does not cause an error."""
        from omegaconf import OmegaConf

        exp = self._make_experiment()
        ds = OmegaConf.create({"args": {"num_workers": 4}})
        # Should not raise
        exp._prepare_ddp_data_args("ds", ds)
        self.assertEqual(ds.args.num_workers, 0)

    def test_num_workers_already_zero_is_unchanged(self):
        """If num_workers is already 0, no warning is issued and args unchanged."""
        exp = self._make_experiment()
        ds = self._make_data_source(num_workers=0, prefetch_factor=2)
        exp._prepare_ddp_data_args("ds", ds)
        # num_workers stays 0; prefetch_factor is NOT touched (it stays at 2 from
        # config but caller must pass None to DataLoader when workers=0).
        self.assertEqual(ds.args.num_workers, 0)

    # ------------------------------------------------------------------
    # DDP with ddp_num_workers override → workers kept, prefetch untouched
    # ------------------------------------------------------------------

    def test_ddp_num_workers_overrides_num_workers(self):
        """ddp_num_workers replaces num_workers when values differ."""
        exp = self._make_experiment()
        ds = self._make_data_source(num_workers=4, prefetch_factor=2, ddp_num_workers=2)
        exp._prepare_ddp_data_args("ds", ds)
        self.assertEqual(ds.args.num_workers, 2)

    def test_ddp_num_workers_keeps_prefetch_factor(self):
        """prefetch_factor is NOT reset when ddp_num_workers keeps workers alive."""
        exp = self._make_experiment()
        ds = self._make_data_source(num_workers=4, prefetch_factor=2, ddp_num_workers=4)
        exp._prepare_ddp_data_args("ds", ds)
        # num_workers already equals ddp_num_workers: no change
        self.assertEqual(ds.args.num_workers, 4)
        self.assertEqual(ds.args.prefetch_factor, 2)

    def test_ddp_num_workers_same_value_no_change(self):
        """ddp_num_workers equal to num_workers leaves both args untouched."""
        exp = self._make_experiment()
        ds = self._make_data_source(num_workers=4, prefetch_factor=2, ddp_num_workers=4)
        exp._prepare_ddp_data_args("ds", ds)
        self.assertEqual(ds.args.num_workers, 4)
        self.assertEqual(ds.args.prefetch_factor, 2)



if __name__ == '__main__':
    unittest.main()
