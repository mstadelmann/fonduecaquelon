"""Unit and end-to-end tests for the automatic (non-interactive) model dump mode."""

import os
import unittest
from unittest.mock import MagicMock, patch

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from fdq.dump import (
    dump_model_auto,
    export_onnx_model_auto,
    get_example_tensor_auto,
    select_checkpoint_auto,
    select_models_auto,
)
from fdq.experiment import fdqExperiment
from fdq.misc import build_dummy_hydra_paths
from fdq.run_experiment import expand_paths


class TestSelectCheckpointAuto(unittest.TestCase):
    """select_checkpoint_auto resolves a config string to the right experiment.mode setter."""

    def test_best_val_aliases(self):
        for alias in ("best", "best_val", "val", "validation", "BEST_VAL"):
            with self.subTest(alias=alias):
                experiment = MagicMock()
                select_checkpoint_auto(experiment, alias)
                experiment.mode.best_val.assert_called_once()
                experiment.mode.best_train.assert_not_called()
                experiment.mode.last.assert_not_called()

    def test_best_train_aliases(self):
        for alias in ("best_train", "train"):
            with self.subTest(alias=alias):
                experiment = MagicMock()
                select_checkpoint_auto(experiment, alias)
                experiment.mode.best_train.assert_called_once()

    def test_last(self):
        experiment = MagicMock()
        select_checkpoint_auto(experiment, "last")
        experiment.mode.last.assert_called_once()

    def test_invalid_checkpoint_raises(self):
        experiment = MagicMock()
        with self.assertRaises(ValueError):
            select_checkpoint_auto(experiment, "not_a_real_checkpoint")


class TestSelectModelsAuto(unittest.TestCase):
    """select_models_auto picks either a single named model or all of them."""

    def _make_experiment(self):
        experiment = MagicMock()
        experiment.device = "cpu"
        model_a = MagicMock()
        model_b = MagicMock()
        model_a.to.return_value = model_a
        model_b.to.return_value = model_b
        model_a.eval.return_value = model_a
        model_b.eval.return_value = model_b
        experiment.models = {"a": model_a, "b": model_b}
        return experiment, model_a, model_b

    def test_no_model_name_selects_all(self):
        experiment, model_a, model_b = self._make_experiment()
        selected = select_models_auto(experiment, None)
        self.assertEqual(set(selected.keys()), {"a", "b"})

    def test_explicit_model_name_selects_one(self):
        experiment, model_a, _ = self._make_experiment()
        selected = select_models_auto(experiment, "a")
        self.assertEqual(list(selected.keys()), ["a"])

    def test_unknown_model_name_raises(self):
        experiment, _, _ = self._make_experiment()
        with self.assertRaises(ValueError):
            select_models_auto(experiment, "does_not_exist")


class TestGetExampleTensorAuto(unittest.TestCase):
    """get_example_tensor_auto builds an example tensor from config, without prompting."""

    def _make_experiment(self, batch):
        experiment = MagicMock()
        experiment.device = "cpu"
        data_source = MagicMock()
        data_source.train_data_loader = [batch]
        experiment.data = {"MyData": data_source}
        return experiment

    def test_missing_input_source_raises(self):
        experiment = self._make_experiment(torch.rand(2, 3))
        with self.assertRaises(ValueError):
            get_example_tensor_auto(experiment, OmegaConf.create({}))

    def test_unknown_input_source_raises(self):
        experiment = self._make_experiment(torch.rand(2, 3))
        with self.assertRaises(ValueError):
            get_example_tensor_auto(experiment, OmegaConf.create({"input_source": "NotThere"}))

    def test_uses_real_batch_by_default(self):
        batch = torch.ones(2, 3)
        experiment = self._make_experiment(batch)
        result = get_example_tensor_auto(experiment, OmegaConf.create({"input_source": "MyData"}))
        self.assertTrue(torch.equal(result, batch))

    def test_random_input_replaces_values_but_keeps_shape(self):
        batch = torch.ones(2, 3)
        experiment = self._make_experiment(batch)
        result = get_example_tensor_auto(experiment, OmegaConf.create({"input_source": "MyData", "random_input": True}))
        self.assertEqual(result.shape, batch.shape)

    def test_unwraps_tuple_batches(self):
        batch = (torch.ones(2, 3), torch.zeros(2))
        experiment = self._make_experiment(batch)
        result = get_example_tensor_auto(experiment, OmegaConf.create({"input_source": "MyData"}))
        self.assertEqual(result.shape, (2, 3))

    def test_dtype_cast(self):
        batch = torch.ones(2, 3)
        experiment = self._make_experiment(batch)
        result = get_example_tensor_auto(
            experiment, OmegaConf.create({"input_source": "MyData", "input_dtype": "float64"})
        )
        self.assertEqual(result.dtype, torch.float64)

    def test_invalid_dtype_raises(self):
        experiment = self._make_experiment(torch.ones(2, 3))
        with self.assertRaises(ValueError):
            get_example_tensor_auto(
                experiment, OmegaConf.create({"input_source": "MyData", "input_dtype": "not_a_dtype"})
            )


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, x):
        return self.linear(x)


class TestExportOnnxModelAuto(unittest.TestCase):
    """export_onnx_model_auto writes a real ONNX file based on config, without prompting."""

    def test_torchscript_export_writes_file(self):
        with self._tmp_results_dir() as results_dir:
            experiment = MagicMock()
            experiment.results_dir = results_dir
            model = _TinyModel().eval()
            example = torch.rand(1, 3)

            path = export_onnx_model_auto(experiment, example, model, "tiny", OmegaConf.create({}))

            self.assertTrue(path.endswith("tiny_torchscript.onnx"))
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)

    def test_dynamo_export_writes_file(self):
        with self._tmp_results_dir() as results_dir:
            experiment = MagicMock()
            experiment.results_dir = results_dir
            model = _TinyModel().eval()
            example = torch.rand(1, 3)

            path = export_onnx_model_auto(experiment, example, model, "tiny", OmegaConf.create({"use_dynamo": True}))

            self.assertTrue(path.endswith("tiny_dynamo.onnx"))
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)

    def _tmp_results_dir(self):
        import tempfile

        return tempfile.TemporaryDirectory()


class TestDumpModelAutoOrchestration(unittest.TestCase):
    """dump_model_auto wires config -> checkpoint/model/example selection -> export, per model."""

    def test_raises_when_distributed(self):
        experiment = MagicMock()
        experiment.is_distributed.return_value = True
        with self.assertRaises(ValueError):
            dump_model_auto(experiment)

    def test_exports_every_model_and_raises_after_collecting_failures(self):
        experiment = MagicMock()
        experiment.is_distributed.return_value = False
        experiment.cfg.get.return_value = OmegaConf.create({"model_name": None})
        experiment.models = {"a": MagicMock(), "b": MagicMock()}

        with (
            patch("fdq.dump.select_checkpoint_auto") as mock_select_ckpt,
            patch("fdq.dump.select_models_auto", return_value={"a": MagicMock(), "b": MagicMock()}),
            patch("fdq.dump.get_example_tensor_auto", side_effect=[torch.rand(1), ValueError("boom")]),
            patch("fdq.dump.export_onnx_model_auto") as mock_export,
        ):
            with self.assertRaises(RuntimeError):
                dump_model_auto(experiment)

        mock_select_ckpt.assert_called_once()
        self.assertEqual(mock_export.call_count, 1)

    def test_succeeds_when_no_export_fails(self):
        experiment = MagicMock()
        experiment.is_distributed.return_value = False
        experiment.cfg.get.return_value = OmegaConf.create({"model_name": None})

        with (
            patch("fdq.dump.select_checkpoint_auto"),
            patch("fdq.dump.select_models_auto", return_value={"a": MagicMock()}),
            patch("fdq.dump.get_example_tensor_auto", return_value=torch.rand(1)),
            patch("fdq.dump.export_onnx_model_auto") as mock_export,
        ):
            dump_model_auto(experiment)

        mock_export.assert_called_once()


class TestDumpModelAutoEndToEnd(unittest.TestCase):
    """Trains a tiny MNIST classifier, then dumps it to ONNX via mode.dump_model=true, config-driven."""

    def setUp(self):
        os.environ["FDQ_UNITTEST"] = "1"
        os.environ["FDQ_UNITTEST_DIR"] = "1"
        os.environ["FDQ_UNITTEST_CONF"] = "1"

        self.config_dir = os.path.join(os.path.dirname(__file__), "test_experiment")
        self.conf_name = "mnist_testexp_dense_ci" if os.getenv("GITHUB_ACTIONS") else "mnist_testexp_dense"

        os.environ["FDQ_UNITTEST_DIR"] = self.config_dir
        os.environ["FDQ_UNITTEST_CONF"] = self.conf_name

    def _compose_cfg(self) -> DictConfig:
        with initialize_config_dir(version_base=None, config_dir=self.config_dir):
            cfg: DictConfig = compose(
                config_name=self.conf_name,
                overrides=["hydra.run.dir=.", "hydra.job.chdir=False"],
            )
        with open_dict(cfg):
            cfg.hydra_paths = build_dummy_hydra_paths(self.config_dir, self.conf_name)
        return cfg

    def test_dump_model_auto_exports_onnx_after_training(self):
        cfg = self._compose_cfg()
        cfg = expand_paths(cfg)

        train_experiment = fdqExperiment(cfg, rank=0)
        getattr(train_experiment.mode, "unittest")()
        train_experiment.prepareTraining()
        train_experiment.trainer.fdq_train(train_experiment)

        # Fresh experiment instance, mirroring a standalone `mode.dump_model=true` process
        # that loads the just-trained checkpoint from disk instead of reusing live weights.
        dump_experiment = fdqExperiment(cfg, rank=0)
        dump_model_auto(dump_experiment)

        onnx_path = os.path.join(dump_experiment.results_dir, "simpleNet_torchscript.onnx")
        self.assertTrue(os.path.exists(onnx_path), f"Expected ONNX file at {onnx_path}")
        self.assertGreater(os.path.getsize(onnx_path), 0)


if __name__ == "__main__":
    unittest.main()
