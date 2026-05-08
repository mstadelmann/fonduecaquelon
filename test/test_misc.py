"""Unit tests for utility functions in fdq.misc module."""

import os
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import torch
from omegaconf import OmegaConf

from fdq.misc import _log_wandb_images, init_wandb


class TestLogWandbImages(unittest.TestCase):
    """Tests for _log_wandb_images function."""

    def test_none_images_does_nothing(self):
        """Passing None should not call wandb.log at all."""
        with patch("fdq.misc.wandb") as mock_wandb:
            _log_wandb_images(None)
            mock_wandb.log.assert_not_called()

    def test_single_3d_tensor_image(self):
        """A 3D tensor (C×H×W) should be passed directly to wandb.Image."""
        img = torch.zeros(1, 28, 28)
        images = {"name": "test_img", "data": img}
        with patch("fdq.misc.wandb") as mock_wandb:
            mock_wandb.Image.return_value = MagicMock()
            _log_wandb_images(images)
            mock_wandb.Image.assert_called_once_with(img, caption=None)
            mock_wandb.log.assert_called_once()

    def test_4d_tensor_image_without_captions(self):
        """A 4D tensor (N×C×H×W) should be split into N individual wandb.Image objects."""
        batch_size = 8
        img = torch.zeros(batch_size, 1, 28, 28)
        images = {"name": "val_samples", "data": img}

        with patch("fdq.misc.wandb") as mock_wandb:
            mock_wandb.Image.return_value = MagicMock()
            _log_wandb_images(images)

            # wandb.Image should be called once per image in the batch
            self.assertEqual(mock_wandb.Image.call_count, batch_size)
            # Each call should use the individual slice with the same (None) caption
            for i in range(batch_size):
                torch.testing.assert_close(mock_wandb.Image.call_args_list[i][0][0], img[i])
                self.assertIsNone(mock_wandb.Image.call_args_list[i][1]["caption"])

            # wandb.log should be called once with a list of images
            mock_wandb.log.assert_called_once()
            logged_value = mock_wandb.log.call_args[0][0]["val_samples"]
            self.assertIsInstance(logged_value, list)
            self.assertEqual(len(logged_value), batch_size)

    def test_4d_tensor_image_with_list_captions(self):
        """A 4D tensor with a list of captions should assign each caption to its image."""
        batch_size = 4
        img = torch.zeros(batch_size, 1, 28, 28)
        captions = [f"caption_{i}" for i in range(batch_size)]
        images = {"name": "val_samples", "data": img, "captions": captions}

        with patch("fdq.misc.wandb") as mock_wandb:
            mock_wandb.Image.return_value = MagicMock()
            _log_wandb_images(images)

            self.assertEqual(mock_wandb.Image.call_count, batch_size)
            for i in range(batch_size):
                self.assertEqual(mock_wandb.Image.call_args_list[i][1]["caption"], captions[i])

    def test_4d_tensor_image_with_scalar_caption(self):
        """A 4D tensor with a single string caption should apply it to all images."""
        batch_size = 3
        img = torch.zeros(batch_size, 1, 28, 28)
        caption = "shared caption"
        images = {"name": "val_samples", "data": img, "captions": caption}

        with patch("fdq.misc.wandb") as mock_wandb:
            mock_wandb.Image.return_value = MagicMock()
            _log_wandb_images(images)

            self.assertEqual(mock_wandb.Image.call_count, batch_size)
            for i in range(batch_size):
                self.assertEqual(mock_wandb.Image.call_args_list[i][1]["caption"], caption)

    def test_4d_normalized_float_tensor_is_rescaled_for_display(self):
        """A normalized float tensor should be rescaled before being passed to wandb.Image."""
        img = torch.tensor(
            [
                [[[-0.4, 0.0], [0.4, 0.8]]],
                [[[1.0, 1.6], [2.2, 2.8]]],
            ],
            dtype=torch.float32,
        )
        images = {"name": "val_samples", "data": img}

        with patch("fdq.misc.wandb") as mock_wandb:
            mock_wandb.Image.return_value = MagicMock()
            _log_wandb_images(images)

            for idx in range(img.shape[0]):
                logged_img = mock_wandb.Image.call_args_list[idx][0][0]
                self.assertGreaterEqual(float(logged_img.min()), 0)
                self.assertLessEqual(float(logged_img.max()), 1)
                self.assertAlmostEqual(float(logged_img.min()), 0)
                self.assertAlmostEqual(float(logged_img.max()), 1)

    def test_path_based_image_is_passed_directly(self):
        """An image dict with a file path should pass the path to wandb.Image unchanged."""
        images = {"name": "file_img", "path": "/tmp/image.png"}
        with patch("fdq.misc.wandb") as mock_wandb:
            mock_wandb.Image.return_value = MagicMock()
            _log_wandb_images(images)
            mock_wandb.Image.assert_called_once_with("/tmp/image.png", caption=None)

    def test_list_of_images(self):
        """A list of image dicts should each be logged separately."""
        img_a = torch.zeros(1, 8, 8)
        img_b = torch.zeros(1, 8, 8)
        images = [
            {"name": "img_a", "data": img_a},
            {"name": "img_b", "data": img_b},
        ]
        with patch("fdq.misc.wandb") as mock_wandb:
            mock_wandb.Image.return_value = MagicMock()
            _log_wandb_images(images)
            self.assertEqual(mock_wandb.log.call_count, 2)

    def test_invalid_images_type_raises(self):
        """Non-dict, non-list input should raise ValueError."""
        with self.assertRaises(ValueError):
            _log_wandb_images("not_a_valid_input")


class TestInitWandb(unittest.TestCase):
    """Tests for wandb run naming."""

    def test_parameter_study_wandb_name_uses_one_timestamp_and_run_tag(self):
        """Parameter study names keep the run timestamp once and append the parameter tag."""
        experiment = SimpleNamespace(
            cfg=SimpleNamespace(
                store=SimpleNamespace(
                    wandb_project="mnist_classifier",
                    wandb_entity="stmd",
                    wandb_key="secret",
                )
            ),
            is_slurm=True,
            previous_slurm_job_id=None,
            slurm_job_id="15298",
            creation_time=datetime(2026, 5, 8, 8, 11, 53),
            experimentName="mnist_class_dense_param_study",
            funky_name="clever_chebyshev",
            mode=SimpleNamespace(op_mode=SimpleNamespace(train=True)),
            wandb_initialized=False,
        )

        with patch.dict(os.environ, {"FDQ_PARAMETER_RUN_TAG": "_p001"}, clear=False):
            with patch("fdq.misc.wandb") as mock_wandb:
                mock_wandb.run.dir = "/tmp/wandb"

                self.assertTrue(init_wandb(experiment))

        mock_wandb.init.assert_called_once()
        wandb_name = mock_wandb.init.call_args.kwargs["name"]
        self.assertEqual(
            wandb_name,
            "20260508_081153__mnist_class_dense_pa__clever_chebyshev_p001__15298",
        )
        self.assertEqual(wandb_name.count("20260508_"), 1)

    def test_omegaconf_config_is_converted_before_passing_to_wandb(self):
        """Wandb receives a plain resolved config rather than a DictConfig."""
        cfg = OmegaConf.create(
            {
                "store": {
                    "wandb_project": "mnist_classifier",
                    "wandb_entity": "stmd",
                    "wandb_key": "secret",
                },
                "train": {
                    "args": {
                        "lr": 0.01,
                    }
                },
            }
        )
        experiment = SimpleNamespace(
            cfg=cfg,
            is_slurm=False,
            previous_slurm_job_id=None,
            slurm_job_id=None,
            creation_time=datetime(2026, 5, 8, 8, 11, 53),
            experimentName="mnist_class_dense",
            funky_name="clever_chebyshev",
            mode=SimpleNamespace(op_mode=SimpleNamespace(train=True)),
            wandb_initialized=False,
        )

        with patch("fdq.misc.wandb") as mock_wandb:
            mock_wandb.run.dir = "/tmp/wandb"

            self.assertTrue(init_wandb(experiment))

        wandb_config = mock_wandb.init.call_args.kwargs["config"]
        self.assertIsInstance(wandb_config, dict)
        self.assertEqual(wandb_config["store"]["wandb_project"], "mnist_classifier")
        self.assertEqual(wandb_config["train"]["args"]["lr"], 0.01)


if __name__ == "__main__":
    unittest.main()
