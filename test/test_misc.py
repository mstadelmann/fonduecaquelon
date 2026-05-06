"""Unit tests for utility functions in fdq.misc module."""

import unittest
from unittest.mock import MagicMock, patch, call

import torch

from fdq.misc import _log_wandb_images


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


if __name__ == "__main__":
    unittest.main()
