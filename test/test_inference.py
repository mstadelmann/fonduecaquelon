"""Unit tests for fdq.inference module."""

import unittest
from unittest.mock import MagicMock, patch

from fdq.inference import inference_model


class TestInferenceModelRocmGuard(unittest.TestCase):
    """TensorRT-based inference is NVIDIA-only; it must not run on ROCm builds."""

    def test_skips_on_rocm_build(self):
        experiment = MagicMock()
        experiment.is_distributed.return_value = False

        with patch("fdq.inference.is_rocm_build", return_value=True):
            inference_model(experiment)

        experiment.setupData.assert_not_called()

    def test_proceeds_on_cuda_build(self):
        experiment = MagicMock()
        experiment.is_distributed.return_value = False

        with (
            patch("fdq.inference.is_rocm_build", return_value=False),
            patch("fdq.inference.find_onnx_models", return_value="model.onnx"),
            patch("fdq.inference.get_precision_choice", return_value="fp32"),
            patch("fdq.inference.run_tensorrt_inference") as mock_run,
        ):
            inference_model(experiment)

        experiment.setupData.assert_called_once()
        mock_run.assert_called_once_with("model.onnx", "fp32", experiment=experiment)


if __name__ == "__main__":
    unittest.main()
