"""Tests for parameter-study runtime normalization."""

import unittest

from omegaconf import OmegaConf

from fdq.run_experiment import normalize_parameter_ranges


class TestParameterStudyNormalization(unittest.TestCase):
    """Unit tests for direct fdq execution with unexpanded study markers."""

    def test_disabled_parameter_study_uses_first_categorical_values(self):
        """Disabled categorical markers collapse to the first scalar value."""
        cfg = OmegaConf.create(
            {
                "mode": {"parameter_study": False},
                "data": {"OXPET": {"args": {"shuffle_train": ["true:false"]}}},
                "models": {
                    "simpleNet": {
                        "optimizer": {"class_name": [{"torch.optim.Adam": "torch.optim.SGD"}]},
                    }
                },
            }
        )

        normalized = normalize_parameter_ranges(cfg)

        self.assertIs(normalized.data.OXPET.args.shuffle_train, True)
        self.assertEqual(normalized.models.simpleNet.optimizer.class_name, "torch.optim.Adam")


if __name__ == "__main__":
    unittest.main()
