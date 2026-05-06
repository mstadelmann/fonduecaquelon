"""Tests for parameter-study runtime guards."""

import unittest

from omegaconf import OmegaConf

from fdq.run_experiment import reject_unexpanded_parameter_studies


class TestParameterStudyNormalization(unittest.TestCase):
    """Unit tests for direct fdq execution with unexpanded study markers."""

    def test_parameter_key_marker_must_be_expanded_before_direct_run(self):
        """Keys ending in @p error if they reach fdq without submit expansion."""
        cfg = OmegaConf.create(
            {
                "mode": {},
                "data": {"OXPET": {"args": {"shuffle_train@p": ["true:false"]}}},
                "models": {
                    "simpleNet": {
                        "optimizer": {"class_name@p": [{"torch.optim.Adam": "torch.optim.SGD"}]},
                    }
                },
            }
        )

        with self.assertRaisesRegex(ValueError, "must be expanded by fdq_submit.py"):
            reject_unexpanded_parameter_studies(cfg)

    def test_unmarked_range_like_lists_stay_untouched(self):
        """Unmarked lists are left unchanged."""
        cfg = OmegaConf.create(
            {
                "mode": {},
                "data": {"OXPET": {"args": {"shuffle_train": ["true:false"]}}},
            }
        )

        normalized = reject_unexpanded_parameter_studies(cfg)

        self.assertEqual(OmegaConf.to_container(normalized.data.OXPET.args.shuffle_train), ["true:false"])

    def test_transform_definitions_are_not_normalized_as_parameter_studies(self):
        """One-item transform lists with parameter dictionaries stay untouched."""
        cfg = OmegaConf.create(
            {
                "mode": {},
                "transforms": {
                    "resize_and_pad_bilinear": [
                        {
                            "ResizeMaxDimPad": {
                                "max_dim": 256,
                                "interpol_mode": "bilinear",
                            }
                        }
                    ]
                },
            }
        )

        normalized = reject_unexpanded_parameter_studies(cfg)

        self.assertEqual(
            OmegaConf.to_container(normalized.transforms.resize_and_pad_bilinear, resolve=True),
            [{"ResizeMaxDimPad": {"max_dim": 256, "interpol_mode": "bilinear"}}],
        )


if __name__ == "__main__":
    unittest.main()
