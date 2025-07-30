"""Unit tests for fdqExperiment instantiation.

This module contains simple tests to verify that fdqExperiment objects
can be instantiated without errors.
"""

import argparse
import os
import unittest

from fdq.experiment import fdqExperiment


class TestFdqExperimentInstantiation(unittest.TestCase):
    """Test cases for fdqExperiment instantiation."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Use the CI config file which should have relative paths
        self.config_file = os.path.join(
            os.path.dirname(__file__), "mnist_testexp_dense.json"
        )
        self.assertTrue(
            os.path.exists(self.config_file),
            f"Config file {self.config_file} not found",
        )

    def test_experiment_instantiation(self):
        """Test that fdqExperiment can be instantiated without error."""
        # Create minimal arguments needed for instantiation
        args = argparse.Namespace(
            experimentfile=self.config_file,
            train_model=False,  # Don't actually train
            test_model_ia=False,
            test_model_auto=False,
            dump_model=False,
            print_model=False,
            resume_path=None,
        )

        # Test instantiation with rank 0 (single process)
        try:
            experiment = fdqExperiment(args, rank=0)

            # Basic assertions to verify the object was created properly
            self.assertIsNotNone(experiment)
            self.assertEqual(experiment.rank, 0)
            self.assertIsNotNone(experiment.project)
            self.assertIsNotNone(experiment.experimentName)
            self.assertIsInstance(experiment.inargs, argparse.Namespace)

            # Check that basic attributes are set
            self.assertEqual(experiment.inargs.experimentfile, self.config_file)
            self.assertFalse(experiment.inargs.train_model)

            print("✓ Successfully instantiated fdqExperiment")
            print(f"  Project: {experiment.project}")
            print(f"  Experiment name: {experiment.experimentName}")
            print(f"  Rank: {experiment.rank}")
            print(f"  Device: {experiment.device}")

        except (ImportError, ValueError, FileNotFoundError, AttributeError) as e:
            self.fail(f"Failed to instantiate fdqExperiment: {e}")

    def test_experiment_with_different_ranks(self):
        """Test that fdqExperiment can be instantiated with different ranks."""
        args = argparse.Namespace(
            experimentfile=self.config_file,
            train_model=False,
            test_model_ia=False,
            test_model_auto=False,
            dump_model=False,
            print_model=False,
            resume_path=None,
        )

        # Test with rank 0 and rank 1 (but not distributed mode)
        for rank in [0, 1]:
            with self.subTest(rank=rank):
                try:
                    experiment = fdqExperiment(args, rank=rank)
                    self.assertEqual(experiment.rank, rank)
                    self.assertIsNotNone(experiment)
                except (
                    ImportError,
                    ValueError,
                    FileNotFoundError,
                    AttributeError,
                ) as e:
                    self.fail(
                        f"Failed to instantiate fdqExperiment with rank {rank}: {e}"
                    )

    def test_experiment_attributes_after_instantiation(self):
        """Test that key attributes are properly set after instantiation."""
        args = argparse.Namespace(
            experimentfile=self.config_file,
            train_model=False,
            test_model_ia=False,
            test_model_auto=False,
            dump_model=False,
            print_model=False,
            resume_path=None,
        )

        experiment = fdqExperiment(args, rank=0)

        # Test that essential attributes exist
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

    def tearDown(self):
        """Clean up after each test method."""
        # Clean up any temporary files or resources if needed
        return


if __name__ == "__main__":
    unittest.main()
