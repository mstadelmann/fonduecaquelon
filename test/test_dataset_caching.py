import unittest
import tempfile
import shutil
import os
import numpy as np
import torch
import h5py
from omegaconf import DictConfig, OmegaConf
from unittest.mock import Mock, patch, MagicMock
from torch.utils.data import Dataset, DataLoader

# Add the src directory to the path
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Hydra imports
from omegaconf import DictConfig
from hydra import compose

try:
    from hydra import initialize_config_dir  # Hydra >=1.2
except ImportError:
    from hydra import initialize as initialize_config_dir  # fallback

from fdq.dataset_caching import (
    get_file_size_mb,
    hash_conf,
    find_valid_cache_file,
    cache_datasets_ddp_handler,
    CachedDataset,
    _save_samples_to_hdf5,
    _save_sample_to_group,
    cache_dataloader,
    reconfig_orig_dataloader,
)
from fdq.misc import DictToObj


class MockDataset(Dataset):
    """Mock dataset for testing."""

    def __init__(self, samples):
        """Initialize mock dataset with samples."""
        self.samples = samples

    def __len__(self):
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Get sample at index."""
        return self.samples[idx]


class MockExperiment:
    """Mock experiment object for testing."""

    def __init__(self, cfg: DictConfig | None = None, is_main=True, rank=0):
        """Initialize mock experiment."""
        self._is_main = is_main
        self.rank = rank
        self.barrier_calls = 0
        # Minimal Hydra-like cfg with mode flags so dataset_caching.get_loaders_to_cache works
        if cfg is None:
            self.cfg = DictToObj(
                {
                    "mode": DictToObj(
                        {
                            "run_train": True,
                            "run_test_auto": False,
                            "run_test_interactive": False,
                        }
                    )
                }
            )
        else:
            self.cfg = cfg

    def is_main_process(self):
        return self._is_main

    def is_child_process(self):
        return not self._is_main

    def is_distributed(self):
        return not self._is_main

    def dist_barrier(self):
        self.barrier_calls += 1

    def import_class(self, file_path=None, class_path=None):
        # Return a mock class for testing
        return Mock()


class MockProcessor:
    """Mock data processor for testing."""

    def create_datasets(self, experiment, args):
        # Create mock datasets with 1D label arrays to avoid scalar dataset issues
        train_samples = [
            {"image": torch.randn(3, 32, 32), "label": torch.tensor([0])},
            {"image": torch.randn(3, 32, 32), "label": torch.tensor([1])},
        ]
        val_samples = [
            {"image": torch.randn(3, 32, 32), "label": torch.tensor([0])},
        ]

        train_dataset = MockDataset(train_samples)
        val_dataset = MockDataset(val_samples)

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        return {"train_data_loader": train_loader, "val_data_loader": val_loader, "test_data_loader": None}


class TestDatasetCaching(unittest.TestCase):
    """Test suite for dataset caching functionality."""

    def setUp(self):
        """Set up test environment with Hydra cfg."""
        os.environ["FDQ_UNITTEST"] = "1"
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.temp_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Compose Hydra cfg like other tests
        self.config_dir = os.path.join(os.path.dirname(__file__), "test_experiment")
        self.conf_name = "mnist_testexp_dense_ci" if os.getenv("GITHUB_ACTIONS") else "mnist_testexp_dense"
        try:
            with initialize_config_dir(version_base=None, config_dir=self.config_dir):
                self.cfg: DictConfig = compose(
                    config_name=self.conf_name,
                    overrides=["hydra.run.dir=.", "hydra.job.chdir=False"],
                )
        except Exception:
            from hydra import initialize

            conf_rel = os.path.relpath(self.config_dir, os.getcwd())
            with initialize(version_base=None, config_path=conf_rel):
                self.cfg = compose(
                    config_name=self.conf_name,
                    overrides=["hydra.run.dir=.", "hydra.job.chdir=False"],
                )

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_file_size_mb(self):
        """Test file size calculation."""
        # Test with non-existent file
        self.assertEqual(get_file_size_mb("non_existent_file.txt"), 0.0)

        # Test with existing file
        test_file = os.path.join(self.temp_dir, "test_file.txt")
        with open(test_file, "w") as f:
            f.write("A" * 1024 * 1024)  # 1MB of data

        size_mb = get_file_size_mb(test_file)
        self.assertAlmostEqual(size_mb, 1.0, places=1)

    def test_hash_conf(self):
        """Test configuration hashing."""
        config1 = OmegaConf.create({"param1": "value1", "param2": 42})
        config2 = OmegaConf.create({"param1": "value1", "param2": 42})
        config3 = OmegaConf.create({"param1": "value1", "param2": 43})

        # Same configs should produce same hash
        self.assertEqual(hash_conf(config1), hash_conf(config2))

        # Different configs should produce different hashes
        self.assertNotEqual(hash_conf(config1), hash_conf(config3))

        # Hash should be deterministic
        hash1 = hash_conf(config1)
        hash2 = hash_conf(config1)
        self.assertEqual(hash1, hash2)

    def test_find_valid_cache_file(self):
        """Test finding valid cache files."""
        data_name = "test_dataset"
        split_name = "train"
        expected_hash = "test_hash_123"

        # Test with no cache files
        result = find_valid_cache_file(self.cache_dir, data_name, split_name, expected_hash)
        self.assertIsNone(result)

        # Create a cache file with matching hash
        cache_file = os.path.join(self.cache_dir, f"{data_name}_{split_name}_20230101_120000.h5")
        with h5py.File(cache_file, "w") as f:
            f.attrs["fdq_data_hash"] = expected_hash
            f.attrs["num_samples"] = 10

        result = find_valid_cache_file(self.cache_dir, data_name, split_name, expected_hash)
        self.assertEqual(result, cache_file)

        # Test with non-matching hash
        wrong_hash = "wrong_hash_456"
        result = find_valid_cache_file(self.cache_dir, data_name, split_name, wrong_hash)
        self.assertIsNone(result)

    def test_save_and_load_samples_hdf5(self):
        """Test saving and loading samples to/from HDF5."""
        # Test data
        samples = [
            {"image": np.random.randn(3, 32, 32), "label": 0},
            {"image": np.random.randn(3, 32, 32), "label": 1},
            (np.random.randn(3, 32, 32), 1),
            np.random.randn(3, 32, 32),
            "string_sample",
            42,
        ]

        cache_file = os.path.join(self.temp_dir, "test_cache.h5")
        config_hash = "test_hash"

        # Save samples
        _save_samples_to_hdf5(samples, cache_file, config_hash)

        # Verify file was created and has correct attributes
        self.assertTrue(os.path.exists(cache_file))

        with h5py.File(cache_file, "r") as f:
            self.assertEqual(f.attrs["num_samples"], len(samples))
            self.assertEqual(f.attrs["fdq_data_hash"], config_hash)
            self.assertEqual(len([k for k in f.keys() if k.startswith("sample_")]), len(samples))

    def test_cached_dataset(self):
        """Test CachedDataset functionality."""
        # Create test cache file
        samples = [
            {"image": np.random.randn(3, 32, 32), "label": 0},
            {"image": np.random.randn(3, 32, 32), "label": 1},
        ]

        cache_file = os.path.join(self.temp_dir, "test_dataset_cache.h5")
        _save_samples_to_hdf5(samples, cache_file, "test_hash")

        # Create mock data source
        data_source = DictToObj({"caching": DictToObj({"nondeterministic_transforms": DictToObj({"processor": None})})})

        # Use Hydra-backed MockExperiment
        experiment = MockExperiment(cfg=self.cfg)

        # Test CachedDataset
        cached_dataset = CachedDataset(cache_file, data_source, experiment)

        self.assertEqual(len(cached_dataset), len(samples))

        # Test indexing
        sample = cached_dataset[0]
        self.assertIn("image", sample)
        self.assertIn("label", sample)
        self.assertIsInstance(sample["image"], torch.Tensor)

        # Test out of bounds
        with self.assertRaises(IndexError):
            _ = cached_dataset[len(samples)]

    def test_cache_dataloader(self):
        """Test caching a DataLoader."""
        # Create test dataset
        samples = [
            {"image": torch.randn(3, 32, 32), "label": 0},
            {"image": torch.randn(3, 32, 32), "label": 1},
        ]

        dataset = MockDataset(samples)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

        # Cache the dataloader
        cached_samples = cache_dataloader(dataloader, "train")

        self.assertEqual(len(cached_samples), len(samples))

        # Check that tensors were converted to numpy
        for sample in cached_samples:
            self.assertIsInstance(sample["image"], np.ndarray)
            # Label should be a numpy array (integers get converted when batched)
            self.assertTrue(isinstance(sample["label"], int | np.integer | np.ndarray))

    def test_reconfig_orig_dataloader(self):
        """Test dataloader reconfiguration."""
        # Create original dataloader
        dataset = MockDataset([1, 2, 3])
        original_loader = DataLoader(
            dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
        )

        # Reconfigure it
        new_loader = reconfig_orig_dataloader(original_loader)

        # Check that it was reconfigured correctly
        self.assertEqual(new_loader.batch_size, 1)
        self.assertFalse(new_loader.dataset != original_loader.dataset)  # Same dataset
        self.assertEqual(new_loader.num_workers, original_loader.num_workers)
        self.assertFalse(new_loader.pin_memory)
        self.assertFalse(new_loader.drop_last)

    @patch("fdq.dataset_caching.cache_datasets")
    def test_cache_datasets_ddp_handler(self, mock_cache_datasets):
        """Test DDP handler synchronization."""
        mock_data = DictToObj({"train_data_loader": Mock()})
        mock_cache_datasets.return_value = mock_data

        # Main process with Hydra cfg
        experiment = MockExperiment(cfg=self.cfg, is_main=True)
        processor = MockProcessor()
        data_name = "test_dataset"
        data_source = DictToObj({})

        result = cache_datasets_ddp_handler(experiment, processor, data_name, data_source)

        self.assertEqual(mock_cache_datasets.call_count, 1)
        self.assertEqual(experiment.barrier_calls, 2)
        self.assertEqual(result, mock_data)

        # Reset mocks
        mock_cache_datasets.reset_mock()
        experiment = MockExperiment(cfg=self.cfg, is_main=False)

        # Child process
        result = cache_datasets_ddp_handler(experiment, processor, data_name, data_source)
        self.assertEqual(mock_cache_datasets.call_count, 1)
        self.assertEqual(experiment.barrier_calls, 2)

    def test_save_sample_to_group_dict(self):
        """Test saving dictionary samples to HDF5 group."""
        with h5py.File(os.path.join(self.temp_dir, "test_dict.h5"), "w") as f:
            group = f.create_group("test_group")

            sample = {"image": np.random.randn(3, 32, 32), "label": 5, "name": "test_sample"}

            _save_sample_to_group(sample, group, compression=True)

            # Check that the group has correct attributes and datasets
            self.assertEqual(group.attrs["type"], "dict")
            self.assertEqual(group.attrs["label"], 5)
            self.assertEqual(group.attrs["name"], "test_sample")
            self.assertIn("image_data", group.keys())

    def test_save_sample_to_group_tuple(self):
        """Test saving tuple samples to HDF5 group."""
        with h5py.File(os.path.join(self.temp_dir, "test_tuple.h5"), "w") as f:
            group = f.create_group("test_group")

            sample = (np.random.randn(3, 32, 32), 5, "test")

            _save_sample_to_group(sample, group, compression=True)

            # Check that the group has correct attributes and datasets
            self.assertEqual(group.attrs["type"], "tuple")
            self.assertEqual(group.attrs["num_items"], 3)
            self.assertEqual(group.attrs["item_1_value"], 5)
            self.assertEqual(group.attrs["item_2_value"], "test")
            self.assertIn("item_0_data", group.keys())

    def test_save_sample_to_group_tensor(self):
        """Test saving tensor samples to HDF5 group."""
        with h5py.File(os.path.join(self.temp_dir, "test_tensor.h5"), "w") as f:
            group = f.create_group("test_group")

            sample = np.random.randn(3, 32, 32)

            _save_sample_to_group(sample, group, compression=True)

            # Check that the group has correct attributes and datasets
            self.assertEqual(group.attrs["type"], "tensor")
            self.assertIn("data", group.keys())
            np.testing.assert_array_equal(group["data"][:], sample)


class TestDatasetCachingIntegration(unittest.TestCase):
    """Integration tests for dataset caching."""

    def setUp(self):
        """Set up test environment with Hydra cfg."""
        os.environ["FDQ_UNITTEST"] = "1"
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.temp_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.config_dir = os.path.join(os.path.dirname(__file__), "test_experiment")
        self.conf_name = "mnist_testexp_dense_ci" if os.getenv("GITHUB_ACTIONS") else "mnist_testexp_dense"
        try:
            with initialize_config_dir(version_base=None, config_dir=self.config_dir):
                self.cfg: DictConfig = compose(
                    config_name=self.conf_name,
                    overrides=["hydra.run.dir=.", "hydra.job.chdir=False"],
                )
        except Exception:
            from hydra import initialize

            conf_rel = os.path.relpath(self.config_dir, os.getcwd())
            with initialize(version_base=None, config_path=conf_rel):
                self.cfg = compose(
                    config_name=self.conf_name,
                    overrides=["hydra.run.dir=.", "hydra.job.chdir=False"],
                )

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("fdq.dataset_caching.iprint")
    @patch("fdq.dataset_caching.wprint")
    def test_full_caching_pipeline(self, mock_wprint, mock_iprint):
        """Test the full caching pipeline end-to-end."""
        from fdq.dataset_caching import cache_datasets

        # Hydra-backed MockExperiment and processor
        experiment = MockExperiment(cfg=self.cfg)
        processor = MockProcessor()
        data_name = "test_dataset"

        # Data source configuration (kept minimal; cache_dir comes from temp)
        data_source = OmegaConf.create(
            {
                "caching": {
                    "cache_dir": self.cache_dir,
                    "shuffle_train": True,
                    "shuffle_val": False,
                    "shuffle_test": False,
                    "num_workers": 0,
                    "pin_memory": False,
                    "compress_cache": False,
                    "nondeterministic_transforms": {"processor": None},
                },
                "args": {},
            }
        )

        result = cache_datasets(experiment, processor, data_name, data_source)

        self.assertIsInstance(result, DictToObj)
        self.assertTrue(hasattr(result, "train_data_loader"))
        self.assertTrue(hasattr(result, "val_data_loader"))

        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith(".h5")]
        self.assertGreater(len(cache_files), 0)

        train_loader = result.train_data_loader
        batch = next(iter(train_loader))
        self.assertIsNotNone(batch)


if __name__ == "__main__":
    unittest.main(verbosity=2)
